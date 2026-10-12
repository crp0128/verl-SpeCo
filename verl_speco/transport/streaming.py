# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
MAX_NATIVE_BATCH_SAMPLES = 1024
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import time
from typing import Any, Mapping
PROTOCOL_V3_SCHEMA_VERSION = 3
stream_protocol_IDENTIFIER = re.compile('^[A-Za-z0-9][A-Za-z0-9_.:-]*$')

@dataclass(frozen=True)
class SampleMetadataV3:
    schema_version: int
    run_id: str
    generation: int
    producer_id: str
    producer_incarnation: str
    request_id: str
    sample_id: str
    sequence_no: int
    target_revision: str
    data_version: int
    tokenizer_fingerprint: str
    target_model_fingerprint: str
    target_layer_ids: tuple[int, ...]
    hidden_states_layout: str
    hidden_dtype: str
    token_count: int
    payload_nbytes: int
    rl_step: int | None = None
    dataset_fingerprint: str | None = None
    draft_revision: str | None = None
    plan_id: str | None = None
    prompt_token_count: int | None = None
    supervised_token_count: int | None = None

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PROTOCOL_V3_SCHEMA_VERSION:
            raise ValueError(f'unsupported protocol v3 schema_version={self.schema_version}; expected {PROTOCOL_V3_SCHEMA_VERSION}')
        for name in ('run_id', 'producer_id', 'producer_incarnation', 'request_id', 'sample_id', 'target_revision', 'tokenizer_fingerprint', 'target_model_fingerprint', 'hidden_states_layout', 'hidden_dtype'):
            value = str(getattr(self, name) or '')
            if not value:
                raise ValueError(f'SampleMetadataV3.{name} must not be empty')
        if not stream_protocol_IDENTIFIER.match(self.run_id):
            raise ValueError('SampleMetadataV3.run_id contains unsupported characters')
        if self.generation < 0 or self.sequence_no < 0 or self.data_version < 0:
            raise ValueError('generation, sequence_no and data_version must be non-negative')
        if self.token_count <= 0 or self.payload_nbytes <= 0:
            raise ValueError('token_count and payload_nbytes must be positive')
        if not self.target_layer_ids or any((layer < 0 for layer in self.target_layer_ids)):
            raise ValueError('target_layer_ids must contain non-negative layer IDs')
        if self.rl_step is not None and self.rl_step < 0:
            raise ValueError('rl_step must be non-negative when present')
        for name in ('draft_revision', 'plan_id'):
            value = getattr(self, name)
            if value is not None and (not str(value)):
                raise ValueError(f'{name} must be nonempty when present')
        if (self.prompt_token_count is None) != (self.supervised_token_count is None):
            raise ValueError('prompt and supervised token counts must be supplied together')
        if self.prompt_token_count is not None and (self.prompt_token_count < 0 or self.supervised_token_count <= 0 or self.prompt_token_count + self.supervised_token_count != self.token_count):
            raise ValueError('prompt/supervised token mapping differs from token_count')
        if self.dataset_fingerprint is not None and (not re.fullmatch('[0-9a-f]{64}', self.dataset_fingerprint)):
            raise ValueError('dataset_fingerprint must be a lowercase SHA-256 digest')

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload['target_layer_ids'] = list(self.target_layer_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'SampleMetadataV3':
        try:
            metadata = cls(schema_version=payload['schema_version'], run_id=str(payload['run_id']), generation=int(payload['generation']), producer_id=str(payload['producer_id']), producer_incarnation=str(payload['producer_incarnation']), request_id=str(payload['request_id']), sample_id=str(payload['sample_id']), sequence_no=int(payload['sequence_no']), target_revision=str(payload['target_revision']), data_version=int(payload['data_version']), tokenizer_fingerprint=str(payload['tokenizer_fingerprint']), target_model_fingerprint=str(payload['target_model_fingerprint']), target_layer_ids=tuple((int(item) for item in payload['target_layer_ids'])), hidden_states_layout=str(payload['hidden_states_layout']), hidden_dtype=str(payload['hidden_dtype']), token_count=int(payload['token_count']), payload_nbytes=int(payload['payload_nbytes']), rl_step=stream_protocol_optional_int(payload.get('rl_step')), dataset_fingerprint=str(payload['dataset_fingerprint']) if payload.get('dataset_fingerprint') is not None else None, draft_revision=str(payload['draft_revision']) if payload.get('draft_revision') is not None else None, plan_id=str(payload['plan_id']) if payload.get('plan_id') is not None else None, prompt_token_count=stream_protocol_optional_int(payload.get('prompt_token_count')), supervised_token_count=stream_protocol_optional_int(payload.get('supervised_token_count')))
        except KeyError as exc:
            raise ValueError(f'protocol v3 metadata missing field {exc.args[0]!r}') from exc
        metadata.validate()
        return metadata

def make_v3_sample_key(metadata: SampleMetadataV3) -> str:
    """Return the idempotency key for one logical captured sample."""
    metadata.validate()
    return f'draft:v3:{metadata.run_id}:g{metadata.generation}:{metadata.producer_id}:{metadata.sequence_no:020d}:{metadata.sample_id}'

def make_v3_ready_tag(metadata: SampleMetadataV3, *, payload_checksum: str) -> dict[str, Any]:
    """Create a small tag that can be filtered before downloading payload bytes."""
    metadata.validate()
    checksum = str(payload_checksum).lower()
    if not re.fullmatch('[0-9a-f]{64}', checksum):
        raise ValueError('payload_checksum must be a lowercase SHA-256 digest')
    return {'record_type': 'sample', 'status': 'ready', 'payload_checksum': checksum, **metadata.to_dict()}

def parse_v3_ready_tag(tag: Mapping[str, Any]) -> tuple[SampleMetadataV3, str] | None:
    if tag.get('record_type') != 'sample' or tag.get('status') != 'ready':
        return None
    try:
        metadata = SampleMetadataV3.from_dict(tag)
        checksum = str(tag['payload_checksum']).lower()
    except (KeyError, TypeError, ValueError):
        return None
    if not re.fullmatch('[0-9a-f]{64}', checksum):
        return None
    return (metadata, checksum)

def transaction_set_hash(transaction_ids: tuple[str, ...] | list[str]) -> str:
    """Hash the order-independent transaction identity set for one optimizer step."""
    normalized = tuple(sorted({str(item) for item in transaction_ids if str(item)}))
    if not normalized:
        raise ValueError('transaction set must not be empty')
    return hashlib.sha256(json.dumps(normalized, separators=(',', ':')).encode('utf-8')).hexdigest()

def stable_acquire_id(*, run_id: str, generation: int, consumer_id: str, batch_sequence: int) -> str:
    if generation < 0 or batch_sequence < 0:
        raise ValueError('generation and batch_sequence must be non-negative')
    material = f'{run_id}:{generation}:{consumer_id}:{batch_sequence}'.encode('utf-8')
    return f'acq-{hashlib.sha256(material).hexdigest()[:32]}'

def utc_timestamp() -> float:
    return time.time()

def stream_protocol_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
import base64
from collections.abc import Mapping
import zlib
CODEC = 'zlib-sha256-v1'
MAX_RAW_BYTES = 128 * 1024 ** 2
MAX_ENCODED_BYTES = 12 * 1024 ** 2

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def stream_snapshot_codec_validate(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {'schema_version', 'stream_scope', 'committed_cursor', 'committed_optimizer_steps'} or type(snapshot['schema_version']) is not int or (snapshot['schema_version'] != 1) or (not isinstance(snapshot['stream_scope'], str)) or (not snapshot['stream_scope']) or (type(snapshot['committed_cursor']) is not int) or (snapshot['committed_cursor'] < 0) or (not isinstance(snapshot['committed_optimizer_steps'], dict)):
        raise ValueError('invalid complete stream snapshot for lossless receipt')
    for step, ids in snapshot['committed_optimizer_steps'].items():
        if not isinstance(step, str) or not step or (not isinstance(ids, (list, tuple))) or (not ids) or any((not isinstance(i, str) or not i for i in ids)) or (len(set(ids)) != len(ids)):
            raise ValueError('invalid complete transaction history for lossless receipt')

def encode_stream_snapshot(snapshot):
    stream_snapshot_codec_validate(snapshot)
    raw = canonical(snapshot)
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError('stream snapshot exceeds decoded byte budget')
    encoded = base64.b64encode(zlib.compress(raw, level=1)).decode('ascii')
    if len(encoded) > MAX_ENCODED_BYTES:
        raise ValueError('lossless stream receipt exceeds its encoded budget')
    return dict(schema_version=2, codec=CODEC, stream_scope=snapshot['stream_scope'], committed_cursor=snapshot['committed_cursor'], decoded_nbytes=len(raw), decoded_sha256=hashlib.sha256(raw).hexdigest(), payload_base64=encoded)

def decode_stream_snapshot(snapshot):
    if not isinstance(snapshot, Mapping) or 'codec' not in snapshot:
        return snapshot
    required = {'schema_version', 'codec', 'stream_scope', 'committed_cursor', 'decoded_nbytes', 'decoded_sha256', 'payload_base64'}
    if set(snapshot) != required or type(snapshot['schema_version']) is not int or snapshot['schema_version'] != 2 or (snapshot['codec'] != CODEC) or (type(snapshot['decoded_nbytes']) is not int) or (not 1 <= snapshot['decoded_nbytes'] <= MAX_RAW_BYTES) or (not isinstance(snapshot['payload_base64'], str)) or (len(snapshot['payload_base64']) > MAX_ENCODED_BYTES) or (not isinstance(snapshot['decoded_sha256'], str)) or (len(snapshot['decoded_sha256']) != 64) or (type(snapshot['committed_cursor']) is not int):
        raise ValueError('invalid lossless stream receipt envelope')
    try:
        compressed = base64.b64decode(snapshot['payload_base64'], validate=True)
        inflater = zlib.decompressobj()
        raw = inflater.decompress(compressed, snapshot['decoded_nbytes'] + 1)
    except (ValueError, zlib.error) as error:
        raise ValueError('invalid compressed stream receipt bytes') from error
    if len(raw) != snapshot['decoded_nbytes'] or not inflater.eof or inflater.unconsumed_tail or inflater.unused_data or (hashlib.sha256(raw).hexdigest() != snapshot['decoded_sha256']):
        raise ValueError('stream receipt complete bytes, size or SHA differ')
    result = json.loads(raw)
    stream_snapshot_codec_validate(result)
    if result['stream_scope'] != snapshot['stream_scope'] or result['committed_cursor'] != snapshot['committed_cursor']:
        raise ValueError('stream receipt header differs from its complete history')
    return result

def receipt_stream_codec(environment):
    mode = environment.get('SPECO_STREAM_SNAPSHOT_CODEC', '')
    if mode not in ('', CODEC):
        raise ValueError('unsupported stream snapshot receipt codec')
    return mode
import threading
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class TrainingTransaction:
    transaction_id: str
    lease_ids: tuple[str, ...]
    keys: tuple[str, ...]
    stream_scope: str
    stream_cursor: int
    generation: int
    target_revision: str

    def validate(self) -> None:
        if not self.transaction_id or not self.stream_scope or (not self.target_revision):
            raise ValueError('transaction identity fields must not be empty')
        if self.stream_cursor < 0 or self.generation < 0:
            raise ValueError('transaction cursor and generation must be non-negative')
        if not self.lease_ids or not self.keys or len(self.lease_ids) != len(self.keys):
            raise ValueError('transaction lease_ids and keys must be non-empty and aligned')

@dataclass(frozen=True)
class StreamSnapshot:
    schema_version: int
    stream_scope: str
    committed_cursor: int
    committed_optimizer_steps: Mapping[str, tuple[str, ...]]

class TrainingStreamCoordinator:
    """Advance a restartable stream only through a contiguous committed prefix."""

    def __init__(self, stream_scope: str, *, remembered_steps: int=4096) -> None:
        if not stream_scope:
            raise ValueError('stream_scope must not be empty')
        if remembered_steps <= 0:
            raise ValueError('remembered_steps must be positive')
        self._lock = threading.RLock()
        self._stream_scope = stream_scope
        self._remembered_steps = remembered_steps
        self._committed_cursor = 0
        self._pending: dict[int, TrainingTransaction] = {}
        self._commit_ready_cursors: set[int] = set()
        self._committed_steps: dict[str, tuple[str, ...]] = {}
        self._failed_reason: str | None = None

    def register(self, transaction: TrainingTransaction) -> None:
        transaction.validate()
        if transaction.stream_scope != self._stream_scope:
            raise ValueError('transaction belongs to another stream scope')
        with self._lock:
            self.require_healthy()
            existing = self._pending.get(transaction.stream_cursor)
            if existing is not None and existing != transaction:
                raise ValueError('stream cursor is already assigned to another transaction')
            if transaction.stream_cursor <= self._committed_cursor:
                raise ValueError('transaction cursor is not after the committed prefix')
            if any((pending.transaction_id == transaction.transaction_id and pending.stream_cursor != transaction.stream_cursor for pending in self._pending.values())):
                raise ValueError('transaction ID is already assigned to another cursor')
            self._pending[transaction.stream_cursor] = transaction

    def commit(self, optimizer_step_id: str, transaction_ids: Sequence[str]) -> bool:
        """Commit a transaction set once and return whether it was already committed."""
        normalized = tuple(sorted({str(item) for item in transaction_ids if str(item)}))
        if not optimizer_step_id or not normalized:
            raise ValueError('optimizer_step_id and transaction_ids are required')
        with self._lock:
            self.require_healthy()
            remembered = self._committed_steps.get(optimizer_step_id)
            if remembered is not None:
                if remembered != normalized:
                    raise ValueError('optimizer step ID was reused with different transactions')
                return True
            pending_by_id = {item.transaction_id: item for item in self._pending.values()}
            missing = [item for item in normalized if item not in pending_by_id]
            if missing:
                raise ValueError(f'unknown transaction IDs in stream commit: {missing}')
            for transaction in (pending_by_id[item] for item in normalized):
                self._commit_ready_cursors.add(transaction.stream_cursor)
            self._advance_contiguous_prefix()
            self._committed_steps[optimizer_step_id] = normalized
            while len(self._committed_steps) > self._remembered_steps:
                self._committed_steps.pop(next(iter(self._committed_steps)))
            return False

    def validate_commit(self, optimizer_step_id: str, transaction_ids: Sequence[str]) -> None:
        """Check stream identity and pending transactions before any lease mutation."""
        normalized = tuple(sorted({str(item) for item in transaction_ids if str(item)}))
        if not optimizer_step_id or not normalized:
            raise ValueError('optimizer_step_id and transaction_ids are required')
        with self._lock:
            self.require_healthy()
            remembered = self._committed_steps.get(optimizer_step_id)
            if remembered is not None:
                if remembered != normalized:
                    raise ValueError('optimizer step ID was reused with different transactions')
                return
            pending_ids = {item.transaction_id for item in self._pending.values()}
            missing = [item for item in normalized if item not in pending_ids]
            if missing:
                raise ValueError(f'unknown transaction IDs in stream commit: {missing}')

    def fail(self, reason: str) -> None:
        with self._lock:
            self._failed_reason = str(reason) or 'stream failed; restore last committed checkpoint'

    def require_healthy(self) -> None:
        if self._failed_reason is not None:
            raise RuntimeError(self._failed_reason)

    def snapshot(self) -> StreamSnapshot:
        with self._lock:
            return StreamSnapshot(schema_version=1, stream_scope=self._stream_scope, committed_cursor=self._committed_cursor, committed_optimizer_steps=dict(self._committed_steps))

    def restore(self, snapshot: StreamSnapshot | Mapping[str, Any]) -> None:
        snapshot = decode_stream_snapshot(snapshot)
        if isinstance(snapshot, Mapping):
            snapshot = StreamSnapshot(schema_version=int(snapshot['schema_version']), stream_scope=str(snapshot['stream_scope']), committed_cursor=int(snapshot['committed_cursor']), committed_optimizer_steps={str(key): tuple((str(item) for item in value)) for key, value in dict(snapshot['committed_optimizer_steps']).items()})
        if snapshot.schema_version != 1 or snapshot.stream_scope != self._stream_scope:
            raise ValueError('stream snapshot is incompatible')
        if snapshot.committed_cursor < 0:
            raise ValueError('stream snapshot cursor must be non-negative')
        with self._lock:
            self._committed_cursor = snapshot.committed_cursor
            self._committed_steps = dict(snapshot.committed_optimizer_steps)
            self._pending.clear()
            self._commit_ready_cursors.clear()
            self._failed_reason = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {'stream_scope': self._stream_scope, 'committed_cursor': self._committed_cursor, 'open_transactions': len(self._pending), 'committed_optimizer_steps': len(self._committed_steps), 'failed': self._failed_reason is not None}

    def _advance_contiguous_prefix(self) -> None:
        while self._committed_cursor + 1 in self._commit_ready_cursors:
            cursor = self._committed_cursor + 1
            self._commit_ready_cursors.remove(cursor)
            self._pending.pop(cursor, None)
            self._committed_cursor = cursor
