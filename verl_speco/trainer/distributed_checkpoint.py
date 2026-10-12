# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
import torch
import torch.distributed as dist

@dataclass(frozen=True)
class ParameterDigest:
    value: bytes

    def digest(self):
        return self.value

    def hexdigest(self):
        return self.value.hex()

def exact_device_parameter_consensus(module, *, digest_fn, chunk_bytes=64 * 1024 ** 2):
    """Broadcast isolated scratch bytes, never peer parameters. All peers compare.

    Existing canonical name/byte SHA remains the root digest, after bitwise
    device equality. This removes redundant D2H/SHA on replica ranks without
    replacing an exact comparison by a floating-point norm or sampled checksum.
    """
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024 ** 2 or (not callable(digest_fn)) or (not dist.is_initialized()):
        raise ValueError('device consensus requires a live group and bounded scratch')
    weights = []
    for name, weight in sorted(module.named_parameters()):
        if not weight.requires_grad:
            continue
        value = weight.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        weights.append((name, value))
    if not weights:
        raise ValueError('device consensus requires trainable parameters')
    device = weights[0][1].device
    if any((value.device != device for _, value in weights)):
        raise ValueError('device consensus requires one parameter device per rank')
    schema = [(name, list(v.shape), str(v.dtype)) for name, v in weights]
    schema_bytes = hashlib.sha256(json.dumps(dict(parameters=schema, chunk_bytes=chunk_bytes), separators=(',', ':')).encode()).digest()
    own = torch.tensor(list(schema_bytes), dtype=torch.uint8, device=device)
    peers = [torch.empty_like(own) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, own)
    if any((not torch.equal(own, peer) for peer in peers)):
        raise RuntimeError('DP parameter schema differs before byte consensus')
    maximum = max((v.numel() * v.element_size() for _, v in weights))
    if not maximum:
        raise ValueError('device consensus has no parameter bytes')
    scratch = torch.empty(min(maximum, chunk_bytes), dtype=torch.uint8, device=device)
    identical = True
    rank = dist.get_rank()
    for _, value in weights:
        raw = value.contiguous().reshape(-1).view(torch.uint8)
        for start in range(0, raw.numel(), chunk_bytes):
            local = raw[start:start + chunk_bytes]
            view = scratch[:local.numel()]
            if rank == 0:
                view.copy_(local)
            dist.broadcast(view, src=0)
            if not torch.equal(view, local):
                identical = False
    verdict = torch.tensor([int(identical)], dtype=torch.int32, device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    if verdict.item() != 1:
        raise RuntimeError('DP optimizer parameter bytes differ before checkpoint')
    error = None
    digest = None
    if rank == 0:
        try:
            digest = digest_fn(module).digest()
            if not isinstance(digest, bytes) or len(digest) != 32:
                raise ValueError('canonical parameter digest must be SHA256 bytes')
        except Exception as exc:
            error = exc
    ready = torch.tensor([int(rank != 0 or error is None)], dtype=torch.int32, device=device)
    dist.broadcast(ready, src=0)
    if ready.item() != 1:
        raise RuntimeError('root canonical parameter SHA failed') from error
    payload = torch.tensor(list(digest) if rank == 0 else [0] * 32, dtype=torch.uint8, device=device)
    dist.broadcast(payload, src=0)
    return ParameterDigest(bytes(payload.cpu().tolist()))
from dataclasses import asdict, dataclass
from enum import Enum
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
from verl_speco.transport.control import CommitReceipt, LeaseRegistry, LeaseState, bounded_lease_records
from verl_speco.transport.streaming import transaction_set_hash
from verl_speco.transport.streaming import TrainingStreamCoordinator

class StepState(str, Enum):
    PREPARED = 'STEP_PREPARED'
    COMMITTED = 'STEP_COMMITTED'
    ABORTED = 'STEP_ABORTED'

class RecoveryAction(str, Enum):
    RESUME_COMMITTED = 'resume_committed'
    ROLLBACK_GENERATION = 'rollback_generation'
    FAIL_CLOSED = 'fail_closed'

@dataclass(frozen=True)
class OptimizerTransaction:
    optimizer_step_id: str
    plan_id: str
    generation: int
    stream_scope: str
    transaction_ids: tuple[str, ...]
    lease_ids: tuple[str, ...]
    transaction_set_hash: str
    checkpoint_generation_before: int

    @classmethod
    def create(cls, *, optimizer_step_id: str, plan_id: str, generation: int, stream_scope: str, transaction_ids: Sequence[str], lease_ids: Sequence[str], checkpoint_generation_before: int) -> 'OptimizerTransaction':
        normalized_transactions = tuple(sorted({str(item) for item in transaction_ids if str(item)}))
        normalized_leases = tuple(sorted({str(item) for item in lease_ids if str(item)}))
        tx = cls(optimizer_step_id=str(optimizer_step_id), plan_id=str(plan_id), generation=int(generation), stream_scope=str(stream_scope), transaction_ids=normalized_transactions, lease_ids=normalized_leases, transaction_set_hash=transaction_set_hash(normalized_transactions), checkpoint_generation_before=int(checkpoint_generation_before))
        tx.validate()
        return tx

    def validate(self) -> None:
        if not self.optimizer_step_id or not self.plan_id or (not self.stream_scope):
            raise ValueError('optimizer_step_id, plan_id and stream_scope are required')
        if self.generation < 0 or self.checkpoint_generation_before < 0:
            raise ValueError('generation values must be non-negative')
        if not self.transaction_ids or not self.lease_ids:
            raise ValueError('transaction_ids and lease_ids must not be empty')
        if len(self.transaction_ids) != len(self.lease_ids):
            raise ValueError('transaction_ids and lease_ids must have the same cardinality')
        if self.transaction_set_hash != transaction_set_hash(self.transaction_ids):
            raise ValueError('transaction_set_hash does not match transaction_ids')

@dataclass(frozen=True)
class CandidateCheckpoint:
    uri: str
    checksum: str
    checkpoint_generation: int

    def validate(self) -> None:
        if not self.uri or not self.checksum:
            raise ValueError('candidate checkpoint uri and checksum are required')
        if self.checkpoint_generation < 0:
            raise ValueError('candidate checkpoint generation must be non-negative')

@dataclass(frozen=True)
class StepJournalRecord:
    schema_version: int
    state: StepState
    transaction: OptimizerTransaction
    topology_fingerprint: str
    model_fingerprint: str
    candidate_checkpoint: CandidateCheckpoint | None
    optimizer_started: bool
    created_at: float
    updated_at: float
    reason: str = ''

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError('unsupported optimizer journal schema')
        self.transaction.validate()
        if not self.topology_fingerprint or not self.model_fingerprint:
            raise ValueError('journal fingerprints must not be empty')
        if self.state is StepState.COMMITTED:
            if self.candidate_checkpoint is None:
                raise ValueError('committed journal record requires a candidate checkpoint')
            self.candidate_checkpoint.validate()

@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    optimizer_step_id: str
    reason: str

class JsonStepJournal:
    """One atomic JSON record per optimizer step in a durable journal directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def load(self, optimizer_step_id: str) -> StepJournalRecord | None:
        path = self._path(optimizer_step_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f'cannot read optimizer journal {path}: {exc}') from exc
        return optimizer_transaction_record_from_dict(payload)

    def store(self, record: StepJournalRecord) -> None:
        record.validate()
        payload = optimizer_transaction_record_to_dict(record)
        path = self._path(record.transaction.optimizer_step_id)
        with self._lock:
            fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix='.tmp', dir=path.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY) if os.name != 'nt' else None
                try:
                    os.fsync(directory_fd) if directory_fd is not None else None
                finally:
                    os.close(directory_fd) if directory_fd is not None else None
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)

    def _path(self, optimizer_step_id: str) -> Path:
        if not optimizer_step_id:
            raise ValueError('optimizer_step_id must not be empty')
        digest = hashlib.sha256(optimizer_step_id.encode('utf-8')).hexdigest()
        return self.root / f'step-{digest}.json'

class CommitExecutor:
    """Serialize prepare/candidate/commit state for one training stream."""

    def __init__(self, *, lease_registry: LeaseRegistry, stream: TrainingStreamCoordinator, journal: JsonStepJournal, topology_fingerprint: str, model_fingerprint: str) -> None:
        if not topology_fingerprint or not model_fingerprint:
            raise ValueError('topology_fingerprint and model_fingerprint are required')
        self._lease_registry = lease_registry
        self._stream = stream
        self._journal = journal
        self._topology_fingerprint = topology_fingerprint
        self._model_fingerprint = model_fingerprint
        self._lock = threading.RLock()
        self._pending: dict[str, StepJournalRecord] = {}
        self._commit_attempted: set[str] = set()

    def prepare(self, transaction: OptimizerTransaction) -> StepJournalRecord:
        """Durably mark a transaction before the caller applies the optimizer."""
        transaction.validate()
        with self._lock:
            existing = self._journal.load(transaction.optimizer_step_id)
            if existing is not None:
                self._require_same_transaction(existing, transaction)
                if existing.state is StepState.COMMITTED:
                    return existing
                if existing.state is StepState.PREPARED:
                    if transaction.optimizer_step_id not in self._pending:
                        raise RuntimeError('prepared optimizer transaction belongs to an earlier executor; recover by rolling back the generation')
                    if existing.optimizer_started:
                        raise RuntimeError('prepared optimizer transaction may already have entered optimizer_step; recover by rolling back the generation')
                    self._pending[transaction.optimizer_step_id] = existing
                    return existing
                raise RuntimeError(f'optimizer step {transaction.optimizer_step_id!r} was aborted; start a new generation instead of reusing the step ID')
            now = time.time()
            record = StepJournalRecord(schema_version=1, state=StepState.PREPARED, transaction=transaction, topology_fingerprint=self._topology_fingerprint, model_fingerprint=self._model_fingerprint, candidate_checkpoint=None, optimizer_started=False, created_at=now, updated_at=now)
            self._journal.store(record)
            self._pending[transaction.optimizer_step_id] = record
            return record

    def begin_optimizer(self, optimizer_step_id: str) -> StepJournalRecord:
        """Persist the point after which optimizer execution must never be retried."""
        with self._lock:
            record = self._require_prepared(optimizer_step_id)
            if record.optimizer_started:
                raise RuntimeError('optimizer was already marked started for this step; recover by rolling back the generation')
            updated = StepJournalRecord(schema_version=record.schema_version, state=record.state, transaction=record.transaction, topology_fingerprint=record.topology_fingerprint, model_fingerprint=record.model_fingerprint, candidate_checkpoint=record.candidate_checkpoint, optimizer_started=True, created_at=record.created_at, updated_at=time.time())
            self._journal.store(updated)
            self._pending[optimizer_step_id] = updated
            return updated

    def attach_candidate(self, optimizer_step_id: str, candidate_checkpoint: CandidateCheckpoint) -> StepJournalRecord:
        """Record a validated DCP candidate after one successful optimizer step."""
        candidate_checkpoint.validate()
        with self._lock:
            record = self._require_prepared(optimizer_step_id)
            if not record.optimizer_started:
                raise RuntimeError('begin_optimizer must be recorded before attaching a checkpoint')
            if record.candidate_checkpoint is not None:
                if record.candidate_checkpoint != candidate_checkpoint:
                    raise ValueError('optimizer step already has a different candidate checkpoint')
                return record
            updated = StepJournalRecord(schema_version=record.schema_version, state=StepState.PREPARED, transaction=record.transaction, topology_fingerprint=record.topology_fingerprint, model_fingerprint=record.model_fingerprint, candidate_checkpoint=candidate_checkpoint, optimizer_started=True, created_at=record.created_at, updated_at=time.time())
            self._journal.store(updated)
            self._pending[optimizer_step_id] = updated
            return updated

    def attach_live_update(self, optimizer_step_id: str, candidate: CandidateCheckpoint, *, live_owner_token: str) -> StepJournalRecord:
        """Bind process-owned progress; this descriptor cannot restore optimizer state."""
        from verl_speco.runtime.checkpoint_policy import read_live_update
        live = read_live_update(candidate, owner_token=live_owner_token)
        if live['step_id'] != optimizer_step_id:
            raise ValueError('live update names a different optimizer step')
        return self.attach_candidate(optimizer_step_id, candidate)

    def commit(self, optimizer_step_id: str, *, live_owner_token: str | None=None) -> CommitReceipt:
        """Commit payload state exactly once after a candidate DCP is durable."""
        with self._lock:
            record = self._require_prepared(optimizer_step_id)
            if record.candidate_checkpoint is None:
                raise RuntimeError('candidate checkpoint must be durable before payload commit')
            from verl_speco.runtime.checkpoint_policy import is_live_candidate, read_live_update
            if is_live_candidate(record.candidate_checkpoint):
                if live_owner_token is None:
                    raise RuntimeError('live optimizer update requires the current owner; it is not a durable checkpoint')
                live = read_live_update(record.candidate_checkpoint, owner_token=live_owner_token)
                if live['step_id'] != optimizer_step_id or live['run_id'] != record.transaction.stream_scope or live['generation'] != record.transaction.generation or (live['topology_fingerprint'] != self._topology_fingerprint) or (live['model_fingerprint'] != self._model_fingerprint):
                    raise ValueError('live optimizer participant identity differs')
            if not record.optimizer_started:
                raise RuntimeError('optimizer was not marked started')
            if optimizer_step_id in self._commit_attempted:
                raise RuntimeError('commit was already attempted for this optimizer step; recover by rolling back the generation')
            self._commit_attempted.add(optimizer_step_id)
            self._lease_registry.validate_commit(optimizer_step_id=record.transaction.optimizer_step_id, transaction_set_hash=record.transaction.transaction_set_hash, lease_ids=record.transaction.lease_ids)
            self._stream.validate_commit(record.transaction.optimizer_step_id, record.transaction.transaction_ids)
            receipt = self._lease_registry.commit(optimizer_step_id=record.transaction.optimizer_step_id, transaction_set_hash=record.transaction.transaction_set_hash, lease_ids=record.transaction.lease_ids)
            self._stream.commit(record.transaction.optimizer_step_id, record.transaction.transaction_ids)
            committed = StepJournalRecord(schema_version=record.schema_version, state=StepState.COMMITTED, transaction=record.transaction, topology_fingerprint=record.topology_fingerprint, model_fingerprint=record.model_fingerprint, candidate_checkpoint=record.candidate_checkpoint, optimizer_started=True, created_at=record.created_at, updated_at=time.time())
            self._journal.store(committed)
            self._pending.pop(optimizer_step_id, None)
            return receipt

    def abort(self, optimizer_step_id: str, *, reason: str) -> StepJournalRecord:
        """Mark an uncommitted step terminal; callers must start a new generation."""
        with self._lock:
            record = self._require_prepared(optimizer_step_id)
            self._lease_registry.cancel(record.transaction.lease_ids, reason=reason)
            aborted = StepJournalRecord(schema_version=record.schema_version, state=StepState.ABORTED, transaction=record.transaction, topology_fingerprint=record.topology_fingerprint, model_fingerprint=record.model_fingerprint, candidate_checkpoint=record.candidate_checkpoint, optimizer_started=record.optimizer_started, created_at=record.created_at, updated_at=time.time(), reason=str(reason))
            self._journal.store(aborted)
            self._pending.pop(optimizer_step_id, None)
            return aborted

    def drain(self) -> None:
        """Refuse checkpoint/publish while a prepared step remains unresolved."""
        with self._lock:
            if self._pending:
                pending = ', '.join(sorted(self._pending))
                raise RuntimeError(f'pending optimizer transactions must be resolved: {pending}')

    def recover(self, optimizer_step_id: str, *, live_owner_token: str | None=None) -> RecoveryDecision:
        """Return conservative recovery action without replaying an optimizer step."""
        record = self._journal.load(optimizer_step_id)
        if record is None:
            return RecoveryDecision(RecoveryAction.ROLLBACK_GENERATION, optimizer_step_id, 'journal record missing')
        if record.topology_fingerprint != self._topology_fingerprint:
            return RecoveryDecision(RecoveryAction.FAIL_CLOSED, optimizer_step_id, 'topology fingerprint changed')
        if record.model_fingerprint != self._model_fingerprint:
            return RecoveryDecision(RecoveryAction.FAIL_CLOSED, optimizer_step_id, 'model fingerprint changed')
        if record.state is StepState.COMMITTED:
            if record.candidate_checkpoint is None:
                return RecoveryDecision(RecoveryAction.FAIL_CLOSED, optimizer_step_id, 'committed record has no candidate checkpoint')
            from verl_speco.runtime.checkpoint_policy import is_live_candidate
            if is_live_candidate(record.candidate_checkpoint) and live_owner_token is None:
                return RecoveryDecision(RecoveryAction.ROLLBACK_GENERATION, optimizer_step_id, 'live progress is not restartable; restore the latest full checkpoint')
            return RecoveryDecision(RecoveryAction.RESUME_COMMITTED, optimizer_step_id, 'complete journal record')
        return RecoveryDecision(RecoveryAction.ROLLBACK_GENERATION, optimizer_step_id, f'journal state is {record.state.value}; optimizer must not be retried')

    def recover_consistent(self, optimizer_step_id: str, *, live_owner_token: str | None=None) -> RecoveryDecision:
        """Resume only when journal, DCP manifest, leases and stream agree.

        The caller still restores the full DCP through VeOmniDCPBridge, which
        verifies every checkpoint file before loading model/optimizer state.
        """
        decision = self.recover(optimizer_step_id, live_owner_token=live_owner_token)
        if decision.action is not RecoveryAction.RESUME_COMMITTED:
            return decision
        record = self._journal.load(optimizer_step_id)
        assert record is not None and record.candidate_checkpoint is not None
        candidate = record.candidate_checkpoint
        try:
            from verl_speco.runtime.checkpoint_policy import is_live_candidate, read_live_update
            if is_live_candidate(candidate):
                manifest = read_live_update(candidate, owner_token=live_owner_token)
                if manifest['step_id'] != optimizer_step_id or manifest['run_id'] != record.transaction.stream_scope:
                    raise ValueError('live update transaction identity differs')
            else:
                manifest_path = Path(candidate.uri) / 'manifest.json'
                manifest_bytes = manifest_path.read_bytes()
                digest = hashlib.sha256(manifest_bytes).hexdigest()
                if digest != candidate.checksum:
                    raise ValueError('DCP manifest checksum differs from journal')
                manifest = json.loads(manifest_bytes)
                if manifest.get('generation') != candidate.checkpoint_generation or manifest.get('topology_fingerprint') != self._topology_fingerprint or manifest.get('model_fingerprint') != self._model_fingerprint:
                    raise ValueError('DCP manifest identity differs from journal')
            if manifest.get('topology_fingerprint') != self._topology_fingerprint or manifest.get('model_fingerprint') != self._model_fingerprint:
                raise ValueError('optimizer state identity differs')
            self._lease_registry.validate_commit(optimizer_step_id=record.transaction.optimizer_step_id, transaction_set_hash=record.transaction.transaction_set_hash, lease_ids=record.transaction.lease_ids)
            for lease in bounded_lease_records(self._lease_registry, record.transaction.lease_ids):
                if lease.state is not LeaseState.COMMITTED or not lease.payload_released:
                    raise ValueError('journal names a lease that is not committed')
            committed = self._stream.snapshot().committed_optimizer_steps.get(optimizer_step_id)
            if committed != record.transaction.transaction_ids:
                raise ValueError('stream transaction set differs from journal')
        except Exception as exc:
            return RecoveryDecision(RecoveryAction.ROLLBACK_GENERATION, optimizer_step_id, f'committed step is inconsistent: {exc}')
        return decision

    def _require_prepared(self, optimizer_step_id: str) -> StepJournalRecord:
        record = self._pending.get(optimizer_step_id) or self._journal.load(optimizer_step_id)
        if record is None:
            raise ValueError(f'unknown optimizer transaction {optimizer_step_id!r}')
        if record.state is not StepState.PREPARED:
            raise RuntimeError(f'optimizer transaction is not prepared: {record.state.value}')
        return record

    @staticmethod
    def _require_same_transaction(record: StepJournalRecord, transaction: OptimizerTransaction) -> None:
        if record.transaction != transaction:
            raise ValueError('optimizer step ID was reused with a different transaction')

def optimizer_transaction_record_to_dict(record: StepJournalRecord) -> dict[str, Any]:
    record.validate()
    transaction = asdict(record.transaction)
    transaction['transaction_ids'] = list(record.transaction.transaction_ids)
    transaction['lease_ids'] = list(record.transaction.lease_ids)
    candidate = asdict(record.candidate_checkpoint) if record.candidate_checkpoint else None
    return {'schema_version': record.schema_version, 'state': record.state.value, 'transaction': transaction, 'topology_fingerprint': record.topology_fingerprint, 'model_fingerprint': record.model_fingerprint, 'candidate_checkpoint': candidate, 'optimizer_started': record.optimizer_started, 'created_at': record.created_at, 'updated_at': record.updated_at, 'reason': record.reason}

def optimizer_transaction_record_from_dict(payload: Mapping[str, Any]) -> StepJournalRecord:
    raw_tx = payload.get('transaction')
    if not isinstance(raw_tx, Mapping):
        raise ValueError('journal record transaction must be a mapping')
    transaction = OptimizerTransaction(optimizer_step_id=str(raw_tx['optimizer_step_id']), plan_id=str(raw_tx['plan_id']), generation=int(raw_tx['generation']), stream_scope=str(raw_tx['stream_scope']), transaction_ids=tuple((str(item) for item in raw_tx['transaction_ids'])), lease_ids=tuple((str(item) for item in raw_tx['lease_ids'])), transaction_set_hash=str(raw_tx['transaction_set_hash']), checkpoint_generation_before=int(raw_tx['checkpoint_generation_before']))
    raw_candidate = payload.get('candidate_checkpoint')
    candidate = None
    if raw_candidate is not None:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError('journal candidate_checkpoint must be a mapping')
        candidate = CandidateCheckpoint(uri=str(raw_candidate['uri']), checksum=str(raw_candidate['checksum']), checkpoint_generation=int(raw_candidate['checkpoint_generation']))
    record = StepJournalRecord(schema_version=int(payload['schema_version']), state=StepState(str(payload['state'])), transaction=transaction, topology_fingerprint=str(payload['topology_fingerprint']), model_fingerprint=str(payload['model_fingerprint']), candidate_checkpoint=candidate, optimizer_started=bool(payload.get('optimizer_started', False)), created_at=float(payload['created_at']), updated_at=float(payload['updated_at']), reason=str(payload.get('reason', '')))
    record.validate()
    return record
from verl_speco.runtime.artifacts import read_published_json, publish_json
MAX_NATIVE_JOURNAL_BYTES = 16 * 1024 ** 2

class PublishedNativeStepJournal(JsonStepJournal):

    def load(self, optimizer_step_id):
        with self._lock:
            try:
                value = read_published_json(self._path(optimizer_step_id), max_bytes=MAX_NATIVE_JOURNAL_BYTES)
            except FileNotFoundError:
                return None
        return optimizer_transaction_record_from_dict(value)

    def store(self, record):
        record.validate()
        payload = optimizer_transaction_record_to_dict(record)
        if len(json.dumps(payload, sort_keys=True).encode()) > MAX_NATIVE_JOURNAL_BYTES:
            raise ValueError('native journal exceeds metadata budget')
        with self._lock:
            publish_json(self._path(record.transaction.optimizer_step_id), payload)

def native_step_journal(root, *, shared=False, outputs_root=None):
    if type(shared) is not bool:
        raise ValueError('native journal storage policy must be boolean')
    if not shared:
        return JsonStepJournal(root)
    path = Path(root)
    allowed = Path(outputs_root) if outputs_root is not None else Path(__file__).resolve().parents[2] / 'outputs'
    if not path.is_absolute() or not path.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('shared native journal must remain within owned outputs')
    return PublishedNativeStepJournal(path.resolve())
import dataclasses
import pickle
import queue
from concurrent.futures import ThreadPoolExecutor
from torch.futures import Future
from torch.distributed.checkpoint.filesystem import CURRENT_DCP_VERSION, FileSystemWriter, _write_files_from_queue

class CheckedThreadFileSystemWriter(FileSystemWriter):
    """Join every writer and reject failed, missing or duplicate write results."""

    def _write_data(self, planner, file_queue):
        if self.thread_count == 1:
            return super()._write_data(planner, file_queue)
        with file_queue.mutex:
            expected = [item.index for _, _, items in file_queue.queue for item in items]
        result_queue = queue.Queue()

        def work():
            _write_files_from_queue(create_stream=self.fs.create_stream, file_queue=file_queue, result_queue=result_queue, planner=planner, transforms=self.transforms, inflight_threshhold=self.per_thread_copy_ahead, use_fsync=self.sync_files, thread_count=self.thread_count, serialization_format=self.serialization_format)
        errors = []
        with ThreadPoolExecutor(max_workers=self.thread_count - 1) as pool:
            futures = [pool.submit(work) for _ in range(self.thread_count - 1)]
            try:
                work()
            except BaseException as error:
                errors.append(error)
            for future in futures:
                try:
                    future.result()
                except BaseException as error:
                    errors.append(error)
        if errors:
            raise errors[0]
        results = []
        while True:
            try:
                results.extend(result_queue.get_nowait())
            except queue.Empty:
                break
        if len(results) != len(expected) or {result.index for result in results} != set(expected):
            raise RuntimeError('checkpoint writer results missing or duplicated')
        future = Future()
        future.set_result(results)
        return future

class NoRenameFileSystemWriter(CheckedThreadFileSystemWriter):
    """Create final DCP metadata once, without a temp-file rename."""

    def finish(self, metadata, results) -> None:
        metadata = dataclasses.replace(metadata, version=CURRENT_DCP_VERSION)
        storage_md = {}
        for writes in results:
            storage_md.update({write.index: write.storage_data for write in writes})
        metadata.storage_data = storage_md
        metadata.storage_meta = self.storage_meta()
        metadata_path = self._get_metadata_path(self.rank if not self.use_collectives and self.rank is not None else None)
        with self.fs.create_stream(metadata_path, 'xb') as stream:
            pickle.dump(metadata, stream)
            stream.flush()
            if self.sync_files:
                os.fsync(stream.fileno())
from typing import Any
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

class VeOmniDCPBridge:
    """Commit model/optimizer DCP plus rank-local scheduler/RNG under a manifest."""

    def __init__(self, *, module: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, topology_fingerprint: str, model_fingerprint: str, shared_storage: bool=False, checkpoint_process_group: Any=None, logical_checkpoint_fingerprint: str | None=None, file_hash_workers: int=1, file_write_workers: int=1) -> None:
        if type(file_hash_workers) is not int or not 1 <= file_hash_workers <= 4:
            raise ValueError('checkpoint file hash workers must be an integer from 1 to 4')
        self.file_hash_workers = file_hash_workers
        if type(file_write_workers) is not int or not 1 <= file_write_workers <= 4:
            raise ValueError('checkpoint file write workers must be an integer from 1 to 4')
        self.file_write_workers = file_write_workers
        if not dist.is_initialized():
            raise RuntimeError('DCP bridge requires an initialized process group')
        if not topology_fingerprint or not model_fingerprint:
            raise ValueError('topology and model fingerprints are required')
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.topology_fingerprint = topology_fingerprint
        self.model_fingerprint = model_fingerprint
        self.shared_storage = shared_storage
        self.checkpoint_process_group = checkpoint_process_group
        if logical_checkpoint_fingerprint is not None and (not isinstance(logical_checkpoint_fingerprint, str) or len(logical_checkpoint_fingerprint) != 64 or any((c not in '0123456789abcdef' for c in logical_checkpoint_fingerprint))):
            raise ValueError('logical checkpoint fingerprint must be SHA-256')
        self.logical_checkpoint_fingerprint = logical_checkpoint_fingerprint

    def save_candidate(self, path: str | os.PathLike[str], *, generation: int) -> CandidateCheckpoint:
        if generation < 0:
            raise ValueError('checkpoint generation must be non-negative')
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'manifest.json').exists():
            raise FileExistsError(f'candidate checkpoint already complete: {root}')
        model_state, optim_state = get_state_dict(self.module, self.optimizer)
        write_started = time.perf_counter()
        if self.shared_storage:
            dcp.save({'model': model_state, 'optimizer': optim_state}, storage_writer=NoRenameFileSystemWriter(root / 'dcp', thread_count=self.file_write_workers), process_group=self.checkpoint_process_group)
        elif self.file_write_workers != 1:
            dcp.save({'model': model_state, 'optimizer': optim_state}, storage_writer=CheckedThreadFileSystemWriter(root / 'dcp', thread_count=self.file_write_workers), process_group=self.checkpoint_process_group)
        else:
            dcp.save({'model': model_state, 'optimizer': optim_state}, checkpoint_id=root / 'dcp', process_group=self.checkpoint_process_group)
        rank = dist.get_rank()
        if self.file_write_workers != 1:
            print('checkpoint_file_write_timing ' + json.dumps(dict(generation=generation, rank=rank, workers=self.file_write_workers, save_seconds=time.perf_counter() - write_started)), flush=True)
        error = None
        try:
            extra = {'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None, 'cpu_rng': torch.get_rng_state(), 'npu_rng': torch.npu.get_rng_state() if hasattr(torch, 'npu') else None}
            extra_path = root / f'extra-rank-{rank:04d}.pt'
            torch.save(extra, extra_path)
            with extra_path.open('rb+') as handle:
                os.fsync(handle.fileno())
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'checkpoint rank-local state was rejected by another rank')
        error = None
        try:
            hash_started = time.perf_counter()
            files = veomni_dcp_bridge_checkpoint_file_hashes_with_progress(root, workers=self.file_hash_workers, group=self.checkpoint_process_group)
            if rank == 0:
                if self.file_hash_workers != 1:
                    print('checkpoint_file_hash_timing ' + json.dumps(dict(generation=generation, workers=self.file_hash_workers, files=len(files), hash_seconds=time.perf_counter() - hash_started)), flush=True)
                manifest = {'schema_version': 1, 'generation': generation, 'world_size': dist.get_world_size(), 'topology_fingerprint': self.topology_fingerprint, 'model_fingerprint': self.model_fingerprint, 'files': files}
                if self.logical_checkpoint_fingerprint is not None:
                    manifest['logical_checkpoint_fingerprint'] = self.logical_checkpoint_fingerprint
                if self.shared_storage:
                    veomni_dcp_bridge_exclusive_json(root / 'manifest.json', manifest)
                else:
                    veomni_dcp_bridge_atomic_json(root / 'manifest.json', manifest)
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'checkpoint manifest publication was rejected by another rank')
        dist.barrier()
        digest = veomni_dcp_bridge_sha256_file(root / 'manifest.json')
        candidate = CandidateCheckpoint(str(root), digest, generation)
        candidate.validate()
        return candidate

    def _require_all_ranks_success(self, error: Exception | None, rejected_message: str) -> None:
        """Finish the small collective before a rank-local save/load failure escapes."""
        group = self.checkpoint_process_group
        if dist.get_world_size(group) > 1:
            backend = str(dist.get_backend(group)).lower()
            if backend == 'hccl':
                device = f'npu:{torch.npu.current_device()}'
            elif backend == 'nccl':
                device = f'cuda:{torch.cuda.current_device()}'
            else:
                device = 'cpu'
            verified = torch.tensor([int(error is None)], dtype=torch.int32, device=device)
            dist.all_reduce(verified, op=dist.ReduceOp.MIN, group=group)
            all_verified = bool(verified.item())
        else:
            all_verified = error is None
        if not all_verified:
            if error is not None:
                raise error
            raise ValueError(rejected_message)

    def _verified_candidate(self, candidate: CandidateCheckpoint, *, handoff_generation: int | None=None) -> Path:
        candidate.validate()
        from verl_speco.runtime.checkpoint_policy import is_live_candidate
        if is_live_candidate(candidate):
            raise ValueError('live optimizer progress cannot restore model/optimizer; use the latest full checkpoint')
        root = Path(candidate.uri)
        manifest_path = root / 'manifest.json'
        if veomni_dcp_bridge_sha256_file(manifest_path) != candidate.checksum:
            raise ValueError('candidate manifest checksum mismatch')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or manifest.get('generation') != candidate.checkpoint_generation or manifest.get('world_size') != dist.get_world_size() or (manifest.get('model_fingerprint') != self.model_fingerprint):
            raise ValueError('candidate checkpoint manifest is incompatible')
        same_topology = manifest.get('topology_fingerprint') == self.topology_fingerprint
        saved_logical = manifest.get('logical_checkpoint_fingerprint')
        if handoff_generation is not None:
            if type(handoff_generation) is not int or handoff_generation <= candidate.checkpoint_generation or self.logical_checkpoint_fingerprint is None or (saved_logical != self.logical_checkpoint_fingerprint):
                raise ValueError('checkpoint generation handoff is incompatible')
        elif not same_topology:
            raise ValueError('candidate checkpoint manifest is incompatible')
        if saved_logical is not None and self.logical_checkpoint_fingerprint is not None and (saved_logical != self.logical_checkpoint_fingerprint):
            raise ValueError('logical checkpoint identity changed')
        files = manifest.get('files')
        if not isinstance(files, dict) or not files:
            raise ValueError('candidate checkpoint manifest has no files')
        for name, expected in files.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or (not isinstance(expected, str)):
                raise ValueError('candidate checkpoint manifest has an invalid file entry')
            if veomni_dcp_bridge_sha256_file(root / relative) != expected:
                raise ValueError(f'candidate checkpoint file checksum mismatch: {name}')
        return root

    def restore_candidate(self, candidate: CandidateCheckpoint, *, handoff_generation: int | None=None) -> None:
        root = None
        error = None
        try:
            root = self._verified_candidate(candidate, handoff_generation=handoff_generation)
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'candidate checkpoint verification was rejected by another rank')
        assert root is not None
        model_state, optim_state = get_state_dict(self.module, self.optimizer)
        state = {'model': model_state, 'optimizer': optim_state}
        dcp.load(state, checkpoint_id=root / 'dcp', process_group=self.checkpoint_process_group)
        set_state_dict(self.module, self.optimizer, model_state_dict=state['model'], optim_state_dict=state['optimizer'])
        extra = torch.load(root / f'extra-rank-{dist.get_rank():04d}.pt', map_location='cpu', weights_only=True)
        if self.scheduler is not None and extra['scheduler'] is not None:
            self.scheduler.load_state_dict(extra['scheduler'])
        torch.set_rng_state(extra['cpu_rng'])
        if extra['npu_rng'] is not None:
            torch.npu.set_rng_state(extra['npu_rng'])
        dist.barrier()

def veomni_dcp_bridge_checkpoint_file_hashes_with_progress(root: Path, *, workers: int, group: Any=None, poll_seconds: float=1.0) -> dict[str, str] | None:
    """Keep peers communicating while rank zero hashes the complete checkpoint.

    Hashing runs in an IO thread; collectives remain on the calling thread.
    Completion never substitutes for the subsequent all-rank success gate.
    """
    if poll_seconds <= 0:
        raise ValueError('checkpoint hash polling interval must be positive')
    rank = dist.get_rank()
    if dist.get_world_size(group) == 1:
        return veomni_dcp_bridge_checkpoint_file_hashes(root, workers=workers) if rank == 0 else None
    backend = str(dist.get_backend(group)).lower()
    device = f'npu:{torch.npu.current_device()}' if backend == 'hccl' else f'cuda:{torch.cuda.current_device()}' if backend == 'nccl' else 'cpu'
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(veomni_dcp_bridge_checkpoint_file_hashes, root, workers=workers) if rank == 0 else None
        while True:
            status = 0
            if pending is not None and pending.done():
                status = -1 if pending.exception() is not None else 1
            progress = torch.tensor([status], dtype=torch.int32, device=device)
            dist.broadcast(progress, src=0, group=group)
            status = int(progress.item())
            if status:
                if rank == 0:
                    return pending.result()
                return None
            time.sleep(poll_seconds)

def veomni_dcp_bridge_checkpoint_file_hashes(root: Path, *, workers: int=1) -> dict[str, str]:
    """Hash every finalized file; retain deterministic order and join on failure."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('checkpoint file hash workers must be an integer from 1 to 4')
    paths = [file for file in sorted(root.rglob('*')) if file.is_file() and file.name != 'manifest.json']
    if workers == 1:
        hashes = list(map(veomni_dcp_bridge_sha256_file, paths))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(veomni_dcp_bridge_sha256_file, paths))
    return {str(file.relative_to(root)): digest for file, digest in zip(paths, hashes)}

def veomni_dcp_bridge_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def veomni_dcp_bridge_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix='.manifest-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        from verl_speco.runtime.artifacts import sync_directory
        sync_directory(path.parent, allow_unsupported=False)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

def veomni_dcp_bridge_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a fresh shared-filesystem manifest without unsupported rename."""
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        handle.flush()
        os.fsync(handle.fileno())
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_model_state_dict, set_optimizer_state_dict

class MultiNodeActorDCPBridge(VeOmniDCPBridge):

    def restore_candidate(self, candidate, *, handoff_generation=None):
        root = self._verified_candidate(candidate, handoff_generation=handoff_generation)
        error = None
        metadata = None
        saved_parameters = set()
        try:
            metadata = dcp.FileSystemReader(root / 'dcp').read_metadata()
            paths = metadata.planner_data
            if not isinstance(paths, dict) or set(paths) != set(metadata.state_dict_metadata):
                raise ValueError('actor DCP lacks the complete saved state path schema')
            named = dict(self.module.named_parameters())
            optimizer_parameters = {id(param) for group in self.optimizer.param_groups for param in group['params']}
            eligible = {name for name, param in named.items() if id(param) in optimizer_parameters}
            for path in paths.values():
                if not isinstance(path, tuple) or not path:
                    raise ValueError('actor DCP has an invalid saved state path')
                if len(path) >= 2 and path[:2] == ('optimizer', 'state'):
                    if len(path) < 4 or path[2] not in eligible:
                        raise ValueError('actor DCP has a foreign optimizer parameter state')
                    saved_parameters.add(path[2])
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'actor DCP state schema rejected by another rank')
        state = None
        error = None
        try:
            model, optimizer = get_state_dict(self.module, self.optimizer)
            initialized = optimizer['state']
            if not saved_parameters.issubset(initialized):
                raise ValueError('actor DCP has an unsupported optimizer state schema')
            optimizer['state'] = {name: value for name, value in initialized.items() if name in saved_parameters}
            state = {'model': model, 'optimizer': optimizer}
            planner = dcp.DefaultLoadPlanner()
            planner.set_up_planner(state, metadata, is_coordinator=False)
            if set(planner.state_dict) != set(metadata.state_dict_metadata):
                raise ValueError('actor DCP model/optimizer fields differ from the complete saved schema')
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'actor DCP restore template rejected by another rank')
        reader = actor_dcp_ActorProgressFileSystemReader(root / 'dcp', group=self.checkpoint_process_group)
        dcp.load(state, storage_reader=reader, process_group=self.checkpoint_process_group)
        set_model_state_dict(self.module, state['model'])
        set_optimizer_state_dict(self.module, self.optimizer, state['optimizer'], options=StateDictOptions(strict=False))
        error = None
        try:
            extra = torch.load(root / f'extra-rank-{dist.get_rank():04d}.pt', map_location='cpu', weights_only=True)
            if self.scheduler is not None and extra['scheduler'] is not None:
                self.scheduler.load_state_dict(extra['scheduler'])
            torch.set_rng_state(extra['cpu_rng'])
            if extra['npu_rng'] is not None:
                torch.npu.set_rng_state(extra['npu_rng'])
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'actor DCP rank-local restore rejected by another rank')
        dist.barrier()

    @classmethod
    def collective_create(cls, settings_factory, *, checkpoint_process_group=None):
        """Propagate even rank-local config/read errors before DCP collectives."""
        bridge = cls.__new__(cls)
        bridge.checkpoint_process_group = checkpoint_process_group
        error = None
        try:
            settings = dict(settings_factory())
            if 'checkpoint_process_group' in settings and settings['checkpoint_process_group'] is not checkpoint_process_group:
                raise ValueError('actor checkpoint factory changed the collective group')
            settings['checkpoint_process_group'] = checkpoint_process_group
            bridge.__init__(**settings)
        except Exception as exc:
            error = exc
        bridge._require_all_ranks_success(error, 'actor checkpoint factory rejected by another rank')
        return bridge

    def _verified_candidate(self, candidate, *, handoff_generation=None):
        """Agree on identity, then hash each file once before any DCP mutation."""
        error = None
        root = manifest = None
        try:
            candidate.validate()
            root = Path(candidate.uri)
            manifest_path = root / 'manifest.json'
            if veomni_dcp_bridge_sha256_file(manifest_path) != candidate.checksum:
                raise ValueError('candidate manifest checksum mismatch')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('schema_version') != 1 or manifest.get('generation') != candidate.checkpoint_generation or manifest.get('world_size') != dist.get_world_size() or (manifest.get('model_fingerprint') != self.model_fingerprint):
                raise ValueError('candidate checkpoint manifest is incompatible')
            saved_logical = manifest.get('logical_checkpoint_fingerprint')
            if handoff_generation is not None:
                if type(handoff_generation) is not int or handoff_generation <= candidate.checkpoint_generation or self.logical_checkpoint_fingerprint is None or (saved_logical != self.logical_checkpoint_fingerprint):
                    raise ValueError('checkpoint generation handoff is incompatible')
            elif manifest.get('topology_fingerprint') != self.topology_fingerprint:
                raise ValueError('candidate checkpoint manifest is incompatible')
            if saved_logical is not None and self.logical_checkpoint_fingerprint is not None and (saved_logical != self.logical_checkpoint_fingerprint):
                raise ValueError('logical checkpoint identity changed')
            files = manifest.get('files')
            if not isinstance(files, dict) or not files:
                raise ValueError('candidate checkpoint manifest has no files')
            required = {'dcp/.metadata'} | {f'extra-rank-{rank:04d}.pt' for rank in range(dist.get_world_size())}
            if not required.issubset(files):
                raise ValueError('actor checkpoint manifest omits DCP metadata or rank-local state')
            actual_files = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and path.name != 'manifest.json'}
            if actual_files != set(files):
                raise ValueError('actor checkpoint files differ from its complete manifest')
            for name, expected in files.items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts or (not isinstance(expected, str)) or (len(expected) != 64) or any((c not in '0123456789abcdef' for c in expected)):
                    raise ValueError('candidate checkpoint manifest has an invalid file entry')
                if not (root / relative).resolve().is_relative_to(root.resolve()):
                    raise ValueError('actor checkpoint file escapes its owned directory')
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'actor checkpoint header rejected by another rank')
        identity = (str(root), candidate.checksum, candidate.checkpoint_generation, handoff_generation)
        group = self.checkpoint_process_group
        identities = [None] * dist.get_world_size(group)
        dist.all_gather_object(identities, identity, group=group)
        if any((value != identity for value in identities)):
            raise ValueError('actor ranks requested different checkpoint identities')
        names = sorted(manifest['files'])
        assigned = names[dist.get_rank(group)::dist.get_world_size(group)]
        error = None
        try:
            hashes = actor_dcp_assigned_checkpoint_file_hashes_with_progress([root / name for name in assigned], workers=self.file_hash_workers, group=group)
            for name, actual in zip(assigned, hashes):
                if actual != manifest['files'][name]:
                    raise ValueError(f'candidate checkpoint file checksum mismatch: {name}')
        except Exception as exc:
            error = exc
        self._require_all_ranks_success(error, 'actor checkpoint payload rejected by another rank')
        self.last_verification = dict(manifest_sha256=candidate.checksum, global_file_count=len(names), assigned_file_count=len(assigned), all_files_collectively_verified=True)
        return root

def actor_dcp_assigned_checkpoint_file_hashes_with_progress(paths, *, workers, group=None, poll_seconds=1.0):
    """Hash assigned files once while every rank participates in progress.

    A local IO error stays pending until all other hash futures finish. The
    caller then executes its original all-rank success gate before DCP load.
    Collectives run only on the calling thread, never on the IO workers.
    """
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('checkpoint file hash workers must be an integer from 1 to 4')
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)) or (not 0 < poll_seconds <= 1):
        raise ValueError('checkpoint hash polling interval must be positive and at most one second')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(veomni_dcp_bridge_sha256_file, path) for path in paths]
        if dist.get_world_size(group) > 1:
            backend = str(dist.get_backend(group)).lower()
            device = f'npu:{torch.npu.current_device()}' if backend == 'hccl' else f'cuda:{torch.cuda.current_device()}' if backend == 'nccl' else 'cpu'
            progress = torch.zeros(1, dtype=torch.int32, device=device)
            while True:
                progress.fill_(int(all((future.done() for future in pending))))
                dist.all_reduce(progress, op=dist.ReduceOp.MIN, group=group)
                if int(progress.item()):
                    break
                time.sleep(poll_seconds)
        return [future.result() for future in pending]

class actor_dcp_ActorProgressFileSystemReader(dcp.FileSystemReader):
    """Keep the read-stage collective live while each rank reads its own plan.

    Torch's filesystem reader performs local storage IO and tensor copies; it
    does not issue collectives. Run only that phase on one worker, preserving
    its request order and original Future/error. All progress collectives stay
    on the DCP calling thread. Do not move the whole dcp.load to a worker.
    """

    def __init__(self, path, *, group=None, poll_seconds=1.0):
        super().__init__(path)
        if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)) or (not 0 < poll_seconds <= 1):
            raise ValueError('actor DCP read polling interval must be positive and at most one second')
        self._progress_group = group
        self._poll_seconds = poll_seconds

    def read_data(self, plan, planner):
        initializer = None
        npu = getattr(torch, 'npu', None)
        if npu is not None and getattr(npu, 'is_available', lambda: False)():
            device = npu.current_device()
            initializer = lambda: npu.set_device(device)
        elif torch.cuda.is_available():
            device = torch.cuda.current_device()
            initializer = lambda: torch.cuda.set_device(device)
        with ThreadPoolExecutor(max_workers=1, initializer=initializer) as pool:
            work = pool.submit(super().read_data, plan, planner)
            group = self._progress_group
            if dist.get_world_size(group) > 1:
                backend = str(dist.get_backend(group)).lower()
                target = f'npu:{torch.npu.current_device()}' if backend == 'hccl' else f'cuda:{torch.cuda.current_device()}' if backend == 'nccl' else 'cpu'
                progress = torch.zeros(1, dtype=torch.int32, device=target)
                while True:
                    progress.fill_(int(work.done()))
                    dist.all_reduce(progress, op=dist.ReduceOp.MIN, group=group)
                    if int(progress.item()):
                        break
                    time.sleep(self._poll_seconds)
            return work.result()

def actor_state_sha256(*, module, optimizer, scheduler, chunk_bytes=16 * 1024 ** 2, include_components=False):
    """Hash complete local model/Adam/scheduler/RNG bytes with bounded copies.

    Local DTensor shards are hashed without collecting the full actor. Comparing
    these receipts on all ranks is a complete state check, not a sampled probe.
    The caller must fence optimizer updates and use the same logical rank mesh.
    """
    if type(chunk_bytes) is not int or chunk_bytes < 8:
        raise ValueError('actor state hash chunk must be at least eight bytes')
    digest = hashlib.sha256()
    count = nbytes = 0
    component = None
    components = {}

    def update(data):
        digest.update(data)
        if component is not None:
            component.update(data)

    def visit(value):
        nonlocal count, nbytes, component
        if isinstance(value, torch.Tensor):
            local = value.to_local() if hasattr(value, 'to_local') else value
            local = local.detach()
            update(json.dumps(dict(tensor=True, shape=list(value.shape), local_shape=list(local.shape), dtype=str(value.dtype)), sort_keys=True).encode())
            flat = local.reshape(-1)
            size = max(1, chunk_bytes // local.element_size())
            for start in range(0, flat.numel(), size):
                chunk = flat[start:start + size].cpu().contiguous()
                if chunk.is_floating_point() and (not torch.isfinite(chunk).all()):
                    raise ValueError('actor checkpoint state contains nonfinite values')
                update(memoryview(chunk.view(torch.uint8).numpy()))
            count += 1
            nbytes += local.numel() * local.element_size()
        elif isinstance(value, dict):
            update(b'dict[')
            ordered = sorted(value.items(), key=lambda pair: json.dumps([type(pair[0]).__name__, pair[0]], sort_keys=True, allow_nan=False))
            for key, item in ordered:
                visit(key)
                if include_components and value is state:
                    component = hashlib.sha256()
                    before_count, before_bytes = (count, nbytes)
                    visit(item)
                    components[key] = dict(sha256=component.hexdigest(), tensor_count=count - before_count, local_tensor_nbytes=nbytes - before_bytes)
                    component = None
                else:
                    visit(item)
            update(b']')
        elif isinstance(value, (list, tuple)):
            update(b'list[' if isinstance(value, list) else b'tuple[')
            for item in value:
                visit(item)
            update(b']')
        elif value is None or type(value) in (str, int, float, bool):
            update(json.dumps([type(value).__name__, value], allow_nan=False).encode())
        else:
            raise TypeError(f'unsupported actor checkpoint state type: {type(value).__name__}')
    state = None
    try:
        state = dict(model=module.state_dict(), optimizer=optimizer.state_dict(), scheduler=None if scheduler is None else scheduler.state_dict(), cpu_rng=torch.get_rng_state())
        if hasattr(torch, 'npu') and torch.npu.is_available():
            state['npu_rng'] = torch.npu.get_rng_state()
        visit(state)
        result = dict(state_sha256=digest.hexdigest(), tensor_count=count, local_tensor_nbytes=nbytes, complete_local_model_optimizer_scheduler_rng=True)
        if include_components:
            result['state_components'] = components
        return result
    finally:
        state = None
        visit = None

def recover_empty_adam_representation(optimizer, *, observed_state, expected_sha256, measure_state):
    if observed_state['state_sha256'] == expected_sha256:
        return (observed_state, 0)
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('complete-state empty entry recovery requires the actual AdamW optimizer')
    added = []
    accepted = False
    try:
        for group in optimizer.param_groups:
            for param in group['params']:
                if param not in optimizer.state:
                    optimizer.state[param] = {}
                    added.append(param)
        if not added:
            return (observed_state, 0)
        measured = measure_state()
        if measured['state_sha256'] != expected_sha256 or measured.get('complete_local_model_optimizer_scheduler_rng') is not True or measured['tensor_count'] != observed_state['tensor_count'] or (measured['local_tensor_nbytes'] != observed_state['local_tensor_nbytes']):
            return (observed_state, 0)
        accepted = True
        return (dict(observed_state, **measured), len(added))
    finally:
        if not accepted:
            for param in added:
                del optimizer.state[param]
from typing import Any, Mapping
from safetensors.torch import load_file, save_file
from verl_speco.runtime.artifacts import publish_json, read_published_json, sync_directory

@dataclass(frozen=True)
class DraftParameterSnapshot:
    uri: str
    checksum: str
    draft_revision: int

def trainable_snapshot_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()

def trainable_snapshot_parameters(module, prefix: str):
    parameters = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith(prefix):
            raise ValueError('trainable parameter is outside the declared namespace')
        exported = name[len(prefix):]
        if not exported or exported in parameters:
            raise ValueError('snapshot parameter name mapping is ambiguous')
        local = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
        if tuple(local.shape) != tuple(parameter.shape):
            raise ValueError('draft snapshot requires full replicated parameters')
        parameters[exported] = (name, local)
    if not parameters:
        raise ValueError('draft snapshot has no trainable parameters')
    return parameters

def trainable_snapshot_parameter_checksum(parameters, weights=None) -> str:
    digest = hashlib.sha256()
    for exported, (source_name, parameter) in sorted(parameters.items(), key=lambda item: item[1][0]):
        tensor = weights[exported] if weights is not None else parameter.detach().cpu().contiguous()
        digest.update(source_name.encode())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()

def save_trainable_snapshot(module, directory: Path, *, draft_revision: int, generation: int, source_step_id: str, target_model_fingerprint: str, tokenizer_fingerprint: str, model_config: Mapping[str, Any], namespace_prefix: str='') -> DraftParameterSnapshot:
    """Save once after a committed step; callers own rank/barrier coordination."""
    if type(draft_revision) is not int or draft_revision < 1 or type(generation) is not int or (generation < 1):
        raise ValueError('snapshot revision and generation must be positive integers')
    if not all((isinstance(value, str) and value.strip() for value in (source_step_id, target_model_fingerprint, tokenizer_fingerprint))):
        raise ValueError('snapshot requires committed step and target/tokenizer identities')
    config = json.loads(json.dumps(dict(model_config), allow_nan=False))
    trainable_snapshot_checksum(config)
    parameters = trainable_snapshot_parameters(module, namespace_prefix)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    weights = {name: local.detach().cpu().contiguous().clone() for name, (_, local) in parameters.items()}
    path = directory / 'trainable.safetensors'
    save_file(weights, str(path))
    with path.open('rb+') as handle:
        os.fsync(handle.fileno())
        weight_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest = {'schema_version': 1, 'kind': 'draft_trainable_parameters', 'draft_revision': draft_revision, 'generation': generation, 'source_step_id': source_step_id, 'target_model_fingerprint': target_model_fingerprint, 'tokenizer_fingerprint': tokenizer_fingerprint, 'model_config': config, 'namespace_prefix': namespace_prefix, 'weights_file': path.name, 'weights_sha256': weight_sha, 'trainable_parameter_sha256': trainable_snapshot_parameter_checksum(parameters, weights), 'parameters': {name: {'source_name': parameters[name][0], 'shape': list(weight.shape), 'dtype': str(weight.dtype)} for name, weight in weights.items()}}
    publish_json(directory / 'manifest.json', manifest)
    sync_directory(directory)
    return DraftParameterSnapshot(str(directory), trainable_snapshot_checksum(manifest), draft_revision)

def load_trainable_snapshot(module, snapshot: DraftParameterSnapshot, *, target_model_fingerprint: str, tokenizer_fingerprint: str, model_config: Mapping[str, Any], namespace_prefix: str='') -> dict:
    """Validate every binding and tensor before changing any model parameter."""
    directory = Path(snapshot.uri)
    manifest = read_published_json(directory / 'manifest.json')
    if trainable_snapshot_checksum(manifest) != snapshot.checksum or manifest.get('draft_revision') != snapshot.draft_revision or manifest.get('schema_version') != 1 or (manifest.get('kind') != 'draft_trainable_parameters'):
        raise ValueError('draft snapshot manifest identity differs')
    if manifest.get('target_model_fingerprint') != target_model_fingerprint or manifest.get('tokenizer_fingerprint') != tokenizer_fingerprint or manifest.get('model_config') != json.loads(json.dumps(dict(model_config), allow_nan=False)) or (manifest.get('namespace_prefix') != namespace_prefix):
        raise ValueError('draft snapshot target, tokenizer, geometry or namespace differs')
    if manifest.get('weights_file') != 'trainable.safetensors':
        raise ValueError('draft snapshot weight filename differs')
    path = directory / manifest['weights_file']
    with path.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != manifest['weights_sha256']:
            raise ValueError('draft snapshot weight checksum differs')
    weights = load_file(str(path), device='cpu')
    parameters = trainable_snapshot_parameters(module, namespace_prefix)
    if set(weights) != set(parameters) or set(weights) != set(manifest['parameters']):
        raise ValueError('draft snapshot trainable parameter set differs')
    for name, tensor in weights.items():
        source_name, parameter = parameters[name]
        binding = manifest['parameters'][name]
        if binding != {'source_name': source_name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)} or tuple(tensor.shape) != tuple(parameter.shape) or tensor.dtype != parameter.dtype:
            raise ValueError('draft snapshot tensor binding, shape or dtype differs')
    with torch.no_grad():
        for name, tensor in weights.items():
            parameters[name][1].copy_(tensor)
    if trainable_snapshot_parameter_checksum(parameters) != manifest['trainable_parameter_sha256']:
        raise RuntimeError('loaded draft parameter checksum differs before optimizer use')
    return manifest
from contextlib import contextmanager

def draft_step_seed(base_seed, sample_index, rank):
    if type(base_seed) is not int or not 0 <= base_seed < 2 ** 63 or type(sample_index) is not int or (sample_index < 0) or (type(rank) is not int) or (not 0 <= rank < 128):
        raise ValueError('invalid draft seed, stream position or DP rank')
    return (base_seed + 128 * sample_index + rank) % 2 ** 63

@contextmanager
def draft_training_rng(base_seed, *, sample_index, rank, device):
    if base_seed is None:
        yield None
        return
    seed = draft_step_seed(base_seed, sample_index, rank)
    device = torch.device(device)
    if device.type not in ('cpu', 'npu') or (device.type == 'npu' and device.index is None):
        raise ValueError('draft RNG requires CPU or an explicit NPU device')
    devices = [device.index] if device.type == 'npu' else []
    with torch.random.fork_rng(devices=devices, device_type=device.type):
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'npu':
            torch.npu.manual_seed(seed)
        yield seed
