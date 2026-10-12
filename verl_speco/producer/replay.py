# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations

class ForwardReplayCursor:
    """Read preceding JSONL records once; retain the latest request or batch.

    Callers must revalidate the immutable input fingerprint before selection.
    Repeating the latest index is allowed; rewinding or changing identity is not.
    """

    def __init__(self, factory, *, identity):
        if not callable(factory) or not isinstance(identity, str) or len(identity) != 64 or any((c not in '0123456789abcdef' for c in identity)):
            raise ValueError('invalid replay cursor identity')
        self._factory = factory
        self._identity = identity
        self._iterator = None
        self._next_index = 0
        self._last_index = -1
        self._last = None
        self._exhausted = False
        self._closed = False
        self._failed = False
        self._last_batch_indices = None
        self._last_batch = None

    def select(self, index, *, identity):
        if self._closed or self._failed:
            raise RuntimeError('replay cursor is closed or failed')
        if identity != self._identity:
            raise ValueError('replay cursor input identity changed')
        if type(index) is not int or not 0 <= index < 2 ** 63:
            raise ValueError('invalid replay cursor index')
        if index < self._last_index:
            raise ValueError('replay cursor cannot rewind')
        if index == self._last_index:
            return self._last
        if self._exhausted:
            return None
        self._last_batch_indices = self._last_batch = None
        try:
            if self._iterator is None:
                self._iterator = iter(self._factory())
            while self._next_index <= index:
                item = next(self._iterator)
                if item is None:
                    raise ValueError('replay iterator returned an empty request')
                position = self._next_index
                self._next_index += 1
                if position == index:
                    self._last_index, self._last = (index, item)
                    return item
        except StopIteration:
            self._exhausted = True
            return None
        except BaseException:
            self._failed = True
            raise

    def select_many(self, indices, *, identity):
        """Select at most 16 ordered producer positions as one complete batch.

        Only an exact repeat of the latest batch can use the bounded cache.
        An incomplete batch poisons this owner so no partial prefill is retried.
        """
        if self._closed or self._failed:
            raise RuntimeError('replay cursor is closed or failed')
        if identity != self._identity:
            raise ValueError('replay cursor input identity changed')
        if not isinstance(indices, (list, tuple)) or not 1 <= len(indices) <= 16 or any((type(i) is not int or not 0 <= i < 2 ** 63 for i in indices)) or any((a >= b for a, b in zip(indices, indices[1:]))):
            raise ValueError('replay batch needs 1..16 strictly increasing indices')
        indices = tuple(indices)
        if indices == self._last_batch_indices:
            return self._last_batch
        if indices[0] < self._last_index:
            raise ValueError('replay cursor cannot rewind')
        try:
            requests = []
            for index in indices:
                request = self.select(index, identity=identity)
                if request is None:
                    raise ValueError('physical replay has too few committed sequences')
                requests.append(request)
            self._last_batch_indices = indices
            self._last_batch = tuple(requests)
            return self._last_batch
        except BaseException:
            self._last_batch_indices = self._last_batch = None
            self._failed = True
            raise

    def close(self):
        self._closed = True
        self._last = None
        self._last_batch_indices = self._last_batch = None
        if self._iterator is not None:
            close = getattr(self._iterator, 'close', None)
            if close is not None:
                close()
            self._iterator = None

class IndexedReplayCursor(ForwardReplayCursor):
    """Select only requested epoch occurrences from a validated frozen catalog.

    Reuse the existing 16-row cache, monotonic ownership and poison-on-partial
    failure contract. Sparse producer reads do not parse preceding producers'
    token payloads. Input/plan validation belongs to the catalog supplied by the
    caller; its reader must bind each request to the requested occurrence index.
    """

    def __init__(self, reader, *, identity, total_samples):
        if not callable(reader) or type(total_samples) is not int or (not 1 <= total_samples <= 6000000):
            raise ValueError('indexed replay requires a bounded catalog reader')
        super().__init__(lambda: (), identity=identity)
        self._reader = reader
        self._total_samples = total_samples

    def select(self, index, *, identity):
        if self._closed or self._failed:
            raise RuntimeError('replay cursor is closed or failed')
        if identity != self._identity:
            raise ValueError('replay cursor input identity changed')
        if type(index) is not int or not 0 <= index < 2 ** 63:
            raise ValueError('invalid replay cursor index')
        if index < self._last_index:
            raise ValueError('replay cursor cannot rewind')
        if index == self._last_index:
            return self._last
        if self._exhausted:
            return None
        self._last_batch_indices = self._last_batch = None
        if index >= self._total_samples:
            self._exhausted = True
            return None
        try:
            request = self._reader(index)
            if request is None or type(getattr(request, 'sequence_no', None)) is not int or request.sequence_no != index:
                raise ValueError('indexed replay occurrence differs from requested position')
            self._last_index, self._last = (index, request)
            return request
        except BaseException:
            self._failed = True
            raise

    def close(self):
        super().close()
        self._reader = None

def forward_replay_cursor_enabled(environment):
    policy = environment.get('SPECO_CAPTURE_REUSE_REPLAY_CURSOR', '')
    if policy not in ('', '0', '1'):
        raise ValueError('invalid replay cursor policy')
    if policy == '1' and (environment.get('SPECO_CAPTURE_REUSE_ENGINE') != '1' or environment.get('SPECO_CAPTURE_DATASET_REPLAY') != '1' or (not environment.get('SPECO_SEQUENCE_REPLAY_INPUT', '').strip())):
        raise ValueError('replay cursor requires persistent saved-sequence capture')
    return policy == '1'
import hashlib
import json
import logging
import os
import re
import threading
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping
import torch
from verl_speco.producer.input_reader import SampleFilteredError
from verl_speco.producer.input_reader import TokenizedRequest, _build_tokenized_request
sequence_replay_writer_SAFE_SHARD_ID = re.compile('[A-Za-z0-9][A-Za-z0-9_.-]*')
sequence_replay_writer_SAFE_RUN_ID = re.compile('[A-Za-z0-9][A-Za-z0-9_.:-]*')

def generation_replay_dir(output_root: str | os.PathLike[str], *, run_id: str, generation: int) -> Path:
    """Keep retries within one generation idempotent, but isolate rollbacks."""
    if not sequence_replay_writer_SAFE_RUN_ID.fullmatch(str(run_id)) or generation < 0:
        raise ValueError('replay run ID or generation is invalid')
    return Path(output_root) / str(run_id) / f'g{generation:06d}'

def sequence_replay_writer_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return os.fspath(value)
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, Mapping):
        return {str(key): sequence_replay_writer_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sequence_replay_writer_json_safe(item) for item in value]
    return str(value)

def sequence_replay_writer_shard_file_name(shard_id: str) -> str:
    value = str(shard_id).strip()
    if not sequence_replay_writer_SAFE_SHARD_ID.fullmatch(value):
        raise ValueError('sequence_replay.shard_id must match [A-Za-z0-9][A-Za-z0-9_.-]*')
    return f'part-{value}.jsonl'

def sequence_replay_writer_replay_fingerprint(payload: Mapping[str, Any]) -> str:
    identity = {'id': payload.get('id'), 'algorithm': payload.get('algorithm'), 'input_ids': payload.get('input_ids'), 'loss_mask': payload.get('loss_mask'), 'position_ids': payload.get('position_ids')}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def validate_sequence_replay_row(row, *, algorithm, target_model_id, target_model_revision, tokenizer_fingerprint, context):
    """Validate one complete saved row without allocating feature tensors."""
    if not isinstance(row, Mapping):
        raise ValueError(f'invalid sequence replay object at {context}')
    if row.get('schema_version') != 1 or row.get('sample_type') != 'token_replay' or row.get('algorithm') != algorithm.upper() or (not isinstance(row.get('metadata'), dict)):
        raise ValueError(f'invalid sequence replay schema at {context}')
    metadata = row['metadata']
    if metadata.get('commit_state') != 'sequence_only_generated' or metadata.get('target_generated') is not True or metadata.get('target_model_id') != target_model_id or (metadata.get('target_model_revision') != target_model_revision) or (metadata.get('tokenizer_fingerprint') != tokenizer_fingerprint):
        raise ValueError(f'sequence replay provenance differs at {context}')
    declared = row.get('sequence_fingerprint')
    if declared != f'sha256:{sequence_replay_writer_replay_fingerprint(row)}':
        raise ValueError(f'sequence replay fingerprint differs at {context}')
    ids, mask, positions = (row.get(key) for key in ('input_ids', 'loss_mask', 'position_ids'))
    if not isinstance(ids, list) or len(ids) < 2 or any((type(token) is not int or token < 0 for token in ids)) or (not isinstance(mask, list)) or (len(mask) != len(ids)) or any((value not in (0, 1) for value in mask)) or (not isinstance(positions, list)) or (positions != list(range(len(ids)))):
        raise ValueError(f'sequence replay tokens are invalid at {context}')
    try:
        prompt_length = mask.index(1)
    except ValueError as exc:
        raise ValueError(f'sequence replay has no supervised response at {context}') from exc
    if prompt_length < 1 or mask != [0] * prompt_length + [1] * (len(ids) - prompt_length):
        raise ValueError(f'sequence replay mask is not a response suffix at {context}')
    if not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError(f'sequence replay sample ID is invalid at {context}')
    return (ids, mask, positions, prompt_length, metadata, declared)

def iter_sequence_replay_prefill_requests(path: str | os.PathLike[str], *, config: Mapping[str, Any], algorithm: str, target_model_id: str, target_model_revision: str, tokenizer_fingerprint: str, on_filtered=None) -> Iterator[TokenizedRequest]:
    """Stream generated token shards into standalone full-prefill requests.

    The last sampled token is a label and has no corresponding prefill hidden
    row. Request sequence numbers are assigned deterministically across sorted
    shards; the original generation identity stays in source metadata.
    """
    policy = config.get('sequence_replay_on_filtered', 'raise')
    if policy not in ('raise', 'skip'):
        raise ValueError("sequence_replay_on_filtered must be 'raise' or 'skip'")
    if on_filtered is not None and (not callable(on_filtered)):
        raise TypeError('sequence replay filter observer must be callable')
    source = Path(path)
    from verl_speco.runtime.artifacts import published_json_paths, read_published_json
    manifests = published_json_paths(source, 'part-*.manifest.json') if source.is_dir() else [source] if source.name.endswith('.manifest.json') else []
    if manifests:
        root = source if source.is_dir() else source.parent
        files = []
        seen_files: set[str] = set()
        for manifest_path in manifests:
            manifest = read_published_json(manifest_path)
            if manifest.get('schema_version') != 1 or not isinstance(manifest.get('segments'), list):
                raise ValueError(f'invalid sequence replay manifest: {manifest_path}')
            for descriptor in manifest['segments']:
                name = descriptor.get('file')
                if not isinstance(name, str) or Path(name).name != name or name in seen_files:
                    raise ValueError(f'invalid or duplicate sequence segment: {manifest_path}')
                segment = root / name
                if hashlib.sha256(segment.read_bytes()).hexdigest() != descriptor.get('sha256'):
                    raise ValueError(f'sequence segment checksum differs: {segment}')
                seen_files.add(name)
                files.append(segment)
    elif source.is_dir():
        import re
        files = [item for item in sorted(source.glob('*.jsonl')) if not re.search('-segment-\\d{6}\\.jsonl$', item.name)]
    else:
        files = [source]
    if not files or any((not item.is_file() for item in files)):
        raise FileNotFoundError(f'sequence replay JSONL is unavailable: {source}')
    sequence_no = 0
    for file in files:
        with file.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                ids, mask, positions, prompt_length, metadata, declared = validate_sequence_replay_row(row, algorithm=algorithm, target_model_id=target_model_id, target_model_revision=target_model_revision, tokenizer_fingerprint=tokenizer_fingerprint, context=f'{file}:{line_number}')
                try:
                    request = _build_tokenized_request(sequence_no=sequence_no, sample_id=row['id'], prompt_length=prompt_length, full_ids=ids, source_metadata={'replay_file': str(file), 'replay_line': line_number, 'replay_run_id': row.get('run_id'), 'replay_shard_id': row.get('shard_id'), 'replay_sequence_no': row.get('sequence_no'), 'sequence_fingerprint': declared}, config=config, vllm_prompt_token_ids=ids[:-1], feature_end_limit=len(ids) - 1)
                except SampleFilteredError as exc:
                    if policy == 'raise':
                        raise
                    if on_filtered is not None:
                        on_filtered(exc)
                    logging.getLogger(__name__).warning('Skipping untrainable sequence replay id=%s original_sequence_no=%s file=%s:%s reason=%s', row['id'], row.get('sequence_no'), file, line_number, exc)
                    continue
                yield request
                sequence_no += 1

class JsonlSequenceReplayWriter:
    """Write one idempotent JSONL shard per Producer.

    Existing ``(run_id, sequence_no)`` records are indexed on open and skipped
    on retry. This prevents duplicate future-training rows when a Producer is
    restarted against the same shard. Different Producers must use distinct
    ``shard_id`` values; sharing a shard across processes is intentionally not
    supported.
    """

    def __init__(self, output_dir: str | os.PathLike[str], *, run_id: str, shard_id: str, fsync_interval: int=100) -> None:
        self.output_dir = Path(output_dir)
        file_name = sequence_replay_writer_shard_file_name(shard_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / file_name
        self.run_id = str(run_id)
        self.shard_id = str(shard_id)
        self.fsync_interval = int(fsync_interval)
        if not self.run_id:
            raise ValueError('sequence replay run_id must not be empty')
        if self.fsync_interval <= 0:
            raise ValueError('sequence_replay.fsync_interval must be positive')
        self._lock = threading.Lock()
        self._seen_ids: dict[tuple[str, int], str] = {}
        self._seen = self._load_seen_records()
        self._writes_since_fsync = 0
        self._file = self.path.open('a', encoding='utf-8', newline='\n')

    def _load_seen_records(self) -> dict[tuple[str, int], str]:
        seen: dict[tuple[str, int], str] = {}
        if not self.path.exists():
            return seen
        with self.path.open('r', encoding='utf-8') as replay_file:
            for line_number, line in enumerate(replay_file, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    key = (str(payload['run_id']), int(payload['sequence_no']))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid sequence replay record {self.path}:{line_number}') from exc
                if key in seen:
                    raise ValueError(f'Duplicate sequence replay record in existing shard: run_id={key[0]!r} sequence_no={key[1]}')
                fingerprint = sequence_replay_writer_replay_fingerprint(payload)
                declared = payload.get('sequence_fingerprint')
                if declared is not None and declared != f'sha256:{fingerprint}':
                    raise ValueError(f'Sequence replay fingerprint mismatch in existing shard: {self.path}:{line_number}')
                seen[key] = fingerprint
                self._seen_ids[key] = str(payload.get('id', ''))
        return seen

    @property
    def persisted_record_count(self) -> int:
        return len(self._seen)

    def has_record(self, *, sequence_no: int, sample_id: str) -> bool:
        """Skip a durable generation only when the input identity still agrees."""
        key = (self.run_id, int(sequence_no))
        with self._lock:
            existing = self._seen_ids.get(key)
            if existing is not None and existing != sample_id:
                raise ValueError('sequence replay input changed at an existing sequence number')
            return existing is not None

    def _input_identity(self, input_path: str | os.PathLike[str]) -> dict[str, object]:
        source = Path(input_path).resolve()
        stat = source.stat()
        return {'path': str(source), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

    def validate_progress_input(self, input_path: str | os.PathLike[str]) -> None:
        progress_path = self.path.with_suffix('.progress.json')
        if progress_path.exists():
            previous = json.loads(progress_path.read_text(encoding='utf-8'))
            if previous.get('run_id') != self.run_id or previous.get('shard_id') != self.shard_id or previous.get('input_fingerprint') != self._input_identity(input_path):
                raise ValueError('sequence replay progress input identity changed')

    def checkpoint_progress(self, *, input_path: str | os.PathLike[str], processed_count: int, complete: bool=False) -> Path:
        """Fsync rows before atomically publishing a per-shard progress record."""
        self.validate_progress_input(input_path)
        identity = self._input_identity(input_path)
        progress_path = self.path.with_suffix('.progress.json')
        with self._lock:
            if processed_count < len(self._seen):
                raise ValueError('sequence replay progress cannot trail persisted rows')
            self._file.flush()
            os.fsync(self._file.fileno())
            self._writes_since_fsync = 0
            payload = {'schema_version': 1, 'run_id': self.run_id, 'shard_id': self.shard_id, 'input_fingerprint': identity, 'processed_count': int(processed_count), 'persisted_count': len(self._seen), 'last_sequence_no': max((key[1] for key in self._seen), default=-1), 'replay_size_bytes': self.path.stat().st_size, 'complete': bool(complete)}
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=self.output_dir, prefix=f'.part-{self.shard_id}-progress-', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, progress_path)
        return progress_path

    def write(self, request: TokenizedRequest, *, algorithm: str, metadata: Mapping[str, Any] | None=None) -> bool:
        """Append a replay row, returning ``False`` when it already exists."""
        key = (self.run_id, int(request.sequence_no))
        input_ids = request.input_ids.detach().cpu().long().reshape(-1)
        loss_mask = request.loss_mask.detach().cpu().float().reshape(-1)
        position_ids = request.position_ids.detach().cpu().long().reshape(-1)
        if not input_ids.numel() == loss_mask.numel() == position_ids.numel():
            raise ValueError('sequence replay input_ids/loss_mask/position_ids length mismatch')
        row = {'schema_version': 1, 'sample_type': 'token_replay', 'id': request.sample_id, 'run_id': self.run_id, 'shard_id': self.shard_id, 'sequence_no': int(request.sequence_no), 'algorithm': str(algorithm).upper(), 'input_ids': input_ids.tolist(), 'loss_mask': loss_mask.tolist(), 'attention_mask': torch.ones_like(input_ids, dtype=torch.bool).tolist(), 'position_ids': position_ids.tolist(), 'feature_positions': request.feature_positions.detach().cpu().long().reshape(-1).tolist(), 'draft_position_ids': request.draft_position_ids.detach().cpu().long().reshape(-1).tolist(), 'metadata': {'source': 'standalone_target_generation', 'source_metadata': sequence_replay_writer_json_safe(request.source_metadata), **sequence_replay_writer_json_safe(dict(metadata or {}))}}
        fingerprint = sequence_replay_writer_replay_fingerprint(row)
        row['sequence_fingerprint'] = f'sha256:{fingerprint}'
        encoded = json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        with self._lock:
            if key in self._seen:
                if self._seen[key] != fingerprint:
                    raise ValueError(f'Conflicting sequence replay record for existing key: run_id={key[0]!r} sequence_no={key[1]}')
                return False
            self._file.write(encoded + '\n')
            self._file.flush()
            self._seen[key] = fingerprint
            self._seen_ids[key] = request.sample_id
            self._writes_since_fsync += 1
            if self._writes_since_fsync >= self.fsync_interval:
                os.fsync(self._file.fileno())
                self._writes_since_fsync = 0
        return True

    def close(self) -> None:
        with self._lock:
            if self._file.closed:
                return
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()

    def __enter__(self) -> 'JsonlSequenceReplayWriter':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
from verl_speco.producer.input_reader import _build_tokenized_request
MAX_SEGMENT_BYTES = 16 * 1024 ** 2
MAX_ROWS = 2000000

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def segment_path(root, name):
    if not isinstance(name, str) or Path(name).name != name or re.fullmatch('part-[A-Za-z0-9_.-]+-segment-[0-9]{6}\\.jsonl', name) is None:
        raise ValueError('frozen segment must have a local committed filename')
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('frozen segment escapes its owned sequence directory')
    return path

def read_segment(root, descriptor):
    path = segment_path(root, descriptor['file'])
    with path.open('rb') as handle:
        blob = handle.read(MAX_SEGMENT_BYTES + 1)
    if len(blob) > MAX_SEGMENT_BYTES or hashlib.sha256(blob).hexdigest() != descriptor['sha256']:
        raise ValueError('frozen segment size or complete SHA differs')
    return blob

def build_manifest_index(root, pins, *, target_model_id, target_model_revision, tokenizer_fingerprint):
    root = Path(root).resolve()
    provenance = dict(target_model_id=target_model_id, target_model_revision=target_model_revision, tokenizer_fingerprint=tokenizer_fingerprint)
    if not root.is_dir() or not pins or (not all((isinstance(v, str) and v for v in provenance.values()))):
        raise ValueError('frozen replay requires its sequence directory, pins and model provenance')
    pins = json.loads(json.dumps(pins))
    entries, segments, seen_ids, seen_fingerprints = ([], [], set(), set())
    seen_files, seen_shards, manifest_hashes = (set(), set(), [])
    counts = dict(rows=0, original_tokens=0, prompt_tokens=0, response_tokens=0, no_trainable_response_rows=0, over_16k_rows=0, max_original_tokens=0)
    for pin in pins:
        manifest = pin['manifest']
        if digest(manifest) != pin['sha256']:
            raise ValueError('frozen manifest differs from the saved pause receipt')
        shard = manifest.get('shard_id')
        if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or (not isinstance(shard, str)) or (shard in seen_shards) or (re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', shard) is None) or (not isinstance(manifest.get('run_id'), str)) or (not manifest['run_id']) or (type(manifest.get('committed_rows')) is not int) or (manifest['committed_rows'] < 1) or (not isinstance(manifest.get('segments'), list)) or (not manifest['segments']):
            raise ValueError('invalid or duplicate frozen shard')
        seen_shards.add(shard)
        manifest_hashes.append(dict(shard=shard, sha256=pin['sha256']))
        shard_rows = 0
        seen_sequence_nos = set()
        for descriptor in manifest['segments']:
            name = descriptor.get('file')
            if not isinstance(name, str) or name in seen_files or (not name.startswith(f'part-{shard}-segment-')) or (type(descriptor.get('rows')) is not int) or (not 1 <= descriptor['rows'] <= 4096):
                raise ValueError('invalid or duplicate frozen segment membership')
            seen_files.add(name)
            blob = read_segment(root, descriptor)
            segment_index = len(segments)
            payload_shard = Path(name).stem.removeprefix('part-')
            segments.append(dict(file=name, sha256=descriptor['sha256'], nbytes=len(blob), rows=descriptor['rows'], shard_id=payload_shard, manifest_shard_id=shard, run_id=manifest['run_id']))
            offset, sequence_nos = (0, [])
            for line_no, raw in enumerate(blob.splitlines(keepends=True), 1):
                start = offset
                offset += len(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                ids, mask, positions, prompt, metadata, fingerprint = validate_sequence_replay_row(row, algorithm='DSPARK', context=f'{name}:{line_no}', **provenance)
                sequence_no = row.get('sequence_no')
                if row.get('run_id') != manifest['run_id'] or row.get('shard_id') != payload_shard or type(sequence_no) is not int or (sequence_no < 0) or (sequence_no in seen_sequence_nos) or (row['id'] in seen_ids) or (fingerprint in seen_fingerprints):
                    raise ValueError('replay identity duplicates or differs from its pinned shard')
                seen_sequence_nos.add(sequence_no)
                sequence_nos.append(sequence_no)
                seen_ids.add(row['id'])
                seen_fingerprints.add(fingerprint)
                response = len(ids) - prompt
                entries.append(dict(segment_index=segment_index, offset=start, nbytes=len(raw), row_sha256=hashlib.sha256(raw).hexdigest(), id=row['id'], sequence_fingerprint=fingerprint, original_tokens=len(ids), prompt_tokens=prompt, response_tokens=response, original_sequence_no=sequence_no))
                counts['rows'] += 1
                counts['original_tokens'] += len(ids)
                counts['prompt_tokens'] += prompt
                counts['response_tokens'] += response
                counts['no_trainable_response_rows'] += int(response < 2)
                counts['over_16k_rows'] += int(len(ids) > 16384)
                counts['max_original_tokens'] = max(counts['max_original_tokens'], len(ids))
                if len(entries) > MAX_ROWS:
                    raise ValueError('frozen catalog exceeds its metadata bound')
            if len(sequence_nos) != descriptor['rows'] or type(descriptor.get('first_sequence_no')) is not int or type(descriptor.get('last_sequence_no')) is not int or (descriptor['first_sequence_no'] != sequence_nos[0]) or (descriptor['last_sequence_no'] != sequence_nos[-1]):
                raise ValueError('frozen segment row count or sequence range differs')
            shard_rows += len(sequence_nos)
        if shard_rows != manifest['committed_rows']:
            raise ValueError('frozen shard committed row total differs')
    snapshot = dict(schema_version=1, manifests=manifest_hashes, provenance=provenance)
    index = dict(schema_version=1, source_directory=str(root), snapshot=snapshot, dataset_sha256=digest(snapshot), provenance=provenance, segments=segments, entries=entries, counts=counts, payloads_copied=False, rows_filtered=False)
    index['index_sha256'] = digest(index)
    return index

class FrozenManifestReplayCatalog:

    def __init__(self, root, index, plan):
        self.root = Path(root).resolve()
        self.index = json.loads(json.dumps(index))
        declared = self.index.pop('index_sha256', None)
        if declared != digest(self.index) or self.index.get('schema_version') != 1 or self.index.get('source_directory') != str(self.root) or (digest(self.index['snapshot']) != self.index['dataset_sha256']) or (self.index['dataset_sha256'] != plan.dataset_sha256) or (len(self.index['entries']) != plan.samples_per_epoch):
            raise ValueError('frozen index, dataset and epoch plan differ')
        self.plan = plan
        for number in range(plan.samples_per_epoch):
            item = plan.occurrence(number)
            row = self.index['entries'][item.source_index]
            if row['id'] != item.original_sample_id or row['sequence_fingerprint'] != item.sequence_fingerprint or row['original_tokens'] != item.original_tokens:
                raise ValueError('epoch plan differs from the frozen row identities')

    def request(self, number, *, config):
        item = self.plan.occurrence(number)
        entry = self.index['entries'][item.source_index]
        segment = self.index['segments'][entry['segment_index']]
        blob = read_segment(self.root, segment)
        raw = blob[entry['offset']:entry['offset'] + entry['nbytes']]
        if hashlib.sha256(raw).hexdigest() != entry['row_sha256']:
            raise ValueError('frozen epoch row checksum differs')
        row = json.loads(raw)
        ids, mask, positions, prompt, metadata, fingerprint = validate_sequence_replay_row(row, algorithm='DSPARK', context=segment['file'], **self.index['provenance'])
        if row['id'] != item.original_sample_id or fingerprint != item.sequence_fingerprint or len(ids) != item.original_tokens or (row.get('run_id') != segment['run_id']) or (row.get('shard_id') != segment['shard_id']) or (row.get('sequence_no') != entry['original_sequence_no']):
            raise ValueError('frozen epoch row identity differs')
        return _build_tokenized_request(sequence_no=number, sample_id=item.sample_id, prompt_length=prompt, full_ids=ids, config=config, vllm_prompt_token_ids=ids[:-1], feature_end_limit=len(ids) - 1, source_metadata=dict(replay_file=str(self.root / segment['file']), replay_source_sample_id=row['id'], sequence_fingerprint=fingerprint, replay_epoch=item.epoch, replay_epoch_position=item.epoch_position, replay_plan_sha256=self.plan.plan_sha256, replay_dataset_sha256=self.plan.dataset_sha256))
from dataclasses import dataclass
import random
from collections.abc import Iterator, Mapping, Sequence

def replay_epoch_plan_sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None

@dataclass(frozen=True)
class ReplayOccurrence:
    sequence_no: int
    epoch: int
    epoch_position: int
    source_index: int
    original_sample_id: str
    sample_id: str
    sequence_fingerprint: str
    original_tokens: int

@dataclass(frozen=True)
class ReplayBatch:
    step_index: int
    start_cursor: int
    end_cursor: int
    occurrences: tuple[ReplayOccurrence, ...]

    @property
    def original_tokens(self):
        return sum((item.original_tokens for item in self.occurrences))

class FrozenReplayEpochPlan:
    """Shuffle each epoch once, retaining every occurrence in the final batch.

    Batches may cross an epoch boundary. A final batch may leave some producers
    empty; its assignment is explicit and never pads or drops saved sequences.
    Nothing is dropped or repeated merely to fill the last training batch.
    """

    def __init__(self, rows: Sequence[Mapping], *, dataset_sha256: str, epochs: int=3, seed: int=42, samples_per_step: int=210, producer_count: int=3, prefill_balance_policy: str=''):
        if not replay_epoch_plan_sha(dataset_sha256) or type(epochs) is not int or (not 1 <= epochs <= 65536) or (type(seed) is not int) or (not 0 <= seed < 2 ** 63) or (type(samples_per_step) is not int) or (not 1 <= samples_per_step <= 1024) or (type(producer_count) is not int) or (producer_count not in (3, 6)) or (samples_per_step % producer_count):
            raise ValueError('invalid frozen epoch plan configuration')
        if not rows or len(rows) > 2000000 or len(rows) * epochs > 6000000:
            raise ValueError('frozen replay catalog is empty or exceeds its bound')
        canonical = []
        seen = set()
        seen_fingerprints = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {'id', 'sequence_fingerprint', 'original_tokens'} or (not isinstance(row['id'], str)) or (not row['id']) or (row['id'] in seen) or (len(row['id']) > 4096) or (not isinstance(row['sequence_fingerprint'], str)) or (not row['sequence_fingerprint'].startswith('sha256:')) or (not replay_epoch_plan_sha(row['sequence_fingerprint'][7:])) or (row['sequence_fingerprint'] in seen_fingerprints) or (type(row['original_tokens']) is not int) or (not 2 <= row['original_tokens'] <= 2 ** 31 - 1):
                raise ValueError('invalid or duplicate frozen replay row')
            canonical.append(dict(row))
            seen.add(row['id'])
            seen_fingerprints.add(row['sequence_fingerprint'])
        self._rows = tuple(canonical)
        self.epochs = epochs
        self.seed = seed
        self.samples_per_step = samples_per_step
        self.producer_count = producer_count
        self.dataset_sha256 = dataset_sha256
        self.samples_per_epoch = len(canonical)
        self.total_samples = self.samples_per_epoch * epochs
        self.total_steps = (self.total_samples + samples_per_step - 1) // samples_per_step
        self.total_original_tokens = epochs * sum((r['original_tokens'] for r in canonical))
        from verl_speco.producer.batching import BALANCE_POLICIES, PACKED_BALANCE_POLICY, PACKED_EIGHT_BALANCE_POLICY, balance_physical_batch_size
        if type(prefill_balance_policy) is not str or prefill_balance_policy not in ('', *BALANCE_POLICIES) or (prefill_balance_policy and (producer_count != 6 or (samples_per_step != 480 and (not (prefill_balance_policy in (PACKED_BALANCE_POLICY, PACKED_EIGHT_BALANCE_POLICY) and samples_per_step == 960))))):
            raise ValueError('unsupported frozen prefill balancing policy')
        self.prefill_balance_policy = prefill_balance_policy
        orders = []
        for epoch in range(epochs):
            order = list(range(self.samples_per_epoch))
            epoch_seed = hashlib.sha256(f'speco-replay-epoch-v1:{seed}:{epoch}'.encode()).digest()
            random.Random(int.from_bytes(epoch_seed, 'big')).shuffle(order)
            orders.append(tuple(order))
        if prefill_balance_policy:
            from verl_speco.producer.batching import balanced_prefill_positions, packed_balanced_prefill_positions
            balance_positions = packed_balanced_prefill_positions if prefill_balance_policy in (PACKED_BALANCE_POLICY, PACKED_EIGHT_BALANCE_POLICY) else balanced_prefill_positions
            balanced = []
            for epoch, order in enumerate(orders):
                order = list(order)
                offset = 0
                while offset < self.samples_per_epoch:
                    cursor = epoch * self.samples_per_epoch + offset
                    end = min(self.samples_per_epoch, offset + samples_per_step - cursor % samples_per_step)
                    chunk = order[offset:end]
                    positions = balance_positions([self._rows[i]['original_tokens'] for i in chunk], start_cursor=cursor, physical_batch_size=balance_physical_batch_size(prefill_balance_policy))
                    order[offset:end] = [chunk[i] for i in positions]
                    offset = end
                balanced.append(tuple(order))
            orders = balanced
        self._orders = tuple(orders)
        identity = dict(schema_version=2, dataset_sha256=dataset_sha256, epochs=epochs, seed=seed, samples_per_step=samples_per_step, producer_count=producer_count, rows=canonical, orders=self._orders)
        if prefill_balance_policy:
            identity.update(schema_version=3, prefill_balance_policy=prefill_balance_policy)
        self.plan_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def occurrence(self, sequence_no: int) -> ReplayOccurrence:
        if type(sequence_no) is not int or not 0 <= sequence_no < self.total_samples:
            raise ValueError('replay occurrence is outside the frozen plan')
        epoch, position = divmod(sequence_no, self.samples_per_epoch)
        source_index = self._orders[epoch][position]
        row = self._rows[source_index]
        occurrence_hash = hashlib.sha256(f'{self.plan_sha256}:{epoch}:{source_index}'.encode()).hexdigest()
        return ReplayOccurrence(sequence_no, epoch, position, source_index, row['id'], f'replay-epoch-{epoch}-{occurrence_hash}', row['sequence_fingerprint'], row['original_tokens'])

    def cursor(self, completed_samples: int, *, receipt_sha256: str='') -> dict:
        if type(completed_samples) is not int or not 0 <= completed_samples <= self.total_samples or (completed_samples != self.total_samples and completed_samples % self.samples_per_step) or (completed_samples == 0 and receipt_sha256 != '') or (completed_samples > 0 and (not replay_epoch_plan_sha(receipt_sha256))):
            raise ValueError('cursor must be at a receipt-bound optimizer batch boundary')
        epoch, position = divmod(completed_samples, self.samples_per_epoch)
        return dict(schema_version=1, plan_sha256=self.plan_sha256, completed_samples=completed_samples, completed_optimizer_steps=(completed_samples + self.samples_per_step - 1) // self.samples_per_step, next_epoch=epoch, next_epoch_position=position, receipt_sha256=receipt_sha256)

    def validate_cursor(self, cursor: Mapping) -> None:
        if not isinstance(cursor, Mapping):
            raise ValueError('epoch cursor must be an object')
        expected = self.cursor(cursor.get('completed_samples'), receipt_sha256=cursor.get('receipt_sha256'))
        if set(cursor) != set(expected) or any((cursor[k] != v or type(cursor[k]) is not type(v) for k, v in expected.items())):
            raise ValueError('epoch cursor differs from the frozen plan or optimizer boundary')

    def batches(self, *, cursor: Mapping | None=None) -> Iterator[ReplayBatch]:
        saved = self.cursor(0) if cursor is None else cursor
        self.validate_cursor(saved)
        for start in range(saved['completed_samples'], self.total_samples, self.samples_per_step):
            end = min(start + self.samples_per_step, self.total_samples)
            occurrences = tuple((self.occurrence(i) for i in range(start, end)))
            yield ReplayBatch(start // self.samples_per_step, start, end, occurrences)

    def producer_assignments(self, batch: ReplayBatch) -> tuple[tuple[ReplayOccurrence, ...], ...]:
        """Assign a canonical optimizer batch to all producers, including empty ones.

        An empty assignment is a scheduling no-op, never a synthetic sample.
        Runtime integration must still acknowledge every producer at the barrier.
        """
        if type(batch) is not ReplayBatch or type(batch.step_index) is not int or (not 0 <= batch.step_index < self.total_steps) or (type(batch.start_cursor) is not int) or (type(batch.end_cursor) is not int):
            raise ValueError('producer batch must belong to the frozen plan')
        start = batch.step_index * self.samples_per_step
        end = min(start + self.samples_per_step, self.total_samples)
        expected = tuple((self.occurrence(i) for i in range(start, end)))
        if batch.start_cursor != start or batch.end_cursor != end or type(batch.occurrences) is not tuple or (len(batch.occurrences) != len(expected)) or any((type(item) is not ReplayOccurrence or any((type(getattr(item, field)) is not type(getattr(reference, field)) for field in ReplayOccurrence.__dataclass_fields__)) for item, reference in zip(batch.occurrences, expected))) or (batch.occurrences != expected):
            raise ValueError('producer batch differs from the frozen epoch occurrences')
        return tuple((tuple((item for item in expected if item.sequence_no % self.producer_count == rank)) for rank in range(self.producer_count)))
SPEC_PATH = 'SPECO_EPOCH_REPLAY_SPEC_PATH'
SPEC_SHA = 'SPECO_EPOCH_REPLAY_SPEC_SHA256'
MAX_INDEX_BYTES = 512 * 1024 ** 2

def checked_json(path, sha256, *, outputs_root, max_bytes):
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(outputs_root) or (not isinstance(sha256, str)) or (re.fullmatch('[0-9a-f]{64}', sha256) is None):
        raise ValueError('epoch replay needs a pinned file inside owned outputs')
    with path.open('rb') as handle:
        chunks, total = ([], 0)
        while (chunk := handle.read(min(1024 ** 2, max_bytes + 1 - total))):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError('epoch replay JSON exceeds its byte bound')
            chunks.append(chunk)
    payload = b''.join(chunks)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError('epoch replay file SHA differs from its frozen launch')
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError('epoch replay JSON must be an object')
    return value

def epoch_replay_cursor_factory(environment, *, outputs_root, producer_count, max_sequence_length, provenance):
    configured = [environment.get(key, '') for key in (SPEC_PATH, SPEC_SHA)]
    if configured == ['', '']:
        return None
    if any((not isinstance(value, str) or not value for value in configured)):
        raise ValueError('partial epoch replay configuration cannot use the legacy reader')
    outputs_root = Path(outputs_root).resolve()
    spec = checked_json(configured[0], configured[1], outputs_root=outputs_root, max_bytes=65536)
    from verl_speco.producer.batching import epoch_spec_balance_policy
    balance_policy = epoch_spec_balance_policy(spec)
    required = {'schema_version', 'index_path', 'index_file_sha256', 'index_sha256', 'dataset_sha256', 'plan_sha256', 'epochs', 'seed', 'samples_per_step', 'producer_count', 'max_sequence_length'}
    if type(spec['schema_version']) is not int or type(spec['epochs']) is not int or spec['epochs'] != 3 or (type(spec['producer_count']) is not int) or (spec['producer_count'] != producer_count) or (producer_count not in (3, 6)) or (type(spec['max_sequence_length']) is not int) or (spec['max_sequence_length'] != max_sequence_length):
        raise ValueError('epoch replay launch topology, epochs or context differs')
    source_path = environment.get('SPECO_SEQUENCE_REPLAY_INPUT', '')
    if not isinstance(source_path, str) or not Path(source_path).is_absolute() or Path(source_path).resolve() != Path(spec['index_path']).resolve():
        raise ValueError('epoch replay input differs from the source fingerprint owner')
    index = checked_json(spec['index_path'], spec['index_file_sha256'], outputs_root=outputs_root, max_bytes=MAX_INDEX_BYTES)
    if index.get('index_sha256') != spec['index_sha256'] or index.get('dataset_sha256') != spec['dataset_sha256'] or index.get('provenance') != provenance or (not Path(index['source_directory']).resolve().is_relative_to(outputs_root)):
        raise ValueError('epoch replay index or model provenance differs from the service')
    rows = [{key: entry[key] for key in ('id', 'sequence_fingerprint', 'original_tokens')} for entry in index['entries']]
    plan = FrozenReplayEpochPlan(rows, dataset_sha256=spec['dataset_sha256'], epochs=spec['epochs'], seed=spec['seed'], samples_per_step=spec['samples_per_step'], producer_count=producer_count, prefill_balance_policy=balance_policy)
    if plan.plan_sha256 != spec['plan_sha256']:
        raise ValueError('epoch replay plan changed after frozen launch preparation')
    catalog = FrozenManifestReplayCatalog(index['source_directory'], index, plan)
    if any((entry['response_tokens'] < 2 or entry['original_tokens'] > max_sequence_length for entry in index['entries'])):
        raise ValueError('epoch replay needs an explicit eligibility snapshot before device startup')

    def reader(number):
        return catalog.request(number, config={'min_supervised_tokens': 1, 'max_sequence_length': max_sequence_length})
    return FrozenEpochReplaySource(catalog=catalog, reader=reader, spec_path=Path(configured[0]), spec_file_sha256=configured[1], index_file_sha256=spec['index_file_sha256'])

def labelled_dataset_fingerprint(file_sha256):
    return hashlib.sha256(b'prompt_dataset\x00' + bytes.fromhex(file_sha256)).hexdigest()

@dataclass(frozen=True)
class FrozenEpochReplaySource:
    catalog: object
    reader: object
    spec_path: Path
    spec_file_sha256: str
    index_file_sha256: str

    @property
    def plan(self):
        return self.catalog.plan

    def __call__(self, *, identity):
        return IndexedReplayCursor(self.reader, identity=identity, total_samples=self.plan.total_samples)

    def input_fingerprints(self, *, model_overlay, target_model_dir, model_content_audit=None, verify_shards_full=False):
        from verl_speco.runtime.services import speco_92b_input_fingerprints as speco_92b_input_fingerprints
        result = speco_92b_input_fingerprints(model_overlay=model_overlay, target_model_dir=target_model_dir, prompt_dataset=self.spec_path, model_content_audit=model_content_audit, verify_shards_full=verify_shards_full)
        if result['dataset_fingerprint'] != labelled_dataset_fingerprint(self.spec_file_sha256):
            raise ValueError('frozen epoch launch spec changed after source admission')
        return dict(result, dataset_fingerprint=labelled_dataset_fingerprint(self.index_file_sha256))

def select_prompt_by_index(tokenizer: object, dataset_path: Path, *, sequence_no: int, offset: int=0, max_prompt_tokens: int=8000) -> tuple[tuple[int, ...], str]:
    """Choose a sorted prompt without changing the contiguous stream cursor."""
    if sequence_no < 0 or offset < 0 or max_prompt_tokens <= 0:
        raise ValueError('prompt sequence, offset and token limit are invalid')
    candidates: list[tuple[int, tuple[int, ...], str]] = []
    with dataset_path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            rendered = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
            ids = tuple(rendered['input_ids'] if hasattr(rendered, 'keys') else rendered)
            if ids and len(ids) <= max_prompt_tokens:
                candidates.append((len(ids), ids, str(row['sample_id'])))
    candidates.sort(key=lambda item: (item[0], item[2]))
    index = offset + sequence_no
    if index >= len(candidates):
        raise ValueError('prompt dataset has too few samples within target context length')
    selected = candidates[index]
    sample_id = 'sample-' + hashlib.sha256(selected[2].encode()).hexdigest()[:16]
    return (selected[1], sample_id)
import argparse
from collections import Counter
from typing import Any, Iterator
raw_messages_to_prompts_MEDIA_MARKER = re.compile('<\\|?(?:image|video|audio)(?:_pad)?\\|?>|data:(?:image|video|audio)/', re.IGNORECASE)

@dataclass(frozen=True)
class ConversionReport:
    input_files: int
    scanned: int
    emitted: int
    skipped: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {'input_files': self.input_files, 'scanned': self.scanned, 'emitted': self.emitted, 'skipped': dict(sorted(self.skipped.items()))}

def raw_messages_to_prompts_input_files(input_root: Path) -> Iterator[tuple[Path, str]]:
    if input_root.is_file():
        if input_root.suffix.lower() != '.jsonl':
            raise ValueError('input file must end in .jsonl')
        yield (input_root, input_root.name)
        return
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    files = sorted(input_root.rglob('*.jsonl'))
    if not files:
        raise ValueError(f'no JSONL files under {input_root}')
    for path in files:
        yield (path, path.relative_to(input_root).as_posix())

def raw_messages_to_prompts_convert_row(row: Any, *, version: str, relative_path: str, line_number: int, max_prompt_chars: int) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(row, dict) or not isinstance(row.get('messages'), list):
        return (None, 'invalid_schema')
    if row.get('tools'):
        return (None, 'tools')
    messages = row['messages']
    if len(messages) != 3 or any((not isinstance(message, dict) for message in messages)):
        return (None, 'not_single_turn')
    if tuple((message.get('role') for message in messages)) != ('system', 'user', 'assistant'):
        return (None, 'not_single_turn')
    contents = [message.get('content') for message in messages]
    if any((not isinstance(content, str) or not content.strip() for content in contents)):
        return (None, 'empty_or_nontext_content')
    prompt = [{'role': messages[index]['role'], 'content': contents[index]} for index in (0, 1)]
    if any((raw_messages_to_prompts_MEDIA_MARKER.search(message['content']) for message in prompt)):
        return (None, 'media_marker')
    if sum((len(message['content']) for message in prompt)) > max_prompt_chars:
        return (None, 'prompt_too_long')
    return ({'sample_id': f'{version}:{relative_path}:{line_number}', 'prompt': prompt, 'source_version': version, 'source_file': relative_path, 'source_line': line_number}, None)

def convert_raw_messages(input_root: str | Path, output_path: str | Path, *, version: str, max_records: int=0, max_prompt_chars: int=65536, tokenizer_path: str | Path | None=None, max_sequence_length: int=0, generation_max_tokens: int=0) -> ConversionReport:
    """Write prompt-only JSONL and a completion manifest; never copy answers."""
    source = Path(input_root)
    output = Path(output_path)
    if not version or any((char in version for char in '/\\:')):
        raise ValueError("version must be non-empty and contain no path separators or ':'")
    if max_records < 0 or max_prompt_chars <= 0:
        raise ValueError('max_records must be non-negative and max_prompt_chars positive')
    if tokenizer_path is not None and (max_sequence_length <= 0 or generation_max_tokens <= 0):
        raise ValueError('tokenizer filtering requires positive sequence and generation limits')
    if tokenizer_path is None and (max_sequence_length or generation_max_tokens):
        raise ValueError('token limits require tokenizer_path')
    if output.exists() or output.with_suffix(output.suffix + '.complete.json').exists():
        raise FileExistsError(output)
    tokenizer = None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    scanned = emitted = input_files = 0
    with output.open('w', encoding='utf-8') as writer:
        for path, relative_path in raw_messages_to_prompts_input_files(source):
            input_files += 1
            with path.open('r', encoding='utf-8') as reader:
                for line_number, line in enumerate(reader, 1):
                    if not line.strip():
                        skipped['blank_line'] += 1
                        continue
                    scanned += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f'invalid JSON at {path}:{line_number}') from error
                    converted, reason = raw_messages_to_prompts_convert_row(row, version=version, relative_path=relative_path, line_number=line_number, max_prompt_chars=max_prompt_chars)
                    if reason is not None:
                        skipped[reason] += 1
                        continue
                    if tokenizer is not None:
                        from verl_speco.producer.input_reader import InputRecord, prepare_generation_request
                        record = InputRecord(sequence_no=emitted, sample_id=converted['sample_id'], prompt=tuple(converted['prompt']), response=None, source_metadata={})
                        try:
                            request = prepare_generation_request(record, tokenizer, {'max_sequence_length': max_sequence_length, 'generation_max_tokens': generation_max_tokens})
                        except ValueError:
                            skipped['tokenizer_or_length_rejected'] += 1
                            continue
                        if request.max_tokens < generation_max_tokens:
                            skipped['insufficient_generation_room'] += 1
                            continue
                    writer.write(json.dumps(converted, ensure_ascii=False) + '\n')
                    emitted += 1
                    if max_records and emitted >= max_records:
                        break
            if max_records and emitted >= max_records:
                break
        writer.flush()
    report = ConversionReport(input_files, scanned, emitted, dict(skipped))
    manifest = output.with_suffix(output.suffix + '.complete.json')
    manifest.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + '\n')
    return report

def raw_messages_to_prompts_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--version', required=True)
    parser.add_argument('--max-records', type=int, default=0)
    parser.add_argument('--max-prompt-chars', type=int, default=65536)
    parser.add_argument('--tokenizer-path', type=Path)
    parser.add_argument('--max-sequence-length', type=int, default=0)
    parser.add_argument('--generation-max-tokens', type=int, default=0)
    args = parser.parse_args()
    report = convert_raw_messages(args.input_root, args.output, version=args.version, max_records=args.max_records, max_prompt_chars=args.max_prompt_chars, tokenizer_path=args.tokenizer_path, max_sequence_length=args.max_sequence_length, generation_max_tokens=args.generation_max_tokens)
    print(json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True))
from typing import Any
from verl_speco.runtime.artifacts import publish_file
from verl_speco.producer.input_reader import InputRecord, prepare_generation_request

def prepare_raw_sequence_shards(input_root: Path, output_dir: Path, *, version: str, shard_count: int, tokenizer: Any, max_sequence_length: int, generation_max_tokens: int, max_records: int=0) -> dict[str, Any]:
    if not version or shard_count <= 0 or max_sequence_length <= 0 or (generation_max_tokens <= 0) or (max_records < 0):
        raise ValueError('version, shard count, token limits and record budget are invalid')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'sequence shard directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f'prompts-{index:02d}.jsonl' for index in range(shard_count)]
    temporary = [path.with_suffix('.jsonl.partial') for path in paths]
    writers = [path.open('w', encoding='utf-8') for path in temporary]
    skipped: Counter[str] = Counter()
    scanned = emitted = input_files = 0
    try:
        for path, relative in raw_messages_to_prompts_input_files(input_root):
            input_files += 1
            with path.open('r', encoding='utf-8') as reader:
                for line_number, line in enumerate(reader, 1):
                    if not line.strip():
                        skipped['blank_line'] += 1
                        continue
                    scanned += 1
                    row = json.loads(line)
                    prompt, reason = raw_messages_to_prompts_convert_row(row, version=version, relative_path=relative, line_number=line_number, max_prompt_chars=65536)
                    if reason is not None:
                        skipped[reason] += 1
                        continue
                    record = InputRecord(sequence_no=emitted, sample_id=prompt['sample_id'], prompt=tuple(prompt['prompt']), response=None, source_metadata={})
                    try:
                        request = prepare_generation_request(record, tokenizer, {'max_sequence_length': max_sequence_length, 'generation_max_tokens': generation_max_tokens})
                    except ValueError:
                        skipped['tokenizer_or_length_rejected'] += 1
                        continue
                    if request.max_tokens < generation_max_tokens:
                        skipped['insufficient_generation_room'] += 1
                        continue
                    writers[emitted % shard_count].write(json.dumps(prompt, ensure_ascii=False, separators=(',', ':')) + '\n')
                    emitted += 1
                    if max_records and emitted >= max_records:
                        break
            if max_records and emitted >= max_records:
                break
        for writer in writers:
            writer.flush()
            os.fsync(writer.fileno())
            writer.close()
        for source, target in zip(temporary, paths):
            publish_file(source, target)
    finally:
        for writer in writers:
            if not writer.closed:
                writer.close()
    report = {'schema_version': 1, 'source': str(input_root.resolve()), 'version': version, 'input_files': input_files, 'scanned': scanned, 'emitted': emitted, 'shard_count': shard_count, 'shard_rows': [emitted // shard_count + (index < emitted % shard_count) for index in range(shard_count)], 'max_sequence_length': max_sequence_length, 'generation_max_tokens': generation_max_tokens, 'skipped': dict(sorted(skipped.items()))}
    manifest = output_dir / 'prompts.complete.json'
    manifest.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    return report

def raw_sequence_shards_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--shard-count', type=int, default=16)
    parser.add_argument('--tokenizer-path', type=Path, required=True)
    parser.add_argument('--max-sequence-length', type=int, default=65536)
    parser.add_argument('--generation-max-tokens', type=int, default=32768)
    parser.add_argument('--max-records', type=int, default=0)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, local_files_only=True)
    print(json.dumps(prepare_raw_sequence_shards(args.input_root, args.output_dir, version=args.version, shard_count=args.shard_count, tokenizer=tokenizer, max_sequence_length=args.max_sequence_length, generation_max_tokens=args.generation_max_tokens, max_records=args.max_records), ensure_ascii=False, sort_keys=True), flush=True)
from verl_speco.runtime.artifacts import fcntl
import shutil
from typing import Any, Mapping
from verl_speco.runtime.artifacts import publish_file, publish_json, read_published_json
from verl_speco.producer.input_reader import TokenizedRequest

def sequence_segment_writer_input_fingerprint(path: str | os.PathLike[str]) -> dict[str, object]:
    source = Path(path).resolve()
    stat = source.stat()
    return {'path': str(source), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

class CommittedSequenceSegmentWriter:
    """Publish a usable JSONL segment and manifest every N generated samples.

    A pending segment is hidden from readers and discarded after a crash. The
    manifest is authoritative, so an orphan segment from a crash between two
    atomic renames is never consumed or counted as committed.
    """

    def __init__(self, output_dir: str | os.PathLike[str], *, input_path: str | os.PathLike[str], run_id: str, shard_id: str, segment_size: int, allow_out_of_order: bool=False) -> None:
        if segment_size <= 0:
            raise ValueError('sequence segment size must be positive')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_path = os.fspath(input_path)
        self.run_id = run_id
        self.shard_id = shard_id
        self.segment_size = segment_size
        self.allow_out_of_order = allow_out_of_order
        self.record_order = 'completion' if allow_out_of_order else 'input_prefix'
        self.manifest_path = self.output_dir / f'part-{shard_id}.manifest.json'
        self._lock_file = (self.output_dir / f'.part-{shard_id}.lock').open('a+')
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._lock_file.close()
            raise
        try:
            self._fingerprint = sequence_segment_writer_input_fingerprint(self.input_path)
            self._segments: list[dict[str, Any]] = []
            self._seen_ids: dict[int, str] = {}
            self._complete = False
            self._pending_dir = self.output_dir / f'.part-{shard_id}.pending'
            self._load_manifest()
            if self._pending_dir.exists():
                shutil.rmtree(self._pending_dir)
            self._pending_dir.mkdir()
            self._pending: JsonlSequenceReplayWriter | None = None
            self._pending_count = 0
            self._pending_ids: dict[int, str] = {}
            self._closed = False
        except BaseException:
            self._lock_file.close()
            raise

    def _load_manifest(self) -> None:
        try:
            manifest = read_published_json(self.manifest_path)
        except FileNotFoundError:
            return
        if manifest.get('schema_version') != 1 or manifest.get('run_id') != self.run_id or manifest.get('shard_id') != self.shard_id or (manifest.get('segment_size') != self.segment_size) or (manifest.get('record_order', 'input_prefix') != self.record_order) or (manifest.get('input_fingerprint') != self._fingerprint) or (not isinstance(manifest.get('segments'), list)):
            raise ValueError('sequence segment manifest identity differs')
        for index, descriptor in enumerate(manifest['segments']):
            name = f'part-{self.shard_id}-segment-{index:06d}.jsonl'
            if descriptor.get('file') != name:
                raise ValueError('sequence segment manifest has noncontiguous files')
            path = self.output_dir / name
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != descriptor.get('sha256'):
                raise ValueError(f'committed sequence segment checksum differs: {path}')
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
            if len(rows) != descriptor.get('rows') or not rows:
                raise ValueError(f'committed sequence segment row count differs: {path}')
            for row in rows:
                sequence_no = row.get('sequence_no')
                sample_id = row.get('id')
                if row.get('run_id') != self.run_id or type(sequence_no) is not int or sequence_no < 0 or (sequence_no in self._seen_ids) or (not isinstance(sample_id, str)):
                    raise ValueError(f'committed sequence segment identity differs: {path}')
                self._seen_ids[sequence_no] = sample_id
            self._segments.append(descriptor)
        if not self.allow_out_of_order and sorted(self._seen_ids) != list(range(len(self._seen_ids))):
            raise ValueError('committed sequence segments do not cover a contiguous prefix')
        self._complete = bool(manifest.get('complete'))

    @property
    def persisted_record_count(self) -> int:
        return len(self._seen_ids) + self._pending_count

    def has_record(self, *, sequence_no: int, sample_id: str) -> bool:
        existing = self._seen_ids.get(sequence_no)
        if existing is not None and existing != sample_id:
            raise ValueError('sequence segment input changed at a committed position')
        return existing is not None

    def validate_progress_input(self, input_path: str | os.PathLike[str]) -> None:
        if sequence_segment_writer_input_fingerprint(input_path) != self._fingerprint:
            raise ValueError('sequence segment input fingerprint changed')

    def write(self, request: TokenizedRequest, *, algorithm: str, metadata: Mapping[str, Any] | None=None) -> bool:
        if self._closed or self._complete:
            raise RuntimeError('sequence segment writer is closed or complete')
        if not self.allow_out_of_order and request.sequence_no != len(self._seen_ids) + self._pending_count:
            raise ValueError('sequence segment rows must follow a contiguous input prefix')
        if request.sequence_no < 0:
            raise ValueError('sequence segment number must be nonnegative')
        existing = self._seen_ids.get(request.sequence_no, self._pending_ids.get(request.sequence_no))
        if existing is not None:
            if existing != request.sample_id:
                raise ValueError('sequence segment input changed at an existing position')
            return False
        if self._pending is None:
            self._pending = JsonlSequenceReplayWriter(self._pending_dir, run_id=self.run_id, shard_id=f'{self.shard_id}-segment-{len(self._segments):06d}', fsync_interval=self.segment_size)
        wrote = self._pending.write(request, algorithm=algorithm, metadata=metadata)
        if not wrote:
            raise RuntimeError('pending sequence segment unexpectedly contains a duplicate')
        self._pending_ids[request.sequence_no] = request.sample_id
        self._pending_count += 1
        if self._pending_count == self.segment_size:
            self._commit_pending()
        return True

    def _write_manifest(self, *, complete: bool) -> None:
        payload = {'schema_version': 1, 'run_id': self.run_id, 'shard_id': self.shard_id, 'segment_size': self.segment_size, 'input_fingerprint': self._fingerprint, 'committed_rows': len(self._seen_ids), 'complete': complete, 'segments': self._segments}
        if self.allow_out_of_order:
            payload['record_order'] = self.record_order
        publish_json(self.manifest_path, payload)

    def _commit_pending(self) -> None:
        if self._pending is None or not self._pending_count:
            return
        self._pending.close()
        data = self._pending.path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(rows) != self._pending_count:
            raise RuntimeError('pending sequence segment is incomplete')
        index = len(self._segments)
        name = f'part-{self.shard_id}-segment-{index:06d}.jsonl'
        target = self.output_dir / name
        publish_file(self._pending.path, target)
        for row in rows:
            self._seen_ids[row['sequence_no']] = row['id']
        self._segments.append({'file': name, 'rows': len(rows), 'sha256': hashlib.sha256(data).hexdigest(), 'first_sequence_no': rows[0]['sequence_no'], 'last_sequence_no': rows[-1]['sequence_no']})
        self._pending = None
        self._pending_count = 0
        self._pending_ids.clear()
        self._write_manifest(complete=False)

    def checkpoint_progress(self, *, input_path: str | os.PathLike[str], processed_count: int, complete: bool=False, flush_pending: bool=False) -> Path:
        self.validate_progress_input(input_path)
        if processed_count < len(self._seen_ids) + self._pending_count:
            raise ValueError('sequence segment progress trails generated rows')
        if flush_pending or complete:
            self._commit_pending()
        if complete:
            self._complete = True
            self._write_manifest(complete=True)
        return self.manifest_path

    def close(self) -> None:
        if self._closed:
            return
        if self._pending is not None:
            self._pending.close()
        self._closed = True
        self._lock_file.close()
from collections import Counter, defaultdict, deque
from pathlib import PurePosixPath

def source_weighted_order(source_files, *, completed_indices=()):
    """Choose categories, then files, by largest integer proportional deficit.

    Source counts come from the exact selected inputs. Completed indices retain
    their credit when preparing a continuation; no prompt is duplicated or
    renumbered. This is submission ordering, not a guarantee about asynchronous
    completion order or GPU throughput.
    """
    if not isinstance(source_files, (list, tuple)) or not source_files:
        raise ValueError('nonempty selected source inventory required')
    labels = []
    for name in source_files:
        if not isinstance(name, str) or not name:
            raise ValueError('each record requires its original source file')
        parts = PurePosixPath(name).parts
        if len(parts) < 3:
            raise ValueError('source needs date/category/file provenance')
        labels.append(parts[1])
    completed = list(completed_indices)
    if any((type(i) is not int or not 0 <= i < len(labels) for i in completed)) or len(set(completed)) != len(completed):
        raise ValueError('completed indices must be unique original record IDs')
    completed = set(completed)
    category_counts = Counter(labels)
    source_counts = Counter(source_files)
    category_used = Counter((labels[i] for i in completed))
    source_used = Counter((source_files[i] for i in completed))
    groups = defaultdict(lambda: defaultdict(deque))
    for index, (category, source) in enumerate(zip(labels, source_files)):
        if index not in completed:
            groups[category][source].append(index)
    categories = sorted(groups)
    source_names = {category: sorted(groups[category]) for category in categories}
    result = []
    for total_used in range(len(completed), len(labels)):
        eligible = [c for c in categories if any((groups[c][s] for s in source_names[c]))]
        if not eligible:
            raise RuntimeError('source inventory exhausted before all records')
        category = max(eligible, key=lambda c: category_counts[c] * (total_used + 1) - category_used[c] * len(labels))
        sources = [s for s in source_names[category] if groups[category][s]]
        source = max(sources, key=lambda s: source_counts[s] * (category_used[category] + 1) - source_used[s] * category_counts[category])
        result.append(groups[category][source].popleft())
        category_used[category] += 1
        source_used[source] += 1
    if set(result) != set(range(len(labels))) - completed:
        raise RuntimeError('source order lost or repeated an original record')
    return result
import copy
from pathlib import Path, PurePosixPath

def source_weighted_input_digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('explicit SHA256 identity required')
    return value

class CheckedWeightedInput:

    def __init__(self, plan_path, *, expected_plan_sha256, expected_input_path):
        blob = Path(plan_path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != source_weighted_input_digest(expected_plan_sha256):
            raise ValueError('source order plan checksum differs')
        plan = json.loads(blob)
        if type(plan) is not dict or type(plan.get('schema_version')) is not int or plan.get('schema_version') != 1 or (plan.get('kind') != 'hierarchical-original-source-proportional-v1'):
            raise ValueError('unsupported source order plan')
        self._path = Path(expected_input_path).resolve()
        if Path(plan.get('input_path', '')).resolve() != self._path:
            raise ValueError('order plan points to a foreign input')
        source_sha = source_weighted_input_digest(plan.get('input_sha256'))
        initial_stat = self._path.stat()
        sha = hashlib.sha256()
        source_files, offsets, sizes, line_sha = ([], [], [], [])
        offset = 0
        with self._path.open('rb') as stream:
            for line in stream:
                sha.update(line)
                if line.strip():
                    record = json.loads(line)
                    if type(record) is not dict:
                        raise ValueError('prompt record must be a mapping')
                    source_files.append(record.get('source_file'))
                    offsets.append(offset)
                    sizes.append(len(line))
                    line_sha.append(hashlib.sha256(line).digest())
                offset += len(line)
        final_stat = self._path.stat()
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if signature(initial_stat) != signature(final_stat) or sha.hexdigest() != source_sha:
            raise ValueError('selected input changed or differs from the plan')
        n = len(source_files)
        if type(plan.get('selected_records')) is not int or plan['selected_records'] != n:
            raise ValueError('selected record count differs')
        permutation = plan.get('original_sequence_no')
        if type(permutation) is not list or len(permutation) != n or any((type(i) is not int or not 0 <= i < n for i in permutation)) or (len(set(permutation)) != n):
            raise ValueError('order plan lost or repeated an original identity')
        for key, expected in [('byte_offset_by_original_sequence_no', offsets), ('line_nbytes_by_original_sequence_no', sizes)]:
            value = plan.get(key)
            if type(value) is not list or any((type(i) is not int for i in value)) or value != expected:
                raise ValueError('order plan offsets do not match original record IDs')
        expected_order = source_weighted_order(source_files)
        if permutation != expected_order:
            raise ValueError('source order differs from the declared weighted algorithm')
        for key in ('category_counts', 'source_counts'):
            counts = plan.get(key)
            if type(counts) is not dict or any((type(k) is not str or type(v) is not int or v <= 0 for k, v in counts.items())):
                raise ValueError('order plan weights must be positive integer counts')
        categories = Counter((PurePosixPath(s).parts[1] for s in source_files))
        if plan.get('category_counts') != dict(categories) or plan.get('source_counts') != dict(Counter(source_files)):
            raise ValueError('order plan source weights differ from selected inputs')
        self._plan = copy.deepcopy(plan)
        self._original_sequence_no = tuple(permutation)
        self._offsets, self._sizes, self._line_sha = (tuple(offsets), tuple(sizes), tuple(line_sha))
        self._source_files = tuple(source_files)
        self._signature = signature(final_stat)

    @property
    def plan(self):
        return copy.deepcopy(self._plan)

    @property
    def path(self):
        return self._path

    @property
    def source_files(self):
        return self._source_files

    @property
    def offsets(self):
        return self._offsets

    @property
    def sizes(self):
        return self._sizes

    @property
    def line_sha(self):
        return self._line_sha

    @property
    def signature(self):
        return self._signature

    def iter_records(self, *, completed_indices=(), rebalance_completed_credit=False):
        if type(rebalance_completed_credit) is not bool:
            raise ValueError('continuation policy must be explicit boolean')
        completed = list(completed_indices)
        n = len(self._offsets)
        if any((type(i) is not int or not 0 <= i < n for i in completed)) or len(set(completed)) != len(completed):
            raise ValueError('invalid completed original identity')
        completed = set(completed)
        if rebalance_completed_credit:
            indices = source_weighted_order(self._source_files, completed_indices=completed)
        else:
            indices = [i for i in self._original_sequence_no if i not in completed]
        with self._path.open('rb') as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != self._signature:
                raise ValueError('input changed after source plan validation')
            for original_sequence_no in indices:
                stream.seek(self._offsets[original_sequence_no])
                line = stream.read(self._sizes[original_sequence_no])
                if hashlib.sha256(line).digest() != self._line_sha[original_sequence_no]:
                    raise ValueError('original prompt bytes changed after validation')
                yield (original_sequence_no, json.loads(line))
from collections.abc import Mapping
from verl_speco.config import config_int
from verl_speco.producer.input_reader import iter_input_records, _record_from_payload

def weighted_input_order_enabled(config):
    settings = config.get('input_order')
    if settings is None:
        return False
    if not isinstance(settings, Mapping):
        raise ValueError('input_order must be a mapping')
    unknown = set(settings) - {'enabled', 'mode', 'plan_path', 'plan_sha256', 'rebalance_completed_credit'}
    if unknown:
        raise ValueError('unsupported input_order fields: ' + ', '.join(sorted(unknown)))
    enabled = settings.get('enabled', True)
    if type(enabled) is not bool:
        raise ValueError('input_order.enabled must be boolean')
    if not enabled:
        return False
    if settings.get('mode') != 'source_proportional':
        raise ValueError('unsupported input_order mode')
    if not isinstance(settings.get('plan_path'), (str, Path)) or not str(settings['plan_path']):
        raise ValueError('input_order requires an explicit plan path')
    source_weighted_input_digest(settings.get('plan_sha256'))
    if type(settings.get('rebalance_completed_credit', True)) is not bool:
        raise ValueError('input_order.rebalance_completed_credit must be boolean')
    return True

def sequence_only_input_records(config, *, writer=None):
    """Preserve generation IDs and completed credit; order only submissions."""
    if not weighted_input_order_enabled(config):
        return iter_input_records(str(config['input_path']), on_error=str(config.get('on_error', 'skip') or 'skip'), max_consecutive_errors=config_int(config, 'max_consecutive_errors', 20), parser_strict_roles=bool(config.get('parser_strict_roles', False)))
    settings = config['input_order']
    reader = CheckedWeightedInput(settings['plan_path'], expected_plan_sha256=settings['plan_sha256'], expected_input_path=config['input_path'])
    strict = bool(config.get('parser_strict_roles', False))

    def normalized(pairs):
        for original_index, payload in pairs:
            yield _record_from_payload(payload, f'{reader.path}:original-sequence-{original_index}', original_index, strict)
    completed_records = []
    rebalance = settings.get('rebalance_completed_credit', True)
    if rebalance and writer is not None and writer.persisted_record_count:
        for record in normalized(reader.iter_records()):
            if writer.has_record(sequence_no=record.sequence_no, sample_id=record.sample_id):
                completed_records.append(record)
    completed = [record.sequence_no for record in completed_records]

    def records():
        yield from completed_records
        yield from normalized(reader.iter_records(completed_indices=completed, rebalance_completed_credit=bool(completed)))
    return records()
import asyncio
from typing import Any, Mapping, TYPE_CHECKING
from verl_speco.producer.input_reader import prepare_generation_request, prepare_generated_replay_request, tokenize_record
from verl_speco.producer.vllm_feature_client import VllmEndpoint, VllmFeatureClientPool
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from verl_speco.standalone_tq_producer import ProducerStats

def sequence_only_sequence_only_sampling_params(producer_cfg: Mapping[str, Any]) -> dict[str, Any]:
    generation_cfg = producer_cfg.get('generation') or {}
    if not isinstance(generation_cfg, Mapping):
        raise TypeError('standalone_tq_producer.generation must be a mapping')
    params: dict[str, Any] = {}
    for name in ('temperature', 'top_p', 'top_k', 'seed'):
        value = generation_cfg.get(name)
        if value is not None:
            params[name] = value
    if 'temperature' in params and (not 0 <= float(params['temperature'])):
        raise ValueError('generation.temperature must be non-negative')
    if 'top_p' in params and (not 0 < float(params['top_p']) <= 1):
        raise ValueError('generation.top_p must be in (0, 1]')
    if 'top_k' in params and int(params['top_k']) == 0:
        raise ValueError('generation.top_k must be -1 or a positive integer')
    return params

def sequence_only_validate_sequence_only_config(producer_cfg: Mapping[str, Any], tq_cfg: Mapping[str, Any]) -> None:
    required = ('input_path', 'tokenizer_path', 'vllm_model', 'generation_max_tokens')
    missing = [name for name in required if not producer_cfg.get(name)]
    if missing:
        raise ValueError(f'sequence_only producer missing required fields: {missing}')
    endpoints = producer_cfg.get('vllm_endpoints')
    if not isinstance(endpoints, list) or not endpoints or (not all(endpoints)):
        raise ValueError('sequence_only.vllm_endpoints must be a non-empty list')
    replay_cfg = producer_cfg.get('sequence_replay') or {}
    if not isinstance(replay_cfg, Mapping) or not replay_cfg.get('enabled', False):
        raise ValueError('sequence_only requires sequence_replay.enabled=true')
    if not replay_cfg.get('output_dir'):
        raise ValueError('sequence_only requires sequence_replay.output_dir')
    if not str(replay_cfg.get('shard_id', '')).strip():
        raise ValueError('sequence_only requires a unique sequence_replay.shard_id')
    if not str(replay_cfg.get('run_id') or tq_cfg.get('run_id') or '').strip():
        raise ValueError('sequence_only requires sequence_replay.run_id or transfer_queue.run_id')
    if int(replay_cfg.get('progress_interval_samples', 100)) <= 0:
        raise ValueError('sequence_only.sequence_replay.progress_interval_samples must be positive')
    if int(replay_cfg.get('commit_interval_samples', 0)) < 0:
        raise ValueError('sequence_only.sequence_replay.commit_interval_samples cannot be negative')
    for name in ('max_inflight_requests', 'per_endpoint_concurrency'):
        if int(producer_cfg.get(name, 0)) <= 0:
            raise ValueError(f'sequence_only.{name} must be positive')
    if float(producer_cfg.get('request_timeout', 0)) <= 0:
        raise ValueError('sequence_only.request_timeout must be positive')
    if int(producer_cfg.get('generation_max_tokens', 0)) <= 0:
        raise ValueError('sequence_only.generation_max_tokens must be positive')
    sequence_only_sequence_only_sampling_params(producer_cfg)
    weighted_input_order_enabled(producer_cfg)

async def run_sequence_only(config: Any, *, tokenizer: Any | None=None, client_pool: Any | None=None) -> ProducerStats:
    """Generate target responses and persist replay tokens without hidden/TQ."""
    producer_cfg, _, tq_cfg = sequence_only_config_sections(config)
    sequence_only_validate_sequence_only_config(producer_cfg, tq_cfg)
    replay_cfg = producer_cfg['sequence_replay']
    run_id = str(replay_cfg.get('run_id') or tq_cfg.get('run_id'))
    concurrency = int(producer_cfg['max_inflight_requests'])
    commit_interval = int(replay_cfg.get('commit_interval_samples', 0))
    if commit_interval:
        writer = CommittedSequenceSegmentWriter(str(replay_cfg['output_dir']), input_path=str(producer_cfg['input_path']), run_id=run_id, shard_id=str(replay_cfg['shard_id']), segment_size=commit_interval, allow_out_of_order=concurrency > 1 or weighted_input_order_enabled(producer_cfg))
    else:
        writer = JsonlSequenceReplayWriter(str(replay_cfg['output_dir']), run_id=run_id, shard_id=str(replay_cfg['shard_id']), fsync_interval=int(replay_cfg.get('fsync_interval', 100)))
    writer.validate_progress_input(str(producer_cfg['input_path']))
    pool = client_pool
    from verl_speco.standalone_tq_producer import ProducerStats
    stats = ProducerStats()
    max_samples = int(producer_cfg.get('max_samples', 0) or 0)
    progress_interval = int(replay_cfg.get('progress_interval_samples', 100))
    completed_sequences = 0
    initial_saved_sequences = writer.persisted_record_count
    sampling_params = sequence_only_sequence_only_sampling_params(producer_cfg)
    algorithm = str(sequence_only_config_sections(config)[1].get('speculative_algorithm') or 'EAGLE3')
    replay_token_config = dict(producer_cfg)
    replay_token_config.update(max_feature_length=0, min_supervised_tokens=0, on_truncated_supervision='keep')
    try:
        if tokenizer is None:
            tokenizer = await asyncio.to_thread(sequence_only_load_tokenizer, producer_cfg)
        if pool is None:
            pool = VllmFeatureClientPool([VllmEndpoint(str(url).rstrip('/'), int(producer_cfg['per_endpoint_concurrency'])) for url in producer_cfg['vllm_endpoints']], model=str(producer_cfg['vllm_model']), max_inflight_requests=int(producer_cfg['max_inflight_requests']), request_timeout=float(producer_cfg['request_timeout']), success_log_interval=int(producer_cfg.get('vllm_success_log_interval', 100)))
        source_records = sequence_only_input_records(producer_cfg, writer=writer)
        await pool.start()
        generated_only = bool(replay_cfg.get('generated_only', True))
        pending: dict[asyncio.Task, int] = {}
        checkpoint_due = False

        def checkpoint(*, complete: bool=False, flush_pending: bool=False) -> None:
            processed = max(stats.input_count, writer.persisted_record_count)
            extra = {'flush_pending': flush_pending} if isinstance(writer, CommittedSequenceSegmentWriter) else {}
            writer.checkpoint_progress(input_path=str(producer_cfg['input_path']), processed_count=processed, complete=complete, **extra)

        async def generate(record):
            target_generated = record.response is None
            if target_generated:
                generation_request = prepare_generation_request(record, tokenizer, producer_cfg)
                generated = await pool.generate_tokens(generation_request, sampling_params=sampling_params)
                request = prepare_generated_replay_request(generation_request, generated.generated_token_ids, producer_cfg)
                endpoint_url = generated.endpoint_url
            else:
                request = tokenize_record(record, tokenizer, replay_token_config)
                endpoint_url = None
            return (request, target_generated, endpoint_url)

        def persist(task):
            nonlocal completed_sequences, checkpoint_due
            request, target_generated, endpoint_url = task.result()
            wrote = writer.write(request, algorithm=algorithm, metadata={'commit_state': 'sequence_only_generated', 'target_generated': target_generated, 'target_model_id': producer_cfg.get('target_model_id'), 'target_model_revision': producer_cfg.get('target_model_revision'), 'tokenizer_fingerprint': producer_cfg.get('tokenizer_fingerprint'), 'vllm_endpoint': endpoint_url, 'sampling_params': dict(sampling_params)})
            if wrote:
                stats.saved_sequence_count += 1
            completed_sequences += 1
            if completed_sequences % progress_interval == 0:
                checkpoint_due = True

        async def drain():
            nonlocal checkpoint_due
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            error = None
            for task in sorted(done, key=pending.__getitem__):
                pending.pop(task)
                try:
                    persist(task)
                except BaseException as caught:
                    if error is None:
                        error = caught
            if error is not None:
                raise error
            if checkpoint_due:
                checkpoint(flush_pending=concurrency > 1)
                checkpoint_due = False
        try:
            for record in source_records:
                if max_samples > 0 and initial_saved_sequences + stats.saved_sequence_count + len(pending) >= max_samples:
                    break
                stats.input_count += 1
                if record.response is not None and generated_only:
                    continue
                if writer.has_record(sequence_no=record.sequence_no, sample_id=record.sample_id):
                    completed_sequences += 1
                    if completed_sequences % progress_interval == 0:
                        checkpoint()
                    continue
                pending[asyncio.create_task(generate(record))] = record.sequence_no
                if len(pending) >= concurrency:
                    await drain()
            while pending:
                await drain()
        except BaseException as error:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in sorted(pending, key=pending.__getitem__):
                if not task.cancelled() and task.exception() is None:
                    try:
                        persist(task)
                    except BaseException as save_error:
                        error.add_note(f'Saving a completed sequence also failed: {save_error}')
            if concurrency > 1:
                try:
                    checkpoint(flush_pending=True)
                except BaseException as save_error:
                    error.add_note(f'Saving completed-sequence progress also failed: {save_error}')
            raise
        checkpoint(complete=True)
        logger.info('Sequence-only Producer completed inputs=%s saved_sequences=%s total_sequences=%s output_dir=%s sampling_params=%s', stats.input_count, stats.saved_sequence_count, initial_saved_sequences + stats.saved_sequence_count, replay_cfg['output_dir'], sampling_params)
        return stats
    finally:
        try:
            if pool is not None:
                await pool.close()
        finally:
            writer.close()

def sequence_only_config_sections(config: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plain = sequence_only_plain_config(config)
    try:
        producer_cfg = plain['speco']['standalone_tq_producer']
        drafter = plain.get('actor_rollout_ref', {}).get('rollout', {}).get('drafter', {}) or {}
        training_cfg = drafter.get('training') or {}
        tq_cfg = training_cfg.get('transfer_queue') or {}
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Producer configuration missing section {exc}') from exc
    if not all((isinstance(value, dict) for value in (producer_cfg, training_cfg, tq_cfg))):
        raise TypeError('Producer configuration sections must resolve to mappings')
    return (producer_cfg, {**training_cfg, 'speculative_algorithm': drafter.get('speculative_algorithm')}, tq_cfg)

def sequence_only_plain_config(config: Any) -> dict[str, Any]:
    value = config
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(config):
            value = OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass
    if not isinstance(value, Mapping):
        raise TypeError('Producer configuration must be a mapping')
    return dict(value)

def sequence_only_load_tokenizer(config: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Standalone TQ Producer requires transformers') from exc
    return AutoTokenizer.from_pretrained(str(config['tokenizer_path']), trust_remote_code=bool(config.get('trust_remote_code', False)))
from verl_speco.runtime.artifacts import read_published_json

def sequence_resume_state_sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def active_replica_indices(config: dict) -> tuple[int, ...]:
    indices = config.get('active_replica_indices', list(range(16)))
    if not isinstance(indices, list) or not indices or any((type(index) is not int or not 0 <= index < 16 for index in indices)) or (indices != sorted(set(indices))):
        raise ValueError('active sequence replicas must be sorted unique indices within 0..15')
    return tuple(indices)

def sequence_resume_state_configuration(path: Path) -> tuple[dict, Path]:
    config = json.loads(path.read_text())
    output = Path(config['output_root'])
    hosts = config.get('hosts', [])
    if not output.is_absolute() or len(hosts) != 8 or len(set(hosts)) != 8 or (not all((isinstance(host, str) and host for host in hosts))) or (not isinstance(config.get('run_id'), str)) or (not config['run_id']):
        raise ValueError('paused sequence configuration identity is invalid')
    active_replica_indices(config)
    return (config, output)

def sequence_resume_state_manifest(output: Path, shard: str, run_id: str) -> tuple[dict, int, int]:
    manifest = read_published_json(output / 'sequences' / f'part-{shard}.manifest.json')
    count = manifest.get('committed_rows')
    if manifest.get('run_id') != run_id or manifest.get('shard_id') != shard or type(count) is not int or (count < 0) or (not isinstance(manifest.get('segments'), list)):
        raise ValueError('paused sequence manifest identity or count differs')
    order = manifest.get('record_order', 'input_prefix')
    if order not in ('input_prefix', 'completion'):
        raise ValueError('paused sequence record order is invalid')
    seen = set()
    cursor = reply_tokens = all_tokens = 0
    for segment in manifest['segments']:
        name = segment.get('file', '')
        if not isinstance(name, str) or Path(name).name != name or (not name.startswith(f'part-{shard}-segment-')) or (not name.endswith('.jsonl')) or (type(segment.get('rows')) is not int) or (segment['rows'] <= 0) or (order == 'input_prefix' and (segment.get('first_sequence_no') != cursor or segment.get('last_sequence_no') != cursor + segment['rows'] - 1)):
            raise ValueError('paused sequence segment is outside the contiguous committed prefix')
        path = output / 'sequences' / name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != segment.get('sha256'):
            raise ValueError('paused sequence segment checksum differs')
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        del payload
        if len(rows) != segment['rows']:
            raise ValueError('paused sequence segment row count differs')
        if rows[0].get('sequence_no') != segment.get('first_sequence_no') or rows[-1].get('sequence_no') != segment.get('last_sequence_no'):
            raise ValueError('paused sequence segment endpoint differs')
        for row in rows:
            number = row.get('sequence_no')
            ids, mask = (row.get('input_ids'), row.get('loss_mask'))
            if row.get('run_id') != run_id or row.get('shard_id') != name.removeprefix('part-').removesuffix('.jsonl') or type(number) is not int or (number < 0) or (number in seen) or (order == 'input_prefix' and number != cursor) or (not isinstance(ids, list)) or (not isinstance(mask, list)) or (len(ids) != len(mask)) or (not ids) or any((value not in (0, 1) for value in mask)):
                raise ValueError('paused sequence row identity or tokens differ')
            seen.add(number)
            cursor += 1
            reply_tokens += int(sum(mask))
            all_tokens += len(ids)
    if cursor != count:
        raise ValueError('paused sequence committed prefix count differs')
    return (manifest, reply_tokens, all_tokens)

def sequence_resume_state_committed_or_empty(output: Path, shard: str, run_id: str):
    try:
        read_published_json(output / 'sequences' / f'part-{shard}.manifest.json')
    except FileNotFoundError:
        if list((output / 'sequences').glob(f'part-{shard}-segment-*.jsonl')):
            raise ValueError('sequence segments exist without a committed manifest')
        return (None, 0, 0)
    return sequence_resume_state_manifest(output, shard, run_id)

def create_pause_resume_state(config_path: Path) -> dict[str, Any]:
    """Snapshot only after the caller has stopped and verified all owned writers."""
    config, output = sequence_resume_state_configuration(config_path)
    shards = []
    reply = tokens = rows = 0
    for index in active_replica_indices(config):
        shard = f'node-{index // 2}-replica-{index % 2}'
        manifest, local_reply, local_tokens = sequence_resume_state_committed_or_empty(output, shard, config['run_id'])
        prompt = output / 'prompts' / f'prompts-{index:02d}.jsonl'
        stat = prompt.stat()
        fingerprint = manifest.get('input_fingerprint', {}) if manifest is not None else dict(path=str(prompt.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if fingerprint.get('path') != str(prompt.resolve()) or fingerprint.get('size_bytes') != stat.st_size or fingerprint.get('mtime_ns') != stat.st_mtime_ns:
            raise ValueError('paused prompt fingerprint differs from generation input')
        count = manifest['committed_rows'] if manifest is not None else 0
        resume_cursor = 0 if manifest is None or manifest.get('record_order') == 'completion' else count
        shards.append(dict(shard_id=shard, committed_rows=count, next_sequence_no=resume_cursor, input_fingerprint=fingerprint, input_sha256=sequence_resume_state_sha(prompt), manifest=manifest))
        rows += count
        reply += local_reply
        tokens += local_tokens
    return dict(schema_version=1, state='PAUSED', run_id=config['run_id'], config_path=str(config_path.resolve()), config_sha256=sequence_resume_state_sha(config_path), sequences_directory=str(output / 'sequences'), shards=shards, committed_sequences=rows, generated_tokens=reply, total_tokens_with_prompt=tokens)

def validate_pause_resume_state(state: dict, config_path: Path) -> None:
    """Reject changed input or lost commits; permit a newer committed extension."""
    config, output = sequence_resume_state_configuration(config_path)
    shards = state.get('shards', [])
    if state.get('schema_version') != 1 or state.get('state') != 'PAUSED' or state.get('run_id') != config['run_id'] or (state.get('config_path') != str(config_path.resolve())) or (state.get('config_sha256') != sequence_resume_state_sha(config_path)) or (state.get('sequences_directory') != str(output / 'sequences')) or (len(shards) != len(active_replica_indices(config))):
        raise ValueError('paused resume configuration identity differs')
    count = 0
    for index, saved in zip(active_replica_indices(config), shards):
        shard = f'node-{index // 2}-replica-{index % 2}'
        cursor = saved.get('committed_rows')
        old = saved.get('manifest', {})
        resume_cursor = 0 if old is None or old.get('record_order') == 'completion' else cursor
        if saved.get('shard_id') != shard or type(cursor) is not int or cursor < 0 or (saved.get('next_sequence_no') != resume_cursor):
            raise ValueError('paused resume shard cursor differs')
        prompt = output / 'prompts' / f'prompts-{index:02d}.jsonl'
        if sequence_resume_state_sha(prompt) != saved.get('input_sha256'):
            raise ValueError('paused prompt checksum differs')
        current, _, _ = sequence_resume_state_committed_or_empty(output, shard, config['run_id'])
        if old is None:
            stat = prompt.stat()
            fingerprint = dict(path=str(prompt.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if cursor != 0 or saved.get('input_fingerprint') != fingerprint or (current is not None and current.get('input_fingerprint') != fingerprint):
                raise ValueError('empty paused input or cursor differs')
            continue
        if current is None:
            raise ValueError('paused committed prefix was lost')
        old = saved.get('manifest', {})
        if old.get('committed_rows') != cursor or old.get('shard_id') != shard or old.get('run_id') != config['run_id'] or (saved.get('input_fingerprint') != old.get('input_fingerprint')) or (current.get('record_order', 'input_prefix') != old.get('record_order', 'input_prefix')) or (current.get('input_fingerprint') != old.get('input_fingerprint')) or (current['committed_rows'] < cursor) or (not isinstance(old.get('segments'), list)) or (current['segments'][:len(old['segments'])] != old['segments']) or (sum((segment['rows'] for segment in old['segments'])) != cursor):
            raise ValueError('paused committed prefix was changed or lost')
        count += cursor
    if state.get('committed_sequences') != count:
        raise ValueError('paused resume total count differs')
