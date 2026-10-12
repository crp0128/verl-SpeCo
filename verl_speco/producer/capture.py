# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import torch

class BoundedSourcePublications:

    def __init__(self, *, publish, max_bytes, initialize=None):
        if not callable(publish) or type(max_bytes) is not int or (not 0 < max_bytes <= 4 * 1024 ** 3) or (initialize is not None and (not callable(initialize))):
            raise ValueError('source publication needs a bounded two-capture CPU policy')
        self._publish = publish
        self._budget = max_bytes
        self._jobs = deque()
        self._bytes = 0
        self._peak = 0
        self._failure = None
        self._closed = False
        self._busy = False
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=initialize, thread_name_prefix='speco-source-publication')

    def _enter(self):
        with self._lock:
            if self._closed or self._busy:
                raise RuntimeError('source publication owner is closed or already running')
            self._busy = True

    def _leave(self):
        with self._lock:
            self._busy = False

    def submit(self, key, metadata, fields):
        self._enter()
        try:
            if self._failure is not None:
                raise RuntimeError('source publication owner is poisoned') from self._failure
            if any((job['key'] == key for job in self._jobs)):
                raise ValueError('source publication key is already owned')
            if not isinstance(key, str) or not key or (not isinstance(fields, dict)) or (set(fields) != {'input_ids', 'position_ids', 'loss_mask', 'hidden_states'}) or any((not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or (not t.is_contiguous()) for t in fields.values())):
                raise ValueError('source publication needs independent contiguous CPU fields')
            size = sum((t.numel() * t.element_size() for t in fields.values()))
            if type(metadata.payload_nbytes) is not int or size != metadata.payload_nbytes or size > self._budget:
                raise ValueError('source capture exceeds publication byte budget')
            if len(self._jobs) >= 2 or self._bytes + size > self._budget:
                raise RuntimeError('source publication queue is full')
            future = self._executor.submit(self._publish, key, metadata, fields)
            self._jobs.append(dict(key=key, bytes=size, future=future))
            self._bytes += size
            self._peak = max(self._peak, self._bytes)
        finally:
            self._leave()

    def get(self, key):
        self._enter()
        try:
            if not self._jobs or key != self._jobs[0]['key']:
                raise RuntimeError('source publication manifest must be consumed in FIFO order')
            job = self._jobs[0]
            try:
                result = job['future'].result()
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
                for sibling in self._jobs:
                    try:
                        sibling['future'].result()
                    except BaseException:
                        pass
                raise
            else:
                self._jobs.popleft()
                self._bytes -= job['bytes']
                return result
        finally:
            self._leave()

    def wait_all(self):
        """Fence native mutations while preserving FIFO result ownership."""
        self._enter()
        try:
            failure = self._failure
            for job in self._jobs:
                try:
                    job['future'].result()
                except BaseException as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                self._failure = failure
                raise failure
        finally:
            self._leave()

    @property
    def pending_keys(self):
        return tuple((job['key'] for job in self._jobs))

    @property
    def closed(self):
        return self._closed

    @property
    def pending_bytes(self):
        return self._bytes

    @property
    def peak_pending_bytes(self):
        return self._peak

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('join source publication callback before close')
            if self._closed:
                return
            self._closed = True
        try:
            for job in self._jobs:
                try:
                    job['future'].result()
                except BaseException as error:
                    if self._failure is None:
                        self._failure = error
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._jobs.clear()
            self._bytes = 0
            self._publish = None
import hashlib
import json

def prefill_digest(result, *, token_count=None):
    count = len(result.token_ids) if token_count is None else token_count
    if type(count) is not int or not 0 < count <= len(result.token_ids):
        raise ValueError('prefill digest requires a nonempty captured prefix')
    if len(result.positions) != len(result.token_ids) or len(result.hidden_states) != len(result.layer_ids):
        raise ValueError('prefill digest identity dimensions differ')
    tensors = (*result.hidden_states, result.final_hidden_state)
    if result.final_hidden_state.ndim != 2 or result.final_hidden_state.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('prefill digest requires a floating feature matrix')
    width = result.final_hidden_state.shape[-1]
    dtype = result.final_hidden_state.dtype
    digest = hashlib.sha256(json.dumps({'schema': 'captured-prefill-v1', 'tokens': result.token_ids[:count], 'positions': result.positions[:count], 'layers': result.layer_ids, 'width': width, 'dtype': str(dtype)}, sort_keys=True, separators=(',', ':')).encode())
    for tensor in tensors:
        if tensor.dtype != dtype or tuple(tensor.shape) != (len(result.token_ids), width):
            raise ValueError('prefill digest requires complete consistently typed feature rows')
        for start in range(0, count, 4096):
            chunk = tensor[start:min(start + 4096, count)].detach().to('cpu').contiguous()
            digest.update(memoryview(chunk.view(torch.uint8).numpy()).cast('B'))
    return {'prefill_sha256': digest.hexdigest(), 'prefill_token_count': count}
from collections import OrderedDict
from copy import deepcopy
from verl_speco.producer.replay import ReplayBatch
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key
from verl_speco.trainer.feature_pipeline import packed_control_sha256

class EpochNativeBatches:

    def __init__(self, catalog, *, run_id, generation, input_fingerprints, producer_incarnations, max_sequence_length=16384, cache_batches=2):
        self.catalog, self.plan = (catalog, catalog.plan)
        if type(generation) is not int or generation < 1 or type(cache_batches) is not int or (not 1 <= cache_batches <= 3) or (type(max_sequence_length) is not int) or (not 2 <= max_sequence_length <= 131072) or (not isinstance(producer_incarnations, (tuple, list))) or (len(producer_incarnations) != self.plan.producer_count) or any((not isinstance(v, str) or not v for v in producer_incarnations)) or (len(set(producer_incarnations)) != len(producer_incarnations)) or (set(input_fingerprints) != {'target_model_fingerprint', 'tokenizer_fingerprint', 'dataset_fingerprint'}) or any((not isinstance(v, str) or not v for v in input_fingerprints.values())):
            raise ValueError('invalid bounded native epoch input identity')
        self.run_id, self.generation = (run_id, generation)
        self.identity = deepcopy(input_fingerprints)
        self.incarnations = tuple(producer_incarnations)
        self.context, self.cache_batches = (max_sequence_length, cache_batches)
        self.cache, self.closed, self.failed = (OrderedDict(), False, False)
        self._metadata(self.plan.occurrence(0)).validate()

    def _metadata(self, occurrence):
        row = self.catalog.index['entries'][occurrence.source_index]
        if row['id'] != occurrence.original_sample_id or row['sequence_fingerprint'] != occurrence.sequence_fingerprint or row['original_tokens'] != occurrence.original_tokens or (row['response_tokens'] < 2) or (row['original_tokens'] > self.context):
            raise ValueError('native epoch row differs from the admitted complete training snapshot')
        producer = occurrence.sequence_no % self.plan.producer_count
        return SampleMetadataV3(schema_version=3, run_id=self.run_id, generation=self.generation, producer_id=f'producer-{producer}', producer_incarnation=self.incarnations[producer], request_id='speco-92b-capture-smoke', sample_id=occurrence.sample_id, sequence_no=occurrence.sequence_no, target_revision='92b-sft-v11', data_version=0, target_layer_ids=(1, 12, 22, 32, 43), hidden_states_layout='dflash_aux_plus_last', hidden_dtype='bfloat16', token_count=row['original_tokens'], prompt_token_count=row['prompt_tokens'], supervised_token_count=row['response_tokens'], payload_nbytes=row['original_tokens'] * 30740, **self.identity)

    def batch(self, step_index):
        if self.closed or self.failed:
            raise RuntimeError('bounded native epoch input is closed or failed')
        if type(step_index) is not int or not 0 <= step_index < self.plan.total_steps:
            raise ValueError('native epoch step is outside its frozen plan')
        start = step_index * self.plan.samples_per_step
        end = min(start + self.plan.samples_per_step, self.plan.total_samples)
        return ReplayBatch(step_index, start, end, tuple((self.plan.occurrence(i) for i in range(start, end))))

    def packed_control(self, step_index):
        self.batch(step_index)
        if step_index not in self.cache:
            try:
                samples = []
                for item in self.batch(step_index).occurrences:
                    metadata = self._metadata(item)
                    samples.append(dict(key=make_v3_sample_key(metadata), metadata=metadata.to_dict()))
                control = dict(schema_version=1, run_id=self.run_id, generation=self.generation, samples=samples)
                self.cache[step_index] = control
                while len(self.cache) > self.cache_batches:
                    self.cache.popitem(last=False)
            except BaseException:
                self.cache.clear()
                self.failed = True
                raise
        self.cache.move_to_end(step_index)
        return deepcopy(self.cache[step_index])

    def control_sha256(self, step_index):
        return packed_control_sha256(self.packed_control(step_index))

    def requests(self, step_index):
        """Read complete token rows for this batch only; never cache token tensors."""
        try:
            return tuple((self.catalog.request(item.sequence_no, config={'min_supervised_tokens': 1, 'max_sequence_length': self.context}) for item in self.batch(step_index).occurrences))
        except BaseException:
            self.cache.clear()
            self.failed = True
            raise

    def close(self):
        self.cache.clear()
        self.closed = True
import argparse
import os
from pathlib import Path
import tempfile
from verl_speco.runtime.artifacts import published_json_paths, read_published_json, publish_file, publish_json

def materialize_sequence_snapshot(source_dir: Path, output_path: Path, *, max_rows: int | None=None) -> dict:
    return materialize_sequence_snapshot_materialize((source_dir,), output_path, max_rows=max_rows, verify_identity=False)

def materialize_sequence_snapshots(source_dirs: tuple[Path, ...], output_path: Path, *, max_rows: int | None=None) -> dict:
    """Merge committed generation runs, rejecting duplicate or mixed identities.

    Pin each manifest once before reading payloads. Later generation appends
    cannot expand this training snapshot; uncommitted segments are ignored.
    """
    return materialize_sequence_snapshot_materialize(source_dirs, output_path, max_rows=max_rows, verify_identity=True)

def materialize_sequence_snapshot_materialize(source_dirs, output_path, *, max_rows, verify_identity):
    source_dirs = tuple((Path(p).resolve() for p in source_dirs))
    if not source_dirs or len(set(source_dirs)) != len(source_dirs):
        raise ValueError('snapshot source directories must be nonempty and unique')
    if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
        raise ValueError('max_rows must be positive')
    manifests = []
    for source_dir in source_dirs:
        paths = published_json_paths(source_dir, 'part-*.manifest.json')
        if not paths:
            raise ValueError('committed manifests are required for every source directory')
        manifests.extend(((source_dir, path, read_published_json(path)) for path in paths))
    if not manifests or output_path.exists():
        raise ValueError('committed manifests are required and output must be new')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    rows = 0
    segments = 0
    seen_ids, seen_fingerprints, seen_segments = (set(), set(), set())
    provenance = None
    generated_tokens = total_tokens = 0
    with tempfile.NamedTemporaryFile('wb', dir=output_path.parent, prefix=f'.{output_path.name}-', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for source_dir, manifest_path, manifest in manifests:
                if max_rows is not None and rows >= max_rows:
                    break
                raw_manifest = (json.dumps(manifest, sort_keys=True) + '\n').encode()
                source_digest.update(str(manifest_path).encode())
                source_digest.update(raw_manifest)
                for descriptor in manifest['segments']:
                    if max_rows is not None and rows >= max_rows:
                        break
                    name = descriptor['file']
                    if not isinstance(name, str) or Path(name).name != name:
                        raise ValueError('segment path escapes its manifest directory')
                    segment = source_dir / name
                    if segment.resolve() in seen_segments or not segment.resolve().is_relative_to(source_dir):
                        raise ValueError('duplicate or escaping committed segment')
                    seen_segments.add(segment.resolve())
                    data = segment.read_bytes()
                    if hashlib.sha256(data).hexdigest() != descriptor['sha256']:
                        raise ValueError(f'committed segment checksum differs: {segment}')
                    actual_rows = sum((bool(line.strip()) for line in data.splitlines()))
                    if actual_rows != descriptor['rows']:
                        raise ValueError(f'committed segment row count differs: {segment}')
                    if max_rows is not None:
                        lines = [line for line in data.splitlines(keepends=True) if line.strip()]
                        data = b''.join(lines[:max_rows - rows])
                        actual_rows = min(actual_rows, max_rows - rows)
                    if verify_identity:
                        from verl_speco.producer.replay import sequence_replay_writer_replay_fingerprint as _replay_fingerprint
                        for line in data.splitlines():
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            if not isinstance(row, dict) or not isinstance(row.get('metadata'), dict):
                                raise ValueError('invalid committed sequence schema')
                            md = row['metadata']
                            identity = tuple((md.get(k) for k in ('target_model_id', 'target_model_revision', 'tokenizer_fingerprint')))
                            fp = row.get('sequence_fingerprint')
                            sample_id = row.get('id')
                            ids, mask, positions = (row.get(k) for k in ('input_ids', 'loss_mask', 'position_ids'))
                            if row.get('schema_version') != 1 or row.get('sample_type') != 'token_replay' or row.get('algorithm') != 'DSPARK' or (md.get('commit_state') != 'sequence_only_generated') or (md.get('target_generated') is not True) or any((not isinstance(k, str) or not k for k in identity)) or (not isinstance(sample_id, str)) or (not sample_id) or (fp != 'sha256:' + _replay_fingerprint(row)) or (not all((isinstance(v, list) for v in (ids, mask, positions)))) or (len(ids) < 2) or (len(ids) != len(mask)) or (len(ids) != len(positions)) or any((type(v) is not int or v < 0 for v in ids)) or any((type(v) is not int for v in positions)) or (positions != list(range(len(ids)))) or any((type(v) not in (int, float) or v not in (0, 1) for v in mask)) or (not sum(mask)) or (mask[0] != 0) or (mask != sorted(mask)):
                                raise ValueError('invalid committed sequence identity or alignment')
                            if provenance is not None and identity != provenance:
                                raise ValueError('sequence target/tokenizer provenance differs')
                            if sample_id in seen_ids or fp in seen_fingerprints:
                                raise ValueError('duplicate committed sequence identity')
                            provenance = identity
                            seen_ids.add(sample_id)
                            seen_fingerprints.add(fp)
                            generated_tokens += int(sum(mask))
                            total_tokens += len(ids)
                    if not data.endswith(b'\n'):
                        data += b'\n'
                    handle.write(data)
                    output_digest.update(data)
                    rows += actual_rows
                    segments += 1
            if max_rows is not None and rows < max_rows:
                raise ValueError('too few committed sequence rows for requested snapshot')
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    if rows == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError('no committed sequence rows are available')
    publish_file(temporary, output_path)
    report = {'schema_version': 1, 'source_dir': str(source_dirs[0]), 'source_manifest_sha256': source_digest.hexdigest(), 'output_path': str(output_path.resolve()), 'output_sha256': output_digest.hexdigest(), 'manifests': len(manifests), 'segments': segments, 'rows': rows}
    if verify_identity:
        report.update(source_dirs=[str(p) for p in source_dirs], duplicate_sequences=0, target_model_id=provenance[0], target_model_revision=provenance[1], tokenizer_fingerprint=provenance[2], generated_tokens=generated_tokens, original_tokens=total_tokens)
    report_path = output_path.with_suffix(output_path.suffix + '.source.json')
    publish_json(report_path, report)
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True, action='append', help='Repeat to merge committed runs with strict identity checks')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, help='Freeze exactly this many committed rows')
    args = parser.parse_args()
    materializer = materialize_sequence_snapshot if len(args.source_dir) == 1 else materialize_sequence_snapshots
    source = args.source_dir[0] if len(args.source_dir) == 1 else tuple(args.source_dir)
    print(json.dumps(materializer(source, args.output, max_rows=args.max_rows), sort_keys=True), flush=True)

def native_capture_source_slots(environment):
    policy = environment.get('SPECO_CAPTURE_NATIVE_SOURCE_SLOTS', '')
    if policy not in ('', '1', '2', '4', '8', '16'):
        raise ValueError('native capture source slots must be 1, 2, 4, 8 or 16')
    if policy in ('', '1'):
        return 1
    if environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1' or environment.get('SPECO_CAPTURE_REUSE_ENGINE') != '1' or environment.get('SPECO_CAPTURE_PHYSICAL_BATCH_SIZE') != policy:
        raise ValueError('native source slots require a matching persistent physical capture batch')
    return int(policy)

def native_capture_source_slot_bytes(slot_count):
    if type(slot_count) is not int or slot_count not in (1, 2, 4, 8, 16):
        raise ValueError('invalid fixed native source slot count')
    return 256 * 1024 ** 2 if slot_count == 16 else 512 * 1024 ** 2 if slot_count >= 4 else 2 * 1024 ** 3

def preallocate_native_capture_sources(transport, slot_count, *, slot_bytes=None):
    """Register every fixed source range before any remote lease can connect."""
    fixed = getattr(transport, 'fixed_source_slot_sizes', None)
    if fixed is not None:
        if slot_bytes is not None or type(slot_count) is not int or slot_count != len(fixed) or (transport.max_slots != slot_count) or (transport.max_bytes < sum(fixed)):
            raise ValueError('fixed native source registrations differ from declared capacity')
        try:
            for slot, size in enumerate(fixed):
                transport.preallocate_source(size, slot_id=slot)
        except BaseException:
            transport.close()
            raise
        return
    if slot_bytes is None:
        size = native_capture_source_slot_bytes(slot_count)
        valid = type(slot_count) is int and slot_count in (1, 2, 4, 8, 16)
    else:
        size = slot_bytes
        valid = type(slot_count) is int and 1 <= slot_count <= 8 and (type(size) is int) and (size == native_capture_source_slot_bytes(8) or (slot_count <= 4 and size == 2 * 1024 ** 3))
    if not valid or transport.max_slots < slot_count or transport.max_bytes < size * slot_count:
        raise ValueError('native capture fixed source registrations exceed capacity')
    try:
        for slot in range(slot_count):
            transport.preallocate_source(size, slot_id=slot)
    except BaseException:
        transport.close()
        raise
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class MultiNodeCaptureCapability:
    worker_count: int
    contract_digest: str
    hidden_size: int
    layer_count: int
    layer_ids: tuple[int, ...]
    single_request_only: bool = True
    full_prefill_required: bool = True
    decode_capture_supported: bool = True

def probe_speco_capture_engine(engine: Any, *, expected_workers: int, hidden_size: int, layer_ids: tuple[int, ...]) -> MultiNodeCaptureCapability:
    """Fail closed if API expectations and TP worker contracts differ."""
    if expected_workers <= 0 or hidden_size <= 0 or (not layer_ids):
        raise ValueError('positive worker count/hidden size and layer IDs are required')
    reports = engine.collective_rpc('speco_capture_capability', kwargs={'expected_layer_ids': tuple(layer_ids), 'expected_hidden_size': hidden_size})
    if not isinstance(reports, (tuple, list)) or len(reports) != expected_workers:
        raise RuntimeError('target model capture worker count differs from API TP plan')
    if any((not isinstance(report, dict) or not report.get('target_capture_ready') for report in reports)):
        raise RuntimeError('one or more target model TP workers cannot capture the target')
    digests = {str(report.get('contract_digest', '')) for report in reports}
    if len(digests) != 1 or not next(iter(digests)):
        raise RuntimeError('target model TP workers disagree on the capture contract')
    if any((int(report.get('hidden_size', -1)) != hidden_size or tuple(report.get('expected_layer_ids', ())) != tuple(layer_ids) or int(report.get('layer_count', 0)) <= max(layer_ids) or (not report.get('single_request_only')) or (not report.get('full_prefill_required')) or (not report.get('decode_capture_supported')) or (not report.get('tq_publish_supported')) for report in reports)):
        raise RuntimeError('target model capture worker contract differs from API request')
    return MultiNodeCaptureCapability(worker_count=expected_workers, contract_digest=next(iter(digests)), hidden_size=hidden_size, layer_count=int(reports[0]['layer_count']), layer_ids=tuple(layer_ids))
PLUGIN_VERSION = '0.1.0'
capture_plugin_REGISTERED_PID: int | None = None

@dataclass(frozen=True)
class CaptureCapabilityReport:
    plugin_version: str
    process_id: int
    registered: bool
    target_capture_ready: bool
    reason: str

def register_speco_speco_capture_plugin() -> None:
    """Register safely in API, EngineCore and worker processes.

    vLLM's general-plugin group runs in all three processes. The current
    request-to-worker identity route is not wired, so this intentionally does
    not claim capture readiness or alter model execution.
    """
    global _REGISTERED_PID
    capture_plugin_REGISTERED_PID = os.getpid()

def probe_speco_capture_capability() -> CaptureCapabilityReport:
    pid = os.getpid()
    registered = capture_plugin_REGISTERED_PID == pid
    return CaptureCapabilityReport(plugin_version=PLUGIN_VERSION, process_id=pid, registered=registered, target_capture_ready=False, reason='request identity and worker TQ publisher are not connected' if registered else 'vLLM general plugin has not loaded in this process')
import ipaddress
import re

def validate_source_capture_owners(count, *, target_tp, physical_slots, native_features=True, release_enabled=True):
    if type(count) is not int or count not in (0, 1, 2, 4, 6, 8):
        raise ValueError('source capture owners must be 0, 1, 2, 4, 6 or 8')
    supported = target_tp == 8 and (physical_slots == 8 or (physical_slots == 16 and count == 8) or (physical_slots == 4 and count == 4)) or (target_tp == 16 and physical_slots in (8, 16) and (count in (6, 8)))
    if count and (type(target_tp) is not int or type(physical_slots) is not int or (not supported) or (native_features is not True) or (release_enabled is not True)):
        raise ValueError('distributed capture owners require a supported TP8/TP16 slot layout and release')
    return count

def source_capture_slot_bytes(*, target_tp, physical_slots, owner_ranks, long_context=False):
    """Distributed TP16 captures retain the complete 16K row capacity."""
    validate_source_capture_owners(owner_ranks, target_tp=target_tp, physical_slots=physical_slots)
    if type(long_context) is not bool:
        raise ValueError('long source context must be boolean')
    if long_context:
        if (target_tp, physical_slots, owner_ranks) not in ((8, 8, 8), (8, 16, 8)):
            raise ValueError('64K complete rows require eight distributed TP8 owners')
        return 2 * 1024 ** 3
    return native_capture_source_slot_bytes(8 if owner_ranks else physical_slots)

def source_capture_long_context(environment):
    policy = environment.get('SPECO_CAPTURE_LONG_CONTEXT', '')
    if policy not in ('', '0', '49152', '65536'):
        raise ValueError('invalid complete long source context')
    if policy in ('', '0'):
        return False
    environment_slot_profile(environment)
    banks = environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES')
    slots = environment.get('SPECO_CAPTURE_PHYSICAL_BATCH_SIZE', '8')
    if slots not in ('8', '16') or (slots == '16' and (banks != '2' or policy != '49152')):
        raise ValueError('sixteen-child long capture requires48K two-bank profile')
    if banks not in ('2', '3', '4', '10') or (banks in ('3', '4', '10') and policy != '49152'):
        raise ValueError('complete long source context permits two banks or three at48K')
    exact = {'SPECO_TARGET_TP': '8', 'SPECO_TARGET_MAX_SEQUENCE_LENGTH': policy, 'SPECO_CAPTURE_PHYSICAL_BATCH_SIZE': slots, 'SPECO_CAPTURE_NATIVE_SOURCE_SLOTS': slots, 'SPECO_CAPTURE_SOURCE_OWNER_RANKS': '8', 'SPECO_CAPTURE_NATIVE_BUFFER_BATCHES': banks, 'SPECO_CAPTURE_MAX_BATCHED_TOKENS': '131072', 'SPECO_CAPTURE_GRAPH_MODE': 'FULL_DECODE_ONLY', 'SPECO_SEQUENCE_FULL_DECODE': '1', 'SPECO_NATIVE_FEATURE_TRANSPORT': '1', 'SPECO_CAPTURE_REUSE_ENGINE': '1', 'SPECO_PREFILL_SOURCE_LOOKAHEAD': '1', 'SPECO_MHC_SP_PREPROFILE': '1', 'SPECO_CAPTURE_FULL_BATCH_PREFILL': '1'}
    if any((environment.get(k) != v for k, v in exact.items())):
        raise ValueError('64K source slots require the complete FULL distributed profile')
    from verl_speco.runtime.services import capture_prefill_engine_options
    options = capture_prefill_engine_options(environment)
    from verl_speco.runtime.services import token_budget_engine_tokens
    if options.get('max_num_batched_tokens') != token_budget_engine_tokens(environment) or options.get('enable_chunked_prefill') is not False:
        raise ValueError('64K complete rows require admitted full prefill')
    return True

def source_capture_owner_count(environment):
    policy = environment.get('SPECO_CAPTURE_SOURCE_OWNER_RANKS', '')
    if policy not in ('', '0', '1', '2', '4', '6', '8'):
        raise ValueError('invalid source capture owner policy')
    count = int(policy or '0')
    if not count:
        return 0
    from verl_speco.runtime.services import capture_target_parallelism
    return validate_source_capture_owners(count, target_tp=capture_target_parallelism(environment), physical_slots=native_capture_source_slots(environment), native_features=environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') == '1', release_enabled=environment.get('SPECO_CAPTURE_REUSE_ENGINE') == '1' and environment.get('SPECO_PREFILL_SOURCE_LOOKAHEAD') == '1')

def validate_source_capture_buffers(batches, *, owner_ranks, target_tp, physical_slots, step_lookahead=False, wide_full_profile=False, source_slot_profile=''):
    if type(source_slot_profile) is not str or source_slot_profile not in ('', PROFILE) or bool(source_slot_profile) != (batches == 10):
        raise ValueError('source slots differ from their explicit tiered profile')
    if type(batches) is not int or batches not in (1, 2, 3, 4, 10):
        raise ValueError('source native buffer batches must be 1, 2 or bound three-bank')
    if type(step_lookahead) is not bool or type(wide_full_profile) is not bool:
        raise ValueError('source cross-step lookahead and FULL profile must be boolean')
    if batches in (3, 4, 10) and (not (wide_full_profile and target_tp == 8 and (owner_ranks == 8) and (physical_slots == 8))):
        raise ValueError('three native banks require the explicit FULL TP8 eight-owner profile')
    if batches == 2:
        validate_source_capture_owners(owner_ranks, target_tp=target_tp, physical_slots=physical_slots)
    supported = target_tp == 8 and (owner_ranks == 8 and physical_slots in (8, 16) or (owner_ranks == 4 and physical_slots == 4)) or (owner_ranks in (6, 8) and target_tp == 16 and (physical_slots in (8, 16)))
    overlap_supported = target_tp == 8 and (owner_ranks == 4 and physical_slots == 4 or (owner_ranks == 8 and physical_slots in (8, 16) and wide_full_profile)) or (target_tp == 16 and owner_ranks == 8 and (physical_slots in (8, 16)))
    if batches == 2 and (not supported or (step_lookahead and (not overlap_supported))):
        raise ValueError('double native buffers require a verified distributed layout; cross-step overlap requires TP8 four-owner or TP16 eight-owner slots')
    return batches

def source_capture_buffer_batches(environment):
    slot_profile = environment_slot_profile(environment)
    flag = environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES', '')
    if flag not in ('', '1', '2', '3', '4', '10'):
        raise ValueError('invalid source native buffer batch policy')
    batches = int(flag or '1')
    if batches in (2, 3, 4, 10):
        from verl_speco.runtime.services import capture_target_parallelism, capture_prefill_engine_options
        wide_full_profile = environment.get('SPECO_TARGET_TP') == '8' and environment.get('SPECO_CAPTURE_PHYSICAL_BATCH_SIZE') in ('8', '16') and (environment.get('SPECO_CAPTURE_SOURCE_OWNER_RANKS') == '8') and (environment.get('SPECO_CAPTURE_MAX_BATCHED_TOKENS') == '131072')
        if batches in (3, 4, 10) and (not wide_full_profile or environment.get('SPECO_PREFILL_EARLY_READY') != '1' or environment.get('SPECO_PREFILL_STEP_LOOKAHEAD') != '1'):
            raise ValueError('three native banks require frozen teacher early-ready lookahead')
        if wide_full_profile:
            capture_prefill_engine_options(environment)
        validate_source_capture_buffers(batches, owner_ranks=source_capture_owner_count(environment), target_tp=capture_target_parallelism(environment), physical_slots=native_capture_source_slots(environment), step_lookahead=environment.get('SPECO_PREFILL_STEP_LOOKAHEAD', '') not in ('', '0'), wide_full_profile=wide_full_profile, source_slot_profile=slot_profile)
        if environment.get('SPECO_PREFILL_STEP_LOOKAHEAD', '') not in ('', '0'):
            if environment.get('SPECO_PREFILL_STEP_LOOKAHEAD') != '1' or environment.get('SPECO_PREFILL_EXACT_WAVE_ACQUIRE') != '1':
                raise ValueError('double-buffer cross-step overlap requires exact wave release acknowledgements')
    return batches

@dataclass(frozen=True)
class SourceCaptureLayout:
    owner_count: int
    rank: int
    global_slot_ids: tuple[int, ...]
    slot_bytes: int
    endpoint: str
    slot_sizes: tuple[int, ...] = ()
    slot_profile: str = ''

    @property
    def max_bytes(self):
        return sum(self.slot_sizes) if self.slot_sizes else len(self.global_slot_ids) * self.slot_bytes

def source_capture_layout(environment, rank):
    count = source_capture_owner_count(environment)
    batches = source_capture_buffer_batches(environment)
    if not count:
        return None
    from verl_speco.runtime.services import capture_target_parallelism
    target_tp = capture_target_parallelism(environment)
    physical_slots = native_capture_source_slots(environment)
    if type(rank) is not int or not 0 <= rank < target_tp:
        raise ValueError('source capture rank must match its TP mesh')
    endpoint = environment.get('SPECO_NATIVE_SOURCE_ENDPOINT', '')
    try:
        host, port_text = endpoint.split(':')
        ipaddress.IPv4Address(host)
        if not port_text.isdecimal() or not 1024 <= int(port_text) <= 65536 - target_tp:
            raise ValueError
    except ValueError:
        raise ValueError('distributed source endpoint requires IPv4 and a reserved TP port range') from None
    slots = tuple((batch * physical_slots + child for batch in range(batches) for child in range(rank, physical_slots, count))) if rank < count else ()
    return SourceCaptureLayout(count, rank, slots, source_capture_slot_bytes(target_tp=target_tp, physical_slots=physical_slots, owner_ranks=count, long_context=source_capture_long_context(environment)), f'{host}:{int(port_text) + rank}', slot_sizes=SLOT_SIZES if batches == 10 else (), slot_profile=environment.get('SPECO_CAPTURE_SOURCE_SLOT_PROFILE', ''))

@dataclass(frozen=True)
class SkippedCapture:
    """An inactive worker holds identity only, without hooks or hidden tensors."""
    request_id: str
    engine_request_id: str
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    layer_ids: tuple[int, ...]
    hidden_size: int
    max_payload_bytes: int
    owner_rank: int

    @classmethod
    def from_arm(cls, item, owner_rank, *, slot_bytes=None):
        expected_bytes = native_capture_source_slot_bytes(8) if slot_bytes is None else slot_bytes
        if type(expected_bytes) is not int or expected_bytes not in (512 * 1024 ** 2, 2 * 1024 ** 3):
            raise ValueError('invalid admitted distributed capture slot capacity')
        result = cls(item['request_id'], item['engine_request_id'], tuple(item['token_ids']), tuple(item['positions']), tuple(item['layer_ids']), item['hidden_size'], item['max_payload_bytes'], owner_rank)
        if not isinstance(result.request_id, str) or not result.request_id or (not isinstance(result.engine_request_id, str)) or (not result.engine_request_id) or (not result.token_ids) or any((type(t) is not int or t < 0 for t in result.token_ids)) or (result.positions != tuple(range(len(result.token_ids)))) or any((type(t) is not int for t in result.positions)) or (len(result.layer_ids) != 5) or (tuple(sorted(set(result.layer_ids))) != result.layer_ids) or any((type(t) is not int or t < 0 for t in result.layer_ids)) or (type(result.hidden_size) is not int) or (result.hidden_size != 2560) or (type(result.max_payload_bytes) is not int) or (result.max_payload_bytes != expected_bytes) or (len(result.token_ids) * 30740 > result.max_payload_bytes):
            raise ValueError('distributed capture arm has invalid identity, rows or byte budget')
        return result

    def close(self):
        pass

    def validate_publication(self, metadata, loss_mask, generated_token_ids, expected_prefill_sha256, expected_prefill_token_count):
        from verl_speco.producer.feature_payload import validate_capture_metadata
        validate_capture_metadata(self, metadata)
        if generated_token_ids or len(loss_mask) != len(self.token_ids) or any((value not in (0.0, 1.0) for value in loss_mask)) or (metadata.payload_nbytes != len(self.token_ids) * 30740) or (expected_prefill_sha256 and (re.fullmatch('[0-9a-f]{64}', expected_prefill_sha256) is None or expected_prefill_token_count != len(self.token_ids))):
            raise ValueError('inactive capture publication differs from armed complete prefill')

def validate_child_publication(proofs, *, target_tp, key, owner_rank=None, request_id=None):
    """Return the real publisher; inactive workers never pretend to hash bytes."""
    if not isinstance(proofs, list) or len(proofs) != target_tp or any((not isinstance(p, dict) or type(p.get('rank')) is not int for p in proofs)) or ([p['rank'] for p in proofs] != list(range(target_tp))):
        raise RuntimeError('physical child publication has incomplete TP proof')
    owner = 0 if owner_rank is None else owner_rank
    if type(owner) is not int or not 0 <= owner < target_tp:
        raise RuntimeError('physical child publisher rank is invalid')
    leader = proofs[owner]
    if leader.get('status') != 'published' or leader.get('key') != key or (not isinstance(leader.get('payload_checksum'), str)) or (re.fullmatch('[0-9a-f]{64}', leader['payload_checksum']) is None):
        raise RuntimeError('physical child publisher has invalid key or payload checksum')
    if owner_rank is None:
        if any((p.get('status') != 'rank_skipped' for p in proofs[1:])) or any((p.get('payload_checksum') != leader['payload_checksum'] for p in proofs)):
            raise RuntimeError('physical child publication has inconsistent TP payload checksums')
    else:
        for rank, proof in enumerate(proofs):
            if type(proof.get('capture_owner_rank')) is not int or proof['capture_owner_rank'] != owner or proof.get('request_id') != request_id or (rank != owner and (proof.get('status') != 'owner_skipped' or 'payload_checksum' in proof or 'native_descriptor' in proof or ('key' in proof))):
                raise RuntimeError('distributed publication ownership proof is invalid')
    return leader
from dataclasses import replace
import time
from types import SimpleNamespace
from contextlib import nullcontext
from verl_speco.producer.target_capture import MultiNode92BSingleRequestCapture as MultiNode92BSingleRequestCapture
from verl_speco.producer.feature_payload import capture_fields, payload_checksum, validate_capture_metadata
from verl_speco.transport.streaming import SampleMetadataV3

class SpecoMultiNodeCaptureWorkerExtension:
    """Methods callable through vLLM collective_rpc; no hidden RPC result.

    vLLM dynamically adds this class as a base of its worker when configured
    with ``--worker-extension-cls``. Its model runner and process group remain
    owned by vLLM. The extension holds only one capture session per worker.
    """

    def speco_capture_memory_capacity(self):
        """Observe free device bytes only; no memory allocation or feature data."""
        from vllm.distributed import get_tp_group
        group = get_tp_group()
        if group.world_size != 8:
            raise ValueError('adaptive source capacity requires TP8')
        free, total = torch.npu.mem_get_info(self.device)
        if type(free) is not int or type(total) is not int or (not 0 <= free <= total):
            raise RuntimeError('invalid actual device memory observation')
        return dict(rank=group.rank_in_group, free_device_bytes=free, total_device_bytes=total)

    def speco_enable_mhc_prefill_fusion(self) -> dict[str, Any]:
        """Explicit standalone opt-in; preserve small-prefill and decode methods."""
        if os.environ.get('SPECO_MHC_PREFILL_FUSION') != 'triton-exact':
            raise RuntimeError('mHC prefill fusion requires the configured standalone opt-in')
        from verl_speco.runtime.services import capture_prefill_engine_options
        capture_prefill_engine_options(os.environ)
        cached = getattr(self, '_speco_preprofile_SP_receipt', None)
        if cached is not None:
            if getattr(self, '_speco_mhc_fusion_originals', None) is None:
                raise RuntimeError('preprofile SP adapter has been removed from a live engine')
            return dict(cached)
        return self._speco_install_mhc_prefill_fusion(min_tokens=16384, diagnostic=False)

    def speco_diagnostic_mhc_fusion_begin(self, *, min_tokens: int=4096) -> dict[str, Any]:
        """Install the exact-reduction candidate only in an owned NPU probe."""
        if os.environ.get('SPECO_STAGE_PROBE') != '1':
            raise RuntimeError('mHC candidate requires the owned diagnostic process')
        return self._speco_install_mhc_prefill_fusion(min_tokens=min_tokens, diagnostic=True)

    def _speco_install_mhc_prefill_fusion(self, *, min_tokens: int, diagnostic: bool) -> dict[str, Any]:
        if getattr(self, '_speco_mhc_fusion_originals', None) is not None:
            raise RuntimeError('mHC candidate is already installed')
        modules = [m for m in self.model_runner.model.model.modules() if type(m).__name__ == 'mHCModule' and (not m.merge_layer_only_pre)]
        if not modules:
            raise RuntimeError('no verified 92B mHC modules are loaded')
        self._speco_mhc_fusion_originals = []
        try:
            for module in modules:
                if not diagnostic and os.environ.get('SPECO_MHC_SEQUENCE_PARALLEL') == '1':
                    from vllm.distributed import get_tp_group
                    group = get_tp_group()
                    if group.world_size not in (8, 16) or group.world_size != int(os.environ['SPECO_TARGET_TP']) or len(modules) != 92:
                        raise RuntimeError('mHC sequence parallel requires the complete configured TP8/TP16 model')
                    from verl_speco.runtime.services import capture_mhc_sp_token_bound
                    fusion = MhcSequenceParallelPrefill(module, group, min_tokens=min_tokens, max_tokens=getattr(self, '_speco_sp_profile_max_tokens', capture_mhc_sp_token_bound(os.environ)))
                else:
                    fusion = MhcTritonPrefillFusion(module, min_tokens=min_tokens)
                for method, replacement in (('hc_pre', fusion.pre), ('hc_post', fusion.post)):
                    self._speco_mhc_fusion_originals.append((module, method, method in vars(module), vars(module).get(method)))
                    setattr(module, method, replacement)
        except BaseException:
            self.speco_diagnostic_mhc_fusion_close()
            raise
        return dict(modules=len(modules), min_tokens=min_tokens, original_FP32_phi_and_sinkhorn=True, diagnostic_only=diagnostic, sequence_parallel=not diagnostic and os.environ.get('SPECO_MHC_SEQUENCE_PARALLEL') == '1', sequence_parallel_world=group.world_size if not diagnostic and os.environ.get('SPECO_MHC_SEQUENCE_PARALLEL') == '1' else 1, sequence_parallel_rank=group.rank_in_group if not diagnostic and os.environ.get('SPECO_MHC_SEQUENCE_PARALLEL') == '1' else None, max_tokens=getattr(self, '_speco_sp_profile_max_tokens', capture_mhc_sp_token_bound(os.environ)) if not diagnostic and os.environ.get('SPECO_MHC_SEQUENCE_PARALLEL') == '1' else None)

    def speco_diagnostic_mhc_fusion_close(self) -> dict[str, Any]:
        originals = getattr(self, '_speco_mhc_fusion_originals', None)
        if originals is None:
            return {'restored_methods': 0}
        for module, method, had_override, previous in reversed(originals):
            if had_override:
                setattr(module, method, previous)
            else:
                delattr(module, method)
        self._speco_mhc_fusion_originals = None
        return {'restored_methods': len(originals)}

    def speco_prefill_profile_begin(self, *, min_tokens: int=4096, max_calls: int=4096) -> dict[str, Any]:
        """Owned diagnostic NPU events; do not change parameters or arithmetic."""
        if os.environ.get('SPECO_STAGE_PROBE') != '1':
            raise RuntimeError('prefill profiling requires the owned diagnostic process')
        if getattr(self, '_speco_prefill_profile', None) is not None:
            raise RuntimeError('prefill profile is already installed')
        profile = MultiNodePrefillProfile(self.model_runner.model.model, min_tokens=min_tokens, max_calls=max_calls)
        self._speco_prefill_profile = profile
        return profile.begin()

    def speco_prefill_profile_finish(self) -> dict[str, Any]:
        """Return small timing summaries only and restore all original methods."""
        profile = getattr(self, '_speco_prefill_profile', None)
        if os.environ.get('SPECO_STAGE_PROBE') != '1' or profile is None:
            raise RuntimeError('no owned prefill profile is installed')
        try:
            return profile.finish()
        finally:
            profile.close()
            self._speco_prefill_profile = None

    def speco_diagnostic_segmented_mhc(self) -> dict[str, Any]:
        """Owned diagnostic only; preserve original arithmetic, change its row grouping."""
        if os.environ.get('SPECO_STAGE_PROBE') != '1':
            raise RuntimeError('mHC segmentation is restricted to the owned diagnostic process')
        if getattr(self, '_speco_segmented_mhc_originals', None):
            raise RuntimeError('mHC diagnostic segmentation is already installed')
        modules = [m for m in self.model_runner.model.model.modules() if type(m).__name__ == 'mHCModule']
        if not modules:
            raise RuntimeError('loaded model has no verified mHC modules')
        originals = []
        for module in modules:
            original = module.hc_pre
            module._speco_diagnostic_original_hc_pre = original
            originals.append((module, 'hc_pre' in module.__dict__, original))
            module.hc_pre = lambda x, original=original: request_segmented_mhc_pre(original, x, self._speco_request_row_ranges())
        self._speco_segmented_mhc_originals = originals
        return {'segmented_mhc_modules': len(originals), 'diagnostic_only': True}

    def speco_diagnostic_precision_report(self, *, force_strict: bool=False) -> dict[str, Any]:
        """Report precision; a diagnostic-only process may request a strict comparison."""
        import torch_npu

        def report():
            cube = torch_npu.npu.matmul.cube_math_type
            return {'matmul_allow_hf32': torch_npu.npu.matmul.allow_hf32, 'conv_allow_hf32': torch_npu.npu.conv.allow_hf32, 'cube_math_type': int(cube) if cube is not None else None, 'float32_matmul_precision': torch.get_float32_matmul_precision()}
        before = report()
        if force_strict:
            if os.environ.get('SPECO_STAGE_PROBE') != '1':
                raise RuntimeError('precision override is restricted to the owned diagnostic process')
            torch.set_float32_matmul_precision('highest')
            torch_npu.npu.matmul.allow_hf32 = False
            torch_npu.npu.conv.allow_hf32 = False
            torch_npu.npu.matmul.cube_math_type = torch_npu.npu.CubeMathType.KEEP_DTYPE
        return {'before': before, 'after': report(), 'strict_override': force_strict}

    def speco_stage_probe_begin(self, *, mode: str, engine_request_id: str, token_count: int) -> dict[str, str]:
        """Temporarily observe the existing stage emitter; no model arithmetic changes."""
        import sys
        probe = getattr(self, '_speco_stage_probe', None)
        if probe is None:
            trunk = self.model_runner.model.model
            module = sys.modules[type(trunk).__module__]
            probe = MultiNodeStageProbe(module, row_ranges=self._speco_request_row_ranges)
            self._speco_stage_probe = probe
        probe.begin(mode=mode, engine_request_id=engine_request_id, token_count=token_count)
        return {'status': 'stage_probe_armed', 'mode': mode}

    def speco_stage_probe_finish(self) -> dict[str, Any]:
        probe = getattr(self, '_speco_stage_probe', None)
        if probe is None:
            raise RuntimeError('stage probe is missing')
        return probe.finish()

    def speco_stage_probe_close(self) -> dict[str, bool]:
        probe = getattr(self, '_speco_stage_probe', None)
        if probe is not None:
            probe.close()
            self._speco_stage_probe = None
        for module, had_override, original in getattr(self, '_speco_segmented_mhc_originals', ()):
            delattr(module, '_speco_diagnostic_original_hc_pre')
            if had_override:
                module.hc_pre = original
            else:
                delattr(module, 'hc_pre')
        self._speco_segmented_mhc_originals = []
        return {'stage_probe_closed': True}

    def speco_native_reap_sources(self) -> dict[str, int]:
        self._speco_fence_source_publications()
        store = getattr(self, '_speco_native_store', None)
        if store is None:
            return {'native_retained_sources': 0}
        store.reap_cleared_sources()
        return {'native_retained_sources': len(store._sources)}

    def speco_native_source_capacity(self):
        """Reap cleared leases before admitting the next physical batch."""
        layout = self._speco_source_owner_layout()
        batches = source_capture_buffer_batches(os.environ)
        if batches not in (2, 3, 4, 10) or layout is None:
            raise ValueError('source capacity probes require the explicit double-buffer owner layout')
        slots = len(layout.global_slot_ids)
        self._speco_fence_source_publications()
        store = getattr(self, '_speco_native_store', None)
        retained = 0
        if store is not None:
            store.reap_cleared_sources()
            if not slots or store.transport.max_slots != slots or store.transport.max_bytes != layout.max_bytes or (len(store._sources) > slots):
                raise RuntimeError('native publisher capacity differs from its fixed registration layout')
            retained = len(store._sources)
        if layout.slot_profile:
            return source_capacity_proof(rank=layout.rank, transport=None if store is None else store.transport)
        return dict(rank=layout.rank, source_capture_owner_ranks=layout.owner_count, source_native_buffer_batches=batches, native_source_slots=slots, native_source_bytes=layout.max_bytes, native_retained_sources=retained, native_available_slots=slots - retained, publisher_initialized=store is not None)

    def speco_close_tq_client(self) -> dict[str, bool]:
        """Drain this worker's client before engine shutdown; never close the service."""
        if getattr(self, '_speco_tq_publisher', None) is None:
            return {'client_closed': False}
        if not getattr(self, '_speco_tq_client_closed', False):
            import ray
            from verl_speco.runtime.services import import_transfer_queue
            tq = import_transfer_queue()
            from verl_speco.runtime.services import close_external_tq_client
            native_store = getattr(self, '_speco_native_store', None)
            if native_store is not None:
                if self.speco_native_reap_sources()['native_retained_sources']:
                    raise RuntimeError('native source leases remain before target shutdown')
                native_store.transport.close()
            queue = getattr(self, '_speco_source_publications', None)
            if queue is not None:
                queue.close()
            close_external_tq_client(tq.get_client())
            ray.shutdown()
            self._speco_tq_client_closed = True
        return {'client_closed': True}

    def _speco_device_concat_mode(self) -> bool:
        from verl_speco.runtime.services import capture_device_concat
        enabled = capture_device_concat(os.environ)
        if hasattr(self, '_speco_capture_device_concat') and self._speco_capture_device_concat != enabled:
            raise ValueError('worker capture device concat settings changed within one process')
        self._speco_capture_device_concat = enabled
        return enabled

    def _speco_source_cpu_pack_mode(self) -> bool:
        from verl_speco.runtime.services import capture_source_cpu_pack
        enabled = capture_source_cpu_pack(os.environ)
        if hasattr(self, '_speco_capture_source_cpu_pack') and self._speco_capture_source_cpu_pack != enabled:
            raise ValueError('worker source CPU pack settings changed within one process')
        self._speco_capture_source_cpu_pack = enabled
        return enabled

    def _speco_source_owner_layout(self):
        layout = source_capture_layout(os.environ, getattr(self, 'rank', None))
        if hasattr(self, '_speco_source_owner_signature') and self._speco_source_owner_signature != layout:
            raise ValueError('worker source capture ownership changed within one process')
        self._speco_source_owner_signature = layout
        return layout

    def speco_capture_capability(self, *, expected_layer_ids: tuple[int, ...]=(), expected_hidden_size: int=0) -> dict[str, Any]:
        model = getattr(getattr(self, 'model_runner', None), 'model', None)
        trunk = getattr(model, 'model', None)
        cfg = getattr(trunk, 'cfg', None)
        layers = getattr(trunk, 'layers', ())
        layer_count = len(layers) if layers is not None else 0
        hidden_size = int(getattr(cfg, 'hidden_size', 0) or 0)
        expected = tuple((int(item) for item in expected_layer_ids))
        mhc_available = bool(trunk is not None and hasattr(trunk, 'merge_mhc_module') and (layer_count > 0))
        ready = bool(mhc_available and hidden_size > 0 and (not expected_hidden_size or hidden_size == expected_hidden_size) and all((0 <= item < layer_count for item in expected)))
        contract = {'extension_version': 1, 'hidden_size': hidden_size, 'layer_count': layer_count, 'expected_layer_ids': expected, 'single_request_only': True, 'full_prefill_required': True, 'decode_capture_supported': True, 'tq_publish_supported': True}
        return {**contract, 'model_loaded': model is not None, 'target_mhc_available': mhc_available, 'target_capture_ready': ready, 'contract_digest': hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}

    def _speco_projection_reuse_policy(self, layout) -> bool:
        reuse_flag = os.environ.get('SPECO_CAPTURE_REUSE_MHC_PROJECTION_OUTPUT', '')
        if reuse_flag not in ('', '0', '1'):
            raise ValueError('invalid mHC projection output reuse flag')
        reuse_projection_output = reuse_flag == '1'
        if reuse_projection_output:
            from verl_speco.runtime.services import capture_prefill_engine_options
            options = capture_prefill_engine_options(os.environ)
            if layout is None or not options.get('enable_chunked_prefill') is False:
                raise ValueError('mHC output reuse requires owned full-batch standalone prefill')
        return reuse_projection_output

    def speco_capture_begin(self, *, request_id: str, engine_request_id: str, token_ids: tuple[int, ...], positions: tuple[int, ...], layer_ids: tuple[int, ...], hidden_size: int, max_payload_bytes: int, capture_decode: bool=False, allow_chunked_prefill: bool=False) -> dict[str, Any]:
        self._speco_source_cpu_pack_mode()
        self._speco_device_concat_mode()
        layout = self._speco_source_owner_layout()
        reuse_projection_output = self._speco_projection_reuse_policy(layout)
        if reuse_projection_output and allow_chunked_prefill:
            raise ValueError('mHC output reuse requires complete standalone prefill')
        if getattr(self, '_speco_capture', None) is not None:
            raise RuntimeError('a SpeCo capture is already active')
        model = getattr(getattr(self, 'model_runner', None), 'model', None)
        if model is None:
            raise RuntimeError('vLLM worker model is not loaded')
        if not engine_request_id:
            raise ValueError('vLLM engine request ID is required')
        cfg = getattr(getattr(model, 'model', None), 'cfg', None)
        configured_width = getattr(cfg, 'hidden_size', None)
        if configured_width is not None and int(configured_width) != hidden_size:
            raise ValueError('capture hidden size differs from the loaded target config')
        capture = MultiNode92BSingleRequestCapture(model, layer_ids=tuple(layer_ids), hidden_size=hidden_size, max_payload_bytes=max_payload_bytes, token_ids_supplier=lambda: self.model_runner.input_ids.gpu, request_ids_supplier=lambda: list(self.model_runner.input_batch.req_ids), capture_decode=capture_decode, allow_chunked_prefill=allow_chunked_prefill, reuse_projection_output=reuse_projection_output)
        try:
            capture.begin(request_id=request_id, token_ids=tuple(token_ids), positions=tuple(positions), engine_request_id=engine_request_id)
        except BaseException:
            capture.close()
            raise
        self._speco_capture = capture
        return {'status': 'armed', 'request_id': request_id, 'engine_request_id': engine_request_id, 'layer_ids': list(layer_ids)}

    def _speco_request_row_ranges(self) -> dict[str, tuple[int, int]]:
        """Map each scheduled vLLM request to its contiguous model input rows."""
        runner = self.model_runner
        request_ids = runner.input_batch.req_ids
        starts = runner.query_start_loc.cpu
        return {request_id: (int(starts[index]), int(starts[index + 1])) for index, request_id in enumerate(request_ids) if request_id is not None and index + 1 < starts.numel()}

    def speco_capture_begin_many(self, *, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Arm separate capture hooks for independently attended vLLM requests."""
        self._speco_source_cpu_pack_mode()
        self._speco_device_concat_mode()
        if getattr(self, '_speco_capture', None) is not None or getattr(self, '_speco_batch_captures', None):
            raise RuntimeError('a SpeCo capture is already active')
        from verl_speco.runtime.services import token_budget_engine_batch_size
        coalesced = token_budget_engine_batch_size(os.environ)
        if not (1 if coalesced else 2) <= len(requests) <= (coalesced or 16):
            raise ValueError('batch capture exceeds its explicit independent request bound')
        model = getattr(getattr(self, 'model_runner', None), 'model', None)
        if model is None:
            raise RuntimeError('vLLM worker model is not loaded')
        layout = self._speco_source_owner_layout()
        reuse_projection_output = self._speco_projection_reuse_policy(layout)
        contracts = []
        if layout is not None:
            for index, item in enumerate(requests):
                owner = item.get('capture_owner_rank')
                child_index = item.get('physical_child_index') if coalesced else index
                if type(child_index) is not int or not 0 <= child_index < (8 if coalesced else 16) or type(owner) is not int or (owner != child_index % layout.owner_count):
                    raise ValueError('distributed capture arm owner differs from bounded round robin')
                contracts.append(SkippedCapture.from_arm(item, owner, slot_bytes=layout.slot_bytes))
            if len({c.engine_request_id for c in contracts}) != len(contracts) or len(requests) > (coalesced or native_capture_source_slots(os.environ)):
                raise ValueError('distributed capture arms exceed unique fixed source slots')
            configured_width = getattr(getattr(getattr(model, 'model', None), 'cfg', None), 'hidden_size', None)
            if configured_width is not None and int(configured_width) != 2560:
                raise ValueError('capture hidden size differs from loaded target config')
        elif any(('capture_owner_rank' in item for item in requests)):
            raise ValueError('capture owner arms require the explicit distributed policy')
        captures = {}
        owners = {}
        try:
            for index, item in enumerate(requests):
                engine_id = item['engine_request_id']
                if not engine_id or engine_id in captures:
                    raise ValueError('batch capture engine request IDs must be unique')
                if layout is not None:
                    owners[engine_id] = contracts[index].owner_rank
                    if layout.rank != contracts[index].owner_rank:
                        captures[engine_id] = contracts[index]
                        continue
                capture = MultiNode92BSingleRequestCapture(model, layer_ids=tuple(item['layer_ids']), hidden_size=int(item['hidden_size']), max_payload_bytes=int(item['max_payload_bytes']), token_ids_supplier=lambda: self.model_runner.input_ids.gpu, request_row_ranges_supplier=self._speco_request_row_ranges, capture_decode=True, reuse_projection_output=reuse_projection_output)
                captures[engine_id] = capture
                capture.begin(request_id=item['request_id'], token_ids=tuple(item['token_ids']), positions=tuple(item['positions']), engine_request_id=engine_id)
        except BaseException:
            for capture in captures.values():
                capture.close()
            raise
        self._speco_batch_captures = captures
        self._speco_batch_capture_owners = owners
        return [{'status': 'owner_skipped' if isinstance(capture, SkippedCapture) else 'armed', 'engine_request_id': engine_id, **({'rank': layout.rank, 'capture_owner_rank': owners[engine_id]} if layout is not None else {})} for engine_id, capture in captures.items()]

    def _speco_close_batch_captures(self):
        captures = list((getattr(self, '_speco_batch_captures', None) or {}).values())
        active = getattr(self, '_speco_capture', None)
        if active is not None:
            captures.append(active)
        self._speco_capture = None
        self._speco_batch_captures = None
        self._speco_capture_owner_rank = None
        self._speco_batch_capture_owners = {}
        failure = None
        closed_ids = set()
        for capture in captures:
            if id(capture) in closed_ids:
                continue
            closed_ids.add(id(capture))
            try:
                capture.close()
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def _speco_require_async_source_profile(self):
        layout = self._speco_source_owner_layout()
        if getattr(self, '_speco_async_source_failed', False) or os.environ.get('SPECO_CAPTURE_ASYNC_PUBLICATION') != '1' or os.environ.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1' or (layout is None) or (layout.owner_count != 8) or (source_capture_buffer_batches(os.environ) not in (4, 10)) or (not self._speco_device_concat_mode()) or (not self._speco_source_cpu_pack_mode()):
            raise ValueError('async publication requires the bounded eight-owner four-bank native profile')
        return layout

    def _speco_source_publication_queue(self):
        existing = getattr(self, '_speco_source_publications', None)
        if existing is not None:
            return existing
        device = torch.npu.current_device()
        queue = BoundedSourcePublications(publish=self._speco_publish_owned_fields, max_bytes=4 * 1024 ** 3, initialize=lambda: torch.npu.set_device(device))
        self._speco_source_publications = queue
        return queue

    def _speco_fence_source_publications(self):
        queue = getattr(self, '_speco_source_publications', None)
        if queue is not None and (not queue.closed):
            queue.wait_all()

    def speco_capture_enqueue_many(self, *, requests):
        """Detach all fixed-eight captures with the unchanged default profile."""
        return self._speco_enqueue_owned_publications(requests, fragment=False)

    def speco_capture_enqueue_fragment(self, *, requests):
        """Detach one FIFO logical command, retaining other model captures."""
        from verl_speco.runtime.services import token_budget_engine_batch_size
        if os.environ.get('SPECO_CAPTURE_KV_AWARE_BATCHING') != '1' or os.environ.get('SPECO_CAPTURE_KV_MAX_SEQUENCES') != '16' or token_budget_engine_batch_size(os.environ) != 40 or os.environ.get('SPECO_CAPTURE_CROSS_STEP_TOKEN_BATCHES', ''):
            raise ValueError('async fragment needs the explicit same-step KV-bounded16 profile')
        return self._speco_enqueue_owned_publications(requests, fragment=True)

    def _speco_enqueue_owned_publications(self, requests, *, fragment):
        from verl_speco.transport.streaming import make_v3_sample_key
        self._speco_require_async_source_profile()
        captures = getattr(self, '_speco_batch_captures', None)
        batches = getattr(self, '_speco_pending_publication_batches', [])
        proofs = getattr(self, '_speco_pending_publication_proofs', {})
        try:
            if not isinstance(requests, list) or not isinstance(captures, dict) or (not 1 <= len(requests) <= 8) or (len(batches) >= 2) or (not fragment and len(requests) != len(captures)) or (fragment and (not len(requests) <= len(captures) <= 16)):
                raise ValueError('async batch exceeds its two-capture ownership bound')
            ids = [item['engine_request_id'] for item in requests]
            keys = tuple((make_v3_sample_key(SampleMetadataV3.from_dict(item['metadata'])) for item in requests))
            owners = getattr(self, '_speco_batch_capture_owners', {})
            exact_ids = ids == list(captures)[:len(ids)] if fragment else set(ids) == set(captures)
            if len(set(ids)) != len(ids) or not exact_ids or len(set(keys)) != len(keys) or set(keys).intersection(proofs) or (len({owners[engine_id] for engine_id in ids}) != len(ids)) or any(('defer_publication' in item for item in requests)):
                raise ValueError('async batch differs from unique fixed capture ownership')
            tickets = []
            for item, key in zip(requests, keys):
                engine_id = item['engine_request_id']
                self._speco_capture = captures.pop(engine_id)
                self._speco_capture_owner_rank = owners.pop(engine_id)
                ticket = self.speco_capture_publish(**{k: v for k, v in item.items() if k != 'engine_request_id'}, defer_publication=True)
                if ticket['status'] not in ('owner_skipped', 'publication_pending'):
                    raise RuntimeError('async capture returned an early completed publication')
                proofs[key] = ticket
                tickets.append(ticket)
            batches.append(keys)
            self._speco_pending_publication_batches = batches
            self._speco_pending_publication_proofs = proofs
            return tickets
        except BaseException:
            self._speco_async_source_failed = True
            queue = getattr(self, '_speco_source_publications', None)
            if queue is not None:
                queue.close()
            self._speco_close_batch_captures()
            raise
        finally:
            if not fragment or not captures:
                self._speco_close_batch_captures()

    def speco_capture_collect_publications(self, *, keys):
        self._speco_require_async_source_profile()
        batches = getattr(self, '_speco_pending_publication_batches', [])
        proofs = getattr(self, '_speco_pending_publication_proofs', {})
        if not isinstance(keys, (list, tuple)) or not batches or tuple(keys) != batches[0]:
            raise RuntimeError('source publication tickets must finish in FIFO batch order')
        try:
            result = []
            for key in keys:
                ticket = proofs[key]
                proof = self._speco_source_publication_queue().get(key) if ticket['status'] == 'publication_pending' else ticket
                if proof['status'] not in ('published', 'owner_skipped'):
                    raise RuntimeError('source publication did not produce a completed owner proof')
                result.append(proof)
            for key in keys:
                proofs.pop(key)
            batches.pop(0)
            return result
        except BaseException:
            self._speco_async_source_failed = True
            queue = getattr(self, '_speco_source_publications', None)
            if queue is not None:
                queue.close()
            raise

    def speco_capture_publish_many(self, *, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Publish each captured child under its own TQ key and lease."""
        captures = getattr(self, '_speco_batch_captures', None)
        if not isinstance(captures, dict) or len(requests) != len(captures):
            self._speco_close_batch_captures()
            raise RuntimeError('batch publish does not match armed requests')
        summaries = []
        try:
            engine_ids = [item['engine_request_id'] for item in requests]
            if len(set(engine_ids)) != len(engine_ids) or set(engine_ids) != set(captures):
                raise RuntimeError('batch publish contains an unknown request ID')
            for item in requests:
                engine_id = item['engine_request_id']
                capture = captures.pop(engine_id, None)
                if capture is None:
                    raise RuntimeError('batch publish contains an unknown request ID')
                self._speco_capture = capture
                self._speco_capture_owner_rank = getattr(self, '_speco_batch_capture_owners', {}).get(engine_id)
                summaries.append(self.speco_capture_publish(**{key: value for key, value in item.items() if key != 'engine_request_id'}))
        finally:
            self._speco_close_batch_captures()
        return summaries

    def speco_capture_publish_fragment(self, *, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Publish at most one logical command while retaining other captures.

        The source callback checks free registered slots before every fragment.
        Remaining hooks and tensors survive a successful fragment. Any failure
        aborts the entire engine batch; no surviving capture can be retried.
        """
        from verl_speco.runtime.services import token_budget_engine_batch_size
        captures = getattr(self, '_speco_batch_captures', None)
        try:
            if not token_budget_engine_batch_size(os.environ) or not isinstance(captures, dict) or (not isinstance(requests, list)) or (not 1 <= len(requests) <= 8):
                raise ValueError('fragment publication requires bounded coalesced captures')
            engine_ids = [item['engine_request_id'] for item in requests]
            if len(set(engine_ids)) != len(engine_ids) or engine_ids != list(captures)[:len(requests)]:
                raise RuntimeError('fragment publication differs from FIFO capture ownership')
            summaries = []
            for item in requests:
                engine_id = item['engine_request_id']
                self._speco_capture = captures.pop(engine_id)
                self._speco_capture_owner_rank = self._speco_batch_capture_owners.pop(engine_id)
                summaries.append(self.speco_capture_publish(**{k: v for k, v in item.items() if k != 'engine_request_id'}))
            if not captures:
                self._speco_close_batch_captures()
            return summaries
        except BaseException:
            self._speco_close_batch_captures()
            raise

    def speco_capture_digest_many(self, *, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compare physical batches without publishing training features."""
        captures = getattr(self, '_speco_batch_captures', None)
        if not isinstance(captures, dict) or len(requests) != len(captures):
            self._speco_close_batch_captures()
            raise RuntimeError('batch digest does not match armed requests')
        summaries = []
        try:
            if getattr(self, '_speco_batch_capture_owners', {}):
                raise ValueError('distributed ownership requires publication proofs; use full TP mode for digest diagnostics')
            for item in requests:
                capture = captures.pop(item['engine_request_id'], None)
                if capture is None:
                    raise RuntimeError('batch digest contains an unknown request ID')
                try:
                    result = capture.collect_trajectory(generated_token_ids=tuple(item['generated_token_ids']))
                    digest = hashlib.sha256()
                    for tensor in (*result.hidden_states, result.final_hidden_state):
                        cpu = tensor.detach().to('cpu').contiguous()
                        digest.update(cpu.view(dtype=torch.uint8).numpy().tobytes())
                    summaries.append({'engine_request_id': item['engine_request_id'], 'token_count': len(result.token_ids), 'hidden_sha256': digest.hexdigest()})
                finally:
                    capture.close()
        finally:
            self._speco_close_batch_captures()
        return summaries

    def speco_capture_finish(self) -> dict[str, Any]:
        capture = getattr(self, '_speco_capture', None)
        if capture is None:
            raise RuntimeError('no SpeCo capture is active')
        try:
            result = capture.collect()
            tensors = (*result.hidden_states, result.final_hidden_state)
            digest = hashlib.sha256()
            for tensor in tensors:
                cpu = tensor.detach().to('cpu').contiguous()
                digest.update(cpu.view(dtype=torch.uint8).numpy().tobytes())
            return {'status': 'captured', 'request_id': result.request_id, 'token_count': len(result.token_ids), 'layer_ids': list(result.layer_ids), 'shapes': [list(tensor.shape) for tensor in tensors], 'dtypes': [str(tensor.dtype) for tensor in tensors], 'payload_nbytes': sum((t.numel() * t.element_size() for t in tensors)), 'sha256': digest.hexdigest()}
        finally:
            capture.close()
            self._speco_capture = None

    def speco_capture_abort(self) -> dict[str, str]:
        pending = getattr(self, '_speco_source_publications', None)
        if pending is not None:
            pending.close()
        self._speco_pending_publication_batches = []
        self._speco_pending_publication_proofs = {}
        failure = None
        try:
            self._speco_close_batch_captures()
        except BaseException as error:
            failure = error
        self._speco_baseline = None
        try:
            self.speco_stage_probe_close()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f'stage probe close failed: {type(error).__name__}')
        if failure is not None:
            raise failure
        return {'status': 'aborted'}

    def speco_capture_store_baseline(self) -> dict[str, Any]:
        """Retain prompt-prefill features on CPU for an in-worker replay check."""
        if getattr(self, '_speco_baseline', None) is not None:
            raise RuntimeError('a prompt baseline is already retained')
        capture = getattr(self, '_speco_capture', None)
        if capture is None:
            raise RuntimeError('no SpeCo capture is active')
        try:
            result = capture.collect()
            tensors = tuple((tensor.detach().to('cpu').contiguous() for tensor in (*result.hidden_states, result.final_hidden_state)))
            self._speco_baseline = (result.token_ids, result.layer_ids, tensors)
            return {'status': 'baseline_retained', 'token_count': len(result.token_ids), 'layer_ids': list(result.layer_ids), **prefill_digest(replace(result, hidden_states=tensors[:-1], final_hidden_state=tensors[-1]))}
        finally:
            capture.close()
            self._speco_capture = None

    def speco_capture_prefill_digest(self) -> dict[str, Any]:
        """Return a bounded exact isolated feature fingerprint, then release capture."""
        capture = getattr(self, '_speco_capture', None)
        if capture is None:
            raise RuntimeError('no SpeCo capture is active')
        try:
            return prefill_digest(capture.collect())
        finally:
            capture.close()
            self._speco_capture = None

    def speco_capture_compare_baseline(self) -> dict[str, Any]:
        """Compare a second prompt-only prefill without publishing hidden."""
        capture = getattr(self, '_speco_capture', None)
        if capture is None:
            raise RuntimeError('no SpeCo capture is active')
        try:
            result = capture.collect()
            return self._speco_compare_baseline(result, retain_baseline=True)
        finally:
            capture.close()
            self._speco_capture = None

    def _speco_compare_baseline(self, result, *, retain_baseline: bool=False) -> dict[str, Any]:
        baseline = getattr(self, '_speco_baseline', None)
        if baseline is None:
            raise RuntimeError('prompt-prefill baseline is missing')
        baseline_ids, baseline_layers, baseline_tensors = baseline
        if not retain_baseline:
            self._speco_baseline = None
        if result.token_ids[:len(baseline_ids)] != baseline_ids or result.layer_ids != baseline_layers:
            raise RuntimeError('replay prefill does not share the baseline prompt prefix')
        current_tensors = (*result.hidden_states, result.final_hidden_state)
        if len(current_tensors) != len(baseline_tensors):
            raise RuntimeError('replay feature layer count differs from prompt baseline')
        total_abs = 0.0
        total_values = 0
        maximum = 0.0
        per_layer: list[dict[str, Any]] = []
        for baseline_tensor, current_tensor in zip(baseline_tensors, current_tensors):
            prefix = current_tensor[:len(baseline_ids)].detach().to('cpu').float()
            difference = (prefix - baseline_tensor.float()).abs()
            denominator = float(baseline_tensor.float().square().mean().sqrt())
            relative_rms = float(difference.square().mean().sqrt()) / max(denominator, 1e-12)
            flattened_reference = baseline_tensor.float().reshape(-1)
            flattened_current = prefix.reshape(-1)
            cosine = float(torch.nn.functional.cosine_similarity(flattened_reference, flattened_current, dim=0))
            maximum = max(maximum, float(difference.max()))
            total_abs += float(difference.sum())
            total_values += difference.numel()
            row_max = difference.amax(dim=-1)
            top = torch.topk(row_max, k=min(3, row_max.numel())).indices.tolist()
            per_layer.append({'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()), 'relative_rms': relative_rms, 'cosine_similarity': cosine, 'first64_mean_abs': float(difference[:64].mean()), 'last64_mean_abs': float(difference[-64:].mean()), 'largest_error_rows': [int(item) for item in top]})
        return {'baseline_token_count': len(baseline_ids), 'baseline_max_abs': maximum, 'baseline_mean_abs': total_abs / total_values, 'baseline_per_layer': per_layer}

    def speco_capture_publish(self, *, metadata: dict[str, Any], loss_mask: tuple[float, ...], partition: str, ray_address: str, capacity_bytes: int, control_actor_name: str='', compare_baseline: bool=False, baseline_mean_abs_limit: float | None=None, baseline_max_abs_limit: float | None=None, generated_token_ids: tuple[int, ...]=(), expected_prefill_sha256: str='', expected_prefill_token_count: int=0, defer_publication: bool=False) -> dict[str, Any]:
        """Publish on the explicit owner, or hash all TP ranks in legacy mode.

        The caller must check every rank digest before publishing ready to the
        shared control plane. TransferQueue must already exist in the named
        Ray cluster; this method only attaches a client and never creates TQ.
        """
        capture = getattr(self, '_speco_capture', None)
        if capture is None:
            raise RuntimeError('no SpeCo capture is active')
        try:
            if not partition or capacity_bytes <= 0:
                raise ValueError('TQ partition and positive byte capacity are required')
            layout = self._speco_source_owner_layout()
            owner = getattr(self, '_speco_capture_owner_rank', None)
            if owner is not None and (layout is None or type(owner) is not int or (not 0 <= owner < layout.owner_count)):
                raise ValueError('publication capture owner differs from static layout')
            if isinstance(capture, SkippedCapture):
                if owner != capture.owner_rank or layout.rank == owner:
                    raise ValueError('inactive capture differs from its assigned owner')
                v3 = SampleMetadataV3.from_dict(metadata)
                capture.validate_publication(v3, tuple(loss_mask), generated_token_ids, expected_prefill_sha256, expected_prefill_token_count)
                return {'status': 'owner_skipped', 'rank': layout.rank, 'request_id': capture.request_id, 'capture_owner_rank': owner}
            if owner is not None and layout.rank != owner:
                raise ValueError('active capture exists on an unassigned owner')
            source_cpu_pack = self._speco_source_cpu_pack_mode()
            device_concat = self._speco_device_concat_mode()
            started = time.perf_counter()
            result = capture.collect_trajectory(generated_token_ids=tuple(generated_token_ids)) if generated_token_ids else capture.collect()
            collect_seconds = time.perf_counter() - started
            comparison = self._speco_compare_baseline(result) if compare_baseline else {}
            if expected_prefill_sha256:
                exact = prefill_digest(result, token_count=expected_prefill_token_count)
                if exact['prefill_sha256'] != expected_prefill_sha256:
                    raise RuntimeError('physical child prefill differs from its isolated fingerprint')
                comparison.update(exact)
            if compare_baseline and (baseline_mean_abs_limit is not None and comparison['baseline_mean_abs'] > baseline_mean_abs_limit or (baseline_max_abs_limit is not None and comparison['baseline_max_abs'] > baseline_max_abs_limit)):
                raise RuntimeError(f"batched prefill differs from isolated request beyond numerical limits: mean_abs={comparison['baseline_mean_abs']}, max_abs={comparison['baseline_max_abs']}; comparison=" + json.dumps(comparison, sort_keys=True))
            v3 = SampleMetadataV3.from_dict(metadata)
            validate_capture_metadata(result, v3)
            started = time.perf_counter()
            fields = capture_fields(result, loss_mask=tuple(loss_mask), **{'device_concat': True} if device_concat else {})
            fields_seconds = time.perf_counter() - started
            actual_bytes = sum((t.numel() * t.element_size() for t in fields.values()))
            if v3.payload_nbytes != actual_bytes:
                raise ValueError('v3 payload byte count differs from captured fields')
            if type(defer_publication) is not bool:
                raise ValueError('source publication deferral must be a boolean')
            publisher = None
            if defer_publication:
                self._speco_require_async_source_profile()
                if owner is None or layout.rank != owner:
                    raise ValueError('async publication needs the explicit active capture owner')
                publisher = self._speco_get_tq_publisher(partition=partition, ray_address=ray_address, capacity_bytes=capacity_bytes, producer_endpoint=v3.producer_id, control_actor_name=control_actor_name)
            context = SimpleNamespace(payload_nbytes=actual_bytes, v3=v3, owner=owner, comparison=dict(comparison), request_id=result.request_id, device_concat=device_concat, source_cpu_pack=source_cpu_pack, collect_seconds=collect_seconds, fields_seconds=fields_seconds, partition=partition, ray_address=ray_address, capacity_bytes=capacity_bytes, control_actor_name=control_actor_name, publisher=publisher, native_store=getattr(self, '_speco_native_store', None))
            from verl_speco.transport.streaming import make_v3_sample_key
            key = make_v3_sample_key(v3)
            if defer_publication:
                self._speco_source_publication_queue().submit(key, context, fields)
                return dict(status='publication_pending', rank=int(self.rank), key=key, request_id=result.request_id, capture_owner_rank=owner)
            return self._speco_publish_owned_fields(key, context, fields)
        finally:
            try:
                capture.close()
            finally:
                self._speco_capture = None
                self._speco_capture_owner_rank = None

    def _speco_publish_owned_fields(self, key, context, fields):
        v3 = context.v3
        owner = context.owner
        comparison = context.comparison
        actual_bytes = context.payload_nbytes
        device_concat = context.device_concat
        source_cpu_pack = context.source_cpu_pack
        collect_seconds = context.collect_seconds
        fields_seconds = context.fields_seconds
        partition = context.partition
        ray_address = context.ray_address
        capacity_bytes = context.capacity_bytes
        control_actor_name = context.control_actor_name
        started = time.perf_counter()
        checksum = payload_checksum(fields)
        checksum_seconds = time.perf_counter() - started
        rank = getattr(self, 'rank', None)
        if rank is None:
            raise RuntimeError('vLLM worker rank is unavailable')
        print('speco_capture_worker_timing ' + json.dumps(dict(rank=int(rank), sequence_no=v3.sequence_no, payload_bytes=actual_bytes, device_concat=device_concat, source_cpu_direct_pack=source_cpu_pack, collect_seconds=collect_seconds, fields_seconds=fields_seconds, checksum_seconds=checksum_seconds, deferred_publication=context.publisher is not None)), flush=True)
        publisher_rank = 0 if owner is None else owner
        if int(rank) != publisher_rank:
            return {'status': 'rank_skipped', 'rank': int(rank), 'request_id': context.request_id, 'payload_checksum': checksum, **comparison}
        publisher = context.publisher or self._speco_get_tq_publisher(partition=partition, ray_address=ray_address, capacity_bytes=capacity_bytes, producer_endpoint=v3.producer_id, control_actor_name=control_actor_name)
        native_store = context.native_store
        scope = native_store.transport.publication_scope() if context.publisher is not None and native_store is not None else nullcontext()
        with scope:
            receipt = publisher.publish(v3, fields=fields, payload_checksum=checksum)
        if receipt.payload_checksum != checksum:
            raise RuntimeError('TQ publication checksum differs from worker capture')
        publication = {'status': receipt.status, 'rank': int(rank), 'key': receipt.key, 'request_id': context.request_id, 'payload_checksum': checksum, **comparison}
        if owner is not None:
            publication['capture_owner_rank'] = owner
        native_store = context.native_store if context.publisher is not None else getattr(self, '_speco_native_store', None)
        if native_store is not None:
            publication['native_descriptor'] = native_store.publication_descriptor(receipt.key, partition)
        return publication

    def _speco_get_tq_publisher(self, *, partition: str, ray_address: str, capacity_bytes: int, producer_endpoint: str, control_actor_name: str=''):
        source_slots = native_capture_source_slots(os.environ)
        layout = self._speco_source_owner_layout()
        if layout is not None and (not layout.global_slot_ids):
            raise ValueError('unassigned capture worker cannot initialize a publisher')
        cpu_direct_pack = self._speco_source_cpu_pack_mode()
        from verl_speco.transport.native_transfer import native_descriptor_store_type
        native_store_type = native_descriptor_store_type(os.environ)
        inline_descriptor = os.environ.get('SPECO_NATIVE_INLINE_DESCRIPTOR', '') == '1'
        settings = (partition, ray_address, capacity_bytes, producer_endpoint, control_actor_name)
        if layout is not None:
            settings += (layout,)
        existing = getattr(self, '_speco_tq_publisher', None)
        if existing is not None:
            if getattr(self, '_speco_tq_settings', None) != settings:
                raise ValueError('worker TQ publisher settings changed within one process')
            if getattr(self, '_speco_tq_cpu_direct_pack', False) != cpu_direct_pack:
                raise ValueError('worker native source CPU pack policy changed within one process')
            if getattr(self, '_speco_tq_inline_descriptor', False) != inline_descriptor:
                raise ValueError('worker TQ publisher settings changed within one process')
            if getattr(self, '_speco_tq_native_slot_count', 1) != source_slots:
                raise ValueError('worker native source slot policy changed within one process')
            return existing
        if not ray_address:
            raise ValueError('external Ray address is required for worker TQ publication')
        import ray
        from verl_speco.runtime.services import import_transfer_queue
        tq = import_transfer_queue()
        from verl_speco.transport.control import BoundedFeaturePublisher, RayBoundedFeaturePublisher, TQKeyedStore
        from verl_speco.transport.control import ReadyRouter
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace='transfer_queue')
        try:
            ray.get_actor('TransferQueueController', namespace='transfer_queue')
        except ValueError as exc:
            raise RuntimeError('existing TransferQueueController is required') from exc
        tq.init()
        store = TQKeyedStore(tq)
        if os.environ.get('SPECO_NATIVE_FEATURE_TRANSPORT') == '1':
            from verl_speco.transport.native_transfer import AscendFeatureTransport
            if os.environ.get('SPECO_CAPTURE_ASYNC_PUBLICATION') == '1':
                self._speco_require_async_source_profile()
                from verl_speco.transport.publication_transport import SourcePublicationTransport
                AscendFeatureTransport = SourcePublicationTransport
            transport = AscendFeatureTransport(run_id=os.environ['SPECO_RUN_ID'], generation=int(os.environ['SPECO_GENERATION']), endpoint=layout.endpoint if layout is not None else os.environ['SPECO_NATIVE_SOURCE_ENDPOINT'], metadata_server=os.environ['SPECO_NATIVE_METADATA_SERVER'], device=f'npu:{torch.npu.current_device()}', max_bytes=layout.max_bytes if layout is not None else 4 * 1024 ** 3, max_slots=len(layout.global_slot_ids) if layout is not None else max(2, source_slots), **dict(fixed_source_slot_sizes=layout.slot_sizes) if layout is not None and layout.slot_profile else {}, **dict(local_host_mirror=True) if os.environ.get('SPECO_COLOCATED_FEATURE_HOST_MIRROR') == '1' else {})
            if os.environ.get('SPECO_CAPTURE_REUSE_ENGINE') == '1':
                if layout is None:
                    preallocate_native_capture_sources(transport, source_slots)
                else:
                    preallocate_native_capture_sources(transport, len(layout.global_slot_ids), **{} if layout.slot_profile else dict(slot_bytes=layout.slot_bytes))
            store = native_store_type(tq=tq, transport=transport, **{'cpu_direct_pack': True} if cpu_direct_pack else {})
            self._speco_native_store = store
        if control_actor_name:
            control = ray.get_actor(control_actor_name, namespace='transfer_queue')
            publisher = RayBoundedFeaturePublisher(store=store, control_actor=control, partition=partition, producer_endpoint=producer_endpoint)
        else:
            publisher = BoundedFeaturePublisher(store=store, router=ReadyRouter(), partition=partition, producer_endpoint=producer_endpoint, capacity_bytes=capacity_bytes)
        self._speco_tq_publisher = publisher
        self._speco_tq_settings = settings
        self._speco_tq_inline_descriptor = inline_descriptor
        self._speco_tq_native_slot_count = source_slots
        self._speco_tq_cpu_direct_pack = cpu_direct_pack
        return publisher

def enable_configured_mhc_fusion(engine, *, environment, expected_workers):
    policy = environment.get('SPECO_MHC_PREFILL_FUSION', '')
    if policy not in ('', 'triton-exact'):
        raise ValueError('unsupported mHC prefill fusion policy')
    sequence_parallel = environment.get('SPECO_MHC_SEQUENCE_PARALLEL', '')
    if sequence_parallel not in ('', '0', '1'):
        raise ValueError('invalid mHC sequence parallel policy')
    sp = sequence_parallel == '1'
    if sp and (policy != 'triton-exact' or type(expected_workers) is not int or expected_workers not in (8, 16) or (environment.get('SPECO_TARGET_TP') != str(expected_workers))):
        raise ValueError('mHC sequence parallel requires matching verified TP8/TP16 fusion')
    if sp:
        from verl_speco.runtime.services import capture_prefill_engine_options
        capture_prefill_engine_options(environment)
    binding = (policy, sp, expected_workers)
    previous = getattr(engine, '_speco_mhc_prefill_policy', None)
    if previous is not None:
        if previous != binding:
            raise RuntimeError('reused mHC fusion policy changed')
        return getattr(engine, '_speco_mhc_prefill_receipt')
    if not policy:
        return {'enabled': False}
    if type(expected_workers) is not int or expected_workers not in (8, 16):
        raise ValueError('verified TP8/TP16 engine required')
    receipt = {'enabled': False}
    if policy:
        workers = engine.llm_engine.collective_rpc('speco_enable_mhc_prefill_fusion')
        if len(workers) != expected_workers or any((w.get('modules') != 92 or w.get('min_tokens') != 16384 or w.get('original_FP32_phi_and_sinkhorn') is not True or (w.get('diagnostic_only') is not False) or (w.get('sequence_parallel', False) is not sp) or (sp and w.get('sequence_parallel_world') != expected_workers) for w in workers)):
            raise RuntimeError('mHC prefill fusion did not bind every verified worker')
        if sp and (any((type(w.get('sequence_parallel_rank')) is not int for w in workers)) or {w['sequence_parallel_rank'] for w in workers} != set(range(expected_workers))):
            raise RuntimeError('mHC sequence parallel worker ranks are not unique and complete')
        receipt = {'enabled': True, 'workers': expected_workers, 'min_tokens': 16384, 'sequence_parallel': sp}
    engine._speco_mhc_prefill_policy = binding
    engine._speco_mhc_prefill_receipt = receipt
    return receipt
import torch.nn.functional as F

def split_mix(x, weight, alpha_pre, beta_pre, alpha_post, beta_post, alpha_res, beta_res, *, streams=4, iterations=20, eps=1e-06):
    pre, post, mix = weight.split([streams, streams, streams * streams], dim=-1)
    post = 2 * torch.sigmoid(post * alpha_post.float() + beta_post.float())
    mix = mix.unflatten(-1, (streams, streams))
    mix = mix * alpha_res.float() + beta_res.float().view(streams, streams)
    mix = mix.softmax(-1) + eps
    mix = mix / (mix.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        mix = mix / (mix.sum(-1, keepdim=True) + eps)
        mix = mix / (mix.sum(-2, keepdim=True) + eps)
    pre = torch.sigmoid(pre * alpha_pre.float() + beta_pre.float()) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.unflatten(-1, (streams, -1)), dim=-2)
    return (y, post, mix)

def post_mix(x, residual, post, mix):
    streams = post.shape[-1]
    return (post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(mix.unsqueeze(-1) * residual.unflatten(-1, (streams, -1)).unsqueeze(-2), dim=-3)).reshape(residual.shape).type_as(x)

def fixed_chunks(kernel, tensors, *, rows=2048):
    """Pad only the final token-independent chunk, preserving all original rows."""
    if type(rows) is not int or rows < 1 or (not tensors):
        raise ValueError('positive fixed chunk size required')
    count = tensors[0].shape[0]
    if count < 1 or any((t.ndim < 1 or t.shape[0] != count for t in tensors)):
        raise ValueError('identical nonempty token axes required')
    outputs = None
    for start in range(0, count, rows):
        length = min(rows, count - start)
        inputs = [t[start:start + length] for t in tensors]
        if length != rows:
            inputs = [torch.cat((t, t.new_zeros((rows - length, *t.shape[1:])))) for t in inputs]
        result = kernel(*inputs)
        values = result if isinstance(result, tuple) else (result,)
        if outputs is None:
            outputs = [v.new_empty((count, *v.shape[1:])) for v in values]
        for output, value in zip(outputs, values, strict=True):
            output[start:start + length].copy_(value[:length])
    return tuple(outputs) if isinstance(result, tuple) else outputs[0]

class MhcPrefillFusion:
    """Opt-in candidate wrapping one verified 92B mHC module, never decode."""

    def __init__(self, module, *, compile_kernel=None, rows=2048, min_tokens=4096):
        if module.num_stream != 4 or module.hc_eps != 1e-06 or module.norm_eps != 1e-06 or (module.mhc_recur_norm != 20) or (type(rows) is not int) or (rows < 1) or (type(min_tokens) is not int) or (min_tokens < rows):
            raise ValueError('verified 92B mHC arithmetic and chunk bounds required')
        self.module, self.rows, self.min_tokens = (module, rows, min_tokens)
        self.original_pre, self.original_post = (module.hc_pre, module.hc_post)
        if compile_kernel is None:
            import torchair
            from torchair.configs.compiler_config import CompilerConfig
            backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
            compile_kernel = lambda fn: torch.compile(fn, backend=backend, dynamic=False, fullgraph=True)

        def finishing(x, weight, ap, bp, ao, bo, ar, br):
            return split_mix(x, weight, ap, bp, ao, bo, ar, br)
        self.finish_kernel = compile_kernel(finishing)
        self.post_kernel = compile_kernel(post_mix)

    def pre(self, x):
        m = self.module
        if x.shape[0] < self.min_tokens or m.merge_layer_only_pre:
            return self.original_pre(x)
        dtype = x.dtype
        x = x.float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + m.norm_eps)
        if m.mhc_use_gamma:
            weight = F.linear(x * rsqrt * m.norm_gamma.unsqueeze(0).float(), m.phi.weight.float())
        else:
            weight = F.linear(x, m.phi.weight.float()) * rsqrt
        parameters = [m.branch_alpha_pre, m.branch_beta_pre, m.branch_alpha_post, m.branch_beta_post, m.branch_alpha_res, m.branch_beta_res]

        def kernel(x, weight):
            return self.finish_kernel(x, weight, *parameters)
        y, post, mix = fixed_chunks(kernel, (x, weight), rows=self.rows)
        return (y.to(dtype), post, mix)

    def post(self, x, residual, h_post, h_res):
        if x.shape[0] < self.min_tokens or self.module.merge_layer_only_pre:
            return self.original_post(x, residual, h_post, h_res)
        if h_post.dtype != torch.float32 or h_res.dtype != torch.float32:
            raise ValueError('mHC weights must retain FP32')
        return fixed_chunks(self.post_kernel, (x, residual, h_post, h_res), rows=self.rows)

class MhcTritonPrefillFusion(MhcPrefillFusion):
    """Candidate using direct Ascend kernels; same full-batch phi projection."""

    def __init__(self, module, *, operations=None, min_tokens=4096, original_sinkhorn=True):
        super().__init__(module, compile_kernel=lambda fn: fn, min_tokens=min_tokens)
        if operations is None:
            import verl_speco.producer.accelerator_adapters as operations
        self.operations = operations
        self.original_sinkhorn = original_sinkhorn

    def pre(self, x):
        m = self.module
        if x.shape[0] < self.min_tokens or m.merge_layer_only_pre:
            return self.original_pre(x)
        dtype = x.dtype
        x = x.float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + m.norm_eps)
        if m.mhc_use_gamma:
            weight = F.linear(x * rsqrt * m.norm_gamma.unsqueeze(0).float(), m.phi.weight.float())
        else:
            weight = F.linear(x, m.phi.weight.float()) * rsqrt
        parameters = [m.branch_alpha_pre.float(), m.branch_beta_pre.float(), m.branch_alpha_post.float(), m.branch_beta_post.float(), m.branch_alpha_res.float(), m.branch_beta_res.float()]
        if self.original_sinkhorn:
            pre, post, mix = m.hc_split_sinkhorn(weight)
        else:
            pre, post, mix = self.operations.split_sinkhorn(weight, parameters)
        return (self.operations.weighted_pre(x, pre, dtype=dtype), post, mix)

    def post(self, x, residual, h_post, h_res):
        if x.shape[0] < self.min_tokens or self.module.merge_layer_only_pre:
            return self.original_post(x, residual, h_post, h_res)
        return self.operations.post(x, residual, h_post, h_res)

class MhcSequenceParallelPrefill:

    def __init__(self, module, group, *, local=None, min_tokens=16384, max_tokens=None):
        world, rank = (group.world_size, group.rank_in_group)
        if type(world) is not int or world not in (1, 2, 4, 8, 16) or type(rank) is not int or (not 0 <= rank < world) or (type(min_tokens) is not int) or (min_tokens < 1) or (module.num_stream != 4) or (module.hc_eps != 1e-06) or (module.norm_eps != 1e-06) or (module.mhc_recur_norm != 20):
            raise ValueError('verified four-stream mHC and bounded TP geometry required')
        maximum = max(16384, world * 8192) if max_tokens is None else max_tokens
        if type(maximum) is not int or maximum < min_tokens:
            raise ValueError('invalid bounded prefill token range')
        self.module, self.group, self.world, self.rank = (module, group, world, rank)
        self.original_pre, self.original_post = (module.hc_pre, module.hc_post)
        if local is None:
            local = MhcTritonPrefillFusion(module, min_tokens=4096)
        self.local, self.min_tokens, self.max_tokens = (local, min_tokens, maximum)
        self.pending = None

    def _gather(self, value, tokens):
        if not value.is_contiguous():
            value = value.contiguous()
        result = self.group.all_gather(value, dim=0)
        if result.shape != (self.world * value.shape[0], *value.shape[1:]) or result.dtype != value.dtype or result.device != value.device:
            raise ValueError('TP gather changed mHC rows, dtype or device')
        return result[:tokens].contiguous()

    @staticmethod
    def _local_rows(value, start, width):
        part = value[start:start + width]
        missing = width - part.shape[0]
        if missing:
            part = F.pad(part, (0, 0, 0, missing))
        return part.contiguous()

    def pre(self, x):
        if self.pending is not None:
            raise RuntimeError('mHC prefill pre/post pair is already outstanding')
        if self.world == 1 or self.module.merge_layer_only_pre or x.shape[0] < self.min_tokens:
            return self.original_pre(x)
        if x.ndim != 2 or x.shape[1] != 4 * self.module.hidden_size or x.shape[0] > self.max_tokens:
            raise ValueError('prefill exceeds verified mHC geometry')
        tokens = x.shape[0]
        width = (tokens + self.world - 1) // self.world
        start = self.rank * width
        part = self._local_rows(x, start, width)
        y, post, mix = self.local.pre(part)
        if y.shape != (width, self.module.hidden_size) or post.shape != (width, 4) or mix.shape != (width, 4, 4):
            raise ValueError('local prefill mHC returned inconsistent shapes')
        result = self._gather(y, tokens)
        self.pending = (tokens, width, start, post, mix)
        return (result, post, mix)

    def post(self, x, residual, h_post, h_res):
        pending = self.pending
        if pending is None:
            if self.world > 1 and (not self.module.merge_layer_only_pre) and (x.shape[0] >= self.min_tokens):
                raise RuntimeError('parallel mHC post has no matching pre')
            return self.original_post(x, residual, h_post, h_res)
        try:
            tokens, width, start, post, mix = pending
            if h_post is not post or h_res is not mix or x.shape != (tokens, self.module.hidden_size) or (residual.shape != (tokens, 4 * self.module.hidden_size)):
                raise ValueError('mHC post differs from its outstanding prefill pair')
            x_part = self._local_rows(x, start, width)
            r_part = self._local_rows(residual, start, width)
            return self._gather(self.local.post(x_part, r_part, post, mix), tokens)
        finally:
            self.pending = None
from collections import defaultdict
import functools

class MultiNodePrefillProfile:

    def __init__(self, model, *, min_tokens=4096, max_calls=4096, event_factory=None, synchronize=None):
        if type(min_tokens) is not int or min_tokens < 1 or type(max_calls) is not int or (not 1 <= max_calls <= 16384):
            raise ValueError('prefill profiler requires bounded positive dimensions')
        self.min_tokens = min_tokens
        self.max_calls = max_calls
        self.event_factory = event_factory or (lambda: torch.npu.Event(enable_timing=True))
        self.synchronize = synchronize or torch.npu.synchronize
        self.records = []
        self.originals = []
        self.active = False
        self.closed = False
        targets = []
        for name, module in model.named_modules():
            kind = type(module).__name__
            methods = ('hc_pre', 'hc_post') if kind == 'mHCModule' else ()
            if kind in ('PanguV2VLMLAAttention', 'PanguV2VLMoE', 'PanguV2VLMLP'):
                methods = ('forward',)
            if kind == 'MomeConv':
                methods = ('causal',)
            if name.endswith('.mlp.experts'):
                methods = ('forward',)
            for method in methods:
                original = getattr(module, method, None)
                if not callable(original):
                    raise ValueError('profile target method is missing')
                targets.append((name + '.' + method, module, method, original))
        if not targets:
            raise ValueError('loaded model has no supported 92B prefill targets')
        try:
            for name, module, method, original in targets:
                had_override = method in vars(module)
                previous = vars(module).get(method)
                self.originals.append((module, method, had_override, previous))
                setattr(module, method, self._wrap(name, original))
        except BaseException:
            self.close()
            raise

    def _wrap(self, name, original):

        @functools.wraps(original)
        def call(*args, **kwargs):
            tensor = args[0] if args else kwargs.get('hidden_states', kwargs.get('x', kwargs.get('positions')))
            if not self.active or not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or (tensor.shape[0] < self.min_tokens):
                return original(*args, **kwargs)
            if len(self.records) >= self.max_calls:
                raise RuntimeError('prefill profile event capacity exceeded')
            start, end = (self.event_factory(), self.event_factory())
            record = dict(name=name, tokens=int(tensor.shape[0]), start=start, end=end, failed=False)
            self.records.append(record)
            start.record()
            try:
                return original(*args, **kwargs)
            except BaseException:
                record['failed'] = True
                raise
            finally:
                end.record()
        return call

    def begin(self):
        if self.closed or self.active or self.records:
            raise RuntimeError('prefill profile already started or closed')
        self.active = True
        return dict(profile_methods=len(self.originals), min_tokens=self.min_tokens, max_calls=self.max_calls, no_parameter_or_math_changes=True)

    def finish(self):
        if self.closed or not self.active:
            raise RuntimeError('prefill profile is not active')
        self.active = False
        self.synchronize()
        stats = defaultdict(lambda: dict(calls=0, scheduled_tokens=0, milliseconds=0.0, failures=0))
        for record in self.records:
            row = stats[record['name']]
            row['calls'] += 1
            row['scheduled_tokens'] += record['tokens']
            row['milliseconds'] += float(record['start'].elapsed_time(record['end']))
            row['failures'] += int(record['failed'])
        return dict(event_pairs=len(self.records), timings=[dict(name=name, **row) for name, row in sorted(stats.items())], nested_timings_must_not_be_summed=True, no_end_to_end_throughput_claim=True)

    def close(self):
        if not self.closed:
            self.active = False
            for module, method, had_override, previous in reversed(self.originals):
                if had_override:
                    setattr(module, method, previous)
                else:
                    delattr(module, method)
            self.originals.clear()
            self.records.clear()
            self.closed = True
from verl_speco.producer.input_reader import TokenizedRequest

def full_replay_prefill_rows(request: TokenizedRequest) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Keep the final sampled label in the payload consumed by DSpark.

    The general reader can request N-1 hidden rows while retaining N token IDs.
    target model's v3 payload requires equal row counts for IDs and features, so its
    direct prefill captures all N rows. The final teacher row is unused for a
    next-token label, but its token remains available to the preceding row.
    """
    ids, mask = (request.input_ids, request.loss_mask)
    if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor) or ids.ndim != 1 or (mask.ndim != 1) or (ids.shape != mask.shape) or (ids.numel() < 2):
        raise ValueError('saved replay requires aligned complete token and mask rows')
    token_ids = tuple((int(token) for token in ids.tolist()))
    loss_mask = tuple((float(value) for value in mask.tolist()))
    if any((token < 0 for token in token_ids)) or any((value not in (0.0, 1.0) for value in loss_mask)):
        raise ValueError('saved replay contains invalid token or mask values')
    try:
        prompt_length = loss_mask.index(1.0)
    except ValueError as error:
        raise ValueError('saved replay has no generated response') from error
    if prompt_length < 1 or loss_mask != (0.0,) * prompt_length + (1.0,) * (len(token_ids) - prompt_length):
        raise ValueError('saved replay supervision must be a complete response suffix')
    return (token_ids, loss_mask)

class MultiNodeStageProbe:

    def __init__(self, module, *, row_ranges, layer_ids=(0,), max_bytes=256 * 1024 * 1024):
        if not layer_ids or max_bytes <= 0 or (not callable(getattr(module, '_trace_stage', None))):
            raise ValueError('stage probe requires an existing emitter, layers and byte bound')
        self.module = module
        self.original = module._trace_stage
        self.row_ranges = row_ranges
        self.layers = frozenset(layer_ids)
        self.max_bytes = max_bytes
        self.baseline = {}
        self.metrics = {}
        self.active = None
        self.closed = False
        self.retained_bytes = 0
        module._trace_stage = self._emitter

    def _emitter(self, layer):
        original = self.original(layer)
        if layer not in self.layers:
            return original

        def emit(tag, tensor):
            original(tag, tensor)
            self.observe(layer, tag, tensor)
        return emit

    def begin(self, *, mode, engine_request_id, token_count):
        if self.closed or self.active is not None or mode not in ('baseline', 'compare') or (not engine_request_id) or (type(token_count) is not int) or (token_count <= 0):
            raise ValueError('stage probe mode or request is invalid')
        if mode == 'baseline' and self.baseline:
            raise RuntimeError('stage baseline is already retained')
        if mode == 'compare' and (not self.baseline):
            raise RuntimeError('stage baseline is missing')
        self.metrics = {}
        self.active = (mode, engine_request_id, token_count)

    def observe(self, layer, tag, tensor):
        if self.active is None or layer not in self.layers:
            return
        mode, request_id, count = self.active
        key = (layer, tag)
        ranges = self.row_ranges()
        if request_id not in ranges:
            return
        start, end = ranges[request_id]
        if end - start != count:
            return
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or start < 0 or (end > tensor.shape[0]):
            raise RuntimeError('stage tensor does not contain the owned prefill rows')
        if mode == 'baseline':
            if key in self.baseline:
                raise RuntimeError('stage emitted twice in one full prefill')
            size = count * tensor[0].numel() * tensor.element_size()
            if self.retained_bytes + size > self.max_bytes:
                raise RuntimeError('stage baseline byte capacity exceeded')
            self.baseline[key] = tensor[start:end].detach().to('cpu').contiguous().clone()
            self.retained_bytes += size
        else:
            if key not in self.baseline or key in self.metrics:
                raise RuntimeError('stage set changed or was emitted twice')
            reference = self.baseline[key].float()
            current = tensor[start:end].detach().to('cpu').float()
            if current.shape != reference.shape:
                raise RuntimeError('stage tensor shape differs from isolated baseline')
            difference = (current - reference).abs()
            rms = float(reference.square().mean().sqrt())
            self.metrics[key] = dict(layer=layer, stage=tag, rows=count, shape=list(reference.shape), mean_abs=float(difference.mean()), max_abs=float(difference.max()), relative_rms=float(difference.square().mean().sqrt()) / max(rms, 1e-12))

    def finish(self):
        if self.active is None:
            raise RuntimeError('stage probe has no active request')
        mode, _, count = self.active
        self.active = None
        if not self.baseline or (mode == 'compare' and set(self.metrics) != set(self.baseline)):
            raise RuntimeError('incomplete stage probe; cannot compare partial prefill')
        return dict(mode=mode, token_count=count, retained_bytes=self.retained_bytes, stages=[dict(layer=k[0], stage=k[1]) for k in self.baseline] if mode == 'baseline' else list(self.metrics.values()))

    def close(self):
        if not self.closed:
            self.module._trace_stage = self.original
            self.baseline.clear()
            self.metrics.clear()
            self.active = None
            self.retained_bytes = 0
            self.closed = True

def request_segmented_mhc_pre(original, tensor, ranges):
    """Diagnostic only: call the same stateless mHC arithmetic per request shape."""
    if not ranges:
        return original(tensor)
    spans = sorted(((start, end) for start, end in ranges.values() if end > start))
    cursor = 0
    for start, end in spans:
        if start != cursor or end > tensor.shape[0]:
            raise RuntimeError('mHC diagnostic request rows are not contiguous')
        cursor = end
    if not spans:
        raise RuntimeError('mHC diagnostic has no scheduled token rows')
    if cursor < tensor.shape[0]:
        spans.append((cursor, tensor.shape[0]))
    outputs = [original(tensor[start:end]) for start, end in spans]
    if any((not isinstance(o, tuple) or len(o) != 3 for o in outputs)):
        raise RuntimeError('mHC diagnostic output contract differs')
    result = []
    for column in zip(*outputs):
        if all((value is None for value in column)):
            result.append(None)
        elif any((not isinstance(value, torch.Tensor) for value in column)):
            raise RuntimeError('mHC diagnostic mixed tensor and absent output')
        else:
            result.append(torch.cat(column, dim=0))
    return tuple(result)
from itertools import islice
from verl_speco.producer.replay import iter_sequence_replay_prefill_requests

def select_physical_replay_requests(path, *, indices, offset, max_sequence_length, model_overlay):
    from verl_speco.standalone_tq_training_launcher import _stable_path_identity
    if not indices or indices != list(range(indices[0], indices[0] + len(indices))) or any((type(i) is not int or i < 0 for i in indices)) or (type(offset) is not int) or (offset < 0):
        raise ValueError('physical replay indices must form a nonnegative contiguous prefix')
    iterator = iter_sequence_replay_prefill_requests(Path(path), config={'max_sequence_length': max_sequence_length, 'on_truncated_supervision': 'keep'}, algorithm='DSPARK', target_model_id=str(model_overlay), target_model_revision=_stable_path_identity('target', str(model_overlay)), tokenizer_fingerprint=_stable_path_identity('tokenizer', str(model_overlay)))
    result = list(islice(iterator, offset + indices[0], offset + indices[-1] + 1))
    if len(result) != len(indices):
        raise ValueError('physical replay has too few committed sequences')
    if any((r.input_ids.numel() + 1 > max_sequence_length for r in result)):
        raise ValueError('physical replay full sequence exceeds the prefill context')
    return result

def select_forward_physical_replay_requests(cursor, *, indices, identity, max_sequence_length):
    """Keep sparse global producer ownership while reading committed input once."""
    from verl_speco.producer.replay import ForwardReplayCursor
    if not isinstance(cursor, ForwardReplayCursor) or type(max_sequence_length) is not int or max_sequence_length < 2:
        raise ValueError('physical replay requires a bounded forward cursor and context')
    requests = cursor.select_many(indices, identity=identity)
    for request in requests:
        ids, _mask, _boundary = physical_prefill_fields(request)
        if len(ids) + 1 > max_sequence_length:
            raise ValueError('physical replay full sequence exceeds the prefill context')
    return requests

def physical_prefill_fields(request):
    ids, mask = full_replay_prefill_rows(request)
    positions = request.position_ids.detach().cpu()
    if request.input_ids.dtype != torch.int64 or request.loss_mask.dtype != torch.float32 or positions.dtype != torch.int64 or (positions.shape != request.input_ids.shape) or (not torch.equal(positions, torch.arange(len(ids)))):
        raise ValueError('physical replay requires complete typed absolute positions')
    return (ids, mask, mask.index(1.0))
from verl_speco.producer.replay import ForwardReplayCursor
from verl_speco.runtime.services import build_physical_capture_command

def validate_persistent_capture_config(*, replay_factory, input_fingerprints, run_id, generation, producer_index, producer_count, target_tp, layer_ids, partition, ray_address, control_actor_name, capacity_bytes, max_sequence_length, max_batched_tokens, native_features, max_batch_size, native_slot_bytes, before_publish, source_capture_owner_ranks=0, source_native_buffer_batches=1, producer_incarnation='inc-0', source_slot_profile=''):
    """Check the complete producer constructor before allocating its TP engine."""
    validate_source_capture_owners(source_capture_owner_ranks, target_tp=target_tp, physical_slots=max_batch_size, native_features=native_features, release_enabled=callable(before_publish))
    validate_source_capture_buffers(source_native_buffer_batches, owner_ranks=source_capture_owner_ranks, target_tp=target_tp, physical_slots=max_batch_size, wide_full_profile=target_tp == 8 and max_batch_size == 8 and (max_batched_tokens == 131072), source_slot_profile=source_slot_profile)
    if not isinstance(producer_incarnation, str) or not producer_incarnation or len(producer_incarnation) > 256 or (re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]*', producer_incarnation) is None):
        raise ValueError('invalid physical producer incarnation')
    large_wave_quota = (capacity_bytes == producer_count * 4 * 1024 ** 3 or (capacity_bytes == producer_count * 8 * 1024 ** 3 and max_batched_tokens == 131072 and (max_batch_size == 8))) and producer_count in (3, 6, 7) and (target_tp == 16) and (max_batch_size in (2, 4, 8, 16)) and (native_features is True) and callable(before_publish)
    large_wave_quota |= capacity_bytes == 32 * 1024 ** 3 and producer_count == 14 and (target_tp == 8) and (max_batch_size == 8) and (native_slot_bytes == 512 * 1024 ** 2) and (max_batched_tokens in (8192, 16384, 32768, 49152, 65536)) and (native_features is True) and callable(before_publish)
    large_wave_quota |= capacity_bytes == 32 * 1024 ** 3 and producer_count == 7 and (target_tp == 16) and (max_batch_size == 16) and (source_capture_owner_ranks in (6, 8)) and (native_slot_bytes == 512 * 1024 ** 2) and (max_batched_tokens == 65536) and (native_features is True) and callable(before_publish)
    large_wave_quota |= capacity_bytes == 24 * 1024 ** 3 and producer_count == 6 and (target_tp == 8) and (max_batch_size == 4) and (source_capture_owner_ranks == 4) and (source_native_buffer_batches == 2) and (native_slot_bytes == 512 * 1024 ** 2) and (max_batched_tokens == 65536) and (native_features is True) and callable(before_publish)
    large_wave_quota |= capacity_bytes == 48 * 1024 ** 3 and producer_count == 6 and (target_tp == 8) and (max_batch_size == 8) and (source_capture_owner_ranks == 8) and (source_native_buffer_batches == 2) and ((max_sequence_length, native_slot_bytes) in ((16384, 512 * 1024 ** 2), (49152, 2 * 1024 ** 3), (65536, 2 * 1024 ** 3))) and (max_batched_tokens == 131072) and (native_features is True) and callable(before_publish)
    large_wave_quota |= capacity_bytes == 48 * 1024 ** 3 and producer_count == 6 and (target_tp == 8) and (max_batch_size == 16) and (source_capture_owner_ranks == 8) and (source_native_buffer_batches == 2) and ((max_sequence_length, native_slot_bytes) in ((16384, 512 * 1024 ** 2), (49152, 2 * 1024 ** 3))) and (max_batched_tokens == 131072) and (native_features is True) and callable(before_publish)
    triple_profile = capacity_bytes == 24 * source_native_buffer_batches * 1024 ** 3 and producer_count == 6 and (target_tp == 8) and (max_batch_size == 8) and (source_capture_owner_ranks == 8) and (source_native_buffer_batches in (3, 4)) and ((max_sequence_length, native_slot_bytes) == (16384, 512 * 1024 ** 2) or ((max_sequence_length, native_slot_bytes) == (49152, 2 * 1024 ** 3) and source_native_buffer_batches in (3, 4))) and (max_batched_tokens == 131072) and (native_features is True) and callable(before_publish)
    if source_native_buffer_batches in (3, 4) and (not triple_profile):
        raise ValueError('three-bank capture differs from the exact frozen FULL source profile')
    if source_native_buffer_batches == 10:
        if not (source_slot_profile == PROFILE and capacity_bytes == 96 * 1024 ** 3 and (producer_count == 6) and (target_tp == 8) and (max_batch_size == 8) and (source_capture_owner_ranks == 8) and (max_sequence_length == 49152) and (native_slot_bytes == 2 * 1024 ** 3) and (max_batched_tokens == 131072) and (native_features is True) and callable(before_publish)):
            raise ValueError('tiered slots require the complete frozen48K FULL profile')
        large_wave_quota = True
    large_wave_quota |= triple_profile
    build_physical_capture_command(run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=0, sample_count=1)
    if type(target_tp) is not int or target_tp not in (8, 16) or (not isinstance(layer_ids, tuple)) or (len(layer_ids) != 5) or any((type(i) is not int or i < 0 for i in layer_ids)) or (tuple(sorted(set(layer_ids))) != layer_ids) or (type(max_sequence_length) is not int) or (max_sequence_length not in (8192, 16384, 49152, 65536)) or (type(max_batched_tokens) is not int) or (max_batched_tokens < 1) or (type(capacity_bytes) is not int) or (not (0 < capacity_bytes <= 16 * 1024 ** 3 or large_wave_quota)) or any((not isinstance(s, str) or not s for s in (partition, ray_address, control_actor_name))) or (not callable(input_fingerprints)) or (type(native_features) is not bool) or (before_publish is not None and (not native_features or not callable(before_publish))) or (type(max_batch_size) is not int) or (max_batch_size not in (2, 4, 8, 16)) or (type(native_slot_bytes) is not int) or (native_slot_bytes != source_capture_slot_bytes(target_tp=target_tp, physical_slots=max_batch_size, owner_ranks=source_capture_owner_ranks, long_context=max_sequence_length in (49152, 65536) and max_batch_size in (8, 16) and (native_slot_bytes == 2 * 1024 ** 3))):
        raise ValueError('invalid persistent physical replay capture configuration')

class PersistentPhysicalReplayCapture:
    """The caller owns engine startup, native lease draining and role cleanup.

    Bootstrap checks each initial physical child against its isolated prefill
    fingerprint on every TP rank. Later commands reuse that engine and immutable
    input identity. A partial capture or publication poisons this owner.
    """

    def __init__(self, *, llm, replay_factory, input_fingerprints, run_id, generation, producer_index, producer_count, target_tp, layer_ids, partition, ray_address, control_actor_name, capacity_bytes, max_sequence_length=65536, max_batched_tokens=32768, sampling_factory=None, request_output_type=None, native_features=False, max_batch_size=2, native_slot_bytes=2 * 1024 ** 3, before_publish=None, source_capture_owner_ranks=0, source_native_buffer_batches=1, producer_incarnation='inc-0', replay_cursor_factory=None, coalesced_engine_batch_size=0, before_fragment=None, exact_fragment_admission=False, source_slot_profile=''):
        validate_persistent_capture_config(replay_factory=replay_factory, input_fingerprints=input_fingerprints, run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, target_tp=target_tp, layer_ids=layer_ids, partition=partition, ray_address=ray_address, control_actor_name=control_actor_name, capacity_bytes=capacity_bytes, max_sequence_length=max_sequence_length, max_batched_tokens=131072 if coalesced_engine_batch_size == 40 and max_batched_tokens == 65536 else max_batched_tokens, native_features=native_features, max_batch_size=max_batch_size, native_slot_bytes=native_slot_bytes, before_publish=before_publish, source_capture_owner_ranks=source_capture_owner_ranks, source_native_buffer_batches=source_native_buffer_batches, producer_incarnation=producer_incarnation, source_slot_profile=source_slot_profile)
        self._inputs = input_fingerprints
        self._fingerprints = json.loads(json.dumps(input_fingerprints(), sort_keys=True))
        if any((not isinstance(self._fingerprints.get(k), str) or not self._fingerprints[k] for k in ('tokenizer_fingerprint', 'target_model_fingerprint', 'dataset_fingerprint'))):
            raise ValueError('physical capture requires verified input fingerprints')
        self._identity = hashlib.sha256(json.dumps(dict(inputs=self._fingerprints, run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, context=max_sequence_length, layer_ids=layer_ids, producer_incarnation=producer_incarnation), sort_keys=True).encode()).hexdigest()
        if replay_cursor_factory is not None and (not callable(replay_cursor_factory)):
            raise ValueError('indexed replay cursor factory must be callable')
        self._cursor = ForwardReplayCursor(replay_factory, identity=self._identity) if replay_cursor_factory is None else replay_cursor_factory(identity=self._identity)
        if not isinstance(self._cursor, ForwardReplayCursor):
            raise ValueError('physical capture requires a bounded forward replay cursor')
        self._llm = llm
        self._run_id, self._generation = (run_id, generation)
        self._producer_index, self._producer_count = (producer_index, producer_count)
        self._producer_incarnation = producer_incarnation
        self._tp, self._layers = (target_tp, layer_ids)
        self._partition, self._ray, self._actor = (partition, ray_address, control_actor_name)
        self._capacity = capacity_bytes
        self._context, self._batch_tokens = (max_sequence_length, max_batched_tokens)
        self._sampling_factory, self._output_type = (sampling_factory, request_output_type)
        self._audit_done = False
        self._native_features = native_features
        self._max_batch_size, self._slot_bytes = (max_batch_size, native_slot_bytes)
        self._source_owner_ranks = source_capture_owner_ranks
        self._source_buffer_batches = source_native_buffer_batches
        self._source_slot_profile = source_slot_profile
        self._before_publish = before_publish
        if type(coalesced_engine_batch_size) is not int or coalesced_engine_batch_size not in (0, 40):
            raise ValueError('invalid coalesced engine batch size')
        if coalesced_engine_batch_size and (not (max_batch_size == 8 and target_tp == 8 and (producer_count == 6) and (source_capture_owner_ranks == 8) and (source_native_buffer_batches in (4, 10)) and (max_batched_tokens in (65536, 131072)) and (max_sequence_length in (16384, 49152)) and (native_features is True) and callable(before_publish) and callable(before_fragment))):
            raise ValueError('token coalescing requires the complete four-bank FULL source profile')
        if before_fragment is not None and (not coalesced_engine_batch_size):
            raise ValueError('fragment publication cannot be used by legacy captures')
        self._coalesced_engine_batch_size = coalesced_engine_batch_size
        if type(exact_fragment_admission) is not bool or (exact_fragment_admission and (not coalesced_engine_batch_size)):
            raise ValueError('exact fragment admission needs explicit coalesced capture')
        self._exact_fragment_admission = exact_fragment_admission
        self._before_fragment = before_fragment
        self._pending_source_captures = []
        self._failed = False

    def close(self):
        try:
            if self._pending_source_captures:
                self._llm.llm_engine.collective_rpc('speco_capture_abort')
        finally:
            self._pending_source_captures.clear()
            self._failed = True
            self._cursor.close()

    def _run(self):
        if self._output_type is None:
            from vllm.outputs import RequestOutput
            self._output_type = RequestOutput
        self._last_engine_step_count = None
        engine = self._llm.llm_engine
        step = getattr(engine, 'step', None)
        if os.environ.get('SPECO_CAPTURE_KV_AWARE_BATCHING') != '1' or not callable(step):
            return self._llm._run_engine(self._output_type, use_tqdm=False)
        instance_step = 'step' in vars(engine)
        self._last_engine_step_count = 0

        def counted(*args, **kwargs):
            self._last_engine_step_count += 1
            return step(*args, **kwargs)
        engine.step = counted
        try:
            return self._llm._run_engine(self._output_type, use_tqdm=False)
        finally:
            if instance_step:
                engine.step = step
            else:
                del engine.step

    def _add(self, ids):
        if self._sampling_factory is None:
            from vllm import SamplingParams
            self._sampling_factory = SamplingParams
        engine_id = self._llm._add_request({'prompt_token_ids': list(ids)}, self._sampling_factory(temperature=0, seed=42, max_tokens=1))
        public, separator, _suffix = engine_id.partition('-')
        if not separator or not public.isdecimal():
            raise RuntimeError('locked vLLM engine request ID format changed')
        return (engine_id, public)

    def _arm(self, ids, engine_id):
        return dict(request_id='speco-92b-capture-smoke', engine_request_id=engine_id, token_ids=ids, positions=tuple(range(len(ids))), layer_ids=self._layers, hidden_size=2560, max_payload_bytes=self._slot_bytes)

    def _check_armed(self, proofs, arms, *, batched=False):
        count = len(arms)
        rows = proofs if batched else [[p] for p in proofs]
        if len(rows) != self._tp or any((len(row) != count for row in rows)):
            raise RuntimeError('physical capture did not arm every child on every TP worker')
        for rank, row in enumerate(rows):
            for proof, arm in zip(row, arms):
                owner = arm.get('capture_owner_rank')
                status = 'owner_skipped' if owner is not None and rank != owner else 'armed'
                if not isinstance(proof, dict) or proof.get('status') != status or proof.get('engine_request_id') != arm['engine_request_id'] or (owner is not None and (type(proof.get('rank')) is not int or proof['rank'] != rank or type(proof.get('capture_owner_rank')) is not int or (proof['capture_owner_rank'] != owner))):
                    raise RuntimeError('physical capture did not arm its exact TP owner')

    @staticmethod
    def _write_manifest(path, value):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as writer:
                temporary = Path(writer.name)
                json.dump(value, writer, sort_keys=True)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _complete_owned_publications(self, *, children, paths, rows, references, coalesced, logical_children, phase, started, indices, distributed):
        if len(rows) != self._tp or any((len(r) != len(children) for r in rows)):
            raise RuntimeError('physical publisher did not return every child and rank')
        manifests = []
        for i, c in enumerate(children):
            proofs = [r[i] for r in rows]
            owner_rank = (logical_children[i].capture_owner_rank if coalesced else i % self._source_owner_ranks) if distributed else None
            leader = validate_child_publication(proofs, target_tp=self._tp, key=make_v3_sample_key(c['metadata']), owner_rank=owner_rank, request_id=c['metadata'].request_id)
            manifest = dict(key=leader['key'], metadata=c['metadata'].to_dict(), partition=self._partition, control_actor_name=self._actor, worker_count=self._tp, worker_receipts=proofs, physical_batch_size=logical_children[i].physical_batch_size if coalesced else len(children), isolated_bootstrap_sha256=references[i] if references else '', physical_bootstrap_completed=bool(references) or self._audit_done)
            if distributed:
                manifest.update(source_capture_owner_ranks=self._source_owner_ranks, capture_owner_rank=owner_rank, physical_child_index=logical_children[i].physical_child_index if coalesced else i, payload_digest_workers=1)
            if self._source_buffer_batches in (2, 3, 4, 10):
                manifest.update(source_native_buffer_batches=self._source_buffer_batches, registered_source_buffers=self._max_batch_size * self._source_buffer_batches)
            if self._source_slot_profile:
                manifest.update(source_native_slot_profile=self._source_slot_profile, source_slot_sizes=list(SLOT_SIZES), registered_source_bytes=self._source_owner_ranks * OWNER_BYTES)
            if self._native_features:
                from verl_speco.transport.native_transfer import ascend_feature_transport_descriptor_sha as _descriptor_sha
                descriptor = leader.get('native_descriptor')
                if not isinstance(descriptor, dict) or descriptor.get('run_id') != self._run_id or type(descriptor.get('generation')) is not int or (descriptor['generation'] != self._generation) or (descriptor.get('lease_id') != manifest['key']) or (descriptor.get('descriptor_sha256') != _descriptor_sha({k: v for k, v in descriptor.items() if k != 'descriptor_sha256'})) or (sum((f['nbytes'] for f in descriptor['fields'])) != c['metadata'].payload_nbytes) or (not any((f['name'] == 'hidden_states' and f['dtype'] == 'torch.bfloat16' and (f['shape'] == [len(c['ids']), 6 * 2560]) for f in descriptor['fields']))):
                    raise RuntimeError('physical native descriptor differs from captured child')
                encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()
                if not 0 < len(encoded) <= 65536:
                    raise RuntimeError('physical native descriptor exceeds control budget')
                manifest.update(native_descriptor=descriptor, tq_payload_fields=['speco_native_descriptor_v1'], tq_descriptor_bytes=len(encoded))
            manifests.append(manifest)
        for path, value in zip(paths, manifests):
            self._write_manifest(path, value)
        if references:
            self._audit_done = True
        print('persistent_physical_capture_timing ' + json.dumps(dict(sequence_nos=list(indices), physical_batch_size=len(children), original_tokens=sum((len(c['ids']) for c in children)), token_fill=sum((len(c['ids']) for c in children)) / self._batch_tokens, coalesced=coalesced, bootstrap_audited=bool(references), wallclock_seconds=time.perf_counter() - started, **phase)), flush=True)

    def finish_pending_capture(self, ticket):
        if self._failed or not self._pending_source_captures or ticket is not self._pending_source_captures[0]:
            raise RuntimeError('source capture publications must finish in FIFO ticket order')
        engine = self._llm.llm_engine
        try:
            started = time.perf_counter()
            keys = [make_v3_sample_key(c['metadata']) for c in ticket['children']]
            bounds = ticket.pop('publication_fragment_bounds', None)
            if bounds is None:
                rows = engine.collective_rpc('speco_capture_collect_publications', kwargs={'keys': keys})
            else:
                rows = [[] for _ in range(self._tp)]
                for begin, end in bounds:
                    fragment = engine.collective_rpc('speco_capture_collect_publications', kwargs={'keys': keys[begin:end]})
                    if len(fragment) != self._tp or any((len(row) != end - begin for row in fragment)):
                        raise RuntimeError('source completed fragment is missing a TP child proof')
                    for target, part in zip(rows, fragment):
                        target.extend(part)
            ticket['phase']['publication_rpc_seconds'] += time.perf_counter() - started
            self._complete_owned_publications(rows=rows, **ticket)
            self._pending_source_captures.pop(0)
        except BaseException as failure:
            try:
                engine.collective_rpc('speco_capture_abort')
            except BaseException as error:
                failure.add_note(f'async TP abort failed: {type(error).__name__}')
            self._pending_source_captures.clear()
            self.close()
            raise

    def capture(self, indices, paths, *, logical_children=None, defer_publication=False, finish_pending_before_enqueue=None):
        if self._failed:
            raise RuntimeError('physical capture owner is closed or failed')
        if type(defer_publication) is not bool:
            raise ValueError('source publication deferral must be a boolean')
        if defer_publication and (not (self._tp == 8 and self._source_owner_ranks == 8 and (self._source_buffer_batches in (4, 10)) and (self._max_batch_size == 8) and self._native_features and callable(self._before_publish) and (not logical_children or (self._coalesced_engine_batch_size == 40 and self._exact_fragment_admission and callable(self._before_fragment) and (len(logical_children) <= 16) and callable(finish_pending_before_enqueue) and (os.environ.get('SPECO_CAPTURE_KV_AWARE_BATCHING') == '1') and (os.environ.get('SPECO_CAPTURE_KV_MAX_SEQUENCES') == '16'))))):
            raise ValueError('async source requires the fixed eight-child native four-bank profile')
        if finish_pending_before_enqueue is not None and (not defer_publication or not logical_children or (not callable(finish_pending_before_enqueue))):
            raise ValueError('pending collection callback requires the async coalesced profile')
        if defer_publication and logical_children and (len({c.command_step for c in logical_children}) > 2):
            raise ValueError('async engine capture exceeds two logical command fragments')
        if self._pending_source_captures and (not defer_publication):
            raise RuntimeError('pending source publications must finish before synchronous captures')
        if len(self._pending_source_captures) >= 2:
            raise RuntimeError('source capture exceeds the two-ticket publication bound')
        engine = self._llm.llm_engine
        started = time.perf_counter()
        phase = dict(bootstrap_engine_seconds=0.0, physical_engine_seconds=0.0, publication_wait_seconds=0.0, publication_rpc_seconds=0.0)
        try:
            paths = tuple((Path(p) for p in paths))
            coalesced = logical_children is not None
            if coalesced:
                logical_children = tuple(logical_children)
                if not self._coalesced_engine_batch_size or len(logical_children) != len(indices) or any((c.sequence_no != i or not 1 <= c.physical_batch_size <= 8 or (not 0 <= c.physical_child_index < c.physical_batch_size) or (c.capture_owner_rank != c.physical_child_index % 8) or (i != self._producer_index + (c.command_step + c.physical_child_index) * self._producer_count) for i, c in zip(indices, logical_children))):
                    raise ValueError('coalesced logical capture provenance differs')
            bound = self._coalesced_engine_batch_size if coalesced else self._max_batch_size
            if not isinstance(indices, (tuple, list)) or not 1 <= len(indices) <= bound or len(paths) != len(indices) or (len(set(paths)) != len(paths)) or any((type(i) is not int or i < 0 or i % self._producer_count != self._producer_index for i in indices)) or any((not p.is_absolute() or not p.resolve().is_relative_to('/tmp') or p.name != f'capture-{i:04d}.json' or p.exists() for i, p in zip(indices, paths))):
                raise ValueError('physical capture indices or fresh manifests differ from owner')
            if self._inputs() != self._fingerprints:
                raise ValueError('physical capture immutable inputs changed')
            if coalesced:
                requests = tuple((self._cursor.select(i, identity=self._identity) for i in indices))
                if any((r is None for r in requests)):
                    raise ValueError('coalesced replay has too few committed sequences')
            else:
                requests = select_forward_physical_replay_requests(self._cursor, indices=indices, identity=self._identity, max_sequence_length=self._context + 1)
            fields = [physical_prefill_fields(request) for request in requests]
            if any((len(ids) > self._context or len(ids) * 30740 > self._slot_bytes for ids, _, _ in fields)) or sum((len(ids) * 30740 for ids, _, _ in fields)) > min(self._capacity, 4 * 1024 ** 3) or (len(indices) > 1 and sum((len(ids) for ids, _, _ in fields)) > self._batch_tokens):
                raise ValueError('physical capture exceeds fixed slots or full batched prefill budget')
            if coalesced and any((c.original_tokens != len(ids) for c, (ids, _, _) in zip(logical_children, fields))):
                raise ValueError('coalesced source tokens differ from frozen metadata')
            metadatas = []
            for index, request, (ids, mask, prompt) in zip(indices, requests, fields):
                metadatas.append(SampleMetadataV3(schema_version=3, run_id=self._run_id, generation=self._generation, producer_id=f'producer-{self._producer_index}', producer_incarnation=self._producer_incarnation, request_id='speco-92b-capture-smoke', sample_id=request.sample_id, sequence_no=index, target_revision='92b-sft-v11', data_version=0, target_layer_ids=self._layers, hidden_states_layout='dflash_aux_plus_last', hidden_dtype='bfloat16', token_count=len(ids), prompt_token_count=prompt, supervised_token_count=int(sum(mask)), payload_nbytes=len(ids) * 30740, **{k: self._fingerprints[k] for k in ('tokenizer_fingerprint', 'target_model_fingerprint', 'dataset_fingerprint')}))
            references = []
            if len(indices) > 1 and (not self._audit_done):
                for ids, _, _ in fields:
                    engine_id, public = self._add(ids)
                    arm = self._arm(ids, engine_id)
                    self._check_armed(engine.collective_rpc('speco_capture_begin', kwargs=arm), [arm])
                    engine_started = time.perf_counter()
                    outputs = self._run()
                    phase['bootstrap_engine_seconds'] += time.perf_counter() - engine_started
                    if len(outputs) != 1 or outputs[0].request_id != public:
                        raise RuntimeError('isolated physical bootstrap did not finish alone')
                    digests = engine.collective_rpc('speco_capture_prefill_digest')
                    if len(digests) != self._tp or any((d['prefill_token_count'] != len(ids) for d in digests)) or any((not isinstance(d['prefill_sha256'], str) or re.fullmatch('[0-9a-f]{64}', d['prefill_sha256']) is None for d in digests)) or (len({d['prefill_sha256'] for d in digests}) != 1):
                        raise RuntimeError('physical isolated fingerprints disagree across TP workers')
                    references.append(digests[0]['prefill_sha256'])
            children = []
            for index, request, (ids, mask, prompt) in zip(indices, requests, fields):
                engine_id, public = self._add(ids)
                metadata = metadatas[len(children)]
                children.append(dict(engine_id=engine_id, public=public, ids=ids, mask=mask, metadata=metadata))
            arms = [self._arm(c['ids'], c['engine_id']) for c in children]
            distributed = self._source_owner_ranks > 0 and (len(children) > 1 or (coalesced and logical_children[0].physical_batch_size > 1))
            if distributed:
                for index, arm in enumerate(arms):
                    arm['capture_owner_rank'] = logical_children[index].capture_owner_rank if coalesced else index % self._source_owner_ranks
                    if coalesced:
                        arm['physical_child_index'] = logical_children[index].physical_child_index
            if len(children) == 1:
                arms[0]['allow_chunked_prefill'] = True
            many_arms = len(children) > 1 or distributed
            self._check_armed(engine.collective_rpc('speco_capture_begin_many' if many_arms else 'speco_capture_begin', kwargs={'requests': arms} if many_arms else arms[0]), arms, batched=many_arms)
            engine_started = time.perf_counter()
            outputs = self._run()
            phase['physical_engine_seconds'] = time.perf_counter() - engine_started
            if self._last_engine_step_count is not None:
                phase['physical_engine_steps'] = self._last_engine_step_count
            if len(outputs) != len(children) or {o.request_id for o in outputs} != {c['public'] for c in children}:
                raise RuntimeError('physical prefill did not finish every owned request')
            publication = []
            for i, c in enumerate(children):
                entry = dict(engine_request_id=c['engine_id'], metadata=c['metadata'].to_dict(), loss_mask=c['mask'], partition=self._partition, ray_address=self._ray, control_actor_name=self._actor, capacity_bytes=self._capacity, generated_token_ids=())
                if references:
                    entry.update(expected_prefill_sha256=references[i], expected_prefill_token_count=len(c['ids']))
                publication.append(entry)
            if defer_publication and coalesced:
                finish_pending_before_enqueue()
                if self._pending_source_captures:
                    raise RuntimeError('previous source publication must drain before new field enqueue')
            if self._before_publish is not None:
                wait_started = time.perf_counter()
                if coalesced and self._exact_fragment_admission:
                    first_end = next((i for i, c in enumerate(logical_children) if c.command_step != logical_children[0].command_step), len(logical_children))
                    self._before_publish(owners=tuple((c.capture_owner_rank for c in logical_children[:first_end])), **dict(required_bytes=tuple((c['metadata'].payload_nbytes + 1024 for c in children[:first_end]))) if self._source_slot_profile else {})
                elif self._source_slot_profile:
                    self._before_publish(required_bytes=tuple((c['metadata'].payload_nbytes + 1024 for c in children)))
                else:
                    self._before_publish()
                phase['publication_wait_seconds'] = time.perf_counter() - wait_started
            publication_started = time.perf_counter()
            if defer_publication and self._audit_done and distributed:
                bounds = []
                begin = 0
                while begin < len(children):
                    end = begin + 1
                    while coalesced and end < len(children) and (logical_children[end].command_step == logical_children[begin].command_step):
                        end += 1
                    if not coalesced:
                        end = len(children)
                    bounds.append((begin, end))
                    begin = end
                if len(bounds) > 2:
                    raise ValueError('async engine capture exceeds two logical command fragments')
                for begin, end in bounds:
                    if begin:
                        self._before_fragment(owners=tuple((c.capture_owner_rank for c in logical_children[begin:end])), **dict(required_bytes=tuple((c['metadata'].payload_nbytes + 1024 for c in children[begin:end]))) if self._source_slot_profile else {})
                    method = 'speco_capture_enqueue_fragment' if coalesced else 'speco_capture_enqueue_many'
                    tickets = engine.collective_rpc(method, kwargs={'requests': publication[begin:end]})
                    if len(tickets) != self._tp or any((len(row) != end - begin for row in tickets)):
                        raise RuntimeError('source enqueue did not return every fragment child on every TP worker')
                    for rank, row in enumerate(tickets):
                        for local_index, ticket in enumerate(row):
                            index = begin + local_index
                            owner = logical_children[index].capture_owner_rank if coalesced else index
                            expected_status = 'publication_pending' if rank == owner else 'owner_skipped'
                            if ticket.get('status') != expected_status or type(ticket.get('rank')) is not int or ticket['rank'] != rank or (type(ticket.get('capture_owner_rank')) is not int) or (ticket['capture_owner_rank'] != owner) or (ticket.get('request_id') != children[index]['metadata'].request_id) or (expected_status == 'publication_pending' and ticket.get('key') != make_v3_sample_key(children[index]['metadata'])):
                                raise RuntimeError('source fragment enqueue differs from fixed TP capture ownership')
                phase['publication_rpc_seconds'] = time.perf_counter() - publication_started
                phase['source_async_publication'] = True
                ticket = dict(children=children, paths=paths, references=references, coalesced=coalesced, logical_children=logical_children, phase=phase, distributed=distributed, started=started, indices=tuple(indices))
                if coalesced:
                    ticket['publication_fragment_bounds'] = tuple(bounds)
                self._pending_source_captures.append(ticket)
                return ticket
            if coalesced and distributed:
                rows = [[] for _ in range(self._tp)]
                start = 0
                while start < len(children):
                    end = start + 1
                    while end < len(children) and logical_children[end].command_step == logical_children[start].command_step:
                        end += 1
                    if start:
                        if self._exact_fragment_admission:
                            self._before_fragment(owners=tuple((c.capture_owner_rank for c in logical_children[start:end])), **dict(required_bytes=tuple((c['metadata'].payload_nbytes + 1024 for c in children[start:end]))) if self._source_slot_profile else {})
                        else:
                            self._before_fragment()
                    fragment = engine.collective_rpc('speco_capture_publish_fragment', kwargs={'requests': publication[start:end]})
                    if len(fragment) != self._tp or any((len(r) != end - start for r in fragment)):
                        raise RuntimeError('fragment publisher did not return every child and rank')
                    for row, part in zip(rows, fragment):
                        row.extend(part)
                    start = end
            elif len(children) > 1:
                rows = engine.collective_rpc('speco_capture_publish_many', kwargs={'requests': publication})
            else:
                entry = dict(publication[0])
                entry.pop('engine_request_id')
                rows = [[p] for p in engine.collective_rpc('speco_capture_publish', kwargs=entry)]
            phase['publication_rpc_seconds'] = time.perf_counter() - publication_started
            return self._complete_owned_publications(children=children, paths=paths, rows=rows, references=references, coalesced=coalesced, logical_children=logical_children, phase=phase, started=started, indices=indices, distributed=distributed)
        except BaseException as failure:
            for stage, cleanup in (('replay close', self.close), ('TP capture abort', lambda: engine.collective_rpc('speco_capture_abort'))):
                try:
                    cleanup()
                except BaseException as cleanup_error:
                    failure.add_note(f'{stage} failed: {type(cleanup_error).__name__}')
            raise
from verl_speco.producer.batching import TokenBudgetCommandCoalescer

class PipelinedSourceCommandCoalescer:

    def __init__(self, coalescer):
        if not isinstance(coalescer, TokenBudgetCommandCoalescer) or coalescer.active is not None:
            raise ValueError('pipelined source requires its inactive owned coalescer')
        self.owner = coalescer
        self.pending = deque()

    def next_batch(self, **options):
        if len(self.pending) >= 2:
            raise RuntimeError('source exceeds two reserved engine batches')
        batch = self.owner.next_batch(**options)
        if batch is not None:
            self.pending.append((batch, self.owner.active))
            self.owner.active = None
        return batch

    def commit_batch(self, batch, proofs, *, acknowledge):
        if not self.pending or self.pending[0][0] is not batch or self.owner.active is not None:
            raise RuntimeError('completed source engine batches must commit in FIFO order')
        self.owner.active = self.pending[0][1]
        result = self.owner.commit_batch(proofs, acknowledge=acknowledge)
        self.pending.popleft()
        return result

    def require_complete(self):
        if self.pending or self.owner.pending or self.owner.commands or self.owner.active or self.owner.failed:
            raise RuntimeError('source ended with incomplete reserved capture or publication')

def transition_paused_graph_memory(proof, source_config, target_config):
    before = json.loads(source_config.read_text())
    after = json.loads(target_config.read_text())
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    old, new = (before.get('gpu_memory_utilization'), after.get('gpu_memory_utilization'))
    if changed != {'gpu_memory_utilization'} or type(old) not in (float, int) or type(new) not in (float, int) or (not 0 < new < old < 1):
        raise ValueError('graph config transition only permits more memory headroom')
    source_sha = hashlib.sha256(source_config.read_bytes()).hexdigest()
    target_sha = hashlib.sha256(target_config.read_bytes()).hexdigest()
    if proof.get('state') != 'PAUSED' or proof.get('run_id') != before['run_id'] or proof.get('config_sha256') != source_sha or (proof.get('owned_processes_remaining') != 0) or (proof.get('npus_free') is not True):
        raise ValueError('graph config transition requires the exact saved and released source owner')
    result = deepcopy(proof)
    result['config_sha256'] = target_sha
    result['config_transition'] = dict(source_config=str(source_config), target_config=str(target_config), source_sha256=source_sha, target_sha256=target_sha, changed_fields=sorted(changed), old_gpu_memory_utilization=old, new_gpu_memory_utilization=new)
    return result
from verl_speco.runtime.source_profiles import sequence_schedule_fingerprint
from verl_speco.runtime.artifacts import read_published_json

def graph_tuning_path(config):
    return Path(config['output_root']) / ('serving-tuning.graph-' + sequence_schedule_fingerprint(config) + '.json')

def validate_graph_tuning(config, profile):
    if not isinstance(profile, dict) or set(profile) != {'schema_version', 'run_id', 'schedule_fingerprint', 'prefill_batched_tokens'} or type(profile['schema_version']) is not int or (profile['schema_version'] != 1) or (profile['run_id'] != config['run_id']) or (profile['schedule_fingerprint'] != sequence_schedule_fingerprint(config)):
        raise ValueError('graph tuning must match the complete serving config')
    budget = profile['prefill_batched_tokens']
    if type(budget) is not int or not 2048 <= budget <= 65536:
        raise ValueError('graph prefill budget must be an integer from 2048 to 65536')
    return budget

def graph_prefill_budget(config):
    return validate_graph_tuning(config, read_published_json(graph_tuning_path(config), max_bytes=4096))
PROFILE = 'tiered-512m8-2g2-v1'
SLOT_SIZES = (512 * 1024 ** 2,) * 8 + (2 * 1024 ** 3,) * 2
OWNER_BYTES = sum(SLOT_SIZES)

def require_profile(profile, *, owners, physical_batch_size, slot_count, owner_bytes):
    if profile != PROFILE or type(owners) is not int or owners != 8 or (type(physical_batch_size) is not int) or (physical_batch_size != 8) or (type(slot_count) is not int) or (slot_count != 10) or (type(owner_bytes) is not int) or (owner_bytes != OWNER_BYTES):
        raise ValueError('tiered source profile differs from its pinned geometry')
    return SLOT_SIZES

def match_complete_sources(required_bytes, available_slots, *, slot_sizes=SLOT_SIZES):
    """Best-fit matching, proving every request fits a distinct free source range."""
    if type(slot_sizes) is not tuple or not 1 <= len(slot_sizes) <= 16 or any((type(n) is not int or n <= 0 for n in slot_sizes)) or (type(required_bytes) is not tuple) or (not required_bytes) or any((type(n) is not int or n <= 0 for n in required_bytes)) or (type(available_slots) is not tuple) or (len(set(available_slots)) != len(available_slots)) or any((type(i) is not int or not 0 <= i < len(slot_sizes) for i in available_slots)):
        raise ValueError('source capacity needs typed complete payloads and unique free slots')
    free = set(available_slots)
    selected = []
    for needed in required_bytes:
        suitable = [i for i in free if slot_sizes[i] >= needed]
        if not suitable:
            return None
        slot = min(suitable, key=lambda i: (slot_sizes[i], i))
        free.remove(slot)
        selected.append(slot)
    return tuple(selected)

def source_capacity_proof(*, rank, transport=None):
    """Export bounded per-rank capacity; addresses and full feature bytes stay local."""
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('tiered source rank differs from its TP8 mesh')
    initialized = transport is not None
    occupied = ()
    if initialized:
        if transport.fixed_source_slot_sizes != SLOT_SIZES or transport.max_slots != 10 or transport.max_bytes != OWNER_BYTES or transport._closed or (set(transport._source_buffers) != set(range(10))) or any((transport._source_buffers[i].numel() != n for i, n in enumerate(SLOT_SIZES))):
            raise RuntimeError('source transport differs from complete fixed tier registration')
        occupied = tuple(sorted(transport._slots))
        if any((type(i) is not int or not 0 <= i < 10 for i in occupied)):
            raise RuntimeError('live source slot lies outside its fixed registration')
        if any((type(transport._slots[i][1].get('nbytes')) is not int or not 0 < transport._slots[i][1]['nbytes'] <= SLOT_SIZES[i] for i in occupied)):
            raise RuntimeError('live source descriptor exceeds its fixed registration')
    available = [i for i in range(10) if i not in occupied]
    return dict(rank=rank, source_capture_owner_ranks=8, source_native_buffer_batches=10, source_native_slot_profile=PROFILE, native_source_slots=10, native_source_bytes=OWNER_BYTES, native_retained_sources=len(occupied), native_available_slots=len(available), publisher_initialized=initialized, native_source_slot_sizes=list(SLOT_SIZES), native_available_slot_ids=available)

def validate_capacity_proofs(proofs, *, owner_indices, required_bytes):
    """All TP identities and exact immutable bytes precede distinct-range matching."""
    if type(owner_indices) is not tuple or not 1 <= len(owner_indices) <= 8 or any((type(i) is not int or not 0 <= i < 8 for i in owner_indices)) or (len(set(owner_indices)) != len(owner_indices)) or (type(required_bytes) is not tuple) or (len(required_bytes) != len(owner_indices)) or any((type(n) is not int or not 0 < n <= max(SLOT_SIZES) for n in required_bytes)):
        raise ValueError('tiered capacity requires exact physical owners and complete payload sizes')
    if type(proofs) is not list or len(proofs) != 8:
        raise RuntimeError('tiered source admission requires all TP8 proofs')
    requests = dict(zip(owner_indices, required_bytes))
    admitted = True
    for rank, proof in enumerate(proofs):
        expected = dict(rank=rank, source_capture_owner_ranks=8, source_native_buffer_batches=10, source_native_slot_profile=PROFILE, native_source_slots=10, native_source_bytes=OWNER_BYTES, native_source_slot_sizes=list(SLOT_SIZES))
        if type(proof) is not dict or any((proof.get(k) != v or type(proof.get(k)) is not type(v) for k, v in expected.items())) or any((type(n) is not int for n in proof['native_source_slot_sizes'])) or (type(proof.get('native_retained_sources')) is not int) or (not 0 <= proof['native_retained_sources'] <= 10) or (type(proof.get('native_available_slots')) is not int) or (proof['native_available_slots'] != 10 - proof['native_retained_sources']) or (type(proof.get('publisher_initialized')) is not bool) or (not proof['publisher_initialized'] and proof['native_retained_sources'] != 0) or (type(proof.get('native_available_slot_ids')) is not list) or (len(proof['native_available_slot_ids']) != proof['native_available_slots']) or any((type(i) is not int or not 0 <= i < 10 for i in proof['native_available_slot_ids'])) or (len(set(proof['native_available_slot_ids'])) != proof['native_available_slots']):
            raise RuntimeError('tiered source capacity proof differs from pinned registrations')
        if rank in requests:
            admitted &= match_complete_sources((requests[rank],), tuple(proof['native_available_slot_ids'])) is not None
    return admitted

def config_slot_profile(config):
    profile = config.get('source_native_slot_profile', '')
    if type(profile) is not str or profile not in ('', PROFILE):
        raise ValueError('source fixed slot profile must name its immutable geometry')
    if bool(profile) != (config.get('source_native_buffer_batches', 1) == 10):
        raise ValueError('ten source slots require the explicit tiered fixed slot profile')
    return profile

def environment_slot_profile(environment):
    profile = environment.get('SPECO_CAPTURE_SOURCE_SLOT_PROFILE', '')
    if type(profile) is not str or profile not in ('', PROFILE):
        raise ValueError('source fixed slot environment differs from the named profile')
    if bool(profile) != (environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES', '') == '10'):
        raise ValueError('ten source slots require their explicit frozen profile flag')
    return profile
from dataclasses import asdict, dataclass
from verl_speco.producer.replay import digest

@dataclass(frozen=True)
class TrainingSequenceEligibilityPolicy:
    min_original_tokens: int = 64
    min_response_tokens: int = 8
    max_sequence_length: int = 65536

    def validate(self):
        values = (self.min_original_tokens, self.min_response_tokens, self.max_sequence_length)
        if any((type(v) is not int for v in values)) or not 2 <= self.min_original_tokens <= self.max_sequence_length <= 131072 or (not 2 <= self.min_response_tokens < self.max_sequence_length):
            raise ValueError('invalid training sequence eligibility limits')

def training_sequence_eligibility_counts(entries):
    return dict(rows=len(entries), original_tokens=sum((e['original_tokens'] for e in entries)), prompt_tokens=sum((e['prompt_tokens'] for e in entries)), response_tokens=sum((e['response_tokens'] for e in entries)), no_trainable_response_rows=sum((e['response_tokens'] < 2 for e in entries)), over_16k_rows=sum((e['original_tokens'] > 16384 for e in entries)), max_original_tokens=max((e['original_tokens'] for e in entries), default=0))

def qualify_training_sequence_index(index, *, policy=None):
    """Return bound training/quarantine indexes and an exact token ledger.

    This checks index identity and accounting, without rehashing sequence files.
    Their existing manifest pins and row SHA bindings stay intact. Over-context
    rows fail explicitly; this policy never truncates or discards long sequences.
    """
    policy = TrainingSequenceEligibilityPolicy() if policy is None else policy
    if not isinstance(policy, TrainingSequenceEligibilityPolicy):
        raise ValueError('training eligibility requires its explicit policy type')
    policy.validate()
    if not isinstance(index, dict) or index.get('schema_version') != 1 or (not isinstance(index.get('entries'), list)) or (not 1 <= len(index['entries']) <= 2000000):
        raise ValueError('training eligibility requires a nonempty frozen index')
    body = {k: v for k, v in index.items() if k != 'index_sha256'}
    if index.get('index_sha256') != digest(body) or index.get('dataset_sha256') != digest(index['snapshot']):
        raise ValueError('source index or dataset SHA differs')
    rows = index['entries']
    for e in rows:
        values = (e.get('original_tokens'), e.get('prompt_tokens'), e.get('response_tokens'))
        if any((type(v) is not int for v in values)) or values[0] < 2 or values[1] < 1 or (values[2] < 0) or (values[1] + values[2] != values[0]):
            raise ValueError('source row token accounting differs')
        if e['original_tokens'] > policy.max_sequence_length:
            raise ValueError('source context overflow requires an explicit larger-context plan')
    if index.get('counts') != training_sequence_eligibility_counts(rows):
        raise ValueError('source aggregate token accounting differs')
    selected, quarantine, reasons = ([], [], [])
    for number, e in enumerate(rows):
        failures = []
        if e['original_tokens'] < policy.min_original_tokens:
            failures.append('short_original_sequence')
        if e['response_tokens'] < policy.min_response_tokens:
            failures.append('incomplete_response_block')
        (quarantine if failures else selected).append(e)
        if failures:
            reasons.append(dict(source_row=number, id=e['id'], row_sha256=e['row_sha256'], reasons=failures))
    outputs = {}
    for role, entries in [('training', selected), ('quarantine', quarantine)]:
        value = deepcopy(index)
        value['entries'] = deepcopy(entries)
        value['counts'] = training_sequence_eligibility_counts(entries)
        value['snapshot']['additional_training_eligibility'] = dict(schema_version=1, role=role, source_index_sha256=index['index_sha256'], policy=asdict(policy), payloads_preserved=True, tokens_truncated=False)
        value['dataset_sha256'] = digest(value['snapshot'])
        value['rows_filtered'] = len(entries) != len(rows)
        value.pop('index_sha256')
        value['index_sha256'] = digest(value)
        outputs[role + '_index'] = value
    report = dict(schema_version=1, policy=asdict(policy), source_index_sha256=index['index_sha256'], training_index_sha256=outputs['training_index']['index_sha256'], quarantine_index_sha256=outputs['quarantine_index']['index_sha256'], source_counts=training_sequence_eligibility_counts(rows), training_counts=training_sequence_eligibility_counts(selected), quarantine_counts=training_sequence_eligibility_counts(quarantine), quarantined_rows=reasons, payloads_copied=False, source_payloads_rehashed=False, original_sequence_files_modified=False, tokens_truncated=False, epoch_plan_must_be_refrozen=True, existing_running_plan_modified=False)
    outputs['report'] = report
    return outputs
