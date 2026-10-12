# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from collections.abc import Sequence
import time
from verl_speco.transport.control import LeaseRecord, LeaseState
from verl_speco.transport.control import AcquireResult

def bind_acquired_batch_leases(acquired: AcquireResult, leases: Sequence[LeaseRecord], *, sample_keys: Sequence[str], generation: int, target_revision: str, consumer_id: str) -> tuple[LeaseRecord, ...]:
    """Accept arrival permutations while rejecting missing or foreign samples."""
    expected = tuple(sample_keys)
    records = {record.key: record for record in acquired.records}
    by_key = {lease.key: lease for lease in leases}
    if not expected or len(set(expected)) != len(expected) or acquired.status != 'ready' or (len(acquired.records) != len(expected)) or (len(records) != len(expected)) or (len(leases) != len(expected)) or (len(by_key) != len(expected)) or (set(records) != set(expected)) or (set(by_key) != set(expected)) or (len({lease.lease_id for lease in leases}) != len(expected)):
        raise RuntimeError('acquired records and leases differ from the canonical batch')
    now = time.time()
    for key in expected:
        record, lease = (records[key], by_key[key])
        record.validate()
        if record.metadata.generation != generation or record.metadata.target_revision != target_revision or lease.generation != generation or (lease.target_revision != target_revision) or (lease.acquire_id != acquired.acquire_id) or (lease.consumer_id != consumer_id) or (lease.request_id != record.metadata.request_id) or (lease.state is not LeaseState.LEASED) or lease.payload_released or (lease.expires_at <= now):
            raise RuntimeError('acquired lease identity or state differs from the canonical batch')
    return tuple((by_key[key] for key in expected))
from verl_speco.transport.streaming import make_v3_sample_key
from verl_speco.transport.control import LeaseState

class BoundedWaveMaterializer:

    def __init__(self, samples, *, control, consumer_id, max_wave_bytes):
        if type(max_wave_bytes) is not int or max_wave_bytes <= 0 or (not consumer_id):
            raise ValueError('wave byte budget and consumer identity are required')
        self.samples = tuple(samples)
        if not self.samples or len({key for key, _ in self.samples}) != len(self.samples):
            raise ValueError('wave samples must be nonempty and unique')
        first = self.samples[0][1]
        for index, (key, m) in enumerate(self.samples):
            m.validate()
            if key != make_v3_sample_key(m) or m.sequence_no != first.sequence_no + index or m.payload_nbytes > max_wave_bytes or any((getattr(m, k) != getattr(first, k) for k in ('run_id', 'generation', 'target_revision', 'target_model_fingerprint', 'tokenizer_fingerprint', 'target_layer_ids', 'dataset_fingerprint'))):
                raise ValueError('wave samples exceed budget or differ from stream identity')
        self.control = control
        self.consumer_id = consumer_id
        self.max_wave_bytes = max_wave_bytes
        self._next = 0
        self._leases = []
        self._records = []
        self.state = 'OPEN'

    def stage(self, acquired, leases, *, fetch_verified, materialize, clear_payload, discard, quiesce=lambda: None, clear_payload_batch=None, on_acquired=None, before_payload_release=None, release_chunk_size=0, before_chunk_release=None):
        """Copy to the node owner's RAM before acknowledging source release."""
        if self.state != 'OPEN':
            raise RuntimeError('wave materializer is not open')
        if any((value is not None and (not callable(value)) for value in (on_acquired, before_payload_release))):
            raise ValueError('two-node wave hooks must be callable metadata callbacks')
        if clear_payload_batch is not None and (not callable(clear_payload_batch) or not callable(getattr(self.control, 'release_materialized_batch', None))):
            raise ValueError('batch clear requires explicit owned materialized release support')
        if type(release_chunk_size) is not int or not 0 <= release_chunk_size <= 48 or (release_chunk_size and (clear_payload_batch is None or not callable(before_chunk_release) or on_acquired is not None)) or (not release_chunk_size and before_chunk_release is not None):
            raise ValueError('prefix release requires an explicit verified RAM ownership fence')
        owned_ids = []
        try:
            expected_keys = {key for key, _ in self.samples}
            for lease in leases:
                actual = self.control.get_lease(lease.lease_id)
                if actual.key in expected_keys and actual.consumer_id == self.consumer_id and (actual.generation == self.samples[0][1].generation) and (actual.state in (LeaseState.LEASED, LeaseState.MATERIALIZED)):
                    owned_ids.append(actual.lease_id)
            count = len(acquired.records)
            expected = self.samples[self._next:self._next + count]
            if not count or len(expected) != count or sum((m.payload_nbytes for _, m in expected)) > self.max_wave_bytes:
                raise RuntimeError('wave exceeds remaining prefix or retained byte budget')
            first = expected[0][1]
            bound = bind_acquired_batch_leases(acquired, leases, sample_keys=[key for key, _ in expected], generation=first.generation, target_revision=first.target_revision, consumer_id=self.consumer_id)
            by_key = {record.key: record for record in acquired.records}
            records = tuple((by_key[key] for key, _ in expected))
            if on_acquired is not None:
                on_acquired(tuple(bound), tuple(expected), records)
            released = 0
            for offset, (key, metadata) in enumerate(expected):
                record = by_key[key]
                if record.metadata != metadata:
                    raise RuntimeError('wave ready metadata differs from planned capture')
                fields, digest = fetch_verified(key)
                if digest != record.payload_checksum:
                    raise RuntimeError('wave capture checksum differs from ready record')
                if any((t.device.type != 'cpu' for t in fields.values())):
                    raise RuntimeError('wave requires verified CPU features')
                materialize(fields, metadata)
                del fields
                if release_chunk_size and ((offset + 1) % release_chunk_size == 0 or offset + 1 == len(expected)):
                    chunk = tuple(bound[released:offset + 1])
                    before_chunk_release(chunk, tuple(expected[released:offset + 1]))
                    ids = tuple((lease.lease_id for lease in chunk))
                    keys = tuple((lease.key for lease in chunk))
                    self.control.mark_materialized(ids)
                    clear_payload_batch(keys)
                    self.control.release_materialized_batch(keys=keys, lease_ids=ids, consumer_id=self.consumer_id)
                    released = offset + 1
            if before_payload_release is not None:
                before_payload_release(tuple(bound), tuple(expected), records)
            if release_chunk_size:
                assert released == len(expected)
            else:
                self.control.mark_materialized(tuple((lease.lease_id for lease in bound)))
            if release_chunk_size:
                quiesce()
            elif clear_payload_batch is not None:
                quiesce()
                clear_payload_batch(tuple((lease.key for lease in bound)))
                self.control.release_materialized_batch(keys=tuple((lease.key for lease in bound)), lease_ids=tuple((lease.lease_id for lease in bound)), consumer_id=self.consumer_id)
            else:
                for lease in bound:
                    clear_payload(lease.key)
                    self.control.release(lease.key, lease_ids=(lease.lease_id,))
            self._leases.extend((lease.lease_id for lease in bound))
            self._records.extend((by_key[key] for key, _ in expected))
            self._next += count
        except BaseException:
            self.state = 'FAILED'
            try:
                quiesce()
                cancel_ids = tuple(dict.fromkeys([*self._leases, *owned_ids]))
                if cancel_ids:
                    self.control.cancel_leases(cancel_ids, reason='prefill wave failed before optimizer')
            finally:
                discard()
            raise

    def abort(self, leases=(), *, discard):
        """Cancel only this staging owner's uncommitted leases and drop RAM."""
        if self.state == 'SEALED':
            raise RuntimeError('sealed waves belong to the optimizer transaction')
        self.state = 'FAILED'
        ids = list(self._leases)
        expected_keys = {key for key, _ in self.samples}
        try:
            for lease in leases:
                actual = self.control.get_lease(lease.lease_id)
                if actual.key in expected_keys and actual.consumer_id == self.consumer_id and (actual.generation == self.samples[0][1].generation) and (actual.state in (LeaseState.LEASED, LeaseState.MATERIALIZED)):
                    ids.append(actual.lease_id)
            if ids:
                self.control.cancel_leases(tuple(dict.fromkeys(ids)), reason='prefill wave aborted before optimizer')
        finally:
            discard()

    def seal(self):
        """Return leases for the existing optimizer transaction, with no credit."""
        if self.state != 'OPEN' or self._next != len(self.samples):
            raise RuntimeError('cannot seal incomplete or failed prefill waves')
        for lease_id in self._leases:
            lease = self.control.get_lease(lease_id)
            if lease.state is not LeaseState.MATERIALIZED or not lease.payload_released:
                raise RuntimeError('wave lease was not materialized and payload-released')
        self.state = 'SEALED'
        return (tuple(self._leases), tuple(self._records))

    def snapshot(self):
        return dict(state=self.state, staged_samples=self._next, expected_samples=len(self.samples), lease_ids=list(self._leases), committed_original_tokens=0, optimizer_started=False)
from verl_speco.transport.streaming import MAX_NATIVE_BATCH_SAMPLES
import hashlib
import json
from pathlib import Path
import re
from verl_speco.runtime.artifacts import read_published_json
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key

def packed_control_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def load_packed_sample_control(*, inline='', path='', sha256='', run_id, generation):
    if inline and path:
        raise ValueError('packed samples must use either inline or published metadata')
    if not path:
        if sha256:
            raise ValueError('packed metadata checksum requires a path')
        return json.loads(inline) if inline else None
    root = Path(__file__).resolve().parents[2] / 'outputs'
    source = Path(path).resolve()
    if not source.is_relative_to(root) or re.fullmatch('[0-9a-f]{64}', sha256 or '') is None:
        raise ValueError('packed metadata path or checksum is invalid')
    payload = read_published_json(source, max_bytes=4 * 1024 ** 2)
    if not isinstance(payload, dict) or set(payload) != {'schema_version', 'run_id', 'generation', 'samples'} or type(payload['schema_version']) is not int or (payload['schema_version'] != 1) or (payload['run_id'] != run_id) or (type(payload['generation']) is not int) or (payload['generation'] != generation) or (packed_control_sha256(payload) != sha256) or (not isinstance(payload['samples'], list)) or (not 1 <= len(payload['samples']) <= MAX_NATIVE_BATCH_SAMPLES):
        raise ValueError('published packed metadata checksum or batch identity differs')
    first = None
    keys = []
    for sample in payload['samples']:
        if not isinstance(sample, dict) or set(sample) != {'key', 'metadata'}:
            raise ValueError('packed control sample must contain only key and metadata')
        m = SampleMetadataV3.from_dict(sample['metadata'])
        if json.dumps(m.to_dict(), sort_keys=True) != json.dumps(sample['metadata'], sort_keys=True) or m.run_id != run_id or m.generation != generation or (make_v3_sample_key(m) != sample['key']) or (first is not None and m.sequence_no != first + len(keys)):
            raise ValueError('packed control samples differ from canonical metadata prefix')
        if first is None:
            first = m.sequence_no
        keys.append(sample['key'])
    if len(set(keys)) != len(keys):
        raise ValueError('packed control contains duplicate keys')
    return payload['samples']
from dataclasses import dataclass
import torch
from safetensors import safe_open

@dataclass(frozen=True)
class TargetHeadLocation:
    shard: Path
    tensor_name: str
    shape: tuple[int, int]
    dtype: str
    index_sha256: str

def inspect_target_lm_head(model_dir: str | Path, *, vocab_size: int=151552, hidden_size: int=2560) -> TargetHeadLocation:
    root = Path(model_dir).resolve()
    index_path = root / 'model.safetensors.index.json'
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    key = 'lm_head.weight'
    shard_name = index.get('weight_map', {}).get(key)
    if not isinstance(shard_name, str):
        raise ValueError('92B checkpoint index has no lm_head.weight')
    shard = (root / shard_name).resolve()
    if shard.parent != root or not shard.is_file():
        raise ValueError('92B LM-head shard path is invalid or missing')
    with safe_open(shard, framework='pt', device='cpu') as handle:
        shape = tuple(handle.get_slice(key).get_shape())
        dtype = handle.get_slice(key).get_dtype()
    if shape != (vocab_size, hidden_size) or dtype != 'BF16':
        raise ValueError(f'92B LM head has shape/dtype {shape}/{dtype}; expected {(vocab_size, hidden_size)}/BF16')
    return TargetHeadLocation(shard, key, shape, dtype, hashlib.sha256(index_bytes).hexdigest())

def load_target_lm_head(model_dir: str | Path, *, device: torch.device | str='cpu', vocab_size: int=151552, hidden_size: int=2560) -> tuple[torch.Tensor, TargetHeadLocation]:
    location = inspect_target_lm_head(model_dir, vocab_size=vocab_size, hidden_size=hidden_size)
    with safe_open(location.shard, framework='pt', device='cpu') as handle:
        weight = handle.get_tensor(location.tensor_name)
    if weight.shape != location.shape or weight.dtype != torch.bfloat16:
        raise ValueError('LM-head tensor differs from its indexed metadata')
    weight = weight.to(device=device, non_blocking=False)
    weight.requires_grad_(False)
    return (weight, location)
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Condition, Event, Semaphore, Lock

class ParallelVerificationPools:
    """Reuse idle single-reader lanes; one byte-bounded wave owns them at a time."""

    def __init__(self, *, reader_count, verification_workers, initialize=None):
        if type(reader_count) is not int or reader_count not in (2, 4, 8) or type(verification_workers) is not int or (verification_workers not in (2, 4, 8)) or (reader_count > verification_workers):
            raise ValueError('invalid reusable native reader pool geometry')
        self.reader_count = reader_count
        self.verification_workers = verification_workers
        self._lock = Lock()
        self._owner = None
        self._closed = False
        self.hash_pool = ThreadPoolExecutor(max_workers=verification_workers, thread_name_prefix='speco-parallel-verify')
        self.coordinator = ThreadPoolExecutor(max_workers=1, thread_name_prefix='speco-read-admission')
        self.lanes = []
        try:
            for lane in range(reader_count):
                self.lanes.append(ThreadPoolExecutor(max_workers=1, initializer=initialize, thread_name_prefix=f'speco-native-reader-{lane}'))
        except BaseException:
            self.close()
            raise

    def acquire(self, owner):
        with self._lock:
            if self._closed or self._owner is not None:
                raise RuntimeError('reusable native reader pools are closed or already owned')
            self._owner = owner

    def release(self, owner):
        with self._lock:
            if self._owner is not owner:
                raise RuntimeError('reusable native reader pool owner differs')
            self._owner = None

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._owner is not None:
                raise RuntimeError('cannot close reusable native pools before the active wave joins')
            self._closed = True
        self.coordinator.shutdown(wait=True, cancel_futures=True)
        for lane in self.lanes:
            lane.shutdown(wait=True, cancel_futures=True)
        self.hash_pool.shutdown(wait=True, cancel_futures=True)

class ParallelVerifiedPayloads:

    def __init__(self, samples, *, copies, verify, max_bytes, initialize=None, verification_workers=4, pools=None):
        self.samples = tuple(samples)
        self._copies = tuple(copies)
        if not self.samples or len({k for k, _ in self.samples}) != len(self.samples) or len(self._copies) not in (2, 4, 8) or any((not callable(copy) for copy in self._copies)) or (not callable(verify)) or (type(verification_workers) is not int) or (verification_workers not in (2, 4, 8)) or (len(self._copies) > verification_workers) or (type(max_bytes) is not int) or (not 0 < max_bytes <= (16 if verification_workers == 8 and len(self._copies) == 8 else 8) * 1024 ** 3) or ((verification_workers + 1) * max((m.payload_nbytes for _, m in self.samples)) > max_bytes):
            raise ValueError('parallel readers exceed the verified payload byte or worker budget')
        self._verify = verify
        self._next = 0
        self._closed = False
        self._stop = Event()
        self._condition = Condition()
        self._slots = Semaphore(verification_workers)
        self._pending = {}
        self._error = None
        if pools is not None and (not isinstance(pools, ParallelVerificationPools) or pools.reader_count != len(self._copies) or pools.verification_workers != verification_workers):
            raise ValueError('reusable native pools differ from the admitted wave geometry')
        self._owns_pools = pools is None
        self._pools = pools if pools is not None else ParallelVerificationPools(reader_count=len(self._copies), verification_workers=verification_workers, initialize=initialize)
        self._pools.acquire(self)
        self._lanes = self._pools.lanes
        self._hash_pool = self._pools.hash_pool
        self._coordinator = self._pools.coordinator
        self._producer = None
        try:
            self._producer = self._coordinator.submit(self._produce)
        except BaseException:
            self.close()
            raise

    def _copy_and_schedule_hash(self, lane, key):
        packet = self._copies[lane](key)
        if not isinstance(packet, tuple) or not packet or packet[0] != key:
            raise ValueError('independent native reader returned a foreign packet')
        return self._hash_pool.submit(self._verify, packet)

    def _produce(self):
        try:
            for index, (key, _) in enumerate(self.samples):
                while not self._stop.is_set():
                    if self._slots.acquire(timeout=0.1):
                        break
                else:
                    return
                if self._stop.is_set():
                    self._slots.release()
                    return
                lane = index % len(self._lanes)
                future = self._lanes[lane].submit(self._copy_and_schedule_hash, lane, key)
                with self._condition:
                    self._pending[key] = future
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def get(self, key):
        if self._closed or self._next >= len(self.samples) or key != self.samples[self._next][0]:
            raise RuntimeError('parallel native payload key is out of order or closed')
        with self._condition:
            while key not in self._pending:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise RuntimeError('parallel native payload queue closed')
                if self._producer.done():
                    self._producer.result()
                    raise RuntimeError('parallel native payload queue ended before the planned key')
                self._condition.wait(timeout=0.1)
            future = self._pending[key]
        fields, digest = future.result().result()
        metadata = self.samples[self._next][1]
        if not fields or any((t.device.type != 'cpu' for t in fields.values())) or sum((t.numel() * t.element_size() for t in fields.values())) != metadata.payload_nbytes:
            raise ValueError('parallel native payload queue requires independent verified CPU fields')
        with self._condition:
            del self._pending[key]
        self._next += 1
        self._slots.release()
        return (fields, digest)

    def require_consumed_prefix(self, keys):
        keys = tuple(keys)
        consumed = [k for k, _ in self.samples[:self._next]]
        if self._closed or not keys or len(set(keys)) != len(keys) or any((k not in consumed for k in keys)):
            raise RuntimeError('native prefix release includes an unverified or unconsumed key')
        with self._condition:
            if any((k in self._pending for k in keys)):
                raise RuntimeError('native prefix release still has a pending reader')
        return keys

    def close(self):
        if self._closed:
            return
        with self._condition:
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
        if self._producer is not None:
            self._producer.cancel()
            wait((self._producer,))
        pending = tuple(self._pending.values())
        for future in pending:
            future.cancel()
        wait(pending)
        hashes = [future.result() for future in pending if not future.cancelled() and future.exception() is None]
        for future in hashes:
            future.cancel()
        wait(hashes)
        with self._condition:
            self._pending.clear()
        self._pools.release(self)
        if self._owns_pools:
            self._pools.close()
from verl_speco.producer.feature_payload import payload_checksum

@dataclass(frozen=True)
class AdmissionBytes:
    payload_nbytes: int

class ParallelSelectivePayloads:

    def __init__(self, records, *, routes, owner_rank, readers, partition, replay_light, hidden_size, max_bytes, initialize=None):
        records = tuple(records)
        readers = tuple(readers)
        if type(owner_rank) is not int or owner_rank not in (0, 16) or len(records) != len(routes) or (len(readers) not in (2, 4)):
            raise ValueError('invalid selective parallel ownership')
        transports = [getattr(reader, 'transport', None) for reader in readers]
        if any((t is None for t in transports)) or len({id(t) for t in transports}) != len(readers) or len({t.endpoint for t in transports}) != len(readers) or (len({str(t.device) for t in transports}) != 1) or any((t.cpu_bulk_receive is not True for t in transports)) or (type(max_bytes) is not int) or (not 0 < max_bytes <= 8 * 1024 ** 3):
            raise ValueError('selective lanes require independent bulk transports and byte budget')
        self.records = {r.key: r for r in records}
        self.routes = {r['sample_key']: r for r in routes}
        self.queue = None
        if len(self.records) != len(records) or len(self.routes) != len(routes):
            raise ValueError('duplicate selective parallel sample')
        sizes = []
        for record, route in zip(records, routes):
            record.validate()
            if record.key != route['sample_key'] or record.payload_checksum != route['payload_sha256']:
                raise ValueError('selective parallel route differs from source')
            full = owner_rank in route['required_hidden_owners']
            sizes.append((record.key, AdmissionBytes(record.metadata.payload_nbytes if full else 20 * record.metadata.token_count)))

        def lane(reader):

            def copy(key):
                record = self.records[key]
                route = self.routes[key]
                if owner_rank in route['required_hidden_owners']:
                    packet = reader._copy_cpu_for_verification(key, partition)
                    if packet[0] != key or packet[2] != record.metadata or packet[3] != record.payload_checksum:
                        raise ValueError('parallel full capture binding differs')
                    return (key, 'full', packet, reader)
                fields, digest = reader.get_verified_light(key, partition, expected_metadata=record.metadata, expected_payload_checksum=record.payload_checksum, expected_fields=replay_light[key], expected_light_checksum=route['light_sha256'], hidden_size=hidden_size)
                return (key, 'light', fields, digest)
            return copy

        def verify(packet):
            key, mode, value, verifier = packet
            record = self.records[key]
            route = self.routes[key]
            if mode == 'full':
                fields, digest = verifier._verify_cpu_packet(value, pipelined=True)
                expected = record.payload_checksum
                if set(fields) != {'input_ids', 'position_ids', 'loss_mask', 'hidden_states'}:
                    raise ValueError('parallel full fields differ')
            else:
                fields, digest = (value, verifier)
                expected = route['light_sha256']
                if set(fields) != {'input_ids', 'position_ids', 'loss_mask'} or payload_checksum(fields) != digest:
                    raise ValueError('parallel light fields or checksum differ')
            if digest != expected:
                raise ValueError('parallel selected checksum differs')
            return (fields, digest)
        self.queue = ParallelVerifiedPayloads(sizes, copies=[lane(reader) for reader in readers], verify=verify, max_bytes=max_bytes - (len(readers) + 1) * 4096 - len(readers) * 40 * max((r.metadata.token_count for r in records)), initialize=initialize, verification_workers=len(readers))

    def get(self, key):
        return self.queue.get(key)

    def close(self):
        if self.queue is not None:
            self.queue.close()

def derive_route_template(planner, *, replay_light, hidden_size):
    from verl_speco.trainer.owned_batches import materialize_light_columns
    from verl_speco.transport.streaming import make_v3_sample_key
    if planner.rank != 0 or planner.world_size != 32 or (not planner.metadatas) or (planner._global_parts < 1):
        raise ValueError('selective routing requires the complete global DP32 plan')
    expected_keys = [make_v3_sample_key(m) for m in planner.metadatas]
    if not isinstance(replay_light, dict) or set(replay_light) != set(expected_keys):
        raise ValueError('immutable replay differs from complete selected batch')
    ownership = {m.sequence_no: set() for m in planner.metadatas}
    for row_index, local in enumerate(planner._rows):
        offset = planner._part_offsets[row_index]
        for ordinal, (_, _, sample_index) in enumerate(local._parts_plan):
            metadata = local.metadatas[sample_index]
            ownership[metadata.sequence_no].add((offset + ordinal) % 32 // 16 * 16)
    dummy_nodes = {owner for owner in (0, 16) if any((rank >= planner._global_parts for rank in range(owner, owner + 16)))}
    routes = []
    for key, m in zip(expected_keys, planner.metadatas):
        fields = replay_light[key]
        materialize_light_columns(fields, m, hidden_size=hidden_size)
        owners = ownership[m.sequence_no] | dummy_nodes
        if not owners:
            owners = {0}
        routes.append(dict(sample_key=key, required_hidden_owners=sorted(owners), light_sha256=payload_checksum(fields)))
    return routes

def derive_routes(planner, records, *, replay_light, hidden_size):
    template = derive_route_template(planner, replay_light=replay_light, hidden_size=hidden_size)
    if len(records) != len(template):
        raise ValueError('ready records differ from complete selected batch')
    routes = []
    for m, row, record in zip(planner.metadatas, template, records):
        record.validate()
        if record.key != row['sample_key'] or record.metadata != m:
            raise ValueError('route differs from exact planned capture')
        routes.append(dict(row, payload_sha256=record.payload_checksum))
    return routes

def stage_selected_wave(*, owner_rank, records, routes, reader, partition, copies, replay_light, hidden_size, verification, prefetch=None):
    if type(owner_rank) is not int or owner_rank not in (0, 16) or len(records) != len(routes):
        raise ValueError('selective node wave binding differs')
    if routes != verification.routes:
        raise ValueError('selective routes differ from registered verification')
    entries = []
    try:
        for record, route in zip(records, routes):
            record.validate()
            if route['sample_key'] != record.key or route['payload_sha256'] != record.payload_checksum:
                raise ValueError('selective source transaction differs')
            full = owner_rank in route['required_hidden_owners']
            needed = copies.requires_hidden(record.metadata)
            if needed and (not full):
                raise ValueError('routing omitted locally required hidden')
            if prefetch is not None:
                fields, digest = prefetch.get(record.key)
                expected = record.payload_checksum if full else route['light_sha256']
                names = {'input_ids', 'position_ids', 'loss_mask'} | ({'hidden_states'} if full else set())
                if digest != expected or set(fields) != names:
                    raise ValueError('selected prefetch proof differs')
                mode = 'full_payload' if full else 'immutable_replay_light'
            elif full:
                fields, digest = reader.get_verified_cpu_bound(record.key, partition, expected_metadata=record.metadata, expected_payload_checksum=record.payload_checksum)
                if digest != record.payload_checksum:
                    raise ValueError('selective full payload SHA differs')
                mode = 'full_payload'
            else:
                fields, digest = reader.get_verified_light(record.key, partition, expected_metadata=record.metadata, expected_payload_checksum=record.payload_checksum, expected_fields=replay_light[record.key], expected_light_checksum=route['light_sha256'], hidden_size=hidden_size)
                if digest != route['light_sha256']:
                    raise ValueError('selective light SHA differs')
                mode = 'immutable_replay_light'
            copies.add_selected(fields, record.metadata)
            entries.append(dict(sample_key=record.key, verification=mode, verified_sha256=digest))
        if prefetch is not None:
            prefetch.close()
        copies.close()
        return dict(identity=dict(verification.identity), routing_sha256=verification.routing_sha256, owner_rank=owner_rank, node_incarnation=verification.incarnations[owner_rank], entries=entries, staging_joined=True, optimizer_started=False, committed_original_tokens=0)
    except BaseException:
        if prefetch is not None:
            prefetch.close()
        copies.close()
        verification.abort()
        raise

def stage_selected_session_wave(*, session, value, records, reader, partition, copies, replay_light, hidden_size, verification, prefetch=None):
    try:
        session._validate_selected_wave(value)
        if [r.key for r in records] != value['sample_keys'] or [r.metadata.to_dict() for r in records] != value['metadata'] or [r.payload_checksum for r in records] != value['payload_checksums']:
            raise ValueError('selected staging differs from retained source records')
        if verification.identity != {k: value[k] for k in ('run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index')} or verification.routes != value['selective_routes'] or verification.incarnations[session.owner_rank] != session.incarnation:
            raise ValueError('selected staging proof differs from actual session')
        proof = stage_selected_wave(owner_rank=session.owner_rank, records=records, routes=value['selective_routes'], reader=reader, partition=partition, copies=copies, replay_light=replay_light, hidden_size=hidden_size, verification=verification, prefetch=prefetch)
        session.acknowledge(value, proof)
        return proof
    except BaseException:
        import traceback
        for close in ([prefetch.close] if prefetch is not None else []) + [copies.close, verification.abort, session.abort]:
            try:
                close()
            except BaseException:
                traceback.print_exc()
        raise

class SelectedWaveMaterializer(BoundedWaveMaterializer):

    def stage_selected(self, acquired, leases, *, session, wave_index, materialize_wave, clear_payload_batch, discard, quiesce):
        if self.state != 'OPEN':
            raise RuntimeError('selected materializer is not open')
        if session.owner_rank != 0 or session.selective_plan is None:
            raise ValueError('selected release transaction requires primary selected session')
        try:
            count = len(acquired.records)
            expected = self.samples[self._next:self._next + count]
            if not count or len(expected) != count or sum((m.payload_nbytes for _, m in expected)) > self.max_wave_bytes:
                raise RuntimeError('selected wave exceeds remaining prefix or retained byte budget')
            first = expected[0][1]
            bound = bind_acquired_batch_leases(acquired, leases, sample_keys=[k for k, _ in expected], generation=first.generation, target_revision=first.target_revision, consumer_id=self.consumer_id)
            by_key = {r.key: r for r in acquired.records}
            records = tuple((by_key[k] for k, _ in expected))
            if any((r.metadata != m for r, (_, m) in zip(records, expected))):
                raise ValueError('selected source metadata differs from exact prefix')
            for lease in bound:
                if self.control.get_lease(lease.lease_id) != lease:
                    raise ValueError('selected source lease changed before admission')
            value = session.begin(wave_index=wave_index, leases=bound, expected=expected, records=records)
            proof = materialize_wave(value, records)
            quiesce()
            session.before_release(value, proof)
            ids = tuple((l.lease_id for l in bound))
            keys = tuple((l.key for l in bound))
            self.control.mark_materialized(ids)
            clear_payload_batch(keys)
            self.control.release_materialized_batch(keys=keys, lease_ids=ids, consumer_id=self.consumer_id)
            session.confirm_release()
            self._leases.extend(ids)
            self._records.extend(records)
            self._next += count
        except BaseException:
            self.state = 'FAILED'
            import traceback
            for cleanup in (quiesce, discard, session.abort):
                try:
                    cleanup()
                except BaseException:
                    traceback.print_exc()
            raise

class RayNodeControl:
    _METHODS = frozenset({'acquire', 'get_lease', 'mark_materialized', 'release', 'release_materialized_batch', 'cancel_leases', 'register_node_wave_owner', 'get_node_wave_state', 'get_node_wave_owner_incarnations', 'begin_node_wave', 'acknowledge_node_wave', 'node_wave_release_permit', 'confirm_node_wave_release', 'abort_node_wave', 'reap_expired', 'status', 'snapshot_quiescent', 'publish_materialized_wave_ack'})

    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        if name not in self._METHODS:
            raise AttributeError(name)

        def call(*args, **kwargs):
            import ray
            return ray.get(getattr(self.actor, name).remote(*args, **kwargs), timeout=60)
        return call
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event, Semaphore

class OneAheadVerifiedPayloads:

    def __init__(self, samples, *, fetch, max_bytes, initialize=None):
        self.samples = tuple(samples)
        if not self.samples or len({k for k, _ in self.samples}) != len(self.samples) or type(max_bytes) is not int or (not 0 < max_bytes <= 8 * 1024 ** 3) or (2 * max((m.payload_nbytes for _, m in self.samples)) > max_bytes):
            raise ValueError('native payload prefetch exceeds two-payload CPU budget')
        self._fetch = fetch
        self._next = 0
        self._future = None
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=1, initializer=initialize, thread_name_prefix='speco-native-prefetch')
        try:
            self._future = self._pool.submit(fetch, self.samples[0][0])
        except BaseException:
            self._pool.shutdown(wait=True, cancel_futures=True)
            raise

    def get(self, key):
        if self._closed or self._next >= len(self.samples) or key != self.samples[self._next][0]:
            raise RuntimeError('native payload prefetch key is out of order or closed')
        result = self._future.result()
        self._future = None
        fields, digest = result
        metadata = self.samples[self._next][1]
        if any((t.device.type != 'cpu' for t in fields.values())) or sum((t.numel() * t.element_size() for t in fields.values())) != metadata.payload_nbytes:
            raise ValueError('native payload prefetch requires independent verified CPU payloads')
        self._next += 1
        if self._next < len(self.samples):
            self._future = self._pool.submit(self._fetch, self.samples[self._next][0])
        return (fields, digest)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._future = None

class PipelinedVerifiedPayloads:
    """One native copy worker and an explicit two- or four-verifier budget.

    Only successful verified CPU tensors cross get(). Pending slots plus the
    caller's current payload fit the explicit three- or five-payload budget. The source
    must retain its registration until close() has joined both worker pools.
    """

    def __init__(self, samples, *, copy, verify, max_bytes, initialize=None, verification_workers=2):
        self.samples = tuple(samples)
        if not self.samples or len({k for k, _ in self.samples}) != len(self.samples) or type(max_bytes) is not int or (not 0 < max_bytes <= 8 * 1024 ** 3) or (type(verification_workers) is not int) or (verification_workers not in (2, 4)) or ((verification_workers + 1) * max((m.payload_nbytes for _, m in self.samples)) > max_bytes) or (not callable(copy)) or (not callable(verify)):
            raise ValueError('native verification pipeline exceeds explicit three-payload (two workers) or five-payload (four workers) CPU budget')
        self._copy = copy
        self._verify = verify
        self._next = 0
        self._closed = False
        self._condition = Condition()
        self._stop = Event()
        self._slots = Semaphore(verification_workers)
        self._pending = {}
        self._error = None
        self._hash_pool = ThreadPoolExecutor(max_workers=verification_workers, thread_name_prefix='speco-cpu-verify')
        self._copy_pool = ThreadPoolExecutor(max_workers=1, initializer=initialize, thread_name_prefix='speco-native-copy')
        try:
            self._producer = self._copy_pool.submit(self._produce)
        except BaseException:
            self._stop.set()
            self._copy_pool.shutdown(wait=True, cancel_futures=True)
            self._hash_pool.shutdown(wait=True, cancel_futures=True)
            raise

    def _produce(self):
        try:
            for key, _ in self.samples:
                while not self._stop.is_set():
                    if self._slots.acquire(timeout=0.1):
                        break
                else:
                    return
                if self._stop.is_set():
                    self._slots.release()
                    return
                packet = self._copy(key)
                future = self._hash_pool.submit(self._verify, packet)
                del packet
                with self._condition:
                    self._pending[key] = future
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def get(self, key):
        if self._closed or self._next >= len(self.samples) or key != self.samples[self._next][0]:
            raise RuntimeError('native verification pipeline key is out of order or closed')
        with self._condition:
            while key not in self._pending:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise RuntimeError('native verification pipeline closed')
                if self._producer.done():
                    self._producer.result()
                    raise RuntimeError('native verification pipeline ended before planned key')
                self._condition.wait(timeout=0.1)
            future = self._pending[key]
        fields, digest = future.result()
        metadata = self.samples[self._next][1]
        if not fields or any((t.device.type != 'cpu' for t in fields.values())) or sum((t.numel() * t.element_size() for t in fields.values())) != metadata.payload_nbytes:
            raise ValueError('native verification pipeline requires verified CPU payloads')
        with self._condition:
            del self._pending[key]
        self._next += 1
        self._slots.release()
        return (fields, digest)

    def close(self):
        if self._closed:
            return
        with self._condition:
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
        self._copy_pool.shutdown(wait=True, cancel_futures=True)
        self._hash_pool.shutdown(wait=True, cancel_futures=True)
        with self._condition:
            self._pending.clear()
