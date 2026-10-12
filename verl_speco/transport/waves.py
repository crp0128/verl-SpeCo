# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from hashlib import sha256
import json
from verl_speco.transport.control import LeaseState

def committed_prefill_future_limit(capacity_bytes, prefill_geometry=None):
    if prefill_geometry is None:
        return {72 * 1024 ** 3: 144, 96 * 1024 ** 3: 192}.get(capacity_bytes, 96)
    common = dict(producer_count=6, target_tp=8, owner_ranks=8, max_sequence_length=49152, graph_mode='FULL_DECODE_ONLY')
    required = set(common) | {'physical_batch_size', 'buffer_batches'}
    if isinstance(prefill_geometry, dict) and prefill_geometry.get('buffer_batches') == 10:
        from verl_speco.producer.capture import PROFILE
        if prefill_geometry.get('source_slot_profile') != PROFILE:
            raise ValueError('tiered committed snapshot requires its immutable source slot profile')
        required |= {'source_slot_profile'}
    if not isinstance(prefill_geometry, dict) or set(prefill_geometry) not in (required, required | {'samples_per_step'}) or any((type(prefill_geometry[k]) is not type(v) or prefill_geometry[k] != v for k, v in common.items())) or (type(capacity_bytes) is not int):
        raise ValueError('committed snapshot requires an exact frozen six-TP8 geometry')
    physical = prefill_geometry['physical_batch_size']
    banks = prefill_geometry['buffer_batches']
    if type(physical) is not int or type(banks) is not int:
        raise ValueError('committed snapshot geometry requires integer source sizes')
    profiles = {(16, 2): (48 * 1024 ** 3, 192), (8, 3): (72 * 1024 ** 3, 144), (8, 4): (96 * 1024 ** 3, 192), (8, 10): (96 * 1024 ** 3, 480)}
    admitted = profiles.get((physical, banks))
    if admitted is None or capacity_bytes != admitted[0]:
        raise ValueError('committed snapshot geometry differs from its registered source banks')
    samples = prefill_geometry.get('samples_per_step', 480)
    if banks == 10 and samples != 960:
        raise ValueError('tiered snapshot requires packed960 optimizer batch')
    if type(samples) is not int or samples not in (480, 960):
        raise ValueError('snapshot has unsupported optimizer batch')
    return admitted[1]

def snapshot_geometry_from_environment(environment):
    raw = environment.get('SPECO_COMMITTED_PREFILL_GEOMETRY_JSON')
    if raw is None:
        return None
    geometry = json.loads(raw)
    committed_prefill_future_limit(int(environment['SPECO_TQ_CAPACITY_BYTES']), geometry)
    return geometry

def committed_prefill_clone(owner, *, optimizer_step_id, committed_cursor, committed_lease_ids, committed_sample_keys, prefill_geometry=None):
    from verl_speco.transport.control import RayReadyControl
    if type(committed_cursor) is not int or committed_cursor < 1 or (not isinstance(optimizer_step_id, str)) or (not optimizer_step_id):
        raise ValueError('committed prefill snapshot needs an actual positive stream boundary')
    committed_prefill_future_limit(owner.capacity_bytes, prefill_geometry)
    transaction_limit = 960 if prefill_geometry is not None and prefill_geometry.get('samples_per_step') == 960 else 512
    ids = tuple(committed_lease_ids)
    keys = tuple(committed_sample_keys)
    if not 1 <= len(ids) <= transaction_limit or len(set(ids)) != len(ids) or len(keys) != len(ids) or (len(set(keys)) != len(keys)) or any((not isinstance(v, str) or not v for v in (*ids, *keys))):
        raise ValueError('committed prefill snapshot needs exact unique transaction identities')
    history = owner.leases.snapshot()
    step = history['committed_steps'].get(optimizer_step_id)
    by_id = {v['lease_id']: v for v in history['leases']}
    if step is None or set(step['lease_ids']) != set(ids) or any((i not in by_id for i in ids)) or ({by_id[i]['key'] for i in ids} != set(keys)) or any((v['state'] not in (LeaseState.COMMITTED.value, LeaseState.CANCELLED.value) for v in history['leases'])) or any((by_id[i]['state'] != LeaseState.COMMITTED.value or not by_id[i]['payload_released'] for i in ids)):
        raise RuntimeError('committed prefill snapshot cannot include an active or uncommitted optimizer transaction')
    if owner._reclaim_pending:
        raise RuntimeError('committed prefill snapshot has uncertain payload reclamation')
    with owner.router._lock:
        if owner.router._acquires:
            raise RuntimeError('committed prefill snapshot has an active router acquire')
        delivered = [dict(key=v.key, producer_endpoint=v.producer_endpoint, metadata=v.metadata.to_dict(), payload_checksum=v.payload_checksum) for v in owner.router._delivered.values()]
        ready = dict(owner.router._records)
    delivered_by_key = {v['key']: v for v in delivered}
    if any((k not in delivered_by_key for k in keys)):
        raise RuntimeError('committed transaction has no delivered source proof')
    seq = {delivered_by_key[k]['metadata']['sequence_no'] for k in keys}
    if seq != set(range(committed_cursor - len(keys), committed_cursor)):
        raise RuntimeError('current committed source rows differ from the stream boundary')
    if any((v['metadata']['sequence_no'] >= committed_cursor for v in delivered)):
        raise RuntimeError('future source rows have already been acquired')
    if set(ready) != set(owner._retained):
        raise RuntimeError('future ready and retained payload identities differ')
    future = [(v[0], v[2]) for v in owner._pending.values()] + list(owner._retained.values())
    limit = committed_prefill_future_limit(owner.capacity_bytes, prefill_geometry)
    if len(future) > limit or sum((n for _, n in future)) != owner._used_bytes or (not 0 <= owner._used_bytes <= owner.capacity_bytes):
        raise RuntimeError('future prefill exceeds the fixed double-buffer quota')
    names = ('target_model_fingerprint', 'tokenizer_fingerprint', 'dataset_fingerprint', 'target_revision', 'target_layer_ids', 'hidden_dtype', 'hidden_states_layout', 'data_version')
    current = delivered_by_key[keys[0]]['metadata']
    if any((any((delivered_by_key[k]['metadata'][name] != current[name] for name in names)) for k in keys)):
        raise RuntimeError('current committed source is not one frozen teacher recipe')
    for record, nbytes in future:
        record.validate()
        meta = record.metadata.to_dict()
        if meta['run_id'] != owner.run_id or meta['generation'] != owner.generation or (not committed_cursor <= meta['sequence_no'] < committed_cursor + len(keys)) or (nbytes != meta['payload_nbytes']) or any((meta[name] != current[name] for name in names)) or (owner._producer_incarnations.get(meta['producer_id']) != meta['producer_incarnation']):
            raise RuntimeError('future prefill lies outside the next frozen-teacher batch')
    clone = RayReadyControl(run_id=owner.run_id, generation=owner.generation, partition=owner.partition, capacity_bytes=owner.capacity_bytes, reservation_timeout_s=owner.reservation_timeout_s)
    clone.leases.restore(history)
    clone.router.restore_quiescent(delivered)
    clone._producer_incarnations = dict(owner._producer_incarnations)
    clone._acquired = dict(owner._acquired)
    clone._acquired_required_keys = dict(owner._acquired_required_keys)
    snapshot = clone.snapshot_quiescent()
    assert snapshot['leases'] == history
    projection = dict(schema_version=1, kind='COMMITTED_PREFIX_WITH_LIVE_FUTURE_PREFILL', committed_cursor=committed_cursor, committed_lease_count=len(ids), excluded_future_records=len(future), excluded_future_bytes=owner._used_bytes, excluded_future_keys_sha256=sha256(json.dumps(sorted((v.key for v, _ in future)), separators=(',', ':')).encode()).hexdigest(), future_optimizer_credit=0, live_source_payloads_unchanged=True)
    return (clone, projection)
from verl_speco.transport.streaming import MAX_NATIVE_BATCH_SAMPLES
from collections import OrderedDict
import hashlib
import re
from verl_speco.transport.streaming import make_v3_sample_key

def wave_batch_identity(samples):
    samples = tuple(samples)
    if not 1 <= len(samples) <= MAX_NATIVE_BATCH_SAMPLES or len({k for k, _ in samples}) != len(samples):
        raise ValueError('wave batch requires distinct bounded sample identities')
    first = samples[0][1]
    for key, metadata in samples:
        metadata.validate()
        if key != make_v3_sample_key(metadata) or (metadata.run_id, metadata.generation) != (first.run_id, first.generation):
            raise ValueError('wave batch sample owner differs')
    return hashlib.sha256(json.dumps([key for key, _ in samples], separators=(',', ':')).encode()).hexdigest()
MAX_MATERIALIZED_WAVE_SAMPLES = 112

class MaterializedWaveAckHints:
    """Ephemeral acceleration only; durable ACK files and DCP own recovery."""

    def __init__(self, max_entries=512):
        if type(max_entries) is not int or not 1 <= max_entries <= 512:
            raise ValueError('invalid wave hint bound')
        self._limit = max_entries
        self._entries = OrderedDict()

    @staticmethod
    def _key(batch_id, wave_index):
        if not isinstance(batch_id, str) or re.fullmatch('[0-9a-f]{64}', batch_id) is None or type(wave_index) is not int or (not 0 <= wave_index < MAX_NATIVE_BATCH_SAMPLES):
            raise ValueError('invalid wave hint identity')
        return (batch_id, wave_index)

    def publish(self, *, batch_id, ack, lease_ids, get_lease, run_id, generation):
        if not isinstance(ack, dict):
            raise ValueError('wave hint must be metadata')
        key = self._key(batch_id, ack.get('wave_index'))
        required = {'schema_version', 'run_id', 'generation', 'wave_index', 'state', 'sample_keys', 'staged_samples', 'expected_samples', 'optimizer_started', 'committed_original_tokens'}
        if set(ack) != required or type(ack['schema_version']) is not int or ack['schema_version'] != 1 or (ack['run_id'] != run_id) or (type(ack['generation']) is not int) or (ack['generation'] != generation) or (ack['state'] != 'PREFILL_WAVE_MATERIALIZED') or (ack['optimizer_started'] is not False) or (type(ack['committed_original_tokens']) is not int) or (ack['committed_original_tokens'] != 0) or (not isinstance(ack['sample_keys'], list)) or (not 1 <= len(ack['sample_keys']) <= MAX_MATERIALIZED_WAVE_SAMPLES) or any((not isinstance(k, str) or not k for k in ack['sample_keys'])) or (len(set(ack['sample_keys'])) != len(ack['sample_keys'])) or (type(ack['staged_samples']) is not int) or (type(ack['expected_samples']) is not int) or (not len(ack['sample_keys']) <= ack['staged_samples'] <= ack['expected_samples'] <= MAX_NATIVE_BATCH_SAMPLES) or (ack['wave_index'] >= ack['expected_samples']) or (not isinstance(lease_ids, tuple)) or (len(lease_ids) != len(ack['sample_keys'])) or (len(set(lease_ids)) != len(lease_ids)):
            raise ValueError('wave hint owner, size or credit differs')
        data = json.dumps(ack, sort_keys=True, separators=(',', ':')).encode()
        if len(data) > 65536:
            raise ValueError('wave hint exceeds metadata budget')
        leases = [get_lease(i) for i in lease_ids]
        if any((l.state is not LeaseState.MATERIALIZED or not l.payload_released or l.generation != generation or (l.consumer_id != 'dspark-trainer') for l in leases)) or [l.key for l in leases] != ack['sample_keys']:
            raise RuntimeError('wave hint does not prove owned materialized and released leases')
        old = self._entries.get(key)
        if old is not None and old != data:
            raise RuntimeError('wave hint identity changed')
        self._entries[key] = data
        self._entries.move_to_end(key)
        while len(self._entries) > self._limit:
            self._entries.popitem(last=False)
        return 'WAVE_HINT_PUBLISHED'

    def get(self, *, batch_id, wave_index):
        value = self._entries.get(self._key(batch_id, wave_index))
        return json.loads(value) if value is not None else None

def required_wave_keys(request, keys):
    if keys is None:
        return None
    if not isinstance(keys, tuple) or not 1 <= len(keys) <= MAX_MATERIALIZED_WAVE_SAMPLES or request.min_items != len(keys) or (request.max_items != len(keys)) or any((not isinstance(key, str) or not key for key in keys)) or (len(set(keys)) != len(keys)):
        raise ValueError('exact wave acquisition requires unique keys and its complete item count')
    keys = tuple(sorted(keys))
    if len(json.dumps(keys, ensure_ascii=True, separators=(',', ':')).encode()) > 65536:
        raise ValueError('exact wave identities exceed the bounded control budget')
    return keys
import hashlib, json, re

class TwoTrainerWaveBarrier:

    def __init__(self, samples, *, wave_size, incarnations, plan_sha256, get_lease):
        samples = tuple(samples)
        self.batch_id = wave_batch_identity(samples)
        if type(wave_size) is not int or not 1 <= wave_size <= MAX_MATERIALIZED_WAVE_SAMPLES or (not isinstance(incarnations, tuple)) or (len(incarnations) != 2) or any((not isinstance(v, str) or not v or len(v) > 256 for v in incarnations)) or (len(set(incarnations)) != 2) or (not isinstance(plan_sha256, str)) or (re.fullmatch('[0-9a-f]{64}', plan_sha256) is None) or (not callable(get_lease)):
            raise ValueError('invalid two-trainer wave ownership binding')
        first = samples[0][1]
        for index, (key, m) in enumerate(samples):
            if m.sequence_no != first.sequence_no + index or any((getattr(m, k) != getattr(first, k) for k in ('target_revision', 'target_model_fingerprint', 'tokenizer_fingerprint', 'target_layer_ids', 'dataset_fingerprint'))):
                raise ValueError('wave plan differs from complete ordered batch identity')
        self.samples = samples
        self.wave_size = wave_size
        self.run_id = first.run_id
        self.generation = first.generation
        self.incarnations = dict(zip((0, 16), incarnations))
        self.plan_sha256 = plan_sha256
        self.get_lease = get_lease
        self.state = 'OPEN'
        self.next_wave = 0
        self.current = None
        self.acks = {}

    def _leader(self, rank):
        if type(rank) is not int or rank != 0:
            raise ValueError('only global rank zero owns source release')

    def _open(self):
        if self.state != 'OPEN':
            raise RuntimeError('two-trainer wave group is failed or closed')

    def _leases(self, leased):
        for key, lease_id in zip(self.current['sample_keys'], self.current['lease_ids']):
            lease = self.get_lease(lease_id)
            if lease.key != key or lease.generation != self.generation or lease.consumer_id != 'dspark-trainer':
                raise RuntimeError('wave lease owner differs')
            if leased:
                if lease.state is not LeaseState.LEASED or lease.payload_released:
                    raise RuntimeError('source released before all trainer nodes staged')
            elif lease.state is not LeaseState.MATERIALIZED or not lease.payload_released:
                raise RuntimeError('wave source release not complete')

    def begin(self, *, caller_rank, wave_index, records, lease_ids):
        self._leader(caller_rank)
        self._open()
        if type(wave_index) is not int or wave_index != self.next_wave or self.current is not None or (not isinstance(lease_ids, tuple)) or (not lease_ids) or (len(set(lease_ids)) != len(lease_ids)):
            raise ValueError('wave order or global lease set differs')
        expected = self.samples[wave_index * self.wave_size:(wave_index + 1) * self.wave_size]
        records = tuple(records)
        if not expected or len(records) != len(expected) or len(lease_ids) != len(expected):
            raise ValueError('wave has incomplete global transaction set')
        for record, (key, m) in zip(records, expected):
            record.validate()
            if record.key != key or record.metadata != m:
                raise ValueError('wave capture or producer incarnation differs')
        value = dict(run_id=self.run_id, generation=self.generation, batch_id=self.batch_id, plan_sha256=self.plan_sha256, wave_index=wave_index, sample_keys=[key for key, _ in expected], lease_ids=list(lease_ids), payload_checksums=[r.payload_checksum for r in records], metadata=[m.to_dict() for _, m in expected])
        if len(json.dumps(value).encode()) > 256 * 1024:
            raise ValueError('wave metadata exceeds budget')
        for key, lease_id in zip(value['sample_keys'], lease_ids):
            lease = self.get_lease(lease_id)
            if lease.key != key or lease.generation != self.generation or lease.consumer_id != 'dspark-trainer' or (lease.state is not LeaseState.LEASED) or lease.payload_released:
                raise RuntimeError('wave lease owner or retained state differs')
        self.current = value
        self.acks = {}
        return self.get()

    def get(self):
        self._open()
        return json.loads(json.dumps(self.current)) if self.current is not None else None

    def acknowledge(self, ack):
        self._open()
        fields = {'run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index', 'owner_rank', 'node_incarnation', 'sample_keys', 'payload_checksums', 'ram_staged', 'optimizer_started', 'committed_original_tokens'}
        if not isinstance(ack, dict) or set(ack) != fields or self.current is None:
            raise ValueError('wave ACK must be bounded metadata with an active global transaction')
        rank = ack['owner_rank']
        if type(rank) is not int or rank not in self.incarnations or ack['node_incarnation'] != self.incarnations[rank] or (type(ack['generation']) is not int) or (type(ack['wave_index']) is not int) or (ack['ram_staged'] is not True) or (ack['optimizer_started'] is not False) or (type(ack['committed_original_tokens']) is not int) or (ack['committed_original_tokens'] != 0) or any((ack[k] != self.current[k] for k in ('run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index', 'sample_keys', 'payload_checksums'))):
            raise ValueError('wave ACK owner, retained capture, checksum or credit differs')
        data = json.dumps(ack, sort_keys=True, separators=(',', ':')).encode()
        if len(data) > 65536:
            raise ValueError('wave ACK metadata exceeds budget')
        self._leases(True)
        old = self.acks.get(rank)
        if old is not None and old != data:
            raise RuntimeError('trainer-node ACK changed incarnation or capture')
        self.acks[rank] = data
        return 'ALL_NODES_RAM_STAGED' if len(self.acks) == 2 else 'WAITING_FOR_PEER_NODE'

    def release_permit(self, *, caller_rank):
        self._leader(caller_rank)
        self._open()
        if self.current is None or set(self.acks) != {0, 16}:
            raise RuntimeError('cannot release source before both trainer-node owners staged')
        self._leases(True)
        return dict(run_id=self.run_id, generation=self.generation, batch_id=self.batch_id, wave_index=self.current['wave_index'], lease_ids=list(self.current['lease_ids']), all_node_ACK_sha256=hashlib.sha256(b''.join((self.acks[rank] for rank in (0, 16)))).hexdigest(), optimizer_started=False, committed_original_tokens=0)

    def confirm_release(self, *, caller_rank):
        self._leader(caller_rank)
        self._open()
        if self.current is None or set(self.acks) != {0, 16}:
            raise RuntimeError('release did not have both trainer-node ACKs')
        self._leases(False)
        self.current = None
        self.acks = {}
        self.next_wave += 1
        return 'WAVE_SOURCE_RELEASE_CONFIRMED'

    def abort(self, *, caller_rank, node_incarnation, reason):
        if type(caller_rank) is not int or self.incarnations.get(caller_rank) != node_incarnation or (not isinstance(reason, str)) or (not 1 <= len(reason) <= 1024):
            raise ValueError('wave abort owner or reason differs')
        self.state = 'FAILED'
        return dict(state=self.state, optimizer_started=False, committed_original_tokens=0, owned_lease_ids=[] if self.current is None else list(self.current['lease_ids']))
import verl_speco.transport.waves as m

class SelectiveRetainedWaveBarrier(TwoTrainerWaveBarrier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verification = None

    def begin(self, *, caller_rank, wave_index, records, lease_ids, routes):
        self._leader(caller_rank)
        self._open()
        records = tuple(records)
        expected = self.samples[wave_index * self.wave_size:(wave_index + 1) * self.wave_size]
        identity = dict(run_id=self.run_id, generation=self.generation, batch_id=self.batch_id, plan_sha256=self.plan_sha256, wave_index=wave_index)
        verification = m.SelectiveWaveVerification(identity=identity, routes=routes, incarnations=self.incarnations)
        if [r['sample_key'] for r in routes] != [k for k, _ in expected] or [r['payload_sha256'] for r in routes] != [r.payload_checksum for r in records]:
            raise ValueError('selective routes differ from retained source records')
        value = super().begin(caller_rank=caller_rank, wave_index=wave_index, records=records, lease_ids=lease_ids)
        self.verification = verification
        return dict(value, selective_routes=json.loads(json.dumps(verification.routes)), selective_routing_sha256=verification.routing_sha256)

    def get(self):
        value = super().get()
        if value is None or self.verification is None:
            return value
        return dict(value, selective_routes=json.loads(json.dumps(self.verification.routes)), selective_routing_sha256=self.verification.routing_sha256)

    def acknowledge(self, ack):
        if not isinstance(ack, dict) or 'selective_verification' not in ack or self.verification is None:
            raise ValueError('selective proof missing')
        proof = ack['selective_verification']
        base = {k: v for k, v in ack.items() if k != 'selective_verification'}
        if not isinstance(proof, dict) or proof.get('owner_rank') != base.get('owner_rank') or proof.get('node_incarnation') != base.get('node_incarnation'):
            raise ValueError('selective and retained lease owners differ')
        old = dict(self.verification.acks)
        try:
            self.verification.acknowledge(proof)
            return super().acknowledge(base)
        except BaseException:
            self.verification.acks = old
            raise

    def release_permit(self, *, caller_rank):
        self._leader(caller_rank)
        self._open()
        if self.verification is None:
            raise RuntimeError('no active selective verification')
        proof = self.verification.release_proof()
        value = super().release_permit(caller_rank=caller_rank)
        return dict(value, selective_verification_proof=proof)

    def confirm_release(self, *, caller_rank):
        self.release_permit_after_materialization_check(caller_rank)
        value = super().confirm_release(caller_rank=caller_rank)
        self.verification = None
        return value

    def release_permit_after_materialization_check(self, caller_rank):
        self._leader(caller_rank)
        self._open()
        if self.verification is None:
            raise RuntimeError('missing selective verification')
        self.verification.release_proof()

    def abort(self, **kwargs):
        if self.verification is not None:
            self.verification.abort()
        return super().abort(**kwargs)

def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def digest(v):
    return isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) is not None

class SelectiveWaveVerification:

    def __init__(self, *, identity, routes, incarnations):
        if not isinstance(identity, dict) or set(identity) != {'run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index'} or any((not isinstance(identity[k], str) or not identity[k] for k in ['run_id', 'batch_id'])) or (type(identity['generation']) is not int) or (identity['generation'] < 1) or (type(identity['wave_index']) is not int) or (identity['wave_index'] < 0) or (not digest(identity['plan_sha256'])) or (not isinstance(incarnations, dict)) or (set(incarnations) != {0, 16}) or any((not isinstance(v, str) or not v or len(v) > 256 for v in incarnations.values())) or (len(set(incarnations.values())) != 2) or (not isinstance(routes, list)) or (not 1 <= len(routes) <= MAX_MATERIALIZED_WAVE_SAMPLES):
            raise ValueError('invalid selective verification binding')
        keys = []
        for row in routes:
            if not isinstance(row, dict) or set(row) != {'sample_key', 'required_hidden_owners', 'payload_sha256', 'light_sha256'} or (not isinstance(row['sample_key'], str)) or (not row['sample_key']) or (len(row['sample_key']) > 4096) or (not isinstance(row['required_hidden_owners'], list)) or (row['required_hidden_owners'] not in ([0], [16], [0, 16])) or any((type(v) is not int for v in row['required_hidden_owners'])) or (not digest(row['payload_sha256'])) or (not digest(row['light_sha256'])):
                raise ValueError('invalid selective ownership or payload proof')
            keys.append(row['sample_key'])
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate selective sample identity')
        self.identity = json.loads(json.dumps(identity))
        self.routes = json.loads(json.dumps(routes))
        self.incarnations = dict(incarnations)
        self.routing_sha256 = sha(dict(identity=self.identity, routes=self.routes))
        self.acks = {}
        self.failed = False

    def acknowledge(self, ack):
        if self.failed:
            raise RuntimeError('selective verification aborted')
        fields = {'identity', 'routing_sha256', 'owner_rank', 'node_incarnation', 'entries', 'staging_joined', 'optimizer_started', 'committed_original_tokens'}
        if not isinstance(ack, dict) or set(ack) != fields:
            raise ValueError('selective ACK schema differs')
        rank = ack['owner_rank']
        if type(rank) is not int or rank not in (0, 16) or ack['node_incarnation'] != self.incarnations[rank] or (not isinstance(ack['identity'], dict)) or (ack['identity'] != self.identity) or (type(ack['identity']['generation']) is not int) or (type(ack['identity']['wave_index']) is not int) or (ack['routing_sha256'] != self.routing_sha256) or (ack['staging_joined'] is not True) or (ack['optimizer_started'] is not False) or (type(ack['committed_original_tokens']) is not int) or (ack['committed_original_tokens'] != 0) or (not isinstance(ack['entries'], list)) or (len(ack['entries']) != len(self.routes)):
            raise ValueError('selective ACK binding, joins or credit differ')
        for entry, route in zip(ack['entries'], self.routes):
            full = rank in route['required_hidden_owners']
            mode = 'full_payload' if full else 'immutable_replay_light'
            if not isinstance(entry, dict) or set(entry) != {'sample_key', 'verification', 'verified_sha256'} or entry['sample_key'] != route['sample_key'] or (entry['verification'] != mode) or (entry['verified_sha256'] != route['payload_sha256' if full else 'light_sha256']):
                raise ValueError('selective ACK cannot substitute omitted hidden verification')
        frozen = json.dumps(ack, sort_keys=True, separators=(',', ':')).encode()
        if len(frozen) > 65536:
            raise ValueError('selective ACK exceeds metadata budget')
        old = self.acks.get(rank)
        if old is not None and old != frozen:
            raise RuntimeError('selective ACK changed')
        self.acks[rank] = frozen
        return 'ALL_NODES_VERIFIED' if len(self.acks) == 2 else 'WAITING_FOR_PEER'

    def release_proof(self):
        if self.failed or set(self.acks) != {0, 16}:
            raise RuntimeError('selective source must stay retained until both joined proofs')
        return dict(identity=dict(self.identity), routing_sha256=self.routing_sha256, all_node_ACK_sha256=hashlib.sha256(b''.join((self.acks[r] for r in (0, 16)))).hexdigest(), optimizer_started=False, committed_original_tokens=0)

    def abort(self):
        self.failed = True
import hashlib, ipaddress, json, re
from verl_speco.transport.streaming import SampleMetadataV3

class TwoTrainerWaveGroup:

    def __init__(self, *, samples, wave_size, plan_sha256, owner_addresses, get_lease, selective_plan=None):
        if not isinstance(samples, (list, tuple)) or not 1 <= len(samples) <= MAX_NATIVE_BATCH_SAMPLES or any((not isinstance(v, dict) or set(v) != {'key', 'metadata'} for v in samples)):
            raise ValueError('invalid bounded DP32 sample metadata')
        if len(json.dumps(samples).encode()) > 16 * 1024 ** 2:
            raise ValueError('DP32 sample metadata exceeds budget')
        self.samples = tuple(((v['key'], SampleMetadataV3.from_dict(v['metadata'])) for v in samples))
        self.batch_id = wave_batch_identity(self.samples)
        if type(wave_size) is not int or not 1 <= wave_size <= MAX_MATERIALIZED_WAVE_SAMPLES or (not isinstance(plan_sha256, str)) or (re.fullmatch('[0-9a-f]{64}', plan_sha256) is None) or (not isinstance(owner_addresses, list)) or (len(owner_addresses) != 2) or (len(set(owner_addresses)) != 2):
            raise ValueError('invalid DP32 wave group plan')
        for address in owner_addresses:
            if not isinstance(address, str) or str(ipaddress.IPv4Address(address)) != address:
                raise ValueError('invalid trainer address')
        self.wave_size = wave_size
        self.plan_sha256 = plan_sha256
        self.addresses = dict(zip((0, 16), owner_addresses))
        self.get_lease = get_lease
        self.incarnations = {}
        self.barrier = None
        self.failed = False
        self.binding = hashlib.sha256(json.dumps(dict(samples=[dict(key=k, metadata=m.to_dict()) for k, m in self.samples], wave_size=wave_size, plan_sha256=plan_sha256, owner_addresses=owner_addresses), sort_keys=True).encode()).hexdigest()
        self.selective_plan = None
        if selective_plan is not None:
            if not isinstance(selective_plan, list) or len(selective_plan) != len(self.samples) or len(json.dumps(selective_plan).encode()) > 4 * 1024 ** 2:
                raise ValueError('selective plan sample set or metadata budget differs')
            for row, (key, m) in zip(selective_plan, self.samples):
                if not isinstance(row, dict) or set(row) != {'sample_key', 'required_hidden_owners', 'light_sha256'} or row['sample_key'] != key or (not isinstance(row['required_hidden_owners'], list)) or (row['required_hidden_owners'] not in ([0], [16], [0, 16])) or any((type(rank) is not int for rank in row['required_hidden_owners'])) or (not isinstance(row['light_sha256'], str)) or (re.fullmatch('[0-9a-f]{64}', row['light_sha256']) is None):
                    raise ValueError('selective routing template differs')
            self.selective_plan = json.loads(json.dumps(selective_plan))
            self.binding = hashlib.sha256(json.dumps(dict(original_binding=self.binding, selective_plan=self.selective_plan), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def register(self, *, owner_rank, address, incarnation):
        if self.failed:
            raise RuntimeError('DP32 wave group failed')
        if type(owner_rank) is not int or owner_rank not in self.addresses or address != self.addresses[owner_rank] or (not isinstance(incarnation, str)) or (re.fullmatch('[0-9a-f]{64}', incarnation) is None):
            raise ValueError('DP32 wave owner rank, physical address or process incarnation differs')
        old = self.incarnations.get(owner_rank)
        if old is not None and old != incarnation:
            raise RuntimeError('trainer process incarnation changed within batch')
        if incarnation in [v for rank, v in self.incarnations.items() if rank != owner_rank]:
            raise ValueError('trainer owners must be different process incarnations')
        self.incarnations[owner_rank] = incarnation
        if self.barrier is None and len(self.incarnations) == 2:
            barrier_class = TwoTrainerWaveBarrier
            if self.selective_plan is not None:
                barrier_class = SelectiveRetainedWaveBarrier
            self.barrier = barrier_class(self.samples, wave_size=self.wave_size, incarnations=tuple((self.incarnations[r] for r in (0, 16))), plan_sha256=self.plan_sha256, get_lease=self.get_lease)
        return self.state()

    def _owner(self, owner_rank, incarnation):
        if type(owner_rank) is not int or self.incarnations.get(owner_rank) != incarnation:
            raise ValueError('stale or foreign DP32 wave caller')
        if self.failed or self.barrier is None:
            raise RuntimeError('DP32 wave group failed or owners not registered')

    def state(self):
        if self.failed:
            return dict(state='FAILED', batch_id=self.batch_id, registered_owner_ranks=sorted(self.incarnations), acknowledged_owner_ranks=[], next_wave=0, wave=None)
        barrier = self.barrier
        return dict(state='READY' if barrier else 'WAITING_FOR_OWNERS', batch_id=self.batch_id, registered_owner_ranks=sorted(self.incarnations), acknowledged_owner_ranks=sorted(barrier.acks) if barrier else [], next_wave=barrier.next_wave if barrier else 0, wave=barrier.get() if barrier else None)

    def begin(self, *, owner_rank, incarnation, wave_index, records, lease_ids):
        self._owner(owner_rank, incarnation)
        kwargs = {}
        if self.selective_plan is not None:
            records = tuple(records)
            template = self.selective_plan[wave_index * self.wave_size:(wave_index + 1) * self.wave_size]
            if len(records) != len(template):
                raise ValueError('selective wave source count differs')
            kwargs['routes'] = [dict(row, payload_sha256=record.payload_checksum) for row, record in zip(template, records)]
        return self.barrier.begin(caller_rank=owner_rank, wave_index=wave_index, records=records, lease_ids=lease_ids, **kwargs)

    def acknowledge(self, ack):
        if not isinstance(ack, dict):
            raise ValueError('wave ACK must be metadata')
        self._owner(ack.get('owner_rank'), ack.get('node_incarnation'))
        return self.barrier.acknowledge(ack)

    def release_permit(self, *, owner_rank, incarnation):
        self._owner(owner_rank, incarnation)
        return self.barrier.release_permit(caller_rank=owner_rank)

    def confirm_release(self, *, owner_rank, incarnation):
        self._owner(owner_rank, incarnation)
        return self.barrier.confirm_release(caller_rank=owner_rank)

    def abort(self, *, owner_rank, incarnation, reason):
        if type(owner_rank) is not int or self.incarnations.get(owner_rank) != incarnation:
            raise ValueError('foreign DP32 group abort')
        if not isinstance(reason, str) or not 1 <= len(reason) <= 1024:
            raise ValueError('invalid DP32 abort metadata')
        if self.barrier is not None:
            self.barrier.abort(caller_rank=owner_rank, node_incarnation=incarnation, reason=reason)
        self.failed = True
        return self.state()
