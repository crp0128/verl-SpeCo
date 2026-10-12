# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path
import torch
import verl_speco
from verl_speco.runtime.artifacts import publish_json, read_published_json, sync_directory
from verl_speco.producer.feature_payload import payload_checksum
from verl_speco.trainer.owned_batches import materialize_light_columns
from verl_speco.transport.streaming import make_v3_sample_key
LIMIT = 512 * 1024 ** 2

def root():
    return Path(verl_speco.__file__).resolve().parents[1] / 'outputs'

def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def checked_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to(root().resolve()):
        raise ValueError('light replay must stay under outputs')
    return p

def write_light_replay(path, *, samples, replay_light, hidden_size):
    path = checked_path(path)
    samples = tuple(samples)
    if sys.byteorder != 'little' or not samples or len({k for k, _ in samples}) != len(samples):
        raise ValueError('light replay needs unique samples and little endian host')
    if set(replay_light) != {k for k, _ in samples}:
        raise ValueError('light replay differs from exact complete samples')
    size = sum((20 * m.token_count for _, m in samples))
    if not 0 < size <= LIMIT:
        raise ValueError('light replay exceeds CPU byte budget')
    rows = []
    offset = 0
    hasher = hashlib.sha256()
    for key, m in samples:
        if key != make_v3_sample_key(m):
            raise ValueError('light replay sample key differs')
        materialize_light_columns(replay_light[key], m, hidden_size=hidden_size)
        rows.append(dict(key=key, metadata=m.to_dict(), offset=offset, nbytes=20 * m.token_count, light_sha256=payload_checksum(replay_light[key])))
        offset += 20 * m.token_count
    try:
        existing = read_published_json(path, max_bytes=4 * 1024 ** 2)
    except FileNotFoundError:
        pass
    else:
        if not isinstance(existing, dict) or existing.get('samples') != rows or existing.get('hidden_size') != hidden_size:
            raise ValueError('immutable light replay already exists with different inputs')
        digest = sha(existing)
        verified = load_light_replay(path, expected_sha256=digest, samples=samples, hidden_size=hidden_size)
        if any((payload_checksum(verified[key]) != payload_checksum(replay_light[key]) for key, _ in samples)):
            raise ValueError('immutable light replay differs from actual input tokens')
        return digest
    temporary = path.with_name(path.name + '.payload-staging')
    with temporary.open('xb') as writer:
        for key, _ in samples:
            for name in ('input_ids', 'position_ids', 'loss_mask'):
                data = replay_light[key][name].contiguous().numpy().tobytes()
                writer.write(data)
                hasher.update(data)
        writer.flush()
        os.fsync(writer.fileno())
    payload = path.with_name(path.name + '.' + hasher.hexdigest() + '.bin')
    try:
        os.link(temporary, payload)
    except OSError:
        with temporary.open('rb') as reader, payload.open('xb') as writer:
            while True:
                data = reader.read(1024 ** 2)
                if not data:
                    break
                writer.write(data)
            writer.flush()
            os.fsync(writer.fileno())
    temporary.unlink()
    sync_directory(payload.parent)
    value = dict(schema_version=1, byteorder='little', hidden_size=hidden_size, nbytes=size, payload=payload.name, payload_sha256=hasher.hexdigest(), samples=rows)
    publish_json(path, value)
    return sha(value)

def load_light_replay(path, *, expected_sha256, samples, hidden_size):
    path = checked_path(path)
    samples = tuple(samples)
    v = read_published_json(path, max_bytes=4 * 1024 ** 2)
    if sys.byteorder != 'little' or not isinstance(v, dict) or set(v) != {'schema_version', 'byteorder', 'hidden_size', 'nbytes', 'payload', 'payload_sha256', 'samples'} or (type(v['schema_version']) is not int) or (v['schema_version'] != 1) or (v['byteorder'] != 'little') or (type(hidden_size) is not int) or (hidden_size < 1) or (type(v['hidden_size']) is not int) or (v['hidden_size'] != hidden_size) or (not isinstance(expected_sha256, str)) or (re.fullmatch('[0-9a-f]{64}', expected_sha256) is None) or (sha(v) != expected_sha256) or (type(v['nbytes']) is not int) or (not 0 < v['nbytes'] <= LIMIT) or (not isinstance(v['samples'], list)) or (len(v['samples']) != len(samples)) or (not samples):
        raise ValueError('light replay manifest binding or byte budget differs')
    if not isinstance(v['payload'], str) or Path(v['payload']).name != v['payload']:
        raise ValueError('light replay payload path differs')
    payload = checked_path(path.parent / v['payload'])
    offset = 0
    for row, (key, m) in zip(v['samples'], samples):
        m.validate()
        if not isinstance(row, dict) or set(row) != {'key', 'metadata', 'offset', 'nbytes', 'light_sha256'} or row['key'] != key or (key != make_v3_sample_key(m)) or (row['metadata'] != m.to_dict()) or (type(row['offset']) is not int) or (row['offset'] != offset) or (type(row['nbytes']) is not int) or (row['nbytes'] != 20 * m.token_count):
            raise ValueError('light replay sample prefix or geometry differs')
        offset += row['nbytes']
    if offset != v['nbytes']:
        raise ValueError('light replay total byte count differs')
    with payload.open('rb') as reader:
        data = reader.read(v['nbytes'] + 1)
    if len(data) != v['nbytes'] or hashlib.sha256(data).hexdigest() != v['payload_sha256']:
        raise ValueError('light replay payload SHA differs')
    result = {}
    for row, (key, m) in zip(v['samples'], samples):
        fields = {}
        offset = row['offset']
        for name, dtype, width in [('input_ids', torch.int64, 8), ('position_ids', torch.int64, 8), ('loss_mask', torch.float32, 4)]:
            count = m.token_count * width
            fields[name] = torch.frombuffer(bytearray(data[offset:offset + count]), dtype=dtype).clone()
            offset += count
        materialize_light_columns(fields, m, hidden_size=hidden_size)
        if payload_checksum(fields) != row['light_sha256']:
            raise ValueError('light replay sample SHA differs')
        result[key] = fields
    return result
from verl_speco.trainer.owned_batches import OwnedBuilderCopies
from verl_speco.trainer.feature_pipeline import ParallelSelectivePayloads
from verl_speco.trainer.feature_pipeline import stage_selected_session_wave
from verl_speco.transport.control import ReadyRecord

def make_selected_native_stage(*, session, reader, partition, builders, replay_light, hidden_size, prefetch_bytes, owned_copy_workers):
    lanes = getattr(reader, 'readers', None)
    if session.selective_plan is None or session.owner_rank not in (0, 16) or (not isinstance(lanes, list)) or (len(lanes) not in (2, 4)) or (not isinstance(builders, list)) or (len(builders) != 16) or any((getattr(b, 'builder', b).rank != session.owner_rank + i or getattr(b, 'builder', b).world_size != 32 for i, b in enumerate(builders))) or (type(prefetch_bytes) is not int) or (not 0 < prefetch_bytes <= 8 * 1024 ** 3) or (type(owned_copy_workers) is not int) or (owned_copy_workers not in (1, 2, 4, 8)):
        raise ValueError('selected native runtime requires exact independent DP32 owners and lanes')

    def initialize():
        if reader.device.type == 'npu':
            import torch
            torch.npu.set_device(reader.device)

    def stage(value, records):
        session._validate_selected_wave(value)
        verification = session.verification_for(value)
        reader.prepare_receive(value['sample_keys'], partition)
        copies = OwnedBuilderCopies(builders, workers=owned_copy_workers)
        prefetch = None
        try:
            prefetch = ParallelSelectivePayloads(records, routes=value['selective_routes'], owner_rank=session.owner_rank, readers=lanes, partition=partition, replay_light=replay_light, hidden_size=hidden_size, max_bytes=prefetch_bytes, initialize=initialize)
            return stage_selected_session_wave(session=session, value=value, records=records, reader=lanes[0], partition=partition, copies=copies, replay_light=replay_light, hidden_size=hidden_size, verification=verification, prefetch=prefetch)
        except BaseException:
            import traceback
            for close in ([prefetch.close] if prefetch is not None else []) + [copies.close, verification.abort]:
                try:
                    close()
                except BaseException:
                    traceback.print_exc()
            raise
    return stage

def stage_selected_follower_waves(samples, *, session, stage_wave):
    samples = tuple(samples)
    if samples != session.samples or session.owner_rank != 16 or session.selective_plan is None:
        raise ValueError('selected follower differs from complete registered batch')
    digests = []
    lease_ids = []
    try:
        for wave_index, start in enumerate(range(0, len(samples), session.wave_size)):
            expected = samples[start:start + session.wave_size]
            value = session.wait(lambda state: state['wave'] is not None and state['wave']['wave_index'] == wave_index)['wave']
            session._validate_selected_wave(value)
            records = tuple((ReadyRecord(key, m.producer_id, m, digest) for (key, m), digest in zip(expected, value['payload_checksums'])))
            stage_wave(value, records)
            session.wait(lambda state: state['next_wave'] > wave_index)
            digests.extend(value['payload_checksums'])
            lease_ids.extend(value['lease_ids'])
    except BaseException:
        try:
            session.abort()
        except Exception:
            pass
        raise
    if len(lease_ids) != len(samples) or len(set(lease_ids)) != len(lease_ids):
        raise RuntimeError('selected follower did not stage one exact global lease set')
    return dict(payload_checksums=digests, lease_ids=lease_ids)
import hashlib, time, json, math
from verl_speco.trainer.feature_pipeline import SelectedWaveMaterializer
from verl_speco.runtime.services import check_prefill_abort
from verl_speco.runtime.artifacts import publish_json, read_published_json
from verl_speco.transport.control import AcquireRequest

def stage_selected_primary_native_waves(samples, *, session, control, stage_wave, tq, partition, builders, max_wave_bytes, ack_dir, timeout=1800, poll_seconds=0.5):
    samples = tuple(samples)
    ack_dir = Path(ack_dir).resolve()
    if samples != session.samples or session.owner_rank != 0 or session.selective_plan is None or (not ack_dir.is_relative_to(Path(verl_speco.__file__).resolve().parents[1] / 'outputs')) or (type(timeout) not in (int, float)) or (not math.isfinite(timeout)) or (timeout <= 0) or (type(poll_seconds) not in (int, float)) or (not math.isfinite(poll_seconds)) or (not 0 < poll_seconds <= 30):
        raise ValueError('selected primary placement or wait bounds differ')
    stager = SelectedWaveMaterializer(samples, control=control, consumer_id='dspark-trainer', max_wave_bytes=max_wave_bytes)
    ack_dir.mkdir(parents=True, exist_ok=True)

    def discard():
        for builder in builders:
            if callable(getattr(builder, 'close', None)):
                builder.close()
    try:
        for wave_index, start in enumerate(range(0, len(samples), session.wave_size)):
            check_prefill_abort(ack_dir, samples)
            wave = samples[start:start + session.wave_size]
            keys = [k for k, _ in wave]
            first = wave[0][1]
            ack_path = ack_dir / f'wave-{wave_index:04d}.ack.json'
            try:
                read_published_json(ack_path)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError('selected wave ACK already exists')
            started = time.perf_counter()
            deadline = time.monotonic() + timeout
            request = AcquireRequest('prefill-wave-' + hashlib.sha256('\n'.join(keys).encode()).hexdigest()[:20], 'dspark-trainer', first.generation, len(wave), len(wave), first.target_revision, time.time() + timeout)
            while True:
                check_prefill_abort(ack_dir, samples)
                acquired, leases = control.acquire(request, lease_timeout_s=timeout * 2, required_keys=tuple(keys))
                if acquired.status == 'ready':
                    break
                if acquired.status not in ('busy', 'waiting', 'pending', 'empty'):
                    raise RuntimeError('selected acquire failed: ' + acquired.status)
                if time.monotonic() >= deadline:
                    raise TimeoutError('selected wave exact acquire timed out')
                time.sleep(poll_seconds)

            def clear(keys):
                tq.kv_clear(keys=list(keys), partition_id=partition)
                remaining = tq.kv_list(partition_id=partition).get(partition, {})
                if any((k in remaining for k in keys)):
                    raise RuntimeError('selected TQ keys remain after clear')
            stager.stage_selected(acquired, leases, session=session, wave_index=wave_index, materialize_wave=stage_wave, clear_payload_batch=clear, discard=discard, quiesce=lambda: None)
            ack = dict(schema_version=1, run_id=first.run_id, generation=first.generation, wave_index=wave_index, state='PREFILL_WAVE_MATERIALIZED', sample_keys=keys, staged_samples=stager.snapshot()['staged_samples'], expected_samples=len(samples), optimizer_started=False, committed_original_tokens=0)
            publish_json(ack_path, ack)
            control.publish_materialized_wave_ack(batch_id=session.batch_id, ack=ack, lease_ids=tuple(stager.snapshot()['lease_ids'][-len(wave):]))
            print('selected_native_wave_timing ' + json.dumps(dict(run_id=first.run_id, generation=first.generation, wave_index=wave_index, original_tokens=sum((m.token_count for _, m in wave)), wallclock_seconds=time.perf_counter() - started, verification_scope='distributed full payload plus local immutable light', completed_original_tokens=0)), flush=True)
        ids, records = stager.seal()
        return dict(payload_checksums=[r.payload_checksum for r in records], lease_ids=list(ids))
    except BaseException:
        import traceback
        for cleanup in (discard, session.abort):
            try:
                cleanup()
            except BaseException:
                traceback.print_exc()
        raise
import hashlib, json, os, time, math
from verl_speco.transport.waves import wave_batch_identity

class NodeWaveSession:

    def __init__(self, *, samples, control, owner_rank, address, owner_addresses, plan_sha256, wave_size, ack_dir, timeout=1800, poll_seconds=0.5, incarnation=None, selective_plan=None):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0 or (type(poll_seconds) not in (int, float)) or (not math.isfinite(poll_seconds)) or (not 0 < poll_seconds <= 30):
            raise ValueError('invalid two-trainer peer wait bound')
        self.selective_plan = json.loads(json.dumps(selective_plan)) if selective_plan is not None else None
        self.plan_sha256 = plan_sha256
        self.wave_size = wave_size
        self.samples = tuple(samples)
        self.batch_id = wave_batch_identity(self.samples)
        self.control = control
        self.owner_rank = owner_rank
        self.timeout = timeout
        self.poll_seconds = poll_seconds
        self.ack_dir = ack_dir
        if incarnation is None:
            proc = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
            incarnation = hashlib.sha256((Path('/proc/sys/kernel/random/boot_id').read_text() + ':' + str(os.getpid()) + ':' + proc[19]).encode()).hexdigest()
        self.incarnation = incarnation
        control.register_node_wave_owner(samples=[dict(key=k, metadata=m.to_dict()) for k, m in self.samples], wave_size=wave_size, plan_sha256=plan_sha256, owner_addresses=owner_addresses, owner_rank=owner_rank, address=address, incarnation=incarnation, **{'selective_plan': self.selective_plan} if self.selective_plan is not None else {})
        self.wait(lambda state: state['state'] == 'READY')

    def state(self):
        check_prefill_abort(self.ack_dir, self.samples)
        state = self.control.get_node_wave_state(batch_id=self.batch_id)
        if not isinstance(state, dict) or set(state) != {'state', 'batch_id', 'registered_owner_ranks', 'acknowledged_owner_ranks', 'next_wave', 'wave'} or state['batch_id'] != self.batch_id or (state['state'] not in ('READY', 'WAITING_FOR_OWNERS')) or (type(state['next_wave']) is not int) or (not 0 <= state['next_wave'] <= (len(self.samples) + self.wave_size - 1) // self.wave_size):
            raise RuntimeError('two-trainer wave group failed or changed')
        for key in ('registered_owner_ranks', 'acknowledged_owner_ranks'):
            if not isinstance(state[key], list) or any((type(r) is not int or r not in (0, 16) for r in state[key])) or state[key] != sorted(set(state[key])):
                raise RuntimeError('two-trainer owner membership differs')
        if self.selective_plan is not None and state['wave'] is not None:
            self._validate_selected_wave(state['wave'])
        return state

    def wait(self, predicate):
        deadline = time.monotonic() + self.timeout
        while True:
            state = self.state()
            if predicate(state):
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError('two-trainer wave peer did not acknowledge; source stays retained')
            time.sleep(self.poll_seconds)

    def begin(self, *, wave_index, leases, expected, records):
        value = self.control.begin_node_wave(batch_id=self.batch_id, owner_rank=self.owner_rank, incarnation=self.incarnation, wave_index=wave_index, records=tuple(records), lease_ids=tuple((l.lease_id for l in leases)))
        if value['sample_keys'] != [key for key, _ in expected]:
            raise RuntimeError('global wave transaction differs from materializer')
        return value

    def _validate_selected_wave(self, value):
        from verl_speco.transport.waves import sha
        fields = {'run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index', 'sample_keys', 'lease_ids', 'payload_checksums', 'metadata', 'selective_routes', 'selective_routing_sha256'}
        if not isinstance(value, dict) or set(value) != fields or type(value['wave_index']) is not int or (type(value['generation']) is not int) or (value['generation'] != self.samples[0][1].generation) or (value['run_id'] != self.samples[0][1].run_id) or (value['batch_id'] != self.batch_id) or (value['plan_sha256'] != self.plan_sha256) or (value['wave_index'] < 0):
            raise ValueError('selected session wave binding differs')
        index = value['wave_index']
        expected = self.samples[index * self.wave_size:(index + 1) * self.wave_size]
        template = self.selective_plan[index * self.wave_size:(index + 1) * self.wave_size]
        if not expected or value['sample_keys'] != [k for k, _ in expected] or value['metadata'] != [m.to_dict() for _, m in expected] or (not isinstance(value['payload_checksums'], list)) or (len(value['payload_checksums']) != len(expected)) or (not isinstance(value['lease_ids'], list)) or (len(value['lease_ids']) != len(expected)) or any((not isinstance(i, str) or not i for i in value['lease_ids'])) or (len(set(value['lease_ids'])) != len(expected)):
            raise ValueError('selected session complete sample metadata differs')
        routes = [dict(row, payload_sha256=digest) for row, digest in zip(template, value['payload_checksums'])]
        identity = {k: value[k] for k in ['run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index']}
        if value['selective_routes'] != routes or value['selective_routing_sha256'] != sha(dict(identity=identity, routes=routes)):
            raise ValueError('selected session route or proof binding differs')

    def verification_for(self, value):
        from verl_speco.transport.waves import SelectiveWaveVerification
        if self.selective_plan is None:
            raise ValueError('legacy session has no selected verification')
        self._validate_selected_wave(value)
        incarnations = self.control.get_node_wave_owner_incarnations(batch_id=self.batch_id, owner_rank=self.owner_rank, incarnation=self.incarnation)
        if not isinstance(incarnations, dict) or set(incarnations) != {0, 16} or incarnations[self.owner_rank] != self.incarnation:
            raise ValueError('control physical owner incarnations differ')
        return SelectiveWaveVerification(identity={k: value[k] for k in ('run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index')}, routes=value['selective_routes'], incarnations=incarnations)

    def acknowledge(self, value, selective_verification=None):
        ack = {k: value[k] for k in ('run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index', 'sample_keys', 'payload_checksums')}
        ack.update(owner_rank=self.owner_rank, node_incarnation=self.incarnation, ram_staged=True, optimizer_started=False, committed_original_tokens=0)
        if self.selective_plan is not None:
            self._validate_selected_wave(value)
            if not isinstance(selective_verification, dict) or selective_verification.get('owner_rank') != self.owner_rank or selective_verification.get('node_incarnation') != self.incarnation:
                raise ValueError('selected session requires its actual joined staging proof')
            ack['selective_verification'] = selective_verification
        elif selective_verification is not None:
            raise ValueError('legacy session cannot accept selective proof')
        return self.control.acknowledge_node_wave(batch_id=self.batch_id, ack=ack)

    def before_release(self, value, selective_verification=None):
        self.acknowledge(value, selective_verification)
        self.wait(lambda state: state['acknowledged_owner_ranks'] == [0, 16])
        return self.control.node_wave_release_permit(batch_id=self.batch_id, owner_rank=self.owner_rank, incarnation=self.incarnation)

    def confirm_release(self):
        return self.control.confirm_node_wave_release(batch_id=self.batch_id, owner_rank=self.owner_rank, incarnation=self.incarnation)

    def abort(self):
        return self.control.abort_node_wave(batch_id=self.batch_id, owner_rank=self.owner_rank, incarnation=self.incarnation, reason='two-trainer RAM staging failed before optimizer')

def stage_follower_native_waves(samples, *, session, reader, partition, builders, wave_size, owned_copy_workers=1):
    samples = tuple(samples)
    copies = OwnedBuilderCopies(builders, workers=owned_copy_workers)
    digests = []
    lease_ids = []
    try:
        for wave_index, start in enumerate(range(0, len(samples), wave_size)):
            expected = samples[start:start + wave_size]
            state = session.wait(lambda state: state['wave'] is not None and state['wave']['wave_index'] == wave_index)
            value = state['wave']
            fields_set = {'run_id', 'generation', 'batch_id', 'plan_sha256', 'wave_index', 'sample_keys', 'lease_ids', 'payload_checksums', 'metadata'}
            if not isinstance(value, dict) or set(value) != fields_set or value['run_id'] != samples[0][1].run_id or (type(value['generation']) is not int) or (value['generation'] != samples[0][1].generation) or (type(value['wave_index']) is not int) or (value['wave_index'] != wave_index) or (value['batch_id'] != session.batch_id) or (value['plan_sha256'] != session.plan_sha256) or (value['sample_keys'] != [key for key, _ in expected]) or (json.dumps(value['metadata'], sort_keys=True) != json.dumps([m.to_dict() for _, m in expected], sort_keys=True)) or (not isinstance(value['lease_ids'], list)) or (len(value['lease_ids']) != len(expected)) or (len(set(value['lease_ids'])) != len(expected)) or (not isinstance(value['payload_checksums'], list)) or (len(value['payload_checksums']) != len(expected)):
                raise RuntimeError('follower wave metadata, producer incarnation or global transaction differs')
            reader.prepare_receive(value['sample_keys'], partition)
            prefetch = None
            if callable(getattr(reader, 'copy_lanes', None)):
                from verl_speco.trainer.feature_pipeline import ParallelVerifiedPayloads
                workers = int(os.environ.get('SPECO_PREFILL_NATIVE_VERIFICATION_WORKERS') or '2')
                budget = int(os.environ.get('SPECO_PREFILL_NATIVE_PAYLOAD_PREFETCH_BYTES') or '0')
                if not 0 < budget <= 8 * 1024 ** 3:
                    raise ValueError('parallel follower requires explicit bounded CPU payload prefetch')

                def initialize():
                    device = getattr(reader, 'device', None)
                    if device is not None and device.type == 'npu':
                        import torch
                        torch.npu.set_device(device)
                prefetch = ParallelVerifiedPayloads(expected, copies=reader.copy_lanes(partition), verify=reader._verify_cpu_packet, max_bytes=budget - (workers + 1) * 4096, initialize=initialize, verification_workers=workers)
            try:
                for (key, metadata), checksum in zip(expected, value['payload_checksums']):
                    fields, digest = prefetch.get(key) if prefetch is not None else reader.get_verified_cpu(key, partition)
                    if digest != checksum or any((t.device.type != 'cpu' for t in fields.values())):
                        raise RuntimeError('follower native checksum or CPU capture differs')
                    copies.add(fields, metadata)
                    del fields
            finally:
                if prefetch is not None:
                    prefetch.close()
            session.acknowledge(value)
            session.wait(lambda state: state['next_wave'] > wave_index)
            digests.extend(value['payload_checksums'])
            lease_ids.extend(value['lease_ids'])
    except BaseException:
        try:
            copies.close()
            for builder in builders:
                if callable(getattr(builder, 'close', None)):
                    builder.close()
        finally:
            try:
                session.abort()
            except Exception:
                pass
        raise
    finally:
        copies.close()
    if len(lease_ids) != len(samples) or len(set(lease_ids)) != len(lease_ids):
        raise RuntimeError('follower did not stage one complete global lease set')
    return dict(payload_checksums=digests, lease_ids=lease_ids)
