# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Shared ready owner invariants before running it inside a Ray actor.'
from dataclasses import replace
from copy import deepcopy
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import pytest
from verl_speco.transport.control import RayReadyControl, actor_name_for_run, ready_control_owner_node_id

def test_ready_control_owner_follows_ray_address_and_fails_closed() -> None:
    nodes = [{'Alive': True, 'NodeManagerAddress': '172.16.43.102', 'NodeID': 'owner'}, {'Alive': True, 'NodeManagerAddress': '172.16.43.103', 'NodeID': 'trainer'}]
    assert ready_control_owner_node_id(nodes, '172.16.43.102:63507') == 'owner'
    with pytest.raises(RuntimeError, match='unavailable or ambiguous'):
        ready_control_owner_node_id(nodes, '172.16.43.104:63507')
    with pytest.raises(RuntimeError, match='unavailable or ambiguous'):
        ready_control_owner_node_id(nodes + [nodes[0]], '172.16.43.102:63507')
from verl_speco.transport.control import AcquireRequest, ReadyRecord
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key

def _record(producer: str, sequence: int) -> ReadyRecord:
    metadata = SampleMetadataV3(schema_version=3, run_id='run-control', generation=1, producer_id=producer, producer_incarnation='inc-1', request_id=f'request-{producer}-{sequence}', sample_id=f'sample-{producer}-{sequence}', sequence_no=sequence, target_revision='target-1', data_version=0, tokenizer_fingerprint='tokenizer-1', target_model_fingerprint='model-1', target_layer_ids=(1, 3), hidden_states_layout='token_major', hidden_dtype='bfloat16', token_count=4, payload_nbytes=128)
    return ReadyRecord(make_v3_sample_key(metadata), producer, metadata, 'a' * 64)

def test_shared_reservation_prevents_conflicts_and_enforces_global_quota() -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=256)
    first, second, third = (_record('p1', 1), _record('p2', 1), _record('p3', 1))
    reservation = owner.reserve(first, nbytes=128)
    assert reservation.status == 'reserved'
    assert owner.reserve(first, nbytes=128).status == 'busy'
    changed = replace(first, payload_checksum='b' * 64)
    assert owner.reserve(changed, nbytes=128).status == 'conflict'
    assert owner.publish(first, token=reservation.token) == 'published'
    assert owner.reserve(first, nbytes=128).status == 'duplicate'
    second_reservation = owner.reserve(second, nbytes=128)
    assert second_reservation.status == 'reserved'
    assert owner.reserve(third, nbytes=128).status == 'backpressure'
    assert owner.cancel(second.key, token=second_reservation.token)
    assert owner.reserve(third, nbytes=128).status == 'reserved'
    assert owner.status()['used_bytes'] == 256

def test_shared_acquire_release_and_commit_preserve_lease_state() -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=256)
    records = (_record('p1', 1), _record('p2', 1))
    for record in records:
        reservation = owner.reserve(record, nbytes=128)
        owner.publish(record, token=reservation.token)
    request = AcquireRequest('acquire-1', 'consumer', 1, 2, 2, 'target-1', time.time() + 60)
    result, leases = owner.acquire(request, lease_timeout_s=60)
    assert result.status == 'ready'
    assert owner.acquire(request, lease_timeout_s=60) == (result, leases)
    with pytest.raises(ValueError, match='different request'):
        owner.acquire(replace(request, consumer_id='another-consumer'), lease_timeout_s=60)
    assert {record.producer_endpoint for record in result.records} == {'p1', 'p2'}
    assert len(leases) == 2
    lease_ids = tuple((lease.lease_id for lease in leases))
    owner.mark_materialized(lease_ids)
    for lease in leases:
        owner.release(lease.key, lease_ids=(lease.lease_id,))
    assert owner.status()['used_bytes'] == 0
    receipt = owner.commit(optimizer_step_id='step-1', transaction_set_hash='hash-1', lease_ids=lease_ids)
    assert not receipt.already_committed
    assert owner.commit(optimizer_step_id='step-1', transaction_set_hash='hash-1', lease_ids=lease_ids).already_committed
    with pytest.raises(ValueError, match='different transactions'):
        owner.commit(optimizer_step_id='step-1', transaction_set_hash='other', lease_ids=lease_ids)

def test_wrong_run_is_rejected_and_actor_name_is_stable() -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=128)
    wrong = replace(_record('p1', 1).metadata, run_id='another-run')
    with pytest.raises(ValueError, match='run, generation or payload'):
        owner.reserve(ReadyRecord(make_v3_sample_key(wrong), 'p1', wrong, 'a' * 64), nbytes=128)
    assert actor_name_for_run('run-control', 1) == actor_name_for_run('run-control', 1)
    assert actor_name_for_run('run-control', 1) != actor_name_for_run('run-control', 2)

def test_old_generation_cannot_reserve_or_acquire() -> None:
    owner = RayReadyControl(run_id='run-control', generation=2, partition='samples', capacity_bytes=128)
    old = _record('p1', 1)
    with pytest.raises(ValueError, match='generation'):
        owner.reserve(old, nbytes=128)
    with pytest.raises(ValueError, match='generation'):
        owner.acquire(AcquireRequest('old', 'consumer', 1, 1, 1, 'target-1', time.time() + 60), lease_timeout_s=60)
    assert owner.status()['used_bytes'] == 0
    assert owner.status()['generation'] == 2

def test_out_of_order_sequences_keep_distinct_keys_and_commit_once() -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=256)
    later, earlier = (_record('p1', 2), _record('p1', 1))
    for record in (later, earlier):
        token = owner.reserve(record, nbytes=128).token
        assert owner.publish(record, token=token) == 'published'
        assert owner.reserve(record, nbytes=128).status == 'duplicate'
    result, leases = owner.acquire(AcquireRequest('acquire-out-of-order', 'consumer', 1, 2, 2, 'target-1', time.time() + 60), lease_timeout_s=60)
    assert result.status == 'ready'
    assert {record.key for record in result.records} == {later.key, earlier.key}
    lease_ids = tuple((lease.lease_id for lease in leases))
    owner.mark_materialized(lease_ids)
    for lease in leases:
        owner.release(lease.key, lease_ids=(lease.lease_id,))
    assert not owner.commit(optimizer_step_id='step-out-of-order', transaction_set_hash='hash', lease_ids=lease_ids).already_committed
    assert owner.commit(optimizer_step_id='step-out-of-order', transaction_set_hash='hash', lease_ids=lease_ids).already_committed
    assert owner.status()['used_bytes'] == 0

def test_expired_unmaterialized_lease_clears_tq_before_releasing_quota(monkeypatch) -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=128)
    record = _record('p1', 1)
    reservation = owner.reserve(record, nbytes=128)
    owner.publish(record, token=reservation.token)
    result, (lease,) = owner.acquire(AcquireRequest('expire-1', 'consumer', 1, 1, 1, 'target-1', time.time() + 60), lease_timeout_s=1)
    assert result.status == 'ready'
    payloads = {record.key: object()}
    fail_clear = [True]

    def clear(*, keys: str, partition_id: str) -> None:
        assert partition_id == 'samples'
        if fail_clear[0]:
            raise TimeoutError('injected TQ clear timeout')
        payloads.pop(keys)
    fake_tq = SimpleNamespace(init=lambda: None, kv_list=lambda *, partition_id: {partition_id: dict(payloads)}, kv_clear=clear)
    monkeypatch.setitem(sys.modules, 'transfer_queue', fake_tq)
    with pytest.raises(TimeoutError, match='injected'):
        owner.reap_expired(now=lease.expires_at + 1)
    assert owner.status()['used_bytes'] == 128
    assert owner.status()['reclaim_pending'] == 1
    fail_clear[0] = False
    assert owner.reap_expired(now=lease.expires_at + 2) == (record.key,)
    assert owner.status()['used_bytes'] == 0
    assert owner.status()['reclaim_pending'] == 0
    assert not payloads
    assert owner.reap_expired(now=lease.expires_at + 3) == ()
    with pytest.raises(ValueError, match='materialize'):
        owner.mark_materialized((lease.lease_id,))

def test_quiescent_snapshot_restores_tombstones_and_stable_acquire(tmp_path: Path) -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=128)
    record = _record('p1', 1)
    token = owner.reserve(record, nbytes=128).token
    owner.publish(record, token=token)
    request = AcquireRequest('acquire-snapshot', 'consumer', 1, 1, 1, 'target-1', time.time() + 60)
    result, (lease,) = owner.acquire(request, lease_timeout_s=60)
    with pytest.raises(RuntimeError, match='active or uncertain payloads'):
        owner.snapshot_quiescent()
    owner.mark_materialized((lease.lease_id,))
    owner.release(record.key, lease_ids=(lease.lease_id,))
    owner.commit(optimizer_step_id='step-snapshot', transaction_set_hash='hash-snapshot', lease_ids=(lease.lease_id,))
    path = tmp_path / 'control.json'
    digest = owner.save_quiescent(str(path))
    restored = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=128)
    restored.load_quiescent(str(path), expected_sha256=digest)
    retry_result, retry_leases = restored.acquire(request, lease_timeout_s=60)
    assert retry_result == result
    assert retry_leases[0].lease_id == lease.lease_id
    assert retry_leases[0].state.value == 'committed'
    assert restored.reserve(record, nbytes=128).status == 'stale'
    assert restored.status()['used_bytes'] == 0
    assert restored.status()['leases']['leases_committed'] == 1
    with pytest.raises(ValueError, match='checksum mismatch'):
        restored.load_quiescent(str(path), expected_sha256='0' * 64)
    wrong_generation = RayReadyControl(run_id='run-control', generation=2, partition='samples', capacity_bytes=128)
    with pytest.raises(ValueError, match='identity differs'):
        wrong_generation.load_quiescent(str(path), expected_sha256=digest)

def test_producer_incarnation_change_requires_new_generation(tmp_path: Path) -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=256)
    first = _record('p1', 1)
    token = owner.reserve(first, nbytes=128).token
    owner.publish(first, token=token)
    next_metadata = replace(_record('p1', 2).metadata, producer_incarnation='inc-2')
    changed = ReadyRecord(make_v3_sample_key(next_metadata), 'p1', next_metadata, 'a' * 64)
    with pytest.raises(ValueError, match='incarnation changed'):
        owner.reserve(changed, nbytes=128)
    assert owner.status()['used_bytes'] == 128
    request = AcquireRequest('acquire-inc', 'consumer', 1, 1, 1, 'target-1', time.time() + 60)
    _, (lease,) = owner.acquire(request, lease_timeout_s=60)
    owner.mark_materialized((lease.lease_id,))
    owner.release(first.key, lease_ids=(lease.lease_id,))
    owner.commit(optimizer_step_id='step-inc', transaction_set_hash='hash-inc', lease_ids=(lease.lease_id,))
    path = tmp_path / 'control.json'
    digest = owner.save_quiescent(str(path))
    restored = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=256)
    restored.load_quiescent(str(path), expected_sha256=digest)
    with pytest.raises(ValueError, match='incarnation changed'):
        restored.reserve(changed, nbytes=128)
    assert restored.status()['used_bytes'] == 0
    next_generation = replace(changed.metadata, generation=2)
    new_owner = RayReadyControl(run_id='run-control', generation=2, partition='samples', capacity_bytes=256)
    new_record = ReadyRecord(make_v3_sample_key(next_generation), 'p1', next_generation, 'a' * 64)
    assert new_owner.reserve(new_record, nbytes=128).status == 'reserved'

@pytest.mark.parametrize(('field', 'expected'), [('lease_generation', 'lease from another generation'), ('delivered_generation', 'delivered producer identity differs'), ('acquire_generation', 'acquire generation differs'), ('acquired_incarnation', 'acquired producer identity differs'), ('acquired_lease', 'acquired leases do not match records'), ('committed_payload', 'committed lease snapshot has unreleased payload'), ('committed_step_lease', 'committed step snapshot has inconsistent leases')])
def test_quiescent_restore_rejects_cross_generation_or_inconsistent_identity(field: str, expected: str) -> None:
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=128)
    record = _record('p1', 1)
    token = owner.reserve(record, nbytes=128).token
    owner.publish(record, token=token)
    request = AcquireRequest('acquire-corrupt', 'consumer', 1, 1, 1, 'target-1', time.time() + 60)
    _, (lease,) = owner.acquire(request, lease_timeout_s=60)
    owner.mark_materialized((lease.lease_id,))
    owner.release(record.key, lease_ids=(lease.lease_id,))
    owner.commit(optimizer_step_id='step-corrupt', transaction_set_hash='hash-corrupt', lease_ids=(lease.lease_id,))
    snapshot = deepcopy(owner.snapshot_quiescent())
    if field == 'lease_generation':
        snapshot['leases']['leases'][0]['generation'] = 2
    elif field == 'delivered_generation':
        snapshot['delivered'][0]['metadata']['generation'] = 2
    elif field == 'acquire_generation':
        snapshot['acquired'][0]['request']['generation'] = 2
    elif field == 'acquired_incarnation':
        snapshot['acquired'][0]['result']['records'][0]['metadata']['producer_incarnation'] = 'inc-2'
    elif field == 'acquired_lease':
        snapshot['leases']['leases'][0]['acquire_id'] = 'other-acquire'
    elif field == 'committed_payload':
        snapshot['leases']['leases'][0]['payload_released'] = False
    elif field == 'committed_step_lease':
        snapshot['leases']['committed_steps']['step-corrupt']['lease_ids'] = ['missing-lease']
    restored = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=128)
    with pytest.raises(ValueError, match=expected):
        restored.restore_quiescent(snapshot)
    assert restored.status()['leases']['leases_total'] == 0

def _published_owner_and_tq(monkeypatch):
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=512)
    records = (_record('p1', 1), _record('p2', 2))
    payloads = {}
    for record in records:
        token = owner.reserve(record, nbytes=128).token
        owner.publish(record, token=token)
        payloads[record.key] = object()
    fake = SimpleNamespace(init=lambda: None, kv_list=lambda *, partition_id: {partition_id: dict(payloads)}, kv_clear=lambda *, keys, partition_id: payloads.pop(keys, None))
    monkeypatch.setitem(sys.modules, 'transfer_queue', fake)
    return (owner, records, payloads, fake)

def test_failed_step_unacquired_cleanup_is_exact_idempotent_and_preserves_tombstones(monkeypatch):
    owner, records, payloads, _ = _published_owner_and_tq(monkeypatch)
    assert owner.discard_unacquired(records) == tuple((item.key for item in records))
    assert not payloads
    assert owner.status()['used_bytes'] == 0
    assert owner.status()['router']['ready_records'] == 0
    assert owner.discard_unacquired(records) == ()
    restored = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=512)
    restored.restore_quiescent(owner.snapshot_quiescent())
    assert restored.reserve(records[0], nbytes=128).status == 'stale'
    assert restored.status()['leases']['leases_total'] == 0

@pytest.mark.parametrize('mismatch', ['checksum', 'generation', 'endpoint'])
def test_unacquired_cleanup_rejects_entire_batch_before_mutation(monkeypatch, mismatch):
    owner, records, payloads, _ = _published_owner_and_tq(monkeypatch)
    changed = replace(records[1], payload_checksum='b' * 64) if mismatch == 'checksum' else replace(records[1], producer_endpoint='foreign') if mismatch == 'endpoint' else ReadyRecord(make_v3_sample_key(replace(records[1].metadata, generation=2)), records[1].producer_endpoint, replace(records[1].metadata, generation=2), records[1].payload_checksum)
    with pytest.raises(ValueError):
        owner.discard_unacquired((records[0], changed))
    assert len(payloads) == 2
    assert owner.status()['used_bytes'] == 256
    assert owner.status()['router']['ready_records'] == 2

@pytest.mark.parametrize('committed', [False, True])
def test_unacquired_cleanup_never_discards_leased_or_committed_work(monkeypatch, committed):
    owner, records, payloads, _ = _published_owner_and_tq(monkeypatch)
    _, leases = owner.acquire(AcquireRequest('held', 'trainer', 1, 1, 1, 'target-1', time.time() + 60), lease_timeout_s=60)
    if committed:
        owner.mark_materialized((leases[0].lease_id,))
        owner.release(leases[0].key, lease_ids=(leases[0].lease_id,))
        owner.commit(optimizer_step_id='step', transaction_set_hash='hash', lease_ids=(leases[0].lease_id,))
    before = owner.status()
    with pytest.raises(ValueError, match='acquired'):
        owner.discard_unacquired(records)
    assert owner.status() == before
    assert len(payloads) == 2

def test_unacquired_cleanup_uncertain_transport_keeps_quota_and_retry_state(monkeypatch):
    owner, records, payloads, fake = _published_owner_and_tq(monkeypatch)

    def clear(*, keys, partition_id):
        if keys == records[1].key:
            raise TimeoutError('uncertain clear')
        payloads.pop(keys, None)
    fake.kv_clear = clear
    with pytest.raises(TimeoutError):
        owner.discard_unacquired(records)
    assert owner.status()['used_bytes'] == 256
    assert owner.status()['router']['ready_records'] == 2
    with pytest.raises(RuntimeError, match='active or uncertain'):
        owner.snapshot_quiescent()
    fake.kv_clear = lambda *, keys, partition_id: payloads.pop(keys, None)
    assert len(owner.discard_unacquired(records)) == 2
    assert owner.status()['used_bytes'] == 0
