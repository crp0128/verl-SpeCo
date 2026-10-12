# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Behavioral checks for the protocol-v3 control plane.'
import time
from dataclasses import replace
import pytest
from verl_speco.transport.control import LeaseRegistry, LeaseState
from verl_speco.transport.control import AcquireRequest, AcquireResult, ReadyRecord, ReadyRouter
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_ready_tag, make_v3_sample_key, parse_v3_ready_tag, stable_acquire_id
from verl_speco.transport.streaming import TrainingStreamCoordinator, TrainingTransaction

def _record(producer: str, sequence: int, *, generation: int=1, revision: str='t1') -> ReadyRecord:
    metadata = SampleMetadataV3(schema_version=3, run_id='run1', generation=generation, producer_id=producer, producer_incarnation='inc1', request_id=f'request-{producer}-{sequence}', sample_id=f'sample-{producer}-{sequence}', sequence_no=sequence, target_revision=revision, data_version=0, tokenizer_fingerprint='tok1', target_model_fingerprint='model1', target_layer_ids=(3, 5), hidden_states_layout='token_major', hidden_dtype='bfloat16', token_count=4, payload_nbytes=128)
    return ReadyRecord(make_v3_sample_key(metadata), producer, metadata, 'a' * 64)

def _request(acquire_id: str, *, min_items: int=1, max_items: int=1) -> AcquireRequest:
    return AcquireRequest(acquire_id, 'consumer', 1, max_items, min_items, 't1', time.time() + 60)

def test_v3_tag_roundtrip_and_version_rejection() -> None:
    metadata = _record('p', 1).metadata
    tag = make_v3_ready_tag(metadata, payload_checksum='b' * 64)
    assert parse_v3_ready_tag(tag) == (metadata, 'b' * 64)
    assert parse_v3_ready_tag({**tag, 'schema_version': 2}) is None
    assert parse_v3_ready_tag({**tag, 'payload_checksum': 'invalid'}) is None
    assert parse_v3_ready_tag({key: value for key, value in tag.items() if key != 'request_id'}) is None
    assert stable_acquire_id(run_id='run1', generation=1, consumer_id='c', batch_sequence=2) == stable_acquire_id(run_id='run1', generation=1, consumer_id='c', batch_sequence=2)
    online = replace(metadata, rl_step=3, draft_revision='draft-2', plan_id='plan-7')
    assert parse_v3_ready_tag(make_v3_ready_tag(online, payload_checksum='b' * 64)) == (online, 'b' * 64)
    assert make_v3_sample_key(online) == make_v3_sample_key(metadata)
    with pytest.raises(ValueError, match='draft_revision'):
        replace(online, draft_revision='').validate()
    with pytest.raises(ValueError, match='plan_id'):
        replace(online, plan_id='').validate()
    mapped = replace(online, prompt_token_count=2, supervised_token_count=2)
    assert SampleMetadataV3.from_dict(mapped.to_dict()) == mapped
    with pytest.raises(ValueError, match='token mapping'):
        replace(mapped, supervised_token_count=1).validate()
    with pytest.raises(ValueError, match='supplied together'):
        replace(mapped, supervised_token_count=None).validate()

def test_router_retry_after_busy_is_live_and_acquire_is_stable() -> None:
    router = ReadyRouter()
    request = _request('acq1', min_items=2, max_items=2)
    assert router.acquire(request).status == 'busy'
    first, second = (_record('p1', 1), _record('p2', 1))
    assert router.publish(first) == 'published'
    assert router.acquire(request).status == 'busy'
    assert router.publish(second) == 'published'
    result = router.acquire(request)
    assert result.status == 'ready'
    assert {item.key for item in result.records} == {first.key, second.key}
    assert router.acquire(request) == result
    router.ack(request.acquire_id)
    assert router.publish(first) == 'duplicate'
    assert router.publish(replace(first, metadata=replace(first.metadata, data_version=1))) == 'conflict'
    assert router.acquire(request).status == 'acknowledged'

def test_router_filters_generation_and_revision_and_cancel_requeues() -> None:
    router = ReadyRouter()
    old = _record('p1', 1, generation=0)
    stale = _record('p1', 2, revision='t0')
    current = _record('p2', 3)
    for record in (old, stale, current):
        assert router.publish(record) == 'published'
    request = _request('acq1')
    result = router.acquire(request)
    assert result.records == (current,)
    router.cancel(request.acquire_id)
    assert router.acquire(_request('acq2')).records == (current,)

def test_router_round_robin_across_producers() -> None:
    router = ReadyRouter()
    for sequence in (1, 2):
        for producer in ('p1', 'p2'):
            router.publish(_record(producer, sequence))
    result = router.acquire(_request('acq-fair', min_items=4, max_items=4))
    assert [record.producer_endpoint for record in result.records] == ['p1', 'p2', 'p1', 'p2']

def test_lease_register_conflict_is_atomic_and_lifecycle() -> None:
    registry = LeaseRegistry()
    first, second = (_record('p', 1), _record('p', 2))
    initial = AcquireResult('ready', 'acq-old', (second,))
    old_lease = registry.register(initial, consumer_id='c', lease_timeout_s=10, now=100)[0]
    conflicted = AcquireResult('ready', 'acq-new', (first, second))
    with pytest.raises(ValueError, match='already belongs'):
        registry.register(conflicted, consumer_id='c', lease_timeout_s=10, now=100)
    assert registry.status()['leases_total'] == 1
    assert registry.get(old_lease.lease_id).state is LeaseState.LEASED
    lease = registry.register(AcquireResult('ready', 'acq-first', (first,)), consumer_id='c', lease_timeout_s=10, now=100)[0]
    assert registry.register(AcquireResult('ready', 'acq-first', (first,)), consumer_id='c', lease_timeout_s=10, now=100) == (lease,)
    registry.mark_materialized((lease.lease_id,))
    with pytest.raises(ValueError, match='payload-released'):
        registry.commit(optimizer_step_id='step1', transaction_set_hash='hash', lease_ids=(lease.lease_id,))
    registry.mark_payload_released((lease.lease_id,))
    assert not registry.commit(optimizer_step_id='step1', transaction_set_hash='hash', lease_ids=(lease.lease_id,)).already_committed
    assert registry.commit(optimizer_step_id='step1', transaction_set_hash='hash', lease_ids=(lease.lease_id,)).already_committed
    with pytest.raises(ValueError, match='different transactions'):
        registry.commit(optimizer_step_id='step1', transaction_set_hash='other', lease_ids=(lease.lease_id,))
    assert registry.expire(now=111) == (registry.get(old_lease.lease_id),)
    assert registry.get(old_lease.lease_id).state is LeaseState.CANCELLED

def test_stream_rejects_duplicate_transaction_identity() -> None:
    stream = TrainingStreamCoordinator('scope')
    first = TrainingTransaction('tx1', ('l1',), ('k1',), 'scope', 1, 1, 't1')
    duplicate = TrainingTransaction('tx1', ('l2',), ('k2',), 'scope', 2, 1, 't1')
    stream.register(first)
    with pytest.raises(ValueError, match='another cursor'):
        stream.register(duplicate)
    assert stream.commit('step1', ('tx1',)) is False
    assert stream.commit('step1', ('tx1',)) is True
    assert stream.snapshot().committed_cursor == 1
