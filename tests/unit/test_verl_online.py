# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Real scheduler decisions with deliberately different online version clocks.'
from dataclasses import replace
from unittest.mock import Mock
import pytest
from verl_speco.integration.multinode_online import OnlineCollectionCommit, OnlinePhase, OnlineVersions, MultiNodeOnlineAdapter as MultiNodeVerlOnlineAdapter
from verl_speco.trainer.scheduler.drafter_scheduler import DrafterScheduler
from verl_speco.trainer.scheduler.publish_executor import CallbackDrafterPublishExecutor
from verl_speco.trainer.scheduler.schedule_types import DrafterScheduleConfig, TrainingDataStatus
from verl_speco.trainer.scheduler.worker_executor import CallbackDrafterWorkerExecutor

def _fixture(*, training_interval=1, publish_interval=1):
    events = []
    versions = OnlineVersions('pangu-test', 3, 7, 102, 40, 5)
    commit = OnlineCollectionCommit(replace(versions, rl_step=8, data_version=41), 'collection-41', ('p0', 'p1'), ('tq/a', 'tq/b'))
    binding = {}
    knobs = {'bad_preflight': False, 'bad_result': False, 'changed_incarnation': False, 'no_snapshot': False}

    def inspect(*_):
        return [dict(available=True, current_step=8, current_step_samples=2, buffer_samples=2, trainable_samples=2, trainable_batches=1, batch_size_per_gpu=1, partial_batch_available=False, oldest_sample_step=8, newest_sample_step=8, same_step_data_required=True, target_version=102, data_version=41, worker_id=str(rank), worker_incarnation=f'rank-{rank}-original', buffer_version=9) for rank in range(2)]

    def bind(plan):
        events.append('bind')
        binding['plan'] = plan

    def prepare(plan):
        events.append('prepare')
        assert binding['plan'].plan == plan
        assert plan.required_target_version == 102
        return {}

    def preflight(payload):
        events.append('preflight')
        return [dict(participating=True, ready=not knobs['bad_preflight'], worker_id=str(rank), data_version=payload['data_version'], target_version=payload['required_target_version']) for rank in range(2)]

    def submit(payload):
        events.append('train')
        assert binding['plan'].to_worker_payload()['online_versions'] == dict(run_id='pangu-test', generation=3, rl_step=8, target_revision=102, data_version=41, draft_revision=5, collection_id='collection-41', plan_id=payload['plan_id'])
        return [dict(trained=True, triggered=True, successful_steps=1, attempted_steps=1, optimizer_step=1 + int(knobs['bad_result'] and rank == 1), worker_id=str(rank), worker_incarnation=f"rank-{rank}-{('changed' if knobs['changed_incarnation'] else 'original')}", source_global_step=8, target_version=102, data_version=41, plan_id=payload['plan_id'], is_publish_leader=rank == 0, publish_snapshot_cached=int(rank == 0)) for rank in range(2)]
    worker = CallbackDrafterWorkerExecutor(submit=submit, resolve=lambda value: value, inspect_data=inspect, prepare=prepare, activate=lambda: [], preflight=preflight, abort_preflight=lambda plan_id: events.append('abort_preflight'))
    publisher = CallbackDrafterPublishExecutor(wait=lambda: 0, fetch=lambda: None if knobs['no_snapshot'] else {'snapshot': 'sha'}, update=lambda payload, step, asynchronous: events.append('publish'), normalize_payload=lambda value: value)
    scheduler = DrafterScheduler(worker_executor=worker, publish_executor=publisher)
    aborted = Mock(side_effect=lambda plan, reason: events.append('abort'))
    verify = Mock(side_effect=lambda plan, revision: events.append('verify_acks'))
    adapter = MultiNodeVerlOnlineAdapter(scheduler=scheduler, versions=versions, producer_ids=('p0', 'p1'), abort_plan=aborted, publish_commit=verify, resume_rollout=lambda: events.append('resume'), bind_worker_plan=bind)
    config = DrafterScheduleConfig(training_interval_steps=training_interval, publish_interval_steps=publish_interval)
    return (adapter, commit, config, events, knobs, aborted, verify)

def _prepare(fixture):
    adapter, commit, config, *_ = fixture
    adapter.on_rollout_complete(commit)
    return adapter.before_actor_update(config=config)

def test_real_scheduler_keeps_all_online_clocks_separate_and_orders_events():
    fixture = _fixture()
    adapter, commit, config, events, _, abort, verify = fixture
    binding = _prepare(fixture)
    assert binding.plan.source_global_step == 8
    assert binding.plan.required_target_version == 102
    assert binding.plan.data_version == 41
    assert binding.versions.draft_revision == 5 and binding.versions.generation == 3
    assert adapter.versions.rl_step == 7 and adapter.versions.data_version == 41
    adapter.after_actor_update(rl_step=8)
    assert adapter.versions.rl_step == 8 and adapter.versions.draft_revision == 5
    adapter.at_rollout_safe_point()
    assert adapter.phase is OnlinePhase.IDLE and adapter.versions.draft_revision == 6
    assert events == ['bind', 'prepare', 'preflight', 'train', 'publish', 'verify_acks', 'resume']
    verify.assert_called_once_with(binding, 6)
    abort.assert_not_called()

@pytest.mark.parametrize('field,value', [('run_id', 'other'), ('generation', 4), ('rl_step', 9), ('target_revision', 8), ('data_version', 8), ('draft_revision', 6)])
def test_foreign_or_stale_collection_rejected_before_worker_rpc(field, value):
    adapter, commit, _, events, *_ = _fixture()
    with pytest.raises(ValueError, match='identities'):
        adapter.on_rollout_complete(replace(commit, versions=replace(commit.versions, **{field: value})))
    assert events == [] and adapter.phase is OnlinePhase.IDLE

def test_partial_producer_commit_is_rejected():
    adapter, commit, _, events, *_ = _fixture()
    with pytest.raises(ValueError, match='producers'):
        adapter.on_rollout_complete(replace(commit, producer_ids=('p0',)))
    assert not events

@pytest.mark.parametrize('field', ['producer_ids', 'tq_keys'])
def test_duplicate_collection_identifiers_are_rejected(field):
    _, commit, *_ = _fixture()
    with pytest.raises(ValueError, match='duplicate'):
        replace(commit, **{field: ('same', 'same')})

@pytest.mark.parametrize('knob', ['bad_preflight', 'bad_result', 'changed_incarnation'])
def test_failed_worker_group_never_reaches_publish_or_resume(knob):
    fixture = _fixture()
    adapter, _, _, events, knobs, abort, verify = fixture
    _prepare(fixture)
    knobs[knob] = True
    with pytest.raises(RuntimeError):
        adapter.after_actor_update(rl_step=8)
    assert adapter.phase is OnlinePhase.ABORTED
    assert adapter.versions.rl_step == 8 and adapter.versions.draft_revision == 5
    assert 'publish' not in events and 'resume' not in events
    abort.assert_called_once()
    verify.assert_not_called()

def test_failed_replica_ack_does_not_advance_revision_or_resume():
    fixture = _fixture()
    adapter, _, _, events, _, abort, verify = fixture
    _prepare(fixture)
    adapter.after_actor_update(rl_step=8)
    verify.side_effect = RuntimeError('missing replica acknowledgement')
    with pytest.raises(RuntimeError, match='acknowledgement'):
        adapter.at_rollout_safe_point()
    assert adapter.versions.draft_revision == 5 and 'resume' not in events
    assert adapter.phase is OnlinePhase.ABORTED
    abort.assert_called_once()

def test_missing_publish_snapshot_aborts_before_revision_commit():
    fixture = _fixture()
    adapter, _, _, events, knobs, abort, verify = fixture
    _prepare(fixture)
    adapter.after_actor_update(rl_step=8)
    knobs['no_snapshot'] = True
    with pytest.raises(RuntimeError, match='snapshot'):
        adapter.at_rollout_safe_point()
    assert adapter.versions.draft_revision == 5 and 'publish' not in events and ('resume' not in events)
    verify.assert_not_called()
    abort.assert_called_once()

def test_actor_failure_preserves_committed_data_and_blocks_future_steps():
    fixture = _fixture()
    adapter, commit, _, events, _, abort, _ = fixture
    _prepare(fixture)
    adapter.on_role_failure('critic', 'optimizer failed')
    assert adapter.versions.rl_step == 7 and adapter.versions.data_version == 41
    assert adapter.versions.draft_revision == 5 and 'train' not in events
    adapter.on_role_failure('actor', 'another failure')
    abort.assert_called_once()
    with pytest.raises(RuntimeError, match='aborted'):
        adapter.on_rollout_complete(commit)

def test_wrong_actor_step_and_early_safe_point_cannot_train_or_publish():
    fixture = _fixture()
    adapter, _, _, events, *_ = fixture
    _prepare(fixture)
    with pytest.raises(ValueError, match='RL step'):
        adapter.after_actor_update(rl_step=102)
    with pytest.raises(RuntimeError, match='trained'):
        adapter.at_rollout_safe_point()
    assert 'train' not in events and 'publish' not in events

def test_scheduler_training_interval_is_preserved():
    fixture = _fixture(training_interval=3)
    adapter, _, _, events, _, abort, verify = fixture
    binding = _prepare(fixture)
    assert not binding.plan.launch
    adapter.after_actor_update(rl_step=8)
    adapter.at_rollout_safe_point()
    assert events == ['bind'] and adapter.versions.draft_revision == 5
    assert adapter.phase is OnlinePhase.IDLE
    abort.assert_not_called()
    verify.assert_not_called()

def test_scheduler_publish_interval_is_preserved():
    fixture = _fixture(publish_interval=3)
    adapter, _, _, events, _, abort, verify = fixture
    _prepare(fixture)
    adapter.after_actor_update(rl_step=8)
    adapter.at_rollout_safe_point()
    assert 'train' in events and 'publish' not in events and (adapter.versions.draft_revision == 5)
    abort.assert_not_called()
    verify.assert_not_called()

def test_bound_payload_does_not_expose_mutable_worker_snapshots():
    binding = _prepare(_fixture())
    payload = binding.to_worker_payload()
    payload['worker_snapshots']['0']['worker_incarnation'] = 'corrupted'
    assert binding.plan.worker_snapshots['0']['worker_incarnation'] == 'rank-0-original'
    assert binding.to_worker_payload()['selected_keys'] == ['tq/a', 'tq/b']

@pytest.mark.parametrize('name,value', [('generation', 0), ('rl_step', True), ('data_version', -1), ('draft_revision', '5')])
def test_version_counters_reject_invalid_types_and_values(name, value):
    fixture = _fixture()
    versions = fixture[0].versions
    with pytest.raises(ValueError):
        replace(versions, **{name: value})

def test_explicit_worker_data_revision_is_not_replaced_by_sample_step():
    adapter, *_ = _fixture()
    status = adapter.scheduler.inspect_training_data(global_step=8, config=DrafterScheduleConfig())
    assert status.current_step == 8 and status.newest_sample_step == 8
    assert status.data_version == 41 and status.target_version == 102
