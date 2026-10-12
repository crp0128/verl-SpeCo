# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'WorkerGroup fencing tests; no accelerator capability is claimed by fixtures.'
from copy import deepcopy
from dataclasses import replace
import pytest
from verl_speco.integration.multinode_online import MultiNodeOnlineWorkerGroupExecutor as MultiNodeOnlineWorkerGroupExecutor
from verl_speco.integration.multinode_online import OnlineCollectionCommit, OnlineVersions, MultiNodeOnlineAdapter as MultiNodeVerlOnlineAdapter
from verl_speco.trainer.scheduler.drafter_scheduler import DrafterScheduler
from verl_speco.trainer.scheduler.publish_executor import CallbackDrafterPublishExecutor
from verl_speco.trainer.scheduler.schedule_types import DrafterScheduleConfig

class Group:

    def __init__(self):
        self.payload = None
        self.events = []
        self.capabilities = [dict(worker_id=str(rank), worker_incarnation=f'rank-{rank}', run_id='run', generation=3, engine='veomni', wrapper='fsdp2', initialized=True, online_fencing=True, hidden_transport='native_tq', node_id='trainer', mesh=dict(dp_replicate=2, dp_shard=1, tp=1, pp=1, sp=1, ep=1)) for rank in range(2)]

    def rows(self, **extra):
        return [dict(worker_id=str(rank), worker_incarnation=f'rank-{rank}', run_id='run', generation=3, online_plan_sha256=self.payload['online_plan_sha256'] if self.payload else None, **extra) for rank in range(2)]

    def speco_online_capabilities(self):
        return deepcopy(self.capabilities)

    def speco_bind_online_plan(self, payload):
        self.payload = deepcopy(payload)
        self.events.append('bind')
        return self.rows(bound=True)

    def get_drafter_training_data_status(self, **identity):
        assert identity['run_id'] == 'run' and identity['generation'] == 3
        return self.rows(available=True, current_step=8, current_step_samples=2, buffer_samples=2, trainable_samples=2, trainable_batches=1, batch_size_per_gpu=1, partial_batch_available=False, oldest_sample_step=8, newest_sample_step=8, same_step_data_required=True, target_version=102, data_version=41)

    def speco_prepare_online_target(self, payload):
        assert payload == self.payload
        self.events.append('prepare')
        return self.rows(prepared=True, target_version=102)

    def speco_preflight_drafter_training(self, payload):
        assert payload == self.payload
        self.events.append('preflight')
        return self.rows(participating=True, ready=True, data_version=41, target_version=102)

    def train_drafter(self, payload):
        assert payload == self.payload
        self.events.append('train')
        return [dict(row, trained=True, triggered=True, successful_steps=1, attempted_steps=1, optimizer_step=9, source_global_step=8, data_version=41, target_version=102, plan_id=payload['plan_id'], is_publish_leader=row['worker_id'] == '0', publish_snapshot_cached=int(row['worker_id'] == '0')) for row in self.rows()]

    def speco_abort_drafter_training_preflight(self, payload):
        assert payload == self.payload
        self.events.append('abort_preflight')
        return self.rows()

def executor(group):
    return MultiNodeOnlineWorkerGroupExecutor(worker_group=group, resolve=lambda value: value, run_id='run', generation=3, expected_worker_ids=('0', '1'), rollout_nodes=frozenset({'target'}))

def prepared():
    group = Group()
    port = executor(group)
    scheduler = DrafterScheduler(worker_executor=port, publish_executor=CallbackDrafterPublishExecutor(wait=lambda: 0, fetch=lambda: dict(snapshot='sha'), update=lambda *_: None, normalize_payload=lambda value: value))
    versions = OnlineVersions('run', 3, 7, 102, 40, 5)
    adapter = MultiNodeVerlOnlineAdapter(scheduler=scheduler, versions=versions, producer_ids=('p0',), abort_plan=lambda *_: group.events.append('abort'), publish_commit=lambda *_: None, resume_rollout=lambda: group.events.append('resume'), bind_worker_plan=port.bind_online_plan)
    adapter.on_rollout_complete(OnlineCollectionCommit(replace(versions, rl_step=8, data_version=41), 'collection', ('p0',), ('tq/key',)))
    binding = adapter.before_actor_update(config=DrafterScheduleConfig())
    return (adapter, group, port, binding)

def test_real_scheduler_uses_group_fenced_payload_for_prepare_preflight_and_train():
    adapter, group, port, binding = prepared()
    adapter.after_actor_update(rl_step=8)
    adapter.at_rollout_safe_point()
    assert group.events == ['bind', 'prepare', 'preflight', 'train', 'resume']
    assert group.payload['online_versions'] == dict(run_id='run', generation=3, rl_step=8, target_revision=102, data_version=41, draft_revision=5, collection_id='collection', plan_id=binding.plan.plan_id)
    assert group.payload['selected_keys'] == ['tq/key']
    assert len(group.payload['online_plan_sha256']) == 64
    assert adapter.versions.draft_revision == 6

@pytest.mark.parametrize('field,value', [('engine', 'fsdp'), ('wrapper', 'ddp'), ('initialized', False), ('online_fencing', False), ('hidden_transport', 'ray_object_store'), ('hidden_transport', 'http'), ('hidden_transport', 'hidden_files'), ('run_id', 'foreign'), ('generation', 4), ('node_id', 'target'), ('worker_incarnation', '')])
def test_incompatible_or_uninitialized_group_is_rejected_before_binding(field, value):
    group = Group()
    group.capabilities[0][field] = value
    with pytest.raises(ValueError):
        executor(group)
    assert group.payload is None

@pytest.mark.parametrize('field,value', [('dp_shard', 2), ('dp_replicate', 1), ('tp', 2), ('pp', 2), ('sp', 2), ('ep', 2), ('tp', True)])
def test_unsupported_draft_parallelism_is_rejected(field, value):
    group = Group()
    group.capabilities[0]['mesh'][field] = value
    with pytest.raises(ValueError, match='DP-only'):
        executor(group)

def test_duplicate_rank_cannot_hide_a_missing_worker():
    group = Group()
    group.capabilities[1] = deepcopy(group.capabilities[0])
    with pytest.raises(RuntimeError, match='ranks'):
        executor(group)

@pytest.mark.parametrize('field,value', [('generation', 4), ('worker_incarnation', 'restarted'), ('online_plan_sha256', 'foreign')])
def test_stale_preflight_never_submits_training(field, value):
    adapter, group, port, binding = prepared()
    original = group.speco_preflight_drafter_training

    def bad(payload):
        rows = original(payload)
        rows[0][field] = value
        return rows
    group.speco_preflight_drafter_training = bad
    with pytest.raises(RuntimeError):
        adapter.after_actor_update(rl_step=8)
    assert 'train' not in group.events and 'resume' not in group.events
    assert 'abort_preflight' in group.events and adapter.versions.draft_revision == 5

def test_mutating_a_bound_plan_is_rejected_before_worker_side_effects():
    adapter, group, port, binding = prepared()
    binding.plan.worker_snapshots['0']['worker_incarnation'] = 'changed'
    with pytest.raises(RuntimeError, match='mutated'):
        port.submit_training(binding.plan)
    assert 'train' not in group.events

def test_foreign_training_receipt_is_not_accepted_as_committed():
    adapter, group, port, binding = prepared()
    rows = group.train_drafter(group.payload)
    rows[0]['run_id'] = 'foreign'
    with pytest.raises(RuntimeError, match='run/generation'):
        port.resolve_training(rows)

def test_legacy_speco_worker_group_cannot_silently_select_old_trainer():
    with pytest.raises(AttributeError):
        executor(object())
