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
