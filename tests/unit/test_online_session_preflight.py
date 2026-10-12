# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Actual session ordering: failed draft preflight never invokes actor mutation.'
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import pytest
from _multinode_fixture_online_worker_group import Group
from verl_speco.integration.online_session import MultiNodeOnlinePostTrainingSession as MultiNodeOnlinePostTrainingSession
from verl_speco.integration.multinode_online import OnlineCollectionCommit, OnlineVersions
from verl_speco.trainer.scheduler.schedule_types import DrafterScheduleConfig

@pytest.mark.parametrize('damage', [None, 'raise', 'not_ready', 'target', 'data', 'incarnation'])
def test_draft_inputs_are_preflighted_before_real_actor_port_is_called(damage):

    async def run():
        group = Group()
        original = group.speco_preflight_drafter_training

        def preflight(payload):
            rows = original(payload)
            if damage == 'raise':
                raise ValueError('bad SWA response geometry')
            if damage == 'not_ready':
                rows[0]['ready'] = False
            if damage == 'target':
                rows[0]['target_version'] = 103
            if damage == 'data':
                rows[0]['data_version'] = 42
            if damage == 'incarnation':
                rows[0]['worker_incarnation'] = 'restarted'
            return rows
        group.speco_preflight_drafter_training = preflight
        versions = OnlineVersions('run', 3, 7, 102, 40, 5)
        commit = OnlineCollectionCommit(replace(versions, rl_step=8, data_version=41), 'collection', ('p0',), ('tq/key',))
        collection = SimpleNamespace(versions=versions, pending_collection=commit, producer_id='p0', rpc=None, failed=False)

        async def actor(batch, binding):
            group.events.append('actor')
            raise RuntimeError('actor port reached after preflight')
        session = MultiNodeOnlinePostTrainingSession(collection=collection, worker_group=group, resolve=lambda x: x, trainer_worker_ids=('0', '1'), rollout_nodes={'target'}, rollout_incarnations={0: 'tp0'}, reset_prefix_cache=lambda: True, abort_plan=lambda *_: None, actor_update=actor, publish_actor=None, schedule_config=DrafterScheduleConfig())
        with pytest.raises((RuntimeError, ValueError)):
            await session.update(commit, None)
        assert collection.failed
        if damage is None:
            assert group.events == ['bind', 'prepare', 'preflight', 'actor']
        else:
            assert 'actor' not in group.events and 'train' not in group.events
    asyncio.run(run())
