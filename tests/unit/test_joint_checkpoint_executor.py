# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Exercise actual metadata closure; role tensor save/audit use protocol fixtures.'
import asyncio
from copy import deepcopy
from types import SimpleNamespace
import pytest
from verl_speco.integration.multinode_recovery import MultiNodeJointCheckpointExecutor as MultiNodeJointCheckpointExecutor
from verl_speco.integration.multinode_online import OnlinePhase, OnlineVersions
from _multinode_fixture_joint_checkpoint import setup_case

@pytest.mark.parametrize('fault', [None, 'pending', 'stale-step', 'draft-owner', 'actor-receipt', 'control-change', 'actor-state', 'marker', 'save'])
def test_checkpoint_retains_collection_lock_and_aborts_uncertain_exact_owners(tmp_path, fault):
    store, args = setup_case(tmp_path)
    versions = OnlineVersions(**args['versions'])
    collection = SimpleNamespace(lock=asyncio.Lock(), versions=versions, failed=False, pending_collection=None)
    session = SimpleNamespace(collection=collection, adapter=SimpleNamespace(phase=OnlinePhase.IDLE, versions=versions), publisher=SimpleNamespace(last_snapshot=args['snapshot'], last_acks=args['draft_acks']))
    target = SimpleNamespace(last_proof={'native_ACKs': args['target_acks'], 'immutable_head': args['target_head']})
    result = dict(versions=versions, actor=args['actor_update_receipt'], draft=SimpleNamespace(training_execution=SimpleNamespace(trained=True, successful_steps=1, raw_results=deepcopy(args['draft_rows']))))
    events = []

    def save(request):
        assert collection.lock.locked()
        assert request['optimizer_step'] == args['actor_update_receipt']['optimizer_step']
        events.append('save')
        if fault == 'save':
            raise RuntimeError('injected save')
        rows = deepcopy(args['actor_rows'])
        if fault == 'actor-state':
            rows[0]['optimizer_step'] += 1
        return rows

    def audit():
        assert collection.lock.locked()
        events.append('draft-audit')
        return deepcopy(args['draft_state_rows'])

    async def control(method, kw):
        assert collection.lock.locked() and method == 'snapshot_quiescent' and (kw == {})
        events.append('control')
        val = deepcopy(args['control_snapshot'])
        if fault == 'control-change' and events.count('control') == 2:
            val['generation'] += 1
        return val

    async def abort(owners):
        assert collection.lock.locked()
        assert owners == dict(actor=store.actors, draft=store.drafts, rollout=store.rollout)
        events.append('abort-exact-owners')
    if fault == 'marker':
        directory = store.root / 'joint-checkpoints/generation-1-step-1'
        directory.mkdir(parents=True)
    if fault == 'pending':
        collection.pending_collection = object()
    if fault == 'stale-step':
        result['versions'] = None
    if fault == 'draft-owner':
        result['draft'].training_execution.raw_results[0]['worker_incarnation'] = 'other'
    if fault == 'actor-receipt':
        result['actor'] = dict(result['actor'], generation=2)
    executor = MultiNodeJointCheckpointExecutor(store=store, session=session, actor_group=SimpleNamespace(speco_save_actor_checkpoint=save), draft_group=SimpleNamespace(speco_audit_joint_draft_state=audit), target_publisher=target, control_rpc=control, abort_owned=abort)

    async def run():
        if fault:
            with pytest.raises((RuntimeError, ValueError)):
                await executor.save(result)
            if fault in ('pending', 'stale-step', 'draft-owner', 'actor-receipt'):
                assert events == [] and executor.phase == 'idle' and (not collection.failed)
            else:
                assert events[-1] == 'abort-exact-owners' and executor.phase == 'failed' and collection.failed
                previous = list(events)
                with pytest.raises(RuntimeError):
                    await executor.save(result)
                assert events == previous
        else:
            binding = await executor.save(result)
            assert store.load(binding) == store.build(**args)
            assert executor.last_binding == binding and (not collection.lock.locked())
            previous = list(events)
            with pytest.raises(RuntimeError):
                await executor.save(result)
            assert events == previous
    asyncio.run(run())
