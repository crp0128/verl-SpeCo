# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import asyncio
from copy import deepcopy
from verl_speco.integration.multinode_online import MultiNode92BActorExecutor as MultiNode92BActorExecutor
from verl_speco.integration.multinode_online import OnlineCollectionCoordinator as AsyncMultiNodeCollectionCoordinator
from verl_speco.integration.multinode_recovery import open_joint_parent, validate_joint_release
from verl_speco.integration.online_session import MultiNodeOnlinePostTrainingSession as MultiNodeOnlinePostTrainingSession
from verl_speco.integration.multinode_publication import MultiNodeTargetPublishExecutor as MultiNodeTargetPublishExecutor
from verl_speco.integration.multinode_online import OnlineVersions

async def attach_joint_online_session(*, coordinator, restored, resolve, vocab_size, producer_id, rollout_nodes, save_complete_sequence, abort_collection, admit_collection, abort_plan, schedule_config, parallel_updates=False, colocated_placements=None, phase_control=None):
    """Read actual current role counters; creating ports never applies updates.

    The coordinator supplies the already restored actor/draft/native owners.
    Failure after release requires its exact-owner cleanup port. This function
    cannot attach two independent collection locks to the same live generation.
    """
    async with coordinator.lock:
        if coordinator.phase != 'ready' or getattr(coordinator, '_online_session_attached', False):
            raise RuntimeError('joint session requires a released unattached recovery owner')
        try:
            if any((not callable(port) for port in (resolve, save_complete_sequence, abort_collection, admit_collection, abort_plan))):
                raise ValueError('joint session requires actual collection save, admission and abort ports')
            if restored.get('optimizer_effect_created') is not False or restored.get('proof') != coordinator.proof:
                raise RuntimeError('joint session has no exact verified recovery proof')
            if type(parallel_updates) is not bool or type(coordinator.config.get('parallel_actor_draft_updates', False)) is not bool or parallel_updates != coordinator.config.get('parallel_actor_draft_updates', False):
                raise ValueError('restored execution mode must match the fingerprinted parent recipe')
            if (colocated_placements is None) != (phase_control is None):
                raise ValueError('restored shared session needs both placement and phase control')
            if phase_control is not None and (phase_control is not getattr(coordinator, 'phase_control', None) or phase_control.phase != 'rollout' or phase_control.failed or (phase_control.binding != colocated_placements)):
                raise ValueError('restored shared session needs the actual fully resumed recovery phase')
            if coordinator.config.get('colocated') is True and phase_control is None:
                raise ValueError('restored shared session cannot discard its memory owner')
            parent, _ = open_joint_parent(coordinator.config, coordinator.config['joint_restore_binding'])
            versions = OnlineVersions(**restored['versions'])
            expected = dict(parent['versions'], generation=coordinator.config['generation'])
            if restored['versions'] != expected:
                raise RuntimeError('joint session versions differ from the restored parent')
            validate_joint_release(parent, coordinator.proof, coordinator.owners, run_id=versions.run_id, generation=versions.generation, binding_checksum=coordinator.config['joint_restore_binding']['checksum'])
            actor = MultiNode92BActorExecutor(worker_group=coordinator.actor, expected_incarnations=coordinator.owners['actor'], run_id=versions.run_id, generation=versions.generation, vocab_size=vocab_size, optimizer_step=parent['actor_update_receipt']['optimizer_step'], rl_step=versions.rl_step)
            rows = await asyncio.to_thread(coordinator.actor.speco_actor_audit)
            if not isinstance(rows, list) or any((not isinstance(row, dict) or type(row.get('rank')) is not int or type(row.get('optimizer_step')) is not int for row in rows)):
                raise RuntimeError('joint live actor audit has invalid rank or counter types')
            actor._audit(rows, actor.optimizer_step)
            draft = resolve(await asyncio.to_thread(coordinator.draft.speco_online_capabilities))
            previous = {r['worker_id']: r for r in parent['draft_state_rows']}
            if not isinstance(draft, list) or len(draft) != len(previous) or any((not isinstance(row, dict) for row in draft)) or ({row.get('worker_id') for row in draft} != set(previous)) or any((row.get('optimizer_step') != previous[row['worker_id']]['optimizer_step'] or type(row.get('optimizer_step')) is not int or row.get('worker_incarnation') != coordinator.owners['draft'][row['worker_id']] or (row.get('run_id') != versions.run_id) or (row.get('generation') != versions.generation) for row in draft)):
                raise RuntimeError('joint draft live optimizer counters differ from the restored state')
            collection = AsyncMultiNodeCollectionCoordinator(versions=versions, collective_rpc=coordinator.rpc, expected_ranks=tuple(coordinator.owners['rollout']), producer_id=producer_id, save_complete_sequence=save_complete_sequence, abort_collection=abort_collection, admit_collection=admit_collection)
            publisher = MultiNodeTargetPublishExecutor(actor_group=coordinator.actor, actor_incarnations=coordinator.owners['actor'], draft_group=coordinator.draft, draft_incarnations=coordinator.owners['draft'], collective_rpc=coordinator.rpc, rollout_incarnations=coordinator.owners['rollout'], reset_prefix_cache=coordinator.reset_prefix_cache, tokenizer_fingerprint=parent['fingerprints']['tokenizer'], bucket_bytes=coordinator.bucket_bytes)
            session = MultiNodeOnlinePostTrainingSession(collection=collection, worker_group=coordinator.draft, resolve=resolve, trainer_worker_ids=tuple(coordinator.owners['draft']), rollout_nodes=set(rollout_nodes), rollout_incarnations=coordinator.owners['rollout'], reset_prefix_cache=coordinator.reset_prefix_cache, abort_plan=abort_plan, actor_update=actor, publish_actor=publisher, schedule_config=schedule_config, parallel_updates=parallel_updates, colocated_placements=colocated_placements, phase_control=phase_control)
            if session.worker_executor.incarnations != coordinator.owners['draft']:
                raise RuntimeError('joint draft ownership changed during session attachment')
            coordinator._online_session_attached = True
            return dict(session=session, collection=collection, actor_executor=actor, target_publisher=publisher, versions=versions)
        except BaseException:
            coordinator.phase = 'failed'
            await coordinator.abort_owned(deepcopy(coordinator.owners))
            raise
