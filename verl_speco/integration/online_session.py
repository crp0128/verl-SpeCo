# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import asyncio
from dataclasses import replace
from uuid import uuid4
from verl_speco.integration.multinode_online import async_online_await as _await
from verl_speco.integration.multinode_recovery import drain_owned_call
from verl_speco.integration.multinode_publication import require_draft_update_acks
from verl_speco.integration.multinode_publication import TargetWeightAck, commit_target_weight_acks
from verl_speco.integration.multinode_online import MultiNodeOnlineWorkerGroupExecutor as MultiNodeOnlineWorkerGroupExecutor
from verl_speco.integration.multinode_online import MultiNodeOnlineAdapter as MultiNodeVerlOnlineAdapter
from verl_speco.trainer.scheduler.drafter_scheduler import DrafterScheduler

class MultiNodeSnapshotPublishExecutor:
    """Synchronous scheduler port; native engine RPCs run on the owning loop."""

    def __init__(self, *, worker_group, resolve, collective_rpc, reset_prefix_cache, versions, expected_incarnations, loop):
        self.group = worker_group
        self.resolve = resolve
        self.rpc = collective_rpc
        self.reset_prefix_cache = reset_prefix_cache
        self.versions = versions
        self.incarnations = dict(expected_incarnations)
        self.loop = loop
        self.last_snapshot = None
        self.last_acks = None

    def wait_pending(self):
        return 0

    def fetch_snapshot(self):
        rows = self.resolve(self.group.get_drafter_snapshot())
        if not isinstance(rows, list):
            raise RuntimeError('draft snapshot requires all WorkerGroup receipts')
        leaders = [row for row in rows if row.get('snapshot') is not None]
        if len(leaders) != 1 or leaders[0].get('is_publish_leader') is not True:
            raise RuntimeError('draft snapshot is missing or has multiple publishing leaders')
        return leaders[0]['snapshot']

    def update_rollout_weights(self, payload, *, global_step, asynchronous):
        if asynchronous:
            raise ValueError('async rollout mode still requires fenced all-rank snapshot publication')
        future = asyncio.run_coroutine_threadsafe(self._publish(payload, global_step), self.loop)
        future.result()

    async def _publish(self, snapshot, global_step):
        versions = self.versions()
        if snapshot['draft_revision'] != versions.draft_revision + 1 or global_step != versions.rl_step:
            raise RuntimeError('draft publication does not match completed actor/draft step')
        operation_id = uuid4().hex
        settings = dict(run_id=versions.run_id, generation=versions.generation, operation_id=operation_id, expected_revision=versions.draft_revision, snapshot=snapshot)
        rows = await _await(self.rpc('speco_prepare_draft_update', settings))
        require_draft_update_acks(rows, expected_incarnations=self.incarnations, run_id=versions.run_id, generation=versions.generation, operation_id=operation_id, snapshot=snapshot, phase='prepared')
        if await _await(self.reset_prefix_cache()) is not True:
            raise RuntimeError('native engine refused prefix-cache invalidation at publication safe point')
        rows = await _await(self.rpc('speco_commit_draft_update', dict(operation_id=operation_id)))
        require_draft_update_acks(rows, expected_incarnations=self.incarnations, run_id=versions.run_id, generation=versions.generation, operation_id=operation_id, snapshot=snapshot, phase='applied')
        self.last_snapshot = dict(snapshot)
        self.last_acks = rows
        released = await _await(self.rpc('speco_finish_draft_update', dict(operation_id=operation_id, snapshot_checksum=snapshot['checksum'])))
        require_draft_update_acks(released, expected_incarnations=self.incarnations, run_id=versions.run_id, generation=versions.generation, operation_id=operation_id, snapshot=snapshot, phase='released')

    def require_commit(self, binding, revision):
        if self.last_snapshot is None or not self.last_acks or self.last_snapshot['draft_revision'] != revision or any((row['generation'] != binding.versions.generation or row['run_id'] != binding.versions.run_id for row in self.last_acks)):
            raise RuntimeError('draft revision cannot advance without actual all-rank snapshot ACKs')

class MultiNodeOnlinePostTrainingSession:

    def __init__(self, *, collection, worker_group, resolve, trainer_worker_ids, rollout_nodes, rollout_incarnations, reset_prefix_cache, abort_plan, actor_update, publish_actor, schedule_config, colocated_placements=None, phase_control=None, parallel_updates=False):
        if schedule_config.publish_async:
            raise ValueError('default async rollout support uses synchronous safe-point weight publication')
        self.collection = collection
        self.actor_update = actor_update
        self.publish_actor = publish_actor
        self.schedule_config = schedule_config
        if (colocated_placements is None) != (phase_control is None):
            raise ValueError('shared-card session requires both validated placement and real phase control')
        if phase_control is not None and phase_control.binding != colocated_placements:
            raise ValueError('shared-card session and phase controller have different physical owners')
        if phase_control is not None and any((not callable(getattr(phase_control, name, None)) for name in ('enter_training', 'retire_training_role', 'enter_publication', 'finish_publication', 'mark_uncertain'))):
            raise ValueError('shared-card session requires complete owned memory/recovery phase ports')
        self.phase_control = phase_control
        self.worker_executor = MultiNodeOnlineWorkerGroupExecutor(worker_group=worker_group, resolve=resolve, run_id=collection.versions.run_id, generation=collection.versions.generation, expected_worker_ids=tuple(trainer_worker_ids), rollout_nodes=frozenset(rollout_nodes), colocated_placements=colocated_placements)
        self.publisher = MultiNodeSnapshotPublishExecutor(worker_group=worker_group, resolve=resolve, collective_rpc=collection.rpc, reset_prefix_cache=reset_prefix_cache, versions=lambda: self.adapter.versions, expected_incarnations=rollout_incarnations, loop=asyncio.get_running_loop())
        scheduler = DrafterScheduler(worker_executor=self.worker_executor, publish_executor=self.publisher)
        self.adapter = MultiNodeVerlOnlineAdapter(scheduler=scheduler, versions=collection.versions, producer_ids=(collection.producer_id,), abort_plan=abort_plan, publish_commit=self.publisher.require_commit, resume_rollout=lambda: None, bind_worker_plan=self.worker_executor.bind_online_plan, parallel_updates=parallel_updates)
        self.rollout_incarnations = {str(rank): inc for rank, inc in rollout_incarnations.items()}

    async def update(self, commit, actor_batch):
        """Use the real actor batch; no hidden tensors enter this coordinator."""
        if self.collection.failed:
            raise RuntimeError('failed online session requires explicit complete-checkpoint recovery')
        if self.collection.pending_collection != commit:
            raise RuntimeError('VeRL actor step has no matching committed async collection')
        try:
            if self.phase_control is not None:
                await self.phase_control.enter_training()
            self.adapter.on_rollout_complete(commit)
            binding = await asyncio.to_thread(self.adapter.before_actor_update, config=self.schedule_config)
            if binding.plan.launch:
                ready = await asyncio.to_thread(self.worker_executor.preflight_training, binding.plan)
                if any((row.get('ready') is not True or row.get('participating') is not True or row.get('target_version') != binding.plan.required_target_version or (row.get('data_version') != binding.plan.data_version) for row in ready)):
                    raise RuntimeError('draft inputs did not pass actual preflight before actor update')
            if getattr(self.adapter, 'parallel_updates', False):
                actor_result = await self._join_parallel_updates(actor_batch, binding)
            else:
                actor_result = await drain_owned_call(self._actor_update_and_park, actor_batch, binding)
            if not isinstance(actor_result, dict) or actor_result.get('rl_step') != commit.versions.rl_step or actor_result.get('optimizer_committed') is not True:
                raise RuntimeError('VeRL actor did not acknowledge the real optimizer update')
            if getattr(self.adapter, 'parallel_updates', False):
                draft_result = await asyncio.to_thread(self.adapter.after_actor_update, rl_step=commit.versions.rl_step)
            else:
                draft_result = await drain_owned_call(self._serial_draft_update_and_park, commit.versions.rl_step)
            if self.phase_control is not None:
                await self.phase_control.enter_publication()
            publication = await _await(self.publish_actor(actor_result, binding))
            acks = tuple((TargetWeightAck(**row) for row in publication['acknowledgements']))
            commit_target_weight_acks(self.adapter, acknowledgements=acks, expected_incarnations=self.rollout_incarnations, target_revision=publication['target_revision'], actor_weight_manifest_sha256=publication['actor_weight_manifest_sha256'])
            versions = replace(self.adapter.versions, draft_revision=self.adapter.versions.draft_revision + int(binding.plan.launch and binding.plan.publish_after_success))

            async def publish_draft():
                await asyncio.to_thread(self.adapter.at_rollout_safe_point)
                if self.adapter.versions != versions:
                    raise RuntimeError('async engine and scheduler committed different online versions')
                if self.phase_control is not None:
                    await self.phase_control.finish_publication(dict(versions=self.adapter.versions, target_publication=publication, draft_acks=self.publisher.last_acks, draft_snapshot=self.publisher.last_snapshot))
            await self.collection.commit_safe_point(versions=versions, publish=publish_draft)
            return dict(actor=actor_result, draft=draft_result, versions=self.adapter.versions)
        except BaseException as error:
            self.collection.failed = True
            if self.phase_control is not None:
                self.phase_control.mark_uncertain()
            self.adapter.on_role_failure('async_session', error)
            raise

    async def _park_completed_role(self, role):
        control = getattr(self, 'phase_control', None)
        if control is not None:
            await control.retire_training_role(role)

    async def _actor_update_and_park(self, actor_batch, binding):
        result = await _await(self.actor_update(actor_batch, binding))
        if not isinstance(result, dict) or result.get('rl_step') != binding.versions.rl_step or result.get('optimizer_committed') is not True:
            raise RuntimeError('VeRL actor did not acknowledge the real optimizer update')
        await self._park_completed_role('actor')
        return result

    async def _serial_draft_update_and_park(self, rl_step):
        result = await asyncio.to_thread(self.adapter.after_actor_update, rl_step=rl_step)
        await self._park_completed_role('draft')
        return result

    async def _parallel_draft_update_and_park(self):
        result = await asyncio.to_thread(self.adapter.execute_prepared_draft)
        await self._park_completed_role('draft')
        return result

    async def _join_parallel_updates(self, actor_batch, binding):
        """Drain both actual effects before propagating failure or cancellation.

        Cancelling an asyncio.to_thread waiter cannot stop its optimizer RPC.
        The pair remains shielded until both ports have actually terminated.
        Publication remains in update(), after both real effects pass checks.
        """
        pair = asyncio.gather(self._actor_update_and_park(actor_batch, binding), self._parallel_draft_update_and_park(), return_exceptions=True)
        cancelled = None
        while True:
            try:
                results = await asyncio.shield(pair)
                break
            except asyncio.CancelledError as error:
                cancelled = error
        if cancelled is not None:
            raise cancelled
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results[0]
