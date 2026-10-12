# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Joint coordinator protocol ordering; real control, fixture tensor/RPC roles.'
import asyncio
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
import msgspec
from _multinode_fixture_joint_restore import case
from verl_speco.integration.multinode_recovery import MultiNodeJointRestoreCoordinator as MultiNodeJointRestoreCoordinator
from verl_speco.integration.multinode_publication import manifest_for_state
from verl_speco.transport.control import MultiNodeJointReadyControl as MultiNodeJointReadyControl

@pytest.mark.parametrize('fault', [None, 'control', 'native-binding', 'actor-state', 'actor-generation', 'source-manifest', 'boolean-rank', 'source-wire', 'loaded-wire', 'head', 'prefix', 'target-ack', 'draft-ack', 'actor-release', 'native-release', 'draft-release', 'optimizer-release'])
def test_joint_recovery_has_no_compute_release_before_exact_consensus_and_aborts_every_uncertainty(tmp_path, fault):
    parent, config, owners, proof = case(tmp_path)
    binding = config['joint_restore_binding']
    events = []
    pending = {}
    control = MultiNodeJointReadyControl(run_id='run', generation=2, partition='samples', capacity_bytes=4096)

    def actor_restore(request):
        events.append('actor-restore')
        rows = [dict(row, binding_checksum=binding['checksum'], compute_released=False) for row in proof['actor_rows']]
        if fault == 'actor-state':
            rows[0]['state_sha256'] = '0' * 64
        if fault == 'actor-generation':
            rows[0]['generation'] = 1
        return rows

    def draft_restore(request):
        events.append('draft-restore')
        return [dict(row, binding_checksum=binding['checksum'], compute_released=False) for row in proof['draft_rows']]

    def prepare(request):
        events.append('source-prepare')
        pending['request'] = request
        tensor = torch.zeros(1, dtype=torch.bfloat16)
        manifest = manifest_for_state({f'model.weight-{i}': tensor for i in range(35869)}, run_id='run', generation=2, rl_step=1, target_revision=1, operation_id=request['operation_id'], actor_incarnations=owners['actor'])
        pending['manifest'] = manifest
        if fault == 'source-manifest':
            manifest['target_revision'] = 2
        return [dict(rank=rank, worker_incarnation=owner, prepared=True, header_sha256=manifest['header_sha256'], manifest=manifest if rank == 0 else None, master_metadata={'owned': True} if rank == 0 else None) for rank, owner in owners['actor'].items()]

    def rows(role, phase):
        result = [dict(rank=rank, worker_incarnation=owner, **{phase: True}) for rank, owner in owners[role].items()]
        return result

    def send(operation):
        events.append('source-send')
        wire = dict(header_sha256=pending['manifest']['header_sha256'], actor_weight_manifest_sha256='9' * 64, wire_tensor_count=35869, wire_nbytes=pending['manifest']['total_nbytes'])
        pending['wire'] = wire
        head = dict(parent['target_head'], generation=2)
        head['actor_weight_manifest_sha256'] = '9' * 64
        if fault == 'head':
            head['sha256'] = '0' * 64
        if fault == 'source-wire':
            wire['wire_tensor_count'] = 1
        return [dict(row, wire_receipt=wire if row['rank'] == 0 else None, target_head_binding=head if row['rank'] == 0 else None) for row in rows('actor', 'sent')]

    def actor_release(request):
        events.append('actor-release')
        if fault == 'actor-release':
            raise RuntimeError('injected partial release')
        return [dict(row, compute_released=True, run_id='run', generation=2, optimizer_step=1) for row in rows('actor', 'released')]

    def draft_release(request):
        events.append('draft-release')
        result = [dict(worker_id=key, worker_incarnation=owner, released=True, run_id='run', generation=2, optimizer_step=1, target_revision=1, draft_revision=5) for key, owner in owners['draft'].items()]
        if fault == 'draft-release':
            result[0]['released'] = False
        if fault == 'optimizer-release':
            result[0]['optimizer_step'] = 2
        return result
    actor = SimpleNamespace(speco_restore_joint_actor=actor_restore, speco_prepare_target_transfer=prepare, speco_init_target_transfer=lambda request: rows('actor', 'connected'), speco_send_target_transfer=send, speco_finish_target_transfer=lambda *args: rows('actor', 'released'), speco_release_joint_actor=actor_release)
    draft = SimpleNamespace(speco_restore_joint_draft=draft_restore, speco_release_joint_draft=draft_release)

    async def control_rpc(method, kwargs):
        result = getattr(control, method)(**kwargs)
        if fault == 'control' and method == 'snapshot_quiescent':
            result['capacity_bytes'] += 1
        return result

    async def rpc(method, kwargs):
        assert isinstance(method, str)
        wire = msgspec.msgpack.encode((method, kwargs))
        assert msgspec.msgpack.decode(wire)[0] == method
        if method == 'speco_install_joint_native_restore':
            events.append('native-fence')
            result = [dict(row, binding_checksum=binding['checksum'], compute_released=False) for row in rows('rollout', 'prepared')]
            if fault == 'native-binding':
                result[0]['binding_checksum'] = '0' * 64
            return result
        if method == 'speco_prepare_target_transfer':
            events.append('native-prepare')
            result = [dict(row, header_sha256=pending['manifest']['header_sha256']) for row in rows('rollout', 'prepared')]
            if fault == 'boolean-rank':
                result[1]['rank'] = True
            return result
        if method == 'speco_init_target_transfer':
            return rows('rollout', 'connected')
        if method == 'speco_receive_target_transfer':
            while 'wire' not in pending:
                await asyncio.sleep(0)
            events.append('native-load')
            result = [dict(row, **pending['wire'], complete_native_names_and_expert_slots=True) for row in rows('rollout', 'loaded')]
            if fault == 'loaded-wire':
                result[0]['actor_weight_manifest_sha256'] = '0' * 64
            return result
        if method == 'speco_commit_target_transfer':
            events.append('target-ack')
            result = deepcopy(proof['target_acks'])
            if fault == 'target-ack':
                result[0]['kv_cache_cleared'] = False
            return result
        if method == 'speco_finish_target_transfer':
            return rows('rollout', 'released')
        if method == 'speco_audit_joint_native_draft':
            events.append('draft-audit')
            result = deepcopy(proof['draft_acks'])
            if fault == 'draft-ack':
                result[0]['snapshot_checksum'] = '0' * 64
            return result
        assert method == 'speco_release_joint_native_restore'
        events.append('native-release')
        result = [dict(row, target_revision=1, draft_revision=5) for row in rows('rollout', 'compute_released')]
        if fault == 'native-release':
            result[0]['compute_released'] = False
        return result

    async def prefix():
        events.append('prefix-reset')
        return fault != 'prefix'

    async def abort(value):
        assert value == owners
        events.append('abort-all-exact-owners')
    coordinator = MultiNodeJointRestoreCoordinator(config=config, owners=owners, actor_group=actor, draft_group=draft, collective_rpc=rpc, control_rpc=control_rpc, reset_prefix_cache=prefix, abort_owned=abort, bucket_bytes=4096)

    async def run():
        if fault is not None:
            with pytest.raises((RuntimeError, ValueError)):
                await coordinator.restore()
            assert coordinator.phase == 'failed' and events[-1] == 'abort-all-exact-owners'
            assert events.count('abort-all-exact-owners') == 1
        else:
            result = await coordinator.restore()
            assert result['versions'] == dict(parent['versions'], generation=2)
            assert result['optimizer_effect_created'] is False and coordinator.phase == 'ready'
            assert events.index('native-fence') < events.index('actor-restore') < events.index('source-prepare')
            assert events.index('native-load') < events.index('prefix-reset') < events.index('target-ack') < events.index('draft-audit')
            assert all((events.index('draft-audit') < events.index(item) for item in ('actor-release', 'draft-release', 'native-release')))
            assert 'abort-all-exact-owners' not in events
        before = list(events)
        with pytest.raises(RuntimeError):
            await coordinator.restore()
        assert events == before
    asyncio.run(run())
