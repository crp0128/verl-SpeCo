# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Target publisher all-owner fences; fixtures do not claim real NPU transfer.'
import asyncio
from types import SimpleNamespace
import pytest
from verl_speco.integration.multinode_publication import MultiNodeTargetPublishExecutor as MultiNodeTargetPublishExecutor
from verl_speco.integration.multinode_online import OnlineVersions

class Ports:

    def __init__(self):
        self.events = []
        self.damage = None
        self.operation = None

    def source_rows(self, phase):
        return [dict(rank=i, worker_incarnation='actor' + str(i), **{phase: True}) for i in (0, 1)]

    def speco_prepare_target_transfer(self, request):
        self.events.append('source_prepare')
        self.operation = request['operation_id']
        self.request = request
        rows = self.source_rows('prepared')
        for row in rows:
            row.update(header_sha256='a' * 64, manifest=None, master_metadata=None)
        rows[0].update(manifest=dict(header_sha256='a' * 64), master_metadata={'host': 'source'})
        if self.damage == 'source-owner':
            rows[1]['worker_incarnation'] = 'foreign'
        return rows

    def speco_init_target_transfer(self, request):
        self.events.append('source_connect')
        return self.source_rows('connected')

    def speco_send_target_transfer(self, operation):
        self.events.append('source_send')
        rows = self.source_rows('sent')
        for row in rows:
            row.update(wire_receipt=None, target_head_binding=None)
        rows[0].update(wire_receipt=self.wire(), target_head_binding=dict(actor_weight_manifest_sha256='b' * 64, target_revision=10, uri='/tmp/head', sha256='c' * 64))
        return rows

    def wire(self):
        return dict(header_sha256='a' * 64, actor_weight_manifest_sha256='b' * 64, wire_tensor_count=4, wire_nbytes=28)

    def speco_register_online_target_head(self, head):
        self.events.append('register_next_head')
        return [dict(worker_id='0', worker_incarnation='draft0', registered=self.damage != 'draft-head', target_revision=10)]

    def speco_finish_target_transfer(self, *_):
        self.events.append('source_release')
        return self.source_rows('released')

    async def reset(self):
        self.events.append('actual_prefix_reset')
        return self.damage != 'prefix'

    async def rpc(self, name, kwargs):
        self.events.append(name)
        phase = {'speco_prepare_target_transfer': 'prepared', 'speco_init_target_transfer': 'connected', 'speco_receive_target_transfer': 'loaded', 'speco_commit_target_transfer': 'applied', 'speco_finish_target_transfer': 'released'}[name]
        rows = [dict(rank=i, worker_incarnation='rollout' + str(i), **{phase: True}) for i in (0, 1)]
        if phase == 'prepared':
            for row in rows:
                row['header_sha256'] = 'a' * 64
        if phase == 'loaded':
            for row in rows:
                row.update(self.wire(), complete_native_names_and_expert_slots=True)
            if self.damage == 'stream-bytes':
                rows[1]['actor_weight_manifest_sha256'] = 'f' * 64
            if self.damage == 'missing-expert':
                rows[1]['complete_native_names_and_expert_slots'] = False
        if phase == 'applied':
            for row in rows:
                row.update(run_id='run', generation=1, rl_step=6, target_revision=10, worker_id=str(row['rank']), actor_weight_manifest_sha256='b' * 64, kv_cache_cleared=True, draft_target_head_synced=True)
            if self.damage == 'cache':
                rows[1]['kv_cache_cleared'] = False
            if self.damage == 'head':
                rows[1]['draft_target_head_synced'] = False
            if self.damage == 'revision':
                rows[1]['target_revision'] = 11
        if phase == 'released' and self.damage == 'release':
            rows[1]['released'] = False
        return rows

def inputs(p):
    port = MultiNodeTargetPublishExecutor(actor_group=p, actor_incarnations={0: 'actor0', 1: 'actor1'}, draft_group=p, draft_incarnations={'0': 'draft0'}, collective_rpc=p.rpc, rollout_incarnations={0: 'rollout0', 1: 'rollout1'}, reset_prefix_cache=p.reset, tokenizer_fingerprint='tokenizer')
    binding = SimpleNamespace(versions=OnlineVersions('run', 1, 6, 9, 11, 4), collection_id='collection')
    receipt = dict(run_id='run', generation=1, rl_step=6, target_revision=9, collection_id='collection', optimizer_committed=True, optimizer_step=1, worker_incarnations={0: 'actor0', 1: 'actor1'})
    return (port, binding, receipt)

def test_target_revision_is_returned_only_after_all_real_port_phases_acknowledge():

    async def run():
        p = Ports()
        port, binding, actor = inputs(p)
        result = await port(actor, binding)
        assert result['target_revision'] == 10 and len(result['acknowledgements']) == 2
        assert p.events.index('register_next_head') < p.events.index('actual_prefix_reset') < p.events.index('speco_commit_target_transfer')
        assert p.events[-2:] == ['source_release', 'speco_finish_target_transfer']
        assert port.last_proof['actual_native_prefix_reset']
        assert all((set(row) == {'run_id', 'generation', 'rl_step', 'target_revision', 'worker_id', 'worker_incarnation', 'actor_weight_manifest_sha256', 'kv_cache_cleared', 'draft_target_head_synced'} for row in result['acknowledgements']))
    asyncio.run(run())

@pytest.mark.parametrize('damage', ['source-owner', 'stream-bytes', 'missing-expert', 'draft-head', 'prefix', 'cache', 'head', 'revision', 'release'])
def test_partial_wrong_or_uncertain_target_publication_cannot_resume_next_operation(damage):

    async def run():
        p = Ports()
        p.damage = damage
        port, binding, actor = inputs(p)
        with pytest.raises(RuntimeError):
            await port(actor, binding)
        assert port.failed and port.last_proof is None
        events = list(p.events)
        with pytest.raises(RuntimeError):
            await port(actor, binding)
        assert p.events == events
    asyncio.run(run())

def test_foreign_actor_receipt_is_rejected_before_source_or_native_RPC():

    async def run():
        p = Ports()
        port, binding, actor = inputs(p)
        actor['worker_incarnations'][1] = 'old'
        with pytest.raises(ValueError):
            await port(actor, binding)
        assert not p.events and (not port.failed)
    asyncio.run(run())
