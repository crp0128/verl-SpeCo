# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
import asyncio
from copy import deepcopy
from types import SimpleNamespace
import threading
import pytest
import torch
from verl_speco.integration.multinode_recovery import ColocatedCardCoordinator, validate_shared_cards, digest
from verl_speco.integration.multinode_recovery import ColocatedWorkerMemoryPort

def placements(size=32):
    return {role: [dict(run_id='run', generation=4, role=role, rank=rank, worker_incarnation=role + str(rank), node_ip='node' + str(rank // 16), device_uuid='device' + str(rank % 16), placement_group_id='pg' + str(rank // 16), bundle_index=rank % 16) for rank in range(size)] for role in ('actor', 'draft')}

class Ports:

    def __init__(self, rows):
        self.rows = rows
        self.events = []
        self.corrupt = None

    def memory(self, role, request):
        self.events.append((role, request['location']))
        result = [dict({key: value for key, value in request.items() if key != 'owner_incarnations'}, **{key: owner[key] for key in ('rank', 'worker_incarnation', 'node_ip', 'device_uuid')}, model_on_device=request['location'] != 'cpu', optimizer_on_device=request['location'] == 'training', head_on_device=role == 'draft' and request['location'] != 'cpu', prepared_pending=False) for owner in self.rows[role]]
        if self.corrupt:
            self.corrupt(result)
        return result

    async def sleep(self, request):
        self.events.append(('sleep', request['sleep_level']))
        return dict(request, complete=True)

    async def wake(self, request):
        self.events.append(('wake', tuple(request['tags']), request['resume_generation']))
        return dict(request, complete=True)

    def publish(self, proof):
        self.events.append(('ACKs', proof))
        return proof == 'all-owned-target-and-draft-ACKs'

def coordinator(rows=None):
    rows = rows or placements()
    ports = Ports(rows)
    control = ColocatedCardCoordinator(run_id='run', generation=4, actor_placements=rows['actor'], draft_placements=rows['draft'], memory_transition=ports.memory, rollout_sleep=ports.sleep, rollout_wake=ports.wake, publication_validator=ports.publish)
    return (control, ports)

def test_shared_32_card_default_async_phases_keep_DP32_and_weights_KV_separate():

    async def run():
        control, ports = coordinator()
        assert control.binding['draft_mesh'] == dict(dp_replicate=32, dp_shard=1, tp=1, pp=1, sp=1, ep=1)
        await control.start_rollout()
        await control.enter_training()
        await control.enter_publication()
        await control.finish_publication('all-owned-target-and-draft-ACKs')
        assert ports.events == [('actor', 'cpu'), ('draft', 'cpu'), ('wake', ('weights', 'kv_cache'), True), ('sleep', 1), ('actor', 'training'), ('draft', 'training'), ('draft', 'cpu'), ('actor', 'publishing'), ('wake', ('weights', 'kv_cache'), False), ('ACKs', 'all-owned-target-and-draft-ACKs'), ('actor', 'cpu'), ('draft', 'cpu'), ('wake', ('kv_cache',), True)]
        assert control.phase == 'rollout' and control.epoch == 4 and (not control.failed)
        ports.rows['actor'][0]['device_uuid'] = 'changed'
        assert control.binding['actor'][0]['device_uuid'] == 'device0'
    asyncio.run(run())

@pytest.mark.parametrize('fault', ['different-node', 'different-UUID', 'different-PG', 'different-bundle', 'duplicate-card', 'duplicate-rank', 'duplicate-incarnation', 'same-process', 'generation-bool', 'rank-bool', 'missing'])
def test_shared_claim_requires_exact_observed_physical_pairs_and_independent_processes(fault):
    rows = placements()
    if fault == 'different-node':
        rows['draft'][0]['node_ip'] = 'other'
    elif fault == 'different-UUID':
        rows['draft'][0]['device_uuid'] = 'other'
    elif fault == 'different-PG':
        rows['draft'][0]['placement_group_id'] = 'other'
    elif fault == 'different-bundle':
        rows['draft'][0]['bundle_index'] = 1
    elif fault == 'duplicate-card':
        rows['draft'][1]['device_uuid'] = 'device0'
    elif fault == 'duplicate-rank':
        rows['draft'][1]['rank'] = 0
    elif fault == 'duplicate-incarnation':
        rows['draft'][1]['worker_incarnation'] = 'draft0'
    elif fault == 'same-process':
        rows['draft'][0]['worker_incarnation'] = 'actor0'
    elif fault == 'generation-bool':
        rows['draft'][0]['generation'] = True
    elif fault == 'rank-bool':
        rows['draft'][0]['rank'] = False
    else:
        rows['draft'].pop()
    with pytest.raises(ValueError):
        coordinator(rows)

@pytest.mark.parametrize('fault', ['missing-rank', 'wrong-owner', 'same-node-wrong-UUID', 'generation-bool', 'epoch-bool', 'incomplete-offload', 'prepared-inputs', 'untyped-offload'])
def test_memory_receipt_must_prove_all_actual_ranks_before_waking_rollout(fault):

    async def run():
        control, ports = coordinator()

        def corrupt(rows):
            if fault == 'missing-rank':
                rows.pop()
            elif fault == 'wrong-owner':
                rows[0]['worker_incarnation'] = 'old'
            elif fault == 'same-node-wrong-UUID':
                rows[0]['device_uuid'] = 'other'
            elif fault == 'generation-bool':
                rows[0]['generation'] = True
            elif fault == 'epoch-bool':
                rows[0]['transition_epoch'] = True
            elif fault == 'incomplete-offload':
                rows[0]['optimizer_on_device'] = True
            elif fault == 'prepared-inputs':
                rows[0]['prepared_pending'] = True
            else:
                rows[0]['model_on_device'] = 0
        ports.corrupt = corrupt
        with pytest.raises(RuntimeError):
            await control.start_rollout()
        assert control.phase == 'failed' and control.failed and (not any((event[0] == 'wake' for event in ports.events)))
        with pytest.raises(RuntimeError):
            await control.start_rollout()
    asyncio.run(run())

def test_incomplete_publication_never_resumes_KV_or_generation():

    async def run():
        control, ports = coordinator()
        await control.start_rollout()
        await control.enter_training()
        await control.enter_publication()
        with pytest.raises(RuntimeError, match='ACKs'):
            await control.finish_publication('partial')
        assert not any((event == ('wake', ('kv_cache',), True) for event in ports.events))
        assert control.failed
    asyncio.run(run())

def test_repeated_cancellation_waits_for_actual_blocking_offload_before_failing_owner():

    async def run():
        control, ports = coordinator()
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        original = control.memory

        def blocked(role, request):
            entered.set()
            release.wait(5)
            row = original(role, request)
            finished.set()
            return row
        control.memory = blocked
        task = asyncio.create_task(control.start_rollout())
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done() and (not finished.is_set())
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set() and control.failed and (not any((e[0] == 'wake' for e in ports.events)))
    asyncio.run(run())

def test_parallel_transition_attempts_serialize_and_reject_invalid_predecessor():

    async def run():
        control, ports = coordinator()
        values = await asyncio.gather(control.start_rollout(), control.start_rollout(), return_exceptions=True)
        assert sum((isinstance(v, dict) for v in values)) == 1 and sum((isinstance(v, RuntimeError) for v in values)) == 1
        assert ports.events.count(('wake', ('weights', 'kv_cache'), True)) == 1 and control.phase == 'rollout'
    asyncio.run(run())

def memory_case():
    rows = placements(2)
    binding = validate_shared_cards(run_id='run', generation=4, **rows)
    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(module.parameters())
    calls = []

    class Engine:

        def to(self, **kwargs):
            calls.append(kwargs)
    engine = Engine()
    engine.module = module
    engine.optimizer = optimizer
    port = ColocatedWorkerMemoryPort(binding=binding, rank=0, role='actor', engine=engine, current_owner=lambda: deepcopy(rows['actor'][0]), operation_gate=lambda: dict(idle=True, prepared_pending=False), device_type='npu')
    request = dict(run_id='run', generation=4, role='actor', location='cpu', transition_epoch=1, placement_binding_sha256=digest(binding), owner_incarnations={0: 'actor0', 1: 'actor1'})
    return (port, request, calls, rows)

def test_actual_CPU_tensor_audit_retries_without_repeating_effects():
    port, request, calls, _ = memory_case()
    first = port.transition(request)
    assert not first['model_on_device'] and len(calls) == 1
    assert port.transition(request) == first and len(calls) == 1
    changed = dict(request, location='training')
    with pytest.raises(ValueError, match='epoch'):
        port.transition(changed)
    assert len(calls) == 1

def test_claimed_accelerator_load_without_actual_tensor_movement_is_rejected():
    port, request, calls, _ = memory_case()
    request['location'] = 'training'
    with pytest.raises(RuntimeError, match='model parameters'):
        port.transition(request)
    assert len(calls) == 1 and port.failed
    with pytest.raises(RuntimeError, match='recovery'):
        port.transition(request)

@pytest.mark.parametrize('fault', ['owner', 'uuid', 'bool-generation', 'bool-epoch', 'prepared-input', 'active-operation'])
def test_memory_port_rejects_unknown_owner_or_inflight_work_before_effects(fault):
    port, request, calls, rows = memory_case()
    if fault == 'owner':
        rows['actor'][0]['worker_incarnation'] = 'old'
    elif fault == 'uuid':
        rows['actor'][0]['device_uuid'] = 'wrong'
    elif fault == 'bool-generation':
        request['generation'] = True
    elif fault == 'bool-epoch':
        request['transition_epoch'] = True
    elif fault == 'prepared-input':
        port.gate = lambda: dict(idle=True, prepared_pending=True)
    else:
        port.gate = lambda: dict(idle=False, prepared_pending=False)
    with pytest.raises((ValueError, RuntimeError)):
        port.transition(request)
    assert not calls
