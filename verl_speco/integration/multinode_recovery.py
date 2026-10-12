# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import torch

def colocated_actor_optimizer_require_adamw(optimizer):
    if not isinstance(optimizer, torch.optim.AdamW) or not optimizer.param_groups:
        raise ValueError('shared actor memory policy requires a nonempty AdamW optimizer')
    if any((group.get('fused') or group.get('differentiable') for group in optimizer.param_groups)):
        raise ValueError('shared actor memory policy requires unfused nondifferentiable AdamW')

def configure_colocated_actor_optimizer(optimizer):
    colocated_actor_optimizer_require_adamw(optimizer)
    optimizer.defaults['foreach'] = False
    for group in optimizer.param_groups:
        group['foreach'] = False
    return audit_colocated_actor_optimizer(optimizer)

def audit_colocated_actor_optimizer(optimizer):
    colocated_actor_optimizer_require_adamw(optimizer)
    if optimizer.defaults.get('foreach') is not False or any((group.get('foreach') is not False for group in optimizer.param_groups)):
        raise RuntimeError('shared actor optimizer must retain single-tensor AdamW after restore')
    sizes = []
    for group in optimizer.param_groups:
        for parameter in group['params']:
            local = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
            sizes.append(local.numel() * local.element_size())
    return dict(optimizer_class=type(optimizer).__name__, foreach=False, execution='single_tensor_adamw', parameter_groups=len(optimizer.param_groups), local_parameter_bytes=sum(sizes), largest_local_parameter_bytes=max(sizes, default=0))
from copy import deepcopy
import asyncio
import hashlib
import inspect
import json

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def validate_shared_cards(*, run_id, generation, actor, draft):
    if type(run_id) is not str or not run_id or type(generation) is not int or (generation < 1):
        raise ValueError('shared cards need explicit current run/generation')
    if type(actor) is not list or type(draft) is not list or (not 1 <= len(actor) <= 128) or (len(actor) != len(draft)):
        raise ValueError('shared actor/draft requires complete equally sized rank lists')
    required = {'run_id', 'generation', 'role', 'rank', 'worker_incarnation', 'node_ip', 'device_uuid', 'placement_group_id', 'bundle_index'}
    roles = {}
    for role, rows in [('actor', actor), ('draft', draft)]:
        seen = {}
        physical = set()
        incarnations = set()
        for row in rows:
            if type(row) is not dict or set(row) != required or row['run_id'] != run_id or (type(row['generation']) is not int) or (row['generation'] != generation) or (row['role'] != role) or (type(row['rank']) is not int) or (not 0 <= row['rank'] < len(rows)) or (type(row['bundle_index']) is not int) or (not 0 <= row['bundle_index'] < 16) or any((type(row[name]) is not str or not row[name] for name in ('worker_incarnation', 'node_ip', 'device_uuid', 'placement_group_id'))):
                raise ValueError('shared card placement has malformed or foreign identity')
            key = (row['node_ip'], row['device_uuid'])
            if row['rank'] in seen or key in physical or row['worker_incarnation'] in incarnations:
                raise ValueError('shared card placement duplicates rank/device/worker ownership')
            seen[row['rank']] = deepcopy(row)
            physical.add(key)
            incarnations.add(row['worker_incarnation'])
        roles[role] = seen
    if {x['worker_incarnation'] for x in actor} & {x['worker_incarnation'] for x in draft}:
        raise ValueError('actor and DP-only draft require independent process/mesh owners')
    for rank, owner in roles['actor'].items():
        peer = roles['draft'][rank]
        if any((owner[name] != peer[name] for name in ('node_ip', 'device_uuid', 'placement_group_id', 'bundle_index'))):
            raise ValueError('actor/draft rank pair is not on the same assigned physical card')
    return dict(run_id=run_id, generation=generation, actor=[roles['actor'][rank] for rank in range(len(actor))], draft=[roles['draft'][rank] for rank in range(len(draft))], draft_mesh=dict(dp_replicate=len(actor), dp_shard=1, tp=1, pp=1, sp=1, ep=1))

async def drain_owned_call(port, *args, **kwargs):

    async def invoke():
        if inspect.iscoroutinefunction(port):
            value = port(*args, **kwargs)
        else:
            value = await asyncio.to_thread(port, *args, **kwargs)
        return await value if inspect.isawaitable(value) else value
    task = asyncio.create_task(invoke())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result

class ColocatedCardCoordinator:

    def __init__(self, *, run_id, generation, actor_placements, draft_placements, memory_transition, rollout_sleep, rollout_wake, publication_validator, restore_validator=None, record_transition=None, actor_optimizer_cpu=False):
        if type(actor_optimizer_cpu) is not bool:
            raise ValueError('actor optimizer residency must be explicit boolean')
        self.actor_optimizer_cpu = actor_optimizer_cpu
        if any((not callable(p) for p in (memory_transition, rollout_sleep, rollout_wake, publication_validator))):
            raise ValueError('shared card phases require real memory/rollout/ACK ports')
        self.binding = validate_shared_cards(run_id=run_id, generation=generation, actor=actor_placements, draft=draft_placements)
        self.binding_sha = digest(self.binding)
        self.memory, self.sleep, self.wake, self.validate_publication = (memory_transition, rollout_sleep, rollout_wake, publication_validator)
        if restore_validator is not None and (not callable(restore_validator)):
            raise ValueError('shared restore requires an actual complete release validator')
        self.validate_restore = restore_validator
        self.phase = 'parked'
        self.epoch = 0
        self.failed = False
        self.lock = asyncio.Lock()
        if record_transition is not None and (not callable(record_transition)):
            raise ValueError('shared memory receipts require a callable record port')
        self.record_transition = record_transition
        self.retired_training_roles = set()
        self.training_retirements = {}

    def mark_uncertain(self):
        self.failed = True
        self.phase = 'failed'

    async def _memory(self, role, location):
        owners = {row['rank']: row for row in self.binding[role]}
        request = dict(run_id=self.binding['run_id'], generation=self.binding['generation'], role=role, location=location, transition_epoch=self.epoch, placement_binding_sha256=self.binding_sha, owner_incarnations={rank: row['worker_incarnation'] for rank, row in owners.items()})
        rows = await drain_owned_call(self.memory, role, request)
        required = set(request) - {'owner_incarnations'} | {'rank', 'worker_incarnation', 'node_ip', 'device_uuid', 'model_on_device', 'optimizer_on_device', 'head_on_device', 'prepared_pending'}
        if type(rows) is not list or len(rows) != len(owners) or any((type(row) is not dict or set(row) != required for row in rows)) or any((type(row['rank']) is not int for row in rows)) or ({row['rank'] for row in rows} != set(owners)):
            raise RuntimeError('shared memory transition lacks exact all-rank receipts')
        for row in rows:
            owner = owners[row['rank']]
            if any((type(row[key]) is not type(value) or row[key] != value for key, value in request.items() if key != 'owner_incarnations')) or any((row[key] != owner[key] for key in ('worker_incarnation', 'node_ip', 'device_uuid'))) or any((type(row[key]) is not bool for key in ('model_on_device', 'optimizer_on_device', 'head_on_device', 'prepared_pending'))) or (row['model_on_device'] != (location != 'cpu')) or (row['optimizer_on_device'] != (location == 'training' and (not (role == 'actor' and self.actor_optimizer_cpu)))) or (row['head_on_device'] != (role == 'draft' and location != 'cpu')) or (location == 'cpu' and row['prepared_pending']):
                raise RuntimeError('shared memory transition has foreign owner or incomplete physical offload')
        return rows

    async def _rollout(self, port, *, tags=None, resume_generation=True):
        request = dict(run_id=self.binding['run_id'], generation=self.binding['generation'], transition_epoch=self.epoch, placement_binding_sha256=self.binding_sha)
        if tags is None:
            request['sleep_level'] = 1
        else:
            request.update(tags=tags, resume_generation=resume_generation)
        receipt = await drain_owned_call(port, request)
        if type(receipt) is not dict or set(receipt) != set(request) | {'complete'} or receipt['complete'] is not True or any((type(receipt[key]) is not type(value) or receipt[key] != value for key, value in request.items())):
            raise RuntimeError('shared rollout handoff lacks its exact owned completed receipt')
        return receipt

    async def _change(self, allowed, final, operation):
        async with self.lock:
            if self.failed or self.phase not in allowed:
                raise RuntimeError('shared card phase requires owned recovery or a valid preceding boundary')
            self.epoch += 1
            self.phase = 'transitioning'
            try:
                value = await operation()
                self.phase = final
                return dict(phase=final, transition_epoch=self.epoch, placement_binding_sha256=self.binding_sha, receipts=value)
            except BaseException:
                self.failed = True
                self.phase = 'failed'
                raise

    async def start_rollout(self, *, resume_generation=True):
        if type(resume_generation) is not bool or (not resume_generation and self.validate_restore is None):
            raise ValueError('cold shared restore needs its explicit release validator')

        async def operation():
            actor = await self._memory('actor', 'cpu')
            draft = await self._memory('draft', 'cpu')
            wake = await self._rollout(self.wake, tags=['weights', 'kv_cache'], resume_generation=resume_generation)
            return dict(actor=actor, draft=draft, rollout=wake)
        return await self._change({'parked'}, 'rollout' if resume_generation else 'restore_cold', operation)

    async def enter_training(self):

        async def operation():
            sleep = await self._rollout(self.sleep)
            actor = await self._memory('actor', 'training')
            draft = await self._memory('draft', 'training')
            self.retired_training_roles.clear()
            self.training_retirements = {}
            return dict(rollout=sleep, actor=actor, draft=draft)
        return await self._change({'rollout', 'restore_cold'}, 'training', operation)

    async def retire_training_role(self, role):
        """Offload a completed role while its peer may still be updating.

        The caller must drain that role's actual optimizer RPC first. The
        worker gate independently rejects pending work and every assigned
        rank must prove model, moments, gradients and draft head are on CPU.
        Native generation stays asleep; this is not a publication boundary.
        """
        if role not in ('actor', 'draft'):
            raise ValueError('only an assigned actor or draft can retire training')
        async with self.lock:
            if self.failed or self.phase != 'training' or role in self.retired_training_roles:
                raise RuntimeError('role retirement requires its unretired owned training phase')
            self.epoch += 1
            self.phase = 'transitioning'
            try:
                rows = await self._memory(role, 'cpu')
                receipt = dict(phase='training', role=role, transition_epoch=self.epoch, placement_binding_sha256=self.binding_sha, receipts=rows)
                if self.record_transition is not None:
                    await drain_owned_call(self.record_transition, 'shared-retire-' + role + '-epoch-' + str(self.epoch), receipt)
                self.retired_training_roles.add(role)
                self.training_retirements[role] = deepcopy(receipt)
                self.phase = 'training'
                return receipt
            except BaseException:
                self.failed = True
                self.phase = 'failed'
                raise

    async def enter_publication(self):

        async def operation():
            draft = await self._memory('draft', 'cpu')
            actor = await self._memory('actor', 'publishing')
            wake = await self._rollout(self.wake, tags=['weights', 'kv_cache'], resume_generation=False)
            return dict(draft=draft, actor=actor, rollout=wake)
        return await self._change({'training'}, 'publishing', operation)

    async def finish_publication(self, proof):

        async def operation():
            if await drain_owned_call(self.validate_publication, proof) is not True:
                raise RuntimeError('shared rollout cannot resume before complete target/draft ACKs')
            actor = await self._memory('actor', 'cpu')
            draft = await self._memory('draft', 'cpu')
            wake = await self._rollout(self.wake, tags=['kv_cache'])
            return dict(actor=actor, draft=draft, rollout=wake)
        return await self._change({'publishing'}, 'rollout', operation)

    async def finish_restore(self, proof):

        async def operation():
            if self.validate_restore is None or await drain_owned_call(self.validate_restore, proof) is not True:
                raise RuntimeError('shared recovery cannot resume before complete restored role/cache consensus')
            actor = await self._memory('actor', 'cpu')
            draft = await self._memory('draft', 'cpu')
            wake = await self._rollout(self.wake, tags=['kv_cache'])
            return dict(actor=actor, draft=draft, rollout=wake)
        return await self._change({'publishing'}, 'rollout', operation)

def colocated_worker_memory_local(value):
    return value.to_local() if hasattr(value, 'to_local') else value

class ColocatedWorkerMemoryPort:

    def __init__(self, *, binding, rank, role, engine, current_owner, operation_gate, device_type, head_getter=lambda: None, head_setter=lambda _: None, synchronize_and_empty_cache=lambda: None, optimizer_training_cpu=False):
        if type(optimizer_training_cpu) is not bool or (optimizer_training_cpu and role != 'actor'):
            raise ValueError('CPU optimizer residency is supported only for the shared actor')
        self.optimizer_training_cpu = optimizer_training_cpu
        checked = validate_shared_cards(run_id=binding['run_id'], generation=binding['generation'], actor=binding['actor'], draft=binding['draft'])
        if binding != checked or type(rank) is not int or role not in ('actor', 'draft'):
            raise ValueError('memory port requires its exact shared placement')
        owners = {row['rank']: row for row in binding[role]}
        if rank not in owners or type(device_type) is not str or (not device_type) or (device_type == 'cpu'):
            raise ValueError('memory port requires an assigned accelerator rank')
        if any((not callable(p) for p in (current_owner, operation_gate, head_getter, head_setter, synchronize_and_empty_cache))):
            raise ValueError('memory port requires actual owned engine/device inspection')
        self.binding, self.rank, self.role, self.engine = (deepcopy(binding), rank, role, engine)
        self.owner, self.gate, self.head_getter, self.head_setter, self.sync = (current_owner, operation_gate, head_getter, head_setter, synchronize_and_empty_cache)
        self.device_type = device_type
        self.epoch = 0
        self.request = None
        self.failed = False

    def _validate(self, request):
        required = {'run_id', 'generation', 'role', 'location', 'transition_epoch', 'placement_binding_sha256', 'owner_incarnations'}
        owners = {row['rank']: row for row in self.binding[self.role]}
        if type(request) is not dict or set(request) != required or request['run_id'] != self.binding['run_id'] or (type(request['generation']) is not int) or (request['generation'] != self.binding['generation']) or (request['role'] != self.role) or (request['location'] not in ('cpu', 'training', 'publishing')) or (request['location'] == 'publishing' and self.role != 'actor') or (type(request['transition_epoch']) is not int) or (request['transition_epoch'] < 1) or (request['placement_binding_sha256'] != digest(self.binding)) or (type(request['owner_incarnations']) is not dict) or any((type(rank) is not int for rank in request['owner_incarnations'])) or (request['owner_incarnations'] != {rank: owner['worker_incarnation'] for rank, owner in owners.items()}) or (self.owner() != owners[self.rank]):
            raise ValueError('memory transition has stale owner/placement or malformed identity')
        if self.failed:
            raise RuntimeError('uncertain memory transition requires fresh owned recovery')
        if request['transition_epoch'] < self.epoch or (request['transition_epoch'] == self.epoch and request != self.request):
            raise ValueError('memory transition reuses an epoch with different effects')
        gate = self.gate()
        if type(gate) is not dict or set(gate) != {'idle', 'prepared_pending'} or any((type(value) is not bool for value in gate.values())) or (gate['idle'] is not True) or (request['location'] in ('cpu', 'publishing') and gate['prepared_pending']):
            raise RuntimeError('shared memory handoff still has active optimizer/transfer/prepared inputs')
        return gate

    def _audit(self, request, gate):
        import torch
        location = request['location']
        model_device = self.device_type if location != 'cpu' else 'cpu'
        optimizer_device = self.device_type if location == 'training' and (not self.optimizer_training_cpu) else 'cpu'
        gradient_device = self.device_type if location == 'training' else 'cpu'
        parameters = list(self.engine.module.parameters())
        if not parameters or any((colocated_worker_memory_local(p).device.type != model_device for p in parameters)):
            raise RuntimeError('shared model parameters did not actually reach their requested device')
        for parameter in parameters:
            grad = parameter.grad
            if grad is not None and colocated_worker_memory_local(grad).device.type != gradient_device:
                raise RuntimeError('shared model gradient still retains the wrong accelerator placement')
        for state in self.engine.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step' and (colocated_worker_memory_local(value).device.type != optimizer_device):
                    raise RuntimeError('shared optimizer tensors did not actually complete offload/load')
        head = self.head_getter()
        if self.role == 'draft':
            if not torch.is_tensor(head) or colocated_worker_memory_local(head).device.type != model_device:
                raise RuntimeError('shared draft retains an unaccounted target head on the wrong device')
        elif head is not None:
            raise RuntimeError('actor memory port cannot own an extra draft head')
        owner = self.owner()
        return dict({key: value for key, value in request.items() if key != 'owner_incarnations'}, rank=self.rank, worker_incarnation=owner['worker_incarnation'], node_ip=owner['node_ip'], device_uuid=owner['device_uuid'], model_on_device=location != 'cpu', optimizer_on_device=optimizer_device != 'cpu', head_on_device=self.role == 'draft' and location != 'cpu', prepared_pending=gate['prepared_pending'])

    def transition(self, request):
        gate = self._validate(request)
        try:
            if request != self.request:
                location = request['location']
                if location == 'publishing':
                    self.engine.to(device='cpu', model=True, optimizer=True, grad=True)
                    self.engine.optimizer.zero_grad(set_to_none=True)
                    if any((parameter.grad is not None for parameter in self.engine.module.parameters())):
                        raise RuntimeError('publication cannot reload retained gradient buffers')
                    self.engine.to(device=self.device_type, model=True, optimizer=False, grad=False)
                elif location == 'training' and self.optimizer_training_cpu:
                    self.engine.to(device='cpu', model=False, optimizer=True, grad=False)
                    self.engine.to(device=self.device_type, model=True, optimizer=False, grad=True)
                else:
                    self.engine.to(device='cpu' if location == 'cpu' else self.device_type, model=True, optimizer=True, grad=True)
                if self.role == 'draft':
                    self.head_setter(self.head_getter().to('cpu' if location == 'cpu' else self.device_type))
                self.sync()
            result = self._audit(request, gate)
            self.epoch = request['transition_epoch']
            self.request = deepcopy(request)
            return result
        except BaseException:
            self.failed = True
            raise
from functools import wraps

def install_colocated_actor_cpu_step(engine, *, device_type, synchronize):
    if device_type not in ('npu', 'cuda') or not callable(synchronize) or (not callable(engine.to)):
        raise ValueError('shared actor CPU step requires its accelerator engine and synchronization')
    optimizer = engine.optimizer
    audit_colocated_actor_optimizer(optimizer)
    existing = getattr(optimizer, '_speco_cpu_step_engine', None)
    if existing is not None:
        if existing is not engine:
            raise ValueError('CPU optimizer step cannot change engine owner')
        return
    original = optimizer.step
    state = {'failed': False, 'active': False}

    @wraps(original)
    def step(*args, **kwargs):
        if state['failed'] or state['active']:
            raise RuntimeError('uncertain or reentrant shared CPU optimizer requires fresh recovery')
        if args or kwargs.get('closure') is not None:
            raise ValueError('shared CPU optimizer supports the native closure-free AdamW step')
        audit_colocated_actor_optimizer(optimizer)
        state['active'] = True
        try:
            engine.to(device='cpu', model=True, optimizer=True, grad=True)
            synchronize()
            import torch
            parameters = list(engine.module.parameters())
            if not parameters or any((colocated_worker_memory_local(p).device.type != 'cpu' or (p.grad is not None and colocated_worker_memory_local(p.grad).device.type != 'cpu') for p in parameters)):
                raise RuntimeError('CPU optimizer step requires actual CPU parameters and clipped gradients')
            if any((torch.is_tensor(v) and colocated_worker_memory_local(v).device.type != 'cpu' for values in optimizer.state.values() for v in values.values())):
                raise RuntimeError('CPU optimizer step requires actual CPU moments and counters')
            result = original(**kwargs)
            engine.to(device=device_type, model=True, optimizer=False, grad=True)
            synchronize()
            return result
        except BaseException:
            state['failed'] = True
            raise
        finally:
            state['active'] = False
    optimizer.step = step
    optimizer._speco_cpu_step_engine = engine

def select_colocated_native_replica(binding, tensor_parallel_size=8):
    checked = validate_shared_cards(run_id=binding['run_id'], generation=binding['generation'], actor=binding['actor'], draft=binding['draft'])
    if checked != binding or type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError('native checkpoint topology requires a complete validated shared binding')
    actors = checked['actor']
    source = actors[0]
    source_card = (source['node_ip'], source['device_uuid'])
    for replica in range(len(actors) // tensor_parallel_size):
        rows = actors[replica * tensor_parallel_size:(replica + 1) * tensor_parallel_size]
        cards = {(r['node_ip'], r['device_uuid']) for r in rows}
        if len(cards) == tensor_parallel_size and source_card not in cards and all((r['node_ip'] == source['node_ip'] for r in rows)):
            return replica
    raise ValueError('no complete native TP slice leaves the actor publisher on a distinct die')

def require_distinct_checkpoint_devices(actor_source, native_placements):
    if type(native_placements) is not list or not native_placements:
        raise ValueError('checkpoint topology requires actual native device observations')
    cards = [(r['node_ip'], r['device_uuid']) for r in native_placements]
    if len(set(cards)) != len(cards) or (actor_source['node_ip'], actor_source['device_uuid']) in cards:
        raise ValueError('checkpoint ranks cannot share the same physical NPU in one HCCL group')
    return True
import logging
import os
from pathlib import Path
import time
colocated_memory_diagnostics_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / 'outputs'
colocated_memory_diagnostics_LOGGER = logging.getLogger(__name__)

def record_colocated_memory(*, role, rank, stage, details=None):
    """Record live Torch allocation and device-wide free memory separately.

    Driver/HCCL and other processes are outside Torch's allocator. A zero
    Torch allocation therefore does not prove that a physical die is empty.
    Diagnostics must never change the outcome of a handoff or replace its
    original failure. No device query or file write occurs unless opted in.
    """
    configured = os.environ.get('SPECO_COLOCATED_MEMORY_TRACE_DIR')
    if not configured:
        return None
    try:
        destination = Path(configured)
        if not destination.is_absolute() or not destination.resolve().is_relative_to(colocated_memory_diagnostics_OUTPUT_ROOT.resolve()):
            raise ValueError('memory diagnostics must remain in owned SpeCo outputs')
        run_id = os.environ['SPECO_RUN_ID']
        generation = int(os.environ['SPECO_GENERATION'])
        if not run_id or generation < 1 or role not in ('actor', 'draft', 'native') or (type(rank) is not int) or (rank < 0) or (stage not in ('before', 'after', 'failed')) or (details is not None and type(details) is not dict):
            raise ValueError('memory diagnostics require an explicit phase and current owner')
        import torch
        from verl_speco.runtime.artifacts import publish_json
        free, total = torch.npu.mem_get_info()
        row = dict(run_id=run_id, generation=generation, role=role, rank=rank, stage=stage, details=dict(details or {}), pid=os.getpid(), utc_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(), visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), logical_device=int(torch.npu.current_device()), torch_allocated_bytes=int(torch.npu.memory_allocated()), torch_reserved_bytes=int(torch.npu.memory_reserved()), device_free_bytes=int(free), device_total_bytes=int(total), device_counters_include_other_processes=True, torch_counters_exclude_external_allocator=True)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{role}-rank-{rank:04d}-{row['pid']}-{row['monotonic_ns']}-{stage}.json"
        publish_json(path, row)
        return row
    except Exception as error:
        colocated_memory_diagnostics_LOGGER.warning('Colocated memory diagnostic unavailable: %s', error)
        return None
import asyncio, os

class ColocatedNativeGenerationMixin:

    def _generation_owner(self, request):
        if type(request) is not dict or request.get('run_id') != os.environ.get('SPECO_RUN_ID') or type(request.get('generation')) is not int or (str(request['generation']) != os.environ.get('SPECO_GENERATION')) or (type(request.get('placement_binding_sha256')) is not str) or (len(request['placement_binding_sha256']) != 64) or any((c not in '0123456789abcdef' for c in request['placement_binding_sha256'])) or (request['placement_binding_sha256'] != os.environ.get('SPECO_COLOCATED_PLACEMENT_SHA256')) or getattr(self, '_speco_colocated_failed', False):
            raise ValueError('native request belongs to a foreign or uncertain shared owner')

    async def speco_generate_complete(self, request):
        self._generation_owner(request)
        if set(request) != {'run_id', 'generation', 'placement_binding_sha256', 'request_id', 'prompt_token_ids', 'sampling_params'} or type(request['request_id']) is not str or (not request['request_id']) or (type(request['prompt_token_ids']) is not list) or (not request['prompt_token_ids']) or any((type(t) is not int or t < 0 for t in request['prompt_token_ids'])) or (type(request['sampling_params']) is not dict) or (request['sampling_params'].get('n', 1) != 1) or (getattr(self, '_speco_colocated_phase', None) != 'rollout') or await self.engine.is_paused() or await self.engine.is_sleeping():
            raise ValueError('owned shared generation requires awake rollout and a single typed request')
        active = getattr(self, '_speco_colocated_active_requests', None)
        if active is None:
            active = self._speco_colocated_active_requests = set()
        if request['request_id'] in active:
            raise ValueError('duplicate active native request ID')
        counters = getattr(self, '_speco_acceptance_counters', None)
        if counters is not None and active:
            raise RuntimeError('owned per-request acceptance metrics require serialized generation')
        baseline = counters.snapshot() if counters is not None else None
        active.add(request['request_id'])

        async def complete():
            from vllm import SamplingParams
            params = SamplingParams(**deepcopy(request['sampling_params']))
            final = None
            async for row in self.engine.generate(dict(prompt_token_ids=list(request['prompt_token_ids'])), params, request_id=request['request_id']):
                if row.finished:
                    if final is not None:
                        raise RuntimeError('native engine returned duplicate completed response')
                    final = row
            if final is None or final.request_id != request['request_id'] or list(final.prompt_token_ids) != request['prompt_token_ids'] or (len(final.outputs) != 1) or (final.outputs[0].finish_reason not in ('stop', 'length')):
                raise RuntimeError('native backend did not complete the exact owned prompt/request')
            tokens = list(final.outputs[0].token_ids)
            if not tokens or any((type(t) is not int or t < 0 for t in tokens)):
                raise RuntimeError('native complete response contains invalid token IDs')
            metric = {} if counters is None else dict(spec_decode=counters.since(baseline))
            return dict(**metric, run_id=request['run_id'], generation=request['generation'], placement_binding_sha256=request['placement_binding_sha256'], request_id=final.request_id, prompt_token_ids=list(final.prompt_token_ids), response_token_ids=tokens, finish_reason=final.outputs[0].finish_reason, complete=True)
        try:
            return await drain_owned_call(complete)
        finally:
            active.remove(request['request_id'])

    async def speco_reset_owned_prefix_cache(self, request):
        self._generation_owner(request)
        if set(request) != {'run_id', 'generation', 'placement_binding_sha256'} or getattr(self, '_speco_colocated_phase', None) != 'publication' or getattr(self, '_speco_colocated_active_requests', set()) or (await self.engine.is_paused() is not True):
            raise RuntimeError('shared prefix reset requires actual drained paused publication')
        kwargs = getattr(self, '_speco_native_prefix_reset_kwargs', None)
        if type(kwargs) is not dict:
            raise RuntimeError('native prefix reset lacks constructed engine cache configuration')
        result = await drain_owned_call(self.engine.reset_prefix_cache, **kwargs)
        if result is not True:
            raise RuntimeError('native scheduler did not actually clear prefix cache')
        return True

    async def speco_abort_owned_request(self, request):
        self._generation_owner(request)
        if set(request) != {'run_id', 'generation', 'placement_binding_sha256', 'request_id'}:
            raise ValueError('owned abort requires its exact request ID')
        await drain_owned_call(self.engine.abort, request['request_id'])
        return dict(request, complete=True)

    async def speco_shutdown_owned_engine(self, request):
        failed = getattr(self, '_speco_colocated_failed', False)
        self._speco_colocated_failed = False
        try:
            self._generation_owner(request)
        finally:
            self._speco_colocated_failed = failed
        if getattr(self, '_speco_colocated_active_requests', set()):
            raise RuntimeError('owned engine shutdown cannot race active generation')
        await drain_owned_call(self.engine.shutdown)
        if getattr(self, '_server_task', None) is not None:
            self._server_task.cancel()
        return dict(request, complete=True)
import json, os

def verify_native_placement(*, owners, run_id, generation, rank, incarnation, node_ip, visible_devices, logical_device, device_count, platform_uuid):
    if type(owners) is not list or not owners or type(rank) is not int or (rank < 0) or (type(generation) is not int) or (generation < 1) or (type(incarnation) is not str) or (not incarnation) or (type(logical_device) is not int) or (not 0 <= logical_device < len(owners)) or (type(device_count) is not int) or (device_count != len(owners)) or (rank % len(owners) != logical_device):
        raise ValueError('native shared worker lacks complete actual physical visibility')
    required = {'run_id', 'generation', 'role', 'rank', 'worker_incarnation', 'node_ip', 'device_uuid', 'placement_group_id', 'bundle_index'}
    if any((type(r) is not dict or set(r) != required or r['run_id'] != run_id or (r['generation'] != generation) or (type(r['generation']) is not int) or (r['role'] != 'actor') or (r['node_ip'] != node_ip) or (type(r['rank']) is not int) or (type(r['bundle_index']) is not int) or (type(r['device_uuid']) is not str) or (not r['device_uuid'].startswith('NPU-')) or (not r['device_uuid'][4:].isdigit()) or (type(r['worker_incarnation']) is not str) or (not r['worker_incarnation']) or (type(r['placement_group_id']) is not str) or (not r['placement_group_id']) for r in owners)) or len({r['rank'] for r in owners}) != len(owners) or len({r['device_uuid'] for r in owners}) != len(owners) or (visible_devices != ','.join((r['device_uuid'][4:] for r in owners))):
        raise ValueError('native visible physical cards differ from the reserved actor owner set')
    owner = owners[logical_device]
    if platform_uuid != owner['device_uuid'] or incarnation in {r['worker_incarnation'] for r in owners}:
        raise ValueError('native NPU placement or independent process incarnation differs')
    return dict(run_id=run_id, generation=generation, native_rank=rank, worker_incarnation=incarnation, node_ip=node_ip, device_uuid=platform_uuid, actor_rank=owner['rank'], actor_incarnation=owner['worker_incarnation'], placement_group_id=owner['placement_group_id'], bundle_index=owner['bundle_index'])

def observe_native_placement(worker):
    import torch
    from ray.util import get_node_ip_address
    from verl.workers.rollout.vllm_rollout.utils import get_device_uuid
    device = torch.npu.current_device()
    torch.npu.get_device_properties(device)
    return verify_native_placement(owners=json.loads(os.environ['SPECO_COLOCATED_NATIVE_ACTOR_OWNERS']), run_id=os.environ['SPECO_RUN_ID'], generation=int(os.environ['SPECO_GENERATION']), rank=worker.rank, incarnation=worker._speco_serving_incarnation, node_ip=get_node_ip_address(), visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), logical_device=device, device_count=torch.npu.device_count(), platform_uuid=get_device_uuid(device))
import json, os, sys

def preserve_hf_remote_module_environment(environment, model_config):
    """Make the actually loaded custom HF class importable before Ray unpickles it."""
    env = dict(environment)
    config = getattr(model_config, 'hf_config', None)
    if config is None:
        return env
    name = type(config).__module__
    if not name.startswith('transformers_modules.'):
        return env
    module = sys.modules.get(name)
    filename = getattr(module, '__file__', None)
    if type(filename) is not str or not Path(filename).is_file():
        raise ValueError('custom HF config lacks its actual importable cached module file')
    source = Path(filename).absolute()
    cache = source.parents[len(name.split('.')) - 1]
    if not (cache / 'transformers_modules').is_dir():
        raise ValueError('custom HF module file does not belong to its actual namespace cache')
    entries = [str(cache)] + [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p and p != str(cache)]
    env.update(PYTHONPATH=os.pathsep.join(entries), HF_MODULES_CACHE=str(cache))
    return env

class ColocatedServerClass:
    """Wrap Ray server construction without replacing VeRL launch or engine APIs."""

    def __init__(self, remote_class, *, binding, replica_rank, world_size, gpus_per_node, python, base_env, record_owned=lambda *_: None):
        checked = validate_shared_cards(run_id=binding['run_id'], generation=binding['generation'], actor=binding['actor'], draft=binding['draft'])
        if checked != binding or type(replica_rank) is not int or replica_rank < 0 or (type(world_size) is not int) or (world_size < 1) or (type(gpus_per_node) is not int) or (not 1 <= gpus_per_node <= 16) or world_size % gpus_per_node or ((replica_rank + 1) * world_size > len(binding['actor'])) or (type(python) is not str) or (not os.path.isabs(python)) or (type(base_env) is not dict) or any((type(k) is not str or type(v) is not str for k, v in base_env.items())) or (not callable(record_owned)):
            raise ValueError('owned native server needs validated physical pairing and complete runtime environment')
        self.remote_class, self.binding, self.replica_rank = (remote_class, deepcopy(binding), replica_rank)
        self.world, self.per_node, self.python, self.env = (world_size, gpus_per_node, python, dict(base_env))
        self.record, self.handles = (record_owned, [])

    def options(self, **options):
        outer = self

        class Bound:

            def remote(self, **arguments):
                rank = arguments.get('node_rank')
                if type(rank) is not int or not 0 <= rank < outer.world // outer.per_node or arguments.get('replica_rank') != outer.replica_rank or (arguments.get('gpus_per_node') != outer.per_node) or (arguments.get('nnodes') != outer.world // outer.per_node):
                    raise ValueError('VeRL server launch topology differs from the shared reservation')
                start = outer.replica_rank * outer.world + rank * outer.per_node
                owners = outer.binding['actor'][start:start + outer.per_node]
                if len({r['node_ip'] for r in owners}) != 1:
                    raise ValueError('one native server cannot span different reserved nodes')
                mask = ','.join((r['device_uuid'].removeprefix('NPU-') for r in owners))
                if any((not r['device_uuid'].startswith('NPU-') or not r['device_uuid'][4:].isdigit() for r in owners)):
                    raise ValueError('Ascend shared native server requires actual node-scoped physical NPU identities')
                if arguments.get('cuda_visible_devices') != mask:
                    raise ValueError('actual VeRL worker accelerator IDs differ from independently observed shared cards')
                env = dict(outer.env)
                for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'ASCEND_RT_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
                    env.pop(key, None)
                upstream = deepcopy(options.get('runtime_env', {}))
                if set(upstream) - {'env_vars'} or type(upstream.get('env_vars', {})) is not dict:
                    raise ValueError('native server upstream runtime environment contains an unowned override')
                env.update(upstream.get('env_vars', {}))
                env = preserve_hf_remote_module_environment(env, arguments.get('model_config'))
                env.update(SPECO_RUN_ID=outer.binding['run_id'], SPECO_GENERATION=str(outer.binding['generation']), SPECO_COLOCATED_PLACEMENT_SHA256=digest(outer.binding), SPECO_COLOCATED_NATIVE_ACTOR_OWNERS=json.dumps(owners, sort_keys=True), SPECO_COLOCATED_NODE_IP=owners[0]['node_ip'])
                runtime = dict(py_executable=outer.python, env_vars=env)
                opts = dict(options, num_cpus=0.1, runtime_env=runtime, name=f"{outer.binding['run_id']}-g{outer.binding['generation']}-native-{outer.replica_rank}-{rank}")
                handle = outer.remote_class.options(**opts).remote(**arguments)
                outer.handles.append(handle)
                outer.record(dict(run_id=outer.binding['run_id'], generation=outer.binding['generation'], replica_rank=outer.replica_rank, node_rank=rank, actor_id=handle._actor_id.hex(), node_ip=owners[0]['node_ip'], physical_actor_owners=deepcopy(owners)))
                return handle
        return Bound()

class ColocatedVeRLRolloutPort:

    def __init__(self, *, replica, verify_owned_phase, run_id, generation):
        if not callable(getattr(replica, 'sleep', None)) or not getattr(replica, 'servers', None) or (not callable(verify_owned_phase)) or (type(run_id) is not str) or (not run_id) or (type(generation) is not int) or (generation < 1):
            raise ValueError('colocated rollout port requires an initialized real VeRL replica')
        mode = getattr(getattr(replica, 'rollout_mode', None), 'value', None)
        if mode not in ('hybrid', 'colocated'):
            raise ValueError('colocated rollout requires VeRL hybrid/colocated mode')
        if getattr(getattr(replica, 'config', None), 'free_cache_engine', None) is not True:
            raise ValueError('colocated rollout requires actual cache/weight release support')
        self.replica, self.verify, self.run_id, self.generation = (replica, verify_owned_phase, run_id, generation)
        self.failed = False

    def _request(self, request, kind):
        required = {'run_id', 'generation', 'transition_epoch', 'placement_binding_sha256', kind}
        if kind == 'tags':
            required.add('resume_generation')
        if self.failed or type(request) is not dict or set(request) != required or (request['run_id'] != self.run_id) or (type(request['generation']) is not int) or (request['generation'] != self.generation) or (type(request['transition_epoch']) is not int) or (request['transition_epoch'] < 1) or (type(request['placement_binding_sha256']) is not str) or (len(request['placement_binding_sha256']) != 64):
            raise ValueError('colocated rollout transition has a stale or malformed owner')

    async def sleep(self, request):
        self._request(request, 'sleep_level')
        if type(request['sleep_level']) is not int or request['sleep_level'] != 1:
            raise ValueError('colocated Ascend sleep must preserve the full target/draft weights at level one')
        try:
            for server in self.replica.servers:
                await drain_owned_call(server.speco_prepare_colocated_sleep.remote, request=request)
            await drain_owned_call(self.replica.sleep)
            if await drain_owned_call(self.verify, 'sleeping', request) is not True:
                raise RuntimeError('owned native engine has not actually completed sleep')
            return dict(request, complete=True)
        except BaseException:
            self.failed = True
            raise

    async def wake(self, request):
        self._request(request, 'tags')
        if request['tags'] not in (['weights'], ['kv_cache'], ['weights', 'kv_cache']):
            raise ValueError('shared publication requires explicit weights/KV wake tags')
        if type(request['resume_generation']) is not bool:
            raise ValueError('shared wake requires an explicit generation resume fence')
        if request['resume_generation'] and 'kv_cache' not in request['tags']:
            raise ValueError('shared generation cannot resume before KV is awake')
        try:
            for server in self.replica.servers:
                await drain_owned_call(server.speco_wake_colocated_tags.remote, request=request)
            phase = 'awake:' + ','.join(request['tags']) + ('' if request['resume_generation'] else ':paused')
            if await drain_owned_call(self.verify, phase, request) is not True:
                raise RuntimeError('owned native engine did not finish its exact tagged wake')
            return dict(request, complete=True)
        except BaseException:
            self.failed = True
            raise
from types import SimpleNamespace

class MultiNodeColocatedNativeEngine:

    def __init__(self, *, replica, binding):
        if not replica.servers or replica.rollout_mode.value not in ('hybrid', 'colocated'):
            raise ValueError('shared engine requires an initialized native VeRL replica')
        self.replica, self.binding = (replica, deepcopy(binding))
        self.owner = dict(run_id=binding['run_id'], generation=binding['generation'], placement_binding_sha256=digest(binding))
        self.closed = False
        self._speco_acceptance_by_request = {}

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if self.closed:
            raise RuntimeError('native engine was already closed')
        return await drain_owned_call(self.replica.servers[0].collective_rpc.remote, method=method, timeout=timeout, args=args, kwargs=kwargs)

    def generate(self, prompt, sampling_params, request_id):
        if self.closed or type(prompt) is not dict or set(prompt) != {'prompt_token_ids'}:
            raise ValueError('owned shared generation requires its actual token prompt')
        names = ('temperature', 'top_p', 'top_k', 'max_tokens', 'min_tokens', 'ignore_eos', 'seed', 'n', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'stop', 'stop_token_ids', 'logprobs', 'prompt_logprobs', 'skip_special_tokens', 'detokenize')
        params = {name: deepcopy(getattr(sampling_params, name)) for name in names if hasattr(sampling_params, name)}
        request = dict(self.owner, request_id=request_id, prompt_token_ids=list(prompt['prompt_token_ids']), sampling_params=params)

        async def stream():
            row = await drain_owned_call(self.replica.servers[0].speco_generate_complete.remote, request=request)
            required = set(self.owner) | {'request_id', 'prompt_token_ids', 'response_token_ids', 'finish_reason', 'complete'}
            if type(row) is dict and 'spec_decode' in row:
                required.add('spec_decode')
            if type(row) is not dict or set(row) != required or row['complete'] is not True or any((type(row[k]) is not type(v) or row[k] != v for k, v in self.owner.items())) or (row['request_id'] != request_id) or (row['prompt_token_ids'] != request['prompt_token_ids']) or (row['finish_reason'] not in ('stop', 'length')) or (type(row['response_token_ids']) is not list) or (not row['response_token_ids']) or any((type(t) is not int or t < 0 for t in row['response_token_ids'])):
                raise RuntimeError('native VeRL server returned foreign/incomplete actual generation')
            if 'spec_decode' in row:
                from verl_speco.integration.multinode_online import summarize_serving_acceptance
                summarize_serving_acceptance([row['spec_decode']])
                if request_id in self._speco_acceptance_by_request:
                    raise RuntimeError('completed request acceptance was already recorded')
                self._speco_acceptance_by_request[request_id] = deepcopy(row['spec_decode'])
            yield SimpleNamespace(request_id=row['request_id'], prompt_token_ids=row['prompt_token_ids'], finished=True, outputs=[SimpleNamespace(token_ids=row['response_token_ids'], finish_reason=row['finish_reason'])])
        return stream()

    def step_acceptance_metrics(self, request_ids):
        from verl_speco.integration.multinode_online import summarize_serving_acceptance
        ids = tuple(request_ids)
        if not ids or len(ids) != len(set(ids)) or any((i not in self._speco_acceptance_by_request for i in ids)):
            raise RuntimeError('step acceptance lacks unique completed request observations')
        return summarize_serving_acceptance([self._speco_acceptance_by_request[i] for i in ids])

    async def reset_prefix_cache(self):
        return await drain_owned_call(self.replica.servers[0].speco_reset_owned_prefix_cache.remote, request=dict(self.owner))

    async def abort(self, request_id):
        return await drain_owned_call(self.replica.servers[0].speco_abort_owned_request.remote, request=dict(self.owner, request_id=request_id))

    async def close(self):
        if self.closed:
            return
        import ray
        for server in self.replica.servers:
            row = await drain_owned_call(server.speco_shutdown_owned_engine.remote, request=dict(self.owner))
            if type(row) is not dict or row != dict(self.owner, complete=True):
                raise RuntimeError('native server shutdown lacks exact owner receipt')
        for server in self.replica.servers:
            ray.kill(server, no_restart=True)
        self.closed = True
from contextlib import contextmanager
GRAPH_WORKER = 'verl_speco.integration.accelerator_adapters.SpecoMultiNodeDSparkGraphPublicationWorker'
BASE_GRAPH_WORKER = 'verl_speco.integration.accelerator_adapters.SpecoMultiNodeDSparkGraphWorker'
EAGER_WORKER = 'verl_speco.integration.accelerator_adapters.SpecoMultiNodeTargetUpdateWorker'

def colocated_rollout_engine_kwargs(engine_kwargs):
    """Preserve the selected graph recipe across VeRL's RolloutConfig boundary."""
    result = deepcopy(engine_kwargs)
    result.pop('model', None)
    result['enable_sleep_mode'] = True
    if result.get('enforce_eager') is False:
        if result.get('worker_cls') != GRAPH_WORKER or result.get('compilation_config') != dict(cudagraph_mode='FULL_DECODE_ONLY', cudagraph_capture_sizes=[9]):
            raise ValueError('native colocated rollout lost its explicit FULL K8 capture recipe')
    elif result.get('enforce_eager') is True:
        result.setdefault('compilation_config', dict(cudagraph_mode='FULL_DECODE_ONLY'))
    else:
        raise ValueError('native colocated rollout needs an explicit eager or graph selection')
    return result

def colocated_graph_engine_kwargs(config, baseline):
    result = deepcopy(baseline)
    mode = config.get('rollout_graph_mode')
    if mode is None:
        return result
    if mode != 'FULL_DECODE_ONLY' or config.get('colocated') is not True:
        raise ValueError('colocated graph entry requires explicit FULL_DECODE_ONLY')
    if result.get('worker_cls') not in (EAGER_WORKER, GRAPH_WORKER) or result.get('max_num_seqs') != 1 or result.get('speculative_config', {}).get('method') != 'dflash' or (result['speculative_config'].get('num_speculative_tokens') != 8) or (result['speculative_config'].get('enforce_eager') is not True):
        raise ValueError('graph publication entry lacks verified single-request K8 carrier')
    result.update(enforce_eager=False, worker_cls=GRAPH_WORKER, compilation_config=dict(cudagraph_mode='FULL_DECODE_ONLY', cudagraph_capture_sizes=[9]))
    return result

@contextmanager
def dedicated_colocated_dspark_config(kwargs):
    from verl_speco.integration.accelerator_speculation import dedicated_speco_dspark_config as dedicated_speco_dspark_config
    from verl_speco.integration.accelerator_speculation import dedicated_speco_dspark_graph_config as dedicated_speco_dspark_graph_config
    if kwargs.get('enforce_eager') is True:
        with dedicated_speco_dspark_config(kwargs):
            yield
        return
    if kwargs.get('enforce_eager') is not False or kwargs.get('worker_cls') != GRAPH_WORKER or kwargs.get('compilation_config', {}).get('cudagraph_mode') != 'FULL_DECODE_ONLY':
        raise ValueError('colocated publisher requires its exact FULL graph worker')
    checked = dict(kwargs, worker_cls=BASE_GRAPH_WORKER)
    with dedicated_speco_dspark_graph_config(checked):
        yield
from dataclasses import asdict, is_dataclass
import threading
from verl_speco.runtime.artifacts import publish_json, published_json_paths, read_published_json
from verl_speco.trainer.distributed_checkpoint import optimizer_transaction_record_from_dict as _record_from_dict, StepState
from verl_speco.transport.control import LeaseRegistry, LeaseState
from verl_speco.transport.streaming import TrainingStreamCoordinator

def joint_checkpoint_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def joint_checkpoint_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value)):
        raise ValueError('joint checkpoint requires SHA256 identities')
    return value

def joint_checkpoint_positive(value):
    if type(value) is not int or value < 1:
        raise ValueError('joint checkpoint requires positive counters')
    return value

def joint_checkpoint_rows(rows, owners, id_key, phase=None):
    if not owners or not isinstance(rows, list) or len(rows) != len(owners) or any((not isinstance(row, dict) or type(row.get(id_key)) is not type(next(iter(owners))) for row in rows)) or ({row.get(id_key) for row in rows} != set(owners)):
        raise ValueError('joint checkpoint is missing exact role membership')
    for row in rows:
        if row.get('worker_incarnation') != owners[row[id_key]] or (phase and row.get(phase) is not True):
            raise ValueError('joint checkpoint has a stale role incarnation or phase')
    return rows

class MultiNodeJointCheckpointStore:
    """One immutable complete marker per generation/step; incomplete is fenced.

    The caller owns the coordinator and the role safe-point locks. Exclusive
    directory creation also prevents competing writers from replacing a marker
    on shared filesystems that cannot rename. Existing incomplete directories
    require recovery; they are never adopted or overwritten by a retry.
    """

    def __init__(self, *, run_root, run_id, generation, fingerprints, actor_incarnations, draft_incarnations, rollout_incarnations):
        self.root = Path(run_root)
        if not self.root.is_absolute() or not isinstance(run_id, str) or (not run_id.strip()):
            raise ValueError('joint checkpoint requires an absolute owned run root')
        self.root = self.root.resolve()
        self.run_id = run_id
        self.generation = joint_checkpoint_positive(generation)
        self.fingerprints = deepcopy(fingerprints)
        for name in ('actor_model', 'draft_model', 'target_model', 'tokenizer', 'recipe'):
            joint_checkpoint_sha(self.fingerprints[name])
        if self.fingerprints.get('rl_roles') != ['actor']:
            raise ValueError('joint checkpoint needs the supported actor-only GRPO role contract')
        if type(self.fingerprints.get('tq_schema')) is not int or self.fingerprints['tq_schema'] != 3 or (not isinstance(self.fingerprints.get('target_layers'), list)) or (not self.fingerprints['target_layers']) or any((type(i) is not int or i < 0 for i in self.fingerprints['target_layers'])) or (len(set(self.fingerprints['target_layers'])) != len(self.fingerprints['target_layers'])):
            raise ValueError('joint checkpoint has an invalid feature contract')
        self.actors = dict(actor_incarnations)
        self.drafts = dict(draft_incarnations)
        self.rollout = dict(rollout_incarnations)
        for owners, id_type in ((self.actors, int), (self.drafts, str), (self.rollout, int)):
            if not owners or any((type(i) is not id_type or not isinstance(v, str) or (not v) for i, v in owners.items())):
                raise ValueError('joint checkpoint requires explicit exact role incarnations')
        self.lock = threading.RLock()

    def _owned(self, path):
        path = Path(path)
        if not path.is_absolute() or not path.resolve().is_relative_to(self.root):
            raise ValueError('joint checkpoint artifact is outside the owned run')
        return path

    def _candidate(self, candidate, role, world_size):
        if not isinstance(candidate, dict) or set(candidate) != {'uri', 'checksum', 'checkpoint_generation'} or type(candidate['checkpoint_generation']) is not int or (candidate['checkpoint_generation'] != self.generation):
            raise ValueError('joint checkpoint DCP generation differs')
        path = self._owned(candidate['uri']) / 'manifest.json'
        if path.stat().st_size > 2 * 1024 ** 2:
            raise ValueError('joint DCP manifest exceeds its metadata bound')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != joint_checkpoint_sha(candidate['checksum']):
            raise ValueError('joint DCP manifest checksum differs')
        manifest = json.loads(raw)
        if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or type(manifest.get('generation')) is not int or (manifest['generation'] != self.generation) or (type(manifest.get('world_size')) is not int) or (manifest['world_size'] != world_size) or (manifest.get('model_fingerprint') != self.fingerprints[role + '_model']) or (not manifest.get('topology_fingerprint')) or (not manifest.get('files')):
            raise ValueError('joint DCP model or topology binding differs')
        required = {'dcp/.metadata'} | {f'extra-rank-{i:04d}.pt' for i in range(world_size)}
        if not required.issubset(manifest['files']):
            raise ValueError('joint DCP manifest has incomplete rank-local state')
        for name, sha in manifest['files'].items():
            if not name or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('joint DCP file escaped its owner')
            joint_checkpoint_sha(sha)
            if not self._owned(path.parent / name).is_file():
                raise ValueError('joint DCP manifest references a missing payload')
        if not any((name.startswith('dcp/') and name.endswith('.distcp') for name in manifest['files'])):
            raise ValueError('joint DCP manifest lacks tensor payloads')
        return manifest

    def build(self, *, versions, actor_update_receipt, actor_rows, draft_rows, draft_state_rows, control_snapshot, target_acks, draft_acks, target_head, snapshot, session_idle, pending_commits):
        v = asdict(versions) if is_dataclass(versions) else deepcopy(versions)
        if set(v) != {'run_id', 'generation', 'rl_step', 'target_revision', 'data_version', 'draft_revision'} or v['run_id'] != self.run_id or v['generation'] != self.generation or (session_idle is not True) or (type(pending_commits) is not int) or (pending_commits != 0):
            raise ValueError('joint checkpoint requires the current drained session safe point')
        for name in ('rl_step', 'target_revision', 'data_version', 'draft_revision'):
            joint_checkpoint_positive(v[name])
        update = actor_update_receipt
        if update.get('run_id') != self.run_id or update.get('generation') != self.generation or update.get('rl_step') != v['rl_step'] or (update.get('target_revision') != v['target_revision'] - 1) or (update.get('actual_VeRL_actor') is not True) or (update.get('optimizer_committed') is not True) or ({str(k): value for k, value in update.get('worker_incarnations', {}).items()} != {str(k): value for k, value in self.actors.items()}):
            raise ValueError('joint actor update receipt differs from the actual serving revision')
        optimizer_step = joint_checkpoint_positive(update['optimizer_step'])
        actors = joint_checkpoint_rows(actor_rows, self.actors, 'rank', 'saved')
        for row in actors:
            if row.get('run_id') != self.run_id or row.get('generation') != self.generation or row.get('optimizer_step') != optimizer_step or (row.get('complete_local_model_optimizer_scheduler_rng') is not True):
                raise ValueError('joint actor state is not the committed complete optimizer step')
            joint_checkpoint_sha(row['state_sha256'])
        actor_candidate = actors[0]['candidate']
        if any((type(row.get('optimizer_step')) is not int or type(row.get('generation')) is not int for row in actors)):
            raise ValueError('joint actor counters must remain integers')
        if any((row.get('candidate') != actor_candidate for row in actors)):
            raise ValueError('joint actor ranks saved different candidates')
        actor_header = self._candidate(actor_candidate, 'actor', len(self.actors))
        drafts = joint_checkpoint_rows(draft_rows, self.drafts, 'worker_id', 'trained')
        joint_checkpoint_rows(draft_state_rows, self.drafts, 'worker_id')
        leaders = [row for row in drafts if row.get('is_publish_leader') is True]
        if len(leaders) != 1:
            raise ValueError('joint draft checkpoint needs exactly one durable leader')
        receipt = leaders[0]
        for row in drafts:
            joint_checkpoint_positive(row['generation'])
            if type(row.get('target_version')) is not int:
                raise ValueError('joint draft head revision must remain an integer')
            if row.get('run_id') != self.run_id or row.get('generation') != self.generation or type(row.get('source_global_step')) is not int or (not 1 <= row['source_global_step'] <= v['rl_step']) or (type(row.get('data_version')) is not int) or (not 1 <= row['data_version'] <= v['data_version']) or (not 0 <= row['target_version'] < v['target_revision']) or (row.get('source_global_step') != receipt['source_global_step']) or (row.get('data_version') != receipt['data_version']) or (row.get('target_version') != receipt['target_version']) or (row.get('plan_id') != receipt['plan_id']) or (row.get('candidate') != receipt['candidate']) or (row.get('optimizer_step') != receipt['optimizer_step']) or (row.get('online_plan_sha256') != receipt['online_plan_sha256']) or (row.get('trainable_parameter_sha256') != receipt['trainable_parameter_sha256']):
                raise ValueError('joint draft ranks disagree on the committed plan')
            joint_checkpoint_positive(row['optimizer_step'])
        for row in draft_state_rows:
            joint_checkpoint_positive(row['generation'])
            joint_checkpoint_positive(row['optimizer_step'])
            if row.get('run_id') != self.run_id or row.get('generation') != self.generation or row.get('optimizer_step') != receipt['optimizer_step'] or (row.get('complete_local_model_optimizer_scheduler_rng') is not True):
                raise ValueError('joint draft audit lacks complete committed state')
            joint_checkpoint_sha(row['state_sha256'])
        draft_header = self._candidate(receipt['candidate'], 'draft', len(self.drafts))
        step_id = 'online-' + joint_checkpoint_sha(receipt['online_plan_sha256'])
        journal_path = self._owned(Path(receipt['candidate']['uri']).parents[1] / 'journal')
        durable_receipt = read_published_json(journal_path.parent / 'receipts' / (step_id + '.json'), max_bytes=2 * 1024 ** 2)
        if joint_checkpoint_canonical(durable_receipt) != joint_checkpoint_canonical(receipt):
            raise ValueError('joint draft leader differs from its durable receipt')
        journal_payload = read_published_json(journal_path / ('step-' + hashlib.sha256(step_id.encode()).hexdigest() + '.json'), max_bytes=2 * 1024 ** 2)
        journal = _record_from_dict(journal_payload)
        tx = journal.transaction
        if journal.state is not StepState.COMMITTED or asdict(journal.candidate_checkpoint) != receipt['candidate'] or tx.optimizer_step_id != step_id or (tx.plan_id != receipt['plan_id']) or (tx.generation != self.generation) or (tx.stream_scope != self.run_id) or (journal.model_fingerprint != draft_header['model_fingerprint']) or (journal.topology_fingerprint != draft_header['topology_fingerprint']):
            raise ValueError('joint draft journal differs from the committed candidate')
        if receipt.get('snapshot') != snapshot or snapshot.get('draft_revision') != v['draft_revision']:
            raise ValueError('joint serving snapshot differs from the trained draft')
        manifest = read_published_json(self._owned(snapshot['uri']) / 'manifest.json', max_bytes=2 * 1024 ** 2)
        if hashlib.sha256(joint_checkpoint_canonical(manifest)).hexdigest() != joint_checkpoint_sha(snapshot['checksum']) or manifest.get('schema_version') != 1 or manifest.get('kind') != 'draft_trainable_parameters' or (manifest.get('source_step_id') != step_id) or (manifest.get('generation') != self.generation) or (manifest.get('draft_revision') != v['draft_revision']) or (manifest.get('target_model_fingerprint') != self.fingerprints['target_model']) or (manifest.get('tokenizer_fingerprint') != self.fingerprints['tokenizer']) or (manifest.get('trainable_parameter_sha256') != receipt['trainable_parameter_sha256']):
            raise ValueError('joint draft snapshot provenance differs')
        geometry = manifest.get('model_config')
        if not isinstance(geometry, dict) or hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest() != self.fingerprints['draft_model'] or geometry.get('target_layer_ids') != self.fingerprints['target_layers'] or (manifest.get('namespace_prefix') != 'training_model.'):
            raise ValueError('joint draft geometry or selected target layers differ')
        control = deepcopy(control_snapshot)
        joint_checkpoint_positive(control['generation'])
        if control.get('schema_version') != 1 or control.get('run_id') != self.run_id or control.get('generation') != self.generation:
            raise ValueError('joint control generation differs')
        leases = LeaseRegistry()
        leases.restore(control['leases'])
        if any((item['state'] not in (LeaseState.COMMITTED.value, LeaseState.CANCELLED.value) for item in control['leases']['leases'])):
            raise ValueError('joint control has an active lease')
        by_lease = {item['lease_id']: item for item in control['leases']['leases']}
        if any((lease not in by_lease or by_lease[lease]['state'] != LeaseState.COMMITTED.value or by_lease[lease]['payload_released'] is not True for lease in tx.lease_ids)):
            raise ValueError('joint draft candidate has uncommitted or retained leases')
        committed = control['leases']['committed_steps'].get(step_id)
        if committed != {'transaction_set_hash': tx.transaction_set_hash, 'lease_ids': list(tx.lease_ids)}:
            raise ValueError('joint control transaction set differs from the draft journal')
        stream_payload = receipt['stream_snapshot']
        stream = TrainingStreamCoordinator(self.run_id)
        stream.restore(stream_payload)
        if tuple(sorted(stream_payload['committed_optimizer_steps'].get(step_id, ()))) != tx.transaction_ids:
            raise ValueError('joint stream cursor does not contain the committed draft transaction set')
        for ack in joint_checkpoint_rows(target_acks, self.rollout, 'rank', 'applied'):
            for field in ('generation', 'rl_step', 'target_revision'):
                joint_checkpoint_positive(ack[field])
            if any((ack.get(k) != v[k] for k in ('run_id', 'generation', 'rl_step', 'target_revision'))) or ack.get('kv_cache_cleared') is not True or ack.get('draft_target_head_synced') is not True or (ack.get('actor_weight_manifest_sha256') != target_head.get('actor_weight_manifest_sha256')):
                raise ValueError('joint target serving receipt differs')
        for field in ('generation', 'target_revision'):
            joint_checkpoint_positive(target_head[field])
        if any((target_head.get(k) != v[k] for k in ('run_id', 'generation', 'target_revision'))) or target_head.get('tokenizer_fingerprint') != self.fingerprints['tokenizer']:
            raise ValueError('joint target head binding differs')
        if not self._owned(target_head['uri']).is_file():
            raise ValueError('joint target head is missing')
        joint_checkpoint_sha(target_head['sha256'])
        joint_checkpoint_sha(target_head['actor_weight_manifest_sha256'])
        for ack in joint_checkpoint_rows(draft_acks, self.rollout, 'rank', 'applied'):
            joint_checkpoint_positive(ack['generation'])
            joint_checkpoint_positive(ack['draft_revision'])
            if ack.get('run_id') != self.run_id or ack.get('generation') != self.generation or ack.get('draft_revision') != v['draft_revision'] or (ack.get('snapshot_checksum') != snapshot['checksum']):
                raise ValueError('joint draft serving receipt differs')
        payload = dict(schema_version=1, kind='speco_joint_safe_point', versions=v, fingerprints=self.fingerprints, actor_update_receipt=actor_update_receipt, actor_rows=actor_rows, draft_rows=draft_rows, draft_state_rows=draft_state_rows, control_snapshot=control, target_acks=target_acks, draft_acks=draft_acks, target_head=target_head, snapshot=snapshot, draft_journal=journal_payload, actor_manifest=actor_header, draft_manifest=draft_header, session_idle=True, pending_commits=0)
        return json.loads(joint_checkpoint_canonical(payload))

    def commit(self, payload):
        with self.lock:
            if payload.get('kind') != 'speco_joint_safe_point' or payload.get('fingerprints') != self.fingerprints:
                raise ValueError('joint checkpoint payload identity differs')
            rebuilt = self.build(**{key: payload[key] for key in ('versions', 'actor_update_receipt', 'actor_rows', 'draft_rows', 'draft_state_rows', 'control_snapshot', 'target_acks', 'draft_acks', 'target_head', 'snapshot', 'session_idle', 'pending_commits')})
            if rebuilt != payload:
                raise ValueError('joint checkpoint artifacts changed before the complete marker')
            version = payload['versions']
            directory = self.root / 'joint-checkpoints' / f"generation-{self.generation}-step-{version['rl_step']}"
            path = directory / 'complete.json'
            if directory.exists():
                if published_json_paths(directory, path.name) and read_published_json(path) == payload:
                    return dict(uri=str(path), checksum=hashlib.sha256(joint_checkpoint_canonical(payload)).hexdigest())
                raise RuntimeError('joint checkpoint owner has an incomplete or conflicting publication')
            directory.parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir(exist_ok=False)
            publish_json(path, payload)
            if read_published_json(path) != payload:
                raise RuntimeError('joint complete marker did not survive readback')
            return dict(uri=str(path), checksum=hashlib.sha256(joint_checkpoint_canonical(payload)).hexdigest())

    def load(self, binding):
        """Revalidate every metadata binding before any role mutates a model.

        Role restore ports must still hash the complete DCP/head/snapshot tensor
        payloads and compare all saved local state receipts on fresh owners.
        """
        path = self._owned(binding['uri'])
        payload = read_published_json(path)
        if hashlib.sha256(joint_checkpoint_canonical(payload)).hexdigest() != joint_checkpoint_sha(binding['checksum']):
            raise ValueError('joint complete marker checksum differs')
        v = payload['versions']
        expected = self.root / 'joint-checkpoints' / f"generation-{self.generation}-step-{v['rl_step']}" / 'complete.json'
        if path != expected or payload.get('fingerprints') != self.fingerprints:
            raise ValueError('joint complete marker owner or recipe differs')
        rebuilt = self.build(**{key: payload[key] for key in ('versions', 'actor_update_receipt', 'actor_rows', 'draft_rows', 'draft_state_rows', 'control_snapshot', 'target_acks', 'draft_acks', 'target_head', 'snapshot', 'session_idle', 'pending_commits')})
        if rebuilt != payload:
            raise ValueError('joint checkpoint dependencies changed after publication')
        return payload
from verl_speco.runtime.artifacts import read_published_json
from verl_speco.transport.control import MultiNodeJointReadyControl as MultiNodeJointReadyControl, control_snapshot_sha256

def joint_draft_recipe(config, *, world_size, optimizer):
    """Logical recipe remains stable when only the physical generation changes."""
    if type(world_size) is not int or world_size < 1:
        raise ValueError('joint draft requires its exact DP membership')
    recipe = dict(run_id=config['run_id'], model_config=config['model_config'], target_model_fingerprint=config['target_model_fingerprint'], tokenizer_fingerprint=config['tokenizer_fingerprint'], world_size=world_size, mesh=dict(dp_replicate=world_size, dp_shard=1, tp=1, pp=1, sp=1, ep=1), wrapper='fsdp2', optimizer_class=type(optimizer).__module__ + '.' + type(optimizer).__qualname__, optimizer_defaults=deepcopy(optimizer.defaults), total_steps=config['total_steps'])
    if type(recipe['total_steps']) is not int or recipe['total_steps'] < 1:
        raise ValueError('joint draft scheduler recipe requires a positive step count')
    logical = hashlib.sha256(joint_checkpoint_canonical(recipe)).hexdigest()
    generation = config['generation']
    if type(generation) is not int or generation < 1:
        raise ValueError('joint draft requires a positive physical generation')
    topology = hashlib.sha256(joint_checkpoint_canonical(dict(logical=logical, generation=generation))).hexdigest()
    model = hashlib.sha256(json.dumps(config['model_config'], sort_keys=True).encode()).hexdigest()
    return dict(model_fingerprint=model, logical_checkpoint_fingerprint=logical, topology_fingerprint=topology)

def open_joint_parent(config, binding):
    """Reopen the old complete marker using current explicit recipe identities."""
    if binding != config.get('joint_restore_binding'):
        raise ValueError('joint parent binding differs from the configured recovery owner')
    root = Path(config['joint_run_root']).resolve()
    path = Path(binding['uri'])
    if not path.is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError('joint parent complete marker is outside the owned run')
    raw = read_published_json(path, max_bytes=2 * 1024 ** 2)
    if hashlib.sha256(joint_checkpoint_canonical(raw)).hexdigest() != binding['checksum']:
        raise ValueError('joint parent complete marker checksum differs')
    versions = raw['versions']
    for field in ('generation', 'rl_step', 'target_revision', 'data_version', 'draft_revision'):
        if type(versions.get(field)) is not int or versions[field] < 1:
            raise ValueError('joint parent version counters must remain positive integers')
    if versions['run_id'] != config['run_id'] or type(config['generation']) is not int or (not versions['generation'] < config['generation']):
        raise ValueError('joint restore requires a strictly newer generation of the same run')
    store = MultiNodeJointCheckpointStore(run_root=root, run_id=config['run_id'], generation=versions['generation'], fingerprints=config['joint_fingerprints'], actor_incarnations={r['rank']: r['worker_incarnation'] for r in raw['actor_rows']}, draft_incarnations={r['worker_id']: r['worker_incarnation'] for r in raw['draft_state_rows']}, rollout_incarnations={r['rank']: r['worker_incarnation'] for r in raw['target_acks']})
    parent = store.load(binding)
    control = MultiNodeJointReadyControl(run_id=config['run_id'], generation=config['generation'], partition=config['partition'], capacity_bytes=config['joint_control_capacity_bytes'])
    control.restore_joint_parent(parent['control_snapshot'], expected_sha256=control_snapshot_sha256(parent['control_snapshot']))
    return (parent, control.snapshot_quiescent())

def read_joint_target_head(binding, *, run_root, shape):
    """Hash complete immutable head bytes before loading any model state."""
    import torch
    from safetensors.torch import load_file
    path = Path(binding['uri'])
    if not path.is_absolute() or not path.resolve().is_relative_to(Path(run_root).resolve()):
        raise ValueError('joint target head is outside the owned run')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != binding['sha256']:
            raise ValueError('joint immutable target head payload checksum differs')
    weights = load_file(str(path), device='cpu')
    head = weights.get('lm_head.weight')
    if set(weights) != {'lm_head.weight'} or head is None or tuple(head.shape) != tuple(shape) or (head.dtype != torch.bfloat16):
        raise ValueError('joint target head geometry or dtype differs')
    flat = head.reshape(-1)
    for start in range(0, flat.numel(), 8 * 1024 ** 2):
        if not torch.isfinite(flat[start:start + 8 * 1024 ** 2]).all():
            raise ValueError('joint immutable target head contains nonfinite values')
    return head

def validate_joint_owners(parent, owners):
    if not isinstance(owners, dict) or set(owners) != {'actor', 'draft', 'rollout'}:
        raise ValueError('joint recovery requires all explicit fresh role memberships')
    for role, field, identity, key_type in (('actor', 'actor_rows', 'rank', int), ('draft', 'draft_state_rows', 'worker_id', str), ('rollout', 'target_acks', 'rank', int)):
        prior = {r[identity]: r['worker_incarnation'] for r in parent[field]}
        current = owners[role]
        if not isinstance(current, dict) or set(current) != set(prior) or any((type(k) is not key_type or not isinstance(v, str) or (not v) or (v == prior[k]) for k, v in current.items())) or (len(set(current.values())) != len(current)):
            raise ValueError('joint recovery membership is incomplete, stale or duplicated')

def validate_joint_release(parent, proof, owners, *, run_id, generation, binding_checksum):
    """Every fresh role must actually acknowledge the same recovery safe point.

    TargetWireDigest salts its payload digest with the generation/operation
    header. The fresh source digest therefore need not equal the historical
    digest. Complete actor state must equal the parent on every logical rank;
    all fresh receiver digests must equal the fresh source's actual digest.
    """
    validate_joint_owners(parent, owners)
    if proof.get('run_id') != run_id or type(proof.get('generation')) is not int or proof['generation'] != generation or (proof.get('binding_checksum') != binding_checksum) or (proof.get('native_prefix_cache_reset') is not True):
        raise ValueError('joint recovery release has stale or incomplete safe-point identity')
    versions = parent['versions']
    for role, identity, field, phase in (('actor', 'rank', 'actor_rows', 'restored'), ('draft', 'worker_id', 'draft_rows', 'restored'), ('rollout', 'rank', 'target_acks', 'applied')):
        expected = owners[role]
        rows = proof[field]
        key_type = type(next(iter(expected)))
        if not isinstance(rows, list) or len(rows) != len(expected) or any((not isinstance(r, dict) or type(r.get(identity)) is not key_type for r in rows)) or ({r.get(identity) for r in rows} != set(expected)):
            raise ValueError('joint recovery release is missing a fresh role member')
        for row in rows:
            if row.get('worker_incarnation') != expected[row[identity]] or row.get('run_id') != run_id or type(row.get('generation')) is not int or (row['generation'] != generation) or (row.get(phase) is not True):
                raise ValueError('joint recovery release has a stale role owner or phase')
    for field, parent_field, identity in (('actor_rows', 'actor_rows', 'rank'), ('draft_rows', 'draft_state_rows', 'worker_id')):
        expected = {r[identity]: r for r in parent[parent_field]}
        for row in proof[field]:
            prior = expected[row[identity]]
            if row.get('complete_local_model_optimizer_scheduler_rng') is not True or row.get('state_sha256') != prior['state_sha256'] or type(row.get('optimizer_step')) is not int or (row['optimizer_step'] != prior['optimizer_step']):
                raise ValueError('joint recovery release lacks the complete restored role state')
    source = proof['source_wire']
    sha = source.get('actor_weight_manifest_sha256')
    if not isinstance(sha, str) or len(sha) != 64 or any((c not in '0123456789abcdef' for c in sha)) or (source.get('wire_tensor_count') != 35869) or (type(source.get('wire_nbytes')) is not int) or (source['wire_nbytes'] < 1) or (source.get('head_sha256') != parent['target_head']['sha256']):
        raise ValueError('joint recovery release lacks the actual complete fresh source wire')
    for row in proof['target_acks']:
        if type(row.get('target_revision')) is not int or type(row.get('rl_step')) is not int or row.get('target_revision') != versions['target_revision'] or (row.get('rl_step') != versions['rl_step']) or (row.get('actor_weight_manifest_sha256') != sha) or (row.get('kv_cache_cleared') is not True) or (row.get('draft_target_head_synced') is not True):
            raise ValueError('joint recovery release has an incomplete target weight/cache/head ACK')
    rows = proof['draft_acks']
    expected = owners['rollout']
    if not isinstance(rows, list) or len(rows) != len(expected) or any((not isinstance(r, dict) or type(r.get('rank')) is not int for r in rows)) or ({r.get('rank') for r in rows} != set(expected)):
        raise ValueError('joint recovery release lacks all fresh serving draft ACKs')
    for row in rows:
        if row.get('worker_incarnation') != expected[row['rank']] or row.get('run_id') != run_id or type(row.get('generation')) is not int or (row['generation'] != generation) or (row.get('applied') is not True) or (type(row.get('draft_revision')) is not int) or (row.get('draft_revision') != versions['draft_revision']) or (row.get('snapshot_checksum') != parent['snapshot']['checksum']):
            raise ValueError('joint recovery release has a stale serving draft snapshot ACK')

def preflight_colocated_joint_parent(config, *, serving_view=None):
    binding = config.get('joint_restore_binding')
    if type(binding) is not dict or set(binding) != {'uri', 'checksum'}:
        raise ValueError('shared restore needs an explicit complete parent, never silent DP2 reshard')
    try:
        parent, _ = open_joint_parent(config, binding)
    except (KeyError, TypeError) as exc:
        raise ValueError('shared restore lacks the explicit parent recipe; no DP2 reshard') from exc
    for field, key, members in (('actor_rows', 'rank', set(range(32))), ('draft_state_rows', 'worker_id', {str(i) for i in range(32)}), ('target_acks', 'rank', set(range(8)))):
        rows = parent[field]
        expected_type = type(next(iter(members)))
        if type(rows) is not list or len(rows) != len(members) or any((type(row.get(key)) is not expected_type for row in rows)) or ({row[key] for row in rows} != members):
            raise ValueError('shared parent requires exact actor32, draftDP32 and nativeTP8; no DP2 reshard')
    for role in ('actor', 'draft'):
        if type(parent[role + '_manifest'].get('world_size')) is not int or parent[role + '_manifest']['world_size'] != 32:
            raise ValueError('shared parent rank-local checkpoint membership cannot be resharded')
    if serving_view is not None:
        path = Path(serving_view)
        if not path.is_absolute() or not path.resolve().is_relative_to(Path(config['joint_run_root']).resolve()):
            raise ValueError('shared cold serving view must belong to its parent run')
        view = read_published_json(path / 'snapshot-binding.json')
        if view.get('snapshot') != parent['snapshot']:
            raise ValueError('shared native must cold load the committed parent draft snapshot')
        import json
        hf = json.loads((path / 'config.json').read_text())
        if hf.get('speco_snapshot_checksum') != parent['snapshot']['checksum']:
            raise ValueError('shared native HF view has a different parent snapshot identity')
    return deepcopy(parent)
from verl_speco.integration.multinode_online import OnlinePhase

class MultiNodeJointCheckpointExecutor:
    """A failed save requires exact-owner cleanup; it cannot be retried in place."""

    def __init__(self, *, store, session, actor_group, draft_group, target_publisher, control_rpc, abort_owned):
        if not callable(control_rpc) or not callable(abort_owned):
            raise ValueError('joint checkpoint requires control and exact-owner cleanup ports')
        self.store, self.session = (store, session)
        self.actor, self.draft = (actor_group, draft_group)
        self.target = target_publisher
        self.control_rpc, self.abort_owned = (control_rpc, abort_owned)
        self.phase = 'idle'
        self.last_step = 0
        self.last_binding = None

    async def save(self, completed):
        collection = self.session.collection
        async with collection.lock:
            if self.phase != 'idle':
                raise RuntimeError('uncertain joint checkpoint cannot be replayed')
            versions = collection.versions
            if collection.failed or collection.pending_collection is not None or self.session.adapter.phase is not OnlinePhase.IDLE or (self.session.adapter.versions != versions) or (completed.get('versions') != versions) or (versions.run_id != self.store.run_id) or (versions.generation != self.store.generation) or (type(versions.rl_step) is not int) or (versions.rl_step <= self.last_step):
                raise RuntimeError('joint checkpoint requires the exact completed drained online step')
            training = completed['draft'].training_execution
            if training is None or training.trained is not True or training.successful_steps < 1:
                raise RuntimeError('joint checkpoint has no actual committed draft execution')
            raw = deepcopy(training.raw_results)
            if not isinstance(raw, list) or len(raw) != len(self.store.drafts) or any((not isinstance(row, dict) for row in raw)) or ({row.get('worker_id') for row in raw} != set(self.store.drafts)) or any((row.get('worker_incarnation') != self.store.drafts[row['worker_id']] or row.get('run_id') != versions.run_id or row.get('generation') != versions.generation for row in raw)):
                raise RuntimeError('joint checkpoint draft receipts differ from current owners')
            target = deepcopy(self.target.last_proof)
            snapshot = deepcopy(self.session.publisher.last_snapshot)
            acks = deepcopy(self.session.publisher.last_acks)
            if not target or not snapshot or (not acks):
                raise RuntimeError('joint checkpoint requires actual target and draft serving publications')
            update = deepcopy(completed['actor'])
            if update.get('run_id') != versions.run_id or update.get('generation') != versions.generation or update.get('rl_step') != versions.rl_step or (update.get('target_revision') != versions.target_revision - 1) or (update.get('actual_VeRL_actor') is not True) or (update.get('optimizer_committed') is not True) or (type(update.get('optimizer_step')) is not int) or (update['optimizer_step'] < 1) or ({str(k): v for k, v in update.get('worker_incarnations', {}).items()} != {str(k): v for k, v in self.store.actors.items()}):
                raise RuntimeError('joint checkpoint actor receipt differs from the completed serving step')
            owners = dict(actor=dict(self.store.actors), draft=dict(self.store.drafts), rollout=dict(self.store.rollout))
            self.phase = 'saving'
            try:
                before = await self.control_rpc('snapshot_quiescent', {})
                request = dict(run_id=versions.run_id, generation=versions.generation, optimizer_step=update['optimizer_step'], actor_incarnations={str(k): v for k, v in self.store.actors.items()})
                actor_rows, draft_state = await asyncio.gather(asyncio.to_thread(self.actor.speco_save_actor_checkpoint, request), asyncio.to_thread(self.draft.speco_audit_joint_draft_state))
                after = await self.control_rpc('snapshot_quiescent', {})
                if before != after or collection.versions != versions:
                    raise RuntimeError('joint safe point changed while complete role state was saved')
                payload = await asyncio.to_thread(self.store.build, versions=versions, actor_update_receipt=update, actor_rows=actor_rows, draft_rows=raw, draft_state_rows=draft_state, control_snapshot=after, target_acks=target['native_ACKs'], draft_acks=acks, target_head=target['immutable_head'], snapshot=snapshot, session_idle=True, pending_commits=0)
                binding = await asyncio.to_thread(self.store.commit, payload)
                self.last_step, self.last_binding = (versions.rl_step, deepcopy(binding))
                self.phase = 'idle'
                return binding
            except BaseException:
                self.phase = 'failed'
                collection.failed = True
                await self.abort_owned(owners)
                raise

def joint_diagnostic_config(config, draft_manifest):
    total = config['total_training_steps']
    if type(total) is not int or total < 2:
        raise ValueError('joint diagnostic needs a scheduler horizon including continuation')
    run_root = Path(config['joint_run_root'])
    output = Path(config['output_root'])
    if not run_root.is_absolute() or not output.is_absolute():
        raise ValueError('joint role paths must be explicitly absolute')
    run_root, output = (run_root.resolve(), output.resolve())
    if not output.is_relative_to(run_root):
        raise ValueError('joint role outputs must belong to the same owned run')
    parallel = config.get('parallel_actor_draft_updates', False)
    if type(parallel) is not bool:
        raise ValueError('parallel actor/draft updates require an explicit boolean')
    shared = config.get('colocated', False)
    if type(shared) is not bool:
        raise ValueError('shared topology flag must be explicit boolean')
    if shared:
        if type(config['actor_ips']) is not list or len(config['actor_ips']) != 2 or len(set(config['actor_ips'])) != 2 or (config['trainer_ip'] != config['actor_ips'][0]):
            raise ValueError('shared recipe requires two-node actor32 and draftDP32')
    elif config['trainer_ip'] in config['actor_ips']:
        raise ValueError('joint draft trainer must have its own node')
    model = Path(config['model_path']).resolve()
    logical = dict(run_id=config['run_id'], model_path=str(model), model_config_sha256=hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(), world_size=32, fsdp_full_shard=True, expert_wrap='OpenPanguV2MLP', EP=1, SP=1, learning_rate=config.get('learning_rate', 1e-06), total_steps=total)
    actor = hashlib.sha256(json.dumps(logical, sort_keys=True).encode()).hexdigest()
    geometry = draft_manifest['model_config']
    fingerprints = dict(actor_model=actor, draft_model=hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(), target_model=draft_manifest['target_model_fingerprint'], tokenizer=draft_manifest['tokenizer_fingerprint'], tq_schema=3, target_layers=geometry['target_layer_ids'], rl_roles=['actor'], recipe=hashlib.sha256(json.dumps(dict(actor=logical, draft=geometry, seed=42, draft_lr=config.get('draft_learning_rate', 0.0001)), sort_keys=True).encode()).hexdigest())
    if shared:
        fingerprints['recipe'] = hashlib.sha256(json.dumps(dict(parent_recipe=fingerprints['recipe'], colocated=True, actor_optimizer_execution='single_tensor_adamw', draft_mesh=dict(dp_replicate=32, dp_shard=1, tp=1, pp=1, sp=1, ep=1)), sort_keys=True).encode()).hexdigest()
    if parallel:
        fingerprints['recipe'] = hashlib.sha256(json.dumps(dict(parent_recipe=fingerprints['recipe'], parallel_actor_draft_updates=True), sort_keys=True).encode()).hexdigest()
    if config.get('prompt_input'):
        from verl_speco.runtime.training_runtime import online_data_contract
        contract = online_data_contract(config)
        fingerprints['recipe'] = hashlib.sha256(json.dumps(dict(role_recipe=fingerprints['recipe'], online_data=contract), sort_keys=True).encode()).hexdigest()
    supplied = config.get('joint_fingerprints')
    if supplied is not None and supplied != fingerprints:
        raise ValueError('joint diagnostic recipe differs from its explicit parent fingerprints')
    result = dict(run_id=config['run_id'], generation=config['generation'], joint_run_root=str(run_root), joint_fingerprints=fingerprints, joint_control_capacity_bytes=512 * 1024 ** 2, partition='full-online-' + config['run_id'])
    if config.get('prompt_input'):
        result['online_data_contract'] = contract
    if parallel:
        result['parallel_actor_draft_updates'] = True
    if config.get('joint_restore_binding') is not None:
        result['joint_restore_binding'] = config['joint_restore_binding']
    if shared:
        result['colocated'] = True
        if result.get('joint_restore_binding'):
            preflight_colocated_joint_parent(result)
    return result

def resolved_joint_diagnostic_config(config, draft_manifest):
    """Return launcher settings merged with the exact online joint identity.

    Diagnostic inputs can contain a copied partition or capacity from an older
    launcher. The online runtime derives those values in joint_diagnostic_config.
    Recovery must use that same recipe, while retaining strict fingerprint and
    complete-parent validation. This helper never reserves a new generation.
    Generic non-diagnostic joint configurations must keep their own identities.
    """
    from copy import deepcopy
    resolved = deepcopy(config)
    resolved.update(joint_diagnostic_config(config, draft_manifest))
    return resolved
from types import MethodType
from verl_speco.integration.multinode_publication import TargetWireDigest

def install_joint_native_restore(worker, *, config, binding, owners):
    """Worker-local named RPC after cold channel connection, before requests."""
    from verl_speco.integration.accelerator_adapters import SpecoMultiNodeTargetUpdateWorker as SpecoMultiNodeTargetUpdateWorker
    if not isinstance(worker, SpecoMultiNodeTargetUpdateWorker) or getattr(worker, '_speco_joint_phase', None) is not None or worker._speco_target_update_phase != 'idle' or (worker._speco_target_revision != 0) or (worker._speco_draft_update_phase != 'idle') or (getattr(worker, '_speco_accepted_probe', None) is not None):
        raise ValueError('joint native bootstrap requires a fresh released cold engine')
    parent, _ = open_joint_parent(config, binding)
    validate_joint_owners(parent, owners)
    channel = worker._speco_checkpoint_channel
    if channel.phase != 'ready' or channel.identity['actors'] != owners['actor'] or channel.identity['rollout'] != owners['rollout'] or (channel.identity['run_id'] != config['run_id']) or (channel.identity['generation'] != config['generation']) or (owners['rollout'].get(worker.rank) != worker._speco_serving_incarnation):
        raise ValueError('joint native bootstrap has a foreign cold channel or owner')
    draft = worker.model_runner.drafter.model
    if worker._speco_draft_revision != parent['versions']['draft_revision'] or draft.config.speco_snapshot_checksum != parent['snapshot']['checksum']:
        raise ValueError('joint native engine must cold load the committed draft snapshot')
    worker._speco_joint_phase = 'restoring'
    worker._speco_joint_parent = parent
    worker._speco_joint_config = deepcopy(config)
    worker._speco_joint_owners = deepcopy(owners)
    worker._speco_joint_target_ack = None
    worker._speco_joint_draft_ack = None
    worker._speco_joint_original_execute = worker.execute_model
    worker._speco_joint_original_target_commit = worker.speco_commit_target_transfer
    worker.execute_model = MethodType(joint_native_restore_joint_execute, worker)
    worker.speco_prepare_target_transfer = MethodType(joint_native_restore_joint_prepare_target, worker)
    worker.speco_commit_target_transfer = MethodType(joint_native_restore_joint_commit_target, worker)
    return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, prepared=True, binding_checksum=binding['checksum'], compute_released=False)

def joint_native_restore_joint_execute(worker, *args, **kwargs):
    if worker._speco_joint_phase != 'ready':
        raise RuntimeError('native generation is fenced until complete joint recovery ACKs')
    return worker._speco_joint_original_execute(*args, **kwargs)

def joint_native_restore_joint_prepare_target(worker, *, manifest, bucket_bytes):
    if worker._speco_joint_phase == 'ready':
        from verl_speco.integration.multinode_publication import preconnected_target_native_prepare_target as _native_prepare_target
        return _native_prepare_target(worker, manifest=manifest, bucket_bytes=bucket_bytes)
    if worker._speco_joint_phase != 'restoring':
        raise RuntimeError('joint native target bootstrap has a failed recovery owner')
    parent = worker._speco_joint_parent
    channel = worker._speco_checkpoint_channel
    config = worker._speco_joint_config
    if worker._speco_target_revision != 0 or worker._speco_target_update_phase != 'idle' or worker._speco_draft_update_phase != 'idle' or any((type(manifest[key]) is not int or manifest[key] < 1 for key in ('generation', 'rl_step', 'target_revision'))) or (manifest['run_id'] != config['run_id']) or (manifest['generation'] != config['generation']) or (manifest['rl_step'] != parent['versions']['rl_step']) or (manifest['target_revision'] != parent['versions']['target_revision']) or (manifest['actor_incarnations'] != {str(k): v for k, v in channel.identity['actors'].items()}) or (len(manifest['tensors']) != 35869) or (type(bucket_bytes) is not int) or (bucket_bytes != channel.identity['bucket_bytes']) or (os.environ.get('PANGU_92B_EXPERT_SLOTS_EAGER', '') in ('1', 'true', 'True')) or (bucket_bytes < max((row['nbytes'] for row in manifest['tensors']))):
        raise ValueError('joint native target bootstrap differs from its exact committed parent')
    digest = TargetWireDigest(manifest)
    worker._speco_target_update_phase = 'preparing'
    try:
        engine = channel.begin(channel.identity, manifest['operation_id'])
        worker._speco_target_update = dict(manifest=manifest, digest=digest, engine=engine, receipt=None)
        worker._speco_target_update_phase = 'prepared'
        return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, prepared=True, header_sha256=manifest['header_sha256'])
    except BaseException:
        worker._speco_joint_phase = worker._speco_target_update_phase = 'failed'
        raise

def joint_native_restore_joint_commit_target(worker, **request):
    if worker._speco_joint_phase not in ('restoring', 'ready'):
        raise RuntimeError('joint native target commit has a failed recovery owner')
    ack = worker._speco_joint_original_target_commit(**request)
    if worker._speco_joint_phase == 'restoring':
        worker._speco_joint_target_ack = deepcopy(ack)
    return ack

def audit_joint_native_draft(worker):
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from verl_speco.integration.multinode_publication import read_draft_update, audit_native_draft
    from verl_speco.integration.multinode_publication import invalidate_draft_caches
    import torch
    if worker._speco_joint_phase != 'restoring' or worker._speco_joint_draft_ack is not None:
        raise RuntimeError('joint native draft audit requires its fresh fenced snapshot')
    try:
        parent = worker._speco_joint_parent
        model = worker.model_runner.drafter.model
        cfg = model.config
        snapshot, manifest, weights = read_draft_update(parent['snapshot'], target_fingerprint=cfg.speco_snapshot_target_fingerprint, tokenizer_fingerprint=cfg.speco_snapshot_tokenizer_fingerprint, model_config=cfg.to_dict())
        audit = audit_native_draft(model, weights, config=manifest['model_config'], rank=get_tensor_model_parallel_rank(), world_size=get_tensor_model_parallel_world_size())
        cleared = invalidate_draft_caches(worker)
        torch.npu.synchronize()
        if model.lm_head is not worker.model_runner.model.lm_head:
            raise RuntimeError('joint native draft is not bound to the restored target head')
        ack = dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, run_id=worker._speco_joint_config['run_id'], generation=worker._speco_joint_config['generation'], draft_revision=snapshot.draft_revision, snapshot_checksum=snapshot.checksum, applied=True, kv_cache_cleared=True, cache_tensors_cleared=cleared, **audit)
        worker._speco_joint_draft_ack = deepcopy(ack)
        return ack
    except BaseException:
        worker._speco_joint_phase = 'failed'
        raise

def release_joint_native_restore(worker, *, proof):
    if worker._speco_joint_phase != 'restoring':
        raise RuntimeError('joint native release requires the verified fenced generation')
    try:
        config = worker._speco_joint_config
        validate_joint_release(worker._speco_joint_parent, proof, worker._speco_joint_owners, run_id=config['run_id'], generation=config['generation'], binding_checksum=config['joint_restore_binding']['checksum'])
        if worker._speco_target_update_phase != 'idle' or worker._speco_checkpoint_channel.phase != 'ready' or type(worker._speco_target_revision) is not int or (worker._speco_target_revision != worker._speco_joint_parent['versions']['target_revision']) or (type(worker._speco_draft_revision) is not int) or (worker._speco_draft_revision != worker._speco_joint_parent['versions']['draft_revision']) or (worker._speco_joint_target_ack is None) or (worker._speco_joint_draft_ack is None):
            raise ValueError('joint native release lacks its actual complete local ACKs')
        for field, local in (('target_acks', worker._speco_joint_target_ack), ('draft_acks', worker._speco_joint_draft_ack)):
            remote = next((row for row in proof[field] if row['rank'] == worker.rank))
            if any((remote.get(key) != value for key, value in local.items())):
                raise ValueError('joint native release proof differs from actual local weights/cache ACK')
        worker._speco_joint_phase = 'ready'
        return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, compute_released=True, target_revision=worker._speco_target_revision, draft_revision=worker._speco_draft_revision)
    except BaseException:
        worker._speco_joint_phase = 'failed'
        raise
from uuid import uuid4
from verl_speco.integration.multinode_publication import require_owned_rows
from verl_speco.transport.control import control_snapshot_sha256

def joint_restore_coordinator_strict_owned_rows(rows, owners, *, phase):
    require_owned_rows(rows, owners, phase=phase)
    if any((type(row['rank']) is not int for row in rows)):
        raise RuntimeError('joint recovery role rank must be an integer, not boolean')
    return rows

async def joint_restore_coordinator_join_owned_effects(*effects):
    """An error/cancel must not kill peers still mutating or using HCCL buffers."""

    async def join():
        return await asyncio.gather(*effects, return_exceptions=True)
    task = asyncio.create_task(join())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    results = task.result()
    if cancelled:
        raise asyncio.CancelledError
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results

class MultiNodeJointRestoreCoordinator:
    """All role owners stay fenced until complete state/cache consensus.

    The caller initializes cold workers, the parent draft serving checkpoint
    and generation-owned HCCL channel. On any uncertainty, abort_owned must
    terminate these exact role incarnations before another generation starts.
    No training step, version or historical optimizer effect is manufactured.
    """

    def __init__(self, *, config, owners, actor_group, draft_group, collective_rpc, control_rpc, reset_prefix_cache, abort_owned, bucket_bytes=1024 ** 3, phase_control=None, record=lambda *_: None):
        if not callable(abort_owned) or not callable(control_rpc) or (not callable(collective_rpc)) or (type(bucket_bytes) is not int) or (bucket_bytes < 1):
            raise ValueError('joint restore requires explicit owned cleanup and bounded runtime ports')
        if not callable(record) or (phase_control is not None and any((not callable(getattr(phase_control, name, None)) for name in ('enter_training', 'enter_publication', 'finish_restore', 'mark_uncertain')))):
            raise ValueError('joint shared restore requires complete actual phase ports and durable receipts')
        if config.get('colocated') is True and phase_control is None:
            raise ValueError('shared recovery cannot discard its actual memory phase owner')
        self.phase_control, self.record = (phase_control, record)
        self.restore_phase_receipts = []
        self.config, self.owners = (deepcopy(config), deepcopy(owners))
        self.actor, self.draft = (actor_group, draft_group)
        self.rpc, self.control_rpc = (collective_rpc, control_rpc)
        self.reset_prefix_cache, self.abort_owned = (reset_prefix_cache, abort_owned)
        self.bucket_bytes = bucket_bytes
        self.phase = 'new'
        self.lock = asyncio.Lock()
        self.proof = None

    async def restore(self):
        async with self.lock:
            if self.phase != 'new':
                raise RuntimeError('uncertain or completed joint recovery cannot be replayed')
            self.phase = 'restoring'
            try:
                binding = self.config['joint_restore_binding']
                parent, control_snapshot = open_joint_parent(self.config, binding)
                validate_joint_owners(parent, self.owners)
                run_id, generation = (self.config['run_id'], self.config['generation'])
                restored_control = await drain_owned_call(self.control_rpc, 'restore_joint_parent', dict(snapshot=parent['control_snapshot'], expected_sha256=control_snapshot_sha256(parent['control_snapshot'])))
                if restored_control.get('restored') is not True or restored_control.get('run_id') != run_id or type(restored_control.get('generation')) is not int or (restored_control['generation'] != generation):
                    raise RuntimeError('joint control did not acknowledge its fresh parent restore')
                if await drain_owned_call(self.control_rpc, 'snapshot_quiescent', {}) != control_snapshot:
                    raise RuntimeError('joint control restored different commit history or active payloads')
                prepared = await drain_owned_call(self.rpc, 'speco_install_joint_native_restore', dict(config=self.config, binding=binding, owners=self.owners))
                joint_restore_coordinator_strict_owned_rows(prepared, self.owners['rollout'], phase='prepared')
                if any((row.get('binding_checksum') != binding['checksum'] or row.get('compute_released') is not False for row in prepared)):
                    raise RuntimeError('native joint preparation did not retain the exact parent fence')
                request = dict(run_id=run_id, generation=generation, binding=binding, owners=self.owners)
                if self.phase_control is not None:
                    await self._memory_phase('restore-training', self.phase_control.enter_training)
                actor_rows, draft_rows = await joint_restore_coordinator_join_owned_effects(asyncio.to_thread(self.actor.speco_restore_joint_actor, request), asyncio.to_thread(self.draft.speco_restore_joint_draft, request))
                self._role_restored(actor_rows, self.owners['actor'], 'rank', parent['actor_rows'])
                self._role_restored(draft_rows, self.owners['draft'], 'worker_id', parent['draft_state_rows'])
                if any((row.get('binding_checksum') != binding['checksum'] or row.get('compute_released') is not False for row in actor_rows + draft_rows)):
                    raise RuntimeError('joint restored role released compute or changed its parent binding')
                if self.phase_control is not None:
                    await self._memory_phase('restore-publication', self.phase_control.enter_publication)
                operation = uuid4().hex
                v = parent['versions']
                transfer = dict(run_id=run_id, generation=generation, rl_step=v['rl_step'], target_revision=v['target_revision'], operation_id=operation, actor_incarnations={str(k): value for k, value in self.owners['actor'].items()}, optimizer_step=parent['actor_update_receipt']['optimizer_step'], bucket_bytes=self.bucket_bytes, tokenizer_fingerprint=parent['fingerprints']['tokenizer'])
                source = await drain_owned_call(self.actor.speco_prepare_target_transfer, transfer)
                joint_restore_coordinator_strict_owned_rows(source, self.owners['actor'], phase='prepared')
                master = next((row for row in source if row['rank'] == 0))
                manifest = master['manifest']
                if not isinstance(manifest, dict) or not isinstance(master['master_metadata'], dict) or any((row.get('header_sha256') != manifest['header_sha256'] for row in source)) or any((manifest.get(key) != transfer[key] for key in ('run_id', 'generation', 'rl_step', 'target_revision', 'operation_id', 'actor_incarnations'))):
                    raise RuntimeError('joint source did not bind the exact complete fresh parent manifest')
                TargetWireDigest(manifest)
                if len(manifest['tensors']) != 35869 or type(manifest['total_nbytes']) is not int or manifest['total_nbytes'] < 1:
                    raise RuntimeError('joint source manifest does not describe the complete audited state')
                receiver = await drain_owned_call(self.rpc, 'speco_prepare_target_transfer', dict(manifest=manifest, bucket_bytes=self.bucket_bytes))
                joint_restore_coordinator_strict_owned_rows(receiver, self.owners['rollout'], phase='prepared')
                if any((row.get('header_sha256') != manifest['header_sha256'] for row in receiver)):
                    raise RuntimeError('joint native receiver bound a different parent manifest')
                topology = dict(operation_id=operation, master_metadata=master['master_metadata'], rollout_world_size=len(self.owners['rollout']))
                connected = await joint_restore_coordinator_join_owned_effects(asyncio.to_thread(self.actor.speco_init_target_transfer, topology), self.rpc('speco_init_target_transfer', topology))
                joint_restore_coordinator_strict_owned_rows(connected[0], self.owners['actor'], phase='connected')
                joint_restore_coordinator_strict_owned_rows(connected[1], self.owners['rollout'], phase='connected')
                sent, loaded = await joint_restore_coordinator_join_owned_effects(asyncio.to_thread(self.actor.speco_send_target_transfer, operation), self.rpc('speco_receive_target_transfer', dict(operation_id=operation)))
                joint_restore_coordinator_strict_owned_rows(sent, self.owners['actor'], phase='sent')
                joint_restore_coordinator_strict_owned_rows(loaded, self.owners['rollout'], phase='loaded')
                leader = next((row for row in sent if row['rank'] == 0))
                wire, head = (leader['wire_receipt'], leader['target_head_binding'])
                if wire.get('header_sha256') != manifest['header_sha256'] or type(wire.get('wire_tensor_count')) is not int or wire['wire_tensor_count'] != 35869 or (type(wire.get('wire_nbytes')) is not int) or (wire['wire_nbytes'] != manifest['total_nbytes']):
                    raise RuntimeError('joint complete source wire differs from its exact manifest')
                if not isinstance(head, dict) or head.get('sha256') != parent['target_head']['sha256'] or any((head.get(k) != transfer[k] for k in ('run_id', 'generation', 'target_revision', 'tokenizer_fingerprint'))) or (head.get('actor_weight_manifest_sha256') != wire['actor_weight_manifest_sha256']):
                    raise RuntimeError('joint source immutable head differs from its complete restored parent')
                if any((any((row.get(key) != value for key, value in wire.items())) or row.get('complete_native_names_and_expert_slots') is not True for row in loaded)):
                    raise RuntimeError('joint native load lacks complete fresh source bytes and expert coverage')
                if await drain_owned_call(self.reset_prefix_cache) is not True:
                    raise RuntimeError('joint native scheduler did not reset actual prefix caches')
                identity = dict(operation_id=operation, actor_weight_manifest_sha256=wire['actor_weight_manifest_sha256'])
                target_acks = await drain_owned_call(self.rpc, 'speco_commit_target_transfer', identity)
                joint_restore_coordinator_strict_owned_rows(target_acks, self.owners['rollout'], phase='applied')
                source_closed = await drain_owned_call(self.actor.speco_finish_target_transfer, operation, identity['actor_weight_manifest_sha256'])
                joint_restore_coordinator_strict_owned_rows(source_closed, self.owners['actor'], phase='released')
                native_closed = await drain_owned_call(self.rpc, 'speco_finish_target_transfer', identity)
                joint_restore_coordinator_strict_owned_rows(native_closed, self.owners['rollout'], phase='released')
                draft_acks = await drain_owned_call(self.rpc, 'speco_audit_joint_native_draft', {})
                proof = dict(run_id=run_id, generation=generation, binding_checksum=binding['checksum'], actor_rows=actor_rows, draft_rows=draft_rows, target_acks=target_acks, draft_acks=draft_acks, source_wire=dict(wire, head_sha256=head['sha256']), native_prefix_cache_reset=True)
                validate_joint_release(parent, proof, self.owners, run_id=run_id, generation=generation, binding_checksum=binding['checksum'])
                self.phase = 'releasing'
                actor_release, draft_release, native_release = await joint_restore_coordinator_join_owned_effects(asyncio.to_thread(self.actor.speco_release_joint_actor, dict(proof=proof)), asyncio.to_thread(self.draft.speco_release_joint_draft, proof), self.rpc('speco_release_joint_native_restore', dict(proof=proof)))
                joint_restore_coordinator_strict_owned_rows(actor_release, self.owners['actor'], phase='compute_released')
                joint_restore_coordinator_strict_owned_rows(native_release, self.owners['rollout'], phase='compute_released')
                if any((row.get('run_id') != run_id or type(row.get('generation')) is not int or row['generation'] != generation or (type(row.get('optimizer_step')) is not int) or (row['optimizer_step'] != parent['actor_update_receipt']['optimizer_step']) for row in actor_release)):
                    raise RuntimeError('joint actor release changed its restored optimizer identity')
                if any((type(row.get(key)) is not int or row[key] != v[key] for row in native_release for key in ('target_revision', 'draft_revision'))):
                    raise RuntimeError('joint native release changed the restored serving revisions')
                self._draft_released(draft_release, parent)
                if self.phase_control is not None:
                    await self._memory_phase('restore-resumed', self.phase_control.finish_restore, proof)
                self.proof = deepcopy(proof)
                self.phase = 'ready'
                return dict(versions=dict(v, generation=generation), proof=deepcopy(proof), optimizer_effect_created=False, actor_release=actor_release, draft_release=draft_release, native_release=native_release, restore_phase_receipts=deepcopy(self.restore_phase_receipts))
            except BaseException:
                self.phase = 'failed'
                if self.phase_control is not None:
                    self.phase_control.mark_uncertain()
                await drain_owned_call(self.abort_owned, deepcopy(self.owners))
                raise

    async def _memory_phase(self, name, port, *args):
        receipt = await drain_owned_call(port, *args)
        self.restore_phase_receipts.append(dict(name=name, receipt=deepcopy(receipt)))
        self.record(name, receipt)

    def _role_restored(self, rows, owners, identity, parent_rows):
        prior = {row[identity]: row for row in parent_rows}
        if not isinstance(rows, list) or len(rows) != len(owners) or any((not isinstance(row, dict) or type(row.get(identity)) is not type(next(iter(owners))) for row in rows)) or ({row[identity] for row in rows} != set(owners)):
            raise RuntimeError('joint restored role membership differs')
        for row in rows:
            old = prior[row[identity]]
            if row.get('worker_incarnation') != owners[row[identity]] or row.get('restored') is not True or row.get('run_id') != self.config['run_id'] or (type(row.get('generation')) is not int) or (row['generation'] != self.config['generation']) or (row.get('state_sha256') != old['state_sha256']) or (type(row.get('optimizer_step')) is not int) or (row['optimizer_step'] != old['optimizer_step']) or (row.get('complete_local_model_optimizer_scheduler_rng') is not True):
                raise RuntimeError('joint restored role lacks its exact complete parent state')

    def _draft_released(self, rows, parent):
        owners = self.owners['draft']
        if not isinstance(rows, list) or len(rows) != len(owners) or any((not isinstance(row, dict) or type(row.get('worker_id')) is not str for row in rows)) or ({row['worker_id'] for row in rows} != set(owners)) or any((row.get('worker_incarnation') != owners[row['worker_id']] or row.get('released') is not True for row in rows)):
            raise RuntimeError('joint draft release acknowledgement differs')
        steps = {row['worker_id']: row['optimizer_step'] for row in parent['draft_state_rows']}
        for row in rows:
            if row.get('run_id') != self.config['run_id'] or type(row.get('generation')) is not int or row['generation'] != self.config['generation'] or (type(row.get('optimizer_step')) is not int) or (row['optimizer_step'] != steps[row['worker_id']]) or any((type(row.get(key)) is not int or row[key] != parent['versions'][key] for key in ('target_revision', 'draft_revision'))):
                raise RuntimeError('joint draft release changed its restored optimizer or serving identity')
