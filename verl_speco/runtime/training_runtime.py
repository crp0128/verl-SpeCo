# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from copy import deepcopy
from verl_speco.integration.multinode_recovery import validate_shared_cards, drain_owned_call

class MultiNodeColocatedWorkerBootstrap:

    def __init__(self, *, pool, actor_class, draft_class, actor_settings, draft_settings, group_factory=None, record=lambda *_: None):
        if not callable(record):
            raise ValueError('shared bootstrap requires its durable receipt writer')
        if group_factory is None:
            from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
            import ray

            def group_factory(cls, settings, handles, role):
                return RayWorkerGroup(ray_cls_with_init=RayClassWithInitArgs(ray.remote(cls), config=settings), worker_handles=handles, name_prefix=f'{pool.run_id}-g{pool.generation}-colocated-{role}', use_gpu=False, device_name='npu')
        if not callable(group_factory):
            raise ValueError('shared bootstrap requires actual worker group binding')
        if any((type(value) is not dict or value.get('run_id') != pool.run_id or value.get('generation') != pool.generation for value in [actor_settings, draft_settings])):
            raise ValueError('new shared topology requires fresh optimizer state, not silent DP2 checkpoint reshard')
        actor_parent = actor_settings.get('joint_restore_binding')
        draft_parent = draft_settings.get('joint_restore_binding')
        if actor_parent or draft_parent:
            if actor_parent != draft_parent or any((actor_settings.get(k) != draft_settings.get(k) for k in ('joint_run_root', 'joint_fingerprints', 'partition', 'joint_control_capacity_bytes'))):
                raise ValueError('shared roles need the same exact parent; no silent DP2 reshard')
            from verl_speco.integration.multinode_recovery import preflight_colocated_joint_parent
            preflight_colocated_joint_parent(actor_settings)
            preflight_colocated_joint_parent(draft_settings)
        self.pool, self.actor_class, self.draft_class = (pool, actor_class, draft_class)
        self.settings = {'actor': deepcopy(actor_settings), 'draft': deepcopy(draft_settings)}
        self.factory, self.record = (group_factory, record)
        self.groups = {}
        self.initial = {}
        self.placements = {}
        self.binding = None
        self.failed = False
        self.started = False

    def _owners(self, role, rows):
        world = 16 * len(self.pool.nodes)
        key = 'rank' if role == 'actor' else 'worker_id'
        if type(rows) is not list or len(rows) != world or any((type(row) is not dict for row in rows)) or ({row.get(key) for row in rows} != ({str(i) for i in range(world)} if role == 'draft' else set(range(world)))) or any((row.get('run_id') != self.pool.run_id or type(row.get('generation')) is not int or row['generation'] != self.pool.generation or (type(row.get('worker_incarnation')) is not str) or (not row['worker_incarnation']) or (type(row.get('optimizer_step')) is not int) or (row['optimizer_step'] != 0) for row in rows)) or (len({row['worker_incarnation'] for row in rows}) != world):
            raise RuntimeError('shared fresh initialization lacks the complete current role owner set')
        if role == 'actor':
            if any((row.get('full_parameter_VeRL_actor') is not True or row.get('global_parameter_elements') != 91955618233 or row.get('parameter_tensors') != 35825 for row in rows)):
                raise RuntimeError('shared actor is not the actual complete 92B VeRL actor')
        elif any((row.get('initialized') is not True or row.get('wrapper') != 'fsdp2' or row.get('engine') != 'veomni' or (row.get('mesh') != dict(dp_replicate=world, dp_shard=1, tp=1, pp=1, sp=1, ep=1)) for row in rows)):
            raise RuntimeError('shared draft is not DP equal to training dies with FSDP2 only')
        return {int(row[key]): row['worker_incarnation'] for row in rows}

    def _placements(self, role, rows, owners):
        if type(rows) is not list or len(rows) != len(owners) or any((type(row) is not dict or type(row.get('rank')) is not int for row in rows)) or ({row['rank'] for row in rows} != set(owners)):
            raise RuntimeError('shared worker physical placement membership differs')
        for row in rows:
            rank = row['rank']
            if row.get('worker_incarnation') != owners[rank] or row.get('role') != role or row.get('node_ip') != self.pool.nodes[rank // 16] or (row.get('bundle_index') != rank % 16):
                raise RuntimeError('shared worker observed physical role/node/bundle differs')
        return sorted(rows, key=lambda row: row['rank'])

    async def _role(self, role, cls):
        handles = await drain_owned_call(self.pool.launch_role, role, cls, self.settings[role])
        group = self.factory(cls, self.settings[role], handles, role)
        self.groups[role] = group
        rows = await drain_owned_call(group.init_model)
        owners = self._owners(role, rows)
        self.initial[role] = rows
        self.record(role + '-initialized', dict(workers=rows))
        placements = self._placements(role, await drain_owned_call(group.speco_colocated_placement), owners)
        self.placements[role] = placements
        if role == 'draft':
            for row in rows:
                if row.get('colocated_placement') != placements[int(row['worker_id'])]:
                    raise RuntimeError('draft initialization and current physical placement receipts differ')
        parked = await drain_owned_call(group.speco_colocated_park_bootstrap)
        if type(parked) is not list or len(parked) != len(owners):
            raise RuntimeError('shared bootstrap park lacks all ranks')
        expected = {row['rank']: row for row in placements}
        if any((type(row) is not dict or type(row.get('rank')) is not int for row in parked)) or {row['rank'] for row in parked} != set(expected):
            raise RuntimeError('shared bootstrap park has duplicate/missing physical owners')
        for row in parked:
            proof = {key: value for key, value in row.items() if key in expected[row['rank']]}
            if proof != expected[row['rank']] or set(row) != set(proof) | {'bootstrap_parked', 'model_on_device', 'optimizer_on_device', 'head_on_device'} or row['bootstrap_parked'] is not True or (row['model_on_device'] is not False) or (row['optimizer_on_device'] is not False) or (row['head_on_device'] is not False):
                raise RuntimeError('shared bootstrap park did not prove actual CPU model/optimizer/head placement')
        self.record(role + '-bootstrap-parked', dict(workers=parked))

    async def initialize(self):
        if self.started or self.failed:
            raise RuntimeError('shared bootstrap cannot reuse an uncertain/finalized placement')
        self.started = True
        try:
            await drain_owned_call(self.pool.create)
            await self._role('actor', self.actor_class)
            await self._role('draft', self.draft_class)
            binding = validate_shared_cards(run_id=self.pool.run_id, generation=self.pool.generation, actor=self.placements['actor'], draft=self.placements['draft'])
            for role in ('actor', 'draft'):
                rows = await drain_owned_call(self.groups[role].speco_colocated_bind_memory, binding)
                if type(rows) is not list or len(rows) != len(binding[role]):
                    raise RuntimeError('shared memory bind lacks complete ranks')
                expected = {row['rank']: dict(row, memory_bound=True) for row in binding[role]}
                if any((type(row) is not dict or type(row.get('rank')) is not int for row in rows)) or {row['rank'] for row in rows} != set(expected) or any((row != expected[row['rank']] or row.get('memory_bound') is not True for row in rows)):
                    raise RuntimeError('shared memory binding receipt has foreign/incomplete observed physical owners')
            self.binding = binding
            self.record('colocated-placement-binding', binding)
            return binding
        except BaseException:
            self.failed = True
            await drain_owned_call(self.pool.close)
            raise

    async def memory_transition(self, role, request):
        if self.failed or self.binding is None or role not in self.groups:
            raise RuntimeError('shared memory transition requires initialized current worker pairing')
        return await drain_owned_call(self.groups[role].speco_colocated_memory_transition, request)
from ipaddress import ip_address
import os

class MultiNodeColocatedResourcePool:

    def __init__(self, *, run_id, generation, node_ips, python, master_ports, base_env, ray_api=None, placement_factory=None, remove_placement=None, scheduling_factory=None, record_owned=lambda _: None):
        if type(run_id) is not str or not run_id or type(generation) is not int or (generation < 1) or (type(node_ips) is not tuple) or (not 2 <= len(node_ips) <= 8) or (len(set(node_ips)) != len(node_ips)) or (type(python) is not str) or (not os.path.isabs(python)) or (type(master_ports) is not dict) or (set(master_ports) != {'actor', 'draft'}) or any((type(value) is not int or not 1 <= value <= 65535 for value in master_ports.values())) or (len(set(master_ports.values())) != 2) or (type(base_env) is not dict) or any((type(key) is not str or type(value) is not str for key, value in base_env.items())) or (not callable(record_owned)):
            raise ValueError('shared pool requires exact owned nodes, interpreters, env and distinct rendezvous ports')
        for value in node_ips:
            if type(value) is not str or str(ip_address(value)) != value:
                raise ValueError('shared pool requires canonical explicit node IPs')
        if ray_api is None:
            import ray
            from ray.util.placement_group import placement_group, remove_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            ray_api = ray
            placement_factory = placement_group
            remove_placement = remove_placement_group
            scheduling_factory = PlacementGroupSchedulingStrategy
        if any((not callable(value) for value in (placement_factory, remove_placement, scheduling_factory))):
            raise ValueError('shared pool requires complete placement lifecycle ports')
        self.run_id, self.generation, self.nodes, self.python = (run_id, generation, node_ips, python)
        self.ports, self.env = (deepcopy(master_ports), dict(base_env))
        for key in ('ASCEND_RT_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'):
            self.env.pop(key, None)
        self.env['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, range(16)))
        self.env['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = ''
        self.env['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'] = ''
        self.ray, self.place, self.remove, self.strategy, self.record = (ray_api, placement_factory, remove_placement, scheduling_factory, record_owned)
        self.groups = []
        self.workers = []
        self.roles = set()
        self.failed = False
        self.closed = False

    def ownership(self):
        return dict(run_id=self.run_id, generation=self.generation, placements=[dict(node_ip=ip, placement_group_id=group.id.hex()) for ip, group in self.groups], workers=[dict(role=role, rank=rank, actor_id=handle._actor_id.hex()) for role, rank, handle in self.workers])

    def create(self):
        if self.groups or self.failed or self.closed:
            raise RuntimeError('shared pool cannot recreate existing/uncertain ownership')
        live = {node['NodeManagerAddress'] for node in self.ray.nodes() if node.get('Alive') is True}
        if any((ip not in live for ip in self.nodes)):
            raise ValueError('shared pool includes a node absent from the live reserved Ray cluster')
        try:
            for index, ip in enumerate(self.nodes):
                group = self.place(bundles=[{'CPU': 0.1, 'NPU': 1, 'node:' + ip: 0.001} for _ in range(16)], strategy='STRICT_PACK', name=f'{self.run_id}-g{self.generation}-colocated-{index}')
                self.groups.append((ip, group))
                self.record(self.ownership())
            self.ray.get([group.ready() for _, group in self.groups], timeout=180)
            return self.ownership()
        except BaseException:
            self.failed = True
            self.close()
            raise

    def launch_role(self, role, worker_cls, settings):
        if self.failed or self.closed or len(self.groups) != len(self.nodes) or (role not in ('actor', 'draft')) or (role in self.roles):
            raise RuntimeError('shared role requires fresh ready placement ownership')
        if type(settings) is not dict or settings.get('run_id') != self.run_id or type(settings.get('generation')) is not int or (settings['generation'] != self.generation):
            raise ValueError('shared role settings differ from the reserved run owner')
        self.roles.add(role)
        handles = []
        world = 16 * len(self.nodes)
        try:
            for node_index, (ip, group) in enumerate(self.groups):
                for bundle in range(16):
                    rank = node_index * 16 + bundle
                    assignment = dict(run_id=self.run_id, generation=self.generation, role=role, rank=rank, node_ip=ip, placement_group_id=group.id.hex(), bundle_index=bundle)
                    worker_settings = dict(deepcopy(settings), colocated_assignment=assignment)
                    if role == 'draft':
                        worker_settings['native_endpoint_template'] = f'{ip}:{63000 + rank}'
                    env = dict(self.env, RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK='0', LOCAL_WORLD_SIZE='16', MASTER_ADDR=self.nodes[0], MASTER_PORT=str(self.ports[role]))
                    if role == 'actor' and settings.get('model_path') is not None:
                        env['PANGU_MODEL_PATH'] = settings['model_path']
                    strategy = self.strategy(placement_group=group, placement_group_bundle_index=bundle, placement_group_capture_child_tasks=True)
                    handle = self.ray.remote(worker_cls).options(name=f'{self.run_id}-g{self.generation}-{role}-{rank}', num_cpus=0.05, resources={'NPU': 0.5}, max_concurrency=1, scheduling_strategy=strategy, runtime_env=dict(py_executable=self.python, env_vars=env)).remote(worker_settings)
                    handles.append(handle)
                    self.workers.append((role, rank, handle))
                    self.record(self.ownership())
            return handles
        except BaseException:
            self.failed = True
            self.close()
            raise

    def close(self):
        errors = []
        for role, rank, handle in tuple(reversed(self.workers)):
            try:
                self.ray.kill(handle, no_restart=True)
                self.workers.remove((role, rank, handle))
                self.record(self.ownership())
            except BaseException as error:
                errors.append(error)
        if not self.workers:
            for ip, group in tuple(reversed(self.groups)):
                try:
                    self.remove(group)
                    self.groups.remove((ip, group))
                    self.record(self.ownership())
                except BaseException as error:
                    errors.append(error)
        if errors:
            self.failed = True
            raise RuntimeError('shared pool cleanup retains uncertain owned resources') from errors[0]
        self.closed = True
from verl_speco.integration.multinode_recovery import ColocatedCardCoordinator, drain_owned_call
from verl_speco.integration.multinode_recovery import ColocatedVeRLRolloutPort
from verl_speco.integration.multinode_recovery import MultiNodeColocatedNativeEngine as MultiNodeColocatedNativeEngine

def validate_colocated_entry_config(config, base_env):
    from ipaddress import ip_address
    if config.get('colocated') is not True or type(config.get('actor_ips')) is not list or len(config['actor_ips']) != 2 or (len(set(config['actor_ips'])) != 2) or (config.get('trainer_ip') != config['actor_ips'][0]) or (config.get('rollout_ip') != config['actor_ips'][0]) or config.get('actor_trace_root'):
        raise ValueError('shared entry requires fresh actor32 and draftDP32 on two owned nodes')
    for address in config['actor_ips']:
        if type(address) is not str or str(ip_address(address)) != address:
            raise ValueError('shared entry requires canonical reserved node IPs')
    source = online_endpoint_preflight_endpoint(base_env.get('SPECO_NATIVE_SOURCE_ENDPOINT'), 'shared native source')
    if source['host'] != config['actor_ips'][0]:
        raise ValueError('shared native source endpoint must belong to the actual first actor/rollout node')
    ports = [online_endpoint_preflight_port(config[key], key) for key in ('actor_master_port', 'draft_master_port')]
    if ports[0] == ports[1] or source['port'] in ports:
        raise ValueError('shared meshes and native source require distinct rendezvous endpoints')
    if config.get('joint_restore_binding') is not None:
        from verl_speco.integration.multinode_recovery import preflight_colocated_joint_parent
        if not config.get('draft_model_path'):
            raise ValueError('shared restored entry requires its actual cold parent serving view')
        preflight_colocated_joint_parent(config, serving_view=config['draft_model_path'])
    return dict(actor_world_size=32, draft_mesh=dict(dp_replicate=32, dp_shard=1, tp=1, pp=1, sp=1, ep=1), native_TP8_source_node=config['actor_ips'][0], native_source=source, device_allocation_not_performed=True)

class MultiNodeColocatedTrainingRuntime:

    def __init__(self, *, config, base_env, record):
        validate_colocated_entry_config(config, base_env)
        if config.get('colocated') is not True or type(config.get('actor_ips')) is not list or len(config['actor_ips']) != 2 or (len(set(config['actor_ips'])) != 2) or (config.get('trainer_ip') != config['actor_ips'][0]) or config.get('actor_trace_root') or (not callable(record)):
            raise ValueError('initial shared entry requires fresh actor32/DP32 on exactly two owned nodes')
        base_env = dict(base_env, SPECO_COLOCATED_FEATURE_HOST_MIRROR='1', SPECO_COLOCATED_COLD_RESTORE='1' if config.get('joint_restore_binding') else '0')
        self.config, self.env, self.record = (deepcopy(config), dict(base_env), record)
        self.pool = MultiNodeColocatedResourcePool(run_id=config['run_id'], generation=config['generation'], node_ips=tuple(config['actor_ips']), python=config['python'], base_env=base_env, master_ports=dict(actor=config['actor_master_port'], draft=config['draft_master_port']), record_owned=lambda receipt: record('owned-colocated-pool', receipt))
        self.bootstrap = None
        self.replica = None
        self.backend = None
        self.phase_control = None

    async def initialize_workers(self, actor_settings, draft_settings):
        from verl_speco.workers.accelerator_adapters import MultiNodeColocatedActorWorker as MultiNodeColocatedActorWorker, MultiNodeColocatedDraftWorker as MultiNodeColocatedDraftWorker
        self.bootstrap = MultiNodeColocatedWorkerBootstrap(pool=self.pool, actor_class=MultiNodeColocatedActorWorker, draft_class=MultiNodeColocatedDraftWorker, actor_settings=actor_settings, draft_settings=draft_settings, record=self.record)
        self.binding = await self.bootstrap.initialize()
        return (self.bootstrap.groups['actor'], self.bootstrap.initial['actor'], self.bootstrap.groups['draft'], self.bootstrap.initial['draft'])

    async def initialize_rollout(self, engine_kwargs):
        from verl.workers.config import RolloutConfig, HFModelConfig
        from verl_speco.workers.accelerator_adapters import MultiNodeColocatedNativeReplica as MultiNodeColocatedNativeReplica
        from verl_speco.integration.multinode_recovery import select_colocated_native_replica, require_distinct_checkpoint_devices
        from verl_speco.integration.multinode_recovery import colocated_rollout_engine_kwargs
        kwargs = colocated_rollout_engine_kwargs(engine_kwargs)
        rollout = RolloutConfig(name='vllm', mode='async', tensor_model_parallel_size=8, data_parallel_size=1, pipeline_model_parallel_size=1, expert_parallel_size=1, dtype='bfloat16', load_format='auto', enable_sleep_mode=True, free_cache_engine=True, enforce_eager=kwargs['enforce_eager'], max_model_len=1024, max_num_seqs=1, max_num_batched_tokens=1024, enable_chunked_prefill=True, enable_prefix_caching=False, gpu_memory_utilization=engine_kwargs['gpu_memory_utilization'], disable_log_stats=True, response_length=self.config.get('response_tokens', 32), prompt_length=1024 - self.config.get('response_tokens', 32), engine_kwargs={'vllm': kwargs})
        model = HFModelConfig(path=self.config['model_overlay'], trust_remote_code=True)
        replica_rank = select_colocated_native_replica(self.binding, 8)
        self.replica = MultiNodeColocatedNativeReplica(replica_rank, rollout, model, gpus_per_node=16, placement_binding=self.binding, server_python=self.config['python'], base_env=self.env, record_owned=lambda row: self.record('owned-native-server-' + str(row['node_rank']), row))
        await drain_owned_call(self.replica.init_hybrid, self.bootstrap.groups['actor'])
        require_distinct_checkpoint_devices(self.binding['actor'][0], self.replica._speco_native_bootstrap_receipt['native_placements'])
        self.record('actual-native-bootstrap', self.replica._speco_native_bootstrap_receipt)
        self.backend = MultiNodeColocatedNativeEngine(replica=self.replica, binding=self.binding)
        statuses = await self.backend.collective_rpc('speco_dspark_runtime_status')
        if type(statuses) is not list or len(statuses) != 8 or {r.get('rank') for r in statuses} != set(range(8)) or (len({r.get('worker_incarnation') for r in statuses}) != 8):
            raise RuntimeError('shared rollout lacks complete independent current native owners')
        owners = {r['rank']: r['worker_incarnation'] for r in statuses}

        async def verify(phase, request):
            return await self.replica.verify_speco_phase(phase, request, owner_incarnations=owners)
        port = ColocatedVeRLRolloutPort(replica=self.replica, verify_owned_phase=verify, run_id=self.config['run_id'], generation=self.config['generation'])

        def validate(proof):
            return validate_colocated_publication(proof, owners=owners, run_id=self.config['run_id'], generation=self.config['generation'])
        restore_validator = None
        if self.config.get('joint_restore_binding'):
            from verl_speco.integration.multinode_recovery import preflight_colocated_joint_parent
            from verl_speco.integration.multinode_recovery import validate_joint_release
            parent = preflight_colocated_joint_parent(self.config, serving_view=self.config['draft_model_path'])
            all_owners = dict(actor={r['rank']: r['worker_incarnation'] for r in self.binding['actor']}, draft={str(r['rank']): r['worker_incarnation'] for r in self.binding['draft']}, rollout=owners)

            def restore_validator(proof):
                validate_joint_release(parent, proof, all_owners, run_id=self.config['run_id'], generation=self.config['generation'], binding_checksum=self.config['joint_restore_binding']['checksum'])
                return True
        self.phase_control = ColocatedCardCoordinator(run_id=self.config['run_id'], generation=self.config['generation'], actor_placements=self.binding['actor'], draft_placements=self.binding['draft'], memory_transition=self.bootstrap.memory_transition, rollout_sleep=port.sleep, rollout_wake=port.wake, publication_validator=validate, restore_validator=restore_validator, record_transition=self.record, actor_optimizer_cpu=True)
        self.record('shared-start-rollout', await self.phase_control.start_rollout(resume_generation=not bool(self.config.get('joint_restore_binding'))))
        return (self.backend, statuses)

    async def close(self):
        if self.backend is not None:
            await self.backend.close()
        elif self.replica is not None:
            import ray
            for handle in self.replica.server_class.handles:
                ray.kill(handle, no_restart=True)
        await drain_owned_call(self.pool.close)

def validate_colocated_publication(proof, *, owners, run_id, generation):
    from verl_speco.integration.multinode_publication import TargetWeightAck
    from verl_speco.integration.multinode_publication import require_draft_update_acks
    versions = proof['versions']
    target = proof['target_publication']
    snapshot = proof['draft_snapshot']
    draft = proof['draft_acks']
    acknowledgements = tuple((TargetWeightAck(**row) for row in target['acknowledgements']))
    if versions.run_id != run_id or versions.generation != generation or target['target_revision'] != versions.target_revision or (len(acknowledgements) != len(owners)) or ({ack.worker_id for ack in acknowledgements} != {str(rank) for rank in owners}) or any((ack.worker_incarnation != owners[int(ack.worker_id)] or ack.run_id != versions.run_id or ack.generation != versions.generation or (ack.rl_step != versions.rl_step) or (ack.target_revision != versions.target_revision) or (ack.actor_weight_manifest_sha256 != target['actor_weight_manifest_sha256']) or (ack.kv_cache_cleared is not True) or (ack.draft_target_head_synced is not True) for ack in acknowledgements)) or (snapshot['draft_revision'] != versions.draft_revision) or (type(draft) is not list) or (not draft):
        raise RuntimeError('shared rollout lacks independently validated current target/draft ACKs')
    require_draft_update_acks(draft, expected_incarnations=owners, run_id=versions.run_id, generation=versions.generation, operation_id=draft[0]['operation_id'], snapshot=snapshot, phase='applied')
    return True
import asyncio
import time
from dataclasses import asdict, dataclass
from inspect import isawaitable
from verl_speco.integration.multinode_online import AsyncCollectionRequest
from verl_speco.integration.multinode_online import ActorCollectionBatch

@dataclass(frozen=True)
class MultiNodeOnlineRequestGroup:
    collection_id: str
    requests: tuple[AsyncCollectionRequest, ...]
    prompt_sample_ids: tuple[str, ...] = ()

async def online_task_loop_call(port, *args):
    value = port(*args)
    return await value if isawaitable(value) else value

async def online_task_loop_iterate(groups):
    if hasattr(groups, '__aiter__'):
        async for group in groups:
            yield group
    else:
        for group in groups:
            yield group

class MultiNodeOnlineTaskLoop:

    def __init__(self, *, session, checkpoint_executor, build_actor_batch, persist_step, reap_sources, close_owned, abort_owned, max_steps, checkpoint_interval, queue_capacity_bytes=None):
        if any((not callable(p) for p in (build_actor_batch, persist_step, reap_sources, close_owned, abort_owned))):
            raise ValueError('online task loop requires real batch, persistence and owned lifecycle ports')
        if any((type(v) is not int or v < 1 for v in (max_steps, checkpoint_interval))):
            raise ValueError('online step budget and checkpoint interval must be positive integers')
        if not callable(getattr(checkpoint_executor, 'save', None)):
            raise ValueError('online task loop requires a joint safe-point saver')
        if queue_capacity_bytes is not None and (type(queue_capacity_bytes) is not int or queue_capacity_bytes < 1):
            raise ValueError('online queue capacity must be a positive byte count')
        self.queue_capacity_bytes = queue_capacity_bytes
        self.session, self.saver = (session, checkpoint_executor)
        self.build_batch, self.persist, self.reap = (build_actor_batch, persist_step, reap_sources)
        self.close, self.abort = (close_owned, abort_owned)
        self.max_steps, self.interval = (max_steps, checkpoint_interval)
        self.phase, self.completed_steps = ('new', 0)
        self.lock = asyncio.Lock()

    async def run(self, groups):
        async with self.lock:
            if self.phase != 'new':
                raise RuntimeError('completed or uncertain online loop cannot be replayed')
            self.phase = 'running'
            seen = set()
            try:
                previous = self.session.adapter.versions
                if previous != self.session.collection.versions:
                    raise RuntimeError('online task loop begins with inconsistent serving versions')
                async for group in online_task_loop_iterate(groups):
                    if not isinstance(group, MultiNodeOnlineRequestGroup) or not isinstance(group.collection_id, str) or (not group.collection_id) or (group.collection_id in seen) or (not isinstance(group.requests, tuple)) or (not group.requests) or any((not isinstance(r, AsyncCollectionRequest) for r in group.requests)) or (len({r.request_id for r in group.requests}) != len(group.requests)):
                        raise ValueError('online task loop needs unique complete request groups')
                    seen.add(group.collection_id)
                    step_started = time.perf_counter()
                    commit = await self.session.collection.collect_group(collection_id=group.collection_id, requests=group.requests)
                    collected_at = time.perf_counter()
                    batch = await online_task_loop_call(self.build_batch, commit, group)
                    built_at = time.perf_counter()
                    if not isinstance(batch, ActorCollectionBatch) or batch.collection_id != commit.collection_id or batch.tq_keys != commit.tq_keys or (len(batch.samples) != len(group.requests)) or any((sample.complete is not True for sample in batch.samples)):
                        raise RuntimeError('actor batch differs from its completed collection')
                    result = await self.session.update(commit, batch)
                    updated_at = time.perf_counter()
                    current = result['versions']
                    if current != self.session.adapter.versions or current != self.session.collection.versions or current.run_id != previous.run_id or (current.generation != previous.generation) or (current.rl_step != previous.rl_step + 1) or (current.target_revision != previous.target_revision + 1) or (current.data_version != previous.data_version + 1) or (current.draft_revision - previous.draft_revision not in (0, 1)):
                        raise RuntimeError('online task loop lacks one complete serving safe-point transition')
                    await online_task_loop_call(self.reap)
                    reaped_at = time.perf_counter()
                    completed = self.completed_steps + 1
                    binding = None
                    if current.rl_step % self.interval == 0 or completed == self.max_steps:
                        binding = await self.saver.save(result)
                        if not isinstance(binding, dict) or not isinstance(binding.get('uri'), str) or (not isinstance(binding.get('checksum'), str)) or (len(binding['checksum']) != 64) or any((c not in '0123456789abcdef' for c in binding['checksum'])):
                            raise RuntimeError('joint saver returned no complete checkpoint binding')
                    checkpointed_at = time.perf_counter()
                    original_tokens = sum((len(r.prompt_token_ids) + len(r.response_token_ids) for r in batch.samples))
                    elapsed = checkpointed_at - step_started
                    receipt = dict(collection_id=commit.collection_id, request_ids=tuple((r.request_id for r in group.requests)), versions=asdict(current), actual_completed_steps=completed, checkpoint_binding=binding, complete_sequence_count=len(batch.samples), input_tokens=original_tokens, queue_capacity_bytes=self.queue_capacity_bytes, timing_seconds=dict(collection=collected_at - step_started, actor_batch=built_at - collected_at, update=updated_at - built_at, reap=reaped_at - updated_at, checkpoint=checkpointed_at - reaped_at, total=elapsed), throughput=dict(original_tokens=original_tokens, completed_original_token_s=original_tokens / elapsed if elapsed > 0 else None, scope='online_group_entry_to_verified_safe_point_before_receipt_publication', includes_joint_checkpoint=binding is not None, excludes_prompt_group_construction=True, production_sla_pass=False))
                    await online_task_loop_call(self.persist, receipt)
                    self.completed_steps = completed
                    previous = current
                    if completed == self.max_steps:
                        break
                if self.completed_steps != self.max_steps:
                    raise RuntimeError('online dataset ended before the requested training step budget')
                await online_task_loop_call(self.close)
                self.phase = 'finished'
                return dict(completed_steps=self.completed_steps, versions=asdict(previous), final_checkpoint_binding=binding)
            except BaseException:
                self.phase = 'failed'
                self.session.collection.failed = True
                await online_task_loop_call(self.abort)
                raise
import math
from dataclasses import replace
from verl_speco.integration.multinode_online import ActorSample, validate_actor_samples
from verl_speco.transport.streaming import SampleMetadataV3

class MultiNodeOnlineActorBatchBuilder:

    def __init__(self, *, load_completed, reward, temperature, vocab_size):
        if not callable(load_completed) or not callable(reward) or type(temperature) not in (int, float) or (not math.isfinite(temperature)) or (temperature <= 0) or (type(vocab_size) is not int) or (vocab_size < 1):
            raise ValueError('online actor batches need actual completion and explicit reward ports')
        self.load, self.reward = (load_completed, reward)
        self.temperature, self.vocab_size = (temperature, vocab_size)

    async def __call__(self, commit, group):
        if not isinstance(group, MultiNodeOnlineRequestGroup) or group.collection_id != commit.collection_id or (not group.requests) or (not isinstance(group.prompt_sample_ids, tuple)) or (len(group.prompt_sample_ids) != len(group.requests)) or any((not isinstance(s, str) or not s for s in group.prompt_sample_ids)) or (len(commit.tq_keys) != len(group.requests)) or (len(set(commit.tq_keys)) != len(commit.tq_keys)) or (len({r.request_id for r in group.requests}) != len(group.requests)):
            raise ValueError('online actor batch needs exact collection and original prompt identities')
        rows = []
        versions = commit.versions
        for request, prompt_id in zip(group.requests, group.prompt_sample_ids, strict=True):
            completed = self.load(request.request_id)
            if isawaitable(completed):
                completed = await completed
            if not isinstance(completed, dict) or completed.get('complete') is not True or completed.get('temperature') != self.temperature:
                raise ValueError('actor completion is partial or has a different sampling temperature')
            prompt = tuple(completed['prompt_token_ids'])
            response = tuple(completed['response_token_ids'])
            item = SampleMetadataV3.from_dict(completed['metadata'])
            expected = request.metadata(versions, prompt, response) if callable(request.metadata) else request.metadata
            if isawaitable(expected):
                expected = await expected
            if item != SampleMetadataV3.from_dict(expected):
                raise ValueError('actor completion metadata differs from the complete request provenance')
            if item.request_id != request.request_id or item.sample_id != request.request_id or item.run_id != versions.run_id or (item.generation != versions.generation) or (item.rl_step != versions.rl_step) or (item.data_version != versions.data_version) or (item.target_revision != str(versions.target_revision)) or (item.draft_revision != str(versions.draft_revision)) or (prompt != request.prompt_token_ids) or (item.prompt_token_count != len(prompt)) or (item.token_count != len(prompt) + len(response) - 1) or (item.supervised_token_count != len(response) - 1) or (len(response) < 9):
                raise ValueError('actor completion differs from its exact feature publication boundary')
            rows.append(ActorSample(request.request_id, prompt_id, prompt, response, 0.0, versions.target_revision, self.temperature))
        validate_actor_samples(rows, target_revision=versions.target_revision, vocab_size=self.vocab_size)
        scored = []
        for sample in rows:
            score = self.reward(sample)
            if isawaitable(score):
                score = await score
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('online reward backend returned a nonfinite or nonscalar score')
            scored.append(replace(sample, reward=float(score)))
        return ActorCollectionBatch(commit.collection_id, commit.tq_keys, tuple(scored))
from collections.abc import Mapping
import re
from string import Formatter
from urllib.parse import urlsplit

def online_endpoint_preflight_port(value, label):
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError(f'{label} requires an integer TCP port in 1..65535')
    return value

def online_endpoint_preflight_endpoint(value, label, *, url=False):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f'{label} requires an explicit endpoint')
    parsed = urlsplit(value if url else 'tcp://' + value)
    if parsed.scheme not in (('http', 'https') if url else ('tcp',)) or not parsed.hostname or parsed.username is not None or (parsed.password is not None) or (not url and (parsed.path or parsed.query or parsed.fragment)):
        raise ValueError(f'{label} has an invalid endpoint address')
    if not re.fullmatch('[0-9]+', parsed.netloc.rsplit(':', 1)[-1]):
        raise ValueError(f'{label} requires an explicit numeric TCP port')
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f'{label} requires a TCP port in 1..65535') from exc
    online_endpoint_preflight_port(port, label)
    return dict(host=parsed.hostname.lower(), port=port)

def validate_speco_online_endpoints(config, *, environment=None):
    """Pure range/template checks; never binds sockets or imports a device.

    Validate every supplied endpoint while allowing data-only CPU fixtures.
    The native source address comes from the launch environment; the draft
    receiver addresses come from the functional recipe's DP2 template.
    """
    if not isinstance(config, Mapping):
        raise ValueError('online endpoint preflight requires a mapping')
    environment = os.environ if environment is None else environment
    ports = {key: online_endpoint_preflight_port(config[key], key) for key in ('actor_master_port', 'draft_master_port', 'metadata_rpc_port', 'metadata_http_port') if key in config}
    endpoints = {}
    if 'ray_address' in config and config['ray_address'] != 'auto':
        value = config['ray_address']
        endpoints['ray_address'] = online_endpoint_preflight_endpoint(value.removeprefix('ray://') if isinstance(value, str) else value, 'ray_address')
    if 'metadata_url' in config:
        endpoints['metadata_url'] = online_endpoint_preflight_endpoint(config['metadata_url'], 'metadata_url', url=True)
        if 'metadata_http_port' in ports and endpoints['metadata_url']['port'] != ports['metadata_http_port']:
            raise ValueError('metadata_url port differs from metadata_http_port')
    if 'SPECO_NATIVE_SOURCE_ENDPOINT' in environment:
        endpoints['native_source'] = online_endpoint_preflight_endpoint(environment['SPECO_NATIVE_SOURCE_ENDPOINT'], 'SPECO_NATIVE_SOURCE_ENDPOINT')
    if 'native_draft_endpoint_template' in config:
        template = config['native_draft_endpoint_template']
        if not isinstance(template, str):
            raise ValueError('native draft endpoint template must be a string')
        try:
            fields = [field for _, field, _, conversion in Formatter().parse(template) if field is not None]
            if not fields or any((field != 'rank' for field in fields)):
                raise ValueError('native draft endpoint template requires only {rank}')
            addresses = [online_endpoint_preflight_endpoint(template.format(rank=rank), f'native_draft rank{rank}') for rank in range(2)]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError('invalid native draft endpoint template') from exc
        if len({(x['host'], x['port']) for x in addresses}) != 2:
            raise ValueError('DP2 draft receivers require distinct endpoints')
        endpoints['native_draft'] = addresses
    return dict(ports=ports, endpoints=endpoints, before_NPU_role_creation=True, network_bind_or_liveness_not_checked=True)
import hashlib
import importlib
import json
from pathlib import Path

def online_data_contract(config):
    if config.get('functional_validation') is not True:
        raise ValueError('this entry currently requires explicit functional_validation=true')
    for key in ('steps', 'prompts_per_group', 'responses_per_prompt', 'checkpoint_interval'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError('online data recipe requires positive integer step and group budgets')
    if config['responses_per_prompt'] < 2 or config['prompts_per_group'] * config['responses_per_prompt'] != 32:
        raise ValueError('four-node functional actor needs a complete 32-response GRPO batch')
    for key, default in (('temperature', 0.7), ('top_p', 0.95)):
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('online sampling requires finite positive temperature and top_p')
    if config.get('top_p', 0.95) > 1:
        raise ValueError('online top_p must not exceed one')
    if type(config.get('response_tokens', 32)) is not int or not 9 <= config.get('response_tokens', 32) <= 256:
        raise ValueError('functional response length must fit the bounded native feature slots')
    if type(config.get('seed', 42)) is not int:
        raise ValueError('online sampling seed must be an integer')
    path = Path(config['prompt_input'])
    if not path.is_absolute() or not path.is_file():
        raise ValueError('online real prompt input must be an explicit existing absolute file')
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b''):
            digest.update(block)
    if config.get('dataset_fingerprint') != digest.hexdigest():
        raise ValueError('online prompt bytes differ from the bound dataset fingerprint')
    reward = config.get('reward')
    if not isinstance(reward, str) or not reward or (reward != 'diagnostic_response_sha256' and reward.count(':') != 1):
        raise ValueError('online task needs an explicit diagnostic or module:function reward')
    return dict(dataset_fingerprint=digest.hexdigest(), prompt_input=str(path.resolve()), prompts_per_group=config['prompts_per_group'], responses_per_prompt=config['responses_per_prompt'], response_tokens=config.get('response_tokens', 32), temperature=config.get('temperature', 0.7), top_p=config.get('top_p', 0.95), seed=config.get('seed', 42), reward=reward, functional_validation=True, max_model_len=1024)

def load_reward(config):
    if config['reward'] == 'diagnostic_response_sha256':

        def diagnostic(sample):
            digest = hashlib.sha256(json.dumps(sample.response_token_ids).encode()).digest()
            return int.from_bytes(digest[:8], 'big') / 2 ** 64
        return diagnostic
    module, name = config['reward'].split(':')
    reward = getattr(importlib.import_module(module), name)
    if not callable(reward):
        raise ValueError('online reward module:function did not resolve to a callable')
    return reward
from itertools import islice

def validate_online_prompt_budget(config, *, start_step):
    if type(start_step) is not int or start_step < 0:
        raise ValueError('online prompt cursor requires a nonnegative integer restored step')
    online_data_contract(config)
    from verl_speco.producer.input_reader import iter_input_records
    consumed = start_step * config['prompts_per_group']
    required = config['steps'] * config['prompts_per_group']
    seen = set()
    count = 0
    for record in islice(iter_input_records(config['prompt_input'], on_error='raise'), consumed + required):
        if record.sample_id in seen or record.response is not None:
            raise ValueError('online prompt prefix requires unique prompt-only records')
        seen.add(record.sample_id)
        count += 1
    if count < consumed + required:
        raise ValueError(f'online prompt budget exhausted: restored cursor consumes {consumed} records; {required} new records required, {max(0, count - consumed)} remaining in {count} available records')
    return dict(restored_rl_step=start_step, consumed_prompt_records=consumed, required_new_prompt_records=required, verified_prompt_prefix_records=count, requested_new_steps=config['steps'], prompts_per_group=config['prompts_per_group'])
from hashlib import sha256
from functools import partial
from verl_speco.producer.input_reader import prepare_generation_request

def iter_online_prompt_groups(records, *, tokenizer, run_id, generation, producer_id, producer_incarnation, dataset_fingerprint, tokenizer_fingerprint, target_model_fingerprint, target_layer_ids, target_hidden_size, response_tokens, max_model_len, prompts_per_group, responses_per_prompt, first_rl_step, publication_settings, generate, seed=42):
    lengths = (generation, target_hidden_size, response_tokens, max_model_len, prompts_per_group, responses_per_prompt)
    if any((type(v) is not int or v < 1 for v in lengths)) or responses_per_prompt < 2 or type(first_rl_step) is not int or (first_rl_step < 1) or (type(seed) is not int) or (response_tokens < 9) or (not callable(generate)) or (not isinstance(dataset_fingerprint, str)) or (len(dataset_fingerprint) != 64) or any((c not in '0123456789abcdef' for c in dataset_fingerprint)) or (not target_layer_ids) or any((type(i) is not int or i < 0 for i in target_layer_ids)) or (len(set(target_layer_ids)) != len(target_layer_ids)):
        raise ValueError('online prompt groups require explicit GRPO, context and provenance contracts')
    row_bytes = (len(target_layer_ids) + 1) * target_hidden_size * 2 + 20
    iterator = iter(records)
    seen_samples = set()
    group_index = 0
    while True:
        prompts = []
        for _ in range(prompts_per_group):
            try:
                record = next(iterator)
            except StopIteration:
                if prompts:
                    raise ValueError('online dataset ended inside a complete GRPO collection')
                return
            if record.sample_id in seen_samples or type(record.sequence_no) is not int or record.sequence_no < 0:
                raise ValueError('online prompt source has duplicate or invalid original sample identity')
            seen_samples.add(record.sample_id)
            prepared = prepare_generation_request(record, tokenizer, dict(max_sequence_length=max_model_len, generation_max_tokens=response_tokens))
            if prepared.max_tokens < 9:
                raise ValueError('online prompt leaves less than one complete shifted draft block')
            prompts.append(prepared)
        step = first_rl_step + group_index
        requests = []
        prompt_ids = []
        for prompt_index, prompt in enumerate(prompts):
            identity = sha256(prompt.sample_id.encode()).hexdigest()
            for response_index in range(responses_per_prompt):
                request_id = f'{run_id}-g{generation}-step{step}-{identity}-{response_index}'
                sequence_no = ((first_rl_step - 1 + group_index) * prompts_per_group + prompt_index) * responses_per_prompt + response_index

                def metadata(bound, actual_prompt, response, *, expected_prompt=prompt.prompt_token_ids, request_id=request_id, sequence_no=sequence_no, limit=prompt.max_tokens, expected_step=step):
                    if bound.run_id != run_id or bound.generation != generation or bound.rl_step != expected_step or (tuple(actual_prompt) != expected_prompt) or (not 9 <= len(response) <= limit):
                        raise ValueError('online completion differs from its owned prompt or draft response geometry')
                    rows = len(actual_prompt) + len(response) - 1
                    item = SampleMetadataV3(3, run_id, generation, producer_id, producer_incarnation, request_id, request_id, sequence_no, str(bound.target_revision), bound.data_version, tokenizer_fingerprint, target_model_fingerprint, tuple(target_layer_ids), 'dflash_aux_plus_last', 'bfloat16', rows, rows * row_bytes, rl_step=bound.rl_step, dataset_fingerprint=dataset_fingerprint, draft_revision=str(bound.draft_revision), prompt_token_count=len(actual_prompt), supervised_token_count=len(response) - 1)
                    item.validate()
                    return item.to_dict()
                stream = partial(generate, prompt_token_ids=prompt.prompt_token_ids, request_id=request_id, max_tokens=prompt.max_tokens, seed=seed + sequence_no)
                prompt_ids.append(prompt.sample_id)
                requests.append(AsyncCollectionRequest(request_id, prompt.prompt_token_ids, stream, metadata, dict(publication_settings)))
        yield MultiNodeOnlineRequestGroup(f'{run_id}-g{generation}-collection-{step}', tuple(requests), tuple(prompt_ids))
        group_index += 1
import sys

class MultiNodeOnlineTaskRunner:

    def run(self, config):
        from omegaconf import OmegaConf
        path = OmegaConf.select(config, 'speco.multinode_online_config')
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError('target model TaskRunner needs an absolute speco.multinode_online_config path')
        online = json.loads(Path(path).read_text())
        online_data_contract(online)
        load_reward(online)
        if os.environ.get('SPECO_RUN_ID') != online['run_id'] or os.environ.get('SPECO_GENERATION') != str(online['generation']):
            raise ValueError('target model TaskRunner runtime environment differs from the reserved run owner')
        endpoints = validate_speco_online_endpoints(online)
        if online.get('colocated') is True:
            validate_colocated_entry_config(online, os.environ)
        import ray
        from verl_speco.runtime.services import restore_native_teacher_visibility
        extension = sys.modules.get('torch_npu')
        initialized = False
        if extension is not None:
            check = getattr(getattr(extension, 'npu', None), 'is_initialized', None)
            if not callable(check):
                raise RuntimeError('native teacher needs a known NPU initialization state')
            initialized = check()
        if online.get('colocated') is True:
            visibility = dict(mode='colocated_native_server', private_teacher_mask_restored=False, actual_physical_native_pairing_deferred_to_server_bootstrap=True, driver_node_ip=ray.util.get_node_ip_address())
        else:
            visibility = restore_native_teacher_visibility(online, node_ip=ray.util.get_node_ip_address(), npu_initialized=initialized)
        if online.get('output_root'):
            output_root = Path(online['output_root'])
            if not output_root.is_absolute():
                raise ValueError('native teacher visibility receipt needs an absolute owned output root')
            from verl_speco.runtime.artifacts import publish_json
            publish_json(output_root / 'transport-endpoint-preflight.json', dict(run_id=online['run_id'], generation=online['generation'], **endpoints))
            publish_json(output_root / 'native-teacher-visibility.json', dict(run_id=online['run_id'], generation=online['generation'], **visibility))
        return asyncio.run(run(online))

def run_speco_online_ppo(config):
    """Use VeRL's actual Ray TaskRunner hook without editing the pinned fork."""
    from omegaconf import OmegaConf
    path = OmegaConf.select(config, 'speco.multinode_online_config')
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError('target model entry needs an absolute online configuration file')
    online = json.loads(Path(path).read_text())
    online_data_contract(online)
    validate_speco_online_endpoints(online)
    import ray
    from verl.trainer.main_ppo import run_ppo
    rollout_ip = online.get('rollout_ip')
    from ipaddress import ip_address
    if not isinstance(rollout_ip, str):
        raise ValueError('target model entry needs the reserved rollout node IP')
    ip_address(rollout_ip)
    runner = ray.remote(num_cpus=0.1 if online.get('colocated') is True else 1)(MultiNodeOnlineTaskRunner).options(resources={'node:' + rollout_ip: 0.001})
    run_ppo(config, task_runner_class=runner)
from dataclasses import asdict

async def run(config):
    validate_speco_online_endpoints(config)
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
    from verl_speco.workers.accelerator_adapters import MultiNodeJointActorWorker as MultiNodeJointActorWorker
    from verl_speco.workers.accelerator_adapters import MultiNodeJointDraftWorker as MultiNodeJointDraftWorker
    from verl_speco.integration.multinode_online import ActorCollectionBatch, MultiNode92BActorExecutor as MultiNode92BActorExecutor
    from verl_speco.integration.multinode_online import ActorSample
    from verl_speco.integration.multinode_online import AsyncCollectionRequest, OnlineCollectionCoordinator as AsyncMultiNodeCollectionCoordinator
    from verl_speco.integration.online_session import MultiNodeOnlinePostTrainingSession as MultiNodeOnlinePostTrainingSession
    from verl_speco.integration.multinode_publication import MultiNodeTargetPublishExecutor as MultiNodeTargetPublishExecutor
    from verl_speco.integration.multinode_online import OnlineVersions
    from verl_speco.integration.accelerator_speculation import dedicated_speco_dspark_config as dedicated_speco_dspark_config
    from verl_speco.integration.multinode_online import full_async_engine_kwargs
    from verl_speco.integration.accelerator_speculation import bind_speco_sink_reservation as bind_speco_sink_reservation
    from verl_speco.transport.control import actor_name_for_run
    from verl_speco.transport.control import MultiNodeJointReadyControl as MultiNodeJointReadyControl
    from verl_speco.integration.multinode_recovery import joint_diagnostic_config
    from verl_speco.integration.multinode_recovery import MultiNodeJointCheckpointStore as MultiNodeJointCheckpointStore
    from verl_speco.integration.multinode_recovery import MultiNodeJointCheckpointExecutor as MultiNodeJointCheckpointExecutor
    from verl_speco.integration.multinode_recovery import MultiNodeJointRestoreCoordinator as MultiNodeJointRestoreCoordinator
    from verl_speco.integration.joint_session import attach_joint_online_session
    from verl_speco.transport.streaming import SampleMetadataV3
    from verl_speco.runtime.artifacts import publish_json, read_published_json
    from verl_speco.runtime.services import register_owned_session
    from verl_speco.trainer.scheduler.schedule_types import DrafterScheduleConfig
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    if config.get('actor_trace_root'):
        raise ValueError('actor_trace_root refers to an excluded historical diagnostic; use normal worker metrics')
    run_id = config['run_id']
    generation = config['generation']
    register_owned_session(run_id, generation)
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    for name in ('sequences', 'steps', 'target-heads'):
        (root / name).mkdir(exist_ok=True)
    initial = read_published_json(Path(config['draft_model_path']) / 'snapshot-binding.json')['snapshot']
    manifest = read_published_json(Path(initial['uri']) / 'manifest.json')
    joint = joint_diagnostic_config(config, manifest)
    publish_json(root / 'resolved-joint-config.json', dict(config, **joint))
    if config.get('prompt_input'):
        publish_json(root / 'online-data-contract.json', joint['online_data_contract'])
    if config.get('prompt_input'):
        cursor = 0
        if joint.get('joint_restore_binding'):
            from verl_speco.integration.multinode_recovery import open_joint_parent
            parent, _ = open_joint_parent(joint, joint['joint_restore_binding'])
            cursor = parent['versions']['rl_step']
        budget = validate_online_prompt_budget(config, start_step=cursor)
        publish_json(root / 'online-prompt-budget-preflight.json', budget)
    versions = OnlineVersions(run_id, generation, 0, 0, 0, initial['draft_revision'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_overlay'], trust_remote_code=True)
    if config.get('prompt_input'):
        from verl_speco.producer.input_reader import iter_input_records, prepare_generation_request
        first = next(iter_input_records(config['prompt_input'], on_error='raise'))
        tokens = list(prepare_generation_request(first, tokenizer, dict(max_sequence_length=1024, generation_max_tokens=config.get('response_tokens', 32))).prompt_token_ids)
    else:
        tokens = tokenizer.encode('请简要解释机器学习中训练数据与测试数据的区别。', add_special_tokens=True)
    response_tokens = config.get('response_tokens', 32)
    from verl_speco.integration.multinode_online import validate_online_collection_geometry
    geometry = validate_online_collection_geometry(model_config=manifest['model_config'], prompt_token_count=len(tokens), response_tokens=response_tokens, sample_count=32, max_model_len=full_async_engine_kwargs(config)['max_model_len'])
    publish_json(root / 'collection-geometry-preflight.json', geometry)
    owns_ray_client = not ray.is_initialized()
    if owns_ray_client:
        ray.init(address=config['ray_address'], namespace='transfer_queue')
    elif ray.get_runtime_context().namespace != 'transfer_queue':
        raise ValueError('target model online roles require the reserved transfer_queue Ray namespace')
    live = {node['NodeManagerAddress']: node for node in ray.nodes() if node['Alive']}
    handles = []
    owned = []
    engine = None
    control = None
    shared = None
    base_env = {key: value for key, value in os.environ.items() if key.startswith(('ASCEND', 'HCCL', 'LCCL', 'CANN', 'SPECO', 'PANGU', 'VLLM')) or key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PATH', 'OMP_NUM_THREADS', 'CLOSE_MATMUL_K_SHIFT', 'SWA_OFF_WITH_SINKS', 'PYTHONDONTWRITEBYTECODE')}

    def ownership():
        publish_json(root / 'owned_ray_actors.json', dict(run_id=run_id, generation=generation, actors=owned, control=None if control is None else dict(name=control_name, actor_id=control._actor_id.hex())))

    async def create_group(cls, settings, ips, per_node, label, master_port):
        group_handles = []
        for node_index, ip in enumerate(ips):
            for local_rank in range(per_node):
                rank = node_index * per_node + local_rank
                env = dict(base_env, RANK=str(rank), LOCAL_RANK=str(local_rank), WORLD_SIZE=str(len(ips) * per_node), LOCAL_WORLD_SIZE=str(per_node), MASTER_ADDR=ips[0], MASTER_PORT=str(master_port), ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, range(per_node))), RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES='1')
                if label == 'actor':
                    env['PANGU_MODEL_PATH'] = config['model_path']
                name = run_id + '-g' + str(generation) + '-' + label + '-' + str(rank)
                worker = ray.remote(cls).options(name=name, num_cpus=0.25, resources={'NPU': 1}, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=live[ip]['NodeID'], soft=False), runtime_env=dict(py_executable=config['python'], env_vars=env)).remote(settings)
                handles.append(worker)
                group_handles.append(worker)
                owned.append(dict(name=name, actor_id=worker._actor_id.hex()))
                ownership()
        group = RayWorkerGroup(ray_cls_with_init=RayClassWithInitArgs(ray.remote(cls), config=settings), worker_handles=group_handles, name_prefix=run_id + '-' + label, use_gpu=False, device_name='npu')
        initialized = await asyncio.to_thread(group.init_model)
        publish_json(root / (label + '-initialized.json'), dict(workers=initialized))
        return (group, initialized)
    control_name = actor_name_for_run(run_id, generation)
    partition = 'full-online-' + run_id
    try:
        control = ray.remote(MultiNodeJointReadyControl).options(name=control_name, namespace='transfer_queue', num_cpus=0.1, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=live[config['trainer_ip']]['NodeID'], soft=False)).remote(run_id=run_id, generation=generation, partition=partition, capacity_bytes=512 * 1024 ** 2)
        ownership()
        draft_settings = dict(run_id=run_id, generation=generation, rollout_nodes=[config['rollout_hostname']], initial_snapshot=initial, model_config=manifest['model_config'], target_model_fingerprint=manifest['target_model_fingerprint'], tokenizer_fingerprint=manifest['tokenizer_fingerprint'], target_model_dir=config['model_path'], initial_target_revision=0, ray_address=config['ray_address'], control_actor_name=control_name, partition=partition, native_endpoint_template=config['native_draft_endpoint_template'], metadata_server=config['metadata_url'], output_root=str(root / 'training'), total_steps=config['total_training_steps'], lr=config.get('draft_learning_rate', 0.0001), seed=42, native_max_bytes=128 * 1024 ** 2)
        draft_settings.update(joint)
        actor_settings = dict(run_id=run_id, generation=generation, profile_cache=config['profile_cache'], model_path=config['model_path'], total_steps=config['total_training_steps'], learning_rate=config.get('learning_rate', 1e-06), target_transfer_output_root=str(root / 'target-heads'), actor_checkpoint_root=str(Path(joint['joint_run_root']) / 'actor-checkpoints'))
        actor_settings.update(joint)
        if config.get('actor_trace_root'):
            actor_settings['actor_trace_root'] = config['actor_trace_root']
        if config.get('colocated') is True:
            shared = MultiNodeColocatedTrainingRuntime(config=dict(config, **joint), base_env=base_env, record=lambda name, receipt: publish_json(root / (name + '.json'), receipt))
            actor, actor_init, draft, draft_init = await shared.initialize_workers(actor_settings, draft_settings)
        else:
            draft, draft_init = await create_group(MultiNodeJointDraftWorker, draft_settings, [config['trainer_ip']], 2, 'draft', config['draft_master_port'])
            actor, actor_init = await create_group(MultiNodeJointActorWorker, actor_settings, config['actor_ips'], 16, 'actor', config['actor_master_port'])
        draft_owners = {row['worker_id']: row['worker_incarnation'] for row in draft_init}
        actor_owners = {row['rank']: row['worker_incarnation'] for row in actor_init}
        assert len(actor_owners) == 32 and all((row['optimizer_step'] == 0 for row in actor_init))
        print('FULL_ASYNC_ACTOR_AND_DP_ONLY_DRAFT_INITIALIZED' if shared is not None else 'FULL_ASYNC_ACTOR_AND_INDEPENDENT_DRAFT_INITIALIZED', len(draft_owners), flush=True)
        kwargs = full_async_engine_kwargs(config)
        if shared is not None:
            engine, _ = await shared.initialize_rollout(kwargs)
        else:
            with dedicated_speco_dspark_config(kwargs):
                native = AsyncEngineArgs(**kwargs).create_engine_config()
                publish_json(root / 'native_sink_reservation.json', bind_speco_sink_reservation(native, os.environ))
                engine = AsyncLLM.from_vllm_config(native)

        async def rpc(name, request):
            return await engine.collective_rpc(name, kwargs=request, timeout=3600)
        statuses = await rpc('speco_dspark_runtime_status', {})
        rollout_owners = {row['rank']: row['worker_incarnation'] for row in statuses}
        assert len(rollout_owners) == 8
        from verl_speco.integration.multinode_publication import preconnect_target_channel
        from uuid import uuid4
        channel_identity = dict(run_id=run_id, generation=generation, channel_id=uuid4().hex, actors=actor_owners, rollout=rollout_owners, bucket_bytes=1024 ** 3)
        channel_proof = await preconnect_target_channel(actor_group=actor, collective_rpc=rpc, identity=channel_identity)
        publish_json(root / 'checkpoint-channel-preconnected.json', channel_proof)
        print('FULL_ASYNC_CHECKPOINT_CHANNEL_CONNECTED_BEFORE_FIRST_PPO', flush=True)

        async def save(item, prompt, response):
            publish_json(root / 'sequences' / (item.sample_id + '.json'), dict(metadata=item.to_dict(), prompt_token_ids=list(prompt), response_token_ids=list(response), complete=True, temperature=config.get('temperature', 0.7), top_p=config.get('top_p', 0.95)))

        async def abort(request_id, error):
            await engine.abort(request_id)
            await rpc('speco_accepted_query_abort', {})
            publish_json(root / 'owned-abort.json', dict(request_id=request_id, error=error))

        async def admit(bound, requests):
            settings = requests[0].publication_settings
            assert all((item.publication_settings == settings for item in requests))
            payload_bound = max(((len(item.prompt_token_ids) + response_tokens - 1) * (15360 * 2 + 20) for item in requests))
            rows = await rpc('speco_admit_online_collection', dict(sample_count=len(requests), payload_bytes_per_sample=payload_bound, producer_endpoint='p0', **settings))
            assert len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
            assert all((row['admitted'] is True and row['source_owner'] == (row['rank'] == 0) for row in rows))
            publish_json(root / 'steps' / ('admission-' + str(bound.rl_step) + '.json'), dict(workers=rows))
        collection = AsyncMultiNodeCollectionCoordinator(versions=versions, collective_rpc=rpc, expected_ranks=tuple(range(8)), producer_id='p0', save_complete_sequence=save, abort_collection=abort, admit_collection=admit)
        update = MultiNode92BActorExecutor(worker_group=actor, expected_incarnations=actor_owners, run_id=run_id, generation=generation, vocab_size=151552)

        async def update_actor(batch, binding):
            result = await update(batch, binding)
            publish_json(root / 'steps' / ('actor-update-' + str(binding.versions.rl_step) + '.json'), result)
            print('FULL_ASYNC_ACTUAL_32_DIE_ACTOR_OPTIMIZER_PASS', result['optimizer_step'], flush=True)
            return result
        publisher = MultiNodeTargetPublishExecutor(actor_group=actor, actor_incarnations=actor_owners, draft_group=draft, draft_incarnations=draft_owners, collective_rpc=rpc, rollout_incarnations=rollout_owners, reset_prefix_cache=engine.reset_prefix_cache, tokenizer_fingerprint=manifest['tokenizer_fingerprint'])

        def resolve(value):
            return ray.get(value) if isinstance(value, list) and value and isinstance(value[0], ray.ObjectRef) else value

        async def publish_actor(result, binding):
            proof = await publisher(result, binding)
            publish_json(root / 'steps' / ('target-publication-' + str(binding.versions.rl_step) + '.json'), publisher.last_proof)
            return proof

        def abort_plan(binding, reason):
            publish_json(root / 'session-abort.json', dict(reason=reason, plan=None if binding is None else binding.to_worker_payload()))
        owners = dict(actor=actor_owners, draft=draft_owners, rollout=rollout_owners)

        async def control_rpc(method, kwargs):
            return await getattr(control, method).remote(**kwargs)

        async def abort_owned(expected):
            if expected != owners:
                raise RuntimeError('joint cleanup received foreign role incarnations')
            if shared is not None:
                await shared.close()
            elif engine is not None:
                engine.shutdown()
            for handle in handles:
                if handle._actor_id.hex() not in {row['actor_id'] for row in owned}:
                    raise RuntimeError('joint cleanup Ray actor identity changed')
                ray.kill(handle, no_restart=True)
        if joint.get('joint_restore_binding'):
            restore = MultiNodeJointRestoreCoordinator(config=joint, owners=owners, actor_group=actor, draft_group=draft, collective_rpc=rpc, control_rpc=control_rpc, reset_prefix_cache=engine.reset_prefix_cache, abort_owned=abort_owned, phase_control=shared.phase_control if shared is not None else None, record=lambda name, receipt: publish_json(root / (name + '.json'), receipt))
            restored = await restore.restore()
            publish_json(root / 'joint-restored.json', restored)
            attached = await attach_joint_online_session(coordinator=restore, restored=restored, resolve=resolve, vocab_size=151552, producer_id='p0', rollout_nodes={config['rollout_hostname']}, save_complete_sequence=save, abort_collection=abort, admit_collection=admit, abort_plan=abort_plan, schedule_config=DrafterScheduleConfig(publish_interval_steps=1, train_batches_per_trigger=1), parallel_updates=config.get('parallel_actor_draft_updates', False), **dict(colocated_placements=shared.binding, phase_control=shared.phase_control) if shared is not None else {})
            session = attached['session']
            collection = attached['collection']
            update = attached['actor_executor']
            publisher = attached['target_publisher']
            versions = attached['versions']
            session.actor_update = update_actor
            session.publish_actor = publish_actor
        else:
            session = MultiNodeOnlinePostTrainingSession(collection=collection, worker_group=draft, resolve=resolve, trainer_worker_ids=tuple(draft_owners), rollout_nodes={config['rollout_hostname']}, rollout_incarnations=rollout_owners, reset_prefix_cache=engine.reset_prefix_cache, abort_plan=abort_plan, actor_update=update_actor, publish_actor=publish_actor, schedule_config=DrafterScheduleConfig(publish_interval_steps=1, train_batches_per_trigger=1), parallel_updates=config.get('parallel_actor_draft_updates', False), **dict(colocated_placements=shared.binding, phase_control=shared.phase_control) if shared is not None else {})
        store = MultiNodeJointCheckpointStore(run_root=joint['joint_run_root'], run_id=run_id, generation=generation, fingerprints=joint['joint_fingerprints'], actor_incarnations=actor_owners, draft_incarnations=draft_owners, rollout_incarnations=rollout_owners)
        saver = MultiNodeJointCheckpointExecutor(store=store, session=session, actor_group=actor, draft_group=draft, target_publisher=publisher, control_rpc=control_rpc, abort_owned=abort_owned)
        start_step = versions.rl_step
        observations = []
        if config.get('prompt_input'):
            from itertools import islice

            def generate(**request):
                prompt = request.pop('prompt_token_ids')
                request_id = request.pop('request_id')
                return engine.generate(dict(prompt_token_ids=list(prompt)), SamplingParams(temperature=config.get('temperature', 0.7), top_p=config.get('top_p', 0.95), min_tokens=9, **request), request_id=request_id)
            records = islice(iter_input_records(config['prompt_input'], on_error='raise'), start_step * config['prompts_per_group'], None)
            groups = iter_online_prompt_groups(records, tokenizer=tokenizer, run_id=run_id, generation=generation, producer_id='p0', producer_incarnation='producer-' + run_id, dataset_fingerprint=config['dataset_fingerprint'], tokenizer_fingerprint=manifest['tokenizer_fingerprint'], target_model_fingerprint=manifest['target_model_fingerprint'], target_layer_ids=manifest['model_config']['target_layer_ids'], target_hidden_size=2560, response_tokens=response_tokens, max_model_len=kwargs['max_model_len'], prompts_per_group=config['prompts_per_group'], responses_per_prompt=config['responses_per_prompt'], first_rl_step=start_step + 1, publication_settings=dict(partition=partition, ray_address=config['ray_address'], capacity_bytes=512 * 1024 ** 2, control_actor_name=control_name), generate=generate, seed=config.get('seed', 42))
            build_batch = MultiNodeOnlineActorBatchBuilder(load_completed=lambda request_id: read_published_json(root / 'sequences' / (request_id + '.json')), reward=load_reward(config), temperature=config.get('temperature', 0.7), vocab_size=151552)

            async def persist_step(receipt):
                if shared is not None:
                    receipt = dict(receipt, rollout_acceptance=engine.step_acceptance_metrics(receipt['request_ids']))
                observations.append(receipt)
                publish_json(root / 'steps' / ('step-' + str(receipt['versions']['rl_step']) + '.json'), receipt)
                if receipt['checkpoint_binding'] is not None:
                    publish_json(root / 'joint-binding.json', dict(binding=receipt['checkpoint_binding'], fingerprints=joint['joint_fingerprints'], versions=receipt['versions'], snapshot=session.publisher.last_snapshot))

            async def noop_close():
                pass
            loop = MultiNodeOnlineTaskLoop(session=session, checkpoint_executor=saver, build_actor_batch=build_batch, queue_capacity_bytes=512 * 1024 ** 2, persist_step=persist_step, reap_sources=lambda: rpc('speco_online_reap_sources', {}), close_owned=noop_close, abort_owned=lambda: abort_owned(owners), max_steps=config['steps'], checkpoint_interval=config.get('checkpoint_interval', 1))
            completed = await loop.run(groups)
            versions = session.adapter.versions
            binding = completed['final_checkpoint_binding']
        else:
            for iteration in range(config['steps']):
                step = start_step + iteration

                def request_for(index):
                    request_id = f'{run_id}-step-{step}-response-{index}'
                    sample_id = f'step-{step}-response-{index}'

                    def metadata(bound, prompt, response):
                        rows = len(prompt) + len(response) - 1
                        return SampleMetadataV3(3, run_id, generation, 'p0', 'producer-' + run_id, request_id, sample_id, step * 32 + index, str(bound.target_revision), bound.data_version, manifest['tokenizer_fingerprint'], manifest['target_model_fingerprint'], tuple(manifest['model_config']['target_layer_ids']), 'dflash_aux_plus_last', 'bfloat16', rows, rows * (15360 * 2 + 20), rl_step=bound.rl_step, draft_revision=str(bound.draft_revision), prompt_token_count=len(prompt), supervised_token_count=len(response) - 1).to_dict()

                    def stream():
                        return engine.generate(dict(prompt_token_ids=tokens), SamplingParams(temperature=0.7, top_p=0.95, seed=731 + step * 32 + index, max_tokens=response_tokens, min_tokens=response_tokens, ignore_eos=True), request_id=request_id)
                    return AsyncCollectionRequest(request_id, tuple(tokens), stream, metadata, dict(partition=partition, ray_address=config['ray_address'], capacity_bytes=512 * 1024 ** 2, control_actor_name=control_name))
                commit = await collection.collect_group(collection_id='collection-' + str(step), requests=tuple((request_for(i) for i in range(32))))
                saved = [read_published_json(root / 'sequences' / f'step-{step}-response-{i}.json') for i in range(32)]
                rows = []
                for pair in range(16):
                    responses = [tuple(saved[pair * 2 + i]['response_token_ids']) for i in (0, 1)]
                    hashes = [hashlib.sha256(json.dumps(value).encode()).hexdigest() for value in responses]
                    for index in (0, 1):
                        sample = pair * 2 + index
                        rows.append(ActorSample(f'step-{step}-response-{sample}', f'step-{step}-group-{pair}', tuple(tokens), responses[index], float(hashes[index] == max(hashes)), commit.versions.target_revision, 0.7))
                batch = ActorCollectionBatch(commit.collection_id, commit.tq_keys, tuple(rows))
                print('FULL_ASYNC_REAL_OLD_POLICY_COLLECTION_COMPLETE', step + 1, len(rows), flush=True)
                result = await session.update(commit, batch)
                versions = result['versions']
                assert versions.target_revision == step + 1 and versions.draft_revision == initial['draft_revision'] + iteration + 1
                await rpc('speco_online_reap_sources', {})
                receipt = dict(step=step + 1, versions=asdict(versions), actor=result['actor'], draft=asdict(result['draft']), target_weight_manifest_sha256=publisher.last_proof['source'][0]['wire_receipt']['actor_weight_manifest_sha256'], complete_response_count=32, functional_reward_not_task_quality=True)
                observations.append(receipt)
                publish_json(root / 'steps' / ('step-' + str(step + 1) + '.json'), receipt)
                print('FULL_ASYNC_ACTOR_TARGET_DRAFT_SESSION_STEP_PASS', step + 1, flush=True)
                if iteration == config['steps'] - 1:
                    binding = await saver.save(result)
                    publish_json(root / 'joint-binding.json', dict(binding=binding, fingerprints=joint['joint_fingerprints'], versions=asdict(versions), snapshot=session.publisher.last_snapshot))
                    print('FULL_92B_JOINT_SAFE_POINT_COMPLETE', versions.rl_step, flush=True)
        final = None
        async for item in engine.generate(dict(prompt_token_ids=tokens), SamplingParams(temperature=0.7, top_p=0.95, max_tokens=response_tokens, min_tokens=response_tokens, ignore_eos=True, seed=900), request_id=run_id + '-final-published-versions'):
            if item.finished:
                final = item
        assert final is not None and len(final.outputs[0].token_ids) == response_tokens
        publish_json(root / 'final-generation.json', dict(versions=asdict(versions), response_token_ids=list(final.outputs[0].token_ids), complete=True))
        if config.get('rollout_graph_mode') == 'FULL_DECODE_ONLY':
            from verl_speco.integration.accelerator_speculation import validate_updated_graph_status
            graph_status = await rpc('speco_dspark_runtime_status', {})
            graph_proof = validate_updated_graph_status(graph_status, owners=rollout_owners, target_revision=versions.target_revision, draft_revision=versions.draft_revision)
            publish_json(root / 'final-updated-FULL-graph-status.json', dict(versions=asdict(versions), workers=graph_status, **graph_proof))
        await rpc('speco_close_online_publisher', {})
        from verl_speco.integration.multinode_publication import require_owned_rows
        actor_closed = await asyncio.to_thread(actor.speco_close_checkpoint_channel, channel_identity)
        require_owned_rows(actor_closed, actor_owners, phase='closed')
        native_closed = await rpc('speco_close_checkpoint_channel', dict(identity=channel_identity))
        require_owned_rows(native_closed, rollout_owners, phase='closed')
        publish_json(root / 'checkpoint-channel-closed.json', dict(actor=actor_closed, native=native_closed))
        proof = dict(run_id=run_id, generation=generation, steps=observations, actor_nodes=2, actor_die=32, teacher_TP8=True, draft_DP=len(draft_owners), draft_TP_PP_SP_EP=1, shared_actor_draft_die=32 if shared is not None else 0, actual_VeRL_native_async_colocated_server=shared is not None, independent_draft_DP2=shared is None, real_full_92B_actor_target_HCCL_async_SpeCo_session=True, final_updated_target_and_draft_generation_complete=True, checkpoint_channel_preconnected_before_PPO=True, checkpoint_channel_closed=True, joint_checkpoint_binding=binding, joint_fresh_restore=bool(joint.get('joint_restore_binding')), actual_prompt_input=bool(config.get('prompt_input')), online_task_loop_used=bool(config.get('prompt_input')), production_entry_numeric_serving_parity_graphs_SLA_task_quality_not_claimed=True)
        publish_json(root / 'acceptance.json', proof)
        print('FULL_MULTINODE_ASYNC_92B_SPECO_SESSION_PASS', flush=True)
        return proof
    finally:
        if shared is not None:
            await shared.close()
        elif engine is not None:
            engine.shutdown()
        for handle in handles:
            ray.kill(handle, no_restart=True)
        if control is not None:
            ray.kill(control, no_restart=True)
        if owns_ray_client:
            ray.shutdown()
