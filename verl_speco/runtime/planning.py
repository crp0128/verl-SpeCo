# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from importlib import metadata, util
from typing import Any, Mapping

@dataclass(frozen=True)
class DependencyCapability:
    distribution: str
    import_name: str
    installed: bool
    version: str | None
    expected_version: str | None = None
    version_matches: bool | None = None

@dataclass(frozen=True)
class RuntimeFeatureCapability:
    name: str
    status: str
    reason: str

@dataclass(frozen=True)
class CapabilityReport:
    schema_version: int
    dependencies: tuple[DependencyCapability, ...]
    required_missing: tuple[str, ...]
    notes: tuple[str, ...]
    features: tuple[RuntimeFeatureCapability, ...]

    @property
    def ready_for_probe(self) -> bool:
        return not self.required_missing

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': self.schema_version, 'dependencies': [asdict(item) for item in self.dependencies], 'required_missing': list(self.required_missing), 'notes': list(self.notes), 'features': [asdict(item) for item in self.features], 'ready_for_probe': self.ready_for_probe}

def probe_runtime_capabilities(*, required: Mapping[str, str] | None=None, expected_versions: Mapping[str, str] | None=None) -> CapabilityReport:
    """Report importability and installed versions without importing heavy stacks.

    The returned report deliberately does not claim that hidden capture,
    VeOmni model registration, or TQ transport works. Those require their own
    real-process probes after this lightweight preflight succeeds.
    """
    required = required or {'verl': 'verl', 'veomni': 'veomni', 'vllm': 'vllm', 'vllm-ascend': 'vllm_ascend', 'ray': 'ray', 'torch_npu': 'torch_npu', 'transformers': 'transformers', 'TransferQueue': 'transfer_queue'}
    dependencies: list[DependencyCapability] = []
    missing: list[str] = []
    expected_versions = expected_versions or {}
    for distribution, import_name in required.items():
        installed = util.find_spec(import_name) is not None
        version: str | None = None
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pass
        expected = expected_versions.get(distribution)
        matches = None if expected is None else version == expected
        if not installed or version is None or matches is False:
            missing.append(distribution)
        dependencies.append(DependencyCapability(distribution=distribution, import_name=import_name, installed=installed, version=version, expected_version=expected, version_matches=matches))
    notes = ('This report does not import accelerator runtimes or create external resources.', 'Dependency availability is not hardware or end-to-end validation of this checkout.')
    local_status = 'fail_closed' if missing else 'requires_validation'
    local_reason = 'required dependency missing or version-mismatched' if missing else 'implementation migrated; run model, engine and transport probes on the selected cluster'
    features = (
        RuntimeFeatureCapability('launch_plan_dry_run', 'available', 'pure configuration resolution without accelerator or service startup'),
        *(RuntimeFeatureCapability(name, local_status, local_reason) for name in (
            'native_dspark_training', 'external_ray_tq_attach', 'trainer_packing',
            'multi_node_standalone_training', 'online_rl_training', 'draft_weight_publication')),
        RuntimeFeatureCapability('control_safe_point_restore', 'limited', 'metadata-only restore after payload drain and committed/cancelled leases'),
        RuntimeFeatureCapability('physical_producer_packing', 'requires_validation', 'publish requires the target model numerical capture gate'),
        RuntimeFeatureCapability('active_transaction_recovery', 'fail_closed', 'uncertain optimizer windows require a new generation; live rollback is not implemented'),
    )
    return CapabilityReport(schema_version=1, dependencies=tuple(dependencies), required_missing=tuple(missing), notes=notes, features=features)
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

class NodeRole(str, Enum):
    """One process placement role in a SpeCo run."""
    ROLLOUT = 'rollout'
    DRAFT = 'draft'
    CONTROL = 'control'
    RL_LEARNER = 'rl_learner'

@dataclass(frozen=True)
class RankTableNode:
    """A stable node identity read from the user-provided rank table."""
    node_id: str
    address: str
    device_count: int

@dataclass(frozen=True)
class ProcessPlacement:
    """The deterministic location of one process or service role."""
    role: NodeRole
    node_id: str
    node_rank: int
    local_rank: int
    global_rank: int | None
    device_id: int | None
    resource_label: str
    producer_group: str | None = None

@dataclass(frozen=True)
class RoleEndpoint:
    """An owned, deterministic listen address; resolving it opens no socket."""
    key: str
    role: NodeRole
    node_rank: int
    address: str
    port: int
    protocol: str
    global_rank: int | None = None
    local_rank: int | None = None
    producer_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), 'role': self.role.value}

@dataclass(frozen=True)
class ResolvedLaunchPlan:
    """Canonical, side-effect-free execution plan shared by all run nodes."""
    schema_version: int
    run_id: str
    generation: int
    mode: str
    nodes: tuple[RankTableNode, ...]
    placements: tuple[ProcessPlacement, ...]
    producer_groups: tuple[tuple[str, ...], ...]
    tq_partitions: tuple[str, ...]
    ready_router_shards: tuple[str, ...]
    lease_registry_shards: tuple[str, ...]
    master_addr: str
    master_port: int
    draft_world_size: int
    topology_fingerprint: str
    config_fingerprint: str
    role_endpoints: tuple[RoleEndpoint, ...] = ()
    endpoint_fingerprint: str = ''

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with deterministic ordering."""
        result = {'schema_version': self.schema_version, 'run_id': self.run_id, 'generation': self.generation, 'mode': self.mode, 'nodes': [asdict(node) for node in self.nodes], 'placements': [{**asdict(placement), 'role': placement.role.value} for placement in self.placements], 'producer_groups': [list(group) for group in self.producer_groups], 'tq_partitions': list(self.tq_partitions), 'ready_router_shards': list(self.ready_router_shards), 'lease_registry_shards': list(self.lease_registry_shards), 'master_addr': self.master_addr, 'master_port': self.master_port, 'draft_world_size': self.draft_world_size, 'topology_fingerprint': self.topology_fingerprint, 'config_fingerprint': self.config_fingerprint}
        if self.role_endpoints:
            result['role_endpoints'] = [item.to_dict() for item in self.role_endpoints]
            result['endpoint_fingerprint'] = self.endpoint_fingerprint
        return result

    def for_node(self, node_rank: int) -> dict[str, Any]:
        if type(node_rank) is not int or not 0 <= node_rank < len(self.nodes):
            raise ValueError('node_rank must select an existing physical node')
        rendered = self.to_dict()
        return dict(run_id=self.run_id, generation=self.generation, node=rendered['nodes'][node_rank], node_rank=node_rank, placements=[item for item in rendered['placements'] if item['node_rank'] == node_rank], role_endpoints=[item.to_dict() for item in self.role_endpoints if item.node_rank == node_rank], tq_partitions=list(self.tq_partitions), ready_router_shards=list(self.ready_router_shards), lease_registry_shards=list(self.lease_registry_shards), draft_world_size=self.draft_world_size, draft_rendezvous=dict(address=self.master_addr, port=self.master_port), topology_fingerprint=self.topology_fingerprint, endpoint_fingerprint=self.endpoint_fingerprint)

def resolve_launch_plan(config: Mapping[str, Any], rank_table: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None=None) -> ResolvedLaunchPlan:
    """Resolve a launch plan using only configuration data.

    ``rank_table`` accepts either ``{"nodes": [...]}`` or a node list.  Each
    node must provide a stable ``node_id`` and an ``address``.  Omitting the
    table is supported only for deterministic local dry-runs and synthesizes
    ``node-0 .. node-N`` identities from the configured role counts.
    """
    runtime = launch_plan_runtime_config(config)
    topology = launch_plan_mapping(runtime.get('topology'), 'speco.multinode_runtime.topology')
    queue = launch_plan_mapping(runtime.get('transfer_queue'), 'speco.multinode_runtime.transfer_queue')
    ray = launch_plan_mapping(runtime.get('ray'), 'speco.multinode_runtime.ray')
    run_id = launch_plan_nonempty_string(runtime.get('run_id'), 'speco.multinode_runtime.run_id')
    generation = launch_plan_nonnegative_int(runtime.get('generation', 0), 'generation')
    mode = str(runtime.get('mode', 'standalone'))
    if mode not in {'standalone', 'online_rl'}:
        raise ValueError('speco.multinode_runtime.mode must be standalone or online_rl')
    rollout_nodes = launch_plan_positive_int(topology.get('rollout_nodes', 1), 'rollout_nodes')
    draft_nodes = launch_plan_positive_int(topology.get('draft_nodes', 1), 'draft_nodes')
    draft_local_world_size = launch_plan_positive_int(topology.get('nproc_per_draft_node', 1), 'nproc_per_draft_node')
    target_tp = launch_plan_positive_int(topology.get('target_tp', 1), 'target_tp')
    target_dp = launch_plan_positive_int(topology.get('target_dp', 1), 'target_dp')
    replicas_per_node = topology.get('rollout_replicas_per_node', 1)
    if type(replicas_per_node) is not int or replicas_per_node < 1:
        raise ValueError('rollout_replicas_per_node must be a positive integer')
    rollout_devices = target_tp * replicas_per_node
    configured_draft_start = topology.get('draft_device_start')
    if configured_draft_start is not None:
        configured_draft_start = launch_plan_nonnegative_int(configured_draft_start, 'draft_device_start')
    launch_plan_require_first_release_parallelism(topology)
    explicit_indices = [topology.get(name) for name in ('rollout_node_indices', 'draft_node_indices')]
    minimum_nodes = max(rollout_nodes, draft_nodes)
    for indices in explicit_indices:
        if indices is not None:
            if not isinstance(indices, (list, tuple)) or not indices or any((type(index) is not int or index < 0 for index in indices)):
                raise ValueError('explicit role node indices contain an invalid node index; require nonempty nonnegative integers')
            if rank_table is None:
                minimum_nodes = max(minimum_nodes, max(indices) + 1)
    explicit_disjoint = all((indices is not None for indices in explicit_indices)) and (not set(explicit_indices[0]) & set(explicit_indices[1]))
    default_draft_start = 0 if explicit_disjoint else rollout_devices
    nodes = launch_plan_parse_nodes(rank_table, minimum_nodes=minimum_nodes, default_device_count=max(rollout_devices, (configured_draft_start if configured_draft_start is not None else default_draft_start) + draft_local_world_size))
    if len(nodes) < max(rollout_nodes, draft_nodes):
        raise ValueError(f'rank table has fewer nodes than rollout_nodes/draft_nodes: nodes={len(nodes)} rollout_nodes={rollout_nodes} draft_nodes={draft_nodes}')
    rollout_indices = launch_plan_role_node_indices(topology.get('rollout_node_indices'), count=rollout_nodes, node_count=len(nodes), name='rollout_node_indices')
    draft_indices = launch_plan_role_node_indices(topology.get('draft_node_indices'), count=draft_nodes, node_count=len(nodes), name='draft_node_indices')
    if mode == 'standalone' and len(nodes) > 1 and set(rollout_indices) & set(draft_indices):
        raise ValueError('multi-node standalone requires disjoint rollout and draft nodes')
    for node_rank, node in enumerate(nodes):
        if node_rank in rollout_indices and node.device_count < rollout_devices:
            raise ValueError(f'rollout node {node.node_id!r} cannot fit its complete TP replicas')
        if node_rank in draft_indices:
            draft_start = configured_draft_start if configured_draft_start is not None else rollout_devices if node_rank in rollout_indices else 0
            if draft_start + draft_local_world_size > node.device_count:
                raise ValueError(f'draft placement exceeds device_count on {node.node_id!r}: start={draft_start}, ranks={draft_local_world_size}, devices={node.device_count}')
            if node_rank in rollout_indices and draft_start < rollout_devices:
                raise ValueError(f'draft devices overlap target TP devices on {node.node_id!r}')
    partition = launch_plan_nonempty_string(queue.get('partition', 'speco_drafter_features'), 'partition')
    ready_shards = launch_plan_positive_int(queue.get('ready_shards', 1), 'ready_shards')
    master_addr = str(topology.get('master_addr') or nodes[draft_indices[0]].address)
    master_port = launch_plan_tcp_port(topology.get('master_port', 29500), 'master_port')
    placements: list[ProcessPlacement] = []
    rollout_ids: list[str] = []
    for replica_rank, (node_rank, local_replica) in enumerate(((node_rank, local_replica) for node_rank in rollout_indices for local_replica in range(replicas_per_node))):
        node = nodes[node_rank]
        replica_id = f'rollout-{replica_rank:03d}'
        rollout_ids.append(replica_id)
        placements.append(ProcessPlacement(role=NodeRole.ROLLOUT, node_id=node.node_id, node_rank=node_rank, local_rank=local_replica * target_tp, global_rank=None, device_id=local_replica * target_tp, resource_label=str(ray.get('rollout_resource', 'speco_rollout')), producer_group=f'producer-group-{replica_rank % target_dp:03d}'))
    draft_global_rank = 0
    for node_rank in draft_indices:
        node = nodes[node_rank]
        draft_start = configured_draft_start if configured_draft_start is not None else rollout_devices if node_rank in rollout_indices else 0
        for local_rank in range(draft_local_world_size):
            placements.append(ProcessPlacement(role=NodeRole.DRAFT, node_id=node.node_id, node_rank=node_rank, local_rank=local_rank, global_rank=draft_global_rank, device_id=draft_start + local_rank, resource_label=str(ray.get('draft_resource', 'speco_draft'))))
            draft_global_rank += 1
    placements.append(ProcessPlacement(role=NodeRole.CONTROL, node_id=nodes[0].node_id, node_rank=0, local_rank=0, global_rank=None, device_id=None, resource_label=str(ray.get('control_resource', 'speco_control'))))
    if mode == 'online_rl':
        placements.append(ProcessPlacement(role=NodeRole.RL_LEARNER, node_id=nodes[0].node_id, node_rank=0, local_rank=0, global_rank=None, device_id=None, resource_label=str(ray.get('rl_learner_resource', 'speco_rl_learner'))))
    producer_groups = tuple((tuple((replica_id for index, replica_id in enumerate(rollout_ids) if index % target_dp == group_index)) for group_index in range(target_dp)))
    if any((not group for group in producer_groups)):
        raise ValueError(f'target_dp cannot exceed the number of rollout replicas: target_dp={target_dp} replicas={len(rollout_ids)}')
    topology_payload = {'nodes': [asdict(node) for node in nodes], 'placements': [{**asdict(item), 'role': item.role.value} for item in placements], 'producer_groups': producer_groups, 'target_tp': target_tp, 'target_dp': target_dp, 'master_addr': master_addr, 'master_port': master_port}
    config_payload = dict(runtime)
    config_payload['topology'] = {key: value for key, value in topology.items() if key not in {'node_rank', 'rank_table_path'}}
    endpoints = launch_plan_resolve_role_endpoints(runtime, nodes, placements, master_addr, master_port, target_tp)
    plan = ResolvedLaunchPlan(schema_version=1, run_id=run_id, generation=generation, mode=mode, nodes=tuple(nodes), placements=tuple(placements), producer_groups=producer_groups, tq_partitions=tuple((f'{partition}:g{generation}:s{index}' for index in range(ready_shards))), ready_router_shards=tuple((f'ready-router:g{generation}:s{index}' for index in range(ready_shards))), lease_registry_shards=tuple((f'lease-registry:g{generation}:s{index}' for index in range(ready_shards))), master_addr=master_addr, master_port=master_port, draft_world_size=draft_global_rank, topology_fingerprint=launch_plan_fingerprint(topology_payload), config_fingerprint=launch_plan_fingerprint(config_payload), role_endpoints=endpoints, endpoint_fingerprint=launch_plan_fingerprint({'run_id': run_id, 'generation': generation, 'endpoints': [item.to_dict() for item in endpoints]}) if endpoints else '')
    validate_launch_plan(plan)
    return plan

def validate_launch_plan(plan: ResolvedLaunchPlan) -> None:
    """Validate invariants that must hold identically on every node."""
    if not plan.run_id:
        raise ValueError('launch plan run_id must not be empty')
    if plan.generation < 0:
        raise ValueError('launch plan generation must be non-negative')
    if plan.draft_world_size <= 0:
        raise ValueError('launch plan needs at least one draft rank')
    draft = [item for item in plan.placements if item.role is NodeRole.DRAFT]
    observed_ranks = sorted((item.global_rank for item in draft))
    if observed_ranks != list(range(plan.draft_world_size)):
        raise ValueError('draft global ranks must be exactly contiguous from zero')
    if len(plan.tq_partitions) != len(plan.ready_router_shards):
        raise ValueError('TQ and ReadyRouter shard counts must match')
    if len(plan.tq_partitions) != len(plan.lease_registry_shards):
        raise ValueError('TQ and LeaseRegistry shard counts must match')
    if not plan.topology_fingerprint or not plan.config_fingerprint:
        raise ValueError('launch plan fingerprints must not be empty')
    if plan.role_endpoints:
        expected = launch_plan_fingerprint({'run_id': plan.run_id, 'generation': plan.generation, 'endpoints': [item.to_dict() for item in plan.role_endpoints]})
        if plan.endpoint_fingerprint != expected:
            raise ValueError('launch plan endpoint fingerprint differs')
        launch_plan_validate_endpoints(plan.role_endpoints)
        if any((item.node_rank >= len(plan.nodes) or item.address != plan.nodes[item.node_rank].address for item in plan.role_endpoints)):
            raise ValueError('role endpoint does not belong to its physical rank-table node')

def launch_plan_resolve_role_endpoints(runtime, nodes, placements, master_addr, master_port, target_tp):
    options = runtime.get('role_endpoints')
    if options is None:
        return ()
    options = launch_plan_mapping(options, 'role_endpoints')
    required = {'metadata_rpc_port', 'metadata_http_port', 'source_port_base', 'source_port_stride', 'draft_port_base'}
    optional = {'source_owners_per_replica', 'actor_master_port', 'draft_extra_reader_count', 'external_ray_port', 'draft_ports_per_node'}
    if set(options) - required - optional or not required <= set(options):
        raise ValueError('role_endpoints needs the complete known service port configuration')
    owners = options.get('source_owners_per_replica', 1)
    stride = options['source_port_stride']
    if type(owners) is not int or not 1 <= owners <= target_tp or type(stride) is not int or (stride < owners):
        raise ValueError('source owner count and port stride cannot overlap TP source listeners')
    control = next((item for item in placements if item.role is NodeRole.CONTROL))
    first_draft = next((item for item in placements if item.role is NodeRole.DRAFT))
    if master_addr != nodes[first_draft.node_rank].address:
        raise ValueError('draft rendezvous must belong to the actual first trainer node')
    endpoints = []

    def add(key, role, node_rank, port, protocol, **identity):
        endpoints.append(RoleEndpoint(key, role, node_rank, nodes[node_rank].address, launch_plan_tcp_port(port, key), protocol, **identity))
    add('metadata_rpc', NodeRole.CONTROL, control.node_rank, options['metadata_rpc_port'], 'mooncake_rpc')
    add('metadata_http', NodeRole.CONTROL, control.node_rank, options['metadata_http_port'], 'http')
    add('draft_master', NodeRole.DRAFT, first_draft.node_rank, master_port, 'torch_distributed')
    source_base = launch_plan_tcp_port(options['source_port_base'], 'source_port_base')
    draft_base = launch_plan_tcp_port(options['draft_port_base'], 'draft_port_base')
    for replica, item in enumerate((p for p in placements if p.role is NodeRole.ROLLOUT)):
        for owner in range(owners):
            add(f'source:{replica}:{owner}', NodeRole.ROLLOUT, item.node_rank, source_base + replica * stride + owner, 'mooncake_rpc', global_rank=replica * owners + owner, local_rank=item.local_rank + owner, producer_group=item.producer_group)
    local_ports = options.get('draft_ports_per_node', False)
    if type(local_ports) is not bool:
        raise ValueError('draft ports per node must be boolean')
    draft_placements = [p for p in placements if p.role is NodeRole.DRAFT]
    node_owners = [p for p in draft_placements if p.local_rank == 0]
    if local_ports and (len(node_owners) != 2 or len(draft_placements) != 32 or any((len([p for p in draft_placements if p.node_rank == owner.node_rank]) != 16 for owner in node_owners))):
        raise ValueError('node-local native listeners require two independent local16 DP trainers')
    for item in draft_placements:
        add(f'draft:{item.global_rank}', NodeRole.DRAFT, item.node_rank, draft_base + (item.local_rank if local_ports else item.global_rank), 'mooncake_rpc', global_rank=item.global_rank, local_rank=item.local_rank)
    readers = options.get('draft_extra_reader_count', 1)
    if type(readers) is not int or readers not in (1, 2, 4, 8):
        raise ValueError('independent draft reader count must be 1, 2, 4 or 8')
    if readers == 8 and (local_ports or len(draft_placements) != 16):
        raise ValueError('eight native reader endpoints require one DP16 trainer')
    for owner in node_owners if local_ports else [first_draft]:
        for lane in range(1, readers):
            key = f'draft_reader:{owner.global_rank}:{lane}' if local_ports else f'draft_reader:{lane}'
            add(key, NodeRole.DRAFT, owner.node_rank, draft_base + 15 + lane, 'mooncake_rpc', global_rank=owner.global_rank, local_rank=0)
    if 'external_ray_port' in options:
        add('external_ray', NodeRole.CONTROL, control.node_rank, options['external_ray_port'], 'ray_gcs')
    learners = [p for p in placements if p.role is NodeRole.RL_LEARNER]
    if bool(learners) != ('actor_master_port' in options):
        raise ValueError('actor rendezvous is required exactly for online_rl')
    if learners:
        add('actor_master', NodeRole.RL_LEARNER, learners[0].node_rank, options['actor_master_port'], 'torch_distributed')
    result = tuple(endpoints)
    launch_plan_validate_endpoints(result)
    return result

def launch_plan_validate_endpoints(endpoints):
    for item in endpoints:
        launch_plan_tcp_port(item.port, item.key)
        if not isinstance(item.role, NodeRole) or not item.key or type(item.node_rank) is not int or (item.node_rank < 0) or (item.protocol not in {'mooncake_rpc', 'http', 'torch_distributed', 'ray_gcs'}):
            raise ValueError('role endpoint has an invalid identity or protocol')
    if len({item.key for item in endpoints}) != len(endpoints):
        raise ValueError('role endpoint identities must be unique')
    sockets = [(item.address, item.port) for item in endpoints]
    if len(set(sockets)) != len(sockets):
        raise ValueError('role listen endpoints overlap on the same physical host')

def launch_plan_tcp_port(value, name):
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError(f'{name} must be an integer TCP port in 1..65535')
    return value

def format_launch_plan(plan: ResolvedLaunchPlan) -> str:
    """Format the canonical plan for dry-run logs without side effects."""
    return json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

def launch_plan_runtime_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    speco = launch_plan_mapping(config.get('speco'), 'speco')
    if sum(key in speco for key in ('multinode_runtime', 'speco_runtime')) > 1:
        raise ValueError('choose one runtime configuration, not both aliases')
    return launch_plan_mapping(speco.get('multinode_runtime', speco.get('speco_runtime')), 'speco.multinode_runtime')

def launch_plan_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value

def launch_plan_parse_nodes(rank_table: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None, *, minimum_nodes: int, default_device_count: int) -> list[RankTableNode]:
    if rank_table is None:
        return [RankTableNode(node_id=f'node-{index}', address=f'127.0.0.{index + 1}', device_count=default_device_count) for index in range(minimum_nodes)]
    raw_nodes: Any = rank_table.get('nodes') if isinstance(rank_table, Mapping) else rank_table
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
        raise ValueError('rank table must contain a nodes list')
    nodes: list[RankTableNode] = []
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise ValueError(f'rank table node {index} must be a mapping')
        node_id = launch_plan_nonempty_string(raw.get('node_id', raw.get('id')), f'nodes[{index}].node_id')
        address = launch_plan_nonempty_string(raw.get('address', raw.get('host')), f'nodes[{index}].address')
        device_count = launch_plan_positive_int(raw.get('device_count', raw.get('nproc_per_node', 1)), f'nodes[{index}].device_count')
        nodes.append(RankTableNode(node_id=node_id, address=address, device_count=device_count))
    if len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError('rank table node_id values must be unique')
    if len({node.address for node in nodes}) != len(nodes):
        raise ValueError('physical rank-table addresses must be unique; use explicit local TP replicas')
    return nodes

def launch_plan_require_first_release_parallelism(topology: Mapping[str, Any]) -> None:
    unsupported = {'draft_tp': topology.get('draft_tp', 1), 'draft_pp': topology.get('draft_pp', 1), 'draft_sp': topology.get('draft_sp', 1), 'draft_ep': topology.get('draft_ep', 1)}
    enabled = {name: value for name, value in unsupported.items() if int(value) != 1}
    if enabled:
        raise ValueError(f'first-release draft parallelism supports DP/FSDP2 only; unsupported settings: {enabled}')

def launch_plan_role_node_indices(value: Any, *, count: int, node_count: int, name: str) -> tuple[int, ...]:
    """Resolve explicit role placement without silently duplicating a node."""
    if value is None:
        return tuple(range(count))
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f'{name} must contain exactly {count} node indices')
    if any((type(item) is not int or item < 0 or item >= node_count for item in value)):
        raise ValueError(f'{name} contains an invalid node index')
    if len(set(value)) != count:
        raise ValueError(f'{name} contains a duplicate node index')
    return tuple(value)

def launch_plan_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if parsed <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return parsed

def launch_plan_nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a non-negative integer') from exc
    if parsed < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return parsed

def launch_plan_nonempty_string(value: Any, name: str) -> str:
    result = str(value or '').strip()
    if not result:
        raise ValueError(f'{name} must not be empty')
    return result

def launch_plan_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class OwnedResource:
    kind: str
    identity: str

@dataclass(frozen=True)
class CleanupResult:
    resource: OwnedResource
    cleaned: bool
    error: str | None = None

class ResourceOwner:
    """Record cleanup callbacks without ever discovering global resources.

    External Ray clusters, TQ owners, and vLLM services are dependencies and
    must not be registered here unless this run created them itself.
    """

    def __init__(self) -> None:
        self._callbacks: list[tuple[OwnedResource, Callable[[], None]]] = []
        self._identities: set[tuple[str, str]] = set()

    def register(self, kind: str, identity: str, cleanup: Callable[[], None]) -> None:
        key = (str(kind), str(identity))
        if not key[0] or not key[1]:
            raise ValueError('owned resource kind and identity must not be empty')
        if key in self._identities:
            raise ValueError(f'resource already registered: {key[0]}:{key[1]}')
        self._identities.add(key)
        self._callbacks.append((OwnedResource(*key), cleanup))

    def resources(self) -> tuple[OwnedResource, ...]:
        return tuple((resource for resource, _ in self._callbacks))

    def cleanup_owned(self) -> tuple[CleanupResult, ...]:
        """Run callbacks in reverse creation order and report every outcome."""
        results: list[CleanupResult] = []
        for resource, callback in reversed(self._callbacks):
            try:
                callback()
            except Exception as exc:
                results.append(CleanupResult(resource=resource, cleaned=False, error=str(exc)))
            else:
                results.append(CleanupResult(resource=resource, cleaned=True))
        self._callbacks.clear()
        self._identities.clear()
        return tuple(results)
