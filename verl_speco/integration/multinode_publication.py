# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from dataclasses import asdict

def checkpoint_channel_identity(request):
    if not isinstance(request, dict) or set(request) != {'run_id', 'generation', 'channel_id', 'actors', 'rollout', 'bucket_bytes'}:
        raise ValueError('checkpoint channel requires its complete owned identity')
    for key in ('run_id', 'channel_id'):
        if not isinstance(request[key], str) or not request[key]:
            raise ValueError('checkpoint channel requires nonempty identity fields')
    if type(request['generation']) is not int or request['generation'] < 1:
        raise ValueError('checkpoint channel generation must be a positive integer')
    if type(request['bucket_bytes']) is not int or request['bucket_bytes'] < 1:
        raise ValueError('checkpoint channel bucket must be positive')
    for role in ('actors', 'rollout'):
        owners = request[role]
        if not isinstance(owners, dict) or not owners or any((type(rank) is not int for rank in owners)) or (set(owners) != set(range(len(owners)))) or any((not isinstance(owner, str) or not owner for owner in owners.values())) or (len(set(owners.values())) != len(owners)):
            raise ValueError('checkpoint channel requires exact unique role memberships')
    if len(request['rollout']) > 126:
        raise ValueError('checkpoint channel exceeds the bounded initialization SUM membership')
    return {**request, 'actors': dict(request['actors']), 'rollout': dict(request['rollout'])}

class MultiNodeCheckpointChannel:
    """One HCCL communicator per generation, independent of weight revisions."""

    def __init__(self, *, identity, role, rank, incarnation, engine_factory=None):
        self.identity = checkpoint_channel_identity(identity)
        if role not in ('actors', 'rollout') or type(rank) is not int:
            raise ValueError('checkpoint channel has a foreign role or rank')
        if self.identity[role].get(rank) != incarnation:
            raise ValueError('checkpoint channel has a foreign worker incarnation')
        self.role, self.rank, self.incarnation = (role, rank, incarnation)
        self.wire_rank = (0 if rank == 0 else -1) if role == 'actors' else rank + 1
        self.world_size = len(self.identity['rollout']) + 1
        self.phase = 'new'
        self.engine = None
        self.master_metadata = None
        self.operation_id = None
        self.completed_operations = set()
        self.factory = engine_factory

    def _check(self, identity):
        if checkpoint_channel_identity(identity) != self.identity:
            raise ValueError('checkpoint channel identity changed within its generation')

    def prepare(self, identity):
        self._check(identity)
        if self.phase != 'new':
            raise RuntimeError('checkpoint channel preparation cannot be replayed')
        self.phase = 'preparing'
        try:
            factory = self.factory
            if factory is None:
                from verl.checkpoint_engine.hccl_checkpoint_engine import HCCLCheckpointEngine
                from verl_speco.integration.multinode_online import install_hccl_checkpoint_compat
                install_hccl_checkpoint_compat()
                factory = HCCLCheckpointEngine
            self.engine = factory(bucket_size=self.identity['bucket_bytes'], group_name=self.identity['channel_id'], rebuild_group=False, is_master=self.wire_rank == 0)
            if self.wire_rank >= 0:
                metadata = self.engine.prepare()
                if self.wire_rank == 0:
                    self.master_metadata = asdict(metadata)
            self.phase = 'prepared'
            return self.master_metadata
        except BaseException:
            self.phase = 'failed'
            raise

    def connect(self, identity, master_metadata):
        self._check(identity)
        if self.phase != 'prepared':
            raise RuntimeError('checkpoint channel connect requires its fresh prepared owner')
        if self.wire_rank == 0 and master_metadata != self.master_metadata:
            raise ValueError('checkpoint channel master metadata belongs to another source')
        self.phase = 'connecting'
        try:
            from verl.checkpoint_engine.hccl_checkpoint_engine import MasterMetadata
            metadata = MasterMetadata(**master_metadata)
            self.engine.init_process_group(rank=self.wire_rank, world_size=self.world_size, master_metadata=metadata)
            self.master_metadata = dict(master_metadata)
            self.engine.finalize()
            self.phase = 'ready'
        except BaseException:
            self.phase = 'failed'
            raise

    def begin(self, identity, operation_id):
        self._check(identity)
        if not isinstance(operation_id, str) or not operation_id or operation_id in self.completed_operations:
            raise ValueError('checkpoint channel operation must be fresh and explicit')
        if self.phase != 'ready':
            raise RuntimeError('checkpoint channel has an unfinished or uncertain transfer')
        self.phase = 'preparing_transfer'
        self.operation_id = operation_id
        try:
            if self.wire_rank >= 0:
                metadata = self.engine.prepare()
                if self.wire_rank == 0 and asdict(metadata) != self.master_metadata:
                    raise RuntimeError('checkpoint channel source metadata changed')
            self.phase = 'active'
            return self.engine
        except BaseException:
            self.phase = 'failed'
            raise

    def finish(self, identity, operation_id, *, whole_revision_acknowledged):
        self._check(identity)
        if self.phase != 'active' or operation_id != self.operation_id or whole_revision_acknowledged is not True:
            raise RuntimeError('checkpoint channel release lacks the exact whole revision ACK')
        try:
            self.engine.finalize()
            self.completed_operations.add(operation_id)
            self.operation_id = None
            self.phase = 'ready'
        except BaseException:
            self.phase = 'failed'
            raise

    def abort(self, identity, operation_id):
        self._check(identity)
        if self.phase != 'active' or self.operation_id != operation_id:
            raise RuntimeError('checkpoint channel abort lacks its exact active operation')
        self.phase = 'failed'

    def close(self, identity):
        self._check(identity)
        if self.phase not in ('ready', 'failed'):
            raise RuntimeError('checkpoint channel close requires an idle or failed safe point')
        self.phase = 'closing'
        try:
            if self.engine is not None:
                comm = self.engine.pyhccl
                if comm is not None:
                    comm.destroyComm(comm.comm)
                    self.engine.pyhccl = None
                self.engine.finalize()
                socket = getattr(self.engine, 'socket', None)
                if socket is not None:
                    socket.close(linger=0)
            self.phase = 'closed'
        except BaseException:
            self.phase = 'failed'
            raise
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from verl_speco.runtime.artifacts import read_published_json
from verl_speco.trainer.distributed_checkpoint import DraftParameterSnapshot, trainable_snapshot_checksum as _checksum

def draft_snapshot_update_effective_rope_theta(config):
    """Compare the same default RoPE across legacy and Transformers 5 config."""
    legacy = config.get('rope_theta')
    parameters = config.get('rope_parameters')
    if parameters is None:
        return legacy
    if not isinstance(parameters, dict) or parameters.get('rope_type', 'default') != 'default' or set(parameters) - {'rope_type', 'rope_theta'}:
        raise ValueError('draft update changes serving/training geometry: unsupported RoPE')
    theta = parameters.get('rope_theta', legacy)
    if legacy is not None and theta != legacy:
        raise ValueError('draft update changes serving/training geometry: conflicting RoPE')
    return theta

def read_draft_update(snapshot, *, target_fingerprint, tokenizer_fingerprint, model_config):
    if not isinstance(snapshot, DraftParameterSnapshot):
        snapshot = DraftParameterSnapshot(**snapshot)
    manifest = read_published_json(Path(snapshot.uri) / 'manifest.json')
    if _checksum(manifest) != snapshot.checksum or manifest.get('schema_version') != 1 or manifest.get('kind') != 'draft_trainable_parameters' or (manifest.get('draft_revision') != snapshot.draft_revision) or (manifest.get('target_model_fingerprint') != target_fingerprint) or (manifest.get('tokenizer_fingerprint') != tokenizer_fingerprint) or (manifest.get('weights_file') != 'trainable.safetensors'):
        raise ValueError('draft update manifest or immutable target/tokenizer binding differs')
    source_config = manifest['model_config']
    geometry = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size', 'target_hidden_size', 'target_layer_ids', 'block_size', 'num_anchors', 'markov_rank', 'markov_head_type', 'sliding_window', 'use_sliding_window', 'rms_norm_eps', 'mask_token_id')
    if any((source_config.get(key) != model_config.get(key) for key in geometry)) or draft_snapshot_update_effective_rope_theta(source_config) != draft_snapshot_update_effective_rope_theta(model_config):
        raise ValueError('draft update changes serving/training geometry')
    path = Path(snapshot.uri) / manifest['weights_file']
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['weights_sha256']:
            raise ValueError('draft update safetensors checksum differs')
    weights = load_file(str(path), device='cpu')
    bindings = manifest['parameters']
    if not weights or set(weights) != set(bindings):
        raise ValueError('draft update tensor set differs from manifest')
    digest = hashlib.sha256()
    for name in sorted(weights, key=lambda name: bindings[name]['source_name']):
        tensor = weights[name]
        binding = bindings[name]
        if not name.startswith('draft_model.') or binding != dict(source_name=manifest['namespace_prefix'] + name, shape=list(tensor.shape), dtype=str(tensor.dtype)) or tensor.dtype != torch.float32 or (not torch.isfinite(tensor).all()):
            raise ValueError('draft update requires finite full FP32 trainable tensors with exact namespace')
        digest.update(binding['source_name'].encode())
        digest.update(memoryview(tensor.contiguous().view(torch.uint8).numpy()))
    if digest.hexdigest() != manifest['trainable_parameter_sha256']:
        raise ValueError('draft update trainable parameter checksum differs')
    return (snapshot, manifest, weights)

def expected_native_shards(weights, *, config, rank, world_size):
    """Map source DSpark geometry onto dense unquantized DFlash TP parameters."""
    if type(rank) is not int or type(world_size) is not int or world_size < 1 or (not 0 <= rank < world_size) or (config['num_hidden_layers'] != 1) or config['num_attention_heads'] % world_size or config['intermediate_size'] % world_size or config['vocab_size'] % world_size:
        raise ValueError('unsupported native DSpark TP geometry')
    source = {name.removeprefix('draft_model.'): value for name, value in weights.items()}
    head_dim = config['head_dim']
    q_rows = config['num_attention_heads'] * head_dim
    kv_heads = config['num_key_value_heads']
    if kv_heads >= world_size:
        if kv_heads % world_size:
            raise ValueError('draft KV heads do not divide the TP mesh')
        kv_rank, kv_world = (rank, world_size)
    else:
        if world_size % kv_heads:
            raise ValueError('draft TP mesh cannot replicate KV heads')
        kv_rank, kv_world = (rank // (world_size // kv_heads), kv_heads)
    layer = 'layers.0.'
    split_names = {layer + 'self_attn.' + name + '_proj.weight' for name in ('q', 'k', 'v')}
    split_names |= {layer + 'mlp.' + name + '_proj.weight' for name in ('gate', 'up')}
    expected = {'model.' + name: value for name, value in source.items() if name not in split_names}

    def row_shard(name, shard_rank=rank, shard_world=world_size):
        tensor = source[name]
        if tensor.shape[0] % shard_world:
            raise ValueError('draft tensor rows cannot be sharded')
        return tensor.chunk(shard_world, dim=0)[shard_rank]
    expected['model.' + layer + 'self_attn.qkv_proj.weight'] = torch.cat([row_shard(layer + 'self_attn.q_proj.weight'), row_shard(layer + 'self_attn.k_proj.weight', kv_rank, kv_world), row_shard(layer + 'self_attn.v_proj.weight', kv_rank, kv_world)], dim=0)
    expected['model.' + layer + 'mlp.gate_up_proj.weight'] = torch.cat([row_shard(layer + 'mlp.gate_proj.weight'), row_shard(layer + 'mlp.up_proj.weight')], dim=0)
    expected['model.embed_tokens.weight'] = row_shard('embed_tokens.weight')
    for name, cols in ((layer + 'self_attn.o_proj.weight', q_rows), (layer + 'mlp.down_proj.weight', config['intermediate_size'])):
        if source[name].shape[1] != cols:
            raise ValueError('draft tensor column geometry differs')
        expected['model.' + name] = source[name].chunk(world_size, dim=1)[rank]
    return expected

def audit_native_draft(model, weights, *, config, rank, world_size):
    expected = expected_native_shards(weights, config=config, rank=rank, world_size=world_size)
    parameters = {name: param for name, param in model.named_parameters() if not name.startswith('lm_head.')}
    if set(parameters) != set(expected):
        raise ValueError('native draft trainable parameter set differs from complete snapshot')
    digest = hashlib.sha256()
    for name, source in sorted(expected.items()):
        actual = parameters[name].detach().cpu().contiguous()
        reference = source.to(dtype=actual.dtype).contiguous()
        if actual.shape != reference.shape or not torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)):
            raise RuntimeError('native draft parameter bytes differ after load: ' + name)
        if ('.markov_head.' in name or '.confidence_head.' in name) and actual.dtype != torch.float32:
            raise RuntimeError('native draft distribution heads lost FP32 precision')
        digest.update(json.dumps((name, str(actual.dtype), list(actual.shape))).encode())
        digest.update(memoryview(actual.view(torch.uint8).numpy()))
    attention = model.model.layers[0].self_attn
    fused = model.model._fused_kv_weight.detach().cpu().contiguous()
    reference = expected['model.layers.0.self_attn.qkv_proj.weight'][attention.q_size:].to(fused.dtype).contiguous()
    if not torch.equal(fused.view(torch.uint8), reference.view(torch.uint8)):
        raise RuntimeError('native draft fused context KV projection was not rebuilt')
    return dict(native_parameter_sha256=digest.hexdigest(), native_parameters=len(parameters), fused_context_KV_verified=True)
import os
from uuid import uuid4

def initialize_worker_publication(worker):
    from pathlib import Path
    binding = read_published_json(Path(os.environ['SPECO_DSPARK_MODEL_PATH']) / 'snapshot-binding.json')
    config = worker.model_runner.drafter.model.config
    if binding['snapshot']['checksum'] != config.speco_snapshot_checksum:
        raise RuntimeError('loaded DSpark checkpoint view differs from immutable snapshot binding')
    worker._speco_serving_incarnation = uuid4().hex
    worker._speco_draft_revision = binding['snapshot']['draft_revision']
    worker._speco_draft_update_phase = 'idle'
    worker._speco_pending_draft_update = None
    worker._speco_last_draft_ack = None

def prepare_worker_draft_update(worker, *, run_id, generation, operation_id, expected_revision, snapshot):
    probe = getattr(worker, '_speco_accepted_probe', None)
    if run_id != os.environ.get('SPECO_RUN_ID') or generation != int(os.environ.get('SPECO_GENERATION', '-1')) or (not operation_id) or (type(expected_revision) is not int) or (expected_revision != worker._speco_draft_revision) or (probe is not None and (probe.capture is not None or probe.finished_features is not None)):
        raise ValueError('draft update requires an owned idle rollout with released private features')
    identity = dict(run_id=run_id, generation=generation, operation_id=operation_id, expected_revision=expected_revision, snapshot=dict(snapshot))
    pending = worker._speco_pending_draft_update
    if pending is not None:
        if pending['identity'] != identity:
            raise RuntimeError('a different draft update already owns this worker')
        return pending['prepared_receipt']
    if worker._speco_draft_update_phase != 'idle':
        raise RuntimeError('uncertain draft update requires rollout recovery')
    config = worker.model_runner.drafter.model.config
    loaded, manifest, weights = read_draft_update(snapshot, target_fingerprint=config.speco_snapshot_target_fingerprint, tokenizer_fingerprint=config.speco_snapshot_tokenizer_fingerprint, model_config=config.to_dict())
    if loaded.draft_revision != expected_revision + 1 or manifest['generation'] != generation:
        raise ValueError('draft update must advance exactly one revision in the owned generation')
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    tp_rank = get_tensor_model_parallel_rank()
    tp_world = get_tensor_model_parallel_world_size()
    expected = expected_native_shards(weights, config=manifest['model_config'], rank=tp_rank, world_size=tp_world)
    actual = {name: param for name, param in worker.model_runner.drafter.model.named_parameters() if not name.startswith('lm_head.')}
    if set(actual) != set(expected) or any((actual[name].shape != tensor.shape for name, tensor in expected.items())):
        raise ValueError('draft update cannot cover the actual native TP parameter geometry')
    receipt = dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, run_id=run_id, generation=generation, operation_id=operation_id, snapshot_checksum=loaded.checksum, draft_revision=loaded.draft_revision, prepared=True)
    worker._speco_pending_draft_update = dict(identity=identity, snapshot=loaded, manifest=manifest, weights=weights, tp_rank=tp_rank, tp_world=tp_world, prepared_receipt=receipt)
    worker._speco_draft_update_phase = 'prepared'
    return receipt

def invalidate_draft_caches(worker):
    """Invalidate draft-owned state while retaining target parameter sink KV."""
    runner = worker.model_runner
    histories = worker._speco_mome_histories
    if any((history._active for history in histories)):
        raise RuntimeError('cannot invalidate an active MoME convolution')
    caches = [getattr(layer.self_attn.attn, 'kv_cache', ()) for layer in runner.drafter.model.model.layers]
    cleared = set()

    def clear(value):
        if isinstance(value, torch.Tensor):
            if value.numel() and value.data_ptr() not in cleared:
                value.zero_()
                cleared.add(value.data_ptr())
        elif isinstance(value, (tuple, list)):
            for item in value:
                clear(item)
    with torch.inference_mode():
        for cache in caches:
            clear(cache)
        if not cleared:
            raise RuntimeError('draft publication could not locate actual allocated draft caches')
        for history in histories:
            history.windows.clear()
            clear(getattr(history.conv, '_state', None))
        clear(getattr(runner.drafter, '_dflash_hidden_states', None))
    return len(cleared)

def commit_worker_draft_update(worker, *, operation_id):
    pending = worker._speco_pending_draft_update
    if pending is None:
        if worker._speco_last_draft_ack is not None and worker._speco_last_draft_ack['operation_id'] == operation_id:
            return worker._speco_last_draft_ack
        raise RuntimeError('draft commit lacks a validated pending snapshot')
    if pending['identity']['operation_id'] != operation_id or worker._speco_draft_update_phase != 'prepared':
        raise RuntimeError('draft commit has a foreign or uncertain operation')
    worker._speco_draft_update_phase = 'applying'
    try:
        model = worker.model_runner.drafter.model
        model.load_weights(pending['weights'].items())
        audit = audit_native_draft(model, pending['weights'], config=pending['manifest']['model_config'], rank=pending['tp_rank'], world_size=pending['tp_world'])
        runner = worker.model_runner
        cleared = invalidate_draft_caches(worker)
        torch.npu.synchronize()
        if model.lm_head is not runner.model.lm_head or model.model.embed_tokens is runner.model.model.embed_tokens:
            raise RuntimeError('draft hot update lost target head binding or independent embedding')
        receipt = dict(pending['prepared_receipt'], prepared=False, applied=True, **audit, kv_cache_cleared=True, cache_tensors_cleared=cleared, draft_target_head_synced=True, target_parameter_sink_cache_preserved=True)
        worker._speco_draft_revision = pending['snapshot'].draft_revision
        worker._speco_last_draft_ack = receipt
        worker._speco_pending_draft_update = None
        worker._speco_draft_update_phase = 'ack_pending'
        return receipt
    except BaseException:
        worker._speco_draft_update_phase = 'failed'
        raise

def finish_worker_draft_update(worker, *, operation_id, snapshot_checksum):
    ack = worker._speco_last_draft_ack
    if worker._speco_draft_update_phase not in ('ack_pending', 'idle') or ack is None or ack['operation_id'] != operation_id or (ack['snapshot_checksum'] != snapshot_checksum):
        raise RuntimeError('draft update cannot resume before matching all-rank commit acknowledgement')
    worker._speco_draft_update_phase = 'idle'
    return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, operation_id=operation_id, snapshot_checksum=snapshot_checksum, released=True)

def require_draft_update_acks(rows, *, expected_incarnations, run_id, generation, operation_id, snapshot, phase):
    if not isinstance(rows, list) or len(rows) != len(expected_incarnations) or any((not isinstance(row, dict) for row in rows)) or ({row.get('rank') for row in rows} != set(expected_incarnations)):
        raise RuntimeError('draft update returned missing, duplicate or foreign TP ranks')
    for row in rows:
        if row.get('worker_incarnation') != expected_incarnations[row['rank']] or row.get('operation_id') != operation_id or row.get('snapshot_checksum') != snapshot['checksum'] or (phase != 'released' and (row.get('run_id') != run_id or row.get('generation') != generation or row.get('draft_revision') != snapshot['draft_revision'])):
            raise RuntimeError('draft update ACK changed worker, version or snapshot identity')
        if row.get(phase) is not True:
            raise RuntimeError('draft update phase was not acknowledged')
        if phase == 'applied' and (row.get('native_parameters') != 16 or not row.get('fused_context_KV_verified') or (not row.get('kv_cache_cleared')) or (not row.get('draft_target_head_synced')) or (not isinstance(row.get('native_parameter_sha256'), str)) or (len(row['native_parameter_sha256']) != 64)):
            raise RuntimeError('draft update lacks actual native shard/cache/head verification')
    return rows
from types import MethodType
from verl_speco.integration.accelerator_speculation import verify_model_storage
from verl_speco.integration.accelerator_speculation import capture_graph_kv, verify_graph_kv

def audit_graph_publication(worker):
    runner = worker.model_runner
    cold = getattr(worker, '_speco_joint_phase', None) == 'restoring'
    if runner.model_config.enforce_eager or runner.compilation_config.cudagraph_mode.name != 'FULL_DECODE_ONLY' or (not cold and (worker._speco_graph_replay_calls < 1 or worker._speco_graph_live_forward_calls < 1)):
        raise RuntimeError('graph publication requires owned FULL replay or fenced cold restore')
    if cold and (worker._speco_target_revision != 0 or worker._speco_target_update_phase != 'idle' or worker._speco_draft_update_phase != 'idle' or (worker._speco_joint_target_ack is not None) or (not isinstance(worker._speco_joint_parent, dict)) or (worker._speco_checkpoint_channel.phase != 'ready')):
        raise RuntimeError('graph cold bootstrap lacks its fresh fenced joint owner')
    verify_model_storage(runner.model, worker._speco_graph_model_bindings, allow_expert_loader_views=True)
    layers, captured = (worker._graph_attention_layers(), worker._graph_captured_attention())
    if worker._speco_graph_publication_kv is None:
        worker._speco_graph_publication_kv = capture_graph_kv(layers, captured)
    else:
        verify_graph_kv(layers, captured, worker._speco_graph_publication_kv)

def install_graph_prepare_guard(worker):
    original = worker.speco_prepare_target_transfer
    if not callable(original):
        raise ValueError('graph prepare requires the original owned native port')
    if getattr(getattr(original, '__func__', original), '_speco_graph_prepare_guard', False):
        return False

    def prepare(owner, **settings):
        try:
            audit_graph_publication(owner)
            return original(**settings)
        except BaseException:
            owner._speco_target_update_phase = 'failed'
            raise
    prepare._speco_graph_prepare_guard = True
    worker.speco_prepare_target_transfer = MethodType(prepare, worker)
    return True
from copy import deepcopy

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def wire_dtype(name, tensor):
    if tensor.is_floating_point():
        return torch.float32 if '_mhc_module.' in name or name.endswith('e_score_correction_bias') else torch.bfloat16
    return tensor.dtype

def manifest_for_state(state, *, run_id, generation, rl_step, target_revision, operation_id, actor_incarnations):
    rows = []
    for name, tensor in state.items():
        if not name or not isinstance(tensor, torch.Tensor) or tensor.numel() < 1:
            raise ValueError('target wire requires actual nonempty global state tensors')
        dtype = wire_dtype(name, tensor)
        rows.append(dict(name=name, shape=list(tensor.shape), dtype=str(dtype), nbytes=tensor.numel() * dtype.itemsize))
    identity = dict(schema_version=1, run_id=run_id, generation=generation, rl_step=rl_step, target_revision=target_revision, operation_id=operation_id, actor_incarnations={str(key): value for key, value in actor_incarnations.items()}, tensors=rows, total_nbytes=sum((row['nbytes'] for row in rows)))
    return dict(identity, header_sha256=hashlib.sha256(canonical(identity)).hexdigest())

class TargetWireDigest:
    """Digest the actual whole stream, never just loaded names or finite probes."""

    def __init__(self, manifest, *, chunk_bytes=16 * 1024 ** 2):
        self.manifest = deepcopy(manifest)
        header = {key: value for key, value in manifest.items() if key != 'header_sha256'}
        if hashlib.sha256(canonical(header)).hexdigest() != manifest.get('header_sha256'):
            raise ValueError('target wire manifest was mutated')
        names = [row['name'] for row in manifest['tensors']]
        if not names or len(set(names)) != len(names) or chunk_bytes < 8:
            raise ValueError('target wire manifest has duplicate or missing tensors')
        self.digest = hashlib.sha256(canonical(header))
        self.count = 0
        self.total = 0
        self.chunk_bytes = chunk_bytes

    def update(self, name, value):
        if self.count >= len(self.manifest['tensors']):
            raise RuntimeError('target wire emitted extra tensor')
        row = self.manifest['tensors'][self.count]
        if name != row['name'] or list(value.shape) != row['shape'] or str(value.dtype) != row['dtype'] or (value.numel() * value.element_size() != row['nbytes']):
            raise ValueError('actual target wire tensor differs from bound manifest')
        self.digest.update(canonical(row))
        flat = value.detach().reshape(-1)
        size = max(1, self.chunk_bytes // value.element_size())
        for start in range(0, flat.numel(), size):
            chunk = flat[start:start + size].cpu().contiguous()
            if chunk.is_floating_point() and (not torch.isfinite(chunk).all()):
                raise ValueError('actual target wire contains nonfinite weights')
            self.digest.update(memoryview(chunk.view(torch.uint8).numpy()))
        self.count += 1
        self.total += row['nbytes']

    def finish(self):
        if self.count != len(self.manifest['tensors']) or self.total != self.manifest['total_nbytes']:
            raise RuntimeError('target wire ended before full state coverage')
        return dict(header_sha256=self.manifest['header_sha256'], actor_weight_manifest_sha256=self.digest.hexdigest(), wire_tensor_count=self.count, wire_nbytes=self.total)

def target_wire_native_coverage_owner(model):
    """Use the native owner behind Ascend's read-only attribute proxy.

    ACLGraphWrapper delegates reads but not assignments. Writing its ready
    flag would shadow the native flag updated by the bound readiness method.
    No other wrapper or foreign bound method is accepted as an owner.
    """
    seen = set()
    while type(model).__name__ == 'ACLGraphWrapper' and type(model).__module__ == 'vllm_ascend.compilation.acl_graph':
        if id(model) in seen:
            raise ValueError('cyclic native graph coverage owner')
        seen.add(id(model))
        if any((field in vars(model) for field in ('_loaded_cumulative', '_pangu_ready', '_expert_slots_seen', '_expert_declined'))):
            raise ValueError('graph wrapper shadows native coverage state')
        model = vars(model)['runnable']
    owner = getattr(getattr(model, '_assert_ready', None), '__self__', model)
    if owner is not model:
        raise ValueError('native readiness method belongs to a foreign owner')
    return model

def reset_native_weight_coverage(model):
    """Each target revision must independently satisfy the native guards."""
    model = target_wire_native_coverage_owner(model)
    for field in ('_loaded_cumulative', '_pangu_ready', '_expert_slots_seen', '_expert_declined'):
        if not hasattr(model, field):
            raise ValueError('actual target model native target has no audited coverage contract')
    names = model._loaded_cumulative
    slots = model._expert_slots_seen
    declined = model._expert_declined
    if not isinstance(names, set) or not isinstance(slots, set) or (not isinstance(declined, list)):
        raise ValueError('actual target model native coverage containers changed their contract')
    names.clear()
    slots.clear()
    declined.clear()
    model._pangu_ready = False

def verify_native_weight_coverage(model):
    model = target_wire_native_coverage_owner(model)
    model._assert_ready()
    params = dict(model.named_parameters())
    missing = [name for name in params if not model._exempt(name) and name not in model._loaded_cumulative]
    if missing or not model._pangu_ready:
        raise RuntimeError(f'updated target has incomplete native parameter coverage: missing={missing[:12]}, ready={model._pangu_ready}, loaded={len(model._loaded_cumulative)}, parameters={len(params)}')
    model._assert_expert_slots_complete()
    return dict(native_parameter_count=len(params), complete_native_names_and_expert_slots=True, full_native_parameter_bytes_not_claimed=True)

async def load_received_target_buckets(stream, model, digest):
    """Load before HCCL double-buffer storage can be reused by the next bucket."""
    storage = None
    batch = []
    with torch.inference_mode():
        async for name, value in stream:
            current = (value.device, value.untyped_storage().data_ptr())
            if storage is not None and current != storage:
                model.load_weights(batch)
                batch = []
            storage = current
            digest.update(name, value)
            batch.append((name, value.clone()))
        if batch:
            model.load_weights(batch)
    return digest.finish()
import asyncio
from verl_speco.integration.accelerator_speculation import capture_graph_expert_bindings, prepare_graph_expert_reload, finish_graph_expert_reload
from verl_speco.integration.accelerator_speculation import capture_model_storage, verify_model_storage
from verl_speco.integration.accelerator_speculation import capture_graph_kv, verify_graph_kv, refresh_graph_target_state
from verl_speco.integration.accelerator_speculation import GraphAcceptedTargetQueryProbe
from verl_speco.integration.accelerator_speculation import select_target_graph_metadata

class GraphTargetPublicationMixin:

    def load_model(self):
        super().load_model()
        model = self.model_runner.model
        self._speco_graph_expert_bindings = capture_graph_expert_bindings(model)
        self._speco_graph_model_bindings = capture_model_storage(model, expert_names=self._speco_graph_expert_bindings.tensors.keys())
        self._speco_graph_publication_kv = None
        self._speco_graph_last_publication = None

    def _graph_attention_layers(self):
        return tuple(((layer.self_attn.attn.layer_name, layer.self_attn.attn) for layer in self._speco_aux_target.model.model.layers))

    def _graph_captured_attention(self):
        from vllm_ascend.attention.attention_v1 import get_graph_params
        query = self._speco_dspark_adapter.block_size + 1
        return get_graph_params().attn_params[query]

    def speco_install_checkpoint_channel(self, **settings):
        receipt = super().speco_install_checkpoint_channel(**settings)
        install_graph_prepare_guard(self)
        return receipt

    def speco_install_joint_native_restore(self, **settings):
        receipt = super().speco_install_joint_native_restore(**settings)
        install_graph_prepare_guard(self)
        return receipt

    def speco_prepare_target_transfer(self, **settings):
        try:
            audit_graph_publication(self)
            return super().speco_prepare_target_transfer(**settings)
        except BaseException:
            self._speco_target_update_phase = 'failed'
            raise

    @torch.inference_mode()
    def speco_receive_target_transfer(self, *, operation_id):
        pending = self._speco_target_update
        if self._speco_target_update_phase != 'connected' or pending['manifest']['operation_id'] != operation_id or self._speco_graph_publication_kv is None:
            raise RuntimeError('graph target receive lacks a connected owned stream')
        self._speco_target_update_phase = 'applying'
        model = self.model_runner.model
        try:
            verify_model_storage(model, self._speco_graph_model_bindings, allow_expert_loader_views=True)
            layout = prepare_graph_expert_reload(model, self._speco_graph_expert_bindings)
            reset_native_weight_coverage(model)
            stream = pending['engine'].receive_weights(global_steps=pending['manifest']['rl_step'])
            receipt = asyncio.run(load_received_target_buckets(stream, model, pending['digest']))
            receipt.update(verify_native_weight_coverage(model))
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            process_weights_after_loading(model, self.model_runner.model_config, self.model_runner.device)
            model.post_weight_load()
            receipt.update(finish_graph_expert_reload(model, self._speco_graph_expert_bindings, layout))
            receipt.update(verify_model_storage(model, self._speco_graph_model_bindings))
            receipt.update(verify_graph_kv(self._graph_attention_layers(), self._graph_captured_attention(), self._speco_graph_publication_kv))
            receipt['native_expert_loader_layout_restored_and_reprocessed'] = True
            torch.npu.synchronize()
            pending['receipt'] = receipt
            self._speco_target_update_phase = 'loaded'
            return dict(receipt, rank=self.rank, worker_incarnation=self._speco_serving_incarnation, loaded=True)
        except BaseException:
            self._speco_target_update_phase = 'failed'
            raise

    def speco_commit_target_transfer(self, **settings):
        try:
            verify_model_storage(self.model_runner.model, self._speco_graph_model_bindings)
            receipt = super().speco_commit_target_transfer(**settings)
            proof = refresh_graph_target_state(self._graph_attention_layers(), self._graph_captured_attention(), self._speco_graph_publication_kv, self._speco_mome_histories)
            proof.update(verify_model_storage(self.model_runner.model, self._speco_graph_model_bindings))
            torch.npu.synchronize()
            self._speco_graph_last_publication = proof
            self._speco_graph_replays_at_publication = self._speco_graph_replay_calls
            return dict(receipt, **proof)
        except BaseException:
            self._speco_target_update_phase = 'failed'
            raise

    def speco_accepted_query_arm(self, **settings):
        if not hasattr(self, '_speco_accepted_probe'):
            from vllm.forward_context import get_forward_context
            self._speco_accepted_probe = GraphAcceptedTargetQueryProbe(self.model_runner, metadata_supplier=lambda: select_target_graph_metadata(get_forward_context().attn_metadata, self._speco_target_attention_layer_names))
        receipt = super().speco_accepted_query_arm(**settings)
        receipt.update(single_request_eager_only=False, actual_runner_return_feature_capture=True)
        return receipt

    def speco_graph_publication_readiness(self):
        probe = getattr(self, '_speco_accepted_probe', None)
        if self._speco_target_update_phase != 'idle' or self._speco_draft_update_phase != 'idle' or (probe is not None and (probe.capture is not None or probe.finished_features is not None)):
            raise RuntimeError('graph readiness requires released target and draft owners')
        audit_graph_publication(self)
        proof = verify_model_storage(self.model_runner.model, self._speco_graph_model_bindings)
        proof.update(verify_graph_kv(self._graph_attention_layers(), self._graph_captured_attention(), self._speco_graph_publication_kv))
        return dict(proof, rank=self.rank, worker_incarnation=self._speco_serving_incarnation, publication_not_performed=True, actual_graph_replays=self._speco_graph_replay_calls)

    def speco_dspark_runtime_status(self):
        receipt = super().speco_dspark_runtime_status()
        receipt.update(graph_last_weight_publication=self._speco_graph_last_publication, target_weight_revision=self._speco_target_revision, target_replays_after_last_publication=self._speco_graph_replay_calls - getattr(self, '_speco_graph_replays_at_publication', self._speco_graph_replay_calls), full_training_or_joint_restore_acceptance_not_claimed=True)
        return receipt
from dataclasses import dataclass, replace
from verl_speco.integration.multinode_online import OnlinePhase, MultiNodeOnlineAdapter as MultiNodeVerlOnlineAdapter

@dataclass(frozen=True)
class TargetWeightAck:
    run_id: str
    generation: int
    rl_step: int
    target_revision: int
    worker_id: str
    worker_incarnation: str
    actor_weight_manifest_sha256: str
    kv_cache_cleared: bool
    draft_target_head_synced: bool

    def __post_init__(self):
        for field in ('run_id', 'worker_id', 'worker_incarnation'):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError('target ACK requires nonempty immutable worker identities')
        if any((type(getattr(self, field)) is not int or getattr(self, field) < 1 for field in ('generation', 'rl_step', 'target_revision'))):
            raise ValueError('target ACK requires positive independent revision counters')
        digest = self.actor_weight_manifest_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any((character not in '0123456789abcdef' for character in digest)):
            raise ValueError('target ACK requires the actor weight manifest SHA256')
        if type(self.kv_cache_cleared) is not bool or type(self.draft_target_head_synced) is not bool:
            raise ValueError('target ACK cache/head flags must be explicit booleans')

def commit_target_weight_acks(adapter: MultiNodeVerlOnlineAdapter, *, acknowledgements: tuple[TargetWeightAck, ...], expected_incarnations: dict[str, str], target_revision: int, actor_weight_manifest_sha256: str):
    """Map the actor-to-serving ACK event after frozen-head draft training.

    The collection/training binding keeps the old target revision. The next
    rollout observes the newly acknowledged revision. No RPC is sent and no
    request is resumed here; the caller must supply actual per-rank receipts
    from the weight-transfer/cache/head synchronization executor.
    """
    adapter._require(OnlinePhase.TRAINED)
    try:
        if type(target_revision) is not int or target_revision != adapter.versions.target_revision + 1:
            raise ValueError('target serving revision must advance exactly once')
        if not isinstance(expected_incarnations, dict) or not expected_incarnations or any((not isinstance(key, str) or not key or (not isinstance(value, str)) or (not value) for key, value in expected_incarnations.items())):
            raise ValueError('target update requires every expected rank incarnation')
        if not isinstance(acknowledgements, tuple) or any((not isinstance(ack, TargetWeightAck) for ack in acknowledgements)) or len(acknowledgements) != len(expected_incarnations) or ({ack.worker_id for ack in acknowledgements} != set(expected_incarnations)):
            raise RuntimeError('target weight update returned missing, duplicate or foreign ranks')
        for ack in acknowledgements:
            if ack.run_id != adapter.versions.run_id or ack.generation != adapter.versions.generation or ack.rl_step != adapter.versions.rl_step or (ack.target_revision != target_revision) or (ack.worker_incarnation != expected_incarnations[ack.worker_id]) or (ack.actor_weight_manifest_sha256 != actor_weight_manifest_sha256):
                raise RuntimeError('target weight ACK belongs to a different actor update or worker incarnation')
            if not ack.kv_cache_cleared or not ack.draft_target_head_synced:
                raise RuntimeError('target update has incomplete KV invalidation or draft target-head synchronization')
        adapter.versions = replace(adapter.versions, target_revision=target_revision)
        return dict(target_revision=target_revision, acknowledged_workers=len(acknowledgements), actor_weight_manifest_sha256=actor_weight_manifest_sha256)
    except Exception as error:
        adapter.on_role_failure('target_weight_ack', error)
        raise
from dataclasses import fields
from verl_speco.integration.multinode_online import async_online_await as _await

def require_owned_rows(rows, owners, *, phase):
    if not isinstance(rows, list) or len(rows) != len(owners) or any((not isinstance(row, dict) for row in rows)) or ({row.get('rank') for row in rows} != set(owners)):
        raise RuntimeError('target transfer returned missing, duplicate or foreign ranks')
    for row in rows:
        if row.get('worker_incarnation') != owners[row['rank']] or row.get(phase) is not True:
            raise RuntimeError('target transfer owner or phase acknowledgement differs')
    return rows

class MultiNodeTargetPublishExecutor:
    """A safe-point session port; target publication is a fenced whole revision."""

    def __init__(self, *, actor_group, actor_incarnations, draft_group, draft_incarnations, collective_rpc, rollout_incarnations, reset_prefix_cache, tokenizer_fingerprint, bucket_bytes=1024 ** 3):
        for owners in (actor_incarnations, rollout_incarnations):
            if not owners or set(owners) != set(range(len(owners))) or any((type(rank) is not int or not incarnation for rank, incarnation in owners.items())):
                raise ValueError('target publisher requires exact contiguous HCCL memberships')
        self.group = actor_group
        self.actors = dict(actor_incarnations)
        self.draft = draft_group
        self.draft_owners = dict(draft_incarnations)
        self.rpc = collective_rpc
        self.rollout = dict(rollout_incarnations)
        self.reset_prefix_cache = reset_prefix_cache
        self.tokenizer = tokenizer_fingerprint
        self.bucket_bytes = bucket_bytes
        self.failed = False
        self.lock = asyncio.Lock()
        self.last_proof = None

    async def __call__(self, actor_result, binding):
        async with self.lock:
            if self.failed:
                raise RuntimeError('uncertain target transfer requires serving and actor recovery')
            versions = binding.versions
            if actor_result.get('run_id') != versions.run_id or actor_result.get('generation') != versions.generation or actor_result.get('rl_step') != versions.rl_step or (actor_result.get('target_revision') != versions.target_revision) or (actor_result.get('collection_id') != binding.collection_id) or (actor_result.get('optimizer_committed') is not True) or (actor_result.get('worker_incarnations') != self.actors):
                raise ValueError('target publisher has no exact actual actor optimizer receipt')
            operation_id = uuid4().hex
            request = dict(run_id=versions.run_id, generation=versions.generation, rl_step=versions.rl_step, target_revision=versions.target_revision + 1, operation_id=operation_id, actor_incarnations={str(rank): incarnation for rank, incarnation in self.actors.items()}, optimizer_step=actor_result['optimizer_step'], bucket_bytes=self.bucket_bytes, tokenizer_fingerprint=self.tokenizer)
            try:
                source = await asyncio.to_thread(self.group.speco_prepare_target_transfer, request)
                require_owned_rows(source, self.actors, phase='prepared')
                master = next((row for row in source if row['rank'] == 0))
                manifest = master['manifest']
                if manifest is None or master['master_metadata'] is None or any((row['header_sha256'] != manifest['header_sha256'] for row in source)):
                    raise RuntimeError('actor ranks disagreed on the complete target manifest')
                prepared = await _await(self.rpc('speco_prepare_target_transfer', dict(manifest=manifest, bucket_bytes=self.bucket_bytes)))
                require_owned_rows(prepared, self.rollout, phase='prepared')
                if any((row['header_sha256'] != manifest['header_sha256'] for row in prepared)):
                    raise RuntimeError('native target prepare did not bind the full actor manifest')
                topology = dict(operation_id=operation_id, master_metadata=master['master_metadata'], rollout_world_size=len(self.rollout))
                connected = await asyncio.gather(asyncio.to_thread(self.group.speco_init_target_transfer, topology), _await(self.rpc('speco_init_target_transfer', topology)))
                require_owned_rows(connected[0], self.actors, phase='connected')
                require_owned_rows(connected[1], self.rollout, phase='connected')
                sent, loaded = await asyncio.gather(asyncio.to_thread(self.group.speco_send_target_transfer, operation_id), _await(self.rpc('speco_receive_target_transfer', dict(operation_id=operation_id))))
                require_owned_rows(sent, self.actors, phase='sent')
                require_owned_rows(loaded, self.rollout, phase='loaded')
                source = next((row for row in sent if row['rank'] == 0))
                receipt = source['wire_receipt']
                for row in loaded:
                    if any((row.get(key) != value for key, value in receipt.items())) or row.get('complete_native_names_and_expert_slots') is not True:
                        raise RuntimeError('native target did not receive and cover the actual whole actor state')
                sha = receipt['actor_weight_manifest_sha256']
                head = source['target_head_binding']
                if head is None or head.get('actor_weight_manifest_sha256') != sha or head.get('target_revision') != request['target_revision']:
                    raise RuntimeError('actual actor target head is missing or foreign')
                registered = await asyncio.to_thread(self.draft.speco_register_online_target_head, head)
                if not isinstance(registered, list) or len(registered) != len(self.draft_owners) or {row.get('worker_id') for row in registered} != set(self.draft_owners) or any((row.get('worker_incarnation') != self.draft_owners[row['worker_id']] or row.get('registered') is not True or row.get('target_revision') != request['target_revision'] for row in registered)):
                    raise RuntimeError('draft trainer did not retain the immutable next target head')
                if await _await(self.reset_prefix_cache()) is not True:
                    raise RuntimeError('native scheduler refused actual target prefix-cache reset')
                identity = dict(operation_id=operation_id, actor_weight_manifest_sha256=sha)
                acknowledgements = await _await(self.rpc('speco_commit_target_transfer', identity))
                require_owned_rows(acknowledgements, self.rollout, phase='applied')
                for row in acknowledgements:
                    if row.get('run_id') != versions.run_id or row.get('generation') != versions.generation or row.get('rl_step') != versions.rl_step or (row.get('target_revision') != request['target_revision']) or (row.get('actor_weight_manifest_sha256') != sha) or (row.get('kv_cache_cleared') is not True) or (row.get('draft_target_head_synced') is not True):
                        raise RuntimeError('target commit has incomplete actual weight/cache/head acknowledgement')
                source_released = await asyncio.to_thread(self.group.speco_finish_target_transfer, operation_id, sha)
                require_owned_rows(source_released, self.actors, phase='released')
                native_released = await _await(self.rpc('speco_finish_target_transfer', identity))
                require_owned_rows(native_released, self.rollout, phase='released')
                names = {field.name for field in fields(TargetWeightAck)}
                result = dict(target_revision=request['target_revision'], actor_weight_manifest_sha256=sha, acknowledgements=[{key: value for key, value in row.items() if key in names} for row in acknowledgements])
                self.last_proof = dict(operation_id=operation_id, manifest=manifest, source=sent, loaded=loaded, immutable_head=head, registered_head=registered, native_ACKs=acknowledgements, actual_native_prefix_reset=True, source_released=source_released, native_released=native_released, full_native_shard_bytes_or_actor_DCP_recovery_not_claimed=True)
                return result
            except BaseException:
                self.failed = True
                raise

def install_native_checkpoint_channel(worker, *, identity):
    """Callable vLLM collective RPC; installed packages and worker class stay intact."""
    from verl_speco.integration.accelerator_adapters import SpecoMultiNodeTargetUpdateWorker as SpecoMultiNodeTargetUpdateWorker
    if not isinstance(worker, SpecoMultiNodeTargetUpdateWorker) or identity['run_id'] != os.environ.get('SPECO_RUN_ID') or identity['generation'] != int(os.environ.get('SPECO_GENERATION', '-1')) or (getattr(worker, '_speco_checkpoint_channel', None) is not None) or (worker._speco_target_revision != 0) or (worker._speco_target_update_phase != 'idle') or (worker._speco_draft_update_phase != 'idle') or (getattr(worker, '_speco_accepted_probe', None) is not None):
        raise ValueError('native preconnection requires its fresh dedicated pre-collection owner')
    channel = MultiNodeCheckpointChannel(identity=identity, role='rollout', rank=worker.rank, incarnation=worker._speco_serving_incarnation)
    worker._speco_checkpoint_channel = channel
    worker._speco_target_update_phase = 'preconnecting'
    try:
        channel.prepare(identity)
        worker.speco_prepare_target_transfer = MethodType(preconnected_target_native_prepare_target, worker)
        worker.speco_init_target_transfer = MethodType(preconnected_target_native_target_connected, worker)
        original_finish = worker.speco_finish_target_transfer
        worker._speco_preconnected_original_finish = original_finish
        worker.speco_finish_target_transfer = MethodType(preconnected_target_native_finish_target, worker)
        return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, prepared=True)
    except BaseException:
        worker._speco_target_update_phase = 'failed'
        raise

def connect_native_checkpoint_channel(worker, *, identity, master_metadata):
    if worker._speco_target_update_phase != 'preconnecting':
        raise RuntimeError('native checkpoint connect has no fresh fenced preparation')
    try:
        worker._speco_checkpoint_channel.connect(identity, master_metadata)
        worker._speco_target_update_phase = 'idle'
        return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, connected=True)
    except BaseException:
        worker._speco_target_update_phase = 'failed'
        raise

def preconnected_target_native_prepare_target(worker, *, manifest, bucket_bytes):
    channel = worker._speco_checkpoint_channel
    probe = getattr(worker, '_speco_accepted_probe', None)
    if manifest['run_id'] != channel.identity['run_id'] or any((type(manifest[key]) is not int or manifest[key] < 1 for key in ('generation', 'rl_step', 'target_revision'))) or manifest['generation'] != channel.identity['generation'] or (manifest['actor_incarnations'] != {str(k): v for k, v in channel.identity['actors'].items()}) or (manifest['target_revision'] != worker._speco_target_revision + 1) or (worker._speco_target_update_phase != 'idle') or (worker._speco_draft_update_phase != 'idle') or (probe is not None and (probe.capture is not None or probe.finished_features is not None)):
        raise ValueError('preconnected native target requires its released exact collection owner')
    digest = TargetWireDigest(manifest)
    if len(manifest['tensors']) != 35869 or type(bucket_bytes) is not int or bucket_bytes != channel.identity['bucket_bytes'] or (bucket_bytes < max((row['nbytes'] for row in manifest['tensors']))) or (os.environ.get('PANGU_92B_EXPERT_SLOTS_EAGER', '') in ('1', 'true', 'True')):
        raise ValueError('preconnected native target requires the bounded complete bucketed state')
    worker._speco_target_update_phase = 'preparing'
    try:
        engine = channel.begin(channel.identity, manifest['operation_id'])
        worker._speco_target_update = dict(manifest=manifest, digest=digest, engine=engine, receipt=None)
        worker._speco_target_update_phase = 'prepared'
        return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, header_sha256=manifest['header_sha256'], prepared=True)
    except BaseException:
        worker._speco_target_update_phase = 'failed'
        raise

def preconnected_target_native_target_connected(worker, *, operation_id, master_metadata, rollout_world_size):
    channel = worker._speco_checkpoint_channel
    if worker._speco_target_update_phase != 'prepared' or channel.phase != 'active' or operation_id != channel.operation_id or (master_metadata != channel.master_metadata) or (rollout_world_size != len(channel.identity['rollout'])):
        raise RuntimeError('preconnected native target topology differs from its generation')
    worker._speco_target_update_phase = 'connected'
    return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, connected=True)

def preconnected_target_native_finish_target(worker, *, operation_id, actor_weight_manifest_sha256):
    try:
        result = worker._speco_preconnected_original_finish(operation_id=operation_id, actor_weight_manifest_sha256=actor_weight_manifest_sha256)
        channel = worker._speco_checkpoint_channel
        channel.finish(channel.identity, operation_id, whole_revision_acknowledged=True)
        return result
    except BaseException:
        worker._speco_target_update_phase = 'failed'
        raise

def close_native_checkpoint_channel(worker, *, identity):
    if worker._speco_target_update_phase != 'idle' or worker._speco_target_update is not None:
        raise RuntimeError('native checkpoint close requires the fully released publication safe point')
    try:
        worker._speco_checkpoint_channel.close(identity)
        worker._speco_target_update_phase = 'closed'
        return dict(rank=worker.rank, worker_incarnation=worker._speco_serving_incarnation, closed=True)
    except BaseException:
        worker._speco_target_update_phase = 'failed'
        raise

async def preconnect_target_channel(*, actor_group, collective_rpc, identity):
    """All owners must connect before admitting any PPO collection/update."""
    source = await asyncio.to_thread(actor_group.speco_prepare_checkpoint_channel, identity)
    require_owned_rows(source, identity['actors'], phase='prepared')
    master = next((row for row in source if row['rank'] == 0))['master_metadata']
    if not isinstance(master, dict):
        raise RuntimeError('preconnected source lacks its actual master metadata')
    receivers = await collective_rpc('speco_install_checkpoint_channel', dict(identity=identity))
    require_owned_rows(receivers, identity['rollout'], phase='prepared')
    connected = await asyncio.gather(asyncio.to_thread(actor_group.speco_connect_checkpoint_channel, dict(identity=identity, master_metadata=master)), collective_rpc('speco_connect_checkpoint_channel', dict(identity=identity, master_metadata=master)))
    require_owned_rows(connected[0], identity['actors'], phase='connected')
    require_owned_rows(connected[1], identity['rollout'], phase='connected')
    return dict(identity=identity, master_metadata=master, actor_connected=connected[0], native_connected=connected[1], target_revision_applied=False)
