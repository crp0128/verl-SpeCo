# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import functools
import hashlib
import logging
import os
import uuid
import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import TrainingWorker

def model_92b_actor_worker_local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor

def model_92b_actor_worker_probe(tensor):
    value = model_92b_actor_worker_local(tensor).detach().reshape(-1)[:256].cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite actual actor parameter probe')
    return dict(shape=list(tensor.shape), local_shape=list(model_92b_actor_worker_local(tensor).shape), dtype=str(tensor.dtype), sampled_elements=value.numel(), sample_sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())

class MultiNode92BActorWorker(TrainingWorker):

    def __init__(self, config):
        from verl_speco.integration.multinode_online import install_speco_92b_training_profile as install_speco_92b_training_profile
        self.speco_config = dict(config)
        self.worker_incarnation = uuid.uuid4().hex
        install_speco_92b_training_profile(cache_dir=config['profile_cache'], model_path=config['model_path'])
        from verl.workers.config import HFModelConfig, TrainingWorkerConfig, VeOmniActorConfig
        from verl.workers.config.engine import VeOmniEngineConfig
        from verl.workers.config.optimizer import VeOmniOptimizerConfig
        from verl.trainer.config import CheckpointConfig
        from verl.workers.utils.losses import ppo_loss
        world = int(os.environ['WORLD_SIZE'])
        if world < 32 or world % 16:
            raise ValueError('full 92B actor requires at least two complete 16-die nodes')
        engine = VeOmniEngineConfig(fsdp_size=world, ulysses_parallel_size=1, expert_parallel_size=1, init_device='meta', enable_full_shard=True, mixed_precision=True, attn_implementation='sdpa', moe_implementation='eager', basic_modules=['OpenPanguV2MLP'], use_torch_compile=False, use_dynamic_bsz=False, micro_batch_size_per_gpu=1, infer_micro_batch_size_per_gpu=1, param_offload=False, optimizer_offload=False, enable_fsdp_offload=False, enable_reentrant=False, forward_prefetch=False)
        model = HFModelConfig(path=config['model_path'], trust_remote_code=True, load_tokenizer=False, override_config={'attn_implementation': 'sdpa'}, use_remove_padding=False, use_fused_kernels=False, enable_gradient_checkpointing=True)
        optimizer = VeOmniOptimizerConfig(lr=config.get('learning_rate', 1e-06), total_training_steps=config.get('total_steps', 2), lr_warmup_steps_ratio=0, weight_decay=0.01, clip_grad=1)
        actor = VeOmniActorConfig(veomni=engine, optim=optimizer, ppo_micro_batch_size_per_gpu=1, ppo_mini_batch_size=world, rollout_n=2, entropy_coeff=0, use_kl_loss=False)
        super().__init__(TrainingWorkerConfig(model_type='language_model', model_config=model, engine_config=engine, optimizer_config=optimizer, checkpoint_config=CheckpointConfig(save_contents=['model', 'optimizer', 'extra'])))
        self.set_loss_fn(functools.partial(ppo_loss, config=actor))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        for name in tuple(logging.Logger.manager.loggerDict):
            if name.startswith('veomni.'):
                logging.getLogger(name).setLevel(logging.WARNING)
        self.reset()
        from torch.distributed.fsdp import FSDPModule
        experts = [module for module in self.engine.module.modules() if any((cls.__name__ == 'OpenPanguV2MLP' for cls in type(module).__mro__))]
        if not experts or not all((isinstance(module, FSDPModule) for module in experts)):
            raise RuntimeError('92B MoE expert modules were not individually wrapped with FSDP2')
        return self.speco_actor_audit()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_actor_audit(self):
        parameters = dict(self.engine.module.named_parameters())
        families = dict(attention='model.layers.0.self_attn.q_a_proj.weight', mhc='model.layers.0.attn_mhc_module.branch_alpha_pre', moe='model.layers.2.mlp.experts.0.gate_proj.weight', head='lm_head.weight')
        if families['mhc'] not in parameters:
            matches = [name for name in parameters if name.startswith('model.layers.0.') and name.endswith('mhc_module.branch_alpha_pre')]
            if len(matches) != 2:
                raise RuntimeError('actual 92B mHC parameter family is absent or ambiguous')
            families['mhc'] = sorted(matches)[0]
        probes = {family: dict(name=name, **model_92b_actor_worker_probe(parameters[name])) for family, name in families.items()}
        steps = set()
        for state in self.engine.optimizer.state.values():
            if state == {}:
                continue
            if 'step' not in state:
                raise RuntimeError('actual actor Adam state has no step counter')
            steps.add(int(model_92b_actor_worker_local(state['step']).item()))
        if len(steps) > 1:
            raise RuntimeError('actual actor Adam counters disagree')
        device = torch.npu
        return dict(run_id=self.speco_config['run_id'], generation=self.speco_config['generation'], worker_incarnation=self.worker_incarnation, rank=self.rank, hostname=os.uname().nodename, model_class=type(self.engine.module).__name__, parameter_tensors=len(parameters), global_parameter_elements=sum((param.numel() for param in parameters.values())), optimizer_parameters=len(self.engine.optimizer.state), optimizer_step=next(iter(steps), 0), probes=probes, memory_allocated=device.memory_allocated(), memory_reserved=device.memory_reserved(), full_parameter_VeRL_actor=True, bounded_shard_probe_not_full_checkpoint_hash=True)
import asyncio
from dataclasses import asdict
from pathlib import Path
import tempfile
from verl_speco.integration.multinode_publication import manifest_for_state, wire_dtype, TargetWireDigest

class MultiNode92BPublishingActorWorker(MultiNode92BActorWorker):

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_prepare_target_transfer(self, request):
        """Bind the actual post-optimizer state before any full-tensor gather."""
        from verl.checkpoint_engine.hccl_checkpoint_engine import HCCLCheckpointEngine
        from verl.utils.model import convert_weight_keys
        from verl_speco.integration.multinode_online import install_hccl_checkpoint_compat
        install_hccl_checkpoint_compat()
        if getattr(self, '_speco_target_transfer', None) is not None:
            raise RuntimeError('actor already owns an unfinished target transfer')
        audit = self.speco_actor_audit()
        owners = request['actor_incarnations']
        if request['run_id'] != self.speco_config['run_id'] or request['generation'] != self.speco_config['generation'] or owners.get(str(self.rank)) != self.worker_incarnation or (request['optimizer_step'] != audit['optimizer_step']) or (audit['optimizer_step'] < 1) or (type(request['rl_step']) is not int) or (request['rl_step'] < 1) or (type(request['target_revision']) is not int) or (request['target_revision'] < 1) or (not request['operation_id']):
            raise ValueError('target transfer belongs to a foreign or uncommitted actor update')
        state = convert_weight_keys(self.engine.module.state_dict(), getattr(self.engine.module, '_fsdp_wrapped_module', self.engine.module))
        manifest = manifest_for_state(state, run_id=request['run_id'], generation=request['generation'], rl_step=request['rl_step'], target_revision=request['target_revision'], operation_id=request['operation_id'], actor_incarnations=owners)
        if len(state) != 35869 or audit['global_parameter_elements'] != 91955618233:
            raise ValueError('target publication requires the audited complete 92B state')
        size = request['bucket_bytes']
        if type(size) is not int or size < max((row['nbytes'] for row in manifest['tensors'])):
            raise ValueError('native HCCL bucket cannot hold the largest actual wire tensor')
        output = Path(self.speco_config['target_transfer_output_root'])
        if not output.is_absolute():
            raise ValueError('target head publication requires an absolute owned output directory')
        engine = HCCLCheckpointEngine(bucket_size=size, group_name=request['operation_id'], rebuild_group=True, is_master=self.rank == 0)
        metadata = engine.prepare()
        self._speco_target_transfer = dict(manifest=manifest, engine=engine, request=dict(request), output=output, phase='prepared', receipt=None)
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, header_sha256=manifest['header_sha256'], prepared=True, manifest=manifest if self.rank == 0 else None, master_metadata=asdict(metadata) if metadata is not None else None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_init_target_transfer(self, request):
        from verl.checkpoint_engine.hccl_checkpoint_engine import MasterMetadata
        pending = self._speco_target_transfer
        if pending['phase'] != 'prepared' or request['operation_id'] != pending['manifest']['operation_id']:
            raise RuntimeError('actor HCCL topology has a foreign or uncertain owner')
        pending['engine'].init_process_group(rank=0 if self.rank == 0 else -1, world_size=request['rollout_world_size'] + 1, master_metadata=MasterMetadata(**request['master_metadata']))
        pending['phase'] = 'connected'
        return dict(rank=self.rank, connected=True, worker_incarnation=self.worker_incarnation)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_send_target_transfer(self, operation_id):
        """All FSDP ranks gather; only checkpoint rank 0 sends real full weights."""
        pending = self._speco_target_transfer
        if pending['manifest']['operation_id'] != operation_id or pending['phase'] != 'connected':
            raise RuntimeError('actor target send lacks a connected owned transfer')
        pending['phase'] = 'sending'
        digest = TargetWireDigest(pending['manifest']) if self.rank == 0 else None
        head = None
        try:
            weights, peft = self.engine.get_per_tensor_param()
            if peft is not None:
                raise ValueError('full 92B publication does not support a partial PEFT state')

            def wire():
                nonlocal head
                with torch.inference_mode():
                    for name, value in weights:
                        value = value.detach().to(wire_dtype(name, value)).contiguous()
                        if digest is not None:
                            digest.update(name, value)
                            if name == 'lm_head.weight':
                                from safetensors.torch import save_file
                                from verl_speco.runtime.artifacts import publish_file
                                directory = pending['output'] / operation_id
                                directory.mkdir(parents=True, exist_ok=True)
                                destination = directory / 'lm_head.safetensors'
                                if destination.exists():
                                    raise RuntimeError('immutable actor target head already exists')
                                with tempfile.NamedTemporaryFile(dir='/tmp', prefix='speco-actor-head-', delete=False) as handle:
                                    temporary = Path(handle.name)
                                try:
                                    save_file({'lm_head.weight': value.cpu()}, str(temporary))
                                    publish_file(temporary, destination)
                                finally:
                                    temporary.unlink(missing_ok=True)
                                with destination.open('rb') as handle:
                                    sha = hashlib.file_digest(handle, 'sha256').hexdigest()
                                head = dict(uri=str(destination), sha256=sha)
                        yield (name, value)
            asyncio.run(pending['engine'].send_weights(wire(), global_steps=pending['manifest']['rl_step']))
            receipt = digest.finish() if digest is not None else None
            if digest is not None and head is None:
                raise RuntimeError('complete actor target wire omitted the actual LM head')
            if head is not None:
                head.update(run_id=pending['manifest']['run_id'], generation=pending['manifest']['generation'], target_revision=pending['manifest']['target_revision'], tokenizer_fingerprint=pending['request']['tokenizer_fingerprint'], actor_weight_manifest_sha256=receipt['actor_weight_manifest_sha256'])
            pending.update(phase='sent', receipt=receipt)
            return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, sent=True, wire_receipt=receipt, target_head_binding=head)
        except BaseException:
            pending['phase'] = 'failed'
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_finish_target_transfer(self, operation_id, actor_weight_manifest_sha256):
        pending = self._speco_target_transfer
        if pending['manifest']['operation_id'] != operation_id or pending['phase'] != 'sent' or (pending['receipt'] is not None and pending['receipt']['actor_weight_manifest_sha256'] != actor_weight_manifest_sha256):
            raise RuntimeError('actor target transfer finalization lacks matching full-stream receipt')
        pending['engine'].finalize()
        self._speco_target_transfer = None
        return dict(rank=self.rank, released=True, worker_incarnation=self.worker_incarnation)
import json
from verl_speco.trainer.distributed_checkpoint import CandidateCheckpoint
from verl_speco.trainer.distributed_checkpoint import MultiNodeActorDCPBridge as MultiNodeActorDCPBridge, actor_state_sha256

class MultiNode92BCheckpointActorWorker(MultiNode92BPublishingActorWorker):

    def _checkpoint_bridge(self):
        return MultiNodeActorDCPBridge.collective_create(self._checkpoint_settings)

    def _checkpoint_settings(self):
        config = self.speco_config
        logical = dict(run_id=config['run_id'], model_path=str(Path(config['model_path']).resolve()), model_config_sha256=hashlib.sha256((Path(config['model_path']) / 'config.json').read_bytes()).hexdigest(), world_size=self.world_size, fsdp_full_shard=True, expert_wrap='OpenPanguV2MLP', EP=1, SP=1, learning_rate=config.get('learning_rate', 1e-06), total_steps=config.get('total_steps', 2))
        fingerprint = hashlib.sha256(json.dumps(logical, sort_keys=True).encode()).hexdigest()
        topology = hashlib.sha256(json.dumps(dict(logical, generation=config['generation']), sort_keys=True).encode()).hexdigest()
        return dict(module=self.engine.module, optimizer=self.engine.optimizer, scheduler=self.engine.lr_scheduler, topology_fingerprint=topology, model_fingerprint=fingerprint, logical_checkpoint_fingerprint=fingerprint, shared_storage=True, file_hash_workers=4)

    def _require_checkpoint_owner(self, request):
        config = self.speco_config
        if request.get('run_id') != config['run_id'] or request.get('generation') != config['generation'] or request.get('actor_incarnations', {}).get(str(self.rank)) != self.worker_incarnation or getattr(self, '_speco_checkpoint_failed', False) or (getattr(self, '_speco_target_transfer', None) is not None):
            raise ValueError('actor checkpoint requires the current unfenced owner and idle target port')
        configured = config.get('actor_checkpoint_root')
        if not configured:
            raise ValueError('actor checkpoint output root must be configured explicitly')
        root = Path(configured)
        output_root = Path(__file__).resolve().parents[2] / 'outputs'
        if not root.is_absolute() or not root.resolve().is_relative_to(output_root):
            raise ValueError('actor checkpoint must remain in owned SpeCo outputs')
        return root

    def _complete_state(self):
        return actor_state_sha256(module=self.engine.module, optimizer=self.engine.optimizer, scheduler=self.engine.lr_scheduler, include_components=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_actor_checkpoint_audit(self):
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, run_id=self.speco_config['run_id'], generation=self.speco_config['generation'], **self._complete_state())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_save_actor_checkpoint(self, request):
        bridge = self._checkpoint_bridge()
        error = None
        root = None
        try:
            root = self._require_checkpoint_owner(request)
            step = request.get('optimizer_step')
            if type(step) is not int or step < 1 or self.speco_actor_audit()['optimizer_step'] != step:
                raise ValueError('actor checkpoint does not match the actual optimizer step')
        except Exception as exc:
            error = exc
        bridge._require_all_ranks_success(error, 'actor checkpoint owner rejected by another rank')
        try:
            candidate = bridge.save_candidate(root / f"generation-{request['generation']}-step-{step}", generation=request['generation'])
            return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, run_id=request['run_id'], generation=request['generation'], optimizer_step=step, candidate=asdict(candidate), saved=True, **self._complete_state())
        except BaseException:
            self._speco_checkpoint_failed = True
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_restore_actor_checkpoint(self, request):
        bridge = self._checkpoint_bridge()
        error = None
        candidate = None
        try:
            root = self._require_checkpoint_owner(request)
            candidate = CandidateCheckpoint(**request['candidate'])
            if not Path(candidate.uri).resolve().is_relative_to(root.resolve()):
                raise ValueError('actor checkpoint candidate is outside the configured owner root')
            if self.speco_actor_audit()['optimizer_step'] != 0 or candidate.checkpoint_generation > request['generation'] or type(request.get('optimizer_step')) is not int or (request['optimizer_step'] < 1):
                raise ValueError('actor restore requires a fresh optimizer owner and a committed parent')
        except Exception as exc:
            error = exc
        bridge._require_all_ranks_success(error, 'actor restore owner rejected by another rank')
        try:
            bridge.restore_candidate(candidate, handoff_generation=request['generation'] if request['generation'] > candidate.checkpoint_generation else None)
            if self.speco_actor_audit()['optimizer_step'] != request['optimizer_step']:
                raise RuntimeError('restored actor Adam step differs from the committed checkpoint')
            return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, run_id=request['run_id'], generation=request['generation'], optimizer_step=request['optimizer_step'], candidate=asdict(candidate), restored=True, verification=bridge.last_verification, **self._complete_state())
        except BaseException:
            self._speco_checkpoint_failed = True
            raise
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl_speco.trainer.distributed_checkpoint import MultiNodeActorDCPBridge as MultiNodeActorDCPBridge

class MultiNode92BFencedCheckpointActorWorker(MultiNode92BCheckpointActorWorker):

    def _require_healthy_checkpoint_owner(self, *, collective=False, compute=False):
        error = None
        if getattr(self, '_speco_checkpoint_failed', False):
            error = RuntimeError('uncertain actor checkpoint requires a fresh optimizer owner')
        elif compute and getattr(self, '_speco_target_transfer', None) is not None:
            error = RuntimeError('actor compute cannot overlap an unfinished target transfer')
        if collective:
            gate = MultiNodeActorDCPBridge.__new__(MultiNodeActorDCPBridge)
            gate.checkpoint_process_group = None
            gate._require_all_ranks_success(error, 'actor compute fenced by another checkpoint owner')
        elif error is not None:
            raise error

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_actor_audit(self):
        self._require_healthy_checkpoint_owner()
        return super().speco_actor_audit()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='train'), blocking=False)
    def infer_batch(self, data):
        self._require_healthy_checkpoint_owner(collective=True, compute=True)
        return super().infer_batch(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='train'), blocking=False)
    def train_mini_batch(self, data):
        self._require_healthy_checkpoint_owner(collective=True, compute=True)
        return super().train_mini_batch(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_actor_checkpoint_audit(self):
        self._require_healthy_checkpoint_owner(collective=True)
        result = super().speco_actor_checkpoint_audit()
        result['optimizer_step'] = self.speco_actor_audit()['optimizer_step']
        return result
from verl_speco.integration.multinode_publication import MultiNodeCheckpointChannel as MultiNodeCheckpointChannel
from verl_speco.integration.multinode_publication import manifest_for_state

class MultiNode92BPreconnectedActorWorker(MultiNode92BPublishingActorWorker):

    def _require_ready_checkpoint_channel(self):
        from verl_speco.trainer.distributed_checkpoint import MultiNodeActorDCPBridge as MultiNodeActorDCPBridge
        channel = getattr(self, '_speco_checkpoint_channel', None)
        error = None
        if channel is None or channel.phase != 'ready' or getattr(self, '_speco_target_transfer', None) is not None:
            error = RuntimeError('actor compute requires its connected idle generation channel')
        gate = MultiNodeActorDCPBridge.__new__(MultiNodeActorDCPBridge)
        gate.checkpoint_process_group = None
        gate._require_all_ranks_success(error, 'actor peer checkpoint channel is not ready')

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='train'), blocking=False)
    def infer_batch(self, data):
        self._require_ready_checkpoint_channel()
        return super().infer_batch(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='train'), blocking=False)
    def train_mini_batch(self, data):
        self._require_ready_checkpoint_channel()
        return super().train_mini_batch(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_prepare_checkpoint_channel(self, request):
        if getattr(self, '_speco_checkpoint_channel', None) is not None or getattr(self, '_speco_target_transfer', None) is not None or request['run_id'] != self.speco_config['run_id'] or (request['generation'] != self.speco_config['generation']) or (len(request['actors']) != self.world_size) or (self.speco_actor_audit()['optimizer_step'] != 0):
            raise ValueError('actor preconnection requires its fresh pre-PPO generation owner')
        channel = MultiNodeCheckpointChannel(identity=request, role='actors', rank=self.rank, incarnation=self.worker_incarnation)
        self._speco_checkpoint_channel = channel
        metadata = channel.prepare(request)
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, prepared=True, master_metadata=metadata)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_connect_checkpoint_channel(self, request):
        channel = self._speco_checkpoint_channel
        channel.connect(request['identity'], request['master_metadata'])
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, connected=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_prepare_target_transfer(self, request):
        from pathlib import Path
        from verl.utils.model import convert_weight_keys
        channel = self._speco_checkpoint_channel
        audit = self.speco_actor_audit()
        owners = {str(rank): owner for rank, owner in channel.identity['actors'].items()}
        if getattr(self, '_speco_target_transfer', None) is not None or request['run_id'] != channel.identity['run_id'] or request['generation'] != channel.identity['generation'] or (request['actor_incarnations'] != owners) or any((type(request[key]) is not int for key in ('generation', 'optimizer_step', 'rl_step', 'target_revision'))) or (request['optimizer_step'] != audit['optimizer_step']) or (audit['optimizer_step'] < 1) or (type(request['rl_step']) is not int) or (request['rl_step'] < 1) or (type(request['target_revision']) is not int) or (request['target_revision'] < 1):
            raise ValueError('preconnected target requires the actual committed actor update')
        state = convert_weight_keys(self.engine.module.state_dict(), getattr(self.engine.module, '_fsdp_wrapped_module', self.engine.module))
        manifest = manifest_for_state(state, run_id=request['run_id'], generation=request['generation'], rl_step=request['rl_step'], target_revision=request['target_revision'], operation_id=request['operation_id'], actor_incarnations=owners)
        size = request['bucket_bytes']
        if len(state) != 35869 or audit['global_parameter_elements'] != 91955618233 or type(size) is not int or (size != channel.identity['bucket_bytes']) or (size < max((row['nbytes'] for row in manifest['tensors']))):
            raise ValueError('preconnected target requires the entire audited state and bounded bucket')
        output = Path(self.speco_config['target_transfer_output_root'])
        if not output.is_absolute():
            raise ValueError('preconnected target head requires its absolute owned output')
        engine = channel.begin(channel.identity, request['operation_id'])
        self._speco_target_transfer = dict(manifest=manifest, engine=engine, request=dict(request), output=output, phase='prepared', receipt=None)
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, header_sha256=manifest['header_sha256'], prepared=True, manifest=manifest if self.rank == 0 else None, master_metadata=channel.master_metadata if self.rank == 0 else None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_init_target_transfer(self, request):
        channel = self._speco_checkpoint_channel
        pending = self._speco_target_transfer
        if channel.phase != 'active' or pending['phase'] != 'prepared' or request['operation_id'] != channel.operation_id or (request['master_metadata'] != channel.master_metadata) or (request['rollout_world_size'] != len(channel.identity['rollout'])):
            raise RuntimeError('preconnected target topology differs from its owned generation')
        pending['phase'] = 'connected'
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, connected=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_finish_target_transfer(self, operation_id, actor_weight_manifest_sha256):
        pending = self._speco_target_transfer
        if pending['manifest']['operation_id'] != operation_id or pending['phase'] != 'sent' or (pending['receipt'] is not None and pending['receipt']['actor_weight_manifest_sha256'] != actor_weight_manifest_sha256):
            raise RuntimeError('preconnected source release lacks its exact full wire receipt')
        channel = self._speco_checkpoint_channel
        channel.finish(channel.identity, operation_id, whole_revision_acknowledged=True)
        self._speco_target_transfer = None
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, released=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_close_checkpoint_channel(self, request):
        channel = self._speco_checkpoint_channel
        if getattr(self, '_speco_target_transfer', None) is not None:
            raise RuntimeError('preconnected source close has an unfinished publication')
        channel.close(request)
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, closed=True)
from copy import deepcopy
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica, _RESET_PREFIX_CACHE_KWARGS
from verl_speco.integration.multinode_recovery import drain_owned_call
from verl_speco.integration.multinode_recovery import ColocatedNativeGenerationMixin
from verl_speco.integration.multinode_recovery import ColocatedServerClass

class MultiNodeColocatedNativeServer(ColocatedNativeGenerationMixin, vLLMHttpServer):

    async def run_server(self, args):
        import inspect
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.usage.usage_lib import UsageContext
        from vllm.entrypoints.openai.api_server import build_app, init_app_state
        from verl.workers.rollout.utils import run_uvicorn
        from verl_speco.integration.multinode_recovery import dedicated_colocated_dspark_config
        from verl_speco.integration.accelerator_speculation import bind_speco_sink_reservation as bind_speco_sink_reservation
        kwargs = dict(self.config.engine_kwargs.get('vllm', {}), model=self.model_config.local_path)
        with dedicated_colocated_dspark_config(kwargs):
            engine_args = AsyncEngineArgs.from_cli_args(args)
            from verl_speco.integration.accelerator_speculation import bind_native_cache_policy, verify_native_cache_policy
            cache_policy = bind_native_cache_policy(engine_args, rollout=self.config, engine_kwargs=kwargs)
            usage = UsageContext.OPENAI_API_SERVER
            native = engine_args.create_engine_config(usage_context=usage)
            self._speco_native_cache_policy = verify_native_cache_policy(native, cache_policy)
            self._speco_native_sink_reservation = bind_speco_sink_reservation(native, os.environ)
            from verl_speco.integration.accelerator_speculation import native_prefix_reset_kwargs
            self._speco_native_prefix_reset_kwargs = native_prefix_reset_kwargs(native, _RESET_PREFIX_CACHE_KWARGS)
            native.parallel_config.data_parallel_master_port = self._dp_master_port
            accepted = set(inspect.signature(AsyncLLM.from_vllm_config).parameters)
            extra = {k: getattr(engine_args, k) for k in ('enable_log_requests', 'disable_log_stats') if k in accepted}
            from verl_speco.integration.multinode_online import ServingAcceptanceCounters
            self._speco_acceptance_counters = ServingAcceptanceCounters()

            def acceptance_logger_factory(vllm_config, engine_index=0):
                if engine_index != 0:
                    raise ValueError('shared acceptance logger requires one actual TP8 engine')
                return self._speco_acceptance_counters
            if 'stat_loggers' not in accepted:
                raise RuntimeError('installed vLLM lacks per-engine acceptance statistics')
            extra['stat_loggers'] = [acceptance_logger_factory]
            self.engine = AsyncLLM.from_vllm_config(vllm_config=native, usage_context=usage, **extra)
        await self.engine.reset_mm_cache()
        await self.engine.collective_rpc(method='monkey_patch_model', kwargs={'vocab_size': len(self.model_config.tokenizer)})
        await self._fence_cold_restore()
        supported = ()
        app_kwargs = {}
        app_args = set(inspect.signature(build_app).parameters)
        if 'supported_tasks' in app_args:
            supported = await self.engine.get_supported_tasks()
            app_kwargs['supported_tasks'] = supported
        if 'model_config' in app_args:
            app_kwargs['model_config'] = self.engine.model_config
        app = build_app(args, **app_kwargs)
        init_args = set(inspect.signature(init_app_state).parameters)
        if 'vllm_config' in init_args:
            await init_app_state(self.engine, native, app.state, args)
        elif 'supported_tasks' in init_args:
            await init_app_state(self.engine, app.state, args, supported)
        else:
            await init_app_state(self.engine, app.state, args)
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def _fence_cold_restore(self):
        if os.environ.get('SPECO_COLOCATED_COLD_RESTORE') == '1':
            await drain_owned_call(self.engine.pause_generation, mode='wait', clear_cache=True)
            if await self.engine.is_paused() is not True:
                raise RuntimeError('cold joint native server did not fence generation before API startup')

    async def speco_native_bootstrap_status(self):
        rows = await drain_owned_call(self.collective_rpc, 'speco_colocated_native_placement')
        return dict(native_placements=rows, sink_reservation=deepcopy(self._speco_native_sink_reservation), actual_VeRL_native_async_server=True, cache_policy=deepcopy(self._speco_native_cache_policy))

    def _owned_request(self, request, kind):
        required = {'run_id', 'generation', 'transition_epoch', 'placement_binding_sha256', kind}
        if kind == 'tags':
            required.add('resume_generation')
        if type(request) is not dict or set(request) != required or request['run_id'] != os.environ.get('SPECO_RUN_ID') or (type(request['generation']) is not int) or (str(request['generation']) != os.environ.get('SPECO_GENERATION')) or (type(request['transition_epoch']) is not int) or (request['transition_epoch'] < 1) or (type(request['placement_binding_sha256']) is not str) or (len(request['placement_binding_sha256']) != 64) or any((c not in '0123456789abcdef' for c in request['placement_binding_sha256'])) or (self.rollout_mode.value not in ('hybrid', 'colocated')) or (self.config.free_cache_engine is not True):
            raise ValueError('SpeCo colocated server received foreign/invalid ownership')
        binding = getattr(self, '_speco_colocated_binding', None)
        if binding is not None and binding != request['placement_binding_sha256']:
            raise ValueError('SpeCo colocated server placement binding changed')
        if getattr(self, '_speco_colocated_failed', False):
            raise RuntimeError('uncertain colocated server requires owned recovery')
        prior = getattr(self, '_speco_colocated_request', None)
        if prior is not None and (request['transition_epoch'] < prior['transition_epoch'] or (request['transition_epoch'] == prior['transition_epoch'] and request != prior)):
            raise ValueError('colocated server received a stale or reused transition epoch')

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return await self.engine.collective_rpc(method=method, timeout=timeout, args=args, kwargs=kwargs)

    async def speco_prepare_colocated_sleep(self, request):
        self._owned_request(request, 'sleep_level')
        if type(request['sleep_level']) is not int or request['sleep_level'] != 1:
            raise ValueError('Ascend colocated sleep must preserve target/draft weight backups')
        if self.node_rank != 0:
            return dict(request, complete=True, coordinator_rank_skipped=True)
        self._speco_colocated_request = deepcopy(request)
        self._speco_colocated_awake_witness = None
        try:
            await drain_owned_call(self.engine.pause_generation, mode='wait', clear_cache=True)
            if await self.engine.is_paused() is not True:
                raise RuntimeError('colocated generation was not actually paused after drain')
            self._speco_colocated_binding = request['placement_binding_sha256']
            self._speco_colocated_phase = 'prepared_sleep'
            return dict(request, complete=True)
        except BaseException:
            self._speco_colocated_failed = True
            raise

    async def speco_wake_colocated_tags(self, request):
        self._owned_request(request, 'tags')
        tags = request['tags']
        if tags not in (['weights'], ['kv_cache'], ['weights', 'kv_cache']):
            raise ValueError('SpeCo requires explicit tagged colocated wake')
        if type(request['resume_generation']) is not bool:
            raise ValueError('SpeCo wake requires an explicit generation resume fence')
        if request['resume_generation'] and 'kv_cache' not in tags:
            raise ValueError('SpeCo generation cannot resume before KV is awake')
        if self.node_rank != 0:
            return dict(request, complete=True, coordinator_rank_skipped=True)
        self._speco_colocated_request = deepcopy(request)
        try:
            if tags == ['kv_cache'] and getattr(self, '_speco_colocated_phase', None) not in ('weights_awake', 'publication'):
                raise RuntimeError('KV wake requires prior weights-only publication boundary')
            if tags == ['weights', 'kv_cache'] and request['resume_generation'] is False:
                await drain_owned_call(self.engine.wake_up, tags=['weights'])
                await drain_owned_call(self.engine.wake_up, tags=['kv_cache'])
            else:
                await drain_owned_call(self.engine.wake_up, tags=tags)
            if await drain_owned_call(self.engine.reset_prefix_cache, **self._speco_native_prefix_reset_kwargs) is not True:
                raise RuntimeError('native scheduler refused prefix reset after tagged wake')
            self._speco_colocated_awake_witness = None
            if 'kv_cache' in tags:
                if await self.engine.is_sleeping() is not False:
                    raise RuntimeError('colocated native engine did not complete its full tagged wake')
                self._speco_colocated_awake_witness = deepcopy(request)
            if request['resume_generation'] is False:
                await drain_owned_call(self.engine.pause_generation, mode='wait', clear_cache=False)
                if await self.engine.is_paused() is not True:
                    raise RuntimeError('weights-only wake accidentally admitted new generation')
                self._speco_colocated_phase = 'weights_awake' if tags == ['weights'] else 'publication'
            else:
                await drain_owned_call(self.engine.resume_generation)
                if await self.engine.is_paused() is not False:
                    raise RuntimeError('colocated generation did not actually resume')
                self._speco_colocated_phase = 'rollout'
            self._speco_colocated_binding = request['placement_binding_sha256']
            return dict(request, complete=True)
        except BaseException:
            self._speco_colocated_failed = True
            raise

    async def speco_colocated_phase_status(self, request, phase, owner_incarnations):
        kind = 'sleep_level' if phase == 'sleeping' else 'tags'
        self._owned_request(request, kind)
        if self.node_rank != 0:
            return dict(request, complete=True, coordinator_rank_skipped=True)
        if type(owner_incarnations) is not dict or not owner_incarnations or any((type(rank) is not int or type(owner) is not str or (not owner) for rank, owner in owner_incarnations.items())):
            raise ValueError('colocated phase audit requires independent exact native owners')
        rows = await drain_owned_call(self.collective_rpc, 'speco_dspark_runtime_status')
        if type(rows) is not list or len(rows) != len(owner_incarnations) or any((type(row) is not dict or type(row.get('rank')) is not int for row in rows)) or ({row['rank']: row.get('worker_incarnation') for row in rows} != owner_incarnations):
            raise RuntimeError('colocated server native membership changed during memory handoff')
        paused = await self.engine.is_paused()
        sleeping = await self.engine.is_sleeping()
        if type(paused) is not bool or type(sleeping) is not bool:
            raise RuntimeError('colocated native engine has unknown physical sleep/pause status')
        if phase == 'sleeping':
            complete = sleeping and paused
        elif phase == 'awake:weights:paused':
            complete = paused and self._speco_colocated_phase == 'weights_awake'
        elif phase == 'awake:weights,kv_cache:paused':
            complete = paused and self._speco_colocated_phase == 'publication' and (getattr(self, '_speco_colocated_awake_witness', None) == request) and (self._speco_colocated_request == request)
        elif phase in ('awake:kv_cache', 'awake:weights,kv_cache'):
            complete = not sleeping and (not paused) and (self._speco_colocated_phase == 'rollout')
        else:
            raise ValueError('unrecognized colocated physical phase audit')
        return dict(request, complete=complete)

class MultiNodeColocatedNativeReplica(vLLMReplica):

    def __init__(self, *args, placement_binding, server_python, base_env, record_owned=lambda *_: None, **kwargs):
        super().__init__(*args, **kwargs)
        import ray
        self._speco_placement_binding = deepcopy(placement_binding)
        self.server_class = ColocatedServerClass(ray.remote(MultiNodeColocatedNativeServer), binding=placement_binding, replica_rank=self.replica_rank, world_size=self.world_size, gpus_per_node=self.gpus_per_replica_node, python=server_python, base_env=base_env, record_owned=record_owned)

    async def launch_servers(self):
        if self.rollout_mode.value != 'hybrid' or len(self.workers) != self.world_size:
            raise ValueError('shared native replica must reuse the initialized actor handles')
        first = self.replica_rank * self.world_size
        expected = self._speco_placement_binding['actor'][first:first + self.world_size]

        async def read():
            return await asyncio.gather(*[w.speco_colocated_placement.remote() for w in self.workers])
        rows = await drain_owned_call(read)
        if rows != expected:
            raise RuntimeError('actual current actor physical pairing differs before native launch')
        await super().launch_servers()
        receipt = await drain_owned_call(self.servers[0].speco_native_bootstrap_status.remote)
        native = receipt.get('native_placements')
        by_rank = {i: r for i, r in enumerate(expected)}
        if type(native) is not list or len(native) != len(expected) or any((type(r) is not dict or type(r.get('native_rank')) is not int for r in native)) or ({r['native_rank'] for r in native} != set(by_rank)) or (len({r.get('worker_incarnation') for r in native}) != len(expected)) or any((not r.get('worker_incarnation') or any((r.get(k) != by_rank[r['native_rank']][k] for k in ('node_ip', 'device_uuid', 'placement_group_id', 'bundle_index'))) or r.get('actor_rank') != by_rank[r['native_rank']]['rank'] or (r.get('actor_incarnation') != by_rank[r['native_rank']]['worker_incarnation']) for r in native)):
            raise RuntimeError('native MP workers did not actually reuse the reserved physical actor cards')
        self._speco_native_bootstrap_receipt = deepcopy(receipt)

    async def verify_speco_phase(self, phase, request, *, owner_incarnations):
        results = []
        for server in self.servers:
            results.append(await drain_owned_call(server.speco_colocated_phase_status.remote, request=request, phase=phase, owner_incarnations=deepcopy(owner_incarnations)))
        return bool(results) and all((result.get('complete') is True for result in results))
from verl_speco.integration.multinode_recovery import open_joint_parent, validate_joint_owners, validate_joint_release

class MultiNodeJointActorWorker(MultiNode92BPreconnectedActorWorker, MultiNode92BFencedCheckpointActorWorker):

    def __init__(self, config):
        super().__init__(config)
        self._joint_parent = None
        self._joint_owners = None
        self._joint_restored = False
        self._joint_phase = 'restoring' if config.get('joint_restore_binding') else 'ready'

    def _joint_collective(self, operation):
        error = None
        result = None
        try:
            result = operation()
        except Exception as exc:
            error = exc
        gate = MultiNodeActorDCPBridge.__new__(MultiNodeActorDCPBridge)
        gate.checkpoint_process_group = None
        try:
            gate._require_all_ranks_success(error, 'joint actor operation rejected by another owner')
        except BaseException:
            self._joint_phase = 'failed'
            raise
        return result

    def _require_ready_checkpoint_channel(self):
        error = RuntimeError('joint actor compute remains fenced until all roles restore') if self._joint_phase != 'ready' else None
        gate = MultiNodeActorDCPBridge.__new__(MultiNodeActorDCPBridge)
        gate.checkpoint_process_group = None
        gate._require_all_ranks_success(error, 'joint actor peer restore is not released')
        super()._require_ready_checkpoint_channel()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_restore_joint_actor(self, request):

        def validate():
            config = self.speco_config
            owners = request['owners']
            if self._joint_phase != 'restoring' or self._joint_restored or request.get('run_id') != config['run_id'] or (type(request.get('generation')) is not int) or (request['generation'] != config['generation']) or (owners['actor'].get(self.rank) != self.worker_incarnation) or (set(owners['actor']) != set(range(self.world_size))) or (self._speco_checkpoint_channel.phase != 'ready') or (self._speco_checkpoint_channel.identity['actors'] != owners['actor']) or (self._speco_checkpoint_channel.identity['rollout'] != owners['rollout']) or (self.speco_actor_audit()['optimizer_step'] != 0):
                raise ValueError('joint actor restore requires the fresh preconnected fenced owner')
            parent, _ = open_joint_parent(config, request['binding'])
            validate_joint_owners(parent, owners)
            prior = next((row for row in parent['actor_rows'] if row['rank'] == self.rank))
            return (parent, prior, deepcopy(owners))
        parent, prior, owners = self._joint_collective(validate)
        try:
            restored = self.speco_restore_actor_checkpoint(dict(run_id=self.speco_config['run_id'], generation=self.speco_config['generation'], actor_incarnations={str(rank): owner for rank, owner in owners['actor'].items()}, candidate=prior['candidate'], optimizer_step=prior['optimizer_step']))

            def verify():
                nonlocal restored
                from verl_speco.trainer.distributed_checkpoint import recover_empty_adam_representation
                restored, empty_entries_recovered = recover_empty_adam_representation(self.engine.optimizer, observed_state=restored, expected_sha256=prior['state_sha256'], measure_state=self._complete_state)
                from verl_speco.runtime.artifacts import publish_json
                import json
                fields = ('state_sha256', 'tensor_count', 'local_tensor_nbytes', 'complete_local_model_optimizer_scheduler_rng', 'state_components')
                comparison = dict(run_id=request['run_id'], generation=request['generation'], rank=self.rank, worker_incarnation=self.worker_incarnation, binding_checksum=request['binding']['checksum'], expected={key: prior[key] for key in fields if key in prior}, actual={key: restored[key] for key in fields if key in restored}, payload_verification=restored.get('verification'), compute_released=False, empty_adam_entries_recovered=empty_entries_recovered)
                root = self._require_checkpoint_owner(dict(request, actor_incarnations={str(rank): owner for rank, owner in owners['actor'].items()}))
                directory = root / 'restore-diagnostics' / f"generation-{request['generation']}"
                directory.mkdir(parents=True, exist_ok=True)
                publish_json(directory / f'rank-{self.rank:04d}.json', comparison)
                if restored['state_sha256'] != prior['state_sha256'] or restored.get('complete_local_model_optimizer_scheduler_rng') is not True:
                    raise RuntimeError('joint actor restored complete state differs from its committed parent: ' + json.dumps(comparison, sort_keys=True))
            self._joint_collective(verify)
            self._joint_parent, self._joint_owners = (parent, owners)
            self._joint_restored = True
            return dict(restored, binding_checksum=request['binding']['checksum'], compute_released=False)
        except BaseException:
            self._joint_phase = 'failed'
            self._speco_checkpoint_failed = True
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_release_joint_actor(self, request):

        def validate():
            if self._joint_phase != 'restoring' or not self._joint_restored:
                raise RuntimeError('joint actor release requires its verified fenced parent')
            validate_joint_release(self._joint_parent, request['proof'], self._joint_owners, run_id=self.speco_config['run_id'], generation=self.speco_config['generation'], binding_checksum=self.speco_config['joint_restore_binding']['checksum'])
        self._joint_collective(validate)
        self._joint_phase = 'ready'
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation, run_id=self.speco_config['run_id'], generation=self.speco_config['generation'], compute_released=True, optimizer_step=self.speco_actor_audit()['optimizer_step'])
import socket
import time
from uuid import uuid4
from verl.single_controller.base import Worker
from verl_speco.integration.multinode_online import OnlineWorkerPlanFence

class MultiNodeOnlineDraftWorker(Worker):

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.incarnation = uuid4().hex
        self.initialized = False
        self.prepared = None
        self.optimizer_step = 0
        self.snapshot = None
        self.committed_receipt = None
        self.target_heads = {}
        self.fence = OnlineWorkerPlanFence(run_id=self.config['run_id'], generation=self.config['generation'], worker_id=str(self.rank), incarnation=self.incarnation)

    def _receipt(self, **fields):
        return dict(worker_id=str(self.rank), worker_incarnation=self.incarnation, run_id=self.config['run_id'], generation=self.config['generation'], online_plan_sha256=(self.fence.payload or {}).get('online_plan_sha256'), **fields)

    def _collective_call(self, function):
        """Propagate local validation errors before the next tensor collective."""
        import torch
        import torch.distributed as dist
        error = None
        value = None
        try:
            value = function()
        except Exception as exc:
            error = exc
            import traceback
            traceback.print_exc()
        flag = torch.tensor(int(error is None), dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            self.fence.phase = 'aborted'
            if error is not None:
                raise error
            raise RuntimeError('online draft peer rejected the collective operation')
        return value

    def _allow_online_colocation(self):
        return False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import torch
        import torch_npu
        import torch.distributed as dist
        import ray
        from verl.trainer.config import CheckpointConfig
        from verl.workers.config import VeOmniEngineConfig, VeOmniOptimizerConfig
        from verl_speco.backends.dspark_multinode_backend import MultiNodeDSparkTrainingModel as DSparkTrainingModel
        from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
        from verl_speco.integration.multinode_publication import read_draft_update
        from verl_speco.trainer.distributed_checkpoint import load_trainable_snapshot
        from verl_speco.trainer.veomni_dspark_adapter import build_veomni_dspark_engine_class, DSparkVeOmniModelConfig
        from verl_speco.trainer.feature_pipeline import load_target_lm_head
        from verl_speco.trainer.distributed_checkpoint import VeOmniDCPBridge
        from verl_speco.transport.native_transfer import AscendFeatureTransport
        from verl_speco.transport.native_transfer import NativeDescriptorStore
        from verl_speco.transport.streaming import TrainingStreamCoordinator
        from verl_speco.runtime.services import import_transfer_queue
        c = self.config
        if self.initialized or (socket.gethostname() in c['rollout_nodes'] and (not self._allow_online_colocation())):
            raise RuntimeError('online draft initialization requires a separate uninitialized trainer node')
        self.local_device_rank = self._local_rank
        self.device = f'npu:{self.local_device_rank}'
        torch.npu.set_device(self.local_device_rank)
        if not dist.is_initialized():
            dist.init_process_group('hccl', rank=self.rank, world_size=self.world_size)
        if dist.get_world_size() != self.world_size or dist.get_rank() != self.rank or dist.get_backend() != 'hccl':
            raise RuntimeError('online draft process group does not match the assigned WorkerGroup')
        snapshot, manifest, _ = read_draft_update(c['initial_snapshot'], target_fingerprint=c['target_model_fingerprint'], tokenizer_fingerprint=c['tokenizer_fingerprint'], model_config=c['model_config'])
        cfg = DSparkConfig(**manifest['model_config'])
        if cfg.draft_response_plan != 'dmtp_swa_v1' or cfg.draft_loss_reduction != 'part_mean' or cfg.sliding_window != 2048 or (cfg.block_size != 8):
            raise ValueError('online DSpark checkpoint requires the verified DMTP SWA block8 contract')

        def factory():
            return DSparkTrainingModel(draft_model=DSparkDraftModel(cfg), block_size=cfg.block_size, num_anchors=cfg.num_anchors, loss_decay_gamma=cfg.loss_decay_gamma, ce_loss_alpha=cfg.ce_loss_alpha, l1_loss_alpha=cfg.l1_loss_alpha, confidence_head_alpha=cfg.confidence_head_alpha, l1_chunk_size=512, distribution_loss_impl='eager', anchor_sampling_policy='dmtp_cpu')
        self.engine = build_veomni_dspark_engine_class()(model_factory=factory, model_config=DSparkVeOmniModelConfig(), engine_config=VeOmniEngineConfig(init_device='meta', fsdp_size=1, ulysses_parallel_size=1, expert_parallel_size=1, use_torch_compile=False, enable_full_shard=True, mixed_precision=True, activation_gpu_limit=0.0), optimizer_config=VeOmniOptimizerConfig(lr=c.get('lr', 0.0001), total_training_steps=c['total_steps']), checkpoint_config=CheckpointConfig())
        self.engine.initialize()
        from veomni.distributed import parallel_state
        ps = parallel_state.get_parallel_state()
        if ps.dp_replicate_size != self.world_size or ps.dp_shard_size != 1 or ps.ulysses_size != 1:
            raise RuntimeError('actual VeOmni mesh is not DP-only with FSDP2')
        load_trainable_snapshot(self.engine.module, snapshot, target_model_fingerprint=c['target_model_fingerprint'], tokenizer_fingerprint=c['tokenizer_fingerprint'], model_config=self.engine.module.config.to_dict(), namespace_prefix='training_model.')
        self.head, self.head_location = load_target_lm_head(c['target_model_dir'], device=self.device)
        self.target_revision = c['initial_target_revision']
        self.draft_revision = snapshot.draft_revision
        if not ray.is_initialized():
            ray.init(address=c['ray_address'], namespace='transfer_queue')
        self.control = ray.get_actor(c['control_actor_name'], namespace='transfer_queue')
        self.tq = import_transfer_queue()
        self.tq.init()
        endpoint = c['native_endpoint_template'].format(rank=self.rank, local_rank=self.local_device_rank)
        self.transport = AscendFeatureTransport(run_id=c['run_id'], generation=c['generation'], endpoint=endpoint, metadata_server=c['metadata_server'], device=self.device, max_bytes=c.get('native_max_bytes', 512 * 1024 ** 2), **dict(local_host_mirror=True) if os.environ.get('SPECO_COLOCATED_FEATURE_HOST_MIRROR') == '1' else {})
        self.reader = NativeDescriptorStore(tq=self.tq, transport=self.transport, receive_capacity_bytes=self.transport.max_bytes)
        self.root = Path(c['output_root'])
        if not self.root.is_absolute():
            raise ValueError('online durable output path must be absolute')
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ('journal', 'dcp', 'snapshots', 'receipts'):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.stream = TrainingStreamCoordinator(c['run_id'])
        self.bridge = VeOmniDCPBridge(module=self.engine.module, optimizer=self.engine.optimizer, scheduler=self.engine.lr_scheduler, topology_fingerprint=f'online-dp{self.world_size}-shard1', model_fingerprint=hashlib.sha256(json.dumps(c['model_config'], sort_keys=True).encode()).hexdigest(), shared_storage=True)
        self.initialized = True
        return self.speco_online_capabilities()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_online_capabilities(self):
        return self._receipt(engine='veomni', wrapper='fsdp2', initialized=self.initialized, online_fencing=True, hidden_transport='native_tq', node_id=socket.gethostname(), mesh=dict(dp_replicate=self.world_size, dp_shard=1, tp=1, pp=1, sp=1, ep=1), optimizer_step=self.optimizer_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_bind_online_plan(self, payload):
        if not self.initialized:
            raise RuntimeError('online draft engine is not initialized')
        self._collective_call(lambda: self.fence.bind(payload))
        return self._receipt(bound=True)

    def _ready_metadata(self, keys=None):
        from verl_speco.transport.streaming import parse_v3_ready_tag
        tags = self.tq.kv_list(partition_id=self.config['partition']).get(self.config['partition'], {})
        result = []
        for key in keys if keys is not None else tags:
            parsed = parse_v3_ready_tag(tags.get(key, {}))
            if parsed is None:
                if keys is not None:
                    raise RuntimeError('online plan references missing committed native TQ data')
                continue
            metadata, sha = parsed
            if metadata.run_id == self.config['run_id'] and metadata.generation == self.config['generation']:
                result.append((key, metadata, sha))
            elif keys is not None:
                raise ValueError('online TQ sample belongs to a foreign run/generation')
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_drafter_training_data_status(self, *, sample_last_n_steps, require_full_batch, run_id, generation):
        if run_id != self.config['run_id'] or generation != self.config['generation'] or (not self.initialized):
            raise ValueError('online data inspection has a foreign run/generation')
        samples = self._ready_metadata()
        targets = {int(item.target_revision) for _, item, _ in samples}
        versions = {item.data_version for _, item, _ in samples}
        steps = [item.rl_step for _, item, _ in samples]
        newest = max(steps, default=0)
        return self._receipt(available=True, current_step=newest, current_step_samples=sum((step == newest for step in steps)), buffer_samples=len(samples), trainable_samples=len(samples), trainable_batches=int(bool(samples)), batch_size_per_gpu=1, partial_batch_available=False, oldest_sample_step=min(steps, default=None), newest_sample_step=max(steps, default=None), same_step_data_required=True, target_version=next(iter(targets)) if len(targets) == 1 else None, target_version_consistent=len(targets) <= 1, data_version=next(iter(versions)) if len(versions) == 1 else None, data_version_consistent=len(versions) <= 1, buffer_version=self.optimizer_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_prepare_online_target(self, payload):

        def prepare():
            self.fence.require(payload, ('bound', 'prepared'))
            if payload['required_target_version'] != self.target_revision:
                binding = self.target_heads.get(payload['required_target_version'])
                if binding is None:
                    raise RuntimeError('collection target head is unavailable; an immutable actor head snapshot is required')
                from safetensors.torch import load_file
                import torch
                path = Path(binding['uri'])
                with path.open('rb') as handle:
                    if hashlib.file_digest(handle, 'sha256').hexdigest() != binding['sha256']:
                        raise ValueError('immutable actor target head changed before online training')
                weights = load_file(str(path), device='cpu')
                head = weights.get('lm_head.weight')
                if set(weights) != {'lm_head.weight'} or head.shape != self.head.shape or head.dtype != torch.bfloat16 or (not torch.isfinite(head).all()):
                    raise ValueError('immutable actor target head geometry/dtype differs')
                self.head = head.to(self.device)
                self.head.requires_grad_(False)
                self.target_revision = payload['required_target_version']
            self.fence.phase = 'prepared'
        self._collective_call(prepare)
        return self._receipt(prepared=True, target_version=self.target_revision)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_preflight_drafter_training(self, payload):
        import torch
        from verl_speco.trainer.owned_batches import materialize_online_swa_part_shard
        from verl_speco.trainer.distributed_batches import move_dmtp_swa_part_shard

        def prepare():
            self.fence.require(payload, ('prepared', 'preflight'))
            samples = self._ready_metadata(payload['selected_keys'])
            if len(samples) != len(payload['selected_keys']):
                raise RuntimeError('online batch lost a selected key')
            for _, item, _ in samples:
                if int(item.target_revision) != payload['required_target_version'] or item.data_version != payload['data_version'] or item.rl_step != payload['source_global_step'] or (item.target_layer_ids != tuple(self.config['model_config']['target_layer_ids'])) or (item.target_model_fingerprint != self.config['target_model_fingerprint']) or (item.tokenizer_fingerprint != self.config['tokenizer_fingerprint']):
                    raise ValueError('native TQ sample changed its collection identity')
            self.reader.prepare_receive(payload['selected_keys'], self.config['partition'])
            fields = []
            for key, item, sha in samples:
                data, actual = self.reader.get_verified_cpu(key, self.config['partition'])
                if actual != sha:
                    raise RuntimeError('native online feature checksum differs')
                fields.append((data, item))
            stub = torch.empty((1, self.config['model_config']['hidden_size']), dtype=torch.bfloat16)
            cfg = self.engine.module.config
            shard, rows = materialize_online_swa_part_shard(fields, lm_head_weight=stub, target_hidden_size=self.config['model_config']['hidden_size'], target_length=self.config.get('packing_length', 16384), max_packed_bytes=self.config.get('native_max_bytes', 512 * 1024 ** 2), seq_length=cfg.max_position_embeddings, draft_train_length=cfg.block_size * cfg.num_anchors, block_size=cfg.block_size, chunk_cap=cfg.num_swa_chunk, window_size=cfg.sliding_window, seed=self.config.get('seed', 42) + payload['data_version'], rank=self.rank, world_size=self.world_size)
            for part in shard.parts:
                part.batch['lm_head_weight'] = self.head
            self.prepared = (samples, move_dmtp_swa_part_shard(shard, self.device))
            self.preflight_packing_rows = rows
            self.fence.phase = 'preflight'
        self._collective_call(prepare)
        return self._receipt(participating=True, ready=True, target_version=self.target_revision, data_version=payload['data_version'], buffer_version=self.optimizer_step, packing_rows=len(self.preflight_packing_rows.rows), selected_sample_count=self.preflight_packing_rows.sample_count)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def train_drafter(self, payload):
        import ray
        import torch.distributed as dist
        from verl_speco.trainer.distributed_checkpoint import CommitExecutor, OptimizerTransaction
        from verl_speco.integration.multinode_online import PublishedOnlineStepJournal
        from verl_speco.integration.multinode_online import online_stream_transactions
        from verl_speco.transport.control import RayLeaseProxy
        from verl_speco.transport.control import AcquireRequest
        from verl_speco.trainer.feature_pipeline import bind_acquired_batch_leases
        from verl_speco.trainer.distributed_checkpoint import save_trainable_snapshot
        from verl_speco.runtime.artifacts import publish_json
        if self.fence.phase == 'committed':
            self.fence.require(payload, ('committed',))
            return self.fence.completed[payload['plan_id']]
        self._collective_call(lambda: self.fence.require(payload, ('preflight',)))
        samples, shard = self.prepared
        keys = tuple((key for key, _, _ in samples))
        step_id = 'online-' + payload['online_plan_sha256']
        executor = None

        def prepare_transaction():
            nonlocal executor
            if self.rank != 0:
                return
            acquired, leases = ray.get(self.control.acquire.remote(AcquireRequest(step_id, 'dspark-trainer', self.config['generation'], len(keys), len(keys), str(self.target_revision), time.time() + 600), lease_timeout_s=3600, required_keys=keys))
            leases = bind_acquired_batch_leases(acquired, leases, sample_keys=keys, generation=self.config['generation'], target_revision=str(self.target_revision), consumer_id='dspark-trainer')
            lease_ids = tuple((lease.lease_id for lease in leases))
            ray.get(self.control.mark_materialized.remote(lease_ids))
            transactions = online_stream_transactions(samples, leases, stream_scope=self.config['run_id'], generation=self.config['generation'], target_revision=str(self.target_revision), committed_cursor=self.stream.snapshot().committed_cursor)
            transaction_ids = tuple((tx.transaction_id for tx in transactions))
            for transaction in transactions:
                self.stream.register(transaction)
            executor = CommitExecutor(lease_registry=RayLeaseProxy(self.control), stream=self.stream, journal=PublishedOnlineStepJournal(self.root / 'journal'), topology_fingerprint=self.bridge.topology_fingerprint, model_fingerprint=self.bridge.model_fingerprint)
            tx = OptimizerTransaction.create(optimizer_step_id=step_id, plan_id=payload['plan_id'], generation=self.config['generation'], stream_scope=self.config['run_id'], transaction_ids=transaction_ids, lease_ids=lease_ids, checkpoint_generation_before=self.config['generation'] if self.optimizer_step else 0)
            executor.prepare(tx)
            self.tq.kv_clear(keys=list(keys), partition_id=self.config['partition'])
            for key, lease in zip(keys, leases):
                ray.get(self.control.release.remote(key, lease_ids=(lease.lease_id,)))
            executor.begin_optimizer(step_id)
        self._collective_call(prepare_transaction)
        self.fence.phase = 'updating'
        started = time.monotonic()
        try:
            loss, grad = self.engine.train_dspark_swa_part_shard(shard)
            if not __import__('math').isfinite(loss) or not __import__('math').isfinite(grad):
                raise RuntimeError('online draft optimizer produced nonfinite output')
            from verl_speco.trainer.distributed_checkpoint import trainable_snapshot_parameter_checksum as _parameter_checksum, trainable_snapshot_parameters as _parameters
            import torch
            sha = _parameter_checksum(_parameters(self.engine.module, 'training_model.'))
            own = torch.tensor(list(bytes.fromhex(sha)), dtype=torch.uint8, device=self.device)
            peers = [torch.empty_like(own) for _ in range(self.world_size)]
            dist.all_gather(peers, own)
            if any((not torch.equal(own, peer) for peer in peers)):
                raise RuntimeError('DP-only online draft ranks committed different parameter bytes')
            candidate = self.bridge.save_candidate(self.root / 'dcp' / step_id, generation=self.config['generation'])

            def commit():
                if self.rank == 0:
                    executor.attach_candidate(step_id, candidate)
                    executor.commit(step_id)
            self._collective_call(commit)
            self.optimizer_step += 1

            def export():
                if self.rank == 0:
                    self.snapshot = save_trainable_snapshot(self.engine.module, self.root / 'snapshots' / step_id, draft_revision=self.draft_revision + 1, generation=self.config['generation'], source_step_id=step_id, target_model_fingerprint=self.config['target_model_fingerprint'], tokenizer_fingerprint=self.config['tokenizer_fingerprint'], model_config=self.engine.module.config.to_dict(), namespace_prefix='training_model.')
                    publish_json(self.root / 'snapshots' / (step_id + '.json'), asdict(self.snapshot))
            self._collective_call(export)
            dist.barrier()
            self.draft_revision += 1
            receipt = self._receipt(trained=True, triggered=True, successful_steps=1, attempted_steps=1, optimizer_step=self.optimizer_step, source_global_step=payload['source_global_step'], data_version=payload['data_version'], target_version=self.target_revision, plan_id=payload['plan_id'], is_publish_leader=self.rank == 0, publish_snapshot_cached=self.rank == 0, buffer_size_before=len(keys), buffer_size_after=0, elapsed_sec=time.monotonic() - started, execution_strategy='sync', snapshot=asdict(self.snapshot) if self.rank == 0 else None, candidate=asdict(candidate), loss=loss, grad_norm=grad, stream_snapshot=asdict(self.stream.snapshot()) if self.rank == 0 else None)
            receipt['trainable_parameter_sha256'] = sha

            def persist_receipt():
                if self.rank == 0:
                    publish_json(self.root / 'receipts' / (step_id + '.json'), receipt)
                    self.committed_receipt = receipt
            self._collective_call(persist_receipt)
            self.fence.commit(receipt)
            self.prepared = None
            return receipt
        except BaseException:
            self.fence.phase = 'aborted'
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_drafter_snapshot(self):
        if self.fence.phase != 'committed':
            raise RuntimeError('draft snapshot is unavailable before durable optimizer commit')
        return self._receipt(snapshot=asdict(self.snapshot) if self.rank == 0 else None, is_publish_leader=self.rank == 0, optimizer_step=self.optimizer_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_audit_online_optimizer(self):
        from verl_speco.integration.multinode_online import audit_online_optimizer
        audit = self._collective_call(lambda: audit_online_optimizer(self.engine.module, self.engine.optimizer, expected_step=self.optimizer_step))
        return self._receipt(audit=audit)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_register_online_target_head(self, binding):
        """Register an immutable actor-exported head; never replace an active head."""
        if binding.get('run_id') != self.config['run_id'] or binding.get('generation') != self.config['generation'] or binding.get('tokenizer_fingerprint') != self.config['tokenizer_fingerprint'] or (type(binding.get('target_revision')) is not int) or (binding['target_revision'] != self.target_revision + 1) or (self.fence.phase not in ('idle', 'committed')) or (not Path(binding.get('uri', '')).is_absolute()) or any((not isinstance(binding.get(key), str) or len(binding[key]) != 64 or any((c not in '0123456789abcdef' for c in binding[key])) for key in ('sha256', 'actor_weight_manifest_sha256'))):
            raise ValueError('actor target head registration has stale, foreign or incomplete bindings')
        old = self.target_heads.get(binding['target_revision'])
        if old is not None and old != binding:
            raise RuntimeError('target head revision already owns different bytes')
        self.target_heads[binding['target_revision']] = dict(binding)
        return self._receipt(registered=True, target_revision=binding['target_revision'])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_committed_step(self, receipt):
        """Restore a complete parent on fresh ranks; uncertain current updates fail closed."""
        from verl_speco.trainer.distributed_checkpoint import CandidateCheckpoint, CommitExecutor, RecoveryAction, StepState
        from verl_speco.integration.multinode_online import PublishedOnlineStepJournal
        from verl_speco.transport.control import RayLeaseProxy
        from verl_speco.trainer.distributed_checkpoint import DraftParameterSnapshot, trainable_snapshot_parameter_checksum as _parameter_checksum, trainable_snapshot_parameters as _parameters
        from verl_speco.runtime.artifacts import read_published_json

        def validate_parent():
            if not self.initialized or self.optimizer_step != 0 or self.fence.phase != 'idle' or (receipt.get('run_id') != self.config['run_id']) or (receipt.get('generation') != self.config['generation']) or (receipt.get('snapshot') is None) or (receipt.get('candidate') is None) or (receipt.get('target_version') != self.target_revision) or (type(receipt.get('optimizer_step')) is not int) or (receipt['optimizer_step'] < 1):
                raise ValueError('online restore requires a fresh owned same-generation WorkerGroup and committed parent head')
            step_id = 'online-' + receipt['online_plan_sha256']
            durable = read_published_json(self.root / 'receipts' / (step_id + '.json'))
            if json.dumps(durable, sort_keys=True) != json.dumps(receipt, sort_keys=True):
                raise RuntimeError('online restore receipt differs from durable committed bytes')
            prior = PublishedOnlineStepJournal(self.root / 'journal').load(step_id)
            if prior is None or prior.state is not StepState.COMMITTED or asdict(prior.candidate_checkpoint) != receipt['candidate']:
                raise RuntimeError('online restore parent DCP differs from committed journal')
            self.stream.restore(receipt['stream_snapshot'])
            verifier = CommitExecutor(lease_registry=RayLeaseProxy(self.control), stream=self.stream, journal=PublishedOnlineStepJournal(self.root / 'journal'), topology_fingerprint=self.bridge.topology_fingerprint, model_fingerprint=self.bridge.model_fingerprint)
            if verifier.recover_consistent(step_id).action is not RecoveryAction.RESUME_COMMITTED:
                raise RuntimeError('online restore DCP, journal, lease and stream participants disagree')
        self._collective_call(validate_parent)
        self.bridge.restore_candidate(CandidateCheckpoint(**receipt['candidate']))

        def validate_weights():
            if _parameter_checksum(_parameters(self.engine.module, 'training_model.')) != receipt['trainable_parameter_sha256']:
                raise RuntimeError('restored online draft parameter bytes differ from committed parent')
        self._collective_call(validate_weights)
        self.optimizer_step = receipt['optimizer_step']
        self.draft_revision = receipt['snapshot']['draft_revision']
        self.snapshot = DraftParameterSnapshot(**receipt['snapshot']) if self.rank == 0 else None
        self.committed_receipt = receipt if self.rank == 0 else None
        return self._receipt(restored=True, optimizer_step=self.optimizer_step, draft_revision=self.draft_revision)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_abort_drafter_training_preflight(self, payload):
        self.fence.require(payload, ('bound', 'prepared', 'preflight', 'aborted'))
        self.prepared = None
        self.fence.phase = 'aborted'
        return self._receipt(preflight_aborted=True)
from verl_speco.integration.multinode_recovery import joint_draft_recipe, open_joint_parent, read_joint_target_head, validate_joint_release, validate_joint_owners

class MultiNodeJointDraftWorker(MultiNodeOnlineDraftWorker):

    def __init__(self, config):
        super().__init__(config)
        self._joint_parent = None
        self._joint_owners = None
        self._joint_restored = False
        if self.config.get('joint_restore_binding'):
            self.fence.phase = 'restoring'

    def _joint_checkpoint_settings(self):
        return dict(module=self.engine.module, optimizer=self.engine.optimizer, scheduler=self.engine.lr_scheduler, shared_storage=True, **joint_draft_recipe(self.config, world_size=self.world_size, optimizer=self.engine.optimizer))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self.bridge = MultiNodeActorDCPBridge.collective_create(self._joint_checkpoint_settings)
        return self.speco_online_capabilities()

    def _joint_state(self):
        return actor_state_sha256(module=self.engine.module, optimizer=self.engine.optimizer, scheduler=self.engine.lr_scheduler)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_audit_joint_draft_state(self):

        def audit():
            if self.optimizer_step < 1 or self.fence.phase not in ('committed', 'restoring'):
                raise RuntimeError('joint draft complete state requires a committed or restoring safe point')
            return self._joint_state()
        return self._receipt(optimizer_step=self.optimizer_step, **self._collective_call(audit))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_restore_joint_draft(self, request):
        import ray
        from verl_speco.integration.multinode_online import PublishedOnlineStepJournal
        from verl_speco.trainer.distributed_checkpoint import CandidateCheckpoint, CommitExecutor, RecoveryAction
        from verl_speco.trainer.distributed_checkpoint import DraftParameterSnapshot, trainable_snapshot_parameter_checksum as _parameter_checksum, trainable_snapshot_parameters as _parameters
        from verl_speco.transport.control import RayLeaseProxy
        from verl_speco.transport.streaming import TrainingStreamCoordinator

        def validate():
            owners = request['owners']
            if not self.initialized or self.optimizer_step != 0 or self.fence.phase != 'restoring' or self._joint_restored or (request.get('run_id') != self.config['run_id']) or (type(request.get('generation')) is not int) or (request['generation'] != self.config['generation']) or (owners['draft'].get(str(self.rank)) != self.incarnation) or (set(owners['draft']) != {str(i) for i in range(self.world_size)}):
                raise ValueError('joint draft restore requires exact fresh fenced DP owners')
            parent, expected_control = open_joint_parent(self.config, request['binding'])
            validate_joint_owners(parent, owners)
            if ray.get(self.control.snapshot_quiescent.remote()) != expected_control:
                raise ValueError('joint draft control has not restored the exact drained parent history')
            prior = next((r for r in parent['draft_state_rows'] if r['worker_id'] == str(self.rank)))
            receipt = next((r for r in parent['draft_rows'] if r['is_publish_leader']))
            if parent['snapshot'] != self.config['initial_snapshot'] or parent['versions']['draft_revision'] != self.draft_revision:
                raise ValueError('joint draft initial serving snapshot differs from the committed parent')
            stream = TrainingStreamCoordinator(self.config['run_id'])
            stream.restore(receipt['stream_snapshot'])
            root = Path(receipt['candidate']['uri']).parents[1]
            header = parent['draft_manifest']
            verifier = CommitExecutor(lease_registry=RayLeaseProxy(self.control), stream=stream, journal=PublishedOnlineStepJournal(root / 'journal'), topology_fingerprint=header['topology_fingerprint'], model_fingerprint=self.bridge.model_fingerprint)
            step = 'online-' + receipt['online_plan_sha256']
            if verifier.recover_consistent(step).action is not RecoveryAction.RESUME_COMMITTED:
                raise ValueError('joint draft parent journal, DCP, leases and stream disagree')
            head = read_joint_target_head(parent['target_head'], run_root=self.config['joint_run_root'], shape=self.head.shape)
            return (parent, prior, receipt, stream, head, deepcopy(owners))
        parent, prior, receipt, stream, head, owners = self._collective_call(validate)
        try:
            self.bridge.restore_candidate(CandidateCheckpoint(**receipt['candidate']), handoff_generation=self.config['generation'])

            def verify_state():
                audit = self._joint_state()
                if audit['state_sha256'] != prior['state_sha256']:
                    raise RuntimeError('joint restored draft complete model/Adam/scheduler/RNG state differs')
                if _parameter_checksum(_parameters(self.engine.module, 'training_model.')) != receipt['trainable_parameter_sha256']:
                    raise RuntimeError('joint restored draft parameters differ from the serving snapshot')
                return audit
            audit = self._collective_call(verify_state)
            self.head = self._collective_call(lambda: head.to(self.device))
            self.head.requires_grad_(False)
            self.target_revision = parent['versions']['target_revision']
            self.target_heads = {self.target_revision: deepcopy(parent['target_head'])}
            self.optimizer_step = receipt['optimizer_step']
            self.draft_revision = parent['versions']['draft_revision']
            self.stream = stream
            self.snapshot = DraftParameterSnapshot(**parent['snapshot']) if self.rank == 0 else None
            self.committed_receipt = deepcopy(receipt) if self.rank == 0 else None
            self._joint_parent, self._joint_owners = (parent, owners)
            self._joint_restored = True
            return self._receipt(restored=True, binding_checksum=request['binding']['checksum'], optimizer_step=self.optimizer_step, target_revision=self.target_revision, draft_revision=self.draft_revision, compute_released=False, **audit)
        except BaseException:
            self.fence.phase = 'aborted'
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_release_joint_draft(self, proof):

        def validate():
            if not self._joint_restored or self.fence.phase != 'restoring':
                raise RuntimeError('joint draft release requires verified complete local restoration')
            validate_joint_release(self._joint_parent, proof, self._joint_owners, run_id=self.config['run_id'], generation=self.config['generation'], binding_checksum=self.config['joint_restore_binding']['checksum'])
        self._collective_call(validate)
        self.fence.phase = 'committed'
        return self._receipt(released=True, optimizer_step=self.optimizer_step, target_revision=self.target_revision, draft_revision=self.draft_revision)
import inspect
import threading
from verl.single_controller.base.decorator import Dispatch, MAGIC_ATTR, register
from verl_speco.integration.multinode_recovery import validate_shared_cards
from verl_speco.integration.multinode_recovery import ColocatedWorkerMemoryPort, colocated_worker_memory_local as _local
from verl_speco.workers.worker_identity import observe_worker_placement, validate_worker_assignment
from verl_speco.integration.multinode_recovery import configure_colocated_actor_optimizer, audit_colocated_actor_optimizer
from verl_speco.integration.multinode_recovery import install_colocated_actor_cpu_step
from verl_speco.integration.multinode_recovery import record_colocated_memory

def guard_registered_rpcs(cls):
    """Keep inherited dispatch metadata while fencing every model/transfer RPC.

    These are synchronous Ray actors (max_concurrency=1). If a parent adds an
    async RPC, reject that class rather than accidentally use a thread RLock
    across multiple coroutines. Cross-role actor/draft concurrency is separate.
    """
    for name in dir(cls):
        function = getattr(cls, name)
        if name.startswith('speco_colocated_') or not callable(function) or (not hasattr(function, MAGIC_ATTR)):
            continue
        if inspect.iscoroutinefunction(function):
            raise TypeError('shared training worker requires serialized synchronous RPCs')

        def wrap(original):

            @functools.wraps(original)
            def guarded(self, *args, **kwargs):
                with self._colocated_lock:
                    if self._colocated_failed:
                        raise RuntimeError('shared worker operation remains fenced after uncertain effects')
                    self._colocated_active += 1
                    try:
                        if original.__name__ == 'train_mini_batch' and self.colocated_role == 'actor':
                            audit_colocated_actor_optimizer(self.engine.optimizer)
                        return original(self, *args, **kwargs)
                    except BaseException:
                        self._colocated_failed = True
                        raise
                    finally:
                        self._colocated_active -= 1
            return guarded
        setattr(cls, name, wrap(function))
    return cls

class ColocatedTrainingWorkerMixin:

    def _init_colocated(self, config):
        self._colocated_settings = deepcopy(config)
        self._colocated_lock = threading.RLock()
        self._colocated_active = 0
        self._colocated_failed = False
        self._colocated_memory = None
        self._colocated_bootstrap_parked = False

    def _observed_colocated_owner(self):
        return observe_worker_placement(self, self._colocated_settings, self.colocated_role)

    def _colocated_gate(self):
        active = self._colocated_active != 0
        if self.colocated_role == 'actor':
            channel = getattr(self, '_speco_checkpoint_channel', None)
            active |= getattr(self, '_speco_target_transfer', None) is not None or (channel is not None and channel.phase not in ('new', 'prepared', 'ready', 'closed')) or getattr(self, '_joint_phase', 'ready') == 'failed' or getattr(self, '_speco_checkpoint_failed', False)
            prepared = False
        else:
            prepared = self.prepared is not None
            active |= self.fence.phase in ('updating', 'aborted')
        return dict(idle=not (active or self._colocated_failed), prepared_pending=prepared)

    def _colocated_sync(self):
        import gc
        import torch
        torch.npu.synchronize()
        gc.collect()
        torch.npu.empty_cache()

    def _colocated_set_head(self, head):
        self.head = head

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_colocated_placement(self):
        return self._observed_colocated_owner()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_colocated_park_bootstrap(self):
        if not self._colocated_lock.acquire(blocking=False):
            raise RuntimeError('shared bootstrap has an active RPC')
        try:
            owner = self._observed_colocated_owner()
            gate = self._colocated_gate()
            if not gate['idle'] or gate['prepared_pending'] or (not hasattr(self, 'engine')):
                raise RuntimeError('shared bootstrap requires initialized idle engine without prepared inputs')
            self.engine.to(device='cpu', model=True, optimizer=True, grad=True)
            if self.colocated_role == 'draft':
                self.head = self.head.to('cpu')
            self._colocated_sync()
            parameters = list(self.engine.module.parameters())
            if not parameters or any((_local(p).device.type != 'cpu' or (p.grad is not None and _local(p.grad).device.type != 'cpu') for p in parameters)):
                raise RuntimeError('shared bootstrap model/grad did not actually offload')
            import torch
            for state in self.engine.optimizer.state.values():
                if any((torch.is_tensor(v) and _local(v).device.type != 'cpu' for v in state.values())):
                    raise RuntimeError('shared bootstrap optimizer did not actually offload')
            if self.colocated_role == 'draft' and _local(self.head).device.type != 'cpu':
                raise RuntimeError('shared bootstrap target head did not actually offload')
            if self._observed_colocated_owner() != owner:
                raise RuntimeError('shared physical owner changed during bootstrap')
            self._colocated_bootstrap_parked = True
            return dict(owner, bootstrap_parked=True, model_on_device=False, optimizer_on_device=False, head_on_device=False)
        except BaseException:
            self._colocated_failed = True
            raise
        finally:
            self._colocated_lock.release()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_colocated_bind_memory(self, binding):
        with self._colocated_lock:
            checked = validate_shared_cards(run_id=self._colocated_settings['run_id'], generation=self._colocated_settings['generation'], actor=binding['actor'], draft=binding['draft'])
            if binding != checked or self._colocated_failed or (not self._colocated_bootstrap_parked) or (len(checked[self.colocated_role]) != self.world_size) or (self._observed_colocated_owner() != checked[self.colocated_role][self.rank]):
                raise ValueError('memory binding differs from independently observed complete shared world')
            if self._colocated_memory is not None:
                if self._colocated_memory.binding != checked:
                    raise ValueError('shared memory binding cannot change owners')
            else:
                self._colocated_memory = ColocatedWorkerMemoryPort(binding=checked, rank=self.rank, role=self.colocated_role, engine=self.engine, current_owner=self._observed_colocated_owner, operation_gate=self._colocated_gate, device_type='npu', head_getter=(lambda: self.head) if self.colocated_role == 'draft' else lambda: None, head_setter=self._colocated_set_head, synchronize_and_empty_cache=self._colocated_sync, optimizer_training_cpu=self.colocated_role == 'actor')
            return dict(self._observed_colocated_owner(), memory_bound=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_colocated_memory_transition(self, request):
        if not self._colocated_lock.acquire(blocking=False):
            raise RuntimeError('shared memory transition has an active RPC')
        try:
            if self._colocated_memory is None:
                raise RuntimeError('shared memory transition is unbound')
            fields = request if isinstance(request, dict) else {}
            details = dict(operation='memory_transition', location=fields.get('location'), transition_epoch=fields.get('transition_epoch'))
            record_colocated_memory(role=self.colocated_role, rank=self.rank, stage='before', details=details)
            try:
                result = self._colocated_memory.transition(request)
            except BaseException as error:
                record_colocated_memory(role=self.colocated_role, rank=self.rank, stage='failed', details=dict(details, error=str(error)[:1000]))
                raise
            record_colocated_memory(role=self.colocated_role, rank=self.rank, stage='after', details=details)
            return result
        except BaseException:
            self._colocated_failed = True
            raise
        finally:
            self._colocated_lock.release()

@guard_registered_rpcs
class MultiNodeColocatedActorWorker(ColocatedTrainingWorkerMixin, MultiNodeJointActorWorker):
    colocated_role = 'actor'

    def __init__(self, config):
        self._init_colocated(config)
        super().__init__(config)
        validate_worker_assignment(config, role='actor', rank=self.rank, world_size=self.world_size)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._observed_colocated_owner()
        receipt = super().init_model()
        policy = configure_colocated_actor_optimizer(self.engine.optimizer)
        install_colocated_actor_cpu_step(self.engine, device_type='npu', synchronize=self._colocated_sync)
        return dict(receipt, optimizer_execution=dict(policy, optimizer_step_device='cpu'))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_restore_joint_actor(self, request):
        receipt = super().speco_restore_joint_actor(request)
        policy = configure_colocated_actor_optimizer(self.engine.optimizer)
        install_colocated_actor_cpu_step(self.engine, device_type='npu', synchronize=self._colocated_sync)
        return dict(receipt, optimizer_execution=dict(policy, optimizer_step_device='cpu'))

@guard_registered_rpcs
class MultiNodeColocatedDraftWorker(ColocatedTrainingWorkerMixin, MultiNodeJointDraftWorker):
    colocated_role = 'draft'

    def __init__(self, config):
        self._init_colocated(config)
        super().__init__(config)
        validate_worker_assignment(config, role='draft', rank=self.rank, world_size=self.world_size)

    def _allow_online_colocation(self):
        self._observed_colocated_owner()
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def speco_online_capabilities(self):
        return dict(super().speco_online_capabilities(), colocated_placement=self._observed_colocated_owner())
