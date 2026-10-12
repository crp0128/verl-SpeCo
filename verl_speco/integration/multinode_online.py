# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ActorSample:
    sample_id: str
    prompt_group: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    reward: float
    target_revision: int
    temperature: float
    complete: bool = True

def validate_actor_samples(samples, *, target_revision, vocab_size):
    """Reject mixed policies, incomplete outputs and invalid GRPO groupings."""
    if not samples or len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError('actor batch requires unique, nonempty sample identities')
    groups = {}
    for sample in samples:
        if not sample.sample_id or not sample.prompt_group or sample.complete is not True:
            raise ValueError('actor requires complete outputs with owned sample and prompt identities')
        if sample.target_revision != target_revision or target_revision < 0:
            raise ValueError('actor batch contains a different target revision')
        if not math.isfinite(sample.reward):
            raise ValueError('actor reward must be finite')
        if not math.isfinite(sample.temperature) or sample.temperature <= 0:
            raise ValueError('actor log probabilities require a positive sampling temperature')
        if not sample.prompt_token_ids or not sample.response_token_ids:
            raise ValueError('actor requires nonempty prompts and responses')
        if any((type(token) is not int or not 0 <= token < vocab_size for token in sample.prompt_token_ids + sample.response_token_ids)):
            raise ValueError('actor token is outside the actual checkpoint vocabulary')
        groups.setdefault(sample.prompt_group, []).append(sample)
    for group in groups.values():
        if len(group) < 2 or len({(sample.prompt_token_ids, sample.temperature) for sample in group}) != 1:
            raise ValueError('GRPO requires at least two responses to the same prompt and temperature')
    return groups

def build_actor_batch(samples, *, target_revision, vocab_size, old_log_probs=None):
    """Use VeRL's actual jagged input contract and GRPO advantage estimator.

    Old log probabilities come from the actor's inference result, already sliced
    to response tokens. They are never guessed from generation token IDs.
    """
    validate_actor_samples(samples, target_revision=target_revision, vocab_size=vocab_size)
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.utils import tensordict_utils as tu
    prompts = [torch.tensor(sample.prompt_token_ids, dtype=torch.int64) for sample in samples]
    responses = [torch.tensor(sample.response_token_ids, dtype=torch.int64) for sample in samples]
    sequences = [torch.cat((prompt, response)) for prompt, response in zip(prompts, responses, strict=True)]
    nested = lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    lengths = [len(response) for response in responses]
    response_mask = torch.zeros(len(samples), max(lengths))
    rewards = torch.zeros_like(response_mask)
    for row, (sample, length) in enumerate(zip(samples, lengths, strict=True)):
        response_mask[row, :length] = 1
        rewards[row, length - 1] = sample.reward
    advantages, _ = compute_grpo_outcome_advantage(rewards, response_mask, np.asarray([sample.prompt_group for sample in samples]))
    tensors = dict(input_ids=nested(sequences), position_ids=nested([torch.arange(len(seq)) for seq in sequences]), prompts=nested(prompts), responses=nested(responses), response_mask=response_mask, advantages=advantages, temperature=torch.tensor([sample.temperature for sample in samples]))
    tensors['loss_mask'] = tensors['response_mask']
    if old_log_probs is not None:
        if len(old_log_probs) != len(samples):
            raise ValueError('old log probability rows do not match actor samples')
        rows = []
        for values, length in zip(old_log_probs, lengths, strict=True):
            values = torch.as_tensor(values).detach().float().cpu()
            if values.ndim != 1 or values.numel() != length or (not torch.isfinite(values).all()):
                raise ValueError('old log probabilities do not cover exactly the response tokens')
            rows.append(values)
        padded_old = torch.zeros_like(response_mask)
        for row, (values, length) in enumerate(zip(rows, lengths, strict=True)):
            padded_old[row, :length] = values
        tensors['old_log_probs'] = padded_old
    return tu.get_tensordict(tensor_dict=tensors, non_tensor_dict=dict(compute_loss=False, calculate_entropy=False, use_remove_padding=False, use_fused_kernels=False, use_dynamic_bsz=False, micro_batch_size_per_gpu=1, global_token_num=[len(seq) for seq in sequences], num_mini_batch=1, epochs=1, global_batch_size=len(samples), max_response_len=max(lengths), pad_token_id=0))

def resolve_actor_batch(value):
    import ray
    from tensordict import TensorDict
    from verl.protocol import DataProtoFuture
    from verl.utils.tensordict_utils import concat_tensordict
    if isinstance(value, DataProtoFuture):
        value = value.get()
    elif isinstance(value, ray.ObjectRef):
        value = ray.get(value)
    elif isinstance(value, list) and value and all((isinstance(row, ray.ObjectRef) for row in value)):
        value = concat_tensordict(ray.get(value))
    if not isinstance(value, TensorDict):
        raise TypeError('actual VeRL actor batch result must resolve to TensorDict')
    return value
import asyncio
from copy import deepcopy

@dataclass(frozen=True)
class ActorCollectionBatch:
    collection_id: str
    tq_keys: tuple[str, ...]
    samples: tuple

class MultiNode92BActorExecutor:
    """Session port for one real mini-batch update per collected old policy.

    Receipts are process-local. An uncertain update permanently fences this
    executor; recovery requires a new owner and actual durable actor state.
    This class does not manufacture DCP commits or target publication ACKs.
    """

    def __init__(self, *, worker_group, expected_incarnations, run_id, generation, vocab_size, optimizer_step=0, rl_step=0):
        if not expected_incarnations or any((type(rank) is not int or rank < 0 or (not isinstance(incarnation, str)) or (not incarnation) for rank, incarnation in expected_incarnations.items())) or set(expected_incarnations) != set(range(len(expected_incarnations))) or (not run_id) or (type(generation) is not int) or (generation < 1) or (type(optimizer_step) is not int) or (optimizer_step < 0) or (type(rl_step) is not int) or (rl_step < 0):
            raise ValueError('actor executor requires exact owned DP membership and initial step')
        self.group = worker_group
        self.incarnations = dict(expected_incarnations)
        self.run_id = run_id
        self.generation = generation
        self.vocab_size = vocab_size
        self.optimizer_step = optimizer_step
        self.rl_step = rl_step
        self.lock = asyncio.Lock()
        self.failed = False
        self.completed = {}

    def _audit(self, rows, step):
        if not isinstance(rows, list) or len(rows) != len(self.incarnations) or {row.get('rank') for row in rows} != set(self.incarnations):
            raise RuntimeError('actor optimizer receipt is missing, duplicate or foreign')
        for row in rows:
            if row.get('run_id') != self.run_id or row.get('generation') != self.generation or row.get('worker_incarnation') != self.incarnations[row['rank']] or (row.get('optimizer_step') != step) or (row.get('full_parameter_VeRL_actor') is not True) or (step > 0 and row.get('optimizer_parameters', 0) < 1):
                raise RuntimeError('actor optimizer owner or actual step differs from session')

    async def __call__(self, batch, binding):
        async with self.lock:
            if self.failed:
                raise RuntimeError('uncertain actor update requires durable recovery with a new owner')
            versions = binding.versions
            if versions.run_id != self.run_id or versions.generation != self.generation or (not isinstance(batch, ActorCollectionBatch)) or (batch.collection_id != binding.collection_id) or (not isinstance(batch.samples, tuple)) or (not isinstance(batch.tq_keys, tuple)) or (not batch.tq_keys) or (len(batch.tq_keys) != len(batch.samples)) or (len(set(batch.tq_keys)) != len(batch.tq_keys)) or (tuple(binding.plan.selected_keys) != batch.tq_keys):
                raise ValueError('actor batch is not bound to the exact committed feature collection')
            validate_actor_samples(batch.samples, target_revision=versions.target_revision, vocab_size=self.vocab_size)
            if len(batch.samples) % len(self.incarnations):
                raise ValueError('actor batch must contain complete DP micro-batches')
            identity = (versions, batch, binding.plan.plan_id)
            old = self.completed.get(versions.rl_step)
            if old is not None:
                if old[0] != identity:
                    raise RuntimeError('actor retry changed committed samples or training plan')
                return deepcopy(old[1])
            if versions.rl_step != self.rl_step + 1:
                raise ValueError('actor RL step must advance exactly once')
            try:
                before = await asyncio.to_thread(self.group.speco_actor_audit)
                self._audit(before, self.optimizer_step)
                infer = build_actor_batch(batch.samples, target_revision=versions.target_revision, vocab_size=self.vocab_size)

                def compute():
                    import torch
                    from verl.utils import tensordict_utils as tu
                    from verl.workers.utils.padding import no_padding_2_padding
                    result = resolve_actor_batch(self.group.infer_batch(infer))
                    log_probs = no_padding_2_padding(result['log_probs'], infer)
                    rows = [log_probs[index, :len(sample.response_token_ids)] for index, sample in enumerate(batch.samples)]
                    if any((not torch.isfinite(row).all() for row in rows)):
                        raise RuntimeError('actor inference returned nonfinite response log probabilities')
                    training = build_actor_batch(batch.samples, target_revision=versions.target_revision, vocab_size=self.vocab_size, old_log_probs=rows)
                    tu.assign_non_tensor(training, compute_loss=True)
                    resolve_actor_batch(self.group.train_mini_batch(training))
                await asyncio.to_thread(compute)
                after = await asyncio.to_thread(self.group.speco_actor_audit)
                self._audit(after, self.optimizer_step + 1)
                self.optimizer_step += 1
                self.rl_step = versions.rl_step
                receipt = dict(run_id=self.run_id, generation=self.generation, rl_step=versions.rl_step, target_revision=versions.target_revision, collection_id=batch.collection_id, optimizer_step=self.optimizer_step, optimizer_committed=True, worker_incarnations=dict(self.incarnations), actual_VeRL_actor=True, full_actor_DCP_commit_not_claimed=True)
                self.completed[versions.rl_step] = (identity, receipt)
                return deepcopy(receipt)
            except BaseException:
                self.failed = True
                raise
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
UPSTREAM_COMMIT = '2ddbed9221512b3fa55deb131d132742175e0a93'
UPSTREAM_PROFILE = 'patch/architectures/openpangu_v2_vl/'
PINNED_FILES = {'modeling/__init__.py': 'b67b5c8d37042684d6ea7986d7b9246e44bea324540d64c7e91a3ff2d2c10f72', 'modeling/modeling_openpangu_v2_vl.py': '977e8541fbacaee01344dbef30ead8df37c6c01e4e983708f175c1ab4893c5f2', 'runtime/veomni_plugin/__init__.py': '826ec4c26ad991dcb3da18c43c4771de653f863e8bbfe67cab0bdf18107b898e', 'tools/check_weight_coverage.py': '877a377c46b32cffd77064cc87aa62147b6ffaea0a72957f4234ac03e1dcbb54', 'profile.yaml': '4cf1759d1e77d7be4e58c9ab0c46fd86ad086dafffeb050b1beb00e29f93ea2c', 'README.md': '636372502d829cedc2b002e231a4c84b1c5496344b232e58f95aca37f8162e0d'}

def verify_speco_92b_training_profile(cache_dir):
    root = Path(cache_dir).resolve()
    for relative, expected in PINNED_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('92B training source differs from pinned upstream Git bytes: ' + relative)
    return root

def materialize_speco_92b_training_profile(*, repository, cache_dir):
    """Read-only Git extraction; never checkout, patch or install a sibling."""
    repository = Path(repository).resolve()
    root = Path(cache_dir).resolve()
    payloads = {name: subprocess.check_output(['git', '-C', str(repository), 'show', UPSTREAM_COMMIT + ':' + UPSTREAM_PROFILE + name]) for name in PINNED_FILES}
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != PINNED_FILES[name]:
            raise ValueError('92B Git object SHA256 differs: ' + name)
    for name, payload in payloads.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open('xb') as stream:
                stream.write(payload)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != payload:
                raise ValueError('refusing to overwrite different cached 92B source: ' + name)
    return verify_speco_92b_training_profile(root)

def install_speco_92b_training_profile(*, cache_dir, model_path):
    """Register upstream training hooks before constructing the actual actor.

    The upstream rollout hook is deliberately excluded: SpeCo's dedicated
    inference process loads its own pinned native worker and draft adapter.
    """
    root = verify_speco_92b_training_profile(cache_dir)
    model_path = Path(model_path).resolve()
    raw = json.loads((model_path / 'config.json').read_text())
    if raw.get('architectures') != ['OpenPanguV2VLForConditionalGeneration'] or raw.get('model_type') != 'openpangu_v2_vl_moe':
        raise ValueError('92B training profile requires its actual checkpoint architecture')
    expected = root / 'modeling/__init__.py'
    existing = sys.modules.get('modeling')
    if existing is not None and Path(getattr(existing, '__file__', '') or '.').resolve() != expected:
        raise RuntimeError('foreign modeling package is already imported in the actor process')
    name = '_speco_target_training_plugin'
    plugin_path = root / 'runtime/veomni_plugin/__init__.py'
    plugin = sys.modules.get(name)
    if plugin is not None:
        if Path(plugin.__file__).resolve() != plugin_path or plugin._MODEL_PATH != str(model_path):
            raise RuntimeError('actor process already owns another 92B training profile or model')
        if getattr(plugin, '_speco_install_complete', False) is not True:
            raise RuntimeError('failed 92B profile initialization requires a fresh actor process')
        return plugin._resolve_and_patch(str(model_path))
    spec = importlib.util.spec_from_file_location(name, plugin_path)
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[name] = plugin
    spec.loader.exec_module(plugin)
    plugin._MODEL_PATH = str(model_path)
    plugin._guard_signal_outside_main_thread()
    plugin._arm_packed_signal_strictness()
    hook = plugin._PostImportHook()
    plugin._HOOK = hook
    targets = {'transformers': '_patch_transformers', 'veomni.models.loader': '_patch_veomni_loader', 'veomni.models.module_utils': '_patch_veomni_strict_load', 'verl.workers.engine.veomni.utils': '_patch_verl_moe', 'verl.workers.engine.veomni.transformer_impl': '_patch_verl_veomni_transformer', 'verl.utils.checkpoint.fsdp_checkpoint_manager': '_patch_verl_checkpoint_auto_map', 'verl.utils.debug.metrics': '_patch_verl_debug_metrics', 'verl.workers.reward_manager.naive': '_patch_verl_reward_manager', 'verl.experimental.reward_loop.reward_manager.base': '_patch_verl_reward_loop_base', 'verl.plugin.platform.platform_npu': '_patch_npu_ipc', 'verl.utils.tokenizer': '_patch_verl_hf_processor'}
    if set(targets) != set(plugin.HOOK_TARGETS) - {'vllm'}:
        raise RuntimeError('pinned 92B training plugin hook contract changed')
    sys.meta_path.insert(0, hook)
    for target, callback in targets.items():
        function = getattr(plugin, callback)
        hook.register(target, function)
        module = sys.modules.get(target)
        if module is not None:
            function(module)
    model_class = plugin._resolve_and_patch(str(model_path))
    from veomni.models.loader import get_model_class
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if type(config) is not plugin._import_modeling().OpenPanguV2VLConfig or get_model_class(config) is not model_class:
        raise RuntimeError('92B actor AutoConfig or VeOmni model routing resolved another implementation')
    plugin._speco_install_complete = True
    return model_class
import torch
from verl_speco.producer.target_capture import Captured92BFeatures

@dataclass
class accepted_query_features_Chunk:
    ids: tuple[int, ...]
    states: tuple[torch.Tensor, ...]

class AcceptedQueryFeatures:

    def __init__(self, *, request_id: str, prompt_token_ids: tuple[int, ...], layer_ids: tuple[int, ...], hidden_size: int, max_capture_tokens: int, max_query_tail: int, dtype: torch.dtype=torch.bfloat16):
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError('capture request ID is required')
        if not isinstance(prompt_token_ids, tuple) or not prompt_token_ids or any((type(token) is not int or token < 0 for token in prompt_token_ids)):
            raise ValueError('immutable nonempty prompt token IDs are required')
        if not isinstance(layer_ids, tuple) or not layer_ids or any((type(layer) is not int or layer < 0 for layer in layer_ids)) or (tuple(sorted(set(layer_ids))) != layer_ids):
            raise ValueError('selected output layer IDs must be ascending unique integers')
        if any((type(value) is not int or value < 1 for value in (hidden_size, max_capture_tokens, max_query_tail))):
            raise ValueError('capture geometry and bounds must be positive integers')
        if len(prompt_token_ids) > max_capture_tokens or dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('prompt or feature dtype exceeds capture capability')
        self.request_id = request_id
        self.prompt_ids = prompt_token_ids
        self.layer_ids = layer_ids
        self.hidden_size = hidden_size
        self.max_tokens = max_capture_tokens
        self.max_query_tail = max_query_tail
        self.dtype = dtype
        self._chunks = []
        self._ids = ()
        self._device = None
        self._closed = False
        self.observed_forwards = 0
        self.discarded_query_rows = 0

    def observe(self, *, token_ids: tuple[int, ...], positions: tuple[int, ...], auxiliary: tuple[torch.Tensor, ...], final_hidden: torch.Tensor):
        if self._closed:
            raise RuntimeError('accepted feature capture is closed')
        if not isinstance(token_ids, tuple) or not token_ids or any((type(token) is not int or token < 0 for token in token_ids)) or (not isinstance(positions, tuple)) or (len(positions) != len(token_ids)) or any((type(position) is not int or position < 0 for position in positions)):
            raise ValueError('query tokens and absolute positions are invalid')
        start = positions[0]
        count = len(token_ids)
        end = start + count
        if positions != tuple(range(start, end)) or start > len(self._ids):
            raise RuntimeError('target query has noncontiguous positions or an input gap')
        if self._ids and start == 0:
            raise RuntimeError('new request cannot overwrite an armed capture')
        if end > self.max_tokens:
            raise RuntimeError('target query exceeds bounded feature storage')
        if len(self._ids) - start > self.max_query_tail:
            raise RuntimeError('target rejection exceeds declared speculative tail')
        if any((token != self.prompt_ids[position] for token, position in zip(token_ids, positions) if position < len(self.prompt_ids))):
            raise RuntimeError('target query prompt identity differs')
        if not isinstance(auxiliary, tuple) or len(auxiliary) != len(self.layer_ids):
            raise ValueError('target query is missing selected auxiliary outputs')
        states = (*auxiliary, final_hidden)
        if any((not isinstance(value, torch.Tensor) or value.shape != (count, self.hidden_size) or value.dtype != self.dtype or value.requires_grad for value in states)):
            raise ValueError('target query feature shape/dtype/no-grad contract differs')
        device = final_hidden.device
        if any((value.device != device for value in states)) or (self._device is not None and device != self._device):
            raise ValueError('target query feature device identity differs')
        retained = []
        cursor = 0
        for chunk in self._chunks:
            keep = min(len(chunk.ids), max(start - cursor, 0))
            if keep:
                retained.append(accepted_query_features_Chunk(chunk.ids[:keep], tuple((value[:keep] for value in chunk.states))))
            cursor += len(chunk.ids)
        self.discarded_query_rows += len(self._ids) - start
        retained.append(accepted_query_features_Chunk(token_ids, tuple((value.detach().clone() for value in states))))
        self._chunks = retained
        self._ids = self._ids[:start] + token_ids
        self._device = device
        self.observed_forwards += 1

    def finish(self, generated_token_ids: tuple[int, ...]) -> Captured92BFeatures:
        if self._closed:
            raise RuntimeError('accepted feature capture is closed')
        if not isinstance(generated_token_ids, tuple) or not generated_token_ids or any((type(token) is not int or token < 0 for token in generated_token_ids)):
            raise ValueError('completed generation token IDs are required')
        expected = self.prompt_ids + generated_token_ids[:-1]
        terminal_query_bound = self.max_query_tail + 1
        if len(self._ids) < len(expected) or self._ids[:len(expected)] != expected or len(self._ids) - len(expected) > terminal_query_bound:
            mismatch = next((index for index, pair in enumerate(zip(self._ids, expected)) if pair[0] != pair[1]), None)
            raise RuntimeError(f'completed response does not match accepted target input features: request={self.request_id}, observed_rows={len(self._ids)}, expected_rows={len(expected)}, first_mismatch={mismatch}, observed_token={(None if mismatch is None else self._ids[mismatch])}, expected_token={(None if mismatch is None else expected[mismatch])}, forwards={self.observed_forwards}, chunks={[len(chunk.ids) for chunk in self._chunks]}, discarded_rows={self.discarded_query_rows}, max_tail={self.max_query_tail}, terminal_query_bound={terminal_query_bound}')
        merged = tuple((torch.cat([chunk.states[index] for chunk in self._chunks], dim=0)[:len(expected)].clone() for index in range(len(self.layer_ids) + 1)))
        result = Captured92BFeatures(self.request_id, expected, tuple(range(len(expected))), self.layer_ids, merged[:-1], merged[-1])
        self.discarded_query_rows += len(self._ids) - len(expected)
        self.abort()
        return result

    def abort(self):
        self._chunks.clear()
        self._ids = ()
        self._closed = True

def accepted_query_probe_ids_sha(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()

def accepted_query_probe_tensor_sha(value):
    local = value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(local.view(torch.uint8).numpy())).hexdigest()

class AcceptedTargetQueryProbe:

    def __init__(self, runner, *, metadata_supplier, dtype=torch.bfloat16):
        self.runner = runner
        self.metadata_supplier = metadata_supplier
        self.dtype = dtype
        self.capture = None
        self.handle = None
        self.engine_request_id = None
        self.finished_features = None

    def arm(self, *, request_id, prompt_token_ids, layer_ids, hidden_size, max_capture_tokens, max_query_tail):
        if self.capture is not None or self.finished_features is not None:
            raise RuntimeError('previous accepted query capture must be released')
        if not self.runner.use_aux_hidden_state_outputs or not getattr(self.runner.model, 'supports_eagle3', False):
            raise ValueError('loaded target has no verified auxiliary-output contract')
        self.capture = AcceptedQueryFeatures(request_id=request_id, prompt_token_ids=tuple(prompt_token_ids), layer_ids=tuple(layer_ids), hidden_size=hidden_size, max_capture_tokens=max_capture_tokens, max_query_tail=max_query_tail, dtype=self.dtype)
        self.engine_request_id = None
        try:
            self.handle = self.runner.model.register_forward_hook(self._observe, with_kwargs=True)
        except BaseException:
            self.capture = None
            raise

    def _observe(self, _model, args, kwargs, output):
        req_ids = tuple(self.runner.input_batch.req_ids)
        if len(req_ids) != 1:
            raise RuntimeError('accepted query probe supports one actual engine request')
        if self.engine_request_id is not None and self.engine_request_id != req_ids[0]:
            raise RuntimeError('engine request changed during accepted query capture')
        metadata = self.metadata_supplier()
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()), None)
        bounds = getattr(metadata, 'query_start_loc', None)
        if not isinstance(bounds, torch.Tensor):
            raise RuntimeError('accepted query probe lacks actual query row metadata')
        bounds = tuple((int(value) for value in bounds.detach().cpu().tolist()))
        if len(bounds) < 2 or bounds[0] != 0 or bounds[1] < 1 or any((value != bounds[1] for value in bounds[2:])):
            raise RuntimeError('accepted query metadata contains foreign or empty rows')
        rows = bounds[1]
        if not isinstance(output, tuple) or len(output) != 2 or (not isinstance(output[1], (list, tuple))):
            raise RuntimeError('accepted query target did not return final and auxiliary features')
        final, auxiliary = output
        if not isinstance(final, torch.Tensor) or final.ndim != 2 or final.shape[0] < rows or any((not isinstance(value, torch.Tensor) or value.shape != final.shape for value in auxiliary)):
            raise RuntimeError('accepted query target feature rows differ')
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None:
            ids = self.runner.input_ids.gpu
        positions = kwargs.get('positions', args[1] if len(args) > 1 else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.numel() < rows or (ids.dtype not in (torch.int32, torch.int64)) or (not isinstance(positions, torch.Tensor)) or (positions.dtype not in (torch.int32, torch.int64)):
            raise RuntimeError('accepted query token or position buffer differs')
        if positions.ndim == 2 and positions.shape[0] == 3:
            if not torch.equal(positions[0, :rows], positions[1, :rows]) or not torch.equal(positions[0, :rows], positions[2, :rows]):
                raise RuntimeError('accepted query functional probe currently requires text positions')
            positions = positions[0]
        if positions.ndim != 1 or positions.numel() < rows:
            raise RuntimeError('accepted query absolute position layout differs')
        self.capture.observe(token_ids=tuple((int(value) for value in ids[:rows].detach().cpu().tolist())), positions=tuple((int(value) for value in positions[:rows].detach().cpu().tolist())), auxiliary=tuple((value[:rows] for value in auxiliary)), final_hidden=final[:rows])
        self.engine_request_id = req_ids[0]

    def finish(self, generated_token_ids):
        if self.capture is None:
            raise RuntimeError('accepted query capture is not armed')
        capture = self.capture
        features = capture.finish(tuple(generated_token_ids))
        self.handle.remove()
        self.handle = None
        self.capture = None
        self.finished_features = features
        return dict(request_id=features.request_id, engine_request_id=self.engine_request_id, rows=len(features.token_ids), layer_ids=list(features.layer_ids), input_ids_sha256=accepted_query_probe_ids_sha(features.token_ids), positions_sha256=accepted_query_probe_ids_sha(features.positions), selected_sha256=[accepted_query_probe_tensor_sha(value) for value in features.hidden_states], final_sha256=accepted_query_probe_tensor_sha(features.final_hidden_state), observed_forwards=capture.observed_forwards, discarded_query_rows=capture.discarded_query_rows, feature_bytes=sum((value.numel() * value.element_size() for value in (*features.hidden_states, features.final_hidden_state))), accepted_feature_capture_only=True, TQ_or_training_or_publish_not_claimed=True)

    def release(self):
        if self.capture is not None:
            raise RuntimeError('cannot release an unfinished accepted query capture')
        if self.finished_features is None:
            raise RuntimeError('accepted query features are not available')
        self.finished_features = None

    def close(self):
        if self.handle is not None:
            self.handle.remove()
        if self.capture is not None:
            self.capture.abort()
        self.handle = None
        self.capture = None
        self.finished_features = None
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable
from threading import Lock
from verl_speco.trainer.scheduler.drafter_runtime_state import DrafterRuntimeState
from verl_speco.trainer.scheduler.drafter_scheduler import DrafterScheduler
from verl_speco.trainer.scheduler.lifecycle import AfterActorUpdateContext, AfterWeightUpdateContext
from verl_speco.trainer.scheduler.schedule_types import DrafterExecutionStrategy, DrafterScheduleConfig, DrafterScheduleContext, DrafterTrainingDataSource, TrainingDataStatus, TrainingPlan

def verl_online_counter(name: str, value: int, *, positive: bool=False) -> None:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be a {('positive' if positive else 'non-negative')} integer")

@dataclass(frozen=True)
class OnlineVersions:
    run_id: str
    generation: int
    rl_step: int
    target_revision: int
    data_version: int
    draft_revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError('run_id is required')
        for name in ('generation', 'rl_step', 'target_revision', 'data_version', 'draft_revision'):
            verl_online_counter(name, getattr(self, name), positive=name == 'generation')

@dataclass(frozen=True)
class OnlineCollectionCommit:
    """Receipt issued only after all feature producers committed their TQ stage."""
    versions: OnlineVersions
    collection_id: str
    producer_ids: tuple[str, ...]
    tq_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.collection_id:
            raise ValueError('collection_id is required')
        for name in ('producer_ids', 'tq_keys'):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or any((not isinstance(v, str) or not v for v in values)):
                raise ValueError(f'{name} must contain immutable non-empty string identities')
            if len(set(values)) != len(values):
                raise ValueError(f'duplicate {name}')

@dataclass(frozen=True)
class OnlineTrainingBinding:
    """Separate online identities frozen around the scheduler-produced plan."""
    versions: OnlineVersions
    collection_id: str
    plan: TrainingPlan

    def to_worker_payload(self) -> dict[str, object]:
        payload = self.plan.to_worker_payload()
        payload['online_versions'] = dict(run_id=self.versions.run_id, generation=self.versions.generation, rl_step=self.versions.rl_step, target_revision=self.versions.target_revision, data_version=self.versions.data_version, draft_revision=self.versions.draft_revision, collection_id=self.collection_id, plan_id=self.plan.plan_id)
        return deepcopy(payload)

class OnlinePhase(str, Enum):
    IDLE = 'idle'
    COLLECTED = 'collected'
    PREPARED = 'prepared'
    ACTOR_UPDATED = 'actor_updated'
    TRAINED = 'trained'
    ABORTED = 'aborted'

class MultiNodeOnlineAdapter:
    """Validate events without duplicating interval, batch, or publish decisions.

    ``publish_commit`` must verify every replica's revision/checksum ACK and
    keep generation blocked until this adapter commits its draft revision.
    ``resume_rollout`` runs only afterwards. An uncertain or failed operation
    calls the plan abort port and leaves this adapter ABORTED: the caller must
    recover from durable actor/draft/TQ state before creating a new adapter.
    """

    def __init__(self, *, scheduler: DrafterScheduler, versions: OnlineVersions, producer_ids: tuple[str, ...], abort_plan: Callable[[OnlineTrainingBinding | None, str], None], publish_commit: Callable[[OnlineTrainingBinding, int], None], resume_rollout: Callable[[], None], bind_worker_plan: Callable[[OnlineTrainingBinding], None], parallel_updates: bool=False) -> None:
        if not producer_ids or len(set(producer_ids)) != len(producer_ids):
            raise ValueError('expected producer identities must be non-empty and unique')
        if type(parallel_updates) is not bool:
            raise ValueError('parallel_updates must be an explicit boolean')
        self.parallel_updates = parallel_updates
        self._execution_lock = Lock()
        self._execution_claimed = False
        self._execution_finished = False
        self._prepared_outcome = None
        self._prepared_error = None
        self.scheduler = scheduler
        self.versions = versions
        self.producer_ids = frozenset(producer_ids)
        self.abort_plan = abort_plan
        self.publish_commit = publish_commit
        self.resume_rollout = resume_rollout
        self.bind_worker_plan = bind_worker_plan
        self.phase = OnlinePhase.IDLE
        self.runtime_state = DrafterRuntimeState()
        self.collection: OnlineCollectionCommit | None = None
        self.binding: OnlineTrainingBinding | None = None
        self._schedule_config: DrafterScheduleConfig | None = None

    def _require(self, phase: OnlinePhase) -> None:
        if self.phase is not phase:
            raise RuntimeError(f'expected online phase {phase.value}, got {self.phase.value}')

    def on_rollout_complete(self, commit: OnlineCollectionCommit) -> None:
        self._require(OnlinePhase.IDLE)
        expected = replace(self.versions, rl_step=self.versions.rl_step + 1, data_version=self.versions.data_version + 1)
        if commit.versions != expected:
            raise ValueError('collection run/generation/RL/target/data/draft identities do not match')
        if frozenset(commit.producer_ids) != self.producer_ids:
            raise ValueError('collection is missing or includes unexpected producers')
        self.collection = commit
        self.versions = replace(self.versions, data_version=commit.versions.data_version)
        self.phase = OnlinePhase.COLLECTED

    def before_actor_update(self, *, config: DrafterScheduleConfig, data_status: TrainingDataStatus | None=None) -> OnlineTrainingBinding:
        self._require(OnlinePhase.COLLECTED)
        if config.publish_async:
            raise ValueError('target model synchronous MVP requires publish_async=false')
        assert self.collection is not None
        commit = self.collection
        try:
            status = data_status or self.scheduler.inspect_training_data(global_step=commit.versions.rl_step, config=config)
            if not status.target_version_consistent or not status.data_version_consistent or status.target_version != commit.versions.target_revision or (status.data_version != commit.versions.data_version):
                raise ValueError('worker data does not match committed collection revisions')
            plan = self.scheduler.prepare_training_plan(DrafterScheduleContext(global_step=commit.versions.rl_step, training_mode='online', collected_samples_this_step=len(commit.tq_keys), oldlogprob_collection_requested=False, data_status=status), config)
            if plan.execution_strategy is not DrafterExecutionStrategy.SYNC:
                raise ValueError('target model synchronous MVP requires a synchronous scheduler plan')
            plan = replace(plan, required_target_version=commit.versions.target_revision, data_version=commit.versions.data_version, data_source=DrafterTrainingDataSource.TRANSFER_QUEUE, selected_keys=commit.tq_keys, worker_snapshots=deepcopy(plan.worker_snapshots))
            if plan.launch and (not plan.worker_snapshots):
                raise ValueError('online draft plan requires all WorkerGroup incarnation snapshots')
            self.binding = OnlineTrainingBinding(commit.versions, commit.collection_id, plan)
            self._schedule_config = config
            self.bind_worker_plan(self.binding)
            self.scheduler.prepare_training_execution(plan)
            self.phase = OnlinePhase.PREPARED
            return self.binding
        except Exception as error:
            self.on_role_failure('prepare', error)
            raise

    def execute_prepared_draft(self):
        """Claim exactly one bound plan without manufacturing an actor ACK.

        Failures are returned to the joining session before it aborts roles;
        callbacks must not release a generation while the actor is running.
        """
        with self._execution_lock:
            self._require(OnlinePhase.PREPARED)
            if not self.parallel_updates or self.binding is None:
                raise RuntimeError('parallel draft execution requires an explicit prepared binding')
            if self._execution_claimed:
                raise RuntimeError('prepared draft plan was already claimed')
            self._execution_claimed = True
            plan = self.binding.plan
        try:
            outcome = self.scheduler.execute_prepared_training(plan, runtime_state=self.runtime_state)
        except BaseException as error:
            with self._execution_lock:
                self._prepared_error = error
                self._execution_finished = True
            raise
        with self._execution_lock:
            self._prepared_outcome = outcome
            self._execution_finished = True
        return outcome

    def after_actor_update(self, *, rl_step: int):
        self._require(OnlinePhase.PREPARED)
        assert self.binding is not None
        if type(rl_step) is not int or rl_step != self.binding.versions.rl_step:
            raise ValueError('actor update acknowledged a different RL step')
        with self._execution_lock:
            if self.parallel_updates and (not self._execution_finished):
                raise RuntimeError('actual draft execution must finish before consuming the actor ACK')
        self.versions = replace(self.versions, rl_step=rl_step)
        self.phase = OnlinePhase.ACTOR_UPDATED
        try:
            if self.parallel_updates:
                with self._execution_lock:
                    if self._prepared_error is not None:
                        raise self._prepared_error
                    outcome = self._prepared_outcome
                if outcome is None or outcome.training_plan != self.binding.plan:
                    raise RuntimeError('prepared execution outcome differs from the actual bound plan')
            else:
                outcome = self.scheduler.on_after_actor_update(AfterActorUpdateContext(self.binding.plan, self.runtime_state))
            if self.binding.plan.launch:
                training = outcome.training_execution
                if training is None or not training.trained or training.successful_steps < 1:
                    raise RuntimeError('draft WorkerGroup did not commit a consistent training update')
                snapshots = self.binding.plan.worker_snapshots or {}
                for worker in training.worker_results:
                    if worker.worker_id in snapshots and worker.worker_incarnation != snapshots[worker.worker_id].get('worker_incarnation'):
                        raise RuntimeError('draft worker incarnation changed during the training plan')
            self.phase = OnlinePhase.TRAINED
            return outcome
        except Exception as error:
            self.on_role_failure('actor_or_draft', error)
            raise

    def at_rollout_safe_point(self):
        self._require(OnlinePhase.TRAINED)
        assert self.binding is not None and self._schedule_config is not None
        try:
            outcome = self.scheduler.on_safe_point(AfterWeightUpdateContext(global_step=self.binding.versions.rl_step, drafter_trained=self.binding.plan.launch, config=self._schedule_config, training_plan=self.binding.plan))
            if outcome.publish_plan is not None and outcome.publish_plan.publish:
                if outcome.publish_outcome is None or not outcome.publish_outcome.published:
                    raise RuntimeError('scheduled draft publication has no verified snapshot')
                next_revision = self.versions.draft_revision + 1
                self.publish_commit(self.binding, next_revision)
                self.versions = replace(self.versions, draft_revision=next_revision)
                self.resume_rollout()
            self.phase = OnlinePhase.IDLE
            self.collection = None
            self.binding = None
            self._schedule_config = None
            with self._execution_lock:
                self._execution_claimed = False
                self._execution_finished = False
                self._prepared_outcome = None
                self._prepared_error = None
            return outcome
        except Exception as error:
            self.on_role_failure('publish', error)
            raise

    def on_role_failure(self, role: str, error: BaseException | str) -> None:
        if self.phase is OnlinePhase.ABORTED:
            return
        self.phase = OnlinePhase.ABORTED
        self.abort_plan(self.binding, f'{role}: {error}')
import inspect
from verl_speco.integration.accepted_publication import require_accepted_publication_consensus
from verl_speco.transport.streaming import SampleMetadataV3

async def async_online_await(value):
    return await value if inspect.isawaitable(value) else value

@dataclass(frozen=True)
class AsyncCollectionRequest:
    """One owned stream in a collection using a single unchanged policy."""
    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_stream: object
    metadata: object
    publication_settings: dict

class OnlineCollectionCoordinator:

    def __init__(self, *, versions: OnlineVersions, collective_rpc, expected_ranks: tuple[int, ...], producer_id: str, save_complete_sequence, abort_collection, admit_collection=None):
        if not expected_ranks or len(set(expected_ranks)) != len(expected_ranks) or any((type(rank) is not int or rank < 0 for rank in expected_ranks)) or (0 not in expected_ranks) or (not producer_id):
            raise ValueError('async collection requires explicit producer and TP membership')
        self.versions = versions
        self.rpc = collective_rpc
        self.expected_ranks = expected_ranks
        self.producer_id = producer_id
        self.save_complete_sequence = save_complete_sequence
        self.abort_collection = abort_collection
        if admit_collection is not None and (not callable(admit_collection)):
            raise ValueError('async admission must be a callable')
        self.admit_collection = admit_collection
        self.lock = asyncio.Lock()
        self.failed = False
        self.pending_collection = None

    async def collect(self, *, collection_id, prompt_token_ids, output_stream, metadata, publication_settings, request_id=None):
        """Commit one complete native stream; retain the historical single API."""
        if not callable(metadata):
            item = SampleMetadataV3.from_dict(metadata)
            if request_id is not None and request_id != item.request_id:
                raise ValueError('explicit request ID differs from metadata')
            request_id = item.request_id
        return await self.collect_group(collection_id=collection_id, requests=(AsyncCollectionRequest(request_id, tuple(prompt_token_ids), output_stream, metadata, publication_settings),))

    async def collect_group(self, *, collection_id, requests):
        """Commit all response features at the same old target/draft boundary.

        The native accepted-query probe owns one request at a time. Streams run
        serially under the same version lock; no weight safe point or partial
        group commit is exposed between requests. This supports grouped GRPO,
        not concurrent multi-request capture or fully-async policy updates.
        """
        async with self.lock:
            if self.failed or self.pending_collection is not None:
                raise RuntimeError('async collection requires recovery or preceding safe point')
            if not isinstance(collection_id, str) or not collection_id or (not isinstance(requests, tuple)) or (not requests) or any((not isinstance(item, AsyncCollectionRequest) for item in requests)):
                raise ValueError('async group requires an immutable nonempty request tuple')
            ids = [item.request_id for item in requests]
            if any((not isinstance(item, str) or not item for item in ids)) or len(set(ids)) != len(ids):
                raise ValueError('async group requires unique stable request IDs')
            expected = replace(self.versions, rl_step=self.versions.rl_step + 1, data_version=self.versions.data_version + 1)
            prepared = []
            for item in requests:
                if not isinstance(item.prompt_token_ids, tuple) or not item.prompt_token_ids or any((type(token) is not int or token < 0 for token in item.prompt_token_ids)) or (not callable(item.output_stream)) or (not isinstance(item.publication_settings, dict)) or {'metadata', 'loss_mask'} & item.publication_settings.keys():
                    raise ValueError('async group has invalid input or publication settings')
                v3 = None if callable(item.metadata) else SampleMetadataV3.from_dict(item.metadata)
                if v3 is not None:
                    self._validate_request_metadata(v3, item, expected)
                prepared.append((item, v3))
            attempted = []
            keys = []
            try:
                if self.admit_collection is not None:
                    await async_online_await(self.admit_collection(expected, tuple((item for item, _ in prepared))))
                for item, v3 in prepared:
                    attempted.append(item.request_id)
                    key = await self._collect_request(item, v3, expected)
                    if key in keys:
                        raise RuntimeError('async group published a duplicate sample key')
                    keys.append(key)
                commit = OnlineCollectionCommit(expected, collection_id, (self.producer_id,), tuple(keys))
                self.pending_collection = commit
                return commit
            except BaseException as error:
                self.failed = True

                async def abort_owned(cause=error):
                    errors = []
                    for request_id in reversed(attempted):
                        try:
                            await async_online_await(self.abort_collection(request_id, str(cause)))
                        except BaseException as cleanup_error:
                            errors.append(cleanup_error)
                    if errors:
                        raise errors[0]
                cleanup = asyncio.create_task(abort_owned())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                raise

    def _validate_request_metadata(self, v3, item, expected):
        if v3.request_id != item.request_id or v3.run_id != expected.run_id or v3.generation != expected.generation or (v3.rl_step != expected.rl_step) or (v3.data_version != expected.data_version) or (v3.target_revision != str(expected.target_revision)) or (v3.draft_revision != str(expected.draft_revision)) or (v3.producer_id != self.producer_id) or (v3.prompt_token_count != len(item.prompt_token_ids)):
            raise ValueError('async request metadata belongs to a different version boundary')

    async def _collect_request(self, item, v3, expected):
        request_id = item.request_id
        prompt = item.prompt_token_ids
        rows = await async_online_await(self.rpc('speco_accepted_query_arm', dict(request_id=request_id, prompt_token_ids=list(prompt))))
        if len(rows) != len(self.expected_ranks) or {row.get('rank') for row in rows} != set(self.expected_ranks) or any((row.get('armed') is not True or row.get('request_id') != request_id for row in rows)):
            raise RuntimeError('async engine did not arm all owned TP ranks')
        finished = None
        async for output in item.output_stream():
            if getattr(output, 'finished', False):
                if finished is not None:
                    raise RuntimeError('async engine produced duplicate completed output')
                finished = output
        if finished is None or len(finished.outputs) != 1 or getattr(finished.outputs[0], 'finish_reason', None) is None:
            raise RuntimeError('async engine did not finish one complete response')
        generated = tuple(finished.outputs[0].token_ids)
        if tuple(finished.prompt_token_ids) != prompt:
            raise RuntimeError('async engine returned a different prompt identity')
        if v3 is None:
            v3 = SampleMetadataV3.from_dict(await async_online_await(item.metadata(expected, prompt, generated)))
            self._validate_request_metadata(v3, item, expected)
        if len(prompt) + len(generated) - 1 != v3.token_count or len(generated) < 2:
            raise RuntimeError('async output token count differs from committed training rows')
        await async_online_await(self.save_complete_sequence(v3, prompt, generated))
        rows = await async_online_await(self.rpc('speco_accepted_query_finish', dict(generated_token_ids=list(generated))))
        if len(rows) != len(self.expected_ranks) or {row.get('rank') for row in rows} != set(self.expected_ranks) or any((row.get('rows') != v3.token_count or row.get('request_id') != request_id for row in rows)):
            raise RuntimeError('async target capture did not finish every TP rank')
        rows = await async_online_await(self.rpc('speco_accepted_query_publish', dict(metadata=v3.to_dict(), loss_mask=[0.0] * len(prompt) + [1.0] * (len(generated) - 1), **item.publication_settings)))
        consensus = require_accepted_publication_consensus(rows, metadata=v3.to_dict(), expected_ranks=self.expected_ranks)
        rows = await async_online_await(self.rpc('speco_accepted_query_release', {}))
        if len(rows) != len(self.expected_ranks) or {row.get('rank') for row in rows} != set(self.expected_ranks) or any((row.get('released') is not True for row in rows)):
            raise RuntimeError('async target did not release all private captures')
        return consensus['key']

    async def commit_safe_point(self, *, versions: OnlineVersions, publish):
        """Hold the same lock until actual target/draft all-rank ACKs are verified."""
        async with self.lock:
            if self.failed or self.pending_collection is None:
                raise RuntimeError('async publication requires a completed collection')
            pending = self.pending_collection.versions
            if versions.run_id != pending.run_id or versions.generation != pending.generation or versions.rl_step != pending.rl_step or (versions.data_version != pending.data_version) or (versions.target_revision not in (pending.target_revision, pending.target_revision + 1)) or (versions.draft_revision not in (pending.draft_revision, pending.draft_revision + 1)):
                raise ValueError('async safe point proposed a foreign or skipped version')
            try:
                await async_online_await(publish())
                self.versions = versions
                self.pending_collection = None
            except BaseException:
                self.failed = True
                raise

def full_async_engine_kwargs(config):
    utilization = config.get('rollout_gpu_memory_utilization', 0.55 if config.get('colocated') is True else 0.7)
    if type(utilization) not in (int, float) or not math.isfinite(utilization) or (not 0 < utilization < 1):
        raise ValueError('rollout GPU memory utilization must be finite and between zero and one')
    baseline = dict(model=config['model_overlay'], tokenizer=config['model_overlay'], trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=8, data_parallel_size=1, distributed_executor_backend='mp', max_model_len=1024, max_num_seqs=1, gpu_memory_utilization=float(utilization), block_size=128, enable_prefix_caching=False, enable_chunked_prefill=True, max_num_batched_tokens=1024, enforce_eager=True, limit_mm_per_prompt={'image': 0}, worker_cls='verl_speco.integration.accelerator_adapters.SpecoMultiNodeTargetUpdateWorker', speculative_config=dict(method='dflash', model=config['draft_model_path'], num_speculative_tokens=8, draft_tensor_parallel_size=8, enforce_eager=True, max_model_len=1024))
    if 'rollout_kv_cache_memory_bytes' in config:
        budget = config['rollout_kv_cache_memory_bytes']
        if config.get('colocated') is not True or type(budget) is not int or budget < 1:
            raise ValueError('shared rollout KV cache memory bytes must be a positive integer')
        baseline['kv_cache_memory_bytes'] = budget
    from verl_speco.integration.multinode_recovery import colocated_graph_engine_kwargs
    return colocated_graph_engine_kwargs(config, baseline)
from types import MethodType

def hccl_checkpoint_compat_destroy_comm(communicator, comm):
    import torch
    torch.npu.synchronize(communicator.device)
    communicator.hccl.hcclCommDestroy(comm)

def release_owned_checkpoint_cache(device):
    """Return idle Torch allocator memory before external HCCL allocations.

    Called only on the owned, fenced checkpoint safe point. Synchronize first;
    live model/optimizer/bucket tensors stay allocated at the same addresses.
    """
    import torch
    torch.npu.synchronize(device)
    before = dict(allocated=torch.npu.memory_allocated(device), reserved=torch.npu.memory_reserved(device))
    torch.npu.empty_cache()
    after = dict(allocated=torch.npu.memory_allocated(device), reserved=torch.npu.memory_reserved(device))
    if after['allocated'] != before['allocated'] or after['reserved'] > before['reserved']:
        raise RuntimeError('owned checkpoint cache release changed live tensor accounting')
    return dict(before=before, after=after, idle_cache_released_bytes=before['reserved'] - after['reserved'])

def checkpoint_control_sum(communicator, tensor, *, op=None, stream=None):
    """Known uniform initialization controls without asymmetric reductions.

    PyHccl warms up with float32 zero and VeRL connects with int8 one. Every
    participant supplies that same known value. Broadcast plus local scaling
    returns their exact SUM, while keeping strict actor reductions untouched.
    Weight payloads continue through the native broadcast implementation.
    """
    import torch
    from torch.distributed import ReduceOp
    expected = {torch.float32: 0, torch.int8: 1}
    if op is not None and op != ReduceOp.SUM or tensor.numel() != 1:
        raise ValueError('owned checkpoint accepts only its uniform initialization SUM controls')
    if tensor.dtype not in expected or tensor.item() != expected[tensor.dtype]:
        raise ValueError('owned checkpoint received an unknown initialization control')
    if not 2 <= communicator.world_size <= 127:
        raise ValueError('checkpoint control SUM requires the bounded cluster rank count')
    result = tensor.clone()
    communicator.broadcast(result, src=0, stream=stream)
    return result.mul(communicator.world_size)

def install_hccl_checkpoint_compat(*, checkpoint_module=None, group_class=None, communicator_class=None, checkpoint_control_broadcast=True, checkpoint_reclaim_cache=True):
    """Adapt the owned checkpoint engine; leave the installed packages intact.

    Older VeRL copies a vLLM constructor accepting ``socket``. New vLLM owns
    TCPStore construction in ``create`` and removed that constructor argument.
    Patch only the checkpoint module's imported alias, preserving other users
    of VeRL's distributed helpers and the training process group.
    """
    if checkpoint_module is None:
        from verl.checkpoint_engine import hccl_checkpoint_engine as checkpoint_module
    if group_class is None:
        from vllm.distributed.utils import StatelessProcessGroup as group_class
    original = checkpoint_module.stateless_init_process_group
    if getattr(original, '_speco_stateless_store_compat', False):
        if original._speco_checkpoint_control_broadcast != checkpoint_control_broadcast or original._speco_checkpoint_reclaim_cache != checkpoint_reclaim_cache:
            raise ValueError('owned process already installed a different checkpoint communicator mode')
        return False
    if 'socket' in inspect.signature(group_class).parameters:
        return False
    factory = group_class.create
    if not callable(factory):
        raise TypeError('vLLM stateless store requires a callable create factory')
    native_communicator = communicator_class is None
    if native_communicator:
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
        if checkpoint_control_broadcast:

            class OwnedCheckpointCommunicator(PyHcclCommunicator):

                def all_reduce(self, in_tensor, op=None, stream=None):
                    return checkpoint_control_sum(self, in_tensor, op=op, stream=stream)
            communicator_class = OwnedCheckpointCommunicator
        else:
            communicator_class = PyHcclCommunicator

    def initialize(master_address, master_port, rank, world_size, device):
        if ':' in master_address:
            raise ValueError('SpeCo HCCL compatibility requires an IPv4 address')
        group = factory(host=master_address, port=master_port, rank=rank, world_size=world_size)
        cache = release_owned_checkpoint_cache(device) if native_communicator and checkpoint_reclaim_cache else None
        communicator = communicator_class(group, device=device)
        if native_communicator:
            communicator._speco_checkpoint_cache_release = cache
        if getattr(communicator, 'disabled', False) or not getattr(communicator, 'available', True):
            raise RuntimeError('owned HCCL communicator is unavailable')
        if not hasattr(communicator, 'destroyComm') and hasattr(communicator, 'hccl'):
            if not callable(getattr(communicator.hccl, 'hcclCommDestroy', None)):
                raise TypeError('owned HCCL communicator cannot be safely finalized')
            communicator.destroyComm = MethodType(hccl_checkpoint_compat_destroy_comm, communicator)
        return communicator
    initialize._speco_stateless_store_compat = True
    initialize._speco_checkpoint_control_broadcast = checkpoint_control_broadcast
    initialize._speco_checkpoint_reclaim_cache = checkpoint_reclaim_cache
    checkpoint_module.stateless_init_process_group = initialize
    return True
from verl_speco.trainer.distributed_batches import enumerate_packed_response_parts

def validate_online_collection_geometry(*, model_config, prompt_token_count, response_tokens, sample_count, max_model_len):
    if any((type(value) is not int or value < 1 for value in (prompt_token_count, response_tokens, sample_count, max_model_len))):
        raise ValueError('online geometry requires explicit positive integer lengths')
    block = model_config['block_size']
    if response_tokens - 1 < block:
        raise ValueError('shifted online response must contain at least one complete draft block')
    if prompt_token_count + response_tokens > max_model_len:
        raise ValueError('online response exceeds the actual teacher context capacity')
    lengths = torch.full((sample_count,), response_tokens - 1, dtype=torch.int64, device='cpu')
    starts, ends, samples = enumerate_packed_response_parts(response_length=lengths, seq_length=model_config['max_position_embeddings'], draft_train_length=block * model_config['num_anchors'], block_size=block, num_swa_chunk=model_config['num_swa_chunk'], adaptive_num_swa_chunk=True)
    if set(samples.tolist()) != set(range(sample_count)):
        raise ValueError('online collection cannot cover every response under the draft SWA chunk cap')
    for sample in range(sample_count):
        indices = (samples == sample).nonzero(as_tuple=True)[0]
        if int(starts[indices[0]]) != 0 or int(ends[indices[-1]]) != response_tokens - 1 or any((int(ends[a]) != int(starts[b]) for a, b in zip(indices, indices[1:]))):
            raise ValueError('online SWA plan does not cover a contiguous complete response')
    return dict(sample_count=sample_count, prompt_tokens=prompt_token_count, response_tokens=response_tokens, supervised_tokens_per_response=response_tokens - 1, block_size=block, response_parts=samples.numel(), complete_response_coverage=True)
from verl_speco.runtime.artifacts import publish_json, published_json_paths, read_published_json
from verl_speco.trainer.distributed_checkpoint import JsonStepJournal, optimizer_transaction_record_from_dict as _record_from_dict, optimizer_transaction_record_to_dict as _record_to_dict

class PublishedOnlineStepJournal(JsonStepJournal):

    def load(self, optimizer_step_id):
        path = self._path(optimizer_step_id)
        with self._lock:
            if not published_json_paths(self.root, path.name):
                return None
            return _record_from_dict(read_published_json(path))

    def store(self, record):
        record.validate()
        with self._lock:
            publish_json(self._path(record.transaction.optimizer_step_id), _record_to_dict(record))

def online_optimizer_audit_tensor_identity(value):
    if hasattr(value, 'to_local'):
        local = value.to_local()
        if local.shape != value.shape:
            raise ValueError('online audit requires full DP-only replicated tensors')
        value = local
    if not isinstance(value, torch.Tensor):
        raise ValueError('online Adam state is not a tensor')
    cpu = value.detach().cpu().contiguous()
    return dict(shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=hashlib.sha256(memoryview(cpu.reshape(-1).view(torch.uint8).numpy())).hexdigest())

def audit_online_optimizer(module, optimizer, *, expected_step):
    if type(expected_step) is not int or expected_step < 1:
        raise ValueError('online audit requires a committed optimizer step')
    parameters = {name: p for name, p in module.named_parameters() if p.requires_grad}
    if not parameters:
        raise ValueError('online audit has no trainable parameters')
    states = {}
    weights = {}
    if set(optimizer.state) != set(parameters.values()):
        raise ValueError('online optimizer state does not cover every trainable parameter exactly')
    for name, p in sorted(parameters.items()):
        weights[name] = online_optimizer_audit_tensor_identity(p)
        state = optimizer.state[p]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('online audit requires complete Adam state')
        if not isinstance(state['step'], torch.Tensor) or state['step'].numel() != 1 or state['step'].item() != expected_step:
            raise ValueError('online Adam step differs from committed receipt')
        for field, value in sorted(state.items()):
            states[name + '.' + field] = online_optimizer_audit_tensor_identity(value)
    return dict(optimizer_step=expected_step, trainable_parameters=len(weights), optimizer_tensors=len(states), parameters=weights, optimizer_state=states)
import os
from urllib.parse import urlsplit
from verl_speco.transport.native_transfer import AscendFeatureTransport
from verl_speco.transport.native_transfer import NativeDescriptorStore, native_descriptor_store_DESCRIPTOR_FIELD as _DESCRIPTOR_FIELD, native_descriptor_store_MAX_DESCRIPTOR_BYTES as _MAX_DESCRIPTOR_BYTES

class OnlineSourcePool:

    def __init__(self, *, run_id, generation, endpoint, metadata_server, device, max_slots=32, slot_bytes=32 * 1024 ** 2, engine_factory=None, local_host_mirror=False):
        if type(max_slots) is not int or not 1 <= max_slots <= 128 or type(slot_bytes) is not int or (slot_bytes < 2 * 1024 ** 2) or slot_bytes % (2 * 1024 ** 2) or (max_slots * slot_bytes > 8 * 1024 ** 3):
            raise ValueError('invalid bounded online source pool geometry')
        if type(local_host_mirror) is not bool:
            raise ValueError('online source host mirror must be boolean')
        parsed = urlsplit('//' + endpoint)
        count = math.ceil(max_slots / 16)
        if not parsed.hostname or ':' in parsed.hostname or parsed.port is None or (not 1 <= parsed.port <= 65535 - count + 1):
            raise ValueError('online source pool requires bounded IPv4 endpoint ports')
        self.run_id, self.generation = (run_id, generation)
        self.device = torch.device(device)
        self.max_slots, self.slot_bytes = (max_slots, slot_bytes)
        self.max_bytes = max_slots * slot_bytes
        self.children, self.leases = ([], {})
        self._closed = False
        try:
            for index in range(count):
                slots = min(16, max_slots - index * 16)
                child = AscendFeatureTransport(run_id=run_id, generation=generation, endpoint=f'{parsed.hostname}:{parsed.port + index}', metadata_server=metadata_server, device=device, max_slots=slots, max_bytes=slots * slot_bytes, engine_factory=engine_factory, local_host_mirror=local_host_mirror)
                self.children.append(child)
                for local in range(slots):
                    child.preallocate_source(slot_bytes, slot_id=local)
        except BaseException:
            self.close()
            raise

    def available_slot(self):
        if self._closed:
            raise RuntimeError('online source pool is closed')
        used = set(self.leases.values())
        for slot in range(self.max_slots):
            if slot not in used:
                return slot
        raise RuntimeError('native online source slots are still leased')

    def pack(self, fields, *, lease_id, slot_id):
        if self._closed:
            raise RuntimeError('online source pool is closed')
        if type(slot_id) is not int or not 0 <= slot_id < self.max_slots or (not isinstance(lease_id, str)) or (not lease_id):
            raise ValueError('invalid online source lease or slot')
        if lease_id in self.leases or slot_id in self.leases.values():
            raise RuntimeError('native online source slot or lease is still leased')
        size = 256
        for name in sorted(fields):
            size = math.ceil(size / 256) * 256 + fields[name].numel() * fields[name].element_size()
        if size > self.slot_bytes:
            raise RuntimeError('online payload exceeds fixed source slot capacity')
        child = self.children[slot_id // 16]
        descriptor = child.pack(fields, lease_id=lease_id, slot_id=slot_id % 16)
        self.leases[lease_id] = slot_id
        return descriptor

    def _validate(self, descriptor):
        if not isinstance(descriptor, dict):
            raise ValueError('invalid online source descriptor')
        slot = self.leases.get(descriptor.get('lease_id'))
        if slot is None:
            raise ValueError('online source descriptor is not owned')
        child = self.children[slot // 16]
        child._validate(descriptor)
        if descriptor['segment'] != child.endpoint or descriptor['slot_id'] != slot % 16 or descriptor != child._slots[slot % 16][1]:
            raise ValueError('online source descriptor identity differs from retained lease')

    def release(self, *, slot_id, lease_id):
        slot = self.leases.get(lease_id)
        if slot is None or type(slot_id) is not int or slot_id != slot % 16:
            raise ValueError('online source release identity differs from retained lease')
        self.children[slot // 16].release(slot_id=slot_id, lease_id=lease_id)
        del self.leases[lease_id]

    def close(self):
        errors = []
        for child in self.children:
            try:
                child.close()
            except BaseException as error:
                errors.append(error)
        self._closed = True
        if errors:
            raise errors[0]
        self.leases.clear()

class OnlineDescriptorStore(NativeDescriptorStore):

    def put(self, key, partition, fields, tag):
        metadata, checksum = self._metadata(key, tag)
        if self.get_tag(key, partition) is not None or (partition, key) in self._sources:
            raise ValueError('native TQ source key is already published')
        self._validate_fields(fields, metadata, checksum)
        slot = self.transport.available_slot()
        descriptor = self.transport.pack({name: value.to(self.transport.device) for name, value in fields.items()}, lease_id=key, slot_id=slot)
        self._sources[partition, key] = descriptor
        try:
            encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()
            if len(encoded) > _MAX_DESCRIPTOR_BYTES:
                raise ValueError('native TQ descriptor exceeds control limit')
            self._tq.kv_put(key=key, partition_id=partition, fields={_DESCRIPTOR_FIELD: torch.tensor(list(encoded), dtype=torch.uint8)}, tag=dict(tag))
        except BaseException:
            if self.get_tag(key, partition) is None:
                self.transport.release(slot_id=descriptor['slot_id'], lease_id=key)
                del self._sources[partition, key]
            raise

def online_pool_geometry():
    return (int(os.environ.get('SPECO_ONLINE_SOURCE_SLOTS', '32')), int(os.environ.get('SPECO_ONLINE_SOURCE_SLOT_BYTES', str(32 * 1024 ** 2))))

def admit_pool_collection(pool, *, sample_count, payload_bytes_per_sample, capacity_bytes):
    if type(sample_count) is not int or sample_count < 1 or type(payload_bytes_per_sample) is not int or (payload_bytes_per_sample < 1) or (type(capacity_bytes) is not int) or (capacity_bytes < 1):
        raise ValueError('online collection requires explicit positive capacity bounds')
    if sample_count > pool.max_slots - len(pool.leases) or payload_bytes_per_sample + 1024 > pool.slot_bytes or sample_count * payload_bytes_per_sample > capacity_bytes:
        raise RuntimeError('online collection exceeds available native or control capacity')
    return dict(native_source_slots=pool.max_slots, native_available_slots=pool.max_slots - len(pool.leases), native_source_bytes=pool.max_bytes, sample_count=sample_count, payload_bytes_per_sample=payload_bytes_per_sample)
from verl_speco.producer.capture import SpecoMultiNodeCaptureWorkerExtension as SpecoMultiNodeCaptureWorkerExtension

class OnlinePublisherContext(SpecoMultiNodeCaptureWorkerExtension):

    def _speco_get_tq_publisher(self, *, partition, ray_address, capacity_bytes, producer_endpoint, control_actor_name=''):
        slots, slot_bytes = online_pool_geometry()
        local_host_mirror = os.environ.get('SPECO_COLOCATED_FEATURE_HOST_MIRROR') == '1'
        settings = (partition, ray_address, capacity_bytes, producer_endpoint, control_actor_name, slots, slot_bytes, os.environ.get('SPECO_NATIVE_SOURCE_ENDPOINT'), os.environ.get('SPECO_NATIVE_METADATA_SERVER'), os.environ.get('SPECO_RUN_ID'), os.environ.get('SPECO_GENERATION'), local_host_mirror)
        existing = getattr(self, '_speco_tq_publisher', None)
        if existing is not None:
            if settings != self._speco_tq_settings:
                raise ValueError('online publisher settings changed within one process')
            return existing
        if not partition or not ray_address or (not producer_endpoint) or (not control_actor_name) or (type(capacity_bytes) is not int) or (capacity_bytes < 1) or (os.environ.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1'):
            raise ValueError('online source pool requires an owned native shared publisher')
        import ray
        from verl_speco.runtime.services import import_transfer_queue
        from verl_speco.transport.control import RayBoundedFeaturePublisher
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace='transfer_queue')
        ray.get_actor('TransferQueueController', namespace='transfer_queue')
        control = ray.get_actor(control_actor_name, namespace='transfer_queue')
        tq = import_transfer_queue()
        tq.init()
        pool = OnlineSourcePool(run_id=os.environ['SPECO_RUN_ID'], generation=int(os.environ['SPECO_GENERATION']), endpoint=os.environ['SPECO_NATIVE_SOURCE_ENDPOINT'], metadata_server=os.environ['SPECO_NATIVE_METADATA_SERVER'], device=f'npu:{torch.npu.current_device()}', max_slots=slots, slot_bytes=slot_bytes, local_host_mirror=local_host_mirror)
        try:
            store = OnlineDescriptorStore(tq=tq, transport=pool)
            publisher = RayBoundedFeaturePublisher(store=store, control_actor=control, partition=partition, producer_endpoint=producer_endpoint)
        except BaseException:
            pool.close()
            raise
        self._speco_native_store = store
        self._speco_tq_publisher = publisher
        self._speco_tq_settings = settings
        return publisher

def online_publisher_context(worker):
    context = getattr(worker, '_speco_online_publisher_context', None)
    if context is None:
        context = OnlinePublisherContext()
        context.rank = worker.rank
        context.model_runner = worker.model_runner
        worker._speco_online_publisher_context = context
    return context
from verl_speco.transport.streaming import TrainingTransaction

def online_stream_transactions(samples, leases, *, stream_scope, generation, target_revision, committed_cursor):
    if type(committed_cursor) is not int or committed_cursor < 0:
        raise ValueError('online stream requires a committed parent cursor')
    keys = tuple((row[0] for row in samples))
    if not keys or len(keys) != len(leases) or len(set(keys)) != len(keys):
        raise ValueError('online stream batch requires unique aligned sample keys')
    transactions = []
    for offset, ((key, metadata, _), lease) in enumerate(zip(samples, leases), start=1):
        if lease.key != key or metadata.run_id != stream_scope or metadata.generation != generation or (metadata.target_revision != target_revision) or (lease.generation != generation) or (lease.target_revision != target_revision):
            raise ValueError('online stream batch changed run, generation, target or lease key')
        transactions.append(TrainingTransaction('online-tx-' + hashlib.sha256(key.encode()).hexdigest(), (lease.lease_id,), (key,), stream_scope, committed_cursor + offset, generation, target_revision))
    return tuple(transactions)

def plan_digest(payload):
    value = {key: val for key, val in payload.items() if key != 'online_plan_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

class OnlineWorkerPlanFence:

    def __init__(self, *, run_id, generation, worker_id, incarnation):
        self.run_id = run_id
        self.generation = generation
        self.worker_id = worker_id
        self.incarnation = incarnation
        self.payload = None
        self.phase = 'idle'
        self.completed = {}

    def bind(self, payload):
        versions = payload.get('online_versions', {})
        snapshot = payload.get('worker_snapshots', {}).get(self.worker_id, {})
        if versions.get('run_id') != self.run_id or versions.get('generation') != self.generation or snapshot.get('worker_incarnation') != self.incarnation or (not payload.get('plan_id')) or (versions.get('plan_id') != payload['plan_id']) or (versions.get('target_revision') != payload.get('required_target_version')) or (versions.get('data_version') != payload.get('data_version')) or (versions.get('rl_step') != payload.get('source_global_step')) or (payload.get('execution_strategy') != 'sync') or (payload.get('data_source') != 'transfer_queue') or (payload.get('online_plan_sha256') != plan_digest(payload)):
            raise ValueError('online worker plan identity, incarnation or digest differs')
        keys = payload.get('selected_keys')
        if payload.get('launch') is not True or not isinstance(keys, list) or (not keys) or (len(set(keys)) != len(keys)) or any((not isinstance(key, str) or not key for key in keys)) or (type(payload.get('max_batches')) is not int) or (payload['max_batches'] < 1) or (payload.get('min_batches') != 1):
            raise ValueError('online functional executor requires one feasible committed global batch')
        if self.phase in ('aborted', 'updating'):
            raise RuntimeError('uncertain optimizer owner requires durable generation recovery')
        if self.payload == payload:
            return
        if self.phase not in ('idle', 'committed'):
            raise RuntimeError('online worker already owns a different pending plan')
        if payload['plan_id'] in self.completed:
            raise RuntimeError('completed online plan ID was reused with different bindings')
        self.payload = deepcopy(payload)
        self.phase = 'bound'

    def require(self, payload, phases):
        if self.payload != payload or plan_digest(payload) != payload.get('online_plan_sha256'):
            raise RuntimeError('online worker received an unbound or mutated plan')
        if self.phase not in phases:
            raise RuntimeError('online worker execution is out of order: ' + self.phase)

    def commit(self, receipt):
        if self.phase != 'updating':
            raise RuntimeError('online worker cannot commit before optimizer execution')
        self.completed[self.payload['plan_id']] = deepcopy(receipt)
        self.phase = 'committed'
from verl_speco.trainer.scheduler.schedule_types import TrainingDataStatus, TrainingPlan

def online_worker_group_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

class MultiNodeOnlineWorkerGroupExecutor:

    def __init__(self, *, worker_group, resolve: Callable, run_id: str, generation: int, expected_worker_ids: tuple[str, ...], rollout_nodes: frozenset[str], colocated_placements=None):
        if not run_id or type(generation) is not int or generation < 1 or (not expected_worker_ids) or (len(set(expected_worker_ids)) != len(expected_worker_ids)):
            raise ValueError('WorkerGroup requires valid run/generation and unique expected worker IDs')
        self.group = worker_group
        self.resolve = resolve
        self.run_id = run_id
        self.generation = generation
        self.expected_ids = frozenset(expected_worker_ids)
        self.binding = None
        self.plan_sha256 = None
        colocated = None
        if colocated_placements is not None:
            from verl_speco.integration.multinode_recovery import validate_shared_cards
            checked = validate_shared_cards(run_id=run_id, generation=generation, actor=colocated_placements['actor'], draft=colocated_placements['draft'])
            if checked != colocated_placements:
                raise ValueError('online draft requires the exact independently validated shared-card binding')
            colocated = {str(row['rank']): row for row in checked['draft']}
            if set(colocated) != self.expected_ids:
                raise ValueError('shared-card draft placement lacks the full DP-only WorkerGroup')
        capabilities = self._rows(self.group.speco_online_capabilities())
        self._check_members(capabilities)
        self.incarnations = {}
        for row in capabilities:
            if row.get('run_id') != run_id or row.get('generation') != generation or row.get('engine') != 'veomni' or (row.get('wrapper') != 'fsdp2') or (row.get('initialized') is not True) or (row.get('online_fencing') is not True) or (row.get('hidden_transport') not in ('native_tq', 'transfer_queue')):
                raise ValueError('WorkerGroup lacks initialized VeOmni DP-only online fencing capability')
            mesh = row.get('mesh', {})
            expected_mesh = dict(dp_replicate=len(expected_worker_ids), dp_shard=1, tp=1, pp=1, sp=1, ep=1)
            if mesh != expected_mesh or any((type(value) is not int for value in mesh.values())):
                raise ValueError('online draft WorkerGroup must use a DP-only mesh, retaining FSDP2')
            if not row.get('node_id') or (row['node_id'] in rollout_nodes and colocated is None):
                raise ValueError('draft WorkerGroup shares a target rollout node')
            if colocated is not None and row.get('colocated_placement') != colocated[row['worker_id']]:
                raise ValueError('draft capability differs from its observed physical shared-card owner')
            if not isinstance(row.get('worker_incarnation'), str) or not row['worker_incarnation']:
                raise ValueError('draft worker incarnation is missing')
            self.incarnations[row['worker_id']] = row['worker_incarnation']

    def _rows(self, value):
        rows = self.resolve(value)
        if not isinstance(rows, list) or any((not isinstance(row, dict) for row in rows)):
            raise RuntimeError('WorkerGroup must return explicit per-rank metadata receipts')
        return rows

    def _check_members(self, rows):
        if len(rows) != len(self.expected_ids) or {row.get('worker_id') for row in rows} != self.expected_ids:
            raise RuntimeError('WorkerGroup returned missing, duplicate or unexpected ranks')

    def _check_identity(self, rows, *, plan_bound=False):
        self._check_members(rows)
        for row in rows:
            if row.get('run_id') != self.run_id or row.get('generation') != self.generation or row.get('worker_incarnation') != self.incarnations[row['worker_id']]:
                raise RuntimeError('WorkerGroup run/generation/incarnation changed')
            if plan_bound and row.get('online_plan_sha256') != self.plan_sha256:
                raise RuntimeError('WorkerGroup returned a foreign online plan digest')

    def bind_online_plan(self, binding: OnlineTrainingBinding):
        if binding.versions.run_id != self.run_id or binding.versions.generation != self.generation:
            raise ValueError('online plan belongs to a different run/generation')
        if binding.plan.launch:
            snapshots = binding.plan.worker_snapshots or {}
            if set(snapshots) != self.expected_ids or any((snapshots[rank].get('worker_incarnation') != incarnation for rank, incarnation in self.incarnations.items())):
                raise ValueError('online plan WorkerGroup incarnation snapshots differ')
        payload = binding.to_worker_payload()
        self.binding = binding
        self.plan_sha256 = online_worker_group_digest(payload)
        rows = self._rows(self.group.speco_bind_online_plan(dict(payload, online_plan_sha256=self.plan_sha256)))
        self._check_identity(rows, plan_bound=True)
        if any((row.get('bound') is not True for row in rows)):
            raise RuntimeError('a draft rank did not bind the online plan')

    def _payload(self, plan: TrainingPlan):
        if self.binding is None or plan != self.binding.plan:
            raise RuntimeError('draft execution requires the bound scheduler plan')
        payload = self.binding.to_worker_payload()
        if online_worker_group_digest(payload) != self.plan_sha256:
            raise RuntimeError('bound scheduler plan mutated after WorkerGroup acknowledgement')
        return dict(payload, online_plan_sha256=self.plan_sha256)

    def get_training_data_status(self, *, sample_last_n_steps: int, require_full_batch: bool):
        rows = self._rows(self.group.get_drafter_training_data_status(sample_last_n_steps=sample_last_n_steps, require_full_batch=require_full_batch, run_id=self.run_id, generation=self.generation))
        self._check_identity(rows)
        if any((row.get('available') is not True for row in rows)):
            raise RuntimeError('a draft rank cannot inspect committed TQ feature data')
        return [TrainingDataStatus.from_mapping(row) for row in rows]

    def prepare_training(self, plan: TrainingPlan):
        rows = self._rows(self.group.speco_prepare_online_target(self._payload(plan)))
        self._check_identity(rows, plan_bound=True)
        if any((row.get('target_version') != plan.required_target_version or row.get('prepared') is not True for row in rows)):
            raise RuntimeError('a draft rank did not prepare the frozen collection target head')
        return {'drafter/online_worker_group_prepared': len(rows)}

    def preflight_training(self, plan: TrainingPlan):
        rows = self._rows(self.group.speco_preflight_drafter_training(self._payload(plan)))
        self._check_identity(rows, plan_bound=True)
        return rows

    def submit_training(self, plan: TrainingPlan):
        return self.group.train_drafter(self._payload(plan))

    def resolve_training(self, submission):
        rows = self._rows(submission)
        self._check_identity(rows, plan_bound=True)
        return rows

    def abort_training_preflight(self, plan: TrainingPlan):
        rows = self._rows(self.group.speco_abort_drafter_training_preflight(self._payload(plan)))
        self._check_identity(rows, plan_bound=True)
        return rows

    def activate_training_workers(self):
        return [dict(worker_id=rank, initialized=True) for rank in sorted(self.expected_ids)]

class ServingAcceptanceCounters:

    def __init__(self):
        self.drafts = 0
        self.accepted = 0

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        stats = getattr(scheduler_stats, 'spec_decoding_stats', None)
        if stats is None:
            return
        drafts, accepted = (stats.num_drafts, stats.num_accepted_tokens)
        if type(drafts) is not int or type(accepted) is not int or drafts < 0 or (accepted < 0):
            raise ValueError('invalid actual vLLM speculative counters')
        if accepted and (not drafts):
            raise ValueError('accepted tokens have no actual proposals')
        self.drafts += drafts
        self.accepted += accepted

    def log_engine_initialized(self):
        pass

    def log(self):
        pass

    def record_sleep_state(self, is_awake, level):
        pass

    def snapshot(self):
        return (self.drafts, self.accepted)

    def since(self, baseline):
        drafts, accepted = (self.drafts - baseline[0], self.accepted - baseline[1])
        if drafts < 0 or accepted < 0:
            raise RuntimeError('serving acceptance counters moved backwards')
        return dict(drafts=drafts, accepted_tokens=accepted)

def summarize_serving_acceptance(rows):
    if not rows:
        raise ValueError('completed request acceptance counters are required')
    drafts = accepted = 0
    for row in rows:
        if set(row) != {'drafts', 'accepted_tokens'}:
            raise ValueError('invalid serving acceptance fields')
        d, a = (row['drafts'], row['accepted_tokens'])
        if type(d) is not int or type(a) is not int or d < 0 or (a < 0) or (a and (not d)):
            raise ValueError('invalid serving acceptance counts')
        drafts += d
        accepted += a
    return {'drafter/spec_decode/mean_acceptance_length': 1.0 + accepted / drafts if drafts else None, 'drafter/spec_decode/drafts': drafts, 'drafter/spec_decode/accepted_tokens': accepted, 'drafter/spec_decode/acceptance_scope': 'actual_vllm_verification_completed_step', 'drafter/spec_decode/includes_target_token': True}
