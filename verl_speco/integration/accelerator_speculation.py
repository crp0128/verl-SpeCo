# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
dspark_config_TARGET = 'openpangu_v2_vl_moe'
dspark_config_DEDICATED_WORKERS = frozenset(('verl_speco.integration.accelerator_adapters.SpecoMultiNodeDSparkWorker', 'verl_speco.integration.accelerator_adapters.SpecoMultiNodeTargetUpdateWorker'))
dspark_config_NATIVE_FAMILIES = ('llama', 'qwen', 'minicpm', 'gpt_oss', 'hunyuan_vl', 'hunyuan_v1_dense', 'afmoe', 'nemotron_h', 'deepseek_v2', 'deepseek_v3', 'kimi_k2', 'kimi_k25', 'minimax_m2', 'gemma4')

@contextmanager
def dspark_config_verified_family_constant(function):
    original = function.__code__
    constants = list(original.co_consts)
    matches = [index for index, value in enumerate(constants) if isinstance(value, tuple) and value in (dspark_config_NATIVE_FAMILIES, dspark_config_NATIVE_FAMILIES + (dspark_config_TARGET,))]
    if len(matches) != 1:
        if not matches and function.__module__ == 'vllm_ascend.patch.platform.patch_minimax_m2_config' and (function.__name__ == '_patched_verify_args') and (getattr(function, '_vllm_ascend_minimax_eagle3_patched', False) is True) and (original.co_freevars == ('inner_verify',)) and (function.__closure__ is not None):
            inner = function.__closure__[0].cell_contents
            if callable(inner) and hasattr(inner, '__code__') and (inner is not function):
                with dspark_config_verified_family_constant(inner):
                    yield
                return
        raise RuntimeError('installed vLLM auxiliary target validator differs from the verified layout')
    constants[matches[0]] = dspark_config_NATIVE_FAMILIES + (dspark_config_TARGET,)
    modified = original.replace(co_consts=tuple(constants))
    function.__code__ = modified
    try:
        yield
    finally:
        if function.__code__ is not modified:
            raise RuntimeError('auxiliary target validator changed during dedicated engine construction')
        function.__code__ = original

@contextmanager
def dedicated_speco_dspark_config(engine_kwargs):
    if engine_kwargs.get('worker_cls') not in dspark_config_DEDICATED_WORKERS or engine_kwargs.get('enforce_eager') is not True:
        raise ValueError('target model target compatibility is restricted to its dedicated eager DSpark worker')
    spec = engine_kwargs.get('speculative_config', {})
    if spec.get('method') != 'dflash' or spec.get('num_speculative_tokens') != 8 or spec.get('enforce_eager') is not True:
        raise ValueError('dedicated target model compatibility requires the verified K=8 eager carrier')
    target = json.loads((Path(engine_kwargs['model']) / 'config.json').read_text())
    draft = json.loads((Path(spec['model']) / 'config.json').read_text())
    if target.get('model_type') != dspark_config_TARGET or target.get('architectures') != ['OpenPanguV2VLForConditionalGeneration'] or draft.get('markov_head_type') != 'vanilla' or (draft.get('target_layer_ids') != [1, 12, 22, 32, 43]) or (not draft.get('speco_snapshot_checksum')):
        raise ValueError('target/draft geometry is outside the verified target model DSpark binding')
    from vllm.config.speculative import SpeculativeConfig
    with dspark_config_verified_family_constant(SpeculativeConfig._verify_args):
        yield

def select_target_graph_metadata(metadata, layer_names):
    """Select declared target owners in capture order, leaving draft keys alone."""
    if not isinstance(metadata, dict) or not isinstance(layer_names, tuple) or (not layer_names) or (len(set(layer_names)) != len(layer_names)) or any((not isinstance(name, str) or not name for name in layer_names)) or any((name not in metadata for name in layer_names)):
        raise ValueError('graph metadata lacks exact unique target attention owners')
    return {name: metadata[name] for name in layer_names}

@contextmanager
def dedicated_speco_dspark_graph_config(kwargs):
    if kwargs.get('worker_cls') != 'verl_speco.integration.accelerator_adapters.SpecoMultiNodeDSparkGraphWorker' or kwargs.get('enforce_eager') is not False or kwargs.get('compilation_config', {}).get('cudagraph_mode') != 'FULL_DECODE_ONLY':
        raise ValueError('DSpark graph diagnostic requires explicit target FULL_DECODE_ONLY')
    checked = dict(kwargs, enforce_eager=True, worker_cls='verl_speco.integration.accelerator_adapters.SpecoMultiNodeTargetUpdateWorker')
    with dedicated_speco_dspark_config(checked):
        yield

def accepted_graph_contexts(metadata, layer_names, *, request_count, query, sink_len):
    if type(request_count) is not int or request_count < 1 or type(query) is not int or (query < 1) or (type(sink_len) is not int) or (sink_len < 0):
        raise ValueError('invalid accepted graph request geometry')
    selected = select_target_graph_metadata(metadata, layer_names)
    first = selected[layer_names[0]]
    seq_lens = getattr(first, 'seq_lens', None)
    query_start = getattr(first, 'query_start_loc', None)
    if seq_lens is None or query_start is None or seq_lens.ndim != 1 or (query_start.ndim != 1) or (seq_lens.numel() < request_count) or (query_start.numel() < request_count + 1):
        raise ValueError('accepted graph lacks native target device coordinates')
    bounds = query_start[:request_count + 1].tolist()
    lengths = seq_lens[:request_count].tolist()
    if bounds[0] != 0 or any((type(v) is not int for v in bounds + lengths)) or any((bounds[i + 1] - bounds[i] != query for i in range(request_count))) or any((length < 0 for length in lengths)):
        raise ValueError('accepted graph target query is not uniform K+1')
    return [max(length - query - sink_len, 0) for length in lengths]
import torch

def speculative_graph_conv(weight, ring, x, slots, contexts, *, mask_start):
    """Convolve a uniform [batch, query, channels] verification on device.

    ``ring`` is [slot, K + left_pad, channels]; slots and absolute contexts are
    stable device buffers updated before replay. Negative history is zero.
    Callers must validate request ownership and rejection bounds before entry.
    No CPU tensor reads or request-dependent Python branches occur here.
    """
    batch, query, channels = x.shape
    pad = weight.shape[-1] - 1
    window = ring.shape[1]
    if weight.shape != (channels, 1, pad + 1) or pad < 1 or ring.ndim != 3 or (ring.shape[2] != channels) or (query < 1) or (query > window - pad + 1) or (slots.shape != (batch,)) or (contexts.shape != (batch,)) or (slots.dtype != torch.int64) or (contexts.dtype != torch.int64) or any((t.device != x.device for t in (weight, ring, slots, contexts))) or (ring.dtype != x.dtype) or (weight.dtype != x.dtype):
        raise ValueError('invalid speculative graph convolution geometry')
    positions = contexts[:, None] + torch.arange(-pad, 0, device=x.device)
    history = ring[slots[:, None], positions.remainder(window)]
    history = torch.where((positions >= 0)[:, :, None], history, 0)
    values = torch.cat((history, x), dim=1)
    taps = weight[:, 0].transpose(0, 1)
    windows = torch.stack([values[:, tap:tap + query] for tap in range(pad + 1)], dim=2)
    output = (windows.reshape(batch * query, pad + 1, channels) * taps[None]).sum(dim=1).reshape(batch, query, channels)
    absolute = contexts[:, None] + torch.arange(query, device=x.device)
    output = output * (absolute >= mask_start)[:, :, None].to(output.dtype)
    ring[slots[:, None], absolute.remainder(window)] = x
    return output

class SpeculativeHistoryBoundaries:
    """Host validation before graph execution, separate from tensor capture."""

    def __init__(self, *, slots, block_size, left_pad):
        if any((type(v) is not int or v < 1 for v in (slots, block_size, left_pad))):
            raise ValueError('positive history capacity and rejection bound required')
        self.slots = slots
        self.block_size = block_size
        self.left_pad = left_pad
        self.boundaries = {}

    def prepare(self, requests, slots, contexts, query_lengths):
        """Validate the whole batch atomically, then advance prepared boundaries.

        A failed device forward invalidates its prepared boundaries via clear().
        Fresh context zero is the sole permitted way to recycle an owned slot.
        """
        if not len(requests) == len(slots) == len(contexts) == len(query_lengths):
            raise ValueError('history routing lengths differ')
        if len(set(slots)) != len(slots) or len(set(requests)) != len(requests):
            raise ValueError('history routing must have unique requests and slots')
        updates = {}
        for request, slot, context, query in zip(requests, slots, contexts, query_lengths):
            if not isinstance(request, str) or not request or type(slot) is not int or (not 0 <= slot < self.slots) or (type(context) is not int) or (context < 0) or (type(query) is not int) or (not 1 <= query <= self.block_size + 1):
                raise ValueError('invalid speculative request boundary')
            prior = self.boundaries.get(slot)
            if context:
                if prior is None or prior[0] != request:
                    raise RuntimeError('speculative history has no owned request boundary')
                end = prior[1]
                if context > end or end - context > self.block_size:
                    raise RuntimeError('speculative rejection exceeds retained boundary')
            updates[slot] = (request, context + query)
        self.boundaries.update(updates)

    def clear(self, slots):
        for slot in slots:
            self.boundaries.pop(slot, None)
import copy
from types import MethodType

def greedy_dspark_block(base_logits, anchor_token_ids, markov_head):
    """Condition row zero on the bonus/anchor, then each row on its predecessor."""
    if not isinstance(base_logits, torch.Tensor) or base_logits.ndim != 3 or base_logits.shape[1] < 1 or (base_logits.shape[2] < 1) or (not base_logits.is_floating_point()):
        raise ValueError('DSpark base logits require [batch, K, full vocabulary]')
    if not isinstance(anchor_token_ids, torch.Tensor) or anchor_token_ids.shape != (base_logits.shape[0],) or anchor_token_ids.dtype not in (torch.int32, torch.int64) or (anchor_token_ids.device != base_logits.device):
        raise ValueError('DSpark anchor IDs must match batch/device and have integer dtype')
    vocab = base_logits.shape[-1]
    if bool(((anchor_token_ids < 0) | (anchor_token_ids >= vocab)).any()):
        raise ValueError('DSpark anchor is outside the full target vocabulary')
    previous = anchor_token_ids.long()
    result = []
    for index in range(base_logits.shape[1]):
        if callable(getattr(markov_head, 'compute_step_bias', None)):
            bias = markov_head.compute_step_bias(previous)
        else:
            value = markov_head(previous)
            bias = value[0] if isinstance(value, tuple) else value
        if not isinstance(bias, torch.Tensor) or bias.shape != (base_logits.shape[0], vocab) or bias.device != base_logits.device or (not bias.is_floating_point()):
            raise RuntimeError('Markov head must return gathered full-vocabulary logits')
        previous = (base_logits[:, index].float() + bias.float()).argmax(dim=-1)
        result.append(previous)
    return torch.stack(result, dim=1)

class DSparkKQueryEagerAdapter:
    """Reuse native slot/KV preparation with K-1 masks and sample all K rows.

    A shallow proxy changes only the native first-pass query count; all device
    buffers are shared. The real proposer's K remains unchanged, so native
    verification still receives K proposed tokens. Graphs, PCP/DCP, split LM
    heads and non-vanilla Markov heads need separate implementations.
    """

    def __init__(self, proposer, *, block_size: int, loaded_parameter_names: frozenset[str], lmhead_tp_enabled: bool=False):
        if type(block_size) is not int or block_size < 1:
            raise ValueError('DSpark block size must be a positive integer')
        if hasattr(proposer, '_speco_dspark_eager_adapter'):
            raise RuntimeError('DSpark eager adapter is already installed')
        if getattr(proposer, 'method', None) != 'dflash' or proposer.num_speculative_tokens != block_size or (not proposer.parallel_drafting):
            raise ValueError('DSpark adapter requires parallel DFlash carrier with matching K')
        if proposer.use_cuda_graph or lmhead_tp_enabled or proposer.pcp_size != 1 or (proposer.dcp_size != 1) or proposer.uses_mrope:
            raise ValueError('DSpark functional adapter requires eager TP-only draft and unsplit LM head')
        config = proposer.model.config
        if getattr(config, 'markov_head_type', None) != 'vanilla':
            raise ValueError('DSpark eager adapter currently supports vanilla Markov heads only')
        self.markov_head = getattr(proposer.model.model, 'markov_head', None)
        required = {'markov_head.markov_w1.weight', 'markov_head.markov_w2.weight'}
        if self.markov_head is None or not isinstance(loaded_parameter_names, frozenset) or (not required.issubset(loaded_parameter_names)):
            raise ValueError('both DSpark Markov weights must be loaded and verified before proposal')
        if getattr(proposer.model, 'draft_id_to_target_id', None) is not None:
            raise ValueError('DSpark adapter requires the full target vocabulary')
        self.proposer = proposer
        self.block_size = block_size
        self._native_first_pass = proposer.set_inputs_first_pass
        if getattr(self._native_first_pass, '__self__', None) is not proposer:
            raise ValueError('native first-pass slot preparation must be a bound method')
        self._anchors = None
        self._closed = False

    def set_inputs_first_pass(self, **kwargs):
        if self._closed:
            raise RuntimeError('DSpark eager adapter is closed')
        if self._anchors is not None:
            raise RuntimeError('previous DSpark query has not completed')
        proxy = copy.copy(self.proposer)
        proxy.num_speculative_tokens = self.block_size - 1
        result = self._native_first_pass.__func__(proxy, **kwargs)
        count, _mask_indices, cad, long_sequence = result
        if count != cad.num_reqs * self.block_size or long_sequence is not None:
            raise RuntimeError('native query preparation does not implement isolated K-query geometry')
        self.proposer._dflash_num_context = proxy._dflash_num_context
        anchors = kwargs['next_token_ids']
        if anchors.shape != (cad.num_reqs,):
            raise ValueError('DSpark anchor count differs from prepared requests')
        self._anchors = anchors.detach().clone()
        indices = torch.arange(count, dtype=torch.int32, device=self.proposer.device)
        return (count, indices, cad, None)

    @torch.inference_mode()
    def run_merged_draft(self, *, num_input_tokens, batch_size, token_indices_to_sample, target_positions, inputs_embeds, multi_steps_attn_metadata, num_tokens, is_prefill=None):
        if self._closed or self._anchors is None:
            raise RuntimeError('DSpark proposal has no prepared anchor identity')
        anchors = self._anchors
        try:
            count = batch_size * self.block_size
            expected = torch.arange(count, dtype=torch.int32, device=self.proposer.device)
            if anchors.shape != (batch_size,) or num_input_tokens < count or (not torch.equal(token_indices_to_sample, expected)):
                raise RuntimeError('DSpark sample rows differ from the prepared K-query identity')
            kwargs = self.proposer.build_model_inputs_first_pass(num_input_tokens)
            hidden = self.proposer.model(**kwargs)
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[0] < count:
                raise RuntimeError('DSpark carrier returned an unsupported hidden-state layout')
            logits = self.proposer.model.compute_logits(hidden[:count])
            vocab = int(self.proposer.model.config.vocab_size)
            if not isinstance(logits, torch.Tensor) or logits.shape != (count, vocab):
                raise RuntimeError('DSpark proposal requires gathered full-vocabulary LM logits')
            return greedy_dspark_block(logits.reshape(batch_size, self.block_size, vocab), anchors, self.markov_head)
        finally:
            self._anchors = None

    def close(self):
        if self._anchors is not None:
            raise RuntimeError('cannot restore a prepared DSpark query')
        self._closed = True

def install_dspark_k_query_eager(proposer, **kwargs):
    """Install on one quiescent eager proposer; return exact restoration callback."""
    adapter = DSparkKQueryEagerAdapter(proposer, **kwargs)
    names = ('set_inputs_first_pass', '_run_merged_draft', '_runnable')
    previous = {name: (name in proposer.__dict__, proposer.__dict__.get(name)) for name in names}

    def first_pass(_self, **values):
        return adapter.set_inputs_first_pass(**values)

    def run(_self, **values):
        return adapter.run_merged_draft(**values)
    proposer.set_inputs_first_pass = MethodType(first_pass, proposer)
    proposer._run_merged_draft = MethodType(run, proposer)
    proposer._runnable = proposer._run_merged_draft
    proposer._speco_dspark_eager_adapter = adapter

    def restore():
        adapter.close()
        for name, (existed, value) in previous.items():
            if existed:
                setattr(proposer, name, value)
            elif name in proposer.__dict__:
                delattr(proposer, name)
        delattr(proposer, '_speco_dspark_eager_adapter')
    return (adapter, restore)

class MultiNode92BAuxHiddenAdapter:

    def __init__(self, model, *, layer_ids: tuple[int, ...]):
        trunk = getattr(model, 'model', None)
        layers = getattr(trunk, 'layers', None)
        cfg = getattr(trunk, 'cfg', None)
        if trunk is None or layers is None or cfg is None or (not getattr(cfg, 'use_mhc', False)):
            raise ValueError('92B mHC text trunk is required')
        if not isinstance(layer_ids, tuple) or not layer_ids or any((type(index) is not int for index in layer_ids)) or (tuple(sorted(set(layer_ids))) != layer_ids):
            raise ValueError('selected decoder output layer IDs must be ascending unique integers')
        if getattr(trunk, 'start_layer', 0) != 0 or getattr(trunk, 'end_layer', len(layers)) != len(layers):
            raise ValueError('92B auxiliary output currently requires TP-only target, PP=1')
        self.hidden_size = int(cfg.hidden_size)
        if self.hidden_size <= 0:
            raise ValueError('target hidden size must be positive')
        self.model = model
        self.layer_ids = layer_ids
        self._active = False
        self._closed = False
        self._outputs = {}
        self._handles = []
        self._original_forward = model.forward
        sites = []
        for index in layer_ids:
            if index < 0 or index >= len(layers):
                raise ValueError('selected layer is outside the target trunk')
            projection = getattr(trunk, 'merge_mhc_module', None) if index == len(layers) - 1 else getattr(layers[index + 1], 'attn_mhc_module', None)
            if projection is None or not callable(getattr(projection, 'hc_pre', None)):
                raise ValueError('target layer is missing its verified mHC projection')
            sites.append((index, projection))
        for index, projection in sites:
            self._handles.append(layers[index].register_forward_hook(self._hook(index, projection)))

    def _hook(self, index, projection):

        def capture(_module, _args, output):
            if not self._active:
                return
            if index in self._outputs:
                raise RuntimeError('target decoder layer executed twice in one auxiliary forward')
            project = getattr(projection, '_speco_diagnostic_original_hc_pre', projection.hc_pre)
            hidden = project(output)[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
                raise RuntimeError('target mHC projection returned invalid auxiliary geometry')
            self._outputs[index] = hidden.detach()
        return capture

    def forward(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError('auxiliary target adapter is closed')
        if self._active:
            raise RuntimeError('auxiliary target forward cannot be reentered')
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError('auxiliary target adapter requires eval and no-grad inference')
        self._active = True
        self._outputs.clear()
        try:
            final = self._original_forward(*args, **kwargs)
            if not isinstance(final, torch.Tensor) or final.ndim != 2 or final.shape[-1] != self.hidden_size:
                raise RuntimeError('target final hidden output does not match the auxiliary contract')
            if tuple(sorted(self._outputs)) != self.layer_ids:
                raise RuntimeError('target forward omitted a selected auxiliary layer')
            auxiliary = [self._outputs[index] for index in self.layer_ids]
            if any((hidden.shape != final.shape or hidden.device != final.device or hidden.dtype != final.dtype for hidden in auxiliary)):
                raise RuntimeError('target auxiliary row/device/dtype identity differs from final hidden')
            return (final, auxiliary)
        finally:
            self._active = False
            self._outputs.clear()

    def close(self):
        if self._active:
            raise RuntimeError('cannot close an active auxiliary target forward')
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._outputs.clear()
        self._closed = True

def install_speco_92b_eagle3_target(model, *, layer_ids: tuple[int, ...]):
    """Bind the protocol on one loaded instance, preserving class/source code."""
    if hasattr(model, '_speco_92b_aux_target'):
        raise RuntimeError('target already has a SpeCo auxiliary adapter')
    adapter = MultiNode92BAuxHiddenAdapter(model, layer_ids=layer_ids)
    names = ('forward', 'supports_eagle3', 'has_own_lm_head', 'has_own_embed_tokens', 'set_aux_hidden_state_layers', 'get_eagle3_default_aux_hidden_state_layers')
    previous = {name: (name in model.__dict__, model.__dict__.get(name)) for name in names}

    def forward(_self, *args, **kwargs):
        return adapter.forward(*args, **kwargs)

    def select(_self, layers):
        if tuple(layers) != adapter.layer_ids:
            raise ValueError('auxiliary layer selection differs from the trained draft identity')
    model.forward = MethodType(forward, model)
    model.supports_eagle3 = True
    model.has_own_lm_head = False
    model.has_own_embed_tokens = False
    model.set_aux_hidden_state_layers = MethodType(select, model)
    model.get_eagle3_default_aux_hidden_state_layers = MethodType(lambda _self: adapter.layer_ids, model)
    model._speco_92b_aux_target = adapter

    def restore():
        adapter.close()
        for name, (had_value, value) in previous.items():
            if had_value:
                setattr(model, name, value)
            elif name in model.__dict__:
                delattr(model, name)
        delattr(model, '_speco_92b_aux_target')
    return (adapter, restore)
from verl_speco.integration.multinode_online import AcceptedTargetQueryProbe

class graph_accepted_query_probe_RunnerHook:

    def __init__(self, runner, original, installed):
        self.runner, self.original, self.installed = (runner, original, installed)
        self.removed = False

    def remove(self):
        if self.removed:
            return
        if self.runner._model_forward is not self.installed:
            raise RuntimeError('accepted graph probe lost its owned runner hook')
        self.runner._model_forward = self.original
        self.removed = True

class GraphAcceptedTargetQueryProbe(AcceptedTargetQueryProbe):

    def arm(self, **settings):
        original = getattr(self.runner, '_model_forward', None)
        if not callable(original):
            raise ValueError('graph accepted query requires the actual target runner return')
        super().arm(**settings)
        self.handle.remove()
        self.handle = None
        try:

            def observe_return(runner, num_tokens_padded, *args, **kwargs):
                result = original(num_tokens_padded, *args, **kwargs)
                ids = kwargs.get('input_ids', args[0] if args else None)
                positions = kwargs.get('positions', args[1] if len(args) > 1 else None)
                if positions is None:
                    getter = getattr(runner, '_get_positions', None)
                    if not callable(getter):
                        raise RuntimeError('graph accepted query lacks actual target positions')
                    positions = getter(num_tokens_padded)
                self._observe(runner.model, (), dict(input_ids=ids, positions=positions), result)
                return result
            installed = MethodType(observe_return, self.runner)
            self.handle = graph_accepted_query_probe_RunnerHook(self.runner, original, installed)
            self.runner._model_forward = installed
        except BaseException:
            self.close()
            raise

def native_target_reload_layers(model, expected_layers, *, allow_loader_layout=False):
    rows = []
    layouts = set()
    for name, layer in model.named_modules():
        if not hasattr(layer, 'w13_weight') and (not hasattr(layer, 'w2_weight')):
            continue
        method = getattr(layer, 'quant_method', None)
        if type(method).__name__ != 'AscendUnquantizedFusedMoEMethod' or getattr(layer, 'quant_config', None) is not None:
            raise ValueError('native target reload requires the audited Ascend unquantized expert method')
        e, h, i = (layer.local_num_experts, layer.hidden_size, layer.intermediate_size_per_partition)
        if any((type(v) is not int or v <= 0 for v in (e, h, i))):
            raise ValueError('native expert reload has invalid local geometry')
        serving = dict(w13_weight=(e, h, 2 * i), w2_weight=(e, i, h))
        raw = dict(w13_weight=(e, 2 * i, h), w2_weight=(e, h, i))
        local_layouts = set()
        for key, shape in serving.items():
            value = getattr(layer, key, None)
            if not isinstance(value, torch.nn.Parameter) or value.dtype != torch.bfloat16 or value.requires_grad or getattr(value, 'is_transposed', False):
                raise ValueError('native target is not in its exact postprocessed BF16 serving layout')
            if tuple(value.shape) == shape:
                local_layouts.add('serving')
            elif allow_loader_layout and tuple(value.shape) == raw[key]:
                local_layouts.add('loader')
            else:
                raise ValueError('native target has a foreign expert weight layout')
        if len(local_layouts) != 1:
            raise ValueError('native target has mixed expert weight layouts')
        layouts.update(local_layouts)
        rows.append((name, layer, serving, raw))
    if type(expected_layers) is not int or expected_layers < 1 or len(rows) != expected_layers:
        raise ValueError('native target reload lacks all expected expert layers')
    if len(layouts) != 1:
        raise ValueError('native target has mixed expert layer layouts')
    return rows

def prepare_native_target_reload(model, *, expected_layers=44):
    """Inverse only the audited transpose; retain values and parameter objects.

    Ascend postprocessing turns [E,2I,H]/[E,H,I] into [E,H,2I]/[E,I,H].
    The primary HF expert loader still expects the former. All layers are
    validated before mutation. Raw loader views share serving storage: the native
    loader writes through their strides, avoiding full expert clones at wake-up.
    The caller must fence generation through failure.
    """
    if getattr(model, '_speco_native_reload_phase', 'serving') != 'serving':
        raise RuntimeError('uncertain native target layout requires a fresh serving owner')
    rows = native_target_reload_layers(model, expected_layers, allow_loader_layout=True)
    starting_layout = 'loader' if tuple(rows[0][1].w13_weight.shape) == rows[0][3]['w13_weight'] else 'serving'
    model._speco_native_reload_phase = 'preparing'
    try:
        with torch.no_grad():
            for _, layer, _, raw in rows:
                for key, shape in raw.items():
                    param = getattr(layer, key)
                    value = param.detach()
                    param.data = value.transpose(1, 2) if starting_layout == 'serving' else value
                    if tuple(param.shape) != shape:
                        raise RuntimeError('inverse native expert layout has unexpected geometry')
        model._speco_native_reload_phase = 'loading'
        return dict(layers=len(rows), starting_layout=starting_layout, serving_shapes={name: shapes for name, _, shapes, _ in rows}, raw_shapes={name: shapes for name, _, _, shapes in rows})
    except BaseException:
        model._speco_native_reload_phase = 'failed'
        raise

def finish_native_target_reload(model, receipt):
    """Called after the actual whole-model postprocessor; never transpose twice."""
    if getattr(model, '_speco_native_reload_phase', None) != 'loading':
        raise RuntimeError('native target layout finish lacks an owned prepared reload')
    try:
        rows = native_target_reload_layers(model, receipt['layers'])
        if {name: shapes for name, _, shapes, _ in rows} != receipt['serving_shapes']:
            raise RuntimeError('native postprocessing changed the expert serving contract')
        model._speco_native_reload_phase = 'serving'
    except BaseException:
        model._speco_native_reload_phase = 'failed'
        raise
from dataclasses import dataclass

@dataclass(frozen=True)
class GraphExpertBindings:
    model_identity: int
    layers: int
    tensors: dict

def capture_graph_expert_bindings(model, *, expected_layers=44):
    if getattr(model, '_speco_native_reload_phase', 'serving') != 'serving':
        raise RuntimeError('graph bindings require an unambiguous serving owner')
    rows = native_target_reload_layers(model, expected_layers)
    tensors = {}
    for name, layer, serving, _ in rows:
        for key in serving:
            value = getattr(layer, key)
            if not value.is_contiguous() or value.storage_offset() != 0:
                raise ValueError('graph expert bindings require the exact contiguous serving allocation')
            tensors[name + '.' + key] = value.detach()
    return GraphExpertBindings(id(model), expected_layers, tensors)

def prepare_graph_expert_reload(model, bindings):
    """Expose raw loader views without reallocating the captured serving data."""
    if not isinstance(bindings, GraphExpertBindings) or bindings.model_identity != id(model) or getattr(model, '_speco_native_reload_phase', 'serving') != 'serving':
        raise RuntimeError('graph reload lacks the original live serving owner')
    rows = native_target_reload_layers(model, bindings.layers, allow_loader_layout=True)
    expected = {name + '.' + key for name, _, shapes, _ in rows for key in shapes}
    if expected != bindings.tensors.keys():
        raise ValueError('graph expert bindings do not cover all native layers')
    for name, layer, serving, raw in rows:
        for key in serving:
            old = bindings.tensors[name + '.' + key]
            value = getattr(layer, key)
            loader = old.transpose(1, 2)
            if tuple(old.shape) != serving[key] or tuple(loader.shape) != raw[key] or value.dtype != old.dtype or (value.device != old.device) or (value.data_ptr() != old.data_ptr()) or (value.storage_offset() != old.storage_offset()) or ((tuple(value.shape), value.stride()) not in ((tuple(old.shape), old.stride()), (tuple(loader.shape), loader.stride()))):
                raise ValueError('native wake changed captured expert storage or strides')
    model._speco_native_reload_phase = 'preparing'
    try:
        with torch.no_grad():
            for name, layer, serving, _ in rows:
                for key in serving:
                    getattr(layer, key).data = bindings.tensors[name + '.' + key].transpose(1, 2)
        model._speco_native_reload_phase = 'loading'
        return dict(layers=bindings.layers, serving_shapes={name: serving for name, _, serving, _ in rows}, raw_shapes={name: raw for name, _, _, raw in rows}, graph_expert_reload=True)
    except BaseException:
        model._speco_native_reload_phase = 'failed'
        raise

def finish_graph_expert_reload(model, bindings, receipt):
    if not isinstance(bindings, GraphExpertBindings) or bindings.model_identity != id(model) or receipt.get('graph_expert_reload') is not True or (receipt.get('layers') != bindings.layers):
        raise RuntimeError('graph expert finish lacks matching original bindings')
    try:
        finish_native_target_reload(model, receipt)
        rows = native_target_reload_layers(model, bindings.layers)
        for name, layer, serving, _ in rows:
            for key in serving:
                value, old = (getattr(layer, key), bindings.tensors[name + '.' + key])
                if value.data_ptr() != old.data_ptr() or value.stride() != old.stride() or value.storage_offset() != old.storage_offset() or (value.dtype != old.dtype) or (value.device != old.device):
                    raise RuntimeError('postprocessing replaced a captured expert allocation')
        return dict(layers=bindings.layers, expert_storage_pointers_and_strides_preserved=True, full_graph_publication_acceptance_not_claimed=True)
    except BaseException:
        model._speco_native_reload_phase = 'failed'
        raise

@dataclass(frozen=True)
class ModelStorageBindings:
    owner: object
    parameters: dict
    buffers: dict
    expert_names: frozenset
    gradient_flags: dict

def graph_model_storage_signature(value):
    return (tuple(value.shape), value.stride(), value.storage_offset(), value.data_ptr(), value.untyped_storage().data_ptr(), value.dtype, value.device)

def capture_model_storage(model, *, expert_names=()):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    experts = frozenset(expert_names)
    if not parameters or not experts.issubset(parameters):
        raise ValueError('graph storage requires complete parameters and declared expert owners')
    for name in experts:
        value = parameters[name]
        if value.ndim != 3 or not value.is_contiguous() or value.storage_offset():
            raise ValueError('expert storage requires contiguous serving geometry')
    for value in (*parameters.values(), *buffers.values()):
        if value.is_meta or value.layout != torch.strided:
            raise ValueError('graph storage requires materialized inference tensors')
    flags = {(kind, name): value.requires_grad for kind, group in (('parameter', parameters), ('buffer', buffers)) for name, value in group.items()}
    return ModelStorageBindings(model, {name: value.detach() for name, value in parameters.items()}, {name: value.detach() for name, value in buffers.items()}, experts, flags)

def verify_model_storage(model, bindings, *, allow_expert_loader_views=False):
    if not isinstance(bindings, ModelStorageBindings) or model is not bindings.owner or type(allow_expert_loader_views) is not bool:
        raise ValueError('graph storage has a foreign model or invalid phase')
    actual = (dict(model.named_parameters()), dict(model.named_buffers()))
    for group, saved, kind in zip(actual, (bindings.parameters, bindings.buffers), ('parameter', 'buffer')):
        if group.keys() != saved.keys():
            raise RuntimeError(f'graph {kind} owners changed during publication')
        for name, old in saved.items():
            value = group[name]
            expected = [graph_model_storage_signature(old)]
            if kind == 'parameter' and allow_expert_loader_views and (name in bindings.expert_names):
                expected.append(graph_model_storage_signature(old.transpose(1, 2)))
            if value.is_meta or value.requires_grad != bindings.gradient_flags[kind, name] or value.layout != torch.strided or (graph_model_storage_signature(value) not in expected):
                raise RuntimeError(f'graph {kind} storage changed: {name}')
    return dict(parameters=len(bindings.parameters), buffers=len(bindings.buffers), all_model_storage_pointers_and_strides_preserved=True, expert_loader_views_allowed=allow_expert_loader_views, full_graph_publication_acceptance_not_claimed=True)

def graph_kv_refresh_cache(layer):
    value = layer.kv_cache
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (tuple, list, torch.Tensor)) or len(value) != 2:
        raise ValueError('graph sink requires actual key/value cache pair')
    key, val = (value[0], value[1])
    if any((not isinstance(t, torch.Tensor) or not t.numel() or t.is_meta or t.requires_grad or (t.layout != torch.strided) for t in (key, val))):
        raise ValueError('graph sink caches must be allocated inference tensors')
    return (value, key, val)

def graph_kv_refresh_captured_signatures(params):
    if len(params) not in (14, 20):
        raise ValueError('graph KV audit requires the verified FIA capture ABI')
    return tuple((graph_model_storage_signature(value) if isinstance(value, torch.Tensor) else None for value in params))

@dataclass(frozen=True)
class GraphKVBindings:
    layers: tuple
    caches: tuple
    captured: tuple

def capture_graph_kv(layers, captured, *, expected_layers=46):
    layers, captured = (tuple(layers), tuple(captured))
    if type(expected_layers) is not int or expected_layers < 1 or len(layers) != expected_layers or (len(captured) != expected_layers) or (len({name for name, _ in layers}) != expected_layers) or (len({id(layer) for _, layer in layers}) != expected_layers):
        raise ValueError('graph KV requires complete unique target attention owners')
    caches, fingerprints, pointers = ([], [], set())
    for (name, layer), params in zip(layers, captured):
        if not isinstance(name, str) or not name or name != layer.layer_name:
            raise ValueError('graph KV target attention owner differs')
        _, key, value = graph_kv_refresh_cache(layer)
        signatures = graph_kv_refresh_captured_signatures(params)
        expected = [(graph_model_storage_signature(key), graph_model_storage_signature(value))]
        if key.ndim == 4 and value.ndim == 4 and key.is_contiguous() and value.is_contiguous():
            expected.append((graph_model_storage_signature(key.view(key.shape[0], key.shape[1], -1)), graph_model_storage_signature(value.view(value.shape[0], value.shape[1], -1))))
        if signatures[1:3] not in expected:
            raise ValueError('captured attention does not reference its target KV caches')
        pair = (key.data_ptr(), value.data_ptr())
        if len(set(pair)) != 2 or any((ptr in pointers for ptr in pair)):
            raise ValueError('graph KV aliases a foreign target layer')
        pointers.update(pair)
        caches.append((graph_model_storage_signature(key), graph_model_storage_signature(value)))
        fingerprints.append(signatures)
    return GraphKVBindings(layers, tuple(caches), tuple(fingerprints))

def verify_graph_kv(layers, captured, bindings):
    if not isinstance(bindings, GraphKVBindings):
        raise ValueError('graph KV audit lacks original bindings')
    current = capture_graph_kv(layers, captured, expected_layers=len(bindings.layers))
    if any((a != b or x is not y for (a, x), (b, y) in zip(current.layers, bindings.layers))) or current.caches != bindings.caches or current.captured != bindings.captured:
        raise RuntimeError('native wake or publication changed captured KV references')
    return dict(target_attention_layers=len(current.layers), captured_KV_addresses_preserved=True)

def refresh_graph_target_state(layers, captured, bindings, histories, *, expected_convs=138):
    layers, histories = (tuple(layers), tuple(histories))
    proof = verify_graph_kv(layers, captured, bindings)
    if type(expected_convs) is not int or expected_convs < 1 or len(histories) != expected_convs or (len({id(h.conv) for h in histories}) != expected_convs):
        raise ValueError('graph publication lacks every unique target MoME owner')
    for name, layer in layers:
        _, key, _ = graph_kv_refresh_cache(layer)
        sinks = (layer.sink_key, layer.sink_value)
        if layer.sink_populated is not False or type(layer.sink_len) is not int or layer.sink_len < 1 or (not callable(layer.populate_sink_kv)) or any((not isinstance(t, torch.Tensor) or t.ndim != 3 or t.shape[0] != layer.sink_len or t.requires_grad or (t.device != key.device) or (t.dtype != key.dtype) for t in sinks)) or (sinks[0].shape != sinks[1].shape):
            raise RuntimeError(f"graph publication requires freshly recomputed parameter sinks: layer={name}, populated={layer.sink_populated}, key_grad={getattr(sinks[0], 'requires_grad', None)}, value_grad={getattr(sinks[1], 'requires_grad', None)}")
    for history in histories:
        conv = history.conv
        ring = getattr(conv, '_speco_graph_ring', None)
        state = getattr(conv, '_state', None)
        if history._active or not isinstance(ring, torch.Tensor) or ring.ndim != 3 or (ring.shape[2] != conv.channels) or (ring.shape[1] != history.max_rejected_tokens + conv.left_pad) or (ring.device != conv.weight.device) or (ring.dtype != conv.weight.dtype) or (state is not None and (not isinstance(state, torch.Tensor) or state.device != ring.device or state.dtype != ring.dtype)):
            raise RuntimeError('graph publication lacks inactive native history buffers')
    with torch.inference_mode():
        for _, layer in layers:
            layer.populate_sink_kv(graph_kv_refresh_cache(layer)[0])
            if layer.sink_populated is not True:
                raise RuntimeError('actual sink population did not complete')
        for history in histories:
            history.conv._speco_graph_ring.zero_()
            state = getattr(history.conv, '_state', None)
            if state is not None:
                state.zero_()
            history.windows.clear()
    proof.update(refreshed_parameter_sink_layers=len(layers), cleared_graph_MoME_layers=len(histories), full_graph_publication_acceptance_not_claimed=True)
    return proof
from functools import wraps
FLAG = 'SPECO_MHC_SP_PREPROFILE'

def install_preprofile_sp(worker_class=None):
    if worker_class is None:
        from vllm_ascend.worker.worker import NPUWorker
        worker_class = NPUWorker
    if getattr(worker_class, '_speco_preprofile_sp_original', None) is not None:
        return False
    original = worker_class.load_model

    @wraps(original)
    def load(worker, *args, **kwargs):
        import os
        flag = os.environ.get(FLAG, '')
        if flag not in ('', '0', '1'):
            raise ValueError('invalid mHC preprofile policy')
        if flag != '1':
            return original(worker, *args, **kwargs)
        from verl_speco.runtime.services import capture_prefill_engine_options
        options = capture_prefill_engine_options(os.environ)
        from verl_speco.runtime.services import token_budget_engine_tokens
        engine_tokens = token_budget_engine_tokens(os.environ)
        if os.environ.get('SPECO_CAPTURE_GRAPH_MODE') != 'FULL_DECODE_ONLY' or os.environ.get('SPECO_MHC_SEQUENCE_PARALLEL') != '1' or os.environ.get('SPECO_TARGET_TP') != '8' or (options.get('max_num_batched_tokens') != engine_tokens) or (os.environ.get('SPECO_CAPTURE_PHYSICAL_BATCH_SIZE') not in ('8', '16')):
            raise ValueError('preprofile SP requires admitted FULL TP8 eight-slot 128K profile')
        from verl_speco.runtime.services import token_budget_engine_batch_size
        engine_sequences = token_budget_engine_batch_size(os.environ) or int(os.environ['SPECO_CAPTURE_PHYSICAL_BATCH_SIZE'])
        result = original(worker, *args, **kwargs)
        scheduler = worker.vllm_config.scheduler_config
        actual = worker.model_runner.max_num_tokens
        if scheduler.max_num_seqs != engine_sequences or type(actual) is not int or (not engine_tokens <= actual <= engine_tokens + 2 * scheduler.max_num_seqs):
            raise ValueError('actual warmup geometry differs from admitted bounded SP profile')
        worker._speco_sp_profile_max_tokens = actual
        receipt = worker.speco_enable_mhc_prefill_fusion()
        if not isinstance(receipt, dict) or receipt.get('modules') != 92 or receipt.get('sequence_parallel_world') != 8 or (receipt.get('max_tokens') != actual):
            raise RuntimeError('preprofile did not install full92B bounded TP8 SP')
        worker._speco_preprofile_SP_receipt = receipt
        return result
    worker_class._speco_preprofile_sp_original = original
    worker_class.load_model = load
    return True

def install_mome_native_decode_math(convolutions):
    convolutions = tuple(convolutions)
    if not convolutions or len({id(conv) for conv in convolutions}) != len(convolutions):
        raise ValueError('unique nonempty MoME convolutions are required')
    if any((hasattr(conv, '_speco_native_decode_math') for conv in convolutions)):
        raise RuntimeError('native decode math is already installed')
    saved = []
    for conv in convolutions:
        native = conv._causal_stateful
        saved.append((conv, '_causal_stateful' in conv.__dict__, conv.__dict__.get('_causal_stateful')))

        def causal(_conv, x, plan, *, _native=native):
            if plan.decode_only or not any((context > 0 and length > 1 for context, length in zip(plan.ctx_lens, plan.q_lens))):
                return _native(x, plan)
            _conv._speco_native_decode_batches += 1
            out = torch.empty_like(x)
            for slot, start, length, context in zip(plan.slots, plan.starts, plan.q_lens, plan.ctx_lens):
                if length <= 0:
                    continue
                pieces = ((start, length, context),) if context == 0 else ((start + offset, 1, context + offset) for offset in range(length))
                for begin, count, position in pieces:
                    single = copy.copy(plan)
                    single.slots = [slot]
                    single.starts = [0]
                    single.q_lens = [count]
                    single.ctx_lens = [position]
                    single.num_tokens = count
                    single.decode_only = count == 1
                    single.mask = None if plan.mask is None else plan.mask[begin:begin + count]
                    out[begin:begin + count] = _native(x[begin:begin + count], single)
                    if context > 0:
                        _conv._speco_native_decode_rows += count
            return out
        conv._causal_stateful = MethodType(causal, conv)
        conv._speco_native_decode_math = True
        conv._speco_native_decode_batches = 0
        conv._speco_native_decode_rows = 0

    def restore():
        for conv, existed, value in saved:
            if existed:
                conv._causal_stateful = value
            else:
                del conv._causal_stateful
            del conv._speco_native_decode_math
            del conv._speco_native_decode_batches
            del conv._speco_native_decode_rows
    return restore

@dataclass
class mome_speculation_Window:
    start: int
    end: int
    inputs: torch.Tensor

class MomeSpeculationHistory:

    def __init__(self, conv, *, max_rejected_tokens: int):
        if type(max_rejected_tokens) is not int or max_rejected_tokens < 1:
            raise ValueError('positive rejection bound is required')
        if type(conv.left_pad) is not int or conv.left_pad < 1 or (not callable(getattr(conv, '_ensure_state', None))):
            raise ValueError('native stateful MoME convolution is required')
        self.conv = conv
        self.max_rejected_tokens = max_rejected_tokens
        self.windows = {}
        self.rollback_calls = 0
        self.rolled_back_tokens = 0
        self._active = False

    def run(self, native, x, plan):
        if self._active:
            raise RuntimeError('MoME history cannot be reentered')
        if not plan.stateful:
            raise ValueError('MoME history requires request-local state')
        slots, starts, lengths, contexts = map(tuple, (plan.slots, plan.starts, plan.q_lens, plan.ctx_lens))
        if len(set(slots)) != len(slots) or not len(slots) == len(starts) == len(lengths) == len(contexts) or x.ndim != 2 or (x.shape[1] != self.conv.channels):
            raise ValueError('invalid MoME request/query geometry')
        if any((type(v) is not int or v < 0 for values in (slots, starts, lengths, contexts) for v in values)):
            raise ValueError('MoME request coordinates must be nonnegative integers')
        if any((start + length > x.shape[0] for start, length in zip(starts, lengths))):
            raise ValueError('MoME query exceeds input rows')
        histories = {}
        pad = self.conv.left_pad
        for slot, length, context in zip(slots, lengths, contexts):
            if length == 0:
                continue
            if context == 0:
                histories[slot] = x.new_zeros((pad, self.conv.channels))
                continue
            previous = self.windows.get(slot)
            if previous is None or context > previous.end:
                raise RuntimeError('MoME continuation has no committed input boundary')
            rejected = previous.end - context
            if rejected > self.max_rejected_tokens or max(context - pad, 0) < previous.start:
                raise RuntimeError('MoME rejection exceeds retained input window')
            left = max(context - pad, 0) - previous.start
            right = context - previous.start
            history = previous.inputs[left:right]
            if history.shape[0] < pad:
                history = torch.cat((x.new_zeros((pad - history.shape[0], self.conv.channels)), history))
            histories[slot] = history.to(x.dtype)
        self.conv._ensure_state(max(slots, default=0) + 1)
        for slot, history in histories.items():
            context = contexts[slots.index(slot)]
            previous = self.windows.get(slot)
            if context and previous.end > context:
                self.rollback_calls += 1
                self.rolled_back_tokens += previous.end - context
            self.conv._state[slot].copy_(history)
        self._active = True
        try:
            result = native(x, plan)
        except BaseException:
            for slot in histories:
                self.windows.pop(slot, None)
            raise
        finally:
            self._active = False
        for slot, start, length, context in zip(slots, starts, lengths, contexts):
            if length == 0:
                continue
            history = histories[slot][-min(pad, context):] if context else x[:0]
            values = torch.cat((history, x[start:start + length]), dim=0)
            retained = values[-(self.max_rejected_tokens + pad):].detach().clone()
            end = context + length
            self.windows[slot] = mome_speculation_Window(end - retained.shape[0], end, retained)
        return result

def install_mome_speculation_history(convolutions, *, max_rejected_tokens):
    """Bind on selected loaded instances; return histories and exact restoration."""
    convolutions = tuple(convolutions)
    if not convolutions or len({id(conv) for conv in convolutions}) != len(convolutions):
        raise ValueError('unique nonempty native MoME convolutions are required')
    if any((hasattr(conv, '_speco_mome_history') for conv in convolutions)):
        raise RuntimeError('MoME history is already installed')
    histories = tuple((MomeSpeculationHistory(conv, max_rejected_tokens=max_rejected_tokens) for conv in convolutions))
    saved = []
    for conv, history in zip(convolutions, histories):
        native = conv._causal_stateful
        saved.append((conv, '_causal_stateful' in conv.__dict__, conv.__dict__.get('_causal_stateful')))

        def causal(_conv, x, plan, *, _native=native, _history=history):
            return _history.run(_native, x, plan)
        conv._causal_stateful = MethodType(causal, conv)
        conv._speco_mome_history = history

    def restore():
        if any((history._active for history in histories)):
            raise RuntimeError('cannot restore active MoME history')
        for conv, existed, value in saved:
            if existed:
                conv._causal_stateful = value
            else:
                del conv._causal_stateful
            del conv._speco_mome_history
        for history in histories:
            history.windows.clear()
    return (histories, restore)

def bind_native_cache_policy(engine_args, *, rollout, engine_kwargs):
    requested = getattr(rollout, 'enable_prefix_caching', None)
    override = engine_kwargs.get('enable_prefix_caching', requested)
    if type(requested) is not bool or type(override) is not bool or requested != override:
        raise ValueError('native prefix cache recipe must be an explicit consistent boolean')
    engine_args.enable_prefix_caching = requested
    return dict(enable_prefix_caching=requested, bound_before_native_config=True)

def verify_native_cache_policy(native, policy):
    actual = native.cache_config.enable_prefix_caching
    if type(actual) is not bool or actual != policy['enable_prefix_caching']:
        raise RuntimeError('native engine lost the explicit prefix cache recipe')
    return dict(policy, actual_enable_prefix_caching=actual, verified=True)
from collections.abc import Mapping

def native_prefix_reset_kwargs(vllm_config, upstream_kwargs):
    """Keep connector reset only when the constructed engine has a connector.

    Recent VeRL asks for reset_connector unconditionally. Current vLLM returns
    False for that request if no connector is configured, even after its real
    local prefix cache was cleared. Preserve the native boolean acknowledgement.
    """
    if not isinstance(upstream_kwargs, Mapping):
        raise ValueError('native prefix reset defaults must be a mapping')
    result = dict(upstream_kwargs)
    transfer = getattr(vllm_config, 'kv_transfer_config', None)
    connector = getattr(transfer, 'kv_connector', None) if transfer is not None else None
    if transfer is not None and (not isinstance(connector, str) or not connector.strip()):
        raise ValueError('configured KV transfer lacks its actual connector identity')
    if transfer is None:
        result.pop('reset_connector', None)
    return result

def replay_attention_call(param, metadata, hf_config, workspace):
    """Preserve capture pointers and select v1/v2 from its actual parameter ABI."""
    if len(param) not in (14, 20):
        raise RuntimeError('unsupported captured sequence attention parameter ABI')
    query, key, value, table, mask, block_size = param[:6]
    common = dict(query=query, key=key, value=value, block_table=table, atten_mask=mask, input_layout='TND', block_size=block_size, workspace=workspace)
    if len(param) == 14:
        _, kv_heads, heads, scale, window, sinks, output, lse = param[6:]
        return ('v2', dict(common, actual_seq_qlen=metadata.actual_seq_lengths_q, actual_seq_kvlen=metadata.seq_lens_list, num_key_value_heads=kv_heads, num_query_heads=heads, softmax_scale=scale, sparse_mode=4 if window is not None else 3, pre_tokens=window if window is not None else 2147483647, next_tokens=0, learnable_sink=sinks, out=[output, lse]))
    _, _, kv_heads, heads, scale, output, lse, sparse, pre, nxt, k_scale, k_offset, v_scale, v_offset = param[6:]
    if not hasattr(hf_config, 'sliding_window'):
        common['block_table'] = metadata.block_tables
    if k_scale is not None:
        common.update(input_layout='BNSD', key_antiquant_scale=k_scale, key_antiquant_offset=k_offset, value_antiquant_scale=v_scale, value_antiquant_offset=v_offset, key_antiquant_mode=0, value_antiquant_mode=0)
        sparse = 0
    return ('v1', dict(common, actual_seq_lengths=metadata.actual_seq_lengths_q, actual_seq_lengths_kv=metadata.seq_lens_list, num_key_value_heads=kv_heads, num_heads=heads, scale=scale, sparse_mode=sparse, pre_tokens=pre, next_tokens=nxt, out=[output, lse]))

def install_sequence_graph_attention():
    import os
    if os.environ.get('SPECO_SEQUENCE_FULL_DECODE') != '1':
        raise RuntimeError('sequence attention adapter requires explicit graph serving')
    from vllm_ascend.attention import attention_v1 as backend
    impl = backend.AscendAttentionBackendImpl
    if getattr(impl, '_speco_sequence_graph_attention', False):
        return
    original = impl.update_graph_params

    def update(stream, context, num_tokens, config, speculative_config=None, num_dcp_pcp_tokens=None, draft_attn_metadatas=None):
        if backend._EXTRA_CTX.is_draft_model or not backend._EXTRA_CTX.sinks:
            return original(stream, context, num_tokens, config, speculative_config, num_dcp_pcp_tokens, draft_attn_metadatas)
        params = backend.get_graph_params()
        captured = params.attn_params[num_tokens]
        metadata = list(context.attn_metadata.values())
        handles = params.handles[num_tokens]
        events = params.events[num_tokens]
        if not len(captured) == len(metadata) == len(handles) == len(events):
            raise RuntimeError('sequence graph attention layer ownership differs')
        calls = [replay_attention_call(p, m, config.model_config.hf_text_config, params.workspaces.get(num_tokens)) for p, m in zip(captured, metadata)]
        with backend.torch.npu.stream(stream):
            for (version, kwargs), handle, event in zip(calls, handles, events):
                backend.torch.npu.graph_task_update_begin(stream, handle)
                op = backend.torch_npu.npu_fused_infer_attention_score_v2 if version == 'v2' else backend.torch_npu.npu_fused_infer_attention_score
                op.out(**kwargs)
                backend.torch.npu.graph_task_update_end(stream)
                event.record(stream)
    impl.update_graph_params = staticmethod(update)
    impl._speco_sequence_graph_attention = True
from types import SimpleNamespace

def device_decode_conv(conv, x, plan):
    import torch
    if conv._state is None or conv._state.shape[0] < plan.state_capacity:
        raise RuntimeError('MoME graph history was not allocated before capture')
    weight = conv.weight.view(conv.channels, conv.win_size).to(x.dtype)
    history = torch.cat([conv._state[plan.slots].to(x.dtype), x.unsqueeze(1)], dim=1)
    out = (history * weight.t().unsqueeze(0)).sum(dim=1)
    conv._state[plan.slots] = history[:, 1:].to(conv._state.dtype)
    return out if plan.mask is None else out * plan.mask.to(out.dtype)

def install_sequence_graph_mome():
    import os
    if os.environ.get('SPECO_SEQUENCE_FULL_DECODE') != '1':
        raise RuntimeError('MoME graph adapter is restricted to explicitly owned sequence serving')
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from pangu_infer.models._92b_vl import openpangu_v2_vl_text as module
    if getattr(NPUModelRunner, '_speco_sequence_graph_mome', False):
        return
    original_forward = NPUModelRunner._model_forward
    original_plan = module._build_mome_plan
    original_conv = module.MomeConv._causal_stateful
    active = {'plan': None}

    def build_plan(*args, **kwargs):
        plan = active['plan']
        return plan if plan is not None else original_plan(*args, **kwargs)

    def causal(conv, x, plan):
        if isinstance(plan.slots, torch.Tensor):
            return device_decode_conv(conv, x, plan)
        return original_conv(conv, x, plan)

    def forward(runner, num_tokens_padded, *args, **kwargs):
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return original_forward(runner, num_tokens_padded, *args, **kwargs)
        trunk = runner.model.model
        cfg = trunk.cfg
        capacity = int(runner.vllm_config.scheduler_config.max_num_seqs)
        if num_tokens_padded > capacity:
            raise RuntimeError('MoME FULL_DECODE_ONLY requires at most one token per request')
        device = next(trunk.parameters()).device
        dtype = next(trunk.parameters()).dtype
        for reference in list(module._MOME_CONVS):
            conv = reference()
            if conv is not None:
                conv._ensure_state(2 * capacity)
        req_ids = module._current_req_ids()
        contexts = []
        if req_ids:
            query = getattr(runner, 'query_lens', None)
            computed = getattr(runner.input_batch, 'num_computed_tokens_cpu', None)
            if query is None or computed is None:
                raise RuntimeError('MoME graph lacks actual CPU scheduler metadata')
            if hasattr(query, 'device') and query.device.type != 'cpu':
                raise RuntimeError('MoME graph query routing must remain on CPU')
            lengths = query[:len(req_ids)].tolist()
            contexts = computed[:len(req_ids)].tolist()
            if len(lengths) != len(req_ids) or len(contexts) != len(req_ids) or any((type(q) is not int or q != 1 for q in lengths)) or any((type(c) is not int or c < 0 for c in contexts)) or (len(set(req_ids)) != len(req_ids)):
                raise RuntimeError('MoME graph requires exact unpadded one-token requests')
            slots, clear, cap = module._mome_slots_for(req_ids, [c == 0 for c in contexts], len(req_ids))
            module._mome_clear_slots(clear, cap)
        else:
            slots = []
        if len(slots) > num_tokens_padded or any((s < 0 or s >= capacity for s in slots)):
            raise RuntimeError('MoME graph request slots exceed their reserved capacity')
        slots += list(range(capacity, capacity + num_tokens_padded - len(slots)))
        buffers = getattr(runner, '_speco_mome_decode_buffers', None)
        if buffers is None:
            buffers = {}
            runner._speco_mome_decode_buffers = buffers
        if num_tokens_padded not in buffers:
            buffers[num_tokens_padded] = SimpleNamespace(slots=torch.empty(num_tokens_padded, device=device, dtype=torch.long), mask=torch.empty((num_tokens_padded, 1), device=device, dtype=dtype), state_capacity=2 * capacity, stateful=True, decode_only=True)
        current = buffers[num_tokens_padded]
        current.slots.copy_(torch.tensor(slots, device=device, dtype=torch.long))
        current.mask.fill_(1)
        if contexts:
            valid = torch.tensor([c >= max(cfg.router_sliding_window - 1, 0) for c in contexts], device=device, dtype=dtype)
            current.mask[:len(contexts), 0].copy_(valid)
        active['plan'] = current
        try:
            return original_forward(runner, num_tokens_padded, *args, **kwargs)
        finally:
            active['plan'] = None
    module._build_mome_plan = build_plan
    module.MomeConv._causal_stateful = causal
    NPUModelRunner._model_forward = forward
    NPUModelRunner._speco_sequence_graph_mome = True

def bind_speco_sink_reservation(vllm_config, environ):
    model = vllm_config.model_config
    if tuple(model.hf_config.architectures) != ('OpenPanguV2VLForConditionalGeneration',):
        raise ValueError('sink reservation requires the dedicated actual target model92B architecture')
    sinks = getattr(model.hf_text_config, 'param_sink_number', None)
    block = vllm_config.cache_config.block_size
    if type(sinks) is not int or type(block) is not int or sinks <= 0 or (block <= 0) or sinks % block:
        raise ValueError('target model static sink count must cover complete actual native KV blocks')
    value = str(sinks // block)
    key = 'PANGU_92B_SINK_RESERVED_BLOCKS'
    if environ.get(key) not in (None, value):
        raise ValueError('stale sink reservation differs from actual model/cache geometry')
    environ[key] = value
    return dict(param_sink_tokens=sinks, kv_block_size=block, reserved_blocks=sinks // block)

def validate_updated_graph_status(rows, *, owners, target_revision, draft_revision):
    if type(rows) is not list or len(rows) != 8 or type(owners) is not dict or (set(owners) != set(range(8))) or (type(target_revision) is not int) or (target_revision < 1) or (type(draft_revision) is not int) or (draft_revision < 1):
        raise RuntimeError('updated graph acceptance needs all owned ranks and published versions')
    seen = set()
    for row in rows:
        if type(row) is not dict:
            raise RuntimeError('updated graph status lacks a native receipt')
        rank = row.get('rank')
        proof = row.get('graph_last_weight_publication')
        if type(rank) is not int or rank not in owners or rank in seen or (row.get('worker_incarnation') != owners[rank]) or (type(row.get('target_weight_revision')) is not int) or (row['target_weight_revision'] != target_revision) or (type(row.get('draft_revision')) is not int) or (row['draft_revision'] != draft_revision) or (type(row.get('target_replays_after_last_publication')) is not int) or (row['target_replays_after_last_publication'] < 1) or (row.get('K') != 8) or (row.get('eager_only') is not False) or (row.get('target_graph_mode') not in ('FULL_DECODE_ONLY', 'CUDAGraphMode.FULL_DECODE_ONLY')) or (type(proof) is not dict) or (proof.get('captured_KV_addresses_preserved') is not True) or (proof.get('all_model_storage_pointers_and_strides_preserved') is not True) or (proof.get('refreshed_parameter_sink_layers') != 46) or (proof.get('cleared_graph_MoME_layers') != 138):
            raise RuntimeError('updated graph lacks exact owners, versions, new replay or refreshed state')
        seen.add(rank)
    return dict(passed=True, ranks=8, target_revision=target_revision, draft_revision=draft_revision, actual_replay_after_publication=True, actual_refreshed_state_and_storage=True, joint_checkpoint_and_cold_restore_not_claimed=True)
