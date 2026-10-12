# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib

def local_gradient_summary(module):
    import torch
    rows = []
    for name, parameter in sorted(module.named_parameters()):
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            rows.append(dict(name=name, missing=True))
            continue
        local = gradient.detach()
        if hasattr(local, 'to_local'):
            local = local.to_local()
        local = local.cpu().contiguous()
        raw = memoryview(local.view(torch.uint8).numpy())
        finite, maximum, squared = (True, 0.0, 0.0)
        for chunk in local.reshape(-1).split(1048576):
            values = chunk.to(torch.float64)
            finite = finite and bool(torch.isfinite(values).all())
            if values.numel():
                maximum = max(maximum, float(values.abs().max()))
                squared += float(values.square().sum())
        rows.append(dict(name=name, missing=False, local_shape=list(local.shape), dtype=str(local.dtype), elements=local.numel(), sha256=hashlib.sha256(raw).hexdigest(), finite=finite, max_abs=maximum if finite else None, squared_l2=squared if finite else None))
    return dict(schema_version=1, scope='local_gradients_before_clip', parameters=rows, all_present_finite=all((row.get('finite', True) for row in rows)), deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled())

@contextmanager
def observe_optimizer_gradients(engine, observer=None):
    """Restore the exact method binding even when observation/update fails."""
    if observer is None:
        yield
        return
    if not callable(observer):
        raise TypeError('gradient observer must be callable')
    original = engine.optimizer_step
    had_override = 'optimizer_step' in vars(engine)
    override = vars(engine).get('optimizer_step')

    def observed_step(*args, **kwargs):
        observer(local_gradient_summary(engine.module))
        return original(*args, **kwargs)
    engine.optimizer_step = observed_step
    try:
        yield
    finally:
        if had_override:
            engine.optimizer_step = override
        else:
            del engine.optimizer_step

@contextmanager
def draft_deterministic_algorithms(value=''):
    """Diagnostic-only strict switch, restoring the caller's torch policy."""
    if value == '':
        yield
        return
    if type(value) is not str or value not in ('0', '1'):
        raise ValueError('draft deterministic algorithms must be empty, 0 or 1')
    import torch
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(value == '1', warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
import torch

@contextmanager
def collect_training_acceptance(module):
    """Observe detached prefix counters; never retain logits or add a forward."""
    counts = {}

    def observe(_module, _inputs, output):
        if not isinstance(output, (tuple, list)) or not isinstance(output[-1], dict):
            raise RuntimeError('DSpark acceptance requires forward diagnostics')
        diagnostics = output[-1]
        for key in ('accepted_length_sum', 'scored_block_count'):
            value = diagnostics.get(key)
            if not torch.is_tensor(value) or value.numel() != 1:
                raise RuntimeError('missing scalar DSpark acceptance counter: ' + key)
            value = value.detach().reshape(()).to(dtype=torch.float32)
            counts[key] = value.clone() if key not in counts else counts[key] + value
    handle = module.register_forward_hook(observe)
    try:
        yield counts
    finally:
        handle.remove()

def summarize_training_acceptance(counts, *, reduce_sum=None):
    """Reduce sums before division; dummy ranks contribute zero observations."""
    if set(counts) != {'accepted_length_sum', 'scored_block_count'}:
        raise ValueError('incomplete training acceptance counters')
    pair = torch.stack([counts['accepted_length_sum'], counts['scored_block_count']]).detach().clone()
    if pair.shape != (2,) or not bool(torch.isfinite(pair).all()) or bool((pair < 0).any()):
        raise ValueError('invalid training acceptance counters')
    if reduce_sum is not None:
        reduce_sum(pair)
    accepted, blocks = pair.cpu().tolist()
    if not all((__import__('math').isfinite(v) and v >= 0 and (v == int(v)) for v in (accepted, blocks))):
        raise ValueError('invalid reduced acceptance counters')
    if accepted > blocks * 8 or (blocks == 0 and accepted != 0):
        raise ValueError('acceptance exceeds the K8 draft proposal bound')
    return dict(draft_acceptance_scope='offline_greedy_saved_response_prefix_before_update', draft_accepted_length_sum=int(accepted), draft_scored_block_count=int(blocks), draft_mean_accepted_tokens=accepted / blocks if blocks else None, draft_mean_acceptance_length=1.0 + accepted / blocks if blocks else None)

def observe_training_acceptance(method):
    """Enable counts for native standalone without changing optimizer returns."""
    from functools import wraps
    import os

    @wraps(method)
    def wrapped(engine, *args, **kwargs):
        engine.last_acceptance_counts = None
        policy = os.environ.get('SPECO_DRAFT_ACCEPTANCE_METRICS', '')
        if policy not in ('', '0', '1'):
            raise ValueError('invalid training acceptance policy')
        if policy != '1':
            return method(engine, *args, **kwargs)
        with collect_training_acceptance(engine.module) as counts:
            result = method(engine, *args, **kwargs)
        if not counts:
            raise RuntimeError('training step did not observe acceptance counters')
        engine.last_acceptance_counts = counts
        return result
    return wrapped
import json
import os

def compact_training_rank_log(summary):
    """Keep full keys and checkpoint paths in the committed receipt only."""
    allowed = {'status', 'rank', 'loss', 'grad_norm', 'draft_weights_sha256', 'train_seconds', 'tq_get_seconds', 'materialize_seconds', 'dcp_seconds', 'weight_consensus_seconds', 'peak_npu_allocated_bytes', 'original_tokens', 'target_head_index_sha256', 'checkpoint_checksum', 'journal_state', 'draft_acceptance_scope', 'draft_accepted_length_sum', 'draft_scored_block_count', 'draft_mean_accepted_tokens', 'draft_mean_acceptance_length'}
    result = {key: summary[key] for key in allowed if key in summary}
    keys = tuple(summary['packed_sample_keys'])
    result.pop('sample_key', None)
    if not keys or any((not isinstance(k, str) or not k for k in keys)):
        raise ValueError('training summary requires nonempty sample identities')
    result['packed_sample_count'] = len(keys)
    result['packed_sample_keys_sha256'] = hashlib.sha256(json.dumps(keys, separators=(',', ':')).encode()).hexdigest()
    uri = summary['checkpoint']
    result['checkpoint_uri_sha256'] = hashlib.sha256(uri.encode()).hexdigest()
    seconds = result['train_seconds']
    count = result['original_tokens']
    if type(count) is not int or count < 1:
        raise ValueError('training summary requires positive original token count')
    result.pop('train_tokens_per_second', None)
    result['isolated_train_original_token_equivalent_per_second'] = count / seconds if seconds > 0 else None
    result['throughput_scope'] = 'draft_train_only_excludes_prefill_transfer_checkpoint'
    return result

def write_training_rank_log(summary, *, fd):
    data = (json.dumps(compact_training_rank_log(summary), separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(data) > os.fpathconf(fd, 'PC_PIPE_BUF'):
        raise ValueError('training summary exceeds atomic pipe write budget')
    if os.write(fd, data) != len(data):
        raise RuntimeError('training summary write was incomplete')
