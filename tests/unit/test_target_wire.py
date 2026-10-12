# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Whole target content binding and actual double-buffer overwrite regression.'
import asyncio
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from verl_speco.integration.multinode_publication import manifest_for_state, wire_dtype, TargetWireDigest, load_received_target_buckets, reset_native_weight_coverage, verify_native_weight_coverage

def state():
    return {'model.weight': torch.arange(8, dtype=torch.float32).reshape(2, 4), 'model.attn_mhc_module.branch_alpha_pre': torch.tensor([0.4]), 'model.e_score_correction_bias': torch.tensor([0.7, 0.9])}

def manifest(tensors=None):
    return manifest_for_state(tensors or state(), run_id='run', generation=1, rl_step=1, target_revision=1, operation_id='op', actor_incarnations={0: 'inc'})

def finish(tensors):
    digest = TargetWireDigest(manifest(tensors), chunk_bytes=8)
    for name, value in tensors.items():
        digest.update(name, value.to(wire_dtype(name, value)))
    return digest.finish()

def test_wire_content_not_names_or_shapes_defines_actor_manifest_checksum():
    tensors = state()
    first = finish(tensors)
    changed = deepcopy(tensors)
    changed['model.weight'][0, 0] += 1
    second = finish(changed)
    assert first['header_sha256'] == second['header_sha256']
    assert first['actor_weight_manifest_sha256'] != second['actor_weight_manifest_sha256']
    assert first['wire_nbytes'] == 28 and first['wire_tensor_count'] == 3
    assert first == finish(tensors)

@pytest.mark.parametrize('damage', ['header', 'missing', 'extra', 'order', 'shape', 'dtype', 'nonfinite'])
def test_full_wire_refuses_mutation_truncation_or_invalid_actual_values(damage):
    tensors = state()
    m = manifest()
    digest = TargetWireDigest(m)
    if damage == 'header':
        m['target_revision'] = 2
        with pytest.raises(ValueError):
            TargetWireDigest(m)
        return
    pairs = [(name, value.to(wire_dtype(name, value))) for name, value in tensors.items()]
    if damage == 'missing':
        pairs.pop()
    elif damage == 'extra':
        pairs.append(pairs[0])
    elif damage == 'order':
        pairs.reverse()
    elif damage == 'shape':
        pairs[0] = (pairs[0][0], pairs[0][1].reshape(-1))
    elif damage == 'dtype':
        pairs[0] = (pairs[0][0], pairs[0][1].float())
    elif damage == 'nonfinite':
        pairs[0][1][0, 0] = float('nan')
    with pytest.raises((RuntimeError, ValueError)):
        for name, value in pairs:
            digest.update(name, value)
        digest.finish()

def test_native_HCCL_prefetch_overwrite_cannot_corrupt_pending_load_batch():

    async def run():
        values = [torch.tensor([float(i), float(i + 1)], dtype=torch.bfloat16) for i in range(5)]
        tensors = {'weight' + str(i): value for i, value in enumerate(values)}
        buffers = [torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.bfloat16)]
        buffers[0].copy_(values[0])

        async def receive():
            for index in range(len(values)):
                if index + 1 < len(values):
                    buffers[(index + 1) % 2].copy_(values[index + 1])
                yield ('weight' + str(index), buffers[index % 2])
        loaded = {}

        class Native:

            def load_weights(self, batch):
                for name, value in batch:
                    loaded[name] = value.clone()
        actual = await load_received_target_buckets(receive(), Native(), TargetWireDigest(manifest(tensors), chunk_bytes=8))
        assert actual == finish(tensors)
        assert set(loaded) == set(tensors)
        assert all((torch.equal(loaded[name], value) for name, value in tensors.items()))
    asyncio.run(run())

def test_native_coverage_is_rechecked_each_revision_including_expert_slots():
    calls = []
    model = SimpleNamespace(_loaded_cumulative={'old'}, _pangu_ready=True, _expert_slots_seen={('old', 0, 'w1')}, _expert_declined=['old'])
    reset_native_weight_coverage(model)
    assert not model._loaded_cumulative and (not model._pangu_ready) and (not model._expert_slots_seen) and (not model._expert_declined)
    model.named_parameters = lambda: [('new', None)]
    model._exempt = lambda name: False

    def ready():
        model._pangu_ready = True
    model._assert_ready = ready
    model._assert_expert_slots_complete = lambda: calls.append('expert-slots')
    with pytest.raises(RuntimeError):
        verify_native_weight_coverage(model)
    model._loaded_cumulative = {'new'}
    result = verify_native_weight_coverage(model)
    assert result['complete_native_names_and_expert_slots'] and calls == ['expert-slots']

def test_coverage_reset_preserves_getter_only_native_records_and_weights():

    class Native:

        def __init__(self):
            self.names = {'old'}
            self.slots = {('old', 0, 'w1')}
            self.declined = ['old']
            self._pangu_ready = True
            self.weight = torch.tensor([1.0, 2.0])
        _loaded_cumulative = property(lambda self: self.names)
        _expert_slots_seen = property(lambda self: self.slots)
        _expert_declined = property(lambda self: self.declined)
    model = Native()
    containers = (model.names, model.slots, model.declined)
    before = model.weight.clone()
    reset_native_weight_coverage(model)
    assert all((actual is expected for actual, expected in zip((model._loaded_cumulative, model._expert_slots_seen, model._expert_declined), containers)))
    assert not any(containers) and (not model._pangu_ready)
    assert torch.equal(model.weight, before)

def test_coverage_container_contract_failure_does_not_partly_clear_records():
    model = SimpleNamespace(_loaded_cumulative={'old'}, _pangu_ready=True, _expert_slots_seen={('old', 0, 'w1')}, _expert_declined=('foreign',))
    with pytest.raises(ValueError, match='containers'):
        reset_native_weight_coverage(model)
    assert model._loaded_cumulative == {'old'} and model._expert_slots_seen == {('old', 0, 'w1')}
    assert model._pangu_ready

@pytest.mark.parametrize('damage', ['none', 'missing', 'expert', 'shadow'])
def test_installed_acl_proxy_and_native_readiness_keep_one_coverage_owner(damage):
    """Execute the installed proxy and primary native guard without NPU imports."""
    import ast
    from pathlib import Path
    proxy_source = Path('/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py')
    native_source = Path(__file__).resolve().parents[3] / 'pangu-vllm-ascend/pangu_infer/models/_92b_vl/openpangu_v2_vl_model.py'
    if not proxy_source.is_file():
        pytest.skip('installed Ascend ACLGraphWrapper source unavailable')

    def method(path, name):
        tree = ast.parse(path.read_text())
        node = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name))
        node.decorator_list = []
        namespace = {'Callable': object, '_log': lambda *args: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
        return namespace[name]
    Native = type('Native', (), {'_assert_ready': method(native_source, '_assert_ready')})
    native = Native()
    native._loaded_cumulative = {'old'}
    native._expert_slots_seen = {('old', 0, 'w1')}
    native._expert_declined = ['old']
    native._pangu_ready = True
    native.named_parameters = lambda: [('new', None)]
    native._exempt = lambda name: False
    native._log_weight_fingerprint = lambda: None

    def check_experts():
        if damage == 'expert':
            raise RuntimeError('incomplete actual expert slots')
    native._assert_expert_slots_complete = check_experts
    Proxy = type('ACLGraphWrapper', (), {'__module__': 'vllm_ascend.compilation.acl_graph', '__getattr__': method(proxy_source, '__getattr__')})
    wrapper = Proxy()
    wrapper.runnable = native
    wrapper.is_debugging_mode = False
    if damage == 'shadow':
        wrapper._pangu_ready = False
        with pytest.raises(ValueError, match='shadows'):
            reset_native_weight_coverage(wrapper)
        assert native._loaded_cumulative == {'old'} and native._pangu_ready
        return
    reset_native_weight_coverage(wrapper)
    assert not native._pangu_ready and (not native._loaded_cumulative)
    assert '_pangu_ready' not in vars(wrapper)
    if damage != 'missing':
        native._loaded_cumulative.add('new')
    if damage in ('missing', 'expert'):
        with pytest.raises(RuntimeError):
            verify_native_weight_coverage(wrapper)
    else:
        result = verify_native_weight_coverage(wrapper)
        assert result['complete_native_names_and_expert_slots']
        assert native._pangu_ready and wrapper._pangu_ready
    assert '_pangu_ready' not in vars(wrapper)
