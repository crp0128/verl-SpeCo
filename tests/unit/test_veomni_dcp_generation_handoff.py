# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Actual CPU DCP model/AdamW/scheduler/RNG restore with a generation gate.'
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
from verl_speco.trainer.distributed_checkpoint import VeOmniDCPBridge

@pytest.fixture
def saved(tmp_path, monkeypatch):
    assert not dist.is_initialized()
    store = dist.TCPStore('127.0.0.1', 0, 1, True, use_libuv=False)
    dist.init_process_group('gloo', store=store, rank=0, world_size=1)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: False, get_rng_state=lambda: torch.zeros(4, dtype=torch.uint8), set_rng_state=lambda value: None), raising=False)
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        loss = model(torch.tensor([[1.0, -2.0, 3.0]])).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        reference = copy.deepcopy(model.state_dict())
        optim = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        schedule = copy.deepcopy(scheduler.state_dict())
        source = VeOmniDCPBridge(module=model, optimizer=optimizer, scheduler=scheduler, topology_fingerprint='original-port', model_fingerprint='model', logical_checkpoint_fingerprint='a' * 64)
        candidate = source.save_candidate(tmp_path / 'candidate', generation=1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(100)
        optimizer.state.clear()
        scheduler.step()
        torch.rand(37)
        yield dict(model=model, optimizer=optimizer, scheduler=scheduler, candidate=candidate, reference=reference, optim=optim, schedule=schedule, rng=rng)
    finally:
        dist.destroy_process_group()

def bridge(saved, **changes):
    kwargs = dict(module=saved['model'], optimizer=saved['optimizer'], scheduler=saved['scheduler'], topology_fingerprint='new-port', model_fingerprint='model', logical_checkpoint_fingerprint='a' * 64)
    kwargs.update(changes)
    return VeOmniDCPBridge(**kwargs)

def test_explicit_new_generation_restores_actual_model_optimizer_scheduler_and_rng(saved):
    bridge(saved).restore_candidate(saved['candidate'], handoff_generation=2)
    assert all((torch.equal(value, saved['reference'][name]) for name, value in saved['model'].state_dict().items()))
    state = saved['optimizer'].state_dict()
    assert state['param_groups'] == saved['optim']['param_groups']
    for key, values in state['state'].items():
        for name, value in values.items():
            assert torch.equal(value, saved['optim']['state'][key][name])
    assert saved['scheduler'].state_dict() == saved['schedule']
    assert torch.equal(torch.get_rng_state(), saved['rng'])

@pytest.mark.parametrize('generation', [None, 1, 0, True])
def test_changed_port_cannot_restore_without_advancing_generation(saved, generation):
    before = copy.deepcopy(saved['model'].state_dict())
    with pytest.raises(ValueError, match='incompatible'):
        bridge(saved).restore_candidate(saved['candidate'], handoff_generation=generation)
    assert all((torch.equal(value, before[name]) for name, value in saved['model'].state_dict().items()))
    assert not saved['optimizer'].state

@pytest.mark.parametrize('changes', [{'logical_checkpoint_fingerprint': 'b' * 64}, {'logical_checkpoint_fingerprint': None}, {'model_fingerprint': 'foreign-model'}])
def test_new_generation_cannot_override_model_or_mesh_identity(saved, changes):
    with pytest.raises(ValueError, match='incompatible'):
        bridge(saved, **changes).restore_candidate(saved['candidate'], handoff_generation=2)
    assert not saved['optimizer'].state

def test_corrupt_dcp_file_is_rejected_before_mutating_current_model(saved):
    root = Path(saved['candidate'].uri)
    extra = root / 'extra-rank-0000.pt'
    extra.write_bytes(extra.read_bytes() + b'corrupt')
    before = copy.deepcopy(saved['model'].state_dict())
    with pytest.raises(ValueError, match='checksum mismatch'):
        bridge(saved).restore_candidate(saved['candidate'], handoff_generation=2)
    assert all((torch.equal(value, before[name]) for name, value in saved['model'].state_dict().items()))
