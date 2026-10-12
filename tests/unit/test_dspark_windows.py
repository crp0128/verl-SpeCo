# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Long sequences retain all supervised pairs and original target positions.'
import pytest
import torch
from verl_speco.trainer.distributed_batches import dspark_training_windows, backward_dspark_windows

def _batch(length, prompt):
    return dict(input_ids=torch.arange(length).unsqueeze(0), loss_mask=(torch.arange(length) >= prompt).float().unsqueeze(0), position_ids=(torch.arange(length) + 100).unsqueeze(0), hidden_states_list=[torch.arange(length).float().reshape(1, length, 1)], target_last_hidden_states=torch.arange(length).float().reshape(1, length, 1), lm_head_weight=torch.ones(8, 1))

def test_32k_response_visits_every_2048_window_and_preserves_positions():
    batch = _batch(257 + 32768, 257)
    windows = dspark_training_windows(batch, 2048)
    pairs = []
    for window in windows:
        assert window.end - window.start <= 2048
        assert window.batch['lm_head_weight'] is batch['lm_head_weight']
        assert window.batch['position_ids'][0, 0] == 100 + window.start
        assert window.batch['hidden_states_list'][0][0, -1, 0] == window.end - 1
        pairs.extend((i for i in range(window.start, window.end - 1) if i >= 257))
    assert pairs == list(range(257, 257 + 32768 - 1))
    assert sum((w.eligible_pairs for w in windows)) == 32767

def test_windowed_backward_includes_later_windows_before_one_update():
    batch = _batch(10, 1)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    calls = []

    def module(**window):
        calls.append(window['input_ids'][0, 0].item())
        return (parameter * window['input_ids'].float().mean(),)
    loss = backward_dspark_windows(module, batch, 4)
    assert calls == [0, 3, 6]
    assert loss > 4 and parameter.grad.item() == pytest.approx(loss)

def test_window_pair_count_respects_packing_boundaries():
    batch = _batch(10, 0)
    batch['document_ids'] = torch.tensor([[0, 0, 0, -1, 1, 1, 1, 1, 1, 1]])
    windows = dspark_training_windows(batch, 4)
    assert sum((w.eligible_pairs for w in windows)) == 7
    with pytest.raises(ValueError, match='at least two'):
        dspark_training_windows(batch, 1)

def test_fsdp_windows_reduce_gradients_only_after_the_last_local_window():
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class Module:

        def __init__(self):
            self.states = []

        def set_requires_gradient_sync(self, enabled):
            self.states.append(enabled)

        def __call__(self, **batch):
            return (parameter * batch['input_ids'].float().mean(),)
    module = Module()
    backward_dspark_windows(module, _batch(10, 1), 4)
    assert module.states == [False, False, True, True]
    assert parameter.grad is not None
    failed = Module()
    failed.__class__.__call__ = lambda self, **_: (parameter * float('nan'),)
    with pytest.raises(RuntimeError, match='not finite'):
        backward_dspark_windows(failed, _batch(10, 1), 4)
    assert failed.states == [False, True]
