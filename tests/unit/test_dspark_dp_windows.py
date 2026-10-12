# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'DP windows conserve pair ownership and the global weighted gradient.'
import pytest
import torch
from verl_speco.trainer.distributed_batches import shard_dspark_windows, backward_dspark_window_shard
from verl_speco.trainer.distributed_batches import move_dspark_window_shard

def batch():
    ids = torch.arange(20).unsqueeze(0)
    return dict(input_ids=ids, position_ids=ids + 1000, loss_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1.0]]), document_ids=torch.tensor([[0] * 10 + [1] * 10]), hidden_states_list=[ids.unsqueeze(-1).float()], target_last_hidden_states=ids.unsqueeze(-1).float(), lm_head_weight=torch.ones(30, 1))

def eligible(part):
    mask, docs = (part['loss_mask'], part['document_ids'])
    return (mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5) & (docs[:, :-1] == docs[:, 1:]) & (docs[:, :-1] >= 0)

@pytest.mark.parametrize('rank', [0, 1, 15])
def test_moved_owned_windows_keep_pairs_and_compact_storage_without_full_capture_alias(rank):
    original = batch()
    shard = shard_dspark_windows(original, window_size=5, rank=rank, world_size=16)
    moved = move_dspark_window_shard(shard, 'cpu')
    assert moved.window_indices == shard.window_indices
    assert moved.loss_scale == shard.loss_scale and moved.eligible_pairs == shard.eligible_pairs
    for before, after in zip(shard.windows, moved.windows):
        assert (after.start, after.end, after.eligible_pairs) == (before.start, before.end, before.eligible_pairs)
        assert after.batch['lm_head_weight'] is original['lm_head_weight']
        for name in ('input_ids', 'position_ids', 'loss_mask', 'document_ids', 'target_last_hidden_states'):
            value = after.batch[name]
            torch.testing.assert_close(value, before.batch[name])
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
            assert value.untyped_storage().data_ptr() != original[name].untyped_storage().data_ptr()
        torch.testing.assert_close(after.batch['hidden_states_list'][0], before.batch['hidden_states_list'][0])
    saved = moved.windows[0].batch['input_ids'].clone()
    original['input_ids'].zero_()
    torch.testing.assert_close(moved.windows[0].batch['input_ids'], saved)

@pytest.mark.parametrize('world', [1, 2, 4, 8, 64])
def test_every_pair_owned_once_with_absolute_positions_and_teacher_features(world):
    original = batch()
    expected = original['input_ids'][:, :-1][eligible(original)].tolist()
    observed = []
    shards = [shard_dspark_windows(original, window_size=5, rank=r, world_size=world) for r in range(world)]
    for shard in shards:
        for window in shard.windows:
            part = window.batch
            assert part['input_ids'].shape[1] <= 5
            assert torch.equal(part['position_ids'], part['input_ids'] + 1000)
            assert torch.equal(part['hidden_states_list'][0][..., 0], part['input_ids'].float())
            assert part['lm_head_weight'] is original['lm_head_weight']
            if shard.window_indices:
                observed.extend(part['input_ids'][:, :-1][eligible(part)].tolist())
    assert sorted(observed) == expected
    assert sum((s.eligible_pairs for s in shards)) == len(expected)
    assert all((s.global_eligible_pairs == len(expected) for s in shards))

@pytest.mark.parametrize('world', [2, 4, 8, 64])
def test_dp_average_matches_global_pair_gradient_including_empty_ranks(world):
    original = batch()
    reference = original['input_ids'][:, :-1][eligible(original)].float().mean().item()
    gradients = []
    for rank in range(world):
        shard = shard_dspark_windows(original, window_size=5, rank=rank, world_size=world)
        parameter = torch.nn.Parameter(torch.tensor(1.0))

        def model(**part):
            return (parameter * part['input_ids'][:, :-1][eligible(part)].float().mean(),)
        backward_dspark_window_shard(model, shard)
        gradients.append(parameter.grad.item())
    assert sum(gradients) / world == pytest.approx(reference)

def test_fewer_documents_still_use_all_ranks_when_enough_windows():
    shards = [shard_dspark_windows(batch(), window_size=3, rank=r, world_size=4) for r in range(4)]
    assert all((shard.eligible_pairs > 0 for shard in shards))

def test_sync_is_restored_after_nonfinite_window_loss():
    shard = shard_dspark_windows(batch(), window_size=5, rank=0, world_size=1)

    class Broken:

        def __init__(self):
            self.sync = []

        def set_requires_gradient_sync(self, enabled):
            self.sync.append(enabled)

        def __call__(self, **part):
            return (torch.tensor(float('nan')),)
    model = Broken()
    with pytest.raises(RuntimeError, match='not finite'):
        backward_dspark_window_shard(model, shard)
    assert model.sync[-1] is True
