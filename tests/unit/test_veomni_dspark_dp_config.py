# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
from types import SimpleNamespace
import pytest
from verl_speco.trainer.veomni_dspark_adapter import validate_dp_only_engine_config

def _config(**changes):
    values = dict(init_device='meta', fsdp_size=1, ulysses_parallel_size=1, expert_parallel_size=1)
    values.update(changes)
    return SimpleNamespace(**values)

def test_dp_only_draft_mesh_accepts_shard_size_one_without_other_parallelism():
    validate_dp_only_engine_config(_config())

@pytest.mark.parametrize('changes', [{'init_device': 'cpu'}, {'fsdp_size': -1}, {'fsdp_size': 2}, {'ulysses_parallel_size': 2}, {'expert_parallel_size': 2}])
def test_dp_only_draft_mesh_rejects_other_parallelism_before_engine_init(changes):
    with pytest.raises(ValueError):
        validate_dp_only_engine_config(_config(**changes))


@pytest.mark.parametrize('policy, confidence', [('random', 0.0), ('dmtp_cpu', 0.2)])
def test_multinode_model_constructs_and_backpropagates(policy, confidence):
    import torch
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    from verl_speco.backends.dspark_multinode_backend import MultiNodeDSparkTrainingModel

    config = DSparkConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=32,
        num_target_layers=4, num_context_layers=2, target_hidden_size=8,
        target_num_hidden_layers=4, target_layer_ids=[1, 3], mask_token_id=31,
        markov_rank=4, markov_head_type='vanilla', enable_confidence_head=confidence > 0)
    model = MultiNodeDSparkTrainingModel(DSparkDraftModel(config), block_size=2,
        num_anchors=2, l1_loss_alpha=0.0, confidence_head_alpha=confidence,
        anchor_sampling_policy=policy, distribution_loss_impl='eager')
    loss = model(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        hidden_states_list=[torch.randn(1, 6, 8), torch.randn(1, 6, 8)],
        loss_mask=torch.ones(1, 6), lm_head_weight=torch.randn(32, 8),
        target_last_hidden_states=torch.randn(1, 6, 8),
        position_ids=torch.arange(100, 106).unsqueeze(0))[0]
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if confidence:
        assert model.draft_model.confidence_head.proj.weight.grad is not None
