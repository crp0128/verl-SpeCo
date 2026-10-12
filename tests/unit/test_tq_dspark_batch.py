# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'A TQ aux-plus-last payload becomes the exact DSpark VeOmni input contract.'
import pytest
import torch
from dataclasses import replace
from verl_speco.trainer.distributed_batches import materialize_dspark_batch, materialize_packed_dspark_batch
from verl_speco.trainer.distributed_batches import PACKING_TIERS, plan_packed_prefix, require_packing_tier
from verl_speco.transport.streaming import SampleMetadataV3

def _sample():
    fields = {'input_ids': torch.tensor([3, 4, 5], dtype=torch.int64), 'position_ids': torch.tensor([0, 1, 2], dtype=torch.int64), 'loss_mask': torch.tensor([0, 1, 1], dtype=torch.float32), 'hidden_states': torch.arange(36, dtype=torch.float32).reshape(3, 12).bfloat16()}
    metadata = SampleMetadataV3(schema_version=3, run_id='run-0', generation=1, producer_id='producer-0', producer_incarnation='inc-0', request_id='request-0', sample_id='sample-0', sequence_no=0, target_revision='target-0', data_version=0, tokenizer_fingerprint='tokenizer-0', target_model_fingerprint='model-0', target_layer_ids=(1, 3), hidden_states_layout='dflash_aux_plus_last', hidden_dtype='bfloat16', token_count=3, prompt_token_count=1, supervised_token_count=2, payload_nbytes=sum((t.numel() * t.element_size() for t in fields.values())))
    return (fields, metadata)

def test_materializes_target_layers_and_final_teacher_hidden():
    fields, metadata = _sample()
    head = torch.ones(8, 4, dtype=torch.bfloat16)
    batch = materialize_dspark_batch(fields, metadata, lm_head_weight=head, device='cpu', target_hidden_size=4)
    assert batch['input_ids'].shape == (1, 3)
    assert len(batch['hidden_states_list']) == 2
    torch.testing.assert_close(batch['hidden_states_list'][0][0], fields['hidden_states'][:, :4])
    torch.testing.assert_close(batch['target_last_hidden_states'][0], fields['hidden_states'][:, 8:])
    assert batch['lm_head_weight'] is head

@pytest.mark.parametrize('packed', [False, True])
def test_cpu_capture_materialization_keeps_separately_resident_head(packed):
    fields, metadata = _sample()
    head = torch.empty(8, 4, dtype=torch.bfloat16, device='meta')
    kwargs = dict(lm_head_weight=head, device='cpu', target_hidden_size=4, preserve_lm_head_device=True)
    if packed:
        batch, count = materialize_packed_dspark_batch([(fields, metadata)], target_length=16384, max_packed_bytes=100000, **kwargs)
        assert count == 1
    else:
        batch = materialize_dspark_batch(fields, metadata, **kwargs)
    assert batch['lm_head_weight'] is head
    assert batch['target_last_hidden_states'].device.type == 'cpu'
    torch.testing.assert_close(batch['target_last_hidden_states'][0, :3], fields['hidden_states'][:, 8:])

def test_rejects_misaligned_rows_and_empty_supervision():
    fields, metadata = _sample()
    head = torch.ones(8, 4, dtype=torch.bfloat16)
    fields['position_ids'][1] = 9
    with pytest.raises(ValueError, match='complete prefill'):
        materialize_dspark_batch(fields, metadata, lm_head_weight=head, device='cpu', target_hidden_size=4)
    fields['position_ids'][1] = 1
    fields['loss_mask'].zero_()
    with pytest.raises(ValueError, match='supervised token'):
        materialize_dspark_batch(fields, metadata, lm_head_weight=head, device='cpu', target_hidden_size=4)
    fields['loss_mask'][:] = torch.tensor([1, 0, 1], dtype=torch.float32)
    with pytest.raises(ValueError, match='token mapping'):
        materialize_dspark_batch(fields, metadata, lm_head_weight=head, device='cpu', target_hidden_size=4)

@pytest.mark.parametrize('tier', PACKING_TIERS)
def test_packing_tiers_respect_ordered_prefix_and_byte_budget(tier: int):
    assert require_packing_tier(tier) == tier
    plan = plan_packed_prefix([tier // 4] * 5, [tier] * 5, target_length=tier, block_size=2, max_samples=16, max_packed_bytes=tier * 4)
    assert plan.sample_count == 4
    assert plan.used_tokens == tier and plan.payload_bytes == tier * 4
    assert plan.fill_ratio == 1.0 and (not plan.below_min_fill)
    with pytest.raises(ValueError, match='16K, 32K, 64K or 128K'):
        require_packing_tier(8192)

def test_packed_dspark_batch_keeps_document_and_rope_boundaries():
    first_fields, first_metadata = _sample()
    second_fields, _ = _sample()
    second_fields['input_ids'] += 10
    second_metadata = replace(first_metadata, sequence_no=1, sample_id='sample-1')
    head = torch.ones(8, 4, dtype=torch.bfloat16)
    batch, used = materialize_packed_dspark_batch(((first_fields, first_metadata), (second_fields, second_metadata)), lm_head_weight=head, device='cpu', target_hidden_size=4, target_length=16 * 1024, max_packed_bytes=1024, block_size=2)
    assert used == 2
    assert batch['input_ids'].shape == (1, 8)
    assert batch['position_ids'].tolist() == [[0, 1, 2, 0, 0, 1, 2, 0]]
    assert batch['document_ids'].tolist() == [[0, 0, 0, -1, 1, 1, 1, -1]]
    assert batch['loss_mask'].tolist() == [[0, 1, 1, 0, 0, 1, 1, 0]]
    assert len(batch['hidden_states_list']) == 2
    from verl_speco.backends.dflash_trainer_backend import DFlashTrainingModel

    class PositionHelper:
        block_size = 2

        @staticmethod
        def _cached_arange(_name, count, device, view_shape=None):
            result = torch.arange(count, device=device)
            return result.view(view_shape) if view_shape is not None else result
    context, draft = DFlashTrainingModel._create_position_ids(PositionHelper(), torch.tensor([[1, 5]]), 8, position_ids=batch['position_ids'])
    assert context.tolist() == [[0, 1, 2, 0, 0, 1, 2, 0]]
    assert draft.tolist() == [[1, 2, 1, 2]]

def test_packed_batch_rejects_mixed_generation_and_oversize_sample():
    fields, metadata = _sample()
    head = torch.ones(8, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='different target or generation'):
        materialize_packed_dspark_batch(((fields, metadata), (fields, replace(metadata, generation=2, sequence_no=1))), lm_head_weight=head, device='cpu', target_hidden_size=4, target_length=16 * 1024, max_packed_bytes=1024)
    with pytest.raises(ValueError, match='byte budget'):
        materialize_packed_dspark_batch(((fields, metadata),), lm_head_weight=head, device='cpu', target_hidden_size=4, target_length=16 * 1024, max_packed_bytes=10)
