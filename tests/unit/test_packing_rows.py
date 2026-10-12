# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Per-DP packing metadata keeps ordered original-token accounting explicit.'
import pytest
from verl_speco.trainer.distributed_batches import plan_packed_rows

def plan(lengths, **overrides):
    args = dict(target_length=131072, block_size=8, max_rows=16, max_samples_per_row=32, max_bytes_per_row=8 * 1024 ** 3)
    args.update(overrides)
    return plan_packed_rows(lengths, [n * 30740 for n in lengths], **args)

def test_sixteen_per_dp_128k_rows_are_two_million_original_tokens_not_global_128k():
    result = plan([32768] * 64)
    assert len(result.rows) == 16
    assert all((r.used_tokens == 131072 and r.sample_count == 4 and (r.fill_ratio == 1) for r in result.rows))
    assert result.original_tokens == result.padded_tokens == 16 * 131072
    assert result.sample_count == 64 and result.deferred_samples == 0
    assert result.sample_indices == tuple((tuple(range(i * 4, i * 4 + 4)) for i in range(16)))

@pytest.mark.parametrize('tier', [16384, 32768, 65536, 131072])
def test_fifo_documents_are_not_split_between_rows_and_partial_tail_is_explicit(tier):
    lengths = [tier // 2 - 1] * 7
    result = plan(lengths, target_length=tier, max_rows=2)
    assert result.sample_indices == ((0, 1), (2, 3))
    assert result.original_tokens == sum(lengths[:4])
    assert result.padded_tokens == tier * 2
    assert result.deferred_samples == 3

def test_byte_and_document_caps_apply_per_row():
    lengths = [1024] * 7
    result = plan(lengths, max_rows=2, max_bytes_per_row=1024 * 30740 * 2)
    assert result.sample_indices == ((0, 1), (2, 3))
    result = plan(lengths, max_rows=2, max_samples_per_row=1)
    assert result.sample_indices == ((0,), (1,)) and result.deferred_samples == 5

@pytest.mark.parametrize('changes', [{'max_rows': True}, {'max_samples_per_row': 0}, {'max_bytes_per_row': -1}, {'block_size': False}, {'target_length': 2048}])
def test_invalid_or_boolean_budgets_are_rejected(changes):
    with pytest.raises(ValueError):
        plan([1024], **changes)

def test_oversized_sample_is_rejected_instead_of_skipping_a_stream_position():
    with pytest.raises(ValueError, match='one sample exceeds'):
        plan([200000])
    with pytest.raises(ValueError, match='one sample exceeds'):
        plan([131072], max_bytes_per_row=100)
