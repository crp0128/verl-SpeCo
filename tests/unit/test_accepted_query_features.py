# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
import pytest
import torch
from verl_speco.integration.multinode_online import AcceptedQueryFeatures

def capture(**kwargs):
    return AcceptedQueryFeatures(request_id='run/g1/collection', prompt_token_ids=(10, 11, 12), layer_ids=(1, 3), hidden_size=2, max_capture_tokens=32, max_query_tail=8, dtype=torch.float32, **kwargs)

def observe(c, ids, start, tag=0):
    states = torch.tensor([[tag + position, tag + position + 0.5] for position in range(start, start + len(ids))])
    c.observe(token_ids=tuple(ids), positions=tuple(range(start, start + len(ids))), auxiliary=(states, states + 100), final_hidden=states + 200)
    return states

def test_rejected_query_tails_never_enter_completed_features():
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20, 99, 98, 97], 3, tag=1000)
    observe(c, [21, 88, 87, 86], 4, tag=2000)
    result = c.finish((20, 21, 22))
    assert result.token_ids == (10, 11, 12, 20, 21) and result.positions == tuple(range(5))
    assert result.hidden_states[0][:, 0].tolist() == [0, 1, 2, 1003, 2004]
    assert result.hidden_states[1][:, 0].tolist() == [100, 101, 102, 1103, 2104]
    assert c.discarded_query_rows == 6 and c.observed_forwards == 3
    with pytest.raises(RuntimeError, match='closed'):
        observe(c, [23], 5)

def test_full_acceptance_preserves_all_query_rows():
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20, 21, 22], 3)
    observe(c, [23, 24], 6)
    result = c.finish((20, 21, 22, 23, 24, 25))
    assert result.token_ids == (10, 11, 12, 20, 21, 22, 23, 24)
    assert c.discarded_query_rows == 0

def test_async_terminal_query_excludes_final_anchor_and_all_eight_proposals():
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20, 21], 3)
    observe(c, [22] + [99] * 8, 5, tag=1000)
    result = c.finish((20, 21, 22))
    assert result.token_ids == (10, 11, 12, 20, 21)
    assert result.hidden_states[0][:, 0].tolist() == [0, 1, 2, 3, 4]
    assert c.discarded_query_rows == 9

def test_async_terminal_allowance_still_rejects_an_incorrect_training_prefix():
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20, 99], 3)
    observe(c, [22] + [99] * 8, 5)
    with pytest.raises(RuntimeError, match='first_mismatch=4'):
        c.finish((20, 21, 22))

def test_async_terminal_discard_cannot_exceed_one_complete_verification_query():
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20, 21], 3)
    observe(c, [22] + [99] * 9, 5)
    with pytest.raises(RuntimeError, match='terminal_query_bound=9'):
        c.finish((20, 21, 22))

def test_live_rejection_bound_remains_eight_despite_terminal_allowance():
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20] + [99] * 8, 3)
    with pytest.raises(RuntimeError, match='speculative tail'):
        observe(c, [21], 3)

def test_chunked_prefill_and_extra_final_input_are_aligned():
    c = capture()
    observe(c, [10], 0)
    observe(c, [11, 12], 1)
    observe(c, [20, 21], 3)
    result = c.finish((20, 21))
    assert result.token_ids == (10, 11, 12, 20) and c.discarded_query_rows == 1

def test_capture_owns_tensors_before_native_buffers_are_reused():
    c = capture()
    buffer = observe(c, [10, 11, 12], 0)
    buffer.fill_(9999)
    result = c.finish((20,))
    assert result.hidden_states[0][:, 0].tolist() == [0, 1, 2]
    assert result.hidden_states[0].shape == (3, 2)

@pytest.mark.parametrize('ids,start,message', [([20], 4, 'gap'), ([99], 0, 'new request'), ([99], 1, 'prompt')])
def test_bad_query_identity_refuses_without_losing_prior_observation(ids, start, message):
    c = capture()
    observe(c, [10, 11, 12], 0)
    with pytest.raises(RuntimeError, match=message):
        observe(c, ids, start)
    assert c.finish((20,)).token_ids == (10, 11, 12)

@pytest.mark.parametrize('response', [(20, 22, 23), (20, 21, 22, 23, 24, 25)])
def test_completion_refuses_wrong_or_missing_response_features(response):
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20, 21], 3)
    with pytest.raises(RuntimeError, match='completed response'):
        c.finish(response)

def test_query_storage_bound_is_checked_before_cloning_or_trimming():
    c = capture()
    observe(c, [10, 11, 12], 0)
    with pytest.raises(RuntimeError, match='bounded'):
        observe(c, [20] * 30, 3)
    assert c.finish((20,)).token_ids == (10, 11, 12)

def test_rejection_beyond_verified_tail_is_refused():
    c = capture()
    observe(c, [10, 11, 12], 0)
    observe(c, [20] * 10, 3)
    with pytest.raises(RuntimeError, match='speculative tail'):
        observe(c, [21], 4)

def test_feature_geometry_refused_before_state_change():
    c = capture()
    observe(c, [10, 11, 12], 0)
    bad = torch.ones(1, 3)
    with pytest.raises(ValueError, match='shape'):
        c.observe(token_ids=(20,), positions=(3,), auxiliary=(bad, bad), final_hidden=bad)
    assert c.finish((20,)).token_ids == (10, 11, 12)

def test_noncontiguous_query_positions_are_refused():
    c = capture()
    x = torch.ones(3, 2)
    with pytest.raises(RuntimeError, match='noncontiguous'):
        c.observe(token_ids=(10, 11, 12), positions=(0, 1, 3), auxiliary=(x, x), final_hidden=x)
