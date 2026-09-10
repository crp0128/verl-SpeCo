from verl_speco.trainer.v1.feature_adapter import from_transfer_queue_batch


def test_feature_adapter_keeps_optional_fields_explicit():
    batch = {"input_ids": [1, 2], "attention_mask": [1, 1], "response_mask": [0, 1]}
    view = from_transfer_queue_batch(batch, global_step=7)

    assert view.input_ids == [1, 2]
    assert view.attention_mask == [1, 1]
    assert view.response_mask == [0, 1]
    assert view.hidden_states is None
    assert view.target_logprobs is None
    assert view.global_step == 7
