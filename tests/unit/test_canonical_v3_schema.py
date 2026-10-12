# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
from dataclasses import replace
import pytest
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key
from verl_speco.transport.control import ReadyRecord

def canonical():
    return SampleMetadataV3(schema_version=3, run_id='canonical-v3-schema', generation=1, producer_id='producer-0', producer_incarnation='inc-0', request_id='request-0', sample_id='sample-0', sequence_no=0, target_revision='target-1', data_version=0, tokenizer_fingerprint='a' * 64, target_model_fingerprint='b' * 64, dataset_fingerprint='c' * 64, target_layer_ids=(1, 12, 22, 32, 43), hidden_states_layout='dflash_aux_plus_last', hidden_dtype='bfloat16', token_count=16, payload_nbytes=16 * 30740, prompt_token_count=8, supervised_token_count=8)

@pytest.mark.parametrize('version', [3.0, 3.9, '3', '03', b'3', True, None, 2, 4])
def test_wire_parser_rejects_noncanonical_or_wrong_schema_without_integer_coercion(version):
    payload = canonical().to_dict()
    payload['schema_version'] = version
    with pytest.raises(ValueError):
        SampleMetadataV3.from_dict(payload)

@pytest.mark.parametrize('version', [3.0, 3.9, '3', '03', b'3', True, None, 2, 4])
def test_ready_record_cannot_accept_noncanonical_or_wrong_direct_metadata(version):
    valid = canonical()
    record = ReadyRecord(make_v3_sample_key(valid), 'producer-0', replace(valid, schema_version=version), 'd' * 64)
    with pytest.raises(ValueError):
        record.validate()

def test_canonical_integer_schema_roundtrip_and_ready_identity_are_preserved():
    valid = canonical()
    payload = valid.to_dict()
    assert type(payload['schema_version']) is int
    restored = SampleMetadataV3.from_dict(payload)
    assert restored == valid
    key = make_v3_sample_key(valid)
    assert make_v3_sample_key(restored) == key
    ReadyRecord(key, 'producer-0', restored, 'd' * 64).validate()
