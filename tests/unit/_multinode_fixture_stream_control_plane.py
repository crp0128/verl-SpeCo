# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Behavioral checks for the protocol-v3 control plane.'
import time
from dataclasses import replace
import pytest
from verl_speco.transport.control import LeaseRegistry, LeaseState
from verl_speco.transport.control import AcquireRequest, AcquireResult, ReadyRecord, ReadyRouter
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_ready_tag, make_v3_sample_key, parse_v3_ready_tag, stable_acquire_id
from verl_speco.transport.streaming import TrainingStreamCoordinator, TrainingTransaction

def _record(producer: str, sequence: int, *, generation: int=1, revision: str='t1') -> ReadyRecord:
    metadata = SampleMetadataV3(schema_version=3, run_id='run1', generation=generation, producer_id=producer, producer_incarnation='inc1', request_id=f'request-{producer}-{sequence}', sample_id=f'sample-{producer}-{sequence}', sequence_no=sequence, target_revision=revision, data_version=0, tokenizer_fingerprint='tok1', target_model_fingerprint='model1', target_layer_ids=(3, 5), hidden_states_layout='token_major', hidden_dtype='bfloat16', token_count=4, payload_nbytes=128)
    return ReadyRecord(make_v3_sample_key(metadata), producer, metadata, 'a' * 64)

def _request(acquire_id: str, *, min_items: int=1, max_items: int=1) -> AcquireRequest:
    return AcquireRequest(acquire_id, 'consumer', 1, max_items, min_items, 't1', time.time() + 60)
