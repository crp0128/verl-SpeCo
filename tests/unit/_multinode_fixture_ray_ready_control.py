# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Shared ready owner invariants before running it inside a Ray actor.'
from dataclasses import replace
from copy import deepcopy
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import pytest
from verl_speco.transport.control import RayReadyControl, actor_name_for_run, ready_control_owner_node_id
from verl_speco.transport.control import AcquireRequest, ReadyRecord
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key

def _record(producer: str, sequence: int) -> ReadyRecord:
    metadata = SampleMetadataV3(schema_version=3, run_id='run-control', generation=1, producer_id=producer, producer_incarnation='inc-1', request_id=f'request-{producer}-{sequence}', sample_id=f'sample-{producer}-{sequence}', sequence_no=sequence, target_revision='target-1', data_version=0, tokenizer_fingerprint='tokenizer-1', target_model_fingerprint='model-1', target_layer_ids=(1, 3), hidden_states_layout='token_major', hidden_dtype='bfloat16', token_count=4, payload_nbytes=128)
    return ReadyRecord(make_v3_sample_key(metadata), producer, metadata, 'a' * 64)

def _published_owner_and_tq(monkeypatch):
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=512)
    records = (_record('p1', 1), _record('p2', 2))
    payloads = {}
    for record in records:
        token = owner.reserve(record, nbytes=128).token
        owner.publish(record, token=token)
        payloads[record.key] = object()
    fake = SimpleNamespace(init=lambda: None, kv_list=lambda *, partition_id: {partition_id: dict(payloads)}, kv_clear=lambda *, keys, partition_id: payloads.pop(keys, None))
    monkeypatch.setitem(sys.modules, 'transfer_queue', fake)
    return (owner, records, payloads, fake)
