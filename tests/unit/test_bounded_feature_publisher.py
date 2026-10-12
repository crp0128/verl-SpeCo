# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'KV publisher ordering, idempotency and local byte backpressure.'
from dataclasses import replace
import pytest
import torch
from verl_speco.transport.control import BackpressureError, BoundedFeaturePublisher
from verl_speco.transport.control import AcquireRequest, ReadyRouter
from _multinode_fixture_stream_control_plane import _record

class MemoryStore:

    def __init__(self) -> None:
        self.tags: dict[str, dict] = {}
        self.payloads: dict[str, dict] = {}
        self.put_count = 0
        self.fail_put = False

    def get_tag(self, key, partition):
        del partition
        return self.tags.get(key)

    def put(self, key, partition, fields, tag):
        del partition
        if self.fail_put:
            raise RuntimeError('injected TQ put failure')
        self.tags[key] = dict(tag)
        self.payloads[key] = dict(fields)
        self.put_count += 1

    def clear(self, key, partition):
        del partition
        self.tags.pop(key)
        self.payloads.pop(key)

def _publisher(store, router):
    return BoundedFeaturePublisher(store=store, router=router, partition='features:g1:s0', producer_endpoint='p1', capacity_bytes=256, high_watermark_ratio=0.5, low_watermark_ratio=0.25)

def _payload():
    return {'hidden': torch.zeros(64, dtype=torch.bfloat16)}

def test_payload_first_then_ready_duplicate_and_release() -> None:
    store, router = (MemoryStore(), ReadyRouter())
    publisher = _publisher(store, router)
    first = _record('p1', 1)
    result = publisher.publish(first.metadata, fields=_payload(), payload_checksum=first.payload_checksum)
    assert result.status == 'published'
    assert store.put_count == 1
    assert router.status()['ready_records'] == 1
    assert publisher.publish(first.metadata, fields=_payload(), payload_checksum=first.payload_checksum).status == 'duplicate'
    assert store.put_count == 1
    assert publisher.status()['paused'] is True
    second = _record('p1', 2)
    with pytest.raises(BackpressureError):
        publisher.publish(second.metadata, fields=_payload(), payload_checksum=second.payload_checksum)
    request = AcquireRequest('acq', 'consumer', 1, 1, 1, 't1', 10 ** 12)
    assert router.acquire(request).records[0].key == first.key
    router.ack('acq')
    publisher.release(first.key)
    assert publisher.status()['used_bytes'] == 0
    assert publisher.status()['paused'] is False
    assert publisher.publish(first.metadata, fields=_payload(), payload_checksum=first.payload_checksum).status == 'duplicate'
    assert store.put_count == 1
    assert publisher.publish(second.metadata, fields=_payload(), payload_checksum=second.payload_checksum).status == 'published'

def test_put_failure_leaves_no_ready_or_quota_and_conflict_fails_closed() -> None:
    store, router = (MemoryStore(), ReadyRouter())
    publisher = _publisher(store, router)
    first = _record('p1', 1)
    store.fail_put = True
    with pytest.raises(RuntimeError, match='injected TQ'):
        publisher.publish(first.metadata, fields=_payload(), payload_checksum=first.payload_checksum)
    assert publisher.status()['used_bytes'] == 0
    assert router.status()['ready_records'] == 0
    store.fail_put = False
    publisher.publish(first.metadata, fields=_payload(), payload_checksum=first.payload_checksum)
    with pytest.raises(ValueError, match='rejected sample key'):
        publisher.publish(replace(first.metadata, data_version=2), fields=_payload(), payload_checksum=first.payload_checksum)
    with pytest.raises(ValueError, match='payload_nbytes mismatch'):
        publisher.publish(_record('p1', 2).metadata, fields={'hidden': torch.zeros(1)}, payload_checksum=first.payload_checksum)
