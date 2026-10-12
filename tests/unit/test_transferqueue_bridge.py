# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import sys

import pytest
import torch

from verl_speco.integration import transferqueue_bridge as bridge


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeTQ:
    def __init__(self) -> None:
        self.init_calls = []
        self.close_calls = 0
        self.clear_calls = []
        self.records: dict[str, tuple[dict, dict]] = {}
        self.client = _FakeClient()

    def init(self, config=None):
        self.init_calls.append(config)

    def kv_put(self, *, key, partition_id, fields, tag):
        self.records[key] = (dict(fields), dict(tag))

    def kv_list(self, *, partition_id):
        return {key: tag for key, (_, tag) in self.records.items()}

    def kv_batch_get(self, *, keys, partition_id):
        return {key: self.records[key][0] for key in keys}

    def kv_clear(self, *, keys, partition_id):
        self.clear_calls.append((list(keys), partition_id))
        for key in keys:
            self.records.pop(key, None)

    def get_client(self):
        return self.client

    def close(self):
        self.close_calls += 1


class _FakeRay:
    def __init__(self) -> None:
        self.initialized = False
        self.init_calls = []
        self.shutdown_calls = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.initialized = True
        self.init_calls.append(kwargs)

    def shutdown(self):
        self.initialized = False
        self.shutdown_calls += 1


@pytest.fixture
def fake_runtime(monkeypatch):
    fake_tq = _FakeTQ()
    fake_ray = _FakeRay()
    monkeypatch.setattr(bridge, "tq", fake_tq)
    monkeypatch.setattr(bridge, "_TQ_IMPORTABLE", True)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        bridge,
        "_state",
        {
            "enabled": False,
            "configured": False,
            "initialized": False,
            "config": None,
            "owner": False,
            "ray_initialized_here": False,
            "ray_address": None,
            "ray_namespace": None,
        },
    )
    return fake_tq, fake_ray


def _config():
    return {
        "enable": True,
        "package_version": "0.1.10",
        "partition_id": "speco_drafter_features",
        "run_id": "run-a",
        "schema_version": 1,
        "ray": {"address": "ray-head:6379", "namespace": "speco-drafter"},
        "controller": {"polling_mode": True},
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {"total_storage_size": 16, "num_data_storage_units": 1},
        },
    }


def test_generation_shards_preserve_order_and_isolate_records(fake_runtime, monkeypatch):
    class ShardedTQ(_FakeTQ):
        def kv_put(self, *, key, partition_id, fields, tag):
            self.records[(partition_id, key)] = (fields, tag)
        def kv_list(self, *, partition_id):
            # TransferQueue 0.1.10 returns the partition wrapper, including
            # an empty mapping when a generation has no records yet.
            return {partition_id: {key: tag for (part, key), (_, tag) in self.records.items() if part == partition_id}}
        def kv_batch_get(self, *, keys, partition_id):
            return {key: self.records[(partition_id, key)][0] for key in keys}
        def kv_clear(self, *, keys, partition_id):
            for key in keys:
                del self.records[(partition_id, key)]

    fake = ShardedTQ()
    monkeypatch.setattr(bridge, 'tq', fake)
    config = _config()
    config.update(partition_id='features:g1:s0', partition_ids=['features:g1:s0', 'features:g1:s1'])
    assert bridge.configure_transfer_queue(config)
    keys = [f'key-{i}' for i in range(20)]
    for index, key in enumerate(keys):
        bridge.put_sample(key, {'value': torch.tensor([index])}, tag={'index': index})
    assert {partition for partition, _ in fake.records} == set(config['partition_ids'])
    assert set(bridge.list_samples()) == set(keys)
    fetched = bridge.get_samples(keys[::-1])
    assert [int(fields['value'].item()) for _, fields in fetched] == list(range(19, -1, -1))
    assert int(bridge.get_sample(keys[1])['value'].item()) == 1
    native = bridge._native_tq_config(config)
    assert 'partition_ids' not in native
    monkeypatch.setitem(bridge._state, 'config', dict(config,
        partition_id='features:g2:s0', partition_ids=['features:g2:s0', 'features:g2:s1']))
    assert bridge.list_samples() == {}
    bridge.put_sample(keys[0], {'value': torch.tensor([99])})
    assert int(bridge.get_sample(keys[0])['value'].item()) == 99
    monkeypatch.setitem(bridge._state, 'config', config)
    bridge.clear_samples(keys)
    assert bridge.list_samples() == {}
    assert len(fake.records) == 1  # generation 2 remains owned by its own run.


@pytest.mark.parametrize('parts', [[], ['same', 'same'], [''], 'not-a-list'])
def test_invalid_partition_list_is_rejected(fake_runtime, parts):
    config = _config()
    config['partition_ids'] = parts
    with pytest.raises(ValueError, match='partition_ids'):
        bridge.configure_transfer_queue(config)
    assert fake_runtime[0].init_calls == []


def test_partition_binding_accepts_hydra_config_without_native_field_leaks(fake_runtime):
    from omegaconf import OmegaConf
    config = _config()
    config['partition_ids'] = ['features:g1:s0', 'features:g1:s1']
    assert bridge.configure_transfer_queue(OmegaConf.create(config))
    assert bridge._partition_ids() == tuple(config['partition_ids'])
    assert 'partition_ids' not in bridge._native_tq_config(bridge._state['config'])


def test_owner_connects_ray_and_receives_only_native_tq_config(fake_runtime) -> None:
    fake_tq, fake_ray = fake_runtime
    config = _config()
    assert bridge.configure_transfer_queue(config)
    bridge.connect_ray_cluster("ray-head:6379", "speco-drafter")
    bridge.start_transfer_queue_owner(config)

    assert fake_ray.init_calls == [
        {"address": "ray-head:6379", "namespace": "speco-drafter"}
    ]
    native = bridge._to_plain_dict(fake_tq.init_calls[0])
    assert set(native) == {"controller", "backend"}
    assert bridge._state["owner"] is True

    bridge.close_transfer_queue_owner()
    assert fake_tq.close_calls == 1
    assert fake_ray.shutdown_calls == 1


def test_client_put_list_get_many_clear_and_local_close(fake_runtime) -> None:
    fake_tq, fake_ray = fake_runtime
    assert bridge.configure_transfer_queue(_config())
    bridge.connect_ray_cluster("ray-head:6379", "speco-drafter")
    bridge.connect_transfer_queue_client()
    native = bridge._to_plain_dict(fake_tq.init_calls[0])
    assert set(native) == {"controller", "backend"}
    assert native["backend"]["storage_backend"] == "SimpleStorage"
    assert native["backend"]["SimpleStorage"] == {
        "total_storage_size": 16,
        "num_data_storage_units": 1,
    }

    bridge.put_sample(
        "k0",
        {"hidden_states": torch.ones(2, 4), "ignored": None},
        tag={"status": "ready"},
    )
    bridge.put_sample(
        "k1",
        {"hidden_states": torch.zeros(3, 4)},
        tag={"status": "ready"},
    )

    assert bridge.list_samples() == {
        "k0": {"status": "ready"},
        "k1": {"status": "ready"},
    }
    records = bridge.get_samples(["k1", "k0"])
    assert [key for key, _ in records] == ["k1", "k0"]
    assert records[0][1]["hidden_states"].shape == (3, 4)

    bridge.clear_samples(["k0", "k1"])
    assert bridge.list_samples() == {}
    bridge.close_transfer_queue_client()
    assert fake_tq.client.closed is True
    assert fake_tq.close_calls == 0
    assert fake_ray.shutdown_calls == 1


def test_client_close_cannot_be_used_by_owner(fake_runtime) -> None:
    _, _ = fake_runtime
    config = _config()
    bridge.configure_transfer_queue(config)
    bridge.connect_ray_cluster("ray-head:6379", "speco-drafter")
    bridge.start_transfer_queue_owner(config)
    with pytest.raises(RuntimeError, match="owner"):
        bridge.close_transfer_queue_client()
