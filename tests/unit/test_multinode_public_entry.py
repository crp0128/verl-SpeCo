# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import pytest
from verl_speco.runtime.multinode_cli import main, read_config, resolve
from verl_speco.runtime.external_standalone import ExternalProducerGroup, require_live_nodes
from verl_speco.runtime.planning import resolve_launch_plan
ROOT = Path(__file__).resolve().parents[2]

def test_file_lock_uses_native_module_when_available():
    code = "import sys, types; native=types.ModuleType('fcntl'); sys.modules['fcntl']=native; from verl_speco.runtime.artifacts import fcntl; assert fcntl is native"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

def example():
    config = read_config(ROOT / 'examples/multinode/standalone.example.yaml')
    table = read_config(ROOT / 'examples/multinode/rank_table.example.json')
    return (config, table)

def test_public_dry_run_does_not_import_accelerator_or_ray():
    code = "from verl_speco.runtime.multinode_cli import main; import sys; main(sys.argv[1:]); assert not {'ray','torch','torch_npu','vllm','transfer_queue'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, '-c', code, 'dry-run', '--config', str(ROOT / 'examples/multinode/standalone.example.yaml'), '--rank-table', str(ROOT / 'examples/multinode/rank_table.example.json')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['draft_world_size'] == 2

def test_generic_alias_and_variable_device_counts():
    config, table = example()
    config['speco']['multinode_runtime']['topology']['nproc_per_draft_node'] = 3
    assert resolve(config, table)['draft_world_size'] == 3
    table['nodes'][1]['device_count'] = 2
    with pytest.raises(ValueError):
        resolve(config, table)

def test_runtime_aliases_cannot_conflict():
    config, table = example()
    config['speco']['speco_runtime'] = config['speco']['multinode_runtime']
    with pytest.raises(ValueError, match='one runtime'):
        resolve(config, table)

def test_actual_node_membership_is_checked():
    config, table = example()
    plan = resolve_launch_plan(config, table)
    with pytest.raises(RuntimeError, match='not alive'):
        require_live_nodes(plan, [])
    nodes = [{'Alive': True, 'NodeManagerAddress': n.address, 'NodeID': n.node_id} for n in plan.nodes]
    assert len(require_live_nodes(plan, nodes)) == 2

def test_all_producer_progress_is_merged_without_duplicate_credit():

    class Queue:

        def __init__(self):
            self.events = []

        def put(self, event):
            self.events.append(event)
    queue = Queue()
    group = ExternalProducerGroup([object(), object()], queue)

    async def check():
        await group.emit('0', {'kind': 'samples_published', 'published_total': 4})
        await group.emit('1', {'kind': 'samples_published', 'published_total': 3})
        await group.emit('0', {'kind': 'samples_published', 'published_total': 4})
        with pytest.raises(ValueError, match='backwards'):
            await group.emit('0', {'kind': 'samples_published', 'published_total': 2})
        with pytest.raises(ValueError, match='foreign'):
            await group.emit('other', {'kind': 'samples_published', 'published_total': 1})
    asyncio.run(check())
    assert [event['published_total'] for event in queue.events] == [4, 7, 7]

def test_dry_run_writes_readable_plan(tmp_path):
    output = tmp_path / 'plan.json'
    assert main(['dry-run', '--config', str(ROOT / 'examples/multinode/standalone.example.yaml'), '--rank-table', str(ROOT / 'examples/multinode/rank_table.example.json'), '--output', str(output)]) == 0
    plan = json.loads(output.read_text(encoding='utf-8'))
    assert plan['master_addr'] == '10.0.0.2'

def test_unknown_execution_backend_is_rejected_before_runtime_start():
    config, table = example()
    config['speco']['multinode_runtime']['execution']['backend'] = 'unknown'
    assert resolve(config, table)['draft_world_size'] == 2


def test_effective_tq_binding_matches_plan_and_generation():
    from verl_speco.runtime.external_standalone import build_external_commands
    from verl_speco.standalone_ray_runtime import compose_runtime_config

    config, table = example()
    runtime = config['speco']['multinode_runtime']
    runtime['transfer_queue']['ready_shards'] = 2
    runtime['ray']['namespace'] = 'review-namespace'
    observed = []
    for generation in (1, 2):
        runtime['generation'] = generation
        plan = resolve_launch_plan(config, table)
        commands = build_external_commands(config, plan)
        producer = compose_runtime_config(commands.producer_overrides)
        consumer = compose_runtime_config(commands.consumer_overrides)
        for actual in (producer, consumer):
            tq = actual.actor_rollout_ref.rollout.drafter.training.transfer_queue
            assert list(tq.partition_ids) == list(plan.tq_partitions)
            assert tq.partition_id == plan.tq_partitions[0]
            assert tq.ray.namespace == 'review-namespace'
            assert tq.ray.address == runtime['ray']['address']
        assert resolve(config, table)['external_transfer_queue']['partition_ids'] == list(plan.tq_partitions)
        observed.append(set(plan.tq_partitions))
    assert observed[0].isdisjoint(observed[1])


@pytest.mark.parametrize('device', ['npu', 'cuda'])
def test_actual_worker_keeps_logical_rank_zero_for_second_physical_card(monkeypatch, device):
    import os
    from types import SimpleNamespace
    import ray
    import verl.utils.device as device_api
    import verl.single_controller.base.worker as base_worker
    import verl_speco.trainer.draft_training_loop as loop
    from verl_speco.runtime.external_standalone import placed_drafter_class

    selected = []
    fake_device = SimpleNamespace(set_device=lambda index: selected.append(index))
    monkeypatch.setattr(device_api, 'get_device_name', lambda: device)
    monkeypatch.setattr(device_api, 'get_torch_device', lambda: fake_device)
    monkeypatch.setattr(base_worker, 'get_torch_device', lambda: fake_device)
    monkeypatch.setattr(base_worker, 'is_npu_available', device == 'npu')
    resource = 'NPU' if device == 'npu' else 'GPU'
    monkeypatch.setattr(ray, 'get_runtime_context', lambda: SimpleNamespace(get_accelerator_ids=lambda: {resource: ['1']}))
    for key in list(os.environ):
        if key.startswith('RAY_EXPERIMENTAL_NOSET_') or key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
            monkeypatch.delenv(key)
    variable = 'ASCEND_RT_VISIBLE_DEVICES' if device == 'npu' else 'CUDA_VISIBLE_DEVICES'
    for key, value in dict(WORLD_SIZE='2', RANK='1', LOCAL_WORLD_SIZE='1', LOCAL_RANK='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29500', **{variable: '1'}).items():
        monkeypatch.setenv(key, value)
    cls = placed_drafter_class()
    worker = cls({}, assigned_device_id=1, device_name=device)
    assert worker._local_rank == 0 and os.environ['LOCAL_RANK'] == '0'
    monkeypatch.setattr(loop, 'get_device_name', lambda: device)
    monkeypatch.setattr(loop, 'get_torch_device', lambda: fake_device)
    loop._configure_device(worker._local_rank)
    assert selected == [0]
    with pytest.raises(RuntimeError, match='does not match placement'):
        cls({}, assigned_device_id=2, device_name=device)
    assert selected == [0]  # invalid placement is rejected before touching the device.


@pytest.mark.parametrize('device', ['npu', 'cuda'])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_external_startup_creates_all_replicas_and_passes_bindings(monkeypatch, device, cleanup_fails):
    from types import SimpleNamespace
    import ray
    import ray.util.queue
    import verl.single_controller.ray as controller
    import verl_speco.runtime.external_standalone as external
    import verl_speco.standalone_ray_runtime as runtime_module

    config, table = example()
    runtime = config['speco']['multinode_runtime']
    runtime['topology']['rollout_replicas_per_node'] = 2
    runtime['execution']['device'] = device
    runtime['execution']['vllm_endpoints'] *= 2
    plan = resolve_launch_plan(config, table)
    created, killed, closed, bindings = [], [], [], []

    class Remote:
        def __init__(self, cls, options=None):
            self.cls, self.settings = cls, options or {}
        def options(self, **options):
            return Remote(self.cls, dict(self.settings, **options))
        def remote(self, *args, **kwargs):
            instance = self.cls(*args, **kwargs)
            created.append((instance, self.settings, args, kwargs))
            class Handle:
                def __getattr__(self, name):
                    def call(*arguments):
                        result = getattr(instance, name)(*arguments)
                        return asyncio.run(result) if asyncio.iscoroutine(result) else result
                    return SimpleNamespace(remote=call)
            return Handle()

    class Queue:
        def __init__(self, **kwargs): pass
        def shutdown(self): closed.append('queue')

    class Worker:
        def __init__(self, *args, **kwargs): pass

    class Trainer(runtime_module.StandaloneRayTrainer):
        def run(self):
            self.init_workers()
            bindings.append(self.consumer_config.actor_rollout_ref.rollout.drafter.training.transfer_queue)
            return 0

    monkeypatch.setattr(ray, 'remote', lambda cls=None, **options: Remote(cls, options) if cls is not None else lambda target: Remote(target, options))
    monkeypatch.setattr(ray, 'get', lambda value: value)
    def kill(actor, **kwargs):
        killed.append(actor)
        if cleanup_fails and len(killed) == 1:
            raise OSError('cleanup injected failure')
    monkeypatch.setattr(ray, 'kill', kill)
    monkeypatch.setattr(ray, 'is_initialized', lambda: False)
    monkeypatch.setattr(ray, 'init', lambda **kwargs: bindings.append(kwargs))
    monkeypatch.setattr(ray, 'shutdown', lambda: closed.append('ray'))
    monkeypatch.setattr(ray, 'nodes', lambda: [{'Alive': True, 'NodeManagerAddress': n.address, 'NodeID': f'{index + 1:056x}'} for index, n in enumerate(plan.nodes)])
    monkeypatch.setattr(ray.util.queue, 'Queue', Queue)
    monkeypatch.setattr(controller, 'RayClassWithInitArgs', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(controller, 'RayWorkerGroup', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(runtime_module, 'StandaloneRayTrainer', Trainer)
    monkeypatch.setattr(external, 'placed_drafter_class', lambda: Worker)
    if cleanup_fails:
        with pytest.raises(OSError, match='cleanup injected failure'):
            external.run_external_standalone(config, plan)
    else:
        assert external.run_external_standalone(config, plan) == 0
    producer_rows = [row for row in created if isinstance(row[0], external.PlacedProducer)]
    worker_rows = [row for row in created if isinstance(row[0], Worker)]
    assert len(producer_rows) == 2 and len(worker_rows) == plan.draft_world_size
    assert [row[2][0].speco.standalone_tq_producer.input_shard_index for row in producer_rows] == [0, 1]
    assert all(row[2][0].speco.standalone_tq_producer.input_shard_count == 2 for row in producer_rows)
    assert [row[3]['assigned_device_id'] for row in worker_rows] == [0, 1]
    assert all(row[1]['runtime_env']['env_vars']['LOCAL_RANK'] == '0' for row in worker_rows)
    assert all(not row[1]['runtime_env']['env_vars']['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] for row in worker_rows)
    assert list(bindings[-1].partition_ids) == list(plan.tq_partitions)
    assert len(killed) == len(created) and closed == ['queue', 'queue', 'ray']


@pytest.mark.parametrize('connection', ['fresh', 'attached', 'wrong_namespace'])
def test_preflight_failure_closes_only_its_own_ray_connection(monkeypatch, connection):
    from types import SimpleNamespace
    import ray
    from verl_speco.runtime.external_standalone import run_external_standalone

    config, table = example()
    plan = resolve_launch_plan(config, table)
    events = []
    monkeypatch.setattr(ray, 'is_initialized', lambda: connection != 'fresh')
    monkeypatch.setattr(ray, 'get_runtime_context', lambda: SimpleNamespace(namespace='foreign' if connection == 'wrong_namespace' else 'transfer_queue'))
    monkeypatch.setattr(ray, 'init', lambda **kwargs: events.append('init'))
    monkeypatch.setattr(ray, 'shutdown', lambda: events.append('shutdown'))
    monkeypatch.setattr(ray, 'nodes', lambda: [])
    error = ValueError if connection == 'wrong_namespace' else RuntimeError
    with pytest.raises(error, match='namespace|not alive'):
        run_external_standalone(config, plan)
    assert events == (['init', 'shutdown'] if connection == 'fresh' else [])


def test_standalone_command_rejects_online_plan_before_ray_start(tmp_path):
    config, table = example()
    config['speco']['multinode_runtime']['mode'] = 'online_rl'
    path = tmp_path / 'online.json'
    ranks = tmp_path / 'ranks.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    ranks.write_text(json.dumps(table), encoding='utf-8')
    with pytest.raises(ValueError, match='mode=standalone'):
        main(['standalone', '--config', str(path), '--rank-table', str(ranks)])


def test_partial_queue_bootstrap_cleans_up_first_queue(monkeypatch):
    import ray
    import ray.util.queue
    from verl_speco.runtime.external_standalone import run_external_standalone

    config, table = example()
    plan = resolve_launch_plan(config, table)
    events = []
    class Queue:
        created = 0
        def __init__(self, **kwargs):
            Queue.created += 1
            if Queue.created == 2:
                raise OSError('second queue failed')
            events.append('queue created')
        def shutdown(self): events.append('queue closed')
    monkeypatch.setattr(ray.util.queue, 'Queue', Queue)
    monkeypatch.setattr(ray, 'is_initialized', lambda: False)
    monkeypatch.setattr(ray, 'init', lambda **kwargs: events.append('init'))
    monkeypatch.setattr(ray, 'shutdown', lambda: events.append('shutdown'))
    monkeypatch.setattr(ray, 'nodes', lambda: [{'Alive': True, 'NodeManagerAddress': n.address, 'NodeID': f'{i + 1:056x}'} for i, n in enumerate(plan.nodes)])
    with pytest.raises(OSError, match='second queue failed'):
        run_external_standalone(config, plan)
    assert events == ['init', 'queue created', 'queue closed', 'shutdown']
