# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Standalone SpeCo workers on an explicitly placed external Ray cluster.

This runtime reuses the existing scheduler, feature-store, checkpoint and
training loop. It never starts or stops external Ray, TQ or vLLM services.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


def external_tq_binding(config, plan):
    """Connection and generation shards shared by the external owner and workers."""
    from verl_speco.runtime.planning import launch_plan_runtime_config

    runtime = launch_plan_runtime_config(config)
    namespace = runtime['ray'].get('namespace', 'transfer_queue')
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError('external Ray namespace must be a nonempty string')
    return dict(run_id=plan.run_id, partition_id=plan.tq_partitions[0],
                partition_ids=list(plan.tq_partitions),
                ray=dict(address=runtime['ray']['address'], namespace=namespace))


def build_external_commands(config, plan):
    from verl_speco.runtime.planning import NodeRole, launch_plan_runtime_config
    from verl_speco.standalone_tq_training_launcher import PipelineCommands

    if plan.mode != 'standalone':
        raise ValueError('external standalone requires a standalone launch plan')
    runtime = launch_plan_runtime_config(config)
    execution = runtime['execution']
    producers = [p for p in plan.placements if p.role == NodeRole.ROLLOUT]
    endpoints = tuple(execution['vllm_endpoints'])
    if len(endpoints) != len(producers):
        raise ValueError('one explicit vLLM endpoint is required per producer replica')
    binding = external_tq_binding(config, plan)
    prefix = 'actor_rollout_ref.rollout.drafter.training.transfer_queue'
    overrides = (
        *execution.get('overrides', ()),
        f"speco.draft_training.nnodes={runtime['topology']['draft_nodes']}",
        f"speco.draft_training.nproc_per_node={runtime['topology']['nproc_per_draft_node']}",
        *(f'{prefix}.{key}={json.dumps(binding[key])}' for key in ('run_id', 'partition_id', 'partition_ids')),
        *(f'{prefix}.ray.{key}={json.dumps(value)}' for key, value in binding['ray'].items()),
    )
    return PipelineCommands(None, endpoints, [], [], [], overrides, overrides)


def placed_drafter_class():
    """Check the Ray allocation before the upstream Worker touches a device."""
    from verl_speco.workers.standalone_drafter_worker import StandaloneDrafterWorker

    class PlacedDrafter(StandaloneDrafterWorker):
        def __init__(self, *args, assigned_device_id, device_name, **kwargs):
            import ray
            from verl.utils.device import get_device_name
            from verl.utils.ray_utils import ray_noset_visible_devices

            resource = 'NPU' if device_name == 'npu' else 'GPU'
            allocated = [str(value) for value in ray.get_runtime_context().get_accelerator_ids().get(resource, [])]
            variable = 'ASCEND_RT_VISIBLE_DEVICES' if device_name == 'npu' else 'CUDA_VISIBLE_DEVICES'
            visible = os.environ.get(variable, '').split(',')
            if allocated != [str(assigned_device_id)] or visible != allocated:
                raise RuntimeError(f'Ray device allocation does not match placement: expected={assigned_device_id}, allocated={allocated}, visible={visible}')
            if get_device_name() != device_name or ray_noset_visible_devices():
                raise RuntimeError('placed workers require the selected device and Ray-managed visibility')
            os.environ['LOCAL_RANK'] = '0'
            super().__init__(*args, **kwargs)

    return PlacedDrafter


class ProducerEventSink:
    """Small per-producer events merged by one authoritative coordinator."""
    def __init__(self, actor, producer_id):
        self.actor, self.producer_id = actor, producer_id

    def put(self, event):
        import ray
        return ray.get(self.actor.emit.remote(self.producer_id, event))


class ExternalProducerGroup:
    def __init__(self, producers, events):
        self.producers, self.events = producers, events
        self.published = {str(index): 0 for index in range(len(producers))}

    async def bind(self, producers):
        if self.producers or not producers:
            raise ValueError("producer group can only be bound once")
        self.producers = producers
        self.published = {str(index): 0 for index in range(len(producers))}

    async def emit(self, producer_id, event):
        if producer_id not in self.published or not isinstance(event, dict):
            raise ValueError("foreign producer event")
        event = dict(event)
        if event.get("kind") == "samples_published":
            count = event.get("published_total")
            if type(count) is not int or count < self.published[producer_id]:
                raise ValueError("producer progress cannot move backwards")
            self.published[producer_id] = count
        if "published_total" in event:
            event["published_total"] = sum(self.published.values())
        await asyncio.to_thread(self.events.put, event)

    async def start(self):
        return await asyncio.gather(*(actor.start.remote() for actor in self.producers))

    async def wait(self):
        tasks = [asyncio.ensure_future(actor.wait.remote()) for actor in self.producers]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            await self.stop()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def pause(self):
        return await asyncio.gather(*(actor.pause.remote() for actor in self.producers))

    async def resume(self):
        return await asyncio.gather(*(actor.resume.remote() for actor in self.producers))

    async def stop(self):
        return await asyncio.gather(*(actor.stop.remote() for actor in self.producers), return_exceptions=True)


class PlacedProducer:
    """Create the existing async producer inside its assigned node process."""
    def __init__(self, config, sink):
        from verl_speco.standalone_ray_runtime import StandaloneProducerActor
        self.inner = StandaloneProducerActor(config, sink)

    async def start(self):
        return await self.inner.start()

    async def wait(self):
        return await self.inner.wait()

    async def pause(self):
        return await self.inner.pause()

    async def resume(self):
        return await self.inner.resume()

    async def stop(self):
        return await self.inner.stop()


def require_live_nodes(plan, nodes):
    live = {n["NodeManagerAddress"]: n for n in nodes if n.get("Alive")}
    missing = [n.address for n in plan.nodes if n.address not in live]
    if missing:
        raise RuntimeError(f"planned Ray nodes are not alive: {missing}")
    return live


def run_external_standalone(config: dict, plan) -> int:
    import ray
    from ray.util.queue import Queue
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from omegaconf import open_dict
    from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
    from verl_speco.standalone_ray_runtime import StandaloneRayTrainer
    from verl_speco.runtime.planning import NodeRole, launch_plan_runtime_config

    runtime = launch_plan_runtime_config(config)
    execution = runtime["execution"]
    if execution.get("trainer_engine", "speco_fsdp2") != "speco_fsdp2":
        raise ValueError("external_ray entry uses SpeCo FSDP2; VeOmni/native uses its explicit engine adapter")
    address = runtime["ray"]["address"]
    if address in ("local", "", None):
        raise ValueError("external standalone requires an existing Ray cluster")
    commands = build_external_commands(config, plan)
    endpoints = commands.vllm_endpoints
    producers = [p for p in plan.placements if p.role == NodeRole.ROLLOUT]
    drafts = [p for p in plan.placements if p.role == NodeRole.DRAFT]
    owns_connection = not ray.is_initialized()
    owned = []
    queues = []
    run_prefix = f"speco-{plan.run_id}-g{plan.generation}"
    local_device = execution.get("device", "cuda")
    if local_device not in ("cuda", "npu"):
        raise ValueError("external trainer device must be cuda or npu")
    StandaloneDrafterWorker = placed_drafter_class()

    def strategy(node_rank):
        return NodeAffinitySchedulingStrategy(node_id=live[plan.nodes[node_rank].address]["NodeID"], soft=False)

    def group_factory(consumer_config, *, ray_module, scheduled_commands, training_events):
        handles, names = [], []
        for placement in drafts:
            rank = placement.global_rank
            name = f"{run_prefix}-draft-{rank}"
            env = dict(execution.get("worker_env", {}), RANK=str(rank), LOCAL_RANK="0",
                       WORLD_SIZE=str(plan.draft_world_size), LOCAL_WORLD_SIZE="1",
                       MASTER_ADDR=plan.master_addr, MASTER_PORT=str(plan.master_port))
            # Let Ray mask its allocated card. Upstream Worker must see a
            # single logical device, not reinterpret its physical ID as a rank.
            for family in ('CUDA', 'ROCR', 'HIP', 'ASCEND_RT', 'HABANA', 'NEURON_RT', 'TPU', 'ONEAPI'):
                suffix = 'VISIBLE_MODULES' if family == 'HABANA' else 'VISIBLE_CORES' if family == 'NEURON_RT' else 'VISIBLE_CHIPS' if family == 'TPU' else 'DEVICE_SELECTOR' if family == 'ONEAPI' else 'VISIBLE_DEVICES'
                env[f'RAY_EXPERIMENTAL_NOSET_{family}_{suffix}'] = ''
            for variable in ('CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
                env.pop(variable, None)
            options = dict(name=name, num_cpus=execution.get("worker_cpus", 1),
                           scheduling_strategy=strategy(placement.node_rank), runtime_env={"env_vars": env})
            if local_device == "cuda":
                options["num_gpus"] = 1
            else:
                options["resources"] = {"NPU": 1}
            actor = ray.remote(StandaloneDrafterWorker).options(**options).remote(
                consumer_config, scheduled_commands=scheduled_commands, training_events=training_events,
                assigned_device_id=placement.device_id, device_name=local_device)
            handles.append(actor)
            names.append(name)
            owned.append(actor)
        descriptor = RayClassWithInitArgs(cls=ray.remote(StandaloneDrafterWorker))
        return RayWorkerGroup(resource_pool=None, ray_cls_with_init=descriptor,
                              worker_handles=handles, worker_names=names, device_name=local_device)

    class ExternalTrainer(StandaloneRayTrainer):
        def init_workers(self):
            queue_options = {"scheduling_strategy": strategy(next(p.node_rank for p in plan.placements if p.role == NodeRole.CONTROL))}
            self.commands_queue = Queue(actor_options=queue_options)
            queues.append(self.commands_queue)
            self.events_queue = Queue(actor_options=queue_options)
            queues.append(self.events_queue)
            # Create one group owner before attaching producer handles; producers
            # receive only its small event sink, never each other's payloads.
            self.producer = ray.remote(max_concurrency=32)(ExternalProducerGroup).options(
                name=run_prefix + "-producers", num_cpus=.1).remote([], self.events_queue)
            owned.append(self.producer)
            actor_handles = []
            max_samples = int(self.producer_config.speco.standalone_tq_producer.max_samples)
            if max_samples and max_samples < len(producers):
                raise ValueError("max_samples must allocate at least one sample per producer")
            for index, placement in enumerate(producers):
                cfg = deepcopy(self.producer_config)
                with open_dict(cfg.speco.standalone_tq_producer):
                    cfg.speco.standalone_tq_producer.vllm_endpoints = [endpoints[index]]
                    cfg.speco.standalone_tq_producer.input_shard_index = index
                    cfg.speco.standalone_tq_producer.input_shard_count = len(producers)
                    if max_samples:
                        cfg.speco.standalone_tq_producer.max_samples = max_samples // len(producers) + int(index < max_samples % len(producers))
                sink = ProducerEventSink(self.producer, str(index))
                actor = ray.remote(max_concurrency=4)(PlacedProducer).options(
                    name=f"{run_prefix}-producer-{index}", num_cpus=1,
                    scheduling_strategy=strategy(placement.node_rank)).remote(cfg, sink)
                owned.append(actor)
                actor_handles.append(actor)
            ray.get(self.producer.bind.remote(actor_handles))
            self.consumer_wg = group_factory(self.consumer_config, ray_module=ray,
                scheduled_commands=self.commands_queue, training_events=self.events_queue)

    try:
        namespace = external_tq_binding(config, plan)['ray']['namespace']
        if owns_connection:
            ray.init(address=address, namespace=namespace)
        elif ray.get_runtime_context().namespace != namespace:
            raise ValueError('existing Ray connection namespace differs from the launch plan')
        live = require_live_nodes(plan, ray.nodes())
        trainer = ExternalTrainer(commands, ray_module=ray, worker_group_factory=group_factory)
        with open_dict(trainer.producer_config.speco.standalone_tq_producer):
            trainer.producer_config.speco.standalone_tq_producer.vllm_endpoints = list(endpoints)
        return trainer.run()
    finally:
        # Only explicit handles created above are owned. Never stop the cluster
        # or kill external services by name or by process enumeration.
        primary_error, cleanup_error = sys.exc_info()[1], None
        callbacks = [lambda handle=actor: ray.kill(handle, no_restart=True) for actor in reversed(owned)]
        callbacks.extend(queue.shutdown for queue in queues)
        if owns_connection:
            callbacks.append(ray.shutdown)
        for callback in callbacks:
            try:
                callback()
            except Exception as error:
                cleanup_error = cleanup_error or error
                logger.exception('Failed to clean up an owned standalone resource')
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error
