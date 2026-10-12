# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from copy import deepcopy
import os
import re

def validate_worker_assignment(settings, *, role, rank, world_size):
    assignment = settings.get('colocated_assignment')
    required = {'run_id', 'generation', 'role', 'rank', 'node_ip', 'placement_group_id', 'bundle_index'}
    if type(assignment) is not dict or set(assignment) != required or assignment['run_id'] != settings.get('run_id') or (type(assignment['generation']) is not int) or (assignment['generation'] != settings.get('generation')) or (assignment['generation'] < 1) or (assignment['role'] != role) or (type(rank) is not int) or (type(world_size) is not int) or (not 32 <= world_size <= 128) or (world_size % 16) or (type(assignment['rank']) is not int) or (assignment['rank'] != rank) or (not 0 <= rank < world_size) or (type(assignment['bundle_index']) is not int) or (assignment['bundle_index'] != rank % 16) or any((type(assignment[key]) is not str or not assignment[key] for key in ('run_id', 'node_ip', 'placement_group_id'))):
        raise ValueError('shared worker requires its explicit complete-node role/rank assignment')
    return deepcopy(assignment)

def verify_observed_placement(assignment, *, incarnation, node_ip, placement_group_id, resource_ids, accelerator_ids, visible_devices, logical_device, device_count, platform_uuid):
    if type(incarnation) is not str or not incarnation or type(resource_ids) is not dict or (type(accelerator_ids) is not list) or (len(accelerator_ids) != 1) or (type(accelerator_ids[0]) is not str) or (not re.fullmatch('\\d+', accelerator_ids[0])) or (str(int(accelerator_ids[0])) != accelerator_ids[0]) or (not 0 <= int(accelerator_ids[0]) < 16) or (visible_devices != accelerator_ids[0]) or (type(logical_device) is not int) or (logical_device != 0) or (type(device_count) is not int) or (device_count != 1) or (node_ip != assignment['node_ip']) or (placement_group_id != assignment['placement_group_id']) or (platform_uuid != 'NPU-' + accelerator_ids[0]):
        raise ValueError('actual Ray/device visibility differs from the assigned shared physical die')
    indexed = f"NPU_group_{assignment['bundle_index']}_{placement_group_id}"
    rows = resource_ids.get(indexed)
    indexed_names = [name for name in resource_ids if re.fullmatch('NPU_group_\\d+_[0-9a-zA-Z]+', name)]
    if indexed_names != [indexed] or type(rows) not in (list, tuple) or len(rows) != 1 or (type(rows[0]) not in (list, tuple)) or (len(rows[0]) != 2) or (type(rows[0][1]) not in (float, int)) or (rows[0][1] != 0.5):
        raise ValueError('actual Ray indexed placement bundle does not own the expected half NPU')
    return dict(assignment, worker_incarnation=incarnation, device_uuid=platform_uuid)

def observe_worker_placement(worker, settings, role):
    import ray
    import torch
    from ray.util import get_current_placement_group
    from ray.util import get_node_ip_address
    from verl.workers.rollout.vllm_rollout.utils import get_device_uuid
    assignment = validate_worker_assignment(settings, role=role, rank=worker.rank, world_size=worker.world_size)
    ctx = ray.get_runtime_context()
    group = get_current_placement_group()
    if group is None:
        raise ValueError('shared worker has no actual current Ray placement group')
    torch.npu.get_device_properties(0)
    return verify_observed_placement(assignment, incarnation=worker.worker_incarnation if role == 'actor' else worker.incarnation, node_ip=get_node_ip_address(), placement_group_id=group.id.hex(), resource_ids=ctx.worker.core_worker.resource_ids(), accelerator_ids=ctx.get_accelerator_ids().get('NPU'), visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), logical_device=torch.npu.current_device(), device_count=torch.npu.device_count(), platform_uuid=get_device_uuid(0))
