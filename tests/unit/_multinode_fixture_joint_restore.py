# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Joint recovery recipe, head payload and complete fresh role ACK gates.'
from copy import deepcopy
import hashlib
import pytest
import torch
from safetensors.torch import save_file
from _multinode_fixture_joint_checkpoint import setup_case
from verl_speco.integration.multinode_recovery import joint_draft_recipe, open_joint_parent, read_joint_target_head, validate_joint_owners, validate_joint_release

def case(tmp_path):
    store, args = setup_case(tmp_path)
    args['control_snapshot'].update(partition='samples', capacity_bytes=4096, producer_incarnations={}, delivered=[], acquired=[])
    parent = store.build(**args)
    binding = store.commit(parent)
    config = dict(run_id='run', generation=2, joint_run_root=str(tmp_path), joint_restore_binding=binding, joint_fingerprints=store.fingerprints, partition='samples', joint_control_capacity_bytes=4096)
    owners = dict(actor={0: 'fresh-a0', 1: 'fresh-a1'}, draft={'0': 'fresh-d0', '1': 'fresh-d1'}, rollout={0: 'fresh-r0', 1: 'fresh-r1'})
    proof = dict(run_id='run', generation=2, binding_checksum=binding['checksum'], native_prefix_cache_reset=True)
    proof['actor_rows'] = [dict(r, generation=2, worker_incarnation=owners['actor'][r['rank']], restored=True) for r in parent['actor_rows']]
    proof['draft_rows'] = [dict(r, generation=2, worker_incarnation=owners['draft'][r['worker_id']], restored=True) for r in parent['draft_state_rows']]
    proof['source_wire'] = dict(actor_weight_manifest_sha256='9' * 64, wire_tensor_count=35869, wire_nbytes=123456, head_sha256=parent['target_head']['sha256'])
    proof['target_acks'] = [dict(r, generation=2, worker_incarnation=owners['rollout'][r['rank']], actor_weight_manifest_sha256='9' * 64) for r in parent['target_acks']]
    proof['draft_acks'] = [dict(r, generation=2, worker_incarnation=owners['rollout'][r['rank']]) for r in parent['draft_acks']]
    return (parent, config, owners, proof)
