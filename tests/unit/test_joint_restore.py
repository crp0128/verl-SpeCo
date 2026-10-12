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

def test_actual_native_control_validator_reopens_parent_without_rewriting_lease_generation(tmp_path):
    parent, config, _, _ = case(tmp_path)
    opened, current = open_joint_parent(config, config['joint_restore_binding'])
    assert opened == parent and current['generation'] == 2
    assert not current['leases']['leases']
    assert current['committed_parents'][0]['leases']['leases'][0]['generation'] == 1

@pytest.mark.parametrize('fault', ['same-generation', 'boolean-generation', 'wrong-recipe', 'wrong-partition', 'wrong-capacity', 'wrong-binding'])
def test_parent_reopen_rejects_current_owner_or_recipe_changes(tmp_path, fault):
    _, config, _, _ = case(tmp_path)
    binding = deepcopy(config['joint_restore_binding'])
    if fault == 'same-generation':
        config['generation'] = 1
    if fault == 'boolean-generation':
        config['generation'] = True
    if fault == 'wrong-recipe':
        config['joint_fingerprints']['recipe'] = '0' * 64
    if fault == 'wrong-partition':
        config['partition'] = 'other'
    if fault == 'wrong-capacity':
        config['joint_control_capacity_bytes'] += 1
    if fault == 'wrong-binding':
        binding['checksum'] = '0' * 64
    with pytest.raises(ValueError):
        open_joint_parent(config, binding)

def test_recipe_preserves_generation_handoff_but_fences_optimizer_or_mesh_changes():
    model = torch.nn.Linear(3, 2)
    config = dict(run_id='run', generation=1, model_config={'block_size': 8}, target_model_fingerprint='a' * 64, tokenizer_fingerprint='b' * 64, total_steps=100)
    first = joint_draft_recipe(config, world_size=2, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01))
    changed = joint_draft_recipe(dict(config, generation=2), world_size=2, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01))
    assert first['logical_checkpoint_fingerprint'] == changed['logical_checkpoint_fingerprint']
    assert first['topology_fingerprint'] != changed['topology_fingerprint']
    for recipe in (joint_draft_recipe(config, world_size=1, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01)), joint_draft_recipe(config, world_size=2, optimizer=torch.optim.AdamW(model.parameters(), lr=0.02)), joint_draft_recipe(dict(config, total_steps=101), world_size=2, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01))):
        assert recipe['logical_checkpoint_fingerprint'] != first['logical_checkpoint_fingerprint']

@pytest.mark.parametrize('fault', [None, 'checksum', 'shape', 'dtype', 'nonfinite', 'extra-weight', 'outside'])
def test_entire_immutable_target_head_payload_is_checked(tmp_path, fault):
    tensor = torch.arange(64).reshape(8, 8).bfloat16()
    if fault == 'dtype':
        tensor = tensor.float()
    if fault == 'nonfinite':
        tensor[-1, -1] = float('nan')
    weights = {'lm_head.weight': tensor}
    if fault == 'extra-weight':
        weights['extra'] = torch.ones(1)
    path = tmp_path / 'head.safetensors'
    save_file(weights, str(path))
    binding = dict(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if fault == 'checksum':
        binding['sha256'] = '0' * 64
    if fault == 'outside':
        binding['uri'] = str(tmp_path.parent / 'outside.safetensors')
    shape = (4, 16) if fault == 'shape' else (8, 8)
    if fault is None:
        assert torch.equal(read_joint_target_head(binding, run_root=tmp_path, shape=shape), tensor)
    else:
        with pytest.raises(ValueError):
            read_joint_target_head(binding, run_root=tmp_path, shape=shape)

def test_release_accepts_fresh_wire_identity_only_with_full_restored_state_and_all_ACKs(tmp_path):
    parent, config, owners, proof = case(tmp_path)
    validate_joint_release(parent, proof, owners, run_id='run', generation=2, binding_checksum=config['joint_restore_binding']['checksum'])
    assert proof['source_wire']['actor_weight_manifest_sha256'] != parent['target_acks'][0]['actor_weight_manifest_sha256']

@pytest.mark.parametrize('fault', ['actor-tail-state', 'draft-Adam-state', 'actor-step', 'target-sha', 'head-bytes', 'target-cache', 'draft-revision', 'draft-snapshot', 'missing-actor', 'missing-native-draft', 'old-target-owner', 'old-draft-owner', 'old-generation', 'boolean-rank', 'boolean-generation', 'prefix-cache', 'missing-wire-tensor', 'wrong-binding'])
def test_release_rejects_partial_role_state_or_serving_recovery(tmp_path, fault):
    parent, config, owners, proof = case(tmp_path)
    if fault == 'actor-tail-state':
        proof['actor_rows'][0]['state_sha256'] = '0' * 64
    if fault == 'draft-Adam-state':
        proof['draft_rows'][0]['state_sha256'] = '0' * 64
    if fault == 'actor-step':
        proof['actor_rows'][0]['optimizer_step'] += 1
    if fault == 'target-sha':
        proof['target_acks'][0]['actor_weight_manifest_sha256'] = '0' * 64
    if fault == 'head-bytes':
        proof['source_wire']['head_sha256'] = '0' * 64
    if fault == 'target-cache':
        proof['target_acks'][0]['kv_cache_cleared'] = False
    if fault == 'draft-revision':
        proof['draft_acks'][0]['draft_revision'] += 1
    if fault == 'draft-snapshot':
        proof['draft_acks'][0]['snapshot_checksum'] = '0' * 64
    if fault == 'missing-actor':
        proof['actor_rows'].pop()
    if fault == 'missing-native-draft':
        proof['draft_acks'].pop()
    if fault == 'old-target-owner':
        proof['target_acks'][0]['worker_incarnation'] = 'r0'
    if fault == 'old-draft-owner':
        proof['draft_rows'][0]['worker_incarnation'] = 'd0'
    if fault == 'old-generation':
        proof['generation'] = 1
    if fault == 'boolean-rank':
        proof['actor_rows'][0]['rank'] = False
    if fault == 'boolean-generation':
        proof['draft_rows'][0]['generation'] = True
    if fault == 'prefix-cache':
        proof['native_prefix_cache_reset'] = False
    if fault == 'missing-wire-tensor':
        proof['source_wire']['wire_tensor_count'] -= 1
    if fault == 'wrong-binding':
        proof['binding_checksum'] = '0' * 64
    with pytest.raises(ValueError):
        validate_joint_release(parent, proof, owners, run_id='run', generation=2, binding_checksum=config['joint_restore_binding']['checksum'])

@pytest.mark.parametrize('fault', ['reuse-old-incarnation', 'duplicate-incarnation', 'missing-role', 'boolean-key'])
def test_fresh_membership_cannot_adopt_old_or_duplicate_role_owners(tmp_path, fault):
    parent, _, owners, _ = case(tmp_path)
    if fault == 'reuse-old-incarnation':
        owners['actor'][0] = 'a0'
    if fault == 'duplicate-incarnation':
        owners['draft']['1'] = owners['draft']['0']
    if fault == 'missing-role':
        del owners['actor']
    if fault == 'boolean-key':
        owners['actor'] = {False: 'fresh-a0', 1: 'fresh-a1'}
    with pytest.raises(ValueError):
        validate_joint_owners(parent, owners)
