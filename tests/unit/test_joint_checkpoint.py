# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Joint metadata closure tests; no real actor or NPU acceptance is claimed.'
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pytest
from verl_speco.integration.multinode_recovery import MultiNodeJointCheckpointStore as MultiNodeJointCheckpointStore, joint_checkpoint_canonical as _canonical
from verl_speco.integration.multinode_online import PublishedOnlineStepJournal
from verl_speco.runtime.artifacts import publish_json, read_published_json
from verl_speco.trainer.distributed_checkpoint import CandidateCheckpoint, OptimizerTransaction, StepJournalRecord, StepState
from verl_speco.transport.control import LeaseRegistry
from verl_speco.transport.streaming import TrainingStreamCoordinator, TrainingTransaction

def setup_case(tmp_path):
    fp = dict(actor_model='a' * 64, draft_model='b' * 64, target_model='c' * 64, tokenizer='d' * 64, recipe='e' * 64, tq_schema=3, target_layers=[1, 12, 22, 32, 43], rl_roles=['actor'])
    geometry = dict(target_layer_ids=fp['target_layers'], block_size=8, sliding_window=2048)
    fp['draft_model'] = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    store = MultiNodeJointCheckpointStore(run_root=tmp_path, run_id='run', generation=1, fingerprints=fp, actor_incarnations={0: 'a0', 1: 'a1'}, draft_incarnations={'0': 'd0', '1': 'd1'}, rollout_incarnations={0: 'r0', 1: 'r1'})

    def candidate(role, root):
        files = {}
        for name in ('dcp/.metadata', 'dcp/__0_0.distcp', 'extra-rank-0000.pt', 'extra-rank-0001.pt'):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'protocol fixture only')
            files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        header = dict(schema_version=1, generation=1, world_size=2, model_fingerprint=fp[role + '_model'], topology_fingerprint=role + '-mesh', files=files)
        p = root / 'manifest.json'
        p.write_text(json.dumps(header))
        return dict(uri=str(root), checksum=hashlib.sha256(p.read_bytes()).hexdigest(), checkpoint_generation=1)
    actor = candidate('actor', tmp_path / 'actor-DCP')
    plan_sha = 'f' * 64
    step = 'online-' + plan_sha
    draft = candidate('draft', tmp_path / 'training' / 'dcp' / step)
    tx = OptimizerTransaction.create(optimizer_step_id=step, plan_id='plan', generation=1, stream_scope='run', transaction_ids=['transaction'], lease_ids=['lease'], checkpoint_generation_before=0)
    journal = PublishedOnlineStepJournal(tmp_path / 'training' / 'journal')
    journal.store(StepJournalRecord(1, StepState.COMMITTED, tx, 'draft-mesh', fp['draft_model'], CandidateCheckpoint(**draft), True, 1.0, 2.0))
    snapshot_dir = tmp_path / 'training' / 'snapshots' / step
    snapshot_dir.mkdir(parents=True)
    snapshot_header = dict(schema_version=1, kind='draft_trainable_parameters', model_config=geometry, namespace_prefix='training_model.', generation=1, draft_revision=5, source_step_id=step, target_model_fingerprint=fp['target_model'], tokenizer_fingerprint=fp['tokenizer'], trainable_parameter_sha256='1' * 64)
    publish_json(snapshot_dir / 'manifest.json', snapshot_header)
    snapshot = dict(uri=str(snapshot_dir), checksum=hashlib.sha256(_canonical(snapshot_header)).hexdigest(), draft_revision=5)
    leases = LeaseRegistry()
    from verl_speco.transport.control import LeaseRecord, LeaseState
    leases.restore(dict(schema_version=1, leases=[asdict(LeaseRecord('lease', 'acquire', 'key', 'request', 'trainer', LeaseState.COMMITTED, True, 9.0, 1, '0')) | {'state': 'committed'}], committed_steps={step: dict(transaction_set_hash=tx.transaction_set_hash, lease_ids=['lease'])}))
    stream = TrainingStreamCoordinator('run')
    stream.register(TrainingTransaction('transaction', ('lease',), ('key',), 'run', 1, 1, '0'))
    stream.commit(step, ['transaction'])
    receipt = dict(worker_id='0', worker_incarnation='d0', run_id='run', generation=1, trained=True, is_publish_leader=True, optimizer_step=1, source_global_step=1, data_version=1, target_version=0, plan_id='plan', online_plan_sha256=plan_sha, candidate=draft, snapshot=snapshot, trainable_parameter_sha256='1' * 64, stream_snapshot=asdict(stream.snapshot()))
    (tmp_path / 'training' / 'receipts').mkdir()
    publish_json(tmp_path / 'training' / 'receipts' / (step + '.json'), receipt)
    peer = dict(receipt, worker_id='1', worker_incarnation='d1', is_publish_leader=False, snapshot=None)
    head = tmp_path / 'head.safetensors'
    head.write_bytes(b'head fixture')
    head_binding = dict(uri=str(head), sha256=hashlib.sha256(head.read_bytes()).hexdigest(), run_id='run', generation=1, target_revision=1, tokenizer_fingerprint=fp['tokenizer'], actor_weight_manifest_sha256='2' * 64)
    args = dict(actor_update_receipt=dict(run_id='run', generation=1, rl_step=1, target_revision=0, optimizer_step=1, optimizer_committed=True, actual_VeRL_actor=True, worker_incarnations={0: 'a0', 1: 'a1'}), versions=dict(run_id='run', generation=1, rl_step=1, target_revision=1, data_version=1, draft_revision=5), actor_rows=[dict(rank=i, worker_incarnation=f'a{i}', run_id='run', generation=1, optimizer_step=1, saved=True, candidate=actor, state_sha256=str(i + 3) * 64, complete_local_model_optimizer_scheduler_rng=True) for i in range(2)], draft_rows=[receipt, peer], draft_state_rows=[dict(worker_id=str(i), worker_incarnation=f'd{i}', run_id='run', generation=1, optimizer_step=1, state_sha256='5' * 64, complete_local_model_optimizer_scheduler_rng=True) for i in range(2)], control_snapshot=dict(schema_version=1, run_id='run', generation=1, leases=leases.snapshot()), target_acks=[dict(rank=i, worker_incarnation=f'r{i}', applied=True, run_id='run', generation=1, rl_step=1, target_revision=1, kv_cache_cleared=True, draft_target_head_synced=True, actor_weight_manifest_sha256='2' * 64) for i in range(2)], draft_acks=[dict(rank=i, worker_incarnation=f'r{i}', applied=True, run_id='run', generation=1, draft_revision=5, snapshot_checksum=snapshot['checksum']) for i in range(2)], target_head=head_binding, snapshot=snapshot, session_idle=True, pending_commits=0)
    return (store, args)

def test_joint_complete_marker_reopens_without_rename_and_does_not_mutate_inputs(tmp_path, monkeypatch):
    import errno
    import os
    store, args = setup_case(tmp_path)
    original = deepcopy(args)

    def forbidden_rename(*args, **kwargs):
        raise PermissionError(errno.EPERM, 'shared mount forbids rename')
    monkeypatch.setattr(os, 'replace', forbidden_rename)
    payload = store.build(**args)
    binding = store.commit(payload)
    assert store.load(binding) == payload
    assert store.commit(payload) == binding
    assert args == original
    assert payload['draft_journal']['state'] == 'STEP_COMMITTED'

@pytest.mark.parametrize('fault', ['missing_actor', 'actor_incarnation', 'actor_step', 'actor_boolean_rank', 'actor_candidate', 'missing_draft', 'draft_incarnation', 'draft_target', 'draft_plan', 'draft_audit', 'active_session', 'pending_commit', 'control_generation', 'missing_target_ack', 'target_cache', 'target_head_sha', 'draft_revision', 'draft_snapshot'])
def test_joint_rejects_incomplete_or_mixed_role_state_before_complete_marker(tmp_path, fault):
    store, args = setup_case(tmp_path)
    if fault == 'missing_actor':
        args['actor_rows'].pop()
    elif fault == 'actor_incarnation':
        args['actor_rows'][1]['worker_incarnation'] = 'new'
    elif fault == 'actor_step':
        args['actor_rows'][1]['optimizer_step'] = 2
    elif fault == 'actor_boolean_rank':
        args['actor_rows'][1]['rank'] = True
    elif fault == 'actor_candidate':
        args['actor_rows'][1]['candidate'] = dict(args['actor_rows'][1]['candidate'], checksum='6' * 64)
    elif fault == 'missing_draft':
        args['draft_rows'].pop()
    elif fault == 'draft_incarnation':
        args['draft_rows'][1]['worker_incarnation'] = 'new'
    elif fault == 'draft_target':
        args['draft_rows'][1]['target_version'] = 1
    elif fault == 'draft_plan':
        args['draft_rows'][1]['plan_id'] = 'other'
    elif fault == 'draft_audit':
        args['draft_state_rows'][0]['complete_local_model_optimizer_scheduler_rng'] = False
    elif fault == 'active_session':
        args['session_idle'] = False
    elif fault == 'pending_commit':
        args['pending_commits'] = 1
    elif fault == 'control_generation':
        args['control_snapshot']['generation'] = 2
    elif fault == 'missing_target_ack':
        args['target_acks'].pop()
    elif fault == 'target_cache':
        args['target_acks'][0]['kv_cache_cleared'] = False
    elif fault == 'target_head_sha':
        args['target_head']['actor_weight_manifest_sha256'] = '7' * 64
    elif fault == 'draft_revision':
        args['draft_acks'][0]['draft_revision'] = 4
    elif fault == 'draft_snapshot':
        args['draft_acks'][0]['snapshot_checksum'] = '8' * 64
    with pytest.raises(ValueError):
        store.build(**args)
    assert not (tmp_path / 'joint-checkpoints').exists()

def test_draft_journal_changed_after_build_cannot_be_published(tmp_path):
    store, args = setup_case(tmp_path)
    payload = store.build(**args)
    step = 'online-' + args['draft_rows'][0]['online_plan_sha256']
    journal = PublishedOnlineStepJournal(tmp_path / 'training' / 'journal')
    from dataclasses import replace
    journal.store(replace(journal.load(step), state=StepState.ABORTED, reason='uncertain'))
    with pytest.raises(ValueError, match='journal'):
        store.commit(payload)
    assert not (tmp_path / 'joint-checkpoints').exists()

def test_missing_saved_payload_rejects_before_complete_marker(tmp_path):
    store, args = setup_case(tmp_path)
    (Path(args['actor_rows'][0]['candidate']['uri']) / 'extra-rank-0001.pt').unlink()
    with pytest.raises(ValueError, match='missing payload'):
        store.build(**args)

def test_durable_receipt_mismatch_is_not_repaired_from_rpc_rows(tmp_path):
    store, args = setup_case(tmp_path)
    for row in args['draft_rows']:
        row['plan_id'] = 'forged'
    with pytest.raises(ValueError, match='durable receipt'):
        store.build(**args)

def test_incomplete_marker_directory_is_fenced_without_overwrite(tmp_path):
    store, args = setup_case(tmp_path)
    payload = store.build(**args)
    directory = tmp_path / 'joint-checkpoints' / 'generation-1-step-1'
    directory.mkdir(parents=True)
    (directory / 'unfinished').write_text('preserve')
    with pytest.raises(RuntimeError, match='incomplete'):
        store.commit(payload)
    assert (directory / 'unfinished').read_text() == 'preserve'
    assert not list(directory.glob('complete.json*'))

def test_complete_marker_binding_and_recipe_are_verified_on_load(tmp_path):
    store, args = setup_case(tmp_path)
    binding = store.commit(store.build(**args))
    with pytest.raises(ValueError, match='checksum'):
        store.load(dict(binding, checksum='9' * 64))
    store.fingerprints['recipe'] = '0' * 64
    with pytest.raises(ValueError, match='recipe'):
        store.load(binding)

@pytest.mark.parametrize('location', ['draft', 'draft_audit', 'control', 'target', 'head', 'draft_ack'])
def test_boolean_generation_is_not_an_integer_epoch(tmp_path, location):
    store, args = setup_case(tmp_path)
    if location == 'draft':
        args['draft_rows'][1]['generation'] = True
    elif location == 'draft_audit':
        args['draft_state_rows'][1]['generation'] = True
    elif location == 'control':
        args['control_snapshot']['generation'] = True
    elif location == 'target':
        args['target_acks'][0]['generation'] = True
    elif location == 'head':
        args['target_head']['generation'] = True
    elif location == 'draft_ack':
        args['draft_acks'][0]['generation'] = True
    with pytest.raises(ValueError):
        store.build(**args)

def test_complete_marker_dependency_corruption_is_rejected_on_reopen(tmp_path):
    store, args = setup_case(tmp_path)
    binding = store.commit(store.build(**args))
    path = Path(args['actor_rows'][0]['candidate']['uri']) / 'manifest.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='manifest checksum'):
        store.load(binding)

def test_checkpoint_generation_header_must_be_typed_even_with_valid_checksum(tmp_path):
    store, args = setup_case(tmp_path)
    path = Path(args['actor_rows'][0]['candidate']['uri']) / 'manifest.json'
    header = json.loads(path.read_text())
    header['generation'] = True
    path.write_text(json.dumps(header))
    for row in args['actor_rows']:
        row['candidate']['checksum'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='topology binding'):
        store.build(**args)

def test_unrelated_active_lease_blocks_joint_marker(tmp_path):
    store, args = setup_case(tmp_path)
    extra = dict(args['control_snapshot']['leases']['leases'][0], lease_id='extra', key='extra', acquire_id='extra', state='leased', payload_released=False)
    args['control_snapshot']['leases']['leases'].append(extra)
    with pytest.raises(ValueError, match='active lease'):
        store.build(**args)

def test_required_critic_or_reference_role_cannot_be_silently_omitted(tmp_path):
    store, _ = setup_case(tmp_path)
    with pytest.raises(ValueError, match='role contract'):
        MultiNodeJointCheckpointStore(run_root=tmp_path, run_id='run', generation=1, fingerprints=dict(store.fingerprints, rl_roles=['actor', 'ref']), actor_incarnations=store.actors, draft_incarnations=store.drafts, rollout_incarnations=store.rollout)

def test_actor_optimizer_count_is_independent_of_rl_step(tmp_path):
    store, args = setup_case(tmp_path)
    args['actor_update_receipt']['optimizer_step'] = 3
    for row in args['actor_rows']:
        row['optimizer_step'] = 3
    payload = store.build(**args)
    binding = store.commit(payload)
    assert store.load(binding)['versions']['rl_step'] == 1
    assert store.load(binding)['actor_update_receipt']['optimizer_step'] == 3

def test_lagging_complete_draft_checkpoint_keeps_its_own_data_counters(tmp_path):
    store, args = setup_case(tmp_path)
    args['versions'].update(rl_step=2, data_version=2)
    args['actor_update_receipt'].update(rl_step=2, optimizer_step=2)
    for row in args['actor_rows']:
        row['optimizer_step'] = 2
    for row in args['target_acks']:
        row['rl_step'] = 2
    payload = store.build(**args)
    assert payload['draft_rows'][0]['source_global_step'] == 1
    assert payload['versions']['rl_step'] == 2
