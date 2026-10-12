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
