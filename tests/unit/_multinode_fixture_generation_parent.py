# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'A fresh generation must restore the latest durable prefix, never an uncertain candidate.'
from copy import deepcopy
from dataclasses import asdict
import hashlib, json, time
from pathlib import Path
import pytest
from _multinode_fixture_ray_ready_control import _record
from verl_speco.transport.control import RayReadyControl
from verl_speco.transport.control import AcquireRequest
from verl_speco.transport.streaming import transaction_set_hash
from verl_speco.transport.streaming import TrainingStreamCoordinator, TrainingTransaction
from verl_speco.runtime.recovery import load_generation_parent, require_stopped_committed_parent

@pytest.fixture
def parent(tmp_path):
    owner = RayReadyControl(run_id='run-control', generation=1, partition='samples', capacity_bytes=256)
    records = [_record('producer', i) for i in range(2)]
    for row in records:
        token = owner.reserve(row, nbytes=128).token
        owner.publish(row, token=token)
    _, leases = owner.acquire(AcquireRequest('acquire', 'trainer', 1, 2, 2, 'target-1', time.time() + 60), lease_timeout_s=60)
    ids = tuple((x.lease_id for x in leases))
    owner.mark_materialized(ids)
    for lease in leases:
        owner.release(lease.key, lease_ids=(lease.lease_id,))
    txids = ['tx0', 'tx1']
    owner.commit(optimizer_step_id='parent', transaction_set_hash=transaction_set_hash(txids), lease_ids=ids)
    stream = TrainingStreamCoordinator('run-control')
    for i, lease in enumerate(leases):
        stream.register(TrainingTransaction(txids[i], (lease.lease_id,), (lease.key,), 'run-control', i + 1, 1, 'target-1'))
    stream.commit('parent', txids)
    control = tmp_path / 'control.json'
    control_sha = owner.save_quiescent(str(control))
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'weight.bin').write_bytes(b'committed-weight')
    manifest = dict(schema_version=1, generation=1, world_size=16, topology_fingerprint='old', model_fingerprint='model', logical_checkpoint_fingerprint='a' * 64, files={'weight.bin': hashlib.sha256(b'committed-weight').hexdigest()})
    m = candidate / 'manifest.json'
    m.write_text(json.dumps(manifest))
    ref = dict(uri=str(candidate), checksum=hashlib.sha256(m.read_bytes()).hexdigest(), checkpoint_generation=1)
    receipt = dict(schema_version=1, run_id='run-control', generation=1, sample_index=0, total_steps=3, packed_sample_keys=[x.key for x in leases], draft_world_size=16, step_id='parent', candidate=ref, draft_weights_sha256='c' * 64, stream_snapshot=asdict(stream.snapshot()), control_snapshot=dict(path=str(control), sha256=control_sha), journal_dir='/tmp/journal')
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(receipt))
    committed = dict(state='STEP_COMMITTED', transaction=dict(generation=1, stream_scope='run-control', optimizer_step_id='parent', transaction_ids=txids), candidate_checkpoint=ref)
    uncertain = dict(state='STEP_PREPARED', optimizer_started=True, transaction=dict(generation=1, stream_scope='run-control', optimizer_step_id='uncertain', transaction_ids=['tx2', 'tx3']))
    return dict(path=path, receipt=json.loads(json.dumps(receipt)), manifest=manifest, snapshot=json.loads(json.dumps(owner.snapshot_quiescent())), status=owner.status(), probes=[dict(journals=[committed, uncertain], active_trainer_pids=[])])

def load(parent, **changes):
    kwargs = dict(run_id='run-control', generation=2, sample_index=2, samples=2, total_steps=3, logical_checkpoint_fingerprint='a' * 64)
    kwargs.update(changes)
    return load_generation_parent(parent['path'], **kwargs)
