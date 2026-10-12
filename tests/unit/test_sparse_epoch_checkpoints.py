# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Sparse writes must preserve progress and reject cold recovery of live state.'
from dataclasses import asdict
from types import SimpleNamespace
import hashlib
import json
import pytest
from verl_speco.runtime.checkpoint_policy import checkpoint_due, checkpoint_interval, is_live_candidate, write_live_update, read_live_update
from verl_speco.trainer.distributed_checkpoint import RecoveryAction

def plan():
    return SimpleNamespace(samples_per_epoch=200252, samples_per_step=960, total_samples=600756, total_steps=626)

def test_full_three_epoch_schedule_has_exactly_nine_writes():
    writes = [i + 1 for i in range(626) if checkpoint_due(i, plan=plan(), interval=100, stage_end=626)]
    assert writes == [100, 200, 209, 300, 400, 418, 500, 600, 626]
    assert sum((not checkpoint_due(i, plan=plan(), interval=100, stage_end=626) for i in range(626))) == 617

def test_stage_final_and_legacy_every_step():
    assert [i for i in range(3) if checkpoint_due(i, plan=plan(), interval=100, stage_end=3)] == [2]
    assert all((checkpoint_due(i, plan=plan(), interval=1, stage_end=626) for i in range(626)))

@pytest.mark.parametrize('value', [True, 0, -1, 10001, '100', 100.0])
def test_invalid_checkpoint_interval_is_rejected(value):
    with pytest.raises(ValueError):
        checkpoint_interval({'epoch_checkpoint_interval': value})

def live(tmp_path):
    return write_live_update(tmp_path / 'live.json', run_id='scope', generation=1, step_id='step1', step_index=0, cursor=1, weights_sha256='a' * 64, owner_token='b' * 64, topology_fingerprint='topo1', model_fingerprint='model1', logical_checkpoint_fingerprint='c' * 64)

def test_live_state_cannot_commit_or_cold_resume_without_current_owner(tmp_path):
    from _multinode_fixture_optimizer_transaction_control import _setup
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    candidate = live(tmp_path)
    assert is_live_candidate(candidate)
    executor.prepare(tx)
    executor.begin_optimizer('step1')
    executor.attach_candidate('step1', candidate)
    with pytest.raises(RuntimeError, match='current owner'):
        executor.commit('step1')
    with pytest.raises(ValueError, match='owner'):
        executor.commit('step1', live_owner_token='d' * 64)
    assert executor.commit('step1', live_owner_token='b' * 64).already_committed is False
    assert executor.recover('step1').action is RecoveryAction.ROLLBACK_GENERATION
    assert executor.recover_consistent('step1').action is RecoveryAction.ROLLBACK_GENERATION
    assert executor.recover_consistent('step1', live_owner_token='b' * 64).action is RecoveryAction.RESUME_COMMITTED
    assert executor.recover_consistent('step1', live_owner_token='d' * 64).action is RecoveryAction.ROLLBACK_GENERATION
    assert stream.snapshot().committed_cursor == 1

def test_live_descriptor_tampering_and_full_restore_are_rejected(tmp_path):
    from verl_speco.trainer.distributed_checkpoint import VeOmniDCPBridge
    candidate = live(tmp_path)
    bridge = VeOmniDCPBridge.__new__(VeOmniDCPBridge)
    with pytest.raises(ValueError, match='latest full checkpoint'):
        bridge._verified_candidate(candidate)
    path = tmp_path / 'live.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum'):
        read_live_update(candidate, owner_token='b' * 64)

def test_unsaved_step_excludes_no_time_from_complete_throughput():
    from verl_speco.runtime.services import checkpoint_excluded_steady_stage_timing
    rows = [dict(state='STEP_COMMITTED', step_index=i, original_tokens=100, receipt_sha256=str(i) * 64, completed_at=float(10 * (i + 1)), checkpoint_saved=False, checkpoint_timing=None) for i in range(3)]
    value = checkpoint_excluded_steady_stage_timing(rows, node_hostname='n', boot_id='b')
    assert value['checkpoint_seconds'] == 0 and value['checkpoint_excluded_seconds'] == 20
    assert value['original_tokens'] == 200 and value['original_token_s'] == 10
    rows[1]['checkpoint_timing'] = {}
    with pytest.raises(ValueError, match='unsaved'):
        checkpoint_excluded_steady_stage_timing(rows, node_hostname='n', boot_id='b')

def test_sparse_retention_requires_every_live_progress_record(tmp_path):
    from _multinode_fixture_epoch_checkpoint_retention import fixture
    from verl_speco.runtime.recovery import apply_retirement
    from verl_speco.runtime.artifacts import publish_json
    root, work, records, original, journal = fixture(tmp_path)
    for old, new in zip(records, [1, 3, 5]):
        previous = __import__('pathlib').Path(old['control_snapshot']['path'])
        path = work / 'controls' / ('step-' + f'{new:04d}' + '-control-' + '1' * 32 + '.json')
        path.write_bytes(previous.read_bytes())
        old['control_snapshot']['path'] = str(path)
        old.update(step_index=new, start=old['start'] + 480 * (new - old['step_index']), end=480 * (new + 1), checkpoint_saved=True)
        publish_json(work / f'committed-step-{new:04d}.json', old)
    for r in records:
        r['start'] = 480 * r['step_index']
        publish_json(work / f"committed-step-{r['step_index']:04d}.json", r)
    for index in [2, 4]:
        candidate = write_live_update(root / f'live-{index}.json', run_id=records[0]['run_id'], generation=1, step_id=f'live-{index}', step_index=index, cursor=480 * (index + 1), weights_sha256='a' * 64, owner_token='b' * 64, topology_fingerprint='topo', model_fingerprint='model', logical_checkpoint_fingerprint='c' * 64)
        publish_json(work / f'committed-step-{index:04d}.json', dict(schema_version=1, state='STEP_COMMITTED', run_id=records[0]['run_id'], generation=1, step_index=index, start=480 * index, end=480 * (index + 1), checkpoint_saved=False, receipt_sha256='a' * 64, candidate=asdict(candidate)))
    result = apply_retirement(records, outputs_root=root, work_dir=work, allow_sparse=True)
    assert result['plan']['protected_steps'] == [3, 5]
    assert original.exists() and journal.exists()
from _multinode_fixture_generation_parent import parent

def test_cold_recovery_replays_only_proven_live_progress_after_full_checkpoint(parent, tmp_path):
    from verl_speco.runtime.recovery import require_stopped_committed_parent
    receipt = parent['receipt']
    receipt.update(checkpoint_interval=100, checkpoint_saved=True, live_owner_token='b' * 64)
    candidate = write_live_update(tmp_path / 'later-live.json', run_id='run-control', generation=1, step_id='later', step_index=1, cursor=4, weights_sha256='d' * 64, owner_token='b' * 64, topology_fingerprint='old', model_fingerprint='model', logical_checkpoint_fingerprint='a' * 64)
    later = dict(state='STEP_COMMITTED', candidate_checkpoint=asdict(candidate), transaction=dict(generation=1, stream_scope='run-control', optimizer_step_id='later', transaction_ids=['tx2', 'tx3'], lease_ids=['lease2', 'lease3']))
    parent['probes'][0]['journals'] = [parent['probes'][0]['journals'][0], later]
    parent['status']['leases'].update(committed_steps=2, leases_committed=4)
    require_stopped_committed_parent(receipt=receipt, snapshot=parent['snapshot'], status=parent['status'], probes=parent['probes'], samples=2)
    later['candidate_checkpoint'] = parent['receipt']['candidate']
    with pytest.raises(RuntimeError, match='full checkpoint'):
        require_stopped_committed_parent(receipt=receipt, snapshot=parent['snapshot'], status=parent['status'], probes=parent['probes'], samples=2)
