# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Optimizer journal and participant state must agree on retry boundaries.'
import hashlib
import json
import pytest
from verl_speco.trainer.distributed_checkpoint import CandidateCheckpoint, CommitExecutor, JsonStepJournal, OptimizerTransaction, RecoveryAction, StepState
from verl_speco.transport.control import LeaseRegistry, LeaseState
from verl_speco.transport.control import AcquireResult
from verl_speco.transport.streaming import TrainingStreamCoordinator, TrainingTransaction
from _multinode_fixture_stream_control_plane import _record

def _setup(tmp_path):
    record = _record('producer', 1)
    leases = LeaseRegistry()
    lease = leases.register(AcquireResult('ready', 'acq1', (record,)), consumer_id='consumer', lease_timeout_s=60)[0]
    leases.mark_materialized((lease.lease_id,))
    leases.mark_payload_released((lease.lease_id,))
    stream = TrainingStreamCoordinator('scope')
    stream.register(TrainingTransaction('tx1', (lease.lease_id,), (record.key,), 'scope', 1, 1, 't1'))
    journal = JsonStepJournal(tmp_path / 'journal')
    executor = CommitExecutor(lease_registry=leases, stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    transaction = OptimizerTransaction.create(optimizer_step_id='step1', plan_id='plan1', generation=1, stream_scope='scope', transaction_ids=('tx1',), lease_ids=(lease.lease_id,), checkpoint_generation_before=0)
    return (executor, journal, leases, stream, lease, transaction)

def test_commit_requires_candidate_and_is_idempotent(tmp_path) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    assert executor.prepare(tx).state is StepState.PREPARED
    with pytest.raises(RuntimeError, match='candidate checkpoint'):
        executor.commit('step1')
    executor.begin_optimizer('step1')
    executor.attach_candidate('step1', CandidateCheckpoint('dcp://candidate', 'checksum', 1))
    with pytest.raises(RuntimeError, match='pending optimizer'):
        executor.drain()
    assert executor.commit('step1').already_committed is False
    assert leases.get(lease.lease_id).state is LeaseState.COMMITTED
    assert stream.snapshot().committed_cursor == 1
    assert journal.load('step1').state is StepState.COMMITTED
    assert executor.recover('step1').action is RecoveryAction.RESUME_COMMITTED
    assert executor.prepare(tx).state is StepState.COMMITTED
    executor.drain()
    with pytest.raises(ValueError, match='different transaction'):
        executor.prepare(OptimizerTransaction.create(optimizer_step_id='step1', plan_id='plan1', generation=1, stream_scope='scope', transaction_ids=('tx2',), lease_ids=(lease.lease_id,), checkpoint_generation_before=0))

def test_preflight_rejects_unknown_stream_transaction_before_lease_commit(tmp_path) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    executor.prepare(tx)
    executor.begin_optimizer('step1')
    executor.attach_candidate('step1', CandidateCheckpoint('dcp://candidate', 'checksum', 1))
    stream.restore(stream.snapshot())
    with pytest.raises(ValueError, match='unknown transaction IDs'):
        executor.commit('step1')
    assert leases.get(lease.lease_id).state is LeaseState.MATERIALIZED
    assert journal.load('step1').state is StepState.PREPARED
    assert executor.recover('step1').action is RecoveryAction.ROLLBACK_GENERATION

def test_restart_after_optimizer_boundary_does_not_retry_step(tmp_path) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    executor.prepare(tx)
    executor.begin_optimizer('step1')
    with pytest.raises(RuntimeError, match='optimizer was already marked started'):
        executor.begin_optimizer('step1')
    restarted = CommitExecutor(lease_registry=leases, stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    with pytest.raises(RuntimeError, match='earlier executor'):
        restarted.prepare(tx)
    assert restarted.recover('step1').action is RecoveryAction.ROLLBACK_GENERATION
    changed = CommitExecutor(lease_registry=leases, stream=stream, journal=journal, topology_fingerprint='different', model_fingerprint='model1')
    assert changed.recover('step1').action is RecoveryAction.FAIL_CLOSED

def test_restart_after_prepare_before_optimizer_rolls_back_generation(tmp_path) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    executor.prepare(tx)
    assert executor.prepare(tx).state is StepState.PREPARED
    restarted = CommitExecutor(lease_registry=leases, stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    with pytest.raises(RuntimeError, match='earlier executor'):
        restarted.prepare(tx)
    assert restarted.recover('step1').action is RecoveryAction.ROLLBACK_GENERATION
    assert journal.load('step1').optimizer_started is False

def test_stream_failure_after_lease_commit_never_retries_optimizer(tmp_path, monkeypatch) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    executor.prepare(tx)
    executor.begin_optimizer('step1')
    executor.attach_candidate('step1', CandidateCheckpoint('dcp://candidate', 'checksum', 1))

    def fail_stream_commit(*args, **kwargs):
        raise TimeoutError('injected stream commit timeout')
    monkeypatch.setattr(stream, 'commit', fail_stream_commit)
    with pytest.raises(TimeoutError, match='injected'):
        executor.commit('step1')
    assert leases.get(lease.lease_id).state is LeaseState.COMMITTED
    assert journal.load('step1').state is StepState.PREPARED
    with pytest.raises(RuntimeError, match='commit was already attempted'):
        executor.commit('step1')
    restarted = CommitExecutor(lease_registry=leases, stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    assert restarted.recover('step1').action is RecoveryAction.ROLLBACK_GENERATION
    with pytest.raises(RuntimeError, match='earlier executor'):
        restarted.prepare(tx)

def test_lease_commit_ack_timeout_cannot_be_retried(tmp_path, monkeypatch) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    executor.prepare(tx)
    executor.begin_optimizer('step1')
    executor.attach_candidate('step1', CandidateCheckpoint('dcp://candidate', 'checksum', 1))
    original_commit = leases.commit

    def commit_then_timeout(*args, **kwargs):
        original_commit(*args, **kwargs)
        raise TimeoutError('injected lease commit acknowledgement timeout')
    monkeypatch.setattr(leases, 'commit', commit_then_timeout)
    with pytest.raises(TimeoutError, match='acknowledgement timeout'):
        executor.commit('step1')
    assert leases.get(lease.lease_id).state is LeaseState.COMMITTED
    assert stream.snapshot().committed_cursor == 0
    assert journal.load('step1').state is StepState.PREPARED
    with pytest.raises(RuntimeError, match='commit was already attempted'):
        executor.commit('step1')
    restarted = CommitExecutor(lease_registry=leases, stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    assert restarted.recover('step1').action is RecoveryAction.ROLLBACK_GENERATION

def test_journal_failure_after_participant_commit_never_retries_optimizer(tmp_path, monkeypatch) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    executor.prepare(tx)
    executor.begin_optimizer('step1')
    executor.attach_candidate('step1', CandidateCheckpoint('dcp://candidate', 'checksum', 1))
    original_store = journal.store

    def fail_final_store(record):
        if record.state is StepState.COMMITTED:
            raise TimeoutError('injected final journal timeout')
        original_store(record)
    monkeypatch.setattr(journal, 'store', fail_final_store)
    with pytest.raises(TimeoutError, match='injected'):
        executor.commit('step1')
    assert leases.get(lease.lease_id).state is LeaseState.COMMITTED
    assert stream.snapshot().committed_cursor == 1
    assert journal.load('step1').state is StepState.PREPARED
    with pytest.raises(RuntimeError, match='commit was already attempted'):
        executor.commit('step1')
    restarted = CommitExecutor(lease_registry=leases, stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    assert restarted.recover('step1').action is RecoveryAction.ROLLBACK_GENERATION
    with pytest.raises(RuntimeError, match='earlier executor'):
        restarted.prepare(tx)

def test_consistent_recovery_reconciles_dcp_lease_and_stream(tmp_path, monkeypatch) -> None:
    executor, journal, leases, stream, lease, tx = _setup(tmp_path)
    checkpoint = tmp_path / 'dcp'
    checkpoint.mkdir()
    manifest = checkpoint / 'manifest.json'
    manifest.write_text(json.dumps({'generation': 1, 'topology_fingerprint': 'topo1', 'model_fingerprint': 'model1'}), encoding='utf-8')
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    executor.prepare(tx)
    executor.begin_optimizer('step1')
    executor.attach_candidate('step1', CandidateCheckpoint(str(checkpoint), checksum, 1))
    executor.commit('step1')
    assert executor.recover_consistent('step1').action is RecoveryAction.RESUME_COMMITTED
    manifest.write_text('{}', encoding='utf-8')
    assert executor.recover_consistent('step1').action is RecoveryAction.ROLLBACK_GENERATION
    manifest.write_text(json.dumps({'generation': 1, 'topology_fingerprint': 'topo1', 'model_fingerprint': 'model1'}), encoding='utf-8')
    missing_leases = CommitExecutor(lease_registry=LeaseRegistry(), stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    assert missing_leases.recover_consistent('step1').action is RecoveryAction.ROLLBACK_GENERATION
    empty_stream = TrainingStreamCoordinator('scope')
    missing_stream = CommitExecutor(lease_registry=leases, stream=empty_stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    assert missing_stream.recover_consistent('step1').action is RecoveryAction.ROLLBACK_GENERATION
    wrong_step_leases = LeaseRegistry()
    record = _record('producer', 1)
    wrong_lease = wrong_step_leases.register(AcquireResult('ready', 'acq1', (record,)), consumer_id='consumer', lease_timeout_s=60)[0]
    assert wrong_lease.lease_id == lease.lease_id
    wrong_step_leases.mark_materialized((wrong_lease.lease_id,))
    wrong_step_leases.mark_payload_released((wrong_lease.lease_id,))
    wrong_step_leases.commit(optimizer_step_id='another-step', transaction_set_hash=tx.transaction_set_hash, lease_ids=(wrong_lease.lease_id,))
    wrong_step = CommitExecutor(lease_registry=wrong_step_leases, stream=stream, journal=journal, topology_fingerprint='topo1', model_fingerprint='model1')
    assert wrong_step.recover_consistent('step1').action is RecoveryAction.ROLLBACK_GENERATION

    def remote_owner_unreachable(**kwargs):
        del kwargs
        raise Exception('injected remote lease-owner failure')
    monkeypatch.setattr(leases, 'validate_commit', remote_owner_unreachable)
    assert executor.recover_consistent('step1').action is RecoveryAction.ROLLBACK_GENERATION
