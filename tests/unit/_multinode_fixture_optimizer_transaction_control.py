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
