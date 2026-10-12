# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
from dataclasses import replace
from unittest.mock import Mock
import pytest
from verl_speco.integration.multinode_publication import TargetWeightAck, commit_target_weight_acks
from verl_speco.integration.multinode_online import OnlinePhase, OnlineVersions, MultiNodeOnlineAdapter as MultiNodeVerlOnlineAdapter

@pytest.fixture
def adapter():
    value = MultiNodeVerlOnlineAdapter(scheduler=Mock(), versions=OnlineVersions('run', 3, 8, 102, 41, 5), producer_ids=('source',), abort_plan=Mock(), publish_commit=Mock(), resume_rollout=Mock(), bind_worker_plan=Mock())
    value.phase = OnlinePhase.TRAINED
    value.binding = Mock(versions=value.versions)
    return value

def acks():
    return tuple((TargetWeightAck('run', 3, 8, 103, str(rank), f'birth-{rank}', 'a' * 64, True, True) for rank in range(2)))

def commit(adapter, rows=None, **kwargs):
    options = dict(acknowledgements=acks() if rows is None else rows, expected_incarnations={'0': 'birth-0', '1': 'birth-1'}, target_revision=103, actor_weight_manifest_sha256='a' * 64)
    options.update(kwargs)
    return commit_target_weight_acks(adapter, **options)

def test_all_owned_target_acks_advance_only_serving_revision_and_keep_collection_binding(adapter):
    binding = adapter.binding
    result = commit(adapter)
    assert result['target_revision'] == 103 and result['acknowledged_workers'] == 2
    assert adapter.versions == OnlineVersions('run', 3, 8, 103, 41, 5)
    assert adapter.binding is binding and binding.versions.target_revision == 102
    assert adapter.phase is OnlinePhase.TRAINED
    adapter.resume_rollout.assert_not_called()
    adapter.publish_commit.assert_not_called()
    adapter.abort_plan.assert_not_called()

@pytest.mark.parametrize('field,value', [('run_id', 'foreign'), ('generation', 4), ('rl_step', 9), ('target_revision', 104), ('worker_incarnation', 'restarted'), ('actor_weight_manifest_sha256', 'b' * 64), ('kv_cache_cleared', False), ('draft_target_head_synced', False)])
def test_stale_foreign_or_incomplete_acks_abort_without_revision_or_resume(adapter, field, value):
    rows = acks()
    rows = (rows[0], replace(rows[1], **{field: value}))
    with pytest.raises(RuntimeError):
        commit(adapter, rows)
    assert adapter.versions.target_revision == 102 and adapter.phase is OnlinePhase.ABORTED
    adapter.abort_plan.assert_called_once()
    adapter.resume_rollout.assert_not_called()

@pytest.mark.parametrize('selection', ['missing', 'duplicate', 'foreign', 'mutable'])
def test_every_rank_must_ack_exactly_once(adapter, selection):
    rows = acks()
    if selection == 'missing':
        rows = rows[:1]
    if selection == 'duplicate':
        rows = (rows[0], rows[0])
    if selection == 'foreign':
        rows = (rows[0], replace(rows[1], worker_id='2'))
    if selection == 'mutable':
        rows = list(rows)
    with pytest.raises(RuntimeError, match='missing, duplicate or foreign'):
        commit(adapter, rows)
    assert adapter.versions.target_revision == 102

@pytest.mark.parametrize('revision', [102, 104, True])
def test_target_revision_cannot_skip_repeat_or_use_bool(adapter, revision):
    with pytest.raises(ValueError, match='exactly once'):
        commit(adapter, target_revision=revision)
    assert adapter.versions.target_revision == 102

def test_ack_before_frozen_head_training_does_not_change_online_phase(adapter):
    adapter.phase = OnlinePhase.PREPARED
    with pytest.raises(RuntimeError, match='expected online phase trained'):
        commit(adapter)
    assert adapter.phase is OnlinePhase.PREPARED
    adapter.abort_plan.assert_not_called()

@pytest.mark.parametrize('field,value', [('actor_weight_manifest_sha256', 'x' * 64), ('actor_weight_manifest_sha256', 'a' * 63), ('generation', True), ('kv_cache_cleared', 1), ('draft_target_head_synced', None), ('worker_incarnation', ' ')])
def test_ack_payload_types_and_checksum_are_strict(field, value):
    with pytest.raises(ValueError):
        replace(acks()[0], **{field: value})

def test_unknown_expected_rank_incarnation_aborts(adapter):
    with pytest.raises(ValueError, match='expected rank'):
        commit(adapter, expected_incarnations={'0': ''})
    assert adapter.phase is OnlinePhase.ABORTED
