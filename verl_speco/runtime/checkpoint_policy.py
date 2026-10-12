# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import re
LIVE_PREFIX = 'live+file://'
SPARSE_ENV = frozenset({'SPECO_DRAFT_CHECKPOINT_INTERVAL', 'SPECO_DRAFT_LIVE_OWNER_TOKEN', 'SPECO_DRAFT_CHECKPOINT_STAGE_END'})

def checkpoint_interval(config):
    value = config.get('epoch_checkpoint_interval', 1)
    if type(value) is not int or not 1 <= value <= 10000:
        raise ValueError('epoch checkpoint interval must be an integer in [1,10000]')
    return value

def checkpoint_due(step, *, plan, interval, stage_end):
    checkpoint_interval({'epoch_checkpoint_interval': interval})
    if type(step) is not int or type(stage_end) is not int or (not 0 <= step < stage_end <= plan.total_steps):
        raise ValueError('checkpoint step is outside its execution stage')
    start = step * plan.samples_per_step
    end = min(start + plan.samples_per_step, plan.total_samples)
    crosses_epoch = start // plan.samples_per_epoch != end // plan.samples_per_epoch
    return interval == 1 or (step + 1) % interval == 0 or crosses_epoch or (step + 1 == stage_end)

def validate_sparse_environment(environment, epoch_plan):
    values = [environment.get(k, '') for k in SPARSE_ENV]
    if not any(values):
        return (1, None)
    if not all(values) or epoch_plan is None:
        raise ValueError('sparse checkpoints require a complete bound epoch owner')
    interval = environment['SPECO_DRAFT_CHECKPOINT_INTERVAL']
    end = environment['SPECO_DRAFT_CHECKPOINT_STAGE_END']
    token = environment['SPECO_DRAFT_LIVE_OWNER_TOKEN']
    if not interval.isdecimal() or str(int(interval)) != interval or (not end.isdecimal()) or (str(int(end)) != end) or (not 0 < int(end) <= epoch_plan.total_steps) or (not sparse_epoch_checkpoints_sha(token)):
        raise ValueError('invalid sparse checkpoint owner or execution range')
    return (checkpoint_interval({'epoch_checkpoint_interval': int(interval)}), token)

def sparse_epoch_checkpoints_sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None

def is_live_candidate(candidate):
    uri = candidate.get('uri', '') if isinstance(candidate, dict) else candidate.uri
    return isinstance(uri, str) and uri.startswith(LIVE_PREFIX)

def live_update_path(candidate):
    uri = candidate.get('uri', '') if isinstance(candidate, dict) else candidate.uri
    if not is_live_candidate(candidate):
        raise ValueError('not a live optimizer descriptor')
    path = Path(uri[len(LIVE_PREFIX):])
    if not path.is_absolute() or '..' in path.parts or path.resolve() != path:
        raise ValueError('live optimizer descriptor path must be absolute and canonical')
    return path

def write_live_update(path, *, run_id, generation, step_id, step_index, cursor, weights_sha256, owner_token, topology_fingerprint, model_fingerprint, logical_checkpoint_fingerprint):
    """Write only an update identity, explicitly without model/optimizer state."""
    path = Path(path)
    if not path.is_absolute() or not sparse_epoch_checkpoints_sha(weights_sha256) or (not sparse_epoch_checkpoints_sha(owner_token)):
        raise ValueError('invalid live optimizer identity')
    value = dict(schema_version=1, kind='LIVE_OPTIMIZER_UPDATE', run_id=run_id, generation=generation, step_id=step_id, step_index=step_index, cursor=cursor, weights_sha256=weights_sha256, owner_token=owner_token, topology_fingerprint=topology_fingerprint, model_fingerprint=model_fingerprint, logical_checkpoint_fingerprint=logical_checkpoint_fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return live_update_candidate(path, generation=generation)

def live_update_candidate(path, *, generation):
    from verl_speco.trainer.distributed_checkpoint import CandidateCheckpoint
    raw = Path(path).read_bytes()
    candidate = CandidateCheckpoint(LIVE_PREFIX + str(path), hashlib.sha256(raw).hexdigest(), generation)
    read_live_update(candidate)
    return candidate

def read_live_update(candidate, *, owner_token=None):
    path = live_update_path(candidate)
    raw = path.read_bytes()
    checksum = candidate['checksum'] if isinstance(candidate, dict) else candidate.checksum
    generation = candidate['checkpoint_generation'] if isinstance(candidate, dict) else candidate.checkpoint_generation
    value = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != checksum or value.get('schema_version') != 1 or value.get('kind') != 'LIVE_OPTIMIZER_UPDATE' or (type(value.get('generation')) is not int) or (value['generation'] != generation) or (type(value.get('step_index')) is not int) or (value['step_index'] < 0) or (type(value.get('cursor')) is not int) or (value['cursor'] < 1) or (not all((isinstance(value.get(k), str) and value[k] for k in ('run_id', 'step_id', 'topology_fingerprint', 'model_fingerprint')))) or (not all((sparse_epoch_checkpoints_sha(value.get(k)) for k in ('weights_sha256', 'owner_token', 'logical_checkpoint_fingerprint')))) or ('files' in value) or (owner_token is not None and value['owner_token'] != owner_token):
        raise ValueError('live optimizer update identity/checksum/owner differs')
    return value

def require_live_receipt(receipt, *, logical_checkpoint_fingerprint=None):
    value = read_live_update(receipt['candidate'], owner_token=receipt.get('live_owner_token'))
    if receipt.get('checkpoint_saved') is not False or not sparse_epoch_checkpoints_sha(receipt.get('live_owner_token')) or value['run_id'] != receipt.get('run_id') or (value['generation'] != receipt.get('generation')) or (value['step_id'] != receipt.get('step_id')) or (value['cursor'] != receipt.get('stream_snapshot', {}).get('committed_cursor')) or (value['weights_sha256'] != receipt.get('draft_weights_sha256')) or (logical_checkpoint_fingerprint is not None and value['logical_checkpoint_fingerprint'] != logical_checkpoint_fingerprint):
        raise ValueError('live receipt differs from its process-bound update')
    return value
