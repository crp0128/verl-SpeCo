# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

def native_source_incarnation(plan, ordinal):
    if type(ordinal) is not int or not 0 <= ordinal < len(plan.producer_groups) or type(plan.generation) is not int or (plan.generation < 1):
        raise ValueError('invalid formal producer ordinal or generation')
    identity = [plan.run_id, plan.generation, plan.config_fingerprint, plan.endpoint_fingerprint, ordinal]
    return 'native-' + hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()

def claim_native_source_attempt(environment, owner, *, outputs_root, node_identity=None):
    from verl_speco.runtime.services import verify_native_launch_environment
    from verl_speco.runtime.services import owned_role_session_process as _process
    from verl_speco.runtime.artifacts import publish_json, read_published_json
    binding = verify_native_launch_environment(environment, role='rollout', outputs_root=outputs_root, node_identity=node_identity)
    if binding is None:
        return None
    process = _process(os.getpid())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if not isinstance(owner, dict) or type(owner.get('schema_version')) is not int or owner['schema_version'] != 1 or (owner.get('run_id') != binding['run_id']) or (type(owner.get('generation')) is not int) or (owner['generation'] != binding['generation']) or (owner.get('boot_id') != boot) or (not process) or (process['state'] == 'Z') or (process['session'] != os.getpid()) or any((type(owner.get(k)) is not int or owner[k] != process[k] for k in ('pid', 'start_ticks', 'session'))):
        raise ValueError('formal source claim requires its actual private Linux owner')
    ordinal = int(environment['SPECO_PRODUCER_ID'][9:])
    folder = Path(environment['SPECO_RESOLVED_LAUNCH_CONFIG_PATH']).resolve().parent
    claim = folder / f'source-owner-{ordinal:04d}.claim'
    if claim.is_symlink():
        raise ValueError('formal source claim cannot follow a symlink')
    receipt = dict(schema_version=1, **binding, producer_id=environment['SPECO_PRODUCER_ID'], producer_incarnation=environment['SPECO_PRODUCER_INCARNATION'], boot_id=boot, **{k: process[k] for k in ('pid', 'start_ticks', 'session')})
    try:
        claim.mkdir(mode=448)
    except FileExistsError:
        try:
            saved = read_published_json(claim / 'owner.json', max_bytes=16384)
        except (FileNotFoundError, ValueError, OSError) as error:
            raise RuntimeError('formal source attempt has an incomplete owner claim; use a fresh generation') from error
        if saved != receipt:
            raise RuntimeError('formal source incarnation cannot replace its claimed process; use a fresh generation')
        return saved
    publish_json(claim / 'owner.json', receipt)
    return receipt
from verl_speco.runtime.artifacts import published_json_paths, read_published_json

def sequence_schedule_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def sequence_prefill_budget(config: dict) -> int | None:
    root = Path(config['output_root'])
    fingerprint = sequence_schedule_fingerprint(config)
    path = root / ('serving-tuning.schedule-' + fingerprint + '.json')
    if not published_json_paths(root, path.name):
        path = root / 'serving-tuning.json'
        if not published_json_paths(root, path.name):
            return None
    profile = read_published_json(path, max_bytes=4096)
    if not isinstance(profile, dict) or set(profile) != {'schema_version', 'run_id', 'schedule_fingerprint', 'prefill_batched_tokens'} or type(profile['schema_version']) is not int or (profile['schema_version'] != 1) or (profile['run_id'] != config['run_id']) or (profile['schedule_fingerprint'] != fingerprint):
        raise ValueError('sequence serving tuning does not match its schedule')
    budget = profile['prefill_batched_tokens']
    if type(budget) is not int or not 2048 <= budget <= 65536:
        raise ValueError('sequence serving prefill budget must be an integer from 2048 to 65536')
    return budget
