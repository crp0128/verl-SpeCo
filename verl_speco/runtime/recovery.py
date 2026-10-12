# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from verl_speco.transport.streaming import MAX_NATIVE_BATCH_SAMPLES
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class PackedRecoveryVerdict:
    action: str
    reason: str
    old_generation: int
    next_generation: int | None

def decide_packed_generation_restart(*, run_id: str, generation: int, expected_samples: int, receipt_exists: bool, control_status: Mapping[str, Any] | None, journals_by_trainer: Mapping[str, Sequence[Mapping[str, Any]]], active_trainer_pids: Mapping[str, Sequence[int]]) -> PackedRecoveryVerdict:
    """Allow a new generation only after every old participant proves no commit."""

    def reject(reason: str) -> PackedRecoveryVerdict:
        return PackedRecoveryVerdict('fail_closed', reason, generation, None)
    if not run_id or generation < 1 or expected_samples < 1:
        return reject('invalid run, generation or sample count')
    if not journals_by_trainer or set(journals_by_trainer) != set(active_trainer_pids):
        return reject('journal and process probes do not cover the same trainer nodes')
    if any((pids for pids in active_trainer_pids.values())):
        return reject('an old-generation trainer process is still alive')
    if receipt_exists:
        return reject('old-generation receipt exists; reconcile the committed step')
    if control_status is None:
        return reject('old-generation control owner is unavailable')
    if control_status.get('run_id') != run_id or control_status.get('generation') != generation:
        return reject('control owner run or generation differs')
    leases = control_status.get('leases')
    router = control_status.get('router')
    if not isinstance(leases, Mapping) or not isinstance(router, Mapping):
        return reject('control owner status lacks lease or ready counts')
    if leases.get('leases_total') != expected_samples or leases.get('leases_committed') != 0 or leases.get('committed_steps') != 0 or (control_status.get('retained_items') != 0) or (control_status.get('used_bytes') != 0) or (router.get('ready_records') != 0):
        return reject('old-generation control has a commit or outstanding payload')
    observed_steps: set[str] = set()
    observed_transactions: set[tuple[str, ...]] = set()
    for host, journals in journals_by_trainer.items():
        old_records = []
        for item in journals:
            transaction = item.get('transaction')
            if not isinstance(transaction, Mapping):
                return reject(f'invalid journal transaction on {host}')
            if transaction.get('generation') == generation:
                old_records.append(item)
        if len(old_records) > 1:
            return reject(f'multiple old-generation steps on {host}')
        if not old_records:
            continue
        record = old_records[0]
        transaction = record['transaction']
        step_id = transaction.get('optimizer_step_id')
        ids = transaction.get('transaction_ids')
        if record.get('state') != 'STEP_PREPARED' or transaction.get('stream_scope') != run_id or (not isinstance(step_id, str)) or (not step_id) or (not isinstance(ids, list)) or (len(ids) != expected_samples) or any((not isinstance(item, str) or not item for item in ids)):
            return reject(f'old-generation journal is not one prepared packed step on {host}')
        observed_steps.add(step_id)
        observed_transactions.add(tuple(sorted(ids)))
    if len(observed_steps) > 1 or len(observed_transactions) > 1:
        return reject('trainer journals disagree on old-generation step identity')
    return PackedRecoveryVerdict('restart_new_generation', 'old generation has no committed step or live trainer', generation, generation + 1)

def decide_prefill_generation_restart(*, run_id, generation, expected_samples, receipt_exists, control_status, journals_by_roles, active_role_pids):
    """Restart a partial first prefill only after stopped roles prove no journal.

    Prepared or committed optimizers belong to the existing DCP recovery gate.
    Partially materialized RAM cannot advance a cursor and must be replayed.
    """

    def reject(reason):
        return PackedRecoveryVerdict('fail_closed', reason, generation, None)
    if not isinstance(run_id, str) or not run_id or type(generation) is not int or (generation < 1) or (type(expected_samples) is not int) or (not 1 <= expected_samples <= MAX_NATIVE_BATCH_SAMPLES) or (type(receipt_exists) is not bool):
        return reject('invalid prefill recovery identity')
    if not journals_by_roles or set(journals_by_roles) != set(active_role_pids) or any((not isinstance(rows, (list, tuple)) or rows for rows in journals_by_roles.values())) or any((not isinstance(pids, (list, tuple)) or pids for pids in active_role_pids.values())):
        return reject('prefill recovery needs stopped roles and no optimizer journals')
    if receipt_exists:
        return reject('a receipt exists; reconcile optimizer or checkpoint commit')
    if not isinstance(control_status, Mapping) or control_status.get('run_id') != run_id or type(control_status.get('generation')) is not int or (control_status['generation'] != generation):
        return reject('prefill control owner differs or is unavailable')
    if any((type(control_status.get(k)) is not int or control_status[k] != 0 for k in ('used_bytes', 'retained_items', 'pending_reservations', 'reclaim_pending'))):
        return reject('prefill owner still retains reservations or payloads')
    leases = control_status.get('leases')
    router = control_status.get('router')
    if not isinstance(leases, Mapping) or not isinstance(router, Mapping) or type(router.get('ready_records')) is not int or (router['ready_records'] != 0):
        return reject('prefill ready owner is not quiescent')
    zero = ('leases_leased', 'leases_materialized', 'leases_committed', 'committed_steps')
    if any((type(leases.get(k)) is not int or leases[k] != 0 for k in zero)):
        return reject('prefill has active leases or an optimizer commit')
    total = leases.get('leases_total')
    cancelled = leases.get('leases_cancelled')
    if type(total) is not int or type(cancelled) is not int or (not 0 <= total <= expected_samples) or (cancelled != total):
        return reject('partial prefill leases are not all cancelled')
    return PackedRecoveryVerdict('restart_new_generation', 'stopped partial prefill has no optimizer journal or credit', generation, generation + 1)

def decide_lost_prefill_owner_restart(*, run_id, generation, expected_samples, receipt_exists, control_owner_probe, expected_control_actor_id, journals_by_roles, active_role_pids, tq_remaining_keys, checkpoint_artifacts):
    """Admit a new generation after a permanently dead first-prefill owner.

    Lost actor memory cannot prove cancelled leases. Instead require its exact
    actor identity to be DEAD/unavailable, all old roles stopped, and no durable
    optimizer or checkpoint evidence and no residual partition keys. A prepared
    optimizer belongs to checkpoint reconciliation, never this from-zero gate.
    The caller supplies fresh read-only probes after owned-session shutdown.
    """

    def reject(reason):
        return PackedRecoveryVerdict('fail_closed', reason, generation, None)
    if not isinstance(run_id, str) or not run_id or type(generation) is not int or (generation < 1) or (type(expected_samples) is not int) or (not 1 <= expected_samples <= MAX_NATIVE_BATCH_SAMPLES) or (type(receipt_exists) is not bool):
        return reject('invalid lost-owner prefill recovery identity')
    if not journals_by_roles or set(journals_by_roles) != set(active_role_pids) or any((not isinstance(rows, (list, tuple)) or rows for rows in journals_by_roles.values())) or any((not isinstance(pids, (list, tuple)) or pids for pids in active_role_pids.values())):
        return reject('lost-owner recovery requires stopped roles and no optimizer journals')
    if receipt_exists or not isinstance(checkpoint_artifacts, (list, tuple)) or checkpoint_artifacts:
        return reject('lost-owner recovery has durable checkpoint or optimizer evidence')
    if not isinstance(tq_remaining_keys, (list, tuple)) or tq_remaining_keys:
        return reject('lost-owner generation still has partition payload keys')
    from verl_speco.transport.control import actor_name_for_run
    if not isinstance(expected_control_actor_id, str) or len(expected_control_actor_id) != 32 or any((char not in '0123456789abcdef' for char in expected_control_actor_id)):
        return reject('pre-failure control actor identity is missing')
    fields = {'run_id', 'generation', 'actor_name', 'actor_id', 'namespace', 'state', 'name_unavailable'}
    probe = control_owner_probe
    if not isinstance(probe, Mapping) or set(probe) != fields or probe['run_id'] != run_id or (type(probe['generation']) is not int) or (probe['generation'] != generation) or (probe['actor_name'] != actor_name_for_run(run_id, generation)) or (probe['namespace'] != 'transfer_queue') or (probe['state'] != 'DEAD') or (probe['name_unavailable'] is not True) or (probe['actor_id'] != expected_control_actor_id) or (not isinstance(probe['actor_id'], str)) or (len(probe['actor_id']) != 32) or any((char not in '0123456789abcdef' for char in probe['actor_id'])):
        return reject('old control owner is not proven permanently dead and unavailable')
    return PackedRecoveryVerdict('restart_new_generation', 'dead first-prefill owner has no live roles, payload or durable update evidence', generation, generation + 1)
from collections.abc import Mapping, Sequence
from typing import Any

def decide_completed_capture_resume(*, run_id: str, generation: int, expected_samples: int, receipt_exists: bool, control_status: Mapping[str, Any] | None, journals_by_trainer: Mapping[str, Sequence[Mapping[str, Any]]], active_trainer_pids: Mapping[str, Sequence[int]]) -> PackedRecoveryVerdict:

    def reject(reason: str) -> PackedRecoveryVerdict:
        return PackedRecoveryVerdict('fail_closed', reason, generation, None)
    if not run_id or generation < 1 or expected_samples < 1:
        return reject('invalid capture resume identity')
    if not journals_by_trainer or set(journals_by_trainer) != set(active_trainer_pids) or any(active_trainer_pids.values()) or receipt_exists:
        return reject('capture resume has live participants, incomplete probes or a step receipt')
    if control_status is None or (control_status.get('run_id'), control_status.get('generation')) != (run_id, generation):
        return reject('capture resume control owner identity differs')
    leases, router = (control_status.get('leases'), control_status.get('router'))
    if not isinstance(leases, Mapping) or not isinstance(router, Mapping):
        return reject('capture resume lacks lease or queue status')
    if any((leases.get(key) != 0 for key in ('leases_total', 'leases_committed', 'committed_steps', 'leases_leased', 'leases_materialized'))) or control_status.get('retained_items') != expected_samples or (not isinstance(control_status.get('used_bytes'), int)) or (control_status['used_bytes'] <= 0) or (control_status.get('pending_reservations') != 0) or (control_status.get('reclaim_pending') != 0) or (router.get('ready_records') != expected_samples):
        return reject('capture resume is not one complete unclaimed retained batch')
    for journals in journals_by_trainer.values():
        for record in journals:
            transaction = record.get('transaction')
            if not isinstance(transaction, Mapping) or transaction.get('generation') == generation:
                return reject('capture resume has an invalid or already prepared optimizer journal')
    return PackedRecoveryVerdict('resume_completed_capture', 'all captures retained, no leases or optimizer state, all roles stopped', generation, None)
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Sequence
checkpoint_integrity_SHARD_NAME = re.compile('[A-Za-z0-9][A-Za-z0-9_.-]*\\.safetensors')

def checkpoint_integrity_file_digest(path: Path) -> tuple[str, int]:
    before = path.stat()
    if not path.is_file():
        raise ValueError(f'checkpoint shard is not a regular file: {path}')
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
            count += len(chunk)
    after = path.stat()
    if before.st_ino != after.st_ino or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or (count != after.st_size):
        raise RuntimeError(f'checkpoint shard changed while hashing: {path}')
    return (digest.hexdigest(), count)

def audit_checkpoint_shards(*, model_overlay: str | Path, target_model_dir: str | Path) -> dict:
    """Hash config, index and every referenced shard without loading tensors."""
    overlay = Path(model_overlay)
    target = Path(target_model_dir)
    config_path = overlay / 'config.json'
    index_path = target / 'model.safetensors.index.json'
    index_before = index_path.stat()
    index_bytes_raw = index_path.read_bytes()
    index_after = index_path.stat()
    if index_before.st_ino != index_after.st_ino or index_before.st_size != index_after.st_size or index_before.st_mtime_ns != index_after.st_mtime_ns or (len(index_bytes_raw) != index_after.st_size):
        raise RuntimeError('checkpoint index changed while hashing')
    index = json.loads(index_bytes_raw)
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError('checkpoint index has no weight_map')
    names = set()
    for name in weight_map.values():
        if not isinstance(name, str) or checkpoint_integrity_SHARD_NAME.fullmatch(name) is None:
            raise ValueError('checkpoint index names an invalid shard path')
        names.add(name)
    if not names:
        raise ValueError('checkpoint index references no shard')
    config_digest, config_bytes = checkpoint_integrity_file_digest(config_path)
    index_digest = hashlib.sha256(index_bytes_raw).hexdigest()
    index_bytes = len(index_bytes_raw)
    shards = {}
    for name in sorted(names):
        shard_path = target / name
        if shard_path.is_symlink():
            raise ValueError(f'checkpoint index references a symlinked shard: {name}')
        digest, size = checkpoint_integrity_file_digest(shard_path)
        shards[name] = {'sha256': digest, 'bytes': size}
    identity = {'config_sha256': config_digest, 'index_sha256': index_digest, 'shards': shards}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    return {'schema_version': 1, 'model_content_fingerprint': fingerprint, 'config_sha256': config_digest, 'config_bytes': config_bytes, 'index_sha256': index_digest, 'index_bytes': index_bytes, 'weight_count': len(weight_map), 'shard_count': len(shards), 'shard_bytes': sum((item['bytes'] for item in shards.values())), 'shards': shards}

def save_checkpoint_audit(report: dict, path: str | Path) -> None:
    """Publish the complete report only after all shard reads succeeded."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-audit-', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, sort_keys=True, separators=(',', ':'))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)

def load_checkpoint_audit(path: str | Path, *, model_overlay: str | Path, target_model_dir: str | Path, verify_shards_full: bool=False) -> dict:
    """Validate a completed audit; strict mode re-hashes every shard.

    Fast mode verifies report integrity, live config/index and shard sizes. It
    assumes shard contents stay immutable after the full audit was created.
    """
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if report.get('schema_version') != 1 or not isinstance(report.get('shards'), dict):
        raise ValueError('unsupported checkpoint content audit report')
    shards = report['shards']
    identity = {'config_sha256': report['config_sha256'], 'index_sha256': report['index_sha256'], 'shards': shards}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if report.get('model_content_fingerprint') != expected:
        raise ValueError('checkpoint content audit fingerprint differs')
    overlay = Path(model_overlay)
    target = Path(target_model_dir)
    config_digest, config_bytes = checkpoint_integrity_file_digest(overlay / 'config.json')
    index_path = target / 'model.safetensors.index.json'
    index_before = index_path.stat()
    index_raw = index_path.read_bytes()
    index_after = index_path.stat()
    if index_before.st_ino != index_after.st_ino or index_before.st_size != index_after.st_size or index_before.st_mtime_ns != index_after.st_mtime_ns or (len(index_raw) != index_after.st_size):
        raise RuntimeError('checkpoint index changed during audit validation')
    index_digest = hashlib.sha256(index_raw).hexdigest()
    index_bytes = len(index_raw)
    index = json.loads(index_raw)
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map or any((not isinstance(name, str) or checkpoint_integrity_SHARD_NAME.fullmatch(name) is None for name in weight_map.values())) or (set(weight_map.values()) != set(shards)) or (config_digest != report.get('config_sha256')) or (config_bytes != report.get('config_bytes')) or (index_digest != report.get('index_sha256')) or (index_bytes != report.get('index_bytes')) or (len(weight_map) != report.get('weight_count')) or (len(shards) != report.get('shard_count')):
        raise ValueError('checkpoint content audit index or config differs')
    total_bytes = 0
    for name, entry in shards.items():
        if checkpoint_integrity_SHARD_NAME.fullmatch(name) is None or not isinstance(entry, dict):
            raise ValueError('checkpoint content audit has invalid shard entry')
        shard_path = target / name
        if shard_path.is_symlink() or not shard_path.is_file():
            raise ValueError(f'checkpoint audit shard is missing or symlinked: {name}')
        size = shard_path.stat().st_size
        if size != entry.get('bytes'):
            raise ValueError(f'checkpoint audit shard size differs: {name}')
        total_bytes += size
    if total_bytes != report.get('shard_bytes'):
        raise ValueError('checkpoint audit total shard bytes differ')
    if verify_shards_full:
        actual = audit_checkpoint_shards(model_overlay=overlay, target_model_dir=target)
        if actual != report:
            raise ValueError('checkpoint shard content differs from audit')
    return report

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Hash every referenced checkpoint shard')
    parser.add_argument('--model-overlay', required=True, type=Path)
    parser.add_argument('--target-model-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit_checkpoint_shards(model_overlay=args.model_overlay, target_model_dir=args.target_model_dir)
    save_checkpoint_audit(report, args.output)
    print(json.dumps({key: report[key] for key in ('model_content_fingerprint', 'weight_count', 'shard_count', 'shard_bytes')}, sort_keys=True))
    return 0

def checkpoint_mesh_fingerprint(*, runtime_config, launch_plan, packing, draft_window, inputs):
    mesh = {key: launch_plan[key] for key in ('schema_version', 'mode', 'nodes', 'placements', 'producer_groups', 'master_addr', 'draft_world_size')}
    settings = {key: value for key, value in packing.items() if key not in ('generation', 'step_index')}
    identity = dict(schema_version=1, runtime_config=runtime_config, mesh=mesh, packing=settings, draft_window=draft_window, inputs=inputs)
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
from collections.abc import Mapping

def checkpoint_hash_workers(environment: Mapping[str, str]) -> int:
    flag = environment.get('SPECO_DCP_FILE_HASH_WORKERS', '')
    if type(flag) is not str or flag not in ('', '1', '2', '3', '4'):
        raise ValueError('checkpoint file hash workers flag must be empty or 1..4')
    return int(flag or '1')

def checkpoint_write_workers(environment: Mapping[str, str]) -> int:
    flag = environment.get('SPECO_DCP_FILE_WRITE_WORKERS', '')
    if type(flag) is not str or flag not in ('', '1', '2', '3', '4'):
        raise ValueError('checkpoint file write workers flag must be empty or 1..4')
    return int(flag or '1')
import stat
from verl_speco.runtime.artifacts import publish_json, read_published_json

def epoch_checkpoint_retention_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def epoch_checkpoint_retention_regular(path):
    if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('retirement requires an owned regular file without symlinks')
    return path.stat().st_size

def epoch_checkpoint_retention_safe_checkpoint_name(name):
    if not isinstance(name, str) or not name or name in ('.', '..', 'manifest.json'):
        return False
    parts = Path(name).parts
    if len(parts) == 1:
        return Path(name).name == name
    return len(parts) == 2 and parts[0] == 'dcp' and (name == '/'.join(parts)) and (parts[1] == '.metadata' or re.fullmatch('__[0-9]+_[0-9]+\\.distcp', parts[1]) is not None)

def epoch_checkpoint_retention_checkpoint_tree(directory, filenames, *, interrupted=False):
    """Admit only the manifest's files and its one native DCP directory."""
    if not directory.exists():
        if interrupted:
            return
        raise FileNotFoundError(str(directory))
    expected = {Path(name).parts[0] for name in filenames} | {'manifest.json'}
    actual = {path.name for path in directory.iterdir()}
    if not actual <= expected or (not interrupted and actual != expected):
        raise ValueError('checkpoint has unknown or omitted files or directories')
    nested = {Path(name).name for name in filenames if len(Path(name).parts) == 2}
    folder = directory / 'dcp'
    if nested and folder.exists():
        if folder.resolve() != folder or not stat.S_ISDIR(folder.lstat().st_mode):
            raise ValueError('native DCP folder is not an owned regular directory')
        children = {path.name for path in folder.iterdir()}
        if not children <= nested or (not interrupted and children != nested):
            raise ValueError('native DCP folder has unknown or omitted files')
    elif nested and (not interrupted):
        raise FileNotFoundError(str(folder))

def retirement_plan(records, *, outputs_root, work_dir, _resume_plan=None, allow_sparse=False):
    """Validate three consecutive durable records before planning any deletion.

    Only the oldest DCP and its large shared control snapshot are eligible.
    Journals, committed receipts, original sequences and two newer checkpoints
    remain available. A plan never deletes an in-flight or unknown directory.
    """
    if not isinstance(records, (list, tuple)) or len(records) != 3 or any((not isinstance(r, dict) for r in records)):
        raise ValueError('retirement requires two protected durable successors')
    old, previous, latest = records
    if _resume_plan is not None:
        if _resume_plan.get('records_sha256') != hashlib.sha256(epoch_checkpoint_retention_canonical(records)).hexdigest() or not isinstance(_resume_plan.get('files'), list):
            raise ValueError('retirement intent differs from durable records')
        sizes = {row['path']: row['nbytes'] for row in _resume_plan['files']}
        if len(sizes) != len(_resume_plan['files']) or any((type(size) is not int or size < 1 for size in sizes.values())):
            raise ValueError('retirement intent has invalid file sizes')
    else:
        sizes = {}

    def owned_size(path, *, may_be_absent):
        if path.exists() or path.is_symlink():
            size = epoch_checkpoint_retention_regular(path)
            if sizes and may_be_absent and (size != sizes.get(str(path))):
                raise ValueError('obsolete file changed during interrupted retirement')
            return size
        if not may_be_absent or str(path) not in sizes:
            raise FileNotFoundError(str(path))
        return sizes[str(path)]
    root = Path(outputs_root).resolve()
    work = Path(work_dir)
    run = old.get('run_id')
    generation = old.get('generation')
    if not isinstance(run, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', run) or type(generation) is not int or (generation < 1) or (work != root / f'speco-native-wave-{run}-g{generation}') or (work.resolve() != work):
        raise ValueError('retirement work directory differs from its exact owned generation')
    for record in records:
        if type(record.get('schema_version')) is not int or record['schema_version'] != 1 or record.get('state') != 'STEP_COMMITTED' or (record.get('run_id') != run) or (type(record.get('generation')) is not int) or (record['generation'] != generation) or (type(record.get('step_index')) is not int) or (record['step_index'] < 0) or (type(record.get('start')) is not int) or (type(record.get('end')) is not int) or (not 0 <= record['start'] < record['end']) or (not isinstance(record.get('receipt_sha256'), str)) or (re.fullmatch('[a-f0-9]{64}', record['receipt_sha256']) is None):
            raise ValueError('retirement record is not a matching durable committed step')
        persisted = read_published_json(work / f"committed-step-{record['step_index']:04d}.json")
        if persisted != record:
            raise ValueError('retirement record differs from its published durable receipt reference')
    if type(allow_sparse) is not bool:
        raise ValueError('invalid sparse retention policy')
    if allow_sparse:
        from verl_speco.runtime.checkpoint_policy import is_live_candidate, read_live_update
        if not old['step_index'] < previous['step_index'] < latest['step_index']:
            raise ValueError('sparse durable successors must advance')
        expected = old['end']
        protected = {r['step_index'] for r in records[1:]}
        for index in range(old['step_index'] + 1, latest['step_index'] + 1):
            row = read_published_json(work / f'committed-step-{index:04d}.json')
            if row.get('run_id') != run or row.get('generation') != generation or row.get('step_index') != index or (row.get('state') != 'STEP_COMMITTED') or (row.get('start') != expected) or (type(row.get('end')) is not int) or (row['end'] <= expected):
                raise ValueError('sparse retention omitted progress or cursor')
            if index not in protected:
                if row.get('checkpoint_saved') is not False or not is_live_candidate(row.get('candidate', {})) or re.fullmatch('[a-f0-9]{64}', str(row.get('receipt_sha256', ''))) is None:
                    raise ValueError('sparse retention skipped a durable or unknown checkpoint')
                live = read_live_update(row['candidate'], owner_token=old.get('live_owner_token'))
                if live['run_id'] != run or live['generation'] != generation or live['step_index'] != index or (live['cursor'] != row['end']):
                    raise ValueError('sparse progress descriptor differs from its complete cursor')
            expected = row['end']
        if any((r.get('checkpoint_saved') is not True for r in records)):
            raise ValueError('sparse retention requires three actual full checkpoints')
    elif previous['step_index'] != old['step_index'] + 1 or latest['step_index'] != previous['step_index'] + 1 or old['end'] != previous['start'] or (previous['end'] != latest['start']):
        raise ValueError('retirement successors have an omitted step or stream cursor')
    checkpoint_root = root / f'speco-three-node-{run}'
    candidate_roots = []
    for record in records:
        candidate = record.get('candidate', {})
        directory = Path(candidate.get('uri', ''))
        if directory.parent != checkpoint_root or directory.resolve() != directory or re.fullmatch('step-[a-f0-9]{20}', directory.name) is None or (candidate.get('checkpoint_generation') != generation):
            raise ValueError('retirement checkpoint escaped the exact owned run')
        is_obsolete = record is old
        manifest_path = directory / 'manifest.json'
        if _resume_plan is not None and is_obsolete and (not manifest_path.exists()):
            payload = _resume_plan['obsolete_manifest_utf8'].encode('utf-8')
        else:
            epoch_checkpoint_retention_regular(manifest_path)
            payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        if hashlib.sha256(payload).hexdigest() != candidate.get('checksum') or manifest.get('schema_version') != 1 or manifest.get('generation') != generation or (manifest.get('world_size') != 16) or (not manifest.get('files')):
            raise ValueError('retirement candidate manifest differs from its durable reference')
        for name, checksum in manifest['files'].items():
            if not epoch_checkpoint_retention_safe_checkpoint_name(name) or not isinstance(checksum, str) or re.fullmatch('[a-f0-9]{64}', checksum) is None or (owned_size(directory / name, may_be_absent=is_obsolete and _resume_plan is not None) < 1):
                raise ValueError('protected checkpoint is incomplete or has unsafe files')
        epoch_checkpoint_retention_checkpoint_tree(directory, list(manifest['files']), interrupted=is_obsolete and _resume_plan is not None)
        candidate_roots.append(directory)
    if len(set(candidate_roots)) != 3:
        raise ValueError('obsolete and protected checkpoints alias each other')
    directory = candidate_roots[0]
    manifest_file = directory / 'manifest.json'
    manifest_payload = manifest_file.read_bytes() if manifest_file.exists() else _resume_plan['obsolete_manifest_utf8'].encode('utf-8')
    manifest = json.loads(manifest_payload)
    filenames = list(manifest['files'])
    if any((not epoch_checkpoint_retention_safe_checkpoint_name(name) for name in filenames)):
        raise ValueError('obsolete checkpoint has unsafe manifest filenames')
    files = []
    for name in filenames + ['manifest.json']:
        path = directory / name
        files.append(dict(path=str(path), nbytes=owned_size(path, may_be_absent=_resume_plan is not None)))
    descriptor = old.get('control_snapshot', {})
    control = Path(descriptor.get('path', ''))
    if descriptor.get('storage') != 'shared-filesystem-v1' or control.parent != work / 'controls' or re.fullmatch('step-' + f"{old['step_index']:04d}" + '-control-[a-f0-9]{32}\\.json', control.name) is None or (owned_size(control, may_be_absent=_resume_plan is not None) != descriptor.get('nbytes')):
        raise ValueError('obsolete control snapshot differs from exact owned step')
    if control.exists():
        with control.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != descriptor.get('sha256'):
                raise ValueError('obsolete control snapshot checksum changed')
    if any((record.get('control_snapshot', {}).get('path') == str(control) for record in records[1:])):
        raise ValueError('obsolete and protected control snapshots alias each other')
    files.append(dict(path=str(control), nbytes=owned_size(control, may_be_absent=_resume_plan is not None)))
    result = dict(schema_version=1, kind='OWNED_EPOCH_CHECKPOINT_RETIREMENT', run_id=run, generation=generation, obsolete_step_index=old['step_index'], protected_steps=[r['step_index'] for r in records[1:]], records_sha256=hashlib.sha256(epoch_checkpoint_retention_canonical(records)).hexdigest(), checkpoint_directory=str(directory), intent_path=str(work / f"checkpoint-retirement-step-{old['step_index']:04d}.json"), files=files, planned_deleted_bytes=sum((row['nbytes'] for row in files)), obsolete_manifest_utf8=manifest_payload.decode('utf-8'), keeps_two_latest_durable_checkpoints=True, deletes_journals_or_sequence_payloads=False)
    if _resume_plan is not None and result != _resume_plan:
        raise ValueError('retirement intent differs from reconstructed owned plan')
    return result

def apply_retirement(records, *, outputs_root, work_dir, allow_sparse=False):
    """Resume exact recorded deletion after revalidating both protected successors."""
    intent = Path(work_dir) / f"checkpoint-retirement-step-{records[0]['step_index']:04d}.json"
    try:
        existing = read_published_json(intent)
    except FileNotFoundError:
        existing = None
    if existing is not None and existing.get('state') not in ('RETIREMENT_STARTED', 'CHECKPOINT_RETIRED'):
        raise ValueError('unknown retirement intent state')
    plan = retirement_plan(records, outputs_root=outputs_root, work_dir=work_dir, allow_sparse=allow_sparse, _resume_plan=None if existing is None else existing['plan'])
    intent = Path(plan['intent_path'])
    if existing is not None and existing['state'] == 'CHECKPOINT_RETIRED':
        if Path(plan['checkpoint_directory']).exists() or any((Path(row['path']).exists() or Path(row['path']).is_symlink() for row in plan['files'])):
            raise ValueError('retired artifacts were recreated; refuse a second deletion')
        return existing
    if existing is None:
        publish_json(intent, dict(state='RETIREMENT_STARTED', plan=plan))
    deleted = 0
    already_absent = 0
    for row in plan['files']:
        path = Path(row['path'])
        if not path.exists() and (not path.is_symlink()):
            already_absent += row['nbytes']
            continue
        if epoch_checkpoint_retention_regular(path) != row['nbytes']:
            raise RuntimeError('obsolete file changed after retirement admission')
        path.unlink()
        deleted += row['nbytes']
    directory = Path(plan['checkpoint_directory'])
    if directory.exists():
        folder = directory / 'dcp'
        if any((len(Path(name).parts) == 2 for name in json.loads(plan['obsolete_manifest_utf8'])['files'])) and folder.exists():
            folder.rmdir()
        directory.rmdir()
    result = dict(state='CHECKPOINT_RETIRED', plan=plan, deleted_bytes=deleted + already_absent, deleted_bytes_this_attempt=deleted, already_absent_bytes=already_absent, logical_deleted_bytes_only=True, mounted_df_reclamation_not_claimed=True)
    publish_json(intent, result)
    return result
EPOCH_FIELDS = ('SPECO_EPOCH_REPLAY_SPEC_PATH', 'SPECO_EPOCH_REPLAY_SPEC_SHA256', 'SPECO_EPOCH_REPLAY_PLAN_SHA256')

@dataclass(frozen=True)
class EpochDraftPlan:
    plan: object
    bindings: tuple

    @property
    def total_steps(self):
        return self.plan.total_steps

    @property
    def total_samples(self):
        return self.plan.total_samples

    @property
    def samples_per_step(self):
        return self.plan.samples_per_step

    @property
    def producer_count(self):
        return self.plan.producer_count

    @property
    def plan_sha256(self):
        return self.plan.plan_sha256

def validate_epoch_plan_capacity(value):
    from verl_speco.producer.replay import FrozenReplayEpochPlan
    from verl_speco.transport.streaming import MAX_RAW_BYTES
    if not isinstance(value, EpochDraftPlan) or not isinstance(value.plan, FrozenReplayEpochPlan) or value.plan.epochs != 3 or (not 1 <= value.total_steps <= 6000000):
        raise ValueError('epoch draft requires the admitted frozen three-epoch plan')
    if 40 * value.total_samples + 120 * value.total_steps + 8192 > MAX_RAW_BYTES:
        raise ValueError('complete epoch history exceeds bounded receipt capacity before training')

def validate_epoch_command(environment, *, epoch_plan):
    configured = [environment.get(k, '') for k in EPOCH_FIELDS]
    if not any(configured):
        if epoch_plan is not None:
            raise ValueError('epoch plan requires explicit frozen file binding')
        return
    validate_epoch_plan_capacity(epoch_plan)
    if any((not isinstance(v, str) or not v for v in configured)) or environment.get('SPECO_DRAFT_VARIABLE_BATCH') != '1' or environment.get('SPECO_DRAFT_WORLD_SIZE') != '16' or (environment.get(EPOCH_FIELDS[2]) != epoch_plan.plan_sha256) or any((environment.get(k, '') != v for k, v in epoch_plan.bindings)):
        raise ValueError('epoch draft command differs from its admitted frozen source')

def load_epoch_draft_plan(environment, *, outputs_root):
    configured = [environment.get(k, '') for k in EPOCH_FIELDS]
    if not any(configured):
        return None
    if any((not isinstance(v, str) or not v for v in configured)):
        raise ValueError('partial epoch draft configuration cannot fall back to diagnostics')
    from verl_speco.producer.replay import checked_json, epoch_replay_cursor_factory
    from verl_speco.standalone_tq_training_launcher import _stable_path_identity
    spec = checked_json(configured[0], configured[1], outputs_root=Path(outputs_root).resolve(), max_bytes=65536)
    model = environment.get('SPECO_TARGET_MODEL_OVERLAY', '')
    source_path = environment.get('SPECO_PROMPT_DATASET', '')
    if not model or not source_path:
        raise ValueError('epoch draft source needs the actual model and dataset paths')
    source = epoch_replay_cursor_factory(dict(environment, SPECO_SEQUENCE_REPLAY_INPUT=source_path), outputs_root=outputs_root, producer_count=spec['producer_count'], max_sequence_length=int(environment['SPECO_TARGET_MAX_SEQUENCE_LENGTH']), provenance=dict(target_model_id=model, target_model_revision=_stable_path_identity('target', model), tokenizer_fingerprint=_stable_path_identity('tokenizer', model)))
    keys = (*EPOCH_FIELDS, 'SPECO_TARGET_MODEL_OVERLAY', 'SPECO_TARGET_MODEL_DIR', 'SPECO_PROMPT_DATASET', 'SPECO_TARGET_MAX_SEQUENCE_LENGTH')
    value = EpochDraftPlan(source.plan, tuple(((k, environment.get(k, '')) for k in keys)))
    validate_epoch_command(environment, epoch_plan=value)
    return value

def expected_epoch_sample_keys(value, *, run_id, generation, step_index):
    validate_epoch_plan_capacity(value)
    if type(step_index) is not int or not 0 <= step_index < value.total_steps:
        raise ValueError('epoch batch is outside the admitted plan')
    start = step_index * value.samples_per_step
    end = min(start + value.samples_per_step, value.total_samples)
    return [f'draft:v3:{run_id}:g{generation}:producer-{i % value.producer_count}:{i:020d}:{value.plan.occurrence(i).sample_id}' for i in range(start, end)]
from copy import deepcopy
from verl_speco.producer.capture import EpochNativeBatches
from verl_speco.producer.capture import source_capture_slot_bytes
from verl_speco.transport.streaming import SampleMetadataV3
from verl_speco.trainer.owned_batches import StreamingOwnedSWARows
from verl_speco.runtime.services import build_physical_capture_command

def prepare_epoch_step(config, controls, step_index):
    if not isinstance(controls, EpochNativeBatches):
        raise ValueError('epoch preparation needs the admitted actual lazy controls')
    if type(config.get('max_samples_per_row')) is not int or config['max_samples_per_row'] not in (16, 64):
        raise ValueError('complete64K epoch packing permits only16 or64 declared document segments')
    from verl_speco.runtime.services import draft_training_profile
    profile = draft_training_profile(config.get('draft_training_profile', 'dmtp-92b-block8-swa-confidence'))
    if config['max_samples_per_row'] > profile.swa_chunk_cap:
        raise ValueError('complete64K packing profile has fewer regions than possible documents')
    context = config.get('max_sequence_length')
    if type(context) is not int or context not in (49152, 65536):
        raise ValueError('complete long epoch profile requires its explicit48K or64K engine context')
    physical = config.get('source_physical_batch_size')
    if type(physical) is not int or physical not in (8, 16):
        raise ValueError('invalid epoch physical source width')
    from verl_speco.producer.batching import balance_physical_batch_size
    if physical != balance_physical_batch_size(controls.plan.prefill_balance_policy):
        raise ValueError('epoch physical source width differs from frozen plan')
    from verl_speco.producer.capture import config_slot_profile
    source_slot_profile = config_slot_profile(config)
    banks = config.get('source_native_buffer_batches')
    if type(banks) is not int or banks not in (2, 3, 4, 10) or (banks in (3, 4, 10) and context != 49152) or (physical == 16 and (banks != 2 or context != 49152)):
        raise ValueError('complete epoch source bank profile requires two or admitted48K three-bank')
    wide_receive = banks in (4, 10) or (physical == 16 and config.get('native_receive_workers') == 8)
    exact = {'steps': controls.plan.total_steps, 'samples_per_step': controls.plan.samples_per_step, 'max_sequence_length': context, 'source_tp': 8, 'source_capture_owner_ranks': 8, 'source_physical_batch_size': physical, 'source_native_buffer_batches': banks, 'source_prefill_batch_tokens': 131072, 'source_graph_mode': 'FULL_DECODE_ONLY', 'packing_rows': 32, 'max_samples_per_row': config.get('max_samples_per_row'), 'global_capacity_bytes': (96 if source_slot_profile else 24 * banks) * 1024 ** 3, 'native_payload_prefetch_bytes': (16 if wide_receive else 8) * 1024 ** 3, 'native_verification_workers': 8 if wide_receive else 4, 'source_memory_utilization': 0.85}
    true_flags = ['source_long_context', 'source_early_ready', 'source_step_lookahead', 'resolved_launch_plan', 'source_full_batch_prefill', 'source_mhc_sequence_parallel', 'source_mhc_prefill_fusion']
    if not isinstance(config, dict) or any((type(config.get(k)) is not type(v) or config.get(k) != v for k, v in exact.items())) or any((config.get(k) is not True for k in true_flags)) or (len(config.get('sources', [])) != 6) or (controls.plan.producer_count != 6) or (controls.plan.epochs != 3) or (config.get('run_id') != controls.run_id) or (controls.context != context) or ('second_trainer' in config):
        raise ValueError('complete64K epoch step differs from the admitted FULL DP16 profile')
    batch = controls.batch(step_index)
    packed = controls.packed_control(step_index)
    samples = packed['samples']
    metas = [SampleMetadataV3.from_dict(v['metadata']) for v in samples]
    if [m.sequence_no for m in metas] != list(range(batch.start_cursor, batch.end_cursor)):
        raise ValueError('epoch source metadata differs from exact occurrence order')
    if any((m.sample_id != o.sample_id or m.token_count != o.original_tokens for m, o in zip(metas, batch.occurrences))):
        raise ValueError('complete source lengths or identities differ from the frozen occurrence plan')
    if any((m.supervised_token_count < 8 or m.prompt_token_count is None or m.prompt_token_count < 1 for m in metas)):
        raise ValueError('complete64K training snapshot needs a prompt predecessor and complete block8 response')
    slot = source_capture_slot_bytes(target_tp=8, physical_slots=physical, owner_ranks=8, long_context=True)
    if any((m.token_count > context or m.payload_nbytes + 1024 > slot for m in metas)):
        raise ValueError('complete epoch row exceeds actual native slot capacity')
    pipeline = config.get('native_owned_copy_pipeline', False)
    chunk = config.get('native_owned_release_chunk_size', 0)
    if type(chunk) is not int or chunk not in (0, 6, 12) or (chunk and (not pipeline)):
        raise ValueError('owned prefix release requires its byte-bounded copy pipeline')
    if type(pipeline) is not bool or (pipeline and (not wide_receive or config.get('native_owned_copy_workers') != 16 or physical != 8)):
        raise ValueError('owned copy pipeline requires the exact DP16 eight-reader profile')
    if (exact['native_verification_workers'] + 1) * (max((m.payload_nbytes for m in metas)) + 4096) + (2 * max((m.payload_nbytes for m in metas)) if pipeline else 0) > exact['native_payload_prefetch_bytes']:
        raise ValueError('complete epoch row exceeds actual verification prefetch budget')
    commands = []
    source_tokens = []
    for producer in range(6):
        rows = metas[producer::6]
        rows_commands = []
        for offset in range(0, len(rows), physical):
            chunk = rows[offset:offset + physical]
            tokens = sum((m.token_count for m in chunk))
            if tokens > 131072:
                raise ValueError('actual source batch exceeds 128K before device startup')
            command = build_physical_capture_command(run_id=controls.run_id, generation=controls.generation, producer_index=producer, producer_count=6, step_index=batch.start_cursor // 6 + offset, sample_count=len(chunk))
            if command['sequence_nos'] != [m.sequence_no for m in chunk]:
                raise ValueError('physical source cuts differ from frozen occurrences')
            rows_commands.append(command)
            source_tokens.append(tokens)
        commands.append(rows_commands)
    waves = [sum((m.payload_nbytes for m in metas[i:i + 6 * physical])) for i in range(0, len(metas), 6 * physical)]
    if any((n > 48 * 1024 ** 3 for n in waves)) or any((sum(waves[i:i + 2]) > 48 * 1024 ** 3 for i in range(len(waves) - 1))):
        raise ValueError('two complete source waves exceed the shared ready quota')
    plans = [StreamingOwnedSWARows(metas, target_hidden_size=2560, target_length=131072, max_bytes_per_row=4 * 1024 ** 3, rank=rank, world_size=16, max_rows=32, max_samples_per_row=config['max_samples_per_row'], chunk_cap=profile.swa_chunk_cap, seed=42 + 128 * batch.start_cursor) for rank in range(16)]
    if any((v.row_plan.sample_count != len(metas) or v.planned_frame_upper_bound_bytes > 16 * 1024 ** 3 for v in plans)):
        raise ValueError('actual DP16 packed frame exceeds rank-owned budget')
    return dict(schema_version=1, step_index=step_index, start=batch.start_cursor, end=batch.end_cursor, original_tokens=sum((m.token_count for m in metas)), packed=deepcopy(packed), control_sha256=controls.control_sha256(step_index), source_commands=commands, max_eight_child_batch_tokens=max(source_tokens), wave_payload_bytes=waves, allow_incomplete_producer_round=bool(len(metas) % 6), native_slot_bytes=slot, rank_frame_upper_bounds=[v.planned_frame_upper_bound_bytes for v in plans], rank_owned_label_slots=[v.planned_owned_label_slots for v in plans], epoch_plan_sha256=controls.plan.plan_sha256)
from collections import OrderedDict
from verl_speco.trainer.feature_pipeline import packed_control_sha256

class EpochLaunchInputs:
    """Read-only preparation; optimizer and durable commit remain external gates.

    Never supplies an iterator over the full corpus, a full inline control vector,
    or an optimizer cursor. Every on-disk control is immutable and belongs to the
    admitted run/generation. Metadata previews cannot claim completed tokens.
    """

    def __init__(self, config, controls, *, work_dir, outputs_root):
        if not isinstance(controls, EpochNativeBatches):
            raise ValueError('long launch needs the actual admitted epoch controls')
        root = Path(outputs_root).resolve()
        work = Path(work_dir)
        if not root.is_absolute() or not work.is_absolute() or (not work.resolve().is_relative_to(root)) or (work.name != 'speco-native-wave-' + controls.run_id + '-g' + str(controls.generation)):
            raise ValueError('epoch controls require the exact owned generation directory')
        self.config = deepcopy(config)
        self.controls = controls
        self.work = work
        self.cache = OrderedDict()
        self.failed = self.closed = False
        self.prepare(0)

    def prepare(self, step):
        if self.failed or self.closed:
            raise RuntimeError('long launch inputs are closed or failed')
        try:
            self.controls.batch(step)
            if step not in self.cache:
                self.cache[step] = prepare_epoch_step(self.config, self.controls, step)
                while len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(step)
            value = deepcopy(self.cache[step])
            payload = value['packed']
            self.work.mkdir(parents=True, exist_ok=True)
            path = self.work / f'packed-samples-step-{step:04d}.json'
            try:
                existing = read_published_json(path)
            except FileNotFoundError:
                publish_json(path, payload)
            else:
                if packed_control_sha256(existing) != value['control_sha256']:
                    raise RuntimeError('published epoch control changed after frozen preparation')
            value['control_path'] = str(path)
            value['committed_original_tokens'] = 0
            value['optimizer_started'] = False
            return value
        except BaseException:
            self.cache.clear()
            self.failed = True
            self.controls.close()
            raise

    def sample(self, index):
        if type(index) is not int or not 0 <= index < self.controls.plan.total_samples:
            raise ValueError('epoch sample index is outside the frozen occurrence plan')
        step, offset = divmod(index, self.controls.plan.samples_per_step)
        return self.prepare(step)['packed']['samples'][offset]

    def summary(self):
        if self.failed or self.closed:
            raise RuntimeError('long launch inputs are closed or failed')
        plan = self.controls.plan
        return dict(schema_version=1, kind='FROZEN_EPOCH_INPUT_PREPARATION', run_id=self.controls.run_id, generation=self.controls.generation, plan_sha256=plan.plan_sha256, epochs=plan.epochs, total_samples=plan.total_samples, total_steps=plan.total_steps, planned_original_tokens=plan.total_original_tokens, max_cached_steps=2, control_cache_steps=list(self.cache), committed_original_tokens=0, optimizer_started=False, token_payloads_read_by_launcher=0)

    def __iter__(self):
        raise RuntimeError('long launcher must prepare bounded steps, never iterate the complete corpus')

    def close(self):
        self.cache.clear()
        self.controls.close()
        self.closed = True
from concurrent.futures import ThreadPoolExecutor
import shlex
import subprocess
from verl_speco.producer.replay import epoch_replay_cursor_factory
from verl_speco.standalone_tq_training_launcher import _stable_path_identity
from verl_speco.runtime.source_profiles import native_source_incarnation
from verl_speco.runtime.services import validate_variable_batch_cursor

class epoch_launcher_inputs_StepView:

    def __init__(self, runtime, field):
        self.runtime, self.field = (runtime, field)

    def __getitem__(self, step):
        return self.runtime.prepare(step)[self.field]

    def __iter__(self):
        raise RuntimeError('epoch launcher cannot enumerate the complete corpus')

class EpochLauncherInputs:

    def __init__(self, config, *, execution, work, outputs_root, launch_binding):
        if launch_binding is None:
            raise ValueError('epoch lifecycle needs its actual resolved role plan')
        self.config, self.execution, self.work = (config, execution, Path(work))
        env = self.environment()
        model = config['model_overlay']
        self.source = epoch_replay_cursor_factory(env, outputs_root=outputs_root, producer_count=6, max_sequence_length=config['max_sequence_length'], provenance=dict(target_model_id=model, target_model_revision=_stable_path_identity('target', model), tokenizer_fingerprint=_stable_path_identity('tokenizer', model)))
        self.identity = self.source.input_fingerprints(model_overlay=model, target_model_dir=config['target_model_dir'], model_content_audit=config['model_content_audit'])
        controls = EpochNativeBatches(self.source.catalog, run_id=config['run_id'], generation=execution['generation'], input_fingerprints=self.identity, producer_incarnations=tuple((native_source_incarnation(launch_binding, i) for i in range(6))), max_sequence_length=config['max_sequence_length'])
        self.inputs = EpochLaunchInputs(config, controls, work_dir=work, outputs_root=outputs_root)
        self.epoch_plan = self.bind_draft_plan(env)
        self.packed_batches = epoch_launcher_inputs_StepView(self, 'packed')
        self.metadata_paths = epoch_launcher_inputs_StepView(self, 'control_path')
        self.next_step = execution['start_step']
        self.committed_original_tokens = 0
        self.committed_samples = 0
        self.audit_steps = 0
        self.closed = False

    def environment(self):
        c = self.config
        return dict(SPECO_EPOCH_REPLAY_SPEC_PATH=c['epoch_replay_spec_path'], SPECO_EPOCH_REPLAY_SPEC_SHA256=c['epoch_replay_spec_sha256'], SPECO_EPOCH_REPLAY_PLAN_SHA256=c['epoch_replay_plan_sha256'], SPECO_SEQUENCE_REPLAY_INPUT=c['sequence_file'], SPECO_PROMPT_DATASET=c['sequence_file'], SPECO_TARGET_MODEL_OVERLAY=c['model_overlay'], SPECO_TARGET_MODEL_DIR=c['target_model_dir'], SPECO_TARGET_MAX_SEQUENCE_LENGTH=str(c['max_sequence_length']), SPECO_CAPTURE_LONG_CONTEXT=str(c['max_sequence_length']), SPECO_DRAFT_VARIABLE_BATCH='1', SPECO_DRAFT_WORLD_SIZE='16')

    def bind_draft_plan(self, env):
        keys = (*EPOCH_FIELDS, 'SPECO_TARGET_MODEL_OVERLAY', 'SPECO_TARGET_MODEL_DIR', 'SPECO_PROMPT_DATASET', 'SPECO_TARGET_MAX_SEQUENCE_LENGTH')
        value = EpochDraftPlan(self.source.plan, tuple(((key, env.get(key, '')) for key in keys)))
        validate_epoch_plan_capacity(value)
        return value

    def prepare(self, step):
        if self.closed or type(step) is not int or (not self.execution['start_step'] <= step < self.execution['end_step']):
            raise ValueError('epoch lifecycle step outside the admitted stage')
        return self.inputs.prepare(step)

    def producer_end(self, producer):
        if type(producer) is not int or not 0 <= producer < 6:
            raise ValueError('epoch producer ordinal is not canonical')
        end = min(self.execution['end_step'] * self.source.plan.samples_per_step, self.source.plan.total_samples)
        return (end + 5 - producer) // 6

    def next_source_commands(self, step, *, work, metadata_depth):
        if type(metadata_depth) is not int or not 1 <= metadata_depth <= 16 or Path(work) != self.work:
            raise ValueError('epoch source horizon differs from the owned work directory')
        if step + 1 >= self.execution['end_step']:
            return []
        future = self.prepare(step + 1)
        commands = []
        for ordinal, rows in enumerate(future['source_commands']):
            for payload in rows[:metadata_depth]:
                path = self.work / f'producer-{ordinal}' / f"step-{payload['step_index']:04d}.request.json"
                try:
                    previous = read_published_json(path)
                except FileNotFoundError:
                    pass
                else:
                    if previous != payload:
                        raise RuntimeError('future epoch source command identity changed')
                commands.append((path, payload))
        return commands

    def record_commit(self, step, receipt, acknowledgement):
        """Credit exact originals only after a SHA-bound real durable receipt."""
        if type(step) is not int or step != self.next_step:
            raise ValueError('epoch durable commits must advance once in order')
        batch = self.prepare(step)
        if acknowledgement.get('state') != 'STEP_COMMITTED' or acknowledgement.get('run_id') != self.config['run_id'] or acknowledgement.get('generation') != self.execution['generation'] or (acknowledgement.get('step_index') != step) or (not isinstance(acknowledgement.get('receipt_sha256'), str)) or (len(acknowledgement['receipt_sha256']) != 64) or (receipt.get('packed_sample_keys') != [s['key'] for s in batch['packed']['samples']]):
            raise ValueError('epoch receipt differs from the committed batch acknowledgement')
        from verl_speco.transport.streaming import decode_stream_snapshot
        stream = decode_stream_snapshot(receipt.get('stream_snapshot'))
        keys = [s['key'] for s in batch['packed']['samples']]
        expected_step = 'dspark-step-' + hashlib.sha256('\n'.join(keys).encode()).hexdigest()[:20]
        transactions = sorted(('transaction-' + hashlib.sha256(key.encode()).hexdigest()[:20] for key in keys))
        history = stream.get('committed_optimizer_steps', {}) if isinstance(stream, dict) else {}
        if not isinstance(stream, dict) or stream.get('committed_cursor') != batch['end'] or stream.get('stream_scope') != self.config['run_id'] or (not isinstance(history, dict)) or (len(history) != step + 1) or (sum((len(ids) for ids in history.values())) != batch['end']) or (receipt.get('step_id') != expected_step) or (history.get(expected_step) != transactions) or (receipt.get('sample_index') != batch['start']) or (receipt.get('run_id') != self.config['run_id']) or (receipt.get('generation') != self.execution['generation']) or (receipt.get('total_steps') != self.source.plan.total_steps) or (receipt.get('draft_world_size') != 16) or (receipt.get('step_id') not in stream.get('committed_optimizer_steps', {})):
            raise ValueError('epoch durable stream or true tail cursor differs')
        from verl_speco.runtime.checkpoint_policy import checkpoint_interval, checkpoint_due, require_live_receipt
        interval = checkpoint_interval(self.config)
        if interval > 1:
            saved = checkpoint_due(step, plan=self.source.plan, interval=interval, stage_end=self.execution['end_step'])
            if receipt.get('checkpoint_saved') is not saved:
                raise ValueError('receipt checkpoint decision differs from the frozen schedule')
            if not saved:
                require_live_receipt(dict(receipt, stream_snapshot=stream))
        publish_json(self.work / f'committed-step-{step:04d}.json', dict(schema_version=1, run_id=self.config['run_id'], generation=self.execution['generation'], step_index=step, start=batch['start'], end=batch['end'], original_tokens=batch['original_tokens'], receipt_sha256=acknowledgement['receipt_sha256'], packed_control_sha256=packed_control_sha256(batch['packed']), candidate=receipt['candidate'], control_snapshot=receipt['control_snapshot'], state='STEP_COMMITTED', **{'checkpoint_saved': receipt['checkpoint_saved'], 'live_owner_token': receipt['live_owner_token'], 'checkpoint_interval': receipt['checkpoint_interval'], 'draft_mean_acceptance_length': receipt.get('draft_mean_acceptance_length'), 'draft_mean_accepted_tokens': receipt.get('draft_mean_accepted_tokens'), 'draft_accepted_length_sum': receipt.get('draft_accepted_length_sum'), 'draft_scored_block_count': receipt.get('draft_scored_block_count'), 'draft_acceptance_scope': receipt.get('draft_acceptance_scope')} if interval > 1 else {}))
        if interval > 1 and receipt['checkpoint_saved']:
            record = read_published_json(self.work / f'committed-step-{step:04d}.json')
            publish_json(self.work / 'latest-durable-checkpoint.json', record)
        self.committed_original_tokens += batch['original_tokens']
        self.committed_samples += batch['end'] - batch['start']
        self.next_step += 1

    def retire_committed_checkpoints(self, step):
        """Apply opt-in keep-two retention after the current durable/source audits."""
        if 'epoch_checkpoint_keep_last' not in self.config:
            return None
        if type(self.config['epoch_checkpoint_keep_last']) is not int or self.config['epoch_checkpoint_keep_last'] != 2:
            raise ValueError('epoch checkpoint retention requires exactly two durable successors')
        start = self.execution['start_step']
        if type(step) is not int or not start <= step < self.execution['end_step'] or self.next_step != step + 1 or (self.audit_steps != step - start + 1):
            raise ValueError('checkpoint retention must follow every real durable commit and source audit')
        from verl_speco.runtime.checkpoint_policy import checkpoint_interval, checkpoint_due
        sparse = checkpoint_interval(self.config) > 1
        if sparse:
            saved = [i for i in range(start, step + 1) if checkpoint_due(i, plan=self.source.plan, interval=checkpoint_interval(self.config), stage_end=self.execution['end_step'])]
            if step not in saved or len(saved) < 3:
                return None
            indices = saved[-3:]
        else:
            if step < start + 2:
                return None
            indices = range(step - 2, step + 1)
        records = [read_published_json(self.work / f'committed-step-{i:04d}.json') for i in indices]
        return apply_retirement(records, outputs_root=self.work.parent, work_dir=self.work, **{'allow_sparse': True} if sparse else {})

    def audit_capture_controls(self, step, *, ssh, python_bin):
        batch = self.prepare(step)
        samples = batch['packed']['samples']
        run = self.config['run_id']
        generation = self.execution['generation']
        remote_code = 'from pathlib import Path\nimport json,sys,hashlib\nroot=Path(sys.argv[1]);indices=json.loads(sys.argv[2])\nassert isinstance(indices,list) and 0<len(indices)<=160\nassert all(type(i) is int and i>=0 for i in indices) and indices==sorted(set(indices))\n# The 960-occurrence profile has 160 records per source. Keep each read\n# block bounded to the original 80, within one SSH connection per source.\nfor offset in range(0,len(indices),80):\n for index in indices[offset:offset+80]:\n  raw=(root/f"capture-{index:04d}.json").read_bytes();assert 0<len(raw)<=1024*1024\n  print(json.dumps(dict(index=index,capture=json.loads(raw),sha256=hashlib.sha256(raw).hexdigest()),separators=(",",":")))\n'

        def fetch(ordinal):
            indices = list(range(batch['start'] + ordinal, batch['end'], 6))
            if not indices:
                return []
            directory = Path('/tmp') / f'{run}-g{generation}-producer-{ordinal}'
            command = shlex.join([python_bin, '-c', remote_code, str(directory), json.dumps(indices)])
            result = subprocess.run(ssh + [self.config['sources'][ordinal]['host'], command], capture_output=True, text=True, check=True, timeout=90)
            if len(result.stdout.encode()) > 32 * 1024 ** 2:
                raise ValueError('epoch source evidence exceeds bounded batch reply')
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            if [row['index'] for row in rows] != indices:
                raise ValueError('epoch source evidence omitted or reordered complete occurrences')
            for row in rows:
                expected = samples[row['index'] - batch['start']]
                if row['capture'].get('metadata') != expected['metadata'] or row['capture'].get('key') != expected['key']:
                    raise ValueError('epoch source evidence differs from its full frozen row')
            return [dict(index=row['index'], source_control_sha256=row['sha256']) for row in rows]
        with ThreadPoolExecutor(max_workers=6) as pool:
            records = [record for rows in pool.map(fetch, range(6)) for record in rows]
        records.sort(key=lambda row: row['index'])
        if [row['index'] for row in records] != list(range(batch['start'], batch['end'])):
            raise ValueError('epoch evidence does not cover exactly the committed batch')
        publish_json(self.work / f'capture-evidence-step-{step:04d}.json', dict(schema_version=1, run_id=run, generation=generation, step_index=step, records=records, start=batch['start'], end=batch['end'], all_full_source_metadata_verified=True))
        self.audit_steps += 1

    def result(self, *, control_status, wallclock_seconds):
        expected_steps = self.execution['end_step'] - self.execution['start_step']
        if self.next_step != self.execution['end_step'] or self.audit_steps != expected_steps:
            raise RuntimeError('epoch stage cannot pass before every durable batch and source audit')
        return dict(run_id=self.config['run_id'], generation=self.execution['generation'], updates_this_generation=expected_steps, optimizer_steps=self.execution['end_step'], committed_cursor=min(self.execution['end_step'] * self.source.plan.samples_per_step, self.source.plan.total_samples), original_tokens=self.committed_original_tokens, committed_occurrences=self.committed_samples, profile=self.config['draft_training_profile'], block_size=8, num_anchors=256, packing_rows=32, max_samples_per_row=64, source_context=self.config['max_sequence_length'], epoch_plan_sha256=self.source.plan.plan_sha256, wallclock_seconds=wallclock_seconds, control_status=control_status, production_sla_pass=False, formal_three_epochs_complete=self.next_step == self.source.plan.total_steps, per_dp_128k_claimed=False, metadata_cache_steps=len(self.inputs.cache))

    def close(self):
        self.inputs.close()
        self.closed = True
from verl_speco.runtime.services import publication_gate_from_environment
from verl_speco.runtime.planning import ResolvedLaunchPlan

def source_publication_gate(environment, *, outputs_root, source_factory, input_fingerprints, resolved_plan=None, fetch_parent_permit=None, native_binding=None):
    configured = any((environment.get(k, '') for k in EPOCH_FIELDS))
    if not configured:
        if source_factory is not None:
            raise ValueError('epoch factory cannot fall back to legacy gate')
        return publication_gate_from_environment(environment, outputs_root=outputs_root, fetch_parent_permit=fetch_parent_permit, native_binding=native_binding, frozen_input_fingerprints=input_fingerprints)
    if source_factory is None or not isinstance(resolved_plan, ResolvedLaunchPlan) or environment.get('SPECO_PREFILL_STEP_LOOKAHEAD') != '1' or any((not environment.get(k, '') for k in EPOCH_FIELDS)):
        raise ValueError('epoch source needs its actual admitted factory and resolved launch owner')
    if str(source_factory.spec_path) != environment[EPOCH_FIELDS[0]] or source_factory.spec_file_sha256 != environment[EPOCH_FIELDS[1]] or source_factory.plan.plan_sha256 != environment[EPOCH_FIELDS[2]] or (len(resolved_plan.producer_groups) != source_factory.plan.producer_count) or (resolved_plan.run_id != environment.get('SPECO_RUN_ID')) or (str(resolved_plan.generation) != environment.get('SPECO_GENERATION')):
        raise ValueError('epoch source catalog, resolved owner or exact pins differ')
    producer = environment.get('SPECO_PRODUCER_ID', '')
    count = source_factory.plan.producer_count
    if producer not in [f'producer-{i}' for i in range(count)]:
        raise ValueError('epoch source ordinal is not canonical')
    incarnations = tuple((native_source_incarnation(resolved_plan, i) for i in range(count)))
    if environment.get('SPECO_PRODUCER_INCARNATION') != incarnations[int(producer.removeprefix('producer-'))]:
        raise ValueError('epoch source incarnation differs from its resolved role')
    keys = (*EPOCH_FIELDS, 'SPECO_TARGET_MODEL_OVERLAY', 'SPECO_TARGET_MODEL_DIR', 'SPECO_PROMPT_DATASET', 'SPECO_TARGET_MAX_SEQUENCE_LENGTH')
    binding = EpochDraftPlan(source_factory.plan, tuple(((k, environment.get(k, '')) for k in keys)))
    validate_epoch_command(environment, epoch_plan=binding)
    batches = EpochNativeBatches(source_factory.catalog, run_id=resolved_plan.run_id, generation=resolved_plan.generation, input_fingerprints=input_fingerprints, producer_incarnations=incarnations, max_sequence_length=int(environment['SPECO_TARGET_MAX_SEQUENCE_LENGTH']))
    try:
        gate = publication_gate_from_environment(environment, outputs_root=outputs_root, fetch_parent_permit=fetch_parent_permit, epoch_plan=binding, batch_controls=batches, native_binding=native_binding, frozen_input_fingerprints=input_fingerprints)
        if gate is None:
            raise ValueError('epoch source gate unexpectedly disabled')
        return gate
    except BaseException:
        batches.close()
        raise
from verl_speco.producer.replay import checked_json, MAX_INDEX_BYTES
from verl_speco.producer.replay import FrozenReplayEpochPlan
EPOCH_CONFIG_FIELDS = frozenset({'epoch_replay_spec_path', 'epoch_replay_spec_sha256', 'epoch_replay_plan_sha256', 'source_long_context', 'draft_training_profile', 'epoch_listener_incarnation'})
EPOCH_OPTIONAL_FIELDS = frozenset({'epoch_checkpoint_keep_last', 'epoch_checkpoint_interval'})

def validate_epoch_standalone_config(config, *, legacy_validator, outputs_root):
    if not isinstance(config, dict) or not EPOCH_CONFIG_FIELDS <= config.keys():
        raise ValueError('long native launch needs complete explicit epoch bindings')
    from verl_speco.runtime.checkpoint_policy import checkpoint_interval
    checkpoint_interval(config)
    context = config.get('max_sequence_length')
    if type(context) is not int or context not in (49152, 65536):
        raise ValueError('long epoch must declare48K or64K engine context')
    if 'epoch_checkpoint_keep_last' in config and (type(config['epoch_checkpoint_keep_last']) is not int or config['epoch_checkpoint_keep_last'] != 2):
        raise ValueError('epoch checkpoint retention requires exactly two durable successors')
    samples = config.get('samples_per_step')
    if type(samples) is not int or samples not in (480, 960):
        raise ValueError('long epoch admits only480 or explicit packed960 samples')
    physical = config.get('source_physical_batch_size')
    if type(physical) is not int or physical not in (8, 16):
        raise ValueError('invalid full epoch physical source width')
    from verl_speco.producer.capture import config_slot_profile
    source_slot_profile = config_slot_profile(config)
    banks = config.get('source_native_buffer_batches')
    tail = config.get('source_tail_lookahead', False)
    if type(banks) is not int or banks not in (2, 3, 4, 10) or (banks in (3, 4, 10) and context != 49152) or (type(tail) is not bool) or (tail and banks not in (3, 4, 10)) or (physical == 16 and (banks != 2 or context != 49152)):
        raise ValueError('long epoch overlap requires bound48K three-bank profile')
    wide_receive = banks in (4, 10) or (physical == 16 and config.get('native_receive_workers') == 8)
    exact = dict(source_long_context=True, draft_training_profile='dmtp-92b-block8-swa-confidence-128k', max_sequence_length=context, source_tp=8, source_native_buffer_batches=banks, source_capture_owner_ranks=8, source_physical_batch_size=physical, source_prefill_batch_tokens=131072, source_graph_mode='FULL_DECODE_ONLY', source_early_ready=True, source_step_lookahead=True, resolved_launch_plan=True, source_full_batch_prefill=True, source_mhc_sequence_parallel=True, source_mhc_prefill_fusion=True, packing_rows=32, max_samples_per_row=64, samples_per_step=samples, global_capacity_bytes=(96 if source_slot_profile else 24 * banks) * 1024 ** 3, native_verification_workers=8 if wide_receive else 4, native_receive_workers=8 if wide_receive else 4, native_payload_prefetch_bytes=(16 if wide_receive else 8) * 1024 ** 3)
    if any((type(config.get(k)) is not type(v) or config.get(k) != v for k, v in exact.items())) or len(config.get('sources', [])) != 6 or 'second_trainer' in config or (type(config.get('source_tail_lookahead', False)) is not bool) or (config.get('source_memory_utilization') != 0.85) or (type(config['epoch_listener_incarnation']) is not int) or (not 1 <= config['epoch_listener_incarnation'] <= 123):
        raise ValueError('long native launch differs from the bound DP16 complete64K profile')
    original = deepcopy(config)
    projected = {k: v for k, v in config.items() if k not in EPOCH_CONFIG_FIELDS | EPOCH_OPTIONAL_FIELDS}
    projected.update(steps=1, max_sequence_length=16384)
    projected.pop('execution', None)
    normalized = legacy_validator(projected)
    for key in EPOCH_CONFIG_FIELDS | EPOCH_OPTIONAL_FIELDS | {'steps', 'max_sequence_length', 'execution'}:
        if key in original:
            normalized[key] = original[key]
    root = Path(outputs_root).resolve()
    spec = checked_json(config['epoch_replay_spec_path'], config['epoch_replay_spec_sha256'], outputs_root=root, max_bytes=65536)
    from verl_speco.producer.batching import epoch_spec_balance_policy
    balance_policy = epoch_spec_balance_policy(spec)
    from verl_speco.producer.batching import PACKED_BALANCE_POLICY, PACKED_EIGHT_BALANCE_POLICY
    if samples == 960 and balance_policy not in (PACKED_BALANCE_POLICY, PACKED_EIGHT_BALANCE_POLICY):
        raise ValueError('960 optimizer batch requires explicit packed-order plan')
    from verl_speco.producer.batching import balance_physical_batch_size
    if physical != balance_physical_batch_size(balance_policy):
        raise ValueError('physical capture width differs from frozen epoch balance policy')
    required = {'schema_version', 'index_path', 'index_file_sha256', 'index_sha256', 'dataset_sha256', 'plan_sha256', 'epochs', 'seed', 'samples_per_step', 'producer_count', 'max_sequence_length'}
    if type(spec.get('schema_version')) is not int or any((type(spec.get(k)) is not int or spec[k] != v for k, v in {'epochs': 3, 'seed': 42, 'samples_per_step': samples, 'producer_count': 6, 'max_sequence_length': context}.items())) or spec['plan_sha256'] != config['epoch_replay_plan_sha256'] or (Path(config['sequence_file']).resolve() != Path(spec['index_path']).resolve()):
        raise ValueError('long launch specification or source file differs from its frozen plan')
    index = checked_json(spec['index_path'], spec['index_file_sha256'], outputs_root=root, max_bytes=MAX_INDEX_BYTES)
    if index.get('index_sha256') != spec['index_sha256'] or index.get('dataset_sha256') != spec['dataset_sha256']:
        raise ValueError('long launch index identity differs from its frozen specification')
    entries = index['entries']
    if any((e['response_tokens'] < 8 or e['prompt_tokens'] < 1 or e['original_tokens'] > context for e in entries)):
        raise ValueError('long launch needs the audited block8 eligibility snapshot')
    rows = [{k: e[k] for k in ('id', 'sequence_fingerprint', 'original_tokens')} for e in entries]
    plan = FrozenReplayEpochPlan(rows, dataset_sha256=spec['dataset_sha256'], epochs=3, seed=42, samples_per_step=samples, producer_count=6, prefill_balance_policy=balance_policy)
    if plan.plan_sha256 != spec['plan_sha256'] or type(config['steps']) is not int or config['steps'] != plan.total_steps:
        raise ValueError('long launch step count or plan digest differs from all epoch occurrences')
    execution = native_execution(normalized)
    if config['epoch_listener_incarnation'] < execution['generation']:
        raise ValueError('long launch listener incarnation precedes its generation')
    return normalized

def validate_epoch_or_native_config(config, *, legacy_validator, outputs_root):
    """Select only an explicit frozen epoch binding; reject partial opt-ins."""
    if not isinstance(config, dict):
        raise ValueError('native launch configuration must be an object')
    if (EPOCH_CONFIG_FIELDS | EPOCH_OPTIONAL_FIELDS).intersection(config):
        return validate_epoch_standalone_config(config, legacy_validator=legacy_validator, outputs_root=outputs_root)
    return legacy_validator(config)

def load_generation_parent(path, *, run_id, generation, sample_index, samples, total_steps, logical_checkpoint_fingerprint, variable_batch=False, step_index=None, epoch_plan=None):
    path = Path(path)
    from verl_speco.runtime.artifacts import read_published_json
    receipt = read_published_json(path)
    from verl_speco.transport.streaming import decode_stream_snapshot
    stream = decode_stream_snapshot(receipt.get('stream_snapshot', {}))
    committed = stream.get('committed_optimizer_steps', {})
    old_generation = receipt.get('generation')
    previous_keys = receipt.get('packed_sample_keys')
    weights_sha = receipt.get('draft_weights_sha256')
    if type(variable_batch) is not bool:
        raise ValueError('invalid generation parent batch policy')
    if epoch_plan is not None and (not variable_batch):
        raise ValueError('epoch generation parent requires explicit variable batch policy')
    if variable_batch:
        from verl_speco.runtime.services import validate_variable_batch_cursor
        validate_variable_batch_cursor(run_id=run_id, generation=generation, step_index=step_index, start=sample_index, batch_size=samples, total_steps=total_steps, world_size=receipt.get('draft_world_size'), previous_receipt=receipt, generation_handoff=True, epoch_plan=epoch_plan)
    elif step_index is not None:
        raise ValueError('explicit step index requires variable batch policy')
    fixed_prefix_invalid = not variable_batch and (sample_index < samples or sample_index % samples or receipt.get('sample_index') != sample_index - samples or (not isinstance(previous_keys, list)) or (len(previous_keys) != samples) or (len(set(previous_keys)) != samples) or (not isinstance(committed, dict)) or (len(committed) != sample_index // samples))
    if type(old_generation) is not int or old_generation < 1 or generation != old_generation + 1 or fixed_prefix_invalid or (receipt.get('schema_version') != 1) or (receipt.get('run_id') != run_id) or (receipt.get('total_steps') != total_steps) or (stream.get('stream_scope') != run_id) or (stream.get('committed_cursor') != sample_index) or (receipt.get('step_id') not in committed) or (not isinstance(weights_sha, str)) or (len(weights_sha) != 64) or any((c not in '0123456789abcdef' for c in weights_sha)):
        raise ValueError('generation parent is not the latest committed stream prefix')
    candidate = receipt.get('candidate', {})
    from verl_speco.runtime.checkpoint_policy import is_live_candidate
    if is_live_candidate(candidate):
        raise ValueError('cold recovery cannot restore progress-only updates; choose the latest full checkpoint')
    root = Path(candidate.get('uri', ''))
    manifest_path = root / 'manifest.json'
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    if manifest.get('schema_version') != 1 or hashlib.sha256(manifest_payload).hexdigest() != candidate.get('checksum') or candidate.get('checkpoint_generation') != old_generation or (manifest.get('generation') != old_generation) or (manifest.get('world_size') != receipt.get('draft_world_size')) or (manifest.get('logical_checkpoint_fingerprint') != logical_checkpoint_fingerprint) or (not logical_checkpoint_fingerprint):
        raise ValueError('generation parent checkpoint identity differs')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('generation parent checkpoint has no complete files')
    for name, checksum in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('generation parent checkpoint file escapes its root')
        with (root / relative).open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != checksum:
                raise ValueError('generation parent checkpoint file checksum differs')
    control = receipt.get('control_snapshot', {})
    if control.get('storage'):
        from verl_speco.runtime.services import read_shared_control_snapshot
        snapshot = read_shared_control_snapshot(control, run_id=run_id, generation=old_generation)
    else:
        payload = Path(control.get('path', '')).read_bytes()
        snapshot = json.loads(payload)
        if hashlib.sha256(payload).hexdigest() != control.get('sha256') or snapshot.get('run_id') != run_id or snapshot.get('generation') != old_generation:
            raise ValueError('generation parent control snapshot differs')
    from verl_speco.transport.control import RayReadyControl
    owner = RayReadyControl(run_id=run_id, generation=old_generation, partition=snapshot['partition'], capacity_bytes=snapshot['capacity_bytes'])
    owner.restore_quiescent(snapshot)
    return (receipt, manifest, snapshot)

def require_stopped_committed_parent(*, receipt, snapshot, status, probes, samples):
    old_generation = receipt['generation']
    run_id = receipt['run_id']
    if not probes or any((probe.get('active_trainer_pids') for probe in probes)):
        raise RuntimeError('old-generation role processes are still alive or unprobed')
    expected_committed = snapshot['leases']['committed_steps']
    from verl_speco.runtime.checkpoint_policy import checkpoint_interval, is_live_candidate, read_live_update
    sparse = checkpoint_interval({'epoch_checkpoint_interval': receipt.get('checkpoint_interval', 1)}) > 1
    if sparse and (receipt.get('checkpoint_saved') is not True or is_live_candidate(receipt.get('candidate', {}))):
        raise RuntimeError('cold recovery must select an actual full checkpoint')
    if status.get('run_id') != run_id or status.get('generation') != old_generation or status.get('used_bytes') or status.get('retained_items') or status.get('pending_reservations') or status.get('reclaim_pending') or (status.get('router', {}).get('ready_records') != 0):
        raise RuntimeError('old-generation owner has outstanding data or a later commit')
    found = False
    prepared = set()
    later = {}
    from verl_speco.transport.streaming import decode_stream_snapshot
    stream = decode_stream_snapshot(receipt['stream_snapshot'])
    committed = stream['committed_optimizer_steps']
    for probe in probes:
        if not isinstance(probe.get('journals'), list):
            raise ValueError('old-generation journal probe is incomplete')
        for row in probe['journals']:
            tx = row.get('transaction', {})
            if tx.get('generation') != old_generation:
                continue
            if tx.get('stream_scope') != run_id:
                raise ValueError('old-generation journal belongs to another stream')
            step = tx.get('optimizer_step_id')
            if row.get('state') == 'STEP_COMMITTED':
                if step not in committed:
                    candidate = row.get('candidate_checkpoint', {})
                    if not sparse or not is_live_candidate(candidate):
                        raise RuntimeError('a later old-generation update has already committed a full checkpoint')
                    live = read_live_update(candidate, owner_token=receipt.get('live_owner_token'))
                    if live['run_id'] != run_id or live['generation'] != old_generation or live['step_id'] != step or (not isinstance(tx.get('lease_ids'), list)) or (len(tx['lease_ids']) != len(tx.get('transaction_ids', []))) or (not 0 < len(tx['lease_ids']) <= samples):
                        raise ValueError('later live progress is not a complete owned transaction')
                    if step in later and later[step] != (live, tx):
                        raise ValueError('later live progress journal copies differ')
                    later[step] = (live, tx)
                    continue
                if sorted(tx.get('transaction_ids', [])) != sorted(committed[step]):
                    raise ValueError('parent journal and stream transactions differ')
                if step == receipt['step_id']:
                    if row.get('candidate_checkpoint') != receipt['candidate']:
                        raise ValueError('parent journal and checkpoint candidate differ')
                    found = True
            elif row.get('state') == 'STEP_PREPARED':
                ids = tx.get('transaction_ids')
                if not row.get('optimizer_started') or not isinstance(ids, list) or len(ids) != samples or (len(set(ids)) != samples) or (step in committed):
                    raise ValueError('uncertain old-generation transaction differs')
                prepared.add(step)
            else:
                raise RuntimeError('old-generation journal state is not recoverable')
    if later:
        manifest = json.loads((Path(receipt['candidate']['uri']) / 'manifest.json').read_bytes())
        token = receipt.get('live_owner_token')
        if not isinstance(token, str) or len(token) != 64 or any((c not in '0123456789abcdef' for c in token)):
            raise ValueError('sparse parent has no complete live owner identity')
        for live, tx in later.values():
            if any((live[k] != manifest.get(k) for k in ('topology_fingerprint', 'model_fingerprint', 'logical_checkpoint_fingerprint'))):
                raise ValueError('live rollback topology/model/logical identity differs')
    cursor = stream['committed_cursor']
    index = len(committed)
    ordered = sorted(later.values(), key=lambda value: value[0]['step_index'])
    for live, tx in ordered:
        cursor += len(tx['lease_ids'])
        if live['step_index'] != index or live['cursor'] != cursor:
            raise ValueError('live rollback has an omitted step or cursor')
        index += 1
    if status.get('leases', {}).get('committed_steps') != len(expected_committed) + len(later) or status.get('leases', {}).get('leases_committed') != sum((len(step['lease_ids']) for step in expected_committed.values())) + sum((len(tx['lease_ids']) for live, tx in later.values())):
        raise RuntimeError('old-generation owner has outstanding data or a later commit')
    if not found or len(prepared) > 1 or prepared.intersection(later):
        raise RuntimeError('parent commit is unproven or multiple updates are uncertain')

def probe_generation_parent(*, receipt, snapshot, role_hosts, samples, python_bin, ray_address):
    import subprocess, shlex, sys, ray
    from verl_speco.transport.control import actor_name_for_run
    root = Path(__file__).resolve().parents[2]
    argv = ['-c', REMOTE_PROBE, receipt['journal_dir'], receipt['run_id'], str(receipt['generation']), str(root)]
    probes = []
    for host in role_hosts:
        command = ['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', host, shlex.join([python_bin, *argv])]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=45)
        probes.append(json.loads(result.stdout))
    result = subprocess.run([sys.executable, *argv], capture_output=True, text=True, check=True, timeout=45)
    probes.append(json.loads(result.stdout))
    ray.init(address=ray_address, namespace='transfer_queue', logging_level='ERROR')
    try:
        actor = ray.get_actor(actor_name_for_run(receipt['run_id'], receipt['generation']), namespace='transfer_queue')
        status = ray.get(actor.status.remote())
        require_stopped_committed_parent(receipt=receipt, snapshot=snapshot, status=status, probes=probes, samples=samples)
    finally:
        ray.shutdown()
    print('committed_generation_parent_admitted ' + json.dumps(dict(run_id=receipt['run_id'], old_generation=receipt['generation'], next_generation=receipt['generation'] + 1, committed_cursor=receipt['stream_snapshot']['committed_cursor'])), flush=True)
REMOTE_PROBE = "\nimport json, os, pathlib, sys\nroot, run_id, generation, source_root = sys.argv[1:]\nsys.path.insert(0, source_root)\nfrom verl_speco.runtime.services import owned_session_processes\njournal_root = pathlib.Path(root)\nif journal_root.resolve().is_relative_to(pathlib.Path(source_root).resolve() / 'outputs'):\n    import re\n    from verl_speco.runtime.artifacts import read_published_json, published_json_paths\n    paths = [p for p in published_json_paths(journal_root, 'step-*.json')\n             if re.fullmatch(r'step-[0-9a-f]{64}\\.json', p.name)]\n    if len(paths) > 32:\n        raise ValueError('native shared journal probe exceeds planned step bound')\n    journals = [read_published_json(p, max_bytes=16*1024**2) for p in paths]\nelse:\n    journals = [json.loads(path.read_text(encoding='utf-8'))\n                for path in sorted(journal_root.glob('step-*.json'))]\nactive = []\nfor entry in pathlib.Path('/proc').iterdir():\n    if not entry.name.isdigit() or int(entry.name) == os.getpid():\n        continue\n    try:\n        raw = (entry / 'environ').read_bytes().split(b'\\0')\n        env = dict(item.split(b'=', 1) for item in raw if b'=' in item)\n        if (env.get(b'SPECO_RUN_ID') == run_id.encode()\n                and env.get(b'SPECO_GENERATION') == generation.encode()):\n            active.append(int(entry.name))\n    except (FileNotFoundError, PermissionError, ProcessLookupError):\n        pass\nactive = sorted(set(active) | {p['pid'] for p in owned_session_processes(run_id, int(generation))})\nprint(json.dumps({'journals': journals, 'active_trainer_pids': active}, sort_keys=True))\n"
from datetime import datetime, timezone

def validate_owned_native_exit_proof(proof, *, run_id, generation):
    if not isinstance(proof, dict) or set(proof) != {'schema_version', 'run_id', 'generation', 'hosts', 'checks'} or type(proof['schema_version']) is not int or (proof['schema_version'] != 1) or (proof['run_id'] != run_id) or (type(proof['generation']) is not int) or (proof['generation'] != generation) or (not isinstance(proof['hosts'], list)) or (not 1 <= len(proof['hosts']) <= 8) or any((not isinstance(host, str) or not host for host in proof['hosts'])) or (len(set(proof['hosts'])) != len(proof['hosts'])) or (not isinstance(proof['checks'], list)) or (len(proof['checks']) != len(proof['hosts'])):
        raise ValueError('failed prefill requires exact owned native role exit proof')
    for host, check in zip(proof['hosts'], proof['checks']):
        if not isinstance(check, dict) or set(check) != {'host', 'run_id', 'generation', 'remaining_owned_processes', 'observed_utc'} or check['host'] != host or (check['run_id'] != run_id) or (type(check['generation']) is not int) or (check['generation'] != generation) or (check['remaining_owned_processes'] != []) or (not isinstance(check['remaining_owned_processes'], list)):
            raise ValueError('failed prefill owned roles have not exited or identity differs')
        try:
            observed = datetime.fromisoformat(check['observed_utc'])
            if observed.tzinfo is None:
                raise ValueError
            age = (datetime.now(timezone.utc) - observed).total_seconds()
        except (TypeError, ValueError):
            raise ValueError('invalid native role exit observation time') from None
        if not -5 <= age <= 120:
            raise ValueError('native role exit observation is stale')
    return proof

def collect_owned_native_exit_proof(*, hosts, run_id, generation, root, python_bin):
    """Read exact run/generation sessions on every configured physical host."""
    hosts = tuple(dict.fromkeys(hosts))
    if not hosts or len(hosts) > 8:
        raise ValueError('native exit proof requires configured physical hosts')
    script = 'import sys,json,datetime\nsys.path.insert(0,sys.argv[1])\nfrom verl_speco.runtime.services import owned_session_processes\nrun,generation=sys.argv[2],int(sys.argv[3])\nprint(json.dumps(dict(run_id=run,generation=generation,remaining_owned_processes=owned_session_processes(run,generation),observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))\n'

    def check(host):
        command = shlex.join([python_bin, '-c', script, str(Path(root).resolve()), run_id, str(generation)])
        result = subprocess.run(['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', host, command], capture_output=True, text=True, check=True, timeout=45)
        if len(result.stdout.encode()) > 65536:
            raise ValueError('native exit observation exceeds metadata budget')
        value = json.loads(result.stdout)
        value['host'] = host
        return value
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        checks = list(pool.map(check, hosts))
    proof = dict(schema_version=1, run_id=run_id, generation=generation, hosts=list(hosts), checks=checks)
    return validate_owned_native_exit_proof(proof, run_id=run_id, generation=generation)

def failed_prefill_samples(samples, *, active_step, samples_per_step, stage_end, step_lookahead):
    """Bound failed work to the active batch and one permitted lookahead batch."""
    from verl_speco.transport.streaming import MAX_NATIVE_BATCH_SAMPLES
    if not isinstance(samples, list) or type(active_step) is not int or type(stage_end) is not int or (not 0 <= active_step < stage_end) or (type(samples_per_step) is not int) or (not 1 <= samples_per_step <= MAX_NATIVE_BATCH_SAMPLES) or (type(step_lookahead) is not bool) or (len(samples) < stage_end * samples_per_step):
        raise ValueError('failed prefill requires a valid frozen active step window')
    end_step = min(stage_end, active_step + 1 + int(step_lookahead))
    return samples[active_step * samples_per_step:end_step * samples_per_step]

def cleanup_failed_native_prefill(owner, *, expected_samples, owned_exit_proof):
    """Actor-serialized cleanup of one frozen failed step, after role fencing."""
    from verl_speco.transport.streaming import MAX_NATIVE_BATCH_SAMPLES
    from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key, parse_v3_ready_tag
    from verl_speco.transport.control import LeaseState
    validate_owned_native_exit_proof(owned_exit_proof, run_id=owner.run_id, generation=owner.generation)
    if not isinstance(expected_samples, list) or not 1 <= len(expected_samples) <= 2 * MAX_NATIVE_BATCH_SAMPLES or len(json.dumps(expected_samples, separators=(',', ':')).encode()) > 16 * 1024 ** 2:
        raise ValueError('failed native cleanup requires bounded frozen step metadata')
    expected = {}
    for item in expected_samples:
        if not isinstance(item, dict) or set(item) != {'key', 'metadata'}:
            raise ValueError('invalid frozen failed native sample')
        metadata = SampleMetadataV3.from_dict(item['metadata'])
        if metadata.run_id != owner.run_id or metadata.generation != owner.generation or make_v3_sample_key(metadata) != item['key'] or (item['key'] in expected):
            raise ValueError('failed native sample identity differs from control owner')
        expected[item['key']] = metadata
    for key, entry in (*owner._pending.items(), *owner._retained.items()):
        if key not in expected or entry[0].metadata != expected[key]:
            raise ValueError('failed native cleanup cannot retire work outside its frozen step')
    if any((key not in expected for key in owner._reclaim_pending.values())):
        raise ValueError('failed native cleanup has unrelated reclaim work')
    leases = tuple((owner.leases.get(item['lease_id']) for item in owner.leases.snapshot()['leases']))
    active = tuple((lease for lease in leases if lease.state in (LeaseState.LEASED, LeaseState.MATERIALIZED)))
    if any((lease.key not in expected for lease in active)):
        raise ValueError('failed native cleanup has unrelated active leases')
    from verl_speco.runtime.services import import_transfer_queue
    tq = import_transfer_queue()
    tq.init()
    present = tq.kv_list(partition_id=owner.partition).get(owner.partition, {})
    for key in set(present) & set(expected):
        parsed = parse_v3_ready_tag(present[key])
        if parsed is None or parsed[0] != expected[key]:
            raise ValueError('failed native TQ tag does not prove exact frozen ownership')
    committed = owner.leases.status()['committed_steps']
    cancelled = owner.cancel_leases(tuple((lease.lease_id for lease in active)), reason='owned native roles exited after failed prefill') if active else ()
    reclaimed = owner.reap_expired()
    ready = tuple((entry[0] for entry in owner._retained.values()))
    retired = owner.discard_unacquired(ready) if ready else ()
    pending = []
    for key, entry in tuple(owner._pending.items()):
        if key in tq.kv_list(partition_id=owner.partition).get(owner.partition, {}):
            tq.kv_clear(keys=key, partition_id=owner.partition)
        if key in tq.kv_list(partition_id=owner.partition).get(owner.partition, {}):
            raise RuntimeError('failed native pending payload remains present')
        owner.cancel(key, token=entry[1])
        pending.append(key)
    orphaned = []
    for key in sorted(set(tq.kv_list(partition_id=owner.partition).get(owner.partition, {})) & set(expected)):
        tq.kv_clear(keys=key, partition_id=owner.partition)
        if key in tq.kv_list(partition_id=owner.partition).get(owner.partition, {}):
            raise RuntimeError('failed native orphaned descriptor remains present')
        orphaned.append(key)
    status = owner.status()
    if status['used_bytes'] or status['retained_items'] or owner._pending or owner._reclaim_pending or (status['leases']['committed_steps'] != committed):
        raise RuntimeError('failed native cleanup has uncertain accounting or changed commits')
    return dict(run_id=owner.run_id, generation=owner.generation, state='FAILED_PREFILL_RETIRED_AFTER_OWNED_EXIT', expected_samples=len(expected), cancelled_lease_ids=[lease.lease_id for lease in cancelled], reclaimed_keys=list(reclaimed), retired_unacquired_keys=list(retired), cancelled_pending_keys=pending, retired_orphaned_keys=orphaned, existing_optimizer_commits_preserved=committed, new_optimizer_commits=0, committed_DCP_and_receipts_modified=False, foreign_TQ_keys_preserved=True, control_status=status)
import ipaddress
from urllib.parse import urlsplit
from verl_speco.trainer.distributed_batches import MAX_NATIVE_PREFILL_ROWS
root = Path(__file__).resolve().parents[2]

def validate_native_config(config):
    required = {'schema_version', 'run_id', 'sources', 'trainer', 'sequence_file', 'model_overlay', 'target_model_dir', 'model_content_audit', 'draft_init_snapshot', 'cann_environment', 'ray_address', 'metadata_url', 'metadata_rpc_port', 'metadata_http_port', 'source_port_base', 'trainer_master_port', 'samples_per_step', 'steps', 'packing_rows', 'max_samples_per_row', 'max_sequence_length', 'global_capacity_bytes', 'python_bin'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'source_native_slot_profile', 'frame_pipelined_dispatch', 'native_receive_copy_stream', 'source_kv_aware_batching', 'source_kv_max_sequences', 'source_async_publication', 'native_owned_release_chunk_size', 'native_owned_copy_pipeline', 'draft_weight_consensus', 'source_tail_lookahead', 'source_early_ready', 'source_graph_mode', 'source_reuse_mhc_projection', 'control_poll_seconds', 'source_mhc_sequence_parallel', 'source_full_batch_prefill', 'source_mhc_prefill_fusion', 'native_selective_hidden_routing', 'second_trainer', 'resolved_launch_plan', 'native_receive_workers', 'native_cpu_bulk_receive', 'failed_genesis_recovery', 'source_native_buffer_batches', 'source_capture_owner_ranks', 'native_owned_copy_workers', 'native_frame_seal_workers', 'native_cpu_threads', 'source_metadata_prefetch_depth', 'source_step_lookahead', 'source_tp', 'execution', 'native_verification_workers', 'native_payload_prefetch_bytes', 'source_physical_batch_size', 'source_prefill_batch_tokens', 'source_memory_utilization', 'native_inline_descriptor', 'frame_hugepages', 'source_device_concat', 'source_cpu_direct_pack', 'dcp_file_hash_workers', 'dcp_file_write_workers', 'direct_streaming_frame', 'source_token_fill_target', 'source_engine_batch_size', 'source_cross_step_token_batches', 'source_engine_token_budget'}:
        raise ValueError('native config has missing or unsupported fields')
    c = json.loads(json.dumps(config))
    from verl_speco.producer.capture import config_slot_profile
    source_slot_profile = config_slot_profile(c)
    if type(c.get('source_full_batch_prefill', False)) is not bool:
        raise ValueError('source full batch prefill must be boolean')
    if type(c.get('source_mhc_prefill_fusion', False)) is not bool:
        raise ValueError('source mHC prefill fusion must be boolean')
    if type(c.get('source_mhc_sequence_parallel', False)) is not bool:
        raise ValueError('source mHC sequence parallel must be boolean')
    from verl_speco.runtime.services import config_control_poll_seconds
    config_control_poll_seconds(c)
    selected = c.get('native_selective_hidden_routing', False)
    if type(selected) is not bool:
        raise ValueError('selected hidden routing must be boolean')
    if selected and ('second_trainer' not in c or c.get('native_receive_workers', 1) not in (2, 4) or c.get('native_cpu_bulk_receive') is not True):
        raise ValueError('selected hidden routing requires two trainer nodes and independent bulk receivers')
    if type(c.get('resolved_launch_plan', False)) is not bool:
        raise ValueError('resolved launch plan must be boolean')
    if type(c.get('failed_genesis_recovery', False)) is not bool:
        raise ValueError('failed genesis recovery config must be boolean')
    if type(c.get('source_reuse_mhc_projection', False)) is not bool:
        raise ValueError('mHC projection reuse config must be boolean')
    if c.get('source_reuse_mhc_projection', False) and (not c.get('source_mhc_sequence_parallel', False)):
        raise ValueError('mHC output reuse requires the verified full-batch SP standalone profile')
    if type(c.get('source_step_lookahead', False)) is not bool:
        raise ValueError('source step lookahead config must be boolean')
    if type(c.get('source_early_ready', False)) is not bool:
        raise ValueError('source early-ready policy must be boolean')
    if c.get('source_early_ready', False) and (c.get('source_step_lookahead') is not True or c.get('resolved_launch_plan') is not True or c.get('source_tp') != 8 or (len(c.get('sources', [])) != 6) or (c.get('source_native_buffer_batches') not in (2, 3, 4, 10)) or (c.get('source_capture_owner_ranks') != 8) or (c.get('source_physical_batch_size') not in (8, 16)) or (c.get('source_prefill_batch_tokens') != 131072) or (c.get('source_graph_mode') != 'FULL_DECODE_ONLY')):
        raise ValueError('early frozen prefill requires the bound TP8 FULL double-buffer profile')
    receive_workers = c.get('native_receive_workers', 1)
    if type(receive_workers) is not int or receive_workers not in (1, 2, 4, 8):
        raise ValueError('invalid native receiver worker count')
    bulk = c.get('native_cpu_bulk_receive', False)
    if type(bulk) is not bool or (receive_workers != 1 and (not bulk)):
        raise ValueError('parallel native receivers require explicit bulk CPU receive')
    if 'source_metadata_prefetch_depth' in c and (type(c['source_metadata_prefetch_depth']) is not int or not 1 <= c['source_metadata_prefetch_depth'] <= 16):
        raise ValueError('source metadata prefetch depth must be an integer from 1 to 16')
    integer = lambda key, low, high: type(c[key]) is int and low <= c[key] <= high
    if type(c.get('source_graph_mode', '')) is not str or c.get('source_graph_mode', '') not in ('', 'FULL_DECODE_ONLY'):
        raise ValueError('invalid persistent source graph mode')
    if type(c.get('frame_hugepages', False)) is not bool:
        raise ValueError('frame hugepages config must be boolean')
    if type(c.get('direct_streaming_frame', False)) is not bool:
        raise ValueError('direct streaming frame config must be boolean')
    for key in ('frame_pipelined_dispatch', 'native_receive_copy_stream'):
        if type(c.get(key, False)) is not bool:
            raise ValueError(key + ' config must be boolean')
    if c.get('frame_pipelined_dispatch', False) and c.get('direct_streaming_frame') is not True:
        raise ValueError('pipelined frame dispatch requires direct streaming frame')
    if c.get('native_receive_copy_stream', False) and c.get('native_cpu_bulk_receive') is not True:
        raise ValueError('private receive copy stream requires full bulk CPU receive')
    if (c.get('frame_pipelined_dispatch', False) or c.get('native_receive_copy_stream', False)) and 'second_trainer' in c:
        raise ValueError('frame/stream qualification is limited to one DP16 trainer')
    if type(c.get('source_device_concat', False)) is not bool:
        raise ValueError('source device concat config must be boolean')
    if type(c.get('source_cpu_direct_pack', False)) is not bool:
        raise ValueError('source CPU direct pack config must be boolean')
    kv_aware = c.get('source_kv_aware_batching', False)
    if type(kv_aware) is not bool or ('source_kv_max_sequences' in c and (not kv_aware)):
        raise ValueError('invalid adaptive source KV profile')
    if kv_aware and (not (type(c.get('source_kv_max_sequences')) is int and c['source_kv_max_sequences'] == 16 and (c.get('source_async_publication') is True) and (c.get('source_engine_batch_size') == 40) and (c.get('source_token_fill_target') == 0.9) and (c.get('source_cross_step_token_batches', False) is False) and (c.get('source_engine_token_budget') is None) and (c.get('source_prefill_batch_tokens') == 131072) and (c.get('source_cpu_direct_pack') is True))):
        raise ValueError('adaptive source KV requires async same-step sixteen-child/full128K admission')
    if source_slot_profile and (not (c.get('source_async_publication') is True and c.get('source_cpu_direct_pack') is True and (c.get('source_tail_lookahead') is True) and (c.get('samples_per_step') == 960) and (c.get('source_physical_batch_size') == 8))):
        raise ValueError('tiered source slots require the exact async packed960 complete-source pipeline')
    async_publish = c.get('source_async_publication', False)
    if type(async_publish) is not bool:
        raise ValueError('source async publication config must be boolean')
    if async_publish and (not (c.get('source_tp') == 8 and len(c.get('sources', [])) == 6 and (c.get('source_physical_batch_size') == 8) and (c.get('source_capture_owner_ranks') == 8) and (c.get('source_native_buffer_batches') in (4, 10)) and (c.get('source_early_ready') is True) and (c.get('source_full_batch_prefill') is True) and (c.get('source_graph_mode') == 'FULL_DECODE_ONLY') and (c.get('resolved_launch_plan') is True) and (c.get('source_device_concat') is True) and (c.get('source_cpu_direct_pack') is True) and (not c.get('source_engine_batch_size') or (c.get('source_kv_aware_batching') is True and c.get('source_kv_max_sequences') == 16)) and ('second_trainer' not in c))):
        raise ValueError('source async publication differs from the frozen six-source FULL four-bank profile')
    if type(c.get('dcp_file_hash_workers', 1)) is not int or not 1 <= c.get('dcp_file_hash_workers', 1) <= 4:
        raise ValueError('checkpoint file hash workers config must be an integer from 1 to 4')
    if type(c.get('dcp_file_write_workers', 1)) is not int or not 1 <= c.get('dcp_file_write_workers', 1) <= 4:
        raise ValueError('checkpoint file write workers config must be an integer from 1 to 4')
    if type(c.get('native_inline_descriptor', False)) is not bool:
        raise ValueError('native inline descriptor config must be boolean')
    target_tp = c.get('source_tp', 16)
    if type(target_tp) is not int or target_tp not in (8, 16):
        raise ValueError('native source TP must be 8 or 16')
    engine_tokens = c.get('source_engine_token_budget')
    if engine_tokens is not None and (type(engine_tokens) is not int or engine_tokens != 65536 or c.get('source_cross_step_token_batches') is not True):
        raise ValueError('engine token budget requires explicit64K cross-step coalescing')
    cross_tokens = c.get('source_cross_step_token_batches', False)
    if type(cross_tokens) is not bool or (cross_tokens and c.get('source_tail_lookahead') is not True):
        raise ValueError('cross-step token batching requires admitted tail lookahead')
    if {'source_token_fill_target', 'source_engine_batch_size', 'source_cross_step_token_batches'} & c.keys():
        if type(c.get('source_token_fill_target')) is not float or c['source_token_fill_target'] != (0.95 if cross_tokens else 0.9) or type(c.get('source_engine_batch_size')) is not int or (c['source_engine_batch_size'] != 40) or (c.get('source_tp') != 8) or (c.get('source_physical_batch_size') != 8) or (c.get('source_native_buffer_batches') not in (4, 10)) or (c.get('source_capture_owner_ranks') != 8) or (c.get('samples_per_step') != 960) or (c.get('source_metadata_prefetch_depth', 0) < 6) or (c.get('source_graph_mode') != 'FULL_DECODE_ONLY') or (c.get('source_early_ready') is not True) or (c.get('resolved_launch_plan') is not True) or (c.get('source_step_lookahead') is not True):
            raise ValueError('token coalescing requires explicit40/.9 four-bank packed960 FULL geometry')
    physical_batch_size = c.get('source_physical_batch_size', 4 if target_tp == 8 else 16)
    if type(physical_batch_size) is not int or physical_batch_size not in (2, 4, 8, 16):
        raise ValueError('unsupported source physical batch profile')
    from verl_speco.producer.capture import validate_source_capture_owners
    source_owners = validate_source_capture_owners(c.get('source_capture_owner_ranks', 0), target_tp=target_tp, physical_slots=physical_batch_size)
    from verl_speco.producer.capture import validate_source_capture_buffers
    validate_source_capture_buffers(c.get('source_native_buffer_batches', 1), owner_ranks=source_owners, target_tp=target_tp, physical_slots=physical_batch_size, step_lookahead=c.get('source_step_lookahead', False), wide_full_profile=target_tp == 8 and c.get('source_prefill_batch_tokens') == 131072 and (c.get('source_graph_mode') == 'FULL_DECODE_ONLY'), source_slot_profile=source_slot_profile)
    import math
    tokens = c.get('source_prefill_batch_tokens', 32768 if target_tp == 8 else 65536)
    memory = c.get('source_memory_utilization', 0.85)
    profiles = (8192, 16384, 32768, 49152, 65536, 131072) if target_tp == 8 else (65536, 131072)
    if type(tokens) is not int or tokens not in profiles:
        raise ValueError('unsupported source prefill token profile')
    if target_tp == 8 and physical_batch_size == 16 and (not (tokens == 131072 and source_owners == 8 and (c.get('source_native_buffer_batches') == 2))):
        raise ValueError('TP8 sixteen physical slots require explicit two-bank FULL profile')
    if type(memory) not in (int, float) or not math.isfinite(memory) or (not 0.1 <= memory <= 0.95):
        raise ValueError('invalid source memory profile')
    if tokens == 131072 and (physical_batch_size not in (8, 16) or (physical_batch_size == 16 and target_tp != 8) or memory > 0.85):
        raise ValueError('128K prefill profile requires eight 512MiB slots and explicit memory at most 0.85')
    wide_tp8 = target_tp == 8 and physical_batch_size in (8, 16) and (source_owners == 8) and (tokens == 131072) and (len(c['sources']) == 6) and (c.get('source_graph_mode') == 'FULL_DECODE_ONLY') and (c.get('source_native_buffer_batches') in (2, 3, 4, 10)) and (c.get('source_step_lookahead') is True) and (c.get('source_mhc_sequence_parallel') is True) and (c.get('source_full_batch_prefill') is True) and (c.get('source_mhc_prefill_fusion') is True) and (c['max_sequence_length'] == 16384) and (memory <= 0.85)
    consensus = c.get('draft_weight_consensus', '')
    if consensus not in ('', 'exact_device_root_sha') or (consensus and 'second_trainer' in c):
        raise ValueError('device weight consensus requires the declared single DP16 mode')
    if type(c.get('source_tail_lookahead', False)) is not bool:
        raise ValueError('source tail lookahead policy must be boolean')
    if c.get('source_tail_lookahead', False) and (not wide_tp8 or c.get('source_early_ready') is not True or c.get('source_native_buffer_batches') not in (3, 4, 10) or (c.get('resolved_launch_plan') is not True) or ('second_trainer' in c)):
        raise ValueError('source tail lookahead requires the exact frozen three-bank DP16 profile')
    if c.get('source_native_buffer_batches') in (3, 4, 10) and (not wide_tp8 or c.get('source_early_ready') is not True or c.get('resolved_launch_plan') is not True or ('second_trainer' in c)):
        raise ValueError('three native banks require the exact early-ready standalone DP16 profile')
    if target_tp == 8 and tokens == 131072 and (not wide_tp8):
        raise ValueError('TP8 128K requires explicit six-source FULL eight-owner double-buffer profile')
    if c.get('source_mhc_sequence_parallel', False):
        profile = (target_tp, physical_batch_size, source_owners, tokens)
        supported = profile == (8, 4, 4, 65536) or profile == (16, 8, 8, 131072) or wide_tp8
        ceiling = 0.85 if target_tp == 16 else 0.9
        if not supported or not c.get('source_full_batch_prefill', False) or (not c.get('source_mhc_prefill_fusion', False)) or (c.get('source_native_buffer_batches', 1) != 2 and (not (wide_tp8 and c.get('source_native_buffer_batches') in (3, 4, 10)))) or (memory > ceiling):
            raise ValueError('mHC sequence parallel requires the verified TP8 64K or TP16 128K double-buffer profile')
    if c['schema_version'] != 1 or type(c['schema_version']) is not int:
        raise ValueError('unsupported native config schema')
    import re
    if not isinstance(c['run_id'], str) or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', c['run_id']) is None:
        raise ValueError('invalid native run ID')
    from verl_speco.runtime.services import native_trainer_roles
    trainers = native_trainer_roles(c)
    if len(trainers) == 2 and c.get('failed_genesis_recovery', False):
        raise ValueError('DP32 recovery requires a committed parent; empty-genesis replay is not supported')
    counts = (6,) if len(trainers) == 2 else (2, 6, 14) if target_tp == 8 else (1, 3, 7)
    if not isinstance(c['sources'], list) or len(c['sources']) not in counts:
        raise ValueError('native launcher accepts one/three/seven TP16 sources or two/six/fourteen TP8 sources')
    roles = [*c['sources'], *trainers]
    for role in roles:
        expected_fields = {'host', 'address', 'device_start'} if target_tp == 8 and role is not c['trainer'] else {'host', 'address'}
        if not isinstance(role, dict) or set(role) != expected_fields or (not isinstance(role['host'], str)) or (re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', role['host']) is None):
            raise ValueError('invalid native role host')
        if not isinstance(role['address'], str):
            raise ValueError('role address must be IPv4 text')
        try:
            ipaddress.IPv4Address(role['address'])
        except (ipaddress.AddressValueError, TypeError):
            raise ValueError('role address must be IPv4') from None
    if target_tp == 16:
        if len({r['host'] for r in roles}) != len(roles) or len({r['address'] for r in roles}) != len(roles):
            raise ValueError('prefill and trainer roles require separate physical nodes')
    else:
        hosts = {}
        addresses = {}
        for source in c['sources']:
            if type(source['device_start']) is not int or source['device_start'] not in (0, 8):
                raise ValueError('TP8 source device start must be exactly 0 or 8')
            if source['host'] in hosts and hosts[source['host']] != source['address'] or (source['address'] in addresses and addresses[source['address']] != source['host']):
                raise ValueError('source node host and address aliases must agree')
            hosts[source['host']] = source['address']
            addresses[source['address']] = source['host']
        if c['trainer']['host'] in hosts or c['trainer']['address'] in addresses:
            raise ValueError('prefill and trainer roles require separate physical nodes')
        for index in range(0, len(c['sources']), 2):
            left, right = c['sources'][index:index + 2]
            if left['host'] != right['host'] or left['address'] != right['address'] or left['device_start'] != 0 or (right['device_start'] != 8):
                raise ValueError('each TP8 source node requires ordered disjoint 0/8 groups')
        if len(hosts) != len(c['sources']) // 2:
            raise ValueError('TP8 source node cannot appear in more than one pair')
    for key in ('sequence_file', 'model_overlay', 'target_model_dir', 'model_content_audit', 'draft_init_snapshot', 'cann_environment', 'python_bin'):
        if not isinstance(c[key], str) or not Path(c[key]).is_absolute():
            raise ValueError(key + ' must be absolute')
        c[key] = str(Path(c[key]).absolute() if key == 'python_bin' else Path(c[key]).resolve())
    for key, low, high in [('samples_per_step', 1, 1024), ('steps', 1, 32), ('packing_rows', 1, MAX_NATIVE_PREFILL_ROWS), ('max_samples_per_row', 1, 1024), ('source_port_base', 1024, 65529), ('trainer_master_port', 1024, 65535), ('metadata_rpc_port', 1024, 65535), ('metadata_http_port', 1024, 65535)]:
        if not integer(key, low, high):
            raise ValueError('invalid native config integer: ' + key)
    if c['samples_per_step'] % len(c['sources']):
        raise ValueError('samples_per_step must contain complete producer rounds')
    if c['steps'] * c['samples_per_step'] // len(c['sources']) > 4096:
        raise ValueError('native evidence fetch prefix exceeds its bounded limit')
    if type(c['max_sequence_length']) is not int or c['max_sequence_length'] not in (8192, 16384, 65536):
        raise ValueError('unsupported source sequence length')
    expected = len(c['sources']) * (8 if tokens == 131072 else 4) * 1024 ** 3
    if target_tp == 8:
        expected = (24 if len(c['sources']) == 6 and source_owners == 4 and (physical_batch_size == 4) else 8 if len(c['sources']) == 2 else 16) * 1024 ** 3
    allowed_capacities = {96 * 1024 ** 3 if source_slot_profile else 24 * c.get('source_native_buffer_batches', 2) * 1024 ** 3} if wide_tp8 else {expected}
    if target_tp == 8 and len(c['sources']) == 14 and (physical_batch_size == 8):
        allowed_capacities.add(32 * 1024 ** 3)
    if target_tp == 16 and len(c['sources']) == 7 and source_owners and (physical_batch_size == 16):
        allowed_capacities.add(32 * 1024 ** 3)
    if type(c['global_capacity_bytes']) is not int or c['global_capacity_bytes'] not in allowed_capacities:
        raise ValueError('global quota must match the verified source topology')
    for key in ('ray_address', 'metadata_url'):
        if not isinstance(c[key], str):
            raise ValueError('invalid service address')
    ray_url = urlsplit('//' + c['ray_address'])
    metadata = urlsplit(c['metadata_url'])
    try:
        ipaddress.IPv4Address(ray_url.hostname)
        ipaddress.IPv4Address(metadata.hostname)
        if not ray_url.port or ray_url.path or ray_url.query or ray_url.fragment or ray_url.username or ray_url.password:
            raise ValueError('invalid Ray address')
        if metadata.scheme != 'http' or metadata.port != c['metadata_http_port'] or metadata.path != '/metadata' or metadata.query or metadata.fragment or metadata.username or metadata.password:
            raise ValueError('invalid metadata URL')
    except (ValueError, TypeError):
        raise ValueError('service addresses require explicit IPv4 and valid ports') from None
    if ray_url.hostname != metadata.hostname:
        raise ValueError('metadata master must use the controller coordinator address')
    port_stride = target_tp if source_owners else 1
    if c['source_port_base'] + len(c['sources']) * port_stride - 1 > 65535:
        raise ValueError('source ports exceed valid port range')
    ports = [c['metadata_rpc_port'], c['metadata_http_port'], ray_url.port, c['trainer_master_port'], *[c['source_port_base'] + i for i in range(len(c['sources']) * port_stride)]]
    if len(ports) != len(set(ports)):
        raise ValueError('native service ports must be distinct')
    from verl_speco.runtime.services import native_draft_cpu_threads
    native_draft_cpu_threads(c)
    copies = c.get('native_owned_copy_workers', 1)
    if type(copies) is not int or copies not in (1, 2, 4, 8, 16):
        raise ValueError('invalid native owned copy width')
    seal = c.get('native_frame_seal_workers', 8)
    if type(seal) is not int or seal not in (1, 2, 4, 8, 16):
        raise ValueError('invalid native frame seal width')
    width = c.get('native_verification_workers', 2)
    if type(width) is not int or width not in (2, 4, 8):
        raise ValueError('invalid native verification width')
    if receive_workers > width:
        raise ValueError('native readers exceed verification slots')
    from verl_speco.producer.capture import source_capture_slot_bytes
    source_slot_bytes = source_capture_slot_bytes(target_tp=target_tp, physical_slots=physical_batch_size, owner_ranks=source_owners)
    budget = c.get('native_payload_prefetch_bytes', max((1 if width == 2 else 2) * 1024 ** 3, (width + 1) * source_slot_bytes))
    four_bank = c.get('source_native_buffer_batches') in (4, 10)
    wide_sixteen = physical_batch_size == 16 and c.get('source_native_buffer_batches') == 2 and (width == 8 or receive_workers == 8)
    wide_receive = four_bank or wide_sixteen
    if wide_receive and (not wide_tp8 or (four_bank and c.get('source_tail_lookahead') is not True) or c.get('source_early_ready') is not True or (c.get('resolved_launch_plan') is not True) or ('second_trainer' in c) or (copies not in (8, 16)) or (width != 8) or (receive_workers != 8) or (bulk is not True) or (budget != 16 * 1024 ** 3)):
        raise ValueError('wide source pools require the exact DP16 eight-reader 16GiB profile')
    if (width == 8 or receive_workers == 8) and (not wide_receive):
        raise ValueError('eight native readers require four-bank eight-child or two-bank sixteen-child geometry')
    budget_limit = (16 if wide_receive else 8) * 1024 ** 3
    if type(budget) is not int or not 0 < budget <= budget_limit:
        raise ValueError('invalid native verification CPU budget')
    pipeline = c.get('native_owned_copy_pipeline', False)
    if type(pipeline) is not bool:
        raise ValueError('invalid native owned copy pipeline policy')
    if pipeline and (not four_bank or physical_batch_size != 8 or copies != 16 or (width != 8) or (receive_workers != 8) or (bulk is not True) or ('second_trainer' in c) or (c['global_capacity_bytes'] != 96 * 1024 ** 3) or (budget != 16 * 1024 ** 3)):
        raise ValueError('owned copy pipeline requires the exact DP16 eight-reader profile')
    chunk = c.get('native_owned_release_chunk_size', 0)
    if type(chunk) is not int or chunk not in (0, 6, 12) or (chunk and (not pipeline)):
        raise ValueError('owned prefix release requires its byte-bounded copy pipeline')
    native_execution(c)
    return c

def native_execution(config):
    base = {'generation', 'start_step', 'end_step', 'parent_receipt'}
    e = config.get('execution', dict(generation=1, start_step=0, end_step=config['steps'], parent_receipt=''))
    if not isinstance(e, dict) or set(e) not in (base, base | {'failed_genesis_proof'}):
        raise ValueError('native execution has missing or unsupported fields')
    if any((type(e[k]) is not int for k in ('generation', 'start_step', 'end_step'))) or e['generation'] not in (1, 2) or (not 0 <= e['start_step'] < e['end_step'] <= config['steps']):
        raise ValueError('native execution generation or step interval differs')
    parent = e['parent_receipt']
    if e['generation'] == 1:
        if e['start_step'] != 0 or parent != '' or 'failed_genesis_proof' in e:
            raise ValueError('initial generation must start at zero without a parent')
    elif e['start_step'] == 0:
        proof = e.get('failed_genesis_proof')
        if config.get('failed_genesis_recovery') is not True or parent != '' or (not isinstance(proof, str)) or (not Path(proof).is_absolute()):
            raise ValueError('empty generation two requires explicit typed failed genesis proof')
    elif 'failed_genesis_proof' in e or not isinstance(parent, str) or (not Path(parent).is_absolute()):
        raise ValueError('continued generation requires a committed absolute parent receipt')
    return dict(e)
import math
import time

def native_source_buffer_admission_layout(target_tp, owner_ranks, physical_slots, buffer_batches=2, source_slot_profile=''):
    from verl_speco.producer.capture import validate_source_capture_buffers
    validate_source_capture_buffers(buffer_batches, owner_ranks=owner_ranks, target_tp=target_tp, physical_slots=physical_slots, wide_full_profile=buffer_batches in (3, 4, 10), source_slot_profile=source_slot_profile)
    return tuple((buffer_batches * len(range(rank, physical_slots, owner_ranks)) if rank < owner_ranks else 0 for rank in range(target_tp)))

def source_slot_requirements(child_count, *, target_tp=8, owner_ranks=8, physical_slots=8, buffer_batches=2, slot_bytes=512 * 1024 ** 2, owner_indices=None, source_slot_profile='', required_bytes=None):
    native_source_buffer_admission_layout(target_tp, owner_ranks, physical_slots, buffer_batches, source_slot_profile)
    if type(child_count) is not int or not 1 <= child_count <= physical_slots:
        raise ValueError('source slot admission requires a valid physical child count')
    if buffer_batches == 10:
        from verl_speco.producer.capture import PROFILE
        owners = tuple(range(child_count)) if owner_indices is None else tuple(owner_indices)
        if (target_tp, owner_ranks, physical_slots, slot_bytes) != (8, 8, 8, 2 * 1024 ** 3) or source_slot_profile != PROFILE or len(owners) != child_count or any((type(i) is not int or not 0 <= i < 8 for i in owners)) or (len(set(owners)) != len(owners)) or (type(required_bytes) is not tuple) or (len(required_bytes) != child_count) or any((type(n) is not int or not 0 < n <= slot_bytes for n in required_bytes)):
            raise ValueError('tiered admission requires exact owners and complete payload byte bounds')
        return tuple((int(rank in owners) for rank in range(8)))
    if required_bytes is not None:
        raise ValueError('complete payload capacity binding requires the tiered slot profile')
    if type(slot_bytes) is not int or slot_bytes not in (512 * 1024 ** 2, 2 * 1024 ** 3) or (slot_bytes == 2 * 1024 ** 3 and ((target_tp, owner_ranks, physical_slots) not in ((8, 8, 8), (8, 8, 16)) or buffer_batches not in (2, 3, 4) or (physical_slots == 16 and buffer_batches != 2))):
        raise ValueError('source capacity slot size differs from the bound native profile')
    if owner_indices is not None:
        if (target_tp, owner_ranks, physical_slots, buffer_batches, slot_bytes) != (8, 8, 8, 4, 2 * 1024 ** 3) or not isinstance(owner_indices, (list, tuple)) or len(owner_indices) != child_count or any((type(rank) is not int or not 0 <= rank < 8 for rank in owner_indices)) or (len(set(owner_indices)) != len(owner_indices)):
            raise ValueError('fragment owners differ from exact four-bank logical-eight profile')
        return tuple((int(rank in owner_indices) for rank in range(target_tp)))
    required = [0] * target_tp
    for child in range(child_count):
        required[child % owner_ranks] += 1
    return tuple(required)

def validate_source_capacity(proofs, *, child_count, target_tp=8, owner_ranks=8, physical_slots=8, buffer_batches=2, slot_bytes=512 * 1024 ** 2, owner_indices=None, source_slot_profile='', required_bytes=None):
    slots = native_source_buffer_admission_layout(target_tp, owner_ranks, physical_slots, buffer_batches, source_slot_profile)
    required = source_slot_requirements(child_count, target_tp=target_tp, owner_ranks=owner_ranks, physical_slots=physical_slots, buffer_batches=buffer_batches, slot_bytes=slot_bytes, owner_indices=owner_indices, source_slot_profile=source_slot_profile, required_bytes=required_bytes)
    if buffer_batches == 10:
        from verl_speco.producer.capture import validate_capacity_proofs
        return validate_capacity_proofs(proofs, owner_indices=tuple(range(child_count)) if owner_indices is None else tuple(owner_indices), required_bytes=required_bytes)
    if not isinstance(proofs, list) or len(proofs) != target_tp:
        raise RuntimeError('native source capacity probe requires all TP identities')
    for rank, (proof, capacity) in enumerate(zip(proofs, slots)):
        if not isinstance(proof, dict) or type(proof.get('rank')) is not int or proof['rank'] != rank or (proof.get('source_capture_owner_ranks') != owner_ranks) or (type(proof.get('source_capture_owner_ranks')) is not int) or (proof.get('source_native_buffer_batches') != buffer_batches) or (type(proof.get('source_native_buffer_batches')) is not int) or (proof.get('native_source_slots') != capacity) or (type(proof.get('native_source_slots')) is not int) or (proof.get('native_source_bytes') != capacity * slot_bytes) or (type(proof.get('native_source_bytes')) is not int) or (type(proof.get('native_retained_sources')) is not int) or (not 0 <= proof['native_retained_sources'] <= capacity) or (type(proof.get('native_available_slots')) is not int) or (proof['native_available_slots'] != capacity - proof['native_retained_sources']) or (type(proof.get('publisher_initialized')) is not bool) or (not proof['publisher_initialized'] and proof['native_retained_sources']) or (capacity == 0 and proof['publisher_initialized']):
            raise RuntimeError('native source capacity proof differs from the fixed double-buffer layout')
    return all((proof['native_available_slots'] >= need for proof, need in zip(proofs, required)))

def wait_source_buffer_admission(fetch, *, child_count, timeout=1200, poll_seconds=0.5, target_tp=8, owner_ranks=8, physical_slots=8, buffer_batches=2, slot_bytes=512 * 1024 ** 2, owner_indices=None, source_slot_profile='', required_bytes=None):
    policy = dict(target_tp=target_tp, owner_ranks=owner_ranks, physical_slots=physical_slots, buffer_batches=buffer_batches, slot_bytes=slot_bytes, owner_indices=owner_indices, source_slot_profile=source_slot_profile, required_bytes=required_bytes)
    source_slot_requirements(child_count, **policy)
    if not callable(fetch) or type(timeout) not in (int, float) or (not math.isfinite(timeout)) or (timeout <= 0) or (type(poll_seconds) not in (int, float)) or (not math.isfinite(poll_seconds)) or (not 0 < poll_seconds <= 30):
        raise ValueError('invalid source native capacity polling policy')
    started = time.monotonic()
    deadline = started + timeout
    while True:
        proofs = fetch()
        if validate_source_capacity(proofs, child_count=child_count, **policy):
            result = dict(physical_children=child_count, required_slots=list(source_slot_requirements(child_count, **policy)), capacity_proofs=proofs, wallclock_seconds=time.monotonic() - started, source_native_buffer_batches=buffer_batches, total_source_pool_budget_bytes=64 * 1024 ** 3 if buffer_batches == 10 else physical_slots * buffer_batches * slot_bytes, initialized_source_registration_bytes=sum((p['native_source_bytes'] for p in proofs if p['publisher_initialized'])), existing_leases_preserved=True)
            if source_slot_profile:
                result.update(source_native_slot_profile=source_slot_profile, required_complete_source_bytes=list(required_bytes))
            if owner_indices is not None:
                result['fragment_owner_indices'] = list(owner_indices)
            print('source_native_capacity_admission_timing ' + json.dumps(result), flush=True)
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError('source native slots remain leased before next publication')
        time.sleep(min(poll_seconds, max(0.001, deadline - time.monotonic())))
from verl_speco.runtime.artifacts import fcntl
physical_generation_SHA = re.compile('[0-9a-f]{64}')
physical_generation_RECORD = re.compile('g-([0-9]{20})\\.json')

def physical_generation_counter(value):
    if type(value) is not int or not 1 <= value < 2 ** 63:
        raise ValueError('physical generation requires an explicit positive int64')

def reserve_joint_generation(*, run_root, run_id, parent_generation, parent_binding_checksum):
    """Atomically persist an unused generation above all owned recorded attempts.

    Existing diagnostic configs seed the high-water mark during migration from
    parent+1 allocation. The per-run journal serializes concurrent preflights;
    invalid/corrupted ownership records fail closed before any NPU effect.
    """
    physical_generation_counter(parent_generation)
    if type(run_id) is not str or not run_id or type(parent_binding_checksum) is not str or (physical_generation_SHA.fullmatch(parent_binding_checksum) is None):
        raise ValueError('generation reservation requires the actual run and joint parent SHA')
    root = Path(run_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('generation reservation requires an existing absolute owned run root')
    root = root.resolve()
    journal = root / 'physical-generation-reservations'
    journal.mkdir(exist_ok=True)
    with (journal / '.allocation.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        maximum = parent_generation
        for path in root.glob('*/diagnostic.config.json'):
            attempt = json.loads(path.read_text())
            if not isinstance(attempt, dict):
                raise ValueError('recorded physical attempt is not a configuration object')
            if attempt.get('run_id') != run_id:
                continue
            generation = attempt.get('generation')
            physical_generation_counter(generation)
            owned_root = attempt.get('joint_run_root')
            if owned_root is not None and Path(owned_root).resolve() != root:
                raise ValueError('recorded same-run attempt belongs to a different joint root')
            maximum = max(maximum, generation)
        for path in journal.glob('g-*.json'):
            match = physical_generation_RECORD.fullmatch(path.name)
            if match is None:
                raise ValueError('physical generation reservation has an invalid filename')
            record = json.loads(path.read_text())
            required = {'schema_version', 'run_id', 'generation', 'parent_generation', 'parent_binding_checksum'}
            if not isinstance(record, dict) or set(record) != required or type(record['schema_version']) is not int or (record['schema_version'] != 1) or (record['run_id'] != run_id):
                raise ValueError('physical generation journal has foreign or corrupted ownership')
            for name in ('generation', 'parent_generation'):
                physical_generation_counter(record[name])
            if record['generation'] != int(match[1]) or record['parent_generation'] >= record['generation'] or type(record['parent_binding_checksum']) is not str or (physical_generation_SHA.fullmatch(record['parent_binding_checksum']) is None):
                raise ValueError('physical generation reservation identity is inconsistent')
            maximum = max(maximum, record['generation'])
        generation = maximum + 1
        physical_generation_counter(generation)
        record = dict(schema_version=1, run_id=run_id, generation=generation, parent_generation=parent_generation, parent_binding_checksum=parent_binding_checksum)
        with (journal / f'g-{generation:020d}.json').open('x') as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(journal, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
        return generation
from verl_speco.integration.multinode_recovery import open_joint_parent
from verl_speco.runtime.artifacts import read_published_json
from verl_speco.runtime.training_runtime import validate_speco_online_endpoints as validate_speco_online_endpoints

def joint_continuation_config(config, envelope, acceptance, *, serving_view, output_root, actor_master_port, draft_master_port, steps=1, metadata_rpc_port=None, metadata_http_port=None):
    """Continue a fully successful session, preserving its completion gates."""
    if not isinstance(acceptance, dict):
        raise ValueError('joint continuation requires actual session acceptance')
    return joint_continuation_joint_recovery_config(config, envelope, acceptance, serving_view=serving_view, output_root=output_root, actor_master_port=actor_master_port, draft_master_port=draft_master_port, steps=steps, metadata_rpc_port=metadata_rpc_port, metadata_http_port=metadata_http_port)

def joint_checkpoint_recovery_config(config, envelope, *, serving_view, output_root, actor_master_port, draft_master_port, steps=1, metadata_rpc_port=None, metadata_http_port=None):
    """Recover a committed safe point even if subsequent work failed.

    This prepares a config only. It neither asserts that the old session
    succeeded nor retires workers. The caller must independently release old
    physical roles and verify resources before starting fresh incarnations.
    Complete checkpoint reopening validates model/optimizer/RNG manifests,
    serving ACKs and the quiescent control snapshot exactly as continuation.
    """
    return joint_continuation_joint_recovery_config(config, envelope, None, serving_view=serving_view, output_root=output_root, actor_master_port=actor_master_port, draft_master_port=draft_master_port, steps=steps, metadata_rpc_port=metadata_rpc_port, metadata_http_port=metadata_http_port)

def joint_continuation_joint_recovery_config(config, envelope, acceptance, *, serving_view, output_root, actor_master_port, draft_master_port, steps=1, metadata_rpc_port=None, metadata_http_port=None):
    generation = config['generation']
    if type(generation) is not int or generation < 1 or type(steps) is not int or (steps < 1):
        raise ValueError('joint continuation requires positive integer generation and steps')
    root = Path(config['joint_run_root']).resolve()
    view, output = (Path(serving_view), Path(output_root))
    if not view.is_absolute() or not output.is_absolute() or (not view.resolve().is_relative_to(root)) or (not output.resolve().is_relative_to(root)) or (output.resolve() == Path(config['output_root']).resolve()) or output.exists():
        raise ValueError('joint continuation needs fresh owned paths')
    ports = (actor_master_port, draft_master_port)
    if any((type(p) is not int or not 1024 <= p <= 65535 for p in ports)) or len(set(ports)) != 2 or set(ports) & {config['actor_master_port'], config['draft_master_port']}:
        raise ValueError('joint continuation requires fresh distinct role master ports')
    if acceptance is not None and (acceptance.get('run_id') != config['run_id'] or acceptance.get('generation') != generation or type(acceptance.get('generation')) is not int or (acceptance.get('real_full_92B_actor_target_HCCL_async_SpeCo_session') is not True) or (acceptance.get('checkpoint_channel_closed') is not True) or (acceptance.get('final_updated_target_and_draft_generation_complete') is not True) or (acceptance.get('joint_checkpoint_binding') != envelope['binding'])):
        raise ValueError('joint continuation requires completed actual session and closed channel receipts')
    if config.get('joint_fingerprints') != envelope['fingerprints']:
        raise ValueError('joint continuation cannot change the configured recipe')
    current = deepcopy(config)
    current.update(generation=generation + 1, joint_restore_binding=deepcopy(envelope['binding']), joint_fingerprints=deepcopy(envelope['fingerprints']), output_root=str(output), draft_model_path=str(view), steps=steps, actor_master_port=actor_master_port, draft_master_port=draft_master_port)
    metadata_fields = ('metadata_url', 'metadata_rpc_port', 'metadata_http_port')
    if any((name in config for name in metadata_fields)):
        if not all((name in config for name in metadata_fields)):
            raise ValueError('joint continuation requires a complete parent metadata namespace')
        validate_speco_online_endpoints(config, environment={})
        fresh = (metadata_rpc_port, metadata_http_port)
        old = {config['metadata_rpc_port'], config['metadata_http_port'], config['actor_master_port'], config['draft_master_port']}
        if any((type(p) is not int or not 1024 <= p <= 65535 for p in fresh)) or len(set(fresh)) != 2 or set(fresh) & (old | set(ports)):
            raise ValueError('joint continuation requires fresh distinct metadata master ports')
        previous = urlsplit(config['metadata_url'])
        host = previous.hostname
        if ':' in host:
            host = '[' + host + ']'
        current.update(metadata_rpc_port=metadata_rpc_port, metadata_http_port=metadata_http_port, metadata_url=previous._replace(netloc=f'{host}:{metadata_http_port}').geturl())
        validate_speco_online_endpoints(current, environment={})
    elif metadata_rpc_port is not None or metadata_http_port is not None:
        raise ValueError('fresh metadata master ports require a configured parent namespace')
    parent, _ = open_joint_parent(current, current['joint_restore_binding'])
    if parent['versions'] != envelope['versions'] or parent['snapshot'] != envelope['snapshot'] or parent['versions']['generation'] != generation:
        raise ValueError('joint recovery parent differs from its complete checkpoint')
    if acceptance is not None:
        completed = acceptance.get('steps')
        if not isinstance(completed, list) or len(completed) != config['steps'] or (not completed) or (completed[-1].get('versions') != parent['versions']):
            raise ValueError('joint continuation parent differs from its completed session')
    horizon = config['total_training_steps']
    if type(horizon) is not int or parent['versions']['rl_step'] + steps > horizon:
        raise ValueError('joint continuation exceeds the unchanged scheduler horizon')
    if current.get('prompt_input'):
        from verl_speco.runtime.training_runtime import validate_online_prompt_budget
        validate_online_prompt_budget(current, start_step=parent['versions']['rl_step'])
    serving = read_published_json(view / 'snapshot-binding.json')
    if serving.get('snapshot') != parent['snapshot']:
        raise ValueError('joint continuation cold serving view differs from trained parent')
    current['generation'] = reserve_joint_generation(run_root=root, run_id=config['run_id'], parent_generation=generation, parent_binding_checksum=envelope['binding']['checksum'])
    return current

@dataclass(frozen=True)
class SequenceGenerationConcurrency:
    server_max_num_seqs: int
    client_concurrency: int
    client_max_inflight_requests: int
    gpu_memory_utilization: float = 0.85

def resolve_sequence_generation_concurrency(config: Mapping) -> SequenceGenerationConcurrency:
    server = config.get('server_max_num_seqs', 1)
    client = config.get('client_concurrency', server)
    inflight = config.get('client_max_inflight_requests', client)
    if any((type(value) is not int or value <= 0 for value in (server, client, inflight))):
        raise ValueError('sequence-only concurrency values must be positive integers')
    if not inflight <= client <= server:
        raise ValueError('sequence-only inflight must not exceed client concurrency or server capacity')
    memory = config.get('gpu_memory_utilization', 0.85)
    if type(memory) not in (int, float) or not math.isfinite(memory) or (not 0 < memory < 1):
        raise ValueError('sequence-only GPU memory utilization must be finite and between zero and one')
    return SequenceGenerationConcurrency(server, client, inflight, float(memory))

def sequence_graph_capture_sizes(config, max_num_seqs):
    sizes = config.get('graph_capture_sizes')
    if sizes is None:
        return None
    if type(max_num_seqs) is not int or max_num_seqs < 1 or (not isinstance(sizes, list)) or (not 1 <= len(sizes) <= 16) or any((type(n) is not int or n < 1 for n in sizes)) or (sizes != sorted(set(sizes))) or (sizes[0] != 1) or (sizes[-1] != max_num_seqs):
        raise ValueError('decode graph sizes must be sorted unique positive integers covering 1..max_num_seqs, at most16')
    return tuple(sizes)

def transition_paused_graph_capture(proof, source_config, target_config, max_num_seqs):
    """Only capture sizes may change, after exact source writers are released."""
    before = json.loads(source_config.read_text())
    after = json.loads(target_config.read_text())
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    if changed != {'graph_capture_sizes'}:
        raise ValueError('graph capture transition must preserve all generation and memory settings')
    sizes = sequence_graph_capture_sizes(after, max_num_seqs)
    if sizes is None:
        raise ValueError('graph capture transition needs an explicit bounded profile')
    source_sha = hashlib.sha256(source_config.read_bytes()).hexdigest()
    target_sha = hashlib.sha256(target_config.read_bytes()).hexdigest()
    if proof.get('state') != 'PAUSED' or proof.get('run_id') != before['run_id'] or proof.get('config_sha256') != source_sha or (proof.get('owned_processes_remaining') != 0) or (proof.get('npus_free') is not True):
        raise ValueError('graph capture transition requires the exact saved and released source owner')
    result = deepcopy(proof)
    result['config_sha256'] = target_sha
    result['config_transition'] = dict(source_config=str(source_config), target_config=str(target_config), source_sha256=source_sha, target_sha256=target_sha, changed_fields=sorted(changed), graph_capture_sizes=list(sizes))
    return result

def sequence_input_order_arguments(config, replica_index):
    """Validate the selected shard before launching any server or client."""
    if type(replica_index) is not int or not 0 <= replica_index < 16:
        raise ValueError('sequence input order requires a replica index from 0 to 15')
    policy = config.get('sequence_input_order')
    if policy is None:
        return []
    if not isinstance(policy, Mapping) or set(policy) - {'enabled', 'mode', 'rebalance_completed_credit', 'plans'}:
        raise ValueError('invalid sequence input order policy')
    enabled = policy.get('enabled', True)
    if type(enabled) is not bool:
        raise ValueError('sequence input order enabled must be boolean')
    if not enabled:
        return []
    if policy.get('mode') != 'source_proportional':
        raise ValueError('sequence input order must use selected source proportions')
    credit = policy.get('rebalance_completed_credit', True)
    if type(credit) is not bool:
        raise ValueError('completed source credit must be boolean')
    plans = policy.get('plans')
    if type(plans) is not list or len(plans) != 16:
        raise ValueError('sequence input order must bind all sixteen prompt shards')
    bindings = {}
    for binding in plans:
        if not isinstance(binding, Mapping) or set(binding) != {'replica_index', 'plan_path', 'plan_sha256'}:
            raise ValueError('invalid per-replica sequence order binding')
        index = binding['replica_index']
        if type(index) is not int or not 0 <= index < 16 or index in bindings:
            raise ValueError('duplicate or invalid order replica index')
        path, digest = (binding['plan_path'], binding['plan_sha256'])
        if not isinstance(path, str) or not path or (not Path(path).is_absolute()):
            raise ValueError('sequence order plan requires an absolute path')
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('sequence order plan requires an explicit SHA256')
        bindings[index] = (path, digest)
    plan_path, plan_sha = bindings[replica_index]
    from verl_speco.producer.replay import CheckedWeightedInput
    CheckedWeightedInput(plan_path, expected_plan_sha256=plan_sha, expected_input_path=Path(config['output_root']) / 'prompts' / f'prompts-{replica_index:02d}.jsonl')
    return ['--input-order-plan-path', plan_path, '--input-order-plan-sha256', plan_sha, '--input-order-rebalance-completed-credit' if credit else '--no-input-order-rebalance-completed-credit']
