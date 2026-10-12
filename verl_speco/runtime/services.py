# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any

def register_capture_engine_shutdown(llm: Any, *, register: Callable[[Callable[[], None]], Any], timeout: float=60.0, before_shutdown: Callable[[], None] | None=None) -> Callable[[], None]:
    """Register after TQ cleanup so interpreter exit closes the engine first.

    The returned callback also supports explicit shutdown after a successful
    single capture. Failed shutdown remains retryable at interpreter exit.
    """
    if timeout <= 0:
        raise ValueError('capture engine shutdown timeout must be positive')
    shutdown = llm.llm_engine.engine_core.shutdown
    closed = False

    def close() -> None:
        nonlocal closed
        if not closed:
            try:
                if before_shutdown is not None:
                    before_shutdown()
            finally:
                shutdown(timeout=timeout)
            closed = True
    register(close)
    return close
from typing import Mapping
MAX_CAPTURE_PRODUCER_INSTANCES = 14

def capture_device_concat(environment: Mapping[str, str]) -> bool:
    """Validate the explicit feature-copy policy before engine allocation."""
    flag = environment.get('SPECO_CAPTURE_DEVICE_CONCAT', '')
    if type(flag) is not str or flag not in ('', '0', '1'):
        raise ValueError('invalid capture device concat flag')
    return flag == '1'

def capture_source_cpu_pack(environment: Mapping[str, str]) -> bool:
    """Validate final-slot CPU copy before target engine allocation."""
    flag = environment.get('SPECO_NATIVE_SOURCE_CPU_PACK', '')
    if type(flag) is not str or flag not in ('', '0', '1'):
        raise ValueError('invalid native source CPU pack flag')
    if flag == '1' and environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1':
        raise ValueError('native source CPU pack requires native feature transport')
    return flag == '1'

def capture_target_parallelism(environment: Mapping[str, str]) -> int:
    value = environment.get('SPECO_TARGET_TP', '8')
    if value not in ('8', '16'):
        raise ValueError('92B capture target TP must be 8 or 16')
    target_tp = int(value)
    device_start = environment.get('SPECO_CAPTURE_DEVICE_START', '0')
    if device_start not in ('0', '8') or (target_tp == 16 and device_start != '0'):
        raise ValueError('92B capture device start must be 0, or 8 for TP8')
    start = int(device_start)
    expected = ','.join((str(device) for device in range(start, start + target_tp)))
    if environment.get('ASCEND_RT_VISIBLE_DEVICES') != expected:
        raise ValueError('92B capture requires matching logical dies for its explicit device start')
    return target_tp

def capture_prefill_engine_options(environment: Mapping[str, str]) -> dict:
    """Explicit capture scheduling knobs; leave the pinned default untouched."""
    import math
    capture_device_concat(environment)
    from verl_speco.producer.capture import environment_slot_profile
    environment_slot_profile(environment)
    capture_source_cpu_pack(environment)
    if environment.get('SPECO_MHC_SP_PREPROFILE', '') not in ('', '0', '1'):
        raise ValueError('invalid mHC preprofile policy')
    tokens = environment.get('SPECO_CAPTURE_MAX_BATCHED_TOKENS', '')
    if tokens and tokens not in ('8192', '16384', '32768', '49152', '65536', '131072'):
        raise ValueError('capture prefill token batch must be 8K, 16K, 32K, 48K or 64K')
    try:
        memory = float(environment.get('SPECO_CAPTURE_MEMORY_UTILIZATION', '0.85'))
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid capture memory utilization') from exc
    if not math.isfinite(memory) or not 0.1 <= memory <= 0.95:
        raise ValueError('capture memory utilization must be finite within 0.1 to 0.95')
    physical = environment.get('SPECO_CAPTURE_PHYSICAL_BATCH_SIZE')
    wide_child = physical == '16' and environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES') == '2' and (environment.get('SPECO_TARGET_MAX_SEQUENCE_LENGTH') in ('16384', '49152'))
    wide_tp8 = environment.get('SPECO_TARGET_TP') == '8' and tokens == '131072' and (environment.get('SPECO_MHC_SP_PREPROFILE') == '1') and (physical == '8' or wide_child) and (environment.get('SPECO_CAPTURE_NATIVE_SOURCE_SLOTS') == physical) and (environment.get('SPECO_CAPTURE_SOURCE_OWNER_RANKS') == '8') and (environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES') in ('2', '3', '4', '10')) and (environment.get('SPECO_CAPTURE_GRAPH_MODE') == 'FULL_DECODE_ONLY') and (environment.get('SPECO_SEQUENCE_FULL_DECODE') == '1') and (environment.get('SPECO_TARGET_MAX_SEQUENCE_LENGTH') == '16384' or (environment.get('SPECO_TARGET_MAX_SEQUENCE_LENGTH') in ('49152', '65536') and environment.get('SPECO_CAPTURE_LONG_CONTEXT') == environment.get('SPECO_TARGET_MAX_SEQUENCE_LENGTH')))
    if tokens == '131072' and (not (environment.get('SPECO_TARGET_TP') == '16' or wide_tp8) or environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1' or environment.get('SPECO_CAPTURE_REUSE_ENGINE') != '1' or (physical != '8' and (not (wide_tp8 and wide_child))) or (environment.get('SPECO_CAPTURE_NATIVE_SOURCE_SLOTS') != physical) or (memory > 0.85)):
        raise ValueError('128K prefill requires explicit bounded persistent eight-slot TP16 or FULL TP8 profile')
    if environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES') in ('3', '4', '10') and (not wide_tp8 or environment.get('SPECO_PREFILL_EARLY_READY') != '1' or environment.get('SPECO_PREFILL_STEP_LOOKAHEAD') != '1' or (environment.get('SPECO_PREFILL_EXACT_WAVE_ACQUIRE') != '1') or (environment.get('SPECO_CAPTURE_FULL_BATCH_PREFILL') != '1') or (environment.get('SPECO_MHC_SEQUENCE_PARALLEL') != '1')):
        raise ValueError('three source banks require exact frozen FULL TP8 prefill')
    result = dict(gpu_memory_utilization=memory)
    if tokens:
        result['max_num_batched_tokens'] = int(tokens)
    full_batch = environment.get('SPECO_CAPTURE_FULL_BATCH_PREFILL', '')
    if full_batch not in ('', '0', '1'):
        raise ValueError('full batch prefill must be explicitly 0 or 1')
    if full_batch == '1':
        profile = (environment.get('SPECO_TARGET_TP'), tokens, environment.get('SPECO_CAPTURE_PHYSICAL_BATCH_SIZE'), environment.get('SPECO_CAPTURE_NATIVE_SOURCE_SLOTS'), environment.get('SPECO_CAPTURE_SOURCE_OWNER_RANKS'))
        supported = profile == ('8', '65536', '4', '4', '4') or profile == ('16', '131072', '8', '8', '8') or wide_tp8
        if not supported or (environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES') != '2' and (not (wide_tp8 and environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES') in ('3', '4', '10')))) or environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1' or (environment.get('SPECO_CAPTURE_REUSE_ENGINE') != '1') or (environment.get('SPECO_MHC_PREFILL_FUSION') != 'triton-exact'):
            raise ValueError('full batch prefill requires the verified TP8 four-slot or TP16 eight-slot mHC profile')
        result.update(enable_chunked_prefill=False, block_size=128)
    sequence_parallel = environment.get('SPECO_MHC_SEQUENCE_PARALLEL', '')
    if sequence_parallel not in ('', '0', '1'):
        raise ValueError('mHC sequence parallel must be explicitly 0 or 1')
    if sequence_parallel == '1' and (full_batch != '1' or memory > (0.85 if environment.get('SPECO_TARGET_TP') == '16' else 0.9)):
        raise ValueError('mHC sequence parallel requires full TP8 prefill and memory at most 0.90')
    expert_parallel = environment.get('SPECO_CAPTURE_EXPERT_PARALLEL', '')
    if expert_parallel not in ('', '0', '1'):
        raise ValueError('capture expert parallelism must be explicitly 0 or 1')
    if expert_parallel == '1':
        result['enable_expert_parallel'] = True
    if environment.get('SPECO_CAPTURE_ENGINE_TOKEN_BUDGET', ''):
        result['max_num_batched_tokens'] = token_budget_engine_tokens(environment)
        if result['max_num_batched_tokens'] == 65536:
            result['gpu_memory_utilization'] = min(memory, 0.76)
    return result

def validate_same_node_capture_devices(environments):
    """Validate all producer instances on one node before starting any of them."""
    environments = tuple(environments)
    if not 1 <= len(environments) <= 2:
        raise ValueError('one capture node supports one TP16 or up to two TP8 instances')
    groups = []
    used = set()
    for environment in environments:
        tp = capture_target_parallelism(environment)
        start = int(environment.get('SPECO_CAPTURE_DEVICE_START', '0'))
        group = tuple(range(start, start + tp))
        if used.intersection(group):
            raise ValueError('capture instances on the same node overlap their device groups')
        used.update(group)
        groups.append(group)
    return tuple(groups)

def capture_graph_engine_options(environment: Mapping[str, str], *, max_num_seqs: int=1) -> dict:
    """Preserve old snapshots; explicit production FULL has no eager fallback."""
    if type(max_num_seqs) is not int or not 1 <= max_num_seqs <= 128:
        raise ValueError('invalid graph capture sequence bound')
    mode = environment.get('SPECO_CAPTURE_GRAPH_MODE', '')
    if type(mode) is not str or mode not in ('', 'FULL_DECODE_ONLY'):
        raise ValueError('capture graph mode must be FULL_DECODE_ONLY when configured')
    if not mode:
        return dict(enforce_eager=True)
    if environment.get('SPECO_TARGET_TP') not in ('8', '16') or environment.get('SPECO_CAPTURE_REUSE_ENGINE') != '1' or environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1' or (environment.get('SPECO_SEQUENCE_FULL_DECODE') != '1'):
        raise ValueError('FULL graph capture requires explicit persistent native TP8 or TP16')
    sizes = sorted({1, max_num_seqs, *[n for n in (2, 4, 8, 16, 32, 64, 128) if n <= max_num_seqs]})
    return dict(enforce_eager=False, compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': sizes})

def capture_graph_worker_extension(environment: Mapping[str, str]) -> str:
    capture_graph_engine_options(environment)
    if environment.get('SPECO_CAPTURE_GRAPH_MODE') == 'FULL_DECODE_ONLY':
        return 'verl_speco.producer.sequence_graph_worker.SpecoFullDecodeSequenceWorkerExtension'
    return 'verl_speco.producer.capture.SpecoMultiNodeCaptureWorkerExtension'

def capture_mhc_sp_token_bound(environment):
    """Match the admitted engine batch, including the explicit wide FULL TP8 profile."""
    options = capture_prefill_engine_options(environment)
    if environment.get('SPECO_MHC_SEQUENCE_PARALLEL') != '1':
        raise ValueError('mHC SP token bound requires the configured sequence parallel path')
    tp = int(environment['SPECO_TARGET_TP'])
    return max(tp * 8192, options.get('max_num_batched_tokens', 0))
import hashlib
from verl_speco.runtime.artifacts import fcntl
import json
import math
from pathlib import Path
import re
import time
from contextlib import contextmanager
from typing import Callable
from verl_speco.runtime.artifacts import read_published_json, publish_json

@contextmanager
def capture_service_capture_owner_lock(directory):
    with (directory / '.owner.lock').open('a') as owner:
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('persistent capture command owner is already running') from error
        try:
            yield
        finally:
            fcntl.flock(owner.fileno(), fcntl.LOCK_UN)

def validate_capture_command(command, *, run_id, generation, producer_index, producer_count, step_index, manifest_root):
    """Bind commands to the configured source and prevent arbitrary environment edits."""
    if not isinstance(command, dict):
        raise ValueError('capture command must be an object')
    expected = dict(schema_version=1, run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=step_index, sequence_no=producer_index + step_index * producer_count)
    if set(command) != set(expected) or any((command[k] != v or type(command[k]) is not type(v) for k, v in expected.items())):
        raise ValueError('capture command identity or sequence differs from configured owner')
    return Path(manifest_root) / f"capture-{expected['sequence_no']:04d}.json"

def service_step_range(total_steps, start_step=0):
    """Preserve global step/sequence numbering for a fresh resumed owner."""
    if type(total_steps) is not int or total_steps < 1 or type(start_step) is not int or (not 0 <= start_step < total_steps):
        raise ValueError('invalid persistent service step range')
    return range(start_step, total_steps)

def serve_capture_commands(*, command_dir: Path, manifest_root: Path, run_id: str, generation: int, producer_index: int, producer_count: int, total_steps: int, capture: Callable[[int, Path], None], timeout_seconds: float=1800, poll_seconds: float=0.5, after_publish: Callable[[], None] | None=None, start_step: int=0):
    """Capture only explicitly requested batches; acknowledge durable TP receipts.

    This service never trains, executes shell commands or alters model configuration.
    A restarted owner needs a fresh command directory; it cannot pretend its engine
    survived a prior process or replay already acknowledged commands.
    """
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]*', run_id) is None or type(generation) is not int or generation < 1 or (type(producer_count) is not int) or (not 1 <= producer_count <= MAX_CAPTURE_PRODUCER_INSTANCES) or (type(producer_index) is not int) or (not 0 <= producer_index < producer_count) or (type(total_steps) is not int) or (total_steps < 1) or (not math.isfinite(poll_seconds)) or (not math.isfinite(timeout_seconds)) or (not 0 < poll_seconds <= 30) or (timeout_seconds <= 0):
        raise ValueError('invalid persistent capture owner or timing configuration')
    if not command_dir.is_absolute() or not manifest_root.is_absolute() or (not manifest_root.resolve().is_relative_to('/tmp')):
        raise ValueError('capture commands must be absolute and manifests must stay under /tmp')
    steps = service_step_range(total_steps, start_step)
    command_dir.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    with capture_service_capture_owner_lock(command_dir):
        for step in steps:
            command_path = command_dir / f'step-{step:04d}.request.json'
            ack_path = command_dir / f'step-{step:04d}.ack.json'
            try:
                read_published_json(ack_path)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError('capture owner cannot restart with existing acknowledgements')
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    command = read_published_json(command_path)
                    break
                except FileNotFoundError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('persistent capture command timed out')
                    time.sleep(poll_seconds)
            manifest_path = validate_capture_command(command, run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=step, manifest_root=manifest_root)
            if manifest_path.exists():
                raise RuntimeError('capture manifest already exists before this owner executes command')
            started = time.monotonic()
            capture(command['sequence_no'], manifest_path)
            data = manifest_path.read_bytes()
            manifest = json.loads(data)
            metadata = manifest['metadata']
            proofs = manifest['worker_receipts']
            if metadata['run_id'] != run_id or metadata['generation'] != generation or metadata['producer_id'] != f'producer-{producer_index}' or (metadata['sequence_no'] != command['sequence_no']) or (manifest['worker_count'] not in (8, 16)) or (sorted((proof['rank'] for proof in proofs)) != list(range(manifest['worker_count']))) or (proofs[0]['status'] != 'published') or (proofs[0]['key'] != manifest['key']) or (len({proof['payload_checksum'] for proof in proofs}) != 1) or (re.fullmatch('[0-9a-f]{64}', proofs[0]['payload_checksum']) is None):
                raise RuntimeError('persistent capture manifest does not prove complete publication')
            publish_json(ack_path, dict(command, manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(data).hexdigest(), token_count=metadata['token_count'], capture_seconds=time.monotonic() - started, state='PREFILL_PUBLISHED'))
            if after_publish is not None:
                after_publish()
from verl_speco.runtime.artifacts import publish_json, read_published_json

def request_capture_batch(*, command_root: Path, manifest_root: Path, run_id: str, generation: int, step_index: int, producer_count: int, timeout_seconds: float=1200) -> list[dict]:
    if not command_root.is_absolute() or type(producer_count) is not int or (not 1 <= producer_count <= 7) or (type(step_index) is not int) or (step_index < 0) or (not math.isfinite(timeout_seconds)) or (timeout_seconds <= 0):
        raise ValueError('invalid capture service batch configuration')
    expected = []
    unpublished = []
    for index in range(producer_count):
        directory = command_root / f'producer-{index}'
        directory.mkdir(parents=True, exist_ok=True)
        command = dict(schema_version=1, run_id=run_id, generation=generation, producer_index=index, producer_count=producer_count, step_index=step_index, sequence_no=index + step_index * producer_count)
        validate_capture_command(command, run_id=run_id, generation=generation, producer_index=index, producer_count=producer_count, step_index=step_index, manifest_root=manifest_root)
        path = directory / f'step-{step_index:04d}.request.json'
        try:
            previous = read_published_json(path)
        except FileNotFoundError:
            unpublished.append((path, command))
        else:
            if previous != command:
                raise RuntimeError('existing capture request differs from owner identity')
        expected.append((directory / f'step-{step_index:04d}.ack.json', command))
    for path, command in unpublished:
        publish_json(path, command)
    deadline = time.monotonic() + timeout_seconds
    acknowledgements = {}
    while len(acknowledgements) != producer_count:
        for path, command in expected:
            if path in acknowledgements:
                continue
            try:
                ack = read_published_json(path)
            except FileNotFoundError:
                continue
            if not isinstance(ack, dict) or any((ack.get(k) != v or type(ack.get(k)) is not type(v) for k, v in command.items())) or ack.get('state') != 'PREFILL_PUBLISHED' or (ack.get('manifest_path') != str(manifest_root / f"capture-{command['sequence_no']:04d}.json")) or (not isinstance(ack.get('manifest_sha256'), str)) or (re.fullmatch('[0-9a-f]{64}', ack['manifest_sha256']) is None) or (type(ack.get('token_count')) is not int) or (ack['token_count'] <= 0) or (not isinstance(ack.get('capture_seconds'), (int, float))) or (not math.isfinite(ack['capture_seconds'])) or (ack['capture_seconds'] < 0):
                raise RuntimeError('capture service acknowledgement differs from requested batch')
            acknowledgements[path] = ack
        if len(acknowledgements) == producer_count:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('persistent producer batch acknowledgement timed out')
        time.sleep(0.5)
    return [acknowledgements[path] for path, _ in expected]

def build_physical_capture_command(*, run_id, generation, producer_index, producer_count, step_index, sample_count):
    """Bind up to sixteen sparse sequence positions to their existing producer."""
    if not isinstance(run_id, str) or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]*', run_id) is None or type(generation) is not int or (generation < 1) or (type(producer_count) is not int) or (not 1 <= producer_count <= MAX_CAPTURE_PRODUCER_INSTANCES) or (type(producer_index) is not int) or (not 0 <= producer_index < producer_count) or (type(step_index) is not int) or (step_index < 0) or (type(sample_count) is not int) or (not 1 <= sample_count <= 16):
        raise ValueError('invalid physical capture owner or batch')
    indices = [producer_index + (step_index + i) * producer_count for i in range(sample_count)]
    if indices[-1] >= 2 ** 63:
        raise ValueError('physical capture sequence index exceeds bound')
    return dict(schema_version=2, run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=step_index, sequence_nos=indices)

def validate_physical_capture_command(command, *, run_id, generation, producer_index, producer_count, step_index, remaining_steps):
    if not isinstance(command, dict) or not isinstance(command.get('sequence_nos'), list) or (not 1 <= len(command['sequence_nos']) <= min(16, remaining_steps)):
        raise ValueError('physical capture batch exceeds remaining owner rounds')
    expected = build_physical_capture_command(run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=step_index, sample_count=len(command['sequence_nos']))
    if set(command) != set(expected) or any((type(command[k]) is not type(v) or command[k] != v for k, v in expected.items())) or any((type(i) is not int for i in command['sequence_nos'])):
        raise ValueError('physical capture command identity or sequence differs from owner')
    return tuple(expected['sequence_nos'])

def physical_capture_service_manifest_proof(path, *, run_id, generation, producer_index, sequence_no, source_capture_owner_ranks=0, physical_child_index=0, physical_batch_size=1, source_native_buffer_batches=1, physical_slots=8, expected_target_tp=None, source_slot_profile=''):
    data = path.read_bytes()
    manifest = json.loads(data)
    metadata = manifest['metadata']
    proofs = manifest['worker_receipts']
    count = manifest['worker_count']
    if type(count) is not int or count not in (8, 16) or (expected_target_tp is not None and count != expected_target_tp) or (metadata['run_id'] != run_id) or (type(metadata['generation']) is not int) or (metadata['generation'] != generation) or (metadata['producer_id'] != f'producer-{producer_index}') or (type(metadata['sequence_no']) is not int) or (metadata['sequence_no'] != sequence_no) or (type(metadata['token_count']) is not int) or (metadata['token_count'] < 1) or (type(metadata['payload_nbytes']) is not int) or (metadata['payload_nbytes'] < 1):
        raise RuntimeError('physical manifest does not prove complete child publication')
    from verl_speco.producer.capture import validate_child_publication, validate_source_capture_owners
    validate_source_capture_owners(source_capture_owner_ranks, target_tp=count, physical_slots=physical_slots)
    from verl_speco.producer.capture import validate_source_capture_buffers
    validate_source_capture_buffers(source_native_buffer_batches, owner_ranks=source_capture_owner_ranks, target_tp=count, physical_slots=physical_slots, wide_full_profile=source_native_buffer_batches in (3, 4, 10), source_slot_profile=source_slot_profile)
    if source_native_buffer_batches in (2, 3, 4, 10):
        if type(manifest.get('source_native_buffer_batches')) is not int or manifest['source_native_buffer_batches'] != source_native_buffer_batches or type(manifest.get('registered_source_buffers')) is not int or (manifest['registered_source_buffers'] != physical_slots * source_native_buffer_batches):
            raise RuntimeError('physical manifest registered buffer budget differs from its command')
    elif any((k in manifest for k in ('source_native_buffer_batches', 'registered_source_buffers'))):
        raise RuntimeError('unexpected double native buffer manifest')
    if source_slot_profile:
        from verl_speco.producer.capture import SLOT_SIZES, OWNER_BYTES
        if manifest.get('source_native_slot_profile') != source_slot_profile or type(manifest.get('source_slot_sizes')) is not list or manifest['source_slot_sizes'] != list(SLOT_SIZES) or any((type(n) is not int for n in manifest['source_slot_sizes'])) or (type(manifest.get('registered_source_bytes')) is not int) or (manifest['registered_source_bytes'] != 8 * OWNER_BYTES):
            raise RuntimeError('physical manifest tiered registration bytes differ from the frozen profile')
    elif any((k in manifest for k in ('source_native_slot_profile', 'source_slot_sizes', 'registered_source_bytes'))):
        raise RuntimeError('unexpected tiered registration manifest without explicit source profile')
    owner = None
    if source_capture_owner_ranks and physical_batch_size > 1:
        owner = physical_child_index % source_capture_owner_ranks
        if type(manifest.get('source_capture_owner_ranks')) is not int or manifest['source_capture_owner_ranks'] != source_capture_owner_ranks or type(manifest.get('capture_owner_rank')) is not int or (manifest['capture_owner_rank'] != owner) or (type(manifest.get('physical_child_index')) is not int) or (manifest['physical_child_index'] != physical_child_index) or (type(manifest.get('payload_digest_workers')) is not int) or (manifest['payload_digest_workers'] != 1) or (type(manifest.get('physical_batch_size')) is not int) or (manifest['physical_batch_size'] != physical_batch_size):
            raise RuntimeError('physical manifest distributed owner differs from its command')
    elif any((k in manifest for k in ('source_capture_owner_ranks', 'capture_owner_rank', 'physical_child_index', 'payload_digest_workers'))):
        raise RuntimeError('unexpected distributed ownership manifest in full TP capture mode')
    validate_child_publication(proofs, target_tp=count, key=manifest['key'], owner_rank=owner, request_id=metadata.get('request_id'))
    return dict(sequence_no=sequence_no, manifest_path=str(path), manifest_sha256=hashlib.sha256(data).hexdigest(), token_count=metadata['token_count'], payload_nbytes=metadata['payload_nbytes'])

def serve_physical_capture_commands(*, command_dir, manifest_root, run_id, generation, producer_index, producer_count, total_steps, capture_many, start_step=0, max_batch_bytes=4 * 1024 ** 3, timeout_seconds=1800, poll_seconds=0.5, after_publish=None, source_capture_owner_ranks=0, source_native_buffer_batches=1, target_tp=None, physical_slots=8, capture_complete=None, source_slot_profile=''):
    """ACK an entire physical batch only after every child's durable TP proof.

    Global steps advance by the commanded count, including single-child tails
    at optimizer boundaries. Restart requires fresh owner files; partial child
    publication is never mistaken for a complete batch. Lease draining remains
    with the capture owner, after the batch publication ACK.
    """
    build_physical_capture_command(run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=start_step, sample_count=1)
    service_step_range(total_steps, start_step)
    if target_tp is not None and (type(target_tp) is not int or target_tp not in (8, 16)) or type(physical_slots) is not int or physical_slots not in (2, 4, 8, 16):
        raise ValueError('invalid physical capture mesh or fixed slot count')
    policy_tp = 8 if target_tp is None else target_tp
    from verl_speco.producer.capture import validate_source_capture_owners
    validate_source_capture_owners(source_capture_owner_ranks, target_tp=policy_tp, physical_slots=physical_slots)
    from verl_speco.producer.capture import validate_source_capture_buffers
    validate_source_capture_buffers(source_native_buffer_batches, owner_ranks=source_capture_owner_ranks, target_tp=policy_tp, physical_slots=physical_slots, wide_full_profile=source_native_buffer_batches in (3, 4, 10), source_slot_profile=source_slot_profile)
    if type(max_batch_bytes) is not int or max_batch_bytes < 1 or (not callable(capture_many)) or (type(timeout_seconds) not in (int, float)) or (not math.isfinite(timeout_seconds)) or (timeout_seconds <= 0) or (type(poll_seconds) not in (int, float)) or (not math.isfinite(poll_seconds)) or (not 0 < poll_seconds <= 30):
        raise ValueError('invalid physical capture budget or timing')
    if capture_complete is not None and (not (callable(capture_complete) and producer_count == 6 and (policy_tp == 8) and (physical_slots == 8) and (source_capture_owner_ranks == 8) and (source_native_buffer_batches in (4, 10)) and (after_publish is None))):
        raise ValueError('async command service requires the fixed six-source native profile')
    command_dir, manifest_root = (Path(command_dir), Path(manifest_root))
    if not command_dir.is_absolute() or not manifest_root.is_absolute() or (not manifest_root.resolve().is_relative_to('/tmp')):
        raise ValueError('physical capture needs absolute commands and /tmp manifests')
    command_dir.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    with capture_service_capture_owner_lock(command_dir):
        pending = []

        def finish_oldest():
            command, indices, paths, ack_path, started, ticket = pending[0]
            if ticket is not None:
                capture_complete(ticket)
            proofs = [physical_capture_service_manifest_proof(p, run_id=run_id, generation=generation, producer_index=producer_index, sequence_no=i, source_capture_owner_ranks=source_capture_owner_ranks, physical_child_index=index, physical_batch_size=len(indices), source_native_buffer_batches=source_native_buffer_batches, physical_slots=physical_slots, expected_target_tp=target_tp, source_slot_profile=source_slot_profile) for index, (i, p) in enumerate(zip(indices, paths))]
            if sum((p['payload_nbytes'] for p in proofs)) > max_batch_bytes:
                raise RuntimeError('physical capture batch publication exceeds byte budget')
            publish_json(ack_path, dict(command, manifests=proofs, state='PREFILL_BATCH_PUBLISHED', capture_seconds=time.monotonic() - started))
            if after_publish is not None:
                after_publish()
            pending.pop(0)
        step = start_step
        while step < total_steps:
            if capture_complete is not None and step % 160 == 0:
                while pending:
                    finish_oldest()
            command_path = command_dir / f'step-{step:04d}.request.json'
            ack_path = command_dir / f'step-{step:04d}.ack.json'
            try:
                read_published_json(ack_path)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError('physical capture owner cannot restart with acknowledgements')
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    command = read_published_json(command_path)
                    break
                except FileNotFoundError:
                    if pending:
                        finish_oldest()
                        continue
                    if time.monotonic() >= deadline:
                        raise TimeoutError('physical capture command timed out')
                    time.sleep(min(poll_seconds, max(0.001, deadline - time.monotonic())))
            indices = validate_physical_capture_command(command, run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=step, remaining_steps=total_steps - step)
            if source_capture_owner_ranks and len(indices) > physical_slots:
                raise ValueError('physical command exceeds its distributed fixed slot layout')
            paths = tuple((manifest_root / f'capture-{i:04d}.json' for i in indices))
            if any((p.exists() for p in paths)):
                raise RuntimeError('physical capture child manifest already exists')
            started = time.monotonic()
            result = capture_many(indices, paths)
            ticket = result if capture_complete is not None else None
            pending.append((command, indices, paths, ack_path, started, ticket))
            if capture_complete is None or ticket is None or len(pending) >= 2 or ((step + len(indices)) % 160 == 0):
                finish_oldest()
            step += len(indices)
        while pending:
            finish_oldest()
from verl_speco.producer.batching import CaptureCapacityUnavailable, KvAwareCaptureBudget
from verl_speco.producer.batching import TokenBudgetCommandCoalescer

def serve_coalesced_physical_capture_commands(*, command_dir, manifest_root, run_id, generation, producer_index, producer_count, total_steps, start_step, total_samples, token_count, capture_many, timeout_seconds=1800, poll_seconds=0.05, token_fill_target=0.9, cross_step=False, token_budget=131072, capacity_provider=None, budget_policy=None, capture_complete=None, source_native_buffer_batches=4, source_slot_profile=''):
    from verl_speco.producer.capture import validate_source_capture_buffers
    validate_source_capture_buffers(source_native_buffer_batches, owner_ranks=8, target_tp=8, physical_slots=8, wide_full_profile=True, source_slot_profile=source_slot_profile)
    if source_native_buffer_batches not in (4, 10):
        raise ValueError('coalesced service needs the explicit logical-eight source pool')
    service_step_range(total_steps, start_step)
    command_dir = Path(command_dir)
    manifest_root = Path(manifest_root)
    if not command_dir.is_absolute() or not manifest_root.is_absolute() or (not manifest_root.resolve().is_relative_to('/tmp')) or (not callable(token_count)) or (not callable(capture_many)):
        raise ValueError('coalesced capture needs owned paths and admitted frozen callbacks')
    coalescer = TokenBudgetCommandCoalescer(run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, total_samples=total_samples, start_round=start_step, target_fill=token_fill_target, cross_step=cross_step, token_budget=token_budget)
    command_dir.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    if (capacity_provider is None) != (budget_policy is None) or (capacity_provider is not None and (not callable(capacity_provider) or not isinstance(budget_policy, KvAwareCaptureBudget) or cross_step)):
        raise ValueError('adaptive service requires an explicit same-step capacity provider and policy')
    pipeline = None
    if capture_complete is not None:
        if not callable(capture_complete) or capacity_provider is None or budget_policy.max_sequences != 16 or cross_step:
            raise ValueError('async coalescing needs the exact KV-bounded16 completion callback')
        from verl_speco.producer.capture import PipelinedSourceCommandCoalescer
        pipeline = PipelinedSourceCommandCoalescer(coalescer)
    pending = []
    loaded = {}

    def load(step):
        if step >= total_steps:
            raise ValueError('coalesced source reads beyond its admitted execution stage')
        path = command_dir / f'step-{step:04d}.request.json'
        ack = command_dir / f'step-{step:04d}.ack.json'
        try:
            read_published_json(ack)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError('coalesced source cannot reuse an acknowledged command')
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                command = read_published_json(path)
                break
            except FileNotFoundError:
                finish_pending()
                if time.monotonic() >= deadline:
                    raise TimeoutError('coalesced source command timed out')
                time.sleep(min(poll_seconds, max(0.001, deadline - time.monotonic())))
        loaded[step] = time.monotonic()
        return command

    def acknowledge(command, proofs):
        if sum((p['payload_nbytes'] for p in proofs)) > 4 * 1024 ** 3:
            raise RuntimeError('logical command publication exceeds its byte budget')
        publish_json(command_dir / f"step-{command['step_index']:04d}.ack.json", dict(command, manifests=list(proofs), state='PREFILL_BATCH_PUBLISHED', capture_seconds=time.monotonic() - loaded.pop(command['step_index'])))

    def finish_batch(batch, paths, completion_ticket):
        if completion_ticket is not None:
            capture_complete(completion_ticket)
        children = batch['children']
        proofs = [physical_capture_service_manifest_proof(path, run_id=run_id, generation=generation, producer_index=producer_index, sequence_no=c.sequence_no, source_capture_owner_ranks=8, physical_child_index=c.physical_child_index, physical_batch_size=c.physical_batch_size, source_native_buffer_batches=source_native_buffer_batches, physical_slots=8, expected_target_tp=8, source_slot_profile=source_slot_profile) for path, c in zip(paths, children)]
        if pipeline is None:
            coalescer.commit_batch(proofs, acknowledge=acknowledge)
        else:
            pipeline.commit_batch(batch, proofs, acknowledge=acknowledge)

    def finish_pending():
        while pending:
            batch, paths, ticket = pending[0]
            finish_batch(batch, paths, ticket)
            pending.pop(0)
    with capture_service_capture_owner_lock(command_dir):
        capacity_deadline = time.monotonic() + timeout_seconds
        while coalescer.next_round < total_steps or coalescer.pending:
            options = {} if capacity_provider is None else dict(capacity_provider=capacity_provider, budget_policy=budget_policy, now=time.monotonic)
            try:
                batch = (pipeline or coalescer).next_batch(load_command=load, token_count=token_count, **options)
            except CaptureCapacityUnavailable:
                finish_pending()
                if time.monotonic() >= capacity_deadline:
                    raise TimeoutError('complete first source sequence still exceeds actual KV/capture capacity')
                time.sleep(poll_seconds)
                continue
            if batch is None:
                break
            children = batch['children']
            indices = tuple((c.sequence_no for c in children))
            if capacity_provider is not None:
                print('source_KV_capacity_admission ' + json.dumps(dict(run_id=run_id, generation=generation, producer_index=producer_index, sequence_nos=indices, original_tokens=batch['original_tokens'], capacity_admission=batch['capacity_admission'], max_sequences=budget_policy.max_sequences, reserve_kv_blocks=budget_policy.reserve_kv_blocks, reserve_rank_bytes=budget_policy.reserve_rank_bytes, committed_original_tokens=0)), flush=True)
            paths = tuple((manifest_root / f'capture-{i:04d}.json' for i in indices))
            if any((p.exists() for p in paths)):
                raise RuntimeError('coalesced child manifest already exists')
            if pending and batch['optimizer_step'] != pending[0][0]['optimizer_step']:
                finish_pending()
            if pipeline is None:
                capture_many(indices, paths, logical_children=children)
                finish_batch(batch, paths, None)
            else:
                ticket = capture_many(indices, paths, logical_children=children, finish_pending_before_enqueue=finish_pending)
                if pending:
                    raise RuntimeError('source callback did not fence the previous publication before enqueue')
                if ticket is None:
                    finish_batch(batch, paths, None)
                else:
                    pending.append((batch, paths, ticket))
            capacity_deadline = time.monotonic() + timeout_seconds
        finish_pending()
        if pipeline is not None:
            pipeline.require_complete()
        if coalescer.pending or coalescer.commands or coalescer.active or loaded:
            raise RuntimeError('coalesced source ended with incomplete logical publication')
import os
import tempfile
import uuid
SHARED_STORAGE = 'shared-filesystem-v1'
MAX_SHARED_SNAPSHOT_BYTES = 4 * 1024 * 1024 * 1024

def control_snapshot_storage(environment):
    mode = environment.get('SPECO_CONTROL_SNAPSHOT_STORAGE', '')
    if mode not in ('', SHARED_STORAGE):
        raise ValueError('unknown control snapshot storage')
    return mode

def shared_control_directory(*, run_id, generation, project_root=None):
    if not isinstance(run_id, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', run_id) or type(generation) is not int or (generation < 0):
        raise ValueError('shared control snapshot identity is invalid')
    if project_root is None:
        import verl_speco
        project_root = Path(verl_speco.__file__).resolve().parent.parent
    root = Path(project_root).resolve()
    target = root / 'outputs' / f'speco-native-wave-{run_id}-g{generation}' / 'controls'
    if target.resolve() != target:
        raise ValueError('shared control snapshot directory cannot contain symlinks')
    return target

def control_snapshot_handoff_shared_path(path, *, run_id, generation, project_root=None):
    target = Path(path)
    directory = shared_control_directory(run_id=run_id, generation=generation, project_root=project_root)
    if not target.is_absolute() or target.parent != directory or target.resolve() != target or (not re.fullmatch('step-[0-9]{4,7}-control-[0-9a-f]{32}\\.json', target.name)):
        raise ValueError('shared control snapshot path is outside owned run controls')
    return target

def save_shared_control_snapshot(snapshot, *, directory, step_index, run_id, generation, project_root=None):
    """Write an immutable owned snapshot; publish its small descriptor after fsync.

    No rename or large Ray payload is needed. An interrupted write is never
    referenced by a committed receipt. All leases and acquired records survive.
    """
    expected = shared_control_directory(run_id=run_id, generation=generation, project_root=project_root)
    if Path(directory) != expected or type(step_index) is not int or (not 0 <= step_index <= 6000000):
        raise ValueError('shared control snapshot directory or step differs')
    if not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1 or snapshot.get('run_id') != run_id or (snapshot.get('generation') != generation):
        raise ValueError('shared control snapshot identity differs')
    target = control_snapshot_handoff_shared_path(expected / f'step-{step_index:04d}-control-{uuid.uuid4().hex}.json', run_id=run_id, generation=generation, project_root=project_root)
    expected.mkdir(parents=True, exist_ok=True)
    control_snapshot_handoff_shared_path(target, run_id=run_id, generation=generation, project_root=project_root)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
    digest, size = (hashlib.sha256(), 0)
    try:
        with os.fdopen(fd, 'wb') as handle:
            for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(snapshot):
                payload = chunk.encode('utf-8')
                size += len(payload)
                if size > MAX_SHARED_SNAPSHOT_BYTES:
                    raise ValueError('shared control snapshot exceeds bounded file capacity')
                handle.write(payload)
                digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(expected, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return dict(storage=SHARED_STORAGE, path=str(target), sha256=digest.hexdigest(), nbytes=size)

def read_shared_control_snapshot(descriptor, *, run_id, generation, project_root=None):
    """Verify the entire bounded file before handing its state to restore."""
    if not isinstance(descriptor, dict) or set(descriptor) != {'storage', 'path', 'sha256', 'nbytes'} or descriptor.get('storage') != SHARED_STORAGE or (type(descriptor.get('nbytes')) is not int) or (not 0 < descriptor['nbytes'] <= MAX_SHARED_SNAPSHOT_BYTES) or (not isinstance(descriptor.get('sha256'), str)) or (not re.fullmatch('[0-9a-f]{64}', descriptor['sha256'])):
        raise ValueError('shared control snapshot descriptor is invalid')
    target = control_snapshot_handoff_shared_path(descriptor['path'], run_id=run_id, generation=generation, project_root=project_root)
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        if os.fstat(handle.fileno()).st_size != descriptor['nbytes']:
            raise ValueError('shared control snapshot size differs')
        payload = handle.read(descriptor['nbytes'] + 1)
    if len(payload) != descriptor['nbytes'] or hashlib.sha256(payload).hexdigest() != descriptor['sha256']:
        raise ValueError('shared control snapshot checksum or size differs')
    snapshot = json.loads(payload)
    if not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1 or snapshot.get('run_id') != run_id or (snapshot.get('generation') != generation):
        raise ValueError('shared control snapshot identity differs')
    return snapshot

def persist_control_snapshot(path, bundle, *, run_id, generation):
    payload = bundle.get('payload')
    digest = bundle.get('sha256')
    if not isinstance(payload, bytes) or len(payload) > 16 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('control snapshot handoff checksum or size differs')
    snapshot = json.loads(payload)
    if snapshot.get('schema_version') != 1 or snapshot.get('run_id') != run_id or snapshot.get('generation') != generation:
        raise ValueError('control snapshot handoff identity differs')
    target = Path(path)
    if not target.is_absolute() or not target.is_relative_to('/tmp'):
        raise ValueError('control snapshot handoff requires an absolute /tmp path')
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError('local control snapshot conflicts with committed handoff')
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.control-handoff-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)
    return digest

def cross_step_prefill_sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None

def canonical_receipt_sha256(receipt):
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def cross_step_prefill_identity(run_id, generation, step_index, samples_per_step, logical, epoch_plan=None):
    limit = 32
    if epoch_plan is not None:
        from verl_speco.runtime.recovery import validate_epoch_plan_capacity
        validate_epoch_plan_capacity(epoch_plan)
        if samples_per_step != epoch_plan.samples_per_step:
            raise ValueError('cross-step batch differs from frozen epoch plan')
        limit = epoch_plan.total_steps
    if not isinstance(run_id, str) or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]*', run_id) is None or type(generation) is not int or (generation < 1) or (type(step_index) is not int) or (not 0 <= step_index < limit) or (type(samples_per_step) is not int) or (not 1 <= samples_per_step <= 1024) or (not cross_step_prefill_sha(logical)):
        raise ValueError('invalid cross-step prefill identity')

def publish_parent_commit_permit(*, gate_dir, receipt, run_id, generation, parent_step_index, samples_per_step, logical_checkpoint_fingerprint, next_batch_control_sha256, outputs_root, epoch_plan=None):
    cross_step_prefill_identity(run_id, generation, parent_step_index, samples_per_step, logical_checkpoint_fingerprint, epoch_plan)
    if not isinstance(receipt, dict):
        raise ValueError('cross-step parent receipt must be an object')
    root = Path(outputs_root).resolve()
    folder = Path(gate_dir).resolve()
    if not folder.is_relative_to(root) or folder == root or (not cross_step_prefill_sha(next_batch_control_sha256)):
        raise ValueError('cross-step permit must stay inside the role outputs')
    from verl_speco.transport.streaming import decode_stream_snapshot
    stream = decode_stream_snapshot(receipt.get('stream_snapshot', {})) if isinstance(receipt, dict) else {}
    keys = receipt.get('packed_sample_keys', []) if isinstance(receipt, dict) else []
    if not isinstance(stream, dict):
        raise RuntimeError('cross-step parent stream must be an object')
    commits = stream.get('committed_optimizer_steps', {})
    cursor = (parent_step_index + 1) * samples_per_step
    if type(receipt.get('schema_version')) is not int or receipt.get('schema_version') != 1 or receipt.get('run_id') != run_id or (type(receipt.get('generation')) is not int) or (receipt['generation'] != generation) or (type(receipt.get('draft_world_size')) is not int) or (receipt.get('draft_world_size') != 16) or (type(receipt.get('sample_index')) is not int) or (receipt['sample_index'] != parent_step_index * samples_per_step) or (type(stream.get('committed_cursor')) is not int) or (stream['committed_cursor'] != cursor) or (not isinstance(commits, dict)) or (receipt.get('step_id') not in commits) or (len(commits) != parent_step_index + 1) or (not isinstance(keys, list)) or (len(keys) != samples_per_step) or any((not isinstance(k, str) for k in keys)) or (len(set(keys)) != len(keys)) or (not cross_step_prefill_sha(receipt.get('draft_weights_sha256'))):
        raise RuntimeError('cross-step parent is not the expected committed DP16 batch')
    if epoch_plan is not None:
        next_step = parent_step_index + 1
        if next_step >= epoch_plan.total_steps:
            raise RuntimeError('final epoch commit has no future source batch')
        validate_variable_batch_cursor(run_id=run_id, generation=generation, step_index=next_step, start=cursor, batch_size=min(samples_per_step, epoch_plan.total_samples - cursor), total_steps=epoch_plan.total_steps, world_size=16, previous_receipt=receipt, epoch_plan=epoch_plan)
    candidate = receipt.get('candidate', {})
    if not isinstance(candidate, dict):
        raise RuntimeError('cross-step parent candidate must be an object')
    from verl_speco.runtime.checkpoint_policy import is_live_candidate, require_live_receipt
    if is_live_candidate(candidate):
        from verl_speco.runtime.checkpoint_policy import live_update_path
        if not live_update_path(candidate).is_relative_to(root):
            raise RuntimeError('live parent update is outside the role outputs')
        require_live_receipt(dict(receipt, stream_snapshot=stream), logical_checkpoint_fingerprint=logical_checkpoint_fingerprint)
    else:
        uri = Path(str(candidate.get('uri', '')))
        if not uri.is_absolute() or not uri.resolve().is_relative_to(root) or (not cross_step_prefill_sha(candidate.get('checksum'))) or (type(candidate.get('checkpoint_generation')) is not int) or (candidate['checkpoint_generation'] < 1):
            raise RuntimeError('cross-step parent candidate is outside the role outputs')
        raw = (uri / 'manifest.json').read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest.get('schema_version') != 1 or (hashlib.sha256(raw).hexdigest() != candidate['checksum']) or (manifest.get('logical_checkpoint_fingerprint') != logical_checkpoint_fingerprint) or (type(manifest.get('world_size')) is not int) or (manifest.get('world_size') != 16) or (type(manifest.get('generation')) is not int) or (manifest.get('generation') != candidate['checkpoint_generation']):
            raise RuntimeError('cross-step parent DCP manifest identity or checksum differs')
    parent = dict(schema_version=1, run_id=run_id, generation=generation, parent_step_index=parent_step_index, parent_cursor=cursor, next_step_index=parent_step_index + 1, logical_checkpoint_fingerprint=logical_checkpoint_fingerprint, next_batch_control_sha256=next_batch_control_sha256, parent_receipt_sha256=canonical_receipt_sha256(receipt), state='STEP_COMMITTED')
    if epoch_plan is not None:
        parent['epoch_plan_sha256'] = epoch_plan.plan_sha256
    path = folder / f'parent-{parent_step_index:04d}.commit.json'
    try:
        previous = read_published_json(path)
    except FileNotFoundError:
        folder.mkdir(parents=True, exist_ok=True)
        publish_json(path, parent)
    else:
        if previous != parent:
            raise RuntimeError('cross-step parent permit identity changed')
    return parent

class StepPrefillPublicationGate:

    def __init__(self, *, gate_dir, outputs_root, run_id, generation, producer_index, producer_count, samples_per_step, start_step, end_step, logical_checkpoint_fingerprint, batch_control_sha256s, timeout_seconds=1800, poll_seconds=0.5, parent_permit_transport='filesystem', fetch_parent_permit=None, epoch_plan=None, early_ready=False, frozen_input_sha256='', max_capture_children=16, allow_cross_step=False):
        cross_step_prefill_identity(run_id, generation, start_step, samples_per_step, logical_checkpoint_fingerprint, epoch_plan)
        limit = 32
        valid_controls = False
        if epoch_plan is None:
            valid_controls = isinstance(batch_control_sha256s, list) and type(end_step) is int and (end_step <= len(batch_control_sha256s) <= 32) and all((cross_step_prefill_sha(v) for v in batch_control_sha256s))
        else:
            from verl_speco.producer.capture import EpochNativeBatches
            limit = epoch_plan.total_steps
            valid_controls = isinstance(batch_control_sha256s, EpochNativeBatches) and batch_control_sha256s.plan.plan_sha256 == epoch_plan.plan_sha256 and (batch_control_sha256s.run_id == run_id) and (batch_control_sha256s.generation == generation) and (producer_count == epoch_plan.producer_count)
        root = Path(outputs_root).resolve()
        folder = Path(gate_dir).resolve()
        if not folder.is_relative_to(root) or folder == root or type(producer_count) is not int or (not 1 <= producer_count <= 14) or (type(producer_index) is not int) or (not 0 <= producer_index < producer_count) or samples_per_step % producer_count or (type(end_step) is not int) or (not start_step < end_step <= limit) or (not valid_controls) or (type(timeout_seconds) not in (int, float)) or (not math.isfinite(timeout_seconds)) or (timeout_seconds <= 0) or (type(poll_seconds) not in (int, float)) or (not math.isfinite(poll_seconds)) or (not 0 < poll_seconds <= 30):
            raise ValueError('invalid bounded cross-step prefill publication gate')
        if parent_permit_transport not in ('filesystem', 'ray') or (fetch_parent_permit is not None and (not callable(fetch_parent_permit))):
            raise ValueError('invalid parent commit permit transport')
        if type(early_ready) is not bool or (early_ready and (not cross_step_prefill_sha(frozen_input_sha256))) or (not early_ready and frozen_input_sha256):
            raise ValueError('early prefill needs an explicit frozen input identity')
        self.early_ready = early_ready
        self.frozen_input_sha256 = frozen_input_sha256
        self.admitted = set()
        self.parent_permit_transport = parent_permit_transport
        self.fetch_parent_permit = fetch_parent_permit
        self.folder = folder
        self.run_id = run_id
        self.generation = generation
        self.producer_index = producer_index
        self.producer_count = producer_count
        self.samples = samples_per_step
        self.first_step = start_step
        self.end_step = end_step
        self.logical = logical_checkpoint_fingerprint
        self.epoch_plan = epoch_plan
        self.controls = tuple(batch_control_sha256s) if epoch_plan is None else batch_control_sha256s
        self.timeout = timeout_seconds
        self.poll = poll_seconds
        if type(max_capture_children) is not int or max_capture_children not in (16, 40) or (max_capture_children == 40 and (not (epoch_plan is not None and early_ready and (producer_count == 6) and (samples_per_step == 960)))):
            raise ValueError('wide capture gate requires its explicit frozen epoch identity')
        if type(allow_cross_step) is not bool or (allow_cross_step and (not (early_ready and epoch_plan is not None and (max_capture_children == 40)))):
            raise ValueError('cross-step token batches need explicit frozen early-ready40 admission')
        self.allow_cross_step = allow_cross_step
        self._active_steps = ()
        self.max_capture_children = max_capture_children
        self.next_round = start_step * samples_per_step // producer_count
        self.active = None
        self.failed = False
        self.verified = set()

    def begin_capture(self, indices, *, release_previous_sources=None):
        if self.failed or self.active is not None:
            raise RuntimeError('cross-step capture gate is busy or poisoned')
        indices = tuple(indices)
        expected = tuple((self.producer_index + (self.next_round + i) * self.producer_count for i in range(len(indices))))
        if not 1 <= len(indices) <= self.max_capture_children or any((type(i) is not int for i in indices)) or indices != expected or (not self.allow_cross_step and indices[0] // self.samples != indices[-1] // self.samples) or (indices[-1] // self.samples >= self.end_step) or (indices[-1] // self.samples - indices[0] // self.samples > 1) or (not self.first_step <= indices[0] // self.samples < self.end_step) or (self.epoch_plan is not None and indices[-1] >= self.epoch_plan.total_samples):
            self.failed = True
            raise ValueError('cross-step capture sequence or optimizer boundary differs')
        from itertools import groupby
        self._active_steps = tuple(((step, len(tuple(group))) for step, group in groupby(indices, key=lambda i: i // self.samples)))
        self.active = self._active_steps[0][0]
        self.active_samples = self._active_steps[0][1]
        self.next_round += len(indices)
        try:
            if release_previous_sources is not None:
                if not callable(release_previous_sources):
                    raise ValueError('cross-step source release callback must be callable')
                if self.active > self.first_step and indices[0] % self.samples == self.producer_index:
                    self._check_abort()
                    release_previous_sources()
        except BaseException:
            self.failed = True
            raise

    def _check_abort(self):
        try:
            abort = read_published_json(self.folder / 'abort.json', max_bytes=4096)
        except FileNotFoundError:
            return
        expected = dict(schema_version=1, run_id=self.run_id, generation=self.generation, state='ABORTED', committed_original_tokens=0)
        if abort != expected or any((type(abort[k]) is not type(v) for k, v in expected.items())):
            raise RuntimeError('cross-step abort identity differs')
        raise RuntimeError('cross-step prefill aborted by coordinator')

    def _control_sha256(self, step):
        return self.controls[step] if self.epoch_plan is None else self.controls.control_sha256(step)

    def before_publish(self):
        if self.failed or self.active is None:
            raise RuntimeError('cross-step capture gate is inactive or poisoned')
        for step, count in self._active_steps:
            self.active = step
            self.active_samples = count
            self._before_publish_one()
        self._active_steps = ()

    def _before_publish_one(self):
        if self.failed or self.active is None:
            raise RuntimeError('cross-step capture gate is inactive or poisoned')
        try:
            self._check_abort()
            step = self.active
            wait_start = time.perf_counter()
            if self.early_ready and step > self.first_step:
                expected = prefill_admission(run_id=self.run_id, generation=self.generation, next_step_index=step, logical_checkpoint_fingerprint=self.logical, current_batch_control_sha256=self._control_sha256(step - 1), next_batch_control_sha256=self._control_sha256(step), frozen_input_sha256=self.frozen_input_sha256, epoch_plan=self.epoch_plan)
                if step not in self.admitted:
                    deadline = time.monotonic() + self.timeout
                    while True:
                        self._check_abort()
                        try:
                            permit = read_published_json(self.folder / f'prefill-{step:04d}.admit.json', max_bytes=8192)
                            break
                        except FileNotFoundError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError('frozen prefill admission unavailable')
                            time.sleep(min(self.poll, max(0.001, deadline - time.monotonic())))
                    validate_prefill_admission(permit, expected)
                    self.admitted.add(step)
            if not self.early_ready and step > self.first_step and (step not in self.verified):
                print('cross_step_prefill_gate_wait ' + json.dumps(dict(run_id=self.run_id, generation=self.generation, producer_index=self.producer_index, step_index=step, captured_samples=self.active_samples, captured_source_wave_before_publication=True, future_payload_published=False, wallclock_time=time.time(), committed_original_tokens=0)), flush=True)
                deadline = time.monotonic() + self.timeout
                while True:
                    self._check_abort()
                    try:
                        if self.parent_permit_transport == 'ray':
                            if self.fetch_parent_permit is None:
                                raise RuntimeError('Ray parent permit callback unavailable')
                            parent = self.fetch_parent_permit(step - 1)
                            if parent is None:
                                raise FileNotFoundError('parent not committed')
                        else:
                            parent = read_published_json(self.folder / f'parent-{step - 1:04d}.commit.json', max_bytes=4096)
                    except FileNotFoundError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('cross-step prefill parent commit timed out')
                        time.sleep(min(self.poll, max(0.001, deadline - time.monotonic())))
                        continue
                    if not isinstance(parent, dict):
                        raise RuntimeError('cross-step prefill parent permit must be an object')
                    expected = dict(schema_version=1, run_id=self.run_id, generation=self.generation, parent_step_index=step - 1, parent_cursor=step * self.samples, next_step_index=step, logical_checkpoint_fingerprint=self.logical, next_batch_control_sha256=self.controls[step] if self.epoch_plan is None else self.controls.control_sha256(step), parent_receipt_sha256=parent.get('parent_receipt_sha256'), state='STEP_COMMITTED')
                    if self.epoch_plan is not None:
                        expected['epoch_plan_sha256'] = self.epoch_plan.plan_sha256
                    if not isinstance(parent, dict) or set(parent) != set(expected) or (not cross_step_prefill_sha(expected['parent_receipt_sha256'])) or any((parent[k] != v or type(parent[k]) is not type(v) for k, v in expected.items())):
                        raise RuntimeError('cross-step prefill parent permit identity differs')
                    self.verified.add(step)
                    break
            print('cross_step_prefill_gate_timing ' + json.dumps(dict(run_id=self.run_id, generation=self.generation, producer_index=self.producer_index, step_index=step, captured_samples=self.active_samples, parent_commit_required=step > self.first_step and (not self.early_ready), parent_permit_verified=step in self.verified, prefill_only_admitted=self.early_ready and step in self.admitted, parent_commit_permit_wait_seconds=time.perf_counter() - wait_start, wallclock_time=time.time(), committed_original_tokens=0)), flush=True)
            self.active = None
        except BaseException:
            self.failed = True
            raise

    def publish_abort(self):
        publish_step_prefill_abort(gate_dir=self.folder, outputs_root=self.folder.parent, run_id=self.run_id, generation=self.generation)

def publish_step_prefill_abort(*, gate_dir, outputs_root, run_id, generation):
    cross_step_prefill_identity(run_id, generation, 0, 1, '0' * 64)
    root = Path(outputs_root).resolve()
    folder = Path(gate_dir).resolve()
    if not folder.is_relative_to(root) or folder == root:
        raise ValueError('cross-step abort is outside the role outputs')
    path = folder / 'abort.json'
    expected = dict(schema_version=1, run_id=run_id, generation=generation, state='ABORTED', committed_original_tokens=0)
    try:
        previous = read_published_json(path, max_bytes=4096)
    except FileNotFoundError:
        folder.mkdir(parents=True, exist_ok=True)
        publish_json(path, expected)
    else:
        if not isinstance(previous, dict) or set(previous) != set(expected) or any((previous[k] != v or type(previous[k]) is not type(v) for k, v in expected.items())):
            raise RuntimeError('cross-step abort identity changed')

def publication_gate_from_environment(environment, *, outputs_root, fetch_parent_permit=None, epoch_plan=None, batch_controls=None, native_binding=None, frozen_input_fingerprints=None):
    coalesced = token_budget_engine_batch_size(environment)
    early = environment.get('SPECO_PREFILL_EARLY_READY', '')
    if early not in ('', '0', '1'):
        raise ValueError('invalid early frozen prefill policy')
    policy = environment.get('SPECO_PREFILL_STEP_LOOKAHEAD', '')
    if policy not in ('', '0', '1'):
        raise ValueError('invalid cross-step source lookahead policy')
    if policy != '1':
        if early == '1':
            raise ValueError('early prefill requires the bounded cross-step source gate')
        return None
    if environment.get('SPECO_PREFILL_SOURCE_LOOKAHEAD') != '1' or environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1':
        raise ValueError('cross-step source lookahead requires bounded native wave capture')
    if environment.get('SPECO_CAPTURE_NATIVE_BUFFER_BATCHES') in ('2', '3', '4', '10') and environment.get('SPECO_PREFILL_EXACT_WAVE_ACQUIRE') != '1':
        raise ValueError('double-buffer cross-step overlap requires exact wave release acknowledgements')

    def integer(key):
        value = environment.get(key, '')
        if not isinstance(value, str) or re.fullmatch('[0-9]+', value) is None:
            raise ValueError('invalid cross-step source integer ' + key)
        return int(value)
    from verl_speco.runtime.recovery import EPOCH_FIELDS, validate_epoch_plan_capacity
    configured = any((environment.get(k, '') for k in EPOCH_FIELDS))
    if configured != (epoch_plan is not None) or (epoch_plan is None and batch_controls is not None):
        raise ValueError('epoch prefill gate needs its admitted source plan and lazy controls')
    count = integer('SPECO_CAPTURE_PRODUCER_COUNT')
    samples = integer('SPECO_PREFILL_STEP_SAMPLES')
    if count < 1 or samples < 1 or samples % count:
        raise ValueError('cross-step samples need complete producer rounds')
    rounds = samples // count
    start = integer('SPECO_CAPTURE_SERVICE_START_STEP')
    end = integer('SPECO_CAPTURE_SERVICE_STEPS')
    if epoch_plan is None:
        if start % rounds or end % rounds:
            raise ValueError('cross-step service bounds must align with optimizer batches')
        start_step, end_step = (start // rounds, end // rounds)
        try:
            controls = json.loads(environment.get('SPECO_PREFILL_STEP_CONTROL_SHA256S', ''))
        except (ValueError, TypeError) as error:
            raise ValueError('invalid cross-step source control fingerprints') from error
    else:
        validate_epoch_plan_capacity(epoch_plan)
        producer = environment.get('SPECO_PRODUCER_ID', '')
        if re.fullmatch('producer-[0-9]+', producer) is None:
            raise ValueError('epoch gate producer identity differs')
        producer_index = int(producer.removeprefix('producer-'))
        pinned = dict(epoch_plan.bindings)
        keys = (*EPOCH_FIELDS, 'SPECO_TARGET_MODEL_OVERLAY', 'SPECO_TARGET_MODEL_DIR', 'SPECO_PROMPT_DATASET', 'SPECO_TARGET_MAX_SEQUENCE_LENGTH')
        if count != epoch_plan.producer_count or samples != epoch_plan.samples_per_step or any((not pinned.get(k) or environment.get(k, '') != pinned[k] for k in keys)) or (environment.get('SPECO_PREFILL_STEP_CONTROL_SHA256S', '') != ''):
            raise ValueError('epoch prefill environment differs from its pinned source')
        start_step = integer('SPECO_PREFILL_STAGE_START_STEP')
        end_step = integer('SPECO_PREFILL_STAGE_END_STEP')
        global_end = min(end_step * samples, epoch_plan.total_samples)
        expected_end = (global_end + count - 1 - producer_index) // count
        if not 0 <= producer_index < count or not 0 <= start_step < end_step <= epoch_plan.total_steps or start != start_step * rounds or (end != expected_end):
            raise ValueError('epoch source range differs from exact per-producer occurrence bounds')
        controls = batch_controls
    folder = Path(environment.get('SPECO_PREFILL_STEP_GATE_DIR', '')).resolve()
    owner = Path(environment.get('SPECO_CAPTURE_COMMAND_DIR', '')).resolve()
    if folder != owner.parent / 'prefill-parent-commits':
        raise ValueError('cross-step publication gate belongs to another role owner')
    producer = environment.get('SPECO_PRODUCER_ID', '')
    if re.fullmatch('producer-[0-9]+', producer) is None:
        raise ValueError('cross-step source producer ID differs')
    frozen_sha = ''
    if early == '1':
        frozen_sha = admitted_source_fingerprint(environment, outputs_root=outputs_root, native_binding=native_binding, fingerprints=frozen_input_fingerprints)
    return StepPrefillPublicationGate(gate_dir=folder, outputs_root=outputs_root, parent_permit_transport=environment.get('SPECO_PREFILL_STEP_GATE_TRANSPORT', 'filesystem'), fetch_parent_permit=fetch_parent_permit, run_id=environment.get('SPECO_RUN_ID', ''), generation=integer('SPECO_GENERATION'), producer_index=int(producer.removeprefix('producer-')), producer_count=count, samples_per_step=samples, start_step=start_step, end_step=end_step, logical_checkpoint_fingerprint=environment.get('SPECO_PREFILL_STEP_LOGICAL', ''), batch_control_sha256s=controls, epoch_plan=epoch_plan, early_ready=early == '1', frozen_input_sha256=frozen_sha, max_capture_children=coalesced or 16, allow_cross_step=environment.get('SPECO_CAPTURE_CROSS_STEP_TOKEN_BATCHES') == '1')

def native_draft_cpu_threads(config):
    value = config.get('native_cpu_threads', 1)
    if type(value) is not int or value not in (1, 2):
        raise ValueError('native trainer CPU threads must be1 or2')
    if value == 2 and 'second_trainer' in config:
        raise ValueError('CPU2 profile requires one DP16 trainer node')
    return value

def validate_cpu_thread_environment(environment):
    flag = environment.get('SPECO_DRAFT_CPU_THREADS', '')
    if flag not in ('', '2'):
        raise ValueError('invalid explicit draft CPU thread policy')
    if flag and (environment.get('SPECO_DRAFT_WORLD_SIZE') != '16' or environment.get('SPECO_DRAFT_NODE_COUNT') not in ('', '1') or any((environment.get(k) != '1' for k in ('SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_DRAFT_PREFILL_WAVES', 'SPECO_NATIVE_FEATURE_TRANSPORT')))):
        raise ValueError('CPU2 threads require one native DP16 node-wave trainer')
    return int(flag) if flag else None

def verify_actual_cpu_threads(environment, actual):
    expected = validate_cpu_thread_environment(environment)
    if expected is not None and (type(actual) is not int or actual != expected):
        raise RuntimeError('actual trainer CPU thread count differs from the immutable CPU2 policy')
    return expected

def cpu_thread_launch_environment(environment):
    """Forward explicit thread policy before torchrun imports torch."""
    expected = validate_cpu_thread_environment(environment)
    if expected is None:
        return {}
    values = {k: environment.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    if any((v != str(expected) for v in values.values())):
        raise ValueError('explicit CPU2 startup requires matching OMP and MKL thread counts')
    return values
from verl_speco.transport.streaming import MAX_NATIVE_BATCH_SAMPLES
DYNAMIC_ENV = frozenset({'SPECO_CONTROL_SNAPSHOT_STEP_INDEX', 'SPECO_LIGHT_REPLAY_PATH', 'SPECO_LIGHT_REPLAY_SHA256', 'SPECO_TQ_KEY', 'SPECO_TQ_METADATA', 'SPECO_PACKED_TQ_SAMPLES', 'SPECO_SAMPLE_INDEX', 'SPECO_STEP_RECEIPT_PATH', 'SPECO_STEP_CONTROL_SNAPSHOT_PATH', 'SPECO_RESUME_RECEIPT_PATH', 'SPECO_GENERATION_HANDOFF', 'SPECO_PARENT_CONTROL_ACTOR_NAME', 'SPECO_PACKED_TQ_SAMPLES_PATH', 'SPECO_PACKED_TQ_SAMPLES_SHA256', 'SPECO_PREFILL_WAVE_ACK_DIR'})
STATIC_ENV = frozenset({'SPECO_PREFILL_EARLY_READY', 'SPECO_EPOCH_REPLAY_SPEC_PATH', 'SPECO_EPOCH_REPLAY_SPEC_SHA256', 'SPECO_EPOCH_REPLAY_PLAN_SHA256', 'SPECO_NATIVE_LISTENER_INCARNATION', 'SPECO_CONTROL_SNAPSHOT_STORAGE', 'SPECO_CONTROL_SNAPSHOT_DIRECTORY', 'SPECO_STREAM_SNAPSHOT_CODEC', 'SPECO_NATIVE_SELECTIVE_HIDDEN_ROUTING', 'SPECO_RUN_ID', 'SPECO_GENERATION', 'SPECO_DRAFT_WORLD_SIZE', 'SPECO_DRAFT_NODE_COUNT', 'SPECO_DRAFT_NODE_OWNER_ADDRESSES', 'SPECO_DRAFT_WAVE_PLAN_SHA256', 'SPECO_DRAFT_DEVICE_START', 'SPECO_DRAFT_SSH_HOST', 'SPECO_DRAFT_TOPOLOGY_FINGERPRINT', 'SPECO_LOGICAL_CHECKPOINT_FINGERPRINT', 'SPECO_RAY_ADDRESS', 'SPECO_TQ_PARTITION', 'SPECO_CONTROL_ACTOR_NAME', 'SPECO_TARGET_MODEL_DIR', 'SPECO_TARGET_MODEL_OVERLAY', 'SPECO_MODEL_CONTENT_AUDIT', 'SPECO_PROMPT_DATASET', 'SPECO_DCP_SHARED_ROOT', 'SPECO_STEP_JOURNAL_DIR', 'SPECO_DCP_FILE_HASH_WORKERS', 'SPECO_DCP_FILE_WRITE_WORKERS', 'SPECO_TOTAL_STEPS', 'SPECO_DRAFT_TRAINING_WINDOW', 'SPECO_DRAFT_TRAINING_SEED', 'SPECO_DRAFT_DETERMINISTIC_ALGORITHMS', 'SPECO_GRADIENT_DIAGNOSTIC_DIR', 'SPECO_DRAFT_EXPECTED_INIT_CHECKSUM', 'SPECO_DRAFT_SHARD_BATCH', 'SPECO_DRAFT_SHARD_WINDOWS', 'SPECO_DRAFT_INIT_SNAPSHOT', 'SPECO_DRAFT_CPU_WINDOW_MATERIALIZE', 'SPECO_DRAFT_OWNED_WINDOW_MATERIALIZE', 'SPECO_DRAFT_SHARED_WINDOW_MATERIALIZE', 'SPECO_DRAFT_STREAM_OWNED_WINDOW_MATERIALIZE', 'SPECO_DRAFT_TRAINING_PROFILE', 'SPECO_TARGET_LAYER_IDS', 'SPECO_TARGET_MAX_SEQUENCE_LENGTH', 'SPECO_DRAFT_SNAPSHOT_ROOT', 'SPECO_PACKING_TARGET_LENGTH', 'SPECO_PACKING_MAX_BYTES', 'SPECO_VERIFY_SHARDS_FULL', 'SPECO_CAPTURE_DEVICE_CONCAT', 'SPECO_NATIVE_SOURCE_CPU_PACK', 'SPECO_NATIVE_INLINE_DESCRIPTOR', 'SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_NATIVE_METADATA_SERVER', 'SPECO_NATIVE_DRAFT_HOST_IP', 'SPECO_NATIVE_RECEIVE_WORKERS', 'SPECO_NATIVE_CPU_BULK_RECEIVE', 'SPECO_NATIVE_RECEIVER_STEP_INDEX', 'SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_DRAFT_DIRECT_STREAMING_FRAME', 'SPECO_DRAFT_FRAME_HUGEPAGES', 'SPECO_DRAFT_FRAME_SEAL_WORKERS', 'SPECO_DRAFT_PREFILL_WAVES', 'SPECO_DRAFT_VARIABLE_BATCH', 'SPECO_DRAFT_NODE_FRAME_MAX_BYTES', 'SPECO_PACKING_ROW_COUNT', 'SPECO_PACKING_MAX_SAMPLES_PER_ROW', 'SPECO_PREFILL_WAVE_SIZE', 'SPECO_PREFILL_MAX_WAVE_BYTES', 'SPECO_PREFILL_WAVE_ACK_TRANSPORT', 'SPECO_PREFILL_OWNED_COPY_WORKERS', 'SPECO_PREFILL_NATIVE_VERIFICATION_WORKERS', 'SPECO_PREFILL_NATIVE_PAYLOAD_PREFETCH_BYTES', 'SPECO_PREFILL_SOURCE_LOOKAHEAD', 'SPECO_PREFILL_EXACT_WAVE_ACQUIRE', 'SPECO_NATIVE_RECEIVE_PROFILE', 'SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE', 'SPECO_PREFILL_BATCH_RELEASE'})

def build_draft_command(environment, *, run_id, generation, step_index):
    values = {key: str(environment.get(key, '')) for key in STATIC_ENV | DYNAMIC_ENV}
    for key in ('SPECO_DRAFT_PIPELINED_FRAME_DISPATCH', 'SPECO_NATIVE_RECEIVE_COPY_STREAM'):
        value = environment.get(key, '')
        if value not in ('', '0'):
            values[key] = str(value)
    consensus = environment.get('SPECO_DRAFT_WEIGHT_CONSENSUS', '')
    if consensus:
        values['SPECO_DRAFT_WEIGHT_CONSENSUS'] = str(consensus)
    cpu_threads = environment.get('SPECO_DRAFT_CPU_THREADS', '')
    if cpu_threads:
        values['SPECO_DRAFT_CPU_THREADS'] = str(cpu_threads)
    pipeline = environment.get('SPECO_PREFILL_OWNED_COPY_PIPELINE', '')
    if pipeline not in ('', '0'):
        values['SPECO_PREFILL_OWNED_COPY_PIPELINE'] = str(pipeline)
    chunk = environment.get('SPECO_PREFILL_OWNED_RELEASE_CHUNK_SIZE', '')
    if chunk not in ('', '0'):
        values['SPECO_PREFILL_OWNED_RELEASE_CHUNK_SIZE'] = str(chunk)
    from verl_speco.runtime.checkpoint_policy import SPARSE_ENV
    values.update({k: str(environment[k]) for k in SPARSE_ENV if environment.get(k)})
    return dict(schema_version=1, run_id=run_id, generation=generation, step_index=step_index, values=values)

def validate_draft_command(command, *, run_id, generation, step_index, previous_binding=None, epoch_plan=None):
    expected = dict(schema_version=1, run_id=run_id, generation=generation, step_index=step_index)
    if not isinstance(command, dict) or set(command) != set(expected) | {'values'} or any((command[k] != v or type(command[k]) is not type(v) for k, v in expected.items())):
        raise ValueError('draft command owner or step identity differs')
    values = command['values']
    optional = {k for k in ('SPECO_DRAFT_PIPELINED_FRAME_DISPATCH', 'SPECO_NATIVE_RECEIVE_COPY_STREAM', 'SPECO_DRAFT_WEIGHT_CONSENSUS', 'SPECO_DRAFT_CPU_THREADS', 'SPECO_PREFILL_OWNED_COPY_PIPELINE', 'SPECO_PREFILL_OWNED_RELEASE_CHUNK_SIZE') if isinstance(values, dict) and k in values}
    from verl_speco.runtime.checkpoint_policy import SPARSE_ENV, validate_sparse_environment
    optional.update((k for k in SPARSE_ENV if isinstance(values, dict) and k in values))
    if isinstance(values, dict):
        validate_sparse_environment(values, epoch_plan)
    if not isinstance(values, dict) or set(values) != STATIC_ENV | DYNAMIC_ENV | optional or any((not isinstance(v, str) for v in values.values())) or (values['SPECO_RUN_ID'] != run_id) or (values['SPECO_GENERATION'] != str(generation)):
        raise ValueError('draft command environment does not match configured owner')
    if 'SPECO_DRAFT_WEIGHT_CONSENSUS' in optional and (values['SPECO_DRAFT_WEIGHT_CONSENSUS'] != 'exact_device_root_sha' or values['SPECO_DRAFT_WORLD_SIZE'] != '16' or values['SPECO_DRAFT_NODE_COUNT'] not in ('', '1') or (values['SPECO_DRAFT_NODE_OWNED_SWA'] != '1') or (values['SPECO_NATIVE_FEATURE_TRANSPORT'] != '1')):
        raise ValueError('device exact weight consensus requires one native DP16 trainer')
    validate_cpu_thread_environment(values)
    early = values['SPECO_PREFILL_EARLY_READY']
    if early not in ('', '0', '1') or (early == '1' and any((values[k] != '1' for k in ('SPECO_DRAFT_PREFILL_WAVES', 'SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_NATIVE_FEATURE_TRANSPORT')))):
        raise ValueError('early frozen prefill requires immutable native node-wave trainer mode')
    from verl_speco.runtime.recovery import validate_epoch_command
    validate_epoch_command(values, epoch_plan=epoch_plan)
    from verl_speco.transport.streaming import receipt_stream_codec
    receipt_stream_codec(values)
    incarnation = values['SPECO_NATIVE_LISTENER_INCARNATION']
    if incarnation:
        if not incarnation.isdecimal() or incarnation != str(int(incarnation)):
            raise ValueError('invalid native listener incarnation')
        native_draft_endpoint('127.0.0.1', 0, generation, step_index=step_index, listener_incarnation=int(incarnation))
        if int(incarnation) < generation:
            raise ValueError('native listener incarnation predates generation')
    if control_snapshot_storage(values):
        if values['SPECO_CONTROL_SNAPSHOT_STEP_INDEX'] != str(step_index) or values['SPECO_CONTROL_SNAPSHOT_DIRECTORY'] != str(shared_control_directory(run_id=run_id, generation=generation)):
            raise ValueError('shared control snapshot command binding differs')
    elif values['SPECO_CONTROL_SNAPSHOT_DIRECTORY'] or values['SPECO_CONTROL_SNAPSHOT_STEP_INDEX']:
        raise ValueError('shared control snapshot fields require explicit storage mode')
    validate_two_trainer_wave_fields(values)
    from verl_speco.runtime.recovery import checkpoint_hash_workers, checkpoint_write_workers
    checkpoint_hash_workers(values)
    checkpoint_write_workers(values)
    from verl_speco.trainer.owned_batches import resolve_pipelined_frame_dispatch
    from verl_speco.transport.native_transfer import native_receive_copy_stream_policy
    resolve_pipelined_frame_dispatch(values)
    native_receive_copy_stream_policy(values)
    from verl_speco.trainer.owned_batches import resolve_direct_streaming_frame
    resolve_direct_streaming_frame(values['SPECO_DRAFT_DIRECT_STREAMING_FRAME'], node_owned=values['SPECO_DRAFT_NODE_OWNED_SWA'] == '1' and values['SPECO_NATIVE_FEATURE_TRANSPORT'] == '1', prefill_waves=values['SPECO_DRAFT_PREFILL_WAVES'] == '1')
    from verl_speco.trainer.owned_batches import resolve_frame_hugepages
    capture_device_concat(values)
    resolve_frame_hugepages(values['SPECO_DRAFT_FRAME_HUGEPAGES'], node_owned=values['SPECO_DRAFT_NODE_OWNED_SWA'] == '1' and values['SPECO_NATIVE_FEATURE_TRANSPORT'] == '1')
    if values['SPECO_DRAFT_FRAME_SEAL_WORKERS']:
        from verl_speco.trainer.owned_batches import resolve_frame_seal_workers
        resolve_frame_seal_workers(values['SPECO_DRAFT_FRAME_SEAL_WORKERS'], world_size=int(values['SPECO_DRAFT_WORLD_SIZE']), node_owned=values['SPECO_DRAFT_NODE_OWNED_SWA'] == '1' and values['SPECO_NATIVE_FEATURE_TRANSPORT'] == '1')
    if values['SPECO_DRAFT_DETERMINISTIC_ALGORITHMS'] not in ('', '0', '1'):
        raise ValueError('draft deterministic algorithms must be empty, 0 or 1')
    if values['SPECO_DRAFT_VARIABLE_BATCH'] not in ('', '0', '1'):
        raise ValueError('draft variable batch must be empty, 0 or 1')
    if values['SPECO_PREFILL_SOURCE_LOOKAHEAD'] not in ('', '0', '1'):
        raise ValueError('prefill source lookahead must be empty, 0 or 1')
    if values['SPECO_PREFILL_SOURCE_LOOKAHEAD'] == '1' and values['SPECO_DRAFT_PREFILL_WAVES'] != '1':
        raise ValueError('source lookahead requires verified native prefill waves')
    exact = values['SPECO_PREFILL_EXACT_WAVE_ACQUIRE']
    if exact not in ('', '0', '1'):
        raise ValueError('exact wave acquire must be empty, 0 or 1')
    if exact == '1' and any((values[k] != '1' for k in ('SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_DRAFT_PREFILL_WAVES', 'SPECO_PREFILL_BATCH_RELEASE'))):
        raise ValueError('exact wave acquire requires native node waves with batch release')
    from verl_speco.transport.native_transfer import native_descriptor_store_type
    native_descriptor_store_type(values)
    if values['SPECO_NATIVE_RECEIVE_PROFILE'] not in ('', '0', '1'):
        raise ValueError('native receive profile must be empty, 0 or 1')
    if values['SPECO_NATIVE_RECEIVE_PROFILE'] == '1' and values['SPECO_NATIVE_FEATURE_TRANSPORT'] != '1':
        raise ValueError('receive profiling requires native feature transport')
    if values['SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE'] not in ('', '0', '1'):
        raise ValueError('native verification pipeline must be empty, 0 or 1')
    if values['SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE'] == '1' and (values['SPECO_NATIVE_FEATURE_TRANSPORT'] != '1' or values['SPECO_DRAFT_PREFILL_WAVES'] != '1' or (not values['SPECO_PREFILL_NATIVE_PAYLOAD_PREFETCH_BYTES'])):
        raise ValueError('native verification pipeline requires native waves and CPU prefetch budget')
    if values['SPECO_PREFILL_BATCH_RELEASE'] not in ('', '0', '1'):
        raise ValueError('native batch release must be empty, 0 or 1')
    if values['SPECO_PREFILL_BATCH_RELEASE'] == '1' and any((values[k] != '1' for k in ('SPECO_DRAFT_PREFILL_WAVES', 'SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_NODE_OWNED_SWA'))):
        raise ValueError('native batch release requires node-owned native waves')
    copies = values['SPECO_PREFILL_OWNED_COPY_WORKERS']
    if copies not in ('', '1', '2', '4', '8', '16'):
        raise ValueError('invalid owned copy worker width')
    if copies not in ('', '1') and (values['SPECO_NATIVE_FEATURE_TRANSPORT'] != '1' or values['SPECO_DRAFT_NODE_OWNED_SWA'] != '1' or values['SPECO_DRAFT_PREFILL_WAVES'] != '1'):
        raise ValueError('parallel owned copies require native node wave owner')
    width = values['SPECO_PREFILL_NATIVE_VERIFICATION_WORKERS']
    if width not in ('', '2', '4', '8'):
        raise ValueError('native verification width must be empty, 2 or 4')
    from verl_speco.transport.native_transfer import wide_native_receive_profile
    wide_receive = wide_native_receive_profile(values)
    if width == '8' and (not wide_receive):
        raise ValueError('eight-verifier width requires the exact DP16 16GiB profile')
    if width in ('4', '8') and values['SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE'] != '1':
        raise ValueError('native verification width 4 requires verification pipeline')
    verification_workers = int(width or '2')
    prefetch = values['SPECO_PREFILL_NATIVE_PAYLOAD_PREFETCH_BYTES']
    if prefetch and (not prefetch.isdecimal() or not 0 < int(prefetch) <= (16 if wide_receive else 8) * 1024 ** 3 or any((values[k] != '1' for k in ('SPECO_DRAFT_PREFILL_WAVES', 'SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_NODE_OWNED_SWA')))):
        raise ValueError('native payload prefetch needs bounded bytes and native node-owned waves')
    from verl_speco.transport.native_transfer import native_receive_policy
    receive_workers, bulk_receive = native_receive_policy(values)
    pipeline = values.get('SPECO_PREFILL_OWNED_COPY_PIPELINE', '0')
    if pipeline not in ('0', '1'):
        raise ValueError('invalid owned copy pipeline policy')
    if pipeline == '1' and (not wide_receive or copies != '16' or width != '8' or (receive_workers != 8) or (not bulk_receive) or (values['SPECO_PREFILL_WAVE_SIZE'] != '48') or (values['SPECO_PREFILL_MAX_WAVE_BYTES'] != str(96 * 1024 ** 3)) or (values['SPECO_DRAFT_NODE_COUNT'] not in ('', '1'))):
        raise ValueError('owned copy pipeline requires the exact DP16 eight-reader profile')
    chunk = values.get('SPECO_PREFILL_OWNED_RELEASE_CHUNK_SIZE', '0')
    if chunk not in ('0', '6', '12') or (chunk != '0' and pipeline != '1'):
        raise ValueError('owned prefix release requires its byte-bounded copy pipeline')
    binding = {k: values[k] for k in STATIC_ENV}
    binding.update({k: values[k] for k in optional})
    if previous_binding is not None and binding != previous_binding:
        raise RuntimeError('persistent trainer static configuration changed')
    from verl_speco.trainer.feature_pipeline import load_packed_sample_control
    file_mode = bool(values['SPECO_PACKED_TQ_SAMPLES_PATH'])
    waves = values['SPECO_DRAFT_PREFILL_WAVES'] == '1'
    transport = values['SPECO_PREFILL_WAVE_ACK_TRANSPORT']
    if transport not in ('', 'filesystem', 'ray') or (transport == 'ray' and (not waves)):
        raise ValueError('invalid prefill wave ACK transport or trainer mode')
    variable = values['SPECO_DRAFT_VARIABLE_BATCH'] == '1'
    if variable and (not waves):
        raise ValueError('variable batches require verified native prefill waves')
    if waves and (not file_mode or any((values[k] != '1' for k in ('SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_STREAM_OWNED_WINDOW_MATERIALIZE')))):
        raise ValueError('prefill waves require verified metadata and native node-owned SWA')
    selected = values['SPECO_NATIVE_SELECTIVE_HIDDEN_ROUTING']
    if selected not in ('', '0', '1'):
        raise ValueError('selected runtime flag differs')
    if selected == '1':
        import re
        from pathlib import Path
        import verl_speco
        replay = Path(values['SPECO_LIGHT_REPLAY_PATH']).resolve()
        if values['SPECO_DRAFT_NODE_COUNT'] != '2' or values['SPECO_DRAFT_WORLD_SIZE'] != '32' or values['SPECO_NATIVE_RECEIVE_WORKERS'] not in ('2', '4') or (values['SPECO_NATIVE_CPU_BULK_RECEIVE'] != '1') or (not values['SPECO_LIGHT_REPLAY_PATH']) or (not replay.is_relative_to(Path(verl_speco.__file__).resolve().parents[1] / 'outputs')) or (re.fullmatch('[0-9a-f]{64}', values['SPECO_LIGHT_REPLAY_SHA256']) is None):
            raise ValueError('selected runtime requires bound replay and exact DP32 topology')
    elif values['SPECO_LIGHT_REPLAY_PATH'] or values['SPECO_LIGHT_REPLAY_SHA256']:
        raise ValueError('light replay supplied outside selected runtime')
    samples = load_packed_sample_control(inline=values['SPECO_PACKED_TQ_SAMPLES'], path=values['SPECO_PACKED_TQ_SAMPLES_PATH'], sha256=values['SPECO_PACKED_TQ_SAMPLES_SHA256'], run_id=run_id, generation=generation)
    start = int(values['SPECO_SAMPLE_INDEX'])
    if not isinstance(samples, list) or not samples or len(samples) > (MAX_NATIVE_BATCH_SAMPLES if file_mode and waves else 16) or (not variable and start != step_index * len(samples)) or (len({s['key'] for s in samples}) != len(samples)) or any((s['metadata']['run_id'] != run_id or s['metadata']['generation'] != generation or s['metadata']['sequence_no'] != start + i for i, s in enumerate(samples))):
        raise ValueError('draft command samples do not match ordered stream batch')
    payload_slots = verification_workers + 1 if values['SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE'] == '1' else 2
    if prefetch and payload_slots * max((s['metadata']['payload_nbytes'] + (4096 if bulk_receive else 0) for s in samples)) + (2 * max((s['metadata']['payload_nbytes'] for s in samples)) if pipeline == '1' else 0) > int(prefetch):
        if payload_slots == 3:
            raise ValueError('native verification pipeline exceeds three-payload CPU budget before commands')
        if payload_slots == 5:
            raise ValueError('native verification pipeline exceeds five-payload CPU budget before commands')
        raise ValueError('native payload prefetch exceeds two-payload CPU budget before commands')
    bound = values['SPECO_DRAFT_EXPECTED_INIT_CHECKSUM']
    if bound and (not values['SPECO_DRAFT_INIT_SNAPSHOT'] or len(bound) != 64 or any((c not in '0123456789abcdef' for c in bound))):
        raise ValueError('failed genesis initialization checksum binding differs')
    if bool(values['SPECO_RESUME_RECEIPT_PATH']) != (step_index > 0):
        raise ValueError('persistent optimizer commands require their committed predecessor')
    if variable:
        from pathlib import Path
        from verl_speco.runtime.artifacts import read_published_json
        predecessor = read_published_json(Path(values['SPECO_RESUME_RECEIPT_PATH']), max_bytes=16 * 1024 ** 2) if step_index else None
        validate_variable_batch_cursor(run_id=run_id, generation=generation, step_index=step_index, start=start, batch_size=len(samples), total_steps=int(values['SPECO_TOTAL_STEPS']), world_size=int(values['SPECO_DRAFT_WORLD_SIZE']), previous_receipt=predecessor, generation_handoff=values['SPECO_GENERATION_HANDOFF'] == '1', epoch_plan=epoch_plan)
    if epoch_plan is not None:
        from verl_speco.runtime.recovery import expected_epoch_sample_keys
        if [sample['key'] for sample in samples] != expected_epoch_sample_keys(epoch_plan, run_id=run_id, generation=generation, step_index=step_index):
            raise ValueError('epoch command samples differ from frozen occurrence order')
    return (values, binding)

def request_draft_batch(*, command_dir: Path, environment, run_id: str, generation: int, step_index: int, timeout_seconds: float=1200, check_roles=lambda: None, epoch_plan=None):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('draft service timeout must be finite and positive')
    command = build_draft_command(environment, run_id=run_id, generation=generation, step_index=step_index)
    validate_draft_command(command, run_id=run_id, generation=generation, step_index=step_index, **{'epoch_plan': epoch_plan} if epoch_plan is not None else {})
    command_dir.mkdir(parents=True, exist_ok=True)
    path = command_dir / f'step-{step_index:04d}.request.json'
    try:
        previous = read_published_json(path)
    except FileNotFoundError:
        publish_json(path, command)
    else:
        if previous != command:
            raise RuntimeError('draft service request identity changed')
    deadline = time.monotonic() + timeout_seconds
    while True:
        check_roles()
        try:
            ack = read_published_json(command_dir / f'step-{step_index:04d}.ack.json')
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError('persistent draft commit timed out')
            time.sleep(0.5)
    expected = dict(run_id=run_id, generation=generation, step_index=step_index, state='STEP_COMMITTED', receipt_path=command['values']['SPECO_STEP_RECEIPT_PATH'])
    if not isinstance(ack, dict) or any((ack.get(k) != v or type(ack.get(k)) is not type(v) for k, v in expected.items())) or (not isinstance(ack.get('receipt_sha256'), str)) or (re.fullmatch('[0-9a-f]{64}', ack['receipt_sha256']) is None):
        raise RuntimeError('draft service acknowledgement does not prove requested commit')
    return ack
from dataclasses import dataclass

@dataclass(frozen=True)
class DraftTrainingProfile:
    block_size: int
    num_anchors: int
    loss_decay_gamma: float
    ce_loss_alpha: float
    l1_loss_alpha: float
    l1_chunk_size: int
    confidence_loss_alpha: float = 0.0
    anchor_sampling_policy: str = 'random'
    swa_window: int | None = None
    swa_chunk_cap: int | None = None

def draft_training_profile(name: str) -> DraftTrainingProfile:
    if name == 'smoke-dmtp-swa':
        return DraftTrainingProfile(2, 2, 7.0, 0.1, 0.9, 0, 0.0, 'dmtp_cpu', 2048, 32)
    if name == 'dmtp-92b-block8-swa-confidence-128k':
        return DraftTrainingProfile(8, 256, 4.0, 0.5, 0.5, 512, 1.0, 'dmtp_cpu', 2048, 64)
    if name == 'dmtp-92b-block8-swa-confidence':
        return DraftTrainingProfile(8, 256, 4.0, 0.5, 0.5, 512, 1.0, 'dmtp_cpu', 2048, 32)
    if name == 'smoke-dmtp-anchors':
        return DraftTrainingProfile(2, 2, 7.0, 0.1, 0.9, 0, 0.0, 'dmtp_cpu')
    if name == 'dmtp-92b-block8-aligned-confidence':
        return DraftTrainingProfile(8, 256, 4.0, 0.5, 0.5, 512, 1.0, 'dmtp_cpu')
    if name == 'smoke':
        return DraftTrainingProfile(2, 2, 7.0, 0.1, 0.9, 0)
    if name == 'smoke-confidence':
        return DraftTrainingProfile(2, 2, 7.0, 0.1, 0.9, 0, 1.0)
    if name == 'dmtp-92b-block8-confidence':
        return DraftTrainingProfile(8, 256, 4.0, 0.5, 0.5, 512, 1.0)
    if name == 'dmtp-92b-block8':
        return DraftTrainingProfile(8, 256, 4.0, 0.5, 0.5, 512)
    raise ValueError(f'unknown draft training profile: {name}')

def failed_native_genesis_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def require_empty_genesis_owner(status, *, run_id, generation):
    if not isinstance(status, dict) or status.get('run_id') != run_id or status.get('partition') != 'speco-92b-' + run_id or (type(status.get('generation')) is not int) or (status['generation'] != generation) or any((type(status.get(k)) is not int or status[k] != 0 for k in ('used_bytes', 'retained_items', 'pending_reservations', 'reclaim_pending'))) or (not isinstance(status.get('router'), dict)) or (type(status['router'].get('ready_records')) is not int) or (status['router']['ready_records'] != 0) or (not isinstance(status.get('leases'), dict)) or any((type(status.get('leases', {}).get(k)) is not int or status['leases'][k] != 0 for k in ('committed_steps', 'leases_committed', 'leases_leased', 'leases_materialized'))):
        raise ValueError('failed genesis owner still has data, active leases or an optimizer commit')

def initialization_binding(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {'uri', 'checksum', 'draft_revision'} or (not isinstance(snapshot['uri'], str)) or (not Path(snapshot['uri']).is_absolute()) or (not isinstance(snapshot['checksum'], str)) or (len(snapshot['checksum']) != 64) or any((c not in '0123456789abcdef' for c in snapshot['checksum'])) or (type(snapshot['draft_revision']) is not int) or (snapshot['draft_revision'] < 0):
        raise ValueError('failed genesis requires an immutable complete initialization snapshot')
    return dict(snapshot)

def require_uncommitted_genesis_journals(probes, *, run_id, generation, samples):
    if not isinstance(probes, list) or not probes:
        raise ValueError('failed genesis requires every physical host journal and receipt probe')
    prepared = set()
    for probe in probes:
        if not isinstance(probe, dict) or probe.get('active_trainer_pids') != [] or (not isinstance(probe.get('journals'), list)) or (probe.get('published_step_receipts') != []):
            raise ValueError('failed genesis roles are active or optimizer receipts are present')
        for row in probe['journals']:
            if not isinstance(row, dict):
                raise ValueError('invalid failed genesis journal')
            tx = row.get('transaction', {})
            if type(tx.get('generation')) is not int or tx['generation'] != generation or tx.get('stream_scope') != run_id or (row.get('state') != 'STEP_PREPARED') or (row.get('optimizer_started') is not True):
                raise ValueError('failed genesis journal contains a commit or another stream')
            ids = tx.get('transaction_ids')
            step = tx.get('optimizer_step_id')
            if not isinstance(step, str) or not step or (not isinstance(ids, list)) or (len(ids) != samples) or any((not isinstance(i, str) or not i for i in ids)) or (len(set(ids)) != samples):
                raise ValueError('failed genesis uncertain transaction differs from frozen first batch')
            prepared.add(step)
    if len(prepared) > 1:
        raise ValueError('multiple old optimizer updates are uncertain')

def build_failed_genesis_proof(*, run_id, cleanup, exit_proof, recipe, initialization, first_batch):
    from verl_speco.runtime.recovery import validate_owned_native_exit_proof
    validate_owned_native_exit_proof(exit_proof, run_id=run_id, generation=1)
    if not isinstance(cleanup, dict) or cleanup.get('run_id') != run_id or type(cleanup.get('generation')) is not int or (cleanup['generation'] != 1) or (cleanup.get('state') != 'FAILED_PREFILL_RETIRED_AFTER_OWNED_EXIT') or (type(cleanup.get('new_optimizer_commits')) is not int) or (cleanup['new_optimizer_commits'] != 0) or (type(cleanup.get('existing_optimizer_commits_preserved')) is not int) or (cleanup['existing_optimizer_commits_preserved'] != 0) or (cleanup.get('committed_DCP_and_receipts_modified') is not False):
        raise ValueError('failed genesis requires typed zero-commit cleanup after owned role exit')
    require_empty_genesis_owner(cleanup.get('control_status'), run_id=run_id, generation=1)
    initialization = initialization_binding(initialization)
    if not isinstance(recipe, dict) or recipe.get('failed_genesis_recovery') is not True or (not isinstance(first_batch, dict)) or (first_batch.get('run_id') != run_id) or (type(first_batch.get('generation')) is not int) or (first_batch['generation'] != 1) or (not isinstance(first_batch.get('samples'), list)) or (not first_batch['samples']) or (type(cleanup.get('expected_samples')) is not int) or (cleanup['expected_samples'] != len(first_batch['samples'])):
        raise ValueError('failed genesis static recipe or frozen first batch differs')
    return dict(schema_version=1, state='FAILED_NATIVE_GENESIS_FENCED', run_id=run_id, source_generation=1, next_generation=2, committed_cursor=0, optimizer_commits=0, recipe_sha256=failed_native_genesis_digest(recipe), initialization=initialization, first_batch_sha256=failed_native_genesis_digest(first_batch), role_hosts=exit_proof['hosts'], cleanup_sha256=failed_native_genesis_digest(cleanup))

def load_failed_genesis_proof(path, *, work_root, run_id, recipe, initialization, first_batch):
    from verl_speco.runtime.artifacts import read_published_json
    from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key
    from dataclasses import replace
    old_work = Path(work_root) / ('speco-native-wave-' + run_id + '-g1')
    if Path(path).resolve() != (old_work / 'failed-native-genesis.json').resolve():
        raise ValueError('failed genesis proof must belong to the exact old run workspace')
    for name in ('functional-result.json', 'parent-receipt.json'):
        try:
            read_published_json(old_work / name, max_bytes=16 * 1024 ** 2)
        except FileNotFoundError:
            pass
        else:
            raise ValueError('failed genesis already has a published optimizer result')
    proof = read_published_json(Path(path), max_bytes=65536)
    required = {'schema_version', 'state', 'run_id', 'source_generation', 'next_generation', 'committed_cursor', 'optimizer_commits', 'recipe_sha256', 'initialization', 'first_batch_sha256', 'role_hosts', 'cleanup_sha256'}
    if not isinstance(proof, dict) or set(proof) != required or proof['state'] != 'FAILED_NATIVE_GENESIS_FENCED' or (proof['run_id'] != run_id) or any((type(proof[k]) is not int or proof[k] != v for k, v in (('schema_version', 1), ('source_generation', 1), ('next_generation', 2), ('committed_cursor', 0), ('optimizer_commits', 0)))):
        raise ValueError('failed genesis proof identity or zero committed prefix differs')
    if proof['recipe_sha256'] != failed_native_genesis_digest(recipe) or proof['initialization'] != initialization_binding(initialization):
        raise ValueError('failed genesis recipe, data, topology or initialization changed')
    old_recipe = read_published_json(old_work / 'recipe.json', max_bytes=16 * 1024 ** 2)
    old_initialization = read_published_json(old_work / 'genesis-initialization.json', max_bytes=65536)
    old_batch = read_published_json(old_work / 'packed-samples-step-0000.json', max_bytes=16 * 1024 ** 2)
    cleanup = read_published_json(old_work / 'failed-native-prefill-cleanup.json', max_bytes=16 * 1024 ** 2)
    if old_recipe != recipe or old_initialization != proof['initialization'] or proof['first_batch_sha256'] != failed_native_genesis_digest(old_batch) or (proof['cleanup_sha256'] != failed_native_genesis_digest(cleanup)):
        raise ValueError('failed genesis durable evidence changed')
    require_empty_genesis_owner(cleanup.get('control_status'), run_id=run_id, generation=1)
    if not isinstance(proof['role_hosts'], list) or not proof['role_hosts'] or len(proof['role_hosts']) > 8 or any((not isinstance(h, str) or not h for h in proof['role_hosts'])) or (len(set(proof['role_hosts'])) != len(proof['role_hosts'])):
        raise ValueError('failed genesis old physical hosts are incomplete')
    if first_batch.get('run_id') != run_id or first_batch.get('generation') != 2:
        raise ValueError('failed genesis replay must use the same run and generation two')
    restored = []
    for item in first_batch['samples']:
        metadata = SampleMetadataV3.from_dict(item['metadata'])
        if metadata.run_id != run_id or metadata.generation != 2 or make_v3_sample_key(metadata) != item['key']:
            raise ValueError('failed genesis replay sample identity differs')
        prior = replace(metadata, generation=1)
        restored.append(dict(key=make_v3_sample_key(prior), metadata=prior.to_dict()))
    if failed_native_genesis_digest(dict(schema_version=1, run_id=run_id, generation=1, samples=restored)) != proof['first_batch_sha256']:
        raise ValueError('failed genesis replay changes the frozen first batch')
    return proof

def probe_failed_genesis(*, proof, role_hosts, python_bin, ray_address, source_root, first_batch):
    """Fresh read-only probes; durable old exit observations alone do not admit replay."""
    import subprocess, shlex, ray
    from verl_speco.runtime.recovery import REMOTE_PROBE
    from verl_speco.transport.control import actor_name_for_run
    run = proof['run_id']
    hosts = tuple(dict.fromkeys(role_hosts))
    if tuple(proof['role_hosts']) != hosts:
        raise ValueError('failed genesis physical topology changed')
    journal = '/tmp/speco-three-node-' + run + '-journal'
    receipt = '/tmp/speco-pack-receipts-' + run + '-g1'
    code = REMOTE_PROBE.replace("print(json.dumps({'journals': journals, 'active_trainer_pids': active}, sort_keys=True))", '\nfrom verl_speco.runtime.artifacts import read_published_json\nreceipts=[]\nfor step in range(8):\n    try: value=read_published_json(pathlib.Path(' + repr(receipt) + ")/f'step-{step:04d}.json')\n    except FileNotFoundError: continue\n    receipts.append(value)\nprint(json.dumps({'journals': journals, 'active_trainer_pids': active, 'published_step_receipts': receipts}, sort_keys=True))")
    probes = []
    for host in hosts:
        command = shlex.join([python_bin, '-c', code, journal, run, '1', str(source_root)])
        result = subprocess.run(['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', host, command], capture_output=True, text=True, check=True, timeout=45)
        if len(result.stdout.encode()) > 16 * 1024 ** 2:
            raise ValueError('genesis journal probe exceeds metadata budget')
        probes.append(json.loads(result.stdout))
    require_uncommitted_genesis_journals(probes, run_id=run, generation=1, samples=len(first_batch['samples']))
    ray.init(address=ray_address, namespace='transfer_queue', logging_level='ERROR')
    try:
        actor = ray.get_actor(actor_name_for_run(run, 1), namespace='transfer_queue')
        status = ray.get(actor.status.remote(), timeout=30)
        require_empty_genesis_owner(status, run_id=run, generation=1)
        from verl_speco.transport.streaming import SampleMetadataV3, parse_v3_ready_tag, make_v3_sample_key
        from dataclasses import replace
        expected = {make_v3_sample_key(replace(SampleMetadataV3.from_dict(item['metadata']), generation=1)) for item in first_batch['samples']}
        tq = import_transfer_queue()
        tq.init()
        present = tq.kv_list(partition_id=status['partition']).get(status['partition'], {})
        for key, tag in present.items():
            parsed = parse_v3_ready_tag(tag)
            if key in expected or (parsed is not None and parsed[0].run_id == run and (parsed[0].generation == 1)):
                raise ValueError('failed genesis old native TQ data remains present')
    finally:
        ray.shutdown()
    return dict(run_id=run, source_generation=1, next_generation=2, committed_cursor=0, optimizer_commits=0, fresh_role_and_journal_probes=probes, old_owner_status=status, old_run_TQ_empty=True, foreign_TQ_keys_modified=False)

def require_bound_initialization(snapshot, expected):
    if not expected:
        return
    binding = initialization_binding(snapshot)
    if not isinstance(expected, str) or len(expected) != 64 or any((c not in '0123456789abcdef' for c in expected)) or (binding['checksum'] != expected):
        raise ValueError('draft initialization differs from fenced genesis snapshot')
import hashlib, json, re

def fingerprints_sha256(values):
    names = {'target_model_fingerprint', 'tokenizer_fingerprint', 'dataset_fingerprint'}
    if not isinstance(values, dict) or set(values) != names or any((not isinstance(v, str) or not v for v in values.values())):
        raise ValueError('frozen prefill requires all three actual input fingerprints')
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def prefill_admission(*, run_id, generation, next_step_index, logical_checkpoint_fingerprint, current_batch_control_sha256, next_batch_control_sha256, frozen_input_sha256, epoch_plan=None):
    cross_step_prefill_identity(run_id, generation, next_step_index, 1 if epoch_plan is None else epoch_plan.samples_per_step, logical_checkpoint_fingerprint, epoch_plan)
    if next_step_index < 1 or any((not cross_step_prefill_sha(v) for v in (current_batch_control_sha256, next_batch_control_sha256, frozen_input_sha256))):
        raise ValueError('prefill admission needs exact current and next batch identities')
    result = dict(schema_version=1, run_id=run_id, generation=generation, parent_step_index=next_step_index - 1, next_step_index=next_step_index, logical_checkpoint_fingerprint=logical_checkpoint_fingerprint, current_batch_control_sha256=current_batch_control_sha256, next_batch_control_sha256=next_batch_control_sha256, frozen_input_sha256=frozen_input_sha256, state='PREFILL_ONLY_ADMITTED', optimizer_started=False, committed_original_tokens=0)
    if epoch_plan is not None:
        result['epoch_plan_sha256'] = epoch_plan.plan_sha256
    return result

def publish_prefill_admission(*, gate_dir, outputs_root, **identity):
    payload = prefill_admission(**identity)
    root = Path(outputs_root).resolve()
    folder = Path(gate_dir).resolve()
    if not folder.is_relative_to(root) or folder == root:
        raise ValueError('prefill admission outside owned outputs')
    path = folder / f"prefill-{payload['next_step_index']:04d}.admit.json"
    try:
        previous = read_published_json(path, max_bytes=8192)
    except FileNotFoundError:
        folder.mkdir(parents=True, exist_ok=True)
        publish_json(path, payload)
    else:
        if previous != payload or any((type(previous[k]) is not type(payload[k]) for k in payload)):
            raise RuntimeError('prefill admission identity changed')
    return payload

def validate_prefill_admission(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected) or any((actual[k] != v or type(actual[k]) is not type(v) for k, v in expected.items())):
        raise RuntimeError('frozen prefill admission identity differs')

def admitted_source_fingerprint(environment, *, outputs_root, native_binding, fingerprints):
    if not isinstance(native_binding, dict) or native_binding.get('role') != 'rollout':
        raise ValueError('early prefill needs the actual verified native source owner')
    root = Path(outputs_root).resolve()
    config_path = Path(environment.get('SPECO_RESOLVED_LAUNCH_CONFIG_PATH', '')).resolve()
    if not config_path.is_relative_to(root) or config_path.name != 'launch-config.json':
        raise ValueError('early prefill needs its owned native configuration')
    config = read_published_json(config_path, max_bytes=128 * 1024)
    if config.get('source_early_ready') is not True or config.get('run_id') != native_binding.get('run_id') or native_binding.get('run_id') != environment.get('SPECO_RUN_ID') or (str(native_binding.get('generation')) != environment.get('SPECO_GENERATION')) or (config.get('source_step_lookahead') is not True) or (config.get('source_native_buffer_batches') not in (2, 3, 4, 10)) or (config.get('source_tp') != 8) or (len(config.get('sources', [])) != 6) or (config.get('source_capture_owner_ranks') != 8) or (config.get('source_physical_batch_size') not in (8, 16)) or (config.get('source_graph_mode') != 'FULL_DECODE_ONLY'):
        raise ValueError('early prefill is outside the admitted frozen teacher profile')
    from verl_speco.producer.capture import config_slot_profile
    profile = config_slot_profile(config)
    if environment.get('SPECO_CAPTURE_SOURCE_SLOT_PROFILE', '') != profile:
        raise ValueError('early prefill source slot profile differs from its frozen configuration')
    recipe = read_published_json(config_path.with_name('recipe.json'), max_bytes=65536)
    logical = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    actual = fingerprints_sha256(fingerprints)
    if recipe.get('source_early_ready') is not True or logical != environment.get('SPECO_PREFILL_STEP_LOGICAL') or actual != fingerprints_sha256(recipe.get('input_fingerprints')):
        raise ValueError('actual frozen teacher inputs differ from the admitted recipe')
    return actual
from typing import Mapping, Iterable
from verl_speco.transport.streaming import SampleMetadataV3

def fingerprint_files(files: Mapping[str, str | Path]) -> str:
    """Hash named file contents in deterministic order without loading them all."""
    if not files:
        raise ValueError('at least one input file is required')
    digest = hashlib.sha256()
    for name, path_value in sorted(files.items()):
        if not name or '\x00' in name:
            raise ValueError('input file labels must be nonempty and contain no NUL')
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f'fingerprint input {name!r} is missing: {path}')
        digest.update(name.encode('utf-8'))
        digest.update(b'\x00')
        file_digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest()

def speco_92b_input_fingerprints(*, model_overlay: str | Path, target_model_dir: str | Path, prompt_dataset: str | Path, model_content_audit: str | Path | None=None, verify_shards_full: bool=False) -> dict[str, str]:
    """Bind target, tokenizer assets and prompt data to this run."""
    overlay = Path(model_overlay)
    target = Path(target_model_dir)
    if model_content_audit is not None:
        from verl_speco.runtime.recovery import load_checkpoint_audit
        model_fingerprint = load_checkpoint_audit(model_content_audit, model_overlay=overlay, target_model_dir=target, verify_shards_full=verify_shards_full)['model_content_fingerprint']
    else:
        if verify_shards_full:
            raise ValueError('full shard verification requires a content audit report')
        model_fingerprint = fingerprint_files({'config.json': overlay / 'config.json', 'model.safetensors.index.json': target / 'model.safetensors.index.json'})
    return {'target_model_fingerprint': model_fingerprint, 'tokenizer_fingerprint': fingerprint_files({'tokenizer.json': overlay / 'tokenizer.json', 'tokenizer_config.json': overlay / 'tokenizer_config.json', 'special_tokens_map.json': overlay / 'special_tokens_map.json'}), 'dataset_fingerprint': fingerprint_files({'prompt_dataset': prompt_dataset})}

def verify_speco_92b_input_fingerprints(metadata: SampleMetadataV3, *, model_overlay: str | Path, target_model_dir: str | Path, prompt_dataset: str | Path, model_content_audit: str | Path | None=None, verify_shards_full: bool=False) -> None:
    """Fail before optimizer if capture inputs changed before materialization."""
    verify_speco_92b_batch_input_fingerprints((metadata,), model_overlay=model_overlay, target_model_dir=target_model_dir, prompt_dataset=prompt_dataset, model_content_audit=model_content_audit, verify_shards_full=verify_shards_full)

def verify_speco_92b_batch_input_fingerprints(metadatas: Iterable[SampleMetadataV3], *, model_overlay: str | Path, target_model_dir: str | Path, prompt_dataset: str | Path, model_content_audit: str | Path | None=None, verify_shards_full: bool=False) -> None:
    """Read current files once per call and check every capture declaration.

    No digests survive this call: a later training step rechecks file contents.
    Mixed or missing declarations fail before the caller can materialize data.
    """
    metadatas = tuple(metadatas)
    if not metadatas:
        raise ValueError('input fingerprint verification needs capture metadata')
    for metadata in metadatas:
        metadata.validate()
        if metadata.dataset_fingerprint is None:
            raise ValueError('real dataset metadata has no dataset fingerprint')
    actual = speco_92b_input_fingerprints(model_overlay=model_overlay, target_model_dir=target_model_dir, prompt_dataset=prompt_dataset, model_content_audit=model_content_audit, verify_shards_full=verify_shards_full)
    changed = sorted({name for metadata in metadatas for name in actual if actual[name] != getattr(metadata, name)})
    if changed:
        raise ValueError('capture inputs changed before training: ' + ', '.join(changed))
import signal
import subprocess

class NativeCaptureProcess:

    def __init__(self, command, *, environment, timeout_s=1800):
        self.command = command
        self.environment = environment
        self.timeout_s = timeout_s
        self.process = None

    def __enter__(self):
        self.process = subprocess.Popen(self.command, env=self.environment, start_new_session=True)
        return self

    def wait_for_manifests(self, paths):
        deadline = time.monotonic() + self.timeout_s
        while True:
            status = self.process.poll()
            if status is not None:
                raise RuntimeError(f'native producer exited before consumer handoff: {status}')
            if all((path.is_file() for path in paths)):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('native capture manifest deadline exceeded')
            time.sleep(0.25)

    def wait(self, timeout_s=120):
        status = self.process.wait(timeout=timeout_s)
        if status != 0:
            raise RuntimeError(f'native producer shutdown failed: {status}')

    def __exit__(self, *exc):
        if self.process is None:
            return
        if self.process.poll() is None or exc[0] is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
from contextlib import ExitStack
import sys

class NativeCaptureGroup:
    """Only control manifests cross this group's readiness callback.

    The remote cleanup callback must target this run's processes by exact scope;
    terminating an SSH client alone cannot retire its remote NPU registrations.
    """

    def __init__(self, owners, *, load_manifest, retire_remote, timeout_s=1200):
        if not owners or len(owners) > 7:
            raise ValueError('native capture group requires 1..7 independent owners')
        self.owners = owners
        self.load_manifest = load_manifest
        self.retire_remote = retire_remote
        self.timeout_s = timeout_s
        self.stack = ExitStack()
        self.processes = []
        self.complete = False

    def __enter__(self):
        try:
            for command, environment in self.owners:
                owner = self.stack.enter_context(NativeCaptureProcess(command, environment=environment))
                self.processes.append(owner)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def wait_for_manifests(self):
        deadline = time.monotonic() + self.timeout_s
        ready = {}
        while len(ready) < len(self.processes):
            for index, owner in enumerate(self.processes):
                status = owner.process.poll()
                if status is not None:
                    raise RuntimeError(f'native source {index} exited before handoff: {status}')
                if index not in ready:
                    manifest = self.load_manifest(index)
                    if manifest is not None:
                        ready[index] = manifest
            if len(ready) == len(self.processes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('native source group manifest deadline')
            time.sleep(0.25)
        return [ready[i] for i in range(len(self.processes))]

    def wait(self):
        for owner in self.processes:
            owner.wait()
        self.complete = True

    def __exit__(self, *exc):
        try:
            if not self.complete:
                self.retire_remote()
        finally:
            self.stack.__exit__(*exc)

def native_control_poll_checked(value):
    if type(value) not in (int, float) or not math.isfinite(value) or (not 0.01 <= value <= 0.5):
        raise ValueError('standalone control poll seconds must be finite within0.01 to0.5')
    return float(value)

def config_control_poll_seconds(config):
    return native_control_poll_checked(config.get('control_poll_seconds', 0.5))

def environment_control_poll_seconds(environment):
    value = environment.get('SPECO_STANDALONE_CONTROL_POLL_SECONDS', '')
    if value == '':
        return 0.5
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid standalone control poll environment') from error
    return native_control_poll_checked(parsed)
import ipaddress

def native_draft_endpoint(host_ip, local_rank, generation, *, step_index=None, listener_incarnation=None):
    if type(local_rank) is not int or not 0 <= local_rank < 16 or type(generation) is not int or (generation < 1):
        raise ValueError('native draft endpoint requires a generation and local DP rank')
    host = ipaddress.ip_address(host_ip)
    epoch = generation
    if listener_incarnation is not None:
        if type(listener_incarnation) is not int or not 1 <= listener_incarnation <= 123 or (step_index is not None and (type(step_index) is not int or not 0 <= step_index < 6000000)):
            raise ValueError('invalid native epoch listener incarnation or global step')
        epoch = listener_incarnation
    elif step_index is not None:
        if type(step_index) is not int or not 0 <= step_index < 8:
            raise ValueError('native draft endpoint step must be inside the eight-step diagnostic')
        epoch = 8 * (generation - 1) + step_index + 1
    port = 61600 + 32 * (epoch - 1) + local_rank
    if port > 65535:
        raise ValueError('native draft generation exceeds the diagnostic endpoint range')
    return f"{('[' + str(host) + ']' if host.version == 6 else str(host))}:{port}"
import ipaddress, json, re

def native_trainer_roles(config):
    roles = [config.get('trainer')]
    if 'second_trainer' in config:
        roles.append(config['second_trainer'])
    for role in roles:
        if not isinstance(role, dict) or set(role) != {'host', 'address'} or (not isinstance(role['host'], str)) or (re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', role['host']) is None):
            raise ValueError('invalid native trainer host')
        if not isinstance(role['address'], str):
            raise ValueError('trainer address must be IPv4 text')
        try:
            ipaddress.IPv4Address(role['address'])
        except (ipaddress.AddressValueError, TypeError):
            raise ValueError('trainer address must be IPv4') from None
    if len(roles) == 2:
        if config.get('source_tp', 16) != 16 or not isinstance(config.get('sources'), list) or len(config['sources']) != 6 or (config.get('resolved_launch_plan') is not True):
            raise ValueError('two trainer nodes require six TP16 sources and resolved launch binding')
        members = [*config['sources'], *roles]
        if any((not isinstance(r, dict) or not isinstance(r.get('host'), str) or (not isinstance(r.get('address'), str)) for r in members)):
            raise ValueError('invalid native physical role')
        if len({r['host'] for r in members}) != 8 or len({r['address'] for r in members}) != 8:
            raise ValueError('prefill and both trainer roles require separate physical nodes')
    return tuple(roles)

def node_local_native_draft_fields(config, global_rank):
    roles = native_trainer_roles(config)
    world = 16 * len(roles)
    if type(global_rank) is not int or not 0 <= global_rank < world:
        raise ValueError('invalid actual global DP rank')
    role = roles[global_rank // 16]
    return dict(SPECO_DRAFT_SSH_HOST=role['host'], SPECO_NATIVE_DRAFT_HOST_IP=role['address'])

def verify_two_trainer_command_context(environment, values):
    """Shared commands remain identical; restore only independently bound local fields."""
    if environment.get('SPECO_DRAFT_NODE_COUNT') != '2':
        return {}
    binding_fields = ('SPECO_RESOLVED_LAUNCH_PLAN_SHA256', 'SPECO_RESOLVED_ENDPOINT_FINGERPRINT', 'SPECO_RESOLVED_LAUNCH_CONFIG_PATH', 'SPECO_RESOLVED_LAUNCH_PLAN_PATH')
    if environment.get('SPECO_DRAFT_NODE_OWNED_SWA') != '1' and (not any((environment.get(k) for k in binding_fields))):
        return {}
    from pathlib import Path
    proof = verify_native_launch_environment(environment, role='draft', outputs_root=Path(__file__).resolve().parents[2] / 'outputs')
    if proof is None:
        raise ValueError('two-trainer service requires resolved physical binding')
    from verl_speco.runtime.artifacts import read_published_json
    c = read_published_json(Path(environment['SPECO_RESOLVED_LAUNCH_CONFIG_PATH']), max_bytes=128 * 1024)
    rank = int(environment['RANK'])
    local = node_local_native_draft_fields(c, rank)
    expected = dict(SPECO_DRAFT_WORLD_SIZE='32', SPECO_DRAFT_NODE_COUNT='2', SPECO_DRAFT_NODE_OWNER_ADDRESSES=json.dumps([r['address'] for r in native_trainer_roles(c)]), SPECO_DRAFT_WAVE_PLAN_SHA256=environment['SPECO_RESOLVED_LAUNCH_PLAN_SHA256'], SPECO_DRAFT_SSH_HOST=c['trainer']['host'], SPECO_NATIVE_DRAFT_HOST_IP=c['trainer']['address'])
    if any((values.get(k) != v for k, v in expected.items())):
        raise ValueError('common DP32 command differs from actual resolved node plan')
    return local

def validate_two_trainer_wave_fields(values):
    """Require both RAM owners exactly for the resolved native DP32 path."""
    addresses = values.get('SPECO_DRAFT_NODE_OWNER_ADDRESSES', '')
    plan = values.get('SPECO_DRAFT_WAVE_PLAN_SHA256', '')
    active = values.get('SPECO_DRAFT_NODE_COUNT') == '2' and values.get('SPECO_DRAFT_NODE_OWNED_SWA') == '1'
    if not active:
        if addresses or plan:
            raise ValueError('two-trainer wave fields require two native RAM owners')
        return False
    if values.get('SPECO_DRAFT_WORLD_SIZE') != '32' or any((values.get(k) != '1' for k in ('SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_PREFILL_WAVES', 'SPECO_DRAFT_STREAM_OWNED_WINDOW_MATERIALIZE', 'SPECO_PREFILL_BATCH_RELEASE'))) or values.get('SPECO_PREFILL_WAVE_ACK_TRANSPORT') != 'ray':
        raise ValueError('two-trainer RAM staging requires global DP32 native waves and Ray batch release')
    if not isinstance(plan, str) or re.fullmatch('[0-9a-f]{64}', plan) is None:
        raise ValueError('two-trainer wave plan requires its exact resolved SHA256')
    try:
        owners = json.loads(addresses)
        if not isinstance(owners, list) or len(owners) != 2 or any((not isinstance(a, str) for a in owners)):
            raise ValueError('invalid owner address membership')
        parsed = [str(ipaddress.IPv4Address(a)) for a in owners]
        if parsed != owners or len(set(parsed)) != 2:
            raise ValueError('invalid physical owner identities')
    except (TypeError, ValueError):
        raise ValueError('two-trainer waves require two distinct canonical IPv4 owners') from None
    return True
from urllib.parse import urlsplit
from verl_speco.runtime.planning import NodeRole, resolve_launch_plan
from verl_speco.runtime.source_profiles import native_source_incarnation
from verl_speco.transport.native_transfer import native_receiver_endpoint

def epoch_listener_incarnation(config, execution):
    value = config.get('epoch_listener_incarnation')
    if value is not None and (type(value) is not int or not execution['generation'] <= value <= 123):
        raise ValueError('epoch listener incarnation must be fresh and inside reserved ports')
    return value

def resolve_native_launch_plan(config, execution):
    """Resolve the complete listener reservation without opening any resource.

    The caller first validates the native configuration. Ordered sources are
    the actual producer ordinals; each node retains its independent trainer.
    """
    sources = config['sources']
    trainers = native_trainer_roles(config)
    trainer = trainers[0]
    controller = urlsplit(config['metadata_url']).hostname
    roles = [*sources, *trainers]
    controller_role = next((r for r in roles if r['address'] == controller), None)
    table = [dict(node_id=controller_role['host'] if controller_role else 'external-control', address=controller, device_count=16)]
    for role in roles:
        found = next((n for n in table if n['address'] == role['address']), None)
        if found is None:
            table.append(dict(node_id=role['host'], address=role['address'], device_count=16))
        elif found['node_id'] != role['host']:
            raise ValueError('physical node host/address aliases differ from the resolved plan')
    index = {n['address']: i for i, n in enumerate(table)}
    rollout_indices = list(dict.fromkeys((index[r['address']] for r in sources)))
    target_tp = config.get('source_tp', 16)
    replicas = 2 if target_tp == 8 else 1
    if len(sources) != len(rollout_indices) * replicas:
        raise ValueError('resolved source replicas differ from the native producer ordinals')
    primary = native_draft_endpoint(trainer['address'], 0, execution['generation'], step_index=execution['start_step'], listener_incarnation=epoch_listener_incarnation(config, execution))
    owners = config.get('source_capture_owner_ranks', 0)
    runtime = dict(run_id=config['run_id'], generation=execution['generation'], mode='standalone', native_configuration=dict(config), topology=dict(rollout_nodes=len(rollout_indices), draft_nodes=len(trainers), rollout_node_indices=rollout_indices, draft_node_indices=[index[t['address']] for t in trainers], nproc_per_draft_node=16, draft_device_start=0, draft_tp=1, draft_pp=1, draft_sp=1, draft_ep=1, target_tp=target_tp, target_dp=len(sources), rollout_replicas_per_node=replicas, master_addr=trainer['address'], master_port=config['trainer_master_port']), transfer_queue=dict(partition='speco-92b-' + config['run_id'], ready_shards=1), ray=dict(rollout_resource='speco_rollout', draft_resource='speco_draft', control_resource='speco_control'), role_endpoints=dict(metadata_rpc_port=config['metadata_rpc_port'], metadata_http_port=config['metadata_http_port'], external_ray_port=urlsplit('//' + config['ray_address']).port, source_port_base=config['source_port_base'], source_port_stride=target_tp if owners else 1, source_owners_per_replica=owners or 1, draft_port_base=int(primary.rsplit(':', 1)[1]), draft_extra_reader_count=config.get('native_receive_workers', 1)))
    if len(trainers) == 2:
        runtime['role_endpoints']['draft_ports_per_node'] = True
    plan = resolve_launch_plan({'speco': {'multinode_runtime': runtime}}, {'nodes': table})
    placements = [p for p in plan.placements if p.role is NodeRole.ROLLOUT]
    for placement, source in zip(placements, sources, strict=True):
        if plan.nodes[placement.node_rank].address != source['address'] or placement.device_id != source.get('device_start', 0):
            raise ValueError('resolved source placement differs from its actual device slice')
    return plan

def native_logical_launch_fingerprint(plan):
    """Recovery retains logical roles while using fresh transport listeners."""
    payload = plan.to_dict()
    return hashlib.sha256(json.dumps(dict(nodes=payload['nodes'], placements=payload['placements'], producer_groups=payload['producer_groups'], draft_world_size=plan.draft_world_size), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def bind_native_role_environments(config, execution, producers, trainer, plan):
    expected = resolve_native_launch_plan(config, execution)
    if plan != expected or len(producers) != len(config['sources']):
        raise ValueError('native role binding differs from its authoritative resolved launch plan')
    endpoints = {e.key: e for e in plan.role_endpoints}
    rollout = [p for p in plan.placements if p.role is NodeRole.ROLLOUT]
    shared = dict(SPECO_NATIVE_FEATURE_TRANSPORT='1', SPECO_RESOLVED_LAUNCH_PLAN_SHA256=plan.config_fingerprint, SPECO_RESOLVED_ENDPOINT_FINGERPRINT=plan.endpoint_fingerprint, SPECO_NATIVE_METADATA_SERVER=config['metadata_url'], SPECO_TQ_PARTITION=plan.tq_partitions[0])
    if epoch_listener_incarnation(config, execution) is not None:
        shared['SPECO_NATIVE_LISTENER_INCARNATION'] = str(epoch_listener_incarnation(config, execution))
    result = []
    for ordinal, (environment, source, placement) in enumerate(zip(producers, config['sources'], rollout, strict=True)):
        if 'SPECO_NATIVE_FEATURE_TRANSPORT' in environment and environment['SPECO_NATIVE_FEATURE_TRANSPORT'] != '1':
            raise ValueError('formal native role cannot downgrade feature transport')
        incarnation = native_source_incarnation(plan, ordinal)
        if 'SPECO_PRODUCER_INCARNATION' in environment and environment['SPECO_PRODUCER_INCARNATION'] != incarnation:
            raise ValueError('formal producer incarnation differs from its resolved plan')
        values = dict(environment, **shared)
        if config.get('source_native_slot_profile'):
            values['SPECO_CAPTURE_SOURCE_SLOT_PROFILE'] = config['source_native_slot_profile']
        if config.get('source_async_publication', False):
            values['SPECO_CAPTURE_ASYNC_PUBLICATION'] = '1'
        if config.get('source_kv_aware_batching', False):
            values.update(SPECO_CAPTURE_KV_AWARE_BATCHING='1', SPECO_CAPTURE_KV_MAX_SEQUENCES=str(config['source_kv_max_sequences']))
        endpoint = endpoints[f'source:{ordinal}:0']
        values.update(SPECO_NATIVE_SOURCE_ENDPOINT=f'{endpoint.address}:{endpoint.port}', SPECO_PREFILL_EARLY_READY='1' if config.get('source_early_ready', False) else '0', SPECO_CAPTURE_LONG_CONTEXT=str(config['max_sequence_length']) if config.get('source_long_context', False) else '0', SPECO_PRODUCER_ID=f'producer-{ordinal}', SPECO_PRODUCER_INCARNATION=native_source_incarnation(plan, ordinal), SPECO_CAPTURE_DEVICE_START=str(placement.device_id), ASCEND_RT_VISIBLE_DEVICES=','.join((str(d) for d in range(placement.device_id, placement.device_id + config.get('source_tp', 16)))))
        if 'source_engine_batch_size' in config:
            values.update(SPECO_CAPTURE_TOKEN_BUDGET_BATCH_SIZE=str(config['source_engine_batch_size']), SPECO_CAPTURE_TOKEN_FILL_TARGET=str(config['source_token_fill_target']))
            if config.get('source_cross_step_token_batches'):
                values['SPECO_CAPTURE_CROSS_STEP_TOKEN_BATCHES'] = '1'
            if 'source_engine_token_budget' in config:
                values['SPECO_CAPTURE_ENGINE_TOKEN_BUDGET'] = str(config['source_engine_token_budget'])
        result.append(values)
    drafts = bind_all_native_trainer_environments(config, execution, trainer, plan)
    return (result, drafts[0])

def bind_all_native_trainer_environments(config, execution, trainer, plan):
    """Same global DP plan, local listeners and host identity for each owner."""
    if plan != resolve_native_launch_plan(config, execution):
        raise ValueError('trainer binding differs from authoritative physical plan')
    if 'SPECO_NATIVE_FEATURE_TRANSPORT' in trainer and trainer['SPECO_NATIVE_FEATURE_TRANSPORT'] != '1':
        raise ValueError('formal native role cannot downgrade feature transport')
    roles = native_trainer_roles(config)
    endpoints = {e.key: e for e in plan.role_endpoints}
    shared = dict(SPECO_NATIVE_FEATURE_TRANSPORT='1', SPECO_RESOLVED_LAUNCH_PLAN_SHA256=plan.config_fingerprint, SPECO_RESOLVED_ENDPOINT_FINGERPRINT=plan.endpoint_fingerprint, SPECO_NATIVE_METADATA_SERVER=config['metadata_url'], SPECO_TQ_PARTITION=plan.tq_partitions[0])
    if epoch_listener_incarnation(config, execution) is not None:
        shared['SPECO_NATIVE_LISTENER_INCARNATION'] = str(epoch_listener_incarnation(config, execution))
    result = []
    for node, role in enumerate(roles):
        draft = dict(trainer, **shared)
        for key, flag in (('frame_pipelined_dispatch', 'SPECO_DRAFT_PIPELINED_FRAME_DISPATCH'), ('native_receive_copy_stream', 'SPECO_NATIVE_RECEIVE_COPY_STREAM')):
            if config.get(key, False):
                draft[flag] = '1'
        draft.update(MASTER_ADDR=plan.master_addr, MASTER_PORT=str(plan.master_port), SPECO_NATIVE_DRAFT_HOST_IP=role['address'], SPECO_DRAFT_WORLD_SIZE=str(plan.draft_world_size))
        if len(roles) == 2:
            draft.update(SPECO_DRAFT_SSH_HOST=role['host'], SPECO_DRAFT_NODE_COUNT='2', SPECO_DRAFT_NODE_OWNER_ADDRESSES=json.dumps([r['address'] for r in roles]), SPECO_DRAFT_WAVE_PLAN_SHA256=plan.config_fingerprint)
        for local_rank in range(16):
            global_rank = 16 * node + local_rank
            actual = native_draft_endpoint(role['address'], local_rank, execution['generation'], step_index=execution['start_step'], listener_incarnation=epoch_listener_incarnation(config, execution))
            entry = endpoints[f'draft:{global_rank}']
            if actual != f'{entry.address}:{entry.port}':
                raise ValueError('native draft rank listener differs from its resolved reservation')
        primary = native_draft_endpoint(role['address'], 0, execution['generation'], step_index=execution['start_step'], listener_incarnation=epoch_listener_incarnation(config, execution))
        for lane in range(1, config.get('native_receive_workers', 1)):
            key = f'draft_reader:{16 * node}:{lane}' if len(roles) == 2 else f'draft_reader:{lane}'
            entry = endpoints[key]
            if native_receiver_endpoint(primary, lane) != f'{entry.address}:{entry.port}':
                raise ValueError('independent native reader differs from its resolved reservation')
        result.append(draft)
    return result

def verify_native_launch_environment(environment, *, role, outputs_root, node_identity=None):
    """Reopen committed control records on each actual role before device setup.

    Legacy roles opt out only when none of the binding fields are present.
    A partial binding never downgrades to the legacy path.
    """
    from pathlib import Path
    import socket
    from verl_speco.runtime.artifacts import read_published_json
    fields = ('SPECO_RESOLVED_LAUNCH_PLAN_SHA256', 'SPECO_RESOLVED_ENDPOINT_FINGERPRINT', 'SPECO_RESOLVED_LAUNCH_CONFIG_PATH', 'SPECO_RESOLVED_LAUNCH_PLAN_PATH')
    present = [bool(environment.get(k)) for k in fields]
    if not any(present):
        return None
    if not all(present) or role not in ('rollout', 'draft'):
        raise ValueError('native role has a partial or unsupported resolved binding')
    directory = Path(outputs_root).resolve()
    config_path = Path(environment[fields[2]]).resolve()
    plan_path = Path(environment[fields[3]]).resolve()
    if not config_path.is_relative_to(directory) or not plan_path.is_relative_to(directory) or config_path.parent != plan_path.parent or (config_path.name != 'launch-config.json') or (plan_path.name != 'resolved-launch-plan.json'):
        raise ValueError('native resolved control artifacts must stay in their shared owned launch directory')
    from verl_speco.runtime.recovery import validate_native_config, native_execution
    from verl_speco.runtime.recovery import validate_epoch_or_native_config
    config = validate_epoch_or_native_config(read_published_json(config_path, max_bytes=128 * 1024), legacy_validator=validate_native_config, outputs_root=directory)
    for key, flag in (('frame_pipelined_dispatch', 'SPECO_DRAFT_PIPELINED_FRAME_DISPATCH'), ('native_receive_copy_stream', 'SPECO_NATIVE_RECEIVE_COPY_STREAM')):
        actual = environment.get(flag, '')
        if role == 'rollout' and actual not in ('', '0') or (role == 'draft' and (config.get(key, False) and actual != '1' or (not config.get(key, False) and actual not in ('', '0')))):
            raise ValueError('actual frame/receive stream policy differs from frozen role configuration')
    if role == 'rollout':
        if environment.get('SPECO_CAPTURE_SOURCE_SLOT_PROFILE', '') != config.get('source_native_slot_profile', ''):
            raise ValueError('actual fixed source slot profile differs from frozen configuration')
        if environment.get('SPECO_CAPTURE_KV_AWARE_BATCHING', '') != ('1' if config.get('source_kv_aware_batching') else '') or environment.get('SPECO_CAPTURE_KV_MAX_SEQUENCES', '') != str(config.get('source_kv_max_sequences', '')):
            raise ValueError('actual adaptive KV source policy differs from frozen configuration')
        if environment.get('SPECO_CAPTURE_ASYNC_PUBLICATION', '') != ('1' if config.get('source_async_publication') else ''):
            raise ValueError('actual async source publication differs from frozen configuration')
        if environment.get('SPECO_CAPTURE_ENGINE_TOKEN_BUDGET', '') != str(config.get('source_engine_token_budget', '')):
            raise ValueError('actual engine token budget differs from frozen configuration')
        configured = config.get('source_engine_batch_size', 0)
        actual = environment.get('SPECO_CAPTURE_TOKEN_BUDGET_BATCH_SIZE', '')
        fill = environment.get('SPECO_CAPTURE_TOKEN_FILL_TARGET', '')
        if environment.get('SPECO_CAPTURE_CROSS_STEP_TOKEN_BATCHES', '') != ('1' if config.get('source_cross_step_token_batches') else ''):
            raise ValueError('actual cross-step token batch policy differs from frozen config')
        if configured and (actual != str(configured) or fill != str(config['source_token_fill_target'])) or (not configured and (actual not in ('', '0') or fill)):
            raise ValueError('actual token coalescing policy differs from its frozen launch configuration')
    if config.get('resolved_launch_plan') is not True:
        raise ValueError('resolved role cannot use an unbound legacy configuration')
    execution = native_execution(config)
    plan = resolve_native_launch_plan(config, execution)
    saved = read_published_json(plan_path, max_bytes=512 * 1024)
    if saved != plan.to_dict():
        raise ValueError('saved native plan differs from independently resolved launch configuration')
    expected_pe, expected_te = bind_native_role_environments(config, execution, [{} for _ in config['sources']], {}, plan)
    if role == 'rollout':
        producer = environment.get('SPECO_PRODUCER_ID', '')
        if not isinstance(producer, str) or not producer.startswith('producer-') or (not producer[9:].isascii()) or (not producer[9:].isdigit()):
            raise ValueError('resolved producer requires its canonical actual ordinal')
        ordinal = int(producer[9:])
        if producer != f'producer-{ordinal}' or ordinal >= len(expected_pe):
            raise ValueError('resolved producer ordinal differs from actual source membership')
        expected = expected_pe[ordinal]
        physical = config['sources'][ordinal]
    else:
        raw = environment.get('RANK', '')
        if not isinstance(raw, str) or not raw.isascii() or (not raw.isdigit()) or (raw != str(int(raw))) or (not 0 <= int(raw) < plan.draft_world_size):
            raise ValueError('resolved trainer requires its actual global DP rank')
        local = environment.get('LOCAL_RANK', '')
        trainers = native_trainer_roles(config)
        if local != str(int(raw) % 16) or environment.get('WORLD_SIZE') != str(plan.draft_world_size) or (len(trainers) == 2 and environment.get('LOCAL_WORLD_SIZE') != '16'):
            raise ValueError('resolved trainer rank/world differs from its DP-only placement')
        expected = bind_all_native_trainer_environments(config, execution, {}, plan)[int(raw) // 16]
        physical = trainers[int(raw) // 16]
    if environment.get('SPECO_RUN_ID') != plan.run_id or environment.get('SPECO_GENERATION') != str(plan.generation) or any((environment.get(k) != value for k, value in expected.items())):
        raise ValueError('actual native role environment differs from resolved owner, endpoints or device placement')
    identity = node_identity or (socket.gethostname(), socket.gethostbyname(socket.gethostname()))
    if identity[1] != physical['address']:
        raise ValueError('resolved role is running on a foreign physical node')
    return dict(run_id=plan.run_id, generation=plan.generation, role=role, physical_address=identity[1], endpoint_fingerprint=plan.endpoint_fingerprint, launch_config_fingerprint=plan.config_fingerprint)

def parse_npu_process_ids(output):
    """Read process-table PID cells, excluding NPU device-summary rows."""
    return {int(match.group(1)) for match in re.finditer('^\\|\\s*\\d+\\s+\\d+\\s*\\|\\s*(\\d+)\\s*\\|', output, re.MULTILINE)}

def owned_role_session_scope_dir(run_id, generation, root):
    identity = json.dumps([run_id, generation], separators=(',', ':'))
    return Path(root) / hashlib.sha256(identity.encode()).hexdigest()

def owned_role_session_process(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, state=fields[0], session=int(fields[3]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None

def register_owned_session(run_id, generation, *, root='/tmp/speco-owned-role-sessions'):
    """Create a private session before children start; keep its durable receipt."""
    if not isinstance(run_id, str) or not run_id or type(generation) is not int or (generation < 1):
        raise ValueError('invalid owned role identity')
    if os.getsid(0) != os.getpid():
        os.setsid()
    process = owned_role_session_process(os.getpid())
    assert process and process['session'] == os.getpid()
    receipt = dict(schema_version=1, run_id=run_id, generation=generation, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), **process)
    folder = owned_role_session_scope_dir(run_id, generation, root)
    folder.mkdir(parents=True, exist_ok=True, mode=448)
    destination = folder / f"{process['pid']}-{process['start_ticks']}.json"
    if destination.exists():
        if json.loads(destination.read_text()) != receipt:
            raise RuntimeError('owned session receipt changed')
    else:
        with destination.open('x') as output:
            json.dump(receipt, output, sort_keys=True)
            output.write('\n')
    return receipt

def owned_session_processes(run_id, generation, *, root='/tmp/speco-owned-role-sessions'):
    """Match same-boot session members, including orphans lacking env markers."""
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    receipts = []
    for path in owned_role_session_scope_dir(run_id, generation, root).glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('schema_version') != 1 or record.get('run_id') != run_id or record.get('generation') != generation or (record.get('boot_id') != boot) or (record.get('session') != record.get('pid')) or (type(record.get('start_ticks')) is not int):
            raise RuntimeError('owned session receipt identity differs')
        leader = owned_role_session_process(record['pid'])
        if leader and leader['start_ticks'] != record['start_ticks']:
            continue
        receipts.append(record)
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        process = owned_role_session_process(int(path.name))
        if process and process['state'] != 'Z' and (process['pid'] != os.getpid()):
            if any((process['session'] == r['session'] and process['start_ticks'] >= r['start_ticks'] for r in receipts)):
                result.append(process)
    return result

def terminate_owned_sessions(run_id, generation, *, root='/tmp/speco-owned-role-sessions'):
    """Signal only recorded session members, rechecking start times for PID reuse."""
    signalled = set()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in owned_session_processes(run_id, generation, root=root):
            current = owned_role_session_process(process['pid'])
            if current and current['state'] != 'Z' and (current['start_ticks'] == process['start_ticks']) and (current['session'] == process['session']):
                try:
                    os.kill(process['pid'], sig)
                    signalled.add(process['pid'])
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(2)
    deadline = time.monotonic() + 10
    while owned_session_processes(run_id, generation, root=root):
        if time.monotonic() >= deadline:
            raise RuntimeError('owned session processes remain after SIGKILL')
        time.sleep(0.05)
    return sorted(signalled)

def terminate_owned_session_record(record):
    """Reap one captured private session even after its leader has exited."""
    if not isinstance(record, dict) or record.get('pid') != record.get('session') or any((type(record.get(k)) is not int or record[k] < 1 for k in ('pid', 'session', 'start_ticks'))) or (record.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('invalid private session birth record')

    def members():
        leader = owned_role_session_process(record['pid'])
        if leader and leader['start_ticks'] != record['start_ticks']:
            raise RuntimeError('private session leader PID was reused')
        rows = [owned_role_session_process(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
        return [p for p in rows if p and p['state'] != 'Z' and (p['session'] == record['session']) and (p['start_ticks'] >= record['start_ticks']) and (p['pid'] != os.getpid())]
    signalled = set()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for row in members():
            current = owned_role_session_process(row['pid'])
            if current and current['start_ticks'] == row['start_ticks'] and (current['session'] == row['session']):
                try:
                    os.kill(row['pid'], sig)
                    signalled.add(row['pid'])
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(2)
    deadline = time.monotonic() + 10
    while members():
        if time.monotonic() >= deadline:
            raise RuntimeError('private session members remain after SIGKILL')
        time.sleep(0.05)
    return sorted(signalled)
from ipaddress import ip_address

def restore_native_teacher_visibility(config, *, node_ip, environment=None, npu_initialized=False):
    environment = os.environ if environment is None else environment
    expected_ip = config.get('rollout_ip')
    if not isinstance(expected_ip, str) or ip_address(expected_ip) != ip_address(node_ip):
        raise ValueError('native teacher must run on its reserved rollout node')
    if expected_ip in config.get('actor_ips', ()) or expected_ip == config.get('trainer_ip'):
        raise ValueError('native teacher node must be separate from actor and draft nodes')
    devices = config.get('native_visible_devices', list(range(8)))
    if not isinstance(devices, (list, tuple)) or len(devices) != 8 or any((type(d) is not int or not 0 <= d < 16 for d in devices)) or (len(set(devices)) != 8):
        raise ValueError('native TP8 needs eight unique reserved device IDs in 0..15')
    if type(npu_initialized) is not bool:
        raise ValueError('native device initialization state must be known')
    mask = ','.join((str(d) for d in devices))
    before = environment.get('ASCEND_RT_VISIBLE_DEVICES')
    if before != mask and npu_initialized:
        raise RuntimeError('cannot change an initialized NPU context; create a fresh owner')
    environment['ASCEND_RT_VISIBLE_DEVICES'] = mask
    return dict(node_ip=node_ip, before=before, after=mask, native_visible_devices=list(devices), changed=before != mask)
import argparse
import copy
from typing import Any, Mapping, Sequence
from verl_speco.runtime.planning import probe_runtime_capabilities
from verl_speco.runtime.planning import resolve_launch_plan

def dry_run(config: Mapping[str, Any], rank_table: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None=None, *, run_id: str | None=None, mode: str | None=None, generation: int | None=None, with_capabilities: bool=False) -> dict[str, Any]:
    """Resolve a plan without initializing NPU, Ray, TQ or serving processes."""
    if os.environ.get('PANGU_LOG_FULL_IO', '0') != '0':
        raise ValueError('PANGU_LOG_FULL_IO must be 0 before a SpeCo run')
    resolved = copy.deepcopy(dict(config))
    speco = resolved.get('speco')
    if not isinstance(speco, dict):
        raise ValueError('speco must be a mapping')
    runtime = speco.get('multinode_runtime')
    if not isinstance(runtime, dict):
        raise ValueError('speco.multinode_runtime must be a mapping')
    if run_id is not None:
        runtime['run_id'] = run_id
    if mode is not None:
        runtime['mode'] = mode
    if generation is not None:
        runtime['generation'] = generation
    plan = resolve_launch_plan(resolved, rank_table)
    result = {'launch_plan': plan.to_dict()}
    if with_capabilities:
        expected_versions = runtime.get('dependency_versions', {})
        if not isinstance(expected_versions, Mapping):
            raise ValueError('dependency_versions must be a mapping')
        report = probe_runtime_capabilities(expected_versions={str(key): str(value) for key, value in expected_versions.items()})
        if not report.ready_for_probe:
            raise ValueError('required dependencies are missing or version-mismatched: ' + ', '.join(report.required_missing))
        result['capabilities'] = report.to_dict()
        expected_sources = runtime.get('source_revisions', {})
        if not isinstance(expected_sources, Mapping):
            raise ValueError('source_revisions must be a mapping')
        result['source_revisions'] = verify_source_revisions(expected_sources)
    return result

def verify_source_revisions(expected: Mapping[str, str]) -> dict[str, str]:
    """Verify explicitly configured sibling repositories without changing them."""
    workspace = Path(__file__).resolve().parents[3]
    observed: dict[str, str] = {}
    for name, revision in expected.items():
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) is None or re.fullmatch('[0-9a-f]{40}', str(revision)) is None:
            raise ValueError(f'invalid Pangu source revision lock for {name!r}')
        process = subprocess.run(('git', '-C', str(workspace / name), 'rev-parse', 'HEAD'), capture_output=True, text=True, check=False, timeout=15)
        actual = process.stdout.strip()
        if process.returncode != 0 or actual != revision:
            raise ValueError(f"Pangu source revision mismatch: {name}: expected {revision}, got {actual or 'unavailable'}")
        observed[name] = actual
    return observed

def runtime_cli_main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Dry-run the target model/SpeCo runtime plan')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--rank-table', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--mode', choices=('standalone', 'online_rl'))
    parser.add_argument('--generation', type=int)
    parser.add_argument('--capabilities', action='store_true')
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(config, Mapping):
        parser.error('config must be a YAML mapping')
    rank_table = None
    if args.rank_table is not None:
        rank_table = json.loads(args.rank_table.read_text(encoding='utf-8'))
    result = dry_run(config, rank_table, run_id=args.run_id, mode=args.mode, generation=args.generation, with_capabilities=args.capabilities)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0
from typing import Iterable

def resolve_92b_target_layers(model_overlay: str | Path, requested: Iterable[int]) -> tuple[int, ...]:
    """Require five ordered context layers from the verified 92B geometry."""
    config = json.loads((Path(model_overlay) / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'openpangu_v2_vl_moe' or int(config.get('hidden_size', 0)) != 2560 or int(config.get('num_hidden_layers', 0)) != 46:
        raise ValueError('selected model does not have the verified 92B-VL geometry')
    layers = tuple((int(item) for item in requested))
    if len(layers) != 5 or tuple(sorted(set(layers))) != layers:
        raise ValueError('92B DSpark requires five strictly increasing target layers')
    if layers[0] < 0 or layers[-1] >= 46:
        raise ValueError('target layer is outside the 92B model configuration')
    return layers
from verl_speco.transport.streaming import make_v3_sample_key

def prefill_wave_abort_identity(samples):
    samples = tuple(samples)
    if not 1 <= len(samples) <= MAX_NATIVE_BATCH_SAMPLES or len({key for key, _ in samples}) != len(samples):
        raise ValueError('prefill abort needs distinct sample identities')
    first = samples[0][1]
    for key, metadata in samples:
        metadata.validate()
        if key != make_v3_sample_key(metadata) or metadata.run_id != first.run_id or metadata.generation != first.generation:
            raise ValueError('prefill abort sample owner differs')
    return dict(schema_version=1, run_id=first.run_id, generation=first.generation, state='PREFILL_ABORT_REQUESTED', expected_samples=len(samples), sample_keys_sha256=hashlib.sha256(json.dumps([key for key, _ in samples], separators=(',', ':')).encode()).hexdigest())

def publish_prefill_abort(ack_dir, samples):
    """Publish no features, cursors, or optimizer credit; preserve identity."""
    path = Path(ack_dir) / 'abort.json'
    expected = prefill_wave_abort_identity(samples)
    try:
        existing = read_published_json(path, max_bytes=4096)
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        publish_json(path, expected)
    else:
        if not isinstance(existing, dict) or set(existing) != set(expected) or any((existing[k] != v or type(existing[k]) is not type(v) for k, v in expected.items())):
            raise ValueError('prefill abort owner differs')

def check_prefill_abort(ack_dir, samples):
    """Fail closed on a cancellation or malformed notification before reads."""
    try:
        actual = read_published_json(Path(ack_dir) / 'abort.json', max_bytes=4096)
    except FileNotFoundError:
        return
    expected = prefill_wave_abort_identity(samples)
    if not isinstance(actual, dict) or set(actual) != set(expected) or any((actual[k] != v or type(actual[k]) is not type(v) for k, v in expected.items())):
        raise ValueError('prefill abort owner differs')
    raise RuntimeError('native prefill batch aborted by coordinator')
from verl_speco.transport.waves import MAX_MATERIALIZED_WAVE_SAMPLES
from verl_speco.trainer.feature_pipeline import load_packed_sample_control
from verl_speco.trainer.distributed_batches import plan_packed_rows, MAX_NATIVE_PREFILL_ROWS

def prefill_wave_coordinator_publish_immutable(path, payload):
    try:
        previous = read_published_json(path)
    except FileNotFoundError:
        publish_json(path, payload)
    else:
        if previous != payload:
            raise RuntimeError('prefill wave request identity changed')

def request_prefill_wave_batch(*, producer_command_dirs, draft_command_dir, environment, run_id, generation, step_index, timeout_seconds=1800, poll_seconds=0.5, check_roles=lambda: None, after_materialized=None, metadata_prefetch_depth=None, after_final_wave_commands=None, allow_incomplete_producer_round=False, epoch_plan=None):
    """Advance producer commands only after verified node-owner RAM ACKs.

    A wave ACK releases source capacity, without advancing the committed cursor.
    Only the final draft STEP_COMMITTED acknowledgement completes this request.
    Lifecycle and generation recovery remain with the calling role owner.
    """
    if type(allow_incomplete_producer_round) is not bool:
        raise ValueError('incomplete producer round permission must be boolean')
    batch_started = time.monotonic()
    if after_materialized is not None and (not callable(after_materialized)):
        raise ValueError('prefill materialization callback must be callable')
    if after_final_wave_commands is not None and (not callable(after_final_wave_commands)):
        raise ValueError('prefill final-command callback must be callable')
    if metadata_prefetch_depth is not None and (type(metadata_prefetch_depth) is not int or not 1 <= metadata_prefetch_depth <= 16):
        raise ValueError('prefill metadata prefetch depth must be an integer from 1 to 16')
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or (type(poll_seconds) not in (int, float)) or (not math.isfinite(poll_seconds)) or (not 0 < poll_seconds <= 30):
        raise ValueError('invalid prefill wave coordinator timing')
    dirs = tuple((Path(p).resolve() for p in producer_command_dirs))
    draft_dir = Path(draft_command_dir).resolve()
    ack_dir = Path(environment.get('SPECO_PREFILL_WAVE_ACK_DIR', '')).resolve()
    root = Path(__file__).resolve().parents[2] / 'outputs'
    if not 1 <= len(dirs) <= MAX_CAPTURE_PRODUCER_INSTANCES or len(set((*dirs, draft_dir, ack_dir))) != len(dirs) + 2 or any((not p.is_relative_to(root) for p in (*dirs, draft_dir, ack_dir))):
        raise ValueError('prefill wave command owners need distinct SpeCo outputs directories')
    if len(dirs) > 7 and environment.get('SPECO_TARGET_TP') != '8':
        raise ValueError('more than seven prefill instances require explicit target TP8')
    command = build_draft_command(environment, run_id=run_id, generation=generation, step_index=step_index)
    values, _ = validate_draft_command(command, run_id=run_id, generation=generation, step_index=step_index, **{'epoch_plan': epoch_plan} if epoch_plan is not None else {})
    if values['SPECO_DRAFT_PREFILL_WAVES'] != '1':
        raise ValueError('prefill wave coordinator needs native wave trainer mode')
    samples = load_packed_sample_control(path=values['SPECO_PACKED_TQ_SAMPLES_PATH'], sha256=values['SPECO_PACKED_TQ_SAMPLES_SHA256'], run_id=run_id, generation=generation)
    producer_count = len(dirs)
    if epoch_plan is not None:
        from verl_speco.runtime.recovery import validate_epoch_plan_capacity, expected_epoch_sample_keys
        validate_epoch_plan_capacity(epoch_plan)
        expected = expected_epoch_sample_keys(epoch_plan, run_id=run_id, generation=generation, step_index=step_index)
        if producer_count != epoch_plan.producer_count or [s['key'] for s in samples] != expected or allow_incomplete_producer_round != bool(len(expected) % producer_count):
            raise ValueError('epoch coordinator source topology, keys or tail permission differs from frozen plan')
    sample_start = int(values['SPECO_SAMPLE_INDEX'])
    wave_size = int(values['SPECO_PREFILL_WAVE_SIZE'])
    budget = int(values['SPECO_PREFILL_MAX_WAVE_BYTES'])
    metas = [SampleMetadataV3.from_dict(s['metadata']) for s in samples]
    if wave_size > MAX_MATERIALIZED_WAVE_SAMPLES or wave_size not in (producer_count, 2 * producer_count, 4 * producer_count, 8 * producer_count, 16 * producer_count) or (len(samples) % producer_count and (not allow_incomplete_producer_round)) or sample_start % producer_count or (budget <= 0) or any((m.producer_id != f'producer-{(sample_start + i) % producer_count}' for i, m in enumerate(metas))):
        raise ValueError('prefill wave batch must contain complete producer rounds')
    children_per_producer = wave_size // producer_count
    lookahead = values['SPECO_PREFILL_SOURCE_LOOKAHEAD'] == '1'
    if after_final_wave_commands is not None and (not lookahead):
        raise ValueError('final-command prefill admission requires bounded source lookahead')
    command_depth = 1 + int(lookahead) if metadata_prefetch_depth is None else metadata_prefetch_depth
    if command_depth > 1 and (not lookahead):
        raise ValueError('deeper metadata prefetch needs bounded source lookahead')
    if lookahead and children_per_producer not in (2, 4, 8, 16):
        raise ValueError('source lookahead requires bounded physical multi-child capture')
    waves = [samples[i:i + wave_size] for i in range(0, len(samples), wave_size)]
    if any((sum((s['metadata']['payload_nbytes'] for s in wave)) > budget for wave in waves)):
        raise ValueError('prefill wave batch exceeds source quota before commands')
    if lookahead and any((sum((s['metadata']['payload_nbytes'] for w in waves[i:i + 2] for s in w)) > budget for i in range(len(waves) - 1))):
        raise ValueError('source lookahead adjacent waves exceed global byte quota')
    row_count = int(values['SPECO_PACKING_ROW_COUNT'])
    if not 1 <= row_count <= MAX_NATIVE_PREFILL_ROWS:
        raise ValueError(f'prefill wave physical row count must be 1 to {MAX_NATIVE_PREFILL_ROWS}')
    profile = draft_training_profile(values['SPECO_DRAFT_TRAINING_PROFILE'])
    if values['SPECO_DRAFT_TRAINING_WINDOW'] != '2048':
        raise ValueError('DMTP wave coordinator requires configured training window 2048')
    plan = plan_packed_rows([m.token_count for m in metas], [m.payload_nbytes for m in metas], target_length=int(values['SPECO_PACKING_TARGET_LENGTH']), block_size=profile.block_size, max_rows=row_count, max_samples_per_row=int(values['SPECO_PACKING_MAX_SAMPLES_PER_ROW']), max_bytes_per_row=int(values['SPECO_PACKING_MAX_BYTES']))
    if plan.sample_count != len(samples):
        raise ValueError('prefill wave physical row budget defers planned samples')
    from verl_speco.trainer.owned_batches import resolve_node_frame_budget
    from verl_speco.trainer.owned_batches import StreamingOwnedSWARows
    from verl_speco.trainer.distributed_checkpoint import draft_step_seed
    explicit = values['SPECO_DRAFT_NODE_FRAME_MAX_BYTES']
    frame_budget = resolve_node_frame_budget(int(values['SPECO_PACKING_MAX_BYTES']), int(explicit) if explicit else None)
    world = int(values['SPECO_DRAFT_WORLD_SIZE'])
    if not 1 <= world <= 128:
        raise ValueError('invalid wave trainer DP size')
    for rank in range(world):
        estimate = StreamingOwnedSWARows(metas, target_hidden_size=2560, target_length=int(values['SPECO_PACKING_TARGET_LENGTH']), max_bytes_per_row=int(values['SPECO_PACKING_MAX_BYTES']), rank=rank, world_size=world, max_rows=row_count, max_samples_per_row=int(values['SPECO_PACKING_MAX_SAMPLES_PER_ROW']), draft_train_length=profile.block_size * profile.num_anchors, block_size=profile.block_size, window_size=profile.swa_window, chunk_cap=profile.swa_chunk_cap, seed=draft_step_seed(int(values['SPECO_DRAFT_TRAINING_SEED'] or '42'), sample_start, 0))
        if estimate.planned_frame_upper_bound_bytes > frame_budget:
            raise ValueError('owned SWA halos exceed node frame budget before source commands')
    rounds = len(waves)
    ack_transport = values.get('SPECO_PREFILL_WAVE_ACK_TRANSPORT', '') or 'filesystem'
    if ack_transport not in ('filesystem', 'ray'):
        raise ValueError('invalid prefill wave ACK transport')
    ack_actor = None
    if ack_transport == 'ray':
        import ray
        from verl_speco.transport.waves import wave_batch_identity
        batch_id = wave_batch_identity(tuple(((s['key'], m) for s, m in zip(samples, metas))))
        ack_actor = ray.get_actor(values['SPECO_CONTROL_ACTOR_NAME'], namespace='transfer_queue')
        status = ray.get(ack_actor.status.remote(), timeout=30)
        if status['run_id'] != run_id or status['generation'] != generation:
            raise ValueError('prefill wave ACK actor owner differs')
    if wave_size > 56:
        require_live_descriptor_wave_capacity(min(wave_size, len(samples)))
    planned_commands = []
    for wave_index in range(rounds):
        wave_commands = []
        source_step = sample_start // producer_count + wave_index * children_per_producer
        for producer_index, path in enumerate(dirs):
            sample_count = len(range(producer_index, len(waves[wave_index]), producer_count))
            if not sample_count:
                continue
            if children_per_producer == 1:
                payload = dict(schema_version=1, run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=source_step, sequence_no=source_step * producer_count + producer_index)
            else:
                payload = build_physical_capture_command(run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=source_step, sample_count=sample_count)
            wave_commands.append((path / f'step-{source_step:04d}.request.json', payload))
        planned_commands.append(tuple(wave_commands))
    for path, payload in [(draft_dir / f'step-{step_index:04d}.request.json', command), *(entry for wave in planned_commands for entry in wave)]:
        try:
            previous = read_published_json(path)
        except FileNotFoundError:
            continue
        if previous != payload:
            raise RuntimeError('prefill wave request identity changed')
    for path in (*dirs, draft_dir, ack_dir):
        path.mkdir(parents=True, exist_ok=True)
    owned_samples = tuple(((sample['key'], metadata) for sample, metadata in zip(samples, metas)))
    check_prefill_abort(ack_dir, owned_samples)
    prefill_wave_coordinator_publish_immutable(draft_dir / f'step-{step_index:04d}.request.json', command)
    published_waves = set()
    final_commands_notified = False
    try:
        for wave_index, wave in enumerate(waves):
            wave_started = time.perf_counter()
            for source_wave in range(wave_index, min(rounds, wave_index + command_depth)):
                if source_wave in published_waves:
                    continue
                check_prefill_abort(ack_dir, owned_samples)
                for path, payload in planned_commands[source_wave]:
                    prefill_wave_coordinator_publish_immutable(path, payload)
                published_waves.add(source_wave)
            if len(published_waves) == rounds and (not final_commands_notified):
                final_commands_notified = True
                if after_final_wave_commands is not None:
                    after_final_wave_commands()
            publish_seconds = time.perf_counter() - wave_started
            wait_started = time.perf_counter()
            deadline = time.monotonic() + timeout_seconds
            while True:
                check_roles()
                try:
                    if ack_actor is None:
                        ack = read_published_json(ack_dir / f'wave-{wave_index:04d}.ack.json')
                    else:
                        ack = ray.get(ack_actor.get_materialized_wave_ack.remote(batch_id=batch_id, wave_index=wave_index), timeout=30)
                        if ack is None:
                            raise FileNotFoundError('wave hint not published')
                except FileNotFoundError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('prefill wave materialization timed out')
                    time.sleep(min(poll_seconds, max(0.001, deadline - time.monotonic())))
                    continue
                break
            expected = dict(schema_version=1, run_id=run_id, generation=generation, wave_index=wave_index, state='PREFILL_WAVE_MATERIALIZED', sample_keys=[s['key'] for s in wave], staged_samples=min((wave_index + 1) * wave_size, len(samples)), expected_samples=len(samples), optimizer_started=False, committed_original_tokens=0)
            if not isinstance(ack, dict) or set(ack) != set(expected) or any((ack[k] != v or type(ack[k]) is not type(v) for k, v in expected.items())):
                raise RuntimeError('prefill wave acknowledgement differs from planned RAM materialization')
            print('prefill_wave_coordinator_timing ' + json.dumps(dict(run_id=run_id, generation=generation, step_index=step_index, wave_index=wave_index, command_publication_seconds=publish_seconds, ack_wait_seconds=time.perf_counter() - wait_started, metadata_prefetch_depth=command_depth, published_not_materialized_waves=len(published_waves) - wave_index - 1, resident_wave_limit=1 + int(lookahead), global_payload_quota_bytes=budget, wallclock_seconds=time.perf_counter() - wave_started)), flush=True)
        if after_materialized is not None:
            after_materialized()
        committed = request_draft_batch(command_dir=draft_dir, environment=values, run_id=run_id, generation=generation, step_index=step_index, timeout_seconds=timeout_seconds, check_roles=check_roles, **{'epoch_plan': epoch_plan} if epoch_plan is not None else {})
        batch_ended = time.monotonic()
        elapsed = batch_ended - batch_started
        original_tokens = sum((m.token_count for m in metas))
        print('standalone_batch_commit_timing ' + json.dumps(dict(run_id=run_id, generation=generation, step_index=step_index, state=committed['state'], original_tokens=original_tokens, wallclock_seconds=elapsed, original_token_s=original_tokens / elapsed, ack_transport=ack_transport, scope='coordinator_entry_to_verified_step_commit_ack', includes_checkpoint=True, excludes_postcommit_evidence_fetch=True, production_sla_pass=False)), flush=True)
        if 'checkpoint_timing' in committed:
            import socket
            try:
                performance = checkpoint_excluded_batch_timing(batch_start=batch_started, batch_end=batch_ended, original_tokens=original_tokens, checkpoint=committed['checkpoint_timing'], node_hostname=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            except ValueError as error:
                print('standalone_checkpoint_excluded_timing_unavailable ' + json.dumps(dict(run_id=run_id, generation=generation, step_index=step_index, reason=str(error))), flush=True)
            else:
                print('standalone_checkpoint_excluded_timing ' + json.dumps(dict(run_id=run_id, generation=generation, step_index=step_index, state=committed['state'], **performance, **packing_fill_summary(plan, [m.token_count for m in metas]))), flush=True)
        return committed
    except BaseException:
        publish_prefill_abort(ack_dir, owned_samples)
        raise
import shlex

def producer_ssh_command(*, host: str, environment: Mapping[str, str], python_bin: str, script: Path) -> list[str]:
    """Launch an isolated TP8 capture producer on a second node."""
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', host) is None:
        raise ValueError('producer SSH host is invalid')
    capture_target_parallelism(environment)
    if not environment.get('SPECO_EXTERNAL_RAY_ADDRESS') or not environment.get('PANGU_MODEL_PATH'):
        raise ValueError('remote producer needs Ray/TQ and a model overlay')
    forwarded = {key: value for key, value in environment.items() if key.startswith(('SPECO_', 'HCCL_', 'PANGU_', 'VLLM_', 'ATB_')) or key in {'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'NO_PROXY', 'no_proxy', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'CPU_AFFINITY_CONF', 'LCCL_DETERMINISTIC', 'CLOSE_MATMUL_K_SHIFT', 'PYTORCH_NPU_ALLOC_CONF', 'TASK_QUEUE_ENABLE'}}
    args = [python_bin, str(script)]
    remote = 'env ' + ' '.join((f'{key}={shlex.quote(value)}' for key, value in sorted(forwarded.items()))) + ' ' + shlex.join(args)
    return ['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', host, remote]

def draft_ssh_command(*, host: str, environment: Mapping[str, str], python_bin: str, script: Path, world_size: int, master_port: int) -> list[str]:
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', host) is None:
        raise ValueError('draft SSH host is invalid')
    if world_size < 2 or world_size > 16:
        raise ValueError('draft SSH world size must be between 2 and 16')
    if master_port < 1 or master_port > 65535:
        raise ValueError('draft SSH master port is invalid')
    forwarded = {key: value for key, value in environment.items() if key.startswith(('SPECO_', 'HCCL_')) or key in {'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_CUSTOM_OPP_PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'NO_PROXY', 'no_proxy', 'PYTHONDONTWRITEBYTECODE', 'MASTER_ADDR', 'MASTER_PORT', 'PANGU_LOG_FULL_IO', 'CPU_AFFINITY_CONF', 'LCCL_DETERMINISTIC', 'ATB_MATMUL_SHUFFLE_K_ENABLE', 'CLOSE_MATMUL_K_SHIFT', 'PYTORCH_NPU_ALLOC_CONF', 'TASK_QUEUE_ENABLE'}}
    forwarded.update(cpu_thread_launch_environment(environment))
    if not forwarded.get('SPECO_EXTERNAL_RAY_ADDRESS'):
        raise ValueError('cross-node draft requires an external Ray/TQ owner')
    master_addr = environment.get('MASTER_ADDR', '127.0.0.1')
    if not isinstance(master_addr, str) or re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', master_addr) is None:
        raise ValueError('draft rendezvous address is invalid')
    args = [python_bin, '-m', 'torch.distributed.run', f'--nproc_per_node={world_size}', f'--master_addr={master_addr}', f'--master_port={master_port}', str(script)]
    remote = 'env ' + ' '.join((f'{key}={shlex.quote(value)}' for key, value in sorted(forwarded.items()))) + ' ' + shlex.join(args)
    return ['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', host, remote]

def draft_multinode_ssh_command(*, host: str, environment: Mapping[str, str], python_bin: str, script: Path, node_rank: int, node_count: int, local_world_size: int, master_addr: str, master_port: int) -> list[str]:
    """Launch one trainer node in a shared HCCL DP process group."""
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', host) is None or re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', master_addr) is None:
        raise ValueError('draft host or rendezvous address is invalid')
    if node_count < 2 or not 0 <= node_rank < node_count or local_world_size <= 0:
        raise ValueError('multi-node draft ranks are invalid')
    if not 1 <= master_port <= 65535:
        raise ValueError('draft rendezvous port is invalid')
    forwarded = {key: value for key, value in environment.items() if key.startswith(('SPECO_', 'HCCL_')) or key in {'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_CUSTOM_OPP_PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'NO_PROXY', 'no_proxy', 'PYTHONDONTWRITEBYTECODE', 'MASTER_ADDR', 'MASTER_PORT', 'PANGU_LOG_FULL_IO', 'CPU_AFFINITY_CONF', 'LCCL_DETERMINISTIC', 'ATB_MATMUL_SHUFFLE_K_ENABLE', 'CLOSE_MATMUL_K_SHIFT', 'PYTORCH_NPU_ALLOC_CONF', 'TASK_QUEUE_ENABLE'}}
    forwarded.update(cpu_thread_launch_environment(environment))
    if not forwarded.get('SPECO_EXTERNAL_RAY_ADDRESS'):
        raise ValueError('multi-node draft requires an external Ray/TQ owner')
    args = [python_bin, '-m', 'torch.distributed.run', f'--nnodes={node_count}', f'--nproc_per_node={local_world_size}', f'--node_rank={node_rank}', f'--master_addr={master_addr}', f'--master_port={master_port}', str(script)]
    remote = 'env ' + ' '.join((f'{key}={shlex.quote(value)}' for key, value in sorted(forwarded.items()))) + ' ' + shlex.join(args)
    return ['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', host, remote]

def fetch_remote_receipt(*, host: str, path: Path, run_id: str, generation: int, draft_world_size: int, sample_index: int, expected_cursor: int | None=None, expected_receipt_sha256: str | None=None) -> dict:
    """Copy a committed remote step receipt after checking its stream identity."""
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*', host) is None:
        raise ValueError('draft SSH host is invalid')
    if not path.is_absolute() or not path.is_relative_to('/tmp'):
        raise ValueError('cross-node receipt must be under /tmp')
    if expected_receipt_sha256 is not None and (not isinstance(expected_receipt_sha256, str) or re.fullmatch('[a-f0-9]{64}', expected_receipt_sha256) is None):
        raise ValueError('expected remote receipt SHA256 is invalid')
    command = ['ssh', '-p', '2222', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', host, 'cat ' + shlex.quote(str(path))]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    if expected_receipt_sha256 is not None and hashlib.sha256(result.stdout.encode('utf-8')).hexdigest() != expected_receipt_sha256:
        raise RuntimeError('remote receipt differs from verified step acknowledgement SHA256')
    receipt = json.loads(result.stdout)
    from verl_speco.transport.streaming import decode_stream_snapshot
    stream = decode_stream_snapshot(receipt.get('stream_snapshot'))
    committed = stream.get('committed_optimizer_steps') if isinstance(stream, dict) else None
    if expected_cursor is None:
        expected_cursor = sample_index + 1
    if expected_cursor <= sample_index:
        raise ValueError('expected cursor must advance past the sample index')
    if receipt.get('schema_version') != 1 or receipt.get('run_id') != run_id or receipt.get('generation') != generation or (receipt.get('draft_world_size') != draft_world_size) or (receipt.get('sample_index') != sample_index) or (not receipt.get('step_id')) or (not receipt.get('candidate')) or (not isinstance(stream, dict)) or (stream.get('committed_cursor') != expected_cursor) or (not isinstance(committed, dict)) or (receipt['step_id'] not in committed):
        raise RuntimeError('remote receipt does not prove the expected committed step')
    snapshot = receipt.get('control_snapshot', {})
    if snapshot.get('storage'):
        read_shared_control_snapshot(snapshot, run_id=run_id, generation=generation)
    else:
        snapshot_path = Path(str(snapshot.get('path', '')))
        if snapshot_path != path.with_name(path.stem + '-control.json'):
            raise RuntimeError('remote receipt control snapshot is unavailable on the owner node')
        if not snapshot_path.is_file():
            snapshot_command = command[:-1] + ['cat ' + shlex.quote(str(snapshot_path))]
            fetched = subprocess.run(snapshot_command, check=True, capture_output=True, timeout=30)
            persist_control_snapshot(snapshot_path, {'payload': fetched.stdout, 'sha256': snapshot.get('sha256')}, run_id=run_id, generation=generation)
        if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != snapshot.get('sha256'):
            raise RuntimeError('remote receipt control snapshot is unavailable on the owner node')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.remote-receipt-', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(receipt, handle, sort_keys=True, separators=(',', ':'))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    return receipt
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

@dataclass
class ReusedDraftEngine:
    binding: dict | None = None
    value: Any = None
    last_step_id: str | None = None
    last_candidate: dict | None = None
    commits: int = 0
    committed_steps: set[str] = field(default_factory=set)

    def release(self):
        """Drop model/optimizer references before distributed runtime shutdown."""
        self.value = None
        self.binding = None
        self.last_step_id = None
        self.last_candidate = None
        self.commits = 0
        self.committed_steps.clear()

    def get(self, binding: Mapping, factory: Callable):
        if self.binding is None:
            value = factory()
            self.binding = copy.deepcopy(dict(binding))
            self.value = value
            return (value, True)
        if dict(binding) != self.binding:
            raise RuntimeError('persistent draft engine binding changed')
        return (self.value, False)

    def require_previous(self, receipt: Mapping):
        if self.last_step_id is None or receipt.get('step_id') != self.last_step_id or receipt.get('candidate') != self.last_candidate or (receipt.get('run_id') != self.binding['run_id']) or (receipt.get('generation') != self.binding['generation']):
            raise RuntimeError('persistent optimizer does not own the committed predecessor')

    def commit(self, *, step_id: str, candidate: Mapping):
        if self.binding is None or step_id in self.committed_steps:
            raise RuntimeError('persistent engine cannot commit an uninitialized or duplicate update')
        self.last_step_id = step_id
        self.last_candidate = copy.deepcopy(dict(candidate))
        self.committed_steps.add(step_id)
        self.commits += 1

def checkpoint_excluded_batch_timing(*, batch_start, batch_end, original_tokens, checkpoint, node_hostname, boot_id):
    """Use one owner's monotonic clock. Never subtract parallel rank sums."""
    if any((type(x) not in (int, float) or not math.isfinite(x) for x in (batch_start, batch_end))) or batch_start < 0 or batch_end <= batch_start or (type(original_tokens) is not int) or (original_tokens <= 0):
        raise ValueError('invalid completed batch time or original token count')
    expected = {'clock', 'node_hostname', 'boot_id', 'start', 'end'}
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected:
        raise ValueError('checkpoint interval schema differs')
    if checkpoint['clock'] != 'monotonic' or checkpoint['node_hostname'] != node_hostname or checkpoint['boot_id'] != boot_id or (not node_hostname) or (not boot_id):
        raise ValueError('checkpoint and coordinator must share one node boot and clock')
    start, end = (checkpoint['start'], checkpoint['end'])
    if any((type(x) not in (int, float) or not math.isfinite(x) for x in (start, end))) or not batch_start <= start < end <= batch_end:
        raise ValueError('checkpoint interval is outside the committed batch')
    whole = batch_end - batch_start
    excluded = end - start
    elapsed = whole - excluded
    if elapsed <= 0:
        raise ValueError('completed batch must retain prefill and draft computation time')
    return dict(original_tokens=original_tokens, wallclock_seconds=whole, checkpoint_seconds=excluded, checkpoint_excluded_seconds=elapsed, original_token_s=original_tokens / elapsed, includes_prompt=True, includes_response=True, includes_checkpoint=False, excludes_warmup=False, scope='completed_prefill_transfer_draft_update_excludes_checkpoint_save_interval', interval_owner=node_hostname, interval_boot_id=boot_id, interval_clock='monotonic')

def packing_fill_summary(plan, original_token_counts):
    """Report total capacity including the tail; omit no inconvenient rows."""
    rows = tuple(plan.rows)
    if not rows or plan.original_tokens <= 0:
        raise ValueError('packing report requires real original tokens and rows')
    capacity = sum((row.target_length for row in rows))
    used = sum((row.used_tokens for row in rows))
    counts = tuple(original_token_counts)
    indices = tuple((i for group in plan.sample_indices for i in group))
    if len(counts) != plan.sample_count or sorted(indices) != list(range(plan.sample_count)) or any((type(v) is not int or v <= 0 for v in counts)) or (sum(counts) != plan.original_tokens):
        raise ValueError('packing source identities or original accounting differs')
    original_rows = [sum((counts[i] for i in group)) for group in plan.sample_indices]
    if not plan.original_tokens <= used <= capacity:
        raise ValueError('packing original or aligned tokens exceed row capacity')
    return dict(packing_rows=len(rows), packing_capacity_tokens=capacity, packing_original_tokens=plan.original_tokens, packing_alignment_tokens=used - plan.original_tokens, packing_original_fill_ratio=plan.original_tokens / capacity, packing_aligned_fill_ratio=used / capacity, packing_rows_ge90=sum((n / row.target_length >= 0.9 for n, row in zip(original_rows, rows))), packing_nonterminal_rows=len(rows) - 1, packing_nonterminal_rows_ge90=sum((n / row.target_length >= 0.9 for n, row in zip(original_rows[:-1], rows[:-1]))), packing_tail_fill_ratio=original_rows[-1] / rows[-1].target_length, packing_fill_target=0.9, packing_original_token_count_excludes_alignment=True)

def checkpoint_excluded_steady_stage_timing(completions, *, node_hostname, boot_id):
    """Count every interstep control/audit/retirement gap after the first update."""
    import re
    if not isinstance(completions, (list, tuple)) or len(completions) < 2:
        raise ValueError('steady timing requires warmup and a completed successor')
    previous = None
    receipts = set()
    measured = []
    for row in completions:
        if not isinstance(row, dict) or row.get('state') != 'STEP_COMMITTED' or type(row.get('step_index')) is not int or (row['step_index'] < 0) or (type(row.get('original_tokens')) is not int) or (row['original_tokens'] <= 0) or (not isinstance(row.get('receipt_sha256'), str)) or (re.fullmatch('[a-f0-9]{64}', row['receipt_sha256']) is None) or (row['receipt_sha256'] in receipts) or (type(row.get('completed_at')) not in (int, float)) or (not math.isfinite(row['completed_at'])) or (row['completed_at'] < 0):
            raise ValueError('steady timing needs distinct real durable completions')
        receipts.add(row['receipt_sha256'])
        if previous is not None:
            if row['step_index'] != previous['step_index'] + 1:
                raise ValueError('steady timing cannot omit or repeat committed steps')
            if row.get('checkpoint_saved', True) is False:
                if row.get('checkpoint_timing') is not None:
                    raise ValueError('unsaved progress cannot exclude checkpoint time')
                wall = row['completed_at'] - previous['completed_at']
                if wall <= 0:
                    raise ValueError('live progress must advance monotonic time')
                value = dict(original_tokens=row['original_tokens'], wallclock_seconds=wall, checkpoint_seconds=0.0, checkpoint_excluded_seconds=wall, original_token_s=row['original_tokens'] / wall, scope='live_optimizer_progress_no_checkpoint_time_excluded')
            else:
                value = checkpoint_excluded_batch_timing(batch_start=previous['completed_at'], batch_end=row['completed_at'], original_tokens=row['original_tokens'], checkpoint=row['checkpoint_timing'], node_hostname=node_hostname, boot_id=boot_id)
            measured.append(dict(step_index=row['step_index'], receipt_sha256=row['receipt_sha256'], **value))
        previous = row
    tokens = sum((row['original_tokens'] for row in measured))
    seconds = sum((row['checkpoint_excluded_seconds'] for row in measured))
    return dict(original_tokens=tokens, checkpoint_excluded_seconds=seconds, original_token_s=tokens / seconds, checkpoint_seconds=sum((row['checkpoint_seconds'] for row in measured)), wallclock_seconds=completions[-1]['completed_at'] - completions[0]['completed_at'], warmup_step_index=completions[0]['step_index'], measured_step_count=len(measured), includes_prompt=True, includes_response=True, includes_interstep_control=True, includes_postcommit_audit=True, includes_checkpoint_retirement=True, excludes_warmup=True, includes_checkpoint=False, minimum_required=50000, minimum_passed=tokens / seconds >= 50000, scope='durable_audited_completions_including_interstep_gaps_excludes_only_owner_save_intervals', measured_steps=measured)

def token_budget_engine_batch_size(environment):
    flag = environment.get('SPECO_CAPTURE_TOKEN_BUDGET_BATCH_SIZE', '')
    fill = environment.get('SPECO_CAPTURE_TOKEN_FILL_TARGET', '')
    cross = environment.get('SPECO_CAPTURE_CROSS_STEP_TOKEN_BATCHES', '')
    engine_tokens = environment.get('SPECO_CAPTURE_ENGINE_TOKEN_BUDGET', '')
    if engine_tokens not in ('', '65536') or (engine_tokens and cross != '1'):
        raise ValueError('engine token budget needs explicit cross-step coalescing')
    if cross not in ('', '0', '1'):
        raise ValueError('invalid cross-step token batch policy')
    if flag in ('', '0'):
        if fill or cross == '1':
            raise ValueError('token fill target requires explicit coalescing')
        return 0
    from verl_speco.producer.capture import environment_slot_profile
    slot_profile = environment_slot_profile(environment)
    exact = dict(SPECO_TARGET_TP='8', SPECO_CAPTURE_PHYSICAL_BATCH_SIZE='8', SPECO_CAPTURE_NATIVE_SOURCE_SLOTS='8', SPECO_CAPTURE_SOURCE_OWNER_RANKS='8', SPECO_CAPTURE_NATIVE_BUFFER_BATCHES='10' if slot_profile else '4', SPECO_CAPTURE_MAX_BATCHED_TOKENS='131072', SPECO_CAPTURE_GRAPH_MODE='FULL_DECODE_ONLY', SPECO_SEQUENCE_FULL_DECODE='1', SPECO_NATIVE_FEATURE_TRANSPORT='1', SPECO_CAPTURE_REUSE_ENGINE='1', SPECO_PREFILL_SOURCE_LOOKAHEAD='1', SPECO_PREFILL_STEP_LOOKAHEAD='1', SPECO_PREFILL_EARLY_READY='1', SPECO_PREFILL_EXACT_WAVE_ACQUIRE='1', SPECO_CAPTURE_FULL_BATCH_PREFILL='1', SPECO_MHC_SEQUENCE_PARALLEL='1', SPECO_MHC_SP_PREPROFILE='1', SPECO_CAPTURE_TOKEN_FILL_TARGET='0.95' if cross == '1' else '0.9')
    if flag != '40' or any((environment.get(k) != v for k, v in exact.items())):
        raise ValueError('token coalescing requires the exact FULL TP8 four-bank profile')
    context = environment.get('SPECO_TARGET_MAX_SEQUENCE_LENGTH')
    if context not in ('16384', '49152') or (context == '49152' and environment.get('SPECO_CAPTURE_LONG_CONTEXT') != '49152'):
        raise ValueError('token coalescing context differs from the admitted source')
    return 40

def token_budget_engine_tokens(environment):
    if environment.get('SPECO_CAPTURE_ENGINE_TOKEN_BUDGET', ''):
        if token_budget_engine_batch_size(environment) != 40:
            raise ValueError('unadmitted engine token budget')
        return 65536
    return 131072

def close_external_tq_client(client: Any) -> None:
    """Join client threads before context destruction and interpreter finalizers."""
    client.close()
    thread = getattr(client, '_thread', None)
    if thread is not None and thread.is_alive():
        raise RuntimeError('TQ client thread survived explicit shutdown')
    loop = getattr(client, '_loop', None)
    if loop is not None and (not loop.is_closed()):
        raise RuntimeError('TQ client loop survived explicit shutdown')
    context = getattr(client, 'zmq_context', None)
    if context is not None and (not context.closed):
        raise RuntimeError('TQ client context survived explicit shutdown')
import importlib
import importlib.abc
import threading
from types import ModuleType
tq_import_LOCK = threading.RLock()
tq_import_SIMPLE_IMPORT = False

class tq_import_UnselectedMooncake(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mooncake.store' and os.environ.get('SPECO_TQ_STORAGE_BACKEND', 'SimpleStorage') == 'SimpleStorage':
            raise ModuleNotFoundError('SpeCo SimpleStorage does not load optional mooncake.store', name=fullname)
        return None

def import_transfer_queue() -> ModuleType:
    """Honor the selected backend and remove the temporary import guard on failure.

    Backend selection must precede the first import. Switching to Mooncake after
    a SimpleStorage import requires a fresh process because TQ caches its probe.
    This does not repair an already loaded native extension.
    """
    global _SIMPLE_IMPORT
    backend = os.environ.get('SPECO_TQ_STORAGE_BACKEND', 'SimpleStorage')
    if backend not in {'SimpleStorage', 'MooncakeStore'}:
        raise ValueError(f'unsupported TQ storage backend: {backend!r}')
    with tq_import_LOCK:
        if backend == 'MooncakeStore':
            if tq_import_SIMPLE_IMPORT:
                raise RuntimeError('MooncakeStore requires a fresh process after SimpleStorage import')
            return importlib.import_module('transfer_queue')
        if 'mooncake.store' in sys.modules:
            raise RuntimeError('SimpleStorage isolation requires mooncake.store to remain unloaded')
        guard = tq_import_UnselectedMooncake()
        sys.meta_path.insert(0, guard)
        try:
            module = importlib.import_module('transfer_queue')
        finally:
            sys.meta_path.remove(guard)
        tq_import_SIMPLE_IMPORT = True
        return module

def install_simple_storage_import_guard() -> None:
    """Protect eager dependency imports that happen before a SpeCo TQ call.

    Ray/VeRL can import TQ themselves. Install before those dependencies are
    loaded; the finder honors an explicit MooncakeStore environment selection.
    """
    if not any((isinstance(finder, tq_import_UnselectedMooncake) for finder in sys.meta_path)):
        sys.meta_path.insert(0, tq_import_UnselectedMooncake())

def bootstrap_transfer_queue(*, address: str, expected_nodes: int=8, storage_units: int=16, capacity_samples: int=64, max_inflight_samples: int | None=None) -> dict:
    if not address or not 1 <= expected_nodes <= 8 or storage_units < expected_nodes or (capacity_samples < storage_units):
        raise ValueError('invalid external TQ address, node count or storage capacity')
    per_unit_capacity = (capacity_samples + storage_units - 1) // storage_units
    if max_inflight_samples is not None:
        from verl_speco.transport.waves import MAX_MATERIALIZED_WAVE_SAMPLES
        if type(max_inflight_samples) is not int or not 1 <= max_inflight_samples <= MAX_MATERIALIZED_WAVE_SAMPLES:
            raise ValueError('invalid maximum simultaneous descriptor samples')
        if per_unit_capacity < max_inflight_samples:
            raise ValueError('per-unit descriptor capacity cannot hold the admitted wave')
    import ray
    tq = import_transfer_queue()
    from omegaconf import OmegaConf
    from ray.job_config import JobConfig
    if ray.is_initialized():
        raise RuntimeError('bootstrap requires its own Ray session with detached actor lifetime')
    ray.init(address=address, namespace='transfer_queue', job_config=JobConfig(default_actor_lifetime='detached'))
    initialized = False
    try:
        nodes = [node for node in ray.nodes() if node['Alive']]
        if len(nodes) != expected_nodes:
            raise RuntimeError('external Ray cluster does not have the expected alive nodes')
        try:
            ray.get_actor('TransferQueueController', namespace='transfer_queue')
        except ValueError:
            pass
        else:
            raise RuntimeError('external TQ already exists; attach without replacing its owner')
        tq.init(OmegaConf.create({'backend': {'SimpleStorage': {'num_data_storage_units': storage_units, 'total_storage_size': capacity_samples}}}))
        initialized = True
        ray.get_actor('TransferQueueController', namespace='transfer_queue')
        return dict(address=address, namespace='transfer_queue', actor_lifetime='detached', alive_nodes=expected_nodes, storage_units=storage_units, capacity_samples=capacity_samples, per_unit_capacity_samples=per_unit_capacity, max_inflight_samples=max_inflight_samples, node_addresses=sorted((node['NodeManagerAddress'] for node in nodes)))
    finally:
        try:
            if initialized:
                close_external_tq_client(tq.get_client())
        finally:
            ray.shutdown()

def tq_service_bootstrap_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--expected-nodes', type=int, default=8)
    parser.add_argument('--storage-units', type=int, default=16)
    parser.add_argument('--capacity-samples', type=int, default=64)
    parser.add_argument('--max-inflight-samples', type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(bootstrap_transfer_queue(address=args.address, expected_nodes=args.expected_nodes, storage_units=args.storage_units, capacity_samples=args.capacity_samples, max_inflight_samples=args.max_inflight_samples), sort_keys=True), flush=True)
from collections.abc import Mapping

def validate_descriptor_wave_capacity(config, samples, *, resident_waves=1, source_slot_profile=''):
    if type(samples) is not int or not 1 <= samples <= 112:
        raise ValueError('invalid simultaneous descriptor wave size')
    from verl_speco.producer.capture import PROFILE
    if type(source_slot_profile) is not str or source_slot_profile not in ('', PROFILE) or bool(source_slot_profile) != (resident_waves == 10) or (resident_waves == 10 and samples != 48):
        raise ValueError('tiered descriptor quota requires the exact named forty-eight-child wave')
    if type(resident_waves) is not int or resident_waves not in (1, 2, 3, 4, 10):
        raise ValueError('invalid resident descriptor wave count')
    inflight = samples * resident_waves
    if not isinstance(config, Mapping):
        raise ValueError('external TQ storage configuration is unavailable')
    backend = config.get('backend')
    storage = backend.get('SimpleStorage') if isinstance(backend, Mapping) else None
    if not isinstance(storage, Mapping):
        raise ValueError('external descriptor service requires explicit SimpleStorage')
    units = storage.get('num_data_storage_units')
    total = storage.get('total_storage_size')
    if type(units) is not int or not 1 <= units <= 128 or type(total) is not int or (not units <= total <= 2 ** 31):
        raise ValueError('external descriptor service has invalid capacity')
    per_unit = (total + units - 1) // units
    if per_unit < inflight:
        raise ValueError('live per-unit descriptor capacity cannot hold the admitted wave')
    return dict(storage_units=units, capacity_samples=total, per_unit_capacity_samples=per_unit, max_inflight_samples=inflight)

def require_live_descriptor_wave_capacity(samples, *, resident_waves=1, source_slot_profile=''):
    import ray
    if not ray.is_initialized():
        raise ValueError('large descriptor wave requires the attached external Ray service')
    controller = ray.get_actor('TransferQueueController', namespace='transfer_queue')
    config = ray.get(controller.get_config.remote(), timeout=30)
    return validate_descriptor_wave_capacity(config, samples, resident_waves=resident_waves, source_slot_profile=source_slot_profile)

def validate_variable_batch_cursor(*, run_id, generation, step_index, start, batch_size, total_steps, world_size, previous_receipt=None, generation_handoff=False, epoch_plan=None):
    limit = 8
    if epoch_plan is not None:
        from verl_speco.runtime.recovery import validate_epoch_plan_capacity
        validate_epoch_plan_capacity(epoch_plan)
        limit = epoch_plan.total_steps
        expected_end = min((step_index + 1) * epoch_plan.samples_per_step, epoch_plan.total_samples) if type(step_index) is int else -1
        if world_size != 16 or total_steps != limit or type(step_index) is not int or (start != step_index * epoch_plan.samples_per_step) or (batch_size != expected_end - start):
            raise ValueError('epoch variable batch differs from its frozen complete batch')
    if not isinstance(run_id, str) or not run_id or type(generation) is not int or (generation < 1) or (type(step_index) is not int) or (type(start) is not int) or (start < 0) or (type(batch_size) is not int) or (not 1 <= batch_size <= MAX_NATIVE_BATCH_SAMPLES) or (type(total_steps) is not int) or (not 1 <= total_steps <= limit) or (not 0 <= step_index < total_steps) or (type(world_size) is not int) or (not 1 <= world_size <= 128) or (type(generation_handoff) is not bool) or (generation_handoff and generation < 2):
        raise ValueError('invalid variable batch stream identity')
    if step_index == 0:
        if start != 0 or previous_receipt is not None or generation_handoff:
            raise ValueError('variable first batch must start at the empty stream')
        return start + batch_size
    receipt = previous_receipt
    if not isinstance(receipt, Mapping):
        raise ValueError('variable batch needs a committed predecessor receipt')
    keys = receipt.get('packed_sample_keys')
    prior_start = receipt.get('sample_index')
    from verl_speco.transport.streaming import decode_stream_snapshot
    snapshot = decode_stream_snapshot(receipt.get('stream_snapshot'))
    if receipt.get('run_id') != run_id or type(receipt.get('generation')) is not int or receipt['generation'] != generation - int(generation_handoff) or (type(receipt.get('total_steps')) is not int) or (receipt['total_steps'] != total_steps) or (type(receipt.get('draft_world_size')) is not int) or (receipt['draft_world_size'] != world_size) or (type(prior_start) is not int) or (prior_start < 0) or (not isinstance(keys, list)) or (not keys) or (len(keys) > MAX_NATIVE_BATCH_SAMPLES) or any((not isinstance(k, str) or not k for k in keys)) or (len(set(keys)) != len(keys)) or (prior_start + len(keys) != start) or (not isinstance(snapshot, Mapping)) or (snapshot.get('stream_scope') != run_id) or (type(snapshot.get('committed_cursor')) is not int) or (snapshot['committed_cursor'] != start):
        raise ValueError('variable batch predecessor differs from committed stream cursor')
    steps = snapshot.get('committed_optimizer_steps')
    prior_step = receipt.get('step_id')
    if not isinstance(steps, Mapping) or len(steps) != step_index or (not isinstance(prior_step, str)) or (prior_step not in steps) or any((not isinstance(ids, list) or not ids or any((not isinstance(i, str) or not i for i in ids)) or (len(set(ids)) != len(ids)) for ids in steps.values())) or (len(steps[prior_step]) != len(keys)):
        raise ValueError('variable batch predecessor optimizer count differs')
    if epoch_plan is not None:
        from verl_speco.runtime.recovery import expected_epoch_sample_keys
        previous_generation = generation - int(generation_handoff)
        if receipt.get('epoch_plan_sha256') != epoch_plan.plan_sha256 or sum((len(ids) for ids in steps.values())) != start or prior_start != (step_index - 1) * epoch_plan.samples_per_step or (keys != expected_epoch_sample_keys(epoch_plan, run_id=run_id, generation=previous_generation, step_index=step_index - 1)):
            raise ValueError('epoch predecessor differs from the frozen plan and identities')
    return start + batch_size
