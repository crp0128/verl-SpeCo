# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import hashlib
from verl_speco.runtime.artifacts import fcntl
from contextlib import contextmanager, nullcontext
import json
import math
import threading
from pathlib import Path
from urllib.parse import urlsplit
import torch
ascend_feature_transport_DTYPES = {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32, 'torch.int64': torch.int64, 'torch.uint8': torch.uint8}
ascend_feature_transport_ALIGNMENT = 2 * 1024 ** 2
ascend_feature_transport_FIELD_ALIGNMENT = 256

def ascend_feature_transport_descriptor_sha(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

@contextmanager
def ascend_feature_transport_local_engine_init_lock(path='/tmp/speco-ascend-transfer-engine-init.lock'):
    with open(path, 'a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

class AscendFeatureTransport:

    def __init__(self, *, run_id, generation, endpoint, metadata_server, device, max_bytes=4 * 1024 ** 3, max_slots=2, transfer_chunk_bytes=256 * 1024 ** 2, engine_factory=None, cpu_bulk_receive=False, local_host_mirror=False, receive_copy_stream=False, fixed_source_slot_sizes=None):
        if not isinstance(run_id, str) or not run_id or type(generation) is not int or (generation < 1) or (type(max_bytes) is not int) or (max_bytes < ascend_feature_transport_ALIGNMENT) or (type(max_slots) is not int) or (not 1 <= max_slots <= 16):
            raise ValueError('invalid transport scope or buffer limits')
        if type(transfer_chunk_bytes) is not int or transfer_chunk_bytes < ascend_feature_transport_ALIGNMENT or transfer_chunk_bytes % ascend_feature_transport_ALIGNMENT:
            raise ValueError('transport chunks must be positive multiples of 2MiB')
        if type(cpu_bulk_receive) is not bool:
            raise ValueError('native bulk CPU receive must be boolean')
        if type(local_host_mirror) is not bool:
            raise ValueError('same-device host mirror must be boolean')
        if type(receive_copy_stream) is not bool:
            raise ValueError('receive copy stream must be boolean')
        if receive_copy_stream and (torch.device(device).type != 'npu' or not cpu_bulk_receive or local_host_mirror):
            raise ValueError('private receive stream requires NPU bulk-only receiver without host mirror')
        if fixed_source_slot_sizes is not None:
            if type(fixed_source_slot_sizes) is not tuple or len(fixed_source_slot_sizes) != max_slots or any((type(n) is not int or n < ascend_feature_transport_ALIGNMENT or n % ascend_feature_transport_ALIGNMENT for n in fixed_source_slot_sizes)) or (sum(fixed_source_slot_sizes) > max_bytes):
                raise ValueError('fixed source slot sizes must fit the declared registered byte budget')
            if receive_copy_stream or cpu_bulk_receive or local_host_mirror:
                raise ValueError('fixed source slot sizes are publisher-only')
        self._fixed_source_slot_sizes = fixed_source_slot_sizes
        self._receive_copy_stream = None
        self._receive_copy_lock = threading.RLock()
        self.local_host_mirror = None
        self.cpu_bulk_receive = cpu_bulk_receive
        self.transfer_chunk_bytes = transfer_chunk_bytes
        parsed = urlsplit('//' + endpoint)
        if not parsed.hostname or parsed.port is None or (not 1 <= parsed.port <= 65535):
            raise ValueError('transport endpoint requires host and port')
        meta = urlsplit(metadata_server)
        if meta.scheme != 'http' or not meta.hostname or meta.port is None:
            raise ValueError('transport metadata requires an explicit HTTP endpoint')
        native_factory = engine_factory is None
        self._native_factory = native_factory
        self._opened_sources = set()
        if native_factory:
            if 'libjemalloc' not in Path('/proc/self/maps').read_text():
                raise RuntimeError('native Ascend transport requires jemalloc at process startup')
            import torch_npu

            def engine_factory():
                from mooncake.engine import TransferEngine
                return TransferEngine()
            if torch.device(device).type != 'npu':
                raise ValueError('native Ascend transport requires an NPU device')
        self.device = torch.device(device)
        self.run_id, self.generation = (run_id, generation)
        self.endpoint = endpoint
        self.max_bytes, self.max_slots = (max_bytes, max_slots)
        self._slots = {}
        self._published_slots = set()
        self._source_buffers = {}
        self._retired_source_buffers = []
        self._receive_buffers = []
        self._receive_buffer = None
        self._closed = False
        self._engine = None
        try:
            with ascend_feature_transport_local_engine_init_lock() if native_factory else nullcontext():
                if self.device.type == 'npu':
                    torch.npu.set_device(self.device)
                self._engine = engine_factory()
                self._require_zero('initialize', self._engine.initialize(endpoint, metadata_server, 'ascend', ''))
                if receive_copy_stream:
                    self._receive_copy_stream = torch.npu.Stream(device=self.device)
                if local_host_mirror:
                    if not native_factory:
                        raise ValueError('host mirror requires actual observed NPU device ownership')
                    from verl.workers.rollout.vllm_rollout.utils import get_device_uuid
                    self.local_host_mirror = LocalHostFeatureMirror(run_id=run_id, generation=generation, node_ip=parsed.hostname, device_uuid=get_device_uuid(self.device.index if self.device.index is not None else torch.npu.current_device()), max_bytes=max_bytes, max_slots=max_slots)
        except BaseException:
            self._engine = None
            self._closed = True
            raise

    @property
    def fixed_source_slot_sizes(self):
        return self._fixed_source_slot_sizes

    @staticmethod
    def _require_zero(operation, result):
        if result != 0:
            raise RuntimeError(f'Ascend TransferEngine {operation} failed: {result}')

    def _require_open(self):
        if self._closed:
            raise RuntimeError('Ascend feature transport is closed')

    @contextmanager
    def _receive_scope(self):
        if self._receive_copy_stream is None:
            yield
            return
        with self._receive_copy_lock:
            self._require_open()
            with torch.npu.stream(self._receive_copy_stream):
                try:
                    yield
                finally:
                    self._receive_copy_stream.synchronize()

    def _synchronize(self):
        if self._receive_copy_stream is not None:
            self._receive_copy_stream.synchronize()
        elif self.device.type == 'npu':
            torch.npu.synchronize(self.device)

    def _allocate(self, size):
        allocation = torch.empty(size + ascend_feature_transport_ALIGNMENT - 1, dtype=torch.uint8, device=self.device)
        start = -allocation.data_ptr() % ascend_feature_transport_ALIGNMENT
        return allocation[start:start + size]

    def preallocate_source(self, size, *, slot_id=0):
        """Keep a fixed registered source range across variable-length leases."""
        self._require_open()
        if self._receive_copy_stream is not None:
            raise RuntimeError('private receive stream is consumer-only')
        if type(size) is not int or not 0 < size <= self.max_bytes or type(slot_id) is not int or (not 0 <= slot_id < self.max_slots):
            raise ValueError('invalid source preallocation size or slot')
        if self.fixed_source_slot_sizes is not None and size != self.fixed_source_slot_sizes[slot_id]:
            raise ValueError('fixed source slot size differs from its immutable registration layout')
        previous = self._source_buffers.get(slot_id)
        if previous is not None and previous.numel() >= size:
            return
        if slot_id in self._published_slots or slot_id in self._slots:
            raise RuntimeError('source buffer growth after publication requires a new transport')
        if self._retained_bytes() + size > self.max_bytes:
            raise RuntimeError('transport source-buffer byte budget exceeded')
        buffer = self._allocate(size)
        buffer[:64].zero_()
        self._synchronize()
        self._require_zero('register_memory', self._engine.register_memory(buffer.data_ptr(), size))
        if previous is not None:
            self._retired_source_buffers.append(previous)
        self._source_buffers[slot_id] = buffer

    def pack(self, fields, *, lease_id, slot_id=0):
        return self._pack(fields, lease_id=lease_id, slot_id=slot_id, input_device=self.device)

    def pack_from_cpu(self, fields, *, lease_id, slot_id=0):
        """Synchronously copy validated CPU fields into the final leased range.

        This avoids temporary full NPU fields. Packing still completes before
        publishing its lease header, and readers still verify the full payload.
        """
        return self._pack(fields, lease_id=lease_id, slot_id=slot_id, input_device=torch.device('cpu'))

    @staticmethod
    def _source_layout(fields, input_device):
        layouts = []
        size = ascend_feature_transport_FIELD_ALIGNMENT
        for name in sorted(fields):
            tensor = fields[name]
            if not isinstance(name, str) or not name or (not isinstance(tensor, torch.Tensor)) or (tensor.device != input_device) or (str(tensor.dtype) not in ascend_feature_transport_DTYPES) or (tensor.numel() == 0):
                raise ValueError('transport fields require supported nonempty tensors on the input device')
            offset = math.ceil(size / ascend_feature_transport_FIELD_ALIGNMENT) * ascend_feature_transport_FIELD_ALIGNMENT
            nbytes = tensor.numel() * tensor.element_size()
            layouts.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype), offset=offset, nbytes=nbytes))
            size = offset + nbytes
        return (layouts, size)

    def source_size_for_fields(self, fields):
        if not fields:
            raise ValueError('source fields must be nonempty')
        return self._source_layout(fields, next(iter(fields.values())).device)[1]

    def select_source_slot(self, fields, *, unavailable=()):
        """Choose the smallest free immutable registration that fits all fields."""
        self._require_open()
        unavailable = set(unavailable) | set(self._slots)
        free = [i for i in range(self.max_slots) if i not in unavailable]
        if not free:
            raise RuntimeError('native TQ source slots are still leased')
        if self.fixed_source_slot_sizes is None:
            return free[0]
        needed = self.source_size_for_fields(fields)
        suitable = [i for i in free if self.fixed_source_slot_sizes[i] >= needed]
        if not suitable:
            raise RuntimeError('no free fixed source slot fits the complete payload')
        return min(suitable, key=lambda i: (self.fixed_source_slot_sizes[i], i))

    def _pack(self, fields, *, lease_id, slot_id, input_device):
        self._require_open()
        if self._receive_copy_stream is not None:
            raise RuntimeError('private receive stream is consumer-only')
        if not isinstance(lease_id, str) or not lease_id or type(slot_id) is not int or (not 0 <= slot_id < self.max_slots) or (not fields):
            raise ValueError('pack requires a bounded slot, lease and tensor fields')
        if slot_id in self._slots:
            raise RuntimeError('source slot is still leased')
        layouts, size = self._source_layout(fields, input_device)
        if self.fixed_source_slot_sizes is not None:
            if len(self._source_buffers) != self.max_slots or any((self._source_buffers.get(i) is None or self._source_buffers[i].numel() != n for i, n in enumerate(self.fixed_source_slot_sizes))):
                raise RuntimeError('all fixed source slots must be registered before any publication')
            if size > self.fixed_source_slot_sizes[slot_id]:
                raise RuntimeError('complete source payload exceeds its fixed slot capacity')
        buffer = self._source_buffers.get(slot_id)
        needs_registration = buffer is None or buffer.numel() < size
        if needs_registration:
            if slot_id in self._published_slots:
                raise RuntimeError('source buffer growth after publication requires a new transport')
            if self._retained_bytes() + size > self.max_bytes:
                raise RuntimeError('transport retained-buffer byte budget exceeded')
            buffer = self._allocate(size)
        for layout in layouts:
            target = buffer[layout['offset']:layout['offset'] + layout['nbytes']]
            target.view(ascend_feature_transport_DTYPES[layout['dtype']]).reshape(layout['shape']).copy_(fields[layout['name']])
        self._synchronize()
        descriptor = dict(schema_version=1, run_id=self.run_id, generation=self.generation, segment=self.endpoint, address=buffer.data_ptr(), nbytes=size, lease_id=lease_id, slot_id=slot_id, fields=layouts)
        descriptor['descriptor_sha256'] = ascend_feature_transport_descriptor_sha(descriptor)
        header = torch.tensor(list(descriptor['descriptor_sha256'].encode()), dtype=torch.uint8, device=self.device)
        buffer[:64].copy_(header)
        self._synchronize()
        if needs_registration:
            self._require_zero('register_memory', self._engine.register_memory(buffer.data_ptr(), size))
            previous = self._source_buffers.get(slot_id)
            if previous is not None:
                self._retired_source_buffers.append(previous)
            self._source_buffers[slot_id] = buffer
        if self.local_host_mirror is not None:
            payload = buffer[:size].detach().to('cpu', copy=True).numpy().tobytes()
            self.local_host_mirror.publish(descriptor, payload)
        self._slots[slot_id] = (buffer, descriptor)
        self._published_slots.add(slot_id)
        return json.loads(json.dumps(descriptor))

    def _validate(self, descriptor):
        expected = {'schema_version', 'run_id', 'generation', 'segment', 'address', 'nbytes', 'lease_id', 'slot_id', 'fields', 'descriptor_sha256'}
        if not isinstance(descriptor, dict) or set(descriptor) != expected:
            raise ValueError('invalid transport descriptor schema')
        identity = {k: v for k, v in descriptor.items() if k != 'descriptor_sha256'}
        if ascend_feature_transport_descriptor_sha(identity) != descriptor['descriptor_sha256']:
            raise ValueError('transport descriptor checksum mismatch')
        if type(descriptor['schema_version']) is not int or descriptor['schema_version'] != 1 or descriptor['run_id'] != self.run_id or (type(descriptor['generation']) is not int) or (descriptor['generation'] != self.generation):
            raise ValueError('transport descriptor belongs to a foreign run or generation')
        if type(descriptor['address']) is not int or descriptor['address'] <= 0 or descriptor['address'] % ascend_feature_transport_ALIGNMENT or (type(descriptor['nbytes']) is not int) or (not 0 < descriptor['nbytes'] <= self.max_bytes) or (not isinstance(descriptor['lease_id'], str)) or (not descriptor['lease_id']) or (type(descriptor['slot_id']) is not int) or (not 0 <= descriptor['slot_id'] < 16):
            raise ValueError('transport descriptor has invalid bounds or lease')
        parsed = urlsplit('//' + descriptor['segment'])
        if not parsed.hostname or parsed.port is None:
            raise ValueError('invalid source segment')
        layouts = descriptor['fields']
        if not isinstance(layouts, list) or not layouts:
            raise ValueError('transport has no field layouts')
        end = ascend_feature_transport_FIELD_ALIGNMENT
        names = set()
        for field in layouts:
            if not isinstance(field, dict) or set(field) != {'name', 'shape', 'dtype', 'offset', 'nbytes'}:
                raise ValueError('invalid transport field schema')
            if not isinstance(field['name'], str) or not field['name'] or field['name'] in names or (field['dtype'] not in ascend_feature_transport_DTYPES) or (not isinstance(field['shape'], list)) or any((type(n) is not int or n < 1 for n in field['shape'])) or (type(field['offset']) is not int) or (field['offset'] < end) or field['offset'] % ascend_feature_transport_FIELD_ALIGNMENT or (type(field['nbytes']) is not int):
                raise ValueError('invalid transport field geometry')
            expected_bytes = math.prod(field['shape']) * torch.empty((), dtype=ascend_feature_transport_DTYPES[field['dtype']]).element_size()
            if expected_bytes != field['nbytes']:
                raise ValueError('transport field byte count differs from shape')
            end = field['offset'] + field['nbytes']
            if end > descriptor['nbytes']:
                raise ValueError('transport field exceeds registered range')
            names.add(field['name'])
        if end != descriptor['nbytes']:
            raise ValueError('transport descriptor has unaccounted trailing bytes')

    def _retained_bytes(self):
        return sum((b.numel() for b in self._source_buffers.values())) + sum((b.numel() for b in self._retired_source_buffers)) + sum((b.numel() for b in self._receive_buffers))

    def preallocate_receive(self, size):
        with self._receive_scope():
            return self._preallocate_receive(size)

    def _preallocate_receive(self, size):
        """Register the batch upper bound before opening any remote segment.

        Ascend peer connections cache registrations. Growing the destination
        after the first pull failed real variable-length reads (503900).
        """
        self._require_open()
        if type(size) is not int or not 0 < size <= self.max_bytes:
            raise ValueError('invalid receive preallocation size')
        if self._receive_buffer is not None and self._receive_buffer.numel() >= size:
            return
        if self._opened_sources:
            raise RuntimeError('receive buffer growth after connection requires a new transport')
        if self._retained_bytes() + size > self.max_bytes:
            raise RuntimeError('transport receive-buffer byte budget exceeded')
        buffer = self._allocate(size)
        if self._receive_copy_stream is not None:
            self._receive_copy_stream.synchronize()
        self._require_zero('register_memory', self._engine.register_memory(buffer.data_ptr(), size))
        self._receive_buffers.append(buffer)
        self._receive_buffer = buffer

    def _read_receive_buffer(self, descriptor):
        self._require_open()
        self._validate(descriptor)
        if self.local_host_mirror is not None:
            payload = self.local_host_mirror.read_if_same_device(descriptor)
            if payload is not None:
                return torch.frombuffer(payload, dtype=torch.uint8).to(self.device, copy=True)
        size = descriptor['nbytes']
        self.preallocate_receive(size)
        buffer = self._receive_buffer
        for offset in range(0, size, self.transfer_chunk_bytes):
            count = min(self.transfer_chunk_bytes, size - offset)
            first_connect = self._native_factory and descriptor['segment'] not in self._opened_sources
            lock = ascend_feature_transport_local_engine_init_lock('/tmp/speco-ascend-transfer-engine-first-connect.lock') if first_connect else nullcontext()
            with lock:
                self._require_zero('transfer_sync_read', self._engine.transfer_sync_read(descriptor['segment'], buffer.data_ptr() + offset, descriptor['address'] + offset, count))
                self._opened_sources.add(descriptor['segment'])
        self._synchronize()
        if bytes(buffer[:64].cpu().tolist()) != descriptor['descriptor_sha256'].encode():
            raise RuntimeError('source slot lease fingerprint changed before read')
        return buffer

    def read(self, descriptor):
        if self._receive_copy_stream is not None:
            raise RuntimeError('private receive stream requires full CPU bulk read')
        buffer = self._read_receive_buffer(descriptor)
        fields = {f['name']: buffer[f['offset']:f['offset'] + f['nbytes']].view(ascend_feature_transport_DTYPES[f['dtype']]).reshape(f['shape']).clone() for f in descriptor['fields']}
        self._synchronize()
        return fields

    def read_cpu(self, descriptor):
        with self._receive_scope():
            return self._read_cpu(descriptor)

    def _read_cpu(self, descriptor):
        """Return independent CPU fields directly from the verified receive range.

        One reader owns this transport. Every synchronous CPU copy completes
        before its receive buffer may be reused or its source lease released.
        """
        self._require_open()
        self._validate(descriptor)
        if self.local_host_mirror is not None:
            payload = self.local_host_mirror.read_if_same_device(descriptor)
            if payload is not None:
                buffer = torch.frombuffer(payload, dtype=torch.uint8)
                return {field['name']: buffer[field['offset']:field['offset'] + field['nbytes']].view(ascend_feature_transport_DTYPES[field['dtype']]).reshape(field['shape']).clone() for field in descriptor['fields']}
        buffer = self._read_receive_buffer(descriptor)
        if self.cpu_bulk_receive:
            payload_bytes = sum((field['nbytes'] for field in descriptor['fields']))
            if descriptor['nbytes'] - payload_bytes > 4096:
                raise ValueError('bulk native receive layout exceeds its explicit overhead budget')
            cpu = buffer[:descriptor['nbytes']].detach().to('cpu', copy=True)
            self._synchronize()
            return {field['name']: cpu[field['offset']:field['offset'] + field['nbytes']].view(ascend_feature_transport_DTYPES[field['dtype']]).reshape(field['shape']) for field in descriptor['fields']}
        fields = {field['name']: buffer[field['offset']:field['offset'] + field['nbytes']].view(ascend_feature_transport_DTYPES[field['dtype']]).reshape(field['shape']).detach().to('cpu', copy=True).contiguous() for field in descriptor['fields']}
        self._synchronize()
        return fields

    def read_cpu_light(self, descriptor, *, expected_fields):
        if self._receive_copy_stream is not None:
            raise RuntimeError('private receive stream requires full CPU bulk read')
        return self._read_cpu_light(descriptor, expected_fields=expected_fields)

    def _read_cpu_light(self, descriptor, *, expected_fields):
        """Read replay-verified light fields only; omitted hidden is NOT verified.

        The wave coordinator must retain the source until a required hidden
        owner verifies the full payload and every light owner completes staging.
        Receive registration stays at the full descriptor upper bound to avoid
        reconnecting native peers when later samples require hidden fields.
        """
        self._require_open()
        self._validate(descriptor)
        names = {'input_ids', 'position_ids', 'loss_mask'}
        layouts = {f['name']: f for f in descriptor['fields']}
        if set(layouts) != names | {'hidden_states'} or set(expected_fields) != names:
            raise ValueError('selective receive requires the exact capture fields')
        expected = {}
        for name in names:
            value = expected_fields[name]
            field = layouts[name]
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != ascend_feature_transport_DTYPES[field['dtype']] or (list(value.shape) != field['shape']):
                raise ValueError('immutable replay light field geometry differs')
            expected[name] = value.clone()
        self.preallocate_receive(descriptor['nbytes'])
        buffer = self._receive_buffer

        def pull(offset, size):
            first = self._native_factory and descriptor['segment'] not in self._opened_sources
            lock = ascend_feature_transport_local_engine_init_lock('/tmp/speco-ascend-transfer-engine-first-connect.lock') if first else nullcontext()
            with lock:
                for relative in range(0, size, self.transfer_chunk_bytes):
                    count = min(self.transfer_chunk_bytes, size - relative)
                    self._require_zero('transfer_sync_read', self._engine.transfer_sync_read(descriptor['segment'], buffer.data_ptr() + offset + relative, descriptor['address'] + offset + relative, count))
                self._opened_sources.add(descriptor['segment'])
            self._synchronize()

        def header():
            pull(0, 64)
            if bytes(buffer[:64].cpu().tolist()) != descriptor['descriptor_sha256'].encode():
                raise RuntimeError('source slot lease fingerprint changed during light read')
        header()
        fields = {}
        for name in sorted(names):
            field = layouts[name]
            pull(field['offset'], field['nbytes'])
            value = buffer[field['offset']:field['offset'] + field['nbytes']].view(ascend_feature_transport_DTYPES[field['dtype']]).reshape(field['shape']).detach().to('cpu', copy=True).contiguous()
            if not torch.equal(value, expected[name]):
                raise RuntimeError('source light fields differ from immutable replay')
            fields[name] = value
        header()
        return fields

    def release(self, *, slot_id, lease_id):
        self._require_open()
        if slot_id not in self._slots:
            raise ValueError('source slot is not leased')
        buffer, descriptor = self._slots[slot_id]
        if descriptor['lease_id'] != lease_id:
            raise ValueError('source lease identity differs')
        if self.local_host_mirror is not None:
            self.local_host_mirror.release(descriptor)
        buffer[:64].zero_()
        self._synchronize()
        del self._slots[slot_id]

    def close(self):
        if self._receive_copy_stream is None:
            return self._close()
        with self._receive_copy_lock:
            if self._closed:
                return
            self._receive_copy_stream.synchronize()
            return self._close()

    def _close(self):
        if self._closed:
            return
        failures = []
        for slot_id, (_, descriptor) in list(self._slots.items()):
            try:
                self.release(slot_id=slot_id, lease_id=descriptor['lease_id'])
            except Exception as error:
                failures.append(str(error))
        for buffer in [*self._source_buffers.values(), *self._retired_source_buffers, *self._receive_buffers]:
            try:
                self._require_zero('unregister_memory', self._engine.unregister_memory(buffer.data_ptr()))
            except Exception as error:
                failures.append(str(error))
        self._source_buffers.clear()
        self._retired_source_buffers.clear()
        self._receive_buffers.clear()
        self._receive_buffer = None
        if self.local_host_mirror is not None:
            try:
                self.local_host_mirror.close()
            except Exception as error:
                failures.append(str(error))
        self._closed = True
        self._engine = None
        if failures:
            raise RuntimeError('Ascend feature transport close failed: ' + '; '.join(failures))
import time
from verl_speco.transport.control import TQKeyedStore
from verl_speco.transport.streaming import parse_v3_ready_tag, make_v3_sample_key
from verl_speco.producer.feature_payload import payload_checksum
native_descriptor_store_DESCRIPTOR_FIELD = 'speco_native_descriptor_v1'
native_descriptor_store_MAX_DESCRIPTOR_BYTES = 65536

def decode_native_record(record, *, key):
    if set(record.keys()) != {native_descriptor_store_DESCRIPTOR_FIELD}:
        raise ValueError('native TQ key contains unexpected payload fields')
    encoded = record[native_descriptor_store_DESCRIPTOR_FIELD]
    if not isinstance(encoded, torch.Tensor) or encoded.device.type != 'cpu' or encoded.dtype != torch.uint8 or (encoded.ndim != 1) or (not 0 < encoded.numel() <= native_descriptor_store_MAX_DESCRIPTOR_BYTES):
        raise ValueError('native TQ descriptor must be bounded CPU control bytes')
    descriptor = json.loads(bytes(encoded.tolist()))
    if not isinstance(descriptor, dict):
        raise ValueError('native TQ descriptor must be a control object')
    if descriptor.get('lease_id') != key:
        raise ValueError('native TQ descriptor lease differs from sample key')
    return descriptor

class NativeDescriptorStore(TQKeyedStore):

    def __init__(self, *, tq, transport, profile=False, cpu_direct_pack=False, receive_capacity_bytes=None):
        if type(profile) is not bool:
            raise ValueError('native receive profile must be boolean')
        if type(cpu_direct_pack) is not bool:
            raise ValueError('native CPU direct pack must be boolean')
        if receive_capacity_bytes is not None and (type(receive_capacity_bytes) is not int or not 0 < receive_capacity_bytes <= transport.max_bytes):
            raise ValueError('native receive capacity must fit the transport byte budget')
        self.receive_capacity_bytes = receive_capacity_bytes
        self._cpu_direct_pack = cpu_direct_pack
        self._profile = profile
        super().__init__(tq)
        self.transport = transport
        self._sources = {}

    def _metadata(self, key, tag):
        metadata, checksum = parse_v3_ready_tag(tag)
        if make_v3_sample_key(metadata) != key or metadata.run_id != self.transport.run_id or metadata.generation != self.transport.generation:
            raise ValueError('native TQ key belongs to a foreign run or generation')
        return (metadata, checksum)

    @staticmethod
    def _validate_fields(fields, metadata, checksum):
        if not fields or any((not isinstance(v, torch.Tensor) for v in fields.values())) or sum((v.numel() * v.element_size() for v in fields.values())) != metadata.payload_nbytes:
            raise ValueError('native TQ payload byte count differs from metadata')
        cpu = {k: v.detach().cpu().contiguous() for k, v in fields.items()}
        if payload_checksum(cpu) != checksum:
            raise ValueError('native TQ payload checksum differs from ready tag')

    def put(self, key, partition, fields, tag):
        metadata, checksum = self._metadata(key, tag)
        if self.get_tag(key, partition) is not None or (partition, key) in self._sources:
            raise ValueError('native TQ source key is already published')
        self._validate_fields(fields, metadata, checksum)
        slots = {d['slot_id'] for d in self._sources.values()}
        slot = self.transport.select_source_slot(fields, unavailable=slots)
        if self._cpu_direct_pack:
            started = time.perf_counter()
            descriptor = self.transport.pack_from_cpu(fields, lease_id=key, slot_id=slot)
            print('native_source_cpu_pack_timing ' + json.dumps(dict(payload_bytes=metadata.payload_nbytes, cpu_direct_pack=True, pack_seconds=time.perf_counter() - started)), flush=True)
        else:
            npu = {k: v.to(self.transport.device) for k, v in fields.items()}
            descriptor = self.transport.pack(npu, lease_id=key, slot_id=slot)
        self._sources[partition, key] = descriptor
        try:
            encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()
            if len(encoded) > native_descriptor_store_MAX_DESCRIPTOR_BYTES:
                raise ValueError('native TQ descriptor exceeds control limit')
            self._tq.kv_put(key=key, partition_id=partition, fields={native_descriptor_store_DESCRIPTOR_FIELD: torch.tensor(list(encoded), dtype=torch.uint8)}, tag=dict(tag))
        except BaseException:
            if self.get_tag(key, partition) is None:
                self.transport.release(slot_id=descriptor['slot_id'], lease_id=key)
                del self._sources[partition, key]
            raise

    def publication_descriptor(self, key, partition):
        """Return bounded producer-owned control bytes, even after remote clear.

        Publication receipts must not reread a descriptor a consumer can already
        materialize and remove. Ownership lasts until the producer reap point.
        """
        descriptor = self._sources.get((partition, key))
        if descriptor is None or descriptor.get('lease_id') != key:
            raise ValueError('native publication descriptor is not owned')
        self.transport._validate(descriptor)
        encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()
        if not 0 < len(encoded) <= native_descriptor_store_MAX_DESCRIPTOR_BYTES:
            raise ValueError('native publication descriptor exceeds control limit')
        return json.loads(encoded)

    def _read_descriptor(self, key, partition):
        tag = self.get_tag(key, partition)
        if tag is None:
            raise ValueError('native TQ descriptor is not ready')
        return self._read_descriptor_with_tag(key, partition, tag)

    def _checked_descriptor(self, key, record, metadata):
        descriptor = decode_native_record(record, key=key)
        self.transport._validate(descriptor)
        if sum((field['nbytes'] for field in descriptor['fields'])) != metadata.payload_nbytes:
            raise ValueError('native descriptor byte count differs from ready metadata')
        return descriptor

    def _read_descriptor_with_tag(self, key, partition, tag):
        metadata, checksum = self._metadata(key, tag)
        record = self._tq.kv_batch_get(keys=[key], partition_id=partition)[0]
        return (self._checked_descriptor(key, record, metadata), metadata, checksum)

    def prepare_receive(self, keys, partition):
        """Validate one control batch before registering its largest payload."""
        keys = tuple(keys)
        if not keys:
            raise ValueError('native receive preparation requires sample keys')
        tags = self._tq.kv_list(partition_id=partition).get(partition, {})
        metadata = []
        for key in keys:
            tag = tags.get(key)
            if tag is None:
                raise ValueError('native TQ descriptor is not ready')
            metadata.append(self._metadata(key, tag)[0])
        batch = self._tq.kv_batch_get(keys=list(keys), partition_id=partition)
        if tuple(batch.batch_size) != (len(keys),):
            raise ValueError('native descriptor batch sample count differs')
        descriptors = [self._checked_descriptor(key, batch[i], metadata[i]) for i, key in enumerate(keys)]
        size = max((d['nbytes'] for d in descriptors))
        self.prepare_receive_bound(size)
        return size

    def prepare_receive_bound(self, size):
        """Apply a validated batch bound to this independent registration."""
        if type(size) is not int or not 0 < size <= self.transport.max_bytes:
            raise ValueError('native receive bound exceeds independent transport capacity')
        if self.receive_capacity_bytes is not None:
            if size > self.receive_capacity_bytes:
                raise ValueError('native descriptor exceeds configured receive capacity')
            size = self.receive_capacity_bytes
        self.transport.preallocate_receive(size)

    def get(self, key, partition):
        descriptor, metadata, checksum = self._read_descriptor(key, partition)
        fields = self.transport.read(descriptor)
        self._validate_fields(fields, metadata, checksum)
        return fields

    def _copy_cpu_for_verification(self, key, partition):
        started = time.perf_counter() if self._profile else None
        descriptor, metadata, checksum = self._read_descriptor(key, partition)
        descriptor_finished = time.perf_counter() if self._profile else None
        fields = self.transport.read_cpu(descriptor)
        copy_finished = time.perf_counter() if self._profile else None
        return (key, fields, metadata, checksum, started, descriptor_finished, copy_finished)

    def _verify_cpu_packet(self, packet, *, pipelined=False):
        key, fields, metadata, checksum, started, descriptor_finished, copy_finished = packet
        verify_started = time.perf_counter() if self._profile else None
        self._validate_fields(fields, metadata, checksum)
        if self._profile:
            finished = time.perf_counter()
            descriptor_seconds = descriptor_finished - started
            copy_seconds = copy_finished - descriptor_finished
            checksum_seconds = finished - verify_started if pipelined else finished - copy_finished
            timing = dict(schema_version=2 if pipelined else 1, key_sha256=hashlib.sha256(key.encode()).hexdigest(), payload_bytes=metadata.payload_nbytes, descriptor_seconds=descriptor_seconds, transfer_cpu_copy_seconds=copy_seconds, checksum_seconds=checksum_seconds, total_seconds=descriptor_seconds + copy_seconds + checksum_seconds)
            if pipelined:
                timing.update(verification_queue_wait_seconds=verify_started - copy_finished, wallclock_seconds=finished - started)
            print('native_verified_cpu_receive_timing ' + json.dumps(timing), flush=True)
        return (fields, checksum)

    def get_verified_cpu_bound(self, key, partition, *, expected_metadata, expected_payload_checksum):
        """Bind full capture metadata before verifying its complete payload."""
        packet = self._copy_cpu_for_verification(key, partition)
        if packet[0] != key or packet[2] != expected_metadata or packet[3] != expected_payload_checksum:
            raise ValueError('bound full reader capture differs from retained wave')
        return self._verify_cpu_packet(packet)

    def get_verified_light(self, key, partition, *, expected_metadata, expected_payload_checksum, expected_fields, expected_light_checksum, hidden_size):
        """Return ONLY replay-verified light SHA; never claim hidden verification."""
        from verl_speco.trainer.owned_batches import materialize_light_columns
        descriptor, metadata, source_checksum = self._read_descriptor(key, partition)
        if metadata != expected_metadata or source_checksum != expected_payload_checksum:
            raise ValueError('selective reader capture differs from retained wave')
        replay = {name: value.clone() for name, value in expected_fields.items()}
        materialize_light_columns(replay, metadata, hidden_size=hidden_size)
        if payload_checksum(replay) != expected_light_checksum:
            raise ValueError('selective reader replay checksum differs from routing plan')
        layout = next((f for f in descriptor['fields'] if f['name'] == 'hidden_states'), None)
        if layout is None or layout['dtype'] != 'torch.bfloat16' or layout['shape'] != [metadata.token_count, hidden_size * (len(metadata.target_layer_ids) + 1)]:
            raise ValueError('selective reader hidden geometry differs from capture')
        fields = self.transport.read_cpu_light(descriptor, expected_fields=replay)
        materialize_light_columns(fields, metadata, hidden_size=hidden_size)
        digest = payload_checksum(fields)
        if digest != expected_light_checksum:
            raise ValueError('selective reader light checksum differs from immutable replay')
        return (fields, digest)

    def get_verified_cpu(self, key, partition):
        """Return exact independent CPU tensors only after full SHA verification."""
        return self._verify_cpu_packet(self._copy_cpu_for_verification(key, partition))

    def clear(self, key, partition):
        tag = self.get_tag(key, partition)
        if tag is not None:
            self._metadata(key, tag)
        self._tq.kv_clear(keys=key, partition_id=partition)
        self.reap_cleared_sources()

    def reap_cleared_sources(self):
        """Producer safe point after rank-wide materialize/clear acknowledgement.

        The caller must use the TQ lease protocol: clearing a key before all
        consumers finish is invalid. Remote consumers only clear descriptors;
        the producer remains the owner of native registrations and NPU memory.
        """
        from collections.abc import Mapping
        owned = list(self._sources.items())
        if not owned:
            return []
        snapshots = {}
        for (partition, key), descriptor in owned:
            if descriptor.get('lease_id') != key:
                raise ValueError('native source descriptor lease differs from owned key')
            if partition not in snapshots:
                response = self._tq.kv_list(partition_id=partition)
                if not isinstance(response, Mapping):
                    raise ValueError('native source reaping requires a partition mapping')
                tags = response.get(partition, {})
                if not isinstance(tags, Mapping):
                    raise ValueError('native source reaping requires a key mapping')
                snapshots[partition] = tags
        cleared = []
        for (partition, key), descriptor in owned:
            tags = snapshots[partition]
            if key in tags:
                if not isinstance(tags[key], Mapping):
                    raise ValueError('native source ready tag must be a mapping')
                self._metadata(key, tags[key])
            else:
                cleared.append((partition, key, descriptor))
        released = []
        for partition, key, descriptor in cleared:
            self.transport.release(slot_id=descriptor['slot_id'], lease_id=key)
            del self._sources[partition, key]
            released.append(key)
        return released
inline_native_descriptor_store_INLINE = 'speco_native_descriptor_inline_v1'
inline_native_descriptor_store_BOUND = 65536

class InlineControlTQ:

    def __init__(self, tq):
        self._tq = tq

    def __getattr__(self, name):
        return getattr(self._tq, name)

    def kv_put(self, *, key, partition_id, fields, tag):
        if inline_native_descriptor_store_INLINE in tag:
            raise ValueError('inline descriptor must be producer-owned')
        descriptor = decode_native_record(fields, key=key)
        inline = dict(tag, **{inline_native_descriptor_store_INLINE: descriptor})
        try:
            encoded = json.dumps(inline, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        except (TypeError, ValueError) as error:
            raise ValueError('inline native ready control must be JSON') from error
        if not 0 < len(encoded) <= inline_native_descriptor_store_BOUND:
            raise ValueError('inline ready control exceeds 64KiB')
        return self._tq.kv_put(key=key, partition_id=partition_id, fields=fields, tag=json.loads(encoded))

class InlineNativeDescriptorStore(NativeDescriptorStore):

    def __init__(self, *, tq, transport, profile=False, cpu_direct_pack=False):
        super().__init__(tq=InlineControlTQ(tq), transport=transport, profile=profile, cpu_direct_pack=cpu_direct_pack)

    def _read_descriptor(self, key, partition):
        client = self._tq.get_client()
        if not callable(getattr(client, 'kv_retrieve_meta', None)):
            raise ValueError('inline descriptor read requires public fresh metadata API')
        meta = client.kv_retrieve_meta(keys=[key], partition_id=partition, create=False)
        if meta.size != 1 or not meta.is_ready or list(meta.partition_ids) != [partition] or (len(meta.global_indexes) != 1) or (type(meta.global_indexes[0]) is not int) or (meta.global_indexes[0] < 0) or (len(meta.custom_meta) != 1) or (not isinstance(meta.custom_meta[0], dict)) or (list(meta.field_names) != [native_descriptor_store_DESCRIPTOR_FIELD]):
            raise ValueError('inline descriptor metadata not ready or foreign')
        tag = meta.custom_meta[0]
        try:
            encoded = json.dumps(tag, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        except (TypeError, ValueError) as error:
            raise ValueError('inline native ready control must be JSON') from error
        if not 0 < len(encoded) <= inline_native_descriptor_store_BOUND:
            raise ValueError('inline ready control exceeds 64KiB')
        tag = json.loads(encoded)
        metadata, checksum = self._metadata(key, tag)
        descriptor = tag.get(inline_native_descriptor_store_INLINE)
        if not isinstance(descriptor, dict) or descriptor.get('lease_id') != key:
            raise ValueError('inline descriptor missing or foreign lease')
        self.transport._validate(descriptor)
        if sum((f['nbytes'] for f in descriptor['fields'])) != metadata.payload_nbytes:
            raise ValueError('inline descriptor byte count differs from ready metadata')
        return (descriptor, metadata, checksum)

def native_descriptor_store_type(environment):
    """Resolve a static opt-in; reject invalid flags before native allocation."""
    flag = environment.get('SPECO_NATIVE_INLINE_DESCRIPTOR', '')
    if flag not in ('', '0', '1'):
        raise ValueError('native inline descriptor flag must be empty, 0 or 1')
    if flag == '1' and environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1':
        raise ValueError('inline descriptor requires native feature transport')
    return InlineNativeDescriptorStore if flag == '1' else NativeDescriptorStore
import array
import errno
import os
import socket
import stat
import struct
local_host_feature_mirror_HEADER_BYTES = 4096
local_host_feature_mirror_MAGIC = b'SPECO-MEMFD-FRAME-1\n'
local_host_feature_mirror_CONTROL_BYTES = 8192
local_host_feature_mirror_SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL

def local_host_feature_mirror_rights(ancillary):
    result = []
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array('i')
            values.frombytes(data[:len(data) - len(data) % values.itemsize])
            result.extend(values)
    return result

class LocalHostFeatureMirror:

    def __init__(self, *, run_id, generation, node_ip, device_uuid, max_bytes, max_slots, root=Path('/tmp/speco-local-feature-frames')):
        if type(run_id) is not str or not run_id or type(generation) is not int or (generation < 1) or (type(node_ip) is not str) or (not node_ip) or (type(device_uuid) is not str) or (not device_uuid) or (type(max_bytes) is not int) or (max_bytes < 1) or (type(max_slots) is not int) or (max_slots < 1):
            raise ValueError('host feature mirror needs explicit physical owner and bounded capacity')
        root = Path(root)
        if not root.is_absolute() or root.parent != Path('/tmp'):
            raise ValueError('host mirror namespace must be one explicit local /tmp name')
        self.scope = dict(run_id=run_id, generation=generation)
        self.node_ip, self.device_uuid = (node_ip, device_uuid)
        self.max_bytes, self.max_slots = (max_bytes, max_slots)
        self.root = str(root)
        self.active = {}
        self._brokers = {}
        self._lock = threading.Lock()
        self._closed = False

    def _name(self, descriptor):
        if any((type(descriptor.get(k)) is not type(v) or descriptor[k] != v for k, v in self.scope.items())):
            raise ValueError('host frame has a foreign run or generation')
        sha = descriptor['descriptor_sha256']
        if type(sha) is not str or len(sha) != 64 or any((c not in '0123456789abcdef' for c in sha)) or (type(descriptor['nbytes']) is not int) or (not 64 <= descriptor['nbytes'] <= self.max_bytes):
            raise ValueError('host frame descriptor bounds or checksum are invalid')
        return sha + '.frame'

    def _address(self, segment):
        from urllib.parse import urlsplit
        parsed = urlsplit('//' + segment)
        if parsed.hostname != self.node_ip or parsed.port is None:
            raise ValueError('host broker segment differs from its exact local node')
        token = hashlib.sha256(json.dumps(dict(self.scope, segment=segment, root=self.root), sort_keys=True).encode()).hexdigest()
        return '\x00speco-feature-' + token

    def _ensure_broker(self, segment):
        if segment in self._brokers:
            return
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            listener.bind(self._address(segment))
            listener.listen(64)
            listener.settimeout(0.2)
            stop = threading.Event()
            thread = threading.Thread(target=self._serve, args=(segment, listener, stop), daemon=True)
            self._brokers[segment] = dict(listener=listener, stop=stop, thread=thread)
            thread.start()
        except BaseException:
            listener.close()
            self._brokers.pop(segment, None)
            raise

    def _serve(self, segment, listener, stop):
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if stop.is_set():
                    return
                continue
            with connection:
                connection.settimeout(1)
                fd = None
                received = []
                try:
                    _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.geteuid():
                        raise ValueError('foreign broker peer')
                    raw, anc, flags, _ = connection.recvmsg(local_host_feature_mirror_CONTROL_BYTES, socket.CMSG_SPACE(4))
                    received = local_host_feature_mirror_rights(anc)
                    if received or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                        raise ValueError('broker request exceeds control or carries foreign handles')
                    request = json.loads(raw)
                    expected_keys = set(self.scope) | {'segment', 'descriptor_sha256', 'nbytes', 'device_uuid'}
                    if set(request) != expected_keys or request['segment'] != segment:
                        raise ValueError('broker request identity differs')
                    name = self._name(request)
                    if type(request['device_uuid']) is not str or not request['device_uuid']:
                        raise ValueError('broker reader physical owner is missing')
                    with self._lock:
                        if request['device_uuid'] != self.device_uuid:
                            reply = dict(status='native', device_uuid=self.device_uuid)
                        else:
                            owner = self.active.get(name)
                            if owner is None or owner['segment'] != segment:
                                reply = dict(status='expired')
                            else:
                                if owner['size'] != request['nbytes']:
                                    raise ValueError('broker payload bounds differ')
                                fd = os.dup(owner['fd'])
                                reply = dict(status='frame', header=owner['header'])
                    data = json.dumps(reply, sort_keys=True, separators=(',', ':')).encode()
                    assert len(data) <= local_host_feature_mirror_CONTROL_BYTES
                    handles = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))]
                    if connection.sendmsg([data], handles) != len(data):
                        raise RuntimeError('broker control reply was incomplete')
                except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                    pass
                finally:
                    if fd is not None:
                        os.close(fd)
                    for handle in received:
                        os.close(handle)

    def publish(self, descriptor, payload):
        name = self._name(descriptor)
        payload = bytes(payload)
        header = dict(self.scope, node_ip=self.node_ip, device_uuid=self.device_uuid, descriptor_sha256=descriptor['descriptor_sha256'], nbytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest())
        encoded = local_host_feature_mirror_MAGIC + json.dumps(header, sort_keys=True, separators=(',', ':')).encode() + b'\n'
        if len(encoded) > local_host_feature_mirror_HEADER_BYTES:
            raise ValueError('host feature header exceeds fixed bound')
        with self._lock:
            if self._closed:
                raise RuntimeError('host feature mirror is closed')
            if len(payload) != descriptor['nbytes'] or payload[:64] != descriptor['descriptor_sha256'].encode() or name in self.active or (len(self.active) >= self.max_slots) or (sum((row['size'] for row in self.active.values())) + len(payload) > self.max_bytes):
                raise ValueError('host frame payload, lease identity or owned capacity differs')
            self._ensure_broker(descriptor['segment'])
            fd = os.memfd_create('speco-feature-frame', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            try:
                with os.fdopen(fd, 'wb', closefd=False) as stream:
                    stream.write(encoded.ljust(local_host_feature_mirror_HEADER_BYTES, b'\x00'))
                    stream.write(payload)
                    stream.flush()
                if os.fstat(fd).st_size != local_host_feature_mirror_HEADER_BYTES + len(payload):
                    raise RuntimeError('host frame write was incomplete')
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, local_host_feature_mirror_SEALS)
                self.active[name] = dict(fd=fd, size=len(payload), header=header, segment=descriptor['segment'])
            except BaseException:
                os.close(fd)
                raise

    def read_if_same_device(self, descriptor):
        if self._closed:
            raise RuntimeError('host feature mirror is closed')
        name = self._name(descriptor)
        from urllib.parse import urlsplit
        if urlsplit('//' + descriptor['segment']).hostname != self.node_ip:
            return None
        request = dict(self.scope, segment=descriptor['segment'], descriptor_sha256=descriptor['descriptor_sha256'], nbytes=descriptor['nbytes'], device_uuid=self.device_uuid)
        data = json.dumps(request, sort_keys=True, separators=(',', ':')).encode()
        if len(data) > local_host_feature_mirror_CONTROL_BYTES:
            raise ValueError('host request exceeds control bound')
        received = []
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(5)
            try:
                connection.connect(self._address(descriptor['segment']))
            except OSError as error:
                if error.errno in (errno.ENOENT, errno.ECONNREFUSED):
                    return None
                raise
            _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.geteuid():
                raise RuntimeError('host broker has a foreign current owner')
            if connection.send(data) != len(data):
                raise RuntimeError('host request was incomplete')
            raw, anc, flags, _ = connection.recvmsg(local_host_feature_mirror_CONTROL_BYTES, socket.CMSG_SPACE(4), socket.MSG_CMSG_CLOEXEC)
            received = local_host_feature_mirror_rights(anc)
        try:
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise RuntimeError('host reply was truncated')
            reply = json.loads(raw)
            if reply.get('status') == 'native':
                if received or set(reply) != {'status', 'device_uuid'} or type(reply['device_uuid']) is not str or (reply['device_uuid'] == self.device_uuid):
                    raise RuntimeError('host physical owner proof differs')
                return None
            if reply.get('status') == 'expired':
                raise RuntimeError('host frame lease fingerprint is no longer owned')
            if set(reply) != {'status', 'header'} or reply['status'] != 'frame' or len(received) != 1:
                raise RuntimeError('host broker lacks one exact shared frame')
            fd = received[0]
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 0 or info.st_size != local_host_feature_mirror_HEADER_BYTES + descriptor['nbytes'] or (fcntl.fcntl(fd, fcntl.F_GET_SEALS) & local_host_feature_mirror_SEALS != local_host_feature_mirror_SEALS):
                raise RuntimeError('host frame is not an exact sealed anonymous payload')
            expected = dict(self.scope, node_ip=self.node_ip, device_uuid=self.device_uuid, descriptor_sha256=descriptor['descriptor_sha256'], nbytes=descriptor['nbytes'])
            header = reply['header']
            if set(header) != set(expected) | {'payload_sha256'} or any((type(header[k]) is not type(v) or header[k] != v for k, v in expected.items())):
                raise RuntimeError('host frame belongs to a foreign descriptor or physical source')
            prefix = os.pread(fd, local_host_feature_mirror_HEADER_BYTES, 0)
            if not prefix.startswith(local_host_feature_mirror_MAGIC) or json.loads(prefix[len(local_host_feature_mirror_MAGIC):].split(b'\n', 1)[0]) != header:
                raise RuntimeError('host frame header differs from current broker proof')
            payload = bytearray()
            while len(payload) < descriptor['nbytes']:
                block = os.pread(fd, min(8 * 1024 ** 2, descriptor['nbytes'] - len(payload)), local_host_feature_mirror_HEADER_BYTES + len(payload))
                if not block:
                    raise RuntimeError('host frame payload was incomplete')
                payload.extend(block)
            if payload[:64] != descriptor['descriptor_sha256'].encode() or hashlib.sha256(payload).hexdigest() != header['payload_sha256']:
                raise RuntimeError('host frame complete payload or lease fingerprint changed')
            return payload
        finally:
            for fd in received:
                os.close(fd)

    def release(self, descriptor):
        name = self._name(descriptor)
        with self._lock:
            owner = self.active.get(name)
            if owner is None or owner['segment'] != descriptor['segment'] or owner['size'] != descriptor['nbytes']:
                raise ValueError('host frame has no exact active source lease')
            os.close(owner['fd'])
            del self.active[name]

    def close(self):
        with self._lock:
            if self.active:
                raise RuntimeError('host frame close requires original source leases released')
            if self._closed:
                return
            self._closed = True
            brokers = list(self._brokers.values())
            for broker in brokers:
                broker['stop'].set()
                broker['listener'].close()
        for broker in brokers:
            broker['thread'].join(timeout=2)
            if broker['thread'].is_alive():
                raise RuntimeError('owned host broker did not finish')
        self._brokers.clear()

def native_receive_policy(environment):
    workers = environment.get('SPECO_NATIVE_RECEIVE_WORKERS', '')
    bulk = environment.get('SPECO_NATIVE_CPU_BULK_RECEIVE', '')
    if workers not in ('', '1', '2', '4', '8') or bulk not in ('', '0', '1'):
        raise ValueError('invalid independent native receiver policy')
    count = int(workers or '1')
    if count != 1 and (bulk != '1' or any((environment.get(k) != '1' for k in ('SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_DRAFT_PREFILL_WAVES', 'SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE', 'SPECO_PREFILL_BATCH_RELEASE')))):
        raise ValueError('parallel native receivers require bulk CPU copies and verified node waves')
    if count > int(environment.get('SPECO_PREFILL_NATIVE_VERIFICATION_WORKERS') or '2'):
        raise ValueError('native readers exceed verification slots')
    if bulk == '1' and environment.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1':
        raise ValueError('bulk CPU receive requires native transport')
    if count == 8 and (not wide_native_receive_profile(environment)):
        raise ValueError('eight native readers require the frozen DP16 16GiB profile')
    return (count, bulk == '1')

def native_receiver_endpoint(primary, index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('invalid native receiver lane')
    parsed = urlsplit('//' + primary)
    if not parsed.hostname or not parsed.port or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError('native receiver endpoint must have host and port only')
    port = parsed.port if index == 0 else parsed.port + 15 + index
    if not 1 <= port <= 65535:
        raise ValueError('independent native receiver endpoint exceeds port range')
    host = '[' + parsed.hostname + ']' if ':' in parsed.hostname else parsed.hostname
    return f'{host}:{port}'

class ParallelNativeReader:

    def __init__(self, *, count, reader_factory):
        if type(count) is not int or count not in (2, 4, 8) or (not callable(reader_factory)):
            raise ValueError('invalid independent native reader factory')
        self.readers = []
        self._closed = False
        try:
            for index in range(count):
                self.readers.append(reader_factory(index))
            transports = [reader.transport for reader in self.readers]
            if len({id(t) for t in transports}) != count or len({t.endpoint for t in transports}) != count or len({str(t.device) for t in transports}) != 1:
                raise ValueError('parallel native readers require independent engines on the owner device')
            self.device = transports[0].device
            if not all((t.cpu_bulk_receive is True for t in transports)):
                raise ValueError('parallel native readers require owned bulk CPU packets')
            self.cpu_bulk_receive = True
        except BaseException:
            self.close()
            raise

    @property
    def transport(self):
        return self

    def prepare_receive(self, keys, partition):
        if self._closed:
            raise RuntimeError('independent native readers are closed')

        def guard(reader):
            tq = reader._tq
            if isinstance(tq, InlineControlTQ):
                tq = tq._tq
            return (type(reader), type(reader.transport), id(tq), reader.transport.run_id, reader.transport.generation, reader.transport.max_bytes)
        actual = all((type(r) in (NativeDescriptorStore, InlineNativeDescriptorStore) for r in self.readers))
        if actual and all((guard(r) == guard(self.readers[0]) for r in self.readers[1:])):
            size = self.readers[0].prepare_receive(keys, partition)
            for reader in self.readers[1:]:
                reader.prepare_receive_bound(size)
        else:
            for reader in self.readers:
                reader.prepare_receive(keys, partition)

    def copy_lanes(self, partition):
        if self._closed:
            raise RuntimeError('independent native readers are closed')
        return tuple((lambda key, reader=reader: reader._copy_cpu_for_verification(key, partition) for reader in self.readers))

    def _copy_cpu_for_verification(self, key, partition):
        raise RuntimeError('parallel native readers require explicit lane admission')

    def _verify_cpu_packet(self, packet, *, pipelined=False):
        return self.readers[0]._verify_cpu_packet(packet, pipelined=pipelined)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for reader in self.readers:
            try:
                reader.transport.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError('owned independent native receiver cleanup failed') from errors[0]

def wide_native_receive_profile(environment):
    """Use existing immutable command geometry; preserve legacy static identity."""
    exact = dict(SPECO_DRAFT_WORLD_SIZE='16', SPECO_PREFILL_NATIVE_VERIFICATION_WORKERS='8', SPECO_PREFILL_NATIVE_PAYLOAD_PREFETCH_BYTES=str(16 * 1024 ** 3), SPECO_NATIVE_RECEIVE_WORKERS='8', SPECO_NATIVE_CPU_BULK_RECEIVE='1', SPECO_NATIVE_FEATURE_TRANSPORT='1', SPECO_DRAFT_NODE_OWNED_SWA='1', SPECO_DRAFT_PREFILL_WAVES='1', SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE='1', SPECO_PREFILL_BATCH_RELEASE='1', SPECO_PREFILL_EXACT_WAVE_ACQUIRE='1', SPECO_PREFILL_WAVE_ACK_TRANSPORT='ray')
    geometry = (environment.get('SPECO_PREFILL_WAVE_SIZE'), environment.get('SPECO_PREFILL_MAX_WAVE_BYTES'))
    return geometry in (('48', str(96 * 1024 ** 3)), ('96', str(48 * 1024 ** 3))) and environment.get('SPECO_PREFILL_OWNED_COPY_WORKERS') in ('8', '16') and all((environment.get(k) == v for k, v in exact.items()))

def native_receive_copy_stream_policy(environment):
    value = environment.get('SPECO_NATIVE_RECEIVE_COPY_STREAM', '')
    if value not in ('', '0', '1'):
        raise ValueError('invalid native receive copy stream policy')
    enabled = value == '1'
    if enabled and (environment.get('SPECO_NATIVE_CPU_BULK_RECEIVE') != '1' or any((environment.get(k) != '1' for k in ('SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_DRAFT_PREFILL_WAVES', 'SPECO_PREFILL_NATIVE_VERIFICATION_PIPELINE')))):
        raise ValueError('private receive stream requires verified full bulk node waves')
    if enabled:
        native_receive_policy(environment)
    return enabled
