# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import torch

class BoundedOwnedBuilderPipeline:

    def __init__(self, builders, *, max_pending_bytes, max_pending_samples=2):
        if not isinstance(builders, list) or len(builders) != 16 or len({id(b) for b in builders}) != 16 or any((not callable(getattr(b, 'add', None)) for b in builders)) or (type(max_pending_bytes) is not int) or (not 0 < max_pending_bytes <= 4 * 1024 ** 3) or (max_pending_samples != 2) or (type(max_pending_samples) is not int):
            raise ValueError('copy pipeline requires DP16 and two byte-bounded CPU captures')
        self._builders = tuple(builders)
        self._limit_bytes = max_pending_bytes
        self._limit_samples = max_pending_samples
        self._pending = deque()
        self._pending_bytes = 0
        self._peak_bytes = 0
        self._failure = None
        self._closed = False
        self._busy = False
        self._lock = Lock()
        self._lanes = []
        try:
            for i in range(16):
                self._lanes.append(ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'speco-owned-copy-{i}'))
        except BaseException:
            self.close()
            raise

    def _enter(self):
        with self._lock:
            if self._closed or self._busy:
                raise RuntimeError('copy pipeline is closed or already running')
            self._busy = True

    def _leave(self):
        with self._lock:
            self._busy = False

    @staticmethod
    def _copy(builder, fields, metadata):
        if callable(getattr(builder, 'requires_hidden', None)) and callable(getattr(builder, 'add_light_only', None)) and (not builder.requires_hidden(metadata)):
            builder.add_light_only({k: fields[k] for k in ('input_ids', 'position_ids', 'loss_mask')}, metadata)
        else:
            builder.add(fields, metadata)

    def _retire_one(self):
        group = self._pending[0]
        for future in group['futures']:
            try:
                future.result()
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
        self._pending.popleft()
        self._pending_bytes -= group['bytes']

    def _join_all(self):
        while self._pending:
            self._retire_one()

    def submit(self, fields, metadata):
        self._enter()
        try:
            if self._failure is not None:
                raise RuntimeError('copy pipeline is poisoned') from self._failure
            if not isinstance(fields, dict) or set(fields) != {'input_ids', 'position_ids', 'loss_mask', 'hidden_states'} or any((not isinstance(v, torch.Tensor) or v.device.type != 'cpu' for v in fields.values())):
                raise ValueError('copy pipeline requires independently verified full CPU fields')
            size = sum((v.numel() * v.element_size() for v in fields.values()))
            if type(metadata.payload_nbytes) is not int or size != metadata.payload_nbytes or size > self._limit_bytes:
                raise ValueError('verified capture exceeds the owned copy byte budget')
            while self._pending and (len(self._pending) >= self._limit_samples or self._pending_bytes + size > self._limit_bytes or all((f.done() for f in self._pending[0]['futures']))):
                self._retire_one()
                if self._failure is not None:
                    self._join_all()
                    raise RuntimeError('copy pipeline is poisoned') from self._failure
            group = dict(bytes=size, futures=[])
            self._pending.append(group)
            self._pending_bytes += size
            self._peak_bytes = max(self._peak_bytes, self._pending_bytes)
            try:
                for lane, builder in zip(self._lanes, self._builders):
                    group['futures'].append(lane.submit(self._copy, builder, fields, metadata))
            except BaseException as error:
                self._failure = error
                self._join_all()
                raise
        finally:
            self._leave()

    def join(self, *, raise_failure=True):
        if type(raise_failure) is not bool:
            raise ValueError('copy join failure policy must be boolean')
        self._enter()
        try:
            self._join_all()
            if raise_failure and self._failure is not None:
                raise RuntimeError('owned copy failed before materialized release') from self._failure
        finally:
            self._leave()

    @property
    def pending_bytes(self):
        with self._lock:
            return self._pending_bytes

    @property
    def peak_pending_bytes(self):
        with self._lock:
            return self._peak_bytes

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('join copy pipeline before close')
            if self._closed:
                return
            self._closed = True
        self._join_all()
        for lane in self._lanes:
            lane.shutdown(wait=True, cancel_futures=True)
        self._lanes.clear()
        self._builders = ()
from dataclasses import dataclass
import array
from verl_speco.runtime.artifacts import fcntl
import hashlib
import json
import math
import mmap
import os
import re
import socket
import struct
from verl_speco.trainer.distributed_batches import DMTPResponsePart, DMTPPartShard
from verl_speco.producer.feature_payload import payload_checksum
local_swa_frame_DTYPES = {str(t): t for t in (torch.int64, torch.float32, torch.bfloat16, torch.bool)}
local_swa_frame_SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
local_swa_frame_MAX_HEADER = 1024 * 1024

def local_swa_frame_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def local_swa_frame_binding(value):
    if not isinstance(value, dict) or set(value) != {'run_id', 'generation', 'batch_id', 'rank', 'world_size'} or (not all((isinstance(value[k], str) and value[k] for k in ('run_id', 'batch_id')))) or any((type(value[k]) is not int for k in ('generation', 'rank', 'world_size'))) or (value['generation'] < 1) or (not 1 <= value['world_size'] <= 128) or (not 0 <= value['rank'] < value['world_size']):
        raise ValueError('invalid local SWA frame binding')
    return dict(value)

@dataclass
class SealedSWAFrame:
    fd: int
    descriptor: dict

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

class MappedSWAFrame:
    """Views hold their mapping alive until consumers finish H2D copies."""

    def __init__(self, mapping, shard, light, control_metadata=None):
        self._mapping = mapping
        self.shard = shard
        self.light = light
        self.control_metadata = control_metadata

    def close(self):
        self.shard = self.light = None
        self._mapping = None

def local_swa_frame_validate_shard_metadata(part_indices, global_part_count, loss_scale, part_count, binding):
    if type(global_part_count) is not int or global_part_count < 1 or (not isinstance(part_indices, (list, tuple))) or any((type(index) is not int for index in part_indices)) or (type(loss_scale) not in (int, float)) or (not math.isfinite(loss_scale)) or (loss_scale != binding['world_size'] / global_part_count) or (tuple(part_indices) != tuple(range(binding['rank'], global_part_count, binding['world_size']))) or (part_count != (len(part_indices) if part_indices else 1)):
        raise ValueError('local SWA frame shard ownership or reduction differs from binding')

def local_swa_frame_validate_parts_and_fields(parts, fields):
    required = {'document_id', 'response_start', 'response_end', 'context_start', 'context_end', 'context_layers'}
    batch_names = {'input_ids', 'position_ids', 'loss_mask', 'document_ids', 'target_last_hidden_states', 'preplanned_anchor_positions', 'preplanned_anchor_keep_mask'}
    if not isinstance(parts, list) or not parts:
        raise ValueError('local SWA parts metadata is invalid')
    expected = {f'light/{name}' for name in ('input_ids', 'position_ids', 'loss_mask', 'document_ids')}
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or set(part) != required or any((type(v) is not int for v in part.values())) or (part['document_id'] < 0) or (not 1 <= part['context_layers'] <= 128) or (not 0 <= part['context_start'] < part['context_end']) or (not 0 <= part['response_start'] <= part['response_end']) or (part['response_start'] < part['response_end'] and (not part['context_start'] <= part['response_start'] < part['response_end'] <= part['context_end'])):
            raise ValueError('local SWA parts bounds or layers are invalid')
        expected.update((f'part/{index}/{name}' for name in batch_names))
        expected.update((f'part/{index}/hidden_states_list/{layer}' for layer in range(part['context_layers'])))
    if set(fields) != expected:
        raise ValueError('local SWA tensor field schema differs from parts')

def local_swa_frame_validate_control_metadata(value):
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {'payload_checksums', 'lease_ids'} or (not isinstance(value['payload_checksums'], list)) or (not isinstance(value['lease_ids'], list)) or (not 1 <= len(value['lease_ids']) <= 4096) or (len(value['payload_checksums']) != len(value['lease_ids'])) or any((not isinstance(x, str) or re.fullmatch('[0-9a-f]{64}', x) is None for x in value['payload_checksums'])) or any((not isinstance(x, str) or not x for x in value['lease_ids'])) or (len(set(value['lease_ids'])) != len(value['lease_ids'])):
        raise ValueError('local SWA staged control metadata is invalid')

def local_swa_frame_copy_tensor_to_mapping(memory, tensor, *, offset, nbytes):
    """Copy through ATen so independent frame writers can release the GIL."""
    source = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    destination = torch.frombuffer(memory, dtype=torch.uint8, count=nbytes, offset=offset)
    try:
        destination.copy_(source)
    finally:
        del destination

def create_swa_frame(shard, light, *, binding, max_bytes, control_metadata=None, hugepages=False):
    binding = local_swa_frame_binding(binding)
    if type(hugepages) is not bool:
        raise ValueError('local SWA hugepages must be boolean')
    if hugepages and (not hasattr(mmap, 'MADV_HUGEPAGE')):
        raise ValueError('local SWA hugepages require Linux mmap advice support')
    local_swa_frame_validate_control_metadata(control_metadata)
    if type(max_bytes) is not int or max_bytes <= 0 or (not shard.parts) or (shard.global_part_count < 1) or (not math.isfinite(shard.loss_scale)) or (shard.loss_scale != binding['world_size'] / shard.global_part_count):
        raise ValueError('invalid local SWA frame budget or shard reduction')
    local_swa_frame_validate_shard_metadata(shard.part_indices, shard.global_part_count, shard.loss_scale, len(shard.parts), binding)
    fields = {}
    parts = []
    for index, part in enumerate(shard.parts):
        layers = part.batch.get('hidden_states_list')
        if not isinstance(layers, list) or not layers:
            raise ValueError('local SWA frame needs captured context layers')
        parts.append(dict(document_id=part.document_id, response_start=part.response_start, response_end=part.response_end, context_start=part.context_start, context_end=part.context_end, context_layers=len(layers)))
        for name, value in part.batch.items():
            if name == 'lm_head_weight':
                continue
            if name == 'hidden_states_list':
                fields.update({f'part/{index}/{name}/{layer}': t for layer, t in enumerate(value)})
            else:
                fields[f'part/{index}/{name}'] = value
    if set(light) != {'input_ids', 'position_ids', 'loss_mask', 'document_ids', 'lm_head_weight'}:
        raise ValueError('local SWA light metadata has unexpected fields')
    fields.update({f'light/{name}': value for name, value in light.items() if name != 'lm_head_weight'})
    local_swa_frame_validate_parts_and_fields(parts, fields)
    layout = []
    offset = 0
    for name, t in sorted(fields.items()):
        if not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or str(t.dtype) not in local_swa_frame_DTYPES or (not 1 <= t.ndim <= 3) or (not t.numel()) or (not t.is_contiguous()):
            raise ValueError('local SWA payload must be nonempty typed CPU tensors')
        offset = (offset + 63) // 64 * 64
        size = t.numel() * t.element_size()
        if offset + size > max_bytes:
            raise ValueError('local SWA frame exceeds byte budget')
        layout.append(dict(name=name, offset=offset, nbytes=size, shape=list(t.shape), dtype=str(t.dtype)))
        offset += size
    descriptor = dict(schema_version=1, binding=binding, nbytes=offset, fields=layout, parts=parts, part_indices=list(shard.part_indices), global_part_count=shard.global_part_count, loss_scale=shard.loss_scale, payload_sha256='0' * 64)
    if control_metadata is not None:
        descriptor['control_metadata'] = control_metadata
    descriptor['descriptor_sha256'] = local_swa_frame_sha(descriptor)
    if len(json.dumps(descriptor).encode()) > local_swa_frame_MAX_HEADER:
        raise ValueError('local SWA descriptor exceeds control budget')
    fd = os.memfd_create('speco-swa-frame', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(fd, offset)
        digest = hashlib.sha256()
        with mmap.mmap(fd, offset, access=mmap.ACCESS_WRITE) as memory:
            if hugepages:
                memory.madvise(mmap.MADV_HUGEPAGE)
            for item in layout:
                tensor = fields[item['name']]
                descriptor_bytes = json.dumps((item['name'], str(tensor.dtype), list(tensor.shape)), separators=(',', ':')).encode('utf-8')
                digest.update(descriptor_bytes)
                local_swa_frame_copy_tensor_to_mapping(memory, tensor, offset=item['offset'], nbytes=item['nbytes'])
                view = memoryview(memory)[item['offset']:item['offset'] + item['nbytes']]
                try:
                    digest.update(view)
                finally:
                    view.release()
        descriptor['payload_sha256'] = digest.hexdigest()
        descriptor['descriptor_sha256'] = local_swa_frame_sha({k: v for k, v in descriptor.items() if k != 'descriptor_sha256'})
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, local_swa_frame_SEALS)
        return SealedSWAFrame(fd, descriptor)
    except BaseException:
        os.close(fd)
        raise

def map_swa_frame(fd, descriptor, *, expected_binding, max_bytes, lm_head_weight):
    expected_binding = local_swa_frame_binding(expected_binding)
    if not isinstance(descriptor, dict) or descriptor.get('schema_version') != 1 or descriptor.get('binding') != expected_binding or (descriptor.get('descriptor_sha256') != local_swa_frame_sha({k: v for k, v in descriptor.items() if k != 'descriptor_sha256'})) or (type(max_bytes) is not int) or (max_bytes <= 0) or (type(descriptor.get('nbytes')) is not int) or (not 0 < descriptor['nbytes'] <= max_bytes):
        raise ValueError('local SWA frame descriptor, binding or budget differs')
    local_swa_frame_validate_control_metadata(descriptor.get('control_metadata'))
    parts = descriptor.get('parts')
    layout = descriptor.get('fields')
    if not isinstance(layout, list) or not all((isinstance(x, dict) and set(x) == {'name', 'offset', 'nbytes', 'shape', 'dtype'} and isinstance(x['name'], str) for x in layout)):
        raise ValueError('local SWA tensor field layout is invalid')
    local_swa_frame_validate_parts_and_fields(parts, [x['name'] for x in layout])
    local_swa_frame_validate_shard_metadata(descriptor.get('part_indices'), descriptor.get('global_part_count'), descriptor.get('loss_scale'), len(parts), expected_binding)
    if os.fstat(fd).st_size != descriptor['nbytes'] or fcntl.fcntl(fd, fcntl.F_GET_SEALS) & local_swa_frame_SEALS != local_swa_frame_SEALS:
        raise ValueError('local SWA frame is not sealed at its declared length')
    mapping = mmap.mmap(fd, descriptor['nbytes'], access=mmap.ACCESS_COPY)
    fields = {}
    cursor = 0
    try:
        for item in descriptor['fields']:
            name, shape, dtype = (item['name'], item['shape'], local_swa_frame_DTYPES.get(item['dtype']))
            if not isinstance(name, str) or name in fields or dtype is None or (not isinstance(shape, list)) or (not 1 <= len(shape) <= 3) or any((type(n) is not int or n <= 0 for n in shape)) or (type(item['offset']) is not int) or item['offset'] % 64 or (item['offset'] < cursor) or (type(item['nbytes']) is not int) or (item['nbytes'] != math.prod(shape) * torch.empty((), dtype=dtype).element_size()) or (item['offset'] + item['nbytes'] > descriptor['nbytes']):
                raise ValueError('local SWA tensor layout is invalid')
            fields[name] = torch.frombuffer(mapping, dtype=dtype, count=math.prod(shape), offset=item['offset']).reshape(shape)
            cursor = item['offset'] + item['nbytes']
        if cursor != descriptor['nbytes'] or payload_checksum(fields) != descriptor['payload_sha256']:
            raise ValueError('local SWA tensor bytes differ from declared checksum')
        parts = []
        for index, part in enumerate(descriptor['parts']):
            prefix = f'part/{index}/'
            batch = {name[len(prefix):]: value for name, value in fields.items() if name.startswith(prefix) and (not name.startswith(prefix + 'hidden_states_list/'))}
            batch['hidden_states_list'] = [fields[prefix + f'hidden_states_list/{layer}'] for layer in range(part['context_layers'])]
            batch['lm_head_weight'] = lm_head_weight
            parts.append(DMTPResponsePart(**{k: v for k, v in part.items() if k != 'context_layers'}, batch=batch))
        light = {name.removeprefix('light/'): value for name, value in fields.items() if name.startswith('light/')}
        light['lm_head_weight'] = lm_head_weight
        shard = DMTPPartShard(tuple(parts), tuple(descriptor['part_indices']), descriptor['global_part_count'], descriptor['loss_scale'])
        if not parts or shard.loss_scale != expected_binding['world_size'] / shard.global_part_count or tuple(range(expected_binding['rank'], shard.global_part_count, expected_binding['world_size'])) != shard.part_indices:
            raise ValueError('local SWA frame shard ownership differs from binding')
        return MappedSWAFrame(mapping, shard, light, descriptor.get('control_metadata'))
    except BaseException:
        fields.clear()
        raise

def local_swa_frame_receive_exact(sock, count):
    result = bytearray()
    while len(result) < count:
        value = sock.recv(count - len(result))
        if not value:
            raise EOFError('local SWA control stream closed before descriptor completed')
        result.extend(value)
    return bytes(result)

def send_swa_frame(sock, frame):
    encoded = json.dumps(frame.descriptor, sort_keys=True, separators=(',', ':')).encode()
    if not 0 < len(encoded) <= local_swa_frame_MAX_HEADER or frame.fd < 0:
        raise ValueError('local SWA frame is closed or oversized')
    prefix = struct.pack('!I', len(encoded))
    sent = sock.sendmsg([prefix], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [frame.fd]))])
    sock.sendall(prefix[sent:] + encoded)

def receive_swa_frame(sock, *, expected_binding, max_bytes, lm_head_weight):
    prefix, ancillary, flags, _ = sock.recvmsg(4, socket.CMSG_SPACE(array.array('i').itemsize * 2))
    fds = []
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            handles = array.array('i')
            handles.frombytes(data[:len(data) - len(data) % handles.itemsize])
            fds.extend(handles)
    try:
        if len(fds) != 1 or flags & socket.MSG_CTRUNC or (not prefix):
            raise ValueError('local SWA frame did not transfer exactly one file descriptor')
        prefix += local_swa_frame_receive_exact(sock, 4 - len(prefix))
        count = struct.unpack('!I', prefix)[0]
        if not 0 < count <= local_swa_frame_MAX_HEADER:
            raise ValueError('local SWA control descriptor exceeds budget')
        descriptor = json.loads(local_swa_frame_receive_exact(sock, count))
        return map_swa_frame(fds[0], descriptor, expected_binding=expected_binding, max_bytes=max_bytes, lm_head_weight=lm_head_weight)
    finally:
        for fd in fds:
            os.close(fd)
from pathlib import Path
import sys, importlib.util
import verl_speco.trainer.owned_batches as writer_module
import verl_speco.trainer.owned_batches as rows_module

def planned_shape_shard(builder):
    indices = tuple(range(builder.rank, builder._global_parts, builder.world_size))
    if not indices:
        return None
    width = builder._rows[0].hidden_size
    parts = []
    actual_indices = []
    token_offset = 0
    head = torch.empty(1, width, dtype=torch.bfloat16, device='meta')
    for row, local in enumerate(builder._rows):
        doc_offset = builder.row_plan.sample_indices[row][0]
        for ordinal in local._indices:
            lo, hi, sample = local._parts_plan[ordinal]
            prompt = local.metadatas[sample].prompt_token_count
            start = prompt + lo
            end = prompt + hi
            left = max(0, start - local.window_size - 1)
            offset = sum(local.plan.aligned_lengths[:sample]) + token_offset
            count = end - left
            batch = {k: torch.empty(1, count, dtype=dtype, device='meta') for k, dtype in [('input_ids', torch.int64), ('position_ids', torch.int64), ('loss_mask', torch.float32), ('document_ids', torch.int64)]}
            batch['hidden_states_list'] = [torch.empty(1, count, width, dtype=torch.bfloat16, device='meta') for _ in local.metadatas[sample].target_layer_ids]
            batch['target_last_hidden_states'] = torch.empty(1, count, width, dtype=torch.bfloat16, device='meta')
            batch['lm_head_weight'] = head
            anchors = local.labels.shape[-1]
            batch['preplanned_anchor_positions'] = torch.empty(1, anchors, dtype=torch.int64, device='meta')
            batch['preplanned_anchor_keep_mask'] = torch.empty(1, anchors, dtype=torch.bool, device='meta')
            parts.append(DMTPResponsePart(sample + doc_offset, offset + start, offset + end, offset + left, offset + end, batch))
            actual_indices.append(builder._part_offsets[row] + ordinal)
        token_offset += builder.row_plan.rows[row].used_tokens
    if tuple(actual_indices) != indices or token_offset != builder.row_plan.padded_tokens:
        raise ValueError('direct template part or row ownership differs')
    light = {k: torch.empty(1, token_offset, dtype=d, device='meta') for k, d in [('input_ids', torch.int64), ('position_ids', torch.int64), ('loss_mask', torch.float32), ('document_ids', torch.int64)]}
    light['lm_head_weight'] = head
    return (DMTPPartShard(tuple(parts), indices, builder._global_parts, builder.world_size / builder._global_parts), light)

class DirectStreamingOwnedSWARows:

    def __init__(self, metadatas, *, binding, max_frame_bytes, hugepages=False, **kwargs):
        self.writer = None
        self._finished = False
        self._closed = False
        self._binding = binding
        self._max_bytes = max_frame_bytes
        self._hugepages = hugepages
        self._rank = kwargs['rank']
        self._world = kwargs['world_size']
        if binding['rank'] != self._rank or binding['world_size'] != self._world:
            raise ValueError('direct builder binding differs from mesh')
        metadatas = tuple(metadatas)
        self.builder = rows_module.StreamingOwnedSWARows(metadatas, owned_hidden_cloner_factory=self._cloner_factory, **kwargs)
        if self.builder.row_plan.sample_count != len(metadatas):
            raise ValueError('direct row plan would defer a verified capture')
        template = planned_shape_shard(self.builder)
        if template is not None:
            self.writer = writer_module.DirectOwnedFrameWriter(*template, binding=binding, max_bytes=max_frame_bytes, hugepages=hugepages)

    @property
    def row_plan(self):
        return self.builder.row_plan

    @property
    def metadatas(self):
        return self.builder.metadatas

    def _cloner_factory(self, part_offset):

        def clone(ordinal, name, source):
            global_index = part_offset + ordinal
            if self.writer is None or global_index % self._world != self._rank:
                return source.clone()
            index = (global_index - self._rank) // self._world
            self.writer.write_tensor(f'part/{index}/{name}', source)
            return torch.empty(tuple(source.shape), dtype=source.dtype, device='meta')
        return clone

    def add(self, fields, metadata):
        try:
            if self._closed or self._finished:
                raise ValueError('direct streaming builder is closed or finished')
            self.builder.add(fields, metadata)
        except BaseException:
            self.close()
            raise

    def requires_hidden(self, metadata):
        if self._closed or self._finished:
            raise ValueError('direct streaming builder is closed or finished')
        return self.builder.requires_hidden(metadata)

    def add_light_only(self, fields, metadata):
        try:
            if self._closed or self._finished:
                raise ValueError('direct streaming builder is closed or finished')
            self.builder.add_light_only(fields, metadata)
        except BaseException:
            self.close()
            raise

    def finish_frame(self, lm_head_weight, *, control_metadata=None):
        try:
            if self._closed or self._finished:
                raise ValueError('direct streaming builder is closed or finished')
            shard, light, count = self.builder.finish(lm_head_weight)
            self._finished = True
            if self.writer is None:
                return create_swa_frame(shard, light, binding=self._binding, max_bytes=self._max_bytes, hugepages=self._hugepages, control_metadata=control_metadata)
            fields, parts = writer_module.frame_fields(shard, light)
            d = self.writer._descriptor
            if parts != d['parts'] or list(shard.part_indices) != d['part_indices'] or shard.global_part_count != d['global_part_count'] or (shard.loss_scale != d['loss_scale']) or (set(fields) != set(self.writer._layout)):
                raise ValueError('direct streaming actual plan differs from template')
            for name, t in fields.items():
                item = self.writer._layout[name]
                if list(t.shape) != item['shape'] or str(t.dtype) != item['dtype']:
                    raise ValueError('direct actual field shape or dtype differs')
                if name in self.writer._written:
                    if t.device.type != 'meta':
                        raise ValueError('direct hidden field unexpectedly retained actual storage')
                else:
                    self.writer.write_tensor(name, t)
            return self.writer.seal(control_metadata=control_metadata)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._closed = True
        if self.writer is not None:
            self.writer.close()
import os, mmap, hashlib, json, math
from verl_speco.runtime.artifacts import fcntl

def frame_fields(shard, light):
    fields = {}
    parts = []
    for index, part in enumerate(shard.parts):
        layers = part.batch.get('hidden_states_list')
        if not isinstance(layers, list) or not layers:
            raise ValueError('direct frame context layers missing')
        parts.append(dict(document_id=part.document_id, response_start=part.response_start, response_end=part.response_end, context_start=part.context_start, context_end=part.context_end, context_layers=len(layers)))
        for name, value in part.batch.items():
            if name == 'lm_head_weight':
                continue
            if name == 'hidden_states_list':
                fields.update({f'part/{index}/{name}/{layer}': t for layer, t in enumerate(value)})
            else:
                fields[f'part/{index}/{name}'] = value
    if set(light) != {'input_ids', 'position_ids', 'loss_mask', 'document_ids', 'lm_head_weight'}:
        raise ValueError('direct frame light schema differs')
    fields.update({f'light/{name}': value for name, value in light.items() if name != 'lm_head_weight'})
    local_swa_frame_validate_parts_and_fields(parts, fields)
    return (fields, parts)

class DirectOwnedFrameWriter:
    """Plan fixed fields from shapes, populate each once, fully hash and seal.

 Templates may be meta tensors. Incoming tensors are independent verified CPU
 captures; strided hidden windows copy directly into final typed storage. This
 primitive does not perform capture verification or plan ownership itself.
 """

    def __init__(self, shard, light, *, binding, max_bytes, hugepages=False):
        self.fd = -1
        self._mapping = None
        self._written = set()
        self._sealed = False
        binding = local_swa_frame_binding(binding)
        if type(hugepages) is not bool or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('invalid direct frame budget or hugepage flag')
        if hugepages and (not hasattr(mmap, 'MADV_HUGEPAGE')):
            raise ValueError('direct frame requires mmap hugepage support')
        local_swa_frame_validate_shard_metadata(shard.part_indices, shard.global_part_count, shard.loss_scale, len(shard.parts), binding)
        fields, parts = frame_fields(shard, light)
        layout = []
        offset = 0
        for name, t in sorted(fields.items()):
            if not isinstance(t, torch.Tensor) or t.device.type not in ('cpu', 'meta') or str(t.dtype) not in local_swa_frame_DTYPES or (not 1 <= t.ndim <= 3) or (not t.numel()):
                raise ValueError('direct frame template field is invalid')
            offset = (offset + 63) // 64 * 64
            size = t.numel() * t.element_size()
            if offset + size > max_bytes:
                raise ValueError('direct frame exceeds byte budget')
            layout.append(dict(name=name, offset=offset, nbytes=size, shape=list(t.shape), dtype=str(t.dtype)))
            offset += size
        self._descriptor = dict(schema_version=1, binding=binding, nbytes=offset, fields=layout, parts=parts, part_indices=list(shard.part_indices), global_part_count=shard.global_part_count, loss_scale=shard.loss_scale, payload_sha256='0' * 64)
        self._descriptor['descriptor_sha256'] = local_swa_frame_sha(self._descriptor)
        if len(json.dumps(self._descriptor).encode()) > local_swa_frame_MAX_HEADER:
            raise ValueError('direct frame descriptor exceeds control budget')
        self._layout = {x['name']: x for x in layout}
        try:
            self.fd = os.memfd_create('speco-direct-owned-frame', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            os.ftruncate(self.fd, offset)
            self._mapping = mmap.mmap(self.fd, offset, access=mmap.ACCESS_WRITE)
            if hugepages:
                self._mapping.madvise(mmap.MADV_HUGEPAGE)
        except BaseException:
            self.close()
            raise

    def write_tensor(self, name, tensor):
        destination = None
        try:
            if self.fd < 0 or self._sealed or self._mapping is None:
                raise ValueError('direct frame writer is closed')
            item = self._layout.get(name)
            if item is None or name in self._written:
                raise ValueError('direct frame field is unknown or already written')
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or str(tensor.dtype) != item['dtype'] or (list(tensor.shape) != item['shape']):
                raise ValueError('direct frame source shape dtype or device differs')
            destination = torch.frombuffer(self._mapping, dtype=local_swa_frame_DTYPES[item['dtype']], count=tensor.numel(), offset=item['offset']).reshape(item['shape'])
            destination.copy_(tensor.detach())
            del destination
            destination = None
            self._written.add(name)
        except BaseException:
            del destination
            self.close()
            raise

    def seal(self, *, control_metadata=None):
        try:
            if self.fd < 0 or self._sealed or self._mapping is None or (self._written != set(self._layout)):
                raise ValueError('direct frame fields incomplete or writer closed')
            local_swa_frame_validate_control_metadata(control_metadata)
            descriptor = json.loads(json.dumps(self._descriptor))
            if control_metadata is not None:
                descriptor['control_metadata'] = json.loads(json.dumps(control_metadata))
            if len(json.dumps(descriptor).encode()) > local_swa_frame_MAX_HEADER:
                raise ValueError('direct frame descriptor exceeds control budget')
            digest = hashlib.sha256()
            for item in descriptor['fields']:
                digest.update(json.dumps((item['name'], item['dtype'], item['shape']), separators=(',', ':')).encode('utf-8'))
                view = memoryview(self._mapping)[item['offset']:item['offset'] + item['nbytes']]
                try:
                    digest.update(view)
                finally:
                    view.release()
            descriptor['payload_sha256'] = digest.hexdigest()
            descriptor['descriptor_sha256'] = local_swa_frame_sha({k: v for k, v in descriptor.items() if k != 'descriptor_sha256'})
            self._mapping.close()
            self._mapping = None
            fcntl.fcntl(self.fd, fcntl.F_ADD_SEALS, local_swa_frame_SEALS)
            frame = SealedSWAFrame(self.fd, descriptor)
            self.fd = -1
            self._sealed = True
            return frame
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self):
        self.close()
import select
import time
VERIFIED = b'HASHGOOD'
READY = b'SPECORDY'

def wait_dispatch_ready(client):
    """Call only after receive_swa_frame has independently checked the full SHA."""
    client.sendall(VERIFIED)
    if local_swa_frame_receive_exact(client, len(READY)) != READY:
        raise ValueError('frame dispatch did not commit complete verification')

def seal_and_dispatch_frames(server, *, ranks, owner, seal_one, map_owner, seal_workers, timeout_seconds):
    ranks = tuple(ranks)
    if not ranks or len(set(ranks)) != len(ranks) or owner not in ranks or any((type(rank) is not int or rank < 0 for rank in ranks)) or (type(seal_workers) is not int) or (not 1 <= seal_workers <= min(16, len(ranks))) or (not callable(seal_one)) or (not callable(map_owner)) or (type(timeout_seconds) not in (int, float)) or (not math.isfinite(timeout_seconds)) or (timeout_seconds <= 0):
        raise ValueError('invalid bounded frame dispatch')
    started = time.monotonic()
    deadline = started + timeout_seconds
    consumers = set(ranks) - {owner}
    clients, frames, sent, verified = ({}, {}, set(), set())
    owner_future = None
    owner_mapping = None
    succeeded = False
    last_seal_seconds = owner_verified_seconds = 0.0
    pool = ThreadPoolExecutor(max_workers=seal_workers)
    futures = {rank: pool.submit(seal_one, rank) for rank in ranks}
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('sealed-frame dispatch or full verification timed out')
            for rank, future in futures.items():
                if rank not in frames and future.done():
                    frame = future.result()
                    if frame.descriptor['binding']['rank'] != rank:
                        frame.close()
                        raise ValueError('sealed frame belongs to another DP rank')
                    frames[rank] = frame
                    last_seal_seconds = time.monotonic() - started
            if owner in frames and owner_future is None:
                owner_future = pool.submit(map_owner, frames[owner])
            if owner_future is not None and owner_future.done() and (owner_mapping is None):
                owner_mapping = owner_future.result()
                owner_verified_seconds = time.monotonic() - started
            for rank in consumers & clients.keys() & frames.keys() - sent:
                clients[rank].settimeout(max(0.001, remaining))
                send_swa_frame(clients[rank], frames[rank])
                frames[rank].close()
                sent.add(rank)
            if len(frames) == len(ranks) and owner_mapping is not None and (verified == consumers):
                for client in clients.values():
                    client.sendall(READY)
                succeeded = True
                print('pipelined_frame_dispatch_verified ' + json.dumps(dict(frame_count=len(ranks), full_consumer_SHA_count=len(verified), owner_full_SHA_verified=True, last_seal_seconds=last_seal_seconds, owner_verified_seconds=owner_verified_seconds, handoff_verified_seconds=time.monotonic() - started, total_original_frame_bytes=sum((f.descriptor['nbytes'] for f in frames.values())))), flush=True)
                return owner_mapping
            waiting = [clients[rank] for rank in sent - verified]
            watch = waiting + ([server] if clients.keys() != consumers else [])
            if not watch:
                time.sleep(min(0.002, remaining))
                continue
            readable, _, _ = select.select(watch, [], [], min(0.01, remaining))
            for endpoint in readable:
                if endpoint is server:
                    client, _ = server.accept()
                    try:
                        client.settimeout(max(0.001, deadline - time.monotonic()))
                        _, uid, _ = struct.unpack('3i', client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if uid != os.getuid():
                            raise ValueError('local frame consumer UID differs')
                        rank = struct.unpack('!I', local_swa_frame_receive_exact(client, 4))[0]
                        if rank not in consumers or rank in clients:
                            raise ValueError('duplicate or foreign frame consumer')
                        clients[rank] = client
                    except BaseException:
                        client.close()
                        raise
                else:
                    rank = next((rank for rank, client in clients.items() if client is endpoint))
                    if local_swa_frame_receive_exact(endpoint, len(VERIFIED)) != VERIFIED:
                        raise ValueError('consumer did not verify complete frame')
                    verified.add(rank)
    finally:
        for client in clients.values():
            client.close()
        for future in futures.values():
            future.cancel()
        if owner_future is not None:
            owner_future.cancel()
        pool.shutdown(wait=True)
        for rank, future in futures.items():
            if not future.cancelled() and future.exception() is None:
                frames.setdefault(rank, future.result())
        for frame in frames.values():
            frame.close()
        if not succeeded:
            if owner_mapping is None and owner_future is not None:
                if not owner_future.cancelled() and owner_future.exception() is None:
                    owner_mapping = owner_future.result()
            if owner_mapping is not None:
                owner_mapping.close()
from verl_speco.trainer.feature_pipeline import BoundedWaveMaterializer
from verl_speco.runtime.services import check_prefill_abort
from verl_speco.runtime.artifacts import publish_json, read_published_json
from verl_speco.transport.control import AcquireRequest
from verl_speco.transport.waves import MAX_MATERIALIZED_WAVE_SAMPLES

def stage_native_swa_waves(samples, *, control, reader, tq, partition, builders, wave_size, max_wave_bytes, ack_dir, timeout=1800, poll_seconds=0.5, ack_transport='filesystem', prefetch_bytes=0, verification_pipeline=False, batch_release=False, verification_workers=2, owned_copy_workers=1, exact_acquire=False, node_session=None, owned_copy_pipeline=False, release_chunk_size=0):
    samples = tuple(samples)
    if type(wave_size) is not int or not 1 <= wave_size <= MAX_MATERIALIZED_WAVE_SAMPLES or type(timeout) not in (int, float) or (not math.isfinite(timeout)) or (timeout <= 0) or (type(poll_seconds) not in (int, float)) or (not math.isfinite(poll_seconds)) or (not 0 < poll_seconds <= 30) or (not isinstance(builders, list)) or (not builders) or (ack_transport not in ('filesystem', 'ray')) or (type(prefetch_bytes) is not int) or (not 0 <= prefetch_bytes <= (16 if verification_workers == 8 else 8) * 1024 ** 3) or (type(owned_copy_pipeline) is not bool) or (type(exact_acquire) is not bool) or (type(batch_release) is not bool) or (type(owned_copy_workers) is not int) or (owned_copy_workers not in (1, 2, 4, 8, 16)) or (type(verification_workers) is not int) or (verification_workers not in (2, 4, 8)) or (verification_workers != 2 and (not verification_pipeline)) or (type(verification_pipeline) is not bool) or (verification_pipeline and (not prefetch_bytes)):
        raise ValueError('invalid native prefill wave size or timing')
    if batch_release and (not callable(getattr(control, 'release_materialized_batch', None))):
        raise ValueError('native wave batch release requires owned control support')
    if verification_pipeline and any((not callable(getattr(reader, name, None)) for name in ('_copy_cpu_for_verification', '_verify_cpu_packet'))):
        raise ValueError('native verification pipeline requires bounded independent CPU packet reader')
    root = Path(__file__).resolve().parents[2] / 'outputs'
    ack_dir = Path(ack_dir).resolve()
    if not ack_dir.is_relative_to(root):
        raise ValueError('native wave acknowledgements must stay under SpeCo outputs')
    stager = BoundedWaveMaterializer(samples, control=control, consumer_id='dspark-trainer', max_wave_bytes=max_wave_bytes)
    waves = [samples[i:i + wave_size] for i in range(0, len(samples), wave_size)]
    if any((sum((m.payload_nbytes for _, m in wave)) > max_wave_bytes for wave in waves)):
        raise ValueError('native wave exceeds retained budget before source admission')
    copies = getattr(reader, 'copy_lanes', None)
    parallel = callable(copies)
    if parallel and (not verification_pipeline):
        raise ValueError('independent native readers require verification pipeline')
    bulk = getattr(reader.transport, 'cpu_bulk_receive', False) if hasattr(reader, 'transport') else False
    if verification_workers == 8 and (not parallel or bulk is not True or (wave_size, max_wave_bytes) not in ((48, 96 * 1024 ** 3), (96, 48 * 1024 ** 3)) or (len(builders) != 16) or (node_session is not None) or (owned_copy_workers not in (8, 16)) or (not exact_acquire) or (not batch_release) or (ack_transport != 'ray') or (prefetch_bytes != 16 * 1024 ** 3)):
        raise ValueError('eight verifier waves require the exact independent DP16 16GiB profile')
    if type(release_chunk_size) is not int or release_chunk_size not in (0, 6, 12) or (release_chunk_size and (not owned_copy_pipeline)):
        raise ValueError('prefix release needs the bounded DP16 owned copy pipeline')
    copy_pending_bytes = 0
    if owned_copy_pipeline:
        if not parallel or bulk is not True or verification_workers != 8 or (len(getattr(reader, 'readers', ())) != 8) or (owned_copy_workers != 16) or (len(builders) != 16) or (node_session is not None) or ((wave_size, max_wave_bytes) != (48, 96 * 1024 ** 3)) or (not exact_acquire) or (not batch_release) or (ack_transport != 'ray') or (prefetch_bytes != 16 * 1024 ** 3):
            raise ValueError('owned copy pipeline requires the exact DP16 eight-reader profile')
        copy_pending_bytes = 2 * max((m.payload_nbytes for _, m in samples))
        if copy_pending_bytes > 4 * 1024 ** 3:
            raise ValueError('owned copy pipeline exceeds its CPU buffer budget')
    slots = verification_workers + 1 if verification_pipeline else 2
    if prefetch_bytes and slots * max((m.payload_nbytes + (4096 if bulk else 0) for _, m in samples)) + copy_pending_bytes > prefetch_bytes:
        raise ValueError('native payload prefetch exceeds its explicit CPU buffer budget')
    ack_dir.mkdir(parents=True, exist_ok=True)
    from verl_speco.transport.waves import wave_batch_identity
    batch_id = wave_batch_identity(samples) if ack_transport == 'ray' else None
    if owned_copy_pipeline:
        owned_copies = BoundedOwnedBuilderPipeline(builders, max_pending_bytes=copy_pending_bytes)
    else:
        owned_copies = OwnedBuilderCopies(builders, workers=owned_copy_workers)

    def discard_owned_builders():
        owned_copies.close()
        errors = []
        for builder in builders:
            try:
                if callable(getattr(builder, 'close', None)):
                    builder.close()
            except BaseException as error:
                errors.append(error)
        builders.clear()
        if errors:
            raise RuntimeError('owned builder cleanup failed after copy workers joined') from errors[0]
    leases = ()
    parallel_pools = None
    try:
        for wave_index, wave in enumerate(waves):
            check_prefill_abort(ack_dir, samples)
            wave_started = time.perf_counter()
            ack_path = ack_dir / f'wave-{wave_index:04d}.ack.json'
            try:
                read_published_json(ack_path)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError('native prefill wave acknowledgement already exists')
            first = wave[0][1]
            identity = hashlib.sha256('\n'.join((key for key, _ in wave)).encode()).hexdigest()[:20]
            request = AcquireRequest('prefill-wave-' + identity, 'dspark-trainer', first.generation, len(wave), len(wave), first.target_revision, time.time() + timeout)
            deadline = time.monotonic() + timeout
            keys = [key for key, _ in wave]
            while True:
                check_prefill_abort(ack_dir, samples)
                acquired, leases = control.acquire(request, lease_timeout_s=timeout * 2, **{'required_keys': tuple(keys)} if exact_acquire else {})
                if acquired.status == 'ready':
                    break
                if acquired.status not in ('busy', 'waiting', 'pending', 'empty') or time.monotonic() >= deadline:
                    raise TimeoutError('native prefill wave did not become ready before deadline')
                time.sleep(poll_seconds)
            ready_wait_seconds = time.perf_counter() - wave_started
            keys = [key for key, _ in wave]
            phase = dict(native_get_verify_seconds=0.0, owned_parts_copy_seconds=0.0, tq_clear_seconds=0.0)
            stage_started = time.perf_counter()
            check_prefill_abort(ack_dir, samples)
            reader.prepare_receive(keys, partition)

            def materialize(fields, metadata):
                started = time.perf_counter()
                if owned_copy_pipeline:
                    owned_copies.submit(fields, metadata)
                else:
                    owned_copies.add(fields, metadata)
                phase['owned_parts_copy_seconds'] += time.perf_counter() - started

            def verified(key):
                check_prefill_abort(ack_dir, samples)
                started = time.perf_counter()
                value = reader.get_verified_cpu(key, partition)
                phase['native_get_verify_seconds'] += time.perf_counter() - started
                return value

            def clear(key):
                started = time.perf_counter()
                tq.kv_clear(keys=key, partition_id=partition)
                if key in tq.kv_list(partition_id=partition).get(partition, {}):
                    raise RuntimeError('native wave TQ key remains after clear')
                phase['tq_clear_seconds'] += time.perf_counter() - started

            def clear_batch(keys):
                started = time.perf_counter()
                tq.kv_clear(keys=list(keys), partition_id=partition)
                remaining = tq.kv_list(partition_id=partition).get(partition, {})
                if any((key in remaining for key in keys)):
                    raise RuntimeError('native wave TQ key remains after batch clear')
                phase['tq_clear_seconds'] += time.perf_counter() - started
            prefetch = None
            if prefetch_bytes:
                from verl_speco.trainer.feature_pipeline import OneAheadVerifiedPayloads
                device = getattr(getattr(reader, 'transport', None), 'device', None)

                def initialize():
                    if device is not None and device.type == 'npu':
                        import torch
                        torch.npu.set_device(device)
                if verification_pipeline:
                    from verl_speco.trainer.feature_pipeline import PipelinedVerifiedPayloads
                    from threading import Lock
                    phase_lock = Lock()

                    def copy_for_verify(key):
                        check_prefill_abort(ack_dir, samples)
                        started = time.perf_counter()
                        packet = reader._copy_cpu_for_verification(key, partition)
                        with phase_lock:
                            phase['native_get_verify_seconds'] += time.perf_counter() - started
                        return packet

                    def verify_packet(packet):
                        started = time.perf_counter()
                        value = reader._verify_cpu_packet(packet, pipelined=True)
                        with phase_lock:
                            phase['native_get_verify_seconds'] += time.perf_counter() - started
                        return value
                    budget = prefetch_bytes - slots * (4096 if bulk else 0) - copy_pending_bytes
                    if parallel:
                        from verl_speco.trainer.feature_pipeline import ParallelVerifiedPayloads, ParallelVerificationPools

                        def measured_copy(lane):

                            def copy(key):
                                check_prefill_abort(ack_dir, samples)
                                started = time.perf_counter()
                                packet = lane(key)
                                with phase_lock:
                                    phase['native_get_verify_seconds'] += time.perf_counter() - started
                                return packet
                            return copy
                        if parallel_pools is None:
                            parallel_pools = ParallelVerificationPools(reader_count=len(reader.readers), verification_workers=verification_workers, initialize=initialize)
                        prefetch = ParallelVerifiedPayloads(wave, copies=[measured_copy(lane) for lane in copies(partition)], verify=verify_packet, max_bytes=budget, verification_workers=verification_workers, pools=parallel_pools)
                    else:
                        prefetch = PipelinedVerifiedPayloads(wave, copy=copy_for_verify, verify=verify_packet, max_bytes=budget, initialize=initialize, verification_workers=verification_workers)
                else:
                    prefetch = OneAheadVerifiedPayloads(wave, fetch=verified, max_bytes=prefetch_bytes, initialize=initialize)
            node_wave = {}

            def acquired_node_wave(bound, expected, records):
                node_wave.update(node_session.begin(wave_index=wave_index, leases=bound, expected=expected, records=records))

            def quiesce_wave():
                try:
                    if prefetch is not None:
                        prefetch.close()
                finally:
                    if owned_copy_pipeline:
                        started = time.perf_counter()
                        try:
                            owned_copies.join(raise_failure=False)
                        finally:
                            phase['owned_parts_copy_seconds'] += time.perf_counter() - started

            def all_node_ram_staged(bound, expected, records):
                quiesce_wave()
                if owned_copy_pipeline:
                    started = time.perf_counter()
                    owned_copies.join()
                    phase['owned_parts_copy_seconds'] += time.perf_counter() - started
                if node_session is not None:
                    node_session.before_release(node_wave)

            def before_owned_prefix_release(bound, expected):
                if prefetch is None:
                    raise RuntimeError('prefix release lacks independent verified readers')
                prefetch.require_consumed_prefix(tuple((key for key, _ in expected)))
                started = time.perf_counter()
                try:
                    owned_copies.join()
                finally:
                    phase['owned_parts_copy_seconds'] += time.perf_counter() - started
            try:
                stager.stage(acquired, leases, fetch_verified=prefetch.get if prefetch else verified, materialize=materialize, clear_payload=clear, discard=discard_owned_builders, quiesce=quiesce_wave, clear_payload_batch=clear_batch if batch_release else None, on_acquired=acquired_node_wave if node_session is not None else None, before_payload_release=all_node_ram_staged if owned_copy_pipeline or node_session is not None else None, release_chunk_size=release_chunk_size, before_chunk_release=before_owned_prefix_release if release_chunk_size else None)
            finally:
                if prefetch is not None:
                    prefetch.close()
            if node_session is not None:
                node_session.confirm_release()
            stage_seconds = time.perf_counter() - stage_started
            ack_started = time.perf_counter()
            ack = dict(schema_version=1, run_id=first.run_id, generation=first.generation, wave_index=wave_index, state='PREFILL_WAVE_MATERIALIZED', sample_keys=keys, staged_samples=stager.snapshot()['staged_samples'], expected_samples=len(samples), optimizer_started=False, committed_original_tokens=0)
            publish_json(ack_path, ack)
            if ack_transport == 'ray':
                control.publish_materialized_wave_ack(batch_id=batch_id, ack=ack, lease_ids=tuple(stager.snapshot()['lease_ids'][-len(wave):]))
            print('native_swa_wave_timing ' + json.dumps(dict(run_id=first.run_id, generation=first.generation, wave_index=wave_index, samples=len(wave), payload_bytes=sum((m.payload_nbytes for _, m in wave)), ready_wait_seconds=ready_wait_seconds, stage_seconds=stage_seconds, prefetch_bytes=prefetch_bytes, verification_pipeline=verification_pipeline, batch_release=batch_release, verification_workers=verification_workers, owned_copy_workers=owned_copy_workers, exact_acquire=exact_acquire, owned_copy_pipeline=owned_copy_pipeline, release_chunk_size=release_chunk_size, copy_pending_budget_bytes=copy_pending_bytes, copy_peak_pending_bytes=owned_copies.peak_pending_bytes if owned_copy_pipeline else 0, native_receive_workers=len(reader.readers) if parallel else 1, native_cpu_bulk_receive=bool(bulk), ack_publication_seconds=time.perf_counter() - ack_started, wallclock_seconds=time.perf_counter() - wave_started, **phase)), flush=True)
    except BaseException:
        try:
            stager.abort(leases, discard=discard_owned_builders)
        finally:
            if node_session is None:
                control.reap_expired()
            else:
                try:
                    node_session.abort()
                except Exception:
                    pass
        raise
    finally:
        owned_copies.close()
        if parallel_pools is not None:
            parallel_pools.close()
    lease_ids, records = stager.seal()
    return dict(payload_checksums=[record.payload_checksum for record in records], lease_ids=list(lease_ids))

def resolve_node_frame_budget(raw_row_bytes, configured_bytes=None):
    value = raw_row_bytes if configured_bytes is None else configured_bytes
    if type(raw_row_bytes) is not int or raw_row_bytes <= 0 or type(value) is not int or (not 0 < value <= 64 * 1024 ** 3):
        raise ValueError('node SWA frame budget must be positive and at most 64 GiB')
    return value
import concurrent.futures

def resolve_frame_seal_workers(value, *, world_size, node_owned):
    """Opt-in bounded CPU workers; preserve the serial default."""
    if type(value) is not str or value not in ('', '1', '2', '4', '8', '16') or type(world_size) is not int or (not 1 <= world_size <= 128) or (type(node_owned) is not bool):
        raise ValueError('invalid SWA frame seal worker policy')
    workers = int(value or '1')
    if workers > world_size or (workers > 1 and (not node_owned)):
        raise ValueError('parallel SWA sealing requires a node owner and enough DP ranks')
    return workers

def resolve_frame_hugepages(value, *, node_owned):
    """Opt in per-mapping advice without changing global kernel settings."""
    if type(value) is not str or value not in ('', '0', '1') or type(node_owned) is not bool:
        raise ValueError('invalid SWA frame hugepage policy')
    enabled = value == '1'
    if enabled and (not node_owned):
        raise ValueError('SWA frame hugepages require a native node owner')
    if enabled:
        import mmap
        if not hasattr(mmap, 'MADV_HUGEPAGE'):
            raise ValueError('SWA frame hugepages require Linux mmap advice support')
    return enabled

def resolve_direct_streaming_frame(value, *, node_owned, prefill_waves):
    """Direct writes need metadata-planned rows and the verified native owner."""
    if type(value) is not str or value not in ('', '0', '1') or type(node_owned) is not bool or (type(prefill_waves) is not bool):
        raise ValueError('invalid direct streaming frame policy')
    enabled = value == '1'
    if enabled and (not node_owned or not prefill_waves):
        raise ValueError('direct streaming frame requires native node owner and prefill waves')
    return enabled

def node_owned_swa_node_scope(binding, local_world_size=None):
    binding = local_swa_frame_binding(binding)
    local = binding['world_size'] if local_world_size is None else local_world_size
    if type(local) is not int or not 1 <= local <= binding['world_size'] or binding['world_size'] % local:
        raise ValueError('local SWA DP scope must evenly divide the global world')
    owner = binding['rank'] // local * local
    return (binding, owner, tuple(range(owner, owner + local)))

def node_owned_swa_address(binding, local_world_size=None):
    binding, owner, ranks = node_owned_swa_node_scope(binding, local_world_size)
    identity = {k: v for k, v in binding.items() if k != 'rank'}
    if len(ranks) != binding['world_size']:
        identity['node_owner_rank'] = owner
    return '\x00speco-swa-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:48]

def receive_node_owned_swa(*, binding, max_bytes, lm_head_weight, timeout=180, local_world_size=None, pipelined_dispatch=False):
    if type(pipelined_dispatch) is not bool:
        raise ValueError('pipelined dispatch must be boolean')
    binding, owner, ranks = node_owned_swa_node_scope(binding, local_world_size)
    if binding['rank'] == owner:
        raise ValueError('SWA owner must prepare rather than receive')
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        while True:
            try:
                client.connect(node_owned_swa_address(binding, local_world_size))
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('SWA owner did not bind its local endpoint')
                time.sleep(0.05)
        client.settimeout(max(0.01, deadline - time.monotonic()))
        client.sendall(struct.pack('!I', binding['rank']))
        mapped = receive_swa_frame(client, expected_binding=binding, max_bytes=max_bytes, lm_head_weight=lm_head_weight)
        try:
            if pipelined_dispatch:
                wait_dispatch_ready(client)
            return mapped
        except BaseException:
            mapped.close()
            raise

def prepare_node_owned_swa(*, binding, builder_factory, captures, fetch_verified, lm_head_weight, max_bytes, timeout=180, populate=None, seal_workers=1, hugepages=False, direct_streaming=False, local_world_size=None, pipelined_dispatch=False):
    """Fetch each capture exactly once and distribute only after all verify.

    `captures` is an ordered iterable of (key, metadata); fetch_verified must
    reject checksum or ready-tag mismatches. The default fetch path leaves leases
    to the transaction coordinator. With populate, that callback manages wave
    materialization/release and returns lease provenance for the optimizer.
    """
    binding, owner, ranks = node_owned_swa_node_scope(binding, local_world_size)
    if binding['rank'] != owner:
        raise ValueError('node SWA preparation requires its local owner rank')
    if type(seal_workers) is not int or not 1 <= seal_workers <= min(16, len(ranks)):
        raise ValueError('invalid bounded SWA frame seal workers')
    if type(hugepages) is not bool:
        raise ValueError('SWA hugepages must be boolean')
    if type(direct_streaming) is not bool:
        raise ValueError('direct streaming frame must be boolean')
    if type(pipelined_dispatch) is not bool:
        raise ValueError('pipelined dispatch must be boolean')
    builders = {}
    mapped = None
    try:
        for rank in ranks:
            builder = builder_factory(rank)
            builders[rank] = builder
            if direct_streaming:
                if not isinstance(builder, DirectStreamingOwnedSWARows):
                    raise ValueError('direct frame requires a metadata-planned direct builder')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.settimeout(timeout)
            server.bind(node_owned_swa_address(binding, local_world_size))
            server.listen(len(ranks))
            control_metadata = None
            if populate is not None:
                control_metadata = populate([builders[rank] for rank in ranks])
            else:
                for key, metadata in captures:
                    fields = fetch_verified(key, metadata)
                    for builder in builders.values():
                        builder.add(fields, metadata)
                    del fields
            frames = {}
            owner_frame = None
            try:

                def seal_rank(rank):
                    builder = builders[rank]
                    started = time.perf_counter()
                    if direct_streaming:
                        finish_seconds = 0.0
                        frames[rank] = builder.finish_frame(lm_head_weight, control_metadata=control_metadata)
                    else:
                        shard, light, _ = builder.finish(lm_head_weight)
                        finish_seconds = time.perf_counter() - started
                        started = time.perf_counter()
                        frames[rank] = create_swa_frame(shard, light, binding=dict(binding, rank=rank), max_bytes=max_bytes, control_metadata=control_metadata, **{'hugepages': True} if hugepages else {})
                    print('node_swa_frame_preparation_timing ' + json.dumps(dict(rank=rank, finish_seconds=finish_seconds, seal_seconds=time.perf_counter() - started, frame_bytes=frames[rank].descriptor['nbytes'], seal_workers=seal_workers, hugepages=hugepages, direct_streaming=direct_streaming)), flush=True)
                    if callable(getattr(builder, 'close', None)):
                        builder.close()
                    builders[rank] = None
                    return frames[rank]
                if pipelined_dispatch:
                    mapped = seal_and_dispatch_frames(server, ranks=ranks, owner=owner, seal_one=seal_rank, map_owner=lambda frame: map_swa_frame(frame.fd, frame.descriptor, expected_binding=binding, max_bytes=max_bytes, lm_head_weight=lm_head_weight), seal_workers=seal_workers, timeout_seconds=timeout)
                    return mapped
                if seal_workers == 1:
                    for rank in ranks:
                        seal_rank(rank)
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=seal_workers) as pool:
                        futures = [pool.submit(seal_rank, rank) for rank in ranks]
                        try:
                            for future in futures:
                                future.result()
                        finally:
                            for future in futures:
                                future.cancel()
                builders.clear()
                shard = light = builder = None
                owner_frame = frames.pop(owner)
                deadline = time.monotonic() + timeout
                while frames:
                    server.settimeout(max(0.01, deadline - time.monotonic()))
                    client, _ = server.accept()
                    with client:
                        client.settimeout(max(0.01, deadline - time.monotonic()))
                        _, uid, _ = struct.unpack('3i', client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if uid != os.getuid():
                            raise ValueError('local SWA consumer UID differs from owner')
                        rank = struct.unpack('!I', local_swa_frame_receive_exact(client, 4))[0]
                        if rank not in frames:
                            raise ValueError('local SWA consumer rank is duplicate or foreign')
                        send_swa_frame(client, frames[rank])
                        frames.pop(rank).close()
                mapped = map_swa_frame(owner_frame.fd, owner_frame.descriptor, expected_binding=binding, max_bytes=max_bytes, lm_head_weight=lm_head_weight)
                return mapped
            except BaseException:
                if mapped is not None:
                    mapped.close()
                raise
            finally:
                if owner_frame is not None:
                    owner_frame.close()
                for frame in frames.values():
                    frame.close()
    finally:
        for builder in builders.values():
            if builder is not None and callable(getattr(builder, 'close', None)):
                builder.close()

def resolve_pipelined_frame_dispatch(environment):
    value = environment.get('SPECO_DRAFT_PIPELINED_FRAME_DISPATCH', '')
    if value not in ('', '0', '1'):
        raise ValueError('invalid pipelined frame dispatch policy')
    enabled = value == '1'
    if enabled and any((environment.get(k) != '1' for k in ('SPECO_NATIVE_FEATURE_TRANSPORT', 'SPECO_DRAFT_NODE_OWNED_SWA', 'SPECO_DRAFT_PREFILL_WAVES', 'SPECO_DRAFT_DIRECT_STREAMING_FRAME'))):
        raise ValueError('pipelined frame dispatch requires verified direct native node waves')
    return enabled
from dataclasses import replace
from verl_speco.trainer.distributed_batches import dmtp_swa_response_parts, shard_dmtp_swa_parts
from verl_speco.trainer.distributed_batches import plan_packed_rows
from verl_speco.trainer.distributed_batches import tq_dspark_batch_plan_packed_samples as _plan_packed_samples, materialize_dspark_batch

def materialize_online_swa_part_shard(samples, *, lm_head_weight, target_hidden_size, target_length, max_packed_bytes, seq_length, draft_train_length, block_size, chunk_cap, window_size, seed, rank, world_size):
    if not samples:
        raise ValueError('online SWA requires all selected committed samples')
    _plan_packed_samples(samples, target_length=target_length, max_packed_bytes=max_packed_bytes, max_samples=len(samples), block_size=block_size)
    first = samples[0][1]
    if any(((item.data_version, item.rl_step, item.draft_revision) != (first.data_version, first.rl_step, first.draft_revision) for _, item in samples)):
        raise ValueError('online samples differ in data, actor step or draft revision')
    rows = plan_packed_rows([item.token_count for _, item in samples], [item.payload_nbytes for _, item in samples], target_length=target_length, block_size=block_size, max_rows=len(samples), max_samples_per_row=len(samples), max_bytes_per_row=max_packed_bytes)
    if rows.sample_count != len(samples) or rows.deferred_samples:
        raise RuntimeError('online row planning deferred a selected key')
    aligned = tuple((length for row in rows.rows for length in row.aligned_lengths))
    columns = {name: [] for name in ('input_ids', 'position_ids', 'loss_mask', 'document_ids')}
    spans, cursor = ([], 0)
    for document, ((fields, metadata), length) in enumerate(zip(samples, aligned)):
        if any((value.device.type != 'cpu' for value in fields.values())):
            raise ValueError('online materialization requires verified CPU captures')
        piece = materialize_dspark_batch(fields, metadata, lm_head_weight=lm_head_weight, device='cpu', target_hidden_size=target_hidden_size, preserve_lm_head_device=True)
        count = metadata.token_count
        for name in ('input_ids', 'position_ids', 'loss_mask'):
            value = piece[name]
            if length > count:
                value = torch.cat((value, value.new_zeros((1, length - count))), dim=1)
            columns[name].append(value)
        docs = torch.full((1, length), document, dtype=torch.int64)
        docs[:, count:] = -1
        columns['document_ids'].append(docs)
        spans.append((cursor, cursor + count, fields['hidden_states']))
        cursor += length
    light = {name: torch.cat(values, dim=1) for name, values in columns.items()}
    light['lm_head_weight'] = lm_head_weight
    parts = dmtp_swa_response_parts(light, seq_length=seq_length, draft_train_length=draft_train_length, block_size=block_size, chunk_cap=chunk_cap, window_size=window_size, seed=seed)
    shard = shard_dmtp_swa_parts(parts, rank=rank, world_size=world_size)
    owned = []
    for part in shard.parts:
        start, end, hidden = spans[part.document_id]
        if not start <= part.context_start < part.context_end <= end:
            raise ValueError('owned online SWA halo crosses a document boundary')
        source = hidden[part.context_start - start:part.context_end - start].clone().unsqueeze(0)
        chunks = source.split(target_hidden_size, dim=-1)
        owned.append(replace(part, batch=dict(part.batch, hidden_states_list=list(chunks[:-1]), target_last_hidden_states=chunks[-1])))
    return (replace(shard, parts=tuple(owned)), rows)
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

class OwnedBuilderCopies:

    def __init__(self, builders: list[Any], *, workers: int=1) -> None:
        if type(workers) is not int or workers not in (1, 2, 4, 8, 16) or (not isinstance(builders, list)) or (not 1 <= len(builders) <= 16) or (len({id(builder) for builder in builders}) != len(builders)) or any((not callable(getattr(builder, 'add', None)) for builder in builders)):
            raise ValueError('invalid independent owned builder copy configuration')
        self._builders = tuple(builders)
        self._lock = Lock()
        self._busy = False
        self._closed = False
        self._poisoned = False
        self._pool = ThreadPoolExecutor(max_workers=min(workers, len(builders))) if workers != 1 else None

    def requires_hidden(self, metadata: Any) -> bool:
        with self._lock:
            if self._closed or self._poisoned or self._busy:
                raise RuntimeError('owned builder copies cannot plan while busy or closed')
            if any((not callable(getattr(b, 'requires_hidden', None)) or not callable(getattr(b, 'add_light_only', None)) for b in self._builders)):
                raise ValueError('selective copies require explicit ownership-aware builders')
            return any((b.requires_hidden(metadata) for b in self._builders))

    def add_selected(self, fields: Any, metadata: Any) -> None:
        self._add(fields, metadata, selected=True)

    def add(self, fields: Any, metadata: Any) -> None:
        route_owned = isinstance(fields, dict) and set(fields) == {'input_ids', 'position_ids', 'loss_mask', 'hidden_states'} and all((callable(getattr(b, 'requires_hidden', None)) and callable(getattr(b, 'add_light_only', None)) for b in self._builders))
        self._add(fields, metadata, selected=route_owned)

    def _add(self, fields: Any, metadata: Any, *, selected: bool) -> None:
        with self._lock:
            if self._closed or self._poisoned or self._busy:
                raise RuntimeError('owned builder copies are closed, poisoned, or already running')
            self._busy = True
        futures = []
        try:
            operations = []
            if selected:
                light_names = {'input_ids', 'position_ids', 'loss_mask'}
                if set(fields) not in (light_names, light_names | {'hidden_states'}):
                    raise ValueError('selective copies require exact capture field names')
                light = {k: fields[k] for k in light_names}
                for builder in self._builders:
                    if not callable(getattr(builder, 'requires_hidden', None)) or not callable(getattr(builder, 'add_light_only', None)):
                        raise ValueError('selective copies require ownership-aware builders')
                    if builder.requires_hidden(metadata):
                        if 'hidden_states' not in fields:
                            raise ValueError('owned hidden fields cannot be omitted')
                        operations.append((builder.add, fields))
                    else:
                        operations.append((builder.add_light_only, light))
            else:
                operations = [(b.add, fields) for b in self._builders]
            if self._pool is None:
                for operation, payload in operations:
                    operation(payload, metadata)
            else:
                for operation, payload in operations:
                    futures.append(self._pool.submit(operation, payload, metadata))
                for future in futures:
                    future.result()
        except BaseException:
            self._poisoned = True
            if futures:
                wait(futures)
            raise
        finally:
            with self._lock:
                self._busy = False

    def close(self) -> None:
        with self._lock:
            if self._busy:
                raise RuntimeError('owned builder copies must join before close')
            if self._closed:
                return
            self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self._builders = ()

def materialize_light_columns(fields, metadata, *, hidden_size):
    metadata.validate()
    if set(fields) != {'input_ids', 'position_ids', 'loss_mask'} or metadata.hidden_states_layout != 'dflash_aux_plus_last' or metadata.hidden_dtype != 'bfloat16' or (type(hidden_size) is not int) or (hidden_size < 1) or (metadata.prompt_token_count is None) or (metadata.payload_nbytes != metadata.token_count * (20 + 2 * hidden_size * (len(metadata.target_layer_ids) + 1))):
        raise ValueError('light columns need the exact complete capture geometry')
    for name, dtype in [('input_ids', torch.int64), ('position_ids', torch.int64), ('loss_mask', torch.float32)]:
        v = fields[name]
        if not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.dtype != dtype or (v.shape != (metadata.token_count,)):
            raise ValueError('light capture field geometry or CPU ownership differs')
    ids, positions, mask = (fields[k] for k in ['input_ids', 'position_ids', 'loss_mask'])
    boundary = metadata.prompt_token_count
    if not torch.equal(positions, torch.arange(metadata.token_count)) or not bool(torch.all(ids >= 0)) or (not bool(torch.all(mask[:boundary] == 0))) or (not bool(torch.all(mask[boundary:] == 1))) or (metadata.supervised_token_count != metadata.token_count - boundary):
        raise ValueError('light capture positions or supervision differ')
    return {k: v.unsqueeze(0) for k, v in fields.items()}
from verl_speco.trainer.distributed_batches import shard_dspark_windows

def materialize_owned_packed_windows(samples, *, lm_head_weight, target_hidden_size, target_length, max_packed_bytes, window_size, rank, world_size, max_samples=16, block_size=2):
    """Validate full captures, plan on small columns, and copy only owned hidden.

    This is a CPU materialization optimization, not transport sharding: callers
    must still verify every complete TQ payload checksum before invoking it.
    Window ordinals, overlaps, padding, absolute positions, document boundaries
    and loss weights match full packing followed by shard_dspark_windows.
    """
    plan, selected = _plan_packed_samples(samples, target_length=target_length, max_packed_bytes=max_packed_bytes, max_samples=max_samples, block_size=block_size)
    pieces = []
    for fields, metadata in selected:
        if any((tensor.device.type != 'cpu' for tensor in fields.values())):
            raise ValueError('owned packed windows require CPU TQ captures')
        pieces.append(materialize_dspark_batch(fields, metadata, lm_head_weight=lm_head_weight, device='cpu', target_hidden_size=target_hidden_size, preserve_lm_head_device=True))
    columns = {name: [] for name in ('input_ids', 'position_ids', 'loss_mask', 'document_ids')}
    spans = []
    offset = 0
    for document, ((fields, metadata), piece, aligned) in enumerate(zip(selected, pieces, plan.aligned_lengths)):
        count = metadata.token_count
        spans.append((offset, offset + count, fields['hidden_states']))
        for name in ('input_ids', 'position_ids', 'loss_mask'):
            value = piece[name]
            if aligned > count:
                value = torch.cat((value, value.new_zeros((1, aligned - count))), dim=1)
            columns[name].append(value)
        doc = torch.full((1, aligned), document, dtype=torch.int64)
        doc[:, count:] = -1
        columns['document_ids'].append(doc)
        offset += aligned
    light_batch = {name: torch.cat(parts, dim=1) for name, parts in columns.items()}
    light_batch['lm_head_weight'] = lm_head_weight
    shard = shard_dspark_windows(light_batch, window_size=window_size, rank=rank, world_size=world_size)
    width = (len(selected[0][1].target_layer_ids) + 1) * target_hidden_size
    windows = []
    for window in shard.windows:
        hidden = torch.zeros((1, window.end - window.start, width), dtype=torch.bfloat16)
        for start, end, source in spans:
            left, right = (max(start, window.start), min(end, window.end))
            if left < right:
                hidden[:, left - window.start:right - window.start].copy_(source[left - start:right - start])
        chunks = hidden.split(target_hidden_size, dim=-1)
        batch = dict(window.batch, hidden_states_list=list(chunks[:-1]), target_last_hidden_states=chunks[-1])
        windows.append(replace(window, batch=batch))
    return (replace(shard, windows=tuple(windows)), light_batch, plan.sample_count)
from dataclasses import dataclass, replace
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key, parse_v3_ready_tag
from verl_speco.trainer.distributed_batches import move_dspark_window_shard

def metadata_checksum(metadatas):
    return hashlib.sha256(json.dumps([m.to_dict() for m in metadatas], sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def window_shard_checksum(shard):
    fields = {}
    rows = []
    for i, window in enumerate(shard.windows):
        rows.append((window.start, window.end, window.eligible_pairs))
        for name, value in window.batch.items():
            if name == 'lm_head_weight':
                continue
            if name == 'hidden_states_list':
                for layer, tensor in enumerate(value):
                    fields[f'{i}/{name}/{layer}'] = tensor
            elif isinstance(value, torch.Tensor):
                fields[f'{i}/{name}'] = value
            else:
                raise ValueError('shared window contains an unsupported field')
    binding = (shard.window_indices, shard.eligible_pairs, shard.global_eligible_pairs, shard.loss_scale, rows, payload_checksum(fields))
    return hashlib.sha256(json.dumps(binding, separators=(',', ':')).encode()).hexdigest()

@dataclass(frozen=True)
class SharedWindowConfig:
    run_id: str
    generation: int
    partition: str
    world_size: int
    window_size: int
    target_hidden_size: int
    target_length: int
    max_packed_bytes: int
    first_sequence_no: int = 0

    def __post_init__(self):
        if not self.run_id or not self.partition or type(self.generation) is not int or (self.generation < 1) or (type(self.world_size) is not int) or (not 1 <= self.world_size <= 16) or (self.window_size < 2) or (self.target_hidden_size < 1) or (self.max_packed_bytes < 1) or (type(self.first_sequence_no) is not int) or (self.first_sequence_no < 0):
            raise ValueError('invalid shared CPU materializer configuration')

class SharedWindowBatchStore:
    """Single ordered owner; original TQ leases remain owned by the trainer.

    Objects are published only after every original payload and rank shard is
    validated. On commit the owner drops its references; consumers retain theirs
    until done, using Ray's reference counting rather than forced deletion.
    """

    def __init__(self, config, *, get_tags, get_fields, put_object, drop_objects):
        self.config = config
        self.get_tags = get_tags
        self.get_fields = get_fields
        self.put_object = put_object
        self.drop_objects = drop_objects
        self.cursor = config.first_sequence_no
        self.stream_identity = None
        self.prepared = None
        self.closed = False
        self.full_payload_read_calls = 0
        self.prepared_batches = 0

    def prepare(self, keys, metadata_dicts):
        if self.closed:
            raise RuntimeError('shared materializer is closed')
        metas = tuple((SampleMetadataV3.from_dict(m) for m in metadata_dicts))
        keys = tuple(keys)
        if not metas or len(keys) != len(metas) or len(set(keys)) != len(keys) or any((make_v3_sample_key(m) != key for key, m in zip(keys, metas))) or any((m.run_id != self.config.run_id or m.generation != self.config.generation for m in metas)):
            raise ValueError('shared batch keys or stream owner differs')
        mdigest = metadata_checksum(metas)
        batch_id = hashlib.sha256('\n'.join(keys).encode()).hexdigest()[:20]
        if self.prepared is not None:
            if self.prepared['keys'] != keys or self.prepared['metadata_sha256'] != mdigest:
                raise RuntimeError('another shared batch is still retained')
            return self._status()
        if metas[0].sequence_no != self.cursor:
            raise ValueError('shared batch does not start at committed cursor')
        first = metas[0]
        stream = (first.target_model_fingerprint, first.tokenizer_fingerprint, first.dataset_fingerprint, first.target_revision, first.data_version, first.target_layer_ids)
        if self.stream_identity is not None and stream != self.stream_identity:
            raise RuntimeError('shared materializer target or dataset identity changed')
        tags = self.get_tags(self.config.partition)
        captures, checksums = ([], [])
        for key, meta in zip(keys, metas):
            self.full_payload_read_calls += 1
            fields = self.get_fields(key, self.config.partition)
            checksum = payload_checksum(fields)
            parsed = parse_v3_ready_tag(tags.get(key) or {})
            if parsed is None or parsed[0] != meta or parsed[1] != checksum:
                raise RuntimeError('shared TQ payload metadata/checksum mismatch')
            captures.append((fields, meta))
            checksums.append(checksum)
        head_shape_probe = torch.empty((0, self.config.target_hidden_size), dtype=torch.bfloat16)
        refs, descriptors = ([], [])
        try:
            for rank in range(self.config.world_size):
                shard, light, count = materialize_owned_packed_windows(captures, lm_head_weight=head_shape_probe, target_hidden_size=self.config.target_hidden_size, target_length=self.config.target_length, max_packed_bytes=self.config.max_packed_bytes, window_size=self.config.window_size, rank=rank, world_size=self.config.world_size)
                if count != len(keys):
                    raise RuntimeError('shared packing deferred a captured sample')
                shard = replace(shard, windows=tuple((replace(w, batch={k: v for k, v in w.batch.items() if k != 'lm_head_weight'}) for w in shard.windows)))
                shard = move_dspark_window_shard(shard, 'cpu')
                ids = light['input_ids'].contiguous()
                descriptor = dict(schema_version=1, batch_id=batch_id, rank=rank, run_id=self.config.run_id, generation=self.config.generation, world_size=self.config.world_size, keys=keys, metadata_sha256=mdigest, capture_checksums=tuple(checksums), shard_sha256=window_shard_checksum(shard), ids_sha256=payload_checksum({'input_ids': ids}), used_tokens=ids.numel())
                refs.append(self.put_object(dict(shard=shard, input_ids=ids)))
                descriptors.append(descriptor)
        except BaseException:
            self.drop_objects(refs)
            raise
        self.stream_identity = stream
        self.prepared_batches += 1
        self.prepared = dict(batch_id=batch_id, keys=keys, metadata_sha256=mdigest, descriptors=descriptors, refs=refs)
        return self._status()

    def _status(self):
        return dict(batch_id=self.prepared['batch_id'], rank_count=len(self.prepared['refs']), metadata_sha256=self.prepared['metadata_sha256'], total_full_payload_read_calls=self.full_payload_read_calls, prepared_batches=self.prepared_batches)

    def get_rank(self, batch_id, rank):
        if self.closed or self.prepared is None or self.prepared['batch_id'] != batch_id or (type(rank) is not int) or (not 0 <= rank < self.config.world_size):
            raise RuntimeError('shared rank object owner or batch differs')
        return dict(descriptor=dict(self.prepared['descriptors'][rank]), object_ref=self.prepared['refs'][rank])

    def release(self, batch_id, *, optimizer_step_id, committed_cursor):
        if self.prepared is None or self.prepared['batch_id'] != batch_id or optimizer_step_id != 'dspark-step-' + batch_id or (committed_cursor != self.cursor + len(self.prepared['keys'])):
            raise RuntimeError('shared batch release does not match committed optimizer step')
        self.drop_objects(self.prepared['refs'])
        self.prepared = None
        self.cursor = committed_cursor
        return dict(committed_cursor=self.cursor, retained_batches=0)

    def close(self):
        if self.prepared is not None:
            self.drop_objects(self.prepared['refs'])
            self.prepared = None
        self.closed = True

class RaySharedWindowMaterializer:
    """A trainer-node-local Ray owner; no accelerator tensors or hidden files."""

    def __init__(self, config):
        import ray
        from verl_speco.runtime.services import import_transfer_queue
        self.tq = import_transfer_queue()
        self.tq.init()
        self.store = SharedWindowBatchStore(SharedWindowConfig(**config), get_tags=lambda partition: self.tq.kv_list(partition_id=partition).get(partition, {}), get_fields=self._get_fields, put_object=ray.put, drop_objects=lambda refs: refs.clear())

    def _get_fields(self, key, partition):
        values = self.tq.kv_batch_get(keys=key, partition_id=partition)
        return {name: rows[0] for name, rows in values.items()}

    def prepare(self, keys, metadatas):
        import os
        return dict(self.store.prepare(keys, metadatas), reader_pid=os.getpid())

    def get_rank(self, batch_id, rank):
        return self.store.get_rank(batch_id, rank)

    def release(self, batch_id, optimizer_step_id, committed_cursor):
        return self.store.release(batch_id, optimizer_step_id=optimizer_step_id, committed_cursor=committed_cursor)

    def close(self):
        from verl_speco.runtime.services import close_external_tq_client
        self.store.close()
        close_external_tq_client(self.tq.get_client())
        return dict(retained_batches=0, client_closed=True)

def decode_shared_rank_packet(packet, descriptor, *, config, keys, metadatas, rank, lm_head_weight):
    """Verify the received owned bytes before attaching the real resident head."""
    expected = dict(schema_version=1, batch_id=hashlib.sha256('\n'.join(keys).encode()).hexdigest()[:20], rank=rank, run_id=config.run_id, generation=config.generation, world_size=config.world_size, keys=tuple(keys), metadata_sha256=metadata_checksum(metadatas))
    if not isinstance(descriptor, dict) or set(descriptor) != set(expected) | {'capture_checksums', 'shard_sha256', 'ids_sha256', 'used_tokens'} or any((descriptor[k] != v or type(descriptor[k]) is not type(v) for k, v in expected.items())) or (not isinstance(packet, dict)) or (set(packet) != {'shard', 'input_ids'}):
        raise ValueError('shared rank descriptor or packet owner differs')
    ids, shard = (packet['input_ids'], packet['shard'])
    if type(descriptor['used_tokens']) is not int or not 0 < descriptor['used_tokens'] <= config.target_length or ids.device.type != 'cpu' or (ids.dtype != torch.int64) or (ids.shape != (1, descriptor['used_tokens'])) or (payload_checksum({'input_ids': ids}) != descriptor['ids_sha256']) or (window_shard_checksum(shard) != descriptor['shard_sha256']):
        raise ValueError('shared rank object checksum or token count differs')
    digests = descriptor['capture_checksums']
    if type(digests) is not tuple or len(digests) != len(keys) or any((type(d) is not str or len(d) != 64 or any((c not in '0123456789abcdef' for c in d)) for d in digests)) or (lm_head_weight.dtype != torch.bfloat16) or (lm_head_weight.ndim != 2) or (lm_head_weight.shape[1] != config.target_hidden_size):
        raise ValueError('shared capture checksums or resident target head differ')
    if not shard.windows or shard.global_eligible_pairs <= 0 or shard.eligible_pairs < 0 or (shard.loss_scale != config.world_size * shard.eligible_pairs / shard.global_eligible_pairs) or (tuple(sorted(set(shard.window_indices))) != shard.window_indices) or any((i % config.world_size != rank for i in shard.window_indices)) or (bool(shard.window_indices) and (len(shard.windows) != len(shard.window_indices) or sum((w.eligible_pairs for w in shard.windows)) != shard.eligible_pairs)) or (not shard.window_indices and (shard.eligible_pairs != 0 or len(shard.windows) != 1)):
        raise ValueError('shared rank window ownership or loss scale differs')
    windows = []
    for window in shard.windows:
        if not 0 <= window.start < window.end <= ids.numel() or not torch.equal(window.batch['input_ids'], ids[:, window.start:window.end]):
            raise ValueError('shared rank window tokens differ from complete packed input')
        windows.append(replace(window, batch=dict(window.batch, lm_head_weight=lm_head_weight)))
    return (replace(shard, windows=tuple(windows)), {'input_ids': ids}, list(digests))
from dataclasses import asdict

def shared_materializer_name(config):
    return f'SpecoSharedWindowMaterializer:{config.run_id}:g{config.generation}'

def create_local_shared_materializer(config):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ip = ray.util.get_node_ip_address()
    nodes = [n for n in ray.nodes() if n['Alive'] and n['NodeManagerAddress'] == ip]
    if len(nodes) != 1:
        raise RuntimeError('shared materializer requires exactly one trainer-local Ray node')
    name = shared_materializer_name(config)
    try:
        ray.get_actor(name, namespace='transfer_queue')
    except ValueError:
        pass
    else:
        raise RuntimeError('shared materializer already exists for this run/generation')
    repo = Path(__file__).resolve().parents[2]
    return ray.remote(RaySharedWindowMaterializer).options(name=name, namespace='transfer_queue', num_cpus=1, max_restarts=0, max_task_retries=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nodes[0]['NodeID'], soft=False), runtime_env={'worker_process_setup_hook': 'verl_speco.runtime.services.install_simple_storage_import_guard', 'env_vars': {'SPECO_TQ_STORAGE_BACKEND': 'SimpleStorage', 'SPECO_RUN_ID': config.run_id, 'SPECO_GENERATION': str(config.generation), 'SPECO_PROCESS_ROLE': 'shared-window-materializer', 'TORCH_DEVICE_BACKEND_AUTOLOAD': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'PYTHONPATH': str(repo) + os.pathsep + os.environ.get('PYTHONPATH', '')}}).remote(asdict(config))

def get_shared_rank(actor, batch_id, *, config, keys, metadatas, rank, lm_head_weight):
    import ray
    reply = ray.get(actor.get_rank.remote(batch_id, rank))
    packet = ray.get(reply['object_ref'])
    return decode_shared_rank_packet(packet, reply['descriptor'], config=config, keys=keys, metadatas=metadatas, rank=rank, lm_head_weight=lm_head_weight)
from verl_speco.trainer.distributed_batches import plan_response_starts_with_fill
from verl_speco.trainer.distributed_batches import enumerate_packed_response_parts
from verl_speco.trainer.distributed_batches import DMTPResponsePart, DMTPPartShard, dmtp_swa_windows_slice_batch as _slice_batch

class StreamingOwnedSWAParts:
    """Plan all parts from CPU metadata before receiving full teacher captures.

    SWA halo never crosses a document, so captures need no retained hidden tail.
    Global slot planning and seed are shared across DP ranks. Short/unknown
    responses are rejected so committed original tokens cannot include a sample
    that had no complete draft block. The caller still verifies each checksum.
    """

    def __init__(self, metadatas, *, target_hidden_size, target_length, max_packed_bytes, rank, world_size, draft_train_length=2048, block_size=8, window_size=2048, chunk_cap=32, seed=42, max_samples=16, owned_hidden_cloner=None):
        if type(rank) is not int or type(world_size) is not int or world_size < 1 or (not 0 <= rank < world_size) or (type(target_hidden_size) is not int) or (target_hidden_size < 1) or (type(window_size) is not int) or (window_size < 1) or (type(seed) is not int) or (not 0 <= seed < 2 ** 63) or (type(block_size) is not int) or (block_size < 1) or (type(draft_train_length) is not int) or (draft_train_length < block_size) or (draft_train_length % block_size):
            raise ValueError('invalid streaming SWA mesh, context or block-slot budget')
        if owned_hidden_cloner is not None and (not callable(owned_hidden_cloner)):
            raise ValueError('owned hidden cloner must be callable')
        self._owned_hidden_cloner = owned_hidden_cloner
        self.metadatas = tuple(metadatas)
        self.plan, _ = _plan_packed_samples(tuple((({}, m) for m in self.metadatas)), target_length=target_length, max_packed_bytes=max_packed_bytes, max_samples=max_samples, block_size=block_size)
        if self.plan.sample_count != len(self.metadatas):
            raise ValueError('streaming SWA packing would defer a captured sample')
        for m in self.metadatas:
            m.validate()
            if m.prompt_token_count is None or m.prompt_token_count < 1 or m.supervised_token_count < block_size:
                raise ValueError('streaming SWA needs a prompt predecessor and complete response block')
        self.hidden_size = target_hidden_size
        self.rank, self.world_size = (rank, world_size)
        self.window_size = window_size
        self.block_size = block_size
        lengths = torch.tensor([m.supervised_token_count for m in self.metadatas], device='cpu', dtype=torch.int64)
        starts, ends, samples = enumerate_packed_response_parts(response_length=lengths, seq_length=target_length, draft_train_length=draft_train_length, block_size=block_size, num_swa_chunk=chunk_cap, adaptive_num_swa_chunk=True)
        if not samples.numel():
            raise ValueError('streaming SWA has no feasible packed part budget')
        self.labels = plan_response_starts_with_fill(region_starts=starts, region_ends=ends, k_blocks=draft_train_length // block_size, block_size=block_size, generator=torch.Generator(device='cpu').manual_seed(seed))
        self._parts_plan = tuple(zip(starts.tolist(), ends.tolist(), samples.tolist()))
        self._indices = tuple(range(rank, len(self._parts_plan), world_size))
        self._owned = {}
        self._dummy = None
        self._stub_head = torch.empty(1, target_hidden_size, dtype=torch.bfloat16, device='cpu')
        self._columns = {name: [] for name in ('input_ids', 'position_ids', 'loss_mask', 'document_ids')}
        self._received = self._offset = 0
        self._finished = False

    @property
    def global_part_count(self):
        return len(self._parts_plan)

    @property
    def retained_tail_rows(self):
        return 0

    @property
    def planned_owned_context_rows(self):
        """Metadata-only halo cost; excludes empty-rank dummy forwards."""
        total = 0
        for ordinal in self._indices:
            lo, hi, sample = self._parts_plan[ordinal]
            prompt = self.metadatas[sample].prompt_token_count
            total += prompt + hi - max(0, prompt + lo - self.window_size - 1)
        return total

    @property
    def planned_owned_part_count(self):
        return len(self._indices)

    @property
    def planned_owned_anchor_count(self):
        """Valid sampled blocks; excludes unavailable slots and dummy forwards."""
        return int((self.labels[list(self._indices)] >= 0).sum().item())

    @property
    def planned_owned_label_slots(self):
        """Sampled training labels, including overlap; never original-token credit."""
        return self.planned_owned_anchor_count * self.block_size

    @property
    def owned_feature_rows(self):
        return sum((p.context_end - p.context_start for p in self._owned.values()))

    def add_light_only(self, fields, metadata):
        if self._finished or self._received >= len(self.metadatas) or metadata != self.metadatas[self._received]:
            raise ValueError('light capture is out of order or already finished')
        if any((self._parts_plan[i][2] == self._received for i in self._indices)):
            raise ValueError('owned hidden parts cannot be replaced by light columns')
        piece = materialize_light_columns(fields, metadata, hidden_size=self.hidden_size)
        piece['document_ids'] = torch.full_like(piece['input_ids'], self._received)
        self._append_light_columns(piece, metadata)

    def _append_light_columns(self, piece, metadata):
        aligned = self.plan.aligned_lengths[self._received]
        for name in self._columns:
            value = piece[name].clone()
            if aligned > metadata.token_count:
                pad = value.new_full((1, aligned - metadata.token_count), -1 if name == 'document_ids' else 0)
                value = torch.cat((value, pad), dim=1)
            self._columns[name].append(value)
        self._offset += aligned
        self._received += 1

    def add(self, fields, metadata):
        if self._finished or self._received >= len(self.metadatas) or metadata != self.metadatas[self._received]:
            raise ValueError('streaming SWA capture is out of order or already finished')
        if any((t.device.type != 'cpu' for t in fields.values())):
            raise ValueError('streaming SWA requires verified CPU captures')
        piece = materialize_dspark_batch(fields, metadata, lm_head_weight=self._stub_head, device='cpu', target_hidden_size=self.hidden_size, preserve_lm_head_device=True)
        piece['document_ids'] = torch.full_like(piece['input_ids'], self._received)
        prepared = {}
        dummy = None
        for ordinal, (lo, hi, sample) in enumerate(self._parts_plan):
            if sample != self._received or (ordinal not in self._indices and (not (ordinal == 0 and (not self._indices)))):
                continue
            start, end = (metadata.prompt_token_count + lo, metadata.prompt_token_count + hi)
            left = max(0, start - self.window_size - 1)
            batch = _slice_batch(piece, left, end) if self._owned_hidden_cloner is None else streaming_owned_swa_parts_slice_with_owned_hidden(piece, left, end, ordinal=ordinal, clone_hidden=self._owned_hidden_cloner)
            batch['loss_mask'][:, :start - left] = 0
            keep = self.labels[ordinal:ordinal + 1] >= 0
            positions = self.labels[ordinal:ordinal + 1] + metadata.prompt_token_count - 1 - left
            batch['preplanned_anchor_positions'] = torch.where(keep, positions, 0)
            batch['preplanned_anchor_keep_mask'] = keep
            part = DMTPResponsePart(sample, self._offset + start, self._offset + end, self._offset + left, self._offset + end, batch)
            if ordinal in self._indices:
                prepared[ordinal] = part
            else:
                seed = int(positions[0, 0])
                db = _slice_batch({k: v for k, v in batch.items() if not k.startswith('preplanned_')}, seed, seed + 1)
                db['loss_mask'].zero_()
                db['preplanned_anchor_positions'] = torch.zeros_like(positions)
                db['preplanned_anchor_keep_mask'] = torch.zeros_like(keep)
                dummy = DMTPResponsePart(sample, part.response_start, part.response_start, part.context_start + seed, part.context_start + seed + 1, db)
        self._owned.update(prepared)
        if dummy is not None:
            self._dummy = dummy
        self._append_light_columns(piece, metadata)

    def finish(self, lm_head_weight, *, allow_empty_parts=False):
        if type(allow_empty_parts) is not bool:
            raise ValueError('empty local row policy must be boolean')
        if self._finished or self._received != len(self.metadatas):
            raise ValueError('streaming SWA captures are incomplete or already finished')
        if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != self.hidden_size or lm_head_weight.dtype != torch.bfloat16:
            raise ValueError('target LM head differs from streaming SWA hidden width/dtype')
        if set(self._owned) != set(self._indices):
            raise RuntimeError('streaming SWA owned parts are incomplete')
        selected = tuple((self._owned[i] for i in self._indices))
        if not selected and (not allow_empty_parts):
            if self._dummy is None:
                raise RuntimeError('streaming SWA empty-rank forward missing')
            selected = (self._dummy,)
        selected = tuple((replace(p, batch=dict(p.batch, lm_head_weight=lm_head_weight)) for p in selected))
        light = {name: torch.cat(parts, dim=1) for name, parts in self._columns.items()}
        light['lm_head_weight'] = lm_head_weight
        self._finished = True
        return (DMTPPartShard(selected, self._indices, len(self._parts_plan), self.world_size / len(self._parts_plan)), light, self.plan.sample_count)

def streaming_owned_swa_parts_slice_with_owned_hidden(batch, start, end, *, ordinal, clone_hidden):
    shape = batch['input_ids'].shape
    result = {}
    for name, value in batch.items():
        if name == 'lm_head_weight':
            result[name] = value
        elif name == 'hidden_states_list':
            result[name] = [clone_hidden(ordinal, f'hidden_states_list/{layer}', t[:, start:end]) for layer, t in enumerate(value)]
        elif name == 'target_last_hidden_states':
            result[name] = clone_hidden(ordinal, name, value[:, start:end])
        elif isinstance(value, torch.Tensor) and value.ndim >= 2 and (value.shape[:2] == shape):
            result[name] = value[:, start:end].clone()
        else:
            raise ValueError('unsupported field in direct SWA batch: ' + name)
    return result
from verl_speco.trainer.distributed_batches import DMTPPartShard

class StreamingOwnedSWARows:
    """Preserve each row's DMTP chunk budget and average over all real parts.

    The global ordinal assignment is rank::world_size, even when row part
    counts differ. Documents and context bounds retain globally unique indices.
    This CPU materializer does not itself implement native wave scheduling.
    """

    def __init__(self, metadatas, *, target_hidden_size, target_length, max_bytes_per_row, rank, world_size, max_rows, max_samples_per_row=32, draft_train_length=2048, block_size=8, window_size=2048, chunk_cap=32, seed=42, owned_hidden_cloner_factory=None):
        if type(world_size) is not int or not 1 <= world_size <= 128 or type(rank) is not int or (not 0 <= rank < world_size):
            raise ValueError('invalid multirow SWA rank or world size')
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError('multirow SWA seed must be a nonnegative signed 64-bit integer')
        if owned_hidden_cloner_factory is not None and (not callable(owned_hidden_cloner_factory)):
            raise ValueError('hidden cloner factory must be callable')
        metadatas = tuple(metadatas)
        if not metadatas:
            raise ValueError('multirow SWA needs a nonempty stream prefix')
        first = metadatas[0]
        for i, m in enumerate(metadatas):
            m.validate()
            if m.sequence_no != first.sequence_no + i or any((getattr(m, k) != getattr(first, k) for k in ('run_id', 'generation', 'target_revision', 'target_model_fingerprint', 'tokenizer_fingerprint', 'target_layer_ids', 'dataset_fingerprint'))):
                raise ValueError('multirow SWA needs one contiguous stream and target identity')
        self.row_plan = plan_packed_rows([m.token_count for m in metadatas], [m.payload_nbytes for m in metadatas], target_length=target_length, block_size=block_size, max_rows=max_rows, max_samples_per_row=max_samples_per_row, max_bytes_per_row=max_bytes_per_row)
        self.metadatas = metadatas[:self.row_plan.sample_count]
        self.rank, self.world_size = (rank, world_size)
        self._rows = []
        self._part_offsets = []
        self._global_parts = 0
        for row, indices in enumerate(self.row_plan.sample_indices):
            local_rank = (rank - self._global_parts) % world_size
            self._part_offsets.append(self._global_parts)
            builder = StreamingOwnedSWAParts([metadatas[i] for i in indices], target_hidden_size=target_hidden_size, target_length=target_length, max_packed_bytes=max_bytes_per_row, rank=local_rank, world_size=world_size, max_samples=max_samples_per_row, draft_train_length=draft_train_length, block_size=block_size, window_size=window_size, chunk_cap=chunk_cap, seed=(seed + row) % 2 ** 63, owned_hidden_cloner=owned_hidden_cloner_factory(self._global_parts) if owned_hidden_cloner_factory is not None else None)
            self._global_parts += builder.global_part_count
            self._rows.append(builder)
        self._row_for_sample = tuple((row for row, indices in enumerate(self.row_plan.sample_indices) for _ in indices))
        self._received = 0
        self._finished = False

    @property
    def global_part_count(self):
        return self._global_parts

    @property
    def planned_frame_upper_bound_bytes(self):
        """Bound one rank's sealed CPU frame before receiving teacher features.

        Includes copied SWA halos, duplicated light columns, anchors and worst
        case 64-byte tensor alignment. Resident NPU head weights stay outside.
        """
        if any((m.hidden_dtype != 'bfloat16' or m.hidden_states_layout != 'dflash_aux_plus_last' for m in self.metadatas)):
            raise ValueError('SWA frame estimate requires BF16 DSpark features')
        count = sum((row.planned_owned_part_count for row in self._rows))
        context = sum((row.planned_owned_context_rows for row in self._rows))
        if not count:
            count = context = 1
        layers = len(self.metadatas[0].target_layer_ids)
        width = self._rows[0].hidden_size
        anchors = self._rows[0].labels.shape[-1]
        data = context * ((layers + 1) * width * 2 + 28) + count * anchors * 9 + self.row_plan.padded_tokens * 28
        fields = count * (layers + 7) + 4
        return data + fields * 63

    @property
    def planned_owned_part_count(self):
        return sum((row.planned_owned_part_count for row in self._rows))

    @property
    def planned_owned_anchor_count(self):
        return sum((row.planned_owned_anchor_count for row in self._rows))

    @property
    def planned_owned_label_slots(self):
        """Valid block slots across rows; excludes padding and empty-rank dummies."""
        return sum((row.planned_owned_label_slots for row in self._rows))

    @property
    def owned_feature_rows(self):
        return sum((row.owned_feature_rows for row in self._rows))

    @property
    def retained_tail_rows(self):
        return 0

    def requires_hidden(self, metadata):
        index = metadata.sequence_no - self.metadatas[0].sequence_no
        if not 0 <= index < len(self.metadatas) or metadata != self.metadatas[index]:
            raise ValueError('hidden ownership needs exact planned capture metadata')
        if not self.planned_owned_part_count:
            return True
        row = self._row_for_sample[index]
        local = self._rows[row]
        sample = index - self.row_plan.sample_indices[row][0]
        return any((local._parts_plan[i][2] == sample for i in local._indices))

    def add_light_only(self, fields, metadata):
        if self._finished or self._received >= len(self.metadatas) or metadata != self.metadatas[self._received]:
            raise ValueError('multirow light capture is out of order or finished')
        if self.requires_hidden(metadata):
            raise ValueError('owned or global dummy hidden requires verified full capture')
        self._rows[self._row_for_sample[self._received]].add_light_only(fields, metadata)
        self._received += 1

    def add(self, fields, metadata):
        if self._finished or self._received >= len(self.metadatas) or metadata != self.metadatas[self._received]:
            raise ValueError('multirow SWA capture is out of order or already finished')
        self._rows[self._row_for_sample[self._received]].add(fields, metadata)
        self._received += 1

    def finish(self, lm_head_weight):
        if self._finished or self._received != len(self.metadatas):
            raise ValueError('multirow SWA captures are incomplete or already finished')
        parts = []
        indices = []
        columns = {k: [] for k in ('input_ids', 'position_ids', 'loss_mask', 'document_ids')}
        token_offset = 0
        dummy = None
        for row, builder in enumerate(self._rows):
            shard, light, count = builder.finish(lm_head_weight, allow_empty_parts=self.planned_owned_part_count > 0)
            documents = self.row_plan.sample_indices[row]
            assert count == len(documents)
            doc_offset = documents[0]
            for ordinal, part in enumerate(shard.parts):
                batch = dict(part.batch)
                docs = batch['document_ids']
                batch['document_ids'] = torch.where(docs >= 0, docs + doc_offset, docs)
                adjusted = replace(part, document_id=part.document_id + doc_offset, response_start=part.response_start + token_offset, response_end=part.response_end + token_offset, context_start=part.context_start + token_offset, context_end=part.context_end + token_offset, batch=batch)
                if shard.part_indices:
                    parts.append(adjusted)
                    indices.append(self._part_offsets[row] + shard.part_indices[ordinal])
                elif dummy is None:
                    dummy = adjusted
            for name in columns:
                value = light[name]
                if name == 'document_ids':
                    value = torch.where(value >= 0, value + doc_offset, value)
                columns[name].append(value)
            token_offset += self.row_plan.rows[row].used_tokens
        expected = tuple(range(self.rank, self._global_parts, self.world_size))
        if tuple(indices) != expected or token_offset != self.row_plan.padded_tokens:
            raise RuntimeError('multirow SWA ownership or token accounting differs from row plan')
        if not parts:
            if dummy is None:
                raise RuntimeError('multirow empty-rank forward is missing')
            parts = [dummy]
        batch = {k: torch.cat(v, dim=1) for k, v in columns.items()}
        batch['lm_head_weight'] = lm_head_weight
        self._finished = True
        return (DMTPPartShard(tuple(parts), expected, self._global_parts, self.world_size / self._global_parts), batch, len(self.metadatas))

class StreamingOwnedWindows:
    """Consume complete checksum-verified CPU payloads in stream order.

    Integrity verification belongs to the caller. This changes CPU retention,
    not transport or checksum semantics. No full teacher payload survives add().
    Global window ordinals, document gaps, overlap and loss weights match the
    existing complete-packing reference, including zero-weight empty ranks.
    """

    def __init__(self, metadatas, *, target_hidden_size, target_length, max_packed_bytes, window_size, rank, world_size, max_samples=16, block_size=2):
        if type(rank) is not int or type(world_size) is not int or world_size < 1 or (not 0 <= rank < world_size) or (type(window_size) is not int) or (window_size < 2) or (type(target_hidden_size) is not int) or (target_hidden_size < 1):
            raise ValueError('invalid streaming window mesh or dimensions')
        self.metadatas = tuple(metadatas)
        self.plan, _ = _plan_packed_samples(tuple((({}, metadata) for metadata in self.metadatas)), target_length=target_length, max_packed_bytes=max_packed_bytes, max_samples=max_samples, block_size=block_size)
        if self.plan.sample_count != len(self.metadatas):
            raise ValueError('streaming packing would defer a captured sample')
        for metadata in self.metadatas:
            metadata.validate()
        self.hidden_size = target_hidden_size
        self.width = (len(self.metadatas[0].target_layer_ids) + 1) * target_hidden_size
        self.window_size, self.rank, self.world_size = (window_size, rank, world_size)
        self._stub_head = torch.empty((1, target_hidden_size), dtype=torch.bfloat16)
        self._columns = {name: [] for name in ('input_ids', 'position_ids', 'loss_mask', 'document_ids')}
        self._spans = []
        self._owned = {}
        self._dummy = None
        self._owned_indices = []
        self._next_start = self._length = self._received = self._ordinal = self._pairs = 0
        self._finished = False

    @property
    def retained_tail_rows(self):
        return sum((end - start for start, end, _ in self._spans))

    @property
    def owned_feature_rows(self):
        return sum((end - start for start, end in self._owned))

    def _light(self, head):
        return {**{name: torch.cat(parts, dim=1) for name, parts in self._columns.items()}, 'lm_head_weight': head}

    def _hidden(self, start, end, light):
        hidden = torch.zeros((1, end - start, self.width), dtype=torch.bfloat16)
        covered = torch.zeros(end - start, dtype=torch.bool)
        for left_edge, right_edge, source in self._spans:
            left, right = (max(start, left_edge), min(end, right_edge))
            if left < right:
                hidden[:, left - start:right - start].copy_(source[left - left_edge:right - left_edge])
                covered[left - start:right - start] = True
        if not torch.equal(covered, light['document_ids'][0, start:end] >= 0):
            raise RuntimeError('streaming hidden tail does not cover the real document rows')
        return hidden

    def _emit(self, start, end, light):
        mask, docs = (light['loss_mask'], light['document_ids'])
        active = (mask[:, start:end - 1] > 0.5) & (mask[:, start + 1:end] > 0.5) & (docs[:, start:end - 1] == docs[:, start + 1:end]) & (docs[:, start:end - 1] >= 0)
        count = int(active.sum().item())
        if not count:
            return
        if self._dummy is None:
            first = start + int(torch.nonzero(active[0], as_tuple=False)[0].item())
            self._dummy = ((first, first + 2), self._hidden(first, first + 2, light))
        if self._ordinal % self.world_size == self.rank:
            self._owned[start, end] = self._hidden(start, end, light)
            self._owned_indices.append(self._ordinal)
        self._ordinal += 1
        self._pairs += count

    def add(self, fields, metadata):
        if self._finished or self._received >= len(self.metadatas) or metadata != self.metadatas[self._received]:
            raise ValueError('streaming capture is out of order or already finished')
        if any((tensor.device.type != 'cpu' for tensor in fields.values())):
            raise ValueError('streaming windows require verified CPU captures')
        piece = materialize_dspark_batch(fields, metadata, lm_head_weight=self._stub_head, device='cpu', target_hidden_size=self.hidden_size, preserve_lm_head_device=True)
        aligned = self.plan.aligned_lengths[self._received]
        for name in ('input_ids', 'position_ids', 'loss_mask'):
            value = piece[name].clone()
            if aligned > metadata.token_count:
                value = torch.cat((value, value.new_zeros((1, aligned - metadata.token_count))), dim=1)
            self._columns[name].append(value)
        docs = torch.full((1, aligned), self._received, dtype=torch.int64)
        docs[:, metadata.token_count:] = -1
        self._columns['document_ids'].append(docs)
        self._spans.append((self._length, self._length + metadata.token_count, fields['hidden_states']))
        self._length += aligned
        light = self._light(self._stub_head)
        while self._next_start + self.window_size <= self._length:
            self._emit(self._next_start, self._next_start + self.window_size, light)
            self._next_start += self.window_size - 1
        spans = []
        for start, end, source in self._spans:
            left = max(start, self._next_start)
            if left < end:
                spans.append((left, end, source[left - start:].clone()))
        self._spans = spans
        self._received += 1
        if self.retained_tail_rows > self.window_size - 1:
            raise RuntimeError('streaming tail exceeded the window bound')

    def finish(self, lm_head_weight):
        if self._finished or self._received != len(self.metadatas):
            raise ValueError('streaming captures are incomplete or already finished')
        if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != self.hidden_size or lm_head_weight.dtype != torch.bfloat16:
            raise ValueError('target LM head differs from streaming hidden width/dtype')
        light = self._light(lm_head_weight)
        while self._next_start < self._length - 1:
            self._emit(self._next_start, min(self._next_start + self.window_size, self._length), light)
            self._next_start += self.window_size - 1
        shard = shard_dspark_windows(light, window_size=self.window_size, rank=self.rank, world_size=self.world_size)
        if shard.window_indices != tuple(self._owned_indices) or shard.global_eligible_pairs != self._pairs:
            raise RuntimeError('streaming window ownership differs from complete packing')
        windows = []
        for window in shard.windows:
            key = (window.start, window.end)
            hidden = self._owned.get(key)
            if hidden is None and (not shard.window_indices) and (self._dummy is not None):
                if key == self._dummy[0]:
                    hidden = self._dummy[1]
            if hidden is None:
                raise RuntimeError('streaming owned window features are missing')
            chunks = hidden.split(self.hidden_size, dim=-1)
            batch = dict(window.batch, hidden_states_list=list(chunks[:-1]), target_last_hidden_states=chunks[-1])
            windows.append(replace(window, batch=batch))
        self._spans.clear()
        self._finished = True
        return (replace(shard, windows=tuple(windows)), light, self.plan.sample_count)
