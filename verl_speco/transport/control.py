# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import re
import threading
import time
from typing import Mapping
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key

@dataclass(frozen=True)
class ReadyRecord:
    key: str
    producer_endpoint: str
    metadata: SampleMetadataV3
    payload_checksum: str

    def validate(self) -> None:
        self.metadata.validate()
        if self.key != make_v3_sample_key(self.metadata):
            raise ValueError('ReadyRecord key does not match metadata')
        if not self.producer_endpoint:
            raise ValueError('ReadyRecord producer_endpoint must not be empty')
        if re.fullmatch('[0-9a-f]{64}', self.payload_checksum) is None:
            raise ValueError('ReadyRecord payload_checksum must be SHA-256')

@dataclass(frozen=True)
class AcquireRequest:
    acquire_id: str
    consumer_id: str
    generation: int
    max_items: int
    min_items: int
    required_target_revision: str
    deadline_ts: float

    def validate(self) -> None:
        if not self.acquire_id or not self.consumer_id:
            raise ValueError('acquire_id and consumer_id must not be empty')
        if self.generation < 0:
            raise ValueError('generation must be non-negative')
        if self.max_items <= 0 or self.min_items <= 0 or self.min_items > self.max_items:
            raise ValueError('acquire min_items/max_items are invalid')
        if not self.required_target_revision:
            raise ValueError('required_target_revision must not be empty')

@dataclass(frozen=True)
class AcquireResult:
    status: str
    acquire_id: str
    records: tuple[ReadyRecord, ...] = ()
    message: str = ''
    partial: bool = False

class ReadyRouter:
    """Fairly route producer-published ready records without owning payload state."""

    def __init__(self, *, acquire_cache_size: int=2048) -> None:
        if acquire_cache_size <= 0:
            raise ValueError('acquire_cache_size must be positive')
        self._lock = threading.RLock()
        self._acquire_cache_size = acquire_cache_size
        self._ready_by_producer: OrderedDict[str, deque[str]] = OrderedDict()
        self._records: dict[str, ReadyRecord] = {}
        self._delivered: OrderedDict[str, ReadyRecord] = OrderedDict()
        self._acquires: OrderedDict[str, tuple[str, AcquireResult]] = OrderedDict()
        self._terminal_acquires: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._cursor = 0

    def publish(self, record: ReadyRecord) -> str:
        """Publish one idempotent ready record and return published/duplicate/conflict."""
        record.validate()
        with self._lock:
            existing = self._records.get(record.key)
            if existing is not None:
                if existing == record:
                    return 'duplicate'
                return 'conflict'
            delivered_record = self._delivered.get(record.key)
            if delivered_record is not None:
                self._delivered.move_to_end(record.key)
                return 'duplicate' if delivered_record == record else 'conflict'
            queue = self._ready_by_producer.setdefault(record.producer_endpoint, deque())
            queue.append(record.key)
            self._records[record.key] = record
            return 'published'

    def lookup(self, record: ReadyRecord) -> str:
        """Inspect key identity without publishing an unmaterialized payload."""
        record.validate()
        with self._lock:
            existing = self._records.get(record.key)
            if existing is not None:
                return 'ready_duplicate' if existing == record else 'conflict'
            delivered = self._delivered.get(record.key)
            if delivered is not None:
                return 'delivered_duplicate' if delivered == record else 'conflict'
            return 'absent'

    def acquire(self, request: AcquireRequest, *, required_keys=None) -> AcquireResult:
        """Return a stable result for each acquire ID until ack or cancellation."""
        request.validate()
        from verl_speco.transport.waves import required_wave_keys
        required_keys = required_wave_keys(request, required_keys)
        fingerprint = ready_router_acquire_fingerprint(request)
        if required_keys is not None:
            fingerprint = hashlib.sha256(json.dumps(dict(request=fingerprint, required_keys=required_keys), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        allowed_keys = set(required_keys) if required_keys is not None else None
        with self._lock:
            terminal = self._terminal_acquires.get(request.acquire_id)
            if terminal is not None:
                if terminal[0] != fingerprint:
                    return AcquireResult(status='conflict', acquire_id=request.acquire_id, message='terminal acquire_id was reused with a different request')
                self._terminal_acquires.move_to_end(request.acquire_id)
                return AcquireResult(status=terminal[1], acquire_id=request.acquire_id, message='acquire already reached a terminal state')
            cached = self._acquires.get(request.acquire_id)
            if cached is not None:
                if cached[0] != fingerprint:
                    return AcquireResult(status='conflict', acquire_id=request.acquire_id, message='acquire_id was reused with a different request')
                self._acquires.move_to_end(request.acquire_id)
                return cached[1]
            if request.deadline_ts <= time.time():
                result = AcquireResult(status='expired', acquire_id=request.acquire_id, message='acquire deadline passed')
                self._remember_acquire(request.acquire_id, fingerprint, result)
                return result
            selected: list[ReadyRecord] = []
            producer_names = tuple(self._ready_by_producer)
            while producer_names and len(selected) < request.max_items:
                found = False
                for offset in range(len(producer_names)):
                    producer = producer_names[(self._cursor + offset) % len(producer_names)]
                    record = self._pop_matching(producer, generation=request.generation, target_revision=request.required_target_revision, allowed_keys=allowed_keys)
                    if record is None:
                        continue
                    selected.append(record)
                    self._cursor = (self._cursor + offset + 1) % len(producer_names)
                    found = True
                    break
                if not found:
                    break
            if len(selected) < request.min_items:
                self._requeue_front(selected)
                result = AcquireResult(status='busy', acquire_id=request.acquire_id, message='insufficient matching ready records')
            else:
                result = AcquireResult(status='ready', acquire_id=request.acquire_id, records=tuple(selected), partial=len(selected) < request.max_items)
            if result.status == 'ready':
                self._remember_acquire(request.acquire_id, fingerprint, result)
            return result

    def ack(self, acquire_id: str) -> None:
        """Finalize router delivery after LeaseRegistry registered the records."""
        with self._lock:
            cached = self._acquires.get(acquire_id)
            if cached is None:
                terminal = self._terminal_acquires.get(acquire_id)
                if terminal is not None and terminal[1] == 'acknowledged':
                    return
                raise ValueError(f'unknown acquire_id {acquire_id!r}')
            result = cached[1]
            if result.status == 'ready':
                for record in result.records:
                    self._records.pop(record.key, None)
                    self._remember_delivered(record)
            self._acquires.pop(acquire_id, None)
            self._remember_terminal(acquire_id, cached[0], 'acknowledged')

    def cancel(self, acquire_id: str) -> None:
        """Return unregistered records to the front of their producer queues."""
        with self._lock:
            cached = self._acquires.pop(acquire_id, None)
            if cached is None:
                return
            result = cached[1]
            if result.status == 'ready':
                self._requeue_front(result.records)
            self._remember_terminal(acquire_id, cached[0], 'cancelled')

    def discard_unacquired(self, records: tuple[ReadyRecord, ...]) -> None:
        """Retire exact ready records without permitting publication replay."""
        with self._lock:
            reserved = {item.key for _, result in self._acquires.values() for item in result.records}
            for record in records:
                record.validate()
                if self._records.get(record.key) != record or record.key in reserved:
                    raise ValueError('record is not an exact unacquired ready record')
            for record in records:
                self._ready_by_producer[record.producer_endpoint].remove(record.key)
                del self._records[record.key]
                self._remember_delivered(record)

    def status(self) -> dict[str, int]:
        with self._lock:
            return {'ready_records': len(self._records), 'ready_producers': len(self._ready_by_producer), 'cached_acquires': len(self._acquires), 'terminal_acquires': len(self._terminal_acquires), 'delivered_tombstones': len(self._delivered)}

    def snapshot_quiescent(self) -> list[dict]:
        """Export only delivered tombstones after all ready work is drained."""
        with self._lock:
            if self._records or self._acquires:
                raise RuntimeError('ready router is not quiescent')
            return [{'key': record.key, 'producer_endpoint': record.producer_endpoint, 'metadata': record.metadata.to_dict(), 'payload_checksum': record.payload_checksum} for record in self._delivered.values()]

    def restore_quiescent(self, delivered: list[dict]) -> None:
        """Restore old keys as tombstones; never make them ready again."""
        if not isinstance(delivered, list):
            raise ValueError('delivered tombstones must be a list')
        records: list[ReadyRecord] = []
        for raw in delivered:
            record = ReadyRecord(key=str(raw['key']), producer_endpoint=str(raw['producer_endpoint']), metadata=SampleMetadataV3.from_dict(raw['metadata']), payload_checksum=str(raw['payload_checksum']))
            record.validate()
            records.append(record)
        if len({record.key for record in records}) != len(records):
            raise ValueError('delivered tombstones contain duplicate keys')
        with self._lock:
            if self._records or self._acquires or self._delivered:
                raise RuntimeError('ready router must be empty before restore')
            for record in records:
                self._remember_delivered(record)

    def _pop_matching(self, producer: str, *, generation: int, target_revision: str, allowed_keys=None) -> ReadyRecord | None:
        queue = self._ready_by_producer.get(producer)
        if not queue:
            return None
        for _ in range(len(queue)):
            key = queue.popleft()
            record = self._records.get(key)
            if record is None:
                continue
            metadata = record.metadata
            if metadata.generation == generation and metadata.target_revision == target_revision and (allowed_keys is None or key in allowed_keys):
                return record
            queue.append(key)
        return None

    def _requeue_front(self, records: list[ReadyRecord] | tuple[ReadyRecord, ...]) -> None:
        for record in reversed(records):
            queue = self._ready_by_producer.setdefault(record.producer_endpoint, deque())
            queue.appendleft(record.key)

    def _remember_acquire(self, acquire_id: str, fingerprint: str, result: AcquireResult) -> None:
        self._acquires[acquire_id] = (fingerprint, result)
        self._acquires.move_to_end(acquire_id)
        while len(self._acquires) > self._acquire_cache_size:
            evictable = next((key for key, (_, cached_result) in self._acquires.items() if cached_result.status != 'ready'), None)
            if evictable is None:
                break
            self._acquires.pop(evictable, None)

    def _remember_delivered(self, record: ReadyRecord) -> None:
        self._delivered[record.key] = record
        self._delivered.move_to_end(record.key)
        while len(self._delivered) > self._acquire_cache_size:
            self._delivered.popitem(last=False)

    def _remember_terminal(self, acquire_id: str, fingerprint: str, status: str) -> None:
        self._terminal_acquires[acquire_id] = (fingerprint, status)
        self._terminal_acquires.move_to_end(acquire_id)
        while len(self._terminal_acquires) > self._acquire_cache_size:
            self._terminal_acquires.popitem(last=False)

def ready_router_acquire_fingerprint(request: AcquireRequest) -> str:
    payload = {'consumer_id': request.consumer_id, 'generation': request.generation, 'max_items': request.max_items, 'min_items': request.min_items, 'required_target_revision': request.required_target_revision}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()
from typing import Any, Mapping, Protocol
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_ready_tag, make_v3_sample_key, parse_v3_ready_tag

class KeyedStore(Protocol):

    def get_tag(self, key: str, partition: str) -> Mapping[str, Any] | None:
        ...

    def put(self, key: str, partition: str, fields: Mapping[str, Any], tag: Mapping[str, Any]) -> None:
        ...

    def clear(self, key: str, partition: str) -> None:
        ...

class BackpressureError(RuntimeError):
    pass

@dataclass(frozen=True)
class PublishResult:
    key: str
    status: str
    payload_checksum: str

class TQKeyedStore:
    """Attach to an already initialized TransferQueue; never owns its lifecycle."""

    def __init__(self, tq: Any | None=None) -> None:
        if tq is None:
            from verl_speco.runtime.services import import_transfer_queue
            tq = import_transfer_queue()
        self._tq = tq

    def get_tag(self, key: str, partition: str) -> Mapping[str, Any] | None:
        return self._tq.kv_list(partition_id=partition).get(partition, {}).get(key)

    def put(self, key: str, partition: str, fields: Mapping[str, Any], tag: Mapping[str, Any]) -> None:
        self._tq.kv_put(key=key, partition_id=partition, fields=dict(fields), tag=dict(tag))

    def clear(self, key: str, partition: str) -> None:
        self._tq.kv_clear(keys=key, partition_id=partition)

class BoundedFeaturePublisher:
    """Write payload first, then publish only a small ready control record.

    The capacity is local to one producer process.  A distributed owner must
    enforce the aggregate quota and key reservations before multi-producer use.
    """

    def __init__(self, *, store: KeyedStore, router: ReadyRouter, partition: str, producer_endpoint: str, capacity_bytes: int, high_watermark_ratio: float=0.85, low_watermark_ratio: float=0.6) -> None:
        if not partition or not producer_endpoint or capacity_bytes <= 0:
            raise ValueError('partition, producer_endpoint and positive capacity_bytes are required')
        if not 0 < low_watermark_ratio < high_watermark_ratio <= 1:
            raise ValueError('watermarks must satisfy 0 < low < high <= 1')
        self._store = store
        self._router = router
        self._partition = partition
        self._endpoint = producer_endpoint
        self._capacity = capacity_bytes
        self._high = int(capacity_bytes * high_watermark_ratio)
        self._low = int(capacity_bytes * low_watermark_ratio)
        self._lock = threading.RLock()
        self._retained: dict[str, int] = {}
        self._used_bytes = 0
        self._paused = False

    def publish(self, metadata: SampleMetadataV3, *, fields: Mapping[str, Any], payload_checksum: str) -> PublishResult:
        metadata.validate()
        key = make_v3_sample_key(metadata)
        tag = make_v3_ready_tag(metadata, payload_checksum=payload_checksum)
        field_nbytes = bounded_publisher_tensor_nbytes(fields)
        if field_nbytes != metadata.payload_nbytes:
            raise ValueError(f'payload_nbytes mismatch: metadata={metadata.payload_nbytes}, tensors={field_nbytes}')
        record = ReadyRecord(key, self._endpoint, metadata, payload_checksum)
        with self._lock:
            routing_state = self._router.lookup(record)
            if routing_state == 'conflict':
                raise ValueError(f'ReadyRouter rejected sample key {key!r}')
            existing = self._store.get_tag(key, self._partition)
            if existing is not None:
                if parse_v3_ready_tag(existing) != (metadata, payload_checksum):
                    raise ValueError(f'stable sample key {key!r} already has different metadata or checksum')
                status = self._router.publish(record)
                if status == 'conflict':
                    raise ValueError(f'ReadyRouter rejected existing key {key!r}')
                return PublishResult(key, 'duplicate', payload_checksum)
            if routing_state == 'delivered_duplicate':
                return PublishResult(key, 'duplicate', payload_checksum)
            if routing_state == 'ready_duplicate':
                raise RuntimeError(f'ready key {key!r} has no TQ payload')
            if self._paused or self._used_bytes + field_nbytes > self._capacity:
                raise BackpressureError('feature publisher byte quota reached')
            self._store.put(key, self._partition, fields, tag)
            self._retained[key] = field_nbytes
            self._used_bytes += field_nbytes
            if self._used_bytes >= self._high:
                self._paused = True
            status = self._router.publish(record)
            if status == 'conflict':
                raise ValueError(f'ReadyRouter rejected published key {key!r}')
            return PublishResult(key, status, payload_checksum)

    def release(self, key: str) -> None:
        """Clear TQ payload only after all consumers have materialized it."""
        with self._lock:
            if key not in self._retained:
                raise ValueError(f'payload key {key!r} is not retained by this publisher')
            self._store.clear(key, self._partition)
            size = self._retained.pop(key, 0)
            self._used_bytes -= size
            if self._used_bytes <= self._low:
                self._paused = False

    def status(self) -> dict[str, int | bool]:
        with self._lock:
            return {'capacity_bytes': self._capacity, 'used_bytes': self._used_bytes, 'retained_items': len(self._retained), 'paused': self._paused}

class RayBoundedFeaturePublisher:
    """Serialize key reservation and aggregate byte quota through one Ray actor."""

    def __init__(self, *, store: KeyedStore, control_actor: Any, partition: str, producer_endpoint: str) -> None:
        if not partition or not producer_endpoint:
            raise ValueError('partition and producer endpoint are required')
        self._store = store
        self._control = control_actor
        self._partition = partition
        self._endpoint = producer_endpoint

    def publish(self, metadata: SampleMetadataV3, *, fields: Mapping[str, Any], payload_checksum: str) -> PublishResult:
        import ray
        metadata.validate()
        key = make_v3_sample_key(metadata)
        size = bounded_publisher_tensor_nbytes(fields)
        if size != metadata.payload_nbytes:
            raise ValueError('payload_nbytes mismatch')
        record = ReadyRecord(key, self._endpoint, metadata, payload_checksum)
        reservation = ray.get(self._control.reserve.remote(record, nbytes=size))
        if reservation.status == 'duplicate':
            if parse_v3_ready_tag(self._store.get_tag(key, self._partition) or {}) != (metadata, payload_checksum):
                raise RuntimeError('control owner remembers key without matching TQ payload')
            return PublishResult(key, 'duplicate', payload_checksum)
        if reservation.status == 'backpressure':
            raise BackpressureError('shared feature byte quota reached')
        if reservation.status != 'reserved':
            raise ValueError(f'shared key reservation failed: {reservation.status}')
        attempted_put = False
        try:
            if self._store.get_tag(key, self._partition) is not None:
                raise RuntimeError('TQ contains an untracked stable sample key')
            attempted_put = True
            self._store.put(key, self._partition, fields, make_v3_ready_tag(metadata, payload_checksum=payload_checksum))
            status = ray.get(self._control.publish.remote(record, token=reservation.token))
        except BaseException:
            state = ray.get(self._control.publication_status.remote(record))
            if state == 'published':
                return PublishResult(key, 'published', payload_checksum)
            if state != 'reserved':
                raise RuntimeError('publication state is uncertain; payload retained')
            if attempted_put:
                tag = self._store.get_tag(key, self._partition)
                if tag is not None:
                    if parse_v3_ready_tag(tag) != (metadata, payload_checksum):
                        raise RuntimeError('TQ key changed during failed publication')
                    self._store.clear(key, self._partition)
            ray.get(self._control.cancel.remote(key, token=reservation.token))
            raise
        return PublishResult(key, status, payload_checksum)

def bounded_publisher_tensor_nbytes(fields: Mapping[str, Any]) -> int:
    if not fields:
        raise ValueError('feature payload needs at least one tensor field')
    total = 0
    for name, tensor in fields.items():
        if not name or not hasattr(tensor, 'numel') or (not hasattr(tensor, 'element_size')):
            raise TypeError(f'feature field {name!r} must be a tensor')
        total += int(tensor.numel()) * int(tensor.element_size())
    if total <= 0:
        raise ValueError('feature payload must contain bytes')
    return total
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

class LeaseState(str, Enum):
    LEASED = 'leased'
    MATERIALIZED = 'materialized'
    COMMITTED = 'committed'
    CANCELLED = 'cancelled'

@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    acquire_id: str
    key: str
    request_id: str
    consumer_id: str
    state: LeaseState
    payload_released: bool
    expires_at: float
    generation: int
    target_revision: str

@dataclass(frozen=True)
class CommitReceipt:
    optimizer_step_id: str
    transaction_set_hash: str
    committed_lease_ids: tuple[str, ...]
    already_committed: bool = False

class LeaseRegistry:
    """Single-shard state owner; wrap this class in a Ray actor for multi-node use."""

    def __init__(self, *, remembered_steps: int=4096) -> None:
        if remembered_steps <= 0:
            raise ValueError('remembered_steps must be positive')
        self._lock = threading.RLock()
        self._remembered_steps = remembered_steps
        self._leases: dict[str, LeaseRecord] = {}
        self._by_key: dict[str, str] = {}
        self._committed_steps: dict[str, tuple[str, tuple[str, ...]]] = {}

    def register(self, acquire: AcquireResult, *, consumer_id: str, lease_timeout_s: float, now: float | None=None) -> tuple[LeaseRecord, ...]:
        """Create or recover stable leases for a successful ReadyRouter acquire."""
        if acquire.status != 'ready' or not acquire.records:
            raise ValueError('only a ready acquire with records can be registered')
        if not consumer_id or lease_timeout_s <= 0:
            raise ValueError('consumer_id and lease_timeout_s are required')
        deadline = (time.time() if now is None else now) + lease_timeout_s
        with self._lock:
            keys_seen: set[str] = set()
            for record in acquire.records:
                record.validate()
                if record.key in keys_seen:
                    raise ValueError(f'duplicate ready key {record.key!r} in acquire')
                keys_seen.add(record.key)
                lease_id = lease_registry_lease_id(acquire.acquire_id, record)
                existing = self._leases.get(lease_id)
                if existing is not None:
                    if existing.consumer_id != consumer_id or existing.key != record.key:
                        raise ValueError('stable lease ID resolved to incompatible state')
                    continue
                existing_key = self._by_key.get(record.key)
                if existing_key is not None:
                    raise ValueError(f'ready key already belongs to lease {existing_key!r}')
            leases: list[LeaseRecord] = []
            for record in acquire.records:
                lease_id = lease_registry_lease_id(acquire.acquire_id, record)
                existing = self._leases.get(lease_id)
                if existing is not None:
                    leases.append(existing)
                    continue
                lease = LeaseRecord(lease_id=lease_id, acquire_id=acquire.acquire_id, key=record.key, request_id=record.metadata.request_id, consumer_id=consumer_id, state=LeaseState.LEASED, payload_released=False, expires_at=deadline, generation=record.metadata.generation, target_revision=record.metadata.target_revision)
                self._leases[lease_id] = lease
                self._by_key[record.key] = lease_id
                leases.append(lease)
            return tuple(leases)

    def mark_materialized(self, lease_ids: Sequence[str]) -> tuple[LeaseRecord, ...]:
        with self._lock:
            leases = self._require_many(lease_ids)
            for lease in leases:
                if lease.state not in {LeaseState.LEASED, LeaseState.MATERIALIZED}:
                    raise ValueError(f'lease {lease.lease_id!r} cannot materialize from {lease.state.value}')
            updated = tuple((replace(lease, state=LeaseState.MATERIALIZED) for lease in leases))
            self._replace_many(updated)
            return updated

    def mark_payload_released(self, lease_ids: Sequence[str]) -> tuple[LeaseRecord, ...]:
        with self._lock:
            leases = self._require_many(lease_ids)
            for lease in leases:
                if lease.state is not LeaseState.MATERIALIZED:
                    raise ValueError(f'lease {lease.lease_id!r} cannot release payload from {lease.state.value}')
            updated = tuple((replace(lease, payload_released=True) for lease in leases))
            self._replace_many(updated)
            return updated

    def commit(self, *, optimizer_step_id: str, transaction_set_hash: str, lease_ids: Sequence[str]) -> CommitReceipt:
        if not optimizer_step_id or not transaction_set_hash:
            raise ValueError('optimizer_step_id and transaction_set_hash are required')
        normalized_ids = tuple(sorted({str(item) for item in lease_ids if str(item)}))
        if not normalized_ids:
            raise ValueError('lease_ids must not be empty')
        with self._lock:
            remembered = self._committed_steps.get(optimizer_step_id)
            if remembered is not None:
                if remembered != (transaction_set_hash, normalized_ids):
                    raise ValueError('optimizer step ID was reused with different transactions')
                return CommitReceipt(optimizer_step_id=optimizer_step_id, transaction_set_hash=transaction_set_hash, committed_lease_ids=normalized_ids, already_committed=True)
            leases = self._require_many(normalized_ids)
            for lease in leases:
                if lease.state is not LeaseState.MATERIALIZED or not lease.payload_released:
                    raise ValueError(f'lease {lease.lease_id!r} must be materialized and payload-released before commit')
            self._replace_many(tuple((replace(lease, state=LeaseState.COMMITTED) for lease in leases)))
            self._committed_steps[optimizer_step_id] = (transaction_set_hash, normalized_ids)
            while len(self._committed_steps) > self._remembered_steps:
                self._committed_steps.pop(next(iter(self._committed_steps)))
            return CommitReceipt(optimizer_step_id=optimizer_step_id, transaction_set_hash=transaction_set_hash, committed_lease_ids=normalized_ids)

    def validate_commit(self, *, optimizer_step_id: str, transaction_set_hash: str, lease_ids: Sequence[str]) -> None:
        """Check a step without changing lease state before a coordinated commit."""
        normalized_ids = tuple(sorted({str(item) for item in lease_ids if str(item)}))
        if not optimizer_step_id or not transaction_set_hash or (not normalized_ids):
            raise ValueError('optimizer_step_id, transaction_set_hash and lease_ids are required')
        with self._lock:
            remembered = self._committed_steps.get(optimizer_step_id)
            if remembered is not None:
                if remembered != (transaction_set_hash, normalized_ids):
                    raise ValueError('optimizer step ID was reused with different transactions')
                return
            for lease in self._require_many(normalized_ids):
                if lease.state is not LeaseState.MATERIALIZED or not lease.payload_released:
                    raise ValueError(f'lease {lease.lease_id!r} must be materialized and payload-released before commit')

    def cancel(self, lease_ids: Sequence[str], *, reason: str='') -> tuple[LeaseRecord, ...]:
        del reason
        with self._lock:
            leases = self._require_many(lease_ids)
            updated: list[LeaseRecord] = []
            for lease in leases:
                if lease.state is LeaseState.COMMITTED:
                    raise ValueError(f'committed lease {lease.lease_id!r} cannot be cancelled')
                updated.append(replace(lease, state=LeaseState.CANCELLED))
            self._replace_many(tuple(updated))
            return tuple(updated)

    def expire(self, *, now: float | None=None) -> tuple[LeaseRecord, ...]:
        """Cancel only leases that have not crossed the optimizer boundary."""
        current = time.time() if now is None else now
        with self._lock:
            expired = tuple((replace(lease, state=LeaseState.CANCELLED) for lease in self._leases.values() if lease.state is LeaseState.LEASED and lease.expires_at <= current))
            self._replace_many(expired)
            return expired

    def get_many(self, lease_ids: Sequence[str]) -> tuple[LeaseRecord, ...]:
        """Read a bounded batch in caller order under the same registry lock."""
        ids = lease_registry_bounded_lease_ids(lease_ids)
        with self._lock:
            return self._require_many(ids)

    def get(self, lease_id: str) -> LeaseRecord:
        with self._lock:
            return self._require_many((lease_id,))[0]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {'schema_version': 1, 'leases': [{**asdict(lease), 'state': lease.state.value} for lease in sorted(self._leases.values(), key=lambda item: item.lease_id)], 'committed_steps': {step_id: {'transaction_set_hash': value[0], 'lease_ids': list(value[1])} for step_id, value in self._committed_steps.items()}}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if int(snapshot.get('schema_version', 0)) != 1:
            raise ValueError('unsupported lease registry snapshot')
        raw_leases = snapshot.get('leases')
        raw_steps = snapshot.get('committed_steps')
        if not isinstance(raw_leases, list) or not isinstance(raw_steps, Mapping):
            raise ValueError('invalid lease registry snapshot')
        restored: dict[str, LeaseRecord] = {}
        by_key: dict[str, str] = {}
        for raw in raw_leases:
            if not isinstance(raw, Mapping):
                raise ValueError('lease snapshot entry must be a mapping')
            lease = LeaseRecord(lease_id=str(raw['lease_id']), acquire_id=str(raw['acquire_id']), key=str(raw['key']), request_id=str(raw['request_id']), consumer_id=str(raw['consumer_id']), state=LeaseState(str(raw['state'])), payload_released=bool(raw['payload_released']), expires_at=float(raw['expires_at']), generation=int(raw['generation']), target_revision=str(raw['target_revision']))
            if lease.state is LeaseState.COMMITTED and (not lease.payload_released):
                raise ValueError('committed lease snapshot has unreleased payload')
            if lease.lease_id in restored or lease.key in by_key:
                raise ValueError('lease snapshot contains duplicate lease ID or key')
            restored[lease.lease_id] = lease
            by_key[lease.key] = lease.lease_id
        committed: dict[str, tuple[str, tuple[str, ...]]] = {}
        for step_id, raw in raw_steps.items():
            if not isinstance(raw, Mapping):
                raise ValueError('committed step snapshot entry must be a mapping')
            lease_ids = tuple(sorted((str(item) for item in raw['lease_ids'])))
            if not str(step_id) or not str(raw['transaction_set_hash']) or (not lease_ids) or (len(set(lease_ids)) != len(lease_ids)) or any((item not in restored or restored[item].state is not LeaseState.COMMITTED for item in lease_ids)):
                raise ValueError('committed step snapshot has inconsistent leases')
            committed[str(step_id)] = (str(raw['transaction_set_hash']), lease_ids)
        with self._lock:
            self._leases = restored
            self._by_key = by_key
            self._committed_steps = committed

    def status(self) -> dict[str, int]:
        with self._lock:
            states = [lease.state for lease in self._leases.values()]
            return {'leases_total': len(states), 'leases_leased': states.count(LeaseState.LEASED), 'leases_materialized': states.count(LeaseState.MATERIALIZED), 'leases_committed': states.count(LeaseState.COMMITTED), 'leases_cancelled': states.count(LeaseState.CANCELLED), 'committed_steps': len(self._committed_steps)}

    def _require_many(self, lease_ids: Sequence[str]) -> tuple[LeaseRecord, ...]:
        normalized = tuple((str(item) for item in lease_ids))
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError('lease_ids must be non-empty and unique')
        try:
            return tuple((self._leases[item] for item in normalized))
        except KeyError as exc:
            raise ValueError(f'unknown lease {exc.args[0]!r}') from exc

    def _replace_many(self, leases: Sequence[LeaseRecord]) -> None:
        for lease in leases:
            self._leases[lease.lease_id] = lease

def lease_registry_lease_id(acquire_id: str, record: ReadyRecord) -> str:
    material = f'{acquire_id}:{record.key}:{record.payload_checksum}'.encode('utf-8')
    return f'lease-{hashlib.sha256(material).hexdigest()[:32]}'

def lease_registry_bounded_lease_ids(lease_ids):
    if not isinstance(lease_ids, (list, tuple)):
        raise ValueError('lease read batch must be an explicit list or tuple')
    ids = tuple(lease_ids)
    if not 0 < len(ids) <= 128 or any((type(i) is not str or not i for i in ids)) or len(set(ids)) != len(ids):
        raise ValueError('lease read batch must contain 1-128 distinct lease IDs')
    return ids

def bounded_lease_records(registry, lease_ids):
    """Read exact IDs with at most 128 records per RPC; never retry failed reads."""
    if not isinstance(lease_ids, (list, tuple)):
        raise ValueError('lease read requires explicit IDs')
    ids = tuple(lease_ids)
    if not ids or any((type(i) is not str or not i for i in ids)) or len(set(ids)) != len(ids):
        raise ValueError('lease read requires distinct nonempty IDs')
    read_many = getattr(registry, 'get_many', None)
    result = []
    for offset in range(0, len(ids), 128):
        block = ids[offset:offset + 128]
        records = read_many(block) if callable(read_many) else tuple((registry.get(i) for i in block))
        if not isinstance(records, (list, tuple)) or len(records) != len(block) or any((not isinstance(r, LeaseRecord) or r.lease_id != i for i, r in zip(block, records))):
            raise ValueError('lease read reply differs from the exact requested IDs')
        result.extend(records)
    return tuple(result)
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import secrets
import tempfile
from typing import Mapping, Sequence
from verl_speco.transport.streaming import SampleMetadataV3

def ready_control_owner_node_id(ray_nodes: Sequence[Mapping[str, object]], ray_address: str) -> str:
    """Pin the control actor to the Ray owner with the local receipt /tmp."""
    owner_address = ray_address.rsplit(':', 1)[0]
    matching = [node for node in ray_nodes if node.get('Alive') and node.get('NodeManagerAddress') == owner_address]
    if len(matching) != 1 or not isinstance(matching[0].get('NodeID'), str):
        raise RuntimeError('ready control owner Ray node is unavailable or ambiguous')
    return matching[0]['NodeID']

@dataclass(frozen=True)
class KeyReservation:
    status: str
    key: str
    token: str = ''

class RayReadyControl:
    """State owned by one Ray actor for one run and TQ partition."""

    def __init__(self, *, run_id: str, generation: int, partition: str, capacity_bytes: int, reservation_timeout_s: float=300.0, high_watermark_ratio: float=0.85, low_watermark_ratio: float=0.6) -> None:
        if not run_id or generation < 0 or (not partition) or (capacity_bytes <= 0) or (reservation_timeout_s <= 0):
            raise ValueError('run_id, generation, partition, capacity and reservation timeout are required')
        if not 0 < low_watermark_ratio < high_watermark_ratio <= 1:
            raise ValueError('watermarks must satisfy 0 < low < high <= 1')
        self.run_id = run_id
        self.generation = generation
        self.partition = partition
        self.capacity_bytes = capacity_bytes
        self.reservation_timeout_s = reservation_timeout_s
        self._high = max(1, int(capacity_bytes * high_watermark_ratio))
        self._low = int(capacity_bytes * low_watermark_ratio)
        self.router = ReadyRouter()
        self.leases = LeaseRegistry()
        self._pending: dict[str, tuple[ReadyRecord, str, int, float]] = {}
        self._retained: dict[str, tuple[ReadyRecord, int]] = {}
        self._acquired: dict[str, tuple[AcquireRequest, AcquireResult, tuple[LeaseRecord, ...]]] = {}
        self._acquired_required_keys = {}
        self._reclaim_pending: dict[str, str] = {}
        self._producer_incarnations: dict[str, str] = {}
        self._used_bytes = 0
        self._paused = False
        from verl_speco.transport.waves import MaterializedWaveAckHints
        self._wave_ack_hints = MaterializedWaveAckHints()

    def publish_materialized_wave_ack(self, *, batch_id, ack, lease_ids):
        return self._wave_ack_hints.publish(batch_id=batch_id, ack=ack, lease_ids=lease_ids, get_lease=self.get_lease, run_id=self.run_id, generation=self.generation)

    def get_materialized_wave_ack(self, *, batch_id, wave_index):
        return self._wave_ack_hints.get(batch_id=batch_id, wave_index=wave_index)

    def publish_parent_commit_permit(self, *, permit, receipt):
        from verl_speco.runtime.services import canonical_receipt_sha256, cross_step_prefill_sha as _sha
        required = {'schema_version', 'run_id', 'generation', 'parent_step_index', 'parent_cursor', 'next_step_index', 'logical_checkpoint_fingerprint', 'next_batch_control_sha256', 'parent_receipt_sha256', 'state'}
        if not isinstance(permit, dict) or set(permit) != required or (not isinstance(receipt, dict)):
            raise ValueError('parent permit hint must be bounded receipt metadata')
        step = permit['parent_step_index']
        keys = receipt.get('packed_sample_keys')
        if type(step) is not int or not 0 <= step < 32 or type(permit['schema_version']) is not int or (permit['schema_version'] != 1) or (permit['run_id'] != self.run_id) or (type(permit['generation']) is not int) or (permit['generation'] != self.generation) or (permit['state'] != 'STEP_COMMITTED') or (type(permit['next_step_index']) is not int) or (permit['next_step_index'] != step + 1) or (not isinstance(keys, list)) or (not 1 <= len(keys) <= 1024) or any((not isinstance(k, str) or not k for k in keys)) or (len(set(keys)) != len(keys)) or (type(permit['parent_cursor']) is not int) or (permit['parent_cursor'] != (step + 1) * len(keys)) or any((not _sha(permit[k]) for k in ['logical_checkpoint_fingerprint', 'next_batch_control_sha256', 'parent_receipt_sha256'])) or (receipt.get('run_id') != self.run_id) or (receipt.get('generation') != self.generation) or (type(receipt.get('sample_index')) is not int) or (receipt['sample_index'] != step * len(keys)) or (permit['parent_receipt_sha256'] != canonical_receipt_sha256(receipt)):
            raise ValueError('parent permit hint receipt or owner identity differs')
        committed = self.leases.snapshot()['committed_steps'].get(receipt.get('step_id'))
        if committed is None:
            raise RuntimeError('parent permit has no committed optimizer lease proof')
        leases = [self.leases.get(i) for i in committed['lease_ids']]
        if self._used_bytes or self._pending or self._retained or (len(leases) != len(keys)) or ({l.key for l in leases} != set(keys)) or any((l.state is not LeaseState.COMMITTED or not l.payload_released or l.generation != self.generation for l in leases)):
            raise RuntimeError('parent permit requires quiescent committed released leases')
        data = json.dumps(permit, sort_keys=True, separators=(',', ':')).encode()
        if len(data) > 4096:
            raise ValueError('parent permit hint exceeds metadata bound')
        hints = getattr(self, '_parent_commit_permit_hints', None)
        if hints is None:
            hints = {}
            self._parent_commit_permit_hints = hints
        if step in hints and hints[step] != data:
            raise RuntimeError('parent permit hint identity changed')
        hints[step] = data
        return 'PARENT_COMMIT_PERMIT_HINT_PUBLISHED'

    def get_parent_commit_permit(self, *, parent_step_index):
        if type(parent_step_index) is not int or not 0 <= parent_step_index < 32:
            raise ValueError('invalid parent permit hint step')
        value = getattr(self, '_parent_commit_permit_hints', {}).get(parent_step_index)
        return json.loads(value) if value is not None else None

    def register_node_wave_owner(self, *, samples, wave_size, plan_sha256, owner_addresses, owner_rank, address, incarnation, selective_plan=None):
        from verl_speco.transport.waves import TwoTrainerWaveGroup
        candidate = TwoTrainerWaveGroup(samples=samples, wave_size=wave_size, plan_sha256=plan_sha256, owner_addresses=owner_addresses, get_lease=self.get_lease, selective_plan=selective_plan)
        if any((m.run_id != self.run_id or m.generation != self.generation for _, m in candidate.samples)):
            raise ValueError('DP32 batch run or generation differs from control owner')
        groups = getattr(self, '_node_wave_groups', None)
        if groups is None:
            groups = {}
            self._node_wave_groups = groups
        if any((group.failed for group in groups.values())):
            raise RuntimeError('DP32 generation failed; cannot replace transaction set')
        old = groups.get(candidate.batch_id)
        if old is not None and old.binding != candidate.binding:
            raise ValueError('DP32 wave batch binding changed')
        if old is None:
            if len(groups) >= 32:
                raise ValueError('DP32 wave batch registry exceeds bounded run steps')
            result = candidate.register(owner_rank=owner_rank, address=address, incarnation=incarnation)
            groups[candidate.batch_id] = candidate
            return result
        return old.register(owner_rank=owner_rank, address=address, incarnation=incarnation)

    def _node_wave_group(self, batch_id):
        if batch_id not in getattr(self, '_node_wave_groups', {}):
            raise ValueError('DP32 wave batch not registered')
        return self._node_wave_groups[batch_id]

    def get_node_wave_state(self, *, batch_id):
        return self._node_wave_group(batch_id).state()

    def get_node_wave_owner_incarnations(self, *, batch_id, owner_rank, incarnation):
        group = self._node_wave_group(batch_id)
        group._owner(owner_rank, incarnation)
        if group.failed or group.barrier is None or set(group.incarnations) != {0, 16}:
            raise RuntimeError('both selected physical owners must be ready')
        if group.selective_plan is None:
            raise ValueError('owner identities requested outside selected transaction')
        return dict(group.incarnations)

    def begin_node_wave(self, *, batch_id, **values):
        return self._node_wave_group(batch_id).begin(**values)

    def acknowledge_node_wave(self, *, batch_id, ack):
        return self._node_wave_group(batch_id).acknowledge(ack)

    def node_wave_release_permit(self, *, batch_id, **values):
        return self._node_wave_group(batch_id).release_permit(**values)

    def confirm_node_wave_release(self, *, batch_id, **values):
        return self._node_wave_group(batch_id).confirm_release(**values)

    def abort_node_wave(self, *, batch_id, **values):
        return self._node_wave_group(batch_id).abort(**values)

    def reserve(self, record: ReadyRecord, *, nbytes: int) -> KeyReservation:
        record.validate()
        if record.metadata.run_id != self.run_id or record.metadata.generation != self.generation or nbytes != record.metadata.payload_nbytes:
            raise ValueError('reservation run, generation or payload byte count differs')
        previous_incarnation = self._producer_incarnations.get(record.metadata.producer_id)
        if previous_incarnation is not None and previous_incarnation != record.metadata.producer_incarnation:
            raise ValueError('producer incarnation changed within a generation')
        self._expire_reservations()
        retained = self._retained.get(record.key)
        if retained is not None:
            return KeyReservation('duplicate' if retained[0] == record else 'conflict', record.key)
        pending = self._pending.get(record.key)
        if pending is not None:
            return KeyReservation('busy' if pending[0] == record else 'conflict', record.key)
        router_state = self.router.lookup(record)
        if router_state == 'conflict':
            return KeyReservation('conflict', record.key)
        if router_state != 'absent':
            return KeyReservation('stale', record.key)
        if self._paused or self._used_bytes + nbytes > self.capacity_bytes:
            return KeyReservation('backpressure', record.key)
        token = secrets.token_hex(16)
        self._pending[record.key] = (record, token, nbytes, time.monotonic() + self.reservation_timeout_s)
        self._producer_incarnations.setdefault(record.metadata.producer_id, record.metadata.producer_incarnation)
        self._used_bytes += nbytes
        if self._used_bytes >= self._high:
            self._paused = True
        return KeyReservation('reserved', record.key, token)

    def publish(self, record: ReadyRecord, *, token: str) -> str:
        record.validate()
        if record.metadata.run_id != self.run_id or record.metadata.generation != self.generation:
            raise ValueError('publication run or generation differs')
        pending = self._pending.get(record.key)
        if pending is None or pending[0] != record or pending[1] != token:
            raise ValueError('missing or mismatched key reservation')
        if pending[3] <= time.monotonic():
            self.cancel(record.key, token=token)
            raise TimeoutError('key reservation expired before publish')
        status = self.router.publish(record)
        if status != 'published':
            raise RuntimeError(f'ready router returned {status!r} for a reserved key')
        self._retained[record.key] = (record, pending[2])
        del self._pending[record.key]
        return status

    def publication_status(self, record: ReadyRecord) -> str:
        retained = self._retained.get(record.key)
        if retained is not None:
            return 'published' if retained[0] == record else 'conflict'
        pending = self._pending.get(record.key)
        if pending is not None:
            return 'reserved' if pending[0] == record else 'conflict'
        return 'absent'

    def cancel(self, key: str, *, token: str) -> bool:
        pending = self._pending.get(key)
        if pending is None:
            return False
        if pending[1] != token:
            raise ValueError('reservation token does not match')
        self._used_bytes -= pending[2]
        del self._pending[key]
        self._resume_if_low()
        return True

    def discard_unacquired(self, records: tuple[ReadyRecord, ...]) -> tuple[str, ...]:
        """Drain exact failed-step publications, preserving all acquired work.

        Ray serializes this method with acquire. Validate the entire batch before
        clearing any TQ key; retain accounting on uncertain transport cleanup so
        a retry remains safe. Retired keys become publication tombstones.
        """
        if not records or len({record.key for record in records}) != len(records):
            raise ValueError('cleanup requires distinct exact records')
        acquired = {record.key for _, result, _ in self._acquired.values() for record in result.records}
        active = []
        for record in records:
            record.validate()
            if record.metadata.run_id != self.run_id or record.metadata.generation != self.generation:
                raise ValueError('cleanup run or generation differs')
            if record.key in acquired:
                raise ValueError('cleanup cannot discard acquired work')
            retained = self._retained.get(record.key)
            state = self.router.lookup(record)
            if retained is None and state == 'delivered_duplicate':
                continue
            if retained is None or retained[0] != record or state != 'ready_duplicate':
                raise ValueError('cleanup record is not an exact retained ready record')
            active.append(record)
        if not active:
            return ()
        from verl_speco.runtime.services import import_transfer_queue
        tq = import_transfer_queue()
        tq.init()
        for record in active:
            present = tq.kv_list(partition_id=self.partition).get(self.partition, {})
            if record.key in present:
                tq.kv_clear(keys=record.key, partition_id=self.partition)
            if record.key in tq.kv_list(partition_id=self.partition).get(self.partition, {}):
                raise RuntimeError('failed-step TQ publication remains present')
        self.router.discard_unacquired(tuple(active))
        for record in active:
            self._used_bytes -= self._retained.pop(record.key)[1]
        self._resume_if_low()
        return tuple((record.key for record in active))

    def acquire(self, request: AcquireRequest, *, lease_timeout_s: float, required_keys=None):
        request.validate()
        from verl_speco.transport.waves import required_wave_keys
        required_keys = required_wave_keys(request, required_keys)
        if request.generation != self.generation:
            raise ValueError('acquire generation differs from control owner')
        previous = self._acquired.get(request.acquire_id)
        if previous is not None:
            if self._acquired_required_keys.get(request.acquire_id) != required_keys:
                raise ValueError('acquire ID was reused for a different exact wave')
            old = previous[0]
            if (old.consumer_id, old.generation, old.max_items, old.min_items, old.required_target_revision) != (request.consumer_id, request.generation, request.max_items, request.min_items, request.required_target_revision):
                raise ValueError('acquire ID was reused for a different request')
            return (previous[1], tuple((self.leases.get(lease.lease_id) for lease in previous[2])))
        result = self.router.acquire(request, **{'required_keys': required_keys} if required_keys is not None else {})
        if result.status != 'ready':
            return (result, ())
        try:
            leases = self.leases.register(result, consumer_id=request.consumer_id, lease_timeout_s=lease_timeout_s)
        except BaseException:
            self.router.cancel(request.acquire_id)
            raise
        self.router.ack(request.acquire_id)
        self._acquired[request.acquire_id] = (request, result, leases)
        self._acquired_required_keys[request.acquire_id] = required_keys
        return (result, leases)

    def _require_node_wave_materialization(self, lease_ids):
        groups = getattr(self, '_node_wave_groups', {})
        if not groups:
            return
        if any((group.failed for group in groups.values())):
            raise RuntimeError('DP32 generation failed before materialization')
        for lease_id in lease_ids:
            lease = self.get_lease(lease_id)
            if lease.state is LeaseState.MATERIALIZED and lease.payload_released:
                continue
            group = next((g for g in groups.values() if lease.key in {key for key, _ in g.samples}), None)
            if group is None or group.barrier is None or group.barrier.current is None or (lease_id not in group.barrier.current['lease_ids']) or (set(group.barrier.acks) != {0, 16}):
                raise RuntimeError('source cannot be materialized before both DP32 node owners staged')

    def mark_materialized(self, lease_ids: tuple[str, ...]):
        self._require_node_wave_materialization(lease_ids)
        return self.leases.mark_materialized(lease_ids)

    def release(self, key: str, *, lease_ids: tuple[str, ...]) -> None:
        retained = self._retained.get(key)
        if retained is None:
            raise ValueError('key is not retained by this control owner')
        leases = tuple((self.leases.get(item) for item in lease_ids))
        if not leases or any((lease.key != key or lease.state.value != 'materialized' for lease in leases)):
            raise ValueError('all leases must be materialized for this key')
        self.leases.mark_payload_released(lease_ids)
        self._used_bytes -= retained[1]
        del self._retained[key]
        self._resume_if_low()

    def release_materialized_batch(self, *, keys, lease_ids, consumer_id):
        """Prevalidate the entire owned batch before releasing any quota."""
        from verl_speco.transport.waves import MAX_MATERIALIZED_WAVE_SAMPLES
        if not isinstance(keys, tuple) or not isinstance(lease_ids, tuple) or (not 1 <= len(keys) <= MAX_MATERIALIZED_WAVE_SAMPLES) or (len(keys) != len(lease_ids)) or any((not isinstance(key, str) or not key for key in keys)) or any((not isinstance(i, str) or not i for i in lease_ids)) or (len(set(keys)) != len(keys)) or (len(set(lease_ids)) != len(lease_ids)) or (not isinstance(consumer_id, str)) or (not consumer_id):
            raise ValueError('invalid bounded materialized release batch')
        leases = tuple((self.leases.get(i) for i in lease_ids))
        retained = tuple((self._retained.get(key) for key in keys))
        if any((item is None for item in retained)):
            raise ValueError('materialized release key is not retained by this owner')
        if any((lease.key != key or lease.consumer_id != consumer_id or lease.generation != self.generation or (lease.state is not LeaseState.MATERIALIZED) or lease.payload_released for key, lease in zip(keys, leases))):
            raise ValueError('materialized release lease owner or state differs')
        released_bytes = sum((item[1] for item in retained))
        if released_bytes > self._used_bytes:
            raise RuntimeError('materialized release exceeds retained accounting')
        self.leases.mark_payload_released(lease_ids)
        for key in keys:
            del self._retained[key]
        self._used_bytes -= released_bytes
        self._resume_if_low()

    def validate_commit(self, *, optimizer_step_id: str, transaction_set_hash: str, lease_ids: tuple[str, ...]) -> None:
        self._require_completed_node_wave_groups()
        self.leases.validate_commit(optimizer_step_id=optimizer_step_id, transaction_set_hash=transaction_set_hash, lease_ids=lease_ids)

    def _require_completed_node_wave_groups(self):
        for group in getattr(self, '_node_wave_groups', {}).values():
            if group.failed or group.barrier is None or group.barrier.current is not None or (group.barrier.next_wave != (len(group.samples) + group.wave_size - 1) // group.wave_size):
                raise RuntimeError('DP32 wave group incomplete or failed before optimizer commit')

    def commit(self, *, optimizer_step_id: str, transaction_set_hash: str, lease_ids: tuple[str, ...]):
        self._require_completed_node_wave_groups()
        return self.leases.commit(optimizer_step_id=optimizer_step_id, transaction_set_hash=transaction_set_hash, lease_ids=lease_ids)

    def cancel_leases(self, lease_ids: tuple[str, ...], *, reason: str):
        cancelled = self.leases.cancel(lease_ids, reason=reason)
        for lease in cancelled:
            if lease.key in self._retained:
                self._reclaim_pending[lease.lease_id] = lease.key
        return cancelled

    def reap_expired(self, *, now: float | None=None) -> tuple[str, ...]:
        """Reclaim explicitly cancelled or timed-out leases and clear their TQ payloads.

        Quota is released only after TQ confirms that each key is absent. A TQ
        error leaves the key pending so a later call can retry safely.
        """
        if not getattr(self, '_node_wave_safe_retirement', False):
            for group in getattr(self, '_node_wave_groups', {}).values():
                if group.failed or group.barrier is None or group.barrier.current is not None:
                    return ()
        expired = self.leases.expire(now=now)
        for lease in expired:
            if lease.key in self._retained:
                self._reclaim_pending[lease.lease_id] = lease.key
        if not self._reclaim_pending:
            return ()
        from verl_speco.runtime.services import import_transfer_queue
        tq = import_transfer_queue()
        tq.init()
        reclaimed: list[str] = []
        for lease_id, key in tuple(self._reclaim_pending.items()):
            present = tq.kv_list(partition_id=self.partition).get(self.partition, {})
            if key in present:
                tq.kv_clear(keys=key, partition_id=self.partition)
            present = tq.kv_list(partition_id=self.partition).get(self.partition, {})
            if key in present:
                raise RuntimeError(f'expired TQ payload remains present: {key}')
            retained = self._retained.pop(key, None)
            if retained is not None:
                self._used_bytes -= retained[1]
            del self._reclaim_pending[lease_id]
            reclaimed.append(key)
        self._resume_if_low()
        return tuple(reclaimed)

    def get_leases(self, lease_ids):
        return tuple((self.get_lease(i) for i in lease_registry_bounded_lease_ids(lease_ids)))

    def get_lease(self, lease_id: str):
        return self.leases.get(lease_id)

    def cleanup_failed_native_prefill(self, *, expected_samples, owned_exit_proof):
        from verl_speco.runtime.recovery import cleanup_failed_native_prefill
        from verl_speco.runtime.recovery import validate_owned_native_exit_proof
        validate_owned_native_exit_proof(owned_exit_proof, run_id=self.run_id, generation=self.generation)
        self._node_wave_safe_retirement = True
        try:
            return cleanup_failed_native_prefill(self, expected_samples=expected_samples, owned_exit_proof=owned_exit_proof)
        finally:
            self._node_wave_safe_retirement = False

    def snapshot_quiescent(self) -> dict:
        """Export metadata-only state only after all payloads are drained."""
        self._expire_reservations()
        if self._pending or self._retained or self._reclaim_pending or self._used_bytes:
            raise RuntimeError('control owner has active or uncertain payloads')
        lease_state = self.leases.snapshot()
        if any((item['state'] not in (LeaseState.COMMITTED.value, LeaseState.CANCELLED.value) for item in lease_state['leases'])):
            raise RuntimeError('control owner has an active lease')
        delivered = self.router.snapshot_quiescent()
        acquired = []
        for request, result, leases in self._acquired.values():
            acquired.append({'request': asdict(request), 'result': {'status': result.status, 'acquire_id': result.acquire_id, 'records': [{'key': record.key, 'producer_endpoint': record.producer_endpoint, 'metadata': record.metadata.to_dict(), 'payload_checksum': record.payload_checksum} for record in result.records], 'message': result.message, 'partial': result.partial}, 'lease_ids': [lease.lease_id for lease in leases]})
            required = self._acquired_required_keys.get(request.acquire_id)
            if required is not None:
                acquired[-1]['required_keys'] = list(required)
        return {'schema_version': 1, 'run_id': self.run_id, 'generation': self.generation, 'partition': self.partition, 'capacity_bytes': self.capacity_bytes, 'producer_incarnations': dict(self._producer_incarnations), 'delivered': delivered, 'leases': lease_state, 'acquired': acquired}

    def restore_quiescent(self, snapshot: dict) -> None:
        """Restore a complete safe-point snapshot into a new empty owner."""
        if snapshot.get('schema_version') != 1 or snapshot.get('run_id') != self.run_id or snapshot.get('generation') != self.generation or (snapshot.get('partition') != self.partition) or (snapshot.get('capacity_bytes') != self.capacity_bytes):
            raise ValueError('control snapshot identity differs')
        if self._pending or self._retained or self._acquired or self._used_bytes or self.leases.status()['leases_total']:
            raise RuntimeError('control owner must be empty before restore')
        lease_snapshot = snapshot['leases']
        producer_incarnations = snapshot['producer_incarnations']
        if not isinstance(producer_incarnations, dict) or any((not str(key) or not str(value) for key, value in producer_incarnations.items())):
            raise ValueError('control snapshot has invalid producer incarnations')
        if any((item['state'] not in (LeaseState.COMMITTED.value, LeaseState.CANCELLED.value) for item in lease_snapshot['leases'])):
            raise ValueError('control snapshot contains an active lease')
        if any((int(item['generation']) != self.generation for item in lease_snapshot['leases'])):
            raise ValueError('control snapshot contains a lease from another generation')
        for item in snapshot['delivered']:
            metadata = SampleMetadataV3.from_dict(item['metadata'])
            if metadata.run_id != self.run_id or metadata.generation != self.generation or producer_incarnations.get(metadata.producer_id) != metadata.producer_incarnation:
                raise ValueError('control snapshot delivered producer identity differs')
        router = ReadyRouter()
        router.restore_quiescent(snapshot['delivered'])
        leases_owner = LeaseRegistry()
        leases_owner.restore(lease_snapshot)
        restored: dict[str, tuple[AcquireRequest, AcquireResult, tuple[LeaseRecord, ...]]] = {}
        restored_required = {}
        from verl_speco.transport.waves import required_wave_keys
        for raw in snapshot['acquired']:
            request = AcquireRequest(**raw['request'])
            request.validate()
            if request.generation != self.generation:
                raise ValueError('control snapshot acquire generation differs')
            raw_required = raw.get('required_keys')
            if raw_required is not None and (not isinstance(raw_required, list)):
                raise ValueError('control snapshot exact wave keys must be a list')
            required = required_wave_keys(request, tuple(raw_required) if raw_required is not None else None)
            result_raw = raw['result']
            records = tuple((ReadyRecord(key=str(item['key']), producer_endpoint=str(item['producer_endpoint']), metadata=SampleMetadataV3.from_dict(item['metadata']), payload_checksum=str(item['payload_checksum'])) for item in result_raw['records']))
            if required is not None and (len(records) != len(required) or set(required) != {record.key for record in records}):
                raise ValueError('control snapshot exact wave keys differ from acquired records')
            for record in records:
                record.validate()
                if record.metadata.run_id != self.run_id or record.metadata.generation != self.generation or producer_incarnations.get(record.metadata.producer_id) != record.metadata.producer_incarnation:
                    raise ValueError('control snapshot acquired producer identity differs')
            result = AcquireResult(status=str(result_raw['status']), acquire_id=str(result_raw['acquire_id']), records=records, message=str(result_raw['message']), partial=bool(result_raw['partial']))
            if result.acquire_id != request.acquire_id or result.status != 'ready':
                raise ValueError('control snapshot acquired result is incompatible')
            leases = tuple((leases_owner.get(str(item)) for item in raw['lease_ids']))
            if len(leases) != len(records) or any((lease.key != record.key or lease.acquire_id != request.acquire_id or lease.generation != self.generation or (lease.target_revision != record.metadata.target_revision) for lease, record in zip(leases, records))):
                raise ValueError('control snapshot acquired leases do not match records')
            if request.acquire_id in restored:
                raise ValueError('control snapshot duplicates an acquire ID')
            restored[request.acquire_id] = (request, result, leases)
            restored_required[request.acquire_id] = required
        self.router = router
        self.leases = leases_owner
        self._acquired = restored
        self._acquired_required_keys = restored_required
        self._producer_incarnations = {str(key): str(value) for key, value in producer_incarnations.items()}

    def save_quiescent(self, path: str) -> str:
        """Atomically fsync a safe-point snapshot; returns its SHA-256 digest."""
        snapshot = self.snapshot_quiescent()
        payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode('utf-8')
        digest = hashlib.sha256(payload).hexdigest()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix='.control-', suffix='.tmp', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
            from verl_speco.runtime.artifacts import sync_directory
            sync_directory(target.parent, allow_unsupported=False)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return digest

    def save_quiescent_bundle(self, path: str) -> dict:
        """Return the saved metadata so a trainer can persist its local replica."""
        digest = self.save_quiescent(path)
        payload = Path(path).read_bytes()
        if len(payload) > 16 * 1024 * 1024:
            raise RuntimeError('control snapshot exceeds the metadata handoff limit')
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError('saved control snapshot changed before handoff')
        return {'sha256': digest, 'payload': payload}

    def save_quiescent_shared(self, *, directory: str, step_index: int) -> dict:
        from verl_speco.runtime.services import save_shared_control_snapshot
        return save_shared_control_snapshot(self.snapshot_quiescent(), directory=directory, step_index=step_index, run_id=self.run_id, generation=self.generation)

    def save_committed_prefill_shared(self, *, directory, step_index, optimizer_step_id, committed_cursor, committed_lease_ids, committed_sample_keys, prefill_geometry=None):
        from verl_speco.transport.waves import committed_prefill_clone
        clone, projection = committed_prefill_clone(self, optimizer_step_id=optimizer_step_id, committed_cursor=committed_cursor, committed_lease_ids=committed_lease_ids, committed_sample_keys=committed_sample_keys, prefill_geometry=prefill_geometry)
        reference = clone.save_quiescent_shared(directory=directory, step_index=step_index)
        return dict(snapshot=reference, future_prefill_projection=projection)

    def save_committed_prefill_bundle(self, path, *, optimizer_step_id, committed_cursor, committed_lease_ids, committed_sample_keys, prefill_geometry=None):
        from verl_speco.transport.waves import committed_prefill_clone
        clone, projection = committed_prefill_clone(self, optimizer_step_id=optimizer_step_id, committed_cursor=committed_cursor, committed_lease_ids=committed_lease_ids, committed_sample_keys=committed_sample_keys, prefill_geometry=prefill_geometry)
        bundle = clone.save_quiescent_bundle(path)
        bundle['future_prefill_projection'] = projection
        return bundle

    def load_quiescent(self, path: str, *, expected_sha256: str) -> None:
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError('control snapshot checksum mismatch')
        self.restore_quiescent(json.loads(payload))

    def status(self) -> dict[str, object]:
        self._expire_reservations()
        return {'run_id': self.run_id, 'generation': self.generation, 'partition': self.partition, 'used_bytes': self._used_bytes, 'capacity_bytes': self.capacity_bytes, 'paused': self._paused, 'pending_reservations': len(self._pending), 'remembered_acquires': len(self._acquired), 'producer_incarnations': len(self._producer_incarnations), 'reclaim_pending': len(self._reclaim_pending), 'retained_items': len(self._retained), 'router': self.router.status(), 'leases': self.leases.status()}

    def _expire_reservations(self) -> None:
        now = time.monotonic()
        for key, pending in tuple(self._pending.items()):
            if pending[3] <= now:
                self._used_bytes -= pending[2]
                del self._pending[key]
        self._resume_if_low()

    def _resume_if_low(self) -> None:
        if self._used_bytes <= self._low:
            self._paused = False

class RayLeaseProxy:
    """Synchronous LeaseRegistry interface used by CommitExecutor on rank zero."""

    def __init__(self, actor) -> None:
        self.actor = actor

    def validate_commit(self, **kwargs) -> None:
        import ray
        ray.get(self.actor.validate_commit.remote(**kwargs))

    def commit(self, **kwargs):
        import ray
        return ray.get(self.actor.commit.remote(**kwargs))

    def cancel(self, lease_ids, *, reason: str):
        import ray
        return ray.get(self.actor.cancel_leases.remote(tuple(lease_ids), reason=reason))

    def get_many(self, lease_ids):
        import ray
        return ray.get(self.actor.get_leases.remote(lease_registry_bounded_lease_ids(lease_ids)))

    def get(self, lease_id: str):
        import ray
        return ray.get(self.actor.get_lease.remote(lease_id))

def actor_name_for_run(run_id: str, generation: int) -> str:
    if not run_id or generation < 0:
        raise ValueError('run ID and non-negative generation are required')
    return 'SpeCoReady-' + hashlib.sha256(f'{run_id}\x00{generation}'.encode('utf-8')).hexdigest()[:20]
from copy import deepcopy

def control_snapshot_sha256(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

class MultiNodeJointReadyControl(RayReadyControl):

    def __init__(self, **kwargs):
        if type(kwargs.get('generation')) is not int or kwargs['generation'] < 0:
            raise ValueError('joint control generation must be a non-negative integer')
        super().__init__(**kwargs)
        self._joint_parents = []
        self._parent_leases = {}
        self._parent_steps = {}

    def _validated_history(self, snapshots):
        if not isinstance(snapshots, list):
            raise ValueError('joint control history must be a list')
        parents = []
        leases = {}
        steps = {}
        previous = -1
        for snapshot in snapshots:
            generation = snapshot.get('generation')
            if type(generation) is not int or not previous < generation < self.generation or snapshot.get('committed_parents'):
                raise ValueError('joint parent generations must be distinct, ordered and historical')
            previous = generation
            if snapshot.get('run_id') != self.run_id or snapshot.get('partition') != self.partition or snapshot.get('capacity_bytes') != self.capacity_bytes:
                raise ValueError('joint parent control identity differs')
            for row in snapshot['leases']['leases']:
                if type(row.get('generation')) is not int or row['generation'] != generation or type(row.get('payload_released')) is not bool:
                    raise ValueError('joint parent lease has an invalid generation or release flag')
            parent = RayReadyControl(run_id=self.run_id, generation=generation, partition=self.partition, capacity_bytes=self.capacity_bytes)
            parent.restore_quiescent(deepcopy(snapshot))
            for row in snapshot['leases']['leases']:
                lease = parent.get_lease(row['lease_id'])
                if lease.lease_id in leases:
                    raise ValueError('joint history reuses a lease identity')
                leases[lease.lease_id] = lease
            for step, value in snapshot['leases']['committed_steps'].items():
                if step in steps:
                    raise ValueError('joint history reuses an optimizer step identity')
                steps[step] = deepcopy(value)
            parents.append(deepcopy(snapshot))
        return (parents, leases, steps)

    def restore_joint_parent(self, snapshot, *, expected_sha256):
        """Install a verified drained parent into a fresh generation owner."""
        if control_snapshot_sha256(snapshot) != expected_sha256:
            raise ValueError('joint parent snapshot checksum differs')
        if self._joint_parents or self._pending or self._retained or self._acquired or self._producer_incarnations or self._used_bytes or self.leases.status()['leases_total']:
            raise RuntimeError('joint parent handoff requires a fresh empty control owner')
        current = deepcopy(snapshot)
        history = current.pop('committed_parents', []) + [current]
        parents, leases, steps = self._validated_history(history)
        self._joint_parents, self._parent_leases, self._parent_steps = (parents, leases, steps)
        return dict(restored=True, run_id=self.run_id, generation=self.generation, parent_generation=current['generation'], parent_sha256=expected_sha256, historical_lease_count=len(leases), historical_step_count=len(steps))

    def snapshot_quiescent(self):
        snapshot = super().snapshot_quiescent()
        if self._joint_parents:
            snapshot['committed_parents'] = deepcopy(self._joint_parents)
        return snapshot

    def restore_quiescent(self, snapshot):
        if type(snapshot.get('generation')) is not int:
            raise ValueError('joint current control generation must remain an integer')
        if self._joint_parents:
            raise RuntimeError('joint control owner already has committed parent history')
        current = deepcopy(snapshot)
        parents, leases, steps = self._validated_history(current.pop('committed_parents', []))
        if set(leases) & {row['lease_id'] for row in current['leases']['leases']} or set(steps) & set(current['leases']['committed_steps']):
            raise ValueError('current control reuses a historical lease or optimizer step')
        super().restore_quiescent(current)
        self._joint_parents, self._parent_leases, self._parent_steps = (parents, leases, steps)

    def get_lease(self, lease_id):
        if lease_id in self._parent_leases:
            return self._parent_leases[lease_id]
        return super().get_lease(lease_id)

    def validate_commit(self, *, optimizer_step_id, transaction_set_hash, lease_ids):
        parent = self._parent_steps.get(optimizer_step_id)
        if parent is not None:
            if parent['transaction_set_hash'] != transaction_set_hash or tuple(sorted(set(lease_ids))) != tuple(sorted(parent['lease_ids'])):
                raise ValueError('historical optimizer step was reused with different transactions')
            return
        if set(lease_ids) & self._parent_leases.keys():
            raise ValueError('historical leases cannot create a new optimizer step')
        return super().validate_commit(optimizer_step_id=optimizer_step_id, transaction_set_hash=transaction_set_hash, lease_ids=lease_ids)

    def commit(self, *, optimizer_step_id, transaction_set_hash, lease_ids):
        self.validate_commit(optimizer_step_id=optimizer_step_id, transaction_set_hash=transaction_set_hash, lease_ids=lease_ids)
        if optimizer_step_id in self._parent_steps:
            return CommitReceipt(optimizer_step_id, transaction_set_hash, tuple(sorted(set(lease_ids))), already_committed=True)
        return super().commit(optimizer_step_id=optimizer_step_id, transaction_set_hash=transaction_set_hash, lease_ids=lease_ids)
