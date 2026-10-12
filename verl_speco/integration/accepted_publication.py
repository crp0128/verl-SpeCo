# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import os
from verl_speco.producer.feature_payload import capture_fields, payload_checksum, validate_capture_metadata
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key

def publish_accepted_query(worker, *, metadata, loss_mask, partition, ray_address, capacity_bytes, control_actor_name):
    v3 = SampleMetadataV3.from_dict(metadata)
    probe = getattr(worker, '_speco_accepted_probe', None)
    features = None if probe is None else probe.finished_features
    if features is None:
        raise RuntimeError('accepted target features must be finished before publication')
    if os.environ.get('SPECO_NATIVE_FEATURE_TRANSPORT') != '1' or os.environ.get('SPECO_RUN_ID') != v3.run_id or int(os.environ.get('SPECO_GENERATION', '-1')) != v3.generation:
        raise ValueError('online features require owned native TQ transport')
    if not partition or not control_actor_name or (not ray_address) or (type(capacity_bytes) is not int) or (capacity_bytes < v3.payload_nbytes):
        raise ValueError('online features require bounded shared control and explicit partition')
    validate_capture_metadata(features, v3)
    mask = tuple(loss_mask)
    if v3.prompt_token_count is None or v3.prompt_token_count < 1 or mask != (0.0,) * v3.prompt_token_count + (1.0,) * v3.supervised_token_count:
        raise ValueError('online response mask must match the committed prompt boundary')
    fields = capture_fields(features, loss_mask=mask, device_concat=True)
    if sum((t.numel() * t.element_size() for t in fields.values())) != v3.payload_nbytes:
        raise ValueError('accepted query payload size differs from committed metadata')
    checksum = payload_checksum(fields)
    receipt = dict(rank=int(worker.rank), key=make_v3_sample_key(v3), request_id=v3.request_id, run_id=v3.run_id, generation=v3.generation, payload_checksum=checksum, payload_nbytes=v3.payload_nbytes, data_version=v3.data_version, target_revision=v3.target_revision, draft_revision=v3.draft_revision, producer_incarnation=v3.producer_incarnation)
    if worker.rank != 0:
        return dict(receipt, status='rank_skipped')
    from verl_speco.integration.multinode_online import online_publisher_context
    context = online_publisher_context(worker)
    publisher = context._speco_get_tq_publisher(partition=partition, ray_address=ray_address, capacity_bytes=capacity_bytes, producer_endpoint=v3.producer_id, control_actor_name=control_actor_name)
    published = publisher.publish(v3, fields=fields, payload_checksum=checksum)
    if published.key != receipt['key'] or published.payload_checksum != checksum:
        raise RuntimeError('native publisher returned a foreign accepted query identity')
    descriptor = context._speco_native_store.publication_descriptor(published.key, partition)
    return dict(receipt, status=published.status, native_descriptor=descriptor)

def require_accepted_publication_consensus(rows, *, metadata, expected_ranks):
    """Do not release private features or expose collection before TP consensus."""
    v3 = SampleMetadataV3.from_dict(metadata)
    if not isinstance(rows, list) or len(rows) != len(expected_ranks) or any((not isinstance(row, dict) for row in rows)) or ({row.get('rank') for row in rows} != set(expected_ranks)) or (0 not in expected_ranks):
        raise RuntimeError('accepted publication has missing, duplicate or foreign TP ranks')
    checksums = {row.get('payload_checksum') for row in rows}
    if len(checksums) != 1 or any((not isinstance(sha, str) or len(sha) != 64 or any((c not in '0123456789abcdef' for c in sha)) for sha in checksums)):
        raise RuntimeError('accepted query feature checksum differs across TP ranks')
    expected = dict(key=make_v3_sample_key(v3), request_id=v3.request_id, run_id=v3.run_id, generation=v3.generation, payload_nbytes=v3.payload_nbytes, data_version=v3.data_version, target_revision=v3.target_revision, draft_revision=v3.draft_revision, producer_incarnation=v3.producer_incarnation)
    for row in rows:
        if any((row.get(key) != value for key, value in expected.items())):
            raise RuntimeError('accepted publication metadata differs across TP ranks')
        if row['rank'] == 0:
            descriptor = row.get('native_descriptor')
            if row.get('status') not in ('published', 'duplicate') or not isinstance(descriptor, dict) or descriptor.get('lease_id') != expected['key'] or (descriptor.get('run_id') != v3.run_id) or (descriptor.get('generation') != v3.generation):
                raise RuntimeError('accepted query rank zero did not retain owned native publication')
        elif row.get('status') != 'rank_skipped' or 'native_descriptor' in row:
            raise RuntimeError('accepted query was published by more than one TP rank')
    return dict(expected, payload_checksum=next(iter(checksums)))
