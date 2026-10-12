# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import hashlib
import json
from typing import Any
import torch
from verl_speco.producer.target_capture import Captured92BFeatures
from verl_speco.transport.control import BoundedFeaturePublisher, PublishResult
from verl_speco.transport.streaming import SampleMetadataV3

def capture_publication_capacity(*, dataset_mode: bool, configured_bytes: str='536870912') -> int:
    """Fixed publisher quota across varying replay lengths in one worker."""
    capacity = int(configured_bytes) if dataset_mode else 4 * 1024 * 1024
    if capacity <= 0:
        raise ValueError('capture publication capacity must be positive')
    return capacity

def capture_fields(captured: Captured92BFeatures, *, loss_mask: tuple[float, ...], device_concat: bool=False) -> dict[str, torch.Tensor]:
    """Build the exact DSpark aux-plus-last feature layout in CPU memory."""
    if type(device_concat) is not bool:
        raise ValueError('capture device concat must be boolean')
    rows = len(captured.token_ids)
    if len(loss_mask) != rows or any((value not in (0.0, 1.0) for value in loss_mask)):
        raise ValueError('loss mask must contain one binary value per captured token')
    tensors = (*captured.hidden_states, captured.final_hidden_state)
    if not tensors or any((t.ndim != 2 or t.shape[0] != rows for t in tensors)):
        raise ValueError('captured hidden rows are inconsistent')
    width = tensors[0].shape[1]
    if any((t.shape[1] != width or t.dtype != torch.bfloat16 for t in tensors)):
        raise ValueError('all captured hidden tensors must have equal width and bfloat16 dtype')
    if device_concat:
        if any((tensor.device != tensors[0].device for tensor in tensors)):
            raise ValueError('device concat requires one capture device')
        hidden = torch.cat([tensor.detach() for tensor in tensors], dim=-1).to('cpu').contiguous()
    else:
        hidden = torch.cat([tensor.detach().to('cpu').contiguous() for tensor in tensors], dim=-1)
    return {'input_ids': torch.tensor(captured.token_ids, dtype=torch.int64), 'position_ids': torch.tensor(captured.positions, dtype=torch.int64), 'loss_mask': torch.tensor(loss_mask, dtype=torch.float32), 'hidden_states': hidden}

def payload_checksum(fields: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(fields):
        tensor = fields[name]
        if tensor.device.type != 'cpu' or not tensor.is_contiguous():
            raise ValueError('TQ capture payload must be contiguous CPU tensors')
        descriptor = json.dumps((name, str(tensor.dtype), list(tensor.shape)), separators=(',', ':'))
        digest.update(descriptor.encode('utf-8'))
        digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()

def publish_captured_features(captured: Captured92BFeatures, *, loss_mask: tuple[float, ...], metadata: SampleMetadataV3, publisher: BoundedFeaturePublisher) -> PublishResult:
    validate_capture_metadata(captured, metadata)
    fields = capture_fields(captured, loss_mask=loss_mask)
    actual_bytes = sum((t.numel() * t.element_size() for t in fields.values()))
    if metadata.payload_nbytes != actual_bytes:
        raise ValueError(f'v3 payload byte count {metadata.payload_nbytes} != {actual_bytes}')
    return publisher.publish(metadata, fields=fields, payload_checksum=payload_checksum(fields))

def validate_capture_metadata(captured: Captured92BFeatures, metadata: SampleMetadataV3) -> None:
    """Check capture identity and DSpark layout before any TQ mutation."""
    if metadata.request_id != captured.request_id:
        raise ValueError('v3 request ID differs from captured request')
    if metadata.target_layer_ids != captured.layer_ids:
        raise ValueError('v3 target layer IDs differ from captured layers')
    if metadata.token_count != len(captured.token_ids):
        raise ValueError('v3 token count differs from captured rows')
    if metadata.hidden_states_layout != 'dflash_aux_plus_last' or metadata.hidden_dtype != 'bfloat16':
        raise ValueError('DSpark capture requires dflash_aux_plus_last bfloat16 layout')
