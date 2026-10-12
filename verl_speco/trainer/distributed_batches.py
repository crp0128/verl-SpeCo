# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import torch

def plan_response_starts_with_fill(*, region_starts: torch.Tensor, region_ends: torch.Tensor, k_blocks: int, block_size: int, allow_fill: bool=True, generator: torch.Generator | None=None) -> torch.Tensor:
    """Return CPU label starts, with -1 for unavailable block slots.

    Regions must already delimit supervised text within one document. These
    are label starts, not DSpark anchor indices; DSpark's seed is start - 1.
    Planning never transfers hidden states or quietly synchronizes an NPU.
    """
    if type(k_blocks) is not int or k_blocks < 0 or type(block_size) is not int or (block_size <= 0) or (type(allow_fill) is not bool):
        raise ValueError('block size, slot count and fill mode are invalid')
    for tensor in (region_starts, region_ends):
        if tensor.device.type != 'cpu' or tensor.dtype != torch.int64 or tensor.ndim != 1:
            raise ValueError('response regions must be one-dimensional CPU int64 metadata')
    if region_starts.shape != region_ends.shape or (region_starts < 0).any() or (region_ends < region_starts).any():
        raise ValueError('response region bounds differ or are invalid')
    if generator is not None and generator.device.type != 'cpu':
        raise ValueError('response planning requires a CPU generator')
    aligned = region_starts[:, None] + torch.arange(k_blocks, device='cpu')[None, :] * block_size
    starts = torch.where(aligned + block_size <= region_ends[:, None], aligned, -1)
    if not allow_fill or k_blocks == 0:
        return starts
    for row, (lo, hi) in enumerate(zip(region_starts.tolist(), region_ends.tolist())):
        chosen = starts[row][starts[row] >= 0]
        needed = k_blocks - chosen.numel()
        legal_count = hi - lo - block_size + 1
        if needed <= 0 or legal_count <= 0:
            continue
        available = torch.ones(legal_count, dtype=torch.bool, device='cpu')
        available[chosen - lo] = False
        candidates = available.nonzero(as_tuple=True)[0]
        take = min(needed, candidates.numel())
        if take:
            indices = torch.randperm(candidates.numel(), generator=generator, device='cpu')[:take]
            starts[row, chosen.numel():chosen.numel() + take] = lo + candidates[indices]
    return starts

def plan_dspark_response_anchors(*, loss_mask: torch.Tensor, document_ids: torch.Tensor | None, num_anchors: int, block_size: int, generator: torch.Generator | None=None):
    """Plan DSpark seed positions on CPU, keeping full label blocks in one region.

    A prompt token may seed the first supervised response block. The seed and
    every label must belong to one document. Multiple response regions share
    one window's slot budget: aligned starts first, unused legal starts second.
    """
    if loss_mask.device.type != 'cpu' or loss_mask.ndim != 2:
        raise ValueError('anchor planning requires two-dimensional CPU loss metadata')
    if not torch.isfinite(loss_mask).all() or ((loss_mask < 0) | (loss_mask > 1)).any():
        raise ValueError('anchor loss mask must be finite and within zero and one')
    if document_ids is not None and (document_ids.device.type != 'cpu' or document_ids.dtype != torch.int64 or document_ids.shape != loss_mask.shape):
        raise ValueError('anchor documents must match CPU int64 loss metadata')
    empty = torch.empty(0, dtype=torch.int64, device='cpu')
    plan_response_starts_with_fill(region_starts=empty, region_ends=empty, k_blocks=num_anchors, block_size=block_size, generator=generator)
    seeds = torch.zeros(loss_mask.shape[0], num_anchors, dtype=torch.int64, device='cpu')
    keep = torch.zeros_like(seeds, dtype=torch.bool)
    for row in range(loss_mask.shape[0]):
        mask = (loss_mask[row] > 0.5).tolist()
        docs = document_ids[row].tolist() if document_ids is not None else [0] * len(mask)
        regions = []
        pos = 1
        while pos < len(mask):
            if not mask[pos] or docs[pos] < 0 or docs[pos - 1] != docs[pos]:
                pos += 1
                continue
            lo = pos
            while pos < len(mask) and mask[pos] and (docs[pos] == docs[lo]):
                pos += 1
            if pos - lo >= block_size:
                regions.append((lo, pos))
        if len(regions) == 1:
            lo, hi = regions[0]
            starts = plan_response_starts_with_fill(region_starts=torch.tensor([lo], dtype=torch.int64, device='cpu'), region_ends=torch.tensor([hi], dtype=torch.int64, device='cpu'), k_blocks=num_anchors, block_size=block_size, generator=generator)[0]
            selected = starts[starts >= 0]
        else:
            aligned = [p for lo, hi in regions for p in range(lo, hi - block_size + 1, block_size)]
            selected = torch.tensor(aligned[:num_anchors], dtype=torch.int64, device='cpu')
            if selected.numel() < num_anchors:
                used = set(selected.tolist())
                unused = torch.tensor([p for lo, hi in regions for p in range(lo, hi - block_size + 1) if p not in used], dtype=torch.int64, device='cpu')
                order = torch.randperm(unused.numel(), generator=generator, device='cpu')
                selected = torch.cat((selected, unused[order[:num_anchors - selected.numel()]]))
        count = selected.numel()
        seeds[row, :count] = selected - 1
        keep[row, :count] = True
    return (seeds, keep)

def prepare_dmtp_anchor_window_shard(shard, *, num_anchors: int, block_size: int, seed: int):
    """Attach CPU plans to owned windows before their accelerator transfer."""
    from dataclasses import replace
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError('anchor planning seed must be a nonnegative integer')
    windows = []
    for ordinal, window in enumerate(shard.windows):
        batch = dict(window.batch)
        generator = torch.Generator(device='cpu').manual_seed((seed + ordinal) % 2 ** 63)
        positions, keep = plan_dspark_response_anchors(loss_mask=batch['loss_mask'], document_ids=batch.get('document_ids'), num_anchors=num_anchors, block_size=block_size, generator=generator)
        batch['preplanned_anchor_positions'] = positions
        batch['preplanned_anchor_keep_mask'] = keep
        windows.append(replace(window, batch=batch))
    return replace(shard, windows=tuple(windows))

def resolve_num_swa_chunks(*, seq_length: int, draft_train_length: int, num_swa_chunk: int | None=None) -> int:
    if any((type(v) is not int or v <= 0 for v in (seq_length, draft_train_length))):
        raise ValueError('SWA sequence and draft training lengths must be positive integers')
    if num_swa_chunk is not None:
        if type(num_swa_chunk) is not int or num_swa_chunk <= 0:
            raise ValueError('explicit SWA chunk count must be a positive integer')
        return num_swa_chunk
    if seq_length % draft_train_length:
        raise ValueError('default SWA count requires divisible sequence and training lengths')
    return seq_length // draft_train_length

def choose_packed_num_swa_chunks(*, response_length: torch.Tensor | list[int], draft_train_length: int, block_size: int, num_swa_chunk_cap: int, min_swa_blocks: int=1, cp_size: int=1) -> int | None:
    """Choose a feasible capped count; a short response needs a whole block.

    CP alignment is reference-planner arithmetic only. The standalone draft
    trainer remains DP-only. No implicit metadata or hidden-state D2H occurs.
    """
    if any((type(v) is not int or v <= 0 for v in (draft_train_length, block_size, num_swa_chunk_cap, min_swa_blocks, cp_size))):
        raise ValueError('SWA lengths, cap and alignment must be positive integers')
    if isinstance(response_length, torch.Tensor):
        if response_length.device.type != 'cpu' or response_length.dtype != torch.int64 or response_length.ndim != 1:
            raise ValueError('response lengths must be one-dimensional CPU int64 metadata')
        lengths = response_length.tolist()
    elif isinstance(response_length, list):
        lengths = response_length
    else:
        raise ValueError('response lengths must be CPU metadata or an integer list')
    if any((type(v) is not int or v < 0 for v in lengths)):
        raise ValueError('response lengths must be nonnegative integers')
    minimum = block_size * min_swa_blocks
    eligible = [v for v in lengths if v >= minimum]
    if not eligible:
        return None
    maximum = sum((v // minimum for v in eligible))
    desired = min(max((sum(eligible) + draft_train_length - 1) // draft_train_length, len(eligible)), maximum)
    aligned = (desired + cp_size - 1) // cp_size * cp_size
    chunks = min(num_swa_chunk_cap, aligned)
    if chunks > maximum:
        chunks = maximum // cp_size * cp_size
    if chunks <= 0 or chunks < len(eligible):
        return None
    return chunks

def enumerate_packed_response_parts(*, response_length: torch.Tensor, seq_length: int, draft_train_length: int, block_size: int, num_swa_chunk: int | None=None, min_swa_blocks: int=1, adaptive_num_swa_chunk: bool=False, sample_batch_indices: torch.Tensor | None=None):
    """Return CPU response-relative starts, exclusive ends and logical sample IDs.

    Allocate one region per eligible document, then give each extra region to
    the document with the largest current average region length. Remainders
    go to earlier parts, preserving every eligible response token exactly once.
    """
    cap = resolve_num_swa_chunks(seq_length=seq_length, draft_train_length=draft_train_length, num_swa_chunk=num_swa_chunk)
    if type(adaptive_num_swa_chunk) is not bool:
        raise ValueError('adaptive SWA mode must be boolean')
    if not isinstance(response_length, torch.Tensor):
        raise ValueError('packed response lengths must be CPU int64 metadata')
    choose_packed_num_swa_chunks(response_length=response_length, draft_train_length=draft_train_length, block_size=block_size, num_swa_chunk_cap=cap, min_swa_blocks=min_swa_blocks)
    lengths = response_length.tolist()
    groups = [list(range(len(lengths)))]
    if sample_batch_indices is not None:
        if not isinstance(sample_batch_indices, torch.Tensor) or sample_batch_indices.device.type != 'cpu' or sample_batch_indices.dtype != torch.int64 or (sample_batch_indices.shape != response_length.shape) or (not lengths) or (sample_batch_indices < 0).any():
            raise ValueError('packed row IDs must match nonempty CPU int64 response metadata')
        physical = sample_batch_indices.tolist()
        unique_rows = sorted(set(physical))
        if unique_rows[0] != 0 or unique_rows[-1] + 1 != len(unique_rows):
            raise ValueError('packed physical rows must be contiguous and nonempty')
        groups = [[i for i, value in enumerate(physical) if value == row] for row in unique_rows]
    starts, ends, samples = ([], [], [])
    empty = torch.empty(0, dtype=torch.int64, device='cpu')
    minimum = block_size * min_swa_blocks
    for group in groups:
        row_lengths = [lengths[i] for i in group]
        chunks = choose_packed_num_swa_chunks(response_length=row_lengths, draft_train_length=draft_train_length, block_size=block_size, num_swa_chunk_cap=cap, min_swa_blocks=min_swa_blocks) if adaptive_num_swa_chunk else cap
        eligible = [(i, lengths[i]) for i in group if lengths[i] >= minimum]
        if chunks is None or not eligible or len(eligible) > chunks or (sum((length // minimum for _, length in eligible)) < chunks):
            return (empty, empty, empty)
        counts = [1] * len(eligible)
        for _ in range(chunks - len(eligible)):
            candidates = [i for i, (_, length) in enumerate(eligible) if counts[i] < length // minimum]
            best = max(candidates, key=lambda i: eligible[i][1] / counts[i])
            counts[best] += 1
        for (sample, length), count in zip(eligible, counts):
            base, remainder = divmod(length, count)
            cursor = 0
            for part in range(count):
                width = base + int(part < remainder)
                starts.append(cursor)
                ends.append(cursor + width)
                samples.append(sample)
                cursor += width
    return tuple((torch.tensor(values, dtype=torch.int64, device='cpu') for values in (starts, ends, samples)))
from dataclasses import dataclass, replace
import math

@dataclass(frozen=True)
class DMTPResponsePart:
    document_id: int
    response_start: int
    response_end: int
    context_start: int
    context_end: int
    batch: dict

@dataclass(frozen=True)
class DMTPPartShard:
    parts: tuple[DMTPResponsePart, ...]
    part_indices: tuple[int, ...]
    global_part_count: int
    loss_scale: float

def dmtp_swa_windows_slice_batch(batch, start, end):
    shape = batch['input_ids'].shape
    result = {}
    for name, value in batch.items():
        if name == 'lm_head_weight':
            result[name] = value
        elif name == 'hidden_states_list':
            result[name] = [tensor[:, start:end].clone() for tensor in value]
        elif isinstance(value, torch.Tensor) and value.ndim >= 2 and (value.shape[:2] == shape):
            result[name] = value[:, start:end].clone()
        else:
            raise ValueError('unsupported field in SWA batch: ' + name)
    return result

def dmtp_swa_response_parts(batch, *, seq_length, draft_train_length=2048, block_size=8, chunk_cap=32, window_size=2048, seed=42):
    """Prepare full-block anchor labels and preceding halo on CPU before H2D.

    Each document has a prompt followed by one contiguous supervised response.
    Padding uses document -1 and zero loss. An absent predecessor is rejected
    rather than shifting or dropping the first response label. All output rows
    own their compact storage; the separately resident head is shared.
    """
    if type(window_size) is not int or window_size < 1 or type(seed) is not int or (not 0 <= seed < 2 ** 63) or (type(draft_train_length) is not int) or (type(block_size) is not int) or (block_size < 1) or (draft_train_length < block_size) or draft_train_length % block_size:
        raise ValueError('invalid SWA context, seed or block-slot budget')
    ids, mask, docs = (batch[name] for name in ('input_ids', 'loss_mask', 'document_ids'))
    if ids.device.type != 'cpu' or ids.dtype != torch.int64 or ids.ndim != 2 or (ids.shape[0] != 1) or (mask.device.type != 'cpu') or (mask.shape != ids.shape) or (docs.device.type != 'cpu') or (docs.dtype != torch.int64) or (docs.shape != ids.shape) or (not torch.isfinite(mask).all()) or ((mask < 0) | (mask > 1)).any() or (docs < -1).any() or ((docs < 0) & (mask > 0.5)).any():
        raise ValueError('SWA requires one CPU packed stream and valid document/loss metadata')
    for name, value in batch.items():
        tensors = value if name == 'hidden_states_list' else [value]
        for tensor in tensors:
            if name != 'lm_head_weight' and (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.ndim < 2 or (tensor.shape[:2] != ids.shape)):
                raise ValueError('SWA features and positions must match the CPU stream')
    segments, seen = ([], set())
    cursor = 0
    while cursor < ids.shape[1]:
        doc = int(docs[0, cursor])
        if doc < 0:
            cursor += 1
            continue
        if doc in seen:
            raise ValueError('packed document IDs must describe contiguous segments')
        seen.add(doc)
        end = cursor + 1
        while end < ids.shape[1] and int(docs[0, end]) == doc:
            end += 1
        supervised = (mask[0, cursor:end] > 0.5).nonzero(as_tuple=True)[0]
        if supervised.numel():
            prompt = int(supervised[0])
            if prompt == 0 or not (mask[0, cursor + prompt:end] > 0.5).all():
                raise ValueError('SWA requires a predecessor and one contiguous response suffix')
            segments.append((doc, cursor, cursor + prompt, end))
        cursor = end
    lengths = torch.tensor([end - start for _, _, start, end in segments], dtype=torch.int64, device='cpu')
    lo, hi, samples = enumerate_packed_response_parts(response_length=lengths, seq_length=seq_length, draft_train_length=draft_train_length, block_size=block_size, num_swa_chunk=chunk_cap, adaptive_num_swa_chunk=True)
    if not samples.numel():
        raise ValueError('packed response has no feasible SWA part budget')
    labels = plan_response_starts_with_fill(region_starts=lo, region_ends=hi, k_blocks=draft_train_length // block_size, block_size=block_size, generator=torch.Generator(device='cpu').manual_seed(seed))
    parts = []
    for index, sample in enumerate(samples.tolist()):
        doc, doc_start, response_start, _ = segments[sample]
        start, end = (response_start + int(lo[index]), response_start + int(hi[index]))
        context_start = max(doc_start, start - window_size - 1)
        part = dmtp_swa_windows_slice_batch(batch, context_start, end)
        part['loss_mask'][:, :start - context_start] = 0
        keep = labels[index:index + 1] >= 0
        positions = labels[index:index + 1] + response_start - 1 - context_start
        part['preplanned_anchor_positions'] = torch.where(keep, positions, 0)
        part['preplanned_anchor_keep_mask'] = keep
        parts.append(DMTPResponsePart(doc, start, end, context_start, end, part))
    return tuple(parts)

def shard_dmtp_swa_parts(parts, *, rank, world_size):
    """Assign response parts to DP ranks; average normalized part losses equally."""
    if type(rank) is not int or type(world_size) is not int or world_size < 1 or (not 0 <= rank < world_size) or (not parts):
        raise ValueError('invalid SWA part mesh or empty plan')
    indices = tuple(range(rank, len(parts), world_size))
    selected = tuple((parts[i] for i in indices))
    if not selected:
        source = parts[0]
        seed = int(source.batch['preplanned_anchor_positions'][0, 0])
        dummy = dmtp_swa_windows_slice_batch({k: v for k, v in source.batch.items() if not k.startswith('preplanned_')}, seed, seed + 1)
        dummy['loss_mask'].zero_()
        dummy['preplanned_anchor_positions'] = torch.zeros_like(source.batch['preplanned_anchor_positions'])
        dummy['preplanned_anchor_keep_mask'] = torch.zeros_like(source.batch['preplanned_anchor_keep_mask'])
        selected = (DMTPResponsePart(source.document_id, source.response_start, source.response_start, source.context_start + seed, source.context_start + seed + 1, dummy),)
    return DMTPPartShard(selected, indices, len(parts), world_size / len(parts))

def backward_dmtp_swa_parts(module, shard):
    """Accumulate per-part normalized loss; DP averaging yields the global mean.

    Callers must enable the model's SWA mask and DMTP anchor policy and bind this
    reduction/plan identity to checkpoints. Empty ranks run the normal forward
    with no labels. NPU/FSDP integration is a separate acceptance requirement.
    """
    if not math.isfinite(shard.loss_scale) or shard.loss_scale <= 0 or (not shard.parts):
        raise ValueError('invalid SWA loss scale or missing parts')
    sync = getattr(module, 'set_requires_gradient_sync', None)
    result = 0.0
    try:
        for ordinal, part in enumerate(shard.parts):
            if sync is not None:
                sync(ordinal == len(shard.parts) - 1)
            loss = module(**part.batch)[0]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('SWA response-part loss is not finite')
            (loss * shard.loss_scale).backward()
            result += float(loss.detach()) * shard.loss_scale
    finally:
        if sync is not None:
            sync(True)
    return result

def move_dmtp_swa_part_shard(shard, device):
    """Transfer only owned compact parts, retaining the resident target head."""
    parts = []
    for part in shard.parts:
        batch = {}
        for name, value in part.batch.items():
            if name == 'lm_head_weight':
                batch[name] = value
            elif name == 'hidden_states_list':
                batch[name] = [tensor.to(device, copy=True).contiguous() for tensor in value]
            else:
                batch[name] = value.to(device, copy=True).contiguous()
        parts.append(replace(part, batch=batch))
    return replace(shard, parts=tuple(parts))
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class DSparkDPShard:
    batch: dict[str, Any]
    document_ids: tuple[int, ...]
    eligible_pairs: int
    global_eligible_pairs: int
    loss_scale: float

def dspark_dp_sharding_eligible(batch: Mapping[str, Any]) -> torch.Tensor:
    mask = batch['loss_mask']
    documents = batch['document_ids']
    return (mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5) & (documents[:, :-1] == documents[:, 1:]) & (documents[:, :-1] >= 0)

def shard_dspark_documents(batch: Mapping[str, Any], *, rank: int, world_size: int) -> DSparkDPShard:
    """Assign whole documents by ordinal modulo DP world size.

    Original target features, positions and document padding stay together.
    An empty rank evaluates one real pair with zero gradient weight, keeping
    FSDP2 forward/backward collectives matched. The caller still commits the
    original global transaction set once.
    """
    ids, mask, docs = (batch['input_ids'], batch['loss_mask'], batch['document_ids'])
    if type(rank) is not int or type(world_size) is not int or world_size < 1 or (not 0 <= rank < world_size) or (ids.ndim != 2) or (ids.shape[0] != 1) or (ids.shape[1] < 2) or (mask.shape != ids.shape) or (docs.shape != ids.shape) or (docs.dtype != torch.int64):
        raise ValueError('DP sharding requires a valid rank and aligned packed stream')
    order = []
    owners = []
    previous = None
    for document in docs[0].tolist():
        if document < -1:
            raise ValueError('invalid packed document ID')
        if document >= 0 and document != previous:
            if document in order:
                raise ValueError('packed documents must be contiguous')
            order.append(document)
            previous = document
        if previous is None:
            raise ValueError('packed padding precedes the first document')
        owners.append(previous)
    active = dspark_dp_sharding_eligible(batch)
    global_pairs = int(active.sum().item())
    if not global_pairs:
        raise ValueError('global packed batch has no trainable adjacent supervised rows')
    selected_docs = tuple((document for index, document in enumerate(order) if index % world_size == rank))
    indices = [index for index, owner in enumerate(owners) if owner in selected_docs]

    def take(rows: list[int]) -> dict[str, Any]:
        index = torch.tensor(rows, dtype=torch.int64, device=ids.device)
        result = {}
        for name, value in batch.items():
            if name == 'lm_head_weight':
                result[name] = value
            elif name == 'hidden_states_list':
                result[name] = [tensor.index_select(1, index) for tensor in value]
            elif isinstance(value, torch.Tensor) and value.ndim >= 2 and (value.shape[:2] == ids.shape):
                result[name] = value.index_select(1, index)
            else:
                result[name] = value
        return result
    local = take(indices) if indices else None
    if local is not None:
        original_docs = local['document_ids']
        remapped = torch.full_like(original_docs, -1)
        for index, document in enumerate(selected_docs):
            remapped[original_docs == document] = index
        local['document_ids'] = remapped
    local_pairs = int(dspark_dp_sharding_eligible(local).sum().item()) if local is not None else 0
    if not local_pairs:
        first = int(torch.nonzero(active[0], as_tuple=False)[0].item())
        local = take([first, first + 1])
        local['document_ids'] = torch.zeros_like(local['document_ids'])
        local['loss_mask'] = torch.ones_like(local['loss_mask'])
    return DSparkDPShard(local, selected_docs, local_pairs, global_pairs, world_size * local_pairs / global_pairs)

@dataclass(frozen=True)
class DSparkWindow:
    start: int
    end: int
    eligible_pairs: int
    batch: dict[str, Any]

def dspark_training_windows(batch: Mapping[str, Any], window_size: int) -> list[DSparkWindow]:
    """Visit every window, retaining one predecessor row at each boundary.

    Target hidden states come from full-context prefill. Absolute positions are
    preserved; slicing never asks the target model to prefill a bare chunk.
    Pair ownership is disjoint across windows. DSpark still samples anchors
    within each window and enforces its existing document/block boundary mask.
    """
    ids, mask = (batch['input_ids'], batch['loss_mask'])
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != 1:
        raise ValueError('DSpark windows require one complete packed stream row')
    if window_size < 2:
        raise ValueError('DSpark window size must be at least two')
    length = ids.shape[1]
    windows = []
    for start in range(0, max(length - 1, 0), window_size - 1):
        end = min(start + window_size, length)
        active = (mask[:, start:end - 1] > 0.5) & (mask[:, start + 1:end] > 0.5)
        documents = batch.get('document_ids')
        if documents is not None:
            active &= (documents[:, start:end - 1] == documents[:, start + 1:end]) & (documents[:, start:end - 1] >= 0)
        count = int(active.sum().item())
        if not count:
            continue
        part = {}
        for name, value in batch.items():
            if name == 'lm_head_weight':
                part[name] = value
            elif name == 'hidden_states_list':
                part[name] = [tensor[:, start:end] for tensor in value]
            elif isinstance(value, torch.Tensor) and value.ndim >= 2 and (value.shape[:2] == ids.shape):
                part[name] = value[:, start:end]
            else:
                part[name] = value
        windows.append(DSparkWindow(start, end, count, part))
    if not windows:
        raise ValueError('DSpark sequence has no trainable adjacent supervised rows')
    return windows

def backward_dspark_windows(module, batch: Mapping[str, Any], window_size: int, *, loss_scale: float=1.0) -> float:
    """Accumulate all windows before the caller performs one optimizer update."""
    import math
    if not math.isfinite(loss_scale) or loss_scale < 0:
        raise ValueError('DSpark gradient loss scale must be finite and non-negative')
    windows = dspark_training_windows(batch, window_size)
    total = sum((window.eligible_pairs for window in windows))
    result = 0.0
    sync = getattr(module, 'set_requires_gradient_sync', None)
    try:
        for index, window in enumerate(windows):
            if sync is not None:
                sync(index == len(windows) - 1)
            loss = module(**window.batch)[0]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('DSpark window loss is not finite')
            weight = window.eligible_pairs / total
            (loss * weight * loss_scale).backward()
            result += float(loss.detach().item()) * weight
    finally:
        if sync is not None:
            sync(True)
    return result

@dataclass(frozen=True)
class DSparkDPWindowShard:
    windows: tuple[DSparkWindow, ...]
    window_indices: tuple[int, ...]
    eligible_pairs: int
    global_eligible_pairs: int
    loss_scale: float

def move_dspark_window_shard(shard: DSparkDPWindowShard, device) -> DSparkDPWindowShard:
    """Move only owned windows, retaining the separately resident target head.

    Shard on CPU first to avoid copying the full packed teacher capture to every
    NPU. Compact window copies also stop views from retaining full capture storage.
    """
    windows = []
    for window in shard.windows:
        batch = {}
        for name, value in window.batch.items():
            if name == 'lm_head_weight':
                batch[name] = value
            elif name == 'hidden_states_list':
                batch[name] = [tensor.to(device, copy=True).contiguous() for tensor in value]
            elif isinstance(value, torch.Tensor):
                batch[name] = value.to(device, copy=True).contiguous()
            else:
                batch[name] = value
        windows.append(replace(window, batch=batch))
    return replace(shard, windows=tuple(windows))

def shard_dspark_windows(batch: Mapping[str, Any], *, window_size: int, rank: int, world_size: int) -> DSparkDPWindowShard:
    """Assign eligible windows by ordinal; preserve teacher features and positions.

    One predecessor row overlaps adjacent windows, so each supervised pair is
    owned exactly once. Empty ranks execute a real pair with zero gradient weight
    to participate in the final FSDP2 gradient synchronization.
    """
    if type(rank) is not int or type(world_size) is not int or world_size < 1 or (not 0 <= rank < world_size) or (type(window_size) is not int):
        raise ValueError('window DP sharding requires a valid rank, world size and window size')
    all_windows = dspark_training_windows(batch, window_size)
    global_pairs = sum((window.eligible_pairs for window in all_windows))
    indices = tuple(range(rank, len(all_windows), world_size))
    selected = tuple((all_windows[index] for index in indices))
    local_pairs = sum((window.eligible_pairs for window in selected))
    if not selected:
        source = all_windows[0]
        mask = source.batch['loss_mask']
        valid = (mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)
        documents = source.batch.get('document_ids')
        if documents is not None:
            valid &= (documents[:, :-1] == documents[:, 1:]) & (documents[:, :-1] >= 0)
        first = int(torch.nonzero(valid[0], as_tuple=False)[0].item())
        ids = source.batch['input_ids']
        dummy = {}
        for name, value in source.batch.items():
            if name == 'lm_head_weight':
                dummy[name] = value
            elif name == 'hidden_states_list':
                dummy[name] = [tensor[:, first:first + 2] for tensor in value]
            elif isinstance(value, torch.Tensor) and value.ndim >= 2 and (value.shape[:2] == ids.shape):
                dummy[name] = value[:, first:first + 2]
            else:
                dummy[name] = value
        selected = (DSparkWindow(source.start + first, source.start + first + 2, 1, dummy),)
    return DSparkDPWindowShard(selected, indices, local_pairs, global_pairs, world_size * local_pairs / global_pairs)

def backward_dspark_window_shard(module, shard: DSparkDPWindowShard) -> float:
    """Accumulate a shard then synchronize once; the caller updates the optimizer."""
    if not math.isfinite(shard.loss_scale) or shard.loss_scale < 0 or (not shard.windows):
        raise ValueError('invalid DP window shard loss scale or windows')
    total = sum((window.eligible_pairs for window in shard.windows))
    value = 0.0
    sync = getattr(module, 'set_requires_gradient_sync', None)
    try:
        for index, window in enumerate(shard.windows):
            if sync is not None:
                sync(index == len(shard.windows) - 1)
            loss = module(**window.batch)[0]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('DSpark DP window loss is not finite')
            weight = window.eligible_pairs / total
            (loss * weight * shard.loss_scale).backward()
            value += float(loss.detach().item()) * weight
    finally:
        if sync is not None:
            sync(True)
    return value
from typing import Sequence
PACKING_TIERS = (16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024)

@dataclass(frozen=True)
class PackedPrefixPlan:
    target_length: int
    sample_count: int
    aligned_lengths: tuple[int, ...]
    used_tokens: int
    payload_bytes: int
    fill_ratio: float
    below_min_fill: bool

def plan_packed_prefix(token_counts: Sequence[int], payload_bytes: Sequence[int], *, target_length: int, block_size: int, max_samples: int, max_packed_bytes: int, min_fill_ratio: float=0.65) -> PackedPrefixPlan:
    """Take a contiguous prefix, so stream cursors never skip a captured sample."""
    if target_length <= 0 or block_size <= 0 or max_samples <= 0 or (max_packed_bytes <= 0) or (not 0 < min_fill_ratio <= 1):
        raise ValueError('packing budgets and min_fill_ratio must be positive')
    if len(token_counts) != len(payload_bytes) or not token_counts:
        raise ValueError('packing needs equally sized nonempty token and byte lists')
    if any((type(count) is not int or count <= 0 for count in token_counts)):
        raise ValueError('packing token counts must be positive integers')
    if any((type(size) is not int or size <= 0 for size in payload_bytes)):
        raise ValueError('packing payload sizes must be positive integers')
    aligned = tuple(((count + block_size - 1) // block_size * block_size for count in token_counts))
    total_tokens = 0
    total_bytes = 0
    selected: list[int] = []
    for count, length, size in zip(token_counts, aligned, payload_bytes):
        padded_size = (size * length + count - 1) // count
        if length > target_length or padded_size > max_packed_bytes:
            raise ValueError('one sample exceeds the packing token or byte budget')
        if len(selected) >= max_samples or total_tokens + length > target_length or total_bytes + padded_size > max_packed_bytes:
            break
        selected.append(length)
        total_tokens += length
        total_bytes += padded_size
    if not selected:
        raise ValueError('packing selected no samples')
    fill = total_tokens / target_length
    return PackedPrefixPlan(target_length=target_length, sample_count=len(selected), aligned_lengths=tuple(selected), used_tokens=total_tokens, payload_bytes=total_bytes, fill_ratio=fill, below_min_fill=fill < min_fill_ratio)

def require_packing_tier(target_length: int) -> int:
    if target_length not in PACKING_TIERS:
        raise ValueError('packing target_length must be 16K, 32K, 64K or 128K')
    return target_length
MAX_NATIVE_PREFILL_ROWS = 64

@dataclass(frozen=True)
class PackedRowsPlan:
    rows: tuple[PackedPrefixPlan, ...]
    sample_indices: tuple[tuple[int, ...], ...]
    sample_count: int
    original_tokens: int
    padded_tokens: int
    payload_bytes: int
    deferred_samples: int

def plan_packed_rows(token_counts, payload_bytes, *, target_length, block_size, max_rows, max_samples_per_row, max_bytes_per_row, min_fill_ratio=0.65):
    """Plan whole documents in FIFO rows; 128K is a per-row capacity.

    This is admission metadata, not a transport or training implementation.
    A caller can assign one row to each DP rank without calling a global 128K
    pack a per-DP 128K batch. Original token totals exclude alignment padding.
    A deferred tail is explicit, so the stream cursor advances by sample_count.
    """
    require_packing_tier(target_length)
    if any((type(v) is not int or v <= 0 for v in (target_length, block_size, max_rows, max_samples_per_row, max_bytes_per_row))):
        raise ValueError('packing row budgets must be positive integers')
    if len(token_counts) != len(payload_bytes) or not token_counts:
        raise ValueError('packing rows require nonempty aligned token and byte metadata')
    if any((type(v) is not int or v <= 0 for v in (*token_counts, *payload_bytes))):
        raise ValueError('packing rows require positive integer token and byte metadata')
    rows, indices = ([], [])
    cursor = 0
    for _ in range(max_rows):
        if cursor == len(token_counts):
            break
        row = plan_packed_prefix(token_counts[cursor:], payload_bytes[cursor:], target_length=target_length, block_size=block_size, max_samples=max_samples_per_row, max_packed_bytes=max_bytes_per_row, min_fill_ratio=min_fill_ratio)
        indices.append(tuple(range(cursor, cursor + row.sample_count)))
        rows.append(row)
        cursor += row.sample_count
    return PackedRowsPlan(tuple(rows), tuple(indices), cursor, sum(token_counts[:cursor]), sum((r.used_tokens for r in rows)), sum((r.payload_bytes for r in rows)), len(token_counts) - cursor)
from typing import Mapping, Sequence
from verl_speco.transport.streaming import SampleMetadataV3

def materialize_dspark_batch(fields: Mapping[str, torch.Tensor], metadata: SampleMetadataV3, *, lm_head_weight: torch.Tensor, device: torch.device | str, target_hidden_size: int, preserve_lm_head_device: bool=False) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    metadata.validate()
    if metadata.hidden_states_layout != 'dflash_aux_plus_last' or metadata.hidden_dtype != 'bfloat16':
        raise ValueError('DSpark TQ sample needs dflash_aux_plus_last bfloat16 layout')
    count = metadata.token_count
    layer_count = len(metadata.target_layer_ids)
    ids = fields['input_ids']
    positions = fields['position_ids']
    mask = fields['loss_mask']
    hidden = fields['hidden_states']
    if ids.shape != (count,) or ids.dtype != torch.int64 or positions.shape != (count,) or (positions.dtype != torch.int64) or (mask.shape != (count,)) or (mask.dtype != torch.float32) or (hidden.shape != (count, (layer_count + 1) * target_hidden_size)) or (hidden.dtype != torch.bfloat16):
        raise ValueError('DSpark TQ sample field shape or dtype is incompatible')
    actual_bytes = sum((t.numel() * t.element_size() for t in (ids, positions, mask, hidden)))
    if actual_bytes != metadata.payload_nbytes:
        raise ValueError('DSpark TQ sample byte count differs from v3 metadata')
    if not torch.equal(positions.cpu(), torch.arange(count)):
        raise ValueError('DSpark TQ sample requires complete prefill positions')
    if not bool(torch.all((mask == 0) | (mask == 1))) or not bool(torch.any(mask == 1)):
        raise ValueError('DSpark TQ sample requires at least one binary supervised token')
    if metadata.prompt_token_count is not None:
        boundary = metadata.prompt_token_count
        if not bool(torch.all(mask[:boundary] == 0)) or not bool(torch.all(mask[boundary:] == 1)):
            raise ValueError('DSpark TQ loss mask differs from v3 prompt/supervised token mapping')
    if lm_head_weight.shape[1] != target_hidden_size or lm_head_weight.dtype != torch.bfloat16:
        raise ValueError('target LM head width/dtype differs from captured hidden')
    chunks = hidden.split(target_hidden_size, dim=-1)
    return {'input_ids': ids.to(device).unsqueeze(0), 'position_ids': positions.to(device).unsqueeze(0), 'hidden_states_list': [chunk.to(device).unsqueeze(0) for chunk in chunks[:-1]], 'target_last_hidden_states': chunks[-1].to(device).unsqueeze(0), 'loss_mask': mask.to(device).unsqueeze(0), 'lm_head_weight': lm_head_weight if preserve_lm_head_device else lm_head_weight.to(device)}

def tq_dspark_batch_plan_packed_samples(samples, *, target_length, max_packed_bytes, max_samples, block_size):
    require_packing_tier(target_length)
    if not samples:
        raise ValueError('packed DSpark batch needs at least one TQ sample')
    first = samples[0][1]
    for _, metadata in samples:
        if metadata.run_id != first.run_id or metadata.generation != first.generation or metadata.target_revision != first.target_revision or (metadata.target_model_fingerprint != first.target_model_fingerprint) or (metadata.tokenizer_fingerprint != first.tokenizer_fingerprint) or (metadata.target_layer_ids != first.target_layer_ids):
            raise ValueError('packed TQ samples have different target or generation identity')
    if any((metadata.sequence_no != first.sequence_no + index for index, (_, metadata) in enumerate(samples))):
        raise ValueError('packed TQ samples must form a contiguous stream prefix')
    plan = plan_packed_prefix([metadata.token_count for _, metadata in samples], [metadata.payload_nbytes for _, metadata in samples], target_length=target_length, block_size=block_size, max_samples=max_samples, max_packed_bytes=max_packed_bytes)
    return (plan, samples[:plan.sample_count])

def materialize_packed_dspark_batch(samples: Sequence[tuple[Mapping[str, torch.Tensor], SampleMetadataV3]], *, lm_head_weight: torch.Tensor, device: torch.device | str, target_hidden_size: int, target_length: int, max_packed_bytes: int, max_samples: int=16, block_size: int=2, preserve_lm_head_device: bool=False) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], int]:
    """Pack a contiguous captured prefix with per-document masks and positions."""
    plan, selected = tq_dspark_batch_plan_packed_samples(samples, target_length=target_length, max_packed_bytes=max_packed_bytes, max_samples=max_samples, block_size=block_size)
    first = samples[0][1]
    pieces = [materialize_dspark_batch(fields, metadata, lm_head_weight=lm_head_weight, device=device, target_hidden_size=target_hidden_size, preserve_lm_head_device=preserve_lm_head_device) for fields, metadata in selected]
    ids_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    doc_parts: list[torch.Tensor] = []
    hidden_parts: list[list[torch.Tensor]] = [[] for _ in first.target_layer_ids]
    last_parts: list[torch.Tensor] = []
    for doc_index, ((_, metadata), piece, aligned) in enumerate(zip(selected, pieces, plan.aligned_lengths)):
        pad = aligned - metadata.token_count

        def padded(value: torch.Tensor) -> torch.Tensor:
            if pad == 0:
                return value
            shape = (1, pad, *value.shape[2:])
            return torch.cat((value, value.new_zeros(shape)), dim=1)
        ids_parts.append(padded(piece['input_ids']))
        pos_parts.append(padded(piece['position_ids']))
        mask_parts.append(padded(piece['loss_mask']))
        doc = torch.full((1, aligned), doc_index, dtype=torch.int64, device=device)
        if pad:
            doc[:, metadata.token_count:] = -1
        doc_parts.append(doc)
        for output, hidden in zip(hidden_parts, piece['hidden_states_list']):
            output.append(padded(hidden))
        last_parts.append(padded(piece['target_last_hidden_states']))
    batch: dict[str, torch.Tensor | list[torch.Tensor]] = {'input_ids': torch.cat(ids_parts, dim=1), 'position_ids': torch.cat(pos_parts, dim=1), 'loss_mask': torch.cat(mask_parts, dim=1), 'document_ids': torch.cat(doc_parts, dim=1), 'hidden_states_list': [torch.cat(parts, dim=1) for parts in hidden_parts], 'target_last_hidden_states': torch.cat(last_parts, dim=1), 'lm_head_weight': lm_head_weight if preserve_lm_head_device else lm_head_weight.to(device)}
    return (batch, plan.sample_count)
