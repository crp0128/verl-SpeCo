# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from dataclasses import dataclass

class CaptureCapacityUnavailable(RuntimeError):
    """The first complete FIFO sequence cannot fit this capacity snapshot."""

@dataclass(frozen=True)
class CaptureCapacity:
    free_kv_blocks: int
    kv_block_size: int
    rank_free_bytes: tuple[int, ...]
    sampled_at: float
    kv_groups: int = 1

    def __post_init__(self):
        if type(self.free_kv_blocks) is not int or self.free_kv_blocks < 0 or type(self.kv_block_size) is not int or (self.kv_block_size != 128) or (type(self.kv_groups) is not int) or (self.kv_groups != 1) or (not isinstance(self.rank_free_bytes, tuple)) or (len(self.rank_free_bytes) != 8) or any((type(n) is not int or n < 0 for n in self.rank_free_bytes)):
            raise ValueError('capacity requires TP8 and the actual uniform MLA page pool')
        import math
        if type(self.sampled_at) not in (float, int) or not math.isfinite(self.sampled_at):
            raise ValueError('invalid capacity observation clock')

@dataclass(frozen=True)
class KvAwareCaptureBudget:
    max_tokens: int = 131072
    max_sequences: int = 40
    reserve_kv_blocks: int = 8
    reserve_rank_bytes: int = 2 * 1024 ** 3
    capture_bytes_per_token: int = 61440
    max_snapshot_age_seconds: float = 5.0

    def __post_init__(self):
        import math
        fields = (self.max_tokens, self.max_sequences, self.reserve_kv_blocks, self.reserve_rank_bytes, self.capture_bytes_per_token)
        if any((type(n) is not int for n in fields)) or not 1 <= self.max_tokens <= 131072 or (not 1 <= self.max_sequences <= 40) or (self.reserve_kv_blocks < 0) or (self.reserve_rank_bytes < 0) or (self.capture_bytes_per_token < 61440) or (type(self.max_snapshot_age_seconds) not in (int, float)) or (not math.isfinite(self.max_snapshot_age_seconds)) or (self.max_snapshot_age_seconds <= 0):
            raise ValueError('invalid KV/capture memory safety bounds')

    def validate_snapshot(self, snapshot, *, now):
        import math
        if not isinstance(snapshot, CaptureCapacity) or type(now) not in (int, float) or (not math.isfinite(now)) or (not 0 <= now - snapshot.sampled_at <= self.max_snapshot_age_seconds):
            raise ValueError('source capacity is missing, stale or from a future clock')

    def resources(self, children):
        tokens = 0
        blocks = 0
        ranks = [0] * 8
        for child in children:
            n, owner = (child.original_tokens, child.capture_owner_rank)
            if type(n) is not int or not 1 <= n <= 49152 or type(owner) is not int or (not 0 <= owner < 8):
                raise ValueError('invalid full-sequence source ownership')
            tokens += n
            blocks += (n + 1 + 127) // 128
            ranks[owner] += n * self.capture_bytes_per_token
        return dict(original_tokens=tokens, kv_blocks=blocks, rank_capture_bytes=tuple(ranks), sequences=len(children))

    def fits(self, children, snapshot):
        use = self.resources(children)
        return use['sequences'] <= self.max_sequences and use['original_tokens'] <= self.max_tokens and (use['kv_blocks'] + self.reserve_kv_blocks <= snapshot.free_kv_blocks) and all((n + self.reserve_rank_bytes <= free for n, free in zip(use['rank_capture_bytes'], snapshot.rank_free_bytes)))

def read_engine_capacity(engine, *, now):
    """Read the pinned in-process scheduler and real TP8 worker observations.

    No guessed capacity, API utilization percentage, or total-HBM fallback is
    permitted. Remote EngineCore clients need a separate explicit adapter.
    """
    if engine.has_unfinished_requests():
        raise RuntimeError('capacity must be sampled between completed engine batches')
    client = getattr(engine, 'engine_core', None)
    core = getattr(client, 'engine_core', None)
    scheduler = getattr(core, 'scheduler', None)
    manager = getattr(scheduler, 'kv_cache_manager', None)
    config = getattr(manager, 'kv_cache_config', None)
    if config is None or len(config.kv_cache_groups) != 1:
        raise ValueError('capacity requires the pinned in-process uniform KV scheduler')
    size = config.kv_cache_groups[0].kv_cache_spec.block_size
    blocks = manager.block_pool.get_num_free_blocks()
    observations = engine.collective_rpc('speco_capture_memory_capacity')
    if not isinstance(observations, list) or len(observations) != 8 or any((not isinstance(row, dict) or row.get('rank') != i or type(row.get('rank')) is not int or (type(row.get('free_device_bytes')) is not int) or (row['free_device_bytes'] < 0) for i, row in enumerate(observations))):
        raise ValueError('missing or inconsistent actual TP8 memory observations')
    return CaptureCapacity(free_kv_blocks=blocks, kv_block_size=size, rank_free_bytes=tuple((r['free_device_bytes'] for r in observations)), sampled_at=now())
import heapq

def balanced_prefill_positions(lengths, *, start_cursor=0, producer_count=6, physical_batch_size=8, max_batch_tokens=131072):
    lengths = tuple(lengths)
    if type(start_cursor) is not int or start_cursor < 0 or type(producer_count) is not int or (producer_count != 6) or (type(physical_batch_size) is not int) or (physical_batch_size not in (8, 16)) or (not lengths) or (len(lengths) > 960) or any((type(n) is not int or not 2 <= n <= 49152 for n in lengths)):
        raise ValueError('invalid frozen batch metadata for balancing')
    destinations = {}
    for i in range(len(lengths)):
        cursor = start_cursor + i
        group = (cursor // (producer_count * physical_batch_size), cursor % producer_count)
        destinations.setdefault(group, []).append(i)
    bins = {key: [] for key in destinations}
    loads = {key: 0 for key in destinations}
    heap = [(0, 0, key) for key in destinations]
    heapq.heapify(heap)
    for source in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        total, count, key = heapq.heappop(heap)
        bins[key].append(source)
        loads[key] += lengths[source]
        if len(bins[key]) < len(destinations[key]):
            heapq.heappush(heap, (loads[key], len(bins[key]), key))
    result = [None] * len(lengths)
    for key, positions in destinations.items():
        for destination, source in zip(positions, bins[key]):
            result[destination] = source
    if sorted(result) != list(range(len(lengths))):
        raise RuntimeError('prefill balancing lost a source row')
    if max(loads.values()) > max_batch_tokens:
        return tuple(range(len(lengths)))
    return tuple(result)

def wave_token_bounds(lengths, start_cursor=0):
    groups = {}
    for i, n in enumerate(lengths):
        cursor = start_cursor + i
        groups[cursor // 48, cursor % 6] = groups.get((cursor // 48, cursor % 6), 0) + n
    waves = {}
    for (wave, producer), tokens in groups.items():
        waves[wave] = max(waves.get(wave, 0), tokens)
    return dict(maximum_eight_child_tokens=max(groups.values()), sum_wave_straggler_tokens=sum(waves.values()), wave_count=len(waves))
BALANCE_POLICY = 'balanced-eight-owner-v1'
WIDE_BALANCE_POLICY = 'balanced-sixteen-child-v1'
PACKED_BALANCE_POLICY = 'balanced-sixteen-child-packed-order-v1'
PACKED_EIGHT_BALANCE_POLICY = 'balanced-eight-child-packed-order-v1'
BALANCE_POLICIES = (BALANCE_POLICY, WIDE_BALANCE_POLICY, PACKED_BALANCE_POLICY, PACKED_EIGHT_BALANCE_POLICY)

def balance_physical_batch_size(policy):
    if policy not in ('', *BALANCE_POLICIES):
        raise ValueError('unsupported frozen prefill policy')
    return 16 if policy in (WIDE_BALANCE_POLICY, PACKED_BALANCE_POLICY) else 8
SPEC_FIELDS = frozenset({'schema_version', 'index_path', 'index_file_sha256', 'index_sha256', 'dataset_sha256', 'plan_sha256', 'epochs', 'seed', 'samples_per_step', 'producer_count', 'max_sequence_length'})

def epoch_spec_balance_policy(spec):
    if not isinstance(spec, dict) or type(spec.get('schema_version')) is not int:
        raise ValueError('invalid epoch specification schema')
    if spec['schema_version'] == 1 and set(spec) == SPEC_FIELDS:
        return ''
    if spec['schema_version'] == 2 and set(spec) == SPEC_FIELDS | {'prefill_balance_policy'} and (spec['prefill_balance_policy'] in BALANCE_POLICIES) and (type(spec['producer_count']) is int) and (spec['producer_count'] == 6) and (type(spec['samples_per_step']) is int) and (spec['samples_per_step'] == 480 or (spec['samples_per_step'] == 960 and spec['prefill_balance_policy'] in (PACKED_BALANCE_POLICY, PACKED_EIGHT_BALANCE_POLICY))) and (type(spec['max_sequence_length']) is int) and (spec['max_sequence_length'] == 49152):
        return spec['prefill_balance_policy']
    raise ValueError('epoch specification has unknown or incomplete balancing bindings')

def packed_balanced_prefill_positions(lengths, *, start_cursor=0, producer_count=6, physical_batch_size=16, max_batch_tokens=131072):
    if type(physical_batch_size) is not int or physical_batch_size not in (8, 16):
        raise ValueError('packing order requires explicit eight/sixteen-child groups')
    lengths = tuple(lengths)
    original = balanced_prefill_positions(lengths, start_cursor=start_cursor, producer_count=producer_count, physical_batch_size=physical_batch_size, max_batch_tokens=max_batch_tokens)
    available = {}
    for destination, source in enumerate(original):
        cursor = start_cursor + destination
        key = (cursor // (producer_count * physical_batch_size), cursor % producer_count)
        available.setdefault(key, []).append(source)
    remaining = 131072
    documents = 0
    packed = []
    for destination in range(len(lengths)):
        cursor = start_cursor + destination
        key = (cursor // (producer_count * physical_batch_size), cursor % producer_count)
        pool = available[key]
        if documents >= 64:
            remaining = 131072
            documents = 0
        fitting = [i for i in pool if (lengths[i] + 7) // 8 * 8 <= remaining]
        if not fitting:
            remaining = 131072
            documents = 0
            fitting = list(pool)
        selected = max(fitting, key=lambda i: (lengths[i], -i))
        pool.remove(selected)
        packed.append(selected)
        remaining -= (lengths[selected] + 7) // 8 * 8
        documents += 1
    if sorted(packed) != list(range(len(lengths))):
        raise RuntimeError('packing order lost a frozen source')
    return tuple(packed)
from collections import deque
import math

@dataclass(frozen=True)
class Child:
    sequence_no: int
    command_step: int
    physical_child_index: int
    physical_batch_size: int
    original_tokens: int
    capture_owner_rank: int

class TokenBudgetCommandCoalescer:

    def __init__(self, *, run_id, generation, producer_index, producer_count, logical_width=8, token_budget=131072, target_fill=0.9, max_sequences=40, samples_per_step=960, total_samples, start_round=0, cross_step=False):
        ints = (producer_index, producer_count, logical_width, token_budget, max_sequences, samples_per_step, total_samples, start_round)
        if any((type(x) is not int for x in ints)) or not (0 <= producer_index < producer_count == 6 and logical_width == 8 and (token_budget in (65536, 131072)) and (1 <= max_sequences <= 40) and (samples_per_step == 960) and (1 <= total_samples <= 6000000) and (start_round >= 0) and (start_round * producer_count < total_samples)):
            raise ValueError('coalescing requires the bounded six-TP8 logical-eight profile')
        if type(target_fill) not in (int, float) or not math.isfinite(target_fill) or (not 0 < target_fill <= 1):
            raise ValueError('invalid original-token fill target')
        from verl_speco.runtime.services import build_physical_capture_command
        build_physical_capture_command(run_id=run_id, generation=generation, producer_index=producer_index, producer_count=producer_count, step_index=start_round, sample_count=1)
        if type(cross_step) is not bool or (cross_step and (target_fill != 0.95 or max_sequences != 40)) or (token_budget == 65536 and (not cross_step)):
            raise ValueError('cross-step coalescing requires explicit40/.95 admission')
        self.cross_step = cross_step
        self.run_id = run_id
        self.generation = generation
        self.producer = producer_index
        self.count = producer_count
        self.width = logical_width
        self.budget = token_budget
        self.target = target_fill
        self.limit = max_sequences
        self.samples = samples_per_step
        self.total = total_samples
        self.next_round = start_round
        self.pending = deque()
        self.commands = {}
        self.active = None
        self.failed = False
        self.original_tokens_completed = 0

    def _load_command(self, load_command, token_count):
        from verl_speco.runtime.services import validate_physical_capture_command
        remaining = len(range(self.producer + self.next_round * self.count, self.total, self.count))
        if not remaining:
            return False
        command = load_command(self.next_round)
        indices = validate_physical_capture_command(command, run_id=self.run_id, generation=self.generation, producer_index=self.producer, producer_count=self.count, step_index=self.next_round, remaining_steps=remaining)
        if not len(indices) <= self.width or indices[0] // self.samples != indices[-1] // self.samples:
            raise ValueError('logical coalescing command crosses optimizer boundary')
        step = self.next_round
        rows = []
        for i, number in enumerate(indices):
            n = token_count(number)
            if type(n) is not int or not 1 <= n <= self.budget:
                raise ValueError('complete source sequence exceeds token budget')
            rows.append(Child(number, step, i, len(indices), n, i % 8))
        self.commands[step] = dict(command=command, children=tuple(rows), published={})
        self.pending.extend(rows)
        self.next_round += len(indices)
        return True

    def next_batch(self, *, load_command, token_count, capacity_provider=None, budget_policy=None, now=None):
        if self.failed or self.active is not None:
            raise RuntimeError('coalescer is busy or poisoned')
        snapshot = None
        if any((x is not None for x in (capacity_provider, budget_policy, now))):
            if not callable(capacity_provider) or not callable(now) or (not isinstance(budget_policy, KvAwareCaptureBudget)) or self.cross_step:
                raise ValueError('adaptive capture requires complete same-step capacity binding')
            snapshot = capacity_provider()
            budget_policy.validate_snapshot(snapshot, now=now())
        try:
            batch = []
            tokens = 0
            optimizer = None
            effective_limit = min(self.limit, budget_policy.max_sequences) if snapshot is not None else self.limit
            while len(batch) < effective_limit:
                if not self.pending:
                    if snapshot is not None and batch and (len({row.command_step for row in batch}) >= 2):
                        break
                    if batch and (not self.cross_step) and ((self.producer + self.next_round * self.count) // self.samples != optimizer):
                        break
                    if not self._load_command(load_command, token_count):
                        break
                child = self.pending[0]
                if batch and (not self.cross_step and child.sequence_no // self.samples != optimizer or tokens + child.original_tokens > self.budget):
                    break
                if snapshot is not None and now() - snapshot.sampled_at > budget_policy.max_snapshot_age_seconds:
                    snapshot = capacity_provider()
                    budget_policy.validate_snapshot(snapshot, now=now())
                    if batch and (not budget_policy.fits(batch, snapshot)):
                        for selected in reversed(batch):
                            self.pending.appendleft(selected)
                        batch = []
                        raise CaptureCapacityUnavailable('capacity changed while waiting for issued commands')
                if snapshot is not None and batch and (child.command_step != batch[-1].command_step) and (len({row.command_step for row in batch}) >= 2):
                    break
                if snapshot is not None and (not budget_policy.fits((*batch, child), snapshot)):
                    if batch:
                        break
                    raise CaptureCapacityUnavailable('complete FIFO source sequence cannot fit current capacity')
                self.pending.popleft()
                batch.append(child)
                tokens += child.original_tokens
                optimizer = child.sequence_no // self.samples
                if tokens >= self.target * self.budget:
                    break
            if not batch:
                return None
            self.active = tuple(batch)
            result = dict(children=self.active, original_tokens=tokens, token_fill=tokens / self.budget, target_reached=tokens >= self.target * self.budget, optimizer_step=batch[0].sequence_no // self.samples)
            if snapshot is not None:
                result['capacity_admission'] = dict(budget_policy.resources(batch), free_kv_blocks=snapshot.free_kv_blocks, kv_block_size=snapshot.kv_block_size, rank_free_bytes=snapshot.rank_free_bytes, sampled_at=snapshot.sampled_at)
            return result
        except CaptureCapacityUnavailable:
            raise
        except BaseException:
            self.failed = True
            raise

    def commit_batch(self, proofs, *, acknowledge):
        if self.failed or self.active is None:
            raise RuntimeError('coalescer is inactive or poisoned')
        try:
            proofs = tuple(proofs)
            if len(proofs) != len(self.active):
                raise RuntimeError('missing captured child proof')
            for child, proof in zip(self.active, proofs):
                if not isinstance(proof, dict) or type(proof.get('sequence_no')) is not int or proof['sequence_no'] != child.sequence_no or (type(proof.get('token_count')) is not int) or (proof['token_count'] != child.original_tokens):
                    raise RuntimeError('captured child proof differs from original source')
                state = self.commands[child.command_step]
                if child.sequence_no in state['published']:
                    raise RuntimeError('duplicate logical publication')
            for child, proof in zip(self.active, proofs):
                self.commands[child.command_step]['published'][child.sequence_no] = proof
            ready = []
            for step, state in sorted(self.commands.items()):
                if len(state['published']) != len(state['children']):
                    break
                ready.append((step, state))
            for step, state in ready:
                acknowledge(state['command'], tuple((state['published'][c.sequence_no] for c in state['children'])))
                del self.commands[step]
            self.original_tokens_completed += sum((c.original_tokens for c in self.active))
            self.active = None
            return len(ready)
        except BaseException:
            self.failed = True
            raise
