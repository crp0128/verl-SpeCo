# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from typing import Callable, Any
projection_observer_ATTRIBUTE = '_speco_projection_output_observer'

class ProjectionObserver:

    def __init__(self, projection: Any):
        self.projection = projection
        self.callbacks: dict[object, Callable] = {}
        self.original = projection.hc_pre
        self.had_instance_method = 'hc_pre' in vars(projection)
        self.instance_method = vars(projection).get('hc_pre')

        def observed(*args, **kwargs):
            result = self.original(*args, **kwargs)
            for callback in tuple(self.callbacks.values()):
                callback(result[0])
            return result
        self.observed = observed
        projection.hc_pre = observed

    def remove(self, key):
        self.callbacks.pop(key, None)
        if self.callbacks:
            return
        if self.projection.hc_pre is not self.observed:
            delattr(self.projection, projection_observer_ATTRIBUTE)
            raise RuntimeError('mHC projection changed while capture observer was active')
        if self.had_instance_method:
            self.projection.hc_pre = self.instance_method
        else:
            del self.projection.hc_pre
        delattr(self.projection, projection_observer_ATTRIBUTE)

class Subscription:

    def __init__(self, observer, key):
        self.observer, self.key = (observer, key)

    def remove(self):
        if self.key is not None:
            try:
                self.observer.remove(self.key)
            finally:
                self.key = None

def observe_projection(projection, callback):
    observer = getattr(projection, projection_observer_ATTRIBUTE, None)
    if observer is None:
        observer = ProjectionObserver(projection)
        setattr(projection, projection_observer_ATTRIBUTE, observer)
    elif projection.hc_pre is not observer.observed:
        raise RuntimeError('mHC projection changed while capture observer was active')
    key = object()
    observer.callbacks[key] = callback
    return Subscription(observer, key)
from dataclasses import dataclass
from typing import Any, Callable, Mapping
import torch

@dataclass(frozen=True)
class Captured92BFeatures:
    request_id: str
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    layer_ids: tuple[int, ...]
    hidden_states: tuple[torch.Tensor, ...]
    final_hidden_state: torch.Tensor

class MultiNode92BSingleRequestCapture:
    """Capture one full-prefill forward with explicit token/position identity.

    The caller must arrange an isolated request. This class refuses mismatched
    model inputs or row counts; batched/prefix-cache/TP transport are separate
    capabilities and must not silently inherit this contract.
    """

    def __init__(self, model: Any, *, layer_ids: tuple[int, ...], hidden_size: int, max_payload_bytes: int, token_ids_supplier: Callable[[], torch.Tensor] | None=None, request_ids_supplier: Callable[[], list[str]] | None=None, request_row_ranges_supplier: Callable[[], Mapping[str, tuple[int, int]]] | None=None, capture_decode: bool=False, allow_chunked_prefill: bool=False, reuse_projection_output: bool=False) -> None:
        trunk = getattr(model, 'model', None)
        layers = getattr(trunk, 'layers', None)
        if trunk is None or layers is None or (not layer_ids):
            raise ValueError('target model text trunk and selected layers are required')
        if tuple(sorted(set(layer_ids))) != layer_ids:
            raise ValueError('layer_ids must be unique and ascending')
        if hidden_size <= 0 or max_payload_bytes <= 0:
            raise ValueError('hidden size and byte limit must be positive')
        self.model = model
        self.layer_ids = layer_ids
        self.hidden_size = hidden_size
        self.max_payload_bytes = max_payload_bytes
        self.token_ids_supplier = token_ids_supplier
        self.request_ids_supplier = request_ids_supplier
        self.request_row_ranges_supplier = request_row_ranges_supplier
        self.capture_decode = capture_decode
        if allow_chunked_prefill and request_row_ranges_supplier is not None:
            raise ValueError('chunked prefill requires an isolated request')
        self.allow_chunked_prefill = allow_chunked_prefill
        self._prefill_rows = 0
        self._prefill_chunk_size = 0
        self._prefill_chunks: dict[int, list[torch.Tensor]] = {item: [] for item in layer_ids}
        self._prefill_final: list[torch.Tensor] = []
        self._handles: list[Any] = []
        self._expected: tuple[str, tuple[int, ...], tuple[int, ...]] | None = None
        self._captured: dict[int, torch.Tensor] = {}
        self._final: torch.Tensor | None = None
        self._seen_forward = False
        self._completed = False
        self._engine_request_id: str | None = None
        self._capturing_decode = False
        self._skip_forward = False
        self._row_start = 0
        self._decode_token_ids: list[int] = []
        self._decode_positions: list[int] = []
        self._decode_captured: dict[int, list[torch.Tensor]] = {item: [] for item in layer_ids}
        self._decode_final: list[torch.Tensor] = []
        last = len(layers) - 1
        sites: list[tuple[int, Any]] = []
        for layer_id in layer_ids:
            if layer_id < 0 or layer_id > last:
                raise ValueError(f'target layer {layer_id} is outside 0..{last}')
            if layer_id == last:
                projection = getattr(trunk, 'merge_mhc_module', None)
            else:
                projection = getattr(layers[layer_id + 1], 'attn_mhc_module', None)
            if projection is None or not callable(getattr(projection, 'hc_pre', None)):
                raise ValueError(f'92B mHC projection after layer {layer_id} is unavailable')
            sites.append((layer_id, projection))
        try:
            for layer_id, projection in sites:
                if reuse_projection_output:
                    self._handles.append(observe_projection(projection, self._projection_hook(layer_id)))
                else:
                    self._handles.append(layers[layer_id].register_forward_hook(self._layer_hook(layer_id, projection)))
            self._handles.append(trunk.norm.register_forward_hook(self._final_hook))
            self._handles.append(model.register_forward_pre_hook(self._input_hook, with_kwargs=True))
            self._handles.append(model.register_forward_hook(self._complete_hook))
        except BaseException:
            self.close()
            raise

    def begin(self, *, request_id: str, token_ids: tuple[int, ...], positions: tuple[int, ...], engine_request_id: str | None=None) -> None:
        if self._expected is not None:
            raise RuntimeError('a capture request is already active')
        if not request_id or not token_ids or len(token_ids) != len(positions):
            raise ValueError('capture identity and token/position rows are required')
        if positions != tuple(range(len(positions))):
            raise ValueError('first release supports complete prefill from position zero only')
        self._expected = (request_id, token_ids, positions)
        self._engine_request_id = engine_request_id
        self._captured.clear()
        self._final = None
        self._seen_forward = False
        self._completed = False
        self._capturing_decode = False
        self._skip_forward = False
        self._row_start = 0
        self._prefill_rows = 0
        self._prefill_chunk_size = 0
        for chunks in self._prefill_chunks.values():
            chunks.clear()
        self._prefill_final.clear()
        self._decode_token_ids.clear()
        self._decode_positions.clear()
        for rows in self._decode_captured.values():
            rows.clear()
        self._decode_final.clear()

    def _input_hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self._expected is None:
            return
        self._skip_forward = False
        row_end: int | None = None
        if self.request_row_ranges_supplier is not None:
            if self._engine_request_id is None:
                raise RuntimeError('batched capture requires an engine request ID')
            rows = self.request_row_ranges_supplier().get(self._engine_request_id)
            if rows is None or rows[0] == rows[1]:
                self._skip_forward = True
                return
            self._row_start, row_end = rows
            if self._row_start < 0 or row_end <= self._row_start:
                raise RuntimeError('batched capture request rows are invalid')
        prefill_done = self._prefill_rows == len(self._expected[1])
        if prefill_done and (not self.capture_decode):
            return
        if self._engine_request_id is not None and self.request_row_ranges_supplier is None:
            if self.request_ids_supplier is None:
                raise RuntimeError('capture requires worker request IDs')
            active_ids = [item for item in self.request_ids_supplier() if item is not None]
            if active_ids != [self._engine_request_id]:
                raise RuntimeError('capture worker request ID does not match the armed request')
        ids = kwargs.get('input_ids', args[0] if args else None)
        positions = kwargs.get('positions', args[1] if len(args) > 1 else None)
        embeds = kwargs.get('inputs_embeds')
        if ids is None and isinstance(embeds, torch.Tensor) and (self.token_ids_supplier is not None):
            ids = self.token_ids_supplier()
            required_rows = 1 if prefill_done or self.allow_chunked_prefill else len(self._expected[1])
            if embeds.ndim != 2 or embeds.shape[0] < required_rows or (row_end is not None and embeds.shape[0] < row_end):
                raise RuntimeError('capture embedding rows are shorter than the requested prefill')
        if not isinstance(ids, torch.Tensor) or not isinstance(positions, torch.Tensor):
            raise RuntimeError('capture requires runner token IDs and model positions')
        if row_end is not None:
            if ids.numel() < row_end or positions.shape[-1] < row_end:
                raise RuntimeError('batched capture row range exceeds model inputs')
            ids = ids.reshape(-1)[self._row_start:row_end]
            positions = positions[..., self._row_start:row_end]
        if prefill_done:
            if row_end is not None and row_end - self._row_start != 1:
                raise RuntimeError('batched decode must schedule one token per request')
            if not self._completed:
                raise RuntimeError('decode began before the preceding model forward completed')
            next_position = len(self._expected[1]) + len(self._decode_token_ids)
            if positions.ndim == 1 and positions.numel() >= 1:
                decode_positions = (int(positions[0]),)
            elif positions.ndim == 2 and positions.shape[0] == 3 and (positions.shape[1] >= 1):
                decode_positions = tuple((int(positions[axis, 0]) for axis in range(3)))
            else:
                raise RuntimeError(f'decode position layout {tuple(positions.shape)} is unsupported')
            if any((item != next_position for item in decode_positions)):
                raise RuntimeError(f'decode position does not continue the captured prompt: expected={next_position}, actual={decode_positions}')
            if ids.numel() < 1:
                raise RuntimeError('decode token ID is missing from runner buffer')
            self._decode_token_ids.append(int(ids.reshape(-1)[0]))
            self._decode_positions.append(next_position)
            self._capturing_decode = True
            self._completed = False
            return
        _, expected_ids, expected_positions = self._expected
        count = len(expected_ids)
        if self.allow_chunked_prefill:
            if self._seen_forward and (not self._completed):
                raise RuntimeError('prefill chunk began before the preceding forward completed')
            count = min(positions.shape[-1], len(expected_ids) - self._prefill_rows)
            if count <= 0:
                raise RuntimeError('prefill chunk has no token rows')
            expected_ids = expected_ids[self._prefill_rows:self._prefill_rows + count]
            expected_positions = expected_positions[self._prefill_rows:self._prefill_rows + count]
        if row_end is not None and row_end - self._row_start != count:
            raise RuntimeError('batched request prefill was chunked or shortened')
        if positions.ndim == 1:
            position_rows = (positions[:count],)
        elif positions.ndim == 2 and positions.shape[0] == 3:
            position_rows = tuple((positions[axis, :count] for axis in range(3)))
        else:
            raise RuntimeError('capture model positions have an unsupported layout')
        if ids.numel() < count or tuple(ids.reshape(-1)[:count].tolist()) != expected_ids or any((tuple(row.tolist()) != expected_positions for row in position_rows)):
            raise RuntimeError('capture model rows do not match the requested full prefill')
        self._seen_forward = True
        self._capturing_decode = False
        self._prefill_chunk_size = count
        self._completed = False
        self._captured.clear()
        self._final = None

    def _complete_hook(self, _module: Any, _args: tuple[Any, ...], _result: Any) -> None:
        if self._expected is not None and self._seen_forward and (not self._skip_forward):
            if not self._completed and (not self._capturing_decode):
                if tuple(sorted(self._captured)) != self.layer_ids or self._final is None:
                    raise RuntimeError('prefill chunk is missing capture features')
                self._prefill_rows += self._prefill_chunk_size
                if self.allow_chunked_prefill:
                    for layer_id in self.layer_ids:
                        self._prefill_chunks[layer_id].append(self._captured[layer_id].clone())
                    self._prefill_final.append(self._final.clone())
                    if self._prefill_rows == len(self._expected[1]):
                        self._captured = {layer_id: torch.cat(chunks, dim=0) for layer_id, chunks in self._prefill_chunks.items()}
                        self._final = torch.cat(self._prefill_final, dim=0)
                        for chunks in self._prefill_chunks.values():
                            chunks.clear()
                        self._prefill_final.clear()
            self._completed = True

    def _layer_hook(self, layer_id: int, projection: Any):

        def hook(_module: Any, _args: tuple[Any, ...], result: Any) -> None:
            if self._expected is None or self._completed or self._skip_forward:
                return
            rows = 1 if self._capturing_decode else self._prefill_chunk_size
            start = self._row_start if self.request_row_ranges_supplier is not None else 0
            project = getattr(projection, '_speco_diagnostic_original_hc_pre', projection.hc_pre)
            tensor = project(result[start:start + rows])[0]
            validated = self._validate_tensor(tensor, rows=rows)
            if self._capturing_decode:
                self._decode_captured[layer_id].append(validated)
            else:
                self._captured[layer_id] = validated
        return hook

    def _projection_hook(self, layer_id: int):

        def hook(tensor: Any) -> None:
            if self._expected is None or self._completed or self._skip_forward:
                return
            rows = 1 if self._capturing_decode else self._prefill_chunk_size
            start = self._row_start if self.request_row_ranges_supplier is not None else 0
            validated = self._validate_tensor(tensor[start:start + rows], rows=rows)
            if self._capturing_decode:
                self._decode_captured[layer_id].append(validated)
            else:
                self._captured[layer_id] = validated
        return hook

    def _final_hook(self, _module: Any, _args: tuple[Any, ...], result: Any) -> None:
        if self._expected is not None and (not self._completed) and (not self._skip_forward):
            rows = 1 if self._capturing_decode else self._prefill_chunk_size
            start = self._row_start if self.request_row_ranges_supplier is not None else 0
            validated = self._validate_tensor(result[start:start + rows], rows=rows)
            if self._capturing_decode:
                self._decode_final.append(validated)
            else:
                self._final = validated

    def _validate_tensor(self, tensor: Any, *, rows: int) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor) or self._expected is None:
            raise RuntimeError('capture hook did not produce a tensor')
        if tensor.ndim != 2 or tensor.shape != (rows, self.hidden_size):
            raise RuntimeError(f'capture hidden shape {tuple(tensor.shape)} != ({rows}, {self.hidden_size})')
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f'unsupported capture hidden dtype {tensor.dtype}')
        return tensor.detach()

    def collect_trajectory(self, *, generated_token_ids: tuple[int, ...]) -> Captured92BFeatures:
        """Return prompt plus actual decode inputs, aligned to response[:-1]."""
        if not self.capture_decode:
            raise RuntimeError('trajectory collection was not enabled')
        if not generated_token_ids or not self._completed or self._expected is None or (self._prefill_rows != len(self._expected[1])):
            raise RuntimeError('generation has not completed')
        captured_decode_ids = tuple(self._decode_token_ids)
        expected_decode_ids = generated_token_ids[:-1]
        if captured_decode_ids != expected_decode_ids and captured_decode_ids != generated_token_ids:
            raise RuntimeError(f'captured decode input IDs do not match generated response: captured={captured_decode_ids}, expected={expected_decode_ids}')
        if tuple(sorted(self._captured)) != self.layer_ids or self._final is None:
            raise RuntimeError('trajectory is missing prompt prefill features')
        captured_count = len(captured_decode_ids)
        count = len(expected_decode_ids)
        if len(self._decode_final) != captured_count or any((len(self._decode_captured[layer_id]) != captured_count for layer_id in self.layer_ids)):
            raise RuntimeError('trajectory is missing decode feature rows')
        _, prompt_ids, prompt_positions = self._expected
        hidden = tuple((torch.cat((self._captured[layer_id], *self._decode_captured[layer_id][:count]), dim=0) for layer_id in self.layer_ids))
        final = torch.cat((self._final, *self._decode_final[:count]), dim=0)
        nbytes = sum((item.numel() * item.element_size() for item in (*hidden, final)))
        if nbytes > self.max_payload_bytes:
            raise RuntimeError('trajectory exceeds configured payload byte limit')
        result = Captured92BFeatures(self._expected[0], prompt_ids + expected_decode_ids, prompt_positions + tuple(self._decode_positions[:count]), self.layer_ids, hidden, final)
        self.abort()
        return result

    def collect(self) -> Captured92BFeatures:
        if self._expected is None or not self._completed:
            raise RuntimeError('capture has no completed model forward')
        if self._prefill_rows != len(self._expected[1]):
            raise RuntimeError('capture has an incomplete chunked prefill')
        if tuple(sorted(self._captured)) != self.layer_ids or self._final is None:
            raise RuntimeError('capture is missing a selected layer or final norm')
        tensors = tuple((self._captured[layer_id] for layer_id in self.layer_ids))
        byte_count = sum((t.numel() * t.element_size() for t in (*tensors, self._final)))
        if byte_count > self.max_payload_bytes:
            raise RuntimeError('capture exceeds configured payload byte limit')
        request_id, token_ids, positions = self._expected
        result = Captured92BFeatures(request_id, token_ids, positions, self.layer_ids, tensors, self._final)
        self._expected = None
        self._engine_request_id = None
        self._captured = {}
        self._final = None
        return result

    def abort(self) -> None:
        self._expected = None
        self._engine_request_id = None
        self._captured.clear()
        self._final = None
        self._seen_forward = False
        self._completed = False
        self._capturing_decode = False
        self._skip_forward = False
        self._row_start = 0
        self._prefill_rows = 0
        self._prefill_chunk_size = 0
        for chunks in self._prefill_chunks.values():
            chunks.clear()
        self._prefill_final.clear()
        self._decode_token_ids.clear()
        self._decode_positions.clear()
        for rows in self._decode_captured.values():
            rows.clear()
        self._decode_final.clear()

    def close(self) -> None:
        self.abort()
        errors = []
        for handle in self._handles:
            try:
                handle.remove()
            except BaseException as error:
                errors.append(error)
        self._handles.clear()
        if errors:
            raise RuntimeError('capture observer cleanup failed') from errors[0]
