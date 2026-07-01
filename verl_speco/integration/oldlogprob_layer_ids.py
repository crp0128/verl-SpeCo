"""Layer-id selection for SPECO old-logprob hidden-state collection."""

from __future__ import annotations

from typing import Any


def _get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _normalize_layer_ids(layer_ids: Any) -> list[int] | None:
    if layer_ids is None:
        return None
    if isinstance(layer_ids, int):
        return [int(layer_ids)]
    if isinstance(layer_ids, str):
        raw = layer_ids.strip()
        if not raw:
            return None
        if raw.startswith("["):
            import json

            layer_ids = json.loads(raw)
        else:
            layer_ids = [part.strip() for part in raw.split(",") if part.strip()]
    return [int(layer_id) for layer_id in list(layer_ids)]


def _config_architectures(config: Any) -> list[str]:
    architectures = _get_nested(config, ("architectures",), []) or []
    if isinstance(architectures, str):
        return [architectures]
    return [str(architecture) for architecture in architectures]


def _is_dflash_config(drafter_cfg: Any, model_configs: tuple[Any, ...]) -> bool:
    algorithm = str(_get_nested(drafter_cfg, ("speculative_algorithm",), "") or "").upper()
    if algorithm == "DFLASH":
        return True
    return any("DFlashDraftModel" in _config_architectures(config) for config in model_configs)


def _generic_aux_layer_ids_from_config(config: Any) -> list[int] | None:
    candidates = (
        ("model", "eagle_config", "target_hidden_layer_ids"),
        ("model", "eagle_config", "eagle_aux_hidden_state_layer_ids"),
        ("eagle_config", "target_hidden_layer_ids"),
        ("eagle_config", "eagle_aux_hidden_state_layer_ids"),
        ("target_hidden_layer_ids",),
        ("eagle_aux_hidden_state_layer_ids",),
        ("target_layer_ids",),
    )
    for path in candidates:
        layer_ids = _normalize_layer_ids(_get_nested(config, path, None))
        if layer_ids is not None:
            return layer_ids
    return None


def eagle3_num_aux_hidden_states_from_config(config: Any) -> int | None:
    layer_ids = _generic_aux_layer_ids_from_config(config)
    return len(layer_ids) if layer_ids is not None else None


def _dflash_target_layer_ids_from_config(config: Any) -> list[int] | None:
    top_level = _normalize_layer_ids(_get_nested(config, ("target_layer_ids",), None))
    nested = _normalize_layer_ids(_get_nested(config, ("dflash_config", "target_layer_ids"), None))
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError(f"DFlash target_layer_ids conflict with dflash_config.target_layer_ids: {top_level} != {nested}")
    if top_level is not None:
        return top_level
    return nested


def _build_dflash_target_layer_ids(num_context_layers: int, num_hidden_layers: int) -> list[int]:
    num_context_layers = int(num_context_layers)
    num_hidden_layers = int(num_hidden_layers)
    if num_context_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_context_layers - 1))) for i in range(num_context_layers)]


def _dflash_num_context_layers(drafter_cfg: Any, model_configs: tuple[Any, ...]) -> int:
    training_cfg = _get_nested(drafter_cfg, ("training",), {}) or {}
    candidates = (
        _get_nested(training_cfg, ("dflash_num_target_layers",), None),
        _get_nested(drafter_cfg, ("num_context_layers",), None),
        _get_nested(drafter_cfg, ("dflash_config", "num_context_layers"), None),
    )
    for config in model_configs:
        candidates = (
            *candidates,
            _get_nested(config, ("num_context_layers",), None),
            _get_nested(config, ("dflash_config", "num_context_layers"), None),
        )
    for value in candidates:
        if value is not None:
            return int(value)
    return 5


def _default_eagle3_aux_layer_ids(num_hidden_layers: int) -> list[int]:
    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers <= 0:
        raise RuntimeError(f"SPECO cannot derive EAGLE3 aux hidden layers from num_hidden_layers={num_hidden_layers}")
    return [2, num_hidden_layers // 2, num_hidden_layers - 3]


def resolve_oldlogprob_aux_layer_ids(
    drafter_cfg: Any,
    *,
    target_num_hidden_layers: int | None,
    model_configs: list[Any] | tuple[Any, ...] = (),
) -> list[int] | None:
    """Resolve target layer ids to capture during old-logprob hidden collection."""

    model_configs = tuple(config for config in model_configs if config is not None)
    if _is_dflash_config(drafter_cfg, model_configs):
        for config in (drafter_cfg, *model_configs):
            layer_ids = _dflash_target_layer_ids_from_config(config)
            if layer_ids is not None:
                return layer_ids
        if target_num_hidden_layers is None:
            return None
        return _build_dflash_target_layer_ids(
            _dflash_num_context_layers(drafter_cfg, model_configs),
            int(target_num_hidden_layers),
        )

    for config in (drafter_cfg, *model_configs):
        layer_ids = _generic_aux_layer_ids_from_config(config)
        if layer_ids is not None:
            return layer_ids
    if target_num_hidden_layers is None:
        return None
    return _default_eagle3_aux_layer_ids(target_num_hidden_layers)
