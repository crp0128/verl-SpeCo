"""Small, dependency-light V1 feature representation.

This is intentionally limited to ordinary V1 PPO batches in Phase 1.  Agent
request/turn fields and context-plus-assistant slicing are added in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class V1DrafterBatch:
    """Explicit names for the fields needed by future V1 collection hooks."""

    input_ids: Any
    attention_mask: Any
    response_mask: Any | None = None
    hidden_states: Any | None = None
    target_logprobs: Any | None = None
    global_step: int | None = None


def from_transfer_queue_batch(batch: Any, *, global_step: int | None = None) -> V1DrafterBatch:
    """Build a feature view without assuming a legacy ``DataProto`` object.

    ``batch`` may be a TensorDict-like object or a mapping.  Missing optional
    fields stay ``None``; no padding or token-window policy is applied here.
    """

    def get(name: str, default=None):
        if hasattr(batch, "get"):
            return batch.get(name, default)
        try:
            return batch[name]
        except (KeyError, TypeError):
            return default

    return V1DrafterBatch(
        input_ids=get("input_ids"),
        attention_mask=get("attention_mask"),
        response_mask=get("response_mask"),
        hidden_states=get("hidden_states"),
        target_logprobs=get("target_logprobs"),
        global_step=global_step,
    )
