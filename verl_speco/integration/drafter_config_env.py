"""Canonical environment transport for SPECO drafter configuration."""

from __future__ import annotations

import os

SPECO_DRAFTER_CONFIG_ENV = "VERL_SPECO_DRAFTER_CONFIG"
LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV = "VERL_SPECO_SGLANG_DRAFTER_CONFIG"


def get_drafter_config_env(default: str = "") -> str:
    """Return the canonical value, falling back to the legacy environment name."""

    return os.getenv(SPECO_DRAFTER_CONFIG_ENV) or os.getenv(
        LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, default
    )


def set_drafter_config_env(value: str) -> None:
    """Publish a drafter config under the canonical name only."""

    os.environ[SPECO_DRAFTER_CONFIG_ENV] = value
    os.environ.pop(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, None)


def clear_drafter_config_env() -> None:
    """Clear both the canonical and legacy names."""

    os.environ.pop(SPECO_DRAFTER_CONFIG_ENV, None)
    os.environ.pop(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, None)
