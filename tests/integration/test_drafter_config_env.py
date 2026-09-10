from __future__ import annotations

import os

from verl_speco.integration.drafter_config_env import (
    LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV,
    SPECO_DRAFTER_CONFIG_ENV,
    clear_drafter_config_env,
    get_drafter_config_env,
    set_drafter_config_env,
)


def test_drafter_config_env_reads_the_legacy_name(monkeypatch) -> None:
    monkeypatch.delenv(SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.setenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, "legacy-config")

    assert get_drafter_config_env() == "legacy-config"


def test_drafter_config_env_writes_only_the_canonical_name(monkeypatch) -> None:
    monkeypatch.delenv(SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.setenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, "legacy-config")

    set_drafter_config_env("canonical-config")

    assert get_drafter_config_env() == "canonical-config"
    assert os.environ[SPECO_DRAFTER_CONFIG_ENV] == "canonical-config"
    assert LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV not in os.environ


def test_clear_drafter_config_env_removes_both_names(monkeypatch) -> None:
    monkeypatch.setenv(SPECO_DRAFTER_CONFIG_ENV, "canonical-config")
    monkeypatch.setenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, "legacy-config")

    clear_drafter_config_env()

    assert get_drafter_config_env() == ""
