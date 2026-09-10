from __future__ import annotations

import sys
import types

import pytest


def _install_fake_verl_v1(monkeypatch):
    """Provide enough of the V1 API to test factory selection without Ray."""
    verl = types.ModuleType("verl")
    trainer = types.ModuleType("verl.trainer")
    ppo = types.ModuleType("verl.trainer.ppo")
    v1 = types.ModuleType("verl.trainer.ppo.v1")

    class Base:
        pass

    v1.PPOTrainerSync = Base
    v1.PPOTrainerColocateAsync = type("Colocate", (Base,), {})
    v1.PPOTrainerSeparateAsync = type("Separate", (Base,), {})
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.trainer", trainer)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", ppo)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1", v1)


def test_factory_selects_all_v1_modes(monkeypatch):
    _install_fake_verl_v1(monkeypatch)
    from verl_speco.trainer.v1.factory import get_speco_v1_trainer_cls

    assert get_speco_v1_trainer_cls("sync").__name__ == "SpecoV1SyncTrainer"
    assert get_speco_v1_trainer_cls("colocate_async").__name__ == "SpecoV1ColocateAsyncTrainer"
    assert get_speco_v1_trainer_cls("separate_async").__name__ == "SpecoV1SeparateAsyncTrainer"


def test_factory_rejects_unknown_mode(monkeypatch):
    _install_fake_verl_v1(monkeypatch)
    from verl_speco.trainer.v1.factory import get_speco_v1_trainer_cls

    with pytest.raises(ValueError, match="Unknown SPECO V1 trainer mode"):
        get_speco_v1_trainer_cls("unsupported")
