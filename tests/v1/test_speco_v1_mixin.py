from __future__ import annotations

import sys
import types

import pytest

from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _config(*, mode="sync", bypass=False, rollout_name="vllm"):
    return AttrDict(
        actor_rollout_ref=AttrDict(
            actor=AttrDict(),
            rollout=AttrDict(
                name=rollout_name,
                drafter=AttrDict(
                    enable=True,
                    enable_drafter_training=True,
                    training=AttrDict(
                        collect_hidden_states_from_sgl=False,
                        collect_hidden_states_from_old_logprob=True,
                    ),
                ),
            ),
        ),
        algorithm=AttrDict(
            rollout_correction=AttrDict(bypass_mode=bypass),
        ),
        trainer=AttrDict(v1=AttrDict(trainer_mode=mode)),
    )


def test_worker_group_facade_delegates_attach_and_require(monkeypatch):
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    attached = []
    worker_group = object()
    monkeypatch.setattr(
        SpecoRayPPOTrainer,
        "attach_speco_worker_group",
        lambda trainer, group: attached.append((trainer, group)),
    )
    monkeypatch.setattr(
        SpecoRayPPOTrainer,
        "_require_speco_worker_group",
        lambda trainer: worker_group,
    )

    class Harness(SpecoV1Mixin):
        pass

    trainer = Harness()
    trainer.attach_speco_worker_group(worker_group)
    assert attached == [(trainer, worker_group)]
    assert trainer._require_speco_worker_group() is worker_group


def test_update_actor_skips_drafter_execution_when_plan_does_not_launch():
    class Plan:
        launch = False
        reason = "training_interval_not_reached"

    class Event:
        training_plan = Plan()
        metrics = {"drafter/training_plan_launch": 0}

    class Upstream:
        def _update_actor(self, batch, metrics):
            metrics["upstream/updated"] = 1
            return "updated"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()

        def _speco_online_enabled_from_config(self, config):
            return True

        def _speco_on_before_actor_update(self):
            return Event()

        def _speco_train_drafter(self, plan):
            raise AssertionError("non-launch plan must not execute drafter training")

    metrics = {}
    assert Harness()._update_actor(None, metrics) == "updated"
    assert metrics["drafter/trained"] == 0
    assert metrics["drafter/train_no_trainable_batch"] == 0
    assert metrics["upstream/updated"] == 1


def test_online_training_rejects_rollout_correction_bypass():
    with pytest.raises(ValueError, match="bypass_mode=true"):
        SpecoV1Mixin._speco_validate_v1_training_config(_config(bypass=True))


def test_online_training_rejects_async_mode():
    with pytest.raises(ValueError, match="trainer_mode=sync only"):
        SpecoV1Mixin._speco_validate_v1_training_config(
            _config(mode="colocate_async")
        )


def test_setup_resolves_drafter_before_upstream_and_creates_worker_after(monkeypatch):
    events = []
    runtime = types.ModuleType("verl_speco.integration.vllm_runtime")
    runtime.configure_vllm_runtime_from_config = lambda config: events.append("configure")
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    class Upstream:
        def _setup(self):
            events.append("upstream")
            return "ready"

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()

        def _speco_validate_v1_training_config(self, config):
            events.append("validate")

        def _speco_init_state(self):
            events.append("state")

        def _speco_prepare_drafter_checkpoint_for_worker_init(self):
            events.append("resume")

        def _init_v1_speco_drafter_workers(self):
            events.append("workers")

    assert Harness()._setup() == "ready"
    assert events == [
        "validate",
        "state",
        "resume",
        "configure",
        "upstream",
        "workers",
    ]


def test_init_creates_drafter_before_first_rollout_weight_update(monkeypatch):
    events = []
    runtime = types.ModuleType("verl_speco.integration.vllm_runtime")
    runtime.configure_vllm_runtime_from_config = lambda config: events.append("configure")
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    class Upstream:
        def init(self):
            self._setup()
            self.on_init_end()

        def _setup(self):
            events.append("upstream")

        def on_init_end(self):
            events.append("update_weights")

    class Harness(SpecoV1Mixin, Upstream):
        config = _config()
        trainer_mode = "sync"

        def _speco_validate_v1_training_config(self, config):
            events.append("validate")

        def _speco_init_state(self):
            events.append("state")

        def _speco_prepare_drafter_checkpoint_for_worker_init(self):
            events.append("resume")

        def _init_v1_speco_drafter_workers(self):
            events.append("workers")

    Harness().init()
    assert events.index("resume") < events.index("upstream")
    assert events.index("workers") < events.index("update_weights")


def test_checkpoint_result_flattening_is_available_on_v1_trainer() -> None:
    nested = [{"saved": True}, ({"saved": False}, [{"saved": True}])]

    assert SpecoV1Mixin._speco_flatten_checkpoint_results(nested) == [
        {"saved": True},
        {"saved": False},
        {"saved": True},
    ]
