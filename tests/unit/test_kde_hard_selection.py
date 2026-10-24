"""HTML rule and real collector-path contracts, without starting Ray/NPU jobs."""
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verl_speco.trainer.hard_sample_selection import kde_hard_partition, partition_hard_candidates


def population(low=96, high=224):
    return [*np.linspace(1.9, 2.1, low), *np.linspace(3.9, 4.1, high)]


def rows(values):
    return [dict(batch_idx=i, mean_accept_len=float(v), hash=i / len(values))
            for i, v in enumerate(values)]


@pytest.mark.parametrize("multiplier", [0, 0.1, 1, 2])
def test_bimodal_fraction_times_multiplier(multiplier):
    values = population()
    hard, stats = kde_hard_partition(rows(values), acceptance_values=values,
                                     training_multiplier=multiplier)
    assert stats["kde_bimodal"] == 1
    assert len(hard) == 96
    assert stats["effective_candidate_ratio"] == 0.3
    assert stats["effective_sample_ratio"] == pytest.approx(0.3 * multiplier)
    assert 2.1 < stats["kde_boundary"] < 3.9


@pytest.mark.parametrize("values", [[], [2], [2] * 320, population(240, 80)])
def test_single_or_left_dominant_is_ordinary_per_html(values):
    hard, stats = kde_hard_partition(rows(values), acceptance_values=values, training_multiplier=0)
    assert not hard
    assert stats["kde_bimodal"] == 0


def test_multipeak_left_merge_and_population_not_eligible_fraction():
    values = [*np.linspace(0.9, 1.1, 32), *np.linspace(1.9, 2.1, 64),
              *np.linspace(4.9, 5.1, 224)]
    hard, stats = kde_hard_partition(rows(values)[:48], acceptance_values=values, training_multiplier=0.1)
    assert stats["kde_bimodal"] == 1
    assert len(hard) == 48
    assert stats["kde_population_hard_count"] == 96
    assert stats["effective_sample_ratio"] == pytest.approx(0.03)


@pytest.mark.parametrize("multiplier", [-1, float("nan"), float("inf")])
def test_invalid_multiplier(multiplier):
    with pytest.raises(ValueError):
        kde_hard_partition([], acceptance_values=[], training_multiplier=multiplier)


def test_nonfinite_scores_and_repeatability():
    values = population() + [float("nan"), float("inf")]
    first = kde_hard_partition(rows(values), acceptance_values=values, training_multiplier=0)
    assert first == kde_hard_partition(rows(values), acceptance_values=values, training_multiplier=0)
    assert first[1]["kde_population_count"] == 320
    assert len(first[0]) == 96


def collector(values, *, mode="kde_hard", multiplier=0):
    import torch
    source = Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpecoRayPPOTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_speco_build_oldlogprob_collect_plan")
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             ast.ClassDef(name="Subject", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    namespace = dict(torch=torch, math=math, hashlib=hashlib,
                     partition_hard_candidates=partition_hard_candidates, kde_hard_partition=kde_hard_partition,
                     DrafterCollectionSource=SimpleNamespace(OLD_LOGPROB="old_logprob"),
                     _get_nested=lambda cfg, path, default: cfg.get(path[0], default))
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    driver = namespace["Subject"]()
    driver.global_steps = 10
    driver._speco_oldlogprob_collection_enabled = lambda: True
    driver._speco_plan_drafter_collection = lambda source: SimpleNamespace(
        collect=True, sample_rate=1.0, max_samples_per_replica=16, max_tokens_per_replica=16384)
    driver._speco_log_drafter_collection_plan = lambda plan: None
    driver._speco_drafter_training_config = lambda: dict(
        dspark_hard_partition_mode=mode, dspark_hard_training_multiplier=multiplier,
        dspark_hard_candidate_ratio=0.4, dspark_hard_sample_ratio=0.1)
    driver._speco_oldlogprob_window_mode = lambda cfg: "front"
    driver._speco_oldlogprob_window_train_rows = lambda cfg: 512
    driver._speco_owner_bucket_count = lambda: 8
    driver._speco_bubble_collection_owner_ranks = lambda **kw: ()
    driver._speco_rollout_idle_worker_enabled = lambda: False
    driver._speco_drafter_config = lambda: dict(speculative_algorithm="DSPARK")
    driver._speco_request_accept_lengths = lambda batch, size: values
    driver._speco_request_ids = lambda batch, size, **kw: [f"r{i}" for i in range(size)]
    driver._speco_request_completion_indices = lambda batch, size: list(range(size))
    driver._speco_request_elapsed_secs = lambda batch, size: [1.0] * size
    driver._speco_hash_fraction = lambda key: int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    driver._speco_log_request_accept_lens = lambda **kw: None
    size = len(values)
    batch = SimpleNamespace(batch=dict(prompts=torch.ones(size, 4, dtype=torch.long),
        responses=torch.ones(size, 520, dtype=torch.long), attention_mask=torch.ones(size, 524, dtype=torch.long),
        response_mask=torch.ones(size, 520, dtype=torch.long)))
    return driver._speco_build_oldlogprob_collect_plan(batch)


def test_real_collector_zero_multiplier_excludes_hard_not_disables_feature():
    plan = collector(population())
    metrics = plan["selection_metrics"]
    assert metrics["kde_bimodal"] == 1
    assert metrics["candidate_hard_count"] == 96
    assert metrics["target_hard"] == metrics["selected_hard_total"] == 0
    assert metrics["selected_total"] == metrics["selected_unique"] == 80
    assert metrics["selected_normal_total"] == 80
    assert not bool((plan["collect_mask"] & plan["request_is_hard"]).any())
    assert plan["hidden_rows"] == 513


def test_real_collector_ratio_point3_times_point1():
    plan = collector(population(), multiplier=0.1)
    metrics = plan["selection_metrics"]
    assert metrics["effective_sample_ratio"] == pytest.approx(0.03)
    assert metrics["target_hard"] == metrics["selected_hard_total"] == round(80 * 0.03)
    assert metrics["selected_total"] == metrics["selected_unique"] == 80


def test_real_collector_single_peak_restores_original_order():
    plan = collector([3.0] * 320)
    assert plan["selection_metrics"]["kde_bimodal"] == 0
    assert plan["collect_mask"].nonzero().flatten().tolist() == list(range(80))
    assert not plan["hard_enabled"]


def test_real_collector_legacy_fixed_unchanged():
    plan = collector(population(), mode="fixed")
    assert plan["selection_metrics"]["candidate_hard_count"] == 128
    assert plan["selection_metrics"]["selected_hard_total"] == 8
    assert plan["selected_count"] == 80
