# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Pure hard-sample candidate partitioning helpers."""

from __future__ import annotations

import math
from typing import Any


def kde_hard_partition(
    scored_candidates: list[dict[str, Any]],
    *,
    acceptance_values: list[float],
    training_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """HTML-v1 rule: rightmost dominant peak, merged left peaks, KDE valley.

    The population fraction is measured on all finite request scores, not just
    rows eligible for hidden-state collection. A zero multiplier still detects
    and labels the hard pool so the collector can exclude it on bimodal steps.
    Single/degenerate populations return no labels and ordinary collection.
    """
    if not math.isfinite(training_multiplier) or training_multiplier < 0:
        raise ValueError("hard training multiplier must be finite and nonnegative")
    import numpy as np
    from scipy.signal import find_peaks

    values = np.asarray(acceptance_values, dtype=float)
    values = values[np.isfinite(values)]
    stats = {
        "kde_bimodal": 0.0,
        "kde_boundary": -1.0,
        "kde_population_count": float(values.size),
        "kde_population_hard_count": 0.0,
        "effective_candidate_ratio": 0.0,
        "effective_sample_ratio": 0.0,
        "training_multiplier": float(training_multiplier),
    }
    if values.size < 3 or float(np.ptp(values)) <= 1e-12:
        return [], stats
    # Robust univariate Silverman rule (not scipy's multivariate factor).
    std = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    scale = min(std, float(q75 - q25) / 1.34)
    if scale <= 0:
        scale = std
    bandwidth = 0.9 * scale * values.size ** (-0.2)
    if not math.isfinite(bandwidth) or bandwidth <= 1e-12:
        return [], stats
    grid = np.linspace(float(values.min()) - 3 * bandwidth,
                       float(values.max()) + 3 * bandwidth, 1024)
    density = np.exp(-0.5 * ((grid[:, None] - values) / bandwidth) ** 2).mean(axis=1)
    peaks, _ = find_peaks(density, height=float(density.max()) * 0.10,
                          prominence=float(density.max()) * 0.05)
    # Keep the taller peak when two candidates are <= 0.5 tokens apart.
    kept: list[int] = []
    for peak in sorted(peaks.tolist(), key=lambda i: (-density[i], i)):
        if all(abs(float(grid[peak] - grid[other])) > 0.5 for other in kept):
            kept.append(peak)
    kept.sort()
    if len(kept) < 2 or max(kept, key=lambda i: density[i]) != kept[-1]:
        return [], stats
    high = kept[-1]
    low = float(np.average(grid[kept[:-1]], weights=density[kept[:-1]]))
    interior = np.flatnonzero((grid > low) & (grid < grid[high]))
    if not interior.size:
        return [], stats
    boundary = float(grid[interior[np.argmin(density[interior])]])
    count = int(np.count_nonzero(values <= boundary))
    fraction = count / int(values.size)
    if fraction < 0.05:
        return [], stats
    hard = [row for row in scored_candidates
            if math.isfinite(float(row["mean_accept_len"]))
            and float(row["mean_accept_len"]) <= boundary]
    stats.update(kde_bimodal=1.0, kde_boundary=boundary,
                 kde_population_hard_count=float(count),
                 effective_candidate_ratio=fraction,
                 effective_sample_ratio=min(fraction * training_multiplier, 1.0))
    return hard, stats


def partition_hard_candidates(
    scored_candidates: list[dict[str, Any]],
    *,
    candidate_ratio: float,
) -> tuple[list[dict[str, Any]], int]:
    """Return the lowest-acceptance hard candidates and their count."""
    if not math.isfinite(candidate_ratio) or not 0.0 <= candidate_ratio <= 1.0:
        raise ValueError(f"Invalid hard candidate ratio: {candidate_ratio}")

    ordered = sorted(
        scored_candidates,
        key=lambda row: (row["mean_accept_len"], row["hash"]),
    )
    candidate_count = round(len(ordered) * candidate_ratio)
    return ordered[:candidate_count], candidate_count
