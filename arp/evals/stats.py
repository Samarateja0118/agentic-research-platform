"""Paired bootstrap for differences between two systems on the same questions."""

from __future__ import annotations

import random


def paired_bootstrap(
    a: list[float], b: list[float], iters: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (b - a) with a 95% percentile interval."""
    assert len(a) == len(b) and a
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]
