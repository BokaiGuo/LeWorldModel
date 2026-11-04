"""Fail-closed selection for deployable candidates, without outcome access."""
from __future__ import annotations

import math

import numpy as np

ALLOWED = frozenset({"fixed_action", "seeded_random", "rgb_model"})


def build_fixed_bank(seed):
    """Build outcome-free bounded prefixes; never read simulator information."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    prefixes = [{"name": "zero", "source": "fixed_action", "prefix": np.zeros((5, 2)).tolist()}]
    for index in range(8):
        direction = np.array([np.cos(index * np.pi / 4), np.sin(index * np.pi / 4)])
        prefixes.append({"name": f"direction_{index}", "source": "fixed_action",
                         "prefix": np.tile(direction, (5, 1)).tolist()})
    for index, action in enumerate(rng.uniform(-1, 1, (2, 5, 2))):
        prefixes.append({"name": f"random_{index}", "source": "seeded_random", "prefix": action.tolist()})
    return prefixes


def select_candidates(candidates, initial_score):
    """Select by predicted terminal cost; reject privileged candidate sources."""
    if not candidates or not math.isfinite(initial_score):
        raise ValueError("finite initial score and candidates required")
    names = [candidate.get("name") for candidate in candidates]
    if len(set(names)) != len(names) or names.count("zero") != 1:
        raise ValueError("unique candidate names and exactly one zero required")
    for candidate in candidates:
        if candidate.get("source") not in ALLOWED:
            raise ValueError("candidate source is not deployable")
        if set(candidate) != {"name", "source", "prefix", "prefix_score", "terminal_score", "suffix"}:
            raise ValueError("unexpected fields, including potential privileged outcome")
        prefix, suffix = np.asarray(candidate["prefix"]), np.asarray(candidate["suffix"])
        if (prefix.shape != (5, 2) or suffix.shape != (10, 2)
                or not np.isfinite(prefix).all() or not np.isfinite(suffix).all()
                or np.max(np.abs(prefix)) > 1.000001 or np.max(np.abs(suffix)) > 1.000001
                or not math.isfinite(candidate["prefix_score"])
                or not math.isfinite(candidate["terminal_score"])):
            raise ValueError("invalid model-scored action program")
    zero = candidates[names.index("zero")]
    if not np.all(np.asarray(zero["prefix"]) == 0):
        raise ValueError("zero candidate is not zero")
    index = min(range(len(candidates)), key=lambda i: candidates[i]["terminal_score"])
    chosen = candidates[index]
    qualified = (chosen["prefix_score"] >= initial_score
                 and chosen["terminal_score"] < zero["terminal_score"])
    return {"selected_index": index, "selected_name": chosen["name"],
            "qualified_against_zero_only": qualified,
            "abstained_reconfiguration": not qualified}
