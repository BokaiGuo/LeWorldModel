"""Outcome-free finite-budget gradient-plateau admission, not a P1 pass.

Root ranking is frozen before any future physical branch is measured. Both
single-actuator tasks determine admission symmetrically; full-drive is retained
as a control, never removed from the subsequent task panel.
"""
from __future__ import annotations

import math

import numpy as np

TASKS = ("full_drive", "shoulder_only", "wrist_only")
CONTRACT = {
    "schema":"reacher_d2p_model_escape_admission.v1",
    "pool_count":32,"metadata_scan_count":128,"selected_count":8,"horizon_packets":3,
    "gradient_checkpoints":[4,8,16],"cem_samples":128,"cem_steps":3,"cem_elite":16,
    "plateau_absolute_mean_cost_margin":1e-6,"escape_absolute_mean_cost_margin":1e-6,
    "eligibility":"at least one exact single-actuator task has GN8-GN16 improvement <= 1e-6 and GN16-CEM > 1e-6",
    "rank":"descending maximum normalized model escape over eligible single-actuator tasks, then descending minimum over both tasks, then ascending seed",
    "task_retention":list(TASKS),"physical_future_labels_read":False,"P1_admission":False,
    "scope":"Finite-budget model-gradient plateau and sampling escape, not mathematical stationarity, real capability change, calibrated P1 or task success. No outcome-selected roots or tasks.",
}


def metadata_pool(lengths, offsets, excluded, *, seed):
    """Take first32 identity-disjoint histories from a fixed128-seed scan.

    Only episode metadata is used. Rejections are archived before inference;
    a later numerical/render failure cannot replace an admitted root.
    """
    lengths,offsets = np.asarray(lengths),np.asarray(offsets)
    if (lengths.ndim != 1 or offsets.shape != lengths.shape or not len(lengths)
            or lengths.dtype.kind not in "iu" or offsets.dtype.kind not in "iu"
            or np.any(lengths < 0) or np.any(offsets < 0) or type(seed) is not int):
        raise ValueError("complete nonnegative integer source metadata required")
    eligible = np.flatnonzero(lengths >= 20)
    if not len(eligible): raise ValueError("source histories of at least20 rows required")
    roots,rejected,seen = [],[],set()
    for candidate in range(seed,seed+CONTRACT["metadata_scan_count"]):
        rng = np.random.default_rng(candidate ^ 0x5A7E)
        episode = int(eligible[int(rng.integers(len(eligible)))])
        start = int(offsets[episode])+int(rng.integers(0,int(lengths[episode])-20+1))
        row = {"seed":candidate,"source_episode":episode,"source_start_row":start}
        if episode in excluded or episode in seen:
            rejected.append({**row,"reason":"previous_source_episode" if episode in excluded else "pool_source_duplicate"})
            continue
        roots.append(row)
        seen.add(episode)
        if len(roots) == CONTRACT["pool_count"]:
            return {"roots":roots,"metadata_rejections":rejected,"first_seed":seed,
                "scan_limit":CONTRACT["metadata_scan_count"],"source_values_or_outcomes_used":False}
    raise ValueError("fixed metadata scan cannot supply32 disjoint sources; no neural/outcome resampling allowed")


def task_evidence(plan):
    costs = [float(plan[key]["cost"]) for key in ("zero","gn4","gn8","gn16","cem")]
    if any(not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError("finite nonnegative model-only mean costs required")
    zero,g4,g8,g16,cem = costs
    if any(right > left+1e-12 for left,right in zip(costs,costs[1:])):
        raise ValueError("all model search phases must retain their exact incumbent")
    plateau,escape = g8-g16,g16-cem
    eligible = zero > 0 and plateau <= 1e-6 and escape > 1e-6
    return {"gn8_to_gn16_improvement":plateau,"gn16_to_cem_improvement":escape,
        "finite_budget_plateau":plateau <= 1e-6,"sampling_escape":escape > 1e-6,
        "eligible":bool(eligible),"normalized_escape":escape/zero if zero > 0 else 0.,
        "remaining_gn16_fraction":g16/zero if zero > 0 else 0.}


def select(rows):
    if len(rows) != CONTRACT["pool_count"]:
        raise ValueError("complete predeclared32-root model pool required")
    seeds,episodes = [r["seed"] for r in rows],[r["source_episode"] for r in rows]
    if (any(type(s) is not int for s in seeds+episodes) or len(set(seeds)) != 32 or len(set(episodes)) != 32):
        raise ValueError("distinct literal root/source identities required")
    ranked = []
    for row in rows:
        if set(row["tasks"]) != set(TASKS):
            raise ValueError("all three tasks required, no outcome-based task removal")
        evidence = {task:task_evidence(row["tasks"][task]) for task in TASKS}
        active = [evidence[t]["normalized_escape"] if evidence[t]["eligible"] else 0.
            for t in ("shoulder_only","wrist_only")]
        ranked.append({"seed":row["seed"],"source_episode":row["source_episode"],
            "eligible":any(evidence[t]["eligible"] for t in ("shoulder_only","wrist_only")),
            "rank_max":max(active),"rank_min":min(active),"tasks":evidence})
    ranked.sort(key=lambda r:(-r["rank_max"],-r["rank_min"],r["seed"]))
    eligible = [r for r in ranked if r["eligible"]]
    enough = len(eligible) >= CONTRACT["selected_count"]
    return {"status":"model_cohort_admitted" if enough else "insufficient_model_eligible_roots",
        "eligible_count":len(eligible),"selected_seeds":[r["seed"] for r in eligible[:8]] if enough else [],
        "ranked_pool":ranked,"task_retention":list(TASKS),
        "physical_future_labels_read":False,"P1_admission":False}
