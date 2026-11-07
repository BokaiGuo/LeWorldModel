"""Frozen RGB nonlinear chart applied to a previously scored fixed pool."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_nonlinear_chart_20261107_codex as chart
from experiments import pc_tworoom_action_sensitive_metric_20261107_codex as comparison
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments.pc_tworoom_cross_root_shuffle_20261107_codex import action_scaler
from experiments.pc_tworoom_failed_root_full_pool_20261107_codex import candidates_for

PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_NONLINEAR_CHART_SELECTION_PROTOCOL_20261107_CODEX.md"
PLAN = source.ROOT / "results/pc_tworoom_nonlinear_selection_plan_20261107_codex.json"
PHYSICAL = source.ROOT / "results/pc_tworoom_nonlinear_selection_physical_20261107_codex.json"
RESULT = source.ROOT / "results/pc_tworoom_nonlinear_selection_result_20261107_codex.json"


def select_candidate(offsets, costs):
    offsets, costs = np.asarray(offsets, float), np.asarray(costs, float)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or costs.shape != (len(offsets),) or not len(costs):
        raise ValueError("offset and cost shape mismatch")
    if not np.isfinite(offsets).all() or not np.isfinite(costs).all():
        raise ValueError("finite scores required")
    return int(np.lexsort((np.arange(len(costs)), costs, np.square(offsets).sum(axis=1)))[0])


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    records = json.loads(chart.PLAN.read_text())
    prior = json.loads(source.PLAN.read_text())
    base = json.loads(comparison.OUTPUT.read_text())
    if (records["dictionary_sha256"] != file_hash(chart.DICTIONARY)
            or records["prior_plan_sha256"] != file_hash(chart.prior.PLAN)
            or records["forecast_sha256"] != file_hash(comparison.FEATURES)
            or records["protocol_sha256"] != file_hash(chart.PROTOCOL)
            or records["runner_sha256"] != file_hash(chart.__file__)
            or base["source_sha256"] != file_hash(source.PLAN)
            or base["features_sha256"] != file_hash(comparison.FEATURES)
            or base["runner_sha256"] != file_hash(comparison.__file__)):
        raise ValueError("source artifact drift")
    scaler, normalizer = action_scaler()
    if normalizer != prior["normalizer_action_sha256"]:
        raise ValueError("normalizer drift")
    rows = []
    with np.load(chart.DICTIONARY, allow_pickle=False) as dictionaries, \
            np.load(comparison.FEATURES, allow_pickle=False) as features:
        for root, (r, item, base_row) in enumerate(zip(records["rows"], prior["rows"], base["rows"], strict=True)):
            if (r["root"], r["row"]) != (root, item["row"]) or item["row"] != base_row["row"]:
                raise ValueError("root mismatch")
            forecast = features[f"root_{root}_predicted"]
            scores = features[f"root_{root}_scores"][0]
            if forecast.shape[0] != 656 or scores.shape != (656,):
                raise ValueError("candidate pool mismatch")
            if r["valid"]:
                offsets = chart.nearest(dictionaries[f"root_{root}_embedding"], forecast,
                                        dictionaries[f"root_{root}_offsets"])
                index = select_candidate(offsets, scores)
                offset = offsets[index].tolist()
            else:
                index = int(np.argmin(scores))
                offset = None
            chains = [x[1] for x in candidates_for(item)]
            for refinement in item["refinements"]:
                chains.extend(refinement["actions"])
            if len(chains) != 656:
                raise ValueError("chain count mismatch")
            actions = source.to_physical(np.asarray(chains[index], dtype=np.float32), scaler)
            rows.append({"root": root, "row": item["row"], "valid": r["valid"], "candidate_index": index,
                         "selected_offset": offset, "original_cost": float(scores[index]),
                         "original_index": base_row["outcomes"]["original"]["candidate_index"],
                         "action_sensitive_index": base_row["outcomes"]["action_sensitive"]["candidate_index"],
                         "actions": actions.tolist(), "template_count": r.get("count", 0),
                         "template_encoding_seconds": r.get("encoding_seconds", 0)})
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "chart_plan_sha256": file_hash(chart.PLAN), "chart_dictionary_sha256": file_hash(chart.DICTIONARY),
                   "source_sha256": file_hash(source.PLAN), "comparison_sha256": file_hash(comparison.OUTPUT),
                   "features_sha256": file_hash(comparison.FEATURES), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "scope": "exposed fixed pool, not online planner"},
                  stream, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    sealed = json.loads(PLAN.read_text())
    if (sealed["runner_sha256"] != file_hash(Path(__file__)) or sealed["protocol_sha256"] != file_hash(PROTOCOL)
            or sealed["chart_plan_sha256"] != file_hash(chart.PLAN) or sealed["source_sha256"] != file_hash(source.PLAN)):
        raise ValueError("sealed plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(source.common.DATA, "r") as data:
            for row in sealed["rows"]:
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row["row"]])
                actor._set_goal_state(data["proprio"][row["row"] + 25])
                actions = np.asarray(row["actions"], dtype=np.float32)
                if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                    raise ValueError("illegal action")
                trace = []
                for action in actions:
                    _, _, term, trunc, info = actor.step(action.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(term), "truncated": bool(trunc)})
                rows.append({"root": row["root"], "row": row["row"], "trace": trace,
                             "ever_success": any(x["terminated"] for x in trace),
                             "final_success": trace[-1]["terminated"]})
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN)}, stream, indent=2, allow_nan=False)


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    p, q, old = (json.loads(path.read_text()) for path in (PLAN, PHYSICAL, comparison.PHYSICAL))
    if (q["plan_sha256"] != file_hash(PLAN) or old["plan_sha256"] != file_hash(comparison.OUTPUT)
            or p["protocol_sha256"] != file_hash(PROTOCOL) or p["runner_sha256"] != file_hash(Path(__file__))):
        raise ValueError("artifact drift")
    rows = []
    for planned, replay, baselines in zip(p["rows"], q["rows"], old["rows"], strict=True):
        if planned["row"] != replay["row"] or planned["row"] != baselines["row"]:
            raise ValueError("row mismatch")
        for arm in ("original", "action_sensitive"):
            if planned["candidate_index"] == planned[f"{arm}_index"] and replay["trace"] != baselines["outcomes"][arm]["trace"]:
                raise ValueError("identical action replay mismatch")
        rows.append({"root": planned["root"], "row": planned["row"], "valid": planned["valid"],
                     "index": planned["candidate_index"], "ever": replay["ever_success"],
                     "final": replay["final_success"], "distance": replay["trace"][-1]["distance"],
                     "original": baselines["outcomes"]["original"]["ever_success"],
                     "action_sensitive": baselines["outcomes"]["action_sensitive"]["ever_success"]})
    wins = sum(r["ever"] and not r["original"] for r in rows)
    losses = sum(not r["ever"] and r["original"] for r in rows)
    success = sum(r["ever"] for r in rows)
    sensitive = sum(r["action_sensitive"] for r in rows)
    summary = {"roots": len(rows), "valid": sum(r["valid"] for r in rows), "ever": success,
               "final": sum(r["final"] for r in rows), "original": sum(r["original"] for r in rows),
               "action_sensitive": sensitive, "wins": wins, "losses": losses,
               "passes_development_exit": bool(wins >= 2 and losses <= 1 and success >= sensitive)}
    with RESULT.open("x") as stream:
        json.dump({"summary": summary, "rows": rows, "plan_sha256": file_hash(PLAN),
                   "physical_sha256": file_hash(PHYSICAL), "baseline_sha256": file_hash(comparison.PHYSICAL)},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical", "analyze"):
        raise SystemExit("expected plan|physical|analyze")
    {"plan": plan, "physical": physical, "analyze": analyze}[sys.argv[1]]()
