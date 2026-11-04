"""Replay sealed v7 and CEM E0 plans while measuring actual CPU call sites."""
import json
import os
import platform
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import pc_v7_e0_exposed_panel_dev_20261104_codex as panel
from experiments import pc_v7_e0_nonzero_cem_dev_20261104_codex as baseline
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = panel.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_E0_RESOURCE_REPLAY_DEV_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v7_e0_resource_replay_dev_20261104_codex.json"
MANIFEST = OUT.with_name(OUT.stem + "_manifest.json")
ARMS = ("v7", "smooth", "action")


class Timings:
    def __init__(self):
        self.stats = {}

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            started = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                row = self.stats.setdefault(name, {"calls": 0, "inclusive_ns": 0})
                row["calls"] += 1
                row["inclusive_ns"] += time.perf_counter_ns() - started
        return measured


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError("existing result or partial manifest")
    sha = panel.source.sha
    if (json.loads(panel.AUDIT.read_text())["plans_sha256"] != sha(panel.OUT)
            or json.loads(baseline.AUDIT.read_text())["result_sha256"] != sha(baseline.OUT)):
        raise ValueError("source audit mismatch")
    plans, baselines = (json.loads(path.read_text()) for path in (panel.OUT, baseline.OUT))
    if baselines["v7_plans_sha256"] != sha(panel.OUT):
        raise ValueError("baseline not bound to v7 plan")
    if len(plans["rows"]) != 18 or len(baselines["rows"]) != 36:
        raise ValueError("unexpected panel size")
    baseline_rows = {(r["root"], r["arm"]): r for r in baselines["rows"]}
    if len(baseline_rows) != 36:
        raise ValueError("duplicate baseline records")
    deps = seal.capture([Path(__file__), PROTOCOL, panel.OUT, panel.AUDIT, baseline.OUT,
                         baseline.AUDIT, Path(panel.__file__), Path(baseline.__file__),
                         Path(panel.v6.__file__), Path(panel.v7.__file__),
                         Path(panel.v6.v4.__file__), Path(panel.core.__file__),
                         Path(panel.source.exact.__file__), Path(panel.source.exact.wall.__file__),
                         *map(Path, plans["dependencies"]["files"]),
                         *map(Path, baselines["dependencies"]["files"])])
    panel.source.write(MANIFEST, {"dependencies": deps, "source_panel_sha256": sha(panel.OUT),
        "source_baseline_sha256": sha(baseline.OUT), "scope": "instrumented exact E0 replay",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "thread_env": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}})
    print("Replaying sealed v7 and CEM batches; measuring inclusive CPU planning time.", flush=True)
    records = []
    for index, plan in enumerate(plans["rows"]):
        root, task = plan["root"], plan["task"]
        goal = np.asarray(task["goal"])
        arms = ARMS[index % 3:] + ARMS[:index % 3]
        for arm in arms:
            original = plan if arm == "v7" else baseline_rows[(root, arm)]
            if original["task"] != task:
                raise ValueError("task mismatch")
            meter, trace, timings = Meter(1920), [], Timings()

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)
                def evaluate(batch):
                    position = panel.source.exact.endpoints(task, batch)
                    return position, np.linalg.norm(position-goal, axis=1)
                position, scores = meter.predict(raw, evaluate)
                trace.append((requested.copy(), raw.copy(), position.copy(), scores.copy()))
                return position, scores

            v6, core = panel.v6, panel.core
            with patch.object(v6, "response", timings.wrap("response", v6.response)), \
                 patch.object(v6, "cem", timings.wrap("v7_cem", v6.cem)), \
                 patch.object(v6.v4, "retain", timings.wrap("retention", v6.v4.retain)), \
                 patch.object(core, "cem", timings.wrap("baseline_cem", core.cem)), \
                 patch.object(np.linalg, "svd", timings.wrap("svd", np.linalg.svd)):
                started = time.perf_counter_ns()
                if arm == "v7":
                    result = panel.v7.decide("response", "response", root, 0, goal,
                                             predict, meter, core.NOMINALS)
                    selected = np.asarray(result["selected"]["chain"])
                    score = result["selected"]["terminal_score"]
                else:
                    knots = 4 if arm == "smooth" else 15
                    bases = [np.zeros((15, 2))]
                    for restart in (1, 2):
                        rng = np.random.default_rng(2027112900 + root*100 + restart)
                        bases.append(np.clip(core.interpolate(rng.normal(0, .6, size=(1, knots, 2)), 15)[0], -1, 1))
                    _, initial = predict(np.asarray(bases))
                    best_idx = int(np.argmin(initial))
                    selected, score = bases[best_idx].copy(), float(initial[best_idx])
                    for restart, count in enumerate((213, 212, 212)):
                        seed = 2027112800 + root*100 + restart
                        chain, value, _ = core.cem(bases[restart], [], count, np.random.default_rng(seed),
                                                  predict, float(initial[restart]), knots)
                        if value < score:
                            selected, score = chain.copy(), float(value)
                planning_ns = time.perf_counter_ns() - started
            expected_chain = original["result"]["selected"]["chain"] if arm == "v7" else original["selected"]
            expected_score = original["result"]["selected"]["terminal_score"] if arm == "v7" else original["score"]
            np.testing.assert_array_equal(selected, np.asarray(expected_chain))
            np.testing.assert_allclose(score, expected_score, atol=1e-12, rtol=0)
            if len(trace) != len(original["trace"]):
                raise ValueError("prediction call count mismatch")
            for actual, expected in zip(trace, original["trace"], strict=True):
                for actual_array, key in zip(actual, ("requested", "raw", "positions", "scores"), strict=True):
                    np.testing.assert_allclose(actual_array, np.asarray(expected[key]), atol=1e-12, rtol=0)
            recorded = meter.as_dict()
            for key in ("prediction_calls", "trajectory_samples", "prediction_token_samples"):
                if recorded[key] != original["counts"][key]:
                    raise ValueError("resource count mismatch")
            if arm == "v7":
                structure = result["structure"]
                if (timings.stats.get("svd", {}).get("calls", 0) != structure["svd"]
                        or timings.stats.get("response", {}).get("calls", 0)*21 != structure["fd_action_samples"]):
                    raise ValueError("structural accounting mismatch")
            records.append({"root": root, "arm": arm, "position_in_order": arms.index(arm),
                            "score": float(score), "planning_ns": planning_ns,
                            "counts": recorded, "inclusive_call_timing": timings.stats,
                            "trace_batches_verified": len(trace)})
        print(f"root={root}: three matching instrumented plans", flush=True)
    seal.verify(deps)
    summary = {}
    for arm in ARMS:
        items = [r for r in records if r["arm"] == arm]
        summary[arm] = {"n": len(items),
                        "median_planning_seconds": float(np.median([r["planning_ns"] for r in items])/1e9),
                        "median_prediction_seconds": float(np.median([r["counts"]["prediction_seconds"] for r in items])),
                        "median_tokens": float(np.median([r["counts"]["prediction_token_samples"] for r in items]))}
    panel.source.write(OUT, {"manifest_sha256": sha(MANIFEST), "rows": records, "summary": summary,
        "scope": "same exposed tasks and exact backend; CPU planning, instrumented, not GPU total cost"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
