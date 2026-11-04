"""Paired v22 mode-count and shared-bank CEM on new exact-Wall tasks."""
import itertools
import json
from pathlib import Path
import traceback

import numpy as np

from experiments import pc_v17_e0_multistart_baseline_dev_20261104_codex as previous
from experiments import pc_v18_reachability_e0_dev_20261104_codex as prior_panel
from experiments import pc_v19_retention_fix_e0_dev_20261104_codex as v19_panel
from experiments import pc_v19_e0_same_chain_mechanism_dev_20261104_codex as v19_mechanism
from experiments import pc_v20_trust_e0_dev_20261104_codex as v20_panel
from experiments import pc_v21_multichain_e0_dev_20261104_codex as v21_panel
from experiments import pc_v21_e0_same_chain_mechanism_dev_20261104_codex as v21_mechanism
from experiments import pc_v22_e0_same_chain_mechanism_dev_20261104_codex as v22_mechanism
from experiments import pc_v22_reserve_e0_dev_20261104_codex as v22_panel
from experiments import pc_v22_response_retention_fixedbank_e0_dev_20261104_codex as retention_panel
from experiments.spectral_reachability import pc_global_reachability_v18 as v18
from experiments.spectral_reachability import pc_recursive_reachability_v18 as core
from experiments.spectral_reachability import pc_global_reachability_v19 as v19
from experiments.spectral_reachability import pc_recursive_reachability_v19 as core19
from experiments.spectral_reachability import pc_global_reachability_v20 as v20
from experiments.spectral_reachability import pc_recursive_reachability_v20 as core20
from experiments.spectral_reachability import pc_global_reachability_v21 as v21
from experiments.spectral_reachability import pc_recursive_reachability_v21 as core21
from experiments.spectral_reachability import pc_global_reachability_v22 as v22
from experiments.spectral_reachability import pc_recursive_reachability_v22 as core22
from experiments.spectral_reachability import pc_timed_program_execution_v6 as timed
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = previous.ROOT
OUT = ROOT / "results/pc_v22_modecount_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V22_MODECOUNT_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v22_mode1", "v22_mode2", "multistart_cem")
CAP = 1920


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.21, .21), (-1.07, 1.07), (-.18, .18))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior, old = previous.prior, previous.prior.prior
    panels = (v22_panel.tasks(), retention_panel.tasks(), v22_mechanism.tasks(), v21_mechanism.tasks(), v21_panel.tasks(), v20_panel.tasks(),
              v19_mechanism.tasks(), v19_panel.tasks(), prior_panel.tasks(), previous.tasks(), prior.tasks(),
              prior.v15.tasks(), previous.v16.tasks(),
              prior.fixed.tasks(), *prior.v15.source_panels())
    if any(task in panel for task in tasks() for panel in panels):
        raise ValueError("task overlap in checked panels")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(v22_panel.__file__), Path(retention_panel.__file__),
        Path(timed.__file__), Path(v18.__file__),
        Path(core.__file__), Path(v19.__file__), Path(core19.__file__), Path(v20.__file__), Path(core20.__file__),
        Path(v21.__file__), Path(core21.__file__), Path(v22.__file__), Path(core22.__file__),
        Path(prior.compiler.__file__), Path(old.compiler.__file__),
        Path(old.compiler_child.__file__), Path(previous.baseline.__file__), Path(prior.v15.prior.cem.__file__),
        Path(old.exact.__file__), Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
        Path(old.previous.independent.__code__.co_filename)])
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in old.previous.prior.core.NOMINALS)
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": CAP,
                                       "dependencies": deps, "scope": "development; provenance unknown"})
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])

        def predict(actions):
            positions = old.exact.endpoints(task, actions)
            return positions, np.linalg.norm(positions-goal, axis=1)

        for arm in ARMS:
            config = v22.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
                                nominals=nominals, token_cap=CAP, seed=2029501000 + root,
                                max_modes=2 if arm == "v22_mode2" else 1)
            try:
                result = timed.execute(predict, goal, config, arm="v22" if arm.startswith("v22_") else arm,
                                       baseline_seed=2029502000+root*10)
                trace = result.pop("traces")
                error = None
            except Exception:
                result, trace, error = None, [], traceback.format_exc()
            path = OUT / f"root{root:02d}_{arm}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **{f"{name}_{i}": batch[name]
                    for i, batch in enumerate(trace) for name in ("actions", "features", "scores")})
            rows.append({"root": root, "task": task, "arm": arm, "status": "failed" if error else "completed",
                         "error": error, "result": result, "trace": path.name, "trace_sha256": seal.digest(path)})
            print(f"root={root} arm={arm} status={rows[-1]['status']}", flush=True)
    seal.verify(deps)
    old.writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    plans = json.loads((OUT / "plans.json").read_text())
    seal.verify(manifest["dependencies"])
    if (plans["manifest_sha256"] != seal.digest(OUT / "manifest.json")
            or manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS) or len(plans["rows"]) != 24):
        raise ValueError("source binding mismatch")
    old, summary = previous.prior.prior, []
    for root, task in enumerate(tasks()):
        banks = []
        for arm in ARMS:
            matches = [row for row in plans["rows"] if row["root"] == root and row["arm"] == arm]
            if len(matches) != 1:
                raise ValueError("missing arm")
            row = matches[0]
            if row["status"] != "completed":
                summary.append({"root": root, "arm": arm, "status": "failed", "error": row["error"]})
                continue
            if row["task"] != task or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
                raise ValueError("trace mismatch")
            record, tokens, best, found = row["result"], 0, float("inf"), False
            selected = np.asarray(record["result"]["selected"], np.float32)
            with np.load(OUT / row["trace"], allow_pickle=False) as saved:
                banks.append(saved["actions_0"].copy())
                for i, call in enumerate(record["calls"]):
                    actions, positions, scores = (saved[f"{name}_{i}"] for name in ("actions", "features", "scores"))
                    if (actions.dtype != np.float32 or actions.shape[1] % 5 or not np.isfinite(actions).all()
                            or np.any(np.abs(actions) > 1)):
                        raise ValueError("invalid actions")
                    path, _ = old.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions-task["goal"], axis=1),
                                               atol=1e-12, rtol=0)
                    cost = len(actions) * actions.shape[1] // 5
                    if cost != call["prediction_tokens"]:
                        raise ValueError("per-call token mismatch")
                    tokens += cost
                    if actions.shape[1] == 15:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                             & (np.abs(scores-record["result"]["selected_score"]) < 1e-12)))
            if (tokens != record["resource"]["prediction_tokens"] or tokens > CAP or not found
                    or record["result"]["selected_score"] > best + 1e-12):
                raise ValueError("selection or budget mismatch")
            probes = (sum(len(layer["reachability"]) for layer in record["result"]["child"]["layers"])
                      if arm.startswith("v22_") else 0)
            candidates = ([len(layer["candidates"]) for layer in record["result"]["child"]["layers"]]
                          if arm.startswith("v22_") else None)
            summary.append({"root": root, "arm": arm, "status": "completed",
                            "score": record["result"]["selected_score"],
                            "success": record["result"]["selected_score"] < .05,
                            "tokens": tokens, "successor_probes": probes, "layer_candidates": candidates})
        for bank in banks[1:]:
            np.testing.assert_array_equal(bank, banks[0])
    seal.verify(manifest["dependencies"])
    old.writer(output, {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"), "rows": summary,
                        "scope": "alternate same-environment geometry; not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
