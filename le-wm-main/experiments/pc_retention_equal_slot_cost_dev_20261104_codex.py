"""Equal-slot retention-policy comparison with full prediction-token accounting."""
from dataclasses import asdict
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_retention_mechanism_next_dev as prior
from experiments.spectral_reachability import pc_global_discovery_ablation_dev as compiler
from experiments.spectral_reachability import pc_global_shared_cem_v23 as global_cem

ROOT = prior.ROOT
OUT = ROOT / "results/pc_retention_equal_slot_cost_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_RETENTION_EQUAL_SLOT_COST_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("response", "greedy", "diverse", "random", "global_cem")
CAP = 2880


def tasks():
    return [{"start": [-0.5, y, theta], "goal": [0.5, gy]}
            for y, theta, gy in itertools.product((-.225, .225), (-1.03, 1.03),
                                                   (-.12, -.04, .04, .12))]


def config(root, arm):
    values = asdict(prior.config(root, "ridge"))
    values.update(direction_source="spectral", retention_policy="response" if arm == "global_cem" else arm)
    return compiler.Config(**values)


def plan(predict, task, root, arm):
    cfg = config(root, arm)
    if arm == "global_cem":
        return global_cem.plan(predict, cfg, seed=cfg.seed)
    return compiler.compile_program(predict, np.asarray(task["goal"]), cfg,
                                    score_scale=1., score_target=0.)


def dependencies():
    return prior.seal.capture([
        Path(__file__), PROTOCOL, Path(prior.__file__), Path(compiler.__file__),
        Path(compiler.core.__file__), Path(compiler.v13.__file__),
        Path(compiler.core.pc_shared_multistart_cem.__file__), Path(global_cem.__file__),
        Path(prior.journal.__file__), Path(prior.journal.write_json.__code__.co_filename),
        Path(prior.journal_audit.__file__), Path(prior.old.previous.fixed.__file__),
        Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
        Path(prior.exact.wall.physics.__file__),
        Path(prior.old.OLD.previous.independent.__code__.co_filename)])


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    task_list = tasks()
    prior_panels = (*prior.old.tasks(), *prior.old.previous.tasks(), *prior.tasks())
    if any(task in prior_panels for task in task_list):
        raise ValueError("new task family overlaps a prior panel")
    deps = dependencies()
    OUT.mkdir()
    prior.journal.write_json(OUT / "manifest.json", {
        "tasks": task_list, "arms": ARMS, "cap": CAP, "dependencies": deps,
        "scope": "new exposed development task family; not confirmation"})
    print("Comparing equal-slot retention policies under full prediction-token cost.", flush=True)
    for root, task in enumerate(task_list):
        for arm in ARMS:
            directory = OUT / f"plan_{root:02d}_{arm}_calls"
            recorder = prior.journal.PredictionJournal(
                directory, prior.old.previous.fixed.predictor(task),
                action_dim=2, packet_length=5, token_cap=CAP)
            started = time.monotonic()
            row = {"root": root, "task": task, "arm": arm, "cap": CAP}
            try:
                result = plan(recorder, task, root, arm)
                reported = result["prediction_tokens"] if arm == "global_cem" else result["resource"]["prediction_tokens"]
                if reported != recorder.tokens or recorder.tokens > CAP:
                    raise ValueError("planner and journal token mismatch")
                row.update(status="completed", result=result)
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            row.update(tokens=recorder.tokens, calls=recorder.calls,
                       elapsed_seconds=time.monotonic() - started)
            prior.journal.write_json(OUT / f"plan_{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']} tokens={row['tokens']}", flush=True)
    prior.seal.verify(deps)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    prior.seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS):
        raise ValueError("manifest mismatch")
    rows = []
    for root, task in enumerate(tasks()):
        first_bank = None
        for arm in ARMS:
            row = json.loads((OUT / f"plan_{root:02d}_{arm}.json").read_text())
            if (row["root"], row["task"], row["arm"], row["cap"]) != (root, task, arm, CAP):
                raise ValueError("row identity mismatch")
            directory = OUT / f"plan_{root:02d}_{arm}_calls"
            info = prior.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
                raise ValueError("journal mismatch")
            if row["status"] != "completed":
                rows.append({"root": root, "arm": arm, "status": "failed",
                             "tokens": info["tokens"], "error": row["error"]})
                continue
            calls = list(prior.old.previous.fixed.read_calls(directory))
            first_bank = calls[0][0] if first_bank is None else first_bank
            if not np.array_equal(calls[0][0], first_bank):
                raise ValueError("shared initial bank mismatch")
            result = row["result"]
            selected = np.asarray(result["selected"], np.float32)
            best, found, full_samples = float("inf"), False, 0
            for actions, features, scores in calls:
                path, _ = prior.old.OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], features, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(features - task["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                if actions.shape[1] == 15:
                    full_samples += len(actions)
                    best = min(best, float(scores.min()))
                    found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                         & (np.abs(scores - result["selected_score"]) < 1e-12)))
            if not found or result["selected_score"] > best + 1e-12:
                raise ValueError("selected chain is not best evaluated complete chain")
            replay = iter(calls)

            def predict(actions):
                saved, features, scores = next(replay)
                np.testing.assert_array_equal(np.asarray(actions, np.float32), saved)
                return features, scores

            rebuilt = plan(predict, task, root, arm)
            if next(replay, None) is not None:
                raise ValueError("unconsumed prediction")
            for key in ("selected", "selected_score", "incumbent", "comparison",
                        "paired_scores", "abstained"):
                if rebuilt[key] != result[key]:
                    raise ValueError("planner replay mismatch")
            resource = result.get("resource", {})
            rows.append({"root": root, "arm": arm, "status": "completed",
                         "score": result["selected_score"],
                         "success": result["selected_score"] < .05,
                         "abstained": result["abstained"], "tokens": info["tokens"],
                         "calls": len(info["calls"]), "full_horizon_samples": full_samples,
                         "successor_probes": sum(len(x["reachability"]) for x in
                                                 result.get("child", {}).get("layers", [])),
                         "svd_calls": resource.get("svd_calls", 0),
                         "elapsed_seconds": row["elapsed_seconds"]})
        print(f"audited root={root}", flush=True)
    prior.seal.verify(manifest["dependencies"])
    prior.journal.write_json(output, {"status": "passed", "rows": rows,
        "scope": "same-environment deterministic replay; development only"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
