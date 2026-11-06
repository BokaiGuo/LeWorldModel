"""Paired global, spectral, random-direction and retention development ablation."""
from dataclasses import asdict
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_retention_mechanism_next_dev as prior
from experiments.spectral_reachability import pc_global_discovery_ablation_dev as compiler
from experiments.spectral_reachability import pc_global_shared_cem_v23 as baseline

ROOT, journal, seal = prior.ROOT, prior.journal, prior.seal
OUT = ROOT / "results/pc_discovery_attribution_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_DISCOVERY_ATTRIBUTION_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("spectral", "random_direction", "diverse", "global_cem")
SAMPLES = 628


def config(root, arm):
    values = asdict(prior.config(root, "ridge"))
    values["direction_source"] = "random" if arm == "random_direction" else "spectral"
    values["retention_policy"] = "diverse" if arm == "diverse" else "response"
    return compiler.Config(**values)


def plan(predict, task, root, arm):
    cfg = config(root, arm)
    if arm == "global_cem":
        return baseline.plan(predict, cfg, seed=cfg.seed)
    return compiler.compile_program(predict, np.asarray(task["goal"]), cfg, score_scale=1., score_target=0.)


def dependencies():
    return seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(prior.PROTOCOL),
        Path(compiler.__file__), Path(compiler.core.__file__), Path(compiler.v13.__file__),
        Path(compiler.core.pc_shared_multistart_cem.__file__), Path(baseline.__file__),
        Path(journal.__file__), Path(journal.write_json.__code__.co_filename),
        Path(prior.journal_audit.__file__), Path(prior.old.previous.fixed.__file__),
        Path(prior.exact.__file__), Path(prior.exact.wall.__file__), Path(prior.exact.wall.physics.__file__),
        Path(prior.old.OLD.previous.independent.__code__.co_filename)])


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    deps = dependencies()
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": prior.tasks(), "arms": ARMS, "cap": 2880,
        "fixed_samples": SAMPLES, "dependencies": deps, "scope": "exposed-task development"})
    print("Compare full-cost discovery ablations, preserving negative results.", flush=True)
    for root, task in enumerate(prior.tasks()):
        for arm in ARMS:
            recorder = journal.PredictionJournal(OUT / f"plan_{root:02d}_{arm}_calls",
                prior.old.previous.fixed.predictor(task), action_dim=2, packet_length=5, token_cap=2880)
            started = time.monotonic()
            row = {"root": root, "task": task, "arm": arm}
            try:
                result = plan(recorder, task, root, arm)
                reported = result["prediction_tokens"] if arm == "global_cem" else result["resource"]["prediction_tokens"]
                if recorder.tokens != reported:
                    raise ValueError("budget mismatch")
                row.update(status="completed", result=result)
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            row.update(tokens=recorder.tokens, calls=recorder.calls, elapsed=time.monotonic() - started)
            journal.write_json(OUT / f"plan_{root:02d}_{arm}.json", row)
            print(f"plan root={root} arm={arm} status={row['status']}", flush=True)
    seal.verify(deps)


def audit_plans():
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    rows = []
    for root, task in enumerate(prior.tasks()):
        first = []
        for arm in ARMS:
            row = json.loads((OUT / f"plan_{root:02d}_{arm}.json").read_text())
            if (row["root"], row["task"], row["arm"]) != (root, task, arm):
                raise ValueError("identity mismatch")
            directory = OUT / f"plan_{root:02d}_{arm}_calls"
            info = prior.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
                raise ValueError("journal mismatch")
            if row["status"] != "completed":
                rows.append({"root": root, "arm": arm, "status": "failed"})
                continue
            calls = list(prior.old.previous.fixed.read_calls(directory))
            first.append(calls[0][0])
            selected = np.asarray(row["result"]["selected"], np.float32)
            best, found = float("inf"), False
            for actions, features, scores in calls:
                path, _ = prior.old.OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], features, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(features-task["goal"], axis=1), atol=1e-12, rtol=0)
                if actions.shape[1] == 15:
                    best = min(best, float(scores.min()))
                    found |= bool(np.any(np.all(actions == selected, axis=(1, 2)) &
                                         (np.abs(scores-row["result"]["selected_score"]) < 1e-12)))
            if not found or row["result"]["selected_score"] > best + 1e-12:
                raise ValueError("selected not scored incumbent")
            replay = iter(calls)
            def predict(actions):
                saved, features, scores = next(replay)
                np.testing.assert_array_equal(actions, saved)
                return features, scores
            rebuilt = plan(predict, task, root, arm)
            if next(replay, None) is not None:
                raise ValueError("unconsumed calls")
            for key in ("selected", "selected_score", "incumbent", "comparison", "paired_scores", "abstained"):
                if rebuilt[key] != row["result"][key]:
                    raise ValueError("planner replay mismatch")
            rows.append({"root": root, "arm": arm, "status": "completed",
                         "score": row["result"]["selected_score"], "tokens": row["tokens"],
                         "row_sha256": seal.digest(OUT / f"plan_{root:02d}_{arm}.json")})
        for bank in first[1:]:
            np.testing.assert_array_equal(first[0], bank)
        print(f"audited plan root={root}", flush=True)
    seal.verify(manifest["dependencies"])
    journal.write_json(OUT / "plans_audit.json", {"status": "passed", "rows": rows,
        "scope": "same environment geometry and deterministic replay"})


def fixed_starts(root, arm, row):
    cfg = config(root, arm)
    bank = compiler.v13.proposal_bank(cfg)
    scores = row["result"]["bank_scores"] if arm == "global_cem" else row["result"]["global_proposals"]["scores"]
    indices = []
    for i in np.argsort(scores, kind="stable"):
        if not any(np.array_equal(bank[i], bank[j]) for j in indices):
            indices.append(int(i))
        if len(indices) == 4:
            break
    if len(indices) < 4:
        raise ValueError("not enough distinct global starts")
    starts = [bank[indices[0]], bank[indices[1]]]
    labels = [f"global_{indices[0]}", f"global_{indices[1]}"]
    if arm == "global_cem":
        starts += [bank[indices[2]], bank[indices[3]]]
        labels += [f"global_{indices[2]}", f"global_{indices[3]}"]
    else:
        layer = row["result"]["child"]["layers"][-1]
        if len(layer["retained"]) != 2:
            raise ValueError("final beam not two candidates")
        for index in layer["retained"]:
            starts.append(np.asarray(layer["candidates"][index]["chain"], np.float32))
            labels.append(f"retained_{index}")
    return np.stack(starts), labels


def fixed_optimize(predict, starts, root):
    return compiler.core.pc_shared_multistart_cem.optimize(
        lambda actions: predict(actions)[1], np.empty((0, 2), np.float32), starts,
        SAMPLES, seed=config(root, "spectral").seed, batch_size=12, elites=4)


def fixed():
    manifest = json.loads((OUT / "manifest.json").read_text())
    audit = json.loads((OUT / "plans_audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or len(audit["rows"]) != 16 * len(ARMS):
        raise ValueError("plans not audited")
    for root, task in enumerate(prior.tasks()):
        common = None
        for arm in ARMS:
            row = json.loads((OUT / f"plan_{root:02d}_{arm}.json").read_text())
            check = next(x for x in audit["rows"] if x["root"] == root and x["arm"] == arm)
            if row["status"] != "completed" or check["status"] != "completed":
                journal.write_json(OUT / f"fixed_{root:02d}_{arm}.json",
                                   {"root": root, "arm": arm, "status": "plan_failed"})
                continue
            if check["row_sha256"] != seal.digest(OUT / f"plan_{root:02d}_{arm}.json"):
                raise ValueError("plan changed")
            recorder = journal.PredictionJournal(OUT / f"fixed_{root:02d}_{arm}_calls",
                prior.old.previous.fixed.predictor(task), action_dim=2, packet_length=5, token_cap=SAMPLES * 3)
            result_row = {"root": root, "task": task, "arm": arm}
            try:
                starts, labels = fixed_starts(root, arm, row)
                if common is None:
                    common = starts[:2].copy()
                else:
                    np.testing.assert_array_equal(common, starts[:2])
                result = fixed_optimize(recorder, starts, root)
                if recorder.tokens != SAMPLES * 3:
                    raise ValueError("fixed opportunity budget mismatch")
                result_row.update(status="completed", initial_actions=starts.tolist(), labels=labels,
                    score=float(result["score"]), actions=result["actions"].tolist(), starts=result["starts"])
            except Exception:
                result_row.update(status="failed", error=traceback.format_exc())
            result_row.update(tokens=recorder.tokens, calls=recorder.calls)
            journal.write_json(OUT / f"fixed_{root:02d}_{arm}.json", result_row)
            print(f"fixed root={root} arm={arm} status={result_row['status']}", flush=True)
    seal.verify(manifest["dependencies"])


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit_plans", "fixed"):
        raise SystemExit("choose run, audit_plans, or fixed")
    {"run": run, "audit_plans": audit_plans, "fixed": fixed}[sys.argv[1]]()
