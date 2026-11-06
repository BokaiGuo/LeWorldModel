"""Equal-token E0 development comparison of action-effect retention."""
from dataclasses import asdict
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_retention_equal_slot_cost_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_global_action_effect_ranked_dev as compiler
from experiments.spectral_reachability import pc_global_shared_cem_v23 as global_cem

ROOT = prior.ROOT
OUT = ROOT / "results/pc_action_effect_ranked_end_to_end_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_ACTION_EFFECT_RANKED_END_TO_END_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("random_min", "random_effect", "spectral_min", "spectral_effect", "global_cem")
CAP = prior.CAP


def tasks():
    return [{"start": [-0.5, y, theta], "goal": [0.5, gy]}
            for y, theta, gy in itertools.product((-0.2, 0.2), (-0.95, 0.95),
                                                   (-0.1, -0.03, 0.03, 0.1))]


def config(root, arm):
    values = asdict(prior.config(root, "response"))
    values["direction_source"] = "random" if arm.startswith("random_") else "spectral"
    values["retention_policy"] = "response"
    values["successor_value"] = "effect_complement" if arm.endswith("_effect") else "effect_min"
    return compiler.Config(**values)


def plan(predict, task, root, arm):
    cfg = config(root, arm)
    if arm == "global_cem":
        return global_cem.plan(predict, cfg, seed=cfg.seed)
    return compiler.compile_program(predict, np.asarray(task["goal"]), cfg, score_scale=1.0, score_target=0.0)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    dependencies = prior.prior.seal.capture([
        Path(__file__), PROTOCOL, Path(prior.__file__), Path(compiler.__file__),
        Path(compiler.core.__file__), Path(compiler.v13.__file__),
        Path(compiler.core.pc_shared_multistart_cem.__file__), Path(global_cem.__file__),
        Path(prior.prior.journal.__file__), Path(prior.prior.journal.write_json.__code__.co_filename),
        Path(prior.prior.old.previous.fixed.__file__), Path(prior.prior.exact.__file__),
        Path(prior.prior.exact.wall.__file__), Path(prior.prior.exact.wall.physics.__file__),
        Path(prior.prior.old.OLD.previous.independent.__code__.co_filename),
    ])
    task_list = tasks()
    old_tasks = (*prior.tasks(), *prior.prior.tasks(), *prior.prior.old.tasks(),
                 *prior.prior.old.previous.tasks())
    if any(task in old_tasks for task in task_list):
        raise ValueError("new development tasks overlap recorded panels")
    OUT.mkdir()
    prior.prior.journal.write_json(OUT / "manifest.json", {
        "tasks": task_list, "arms": ARMS, "cap": CAP, "dependencies": dependencies,
        "scope": "same-family development tasks, same-token cap, not independent confirmation",
    })
    summary = []
    for root, task in enumerate(task_list):
        for arm in ARMS:
            recorder = prior.prior.journal.PredictionJournal(
                OUT / f"plan_{root:02d}_{arm}_calls",
                prior.prior.old.previous.fixed.predictor(task),
                action_dim=2, packet_length=5, token_cap=CAP,
            )
            started = time.monotonic()
            row = {"root": root, "arm": arm, "task": task}
            try:
                result = plan(recorder, task, root, arm)
                reported = result["prediction_tokens"] if arm == "global_cem" else result["resource"]["prediction_tokens"]
                if recorder.tokens != reported or recorder.tokens > CAP:
                    raise ValueError("journal budget mismatch")
                row.update(status="completed", result=result)
                summary.append({"root": root, "arm": arm, "status": "completed",
                                "score": result["selected_score"], "success": result["selected_score"] < 0.05,
                                "tokens": recorder.tokens})
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
                summary.append({"root": root, "arm": arm, "status": "failed",
                                "success": False, "tokens": recorder.tokens})
            row.update(tokens=recorder.tokens, calls=recorder.calls, elapsed_seconds=time.monotonic() - started)
            prior.prior.journal.write_json(OUT / f"plan_{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']} tokens={row['tokens']}", flush=True)
    prior.prior.seal.verify(dependencies)
    prior.prior.journal.write_json(OUT / "summary.json", {"rows": summary})


if __name__ == "__main__":
    print("Running exposed-E0 same-token action-effect retention development panel.", flush=True)
    main()
