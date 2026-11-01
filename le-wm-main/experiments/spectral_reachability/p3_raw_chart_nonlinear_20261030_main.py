"""Replay a fixed P3 failure roster on the frozen CUDA model; no gate changes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import p3_raw_chart_ridge_20261030_main as ridge

NAME = "p3_raw_chart_nonlinear_20261030_main"
JOURNAL = base.ROOT / "results/p3_dual_branch_host_driver_20261030_codex_v1_RUN_MAIN01/decision/journal.jsonl"
JOURNAL_SHA = "ff67c50f2f76b2d0bc3efcc774e60e42c8a55c564f6beaa4e9d706aa74e61da2"
ROSTER = (49891, 118022, 172473, 245163)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_roster():
    if base.sha(JOURNAL) != JOURNAL_SHA:
        raise ValueError("journal digest mismatch")
    selected, slot, task = [], None, None
    with JOURNAL.open() as stream:
        for line in stream:
            entry = json.loads(line)
            event = entry["payload"]
            nested = event.get("payload", {})
            if event.get("kind") == "dual_slot_attempt":
                slot, task = event, None
            if nested.get("kind") == "native_call_attempt" and task is None:
                task = nested["task"]
            if entry["index"] not in ROSTER:
                continue
            result = nested["result"]
            if (nested.get("kind") != "dual_decision_final" or task is None or slot is None
                    or digest(task) != result["root_sha256"] or slot["policy"] != result["policy"]
                    or result["branch"] != "reconfiguration" or result["status"] != "failed"
                    or len(result["panels"]) != 2
                    or task["binding"]["model_sha256"] != base.MODEL_HASH["original_state_dict"]):
                raise ValueError("decision or task binding")
            for horizon, panel in zip((1, 3), result["panels"], strict=True):
                if panel["bank"]["horizon"] != horizon or len(panel["rows"]) != 8:
                    raise ValueError("panel roster")
                for expected, row in zip(["zero", *range(8)],
                                         [panel["control"], *panel["rows"]], strict=True):
                    if (row["index"] != expected or row["status"] != "complete"
                            or digest(row["postcontext"]) != row["signal"]["parent_context_sha256"]
                            or row["signal"]["horizon"] != 1):
                        raise ValueError("row identity")
                    selected.append((entry["index"], slot, task, horizon, row))
    if len(selected) != 72 or tuple(dict.fromkeys(x[0] for x in selected)) != ROSTER:
        raise ValueError("fixed roster missing or duplicated")
    return selected


def projected_cost(task, latent):
    r = np.asarray(task["projection"], np.float64) @ (
        np.asarray(task["goal"], np.float64) - np.asarray(latent, np.float64))
    result = float(np.mean(r * r))
    if not np.isfinite(result):
        raise ValueError("nonfinite model cost")
    return result


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    roster = load_roster()
    paths = (Path(__file__).resolve(), Path(ridge.__file__).resolve(), JOURNAL)
    sources = base.bindings()
    sources.update({str(p): base.sha(p) for p in paths})
    records = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint mismatch")
        started = time.monotonic()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for index, slot, task, horizon, row in roster:
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                states = tensor(row["postcontext"]["latent_history"])[None]
                history = tensor(row["postcontext"]["action_history"])[None]
                mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                if (states.shape != (1, 3, 192) or history.shape != (1, 2, 10)
                        or mean.shape != (2,) or std.shape != (2,) or not bool((std > 0).all())):
                    raise ValueError("context or normalizer shape")

                def replay(raw):
                    action = tensor(raw).reshape(1, 5, 2)
                    normalized = ((action - mean) / std).reshape(1, 1, 10)
                    prediction = model.predict(states, model.action_encoder(torch.cat((history, normalized), dim=1)))
                    if prediction.shape != states.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError("invalid prediction")
                    return prediction[0, -1].cpu().numpy().astype(np.float64)

                old = replay(row["suffix_actions"])
                saved = np.asarray(row["suffix_path"][-1]["latent_history"][-1], np.float64)
                max_error = float(np.max(np.abs(old - saved)))
                old_score = projected_cost(task, old)
                cost_error = abs(old_score - row["bounded_suffix_cost"])
                if max_error > 1e-5 or cost_error > 1e-6:
                    raise ValueError(f"old replay mismatch {index}/{horizon}/{row['index']}: {max_error}, {cost_error}")
                solved = ridge.solve_raw_ridge(row["signal"]["jacobian"],
                    np.asarray(task["goal"], np.float64) - np.asarray(row["signal"]["nominal"], np.float64))
                new = replay(solved["raw"])
                records.append({"decision_index": index, "slot": slot["slot"], "root": slot["root"],
                                "policy": slot["policy"], "horizon": horizon, "candidate": row["index"],
                                "old_action": row["suffix_actions"], "new_action": solved["raw"].tolist(),
                                "old_terminal": old.tolist(), "new_terminal": new.tolist(),
                                "old_cost": old_score, "new_cost": projected_cost(task, new),
                                "old_terminal_max_error": max_error, "old_cost_abs_error": cost_error,
                                "linear_objective": solved["linear_objective"],
                                "solver_iterations": solved["iterations"]})
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        after = base.model_hash(model)
        if before != after or any(base.sha(Path(k)) != v for k, v in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete_cuda_model_only", "rows": records,
                  "sources": sources, "host": host, "model_before": before, "model_after": after,
                  "calls": {"predict": 144, "action_encoder": 144, "backward": 0},
                  "elapsed_seconds": elapsed, "physical_execution": False,
                  "independent_confirmation": False, "equal_budget_certified": False,
                  "P1_admission": False, "P3_admission": False}
        sha = base.write(output, result)
    print(json.dumps({"status": result["status"], "rows": len(records), "sha256": sha}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replay all fixed old and raw-chart suffixes on the frozen host CUDA model.", flush=True)
    run(args.output)
