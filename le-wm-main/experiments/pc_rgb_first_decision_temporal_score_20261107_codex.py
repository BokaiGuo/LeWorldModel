"""Freeze temporal scores first, then evaluate on existing physical trajectories."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_first_pool_triple_alignment_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_FIRST_DECISION_TEMPORAL_SCORE_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_first_decision_temporal_plan_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_first_decision_temporal_result_20261107_codex.json"
PHYSICAL = ROOT / "results/pc_tworoom_first_decision_full_pool_20261107_codex.json"


def arm_indices(kept):
    if len(kept) != 2 or len(set(kept)) != 2 or any(i not in range(3) for i in kept):
        raise ValueError("invalid retained prefixes")
    return list(range(12)) + [12 + 2 * i + j for i in kept for j in range(2)]


def score_sequences(predicted, goal):
    if predicted.ndim != 4 or predicted.shape[2] != 6 or goal.ndim != 2:
        raise ValueError("expected B,N,initial+5,D and B,D")
    if predicted.shape[0] != goal.shape[0] or predicted.shape[-1] != goal.shape[-1]:
        raise ValueError("goal dimension mismatch")
    values = (predicted[:, :, 1:, :] - goal[:, None, None, :]).square().sum(-1)
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite temporal costs")
    return values


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    saved = json.loads(prior.physical.run.OUT.read_text())
    meta = json.loads(prior.FRAME_META.read_text())
    if meta["frames_sha256"] != file_hash(prior.FRAMES):
        raise ValueError("initial frame artifact drift")
    common = prior.physical.run.prior.source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != saved["model_sha256"] or normalizer != saved["normalizer_sha256"]:
        raise ValueError("model or scaler drift")
    input_path = prior.physical.run.prior.source.block.prior.inputs.INPUT
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, prior.FRAMES, input_path,
                                      common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 TwoRoom action scaler",
                         clock="five packets of five physical steps")
    rows = []
    print("Freeze temporal model-only choices before reading physical labels.", flush=True)
    with np.load(prior.FRAMES, allow_pickle=False) as images, np.load(input_path, allow_pickle=False) as inputs:
        with torch.inference_mode():
            for index, item in enumerate(saved["rows"]):
                root = item["root"]
                if root != meta["rows"][index]["root"]:
                    raise ValueError("frame root mismatch")
                decisions = prior.physical.common_first(item)
                parents, probes, tails = prior.physical.run.search.timed.build(
                    decisions["effect"]["candidate_seed"], None, 10)
                bank = np.concatenate((np.zeros_like(probes[:1]), probes, tails))
                chains = np.concatenate((parents, np.stack(
                    [np.concatenate((parents[i, :10], bank[j])) for i in range(3) for j in range(5)])))
                normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, 18, 5, 10)
                current = transform(images["current"][index])[None, None, None].cuda()
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                boundary = RGBLeWM(model, audit)
                costs, ctx = boundary.score({"pixels": current, "goal": goal,
                                             "action": torch.empty((1, 1, 0, 10), device="cuda")},
                                            torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                temporal = score_sequences(ctx["predicted_emb"], ctx["goal_emb"][:, -1])[0].cpu().numpy()
                endpoint = costs[0].cpu().numpy()
                if not np.allclose(endpoint, temporal[:, -1], rtol=0, atol=1e-3):
                    raise ValueError("temporal endpoint disagrees with cost")
                # Full bank is interleaved by prefix; project into the per-arm pool order.
                ordered = np.concatenate((temporal[:3], temporal[3:].reshape(3, 5, 5)[:, :3].reshape(9, 5),
                                          temporal[3:].reshape(3, 5, 5)[:, 3:].reshape(6, 5)))
                arms = {}
                for arm, d in decisions.items():
                    ids = arm_indices(d["retained"])
                    values = ordered[ids]
                    sealed = np.concatenate((d["parent_cost"], np.asarray(d["probe_cost"]).reshape(-1),
                                             d["tail_cost"]))
                    if not np.allclose(values[:, -1], sealed, rtol=0, atol=1e-3):
                        raise ValueError("recomputed endpoint costs differ from original")
                    end_idx = int(np.argmin(values[:, -1]))
                    temporal_idx = int(np.argmin(values.min(axis=1)))
                    old_idx = {"parent": 0, "probe": 3, "tail": 12}[d["choice"]["source"]] + d["choice"]["index"]
                    if end_idx != old_idx:
                        raise ValueError("endpoint choice changed after batch recomputation")
                    arms[arm] = {"indices": ids, "endpoint_index": end_idx, "temporal_index": temporal_idx}
                rows.append({"root": root, "temporal_costs": ordered.tolist(), "arms": arms,
                             "receipts": boundary.receipts})
                print(f"root={root} model_chains=18", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    with PLAN.open("x") as stream:
        json.dump({"scope": "exposed development, model-only choices",
                   "runner_sha256": file_hash(Path(__file__)), "protocol_sha256": file_hash(PROTOCOL),
                   "source_sha256": file_hash(prior.physical.run.OUT), "frames_sha256": file_hash(prior.FRAMES),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan_data = json.loads(PLAN.read_text())
    if (plan_data["runner_sha256"] != file_hash(Path(__file__))
            or plan_data["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("plan source drift")
    physical = json.loads(PHYSICAL.read_text())
    pool = json.loads(prior.physical.OUT.read_text())
    original = json.loads(prior.physical.run.OUT.read_text())
    rows = []
    for row, trace, bank, saved in zip(plan_data["rows"], physical["rows"], pool["rows"], original["rows"], strict=True):
        if len({row["root"], trace["root"], bank["root"], saved["root"]}) != 1:
            raise ValueError("root mismatch")
        arms = {}
        for arm, choice in row["arms"].items():
            evidence = trace["arms"][arm]
            if choice["endpoint_index"] != evidence["choice_flat_index"]:
                raise ValueError("actual choice mismatch")
            kept = saved["arms"][arm]["decisions"][0]["retained"]
            steps = ([x["success"] for x in trace["parents_physical"]] +
                     [bank["physical"][i][j]["success"] for i in range(3) for j in range(3)] +
                     [bank["physical"][i][j]["success"] for i in kept for j in (3, 4)])
            arms[arm] = {"endpoint": bool(evidence["ever"][choice["endpoint_index"]]),
                         "temporal": bool(evidence["ever"][choice["temporal_index"]]),
                         "endpoint_terminal": bool(evidence["endpoint"][choice["endpoint_index"]]),
                         "temporal_terminal": bool(evidence["endpoint"][choice["temporal_index"]]),
                         "endpoint_first5": any(steps[choice["endpoint_index"]][:5]),
                         "temporal_first5": any(steps[choice["temporal_index"]][:5])}
        rows.append({"root": row["root"], "initial_success": trace["initial_success"], "arms": arms})
    summary = {}
    for cohort in ("all", "initial_unsuccessful"):
        subset = [r for r in rows if cohort == "all" or not r["initial_success"]]
        summary[cohort] = {"n": len(subset)}
        for arm in prior.physical.run.search.ARMS:
            paired = [(r["arms"][arm]["endpoint"], r["arms"][arm]["temporal"]) for r in subset]
            summary[cohort][arm] = {
                "endpoint": sum(a for a, _ in paired), "temporal": sum(b for _, b in paired),
                "temporal_only": sum(b and not a for a, b in paired),
                "endpoint_only": sum(a and not b for a, b in paired),
                "endpoint_terminal": sum(r["arms"][arm]["endpoint_terminal"] for r in subset),
                "temporal_terminal": sum(r["arms"][arm]["temporal_terminal"] for r in subset),
                "endpoint_first5": sum(r["arms"][arm]["endpoint_first5"] for r in subset),
                "temporal_first5": sum(r["arms"][arm]["temporal_first5"] for r in subset)}
    effect = summary["initial_unsuccessful"]["effect"]
    passed = effect["temporal_only"] >= 3 and effect["endpoint_only"] <= 1 and effect["temporal"] >= effect["endpoint"]
    with RESULT.open("x") as stream:
        json.dump({"scope": "exposed open-loop first-decision counterfactual, not closed loop",
                   "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL),
                   "development_exit_passed": passed, "summary": summary, "rows": rows}, stream, indent=2)
    print(json.dumps({"development_exit_passed": passed, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "evaluate"):
        raise SystemExit("expected plan|evaluate")
    {"plan": plan, "evaluate": evaluate}[sys.argv[1]]()
