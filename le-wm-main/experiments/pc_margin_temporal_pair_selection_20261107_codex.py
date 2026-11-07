"""Temporal score of the identical frozen pairs; physical outcomes assessed after sealing."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_effect_margin_gate_20261107_codex as gate
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

source = gate.source
BASE = source.src.prior.ROOT / "results/pc_margin_temporal_pair_selection_20261107_codex"
SCORES, PLAN, RESULT = [Path(str(BASE) + x) for x in ("_scores.npz", "_plan.json", "_result.json")]
PROTOCOL = source.src.prior.ROOT.parent / "docs/PC_MARGIN_TEMPORAL_PAIR_SELECTION_PROTOCOL_20261107_CODEX.md"


def scores_to_choices(scores):
    scores = np.asarray(scores)
    if scores.shape != (8, 9, 5) or not np.isfinite(scores).all():
        raise ValueError("expected finite 8x9x5 scores")
    metrics = {"endpoint": scores[:, :, -1], "all_time": scores.min(axis=-1),
               "post_prefix": scores[:, :, 2:].min(axis=-1)}
    return {name: values[:, 1:] < values[:, :1] for name, values in metrics.items()}


def predict():
    source.src.verify()
    if SCORES.exists() or PLAN.exists():
        raise FileExistsError("temporal output exists")
    meta = json.loads(source.META.read_text())
    for key, path in (("pred_sha256", source.PRED), ("runner_sha256", source.__file__),
                      ("protocol_sha256", source.PROTOCOL), ("source_seal_sha256", source.src.SEAL)):
        if meta[key] != file_hash(path):
            raise ValueError(f"source mismatch: {key}")
    plan, model, scaler, transform = source.setup()
    common = source.src.prior.base.source.block.prior.common
    inp = source.src.prior.base.source.block.prior.inputs.INPUT
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.src.SEAL, source.PRED,
                                     source.src.prior.POOL, inp, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="five packets of five environment steps")
    rows, arrays, errors = [], {}, []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (np.load(source.src.prior.POOL, allow_pickle=False) as pool,
          np.load(inp, allow_pickle=False) as inputs, np.load(source.PRED, allow_pickle=False) as previous,
          torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for root in gate.EVALUATION:
            chains = source.chains_for(pool, root)
            normal = scaler.transform(chains.reshape(-1, 2)).reshape(72, 5, 10)
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            costs = []
            for offset in range(0, 72, 32):
                chunk = torch.as_tensor(normal[None, offset:offset+32], dtype=torch.float32, device="cuda")
                endpoint, output = wrapper.score(ctx, chunk)
                prediction, target = output["predicted_emb"], output["goal_emb"]
                if prediction.shape[2] != 6:
                    raise ValueError("prediction clock is not initial plus five packets")
                allcost = (prediction[:, :, 1:] - target[:, -1, None, None]).square().sum(-1)
                if not torch.allclose(endpoint, allcost[:, :, -1], atol=1e-3, rtol=0):
                    raise ValueError("endpoint disagreement")
                costs.append(allcost[0].cpu().numpy())
            costs = np.concatenate(costs).reshape(8, 9, 5)
            error = float(np.abs(costs[:, :, -1] - previous[f"r{root}_cost"]).max())
            if error > .005:
                raise ValueError(f"stored endpoint mismatch root={root}: {error}")
            selected = scores_to_choices(costs)
            arrays[f"r{root}_scores"] = costs
            rows.append({"root": root, "selection": {k: v.tolist() for k, v in selected.items()},
                         "max_endpoint_error": error})
            errors.append(error)
            print(f"root={root} scored=72", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    with SCORES.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with PLAN.open("x") as handle:
        json.dump({"rows": rows, "scores_sha256": file_hash(SCORES),
                   "source_prediction_sha256": file_hash(source.PRED), "source_meta_sha256": file_hash(source.META),
                   "source_gate_plan_sha256": file_hash(gate.PLAN), "runner_sha256": file_hash(__file__),
                   "protocol_sha256": file_hash(PROTOCOL), "max_endpoint_error": max(errors),
                   "gpu_synchronized_seconds": elapsed, "full_chains": len(rows)*72,
                   "model_packet_steps": len(rows)*72*5},
                  handle, indent=2, allow_nan=False)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, gate_plan = json.loads(PLAN.read_text()), json.loads(gate.PLAN.read_text())
    for key, path in (("scores_sha256", SCORES), ("source_prediction_sha256", source.PRED),
                      ("source_meta_sha256", source.META), ("source_gate_plan_sha256", gate.PLAN),
                      ("runner_sha256", __file__), ("protocol_sha256", PROTOCOL)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding mismatch: {key}")
    meta = json.loads(source.META.read_text())
    if meta["source_physical_sha256"] != file_hash(source.src.PHYS):
        raise ValueError("physical source drift")
    gate_choices = {r["root"]: np.asarray(r["choose_probe"]["gated"], dtype=bool) for r in gate_plan["rows"]}
    rows = []
    with np.load(SCORES, allow_pickle=False) as scores, np.load(source.src.PHYS, allow_pickle=False) as phys:
        for row in plan["rows"]:
            root = row["root"]
            choices = scores_to_choices(scores[f"r{root}_scores"])
            if any(not np.array_equal(choices[k], np.asarray(row["selection"][k])) for k in choices):
                raise ValueError("selection drift")
            choices["old_margin_gate"] = gate_choices[root]
            arms = {}
            for name, selection in choices.items():
                result = gate.outcome(phys[f"r{root}_distance"][:, :, -1], phys[f"r{root}_term"], selection)
                arms[name] = {"success": float(result["ever_success"].mean()),
                              "final_distance": float(result["distance"].mean()),
                              "probe_count": int(selection.sum()),
                              "success_per_pair": result["ever_success"].tolist()}
            rows.append({"root": root, "arms": arms})
    if [r["root"] for r in rows] != list(gate.EVALUATION):
        raise ValueError("evaluation roots drift")
    summary = {}
    for name in ("endpoint", "all_time", "post_prefix", "old_margin_gate"):
        summary[name] = {metric: float(np.mean([r["arms"][name][metric] for r in rows]))
                         for metric in ("success", "final_distance", "probe_count")}
    delta = [r["arms"]["post_prefix"]["success"] - r["arms"]["endpoint"]["success"] for r in rows]
    summary["post_prefix_wins"] = sum(x > 0 for x in delta)
    summary["post_prefix_losses"] = sum(x < 0 for x in delta)
    summary["development_exit"] = bool(
        summary["post_prefix"]["success"] >= summary["endpoint"]["success"] + .01 and
        summary["post_prefix_wins"] >= summary["post_prefix_losses"] + 5 and
        summary["post_prefix"]["final_distance"] <= summary["endpoint"]["final_distance"] + .5)
    with RESULT.open("x") as handle:
        json.dump({"summary": summary, "rows": rows, "plan_sha256": file_hash(PLAN),
                   "physical_sha256": file_hash(source.src.PHYS), "runner_sha256": file_hash(__file__),
                   "scope": "exposed fixed pair post-hoc physical evaluation"}, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "evaluate"):
        raise SystemExit("expected predict|evaluate")
    {"predict": predict, "evaluate": evaluate}[sys.argv[1]]()
