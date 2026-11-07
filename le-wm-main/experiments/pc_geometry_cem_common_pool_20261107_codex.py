"""Score three previously executed RGB plans before opening physical outcomes."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import pc_conditional_selected_alignment_20261107_codex as align
from experiments import pc_conditional_rgb_geometry_baseline_20261107_codex as geo
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = align.ROOT
BASE = ROOT / "results/pc_geometry_cem_common_pool_20261107_codex"
PRED, RESULT = [Path(str(BASE) + suffix) for suffix in ("_prediction.json", "_result.json")]
PROTOCOL = ROOT.parent / "docs/PC_GEOMETRY_CEM_COMMON_POOL_PROTOCOL_20261107_CODEX.md"
NAMES = ("geometry", "endpoint", "temporal")


def select(costs):
    values = np.asarray(costs)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("invalid scores")
    return int(np.argmin(values))


def predict():
    if PRED.exists():
        raise FileExistsError(PRED)
    plan = json.loads(align.src.PLAN.read_text())
    inputs_path = align.src.old.base.source.block.prior.inputs.INPUT
    if ([row["root"] for row in plan["rows"]] != list(range(20, 50))
            or plan["sha256"]["pool"] != file_hash(align.src.POOL)
            or plan["sha256"]["inputs"] != file_hash(inputs_path)):
        raise ValueError("source plan or input drift")
    seal = json.loads(geo.PLAN.read_text())
    if (seal["actions_sha256"] != file_hash(geo.ACTIONS)
            or seal["source_plan_sha256"] != file_hash(align.src.PLAN)
            or seal["inputs_sha256"] != file_hash(inputs_path)):
        raise ValueError("geometry plan drift")
    model, scaler, transform = align.setup(plan)
    common = align.src.old.base.source.block.prior.common
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, geo.ACTIONS, geo.PLAN,
                         align.src.POOL, align.src.PLAN, inputs_path,
                         common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="five packets of five environment steps")
    rows = []
    print("Scoring sealed three-chain pools; physical outcomes are not read.", flush=True)
    with (np.load(inputs_path, allow_pickle=False) as inputs,
          np.load(geo.ACTIONS, allow_pickle=False) as geom,
          np.load(align.src.POOL, allow_pickle=False) as pool, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for task, geo_row in zip(plan["rows"], seal["rows"], strict=True):
            root = task["root"]
            if root != geo_row["root"] or geo_row["abstained"]:
                raise ValueError("missing geometry candidate")
            prior = pool[f"r{root}_selected"]
            for i, name in enumerate(align.src.METHODS):
                if align.src.digest(prior[i]) != task["decisions"][name]["action_sha256"]:
                    raise ValueError("CEM candidate fingerprint mismatch")
            actions = np.concatenate((geom[f"r{root}"][None], prior))
            if actions.shape != (3, 25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                raise ValueError("illegal action pool")
            normal = scaler.transform(actions.reshape(-1, 2)).reshape(1, 3, 5, 10)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            endpoint, result = wrapper.score(context, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
            costs = (result["predicted_emb"][:, :, 1:] -
                     result["goal_emb"][:, -1, None, None]).square().sum(-1)[0].cpu().numpy()
            if costs.shape != (3, 5):
                raise ValueError("model clock mismatch")
            np.testing.assert_allclose(endpoint[0].cpu().numpy(), costs[:, -1], atol=.001, rtol=0)
            original = np.asarray([task["decisions"][name]["cost"] for name in align.src.METHODS])
            error = float(np.max(np.abs(costs[1:, -1] - original)))
            if error > .005:
                raise ValueError(f"old endpoint mismatch for {root}: {error}")
            rows.append({"root": root, "action_sha256": [align.src.digest(a) for a in actions],
                         "costs": costs.tolist(), "endpoint_choice": select(costs[:, -1]),
                         "temporal_choice": select(costs.min(axis=1)), "old_score_error": error})
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with PRED.open("x") as f:
        json.dump({"rows": rows, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "geometry_plan_sha256": file_hash(geo.PLAN), "geometry_actions_sha256": file_hash(geo.ACTIONS),
                   "source_plan_sha256": file_hash(align.src.PLAN), "source_pool_sha256": file_hash(align.src.POOL),
                   "model_sha256": plan["model_sha256"], "normalizer_sha256": plan["normalizer_sha256"]},
                  f, indent=2, allow_nan=False)
    print("Sealed GPU predictions for 30 roots.", flush=True)


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    pred = json.loads(PRED.read_text())
    for name, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                       ("geometry_plan_sha256", geo.PLAN), ("geometry_actions_sha256", geo.ACTIONS),
                       ("source_plan_sha256", align.src.PLAN), ("source_pool_sha256", align.src.POOL)):
        if pred[name] != file_hash(path):
            raise ValueError(f"prediction input drift: {name}")
    _, old_truth, _ = align.source_records()
    geometric = json.loads(geo.RESULT.read_text())
    if geometric["plan_sha256"] != file_hash(geo.PLAN):
        raise ValueError("geometry outcome drift")
    rows = []
    for record, truth, gr in zip(pred["rows"], old_truth["rows"], geometric["rows"], strict=True):
        if record["root"] != truth["root"] or record["root"] != gr["root"]:
            raise ValueError("root mismatch")
        success = [bool(gr["success"])] + [bool(truth["outcomes"][m]["ever"]) for m in align.src.METHODS]
        scores = np.asarray(record["costs"])
        comparisons = {}
        for j, name in enumerate(NAMES[1:], 1):
            comparisons[name] = {"geometry_preferred": bool(scores[0, -1] < scores[j, -1]),
                                 "tie": bool(scores[0, -1] == scores[j, -1]),
                                 "geometry_success": success[0], "other_success": success[j]}
        rows.append({"root": record["root"], "success": success, "comparisons": comparisons,
                     "endpoint_choice": NAMES[record["endpoint_choice"]],
                     "endpoint_selected_success": success[record["endpoint_choice"]],
                     "temporal_choice": NAMES[record["temporal_choice"]],
                     "temporal_selected_success": success[record["temporal_choice"]]})
    summary = {"tasks": len(rows), "endpoint_selected_success": sum(x["endpoint_selected_success"] for x in rows),
               "temporal_selected_success": sum(x["temporal_selected_success"] for x in rows),
               "endpoint_geometry_selected": sum(x["endpoint_choice"] == "geometry" for x in rows),
               "geometry_preferred": {n: sum(x["comparisons"][n]["geometry_preferred"] for x in rows)
                                      for n in NAMES[1:]}}
    with RESULT.open("x") as f:
        json.dump({"summary": summary, "rows": rows, "prediction_sha256": file_hash(PRED),
                   "geometry_physical_sha256": file_hash(geo.RESULT),
                   "cem_physical_sha256": file_hash(align.src.PHYS),
                   "scope": "post-hoc development fixed pool, not deployment or independent evidence"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "analyze"):
        raise SystemExit("expected predict|analyze")
    {"predict": predict, "analyze": analyze}[sys.argv[1]]()
