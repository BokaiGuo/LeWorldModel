"""RGB-synthetic manifold proxy, sealed before evaluator-only physical assessment."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_effect_bank_three_way_alignment_20261107_codex as source
from experiments import pc_tworoom_nonlinear_chart_20261107_codex as chart
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOTS = tuple(range(10, 20))
PROTOCOL = source.src.prior.ROOT.parent / "docs/PC_EFFECT_MANIFOLD_CONFIDENCE_PROTOCOL_20261107_CODEX.md"
BASE = source.src.prior.ROOT / "results/pc_effect_manifold_confidence_20261107_codex"
PLAN, ARRAYS, RESULT = [Path(str(BASE) + x) for x in ("_plan.json", "_arrays.npz", "_result.json")]


def indices(residual, cost, root):
    residual, cost = np.asarray(residual), np.asarray(cost)
    if residual.shape != (8, 9) or cost.shape != (8, 9):
        raise ValueError("invalid shape")
    if not np.isfinite(residual).all() or not np.isfinite(cost).all() or (residual < 0).any():
        raise ValueError("invalid scores")
    confidence = np.maximum(residual[:, :1], residual[:, 1:]).reshape(-1)
    margin = np.abs(cost[:, :1] - cost[:, 1:]).reshape(-1)
    return {"manifold": np.argsort(confidence, kind="stable")[:32].tolist(),
            "margin": np.argsort(-margin, kind="stable")[:32].tolist(),
            "random": np.random.default_rng(31800 + int(root)).permutation(64)[:32].tolist(),
            "all": list(range(64))}


def predict():
    source.src.verify()
    if PLAN.exists() or ARRAYS.exists():
        raise FileExistsError("plan or arrays already exist")
    meta = json.loads(source.META.read_text())
    for key, path in (("pred_sha256", source.PRED), ("runner_sha256", source.__file__),
                      ("protocol_sha256", source.PROTOCOL), ("source_seal_sha256", source.src.SEAL)):
        if meta[key] != file_hash(path):
            raise ValueError(f"source mismatch: {key}")
    plan, model, _, transform = source.setup()
    inputs_path = source.src.prior.base.source.block.prior.inputs.INPUT
    rows, arrays = [], {}
    with (np.load(inputs_path, allow_pickle=False) as inputs,
          np.load(source.PRED, allow_pickle=False) as predictions, torch.inference_mode()):
        for root in ROOTS:
            started = time.perf_counter()
            images, offsets, info = chart.templates(inputs["current"][root], inputs["goal"][root])
            if not info["valid"]:
                rows.append({"root": root, "valid": False, "reason": info["reason"]})
                print(f"root={root} abstained", flush=True)
                continue
            torch.cuda.synchronize()
            encoded_start = time.perf_counter()
            batches = []
            for offset in range(0, len(images), 32):
                batch = torch.stack([transform(im) for im in images[offset:offset+32]])[:, None].cuda()
                batches.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
            templates = np.concatenate(batches)
            torch.cuda.synchronize()
            encoding_seconds = time.perf_counter() - encoded_start
            query = predictions[f"r{root}_predicted"]
            if query.ndim != 2 or query.shape[0] != 72 or templates.shape[1:] != query.shape[1:]:
                raise ValueError("invalid representation shapes")
            residual = np.square(query[:, None, :] - templates[None, :, :]).mean(axis=2).min(axis=1).reshape(8, 9)
            selection = indices(residual, predictions[f"r{root}_cost"], root)
            arrays[f"r{root}_residual"] = residual
            arrays[f"r{root}_templates"] = templates
            arrays[f"r{root}_offsets"] = offsets
            rows.append({"root": root, "valid": True, "template_count": len(images),
                         "encoding_seconds": encoding_seconds, "total_seconds": time.perf_counter() - started,
                         "selected": selection})
            print(f"root={root} templates={len(images)}", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with ARRAYS.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with PLAN.open("x") as handle:
        json.dump({"rows": rows, "arrays_sha256": file_hash(ARRAYS), "prediction_sha256": file_hash(source.PRED),
                   "prediction_meta_sha256": file_hash(source.META), "input_sha256": file_hash(inputs_path),
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "chart_sha256": file_hash(chart.__file__),
                   "reconstruction_sha256": file_hash(chart.recon.__file__),
                   "scope": "RGB only; no true successor or physical labels read during prediction"},
                  handle, indent=2, allow_nan=False)


def paired_stats(cost, distance, selected):
    cost, distance = np.asarray(cost), np.asarray(distance)
    if cost.shape != (8, 9) or distance.shape != (8, 9):
        raise ValueError("invalid shape")
    if not np.isfinite(cost).all() or not np.isfinite(distance).all():
        raise ValueError("nonfinite values")
    sel = np.asarray(selected)
    if sel.ndim != 1 or not np.issubdtype(sel.dtype, np.integer) or np.any(sel < 0) or np.any(sel >= 64):
        raise ValueError("invalid selection")
    if len(np.unique(sel)) != len(sel):
        raise ValueError("duplicate selection")
    p = (cost[:, :1] - cost[:, 1:]).reshape(-1)[sel]
    d = (distance[:, :1] - distance[:, 1:]).reshape(-1)[sel]
    mask = (np.abs(p) > 1e-6) & (np.abs(d) > 1e-6)
    return {"valid": int(mask.sum()), "matches": int(np.sum(np.sign(p[mask]) == np.sign(d[mask]))),
            "physical_magnitude_mean": float(np.mean(np.abs(d[mask]))) if mask.any() else None}


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(PLAN.read_text())
    for key, path in (("arrays_sha256", ARRAYS), ("prediction_sha256", source.PRED),
                      ("prediction_meta_sha256", source.META), ("runner_sha256", __file__),
                      ("protocol_sha256", PROTOCOL), ("chart_sha256", chart.__file__),
                      ("reconstruction_sha256", chart.recon.__file__),
                      ("input_sha256", source.src.prior.base.source.block.prior.inputs.INPUT)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding mismatch: {key}")
    meta = json.loads(source.META.read_text())
    if meta["source_physical_sha256"] != file_hash(source.src.PHYS):
        raise ValueError("physical provenance mismatch")
    rows = []
    with (np.load(source.PRED, allow_pickle=False) as pred,
          np.load(ARRAYS, allow_pickle=False) as arrays,
          np.load(source.src.PHYS, allow_pickle=False) as physical):
        for row in plan["rows"]:
            if not row["valid"]:
                rows.append(row)
                continue
            root = row["root"]
            cost = pred[f"r{root}_cost"]
            selection = indices(arrays[f"r{root}_residual"], cost, root)
            if selection != row["selected"]:
                raise ValueError("selection does not match sealed scores")
            distance = physical[f"r{root}_distance"][:, :, -1]
            metrics = {key: paired_stats(cost, distance, sel) for key, sel in selection.items()}
            eligible = all(value["valid"] > 0 for value in metrics.values())
            rows.append({"root": root, "valid": True, "eligible": eligible, "metrics": metrics})
    valid = [r for r in rows if r.get("eligible")]
    summary = {"roots": len(valid), "abstained": sum(not r["valid"] for r in rows),
               "empty_comparison": sum(r["valid"] and not r["eligible"] for r in rows)}
    for name in ("manifold", "margin", "random", "all"):
        values = [r["metrics"][name] for r in valid]
        summary[name] = {"macro_accuracy": float(np.mean([v["matches"] / v["valid"] for v in values])) if values else None,
                         "pairs": sum(v["valid"] for v in values),
                         "macro_physical_magnitude": float(np.mean([v["physical_magnitude_mean"] for v in values])) if values else None}
    wins = sum(r["metrics"]["manifold"]["matches"] / r["metrics"]["manifold"]["valid"] >
               r["metrics"]["margin"]["matches"] / r["metrics"]["margin"]["valid"] for r in valid)
    summary["wins_vs_margin"] = wins
    summary["development_exit"] = bool(len(valid) >= 6 and wins >= 6 and
        all(summary["manifold"]["macro_accuracy"] >= summary[arm]["macro_accuracy"] + .05
            for arm in ("all", "random", "margin")))
    with RESULT.open("x") as handle:
        json.dump({"summary": summary, "rows": rows, "plan_sha256": file_hash(PLAN),
                   "physical_sha256": file_hash(source.src.PHYS), "runner_sha256": file_hash(__file__),
                   "scope": "exposed development; evaluator-only physical labels"},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "analyze"):
        raise SystemExit("expected predict|analyze")
    {"predict": predict, "analyze": analyze}[sys.argv[1]]()
