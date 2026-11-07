"""Root-level summaries; no source result mutation or gate promotion."""
import json
from pathlib import Path

import numpy as np

from experiments.pc1_layer_audit.metrics import effects, root_summary
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/pc1_layer_audit_synthesis_20261107"
PUSHT = ROOT / "results/pc1_full_token_orthogonal_pusht_20261107/result.json"
CAMERA = ROOT / "results/pc1_camera_layer_transfer_runtime_bound_20261107/analyze.json"


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def sign_agreement(predicted, truth):
    p, t = effects(predicted), effects(truth)
    valid = np.abs(t) > 1e-6
    ties = np.abs(p) <= 1e-8
    return float(np.mean(np.where(ties[valid], .5, (p[valid]*t[valid] > 0).astype(float)))) if valid.any() else None


def grouped(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return {"n_roots": 0, "mean": None, "missing_roots": len(values)}
    return {**root_summary(valid), "missing_roots": len(values)-len(valid)}


def fold_choices(representations, camera=False):
    names = sorted(representations)
    values = np.array([representations[n]["summary"]["horizons"][0]["calibration_fold_regret"]
                       if camera else representations[n]["summary"]["calibration_fold_regret"] for n in names])
    per_fold = [[names[i] for i in np.flatnonzero(np.isclose(values[:, j], values[:, j].min(), atol=1e-12, rtol=0))]
                for j in range(4)]
    return dict(criterion="true successor goal distance regret; offline oracle diagnostic only",
                per_fold_minimizers_including_ties=per_fold,
                common_minimizers=sorted(set.intersection(*(set(a) for a in per_fold))),
                all_calibration_minimizers=[names[i] for i in np.flatnonzero(np.isclose(values.mean(1), values.mean(1).min(), atol=1e-12, rtol=0))])


def summarize_pusht(data):
    result = dict(roots=32, evaluation_roots=16, native_successful_condition_actions=sum(r["native_final_successes"] for r in data["roots"]),
                  condition_actions=32*2*2*3, unique_physical_chains=32*2*3,
                  native_metric="seven-state distance; not object-only control",
                  fold_selection=fold_choices(data["representations"]), transfer={}, representations={})
    for group, rows in (("all", data["predictor_transfer"]), ("evaluation", data["predictor_transfer"][16:])):
        item = {}
        for name in ("predicted", "real_encoded"):
            for metric in ("regret", "agreement"):
                item[f"{name}_{metric}"] = grouped([mean([c[name][metric] for c in r["conditions"]]) for r in rows])
        item["feature_prediction_mse"] = grouped([mean([c["feature_prediction_mse"] for c in r["conditions"]]) for r in rows])
        result["transfer"][group] = item
    for name, value in data["representations"].items():
        rows = value["rows"][16:]
        paired = [mean([c["regret"] for c in r["offline_initial_effect_retrieval"]]) -
                  mean([c["regret"] for c in r["offline_mean_effect_baseline"]]) for r in rows]
        result["representations"][name] = dict(summary=value["summary"], mapping=value["mapping"],
            retrieval_minus_mean_regret=grouped(paired))
    return result


def summarize_camera(data):
    result = dict(roots=32, evaluation_roots=16, track=data["track"],
                  fold_selection=fold_choices(data["representations"], camera=True),
                  representations={name: {"summary": r["summary"], "mapping": r["mapping"]}
                                   for name, r in data["representations"].items()}, transfer={})
    for group, roots in (("all", data["predictor_transfer"]), ("evaluation", data["predictor_transfer"][16:])):
        result["transfer"][group] = []
        for h, steps in enumerate((5, 10, 25)):
            out = {"environment_steps": steps, "cameras": []}
            for v in (0, 1):
                rows = [r["horizons"][h]["views"][v] for r in roots]
                item = {"camera": v}
                for name in ("predicted_visual", "predicted_official", "real_visual", "real_official", "predicted_vs_encoded"):
                    for metric in ("regret", "agreement"):
                        item[f"{name}_{metric}"] = grouped([r[name][metric] for r in rows])
                for name in ("visual", "proprio"):
                    for target in ("prediction", "persistence"):
                        field = f"{name}_{target}_mse"
                        item[field] = grouped([r[field] for r in rows])
                    item[name+"_prediction_better_than_persistence_roots"] = sum(r[name+"_prediction_mse"] < r[name+"_persistence_mse"] for r in rows)
                item["predicted_action_sign_agreement"] = grouped([sign_agreement(r["predicted_official_costs"], r["physical_costs"]) for r in rows])
                item["physical_all_action_tie_roots"] = int(sum(np.ptp(r["physical_costs"]) <= 1e-6 for r in rows))
                out["cameras"].append(item)
            for name in ("predicted_official", "real_official", "predicted_visual", "real_visual"):
                out[name+"_view_disagreement_roots"] = sum(
                    r["horizons"][h]["views"][0][name]["choice"] != r["horizons"][h]["views"][1][name]["choice"] for r in roots)
            result["transfer"][group].append(out)
    return result


def main():
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT)
    pusht, camera = [json.loads(p.read_text()) for p in (PUSHT, CAMERA)]
    for data in (pusht, camera):
        for path, expected in data["sources"].items():
            if file_hash(path) != expected:
                raise ValueError(f"source drift {path}")
    for stage in ("prepare", "predict", "replay", "analyze"):
        path = CAMERA.parent / f"{stage}.json"
        data = json.loads(path.read_text())
        if file_hash(path.with_suffix(".npz")) != data["arrays_sha256"]:
            raise ValueError(f"stage array drift: {stage}")
        for source, digest in data["sources"].items():
            if file_hash(source) != digest:
                raise ValueError(f"dependency drift: {source}")
    result = dict(scope="development evidence synthesis, descriptive intervals, no gate changes",
                  sources={str(p): file_hash(p) for p in (PUSHT, CAMERA, Path(__file__))},
                  pusht=summarize_pusht(pusht), pointmaze=summarize_camera(camera))
    json.dumps(result, allow_nan=False)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "summary.json", result)
    print(json.dumps({"pusht_transfer": result["pusht"]["transfer"],
                      "camera_evaluation": result["pointmaze"]["transfer"]["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
