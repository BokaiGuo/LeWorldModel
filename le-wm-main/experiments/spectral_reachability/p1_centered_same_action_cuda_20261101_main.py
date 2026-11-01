"""All executed packets: real-history prediction versus actual RGB and task."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_centered_target_cuda_20261101_main as target
from . import p1_centered_metric_cuda_20261101_main as metric_source
from . import p1_centered_panel_20261101_main as panel

TAGS = ("MAIN01", "COHERENTMAIN01")


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    base = target.encoder.base
    out = Path(output)
    if not out.is_absolute() or out.parent != panel.ROOT / "results" or out.exists():
        raise ValueError("exclusive output")
    entry, _, sources = target.collect()
    if base.sha(metric_source.TARGET) != metric_source.TARGET_SHA:
        raise ValueError("fixed target")
    encoded = json.loads(metric_source.TARGET.read_bytes())["rows"][0]
    metric_path = panel.ROOT / "results/p1_centered_metric_cuda_20261101_main_MAIN01.json"
    if base.sha(metric_path) != "ef9ae31f143a18d73432fb45bf861db13f9f852baa776fe977962ec9cb2a4818":
        raise ValueError("fixed metric")
    matrix = np.asarray(json.loads(metric_path.read_bytes())["sqrt_matrix"], float)
    goal = np.asarray(encoded["latent"], np.float32)
    history_raw = json.loads((target.DIRECTORY / "report.json").read_bytes())["raw_history"]
    task = json.loads(target.witness.old.PREP.read_bytes())["source"]["tasks"][0]
    branches = []
    for tag in TAGS:
        summary_path = panel.ROOT / "results" / f"p1_centered_panel_20261101_main_{tag}.json"
        summary = json.loads(summary_path.read_bytes())
        sources[str(summary_path)] = base.sha(summary_path)
        if [r["arm"] for r in summary["rows"]] != list(panel.ARMS):
            raise ValueError("full fixed roster")
        for item in summary["rows"]:
            path = Path(item["report"])
            if base.sha(path) != item["report_sha256"]:
                raise ValueError("report hash")
            sources[str(path)] = item["report_sha256"]
            report = json.loads(path.read_bytes())
            row = report["rows"][0]
            if row["full_horizon"]:
                panel.saved.check(path.parent, item["report_sha256"])
            if row["first_error"] or row["arm"] != item["arm"]:
                raise ValueError("infrastructure error")
            actual = {}
            for encoding in row["encodings"]:
                path = Path(encoding["latent"]["path"])
                if base.sha(path) != encoding["latent"]["file_sha256"]:
                    raise ValueError("actual encoding SHA")
                sources[str(path)] = base.sha(path)
                latent = np.load(path, allow_pickle=False)
                if latent.shape != (192,) or latent.dtype != np.float32 or not np.isfinite(latent).all():
                    raise ValueError("actual latent shape")
                actual[encoding["step"]] = latent
            if not np.array_equal(actual[0], np.asarray(encoded["history_latents"][-1], np.float32)):
                raise ValueError("fresh root latent")
            branches.append((tag, row, actual))
    for path in (metric_source.TARGET, metric_path, Path(__file__).resolve()):
        sources[str(path)] = base.sha(path)
    sources.update(base.bindings())
    rows = []
    skipped = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device="cuda:0").eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda:0")
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])

        def score(z):
            residual = (np.asarray(z, np.float32)-goal).astype(float)
            return float(np.mean((residual @ matrix.T)**2))

        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for tag, row, actual in branches:
                history = list(encoded["history_latents"])
                issued = np.asarray(row["executed_raw"], np.float32).reshape(-1, 2)
                all_raw = np.concatenate((np.asarray(history_raw, np.float32), issued))
                points = {p["step"]: p for p in row["checkpoints"]}
                for boundary in (0, 5, 10, 15):
                    if boundary+5 not in actual or boundary not in actual or len(issued) < boundary+5:
                        skipped.append(dict(tag=tag, arm=row["arm"], boundary=boundary,
                                            reason="no_complete_encoded_same_time_packet"))
                        break
                    context = ((tensor(all_raw[boundary:boundary+15])-mean)/std).reshape(1, 3, 10)
                    prediction = model.predict(tensor(history[-3:])[None], model.action_encoder(context))[0, -1]
                    pred = prediction.cpu().numpy()
                    if not np.isfinite(pred).all():
                        raise ValueError("finite prediction")
                    previous, successor = actual[boundary], actual[boundary+5]
                    native_before = target.witness.old.qualification.native(entry["goal_state"], points[boundary]["state"])
                    native_after = target.witness.old.qualification.native(entry["goal_state"], points[boundary+5]["state"])
                    rows.append(dict(tag=tag, arm=row["arm"], boundary=boundary,
                        prediction=pred.tolist(), actual=successor.tolist(),
                        prediction_mse=float(np.mean((pred-successor)**2)), current_Q=score(previous),
                        predicted_delta_Q=score(pred)-score(previous),
                        actual_delta_Q=score(successor)-score(previous),
                        actual_delta_joint_xy=native_after["joint_xy_error"]-native_before["joint_xy_error"]))
                    history.append(successor.tolist())
        torch.cuda.synchronize()
        if base.model_hash(model) != before or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("source/model drift")
        digest = base.write(out, dict(rows=rows, skipped=skipped, sources=sources, host=host,
            model_before=before, model_after=before, encoder_calls=0, predict_calls=len(rows),
            action_encode_calls=len(rows), P1_admission=False, independent_confirmation=False,
            diagnostic_only=True, outcomes_used_for_selection=False))
    print(json.dumps(dict(sha256=digest, packets=len(rows), skipped=skipped,
        predicted_actual_reversals=sum(r["predicted_delta_Q"]*r["actual_delta_Q"] < 0 for r in rows),
        actual_Q_task_reversals=sum(r["actual_delta_Q"]*r["actual_delta_joint_xy"] < 0 for r in rows))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    run(args.output)
