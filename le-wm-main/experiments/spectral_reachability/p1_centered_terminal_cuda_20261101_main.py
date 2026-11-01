"""Fill eight missing terminal RGB encodings; preserve the partial24 receipt."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_centered_same_action_cuda_20261101_main as partial

ROOT = partial.panel.ROOT
PARTIAL = ROOT / "results/p1_centered_same_action_cuda_20261101_main_MAIN01.json"
PARTIAL_SHA = "82425c5a3a101f14985b4737fa2085bcd9aa84b2921c597d2095a4ffa85c02b0"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base = partial.target.encoder.base
    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT / "results" or output.exists():
        raise ValueError("exclusive output")
    if base.sha(PARTIAL) != PARTIAL_SHA:
        raise ValueError("sealed partial24")
    prior = json.loads(PARTIAL.read_bytes())
    sources = dict(prior["sources"])
    sources[str(PARTIAL)] = PARTIAL_SHA
    sources[str(Path(__file__).resolve())] = base.sha(Path(__file__))
    if len(prior["rows"]) != 24 or len(prior["skipped"]) != 8:
        raise ValueError("partial denominator")
    target = json.loads(partial.metric_source.TARGET.read_bytes())["rows"][0]
    metric = json.loads((ROOT / "results/p1_centered_metric_cuda_20261101_main_MAIN01.json").read_bytes())
    matrix, goal = np.asarray(metric["sqrt_matrix"], float), np.asarray(target["latent"], np.float32)
    task = json.loads(partial.target.witness.old.PREP.read_bytes())["source"]["tasks"][0]
    branches = []
    for tag in partial.TAGS:
        summary = json.loads((ROOT / "results" / f"p1_centered_panel_20261101_main_{tag}.json").read_bytes())
        for item in summary["rows"]:
            report_path = Path(item["report"])
            if base.sha(report_path) != item["report_sha256"]:
                raise ValueError("report")
            report = json.loads(report_path.read_bytes())
            row = report["rows"][0]
            if not row["full_horizon"] or row["control_returned"] != 20 or row["first_error"]:
                raise ValueError("complete20 required")
            enc = {e["step"]: e for e in row["encodings"]}
            history = []
            for step in (5, 10, 15):
                path = Path(enc[step]["latent"]["path"])
                if base.sha(path) != enc[step]["latent"]["file_sha256"]:
                    raise ValueError("actual history SHA")
                history.append(np.load(path, allow_pickle=False))
                sources[str(path)] = base.sha(path)
            points = {cp["step"]: cp for cp in row["checkpoints"]}
            cp = points[20]
            path = Path(cp["path"])
            if base.sha(path) != cp["file_sha256"] or cp["goal"] != target["goal_state"]:
                raise ValueError("actual terminal goal/PNG")
            with Image.open(path) as im:
                pixels = np.asarray(im).copy()
            if hashlib.sha256(pixels.tobytes()).hexdigest() != cp["pixel_sha256"]:
                raise ValueError("actual terminal pixels")
            sources[str(path)] = base.sha(path)
            raw = np.asarray(row["executed_raw"], np.float32)
            if raw.shape != (20, 2):
                raise ValueError("actual actions")
            branches.append(dict(tag=tag, arm=row["arm"], history=np.stack(history),
                                 raw=raw[5:20], pixels=pixels, before=points[15]["state"],
                                 after=cp["state"], terminal_pixel_sha256=cp["pixel_sha256"]))
    if any(base.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device="cuda:0").eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH or before != prior["model_before"]:
            raise ValueError("checkpoint")
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda:0")
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])

        def score(z):
            residual = (np.asarray(z, np.float32)-goal).astype(float)
            return float(np.mean((residual @ matrix.T)**2))

        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for b in branches:
                context = ((tensor(b["raw"])-mean)/std).reshape(1, 3, 10)
                pred = model.predict(tensor(b["history"])[None], model.action_encoder(context))[0, -1].cpu().numpy()
                image = p0._image_tensor(b["pixels"], transform=p0.image_transform(), device=torch.device("cuda:0"))
                actual = model.encode({"pixels": image})["emb"][0, 0].cpu().numpy()
                if pred.shape != (192,) or actual.shape != (192,) or not np.isfinite([pred, actual]).all():
                    raise ValueError("finite prediction/encoding")
                previous = b["history"][-1]
                before_task = partial.target.witness.old.qualification.native(target["goal_state"], b["before"])
                after_task = partial.target.witness.old.qualification.native(target["goal_state"], b["after"])
                rows.append(dict(tag=b["tag"], arm=b["arm"], boundary=15, prediction=pred.tolist(),
                    actual=actual.tolist(), prediction_mse=float(np.mean((pred-actual)**2)),
                    current_Q=score(previous), predicted_delta_Q=score(pred)-score(previous),
                    actual_delta_Q=score(actual)-score(previous),
                    actual_delta_joint_xy=after_task["joint_xy_error"]-before_task["joint_xy_error"],
                    terminal_pixel_sha256=b["terminal_pixel_sha256"]))
        torch.cuda.synchronize()
        if base.model_hash(model) != before or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model/source drift")
        combined = prior["rows"] + rows
        expected = {(tag, arm, boundary) for tag in partial.TAGS for arm in partial.panel.ARMS
                    for boundary in (0, 5, 10, 15)}
        if len(combined) != 32 or {(r["tag"], r["arm"], r["boundary"]) for r in combined} != expected:
            raise ValueError("complete32 identities")
        stats = dict(packets=32, predicted_actual_reversals=sum(
            r["predicted_delta_Q"]*r["actual_delta_Q"] < 0 for r in combined),
            actual_Q_task_reversals=sum(r["actual_delta_Q"]*r["actual_delta_joint_xy"] < 0 for r in combined))
        digest = base.write(output, dict(rows=combined, sources=sources, stats=stats, host=host,
            model_before=before, model_after=before, added_encoder_calls=8, added_predict_calls=8,
            added_action_encode_calls=8, P1_admission=False, independent_confirmation=False,
            diagnostic_only=True, outcomes_used_for_selection=False))
    print(json.dumps(dict(sha256=digest, **stats)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    run(args.output)
