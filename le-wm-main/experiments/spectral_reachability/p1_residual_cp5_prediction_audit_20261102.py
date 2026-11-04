"""Post-outcome same-action packet audit; never used for action selection."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_residual_cp5_feedback_20261102 as planning
from . import p1_residual_cp5_physical_20261102 as physical
from . import p1_residual_cp5_saved_20261102 as saved

ROOT = planning.ROOT
AUTHORITY = "RESIDUAL_CP5_SAME_ACTION_DIAGNOSTIC_CUDA"


def run(output, authority):
    planning.require(authority == AUTHORITY and not output.exists(), "authority or output")
    verified = saved.analyze()
    planning.require(verified["frame_references_verified"] == 144, "saved physical chain")
    contract = physical.read(saved.CONTRACT, saved.CONTRACT_SHA)
    report = physical.read(saved.RUN / "report.json", saved.REPORT_SHA)
    source = planning.protocol()

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(source["policy"], device=device).eval().requires_grad_(False)
    hasher = planning.engine.base.blind.host.signal.model_hash
    before = hasher(model)
    planning.require(before == source["model_hash"], "frozen checkpoint identity")
    task = source["task"]
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])[None]
    planning.require(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "action scaling")
    rows = []

    def observed(row, phase, step):
        frames = [o for o in row["observations"] if o["phase"] == phase and o["step"] == step]
        planning.require(len(frames) == 1, "unique observed frame")
        pixels = planning.rgb._frame(saved.RUN, frames[0])
        p0 = planning.engine.base.p0
        image = p0._image_tensor(pixels, transform=p0.image_transform(), device=device)
        z = model.encode({"pixels": image})["emb"]
        planning.require(tuple(z.shape) == (1, 1, 192) and bool(torch.isfinite(z).all()), "encoded frame")
        return z

    def predict(z, history, packet):
        context = torch.cat((history, ((packet - mean) / std).reshape(1, 1, 10)), dim=1)
        out = model.predict(z, model.action_encoder(context))
        planning.require(out.shape == z.shape and bool(torch.isfinite(out).all()), "predicted frame")
        return out[:, -1:], context[:, -2:]

    def mse(a, b):
        return float((a - b).square().mean().item())

    with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        for role, row in zip(contract["roles"], report["rows"], strict=True):
            planning.require(role["arm"] == row["arm"] and row["status"] == "fixed_horizon_completed",
                             "completed paired action")
            cp5, real1, real2 = (observed(row, "prefix", 5), observed(row, "tail", 5),
                                 observed(row, "tail", 10))
            states, actions = planning.encode.contexts(task, role["prefix"], cp5[0, 0].cpu().numpy())
            z, past = tensor(states)[None], tensor(actions)[None]
            packets = tensor(np.asarray(role["tail"], np.float32).reshape(2, 5, 2))
            first, past1 = predict(z, past, packets[0])
            open2, _ = predict(torch.cat((z, first), dim=1)[:, -3:], past1, packets[1])
            refreshed2, _ = predict(torch.cat((z, real1), dim=1)[:, -3:], past1, packets[1])
            rows.append({
                "arm": role["arm"], "metric": role["metric"], "tail_sha256": role["tail_sha256"],
                "first_prediction_mse": mse(first, real1),
                "open_second_prediction_mse": mse(open2, real2),
                "refreshed_second_prediction_mse": mse(refreshed2, real2),
                "goal_distance": {"actual_first": mse(real1, goal), "predicted_first": mse(first, goal),
                                  "actual_second": mse(real2, goal), "predicted_open_second": mse(open2, goal),
                                  "predicted_refreshed_second": mse(refreshed2, goal)},
                "latent_vectors": {name: value.flatten().cpu().tolist() for name, value in
                                   (("actual_first", real1), ("predicted_first", first),
                                    ("actual_second", real2), ("predicted_open_second", open2),
                                    ("predicted_refreshed_second", refreshed2))},
            })
    torch.cuda.synchronize(device)
    after = hasher(model)
    planning.require(before == after and len(rows) == 8, "frozen model and roster")
    return physical.write_new(output, {
        "schema": "p1_residual_cp5_prediction_audit.v1", "physical_report_sha256": saved.REPORT_SHA,
        "physical_contract_sha256": saved.CONTRACT_SHA, "source_sha256": physical.sha(__file__),
        "host": host, "model_before": before, "model_after": after, "rows": rows,
        "classification": "post_outcome_development_diagnostic", "P1_admission": False,
        "independent_confirmation": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authority", required=True)
    args = parser.parse_args()
    with planning.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        print(run(args.output, args.authority))
