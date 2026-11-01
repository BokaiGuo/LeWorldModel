"""Oracle-rendered pusher subspace diagnostic on fixed, previously executed suffixes."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap
from . import codex_p1_pusher_render_intervention_20261030_main as render

NAME = "codex_p1_pusher_projection_diagnostic_20261030_main"
EMB = base.ROOT / "results/codex_p1_multistart20_terminal_gap_20261030_main_RUN01.json"
EMB_SHA = "3cfa240e51b8765e4d80d254d04d4439555581703940384d0366272704eb0235"
RENDER = base.ROOT / "results/codex_p1_pusher_render_intervention_20261030_main_RUN01.json"
RENDER_SHA = "d7d786fc0361b00c42abbb8d5190ffd6abb5309ca2873ce1966ea0e993178227"
OFFSETS = ((0, 0), (-20, 0), (20, 0), (0, -20), (0, 20))


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    fixed, _ = gap.read(EMB, EMB_SHA)
    prior, _ = gap.read(RENDER, RENDER_SHA)
    prepared, panel = render.images()
    if len(fixed["rows"]) != 24 or len(panel) != 52 or len(prior["rows"]) != 52:
        raise ValueError("incomplete panel")
    for item, old in zip(panel, prior["rows"], strict=True):
        if any(item[k] != old[k] for k in ("root_slot", "condition", "offset", "pixel_sha256")):
            raise ValueError("render panel identity")
    bindings = base.bindings()
    bindings.update(prepared["bindings"])
    for path in (Path(__file__).resolve(), Path(render.__file__).resolve(), EMB, RENDER):
        bindings[str(path)] = base.sha(path)
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("model mismatch")
        transform = p0.image_transform()
        groups, rows, excluded = {}, [], []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for item in panel:
                offset = tuple(item["offset"])
                if offset not in OFFSETS:
                    continue
                key = (item["root_slot"], item["condition"])
                z = model.encode({"pixels": p0._image_tensor(item["image"], transform=transform, device=device)})["emb"]
                if z.shape != (1, 1, 192) or not bool(torch.isfinite(z).all()):
                    raise ValueError("encoding ABI")
                groups.setdefault(key, {})[offset] = z[0, 0]
            bases = {}
            for key, z in groups.items():
                if set(z) != set(OFFSETS):
                    raise ValueError("fixed offset set")
                directions = torch.stack((z[(20, 0)] - z[(-20, 0)], z[(0, 20)] - z[(0, -20)]), dim=1)
                basis, singular, _ = torch.linalg.svd(directions, full_matrices=False)
                if float(singular[-1]) <= 1e-6:
                    raise ValueError("rank deficient pusher subspace")
                bases[key] = (z[(0, 0)], basis, singular)
            for index, row in enumerate(fixed["rows"]):
                if row["index"] != index:
                    raise ValueError("row order")
                if row["status"] != "probe_completed":
                    excluded.append({"index": index, "status": row["status"], "steps": row["returned_suffix"]})
                    continue
                key = (row["identity"]["root_slot"], row["identity"]["condition"])
                goal, basis, singular = bases[key]
                stored_goal = torch.as_tensor(row["goal_latent"], device=device, dtype=torch.float32)
                if not torch.equal(goal, stored_goal):
                    raise ValueError("goal binding")
                values = {}
                for label in ("actual", "predicted"):
                    z = torch.as_tensor(row[label + "_latent"], device=device, dtype=torch.float32)
                    residual = z-goal
                    full = residual.square().sum()
                    proj = (basis.T @ residual).square().sum()
                    values[label] = {"goal_mse": float(full/192), "pusher_subspace_mse": float(proj/192),
                                     "fraction_energy": float(proj/full) if float(full) else 0.0}
                if abs(values["actual"]["goal_mse"] - row["actual_goal_mse"]) > 1e-6:
                    raise ValueError("actual score mismatch")
                rows.append({"index": index, "identity": row["identity"], "physical_metrics": row["physical_metrics"],
                             "singular_values": singular.cpu().tolist(), **values})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model) or any(base.sha(Path(p)) != h for p, h in bindings.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "excluded": excluded,
                  "host": host, "sources": bindings, "encode_calls": 20, "model_before": before,
                  "model_after": before, "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Privileged render-only subspace, two known roots and dependent trajectories; "
                                 "subspace is not a physical pusher estimator or deployable score."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "excluded": excluded, "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Diagnose fixed real and predicted residuals in oracle-rendered pusher subspaces.", flush=True)
    run(Path(parser.parse_args().output))
