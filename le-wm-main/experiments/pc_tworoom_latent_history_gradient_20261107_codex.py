"""GPU split-vs-full learned-model trajectory and gradient audit."""
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_latent_history_rollout_20261107_codex as history
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_tworoom_real_vs_imagined_feedback_20261107_codex import prior

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_LATENT_HISTORY_GRADIENT_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_latent_history_gradient_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = prior.source
    source.verify()
    _, model, scaler, transform = source.setup()
    inp = source.b.q.old.base.source.block.prior.inputs.INPUT
    initial_fingerprint = model_fingerprint(model)
    rng = np.random.default_rng(271812)
    rows = []
    with np.load(inp, allow_pickle=False) as data:
        for root in (38, 39):
            with torch.no_grad():
                image = transform(data["current"][root])[None, None].cuda()
                goal_image = transform(data["goal"][root])[None, None].cuda()
                initial = model.encode({"pixels": image})["emb"][:, 0].detach()
                goal = model.encode({"pixels": goal_image})["emb"][:, 0].detach()
            raw = rng.uniform(-0.6, 0.6, (8, 25, 2)).astype(np.float32)
            normal = scaler.transform(raw.reshape(-1, 2)).reshape(8, 5, 10)
            for hs in (1, 3):
                for prefix in (1, 2, 4):
                    a = torch.tensor(normal, device="cuda", requires_grad=True)
                    emb = initial[:, None, :].expand(8, -1, -1)
                    full = history.rollout_history(model, emb, a[:, :0], a, hs)
                    full_cost = (full[:, -1] - goal).square().sum(dim=-1)
                    full_grad = torch.autograd.grad(full_cost.sum(), a, retain_graph=True)[0]
                    prefix_traj = history.rollout_history(model, emb, a[:, :0], a[:, :prefix], hs)
                    split = history.rollout_history(model, prefix_traj, a[:, :prefix], a[:, prefix:], hs)
                    split_cost = (split[:, -1] - goal).square().sum(dim=-1)
                    split_grad = torch.autograd.grad(split_cost.sum(), a, retain_graph=True)[0]
                    detached = history.rollout_history(model, prefix_traj.detach(), a[:, :prefix],
                                                      a[:, prefix:], hs)
                    detached_grad = torch.autograd.grad((detached[:, -1] - goal).square().sum(), a,
                                                         retain_graph=True)[0]
                    short = history.rollout_history(model, prefix_traj[:, -1:], a[:, :0], a[:, prefix:], hs)
                    short_cost = (short[:, -1] - goal).square().sum(dim=-1)
                    short_grad = torch.autograd.grad(short_cost.sum(), a)[0]
                    torch.cuda.synchronize()
                    pred_error = float((split - full).detach().abs().max())
                    cost_error = float((split_cost - full_cost).detach().abs().max())
                    grad_error = float((split_grad - full_grad).abs().max())
                    rows.append({"root": root, "history_size": hs, "prefix_packets": prefix,
                                 "max_prediction_error": pred_error, "max_cost_error": cost_error,
                                 "max_gradient_error": grad_error,
                                 "max_prefix_gradient": float(full_grad[:, :prefix].abs().max()),
                                 "detach_prefix_gradient_error": float(
                                     (detached_grad[:, :prefix] - full_grad[:, :prefix]).abs().max()),
                                 "drop_history_final_prediction_error": float(
                                     (short[:, -1] - full[:, -1]).detach().abs().max()),
                                 "drop_history_gradient_error": float((short_grad - full_grad).abs().max()),
                                 "pass": pred_error <= 1e-4 and cost_error <= 1e-2 and grad_error <= 1e-3})
    if model_fingerprint(model) != initial_fingerprint:
        raise ValueError("model parameters changed")
    paths = [Path(__file__), Path(history.__file__), PROTOCOL, inp, Path(inspect.getsourcefile(type(model)))]
    with OUTPUT.open("x") as f:
        json.dump({"track": "rgb_only_dev", "source_sha256": {str(p): file_hash(p) for p in paths},
                   "model_sha256": initial_fingerprint, "seed": 271812, "rows": rows,
                   "all_pass": all(row["pass"] for row in rows),
                   "scope": "exposed two-root GPU numerical audit, not a planning experiment"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
