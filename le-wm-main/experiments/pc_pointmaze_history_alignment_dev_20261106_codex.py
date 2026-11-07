"""Compare exact observed action history with single-frame and zero-padding."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

OUT = ROOT / "results/pc_pointmaze_history_alignment_dev_20261106_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_HISTORY_ALIGNMENT_DEV_PROTOCOL_20261106_CODEX.md"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    model, dataset, permission = load_model()
    before = model_fingerprint(model)
    raw_actions = torch.load(dataset.data_path / "actions.pth", map_location="cpu", weights_only=True).float()
    states = torch.load(dataset.data_path / "states.pth", map_location="cpu", weights_only=True)
    rows = []
    print("Real history vs single context and zero-history in state-assisted JEPA, GPU.", flush=True)
    with torch.inference_mode():
        for root in range(1884, 1892):
            if int(dataset.seq_lengths[root]) < 16:
                raise ValueError("trajectory too short")
            images = torch.load(dataset.data_path / f"obses/episode_{root:03d}.pth",
                                map_location="cpu", weights_only=True)
            context_images = images[[0, 5, 10]].permute(0, 3, 1, 2)[None].cuda()
            context_states = states[root, [0, 5, 10]][None].cuda()
            normalized = ((raw_actions[root, :15] - dataset.action_mean) / dataset.action_std).cuda()
            packets = normalized.reshape(3, 10)
            past, suffix = packets[:2][None], packets[2:3, None, :]
            encoded = model.encode({"visual": context_images, "proprio": context_states})
            target = model.encode(images[15].permute(2, 0, 1)[None, None].cuda())[0, 0]
            predictions, timings = {}, {}
            for name in ("single", "zero_history", "real_history"):
                torch.cuda.synchronize()
                started = time.perf_counter()
                if name == "single":
                    model.ctxt_window = 1
                    current = {k: encoded[k][:, -1:] for k in ("visual", "proprio")}
                    from tensordict import TensorDict
                    z = TensorDict(current, batch_size=[1, 1])
                    pred = model.unroll(z, suffix)["visual"][-1, 0]
                elif name == "zero_history":
                    model.ctxt_window = 3
                    pred = model.unroll(encoded, suffix)["visual"][-1, 0]
                else:
                    pred = unroll_history(model.model, encoded["visual"], encoded["proprio"], past, suffix, 3)["visual"][-1, 0]
                torch.cuda.synchronize()
                predictions[name] = pred
                timings[name] = time.perf_counter() - started
            mse = {k: float((v - target).square().mean()) for k, v in predictions.items()}
            rows.append({"root": root, "mse": mse, "prediction_rmse_history_vs_zero": float(
                (predictions["real_history"] - predictions["zero_history"]).square().mean().sqrt()),
                         "seconds": timings})
            print(f"root={root}: {mse}", flush=True)
    if before != model_fingerprint(model):
        raise ValueError("weights changed")
    means = {name: float(np.mean([r["mse"][name] for r in rows]))
             for name in ("single", "zero_history", "real_history")}
    wins = sum(r["mse"]["real_history"] < r["mse"]["single"] for r in rows)
    output = {"rows": rows, "mean_mse": means, "real_history_better_than_single_roots": wins,
              "development_exit_met": bool(wins >= 6 and means["real_history"] < means["single"]),
              "checkpoint_audit_hash": permission.fingerprint, "model_fingerprint": before,
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "scope": "state-assisted exposed data; offline actual suffix, not online planning"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("mean_mse", "real_history_better_than_single_roots",
                                              "development_exit_met")}, indent=2))


if __name__ == "__main__":
    main()
