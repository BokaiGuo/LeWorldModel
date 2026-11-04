"""Frozen Cube cached-packet versus reference rollout, one exposed source root."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import cube_d2p_training_source as source
from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock
from experiments.spectral_reachability import pc_v6_cube_packet_adapter as adapter
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as reference
from experiments.spectral_reachability.preflight_cube_d2p_p0 import _validate_cube_action_protocol

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data/cube_source_20261025/cube_single_expert.h5"
RECEIPT = ROOT / "results/cube_d2p_training_source_20261025/cube_training_source.json"
POLICY = Path("/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
OUTPUT = ROOT / "results/pc_v6_cube_packet_reference_v3_dev_20261104_codex.json"
EXPECTED_WEIGHTS = "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89"
EXPECTED_RECEIPT = "328ec561545675c59ca2df311c6686e7a06a6a2e1fc890b8240ec089dbba8ee7"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if digest(POLICY / "weights.pt") != EXPECTED_WEIGHTS or digest(RECEIPT) != EXPECTED_RECEIPT:
        raise ValueError("frozen checkpoint or receipt changed")
    receipt_json = json.loads(RECEIPT.read_text())
    if DATA.resolve() != Path(receipt_json["source"]["dataset"]).resolve():
        raise ValueError("source dataset path mismatch")
    receipt = source.validate_receipt(RECEIPT)
    contexts, provenance = source.sample_source_native_contexts(
        DATA, expected_sha256=receipt_json["source"]["dataset_sha256"],
        history_len=3, goal_offset_packets=1, count=4, seed=202610250,
    )
    context = contexts[0]
    mean = np.asarray(receipt["calibration_action_mean"], dtype=np.float32)
    std = np.asarray(receipt["calibration_action_std"], dtype=np.float32)
    print("Comparing frozen Cube predictions on an exposed source context; not a control evaluation.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        runtime = require_cuda_allocation("cuda")
        device = torch.device("cuda")
        model = load_compatible_pretrained(POLICY, device="cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        _validate_cube_action_protocol(model, mean, std)
        initial_hash = reference._parameter_hash(model)
        transform = _image_transform()
        frames = [*context["frames"][1:], context["root_frame"]]
        pixels = reference._frames_tensor(frames, transform=transform, device=device)
        history = reference._normalize_raw_packets(
            torch.as_tensor(context["history_raw_packets"], device=device), mean, std,
            action_block=5, action_dim=5,
        )[None]
        low, high = torch.full((5,), -1., device=device), torch.full((5,), 1., device=device)
        with torch.no_grad():
            encoded = model.encode({"pixels": pixels})["emb"]
            goal = reference._goal_latent(model, context["goal"], transform=transform, device=device)
            predict, target = adapter.make_predictor(model, encoded, history, goal, mean, std, chart_seed=42)
            chart = adapter.goal_chart(encoded, goal, 42)
            rows = []
            for length in (5, 10, 15):
                actions = np.stack((np.zeros((length, 5), dtype=np.float32),
                                    .3 * np.sin(np.arange(length * 5, dtype=np.float32).reshape(length, 5))))
                positions, scores = predict(actions)
                single_positions = np.concatenate([predict(action[None])[0] for action in actions])
                expected = torch.stack([
                    reference._terminal_prediction(
                        model, context["frames"], context["root_frame"], context["history_raw_packets"],
                        torch.as_tensor(action.reshape(-1, 5, 5), device=device),
                        mean=mean, std=std, action_block=5, action_dim=5, transform=transform,
                        device=device, low=low, high=high, future_packets_are_raw=True,
                    ) for action in actions
                ])
                expected_positions = (expected @ chart).cpu().numpy()
                expected_scores = ((expected - goal) ** 2).mean(dim=-1).cpu().numpy()
                future = adapter.raw_to_packets(actions, mean, std, device=device)
                lower = torch.as_tensor((-1 - mean) / std, device=device)
                upper = torch.as_tensor((1 - mean) / std, device=device)
                cached = adapter.cached_terminal(model, encoded, history, future, lower, upper)
                pos_error = float(np.max(np.abs(positions - expected_positions)))
                score_error = float(np.max(np.abs(scores - expected_scores)))
                rows.append({"length": length, "actions": actions.tolist(), "positions": positions.tolist(),
                             "scores": scores.tolist(), "reference_positions": expected_positions.tolist(),
                             "reference_scores": expected_scores.tolist(), "max_position_error": pos_error,
                             "max_score_error": score_error,
                             "max_full_latent_error": float((cached - expected).abs().max().item()),
                             "max_batch_vs_single_position_error": float(np.max(np.abs(positions-single_positions))),
                             "max_single_vs_reference_position_error": float(np.max(np.abs(single_positions-expected_positions)))})
            torch.cuda.synchronize()
        final_hash = reference._parameter_hash(model)
        if initial_hash != final_hash:
            raise ValueError("frozen model changed")
    record = {"scope": "exposed source RGB root; same frozen model; no physical or efficacy claim",
              "source_rows": context["source_rows"], "source_provenance": provenance[0],
              "dataset_sha256": receipt_json["source"]["dataset_sha256"],
              "checkpoint_sha256": EXPECTED_WEIGHTS, "receipt_sha256": EXPECTED_RECEIPT,
              "model_hash": initial_hash, "runtime": runtime, "chart_target": target.tolist(),
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "rows": rows,
              "within_tolerance": all(r["max_position_error"] <= 2e-5 and r["max_score_error"] <= 2e-6 for r in rows)}
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(OUTPUT), "within_tolerance": record["within_tolerance"]}), flush=True)


if __name__ == "__main__":
    main()
