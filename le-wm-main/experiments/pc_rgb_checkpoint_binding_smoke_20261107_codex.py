"""Actual-GPU smoke test for the opt-in checkpoint-bound RGB boundary."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as source
from experiments.pc_checkpoint_bound_rgb_20261107_codex import CheckpointBoundRGB
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = source.ROOT.parent / "docs/PC_RGB_CHECKPOINT_BINDING_SMOKE_PROTOCOL_20261107_CODEX.md"
OUT = source.ROOT / "results/pc_rgb_checkpoint_binding_smoke_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_record = json.loads(source.OUTPUT.read_text())
    if source_record["input_sha256"] != file_hash(source.INPUT):
        raise ValueError("development input drift")
    model, scaler, transform, normalizer = source.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != source_record["model_sha256"] or normalizer != source_record["normalizer_sha256"]:
        raise ValueError("development model or normalizer drift")
    weights = source.common.CHECKPOINT / "weights.pt"
    config = source.common.CHECKPOINT / "config.json"
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.INPUT, weights, config],
                         preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="five packets of five environment steps")
    baseline = RGBLeWM(model, audit)
    strict = CheckpointBoundRGB(model, audit, weights, config)
    with np.load(source.INPUT, allow_pickle=False) as data, torch.inference_mode():
        inputs = {"pixels": transform(data["current"][0])[None, None, None].cuda(),
                  "goal": transform(data["goal"][0])[None, None, None].cuda(),
                  "action": torch.empty((1, 1, 0, 10), device="cuda")}
        raw = np.zeros((25, 2), dtype=np.float32)
        norm = scaler.transform(raw).reshape(1, 1, 5, 10)
        actions = torch.as_tensor(norm, dtype=torch.float32, device="cuda")
        old_cost, old_context = baseline.score(inputs, actions)
        new_cost, new_context = strict.score(inputs, actions)
        cost_diff = float((old_cost - new_cost).abs().max().item())
        latent_diff = float((old_context["predicted_emb"] - new_context["predicted_emb"]).abs().max().item())
        if cost_diff > 1e-5 or latent_diff > 1e-5:
            raise ValueError("old and strict boundary scoring differs")
        sealed = strict.finish()
        episode = int(data["episodes"][0])
    record = {"scope": "developer smoke test, not scientific or origin certification",
              "episode": episode, "track_id": RGB, "model_sha256": fingerprint,
              "normalizer_sha256": normalizer, "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(Path(__file__)), "source_sha256": file_hash(source.OUTPUT),
              "score": float(new_cost.reshape(-1)[0].item()), "cost_max_abs_diff": cost_diff,
              "predicted_latent_max_abs_diff": latent_diff, "candidate_chain_queries": 2,
              "model_packet_steps": 10, "strict_boundary": sealed}
    with OUT.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
