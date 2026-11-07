"""Post-hoc repeated model JVP/VJP for the preidentified adjoint failure."""
import json

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import pc_tworoom_search_region_latent_jvp_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_LATENT_JVP_ADJOINT_REPLAY_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_latent_jvp_adjoint_replay_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(source.OUTPUT.read_text())
    if (result["runner_sha256"] != file_hash(source.__file__)
            or result["protocol_sha256"] != file_hash(source.PROTOCOL)
            or result["parent_sha256"] != file_hash(source.previous.OUTPUT)):
        raise ValueError("source drift")
    parent = json.loads(source.previous.OUTPUT.read_text())
    row, = (r for r in parent["rows"] if (r["episode"], r["steps"], r["direction"]) == (9000, 25, 0))
    failed, = (r for r in result["rows"] if (r["episode"], r["steps"], r["direction"]) == (9000, 25, 0))
    if failed["adjoint_pass"]:
        raise ValueError("target was not the failed record")
    model, scaler, transform, normalizer = source.previous.source.common.model_setup()
    digest = model_digest(model)
    if digest != result["model_digest"] or normalizer != result["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.OUTPUT, source.previous.OUTPUT,
                         source.previous.source.INPUT, source.previous.source.common.CHECKPOINT / "weights.pt",
                         source.previous.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="five steps per packet")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    with np.load(source.previous.source.INPUT, allow_pickle=False) as inputs:
        indices = np.flatnonzero(inputs["episodes"] == 9000)
        if len(indices) != 1:
            raise ValueError("ambiguous episode")
        i = int(indices[0])
        context = {"pixels": transform(inputs["current"][i])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][i])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
    raw = torch.tensor(row["physical_actions"], dtype=torch.float32, device="cuda")
    direction = torch.tensor(row["direction_vector"], dtype=torch.float32, device="cuda")
    def rollout(x):
        _, ctx = guard.score(context, source.previous.normalize_actions(x, scaler.mean_, scaler.scale_))
        return ctx["predicted_emb"][0, 0, -1]
    rng = np.random.default_rng(27126000 + 9000 * 100 + 25 * 2)
    projection = torch.tensor(rng.choice([-1., 1.], size=192) / np.sqrt(192), dtype=torch.float32, device="cuda")
    rows = []
    print("Repeat selected RGB latent JVP/VJP and projected finite differences 10 times.", flush=True)
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(10):
            value = source.compare(rollout, raw, direction, projection, source.previous.EPSILONS)
            with torch.no_grad():
                finite = {str(eps): float(((rollout(raw + eps * direction)
                                           - rollout(raw - eps * direction)) * projection).sum() / (2 * eps))
                          for eps in source.previous.EPSILONS}
            rows.append({"repeat": i, "jvp": value["adjoint_left"], "vjp": value["adjoint_right"],
                         "adjoint_error": value["adjoint_error"], "projected_fd": finite})
            print(f"repeat={i} adjoint_error={value['adjoint_error']:.8f}", flush=True)
    guard.verify_unchanged()
    with OUTPUT.open("x") as f:
        json.dump({"scope": "post-hoc selected failure", "source_sha256": file_hash(source.OUTPUT),
                   "parent_sha256": file_hash(source.previous.OUTPUT), "runner_sha256": file_hash(__file__),
                   "protocol_sha256": file_hash(PROTOCOL), "model_digest": digest,
                   "normalizer_sha256": normalizer, "rows": rows}, f, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
