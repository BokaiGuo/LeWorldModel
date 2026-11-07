"""Replay exposed RGB latent derivatives under the corrected backend policy."""
import json

import numpy as np
import torch

from experiments import pc_tworoom_search_region_latent_jvp_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_derivative_backend_20261107_codex import precise_action_derivatives
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_PRECISE_DERIVATIVE_BACKEND_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_precise_derivative_backend_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old = json.loads(source.OUTPUT.read_text())
    if (old["runner_sha256"] != file_hash(source.__file__)
            or old["protocol_sha256"] != file_hash(source.PROTOCOL)
            or old["parent_sha256"] != file_hash(source.previous.OUTPUT)):
        raise ValueError("source drift")
    parent = json.loads(source.previous.OUTPUT.read_text())
    if len(parent["rows"]) != len(old["rows"]):
        raise ValueError("source coverage mismatch")
    model, scaler, transform, normalizer = source.previous.source.common.model_setup()
    digest = model_digest(model)
    if digest != old["model_digest"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.OUTPUT, source.previous.OUTPUT,
                         source.previous.source.INPUT, source.previous.source.common.CHECKPOINT / "weights.pt",
                         source.previous.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="five steps per packet")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    rows = []
    print("Replay 24 RGB latent derivatives with scoped TF32-disabled math backend.", flush=True)
    with np.load(source.previous.source.INPUT, allow_pickle=False) as data:
        for row, saved in zip(parent["rows"], old["rows"], strict=True):
            key = (row["episode"], row["steps"], row["direction"])
            if key != (saved["episode"], saved["steps"], saved["direction"]):
                raise ValueError("row identity mismatch")
            episode, steps, direction_id = key
            indexes = np.flatnonzero(data["episodes"] == episode)
            if len(indexes) != 1:
                raise ValueError("ambiguous task")
            i = int(indexes[0])
            context = {"pixels": transform(data["current"][i])[None, None, None].cuda(),
                       "goal": transform(data["goal"][i])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            raw = torch.tensor(row["physical_actions"], dtype=torch.float32, device="cuda")
            direction = torch.tensor(row["direction_vector"], dtype=torch.float32, device="cuda")
            def rollout(x):
                _, ctx = guard.score(context, source.previous.normalize_actions(x, scaler.mean_, scaler.scale_))
                return ctx["predicted_emb"][0, 0, -1]
            rng = np.random.default_rng(27126000 + episode * 100 + steps * 2 + direction_id)
            projection = torch.tensor(rng.choice([-1., 1.], size=192) / np.sqrt(192),
                                      dtype=torch.float32, device="cuda")
            with precise_action_derivatives():
                with torch.no_grad():
                    score, _ = guard.score(context, source.previous.normalize_actions(raw, scaler.mean_, scaler.scale_))
                    score_difference = abs(float(score.item()) - row["results"]["0.001"]["value"])
                current = source.compare(rollout, raw, direction, projection, source.previous.EPSILONS)
            if flags != (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32):
                raise ValueError("backend state not restored")
            rows.append({"episode": episode, "steps": steps, "direction": direction_id,
                         "prior_adjoint_error": saved["adjoint_error"],
                         "prior_default_score_abs_difference": score_difference, **current})
            print(f"episode={episode} steps={steps} direction={direction_id} error={current['adjoint_error']:.3g}",
                  flush=True)
    guard.verify_unchanged()
    with OUTPUT.open("x") as f:
        json.dump({"scope": "post-hoc exposed RGB numerical regression", "track": RGB,
                   "source_sha256": file_hash(source.OUTPUT), "parent_sha256": file_hash(source.previous.OUTPUT),
                   "policy_sha256": file_hash(ROOT / "experiments/pc_rgb_derivative_backend_20261107_codex.py"),
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "model_digest": digest, "normalizer_sha256": normalizer, "rows": rows},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
