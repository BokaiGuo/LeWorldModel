"""Compare original and cached RGB scoring on sealed real-model inputs."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval import img_transform
from experiments import pc_rgb_gradient_endpoint_decomposition_20261107_codex as source
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_shared_encoder_cache_20261107_codex import CachedJEPA
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

PROTOCOL = source.ROOT.parent / "docs/PC_RGB_SHARED_ENCODER_CACHE_PROTOCOL_20261107_CODEX.md"
OUT = source.ROOT / "results/pc_rgb_shared_encoder_cache_benchmark_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(source.PLAN.read_text())
    if plan["public_sha256"] != file_hash(source.PUBLIC):
        raise ValueError("public input drift")
    model = load_compatible_pretrained(str(source.CHECKPOINT), device="cuda")
    digest = model_digest(model)
    if digest != plan["model_digest"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.PUBLIC, source.PLAN,
                         source.CHECKPOINT / "weights.pt", source.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; frozen action scaler",
                         clock="single-frame, five physical steps per packet")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows = []
    print("Test same-input cached and uncached scores, rankings, and synchronized latency.", flush=True)
    with np.load(source.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        source.capture.previous.planner.validate_public(public)
        mean, scale = public["action_mean"], public["action_scale"]
        guard = lambda m: SingleFrameRGB(m, audit, expected_model_digest=digest,
                                        action_mean=mean, action_scale=scale, packet_steps=5,
                                        physical_low=-1, physical_high=1)
        original = guard(model)
        cached_model = CachedJEPA(model)
        cached = guard(cached_model)
        for item in plan["rows"]:
            idx, = np.flatnonzero(public["episodes"] == item["episode"])
            context = {"pixels": transform(public["current"][idx])[None, None, None].cuda(),
                       "goal": transform(public["goal"][idx])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            physical = np.asarray(item["candidates"], dtype=np.float32)
            normal = ((physical - mean) / scale).astype(np.float32)
            for packets in range(1, 6):
                action = torch.as_tensor(normal[:, :5 * packets].reshape(1, 5, packets, 10), device="cuda")
                old_cost, old = original.score(context, action)
                new_cost, new = cached.score(context, action)
                old_pred, new_pred = old["predicted_emb"], new["predicted_emb"]
                if (not torch.allclose(old_cost, new_cost, rtol=1e-5, atol=1e-3)
                        or not torch.allclose(old_pred, new_pred, rtol=1e-5, atol=1e-3)):
                    raise ValueError("cached rollout parity failed")
                left, right = old_cost.cpu().numpy()[0], new_cost.cpu().numpy()[0]
                left_order, right_order = np.argsort(left, kind="stable"), np.argsort(right, kind="stable")
                if not np.array_equal(left_order, right_order):
                    raise ValueError("candidate ordering changed")
                elapsed = {}
                for name, boundary in (("original", original), ("cached", cached)):
                    samples = []
                    for _ in range(30):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        boundary.score(context, action)
                        torch.cuda.synchronize()
                        samples.append(time.perf_counter() - start)
                    elapsed[name] = {"median_seconds": float(np.median(samples)),
                                     "samples_seconds": samples}
                rows.append({"episode": item["episode"], "packets": packets, "candidates": 5,
                             "score_max_abs_error": float((old_cost-new_cost).abs().max().item()),
                             "feature_max_abs_error": float((old_pred-new_pred).abs().max().item()),
                             "same_order": True, "selected": int(left_order[0]), "timing": elapsed})
                print(f"episode={item['episode']} packets={packets} "
                      f"original={elapsed['original']['median_seconds']:.5f} "
                      f"cached={elapsed['cached']['median_seconds']:.5f}", flush=True)
        original.verify_unchanged()
        cached.verify_unchanged()
    with OUT.open("x") as stream:
        json.dump({"scope": "same-observation microbenchmark; not closed-loop cost or method gain",
                   "device": torch.cuda.get_device_name(), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__),
                   "cache_sha256": file_hash(Path(__file__).with_name("pc_rgb_shared_encoder_cache_20261107_codex.py")),
                   "public_sha256": file_hash(source.PUBLIC), "plan_sha256": file_hash(source.PLAN),
                   "model_digest": digest, "encoder_calls": cached_model.encoder_calls, "rows": rows},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
