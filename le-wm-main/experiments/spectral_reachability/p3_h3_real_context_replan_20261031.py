"""Plan fixed P3 suffixes from actual RGB context using the frozen CUDA model."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import p3_h3_real_context_replay_20261031 as source

ROOT, base, need = source.ROOT, source.base, source.need
NAME = "p3_h3_real_context_replan_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P3_H3_REAL_CONTEXT_REPLAN_PROTOCOL_20261031.md"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    entries, excluded, sources = source.collect()
    for path, digest in base.bindings().items():
        need(path not in sources or sources[path] == digest, "binding collision")
        sources[path] = digest
    for path in (PROTOCOL, Path(__file__).resolve()):
        sources[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen checkpoint")
        transform = source.p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for item in entries:
                tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
                with torch.no_grad():
                    embeddings = [model.encode({"pixels": source.p0._image_tensor(
                        image, transform=transform, device=device)})["emb"][0, 0]
                                  for image in item["frames"][:3]]
                    states = torch.stack(embeddings).unsqueeze(0)
                    actions = tensor(item["actions"])
                    mean, std = tensor(item["task"]["action_mean"]), tensor(item["task"]["action_std"])
                    goal, projection = tensor(item["task"]["goal"]), tensor(item["task"]["projection"])
                    need(states.shape == (1, 3, 192) and mean.shape == std.shape == (2,)
                         and bool(torch.isfinite(std).all()) and bool((std > 0).all()), "context ABI")
                    history = ((actions[5:15] - mean) / std).reshape(1, 2, 10)
                suffix = torch.nn.Parameter(actions[15:20].clone())
                optimizer = torch.optim.Adam([suffix], lr=0.05)

                def score():
                    token = ((suffix - mean) / std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), 1)))[0, -1]
                    return (projection @ (goal - pred)).square().mean()

                torch.cuda.synchronize(device)
                start = time.monotonic()
                with torch.no_grad():
                    baseline = float(score())
                for _ in range(20):
                    optimizer.zero_grad(set_to_none=True)
                    loss = score()
                    loss.backward()
                    need(suffix.grad is not None and bool(torch.isfinite(suffix.grad).all()), "gradient")
                    optimizer.step()
                    with torch.no_grad():
                        suffix.clamp_(-1, 1)
                with torch.no_grad():
                    final = float(score())
                need(np.isfinite([baseline, final]).all(), "finite score")
                torch.cuda.synchronize(device)
                original = item["actions"][15:20].astype("<f4")
                chosen = suffix.detach().cpu().numpy().astype("<f4") if final < baseline else original.copy()
                need(chosen.shape == (5, 2) and np.isfinite(chosen).all()
                     and (np.abs(chosen) <= 1).all(), "bounded finite action")
                prefix = item["actions"][:15].astype("<f4").tobytes()
                rows.append({"logical_pair": item["logical_pair"], "variant": item["variant"],
                             "decision_index": item["key"][0], "candidate": item["key"][1],
                             "policy": item["key"][2], "root_slot": item["key"][3],
                             "prefix15_float32_hex": prefix.hex(),
                             "original_suffix_float32_hex": original.tobytes().hex(),
                             "chosen_suffix_float32_hex": chosen.tobytes().hex(),
                             "action_sha256": hashlib.sha256(prefix + chosen.tobytes()).hexdigest(),
                             "context_pixel_sha256": [hashlib.sha256(frame.tobytes()).hexdigest()
                                                       for frame in item["frames"][:3]],
                             "baseline_score": baseline, "optimized_score": final,
                             "chosen": "optimized" if final < baseline else "original",
                             "optimizer_seconds": time.monotonic() - start})
        after = base.model_hash(model)
        need(before == after and all(base.sha(Path(path)) == digest for path, digest in sources.items()),
             "model/source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "excluded": excluded,
                  "host": host, "sources": sources, "model_before": before, "model_after": after,
                  "per_row_calls_code_path": {"encode": 3, "action_encoder": 22,
                                              "predict": 22, "backward": 20},
                  "P1_admission": False, "P3_admission": False, "independent_confirmation": False,
                  "equal_budget_certified": False,
                  "limitations": "Known roots, post-outcome exploratory plans; physical execution pending."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "sha256": digest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Optimize all fixed P3 suffixes from actual RGB context on host CUDA.", flush=True)
    run(args.output)
