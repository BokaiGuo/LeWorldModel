"""Frozen-source AD/FD diagnostic along the actual RGB compiler directions."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc1_multistep_response_dev_20261103_codex as base
from experiments import pc_multidirection_rgb_fixed_terminal_dev_20261103_codex as policy

old, h, ROOT = base.old, base.h, base.ROOT
PROTOCOL = ROOT.parent / "docs/PC1_RGB_SPECTRAL_DIRECTION_ADFD_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc1_rgb_spectral_direction_adfd_20261103_codex.json"


def comparison(reference, observed):
    a, b = np.asarray(reference, dtype=float), np.asarray(observed, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    error = float(np.linalg.norm(a - b))
    return {"absolute_error": error, "reference_norm": float(na),
            "observed_norm": float(nb),
            "relative_error": error / na if na >= 1e-6 else None,
            "cosine": float(a @ b / (na * nb)) if min(na, nb) >= 1e-6 else None}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history mismatch")
    policy.runner.PREFIX = policy.PREFIX
    decisions = [policy.runner.paths(root, "multidirection")[0][0] for root in range(4)]
    input_hashes = {str(p): sha(p) for p in decisions}
    camera, calibration = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("PC1 development: AD versus FD along sealed RGB spectral directions.", flush=True)
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), source["action_mean"]) or not np.array_equal(std.cpu().numpy(), source["action_std"]):
            raise ValueError("normalizer mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        rows, calls = [], {"encoding": 0, "forward": 0, "backward": 0, "forward_tokens": 0}
        started = time.monotonic()
        for root, path in enumerate(decisions):
            decision = json.loads(path.read_text())
            result = decision["result"]
            frames = histories[f"root{root}_history_rgb"]
            if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                raise ValueError("RGB history mismatch")
            if decision["input_rgb_sha256"] != [old.digest_image(frames[t]) for t in decision["ticks"]]:
                raise ValueError("decision context mismatch")
            if decision["model_hash"] != model_hash or decision["history_actions"] != old.WARMUP[-10:].tolist():
                raise ValueError("decision model or history mismatch")
            with torch.no_grad():
                entries = [h.encode_with_state(model, frames[t],
                           h.causal_state(frames[t - 1], frames[t], metadata, offset, .05), device)
                           for t in decision["ticks"]]
            calls["encoding"] += len(entries)
            ctx = {key: torch.cat([entry[key] for entry in entries], dim=1) for key in ("visual", "proprio")}
            history = ((torch.as_tensor(decision["history_actions"], dtype=torch.float32, device=device) - mean) / std).reshape(2, 1, 10)
            branches = [("first", None, result["response"])]
            for i, successor in enumerate(result["successor_response"]):
                parent = result["first_bank"][successor["parent_index"]]["prefix"]
                branches.append((f"successor{i}", parent, successor["response"]))
            if len(branches) != 3:
                raise ValueError("expected two retained parents")
            for name, parent, evidence in branches:
                nominal = np.asarray(evidence["nominal"], dtype=np.float32)
                horizon = 5 if parent is None else 10
                parent_tensor = None if parent is None else torch.as_tensor(parent, dtype=torch.float32, device=device)

                def predict(tail):
                    chain = tail if parent_tensor is None else torch.cat((parent_tensor, tail))
                    packets = ((chain - mean) / std).reshape(1, horizon // 5, 10).transpose(0, 1)
                    out = base.unroll_with_action_history(model, ctx, history, packets, 3)["proprio"][-1]
                    calls["forward"] += 1
                    calls["forward_tokens"] += horizon // 5
                    return (out.reshape(1, 256, 16).mean(1) @ pinv.T)[0, :2] * pos_std

                origin = torch.tensor(nominal, device=device, requires_grad=True)
                center = predict(origin)
                jac = torch.stack([torch.autograd.grad(center[i], origin, retain_graph=i == 0)[0]
                                   for i in range(2)])
                calls["backward"] += 2
                center_np = center.detach().cpu().numpy()
                if not np.allclose(center_np, evidence["center"], atol=2e-5, rtol=0):
                    raise ValueError("sealed nominal prediction mismatch")
                for index, vector in enumerate(evidence["right_directions"]):
                    vector = np.asarray(vector, dtype=np.float32).reshape(5, 2)
                    if not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
                        raise ValueError("non-unit sealed direction")
                    ad = (jac * torch.as_tensor(vector, device=device)).sum(dim=(1, 2)).detach().cpu().numpy()
                    saved_jv = np.asarray(evidence["jacobian"]) @ vector.reshape(-1)
                    measurements = []
                    for label, epsilon in (("small", .005), ("medium", .05),
                                           ("candidate", .5 / float(np.max(np.abs(vector))))):
                        plus, minus = nominal + epsilon * vector, nominal - epsilon * vector
                        if max(np.max(np.abs(plus)), np.max(np.abs(minus))) > 1 + 1e-7:
                            measurements.append({"scale": label, "epsilon": epsilon, "skipped": "action_bounds"})
                            continue
                        with torch.no_grad():
                            pos = predict(torch.as_tensor(plus, device=device)).cpu().numpy()
                            neg = predict(torch.as_tensor(minus, device=device)).cpu().numpy()
                        fd = (pos - neg) / (2 * epsilon)
                        measurements.append({"scale": label, "epsilon": epsilon, "plus": pos.tolist(),
                                             "minus": neg.tolist(), "fd": fd.tolist(),
                                             "comparison": comparison(ad, fd)})
                    rows.append({"root": root, "branch": name, "direction_index": index,
                                 "parent": parent, "nominal": nominal.tolist(), "direction": vector.tolist(),
                                 "center": center_np.tolist(), "ad": ad.tolist(), "saved_jv": saved_jv.tolist(),
                                 "saved_jv_comparison": comparison(ad, saved_jv), "measurements": measurements})
            print(f"root {root}: three contexts complete", flush=True)
        torch.cuda.synchronize()
        if h.parameter_hash(model) != model_hash or any(sha(Path(p)) != v for p, v in input_hashes.items()):
            raise ValueError("model or inputs changed")
        old.write_json(OUTPUT, {"scope": "exposed development roots only", "protocol_sha256": sha(PROTOCOL),
                                "runner_sha256": sha(Path(__file__)), "history_sha256": sha(old.HISTORY),
                                "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT), "model_hash": model_hash,
                                "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(),
                                "source_decisions_sha256": input_hashes, "calls": calls,
                                "elapsed_seconds": time.monotonic() - started, "rows": rows})
    print(f"sealed {len(rows)} directions at {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
