"""Fixed16 actual RGB and same-action open/feedback predictions on host CUDA.

Known roots and already observed actions; no optimization, training, physical
dispatch, or P1 admission. Prediction and encoding costs are logged separately.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_backtrack_cuda_replay_20261030 as replay
from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as physical_consume
from experiments.spectral_reachability import p1_backtrack_physical_runner_20261030 as physical

NAME = "p1_backtrack_rgb_feedback_cuda_20261030"
ROOT = physical.ROOT
REPORT = ROOT / "results/p1_backtrack_physical_runner_20261030_MAIN02/report.json"
REPORT_SHA = "26802255a8192e95ae3d67a089ee673ba6e8a7331786481fefde9f4fd1e84de9"
REPLAY = ROOT / "results/p1_backtrack_cuda_replay_20261030_MAIN01.json"
REPLAY_SHA = "83ea957ecb38025e6b5489a0bf843b112906b0954b8b6c62238c10461b31cfa4"
require = physical.require


def inputs():
    require(physical.sha(REPORT) == REPORT_SHA and physical.sha(REPLAY) == REPLAY_SHA, "fixed source report SHA")
    physical_report = physical.native.read(REPORT, cap=64 * 1024**2)
    binding = physical_report["execution_binding"]
    physical_prepare, data = physical.verify(binding["prepared"], binding["prepare_sha256"])
    physical_consume.check(REPORT.parent, physical_report, physical_prepare)
    action, _, source_report, original = replay.sources()
    model_replay = replay.one.saved.read_verified(REPLAY, REPLAY_SHA)
    rows, bindings = [], deepcopy(physical_prepare["bindings"])
    bindings.update(original["bindings"])
    for path in (REPORT, REPLAY, REPORT.parent / "journal.jsonl", Path(__file__),
                 Path(physical_consume.__file__), Path(replay.__file__)):
        bindings[str(path.resolve())] = physical.sha(path)
    for i, (row, plan, item, predicted) in enumerate(zip(
            physical_report["rows"], physical_prepare["plans"], action["rows"], model_replay["rows"], strict=True)):
        require(row["index"] == i and all(plan[k] == item[k] for k in ("slot", "arm", "role", "candidate_index"))
                and all(predicted["exact"].values()), "fixed replay and physical role")
        task = original["requests"][plan["slot"]]["task"]
        require(task["binding"]["task_id"] == plan["task_id"], "same model task")
        cps = row["checkpoints"]
        require([cp["step"] for cp in cps] == [0, 5, 10, 15], "all same-time real frames")
        images = []
        for cp in cps:
            path = physical.native.paired.flat(REPORT.parent, cp["png"]).resolve()
            require(physical.sha(path) == cp["file_sha256"], "PNG file")
            bindings[str(path)] = cp["file_sha256"]
            images.append(dict(step=cp["step"], path=str(path), file_sha256=cp["file_sha256"],
                               pixel_sha256=cp["pixel_sha256"]))
        packet = data["roots"][plan["root_slot"]]["packets"][plan["condition"]]
        goal = np.asarray(packet["goal_rgb"])
        require(goal.dtype == np.uint8 and goal.shape == (224, 224, 3), "goal RGB")
        goal_sha = hashlib.sha256(goal.tobytes()).hexdigest()
        export = data["exports"][(plan["root_slot"], plan["condition"])]
        require(goal_sha == row["goal_pixel_sha256"], "goal pixels")
        rows.append(dict(index=i, slot=plan["slot"], arm=plan["arm"], role=plan["role"],
                         task_id=plan["task_id"], images=images, goal_path=export["path"],
                         goal_file_sha256=export["sha256"], goal_pixel_sha256=goal_sha,
                         raw15_float32_hex=plan["raw15_float32_hex"],
                         model_history=task["root_context"]["latent_history"],
                         action_history=task["root_context"]["action_history"],
                         action_mean=task["action_mean"], action_std=task["action_std"],
                         open_endpoints=predicted["endpoints"],
                         native_goal=row["goal"], native_responses=row["responses"]))
    for path, expected in bindings.items():
        require(physical.sha(path) == expected, "all inputs/source hashes")
    return dict(schema=NAME + ".prepare", source_report_sha256=REPORT_SHA, replay_sha256=REPLAY_SHA,
                model_hash=source_report["model_before"], policy=original["policy"], rows=rows,
                bindings=bindings, fixed_rows=16, image_encoder_calls=80, predictor_calls=32,
                training=False, P1_admission=False, freshness=False)


def prepare(output):
    path = Path(output)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
            "exclusive prepare")
    return replay.one.saved.write(path, inputs())


def image_bytes(item):
    path = Path(item["path"])
    require(physical.sha(path) == item["file_sha256"], "PNG hash")
    with Image.open(path) as image:
        require(image.mode == "RGB" and image.size == (224, 224), "native RGB")
        rgb = np.asarray(image).copy()
    require(hashlib.sha256(rgb.tobytes()).hexdigest() == item["pixel_sha256"], "PNG pixels")
    return rgb


def feedback_contexts(source, actual):
    """Contexts immediately after cp5 and cp10, with real cp0/cp5/cp10 latents."""
    z = np.asarray(source, dtype=np.float32)
    a = np.asarray(actual, dtype=np.float32)
    require(z.shape == (3, 192) and a.shape == (4, 192) and
            np.isfinite(z).all() and np.isfinite(a).all(), "exact context dimensions")
    return np.stack((np.stack((z[1], a[0], a[1])), np.stack((a[0], a[1], a[2]))))


def run(prepared, expected, output):
    p = replay.one.saved.read_verified(prepared, expected)
    require(p == inputs(), "prepare/source/code identity")
    out = Path(output)
    require(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
            and not out.exists() and not out.is_symlink(), "exclusive result")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    host = require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(p["policy"], device=device).eval().requires_grad_(False)
    before = replay.one.host.signal.model_hash(model)
    require(before == p["model_hash"], "frozen checkpoint identity")
    counts = dict(encoder=0, projector=0, predictor=0, action_encoder=0)
    handles = []
    for name in counts:
        module = getattr(model, name)
        def hook(_module, _args, key=name):
            counts[key] += 1
        handles.append(module.register_forward_pre_hook(hook))
    output_rows = []
    try:
        transform = p0.image_transform()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for row in p["rows"]:
                packet = physical.native.moving.load_blind(row["goal_path"], row["goal_file_sha256"])
                goal = np.asarray(packet["goal_rgb"])
                require(hashlib.sha256(goal.tobytes()).hexdigest() == row["goal_pixel_sha256"], "goal pixel hash")
                latents = []
                for rgb in [image_bytes(item) for item in row["images"]] + [goal]:
                    x = p0._image_tensor(rgb, transform=transform, device=device)
                    z = model.encode({"pixels": x})["emb"]
                    require(z.device.type == "cuda" and z.dtype == torch.float32 and
                            tuple(z.shape) == (1, 1, 192), "native CUDA image encoding")
                    latents.append(z[0, 0].cpu().numpy().copy())
                actual = np.stack(latents[:4])
                contexts = feedback_contexts(row["model_history"], actual)
                raw = np.frombuffer(bytes.fromhex(row["raw15_float32_hex"]), dtype="<f4").reshape(3, 10)
                mean = torch.tensor(np.tile(row["action_mean"], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(row["action_std"], 5), dtype=torch.float32, device=device)
                tokens = (torch.tensor(raw.copy(), device=device) - mean) / std
                history = torch.tensor(row["action_history"], dtype=torch.float32, device=device)
                refreshed = []
                for j in range(2):
                    h = torch.cat((history[-1:], tokens[:2]), dim=0) if j == 0 else tokens
                    z = torch.tensor(contexts[j], dtype=torch.float32, device=device)[None]
                    # j=0 predicts packet 2 with history [old-last, packet1, packet2].
                    # j=1 predicts packet 3 with history [packet1, packet2, packet3].
                    end = model.predict(z, model.action_encoder(h[None]))[:, -1]
                    require(end.device.type == "cuda" and tuple(end.shape) == (1, 192), "feedback prediction")
                    refreshed.append(end[0].cpu().tolist())
                output_rows.append(dict(index=row["index"], slot=row["slot"], arm=row["arm"], role=row["role"],
                                        actual_latents=actual.tolist(), goal_latent=latents[4].tolist(),
                                        feedback_predictions=refreshed,
                                        open_predictions=row["open_endpoints"], native_responses=row["native_responses"]))
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    after = replay.one.host.signal.model_hash(model)
    require(before == after and counts == dict(encoder=80, projector=80, predictor=32, action_encoder=32),
            "model and counted native calls")
    for path, digest in p["bindings"].items():
        require(physical.sha(path) == digest, "final input hash")
    result = dict(schema=NAME + ".report", prepared=str(Path(prepared).resolve()), prepare_sha256=expected,
                  model_before=before, model_after=after, host=host, calls=counts, rows=output_rows,
                  fixed_rows=16, training=False, P1_admission=False, freshness=False)
    return replay.one.saved.write(out, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(dict(path=args.output, sha256=prepare(args.output), P1_admission=False)))
    else:
        print("Encoding fixed actual RGB and comparing same-action feedback predictions on CUDA.", flush=True)
        print(json.dumps(dict(path=args.output, sha256=run(args.prepared, args.sha256, args.output))))


if __name__ == "__main__":
    main()
