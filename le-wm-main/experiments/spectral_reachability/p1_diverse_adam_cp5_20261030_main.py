"""Frozen CUDA Adam suffix planning for each new diverse prefix."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import p1_diverse_h10_20261030_main as h10
from . import codex_p1_adam_cp5_20261030_main as adam
from . import codex_p1_three_prefix_blind_inputs_20261030_v1 as blind

NAME = "p1_diverse_adam_cp5_20261030_main"
ROOT = h10.ROOT
H10_PREP = ROOT / "results/p1_diverse_h10_20261030_main_prepare_RUN01.json"
H10_SHA = "1beac317193419c0e1ff38e4b6b921251c5b2732f9c7e5ea4a6ae129e796c5d1"
PROTOCOL = ROOT.parent / "docs/D2P_P1_DIVERSE_ADAM_CP5_PROTOCOL_20261030_MAIN.md"
base, prior = adam.prior.base, adam.prior


def build():
    p, _ = h10.verify(H10_PREP, H10_SHA)
    _, _, _, model_prepared = h10.physical.source.replay.sources()
    joined = h10.native.read(h10.physical.JOIN, cap=32*1024**2)
    need = h10.need
    rows = []
    for anchor in p["anchors"]:
        slot = 2*anchor["root_slot"] + int(anchor["condition"] == "on")
        req = model_prepared["requests"][slot]
        task = req["task"]
        native_task = joined["task_rows"][slot]
        need((req["root_slot"], req["condition"], task["binding"]["task_id"]) ==
             (anchor["root_slot"], anchor["condition"], anchor["task_id"]), "task identity")
        overlay = 0 if anchor["condition"] == "off" else 1
        receipts = [x["receipt"] for x in anchor["trace"] if x["receipt"]["phase"] == "plan"
                    and x["receipt"]["step"] == 5 and x["receipt"]["overlay_index"] == overlay]
        need(len(receipts) == 1 and all(receipts[0][k] == anchor["checkpoint"][k] for k in
             ("png", "file_sha256", "pixel_sha256", "state", "snapshot")), "cp5 identity")
        path = h10.DIRECTORY / receipts[0]["png"]
        need(h10.sha(path) == receipts[0]["file_sha256"], "cp5 PNG source")
        warm = np.frombuffer(bytes.fromhex(native_task["warmup_raw_float32_hex"]), dtype="<f4").reshape(10, 2)
        rows.append(blind.make_row(anchor, task, warm, path.read_bytes()))
    need([r["index"] for r in rows] == [6, 18, 21], "fixed roster")
    files = dict(p["bindings"])
    for path in (H10_PREP, Path(__file__).resolve(), PROTOCOL, Path(blind.__file__).resolve(),
                 Path(prior.__file__).resolve(), Path(prior.fixed.__file__).resolve(), Path(adam.__file__).resolve()):
        files[str(path)] = h10.sha(path)
    files.update(base.bindings())
    need(all(h10.sha(k) == v for k, v in files.items()), "source drift")
    return dict(schema=NAME+".prepare", rows=rows, source_h10_sha256=H10_SHA, bindings=files,
                P1_admission=False, independent_confirmation=False, physical_execution=False)


def prepare(path):
    path = Path(path)
    h10.need(path.is_absolute() and path.parent == ROOT / "results" and
             path.name.startswith(NAME+"_prepare_") and not path.exists(), "exclusive output")
    return base.write(path, build())


def run(path, digest, output):
    path, output = Path(path), Path(output)
    h10.need(path.is_absolute() and h10.sha(path) == digest and output.is_absolute()
             and output.parent == ROOT / "results" and output.name.startswith(NAME+"_")
             and not output.exists(), "input/output identity")
    prep = json.loads(path.read_bytes())
    h10.need(prep == build() and h10.sha(path) == digest, "prepare drift")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0
    import base64
    import io
    from PIL import Image

    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        h10.need(before == base.MODEL_HASH, "frozen model hash")
        counts = {"encoder": 0, "projector": 0, "action_encoder": 0, "predictor": 0, "backward": 0}
        handles = []
        for name in ("encoder", "projector", "action_encoder", "predictor"):
            def hook(_module, _args, key=name):
                counts[key] += 1
            handles.append(getattr(model, name).register_forward_pre_hook(hook))
        rows = []
        transform = p0.image_transform()
        try:
            with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
                for row in prep["rows"]:
                    blind.validate_row(row)
                    with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as im:
                        pixels = np.asarray(im).copy()
                    with torch.no_grad():
                        cp5 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
                    initial = prior.initial_actions(row["logical_pair"])
                    actions, scores = [], []
                    started = time.monotonic()
                    for init in initial:
                        current = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                        optimizer = torch.optim.Adam([current], lr=adam.LR)
                        for _ in range(adam.STEPS):
                            optimizer.zero_grad(set_to_none=True)
                            predicted, goal = prior.fixed.replay(model, row, cp5, current)
                            (predicted-goal).square().mean().backward()
                            counts["backward"] += 1
                            h10.need(current.grad is not None and bool(torch.isfinite(current.grad).all()), "gradient")
                            optimizer.step()
                            with torch.no_grad():
                                current.clamp_(-1, 1)
                        with torch.no_grad():
                            predicted, goal = prior.fixed.replay(model, row, cp5, current)
                            scores.append(float((predicted-goal).square().mean()))
                        actions.append(current.detach()[0].cpu().numpy().copy())
                    torch.cuda.synchronize(device)
                    winner = prior.pick(scores)
                    selected = np.asarray(actions[winner], dtype="<f4")
                    rows.append(dict(index=row["index"], logical_pair=row["logical_pair"], role=row["role"],
                                     prefix_raw_f32_hex=row["prefix_raw_f32_hex"],
                                     cp5_pixel_sha256=row["cp5_pixel_sha256"],
                                     initial_f32_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                                     candidate_scores=scores, selected_index=winner, selected_actions=selected.tolist(),
                                     selected_action_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                                     planning_wall_seconds=time.monotonic()-started))
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        h10.need(before == after and h10.sha(path) == digest and
                 all(h10.sha(k) == v for k, v in prep["bindings"].items()), "model/source drift")
        h10.need(counts == dict(encoder=3, projector=3, action_encoder=504, predictor=504, backward=240),
                 "call census")
        result = dict(schema=NAME, rows=rows, calls=counts, host=host, model_before=before, model_after=after,
                      prepare=str(path), prepare_sha256=digest, config=dict(starts=4, steps=20, optimizer="Adam",
                      learning_rate=adam.LR, seed=prior.SEED), P1_admission=False,
                      independent_confirmation=False, physical_execution=False, equal_budget_certified=False,
                      limitation="Known roots, no OS isolation, backward count is code-path not independent hook.")
        return base.write(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    args = parser.parse_args()
    if args.command == "prepare":
        digest = prepare(args.output)
    else:
        print("Plan three fixed new prefixes with frozen model on host CUDA.", flush=True)
        digest = run(args.prepared, args.prepare_sha, args.output)
    print(json.dumps({"sha256": digest, "P1_admission": False}))


if __name__ == "__main__":
    main()
