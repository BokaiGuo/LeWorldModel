"""Check real-midpoint CUDA replanning parity before dispatch on all 24 slots."""
import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_split_budget_feedback_physical_20261031 as split

physical, planner = split.physical, split.planner
ROOT, need, sha = split.ROOT, split.need, split.sha
NAME = "p1_split_budget_online_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_SPLIT_BUDGET_ONLINE_PARITY_PROTOCOL_20261031.md"


def persist_runtime_receipt(output, result):
    """Publish post-execution runtime identity into the durable run report."""
    report_path = Path(output) / "run_report.json"
    need(report_path.is_file() and not report_path.is_symlink(), "run report")
    report = json.loads(report_path.read_bytes())
    need(report["status"] == result["status"], "report/result identity")
    report.update(host=result["host"], model_before=result["model_before"],
                  model_after=result["model_after"], online_events=result["online_events"])
    temporary = report_path.with_name(".run_report.json.runtime_tmp")
    need(not temporary.exists() and not temporary.is_symlink(), "exclusive runtime report")
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with temporary.open("xb") as stream:
        need(stream.write(payload) == len(payload), "complete runtime report")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, report_path)
    directory_fd = os.open(report_path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return sha(report_path)


def build():
    fixed = split.build()
    bindings = dict(fixed["bindings"])
    for path in (PROTOCOL, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": fixed["roster"], "bindings": bindings,
            "fixed_rows": 24, "P1_admission": False, "independent_confirmation": False}


def online_execute(original_execute, anchor, item, data, path, factory, entry, model, device, transform, events):
    import torch
    from . import preflight_pusht_d2p_p0 as p0

    base = physical.engine.h10.base
    original_capture = base.capture
    seen = []

    def capture(env, obs, phase, step, row, meter, store, reference=None):
        pixels = original_capture(env, obs, phase, step, row, meter, store, reference)
        if phase != "probe" or step != 5:
            return pixels
        receipt = row["trajectory"][-1]
        need(all(receipt[k] == item["midpoint"][k] for k in ("pixel_sha256", "state", "snapshot"))
             and hashlib.sha256(pixels.tobytes()).hexdigest() == receipt["pixel_sha256"], "midpoint identity")
        blind, raw = entry["row"], entry["action"]
        need(entry["index"] == item["index"] == anchor["index"]
             and raw[0].tobytes().hex() == item["raw10_float32_hex"][:80], "first packet identity")
        with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as image:
            cp_pixels = np.asarray(image.convert("RGB")).copy()
        torch.cuda.synchronize(device)
        start = time.monotonic()
        with torch.no_grad():
            def encode(p):
                return model.encode({"pixels": p0._image_tensor(p, transform=transform, device=device)})["emb"][0, 0]
            cp, mid = encode(cp_pixels), encode(pixels)
            states, history, goal, mean, std = planner.midpoint.feedback.second_context(blind, cp, mid, raw[0])
        prior = planner.midpoint.terminal.curve.planner.adam.prior
        starts = np.asarray(prior.initial_actions(blind["logical_pair"])[:, 1], dtype="<f4")
        need(starts.shape == (4, 5, 2), "initial actions")
        scores, actions = [], []
        for initial in starts:
            action = torch.nn.Parameter(torch.tensor(initial[None], dtype=torch.float32, device=device))
            optimizer = torch.optim.Adam([action], lr=0.05)

            def score():
                token = ((action - mean) / std).reshape(1, 1, 10)
                prediction = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))
                need(prediction.shape == states.shape and bool(torch.isfinite(prediction).all()), "prediction")
                return (prediction[:, -1] - goal).square().mean()

            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                score().backward()
                need(action.grad is not None and bool(torch.isfinite(action.grad).all()), "gradient")
                optimizer.step()
                with torch.no_grad():
                    action.clamp_(-1, 1)
            with torch.no_grad():
                scores.append(float(score()))
            actions.append(action.detach()[0].cpu().numpy().astype("<f4").copy())
        need(np.isfinite(scores).all(), "finite scores")
        winner = prior.pick(scores)
        chosen = actions[winner].tobytes()
        need(chosen == bytes.fromhex(item["raw10_float32_hex"])[40:], "online plan mismatch")
        torch.cuda.synchronize(device)
        event = {"anchor_index": anchor["index"], "role": item["role"], "pixel_sha256": receipt["pixel_sha256"],
                 "second_packet_sha256": hashlib.sha256(chosen).hexdigest(), "winner": winner,
                 "scores": scores, "synchronized_seconds": time.monotonic() - start,
                 "calls_code_path": {"encode": 2, "predict": 44, "action_encoder": 44, "backward": 40}}
        store.emit("online_midpoint_plan", **event)
        events.append(event)
        seen.append(True)
        return pixels

    base.capture = capture
    try:
        report = original_execute(anchor, item, data, path, factory)
    finally:
        base.capture = original_capture
    if len(report["row"]["responses"]) >= 5:
        need(seen == [True], "online plan absent")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--directory")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-cuda-and-physics", action="store_true")
    args = parser.parse_args()
    physical.NAME = physical.engine.NAME = NAME
    physical.build = build
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results" and
             out.name.startswith(NAME + "_prepare_") and not out.exists(), "output")
        result = {"sha256": physical.design.run._write_new(out, build())}
    elif args.command == "consume":
        need(out.is_absolute() and out.parent == ROOT / "results" and
             out.name.startswith(NAME + "_consume_") and not out.exists(), "output")
        value = physical.consume(args.directory, args.report_sha, args.prepared, args.sha256)
        result = {"sha256": physical.design.run._write_new(out, value), "rows": len(value["rows"])}
    else:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from . import preflight_pusht_d2p_p0 as p0

        need(args.authorize_cuda_and_physics and args.prepared and args.sha256
             and shutil.disk_usage(ROOT / "results").free >= 3 * 1024**3, "authorization and disk")
        all_entries, stopped, _ = planner.midpoint.inputs()
        entries = [x[0] for x in all_entries if x[0]["step"] == 10]
        need(len(entries) == 24 and not any(x["step"] == 10 for x in stopped)
             and [e["index"] for e in entries] == list(range(24)), "entry roster")
        base = planner.midpoint.terminal.curve.planner.adam.prior.base
        events = []
        with base.locking.gpu_lock(base.LOCK):
            host = require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
            before = base.model_hash(model)
            need(before == base.MODEL_HASH, "checkpoint")
            original_execute = physical.engine.execute
            physical.engine.execute = lambda anchor, item, data, path, factory: online_execute(
                original_execute, anchor, item, data, path, factory, entries[anchor["index"]],
                model, device, p0.image_transform(), events)
            try:
                with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
                    result = physical.execute(args.prepared, args.sha256, args.output)
            finally:
                physical.engine.execute = original_execute
            torch.cuda.synchronize(device)
            after = base.model_hash(model)
            need(before == after, "model drift")
            result.update(host=host, model_before=before, model_after=after,
                          online_events=len(events))
            result["sha256"] = persist_runtime_receipt(args.output, result)
            if result["status"] == "recorded_not_consumed":
                need(len(events) == 24, "online events")
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
