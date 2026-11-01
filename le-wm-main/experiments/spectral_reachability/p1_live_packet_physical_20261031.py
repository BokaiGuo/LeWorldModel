"""Execute a live-generated second packet on fixed known-root PushT inputs."""
import argparse
import base64
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_split_budget_feedback_physical_20261031 as split
from . import p1_live_packet_dispatch_20261031 as dispatch

ROOT, need, sha = split.ROOT, split.need, split.sha
physical, planner = split.physical, split.planner
h10 = physical.engine.h10
NAME = "p1_live_packet_physical_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_LIVE_PACKET_DISPATCH_PROTOCOL_20261031.md"


def build():
    prior = split.build()
    roster = []
    for item in prior["roster"]:
        # Keep only the first packet; never bind a precomputed second packet to dispatch.
        first = bytes.fromhex(item["raw10_float32_hex"])[:40]
        need(len(first) == 40, "first packet")
        roster.append({k: item[k] for k in ("index", "logical_pair", "root_slot", "condition", "arm",
                                            "role", "task_id", "prefix5_float32_hex",
                                            "cp5_file_sha256", "cp5_pixel_sha256")} |
                      {"first_packet_hex": first.hex(), "first_packet_sha256": hashlib.sha256(first).hexdigest()})
    bindings = dict(prior["bindings"])
    for path in (Path(__file__).resolve(), Path(dispatch.__file__).resolve(), PROTOCOL):
        bindings[str(path)] = sha(path)
    need(len(roster) == 24 and all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings, "fixed_rows": 24,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False}


def plan_second(row, first, pixels, model, device, transform):
    import torch
    from . import preflight_pusht_d2p_p0 as p0

    with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
        cp_pixels = np.asarray(image.convert("RGB")).copy()
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with torch.no_grad():
        def encode(p):
            return model.encode({"pixels": p0._image_tensor(p, transform=transform, device=device)})["emb"][0, 0]
        cp, mid = encode(cp_pixels), encode(pixels)
        states, history, goal, mean, std = planner.midpoint.feedback.second_context(row, cp, mid, first)
    prior = planner.midpoint.terminal.curve.planner.adam.prior
    starts = np.asarray(prior.initial_actions(row["logical_pair"])[:, 1], dtype="<f4")
    need(starts.shape == (4, 5, 2), "initialization")
    scores, candidates = [], []
    for initial in starts:
        action = torch.nn.Parameter(torch.tensor(initial[None], dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([action], lr=0.05)

        def score():
            token = ((action - mean) / std).reshape(1, 1, 10)
            pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))
            need(pred.shape == states.shape and bool(torch.isfinite(pred).all()), "prediction")
            return (pred[:, -1] - goal).square().mean()

        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            score().backward()
            need(action.grad is not None and bool(torch.isfinite(action.grad).all()), "gradient")
            optimizer.step()
            with torch.no_grad():
                action.clamp_(-1, 1)
        with torch.no_grad():
            scores.append(float(score()))
        candidates.append(action.detach()[0].cpu().numpy().astype("<f4").copy())
    need(np.isfinite(scores).all(), "finite scores")
    winner = prior.pick(scores)
    torch.cuda.synchronize(device)
    return candidates[winner], {"scores": scores, "winner": winner,
                                "synchronized_seconds": time.monotonic() - started,
                                "calls_code_path": {"encode": 2, "predict": 44, "action_encoder": 44, "backward": 40}}


def episode(env, root, source_plan, prefix_plan, anchor, item, row, meter, store, blind, model, device, transform):
    base, prior = h10.base, h10.prior
    trace = prior.TraceStore(store)
    body, packet = env.unwrapped, root["packets"][prefix_plan["condition"]]
    obs, native_goal = base.place(env, source_plan, root["slot"], meter, trace)
    frames = [base.capture(env, obs, "warmup", 0, row, meter, trace, root["warm"][0])]
    for step, command in enumerate(base.moving.warmup_commands(), 1):
        need(not body.eval_state(native_goal, base.source.vector(obs["state"]))[0], "warmup terminal")
        obs, info, terminated, truncated, reward, actual = base.dispatch(
            env, command, "warmup", step, row, meter, trace, native_goal)
        image = base.capture(env, obs, "warmup", step, row, meter, trace, root["warm"][step])
        base.paired.check_return(env, command, actual, info, native_goal, reward, terminated)
        need(not terminated and not truncated, "warmup stop")
        if step in (5, 10):
            frames.append(image)
    need(base.moving.exact(np.asarray(frames), packet["rgb_history"]) and
         base.moving.exact(packet["raw_action_history"], base.moving.warmup_commands()), "history")
    goal = base.source.vector(root["goal"])
    need(base.source.root_hash(base.source.vector(obs["state"]), goal) == root["root_sha256"], "root")
    base.bind_goal(body, goal, packet["goal_rgb"], meter, trace)
    bound = base.invariant(body)
    base.capture(env, obs, "plan", 0, row, meter, trace)
    need(not body.eval_state(goal, base.source.vector(obs["state"]))[0], "root terminal")
    for step, command in enumerate(base.previous.commands(prefix_plan["commands"])[:5], 1):
        obs, info, terminated, truncated, reward, actual = base.dispatch(
            env, command, "plan", step, row, meter, trace, goal)
        base.capture(env, obs, "plan", step, row, meter, trace)
        base.paired.check_return(env, command, actual, info, goal, reward, terminated)
        need(not terminated and not truncated, "prefix stop")
    prior.compare_trace(row, anchor, trace, meter)

    def check_bound(info=None):
        current = base.invariant(body)
        need({k: v for k, v in current.items() if k != "snapshot"} ==
             {k: v for k, v in bound.items() if k != "snapshot"}, "goal/renderer drift")
        if info is not None:
            need(base.moving.exact(np.asarray(info["goal"]), packet["goal_rgb"]), "goal RGB drift")

    check_bound()
    start = base.source.vector(obs["state"]).copy()
    row["start_state"], row["start_verified"] = start.tolist(), True
    first = np.frombuffer(bytes.fromhex(item["first_packet_hex"]), dtype="<f4").reshape(5, 2)
    stream = dispatch.LivePacketDispatch(first)
    planned = None
    probe = {"index": 0, "name": "live_generated_packet"}
    for step in range(1, 11):
        command = stream.next_action()
        obs, info, terminated, truncated, reward, actual = base.dispatch(
            env, command, "probe", step, row, meter, trace, goal)
        try:
            pixels = base.capture(env, obs, "probe", step, row, meter, trace)
        except base.moving.Rejected as exc:
            base.paired.check_return(env, command, actual, info, goal, reward, terminated)
            check_bound(info)
            need(planned is not None, "censored before live packet is available")
            evidence = h10.censored_proof(store, anchor, probe, row, exc)
            row["status"], row["censor"] = "censored_out_of_domain", evidence
            store.emit("censored_out_of_domain", **meter.where, proof=evidence)
            return planned
        base.paired.check_return(env, command, actual, info, goal, reward, terminated)
        check_bound(info)
        state = base.source.vector(obs["state"])
        delta = state - start
        row["responses"].append({"step": step, "state": state.tolist(), "delta_native7": delta.tolist(),
                                 "delta_angle_wrapped": float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                                 "terminated": terminated, "truncated": truncated})
        if terminated or truncated:
            row["status"] = "native_stopped"
            return planned
        if step == 5:
            second, plan_info = plan_second(blind, first, pixels, model, device, transform)
            digest = stream.bind_second(second)
            probe["raw10_float32_hex"] = first.tobytes().hex() + second.tobytes().hex()
            probe["raw10_sha256"] = hashlib.sha256(bytes.fromhex(probe["raw10_float32_hex"])).hexdigest()
            planned = {"anchor_index": anchor["index"], "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                       "second_packet_hex": second.tobytes().hex(), "second_packet_sha256": digest, **plan_info}
            store.emit("live_packet_plan", **planned)
    row["status"] = "probe_completed"
    return planned


def execute_row(anchor, item, data, path, factory, blind, model, device, transform):
    store = h10.ObservedStore(h10.Store(path))
    probe = {"index": 0, "name": "live_generated_packet"}
    row = h10.runner.row_for(anchor, probe)
    meter, errors, planned = h10.Meter(store), [], None
    meter.where = dict(branch=0, anchor_index=anchor["index"], condition=row["condition"],
                       arm=row["arm"], candidate_role=row["candidate_role"], probe=row["probe"])
    raw = np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]), dtype="<f4").reshape(5, 2)
    prefix_plan = {"condition": anchor["condition"], "commands": np.concatenate([raw, np.zeros((5, 2), np.float32)]).tolist()}
    env = None
    store.emit("run_started", anchor=anchor["index"], first_packet_sha256=item["first_packet_sha256"])
    try:
        env = meter.call("constructor", factory)
        h10.base.validate_env(env.unwrapped)
        need(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
             env.unwrapped.render_mode == "rgb_array", "renderer")
        with h10.base.native_hooks(env, meter, row["secondary_errors"]):
            planned = episode(env, data["roots"][row["root_slot"]], data["source_plan"], prefix_plan,
                              anchor, item, row, meter, store, blind, model, device, transform)
        need(not row["secondary_errors"], "hook cleanup")
    except BaseException as exc:
        row.update(status="engineering_failed", error=h10.base.error("branch", exc))
        errors.append(row["error"])
    finally:
        if env is not None:
            try:
                meter.call("close", env.close, cleanup=True)
            except BaseException as exc:
                errors.append(h10.base.error("close", exc))
    store.emit("branch_closed", branch=0, status=row["status"])
    status = "recorded_not_consumed" if not errors and row["status"] in (
        "probe_completed", "native_stopped", "censored_out_of_domain") else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = {"schema": NAME + ".row", "status": status, "row": row, "counts": meter.counts,
              "receipts": meter.receipts, "errors": errors, "live_plan": planned,
              "journal_chain_tail": store.previous, "journal_sha256": sha(path / "journal.jsonl"),
              "wall_seconds": time.monotonic() - meter.started, "P1_admission": False}
    store.put("report.json", report, final=True)
    return report


def run(prepared, digest, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    need(sha(prepared) == digest and json.loads(Path(prepared).read_bytes()) == build(), "prepared")
    fixed = build()
    out = Path(output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_run_")
         and not out.exists() and not out.is_symlink() and shutil.disk_usage(out.parent).free >= 3 * 1024**3, "output")
    entries, stopped, _ = planner.midpoint.inputs()
    entries = [x[0] for x in entries if x[0]["step"] == 10]
    need(len(entries) == 24 and not any(x["step"] == 10 for x in stopped), "entries")
    old = physical.design.uphill.runner.native.read(physical.design.PHYS / "report.json", cap=64 * 1024**2)
    binding = old["execution_binding"]
    _, data = physical.design.uphill.verify(binding["prepared"], binding["prepare_sha256"], physical.design.JOIN_SHA)
    anchors = json.loads(physical.bundle.PREP.read_bytes())["anchors"]
    physical.design.uphill.runner.native.constructor_contract()
    base = planner.midpoint.terminal.curve.planner.adam.prior.base
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "checkpoint")
        transform = p0.image_transform()
        out.mkdir(mode=0o700)

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

        shards, failure = [], None
        try:
            with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
                for i, (anchor, item, entry) in enumerate(zip(anchors, fixed["roster"], entries, strict=True)):
                    need(i == anchor["index"] == item["index"] == entry["index"], "row identity")
                    path = out / f"row_{i:02d}"
                    result = execute_row(anchor, item, data, path, factory, entry["row"], model, device, transform)
                    shards.append({"index": i, "status": result["status"], "report_sha256": sha(path / "report.json")})
                    print(json.dumps(shards[-1]), flush=True)
                    if result["status"] != "recorded_not_consumed":
                        failure = result["errors"]
                        break
            torch.cuda.synchronize(device)
            after = base.model_hash(model)
            need(before == after and all(sha(p) == h for p, h in fixed["bindings"].items()), "source/model drift")
        except BaseException as exc:
            after = base.model_hash(model)
            failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        status = "recorded_not_consumed" if len(shards) == 24 and failure is None else "partial_or_failed"
        result = {"schema": NAME + ".run", "status": status, "shards": shards, "failure": failure,
                  "host": host, "model_before": before, "model_after": after, "prepare_sha256": digest,
                  "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False}
        return {"status": status, "sha256": physical.design.run._write_new(out / "run_report.json", result)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authorize-cuda-and-physics", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if args.command == "prepare":
        need(output.is_absolute() and output.parent == ROOT / "results" and
             output.name.startswith(NAME + "_prepare_") and not output.exists(), "output")
        result = {"sha256": physical.design.run._write_new(output, build())}
    else:
        need(args.authorize_cuda_and_physics and args.prepared and args.sha256, "authorization")
        result = run(args.prepared, args.sha256, output)
    print(json.dumps(result), flush=True)
    raise SystemExit(int(result.get("status") == "partial_or_failed"))
