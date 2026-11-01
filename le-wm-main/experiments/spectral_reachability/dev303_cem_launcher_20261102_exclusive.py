"""Sealed dev303 exploratory CUDA/PushT full-horizon CEM launcher.

prepare is CPU-only; run requires two explicit authorizations and the prepare
digest. verify consumes saved artifacts only and does not execute a model or
environment. Neither a contract nor a successful verification proves fairness,
hardware authenticity, independence, or official baseline reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from . import dev303_full_cem_cp5_wiring_20261102_exclusive as cem
from . import dev303_native_host_bridge_20261102_exclusive as bridge
from . import p1_dev301_blind_plan_physical_20261101_codex as base
from . import p1_dev301_blind_plan_physical_saved_20261101_codex as saved
from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as source
from . import p1_dev303_blind_cuda_20261101_codex as blind

NAME = "dev303_cem_launcher_20261102_exclusive"
ROOT = cem.ROOT
CEM_SEED = 30320261102
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
require = base.require


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file binding")
    return blind.b.sha(path)


def contract_from_bindings(bindings, receipt_sha):
    return {
        "schema": NAME + ".prepare.v1", "classification": "dev303_exploratory",
        "seed": source.SEED, "cem_seed": CEM_SEED, "source_receipt_sha256": receipt_sha,
        "task_sha256": cem.TASK_SHA, "bindings": bindings,
        "protocol": cem.contract(), "initial_rgb_encodes": 4, "cp5_rgb_encodes_max": 1,
        "reset_max": 1, "warmup_max": 10, "control_max": 15,
        "model_action_encode_max": 325, "model_predict_max": 325,
        "no_retry_no_padding": True, "equal_budget_claim": False,
        "independent_confirmation": False, "P1_admission": False,
        "physical_execution_authorized_by_contract": False,
        "cuda_execution_authorized_by_contract": False,
        "source_closure_scope": "listed files only; transitive dependencies and hardware not authenticated",
        "training_membership_and_historical_independence": "unknown",
    }


def build():
    """Authenticate the existing root/metadata, without constructing an env."""
    arrays, receipt, _ = source.source_data()
    blind_contract = blind.build()
    task = cem.load_task()
    norm = blind_contract["normalization"]
    require(np.array_equal(np.asarray(task["action_mean"], np.float32),
                           np.asarray(norm["mean"], np.float32))
            and np.array_equal(np.asarray(task["action_std"], np.float32),
                               np.asarray(norm["std"], np.float32)), "normalizer")
    bindings = dict(blind_contract["bindings_sha256"])
    bindings.update(base.native.dependencies())
    paths = [Path(__file__), Path(bridge.__file__), Path(cem.__file__),
             Path(source.__file__), Path(base.__file__), Path(saved.__file__),
             Path(source.bridge.__file__), Path(source.source.__file__),
             source.SOURCE_DIR / "source_receipt.json",
             source.SOURCE_DIR / "reset_receipt.json", cem.TASK]
    for row in [*receipt["history"]["frames"], receipt["history"]["goal_rgb"]]:
        paths.append(source.SOURCE_DIR / row["file"])
    bindings.update({str(p.resolve()): sha(p) for p in paths})
    contract = contract_from_bindings(bindings, blind_contract["source_receipt_sha256"])
    contract.update(policy=blind_contract["policy"],
                    expected_model_hash=blind_contract["expected_model_hash"],
                    normalization=norm,
                    source_pixels=[base.pixel_sha(x) for x in
                                   [*arrays["rgb_history"], arrays["goal_rgb"]]])
    verify_bindings(contract)
    return contract


def verify_bindings(contract):
    require(type(contract["bindings"]) is dict and bool(contract["bindings"]), "binding roster")
    for path, expected in contract["bindings"].items():
        require(sha(path) == expected, "binding drift: " + path)


def write_new(path, value):
    raw = base.canonical(value) + b"\n"
    with Path(path).open("xb") as out:
        require(out.write(raw) == len(raw), "write")
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(raw).hexdigest()


def check_prepare(path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected,
            "external prepare SHA")
    contract = base.read(path)
    require(contract == build(), "prepare contract or source drift")
    return contract


def _setup_call(store, kind, callback):
    store.emit("setup_attempt", resource=kind)
    started = time.monotonic()
    try:
        result = callback()
    except BaseException as exc:
        store.emit("setup_failure", resource=kind, error=type(exc).__name__,
                   wall_seconds=time.monotonic() - started)
        raise
    store.emit("setup_return", resource=kind, wall_seconds=time.monotonic() - started)
    return result


def _warmup(env, store, arrays, receipt, reset):
    history = receipt["history"]
    checkpoints = {p["step"]: p for p in history["frames"]}
    goal = np.asarray(history["goal"], np.float64)
    obs, info = _setup_call(store, "reset", lambda: env.reset(seed=source.SEED))
    require(np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
            and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
            and np.array_equal(np.asarray(info["goal"]), arrays["goal_rgb"]), "reset identity")
    frames = []
    def capture(step):
        point = _setup_call(store, "warmup_render",
                            lambda: base.capture(store, env, "cem", "warmup", step, checkpoints[step]))
        pixels = base.image(store.path / point["file"])
        require(np.array_equal(pixels, arrays["rgb_history"][len(frames)]), "source RGB identity")
        frames.append(pixels)
    capture(0)
    for step in range(1, 11):
        action = np.zeros(2, dtype=np.float32)
        store.emit("warmup_dispatch", step=step, action=action.tolist())
        obs, reward, terminated, truncated, info = _setup_call(
            store, "warmup_step", lambda: env.step(action.copy()))
        actual = source._check_return(env, obs, info, goal, arrays["goal_rgb"], action)
        store.emit("warmup_return", step=step, action=actual.tolist(),
                   terminated=bool(terminated), truncated=bool(truncated))
        require(not terminated and not truncated, "warmup early stop")
        if step in checkpoints:
            capture(step)
    require(np.array_equal(np.asarray(obs["state"], np.float64),
                           np.asarray(history["final_state"], np.float64)), "root state")
    return frames, goal


def execute(prepared, expected, output):
    """Requires caller to hold the cooperative lock and explicit authorizations."""
    contract = check_prepare(prepared, expected)
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(Path(prepared).resolve()),
                                          "prepare_sha256": expected})
    store.emit("run_started", prepare_sha256=expected)
    errors, model, env, model_before, model_after, result = [], None, None, None, None, None
    started = time.monotonic()
    try:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from stable_worldmodel.envs.pusht.env import PushT

        require(sha(inspect.getsourcefile(PushT)) == base.native.ENV_SHA, "native source")
        gpu = _setup_call(store, "cuda_preflight", lambda: require_cuda_allocation("cuda:0"))
        store.emit("cuda_preflight_metadata", metadata=gpu)
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = _setup_call(store, "model_load", lambda: load_compatible_pretrained(
            contract["policy"], device=device).eval().requires_grad_(False))
        blind.b.frozen(model)
        model_before = _setup_call(store, "model_hash", lambda: blind.b.tensor_hash(model))
        require(model_before == contract["expected_model_hash"], "model tensor identity")
        env = _setup_call(store, "env_create", lambda: PushT(
            render_mode="rgb_array", resolution=224, with_target=False))
        arrays, receipt, reset = source.source_data()
        frames, goal_state = _warmup(env, store, arrays, receipt, reset)
        goal_row = store.frame("cem_goal.png", arrays["goal_rgb"])
        store.emit("goal_frame", **goal_row)

        def emit(event):
            store.check()
            store.emit("bridge", event=event)

        capture, encode, step = bridge.native_callbacks(
            env, store, model, device, goal_state, arrays["goal_rgb"], emit)
        values = []
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for role, pixels in zip(("observed0", "observed5", "observed10", "goal"),
                                    [*frames, arrays["goal_rgb"]], strict=True):
                def encode_synchronized():
                    value = encode(pixels)
                    torch.cuda.synchronize(device)
                    return value
                latent = _setup_call(store, "initial_rgb_encode", encode_synchronized)
                require(latent.device.type == "cuda" and latent.dtype == torch.float32
                        and tuple(latent.shape) == (192,) and bool(torch.isfinite(latent).all()),
                        "initial latent")
                values.append(latent)
                name = "latent_" + role + ".json"
                digest = store.json(name, {"role": role, "pixel_sha256": base.pixel_sha(pixels),
                                           "values": latent.cpu().tolist()})
                store.emit("initial_latent_saved", role=role, file=name, sha256_file=digest,
                           pixel_sha256=base.pixel_sha(pixels))
            task = cem.load_task()
            mean = torch.tensor(task["action_mean"], dtype=torch.float32, device=device)
            std = torch.tensor(task["action_std"], dtype=torch.float32, device=device)
            def encode_cp5(pixels):
                latent = encode(pixels)
                torch.cuda.synchronize(device)
                require(latent.device.type == "cuda" and latent.dtype == torch.float32
                        and tuple(latent.shape) == (192,) and bool(torch.isfinite(latent).all()),
                        "CP5 latent")
                digest = store.json("latent_cp5.json", {"role": "cp5",
                                    "pixel_sha256": base.pixel_sha(pixels),
                                    "values": latent.cpu().tolist()})
                store.emit("cp5_latent_saved", file="latent_cp5.json", sha256_file=digest,
                           pixel_sha256=base.pixel_sha(pixels))
                return latent
            host = bridge.Host(model, encode_cp5, capture, step, values[:3], values[3],
                               np.asarray(arrays["raw_action_history"], np.float32).reshape(10, 2),
                               mean, std, emit, base.pixel_sha(frames[-1]))
            result = host.run(seed=CEM_SEED)
            torch.cuda.synchronize(device)
        state = base.native.snapshot(env)["state"]
        store.emit("final_metrics", metrics=base.metrics(env, goal_state, state))
    except BaseException as exc:
        errors.append({"stage": "execution", "type": type(exc).__name__, "message": str(exc)[:1000]})
        store.emit("execution_error", error=errors[-1])
    finally:
        if env is not None:
            try:
                _setup_call(store, "env_close", env.close)
            except BaseException as exc:
                errors.append({"stage": "close", "type": type(exc).__name__, "message": str(exc)[:1000]})
        if model is not None:
            try:
                blind.b.frozen(model)
                model_after = _setup_call(store, "model_hash", lambda: blind.b.tensor_hash(model))
                require(model_after == model_before == contract["expected_model_hash"], "model changed")
            except BaseException as exc:
                errors.append({"stage": "model_closure", "type": type(exc).__name__, "message": str(exc)[:1000]})
        try:
            verify_bindings(contract)
        except BaseException as exc:
            errors.append({"stage": "source_closure", "type": type(exc).__name__, "message": str(exc)[:1000]})
    count = sum(row["kind"] == "bridge" and row["event"]["type"] == "control_return"
                for row in store.events)
    status = ("partial_or_failed" if errors or result is None else
              "horizon_completed" if count == 15 else "native_early_stop")
    store.emit("run_finished", status=status, errors=errors)
    report = {"schema": NAME + ".report.v1", "status": status, "errors": errors,
              "prepare_sha256": expected, "control_returned": count,
              "model_hash_before": model_before, "model_hash_after": model_after,
              "wall_seconds": time.monotonic() - started,
              "journal_sha256": sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"],
              "P1_admission": False, "independent_confirmation": False,
              "equal_budget_claim": False, "hardware_authentication": "not_authenticated",
              "internal_model_costs_flops_energy": "not_measured",
              "bridge_model_call_wall_time": "host_unsynchronized_not_cuda_kernel_time"}
    digest = store.json("report.json", report)
    return report, digest


def load_journal(path, expected_sha, tail):
    require(sha(path) == expected_sha, "journal SHA")
    return saved.journal(path, tail)


def costs_from_events(events):
    counts = {}
    for e in events:
        if e["type"] in ("cost_attempt", "cost_return", "cost_failure"):
            row = counts.setdefault(e["kind"], {"attempted": 0, "returned": 0, "failed": 0})
            row[{"cost_attempt": "attempted", "cost_return": "returned",
                 "cost_failure": "failed"}[e["type"]]] += 1
    return counts


def verify_optimizer(events, horizon, seed):
    scores = iter([e["value"] for e in events if e["type"] == "score_return"] +
                  [e["score"] for e in events if e["type"] == "replay_return"])
    regenerated = []
    try:
        cem.optimize(horizon, lambda _: next(scores), seed=seed, emit=regenerated.append)
    except (StopIteration, ValueError) as exc:
        raise ValueError("optimizer replay") from exc
    require(regenerated == events and next(scores, None) is None, "optimizer events")


def verify_sequence(rows, seed):
    require(rows and rows[0]["kind"] == "run_started" and rows[-1]["kind"] == "run_finished"
            and rows[-1]["status"] in ("horizon_completed", "native_early_stop"), "complete run required")
    events = [r["event"] for r in rows if r["kind"] == "bridge"]
    accepted = {"bank", "score_attempt", "score_return", "update", "selection",
                "replay_attempt", "replay_return"}
    positions = [i for i, e in enumerate(events) if e["type"] == "observation_return"]
    require(len(positions) in (1, 2), "feedback nodes")
    dispatches = [e for e in events if e["type"] == "control_dispatch"]
    returns = [e for e in events if e["type"] == "control_return"]
    require(len(dispatches) == len(returns) and 1 <= len(returns) <= 15, "control counts")
    require([e["step"] for e in dispatches] == list(range(1, len(returns) + 1))
            and all(a["step"] == b["step"] and a["action"] == b["action"]
                    for a, b in zip(dispatches, returns, strict=True)), "control receipts")
    require(not any(e["stopped"] for e in returns[:-1])
            and ((len(returns) == 15 and not returns[-1]["stopped"])
                 or returns[-1]["stopped"]), "stop semantics")
    require(all(events.index(a) < events.index(b)
                and (i + 1 == len(dispatches) or events.index(b) < events.index(dispatches[i + 1]))
                for i, (a, b) in enumerate(zip(dispatches, returns, strict=True))),
            "control event order")
    require([events[pos]["step"] for pos in positions] == ([0, 5] if len(positions) == 2 else [0]),
            "observation steps")
    for node, pos in enumerate(positions):
        end = next(i for i in range(pos + 1, len(events)) if events[i]["type"] == "dispatch_attempt")
        node_events = [e for e in events[pos + 1:end] if e["type"] in accepted]
        verify_optimizer(node_events, (3, 2)[node], seed + node)
        selection = next(e for e in node_events if e["type"] == "selection")
        expected = np.asarray(selection["raw"], np.float32).reshape(-1, 2)[:(5, 10)[node]]
        command = np.asarray(events[end]["actions"], np.float32)
        require(np.array_equal(expected, command), "selected command")
        actual = dispatches[:5] if node == 0 else dispatches[5:]
        require(np.array_equal(np.asarray([e["action"] for e in actual], np.float32),
                               command[:len(actual)])
                and (not actual or end < events.index(actual[0])), "issued selected command")
    if len(positions) == 2:
        feedback = [e for e in events if e["type"] == "feedback"]
        require(len(feedback) == 1 and feedback[0]["step"] == 5
                and feedback[0]["actual_actions"] == [e["action"] for e in returns[:5]]
                and events.index(returns[4]) < events.index(feedback[0]) < positions[1],
                "actual CP5 feedback")
    else:
        require(len(returns) <= 5 and returns[-1]["stopped"], "missing CP5")
    counts = costs_from_events(events)
    expected_calls = 195 + (130 if len(positions) == 2 else 0)
    for kind in ("action_encode", "predict"):
        require(counts.get(kind) == {"attempted": expected_calls, "returned": expected_calls, "failed": 0},
                "model accounting")
    require(counts.get("control") == {"attempted": len(returns), "returned": len(returns), "failed": 0},
            "control accounting")
    for kind in ("render", "rgb_encode"):
        expected_feedback = len(positions) - 1
        require(counts.get(kind, {"attempted": 0, "returned": 0, "failed": 0})
                == {"attempted": expected_feedback, "returned": expected_feedback, "failed": 0},
                "feedback accounting")
    return {"control_returned": len(returns), "nodes": len(positions), "costs": counts}


def verify(prepared, expected, directory, report_sha):
    contract = check_prepare(prepared, expected)
    directory = Path(directory).resolve(strict=True)
    report_path = directory / "report.json"
    require(sha(report_path) == report_sha, "external report SHA")
    report = base.read(report_path)
    require(report["schema"] == NAME + ".report.v1" and report["prepare_sha256"] == expected
            and report["errors"] == [] and report["equal_budget_claim"] is False
            and report["independent_confirmation"] is False and report["P1_admission"] is False
            and report["model_hash_before"] == report["model_hash_after"] == contract["expected_model_hash"],
            "report boundary")
    require(base.read(directory / "execution_binding.json") == {
        "prepared": str(Path(prepared).resolve()), "prepare_sha256": expected}, "execution binding")
    rows = load_journal(directory / "journal.jsonl", report["journal_sha256"], report["journal_tail"])
    require(rows[0]["prepare_sha256"] == expected and rows[-1]["status"] == report["status"],
            "journal binding/status")
    result = verify_sequence(rows, contract["cem_seed"])
    require(result["control_returned"] == report["control_returned"], "report action count")
    warm_dispatch = [r for r in rows if r["kind"] == "warmup_dispatch"]
    warm_return = [r for r in rows if r["kind"] == "warmup_return"]
    require(len(warm_dispatch) == len(warm_return) == 10, "warmup receipts")
    for step, (dispatch, returned) in enumerate(zip(warm_dispatch, warm_return, strict=True), 1):
        require(dispatch["step"] == returned["step"] == step
                and dispatch["action"] == returned["action"] == [0.0, 0.0]
                and not returned["terminated"] and not returned["truncated"]
                and dispatch["index"] < returned["index"]
                and (step == 10 or returned["index"] < warm_dispatch[step]["index"]),
                "warmup action order")
    observations = [r for r in rows if r["kind"] == "observation"]
    warm = [r for r in observations if r["phase"] == "warmup"]
    require([r["step"] for r in warm] == [0, 5, 10], "warmup frames")
    _, receipt, _ = source.source_data()
    for point, reference in zip(warm, receipt["history"]["frames"], strict=True):
        require(all(point[k] == reference[k] for k in ("state", "snapshot", "pixel_sha256")),
                "warmup source identity")
        saved.frame(directory, point)
    require(warm[0]["index"] < warm_dispatch[0]["index"]
            and warm_return[4]["index"] < warm[1]["index"] < warm_dispatch[5]["index"]
            and warm_return[9]["index"] < warm[2]["index"], "warmup frame order")
    goal = next(r for r in rows if r["kind"] == "goal_frame")
    require(goal["file"] == "cem_goal.png"
            and sha(directory / goal["file"]) == goal["file_sha256"]
            and base.pixel_sha(base.image(directory / goal["file"])) == contract["source_pixels"][3],
            "goal frame")
    initial = [r for r in rows if r["kind"] == "initial_latent_saved"]
    require([r["role"] for r in initial] == ["observed0", "observed5", "observed10", "goal"],
            "initial encodings")
    for i, row in enumerate(initial):
        require(row["file"] == "latent_" + row["role"] + ".json"
                and sha(directory / row["file"]) == row["sha256_file"]
                and row["pixel_sha256"] == contract["source_pixels"][i], "latent file")
        latent = base.read(directory / row["file"])
        require(latent["role"] == row["role"] and latent["pixel_sha256"] == row["pixel_sha256"]
                and np.asarray(latent["values"]).shape == (192,)
                and np.isfinite(latent["values"]).all(), "latent contents")
    first_observation = next(r for r in rows if r["kind"] == "bridge"
                             and r["event"]["type"] == "observation_attempt")
    require(warm[2]["index"] < initial[0]["index"] and
            all(a["index"] < b["index"] for a, b in zip(initial, initial[1:]))
            and initial[-1]["index"] < first_observation["index"], "initial encoding order")
    cp = [r for r in observations if r["phase"] == "control"]
    require(len(cp) == (1 if result["nodes"] == 2 else 0), "CP5 frame count")
    for point in cp:
        require(point["step"] == 5, "CP5 frame")
        saved.frame(directory, point)
        feedback = next(r["event"] for r in rows if r["kind"] == "bridge" and r["event"]["type"] == "feedback")
        require(point["pixel_sha256"] == feedback["rgb_sha256"], "CP5 pixel binding")
        latent_events = [r for r in rows if r["kind"] == "cp5_latent_saved"]
        require(len(latent_events) == 1 and latent_events[0]["file"] == "latent_cp5.json"
                and latent_events[0]["pixel_sha256"] == point["pixel_sha256"]
                and sha(directory / "latent_cp5.json") == latent_events[0]["sha256_file"],
                "CP5 latent binding")
        latent = base.read(directory / "latent_cp5.json")
        require(latent["role"] == "cp5" and latent["pixel_sha256"] == point["pixel_sha256"]
                and np.asarray(latent["values"]).shape == (192,)
                and np.isfinite(latent["values"]).all(), "CP5 latent contents")
        fifth = next(r for r in rows if r["kind"] == "bridge"
                     and r["event"]["type"] == "control_return" and r["event"]["step"] == 5)
        feedback_row = next(r for r in rows if r["kind"] == "bridge"
                            and r["event"]["type"] == "feedback")
        require(fifth["index"] < point["index"] < latent_events[0]["index"] < feedback_row["index"],
                "CP5 capture order")
    if not cp:
        require(not any(r["kind"] == "cp5_latent_saved" for r in rows), "unexpected CP5 latent")
    for resource, count in (("reset", 1), ("warmup_step", 10), ("warmup_render", 3),
                            ("initial_rgb_encode", 4)):
        require(sum(r["kind"] == "setup_attempt" and r["resource"] == resource for r in rows) == count
                and sum(r["kind"] == "setup_return" and r["resource"] == resource for r in rows) == count,
                "setup accounting")
    return {"status": "saved_chain_verified_development_only", **result,
            "model_predictions_recomputed": False, "hardware_authentication": "not_authenticated",
            "source_closure": contract["source_closure_scope"], "equal_budget_claim": False,
            "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--report-sha256")
    parser.add_argument("--authorize-development-physics", action="store_true")
    parser.add_argument("--authorize-model-cuda", action="store_true")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "ASCII alphanumeric tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.prepare_sha256 and not args.report_sha256
                and not args.authorize_development_physics and not args.authorize_model_cuda
                and not os.path.lexists(output), "prepare only")
        print(json.dumps({"prepared": str(prepared), "sha256": write_new(prepared, build())}))
    elif args.command == "run":
        require(args.prepare_sha256 and not args.report_sha256 and args.authorize_development_physics
                and args.authorize_model_cuda, "separate explicit CUDA and physical authorization")
        with base.locking.gpu_lock(LOCK):
            report, digest = execute(prepared, args.prepare_sha256, output)
        print(json.dumps({"status": report["status"], "report_sha256": digest, "output": str(output)}))
    else:
        require(args.prepare_sha256 and args.report_sha256 and not args.authorize_development_physics
                and not args.authorize_model_cuda, "verify inputs")
        print(json.dumps(verify(prepared, args.prepare_sha256, output, args.report_sha256)))


if __name__ == "__main__":
    main()
