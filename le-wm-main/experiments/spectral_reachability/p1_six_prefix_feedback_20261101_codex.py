"""Prospective six-prefix actual-RGB H2/H1 feedback continuation."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import p1_coherent_own_suffix_20261101_codex as own
from . import p1_coherent_own_physical_20261101_main as five
from . import p1_coherent_samebank_own_physical_20261101_codex as single
from . import p1_carry_binding_20261101 as carry

NAME = "p1_six_prefix_feedback_20261101_codex"
ROOT = own.ROOT
DOC = ROOT.parent / "docs/D2P_P1_SIX_PREFIX_FEEDBACK_20261101_CODEX.md"
PREFIXES = (*own.PREFIXES, "same_bank_greedy")
FIVE_PLAN = ROOT / "results/p1_coherent_own_suffix_20261101_codex_RUN_MAIN01"
FIVE_PLAN_SHA = "e14a4e701ed4983e48801b2ed813f9daf8c0373db1771af4415118690b19cdc7"
ONE_PLAN = ROOT / "results/p1_coherent_own_suffix_samebank_20261101_codex_RUN_MAIN01"
ONE_PLAN_SHA = "7402564db52792ed3a7027b63aa7328d6c9c7b59b0e7ca881a000a16515919e0"
FIVE_PHYS = ROOT / "results/p1_coherent_own_physical_20261101_main_RUN_MAIN01/report.json"
FIVE_PHYS_SHA = "46a4970a0248486495002b70135a231984f25ddbc4936ab5f447abb2ede9206f"
ONE_PHYS = ROOT / "results/p1_coherent_samebank_own_physical_20261101_codex_RUN_MAIN01/report.json"
ONE_PHYS_SHA = "8ed6bf4cc60a33204908464e3e3a7a69bb7b4f85200d86abcd07e31e112e3501"
AUTH = "MAIN_AUTHORIZES_SIX_PREFIX_FEEDBACK_HOST_CUDA_AND_PHYSICAL"
spec, require, sha = own.spec, own.require, own.sha
h4 = own.coherent.centered.h4
common = five.physical.common


def check_sources(sources):
    require(all(sha(p) == digest for p, digest in sources.items()), "source drift")


def read_physical(path, expected, schedules):
    require(sha(path) == expected, "physical report SHA")
    report = json.loads(path.read_bytes())
    protocol_path = path.parent / "protocol.json"
    require(report["neural_calls"] == 0 and report["P1_admission"] is False
            and report["independent_confirmation"] is False
            and sha(protocol_path) == report["protocol_sha256"], "physical report protocol")
    check_sources(report["sources"])
    require([r["prefix"] for r in report["rows"]] == [s["prefix"] for s in schedules],
            "physical roster")
    rows, sources = [], dict(report["sources"])
    sources.update({str(path): expected, str(protocol_path): report["protocol_sha256"]})
    for row, schedule in zip(report["rows"], schedules, strict=True):
        require(row["status"] == "completed20" and not row.get("error")
                and len(row["executed_raw"]) == 20, "historical complete physical branch")
        require(np.array_equal(np.asarray(row["executed_raw"][:10], np.float32),
                               np.asarray(schedule["commands"][:10], np.float32)),
                "historical returned first ten")
        checkpoints = []
        for step in (0, 5, 10):
            frames = [f for f in row["checkpoints"] if f["step"] == step]
            require(len(frames) == 1, "unique historical checkpoint")
            frame = frames[0]
            image = Path(frame["png"])
            require(image.parent == path.parent and not image.is_symlink()
                    and sha(image) == frame["file_sha256"], "checkpoint PNG")
            with Image.open(image) as im:
                pixels = np.asarray(im).copy()
            require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                    and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"],
                    "checkpoint pixel SHA")
            state = np.asarray(frame["state"], np.float64)
            require(state.shape == (7,) and np.isfinite(state).all(), "checkpoint state")
            checkpoints.append(dict(step=step, state=state.tolist(), pixel_sha256=frame["pixel_sha256"],
                                    png=str(image), file_sha256=frame["file_sha256"]))
            sources[str(image)] = frame["file_sha256"]
        require(checkpoints[1]["pixel_sha256"] == schedule["cp5_pixel_sha256"]
                and spec.digest(checkpoints[1]["state"]) == schedule["cp5_state_identity_sha256"],
                "historical cp5 plan identity")
        rows.append(dict(prefix=row["prefix"], checkpoints=checkpoints))
    return rows, sources


def build(runner):
    schedules5, sources5 = five.collect(FIVE_PLAN, FIVE_PLAN_SHA)
    # The sealed single-plan checker normally configures its own fresh process.
    # This process already configured the identical runner once.
    with patch.object(own, "configure", return_value=(None, runner)):
        schedule1, sources1 = single.collect(ONE_PLAN, ONE_PLAN_SHA)
    physical5, ps5 = read_physical(FIVE_PHYS, FIVE_PHYS_SHA, schedules5)
    physical1, ps1 = read_physical(ONE_PHYS, ONE_PHYS_SHA, schedule1)
    schedules = schedules5 + schedule1
    physical = physical5 + physical1
    reports = [json.loads((d / "report.json").read_bytes()) for d in (FIVE_PLAN, ONE_PLAN)]
    bindings = [json.loads((d / "binding.json").read_bytes()) for d in (FIVE_PLAN, ONE_PLAN)]
    require(bindings[0]["base_task"] == bindings[1]["base_task"]
            and bindings[0]["expected_model_hash"] == bindings[1]["expected_model_hash"]
            and bindings[0]["model_policy"] == bindings[1]["model_policy"], "same task and model")
    plan_rows = reports[0]["rows"] + reports[1]["rows"]
    require(tuple(r["prefix"] for r in plan_rows) == PREFIXES
            and tuple(s["prefix"] for s in schedules) == PREFIXES
            and tuple(r["prefix"] for r in physical) == PREFIXES, "fixed six roster")
    rows = []
    for plan_row, schedule, actual in zip(plan_rows, schedules, physical, strict=True):
        previous = np.asarray(plan_row["selected_suffix"], np.float32)
        commands = np.asarray(schedule["commands"], np.float32)
        require(previous.shape == (3, 5, 2) and commands.shape == (20, 2)
                and np.array_equal(previous.reshape(15, 2), commands[5:]), "old selected plan")
        rows.append(dict(prefix=plan_row["prefix"], first10=commands[:10].tolist(),
                         previous_cp5_plan=previous.tolist(), cp5_task=plan_row["task"],
                         checkpoints=actual["checkpoints"]))
    sources = {}
    for source in (sources5, sources1, ps5, ps1):
        for path, digest in source.items():
            require(path not in sources or sources[path] == digest, "conflicting source hashes")
            sources[path] = digest
    for item in (Path(__file__).resolve(), DOC, Path(carry.__file__).resolve(),
                 Path(five.__file__).resolve(), Path(single.__file__).resolve(),
                 Path(own.durable.__file__).resolve(), Path(own.leafmod.__file__).resolve(),
                 Path(own.consumer.__file__).resolve(), Path(own.journal.__file__).resolve(),
                 Path(checkpoint_compat.__file__).resolve(), Path(gpu_preflight.__file__).resolve()):
        sources[str(item)] = sha(item)
    check_sources(sources)
    return dict(schema=NAME + ".prepare.v1", rows=rows, sources=sources,
                base_task=bindings[0]["base_task"], model_policy=bindings[0]["model_policy"],
                model_sha256=bindings[0]["model_sha256"],
                expected_model_hash=bindings[0]["expected_model_hash"],
                node_cost={str(b): h4.h4.node_cost("structured", b) for b in (10, 15)},
                encode_cost=dict(encode_calls=1, encode_samples=1),
                protocol="fixed_six_known_root_actual_rgb_feedback",
                P1_admission=False, independent_confirmation=False)


def make_task(cp5_task, history_latents, raw10, pixel_sha, previous, boundary, runner):
    require(boundary in (10, 15), "feedback boundary")
    z = np.asarray(history_latents, np.float32)
    raw = np.asarray(raw10, np.float32)
    old = np.asarray(previous, np.float32)
    horizon = h4.h4.dimensions(boundary)[0]
    require(z.shape == (3, 192) and np.isfinite(z).all() and raw.shape == (10, 2)
            and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
            and old.shape == (horizon + 1, 5, 2)
            and np.array_equal(old[0], raw[-5:]) and spec.is_hash(pixel_sha), "actual feedback inputs")
    task = deepcopy(cp5_task)
    task["root_context"]["latent_history"] = z.tolist()
    task["root_context"]["action_history"] = runner.rgb.moving.normalized(task, raw)
    identity = spec.digest([NAME, boundary, pixel_sha, raw.tolist(), old.tolist(), z.tolist()])
    binding = task["binding"]
    binding["context_sha256"] = spec.digest(task["root_context"])
    binding["root_sha256"] = identity
    binding["task_id"] = spec.digest([NAME, binding["task_id"], identity])
    binding["task_sha256"] = spec.digest([NAME, "task", binding["task_id"]])
    binding["task_contract_sha256"] = spec.digest([NAME, "contract", binding["task_id"], old.tolist()])
    runner.p3.core.validate_task(task)
    return task


def contracts(task, model_sha, boundary):
    control = h4.reverse.Q.carry.control
    cost = h4.h4.node_cost("structured", boundary)
    with patch.dict(control.base.CONFIG, h4.boundary_config(boundary)), \
            patch.object(control, "cost", lambda arm: h4.h4.node_cost(arm, boundary)):
        policy, contract, config = control.contract_for(task, model_sha, spec.digest({"device_policy": "host_cuda_0"}))
    require(policy["horizon"] == h4.h4.dimensions(boundary)[0]
            and config.items() >= h4.boundary_config(boundary).items()
            and config["steps"] == 6 and config["learning_rate"] == .05
            and all(contract["levels"][0]["caps"][axis] >= cost[axis] for axis in spec.AXES),
            "boundary contract")
    require(np.array_equal(np.asarray(task["projection"], np.float64), np.eye(192)), "identity projection")
    encoding = deepcopy(contract)
    caps = dict.fromkeys(spec.AXES, 0)
    caps.update(encode_calls=1, encode_samples=1)
    encoding["levels"] = [dict(id="node", caps=caps),
                          dict(id="unused_larger", caps={k: v + 1 for k, v in caps.items()})]
    spec.preflight(encoding, spec.digest(encoding))
    return policy, contract, config, encoding


def exclusive(value):
    path = Path(value)
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
            "exclusive output")
    return path


def account(driver, contract, label, expected):
    ledger = driver.finish()
    actual = own.consumer.consume(ledger["journal"], ledger["sha256"], spec.digest(contract),
                                  "structured", "node", label)
    require(actual["status"] == "closed_unverified" and actual["charged_upper_bounds"] == expected,
            "measured durable charges")
    return ledger, actual


def guard_driver(driver, check_limits):
    dispatch = driver.dispatch

    def checked(*args, **kwargs):
        try:
            check_limits()
        except BaseException:
            if not driver.failed and not driver.closed:
                driver.failed = True
                driver._emit("invalid", {})
            raise
        return dispatch(*args, **kwargs)

    driver.dispatch = checked


def checkpoint(env, output, name, step, goal):
    state = np.asarray(env.unwrapped._get_obs()).copy()
    pixels = np.asarray(env.render()).copy()
    require(state.shape == (7,) and np.isfinite(state).all()
            and pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8, "actual checkpoint")
    path = output / f"{name}_{step:02}.png"
    Image.fromarray(pixels).save(path)
    native = common.metrics.qualification.native(goal, state)
    metrics = dict(**common.metrics.metrics(state, goal), joint_xy=native["joint_xy_error"],
                   native_success=native["initial_success"])
    return dict(step=step, state=state.tolist(), metrics=metrics,
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                png=str(path), file_sha256=sha(path)), pixels


def check_checkpoint(actual, expected):
    require(actual["step"] == expected["step"]
            and np.array_equal(np.asarray(actual["state"]), np.asarray(expected["state"]))
            and actual["pixel_sha256"] == expected["pixel_sha256"], "replayed checkpoint identity")


def packet(env, raw, row, goal):
    for command in np.asarray(raw, np.float32):
        obs, _, stop, trunc, info = env.step(command.copy())
        require(np.array_equal(env.unwrapped.latest_action, command)
                and np.array_equal(info["goal_state"], goal), "returned action/goal")
        row["executed_raw"].append(command.tolist())
        state = np.asarray(obs["state"])
        require(state.shape == (7,) and np.isfinite(state).all(), "physical state")
        outside = not bool(np.all((state[:4] >= 0) & (state[:4] <= 512)))
        if stop or trunc or outside:
            native = common.metrics.qualification.native(goal, state)
            return dict(step=len(row["executed_raw"]), terminated=bool(stop),
                        truncated=bool(trunc), outside=outside,
                        native_success=bool(native["initial_success"]),
                        status="native_stopped" if stop or trunc else "canvas_censored")
    return None


def run_branch(slot, value, output, model, device, matrix, runner, check_limits):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    name = slot["prefix"]
    row = dict(prefix=name, status="failed", stop_reason=None,
               executed_raw=[], checkpoints=[], nodes=[], error=None)
    folder = output / name
    folder.mkdir()
    env = None
    try:
        goal, images, _ = common.target.collect()
        root = json.loads((common.target.DIRECTORY / "report.json").read_bytes())
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                       with_target=False, render_action=False)
        body = env.unwrapped
        env.reset(seed=root["reset_seed"])
        env.reset(seed=0, options=dict(state=np.asarray(root["requested_reset"]),
                                      goal_state=np.asarray(root["placeholder_goal"])))
        for step in range(11):
            if step:
                action = np.asarray(root["raw_history"][step-1], np.float32)
                _, _, stop, trunc, _ = env.step(action.copy())
                require(not stop and not trunc and np.array_equal(body.latest_action, action), "warmup")
            if step in (0, 5, 10):
                require(np.array_equal(env.render(), images[step//5]), "warmup RGB")
        common.native_source.attach_goal(body, np.asarray(goal["goal_state"]), images[-1])
        cp, _ = checkpoint(env, folder, name, 0, goal["goal_state"])
        check_checkpoint(cp, slot["checkpoints"][0])
        row["checkpoints"].append(cp)
        first10 = np.asarray(slot["first10"], np.float32)
        for i in (0, 5):
            status = packet(env, first10[i:i+5], row, np.asarray(goal["goal_state"]))
            cp, pixels = checkpoint(env, folder, name, len(row["executed_raw"]), goal["goal_state"])
            row["checkpoints"].append(cp)
            if status:
                row["status"] = status["status"]
                row["stop_reason"] = status
                return row
            check_checkpoint(cp, slot["checkpoints"][1 + i//5])
        previous = carry.BoundPlan.seal(name, 5, np.asarray(slot["previous_cp5_plan"], np.float32))
        latents = deepcopy(slot["cp5_task"]["root_context"]["latent_history"])
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        for boundary in (10, 15):
            check_limits()
            issued = np.asarray(row["executed_raw"][-5:], np.float32)
            feedback = previous.bind_feedback(episode=name, boundary=boundary,
                                               issued_packet=issued, rgb_sha256=cp["pixel_sha256"])
            raw10 = np.asarray(row["executed_raw"][-10:], np.float32)
            # Contract identity for the encoder does not depend on the unobserved latent.
            placeholder = make_task(slot["cp5_task"], latents[-2:] + [[0.] * 192], raw10,
                                    cp["pixel_sha256"], previous.plan, boundary, runner)
            _, _, _, ec = contracts(placeholder, value["model_sha256"], boundary)
            enc_label = f"{name}_b{boundary}_encode"
            ed = own.durable.Driver(folder / f"b{boundary}_encode", ec, spec.digest(ec),
                                    "structured", "node", enc_label)
            guard_driver(ed, check_limits)
            try:
                leaf = own.leafmod.LeafAdapter(model, ed, ec, spec.digest(ec))
                image = own.p0._image_tensor(pixels, transform=own.p0.image_transform(), device=device)
                with torch.inference_mode():
                    z = leaf.encode({"pixels": image}, phase="setup")["emb"][0, 0].detach().cpu().numpy()
            finally:
                encode_cost = dict.fromkeys(spec.AXES, 0)
                encode_cost.update(encode_calls=1, encode_samples=1)
                enc_ledger, enc_account = account(ed, ec, enc_label, encode_cost)
            latents = latents[-2:] + [z.tolist()]
            task = make_task(slot["cp5_task"], latents, raw10, cp["pixel_sha256"],
                             previous.plan, boundary, runner)
            policy, contract, config, _ = contracts(task, value["model_sha256"], boundary)
            label = f"{name}_b{boundary}_node"
            trace_path = folder / f"b{boundary}_trace.jsonl"
            trace = own.journal.Journal(trace_path, 16 * 1024**2)
            driver = own.durable.Driver(folder / f"b{boundary}_node", contract, spec.digest(contract),
                                        "structured", "node", label)
            guard_driver(driver, check_limits)
            try:
                leaf = own.leafmod.LeafAdapter(model, driver, contract, spec.digest(contract))
                result = own.coherent.node.run_node(
                    leaf, arm="structured", boundary=boundary,
                    states=tensor(task["root_context"]["latent_history"])[None],
                    history=tensor(task["root_context"]["action_history"])[None],
                    goal=tensor(task["goal"])[None], mean=tensor(task["action_mean"]),
                    std=tensor(task["action_std"]), matrix=matrix,
                    previous=tensor(previous.plan.copy()), issued=tensor(issued), emit=trace.emit)
                torch.cuda.synchronize(device)
            finally:
                trace.close()
                ledger, measured = account(driver, contract, label, value["node_cost"][str(boundary)])
            require(result["charges"] == measured["charged_upper_bounds"], "node reported charges")
            selected = result["selected_raw"].detach().cpu().numpy().astype(np.float32)
            new_plan = carry.BoundPlan.seal(name, boundary, selected)
            node = dict(boundary=boundary, task=task, policy=policy, config=config,
                        current_pixel_sha256=cp["pixel_sha256"], latent=z.tolist(),
                        previous_plan=previous.plan.tolist(),
                        feedback={k: v for k, v in feedback.items() if k != "tail"},
                        selected_full_plan=selected.tolist(), selected_plan_sha256=new_plan.plan_sha256,
                        selected_index=result["selected_index"], scores=result["scores"],
                        encoding_ledger=enc_ledger, encoding_account=enc_account,
                        node_ledger=ledger, node_account=measured, trace=str(trace_path),
                        trace_sha256=sha(trace_path))
            row["nodes"].append(node)
            print(json.dumps(dict(prefix=name, boundary=boundary, status="node_complete")), flush=True)
            status = packet(env, selected[0], row, np.asarray(goal["goal_state"]))
            cp, pixels = checkpoint(env, folder, name, len(row["executed_raw"]), goal["goal_state"])
            row["checkpoints"].append(cp)
            if status:
                row["status"] = status["status"]
                row["stop_reason"] = status
                return row
            previous = new_plan
        row["status"] = "completed20"
    except Exception as exc:
        row["error"] = dict(type=type(exc).__name__, message=str(exc)[:1000])
    finally:
        if env is not None:
            env.close()
        row["steps"] = len(row["executed_raw"])
        own.write(folder / "receipt.json", row)
    return row


def dry(value, runner):
    checks = []
    for row in value["rows"]:
        old = np.asarray(row["previous_cp5_plan"], np.float32)
        previous = old
        latents = row["cp5_task"]["root_context"]["latent_history"][-2:] + [[0.] * 192]
        raw10 = np.asarray(row["first10"], np.float32)
        for boundary in (10, 15):
            if boundary == 15:
                previous = np.zeros((2, 5, 2), np.float32)
                raw10 = np.concatenate((raw10[-5:], previous[0]))
            task = make_task(row["cp5_task"], latents, raw10, row["checkpoints"][2]["pixel_sha256"],
                             previous, boundary, runner)
            policy, contract, config, encoding = contracts(task, value["model_sha256"], boundary)
            checks.append(dict(prefix=row["prefix"], boundary=boundary, horizon=policy["horizon"],
                               config=config, node_contract=spec.digest(contract),
                               encoding_contract=spec.digest(encoding)))
    return dict(status="dry_prepared_not_written", roster=list(PREFIXES), contracts=checks,
                node_cost=value["node_cost"], encode_cost=value["encode_cost"],
                build_digest=spec.digest(value), no_model_executed=True, physical_execution=False)


def run(prepared, expected_sha, output, authorization, experiment, runner):
    require(authorization == AUTH, "explicit CUDA and physical authorization")
    prepared = Path(prepared)
    require(prepared.is_absolute() and prepared.parent == ROOT / "results"
            and prepared.is_file() and not prepared.is_symlink()
            and spec.is_hash(expected_sha) and sha(prepared) == expected_sha, "prepare SHA")
    value = json.loads(prepared.read_bytes())
    require(value == build(runner), "prepared/source drift")
    output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= 512 * 1024**2, "disk reserve")
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        output.mkdir()
        binding_sha = own.write(output / "binding.json", value)
        rows = [dict(prefix=name, status="not_attempted", steps=0, executed_raw=[],
                     checkpoints=[], nodes=[]) for name in PREFIXES]
        before = after = host = failure = None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

            def check_limits():
                require(time.monotonic() - started < 3600
                        and torch.cuda.memory_allocated(device) <= 4 * 1024**3
                        and torch.cuda.memory_reserved(device) <= 6 * 1024**3
                        and shutil.disk_usage(output).free >= 128 * 1024**2, "resource limits")

            with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                check_limits()
                model = checkpoint_compat.load_compatible_pretrained(value["model_policy"], device=device).eval().requires_grad_(False)
                before = runner.signal.model_hash(model)
                require(before == value["expected_model_hash"], "model identity")
                matrix = torch.as_tensor(h4.reverse.Q.kernels.MATRIX, dtype=torch.float64, device=device)
                require(value["base_task"]["goal"] == h4.reverse.Q.kernels.GOAL
                        and h4.reverse.Q.kernels.ARTIFACT_SHA == own.coherent.centered.adapter.METRIC_SHA,
                        "centered goal and metric")
                for index, slot in enumerate(value["rows"]):
                    check_limits()
                    check_sources(value["sources"])
                    print(json.dumps(dict(prefix=slot["prefix"], status="started")), flush=True)
                    rows[index] = run_branch(slot, value, output, model, device, matrix, runner, check_limits)
                    print(json.dumps(dict(prefix=slot["prefix"], status=rows[index]["status"],
                                          steps=rows[index]["steps"], nodes=len(rows[index]["nodes"]),
                                          error=rows[index].get("error"))), flush=True)
                after = runner.signal.model_hash(model)
                require(before == after, "model changed")
                check_sources(value["sources"])
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
        report = dict(schema=NAME + ".report.v1",
                      status="complete" if failure is None and all(r["status"] == "completed20" for r in rows)
                      else "partial", error=failure, binding_sha256=binding_sha,
                      prepared=str(prepared), prepare_sha256=expected_sha, rows=rows,
                      model_before=before, model_after=after, host=host,
                      P1_admission=False, independent_confirmation=False,
                      expected_nodes=12, measured_nodes=sum(len(r["nodes"]) for r in rows),
                      physical_execution=True, node_cost=value["node_cost"],
                      limitations="known-root exploratory feedback; censored or failed branches retained")
        report_sha = own.write(output / "report.json", report)
        return dict(status=report["status"], report_sha256=report_sha, output=str(output),
                    measured_nodes=report["measured_nodes"], error=failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    for arg in ("prepared", "sha256", "output", "authorization"):
        execute.add_argument("--" + arg, required=True)
    execute.add_argument("--authorize-host-cuda-and-physical", required=True, action="store_true")
    args = parser.parse_args()
    experiment, runner = own.configure()
    if args.command == "dry-prepare":
        result = dry(build(runner), runner)
    elif args.command == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=own.write(path, build(runner)))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization, experiment, runner)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
