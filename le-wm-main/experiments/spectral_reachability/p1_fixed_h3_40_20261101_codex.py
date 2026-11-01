"""Six-prefix 40-step fixed-H3 actual-RGB exploratory controller."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import p1_six_prefix_feedback_20261101_codex as fb

own, carry, h4, spec = fb.own, fb.carry, fb.h4, fb.spec
ROOT = own.ROOT
NAME = "p1_fixed_h3_40_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P1_FIXED_H3_40_20261101_CODEX.md"
BOUNDARIES = (10, 15, 20, 25, 30, 35)
LOGICAL_BOUNDARY = 5
AUTH = "MAIN_AUTHORIZES_FIXED_H3_40_HOST_CUDA_AND_PHYSICAL"
require, sha = own.require, own.sha


def extended_carry(previous, *, episode, boundary, issued, pixel_sha):
    """Verify real lineage first; append a zero packet only for H3 scoring."""
    require(boundary in BOUNDARIES and previous.plan.shape == (3, 5, 2), "physical H3 lineage")
    linked = previous.bind_feedback(episode=episode, boundary=boundary,
                                    issued_packet=np.asarray(issued, np.float32), rgb_sha256=pixel_sha)
    require(linked["tail"].shape == (2, 5, 2), "two actual old tail packets")
    extended = np.concatenate((previous.plan.copy(), np.zeros((1, 5, 2), np.float32)), axis=0)
    return extended, {k: v for k, v in linked.items() if k != "tail"}


def task_for(cp5_task, latents, raw10, pixel_sha, extended, physical_boundary, runner):
    require(physical_boundary in BOUNDARIES, "physical boundary")
    z, raw, old = np.asarray(latents, np.float32), np.asarray(raw10, np.float32), np.asarray(extended, np.float32)
    require(z.shape == (3, 192) and np.isfinite(z).all() and raw.shape == (10, 2)
            and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
            and old.shape == (4, 5, 2) and np.isfinite(old).all() and (np.abs(old) <= 1).all()
            and np.array_equal(old[0], raw[-5:]) and not np.count_nonzero(old[-1])
            and spec.is_hash(pixel_sha), "actual history and hold-extended carry")
    task = deepcopy(cp5_task)
    task["root_context"]["latent_history"] = z.tolist()
    task["root_context"]["action_history"] = runner.rgb.moving.normalized(task, raw)
    ident = spec.digest([NAME, physical_boundary, LOGICAL_BOUNDARY, pixel_sha,
                         raw.tolist(), old.tolist(), z.tolist()])
    binding = task["binding"]
    binding["context_sha256"] = spec.digest(task["root_context"])
    binding["root_sha256"] = ident
    binding["task_id"] = spec.digest([NAME, binding["task_id"], ident])
    binding["task_sha256"] = spec.digest([NAME, "task", binding["task_id"]])
    binding["task_contract_sha256"] = spec.digest([NAME, "contract", binding["task_id"], old.tolist()])
    runner.p3.core.validate_task(task)
    return task


def contracts(task, model_sha):
    control = h4.reverse.Q.carry.control
    cost = h4.h4.node_cost("structured", LOGICAL_BOUNDARY)
    with patch.dict(control.base.CONFIG, h4.boundary_config(LOGICAL_BOUNDARY)), \
            patch.object(control, "cost", lambda arm: h4.h4.node_cost(arm, LOGICAL_BOUNDARY)):
        policy, contract, config = control.contract_for(task, model_sha, spec.digest({"device_policy": "host_cuda_0"}))
    require(policy["horizon"] == 3 and config.items() >= h4.boundary_config(5).items()
            and config["steps"] == 6 and config["learning_rate"] == .05
            and all(contract["levels"][0]["caps"][k] >= cost[k] for k in spec.AXES),
            "fixed H3 optimizer contract")
    require(np.array_equal(np.asarray(task["projection"], np.float64), np.eye(192)), "identity projection")
    enc = deepcopy(contract)
    caps = dict.fromkeys(spec.AXES, 0)
    caps.update(encode_calls=1, encode_samples=1)
    enc["levels"] = [dict(id="node", caps=caps),
                     dict(id="unused_larger", caps={k: v + 1 for k, v in caps.items()})]
    spec.preflight(enc, spec.digest(enc))
    return policy, contract, config, enc


def build(runner):
    base = fb.build(runner)
    sources = dict(base["sources"])
    for path in (Path(__file__).resolve(), DOC):
        sources[str(path)] = sha(path)
    fb.check_sources(sources)
    return dict(schema=NAME + ".prepare.v1", rows=base["rows"], sources=sources,
                base_task=base["base_task"], model_policy=base["model_policy"],
                model_sha256=base["model_sha256"], expected_model_hash=base["expected_model_hash"],
                physical_boundaries=list(BOUNDARIES), logical_optimizer_boundary=LOGICAL_BOUNDARY,
                carry="hold_extended_actual_issued_old_tail2_zero_not_original_plan",
                node_cost=h4.h4.node_cost("structured", LOGICAL_BOUNDARY),
                encode_cost=dict(encode_calls=1, encode_samples=1),
                P1_admission=False, independent_confirmation=False, matched_budget_comparison=False)


def exclusive(value):
    path = Path(value)
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
            "exclusive output")
    return path


def run_branch(slot, value, output, model, device, matrix, runner, check_limits):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    name = slot["prefix"]
    row = dict(prefix=name, status="failed", stop_reason=None, executed_raw=[],
               checkpoints=[], nodes=[], error=None)
    folder = output / name
    folder.mkdir()
    env = None
    try:
        goal, images, _ = fb.common.target.collect()
        goal_state = np.asarray(goal["goal_state"])
        root = json.loads((fb.common.target.DIRECTORY / "report.json").read_bytes())
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
        fb.common.native_source.attach_goal(body, goal_state, images[-1])
        cp, _ = fb.checkpoint(env, folder, name, 0, goal_state)
        fb.check_checkpoint(cp, slot["checkpoints"][0])
        row["checkpoints"].append(cp)
        first10 = np.asarray(slot["first10"], np.float32)
        for i in (0, 5):
            stopped = fb.packet(env, first10[i:i+5], row, goal_state)
            cp, pixels = fb.checkpoint(env, folder, name, len(row["executed_raw"]), goal_state)
            row["checkpoints"].append(cp)
            if stopped:
                row.update(status=stopped["status"], stop_reason=stopped)
                return row
            fb.check_checkpoint(cp, slot["checkpoints"][1 + i//5])
        previous = carry.BoundPlan.seal(name, 5, np.asarray(slot["previous_cp5_plan"], np.float32))
        latents = deepcopy(slot["cp5_task"]["root_context"]["latent_history"])
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        for boundary in BOUNDARIES:
            check_limits()
            require(len(row["executed_raw"]) == boundary, "physical boundary")
            issued = np.asarray(row["executed_raw"][-5:], np.float32)
            extended, linked = extended_carry(previous, episode=name, boundary=boundary,
                                               issued=issued, pixel_sha=cp["pixel_sha256"])
            raw10 = np.asarray(row["executed_raw"][-10:], np.float32)
            placeholder = task_for(slot["cp5_task"], latents[-2:] + [[0.] * 192],
                                   raw10, cp["pixel_sha256"], extended, boundary, runner)
            _, _, _, enc_contract = contracts(placeholder, value["model_sha256"])
            enc_label = f"{name}_b{boundary}_encode"
            ed = own.durable.Driver(folder / f"b{boundary}_encode", enc_contract,
                                    spec.digest(enc_contract), "structured", "node", enc_label)
            fb.guard_driver(ed, check_limits)
            try:
                leaf = own.leafmod.LeafAdapter(model, ed, enc_contract, spec.digest(enc_contract))
                image = own.p0._image_tensor(pixels, transform=own.p0.image_transform(), device=device)
                with torch.inference_mode():
                    z = leaf.encode({"pixels": image}, phase="setup")["emb"][0, 0].detach().cpu().numpy()
            finally:
                enc_cost = dict.fromkeys(spec.AXES, 0)
                enc_cost.update(encode_calls=1, encode_samples=1)
                enc_ledger, enc_account = fb.account(ed, enc_contract, enc_label, enc_cost)
            latents = latents[-2:] + [z.tolist()]
            task = task_for(slot["cp5_task"], latents, raw10, cp["pixel_sha256"], extended, boundary, runner)
            policy, contract, config, _ = contracts(task, value["model_sha256"])
            label = f"{name}_b{boundary}_node"
            trace_path = folder / f"b{boundary}_trace.jsonl"
            trace = own.journal.Journal(trace_path, 16 * 1024**2)
            driver = own.durable.Driver(folder / f"b{boundary}_node", contract, spec.digest(contract),
                                        "structured", "node", label)
            fb.guard_driver(driver, check_limits)
            try:
                leaf = own.leafmod.LeafAdapter(model, driver, contract, spec.digest(contract))
                result = own.coherent.node.run_node(
                    leaf, arm="structured", boundary=LOGICAL_BOUNDARY,
                    states=tensor(task["root_context"]["latent_history"])[None],
                    history=tensor(task["root_context"]["action_history"])[None],
                    goal=tensor(task["goal"])[None], mean=tensor(task["action_mean"]),
                    std=tensor(task["action_std"]), matrix=matrix,
                    previous=tensor(extended), issued=tensor(issued), emit=trace.emit)
                torch.cuda.synchronize(device)
            finally:
                trace.close()
                ledger, actual = fb.account(driver, contract, label, value["node_cost"])
            require(result["charges"] == actual["charged_upper_bounds"], "node measured charges")
            selected = result["selected_raw"].detach().cpu().numpy().astype(np.float32)
            require(selected.shape == (3, 5, 2), "three selected packets")
            new_plan = carry.BoundPlan.seal(name, boundary, selected)
            row["nodes"].append(dict(physical_boundary=boundary, logical_optimizer_boundary=LOGICAL_BOUNDARY,
                task=task, policy=policy, config=config, current_pixel_sha256=cp["pixel_sha256"],
                latent=z.tolist(), previous_original_plan=previous.plan.tolist(),
                carry_kind=value["carry"], hold_extended_previous=extended.tolist(), feedback=linked,
                selected_full_plan=selected.tolist(), selected_plan_sha256=new_plan.plan_sha256,
                selected_index=result["selected_index"], scores=result["scores"],
                encoding_ledger=enc_ledger, encoding_account=enc_account,
                node_ledger=ledger, node_account=actual, trace=str(trace_path), trace_sha256=sha(trace_path)))
            print(json.dumps(dict(prefix=name, physical_boundary=boundary,
                                  logical_optimizer_boundary=LOGICAL_BOUNDARY, status="node_complete")), flush=True)
            stopped = fb.packet(env, selected[0], row, goal_state)
            cp, pixels = fb.checkpoint(env, folder, name, len(row["executed_raw"]), goal_state)
            row["checkpoints"].append(cp)
            if stopped:
                row.update(status=stopped["status"], stop_reason=stopped)
                return row
            previous = new_plan
        row["status"] = "completed40"
    except Exception as exc:
        row["error"] = dict(type=type(exc).__name__, message=str(exc)[:1000])
    finally:
        if env is not None:
            env.close()
        row["steps"] = len(row["executed_raw"])
        row["observed_cp20"] = next((point for point in row["checkpoints"] if point["step"] == 20), None)
        row["observed_cp40"] = next((point for point in row["checkpoints"] if point["step"] == 40), None)
        own.write(folder / "receipt.json", row)
    return row


def dry(value, runner):
    checks = []
    for slot in value["rows"]:
        original = np.asarray(slot["previous_cp5_plan"], np.float32)
        previous = carry.BoundPlan.seal(slot["prefix"], 5, original)
        raw10 = np.asarray(slot["first10"], np.float32)
        latents = slot["cp5_task"]["root_context"]["latent_history"][-2:] + [[0.] * 192]
        for boundary in BOUNDARIES:
            ext, linked = extended_carry(previous, episode=slot["prefix"], boundary=boundary,
                                         issued=raw10[-5:], pixel_sha=slot["checkpoints"][2]["pixel_sha256"])
            task = task_for(slot["cp5_task"], latents, raw10, slot["checkpoints"][2]["pixel_sha256"],
                            ext, boundary, runner)
            policy, contract, config, enc = contracts(task, value["model_sha256"])
            checks.append(dict(prefix=slot["prefix"], physical_boundary=boundary,
                               logical_optimizer_boundary=LOGICAL_BOUNDARY, horizon=policy["horizon"],
                               node_contract=spec.digest(contract), encoding_contract=spec.digest(enc),
                               previous_plan_sha256=linked["previous_plan_sha256"]))
            # Pure synthetic next packet for contract construction only.
            previous = carry.BoundPlan.seal(slot["prefix"], boundary, np.zeros((3, 5, 2), np.float32))
            raw10 = np.concatenate((raw10[-5:], np.zeros((5, 2), np.float32)))
    return dict(status="dry_prepared_not_written", build_digest=spec.digest(value),
                roster=list(fb.PREFIXES), contracts=checks, node_cost=value["node_cost"],
                encode_cost=value["encode_cost"], no_model_executed=True, physical_execution=False)


def study_summary(rows, failure):
    statuses = ("completed40", "native_stopped", "canvas_censored")
    counts = {status: sum(r["status"] == status for r in rows) for status in statuses}
    counts["failed"] = sum(r["status"] == "failed" for r in rows)
    counts["not_attempted"] = sum(r["status"] == "not_attempted" for r in rows)
    valid = (failure is None and tuple(r["prefix"] for r in rows) == fb.PREFIXES
             and all(r["status"] in statuses and r.get("error") is None for r in rows))
    return dict(status="complete" if valid else "partial",
                all_full_horizon=valid and counts["completed40"] == len(fb.PREFIXES),
                branch_status_counts=counts)


def run(prepared, expected_sha, output, authorization, experiment, runner):
    require(authorization == AUTH, "explicit CUDA and physical authorization")
    prepared = Path(prepared)
    require(prepared.is_absolute() and prepared.parent == ROOT / "results" and prepared.is_file()
            and not prepared.is_symlink() and spec.is_hash(expected_sha)
            and sha(prepared) == expected_sha, "prepare SHA")
    value = json.loads(prepared.read_bytes())
    require(value == build(runner), "prepared/source drift")
    output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= 512 * 1024**2, "disk reserve")
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        output.mkdir()
        binding_sha = own.write(output / "binding.json", value)
        rows = [dict(prefix=name, status="not_attempted", steps=0, stop_reason=None,
                     executed_raw=[], checkpoints=[], nodes=[], observed_cp20=None, observed_cp40=None)
                for name in fb.PREFIXES]
        before = after = host = failure = None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

            def check_limits():
                require(time.monotonic() - started < 7200
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
                    fb.check_sources(value["sources"])
                    print(json.dumps(dict(prefix=slot["prefix"], status="started")), flush=True)
                    rows[index] = run_branch(slot, value, output, model, device, matrix, runner, check_limits)
                    print(json.dumps(dict(prefix=slot["prefix"], status=rows[index]["status"],
                                          steps=rows[index]["steps"], nodes=len(rows[index]["nodes"]),
                                          error=rows[index].get("error"))), flush=True)
                after = runner.signal.model_hash(model)
                require(before == after, "model changed")
                fb.check_sources(value["sources"])
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
        summary = study_summary(rows, failure)
        report = dict(schema=NAME + ".report.v1", **summary,
                      error=failure, binding_sha256=binding_sha,
                      prepared=str(prepared), prepare_sha256=expected_sha, rows=rows,
                      model_before=before, model_after=after, host=host,
                      P1_admission=False, independent_confirmation=False, matched_budget_comparison=False,
                      expected_nodes=36, measured_nodes=sum(len(r["nodes"]) for r in rows),
                      physical_execution=True, node_cost=value["node_cost"], encode_cost=value["encode_cost"],
                      limitations="new known-root 40-step controller; failures and stops retained")
        report_sha = own.write(output / "report.json", report)
        return dict(status=report["status"], report_sha256=report_sha, output=str(output),
                    measured_nodes=report["measured_nodes"], all_full_horizon=report["all_full_horizon"],
                    branch_status_counts=report["branch_status_counts"], error=failure)


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
