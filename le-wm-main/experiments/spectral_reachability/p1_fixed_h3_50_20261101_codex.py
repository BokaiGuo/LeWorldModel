"""Replay verified fixed-H3 40-step histories and continue survivors to 50."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import p1_fixed_h3_40_20261101_codex as prior
from . import p1_fixed_h3_saved_20261101_main as verifier

fb, own, carry, h4, spec = prior.fb, prior.own, prior.carry, prior.h4, prior.spec
ROOT = prior.ROOT
NAME = "p1_fixed_h3_50_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P1_FIXED_H3_50_20261101_CODEX.md"
CONFIG = ROOT / "config/eval/pusht.yaml"
OLD = ROOT / "results/p1_fixed_h3_40_20261101_codex_RUN_MAIN01"
REPORT = OLD / "report.json"
REPORT_SHA = "1106877d113223c2d9bb833930bd1f889b7c1a335c442a19363fae9a1dc4772f"
CHECK = ROOT / "results/p1_fixed_h3_saved_20261101_main_MAIN01.json"
CHECK_SHA = "2785bf839acb6f73303f40dcecb12bdff7a58d3d673c962342d22aee3cc42edd"
BOUNDARIES = (40, 45)
AUTH = "MAIN_AUTHORIZES_FIXED_H3_50_HOST_CUDA_AND_PHYSICAL"
require, sha = own.require, own.sha


def extended_carry(previous, *, episode, boundary, issued, pixel_sha):
    require(boundary in BOUNDARIES and previous.plan.shape == (3, 5, 2), "actual H3 lineage")
    linked = previous.bind_feedback(episode=episode, boundary=boundary,
                                    issued_packet=np.asarray(issued, np.float32), rgb_sha256=pixel_sha)
    require(linked["tail"].shape == (2, 5, 2), "two old tail packets")
    return (np.concatenate((previous.plan.copy(), np.zeros((1, 5, 2), np.float32)), axis=0),
            {k: v for k, v in linked.items() if k != "tail"})


def make_task(cp5_task, latents, raw10, pixel_sha, extended, boundary, runner):
    z, raw, old = np.asarray(latents, np.float32), np.asarray(raw10, np.float32), np.asarray(extended, np.float32)
    require(boundary in BOUNDARIES and z.shape == (3, 192) and np.isfinite(z).all()
            and raw.shape == (10, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
            and old.shape == (4, 5, 2) and np.isfinite(old).all() and (np.abs(old) <= 1).all()
            and np.array_equal(old[0], raw[-5:]) and not np.count_nonzero(old[-1])
            and spec.is_hash(pixel_sha), "actual history and hold-extended carry")
    task = deepcopy(cp5_task)
    task["root_context"]["latent_history"] = z.tolist()
    task["root_context"]["action_history"] = runner.rgb.moving.normalized(task, raw)
    ident = spec.digest([NAME, boundary, 5, pixel_sha, raw.tolist(), old.tolist(), z.tolist()])
    binding = task["binding"]
    binding["context_sha256"] = spec.digest(task["root_context"])
    binding["root_sha256"] = ident
    binding["task_id"] = spec.digest([NAME, binding["task_id"], ident])
    binding["task_sha256"] = spec.digest([NAME, "task", binding["task_id"]])
    binding["task_contract_sha256"] = spec.digest([NAME, "contract", binding["task_id"], old.tolist()])
    runner.p3.core.validate_task(task)
    return task


def build():
    require(sha(REPORT) == REPORT_SHA and sha(CHECK) == CHECK_SHA, "sealed 40 report and verifier")
    report, checked = json.loads(REPORT.read_bytes()), json.loads(CHECK.read_bytes())
    binding_path = OLD / "binding.json"
    require(sha(binding_path) == report["binding_sha256"], "old binding")
    binding = json.loads(binding_path.read_bytes())
    require(sha(report["prepared"]) == report["prepare_sha256"]
            and json.loads(Path(report["prepared"]).read_bytes()) == binding
            and report["status"] == "complete" and report["error"] is None
            and report["model_before"] == report["model_after"] == binding["expected_model_hash"]
            and checked["verified"] is True and checked["neural_calls"] == 0
            and checked["P1_admission"] is False and checked["independent_confirmation"] is False
            and checked["sources"].get(str(REPORT)) == REPORT_SHA, "verified 40 lineage")
    fb.check_sources(checked["sources"])
    require(tuple(r["prefix"] for r in report["rows"]) == fb.PREFIXES
            and tuple(r["prefix"] for r in checked["rows"]) == fb.PREFIXES
            and tuple(r["prefix"] for r in binding["rows"]) == fb.PREFIXES, "fixed roster")
    require("eval_budget: 50" in CONFIG.read_text(), "PushT eval budget")
    rows = []
    for row, summary, original in zip(report["rows"], checked["rows"], binding["rows"], strict=True):
        require(row["prefix"] == summary["prefix"] == original["prefix"]
                and row["status"] == summary["status"] and row["steps"] == summary["steps"]
                and row.get("error") is None, "verifier branch binding")
        if row["status"] != "completed40":
            require(row["status"] in ("native_stopped", "canvas_censored")
                    and row["stop_reason"]["step"] == row["steps"], "inherited terminal status")
            rows.append(dict(prefix=row["prefix"], eligible=False, old_status=row["status"],
                             old_steps=row["steps"], old_stop_reason=row["stop_reason"],
                             old_final_checkpoint=row["checkpoints"][-1]))
            continue
        raw = np.asarray(row["executed_raw"], np.float32)
        require(row["steps"] == 40 and raw.shape == (40, 2) and np.isfinite(raw).all()
                and (np.abs(raw) <= 1).all() and [p["step"] for p in row["checkpoints"]] == list(range(0, 41, 5))
                and len(row["nodes"]) == 6 and row["nodes"][-1]["physical_boundary"] == 35,
                "complete forty actions/checkpoints")
        last = row["nodes"][-1]
        selected = np.asarray(last["selected_full_plan"], np.float32)
        sealed = carry.BoundPlan.seal(row["prefix"], 35, selected)
        require(selected.shape == (3, 5, 2) and sealed.plan_sha256 == last["selected_plan_sha256"]
                and np.array_equal(selected[0], raw[35:40]), "old last plan/returned packet")
        rows.append(dict(prefix=row["prefix"], eligible=True, old_status=row["status"], old_steps=40,
                         old_actions=raw.tolist(), old_checkpoints=row["checkpoints"],
                         cp5_task=original["cp5_task"],
                         prior_latents=last["task"]["root_context"]["latent_history"],
                         previous_plan=selected.tolist(), previous_plan_sha256=sealed.plan_sha256))
    require(sum(r["eligible"] for r in rows) == 5
            and not rows[2]["eligible"] and rows[2]["prefix"] == "random"
            and rows[2]["old_steps"] == 31, "fixed observed cohort")
    sources = dict(checked["sources"])
    for path in (REPORT, CHECK, binding_path, Path(verifier.__file__).resolve(), CONFIG,
                 Path(__file__).resolve(), DOC):
        sources[str(path)] = sha(path)
    fb.check_sources(sources)
    return dict(schema=NAME + ".prepare.v1", rows=rows, sources=sources,
                model_policy=binding["model_policy"], model_sha256=binding["model_sha256"],
                expected_model_hash=binding["expected_model_hash"],
                node_cost=h4.h4.node_cost("structured", 5),
                encode_cost=dict(encode_calls=1, encode_samples=1),
                historical_dispatch_charges=checked["actual_dispatch_charges"],
                physical_boundaries=list(BOUNDARIES), logical_optimizer_boundary=5,
                carry="hold_extended_actual_issued_old_tail2_zero_not_original_plan",
                eval_budget=50, expected_new_nodes=10, expected_new_encodes=10,
                P1_admission=False, independent_confirmation=False, matched_budget_comparison=False)


def exclusive(value):
    path = Path(value)
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
            "exclusive output")
    return path


def study_summary(rows, failure):
    valid = ("completed50", "native_stopped", "canvas_censored", "inherited_censored", "inherited_native_stopped")
    counts = {status: sum(r["status"] == status for r in rows) for status in (*valid, "failed", "not_attempted")}
    complete = (failure is None and tuple(r["prefix"] for r in rows) == fb.PREFIXES
                and all(r["status"] in valid and r.get("error") is None for r in rows))
    return dict(status="complete" if complete else "partial", branch_status_counts=counts,
                all_survivors_full_horizon=complete and counts["completed50"] == 5)


def measured_charges(rows):
    node, encoding = dict.fromkeys(spec.AXES, 0), dict.fromkeys(spec.AXES, 0)
    for row in rows:
        for entry in row["node_ledgers"]:
            for key, amount in entry["account"]["charged_upper_bounds"].items():
                node[key] += amount
        for entry in row["encodings"]:
            for key, amount in entry["account"]["charged_upper_bounds"].items():
                encoding[key] += amount
    return dict(planner=node, rgb_encoding=encoding)


def run_branch(slot, value, output, model, device, matrix, runner, check_limits):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    name = slot["prefix"]
    row = dict(prefix=name, status="failed", old_status=slot["old_status"], stop_reason=None,
               executed_raw=[], checkpoints=[], nodes=[], node_ledgers=[], encodings=[], error=None, physical_replay_steps=0,
               physical_continuation_steps=0, warmup_steps=0)
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
                row["warmup_steps"] += 1
                require(not stop and not trunc and np.array_equal(body.latest_action, action), "warmup")
            if step in (0, 5, 10):
                require(np.array_equal(env.render(), images[step//5]), "warmup RGB")
        fb.common.native_source.attach_goal(body, goal_state, images[-1])
        cp, _ = fb.checkpoint(env, folder, name, 0, goal_state)
        fb.check_checkpoint(cp, slot["old_checkpoints"][0])
        row["checkpoints"].append(cp)
        old_actions = np.asarray(slot["old_actions"], np.float32)
        for index in range(8):
            stop = fb.packet(env, old_actions[5*index:5*index+5], row, goal_state)
            row["physical_replay_steps"] = len(row["executed_raw"])
            cp, pixels = fb.checkpoint(env, folder, name, row["physical_replay_steps"], goal_state)
            row["checkpoints"].append(cp)
            if stop:
                row["stop_reason"] = stop
                raise ValueError("unexpected stop while replaying verified completed40")
            fb.check_checkpoint(cp, slot["old_checkpoints"][index+1])
        previous = carry.BoundPlan.seal(name, 35, np.asarray(slot["previous_plan"], np.float32))
        require(previous.plan_sha256 == slot["previous_plan_sha256"], "old selected plan identity")
        latents = deepcopy(slot["prior_latents"])
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        for boundary in BOUNDARIES:
            check_limits()
            require(len(row["executed_raw"]) == boundary, "actual boundary")
            issued = np.asarray(row["executed_raw"][-5:], np.float32)
            extended, linked = extended_carry(previous, episode=name, boundary=boundary,
                                               issued=issued, pixel_sha=cp["pixel_sha256"])
            raw10 = np.asarray(row["executed_raw"][-10:], np.float32)
            placeholder = make_task(slot["cp5_task"], latents[-2:] + [[0.] * 192], raw10,
                                    cp["pixel_sha256"], extended, boundary, runner)
            _, _, _, enc_contract = prior.contracts(placeholder, value["model_sha256"])
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
                enc_ledger = ed.finish()
                enc_account = own.consumer.consume(enc_ledger["journal"], enc_ledger["sha256"],
                    spec.digest(enc_contract), "structured", "node", enc_label)
                row["encodings"].append(dict(physical_boundary=boundary, ledger=enc_ledger, account=enc_account))
            require(enc_account["status"] == "closed_unverified"
                    and enc_account["charged_upper_bounds"] == enc_cost, "encoding measured charges")
            latents = latents[-2:] + [z.tolist()]
            task = make_task(slot["cp5_task"], latents, raw10, cp["pixel_sha256"], extended, boundary, runner)
            policy, contract, config, _ = prior.contracts(task, value["model_sha256"])
            label = f"{name}_b{boundary}_node"
            trace_path = folder / f"b{boundary}_trace.jsonl"
            trace = own.journal.Journal(trace_path, 16 * 1024**2)
            driver = own.durable.Driver(folder / f"b{boundary}_node", contract, spec.digest(contract),
                                        "structured", "node", label)
            fb.guard_driver(driver, check_limits)
            try:
                leaf = own.leafmod.LeafAdapter(model, driver, contract, spec.digest(contract))
                result = own.coherent.node.run_node(
                    leaf, arm="structured", boundary=5,
                    states=tensor(task["root_context"]["latent_history"])[None],
                    history=tensor(task["root_context"]["action_history"])[None],
                    goal=tensor(task["goal"])[None], mean=tensor(task["action_mean"]),
                    std=tensor(task["action_std"]), matrix=matrix,
                    previous=tensor(extended), issued=tensor(issued), emit=trace.emit)
                torch.cuda.synchronize(device)
            finally:
                trace.close()
                ledger = driver.finish()
                actual = own.consumer.consume(ledger["journal"], ledger["sha256"], spec.digest(contract),
                                              "structured", "node", label)
                row["node_ledgers"].append(dict(physical_boundary=boundary, ledger=ledger, account=actual))
            require(actual["status"] == "closed_unverified"
                    and actual["charged_upper_bounds"] == value["node_cost"]
                    and result["charges"] == actual["charged_upper_bounds"], "node measured charges")
            selected = result["selected_raw"].detach().cpu().numpy().astype(np.float32)
            require(selected.shape == (3, 5, 2), "selected H3")
            new_plan = carry.BoundPlan.seal(name, boundary, selected)
            row["nodes"].append(dict(physical_boundary=boundary, logical_optimizer_boundary=5,
                task=task, policy=policy, config=config, current_pixel_sha256=cp["pixel_sha256"],
                latent=z.tolist(), previous_original_plan=previous.plan.tolist(),
                carry_kind=value["carry"], hold_extended_previous=extended.tolist(), feedback=linked,
                selected_full_plan=selected.tolist(), selected_plan_sha256=new_plan.plan_sha256,
                selected_index=result["selected_index"], scores=result["scores"],
                encoding_ledger=enc_ledger, encoding_account=enc_account,
                node_ledger=ledger, node_account=actual, trace=str(trace_path), trace_sha256=sha(trace_path)))
            print(json.dumps(dict(prefix=name, physical_boundary=boundary, logical_optimizer_boundary=5,
                                  status="node_complete")), flush=True)
            stop = fb.packet(env, selected[0], row, goal_state)
            row["physical_continuation_steps"] = len(row["executed_raw"]) - 40
            cp, pixels = fb.checkpoint(env, folder, name, len(row["executed_raw"]), goal_state)
            row["checkpoints"].append(cp)
            if stop:
                row.update(status=stop["status"], stop_reason=stop)
                return row
            previous = new_plan
        row["status"] = "completed50"
    except Exception as exc:
        row["error"] = dict(type=type(exc).__name__, message=str(exc)[:1000])
    finally:
        if env is not None:
            env.close()
        row["steps"] = len(row["executed_raw"])
        row["physical_replay_steps"] = min(row["steps"], 40)
        row["physical_continuation_steps"] = max(row["steps"] - 40, 0)
        row["observed_cp40"] = next((p for p in row["checkpoints"] if p["step"] == 40), None)
        row["observed_cp50"] = next((p for p in row["checkpoints"] if p["step"] == 50), None)
        own.write(folder / "receipt.json", row)
    return row


def dry(value, runner):
    checks = []
    for slot in value["rows"]:
        if not slot["eligible"]:
            continue
        previous = carry.BoundPlan.seal(slot["prefix"], 35, np.asarray(slot["previous_plan"], np.float32))
        raw10 = np.asarray(slot["old_actions"][-10:], np.float32)
        latent = slot["prior_latents"][-2:] + [[0.] * 192]
        for boundary in BOUNDARIES:
            extended, link = extended_carry(previous, episode=slot["prefix"], boundary=boundary,
                                              issued=raw10[-5:],
                                              pixel_sha=slot["old_checkpoints"][-1]["pixel_sha256"])
            task = make_task(slot["cp5_task"], latent, raw10, slot["old_checkpoints"][-1]["pixel_sha256"],
                             extended, boundary, runner)
            policy, contract, config, enc = prior.contracts(task, value["model_sha256"])
            checks.append(dict(prefix=slot["prefix"], physical_boundary=boundary,
                               logical_optimizer_boundary=5, horizon=policy["horizon"],
                               node_contract=spec.digest(contract), encoding_contract=spec.digest(enc),
                               prior_plan_sha256=link["previous_plan_sha256"]))
            previous = carry.BoundPlan.seal(slot["prefix"], boundary, np.zeros((3, 5, 2), np.float32))
            raw10 = np.concatenate((raw10[-5:], np.zeros((5, 2), np.float32)))
    return dict(status="dry_prepared_not_written", build_digest=spec.digest(value),
                roster=list(fb.PREFIXES), eligible=[r["prefix"] for r in value["rows"] if r["eligible"]],
                inherited=[r["prefix"] for r in value["rows"] if not r["eligible"]],
                contracts=checks, node_cost=value["node_cost"], encode_cost=value["encode_cost"],
                no_model_executed=True, physical_execution=False)


def run(prepared, expected_sha, output, authorization, experiment, runner):
    require(authorization == AUTH, "explicit CUDA and physical authorization")
    prepared = Path(prepared)
    require(prepared.is_absolute() and prepared.parent == ROOT / "results" and prepared.is_file()
            and not prepared.is_symlink() and spec.is_hash(expected_sha)
            and sha(prepared) == expected_sha, "prepare SHA")
    value = json.loads(prepared.read_bytes())
    require(value == build(), "prepared/source drift")
    output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= 512 * 1024**2, "disk reserve")
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        output.mkdir()
        binding_sha = own.write(output / "binding.json", value)
        rows = []
        for slot in value["rows"]:
            if slot["eligible"]:
                rows.append(dict(prefix=slot["prefix"], status="not_attempted", steps=0, error=None,
                                 old_status=slot["old_status"], stop_reason=None, executed_raw=[],
                                 checkpoints=[], nodes=[], node_ledgers=[], encodings=[], physical_replay_steps=0, physical_continuation_steps=0, warmup_steps=0,
                                 observed_cp40=None, observed_cp50=None))
            else:
                state = "inherited_censored" if slot["old_status"] == "canvas_censored" else "inherited_native_stopped"
                rows.append(dict(prefix=slot["prefix"], status=state, old_status=slot["old_status"],
                                 steps=slot["old_steps"], stop_reason=slot["old_stop_reason"],
                                 old_final_checkpoint=slot["old_final_checkpoint"], error=None,
                                 replayed=False, executed_raw=[], checkpoints=[], nodes=[], node_ledgers=[], encodings=[],
                                 physical_replay_steps=0, physical_continuation_steps=0,
                                 warmup_steps=0,
                                 observed_cp40=None, observed_cp50=None))
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
                require(h4.reverse.Q.kernels.ARTIFACT_SHA == own.coherent.centered.adapter.METRIC_SHA,
                        "centered metric")
                require(all(not r["eligible"] or r["cp5_task"]["goal"] == h4.reverse.Q.kernels.GOAL
                            for r in value["rows"]), "centered goal")
                for index, slot in enumerate(value["rows"]):
                    if not slot["eligible"]:
                        print(json.dumps(dict(prefix=slot["prefix"], status=rows[index]["status"],
                                              old_steps=slot["old_steps"])), flush=True)
                        continue
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
                      error=failure, binding_sha256=binding_sha, prepared=str(prepared),
                      prepare_sha256=expected_sha, rows=rows, model_before=before, model_after=after,
                      host=host, P1_admission=False, independent_confirmation=False,
                      matched_budget_comparison=False, expected_new_nodes=10, expected_new_encodes=10,
                      measured_new_nodes=sum(len(r["nodes"]) for r in rows),
                      measured_new_encodes=measured_charges(rows)["rgb_encoding"]["encode_calls"],
                      recorded_node_ledgers=sum(len(r["node_ledgers"]) for r in rows),
                      recorded_encoding_ledgers=sum(len(r["encodings"]) for r in rows),
                      actual_new_dispatch_charges=measured_charges(rows),
                      charge_aggregation_scope="consumed_terminal_ledgers_including_failed; missing terminals require audit",
                      historical_dispatch_charges=value["historical_dispatch_charges"],
                      physical_warmup_steps=sum(r["warmup_steps"] for r in rows),
                      physical_replay_steps=sum(r["physical_replay_steps"] for r in rows),
                      physical_continuation_steps=sum(r["physical_continuation_steps"] for r in rows),
                      node_cost=value["node_cost"], encode_cost=value["encode_cost"],
                      physical_execution=True, limitations="known-root 40-to-50 continuation, censored branch retained")
        report_sha = own.write(output / "report.json", report)
        return dict(status=report["status"], report_sha256=report_sha, output=str(output),
                    measured_new_nodes=report["measured_new_nodes"], error=failure)


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
        result = dry(build(), runner)
    elif args.command == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=own.write(path, build()))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization, experiment, runner)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
