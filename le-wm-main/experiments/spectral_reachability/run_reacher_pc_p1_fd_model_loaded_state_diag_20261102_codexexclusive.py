"""Sealed FD development selection: two prefixes x five common physical tails."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import audit_reacher_pc_p1_fd_suffix_saved_20261102_codexexclusive as audit
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_d2p_context as context_lib
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability import run_reacher_pc_spectral_common_suffix_development as common
from experiments.spectral_reachability import run_reacher_pc_tangent_full_physical as tangent

MODEL_SHA = "b76a55c6b6f8e13e7c4cbba0887e6ec6babc062c1a3245ddabf3991d39b00401"
AUDIT_SHA = "127300623499118458970a433521e5c8a3f2c1e00c55c9d2ba69986c11f7ce9c"
MODEL_PLAN_SHA = "12ff26c6c052e02fe487bde8affd21104b09b8950f6a5340f5538fea766a2307"
ROOTS = (20263160, 20263161)
PREFIXES = (("selected", 4), ("parameter_zero", 0))
HERE = Path(__file__).resolve()
PROTOCOL = HERE.parents[3] / "docs/PC_REACHER_P1_FD_COMMON_PHYSICAL_DEV_20261102_CODEXEXCLUSIVE.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_actions(actions):
    return hashlib.sha256(np.ascontiguousarray(actions, dtype=np.float32).tobytes()).hexdigest()


def runtime_versions():
    values = {"python": platform.python_version()}
    for name in ("numpy", "gymnasium", "mujoco", "dm-control", "torch"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def source_actions(source, mean, std, audited):
    require(source["status"] == "complete", "model root incomplete")
    receipt = source["compiler_receipt"]
    require(receipt["development_selected_index"] == 4 and receipt["strict_selected_index"] is None,
            "frozen selection mismatch")
    require(audited["status"] == "verified_saved_consistency"
            and audited["development_selected_index"] == 4
            and audited["strict_selected_index"] is None, "audit selection mismatch")
    # Recompute all saved-data checks before copying any action witness.
    verified = audit.audit_row(source, mean, std)
    require(verified["development_selected_index"] == 4 and verified["strict_selected_index"] is None,
            "independent saved selector mismatch")
    prefixes = np.asarray(source["prefix_raw"], dtype=np.float32)
    tails = np.asarray(source["tail_raw"], dtype=np.float32)
    require(prefixes.shape == (9, 5, 2) and tails.shape == (5, 10, 2)
            and np.isfinite(prefixes).all() and np.isfinite(tails).all(), "invalid source action arrays")
    return prefixes, tails


def make_branches(prefixes, tails):
    branches = []
    for arm, prefix_index in PREFIXES:
        for tail_index in range(5):
            actions = np.ascontiguousarray(np.concatenate((prefixes[prefix_index], tails[tail_index])), dtype=np.float32)
            require(actions.shape == (15, 2) and np.isfinite(actions).all()
                    and (np.abs(actions) <= 1.0 + 1e-6).all(), "invalid bound actions")
            branches.append({"arm": arm, "prefix_index": prefix_index, "tail_index": tail_index,
                             "raw_actions": actions.tolist(), "actions_sha256": digest_actions(actions)})
    return branches


def prepare(model_path, audit_path, model_plan_path, preflight_path):
    for path, expected in ((model_path, MODEL_SHA), (audit_path, AUDIT_SHA), (model_plan_path, MODEL_PLAN_SHA)):
        require(audit.sha(path) == expected, "sealed input hash mismatch: " + str(path))
    model, saved, model_plan, preflight = [json.loads(p.read_text()) for p in
                                           (model_path, audit_path, model_plan_path, preflight_path)]
    preflight_hash = audit.sha(preflight_path)
    require(saved["status"] == "complete_saved_consistency" and saved["report_sha256"] == MODEL_SHA
            and saved["plan_sha256"] == MODEL_PLAN_SHA and saved["preflight_sha256"] == preflight_hash
            and saved["P1_admission"] is False and saved["physical_execution"] is False
            and saved["consumer_sha256"] == audit.sha(audit.__file__),
            "saved audit binding mismatch")
    audit.audit_binding(model, model_plan, preflight, plan_hash=MODEL_PLAN_SHA,
                        preflight_hash=preflight_hash)
    require([r["seed"] for r in saved["rows"]] == list(ROOTS), "saved audit root roster mismatch")
    stats = Path(preflight["action_stats_path"])
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, history_receipt = setup.load_runtime_support(stats, history_len=3, action_block=5)
    roots = []
    for source, checked, root in zip(model["rows"], saved["rows"], model_plan["roots"], strict=True):
        require(source["seed"] == checked["seed"] == root["seed"], "root binding mismatch")
        prefixes, tails = source_actions(source, mean, std, checked)
        packets, index = setup.sample_history_packets(histories, seed=root["seed"])
        require(index == root["history_index"], "warmup index mismatch")
        roots.append({**root, "warmup_packets": packets.tolist(), "warmup_sha256": digest_actions(packets),
                      "branches": make_branches(prefixes, tails)})
    deps = (HERE, PROTOCOL, Path(audit.__file__), Path(setup.__file__), Path(context_lib.__file__),
            Path(support.__file__), Path(common.__file__), Path(tangent.__file__), Path(setup.p0.__file__))
    return {"schema": "pc_fd_common_physical_dev_v1", "model_sha256": MODEL_SHA, "audit_sha256": AUDIT_SHA,
            "model_plan_sha256": MODEL_PLAN_SHA, "preflight_sha256": preflight_hash,
            "stats_sha256": preflight["action_stats_sha256"], "runtime": runtime_versions(),
            "sources": {str(p): audit.sha(p) for p in deps}, "history_support": history_receipt,
            "expected_branch_count": 20, "roots": roots,
            "scope": "exposed development roots; selected means model development selector only; strict null"}


def _point(env, image):
    point = common.record(env)
    point["rgb"] = context_lib.pack_rgb_observation(np.asarray(image))
    return point


def execute_branch(env, seed, context, actions):
    """Restore exact root for each branch, retaining errors and stop semantics."""
    result = {"status": "started", "trajectory": [], "stop": None, "attempted_steps": 0, "returned_steps": 0}
    try:
        snapshot = context["execution_snapshot"]
        required = ("physics_integration_state", "integration_state_spec", "model_binary_sha256",
                    "target_qpos", "target_marker_pos", "rendered_goal_rgb")
        require(snapshot is not None and all(k in snapshot for k in required), "incomplete integration snapshot")
        image, goal = context_lib.restore_execution_snapshot(env, seed=seed, snapshot=snapshot)
        result["identity"] = setup.p0._verify_execution_identity(image, goal, context["root_frame"], context["goal"])
        context_lib.assert_exact_execution_state(snapshot, context_lib.capture_execution_snapshot(env))
        result["trajectory"].append(_point(env, image))
        for index, action in enumerate(actions):
            result["attempted_steps"] = index + 1
            _, reward, terminated, truncated, _ = env.step(action)
            result["returned_steps"] = index + 1
            if terminated or truncated:
                result["stop"] = {"step": index + 1, "terminated": bool(terminated), "truncated": bool(truncated)}
            point = _point(env, env.render())
            point.update(action=np.asarray(action, dtype=np.float32).tolist(), reward=float(reward),
                         terminated=bool(terminated), truncated=bool(truncated))
            result["trajectory"].append(point)
            if terminated or truncated:
                break
        result["status"] = "stopped" if result["stop"] is not None else "complete"
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc)[:1000])
    return result


def write_report(path, report):
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run(plan, output, plan_sha):
    import gymnasium as gym

    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    report = {"schema": "pc_fd_common_physical_dev_result_v1", "plan_sha256": plan_sha,
              "scope": plan["scope"], "expected_branches": 20, "status": "started", "rows": []}
    write_report(path, report)
    env = None
    try:
        setup.install_missing_field_compat()
        import torch
        require(torch.cuda.is_available(), "host CUDA required")
        pf = json.loads((Path(__file__).resolve().parents[2] / "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json").read_text())
        model = setup.load_compatible_pretrained(Path(pf["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        report["model_parameter_hash"] = setup._parameter_hash(model)
        require(report["model_parameter_hash"] == pf["model_parameter_hash_expected"], "model hash mismatch")
        env = setup.ReacherGoalImageWrapper(
            gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
        require(np.array_equal(np.asarray(env.action_space.low), [-1., -1.])
                and np.array_equal(np.asarray(env.action_space.high), [1., 1.]), "live action box mismatch")
        for root in plan["roots"]:
            row = {"seed": root["seed"], "status": "started", "branches": []}
            report["rows"].append(row)
            try:
                packets = np.asarray(root["warmup_packets"], dtype=np.float32)
                require(packets.shape == (3, 5, 2) and digest_actions(packets) == root["warmup_sha256"],
                        "warmup packet hash mismatch")
                ctx = setup.collect_executed_context(
                    env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
                    warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                row["identity_diagnostic"] = {
                    key: {"actual": ctx[actual], "expected": root[expected],
                          "matches": ctx[actual] == root[expected]}
                    for key, actual, expected in (
                        ("root_rgb", "root_rgb_sha256", "root_rgb_sha256"),
                        ("goal_rgb", "goal_rgb_sha256", "goal_rgb_sha256"),
                        ("warmup", "warmup_actions_sha256", "warmup_sha256"))}
                row["observed_root_state"] = common.record(env)
                row["observed_root_rgb"] = context_lib.pack_rgb_observation(ctx["root_frame"])
                row["observed_goal_rgb"] = context_lib.pack_rgb_observation(ctx["goal"])
                # Diagnostic-only run: never dispatch any candidate actions.
                row["status"] = "root_observed_no_branch_execution"
                write_report(path, report)
                continue
                require(ctx["root_rgb_sha256"] == root["root_rgb_sha256"]
                        and ctx["goal_rgb_sha256"] == root["goal_rgb_sha256"]
                        and ctx["warmup_actions_sha256"] == root["warmup_sha256"], "root/warmup identity mismatch")
                row.update(root_rgb_sha256=ctx["root_rgb_sha256"], goal_rgb_sha256=ctx["goal_rgb_sha256"],
                           warmup_actions_sha256=ctx["warmup_actions_sha256"],
                           root_snapshot=ctx["execution_snapshot"])
                for branch in root["branches"]:
                    actions = np.asarray(branch["raw_actions"], dtype=np.float32)
                    require(digest_actions(actions) == branch["actions_sha256"], "branch hash mismatch")
                    outcome = execute_branch(env, root["seed"], ctx, actions)
                    row["branches"].append({**branch, **outcome})
                    write_report(path, report)
                row["status"] = "branches_recorded"
            except Exception as exc:
                row.update(status="root_error", error_type=type(exc).__name__, error=str(exc)[:1000])
            write_report(path, report)
        report["status"] = "finished"
    except Exception as exc:
        report.update(status="execution_error", error_type=type(exc).__name__, error=str(exc)[:1000])
    finally:
        if env is not None:
            env.close()
        report["recorded_branches"] = sum(len(r["branches"]) for r in report["rows"])
        report["complete_branches"] = sum(b["status"] == "complete" for r in report["rows"] for b in r["branches"])
        write_report(path, report)
    return report


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "saved-audit", "model-plan", "preflight", "plan"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--plan-sha256")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    current = prepare(args.model, args.saved_audit, args.model_plan, args.preflight)
    if args.prepare:
        with args.plan.open("x") as file:
            json.dump(current, file, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
        print(json.dumps({"plan": str(args.plan), "sha256": audit.sha(args.plan)}))
        return
    require(args.output is not None and args.plan_sha256 is not None, "output and frozen plan hash required")
    require(audit.sha(args.plan) == args.plan_sha256 and json.loads(args.plan.read_text()) == current,
            "physical plan or source drift")
    result = run(current, args.output, args.plan_sha256)
    print(json.dumps({"status": result["status"], "recorded": result["recorded_branches"],
                      "complete": result["complete_branches"]}))


if __name__ == "__main__":
    main()
