"""Known-root cp5 CEM host. Preparation and import perform no inference."""
import argparse
import json
import os
import platform
import uuid
from pathlib import Path

import numpy as np

from . import reacher_pc_p1_fd_cp5_cem_bridge_20261102_codexexclusive as bridge
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock
from . import pc_ap_durable_context_adapter_20261102_codexexclusive as durable
from . import reacher_d2p_context as context_lib

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
UUID = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"
AUTHORITY = "COORDINATED_FD_CP5_CEM_DEVELOPMENT"
HERE = Path(__file__).resolve()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_gpu_uuid(value):
    """Canonicalize PyTorch's bare UUID and nvidia-smi's GPU-prefixed UUID."""
    return str(uuid.UUID(str(value).removeprefix("GPU-")))


def plan(model, audit, physical_plan, preflight):
    paths = [Path(p).resolve(strict=True) for p in (model, audit, physical_plan, preflight)]
    fixed = bridge.prepare(*paths[:3])
    for root in fixed["roots"]:
        zero = root["arms"][1]
        require(zero["arm"] == "parameter_zero" and zero["proposal_index"] == 0
                and np.array_equal(np.asarray(zero["params"], dtype=np.float32), np.zeros((1, 10), np.float32)),
                "bank[0] is not parameter zero")
    pf = json.loads(paths[3].read_text())
    weights, stats = Path(pf["weights_path"]).resolve(strict=True), Path(pf["action_stats_path"]).resolve(strict=True)
    require(bridge.sha_file(weights) == pf["weights_sha256"]
            and bridge.sha_file(stats) == pf["action_stats_sha256"]
            and pf["model_parameter_hash_expected"] == fixed["parameter_hash_expected"], "model or stats mismatch")
    base = HERE.parent
    dependencies = [HERE, Path(bridge.__file__), Path(lock.__file__), Path(durable.__file__),
                    Path(context_lib.__file__)]
    for name in ("preflight_reacher_d2p_p1_reconfiguration.py", "run_reacher_d2p_p1_reconfiguration.py",
                 "run_reacher_d2p_p0_goal_progress.py", "reacher_d2p_packet_support.py"):
        dependencies.append(base / name)
    all_paths = paths + [weights, stats] + dependencies
    return {"schema": "fd_cp5_host_v1", "inputs": [str(p) for p in paths],
            "files": {str(p): bridge.sha_file(p) for p in all_paths},
            "weights": str(weights), "stats": str(stats), "bridge": fixed,
            "gpu_uuid": UUID, "lock": str(LOCK), "python": platform.python_version(),
            "source_closure_complete": False, "P1_admission": False,
            "physical_outcome_used_for_selection": False,
            "cost_scope": "each arm has prefix, extra zero-prefix control, suffix, two successor encodings and other model work; partial counts can be unknown; not equal total compute"}


def write_new(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def capture(frames, root, goal, actions):
    frames = [np.asarray(x, dtype=np.uint8) for x in frames]
    root, goal = np.asarray(root, dtype=np.uint8), np.asarray(goal, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.float32)
    require(len(frames) == 3 and actions.shape == (3, 10) and np.isfinite(actions).all(),
            "unexpected planner context")
    return {"frames": [context_lib.pack_rgb_observation(x) for x in frames[1:] + [root]],
            "goal": context_lib.pack_rgb_observation(goal), "history": actions.tolist()}


def instrument(runner, journal, seed):
    """Wrap actual _prepare calls; persist intent before the physical arm."""
    original_two_stage, original_prepare = runner._two_stage, runner.p0._prepare

    def wrapped(**kwargs):
        arm = kwargs["name"]
        journal.emit({"kind": "arm_started", "seed": seed, "arm": arm})
        count = 0
        env = kwargs["env"]
        original_step = env.step
        step_count = 0

        def recorded_step(action):
            nonlocal step_count
            index = step_count
            step_count += 1
            journal.emit({"kind": "step_intent", "seed": seed, "arm": arm, "index": index,
                          "action": np.asarray(action, dtype=np.float32).tolist()})
            returned = original_step(action)
            journal.emit({"kind": "step_returned", "seed": seed, "arm": arm, "index": index,
                          "terminated": bool(returned[2]), "truncated": bool(returned[3])})
            return returned

        def recorded(preprocessor, frames, root, goal, actions, device):
            nonlocal count
            labels = ("root", "successor", "zero_successor")
            require(count < len(labels), "unexpected prepare count")
            journal.emit({"kind": "context", "seed": seed, "arm": arm,
                          "role": labels[count], "value": capture(frames, root, goal, actions)})
            count += 1
            return original_prepare(preprocessor, frames, root, goal, actions, device)

        runner.p0._prepare = recorded
        env.step = recorded_step
        try:
            result = original_two_stage(**kwargs)
            require(count == 3, "missing planner context")
            journal.emit({"kind": "arm_returned", "seed": seed, "arm": arm, "result": result})
            return result
        except Exception as exc:
            journal.emit({"kind": "arm_error", "seed": seed, "arm": arm,
                          "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            raise
        finally:
            runner.p0._prepare = original_prepare
            env.step = original_step

    runner._two_stage = wrapped
    return original_two_stage


def run(plan_path, expected, output, authority):
    require(authority == AUTHORITY and os.environ.get("CUDA_VISIBLE_DEVICES") == UUID,
            "coordinated authority and visible UUID required")
    require(bridge.sha_file(plan_path) == expected, "plan hash")
    saved = json.loads(Path(plan_path).read_text())
    require(saved == plan(*saved["inputs"]), "plan/source drift")
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results" and
            output.name.startswith("reacher_fd_cp5_host_") and not os.path.lexists(output),
            "exclusive results directory required")
    with lock.gpu_lock(LOCK):
        import gymnasium as gym
        import torch
        from experiments.gpu_preflight import require_cuda_allocation
        from . import preflight_reacher_d2p_p1_reconfiguration as setup
        from . import run_reacher_d2p_p1_reconfiguration as runner

        allocation = require_cuda_allocation("cuda:0")
        require(torch.cuda.device_count() == 1 and
                canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) ==
                canonical_gpu_uuid(UUID), "CUDA UUID mismatch")
        output.mkdir(mode=0o700)
        write_new(output / "plan.json", saved)
        journal = durable.Journal(output / "events.jsonl")
        env = None
        try:
            journal.emit({"kind": "run_started", "plan_sha256": expected, "gpu_uuid": UUID,
                          "cuda_allocation": allocation})
            setup.install_missing_field_compat()
            stats = Path(saved["stats"])
            setup.validate_training_semantic_receipt(stats)
            mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
            model = setup.load_compatible_pretrained(Path(saved["weights"]).parent, device="cuda").eval()
            model.interpolate_pos_encoding = True
            model.requires_grad_(False)
            before = setup._parameter_hash(model)
            require(before == saved["bridge"]["parameter_hash_expected"], "parameter hash")
            processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
            env = setup.ReacherGoalImageWrapper(
                gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
            setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
            for root in saved["bridge"]["roots"]:
                journal.emit({"kind": "root_started", "seed": root["seed"]})
                original = instrument(runner, journal, root["seed"])
                try:
                    result = bridge.run_root(root, model=model, env=env, processor=processor, mean=mean,
                                             std=std, lower=lower, upper=upper, device="cuda",
                                             parameter_hash_expected=before)
                    journal.emit({"kind": "root_returned", "seed": root["seed"], "result": result})
                except Exception as exc:
                    journal.emit({"kind": "root_error", "seed": root["seed"],
                                  "error_type": type(exc).__name__, "error": str(exc)[:1000]})
                finally:
                    runner._two_stage = original
            after = setup._parameter_hash(model)
            require(before == after and saved == plan(*saved["inputs"]), "post-run identity drift")
            journal.emit({"kind": "run_finished", "model_before": before, "model_after": after})
        except Exception as exc:
            journal.emit({"kind": "run_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            raise
        finally:
            if env is not None:
                env.close()
            journal.close()
    return {"journal_sha256": bridge.sha_file(output / "events.jsonl"), "output": str(output)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for name in ("model", "audit", "physical-plan", "preflight", "output"):
        prep.add_argument("--" + name, required=True)
    launch = sub.add_parser("run")
    for name in ("plan", "sha256", "output", "authority"):
        launch.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        write_new(args.output, plan(args.model, args.audit, args.physical_plan, args.preflight))
        print(json.dumps({"plan": args.output, "sha256": bridge.sha_file(args.output)}))
    else:
        print(json.dumps(run(args.plan, args.sha256, args.output, args.authority)))


if __name__ == "__main__":
    main()
