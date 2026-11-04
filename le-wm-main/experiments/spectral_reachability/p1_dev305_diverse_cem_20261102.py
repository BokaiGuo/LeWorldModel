"""Prospective development-root RGB-only comparison with two-packet scoring.

Three stages: capture a fixed root, plan from an RGB/action-only export, execute
the first selected packet in the simulator. This is not an independent P1 test.
"""
import argparse
import hashlib
import importlib
import inspect
import os
from pathlib import Path

import numpy as np

from . import p1_cp10_coherent_replan_20261102 as coherent
from . import p1_second_feedback_physical_20261102 as physical
from . import p1_development_diverse_cem_20261102 as diverse

ROOT = coherent.ROOT
NAME = "p1_dev305_diverse_cem_20261102"
SEED = 301000005
AUTHORITY = "DEV305_ROOT_CAPTURE_SIMULATION"
PLAN_AUTHORITY = "DEV305_BLIND_MODEL_CUDA"
EXEC_AUTHORITY = "DEV305_PAIRED_SIMULATION"
base = physical.previous
require = coherent.require


def terminal_cost(last_prediction, goal):
    """Score the final predicted packet, not the intermediate state."""
    return (last_prediction - goal).square().mean(dim=1)


def specification():
    # Full historical exclusion and checkpoint training membership are unknown.
    source = coherent.base.protocol()
    task = source["task"]
    return {"schema": NAME + ".spec.v1", "seed": SEED,
            "classification": "prospective_development_history_and_training_membership_unknown",
            "root": "native reset then 10 zero actions, observe at 0/5/10",
            "methods": ["original", "coherent"],
            "same_budget": {"cem_iterations": 5, "candidates": 64, "variants": 5,
                            "predict_calls_per_arm": 10, "predict_samples_per_arm": 3200,
                            "execution_steps": 5},
            "source_sha256": {str(Path(__file__).resolve()): base.sha(__file__),
                              str(Path(diverse.__file__).resolve()): base.sha(diverse.__file__),
                              str(Path(coherent.revision.__file__).resolve()): base.sha(coherent.revision.__file__)},
            "policy": source["policy"], "model_hash": source["model_hash"],
            "action_mean": task["action_mean"], "action_std": task["action_std"],
            "latent_dim": task["latent_dim"], "checkpoint_normalizer_provenance": "not_fully_certified",
            "environment_sha256": base.base.native.ENV_SHA,
            "selection_inputs": "three observed RGB frames, RGB goal, executed action history only",
            "selection_objective": "two-packet terminal latent squared error",
            "search_rule": "single population versus two fixed populations; global best scored feasible action",
            "p1_admission": False, "independent_confirmation": False}


def capture(contract, digest, output, authority):
    require(authority == AUTHORITY and base.sha(contract) == digest and base.read(contract) == specification(),
            "capture authority and contract")
    store = base.base.Store(output)
    store.emit("capture_started", contract_sha256=digest)
    env = None
    try:
        cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
        require(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "environment SHA")
        env = cls(render_mode="rgb_array", resolution=224, with_target=False)
        obs, info = env.reset(seed=SEED)
        goal = np.asarray(info["goal"], dtype=np.uint8)
        require(goal.shape == (224, 224, 3), "goal image")
        goal_ref = store.frame("goal.png", goal)
        checkpoints = [base.base.capture(store, env, "root", "warmup", 0)]
        actions = []
        for step in range(1, 11):
            action = np.zeros(2, dtype="<f4")
            obs, _, terminated, truncated, _ = env.step(action.copy())
            require(not terminated and not truncated and np.array_equal(env.latest_action, action), "warmup")
            actions.append(action.tolist())
            if step in (5, 10):
                checkpoints.append(base.base.capture(store, env, "root", "warmup", step))
        # Planner manifest contains only images and actual actions, not state.
        blind = {"schema": NAME + ".blind.v1", "seed": SEED, "actions": actions,
                 "images": [{k: c[k] for k in ("file", "file_sha256", "pixel_sha256")}
                            for c in checkpoints], "goal": goal_ref}
        blind_sha = store.json("blind.json", blind)
        store.emit("capture_finished", blind_sha256=blind_sha)
        store.json("report.json", {"schema": NAME + ".capture.v1", "contract_sha256": digest,
                                   "blind_sha256": blind_sha, "checkpoints": checkpoints,
                                   "goal_state_evaluation_only": np.asarray(info["goal_state"]).tolist(),
                                   "journal_sha256": base.sha(store.path/"journal.jsonl"),
                                   "p1_admission": False, "independent_confirmation": False})
        return base.sha(store.path/"report.json")
    finally:
        if env is not None:
            env.close()


def plan_contract(spec_path, spec_sha, capture_dir, capture_sha):
    require(base.sha(spec_path) == spec_sha and base.read(spec_path) == specification(), "specification")
    capture_dir = Path(capture_dir)
    capture_report = base.read(capture_dir/"report.json", capture_sha)
    require(capture_report["contract_sha256"] == spec_sha
            and base.sha(capture_dir/"journal.jsonl") == capture_report["journal_sha256"], "capture binding")
    blind = base.read(capture_dir/"blind.json", capture_report["blind_sha256"])
    require(blind["seed"] == SEED and len(blind["images"]) == 3
            and np.array_equal(np.asarray(blind["actions"], np.float32), np.zeros((10, 2), np.float32)),
            "blind root inputs")
    refs = []
    for value in (*blind["images"], blind["goal"]):
        path = capture_dir/value["file"]
        pixels = coherent.base.first.rgb._frame(capture_dir, value)
        refs.append({"path": str(path.resolve()), "file_sha256": value["file_sha256"],
                     "pixel_sha256": value["pixel_sha256"]})
    geometry = coherent.base.first.engine._geometry(coherent.base.frame(refs[2]))
    spec = base.read(spec_path)
    return {"schema": NAME + ".plan.v1", "classification": spec["classification"],
            "spec_sha256": spec_sha, "capture_sha256": capture_sha, "blind_sha256": capture_report["blind_sha256"],
            "images": refs[:3], "goal": refs[3], "actions": blind["actions"],
            "contact_direction_raw_xy": geometry["contact_direction_raw_xy"],
            "policy": spec["policy"], "model_hash": spec["model_hash"],
            "action_mean": spec["action_mean"], "action_std": spec["action_std"],
            "latent_dim": spec["latent_dim"], "source_sha256": spec["source_sha256"],
            "p1_admission": False, "independent_confirmation": False}


def plan(spec_path, spec_sha, capture_dir, capture_sha, output, authority):
    require(authority == PLAN_AUTHORITY and not os.path.lexists(output), "plan authority and exclusive output")
    spec = plan_contract(spec_path, spec_sha, capture_dir, capture_sha)
    output = Path(output)
    output.mkdir()
    plan_sha = base.write_new(output/"plan_contract.json", spec)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(spec["policy"], device=device).eval().requires_grad_(False)
    hasher = coherent.base.first.engine.base.blind.host.signal.model_hash
    before = hasher(model)
    require(before == spec["model_hash"], "frozen model")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
    require(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
    ledger, phase, handles, rows = {}, {"name": ""}, [], []
    for name, module in (("encoder", model.encoder), ("projector", model.projector),
                         ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, name=name):
            entry = ledger.setdefault(phase["name"] + ":" + name, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            phase["name"] = "encode"
            encoded = []
            for ref in (*spec["images"], spec["goal"]):
                img = coherent.base.first.engine.base.p0._image_tensor(
                    coherent.base.frame(ref), transform=coherent.base.first.engine.base.p0.image_transform(),
                    device=device)
                emb = model.encode({"pixels": img})["emb"]
                require(emb.shape == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(emb).all()),
                        "actual RGB latent")
                encoded.append(emb[0, 0])
            z0 = torch.stack(encoded[:3])[None]
            goal = encoded[3][None]
            actual = tensor(spec["actions"]).reshape(2, 5, 2)
            history = ((actual - mean) / std).reshape(1, 2, 10)
            for label in ("original", "coherent"):
                phase["name"] = label
                def score(variants):
                    batch = 320
                    flat = tensor(variants.reshape(batch, 2, 5, 2))
                    z, past = z0.repeat(batch, 1, 1), history.repeat(batch, 1, 1)
                    for packet in range(2):
                        context = torch.cat((past, ((flat[:, packet] - mean)/std).reshape(batch, 1, 10)), dim=1)
                        prediction = model.predict(z, model.action_encoder(context))
                        require(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()), "prediction")
                        z = torch.cat((z, prediction[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                    return terminal_cost(z[:, -1], goal).reshape(64, 5).cpu().numpy()
                result = diverse.plan(spec["contact_direction_raw_xy"], score,
                                      structured=label == "coherent",
                                      common=coherent.base.first.engine.base.common,
                                      seed=coherent.base.first.planner.parent.SEED)
                selected = np.asarray(result["selected"]["actions"], dtype="<f4")
                require(selected.shape == (2, 5, 2) and np.isfinite(selected).all()
                        and (np.abs(selected) <= 1).all(), "selected action")
                row = {"arm": label, "selected_actions_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                       **result}
                rows.append(row)
                base.write_new(output/(label+".json"), row)
        torch.cuda.synchronize(device)
        after = hasher(model)
        require(before == after and len(rows) == 2, "frozen model and roster")
        for name in ("encoder", "projector"):
            require(ledger["encode:"+name] == {"calls": 4, "samples": 4}, "encoder ledger")
        for label in ("original", "coherent"):
            for name in ("action_encoder", "predictor"):
                require(ledger[label+":"+name] == {"calls": 10, "samples": 3200}, "planner ledger")
        require(spec == plan_contract(spec_path, spec_sha, capture_dir, capture_sha), "postrun source")
        return base.write_new(output/"report.json", {"schema": NAME + ".plan_report.v1",
                 "plan_contract_sha256": plan_sha, "capture_sha256": capture_sha,
                 "model_before": before, "model_after": after, "host": host, "rows": rows, "ledger": ledger,
                 "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False})
    except Exception as exc:
        base.write_new(output/"failure.json", {"type": type(exc).__name__, "message": str(exc),
                                               "phase": phase["name"], "rows": len(rows)})
        raise
    finally:
        for h in handles:
            h.remove()


def execute(spec_path, spec_sha, capture_dir, capture_sha, plan_dir, plan_sha, output, authority):
    require(authority == EXEC_AUTHORITY and base.sha(spec_path) == spec_sha
            and base.read(spec_path) == specification(), "execution authority")
    capture_dir, plan_dir = Path(capture_dir), Path(plan_dir)
    capture_report = base.read(capture_dir/"report.json", capture_sha)
    report = base.read(plan_dir/"report.json", plan_sha)
    plan_spec = plan_contract(spec_path, spec_sha, capture_dir, capture_sha)
    require(base.read(plan_dir/"plan_contract.json", report["plan_contract_sha256"]) == plan_spec
            and report["capture_sha256"] == capture_sha and report["model_before"] == report["model_after"]
            and [r["arm"] for r in report["rows"]] == ["original", "coherent"], "plan binding")
    store = base.base.Store(output)
    store.emit("execute_started", plan_sha256=plan_sha)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    require(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "env SHA")
    rows, errors = [], []
    for plan_row in report["rows"]:
        arm = plan_row["arm"]
        row = {"arm": arm, "status": "running", "transitions": [], "observations": []}
        env = None
        try:
            actions = np.asarray(plan_row["selected"]["actions"], dtype="<f4")
            require(actions.shape == (2, 5, 2) and np.isfinite(actions).all()
                    and hashlib.sha256(actions.tobytes()).hexdigest() == plan_row["selected_actions_sha256"],
                    "action SHA")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            obs, info = env.reset(seed=SEED)
            require(np.array_equal(obs["state"], capture_report["checkpoints"][0]["state"])
                    and np.array_equal(info["goal_state"], capture_report["goal_state_evaluation_only"]),
                    "reset")
            row["observations"].append(base.base.capture(store, env, arm, "warmup", 0,
                                                          capture_report["checkpoints"][0]))
            for step in range(1, 11):
                action = np.zeros(2, dtype="<f4")
                obs, reward, terminated, truncated, _ = env.step(action.copy())
                require(not terminated and not truncated and np.array_equal(env.latest_action, action), "warmup")
                if step in (5, 10):
                    row["observations"].append(base.base.capture(
                        store, env, arm, "warmup", step, capture_report["checkpoints"][[5, 10].index(step)+1]))
            for step, action in enumerate(actions[0], 1):
                obs, reward, terminated, truncated, _ = env.step(action.copy())
                require(np.array_equal(env.latest_action, action), "action")
                row["transitions"].append({"step": step, "action": action.tolist(),
                                           "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                           "terminated": bool(terminated), "truncated": bool(truncated)})
                row["observations"].append(base.base.capture(store, env, arm, "execute", step))
                if terminated or truncated:
                    row["status"] = "native_early_stop"
                    break
            else:
                row["status"] = "fixed_horizon_completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"arm": arm, **row["error"]})
        finally:
            if env is not None:
                env.close()
        store.json(arm+".row.json", row)
        store.emit("branch_closed", arm=arm, status=row["status"])
        rows.append(row)
    store.emit("execute_finished", errors=errors)
    store.json("report.json", {"schema": NAME + ".execution.v1", "status": "recorded" if not errors else "failed",
                               "plan_sha256": plan_sha, "capture_sha256": capture_sha, "rows": rows,
                               "errors": errors, "journal_sha256": base.sha(store.path/"journal.jsonl"),
                               "p1_admission": False, "independent_confirmation": False})
    return base.sha(store.path/"report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "capture", "plan", "execute"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--spec-sha")
    parser.add_argument("--capture-sha")
    parser.add_argument("--plan-sha")
    parser.add_argument("--authority")
    a = parser.parse_args()
    require(a.tag.isascii() and a.tag.isalnum(), "tag")
    spec = ROOT/"results"/f"{NAME}_PREP_{a.tag}.json"
    capture_dir = ROOT/"results"/f"{NAME}_CAPTURE_{a.tag}"
    plan_dir = ROOT/"results"/f"{NAME}_PLAN_{a.tag}"
    execution = ROOT/"results"/f"{NAME}_EXEC_{a.tag}"
    if a.command == "prepare":
        print(base.write_new(spec, specification()))
    elif a.command == "capture":
        print(capture(spec, a.spec_sha, capture_dir, a.authority))
    elif a.command == "plan":
        with coherent.base.first.engine.base.locking.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
            print(plan(spec, a.spec_sha, capture_dir, a.capture_sha, plan_dir, a.authority))
    else:
        print(execute(spec, a.spec_sha, capture_dir, a.capture_sha, plan_dir, a.plan_sha, execution, a.authority))


if __name__ == "__main__":
    main()
