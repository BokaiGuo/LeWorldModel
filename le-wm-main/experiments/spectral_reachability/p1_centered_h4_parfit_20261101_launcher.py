"""Fresh-process translated-root H4 launcher. No CUDA without explicit run."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import p1_centered_h4_parfit_20261101_adapter as adapter
from . import p1_reverse_h4_parfit_20261101 as h4

NAME = "p1_centered_h4_parfit_20261101_launcher"
PROTOCOL = adapter.ROOT.parent / "docs/D2P_P1_CENTERED_CONTROL_PROTOCOL_20261101_MAIN.md"


def source_inputs():
    report, target, _, pixels = adapter.load()
    images = []
    sources = {str(p.resolve()): adapter.sha(p) for p in
               (Path(__file__), Path(adapter.__file__), adapter.TARGET, adapter.METRIC,
                adapter.REPORT, PROTOCOL)}
    for frame in report["history"]:
        path = Path(frame["png"])
        adapter.require(adapter.sha(path) == frame["file_sha256"], "history PNG file SHA")
        image = np.asarray(Image.open(path).convert("RGB")).copy()
        adapter.require(image.shape == (224, 224, 3) and
                        hashlib.sha256(image.tobytes()).hexdigest() == frame["pixel_sha256"],
                        "history PNG pixels")
        images.append(image)
        sources[str(path)] = frame["file_sha256"]
    for p, digest in json.loads(adapter.TARGET.read_bytes())["sources"].items():
        adapter.require(adapter.sha(p) == digest, "target source drift")
        sources[p] = digest
    for p, digest in json.loads(adapter.METRIC.read_bytes())["sources"].items():
        adapter.require(adapter.sha(p) == digest, "metric source drift")
        sources[p] = digest
    return report, target, pixels, np.stack(images), sources


def setup(arm):
    report, target, pixels, images, sources = source_inputs()
    reverse = h4.reverse
    reverse.METRIC, reverse.METRIC_SHA = adapter.METRIC, adapter.METRIC_SHA
    reverse.metric_definition = SimpleNamespace(TARGET=adapter.TARGET, TARGET_SHA=adapter.TARGET_SHA)

    def task_setup(root, selected_arm, target_path, target_sha):
        adapter.require(root == 0 and selected_arm == arm and Path(target_path) == adapter.TARGET
                        and target_sha == adapter.TARGET_SHA, "centered task parameters")
        experiment = reverse.Q.parent.protected.setup(root)
        runner = experiment.runner
        old_roster = runner.rgb.roster
        runner.rgb.roster = lambda: [r for r in old_roster() if r["arm"] == arm]

        def inputs():
            entry = dict(root=0, root_state=deepcopy(target["root_state"]),
                         goal_state=deepcopy(target["goal_state"]), pixels=pixels.copy(),
                         pixel_sha256=target["pixel_sha256"])
            return entry, deepcopy(target), dict(sources)

        experiment.inputs = inputs
        return experiment

    reverse.task_setup = task_setup
    experiment = h4.setup(arm)
    runner = experiment.runner
    moving = runner.rgb.moving
    old_build, old_protocol = runner.build, runner.protocol
    old_encode = moving.encode_boundary
    experiment.rebind = lambda prepared, data: rebind(prepared, data, runner, report, target, images)

    def physical_setup(env, ref, source_plan, row, meter, store):
        del source_plan
        old = moving.old
        body = env.unwrapped
        meter.region = "setup"
        env.reset(seed=report["reset_seed"])
        obs, _ = env.reset(seed=0, options=dict(
            state=np.asarray(report["requested_reset"], np.float64).copy(),
            goal_state=np.asarray(report["placeholder_goal"], np.float64).copy()))
        observed, snapshots = [], []

        def capture(step):
            image, receipt = old.capture(env, obs, row, store, "warmup", step)
            if step in (0, 5, 10):
                i = (0, 5, 10).index(step)
                runner.require(np.array_equal(image, ref["history"][i])
                               and np.allclose(obs["state"], ref["states"][i], rtol=0, atol=1e-7),
                               "translated actual warmup RGB/state")
                observed.append(np.asarray(image).copy())
                snapshots.append(receipt["snapshot"])

        capture(0)
        meter.region = "warmup"
        for step, command in enumerate(old.kernel.raw_history(ref["raw_history"]), 1):
            obs, _, terminated, truncated, _ = env.step(command.copy())
            row["warmup_returned"] += 1
            capture(step)
            value = np.asarray(body.latest_action).copy()
            runner.require(old.kernel.moving.exact(value, command), "warmup action bits")
            store.emit("warmup_return", episode_id=row["episode_id"], step=step,
                       command=value.tolist(), terminated=bool(terminated), truncated=bool(truncated))
            if terminated or truncated:
                row.update(status="native_stopped_warmup", terminated=bool(terminated),
                           truncated=bool(truncated))
                return False
        runner.require(len(observed) == len(snapshots) == 3, "actual history snapshots")
        ref["dynamics"] = deepcopy(snapshots)
        meter.region = "goal_bind"
        state, pose = old.kernel.moving.snapshot(body), np.asarray(body.goal_pose).copy()
        meter.call("goal_bind", lambda: old.kernel.moving.attach_goal(body, ref["goal"], ref["goal_rgb"]))
        runner.require(old.kernel.moving.snapshot(body) == state and np.array_equal(pose, body.goal_pose),
                       "goal bind changed state")
        _, obs0 = old.capture(env, obs, row, store, "plan", 0)
        runner.require(obs0["pixel_sha256"] == target["history_pixel_sha256"][-1]
                       and moving.source.root_hash(moving.source.vector(obs["state"]), ref["goal"])
                       == ref["root_sha256"], "translated root binding")
        row["initial_exact"] = True
        store.put(row["episode_id"] + "_translated_history_binding.json",
                  dict(pixel_sha256=target["history_pixel_sha256"],
                       dynamics=snapshots, target_sha256=adapter.TARGET_SHA))
        return True

    def encode_boundary(handler, runtime, task, layer, row, obs, body, meter, store, step):
        z, receipt = old_encode(handler, runtime, task, layer, row, obs, body, meter, store, step)
        if step == 0:
            runner.require(np.array_equal(np.asarray(z, np.float32),
                                          np.asarray(target["history_latents"][-1], np.float32)),
                           "translated root re-encoding mismatch")
        return z, receipt

    def protocol():
        value = old_protocol()
        value.update(schema=NAME + ".protocol.v1", goal="translated_witness_final_rgb",
                     translated_known_root=True, independent_confirmation=False, P1_admission=False,
                     target_receipt_sha256=adapter.TARGET_SHA, metric_sha256=adapter.METRIC_SHA,
                     actual_history_rgb_sha256=target["history_pixel_sha256"],
                     witness_actions_used_in_planner=False)
        return value

    def build():
        value = old_build()
        runner.require(value["schema"] == h4.NAME + ".prepare.v1", "shared H4 prepare ABI")
        value["centered_task_schema"] = NAME + ".task.v1"
        value["sources"].update(sources)
        value["protocol"] = protocol()
        return value

    moving.setup, moving.encode_boundary = physical_setup, encode_boundary
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_{arm}"
    return experiment


def rebind(prepared, data, runner, report, target, images):
    # Called on copies by the existing cp10 execute wrapper.
    ref, task = data["references"][0], prepared["source"]["tasks"][0]
    raw = np.asarray(report["raw_history"], np.float32)
    runner.require(np.array_equal(raw, np.asarray(ref["raw_history"], np.float32)),
                   "translated warmup action identity")
    normalized = runner.rgb.moving.normalized(task, raw)
    runner.require(normalized == task["root_context"]["action_history"],
                   "original normalized action history changed")
    ref["raw_history"] = raw.copy()
    ref["history"] = images.copy()
    ref["states"] = np.asarray([r["state"] for r in report["history"]], np.float64)
    ref["dynamics"] = None  # replaced only by actual reset/warmup snapshots
    ref["goal"] = np.asarray(target["goal_state"], np.float64)
    ref["goal_rgb"] = np.frombuffer(
        __import__("base64").b64decode(target["pixels_uint8_base64"], validate=True),
        np.uint8).reshape(224, 224, 3).copy()
    ref["root_sha256"] = runner.p3.moving.source.root_hash(ref["states"][-1], ref["goal"])
    task["goal"] = deepcopy(target["latent"])
    task["root_context"]["latent_history"] = deepcopy(target["history_latents"])
    task["root_context"]["action_history"] = normalized
    binding = task["binding"]
    binding["goal_sha256"] = runner.digest(task["goal"])
    binding["context_sha256"] = runner.digest(task["root_context"])
    binding["root_sha256"] = ref["root_sha256"]
    binding["task_id"] = runner.digest([NAME, binding["task_id"], adapter.TARGET_SHA])
    binding["task_sha256"] = runner.digest([NAME, "task", binding["task_id"]])
    binding["task_contract_sha256"] = runner.digest([NAME, "contract", adapter.TARGET_SHA, adapter.METRIC_SHA])
    runner.p3.core.validate_task(task)
    return dict(goal_pixel_sha256=target["pixel_sha256"], root_sha256=ref["root_sha256"],
                goal_latent_sha256=runner.digest(task["goal"]), initial_success=False,
                translated_history_pixel_sha256=target["history_pixel_sha256"],
                normalized_action_history_sha256=runner.digest(normalized))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=h4.h4.ARMS, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    experiment = setup(args.arm)
    runner = experiment.runner
    if args.command == "dry-prepare":
        value = runner.build()
        data = runner.p3.source_data(value["source"])
        rebound = deepcopy(value)
        report, target, _, images, _ = source_inputs()
        probe = rebind(rebound, data, runner, report, target, images)
        control = h4.reverse.Q.carry.control
        contracts = {}
        for boundary in (0, 5, 10, 15):
            with patch.dict(control.base.CONFIG, h4.boundary_config(boundary)), \
                    patch.object(control, "cost", lambda a: h4.h4.node_cost(a, boundary)):
                policy, contract, config = control.contract_for(
                    rebound["source"]["tasks"][0], "a" * 64, "b" * 64)
            runner.require(contract["levels"][0]["caps"] ==
                           value["protocol"]["node_caps_by_boundary"][str(boundary)]
                           and policy["horizon"] == h4.h4.dimensions(boundary)[0]
                           and config.items() >= h4.boundary_config(boundary).items()
                           and control.spec.digest(config) == contract["identity"]["optimizer_config"]
                           and control.spec.digest(policy) == contract["phase_policy_sha256"],
                           "H4 node contract/caps")
            contracts[str(boundary)] = control.spec.digest(contract)
        result = dict(status="dry_prepared_not_written", schema=value["schema"],
                      build_sha256=runner.digest(value), binding=probe,
                      dry_contract_sha256=contracts, protocol=value["protocol"], cuda_used=False)
    elif args.command == "prepare":
        result = runner.prepare(args.output)
    else:
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
