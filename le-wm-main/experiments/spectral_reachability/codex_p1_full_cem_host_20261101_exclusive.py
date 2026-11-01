"""Reviewed-preparation candidate: full CEM through existing P1 host interfaces.

No execution occurs on import. Actual run requires an explicit authorization and
the existing cooperative GPU lock. CPU dry preparation does not run physics.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import codex_p1_full_cem_controller_20261101_exclusive as cem
from . import codex_p1_full_cem_source_20261101_exclusive as source
from . import codex_four_arm_common_suffix_control_20261030 as common
from . import codex_four_arm_durable_20261030 as durable
from . import codex_four_arm_leaf_suffix_20261030 as leafmod
from . import codex_four_arm_consumer_20261030 as consumer
from . import codex_four_arm_full_cost_20261030 as spec
from . import e3_cp10_four_arm_root0_20261031 as root0
from . import e3_cp10_four_arm_root1_20261031 as root1
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as gpu
from .codex_e1_proposal_durable_events_20261101 import read_events

NAME = "codex_p1_full_cem_host_20261101_exclusive"
AUTH = "MAIN_AUTHORIZES_P1_FULL_CEM_HOST_CUDA_AND_PHYSICS"
SEED = 20261101
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_FULL_CEM_HOST_20261101_EXCLUSIVE.md"
TEST = Path(__file__).resolve().parents[2] / "tests/test_codex_p1_full_cem_host_20261101_exclusive.py"


def node_cost(boundary):
    spec.require(boundary in cem.BOUNDARIES, "boundary")
    horizon = cem.HORIZONS[cem.BOUNDARIES.index(boundary)]
    result = dict.fromkeys(spec.AXES, 0)
    calls = (cem.ITERATIONS * cem.SAMPLES + 1) * horizon
    result.update(action_encode_calls=calls, predict_calls=calls, predict_samples=calls)
    return result


def node_contract(task, model_sha, hardware_sha, boundary):
    _, contract, _ = common.contract_for(task, model_sha, hardware_sha)
    contract = deepcopy(contract)
    config = {"seed": SEED, "boundary": boundary, "horizon": cem.HORIZONS[cem.BOUNDARIES.index(boundary)],
              "iterations": cem.ITERATIONS, "samples": cem.SAMPLES, "elites": cem.ELITES}
    optimizer = common.spectral.file_sha(cem.__file__)
    config_sha = spec.digest(config)
    contract["identity"]["optimizer"] = optimizer
    contract["identity"]["optimizer_config"] = config_sha
    contract["phase_policy_sha256"] = spec.digest({"kind": "full_cem", **config})
    for arm in contract["arms"].values():
        arm.update(initializer_sha256=optimizer, optimizer_sha256=optimizer,
                   optimizer_config_sha256=config_sha)
    caps = node_cost(boundary)
    contract["levels"] = [{"id": "node", "caps": caps},
                          {"id": "unused_larger", "caps": {k: v + 1 for k, v in caps.items()}}]
    spec.preflight(contract, spec.digest(contract))
    return contract


def fixed_identity_check(task, identity):
    goal = torch.tensor(task["goal"], dtype=torch.float32).unsqueeze(0)
    normalizer = torch.tensor([task["action_mean"], task["action_std"]], dtype=torch.float32)
    matrix = torch.tensor(task["projection"], dtype=torch.float64)
    spec.require(cem.tensor_sha(goal) == identity["goal_sha256"]
                 and cem.tensor_sha(normalizer) == identity["normalizer_sha256"]
                 and cem.tensor_sha(matrix) == identity["metric_sha256"],
                 "actual task goal/normalizer/projection mismatch")


def verify_node_events(path, boundary):
    rows = [row["payload"]["event"] for row in read_events(path)]
    attempts = [(i, event) for i, event in enumerate(rows)
                if event.get("type") == "attempt" and event.get("operation") == "rollout"
                and event.get("details", {}).get("boundary") == boundary]
    returned = {event["attempt_index"] for event in rows if event.get("type") == "return"}
    scores = [event for event in rows if event.get("type") == "score" and event.get("boundary") == boundary]
    banks = [event for event in rows if event.get("type") == "bank" and event.get("boundary") == boundary]
    replay = [event for event in rows if event.get("type") == "winner_replay"
              and event.get("boundary") == boundary]
    count = cem.ITERATIONS * cem.SAMPLES
    spec.require(len(attempts) == count + 1 and all(i in returned for i, _ in attempts)
                 and [e["index"] for e in scores] == list(range(count))
                 and len(banks) == cem.ITERATIONS and len(replay) == 1,
                 "incomplete CEM event accounting")


class LiveLeaf:
    """Bridges CEM calls to the active node's existing budgeted LeafAdapter."""
    def __init__(self, runtime, encode):
        self.runtime, self.encode_current, self.node = runtime, encode, None

    def encode(self, info, *, phase):
        self.runtime["check_limits"]()
        return self.encode_current()

    def action_encode(self, actions, *, phase):
        self.runtime["check_limits"]()
        spec.require(self.node is not None, "node not active")
        return self.node.action_encode(actions, phase=phase)

    def predict(self, states, actions, *, phase):
        self.runtime["check_limits"]()
        spec.require(self.node is not None, "node not active")
        return self.node.predict(states, actions, phase=phase)


def execute_one(prepared, data, store, visual, runtime, factory, runner, rebind, root):
    prepared, data = deepcopy(prepared), deepcopy(data)
    actual_binding = rebind(prepared, data)
    expected = source.inspect(root)
    spec.require(actual_binding == expected["rebound_goal_binding"], "rebound root/target mismatch")
    base, ref, layer = (prepared["source"]["tasks"][root], data["references"][root],
                        prepared["source"]["layers"][root])
    fixed_identity_check(base, expected["identity"])
    spec.require(ref["root_sha256"] == expected["identity"]["root_sha256"]
                 and base["binding"]["root_sha256"] == expected["identity"]["root_sha256"]
                 and base["binding"]["task_sha256"] == expected["identity"]["task_sha256"]
                 and runtime["model_sha256"] == expected["identity"]["model_sha256"],
                 "fixed root/task/model mismatch")
    rgb, p3, old = runner.rgb, runner.p3, runner.old
    moving = rgb.moving
    row = deepcopy(p3.roster()[0])
    row.update(episode_id=f"s{root}_full_cem", source_slot=root, mode="native", arm="native")
    nodes = rgb.node_roster(row)
    report = {"schema": NAME + ".report.v1", "status": "partial_or_failed", "rows": [row],
              "nodes": {row["episode_id"]: nodes}, "errors": [], "protocol": runner.protocol(),
              "P1_admission": False, "fairness_proven": False, "freshness": False,
              "independent_confirmation": False, "cp10_goal_binding": actual_binding}
    data["live_report"] = report
    store.put("roster.json", {"rows": [row], "nodes": report["nodes"]})
    store.emit("run_started", contract=runner.protocol())
    meter = p3.Meter(store, runtime["check_limits"])
    meter.episode, meter.region = row["episode_id"], "setup"
    env = controller = active_driver = None
    active_boundary = None
    try:
        env = meter.call("constructor", lambda: factory(meter))
        body = env.unwrapped
        spec.require(body._live_meter is meter and body._live_constructor_complete is True
                     and body.with_target is False and body.render_action is False
                     and body.render_size == 224 and body.relative is True
                     and body.action_scale == 100 and body.control_hz == 10 and body.dt == .01,
                     "native environment invariants")
        row["status"] = "running"
        if not moving.setup(env, ref, data["plan"], row, meter, store):
            raise ValueError("native stopped during warmup; retained without planning")
        # setup has authenticated reset/warmup RGB, dynamics, current state,
        # rebound target and root using the same driver as the comparison.
        spec.require(row["initial_exact"] is True, "root setup identity")
        store.emit("full_cem_root_verified", root_sha256=ref["root_sha256"],
                   goal_binding=actual_binding)
        context = deepcopy(base["root_context"])
        observed = row["last_observation"]
        node_state_seal = None

        def encode_current():
            z, observation = moving.encode_boundary(visual, runtime.get("visual_runtime", runtime),
                data["visual_tasks"][0], layer, row, observed, body, meter, store, active_boundary)
            return z, observation

        leaf = LiveLeaf(runtime, encode_current)

        def observe(boundary, metered):
            nonlocal context, active_driver, active_boundary, node_state_seal
            spec.require(active_driver is None and row["control_returned"] == boundary,
                         "one node and actual feedback boundary")
            active_boundary = boundary
            node = next(n for n in nodes if n["boundary"] == boundary)
            node["status"] = "attempted"
            meter.region = "encoding"
            z, observation = metered.encode({}, phase="setup")
            raw10 = np.concatenate((ref["raw_history"],
                       np.asarray(row["executed_raw"], np.float32).reshape(-1, 2)))[-10:]
            if boundary == 0:
                context = deepcopy(context)
                context["latent_history"][-1] = z.tolist()
                context["action_history"] = moving.normalized(base, raw10)
            else:
                context = moving.roll(context, z, raw10, base)
            local_identity = {"episode_id": row["episode_id"], "arm": row["arm"],
                              "boundary": boundary, "raw10": raw10.tolist(),
                              "observation": observation}
            task = p3.local_task(base, context, local_identity)
            fixed_identity_check(task, expected["identity"])
            node_state_seal = old.state_seal(body)
            contract = node_contract(task, runtime["model_sha256"], runtime["hardware_sha256"], boundary)
            ns = store.namespace(row["episode_id"] + f"_b{boundary:02}_full_cem")
            active_driver = durable.Driver(ns / "ledger", contract, spec.digest(contract),
                                          "native", "node", row["episode_id"])
            leaf.node = leafmod.LeafAdapter(runtime["model"], active_driver, contract, spec.digest(contract))
            node["contract_sha256"] = spec.digest(contract)
            node["ledger_contract"] = contract
            node["task_sha256"] = spec.digest(task)
            store.put(row["episode_id"] + f"_b{boundary:02}_context.json",
                      {"identity": local_identity, "task_sha256": spec.digest(task),
                       "contract_sha256": spec.digest(contract)})
            tensor = lambda value: torch.tensor(value, dtype=torch.float32, device=runtime["device"])
            return {"identity": expected["identity"], "rgb_sha256": observation["pixel_sha256"],
                    "provenance": observation["encoding_sha256"], "encoding_mode": "fresh_metered",
                    "states": tensor(context["latent_history"])[None],
                    "history": tensor(context["action_history"])[None], "goal": tensor(task["goal"])[None],
                    "mean": tensor(task["action_mean"]), "std": tensor(task["action_std"]),
                    "matrix": torch.tensor(task["projection"], dtype=torch.float64, device=runtime["device"])}

        def issue(packet):
            nonlocal observed, active_driver
            boundary = active_boundary
            node = next(n for n in nodes if n["boundary"] == boundary)
            spec.require(old.state_seal(body) == node_state_seal, "planner changed physical environment")
            ledger = active_driver.finish()
            contract = node["ledger_contract"]
            account = consumer.consume(ledger["journal"], ledger["sha256"], spec.digest(contract),
                                       "native", "node", row["episode_id"])
            spec.require(account["status"] == "closed_unverified"
                         and account["charged_upper_bounds"] == node_cost(boundary),
                         "node budget settlement")
            verify_node_events(controller.journal.path, boundary)
            active_driver, leaf.node = None, None
            node.update(status="sealed", receipt={"ledger": ledger, "account": account,
                                                  "commands": packet.tolist()})
            store.put(row["episode_id"] + f"_b{boundary:02}_seal.json", node["receipt"])
            store.emit("boundary_sealed", episode_id=row["episode_id"], boundary=boundary,
                       plan_sha256=spec.digest(packet.tolist()))
            before = row["control_returned"]
            observed = moving.packet(env, packet.tolist(), start=boundary, row=row, ref=ref,
                                     meter=meter, store=store)
            steps = row["control_returned"] - before
            if steps == 5:
                row["completed_packets"] += 1
            return {"executed_steps": steps, "stopped": observed is None,
                    "status": row["status"], "physical_events": "main_store_journal"}

        controller = cem.Controller(leaf=leaf, observe=observe, issue=issue,
            journal=store.path / "full_cem_events.jsonl", run_id=row["episode_id"],
            identity=expected["identity"], seed=SEED)
        append = controller.journal.append

        def bounded_append(payload):
            store.check(len(spec.canonical(payload)) + 1024)
            return append(payload)

        controller.journal.append = bounded_append
        for _ in cem.BOUNDARIES:
            controller.step()
            if controller.stopped:
                break
        if not controller.stopped:
            row.update(full_horizon=row["control_returned"] == 20, status="completed20")
    except BaseException as exc:
        old.kernel.retain(row, exc, "episode")
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
    finally:
        if active_driver is not None and not active_driver.closed:
            try:
                ledger = active_driver.finish()
                store.put(row["episode_id"] + f"_b{active_boundary:02}_partial_ledger.json", ledger)
            except BaseException as exc:
                report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        if controller is not None:
            controller.close()
        meter.region = "closing"
        store.finalizing = True
        try:
            if env is not None:
                meter.call("close", env.close)
        except BaseException as exc:
            old.kernel.retain(row, exc, "close")
        store.put(row["episode_id"] + "_episode.json", row)
        store.emit("episode_closed", receipt=row)
    report.update(counts=meter.counts, pending=meter.pending, secondary_errors=meter.secondary)
    complete = (not report["errors"] and not meter.pending and not meter.secondary
                and row["first_error"] is None
                and row["status"] in ("completed20", "native_stopped_warmup", "native_stopped_control"))
    report["status"] = "recorded_known_root_not_admitted" if complete else "partial_or_failed"
    store.finalizing = True
    store.emit("run_finished", status=report["status"], counts=meter.counts,
               nodes_sha256=spec.digest(report["nodes"]))
    report.update(journal_sha256=runner.sha(store.path / "journal.jsonl"), journal_chain_tail=store.previous)
    return report


@contextmanager
def configured(root):
    spec.require(type(root) is int and root in (0, 1), "root")
    target = (root0, root1)[root]
    target.configure()
    runner = target.runner
    original_build, original_protocol = runner.build, runner.protocol

    def protocol():
        value = original_protocol()
        value.update(schema=NAME + f"_root{root}.protocol.v1", arms=["full_cem"],
                     controller_seed=SEED, feedback_horizons=list(cem.HORIZONS),
                     node_costs={str(b): node_cost(b) for b in cem.BOUNDARIES},
                     fairness_proven=False, full_cost_certified=False)
        return value

    def build():
        value = original_build()
        value["schema"] = NAME + f"_root{root}.prepare.v1"
        value["protocol"] = protocol()
        identity = source.inspect(root)
        value["cem_source_identity"] = identity
        value["sources"][identity["prepare"]] = identity["prepare_sha256"]
        report_path = source.ROOT / "results" / f"p1_first_feedback_carry_20261101_main_root{root}_RUN_MAIN02/report.json"
        value["sources"][str(report_path)] = identity["report_sha256"]
        value["sources"][str(source.ROOT / "results/e3_cp10_target_cuda_20261031_RUN01.json")] = identity["target_sha256"]
        for item in (Path(__file__), Path(cem.__file__), Path(source.__file__), DOC, TEST):
            path = item.resolve()
            value["sources"][str(path)] = runner.sha(path)
        return value

    def execute(prepared, data, store, visual, runtime, factory):
        return execute_one(prepared, data, store, visual, runtime, factory, runner, target.rebind, root)

    with patch.object(runner, "NAME", NAME + f"_root{root}"), \
         patch.object(runner, "AUTH", AUTH), patch.object(runner, "protocol", protocol), \
         patch.object(runner, "build", build), patch.object(runner, "execute", execute):
        yield runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for name in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--authorize-host-cuda-and-physics", action="store_true", required=True)
    args = parser.parse_args()
    with configured(args.root) as runner:
        if args.command == "dry-prepare":
            value = runner.build()
            result = {"status": "dry_prepared_no_model_or_physics", "digest": runner.digest(value),
                      "root": args.root, "source_identity": value["cem_source_identity"],
                      "node_costs": value["protocol"]["node_costs"]}
        elif args.command == "prepare":
            result = runner.prepare(args.output)
        else:
            spec.require(args.authorization == AUTH, "explicit authorization")
            with gpu.locking.gpu_lock(gpu.LOCK):
                result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True))
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
