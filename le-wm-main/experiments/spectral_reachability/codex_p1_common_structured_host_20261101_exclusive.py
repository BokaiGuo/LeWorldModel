"""Known-root spectral/common-suffix + first-feedback carry host exploration.

Runs only in a dedicated process; patches are process-local. Not coherent
constant-packet initialization, a baseline comparison, or confirmation.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from . import codex_four_arm_common_rgb_launcher_20261030 as host
from . import codex_p1_common_structured_adapter_20261101_exclusive as adapter
from . import codex_p1_variable_cem_20261101_exclusive as cem
from . import p1_carry_binding_20261101 as binding
from . import p1_rgb_carry_replan_codex_20261101 as selector
from . import p1_common_budget_cem_events_20261101_leibniz as common_v2
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

NAME = "codex_p1_common_structured_host_20261101_exclusive"
METHOD = "spectral8_common_suffix_first_feedback_carry_not_coherent_constant_packet"
BOUNDARIES = (0, 5, 10, 15)


def configure():
    """Install process-local host wiring before prepare or run."""
    old_build, old_protocol = host.build, host.protocol
    old_roster, old_episode = host.rgb.roster, host.rgb.episode
    own_sources = (adapter, *adapter.DEPENDENCIES, adapter.common, common_v2, cem, binding, selector,
                   host, host.rgb, host.bridge, host.consumer, host.spec, host.journal)

    def build():
        value = old_build()
        for module in own_sources:
            path = Path(module.__file__).resolve()
            value["sources"][str(path)] = host.sha(path)
        path = Path(__file__).resolve()
        value["sources"][str(path)] = host.sha(path)
        for name in common_v2.required_sources():
            value["sources"][name] = host.sha(Path(name))
        base = value["source"]["tasks"][0]
        model = value["source"]["expected_model_hash"]
        hardware = host.digest({"device_policy": "host_cuda_0"})
        _, structured, _ = adapter.original.contract_for(base, model, hardware)
        templates = {"structured": {b: deepcopy(structured) for b in BOUNDARIES},
                     "variable_cem": {b: cem.node_contract(base, model, hardware, b, seed=20261101)
                                      for b in BOUNDARIES}}
        protocol = common_v2.protocol(adapter.common.make_protocol(templates, value["sources"]))
        value["common_node_protocol"] = protocol
        value["common_node_protocol_sha256"] = host.digest(protocol)
        return value

    def protocol():
        value = old_protocol()
        value.update(arms=["structured"], method=METHOD, carry_horizons={"0": 3, "5": 2, "10": 1, "15": 1},
                     carry_rescore="current_RGB_context_same_horizon_metered",
                     common_node_caps=deepcopy(common_v2.CAPS),
                     fairness_proven=False, independent_confirmation=False, P1_admission=False)
        return value

    def roster():
        return [row for row in old_roster() if row["arm"] == "structured"]

    def episode(env, ref, data, layer, base, row, nodes, visual, runtime, meter, store, ignored):
        previous = None
        def node(task, arm, boundary, sink, rt):
            nonlocal previous
            host.require(arm == "structured", "structured only")
            issued = np.asarray(row["executed_raw"], np.float32).reshape(-1, 2)
            host.require(len(issued) == boundary, "actual command boundary")
            feedback = None
            horizon, has_carry = {0: (3, False), 5: (2, True), 10: (1, True), 15: (1, False)}[boundary]
            if boundary:
                host.require(previous is not None, "previous plan required")
                feedback = previous.bind_feedback(episode=row["episode_id"], boundary=boundary,
                                                  issued_packet=issued[-5:],
                                                  rgb_sha256=row["last_observation"]["pixel_sha256"])
                host.require(len(feedback["tail"]) == (horizon if has_carry else 0), "carry horizon")
                raw10 = np.concatenate((ref["raw_history"], issued))[-10:]
                host.require(task["root_context"]["action_history"] == host.rgb.moving.normalized(base, raw10),
                             "actual normalized history mismatch")
            contract_template = adapter.original.contract_for(task, rt["model_sha256"], rt["hardware_sha256"])[1]
            protocol_value, protocol_sha = rt["common_protocol"], rt["common_protocol_sha"]
            ns = sink.namespace(f"common_structured_b{boundary:02}")
            trace = host.journal.Journal(ns / "candidates.jsonl", 16 * 1024**2)
            driver = None
            failure = None
            result = None
            receipt = None
            try:
                contract = common_v2.bind(protocol_value, protocol_sha, arm, boundary, contract_template)
                driver = adapter.common.durable.Driver(ns / "ledger", contract, host.digest(contract),
                                                        "structured", "node", f"structured_b{boundary:02}")
                leaf = adapter.leafmod.LeafAdapter(rt["model"], driver, contract, host.digest(contract))
                # The full planner and optional carry score must fit before any model call.
                expected = adapter.original.cost(arm)
                extra = binding.extra_cost(8, horizon, has_carry=has_carry) if boundary else {}
                for key, amount in extra.items():
                    expected[key] += amount
                host.require(all(expected[k] <= driver.caps[k] for k in host.spec.AXES), "carry budget exceeded")
                original_emit = driver._emit
                def checked_emit(kind, payload):
                    if kind == "intent":
                        rt["check_limits"]()
                    return original_emit(kind, payload)
                driver._emit = checked_emit
                def emit(kind, payload):
                    rt["check_limits"]()
                    sink.check()
                    trace.emit(kind, payload)
                result = adapter.plan(leaf, task, protocol_value, protocol_sha, boundary,
                                      device=rt["device"], emit=emit)
                if boundary:
                    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=rt["device"])
                    scored = selector.select(leaf, states=tensor(task["root_context"]["latent_history"])[None],
                                             history=tensor(task["root_context"]["action_history"])[None],
                                             goal=tensor(task["goal"])[None], mean=tensor(task["action_mean"]),
                                             std=tensor(task["action_std"]),
                                             new_candidates=result["optimized_raw"][:, :horizon],
                                             previous_selected=tensor(previous.plan.copy()) if has_carry else None)
                    host.require(scored["incremental_cost"] == extra, "carry charge mismatch")
                    result.update(scored)
                    emit("first_feedback_selection", {"boundary": boundary, "horizon": horizon,
                         "feedback": {k: v for k, v in feedback.items() if k != "tail"},
                         "scores": scored["scores"], "selected_source": scored["selected_source"],
                         "incremental_cost": extra})
                selected = binding.BoundPlan.seal(row["episode_id"], boundary,
                                                  result["selected_raw"].detach().cpu().numpy().astype(np.float32))
                emit("selected_full_plan", {"boundary": boundary, "raw": selected.plan.tolist(),
                                            "plan_sha256": selected.plan_sha256})
                emit("terminal", {"status": "complete_node"})
            except BaseException as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                if driver is not None and not driver.failed and not driver.closed:
                    try:
                        driver.dispatch("setup", "__abort__", {}, lambda: None)
                    except ValueError:
                        pass
                try:
                    trace.emit("terminal", {"status": "failed_or_partial", "error": failure})
                except BaseException:
                    pass
            finally:
                trace.close()
                if driver is not None:
                    receipt = driver.finish()
            account = None if receipt is None else host.consumer.consume(
                receipt["journal"], receipt["sha256"], host.digest(contract), arm, "node", f"structured_b{boundary:02}")
            info = {"status": "failed_or_partial" if failure else "complete", "error": failure,
                    "ledger": receipt, "account": account, "trace": str(ns / "candidates.jsonl"),
                    "trace_sha256": host.sha(ns / "candidates.jsonl"), "method": METHOD}
            sink.put(f"common_structured_b{boundary:02}_receipt.json", info)
            if failure:
                raise ValueError("node failed: " + str(failure))
            host.require(account["status"] == "closed_unverified" and account["charged_upper_bounds"] == expected,
                         "exact node accounting")
            previous = selected
            return dict(info, commands=selected.plan[0].tolist(), selected_full_plan=selected.plan.tolist(),
                        selected_full_plan_sha256=selected.plan_sha256, selected_index=result["selected_index"],
                        scores=result["scores"])
        return old_episode(env, ref, data, layer, base, row, nodes, visual, runtime, meter, store, node)

    old_execute = host.execute
    def execute(prepared, data, store, visual, runtime, factory):
        runtime["common_protocol"] = prepared["common_node_protocol"]
        runtime["common_protocol_sha"] = prepared["common_node_protocol_sha256"]
        store.put("common_protocol.json", {"protocol": runtime["common_protocol"],
                                          "sha256": runtime["common_protocol_sha"]})
        return old_execute(prepared, data, store, visual, runtime, factory)

    host.NAME = NAME
    host.build, host.protocol, host.execute = build, protocol, execute
    host.rgb.roster, host.rgb.episode = roster, episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("alphanumeric tag required")
    configure()
    prefix = host.ROOT / "results" / NAME
    prepared = Path(str(prefix) + "_PREP_" + args.tag + ".json")
    output = Path(str(prefix) + "_RUN_" + args.tag)
    if prepared.exists() or output.exists():
        raise ValueError("exclusive outputs")
    seal = host.prepare(str(prepared))
    print(json.dumps({"prepared": seal}), flush=True)
    with locking.gpu_lock(host.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        result = host.run(str(prepared), seal["sha256"], str(output), host.AUTH)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "recorded_known_root_not_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
