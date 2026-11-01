"""Physical-only replay of sealed cp5 CEM and two fixed same-prefix controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_cp5_cem_baseline_20261101_leibniz as cem
from . import p1_coherent_suffix_family_20261101_leibniz as family

NAME = "p1_cp5_cem_physical_20261101_leibniz"
ROOT = family.ROOT
REPORT = ROOT / "results/p1_cp5_cem_baseline_20261101_leibniz_RUN_MAIN01/report.json"
REPORT_SHA = "898d9a289b57259a2e9e3fec0df7f7f2a014e430f448816758a0775219ba10ca"
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
ARMS = ("cem", "structured_own", "zero")


def verify_selection(result, events):
    """Recompute the causal random banks and choice from the stored scores."""
    scores = result["scores"]
    family.require(len(scores) == 64 and np.isfinite(scores).all(), "CEM scores")
    values = iter([*scores, result["replay_score"]])
    generated = []
    rebuilt = cem.optimize(lambda _: next(values), lambda k, v: generated.append((k, v)))
    family.require(rebuilt == result, "CEM optimizer/selection mismatch")
    family.require(generated == [(e["kind"], e["payload"]) for e in events], "CEM trace mismatch")


def build():
    from . import p1_coherent_intervention_20261101_main as common
    from . import p1_coherent_six_prefix_summary_20261101_main as summary
    from . import p1_coherent_own_physical_20261101_main as five
    own = cem.own
    family.require(family.sha(REPORT) == REPORT_SHA, "CEM report hash")
    report = json.loads(REPORT.read_bytes())
    family.require(report["schema"] == cem.NAME + ".report.v1" and report["status"] == "complete"
                   and report["failure"] is None and report["source_verified"] is True
                   and report["physical_execution"] is False and report["P1_admission"] is False
                   and report["model_before"] == report["model_after"], "CEM report status")
    _, runner = own.configure()
    binding = cem.verify(report["prepared"], report["prepare_sha256"], runner)
    family.require(family.sha(REPORT.parent / "binding.json") == report["binding_sha256"]
                   == report["prepare_sha256"]
                   and json.loads((REPORT.parent / "binding.json").read_bytes()) == binding,
                   "CEM preparation/binding")
    family.require(report["model_before"] == binding["source"]["expected_model_hash"], "CEM model")
    row = binding["row"]
    expected_task = own.make_task(binding["source"]["base_task"], row, report["cp5_latent"],
                                  binding["source"]["warmup_raw"], runner)
    family.require(report["task"] == expected_task, "RGB latent and actual action-history binding")
    contract = cem.contract_for(own.make_task(binding["source"]["base_task"], row, [0.] * 192,
                               binding["source"]["warmup_raw"], runner), binding["source"]["model_sha256"], runner)
    contract_sha = own.spec.digest(contract)
    ledger = report["ledger"]
    family.require(Path(ledger["journal"]) == REPORT.parent / "ledger/events.jsonl"
                   and family.sha(ledger["journal"]) == ledger["sha256"], "ledger path/hash")
    account = own.consumer.consume(ledger["journal"], ledger["sha256"], contract_sha,
                                    "structured", "node", cem.NAME)
    family.require(account == report["account"] and account["status"] == "closed_unverified"
                   and account["charged_upper_bounds"] == cem.costs(), "CEM ledger")
    trace = REPORT.parent / "trace.jsonl"
    family.require(family.sha(trace) == report["trace_sha256"], "CEM trace hash")
    verify_selection(report["result"], summary.common.panel.saved.chain(trace))
    family.require(family.sha(family.SUMMARY) == family.SUMMARY_SHA, "six-prefix summary hash")
    summary_data = json.loads(family.SUMMARY.read_bytes())
    family.require([r["prefix"] for r in summary_data["rows"]] == list(family.PREFIXES),
                   "summary roster")
    sources = dict(summary_data["sources"])
    family.verify_sources(sources)
    plans, plan_sources = five.collect(family.PLAN_FIVE, family.PLAN_FIVE_SHA)
    family.merge_sources(sources, plan_sources)
    family.require([p["prefix"] for p in plans] == list(family.PREFIXES[:5]), "plan roster")
    structured = plans[0]
    goal, images, goal_sources = common.target.collect()
    family.merge_sources(sources, goal_sources)
    observed, observed_sources = summary.read_report(summary.FILES[0], goal["goal_state"], goal["root_state"])
    own_rows, physical_sources = summary.read_report(summary.FILES[2], goal["goal_state"], goal["root_state"])
    family.merge_sources(sources, observed_sources)
    family.merge_sources(sources, physical_sources)
    peers = [r for r in observed if r["prefix"] == "structured"]
    family.require(len(peers) == 2 and {r["suffix"] for r in peers} == {"zero", "common_structured"},
                   "structured physical control roster")
    family.require([r["prefix"] for r in own_rows] == list(family.PREFIXES[:5]), "own physical roster")
    raw = family.actions(structured["commands"], 20)
    prefix = family.actions(row["prefix_raw"], 5)
    cp5 = dict(pixel_sha256=row["pixel_sha256"], state_sha256=row["cp5_state_identity_sha256"])
    point = own_rows[0]["checkpoints"][1]
    family.require(np.array_equal(raw, family.actions(own_rows[0]["executed_raw"], 20))
                   and np.array_equal(raw[:5], prefix)
                   and structured["cp5_pixel_sha256"] == cp5["pixel_sha256"] == point["pixel_sha256"]
                   and structured["cp5_state_identity_sha256"] == cp5["state_sha256"]
                   == own.spec.digest(point["state"]), "physical cp5 input")
    for peer in peers:
        family.require(np.array_equal(family.actions(peer["executed_raw"], 20)[:5], prefix)
                       and peer["checkpoints"][1]["state"] == point["state"]
                       and peer["checkpoints"][1]["pixel_sha256"] == cp5["pixel_sha256"],
                       "control physical cp5 mismatch")
    cem_tail = np.asarray(report["result"]["selected_raw"], np.float32)
    family.require(cem_tail.shape == (3, 5, 2), "CEM H3 shape")
    cem_tail = family.actions(cem_tail.reshape(15, 2), 15)
    schedule = [
        dict(arm="cem", commands=np.concatenate((prefix, cem_tail)).tolist()),
        dict(arm="structured_own", commands=raw.tolist()),
        dict(arm="zero", commands=np.concatenate((prefix, np.zeros((15, 2), np.float32))).tolist()),
    ]
    for slot in schedule:
        family.require(np.array_equal(family.actions(slot["commands"], 20)[:5], prefix), "prefix mismatch")
    for path in (REPORT, Path(report["prepared"]), REPORT.parent / "binding.json", trace,
                 Path(ledger["journal"]), Path(__file__).resolve(), DOC, TEST):
        family.merge_sources(sources, {str(path): family.sha(path)})
    family.merge_sources(sources, binding["sources"])
    family.verify_sources(sources)
    root = json.loads((common.target.DIRECTORY / "report.json").read_bytes())
    return schedule, sources, goal, images, root, cp5


def replay(output):
    from . import p1_coherent_intervention_20261101_main as common
    from PIL import Image
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output = Path(output)
    family.require(output.is_absolute() and output.parent == ROOT / "results"
                   and output.name.startswith(NAME + "_") and not output.exists()
                   and not output.is_symlink(), "exclusive output")
    schedule, sources, goal, images, root, cp5 = build()
    family.require([slot["arm"] for slot in schedule] == list(ARMS), "roster")
    output.mkdir()
    protocol = dict(schema=NAME + ".protocol.v1", schedule=schedule, sources=sources, cp5=cp5,
                    cem_report_sha256=REPORT_SHA, neural_calls=0, P1_admission=False,
                    independent_confirmation=False, fairness_proven=False)
    family.write_once(output / "protocol.json", protocol)
    protocol_sha = family.sha(output / "protocol.json")
    rows = []
    for slot in schedule:
        print(json.dumps(dict(event="start", arm=slot["arm"])), flush=True)
        family.verify_sources(sources)
        family.require(family.sha(output / "protocol.json") == protocol_sha, "protocol drift")
        row = dict(arm=slot["arm"], status="failed", checkpoints=[], executed_raw=[],
                   steps_attempted=0, steps_returned=0, step_events=[])
        env = None
        try:
            env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                           with_target=False, render_action=False)
            body = env.unwrapped
            env.reset(seed=root["reset_seed"])
            env.reset(seed=0, options=dict(state=np.asarray(root["requested_reset"]),
                                          goal_state=np.asarray(root["placeholder_goal"])))
            warmup = family.actions(root["raw_history"], 10)
            for step in range(11):
                if step:
                    action = warmup[step - 1]
                    _, _, stop, trunc, _ = env.step(action.copy())
                    family.require(not stop and not trunc and np.array_equal(body.latest_action, action), "warmup")
                if step in (0, 5, 10):
                    family.require(np.array_equal(env.render(), images[step // 5]), "warmup RGB")
            common.native_source.attach_goal(body, np.asarray(goal["goal_state"]), images[-1])

            def capture(step):
                state = np.asarray(body._get_obs()).copy()
                pixels = np.asarray(env.render()).copy()
                family.require(state.shape == (7,) and np.isfinite(state).all(), "physical state")
                path = output / f"{slot['arm']}_{step:02}.png"
                Image.fromarray(pixels).save(path)
                native = common.metrics.qualification.native(goal["goal_state"], state)
                point = dict(step=step, state=state.tolist(),
                    metrics=dict(**common.metrics.metrics(state, goal["goal_state"]),
                                 joint_xy=native["joint_xy_error"], native_success=native["initial_success"]),
                    png=str(path), file_sha256=family.sha(path),
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    state_sha256=common.panel.saved.spec.digest(state.tolist()))
                row["checkpoints"].append(point)
                if step == 5:
                    family.require(point["pixel_sha256"] == cp5["pixel_sha256"]
                                   and point["state_sha256"] == cp5["state_sha256"], "actual cp5 mismatch")
                return state

            family.require(np.array_equal(capture(0), goal["root_state"]), "root mismatch")
            for step, command in enumerate(family.actions(slot["commands"], 20), 1):
                row["steps_attempted"] += 1
                obs, reward, stop, trunc, info = env.step(command.copy())
                row["steps_returned"] += 1
                row["executed_raw"].append(np.asarray(body.latest_action).tolist())
                family.require(np.array_equal(body.latest_action, command)
                               and np.array_equal(info["goal_state"], goal["goal_state"]), "action/goal mismatch")
                state = np.asarray(obs["state"])
                family.require(state.shape == (7,) and np.isfinite(state).all()
                               and np.isfinite(reward), "invalid step")
                outside = not bool(np.all((state[:4] >= 0) & (state[:4] <= 512)))
                row["step_events"].append(dict(step=step, reward=float(reward), terminated=bool(stop),
                                               truncated=bool(trunc), outside=outside))
                if step % 5 == 0 or stop or trunc or outside:
                    capture(step)
                if stop or trunc or outside:
                    row.update(status="native_stopped" if stop or trunc else "canvas_censored",
                               terminated=bool(stop), truncated=bool(trunc), stop_step=step)
                    break
            else:
                row.update(status="completed20", terminated=False, truncated=False, stop_step=None)
        except Exception as exc:
            row["error"] = dict(type=type(exc).__name__, message=str(exc))
        finally:
            if env is not None:
                env.close()
        row["steps"] = row["steps_returned"]
        family.write_once(output / (slot["arm"] + ".json"), row)
        rows.append(row)
        print(json.dumps(dict(event="end", arm=slot["arm"], status=row["status"],
                              steps=row["steps"], error=row.get("error"))), flush=True)
    family.verify_sources(sources)
    family.require(family.sha(output / "protocol.json") == protocol_sha, "protocol drift")
    result = dict(schema=NAME + ".report.v1", rows=rows, protocol_sha256=protocol_sha,
                  sources_verified=True, neural_calls=0, P1_admission=False,
                  fairness_proven=False, independent_confirmation=False)
    family.write_once(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    report = replay(args.output)
    raise SystemExit(0 if all(row["status"] != "failed" for row in report["rows"]) else 1)


if __name__ == "__main__":
    main()
