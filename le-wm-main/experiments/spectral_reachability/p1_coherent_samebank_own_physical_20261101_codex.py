"""Validate one sealed same-bank plan, optionally replay it physically."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_centered_program_replay_20261101_main as physical
from . import p1_coherent_own_suffix_samebank_20261101_codex as plan

saved = physical.common.panel.saved
PREFIX = plan.PREFIX
DOC = plan.ROOT.parent / "docs/D2P_P1_SAMEBANK_GREEDY_OWN_PHYSICAL_20261101_CODEX.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_selection(events, row):
    def one(kind):
        found = [e["payload"] for e in events if e["kind"] == kind]
        require(len(found) == 1, "unique " + kind)
        return found[0]

    optimized = np.asarray(one("optimized_candidates_before_exchange")["raw"], np.float32)
    exchange = one("suffix_exchange")
    scores = np.asarray(exchange["scores"], float)
    require(optimized.shape == (8, 4, 5, 2) and scores.shape == (64,)
            and np.isfinite(optimized).all() and np.isfinite(scores).all(), "exchange bank")
    tails = [min(range(8), key=lambda j: (scores[8*i+j], j != i, j)) for i in range(8)]
    require(tails == exchange["tail_indices"], "exchange argmin")
    bank = np.stack([np.concatenate((optimized[i, :1], optimized[j, 1:]))[:3]
                     for i, j in enumerate(tails)])
    bank = np.concatenate((bank, np.zeros((1, 3, 5, 2), np.float32)))
    selected = one("selected_full_plan")
    score = np.asarray(row["scores"], float)
    require(score.shape == (9,) and np.isfinite(score).all()
            and selected["scores"] == row["scores"], "selection scores")
    index = min(range(9), key=lambda i: (score[i], i != 8, i))
    chosen = np.asarray(row["selected_suffix"], np.float32)
    require(index == row["selected_index"] == selected["index"]
            and np.array_equal(chosen, bank[index])
            and np.array_equal(chosen, np.asarray(selected["raw"], np.float32)), "selected suffix")
    return chosen


def collect(directory, report_sha):
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    require(saved.sha(report_path) == report_sha, "report SHA")
    report = json.loads(report_path.read_bytes())
    binding_path = directory / "binding.json"
    require(saved.sha(binding_path) == report["binding_sha256"], "binding SHA")
    binding = json.loads(binding_path.read_bytes())
    require(report["schema"] == plan.NAME + ".report.v1"
            and binding["schema"] == plan.NAME + ".prepare.v1"
            and report["status"] == "complete" and report["error"] is None
            and report["model_before"] is not None
            and report["model_before"] == report["model_after"] == binding["expected_model_hash"]
            and report["physical_execution"] is False and report["P1_admission"] is False,
            "sealed model report")
    prepared = Path(report["prepared"])
    require(saved.sha(prepared) == report["prepare_sha256"]
            and json.loads(prepared.read_bytes()) == binding, "prepared identity")
    _, runner = plan.own.configure()
    require(binding == plan.build(runner, binding["physical_report"], binding["physical_report_sha256"]),
            "independently rebuilt preparation")
    require([r["prefix"] for r in report["rows"]] == [PREFIX]
            and [r["prefix"] for r in binding["rows"]] == [PREFIX], "single greedy roster")
    sources = dict(binding["sources"])
    require(all(saved.sha(p) == h for p, h in sources.items()), "source drift")
    row, inp = report["rows"][0], binding["rows"][0]
    require(all(row[k] == inp[k] for k in
                ("prefix", "prefix_raw", "pixel_sha256", "cp5_state_identity_sha256")),
            "prefix identity")
    # Recheck the original greedy argmin and actual cp5 evidence, not just report claims.
    prefix, model, index, original_sources = plan.greedy_source()
    require(index == binding["greedy_index"] and model == report["model_before"]
            and np.array_equal(prefix, np.asarray(row["prefix_raw"], np.float32)),
            "greedy argmin binding")
    _, target, _, _, _ = plan.own.coherent.centered.source_inputs()
    actual, actual_sources = plan.load_physical(
        binding["physical_report"], binding["physical_report_sha256"], prefix,
        target["root_state"], target["history_pixel_sha256"][-1])
    require(actual == inp, "actual cp5 binding")
    sources.update(original_sources)
    sources.update(actual_sources)
    task, base = row["task"], binding["base_task"]
    require(task == plan.own.make_task(base, inp, row["cp5_latent"], binding["warmup_raw"], runner)
            and row["hold_reference"] == binding["hold_reference"]
            and row["hold_plan"] == plan.own.hold_plan(inp).tolist(), "task and hold reference")
    require(all(task[k] == base[k] for k in ("goal", "action_mean", "action_std")), "goal/normalizer")
    raw10 = np.concatenate((np.asarray(binding["warmup_raw"], np.float32), prefix))
    normalized = ((raw10 - np.asarray(base["action_mean"], np.float32)) /
                  np.asarray(base["action_std"], np.float32)).reshape(2, 10).tolist()
    require(task["root_context"]["action_history"] == normalized
            and task["root_context"]["latent_history"] ==
            base["root_context"]["latent_history"][-2:] + [row["cp5_latent"]],
            "actual context alignment")
    for kind, suffix in (("node", "_node"), ("encoding", "_encode")):
        ledger = row[kind + "_ledger"]
        journal = Path(ledger["journal"])
        require(journal.resolve().is_relative_to(directory), "ledger path")
        header = json.loads(journal.read_bytes().splitlines()[0])["payload"]
        digest = saved.spec.digest(header["contract"])
        if kind == "node":
            policy, contract, config = plan.own.node_contract(task, binding["model_sha256"], runner)
            require(row["policy"] == policy and row["config"] == config
                    and row["contract_sha256"] == digest, "node policy/config")
        else:
            placeholder = plan.own.make_task(base, inp, [0.] * 192, binding["warmup_raw"], runner)
            _, contract, _ = plan.own.encode_contract(placeholder, binding["model_sha256"], runner)
        require(header["contract"] == contract, "contract binding")
        account = saved.consumer.consume(journal, ledger["sha256"], digest,
                                         "structured", "node", PREFIX + suffix)
        expected = binding["node_cost"].copy() if kind == "node" else dict.fromkeys(saved.h4.AXES, 0)
        if kind == "encoding":
            expected.update(encode_calls=1, encode_samples=1)
        require(account == row[kind + "_account"] and account["status"] == "closed_unverified"
                and account["charged_upper_bounds"] == expected, "durable charges")
        sources[str(journal)] = ledger["sha256"]
    trace = Path(row["trace"])
    require(trace.resolve().is_relative_to(directory) and saved.sha(trace) == row["trace_sha256"],
            "trace SHA/path")
    chosen = validate_selection(saved.chain(trace), row)
    sources[str(trace)] = row["trace_sha256"]
    for path in (report_path, binding_path, prepared, Path(__file__).resolve(), DOC):
        sources[str(path)] = saved.sha(path)
    return [dict(prefix=PREFIX, commands=np.concatenate((prefix, chosen.reshape(15, 2))).tolist(),
                 cp5_pixel_sha256=row["pixel_sha256"],
                 cp5_state_identity_sha256=row["cp5_state_identity_sha256"])], sources


def execute(schedule, bindings, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    common = physical.common
    output = Path(output)
    require(output.is_absolute() and output.parent == common.panel.ROOT / "results"
            and not output.exists() and not output.is_symlink(), "exclusive output")
    require([s["prefix"] for s in schedule] == [PREFIX], "single greedy roster")
    slot = schedule[0]
    commands = np.asarray(slot["commands"], np.float32)
    require(commands.shape == (20, 2) and np.isfinite(commands).all()
            and (np.abs(commands) <= 1).all(), "bounded commands")
    goal, images, sources = common.target.collect()
    root = json.loads((common.target.DIRECTORY / "report.json").read_bytes())
    sources.update(bindings)
    require(all(saved.sha(p) == h for p, h in sources.items()), "source drift")
    output.mkdir()
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(schedule=schedule, sources=sources, neural_calls=0,
                       P1_admission=False, independent_confirmation=False), stream, indent=2)
    row = dict(prefix=PREFIX, status="failed", checkpoints=[], executed_raw=[])
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                   with_target=False, render_action=False)
    try:
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

        def capture(step):
            state, pixels = np.asarray(body._get_obs()).copy(), np.asarray(env.render()).copy()
            require(state.shape == (7,) and np.isfinite(state).all(), "finite state")
            path = output / f"{PREFIX}_{step:02}.png"
            Image.fromarray(pixels).save(path)
            native = common.metrics.qualification.native(goal["goal_state"], state)
            point = dict(step=step, state=state.tolist(),
                         metrics=dict(**common.metrics.metrics(state, goal["goal_state"]),
                                      joint_xy=native["joint_xy_error"], native_success=native["initial_success"]),
                         png=str(path), file_sha256=saved.sha(path),
                         pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
            row["checkpoints"].append(point)
            if step == 5:
                require(point["pixel_sha256"] == slot["cp5_pixel_sha256"]
                        and saved.spec.digest(point["state"]) == slot["cp5_state_identity_sha256"], "cp5 identity")
            return state

        require(np.array_equal(capture(0), goal["root_state"]), "root")
        for step, command in enumerate(commands, 1):
            obs, _, stop, trunc, info = env.step(command.copy())
            require(np.array_equal(body.latest_action, command)
                    and np.array_equal(info["goal_state"], goal["goal_state"]), "action/goal")
            row["executed_raw"].append(command.tolist())
            state = np.asarray(obs["state"])
            outside = not bool(np.all((state[:4] >= 0) & (state[:4] <= 512)))
            if step % 5 == 0 or stop or trunc or outside:
                capture(step)
            if stop or trunc or outside:
                row.update(status="native_stopped" if stop or trunc else "canvas_censored",
                           terminated=bool(stop), truncated=bool(trunc))
                break
        else:
            row["status"] = "completed20"
    except Exception as exc:
        row["error"] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        env.close()
    row["steps"] = len(row["executed_raw"])
    with (output / f"{PREFIX}.json").open("x") as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
    require(all(saved.sha(p) == h for p, h in sources.items()), "source drift")
    with (output / "report.json").open("x") as stream:
        json.dump(dict(rows=[row], sources=sources, protocol_sha256=saved.sha(output / "protocol.json"),
                       neural_calls=0, P1_admission=False, independent_confirmation=False),
                  stream, indent=2, allow_nan=False)
    return row["status"] != "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output")
    parser.add_argument("--authorize-physical", action="store_true")
    args = parser.parse_args()
    schedule, sources = collect(args.plans, args.report_sha256)
    if not args.authorize_physical:
        require(args.output is None, "output requires physical authorization")
        print(json.dumps(dict(status="validated_no_physical_execution", roster=[PREFIX], sources=len(sources))))
    else:
        require(args.output is not None, "physical output required")
        raise SystemExit(0 if execute(schedule, sources, args.output) else 1)


if __name__ == "__main__":
    main()
