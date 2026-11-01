"""Known-root six-by-seven physical suffix intervention; no neural inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

NAME = "p1_coherent_suffix_family_20261101_leibniz"
ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/p1_coherent_six_prefix_summary_20261101_main_MAIN01.json"
SUMMARY_SHA = "b668efc76320fca501567ad74b35b035f19b8beeb9dd4b079661af23af563ebe"
PLAN_FIVE = ROOT / "results/p1_coherent_own_suffix_20261101_codex_RUN_MAIN01"
PLAN_ONE = ROOT / "results/p1_coherent_own_suffix_samebank_20261101_codex_RUN_MAIN01"
PLAN_FIVE_SHA = "e14a4e701ed4983e48801b2ed813f9daf8c0373db1771af4415118690b19cdc7"
PLAN_ONE_SHA = "7402564db52792ed3a7027b63aa7328d6c9c7b59b0e7ca881a000a16515919e0"
PREFIXES = ("structured", "native", "random", "immediate_progress", "zero", "same_bank_greedy")
TAILS = ("zero15",) + tuple("own15_" + p for p in PREFIXES)
DOC = ROOT.parent / "docs" / (NAME + ".md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources(sources):
    require(isinstance(sources, dict) and bool(sources), "empty source closure")
    for path, digest in sources.items():
        require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64,
                "invalid source entry")
        require(Path(path).is_file() and sha(path) == digest, "source mismatch: " + path)


def merge_sources(target, additions):
    for path, digest in additions.items():
        require(path not in target or target[path] == digest, "conflicting source digest: " + path)
        target[path] = digest


def actions(value, length):
    raw = np.asarray(value, dtype=np.float32)
    require(raw.shape == (length, 2) and np.isfinite(raw).all()
            and bool((np.abs(raw) <= 1).all()), "invalid action shape, bounds or finiteness")
    return raw


def make_schedule(prefixes, tails, cp5):
    """Construct the complete, ordered roster; no outcome-dependent selection."""
    require(list(prefixes) == list(PREFIXES) and list(cp5) == list(PREFIXES), "prefix roster")
    require(list(tails) == list(TAILS), "tail roster")
    require(np.array_equal(actions(tails["zero15"], 15), np.zeros((15, 2), np.float32)),
            "zero tail mismatch")
    for name in PREFIXES:
        actions(prefixes[name], 5)
        require(set(cp5[name]) == {"pixel_sha256", "state_sha256"}
                and all(isinstance(v, str) and len(v) == 64 for v in cp5[name].values()), "cp5 identity")
    for name in TAILS:
        actions(tails[name], 15)
    return [dict(prefix=p, tail=t,
                 commands=np.concatenate((actions(prefixes[p], 5), actions(tails[t], 15))).tolist(),
                 cp5=cp5[p])
            for p in PREFIXES for t in TAILS]


def validate_schedule(schedule):
    require([(s["prefix"], s["tail"]) for s in schedule] ==
            [(p, t) for p in PREFIXES for t in TAILS], "42 branch roster/order")
    prefixes, tails, cp5 = {}, {}, {}
    for slot in schedule:
        raw = actions(slot["commands"], 20)
        p, t = slot["prefix"], slot["tail"]
        if p in prefixes:
            require(np.array_equal(prefixes[p], raw[:5]) and cp5[p] == slot["cp5"], "prefix mismatch")
        else:
            prefixes[p], cp5[p] = raw[:5], slot["cp5"]
        if t in tails:
            require(np.array_equal(tails[t], raw[5:]), "common tail mismatch")
        else:
            tails[t] = raw[5:]
    require(schedule == make_schedule(prefixes, tails, cp5), "schedule reconstruction")


def collect():
    # Existing collectors revalidate model-plan bindings, trace selection and
    # durable ledgers; they do not load a model or execute physics.
    from . import p1_coherent_own_physical_20261101_main as five
    from . import p1_coherent_samebank_own_physical_20261101_codex as one
    from . import p1_coherent_six_prefix_summary_20261101_main as summary

    require(sha(SUMMARY) == SUMMARY_SHA, "summary hash")
    data = json.loads(SUMMARY.read_bytes())
    require([r["prefix"] for r in data["rows"]] == list(PREFIXES)
            and data["neural_calls"] == 0 and data["P1_admission"] is False
            and data["independent_confirmation"] is False, "summary roster/scope")
    sources = dict(data["sources"])
    verify_sources(sources)
    five_schedule, five_sources = five.collect(PLAN_FIVE, PLAN_FIVE_SHA)
    one_schedule, one_sources = one.collect(PLAN_ONE, PLAN_ONE_SHA)
    merge_sources(sources, five_sources)
    merge_sources(sources, one_sources)
    plans = five_schedule + one_schedule
    require([r["prefix"] for r in plans] == list(PREFIXES), "plan roster")
    goal, images, goal_sources = summary.common.target.collect()
    merge_sources(sources, goal_sources)
    # Recheck actual physical reports, including issued actions, PNG pixels,
    # metrics and their protocol/source closure, independently of plan labels.
    reports = []
    for name in summary.FILES:
        rows, bindings = summary.read_report(name, goal["goal_state"], goal["root_state"])
        reports.append(rows)
        merge_sources(sources, bindings)
    observed = reports[0] + reports[1]
    own = reports[2] + reports[3]
    require([r["prefix"] for r in own] == list(PREFIXES), "physical own roster")
    prefixes, tails, cp5 = {}, {"zero15": np.zeros((15, 2), np.float32)}, {}
    for name, plan, physical, summary_row in zip(PREFIXES, plans, own, data["rows"], strict=True):
        planned = actions(plan["commands"], 20)
        actual = actions(physical["executed_raw"], 20)
        require(np.array_equal(actual, planned), "planned/actual own actions: " + name)
        point = physical["checkpoints"][1]
        require(point["step"] == 5 and point["pixel_sha256"] == plan["cp5_pixel_sha256"], "own cp5 pixel")
        require(summary.common.panel.saved.spec.digest(point["state"]) ==
                plan["cp5_state_identity_sha256"], "own cp5 state")
        require(summary_row["cp5"] == point["metrics"], "summary cp5 metrics")
        peers = [r for r in observed if r["prefix"] == name]
        require(len(peers) == 2 and {r["suffix"] for r in peers} in
                ({"zero", "common"}, {"zero", "common_structured"}), "physical peers roster")
        for peer in peers:
            require(np.array_equal(actions(peer["executed_raw"], 20)[:5], planned[:5])
                    and peer["checkpoints"][1]["state"] == point["state"]
                    and peer["checkpoints"][1]["pixel_sha256"] == point["pixel_sha256"],
                    "physical prefix/cp5 mismatch")
        prefixes[name], tails["own15_" + name] = planned[:5], planned[5:]
        cp5[name] = dict(pixel_sha256=plan["cp5_pixel_sha256"],
                         state_sha256=plan["cp5_state_identity_sha256"])
    for path in (SUMMARY, Path(__file__).resolve(), DOC):
        merge_sources(sources, {str(path): sha(path)})
    schedule = make_schedule(prefixes, tails, cp5)
    validate_schedule(schedule)
    verify_sources(sources)
    return schedule, sources, goal, images


def write_once(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def execute(output):
    from . import p1_coherent_intervention_20261101_main as common
    from PIL import Image
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and not output.exists() and not output.is_symlink(), "exclusive output path")
    schedule, sources, goal, images = collect()
    root = json.loads((common.target.DIRECTORY / "report.json").read_bytes())
    verify_sources(sources)
    output.mkdir()
    write_once(output / "protocol.json", dict(schema=NAME + ".v1", schedule=schedule,
        sources=sources, summary_sha256=SUMMARY_SHA, neural_calls=0, P1_admission=False,
        independent_confirmation=False, scope="exploratory known-root intervention"))
    protocol_sha = sha(output / "protocol.json")
    rows = []
    for index, slot in enumerate(schedule):
        if index % len(TAILS) == 0:
            verify_sources(sources)
        print(json.dumps(dict(event="branch_start", index=index + 1, total=42,
                              prefix=slot["prefix"], tail=slot["tail"])), flush=True)
        row = dict(prefix=slot["prefix"], tail=slot["tail"], status="failed",
                   checkpoints=[], executed_raw=[], steps_attempted=0, steps_returned=0)
        env = None
        try:
            require(sha(output / "protocol.json") == protocol_sha, "protocol drift")
            env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                           with_target=False, render_action=False)
            body = env.unwrapped
            env.reset(seed=root["reset_seed"])
            env.reset(seed=0, options=dict(state=np.asarray(root["requested_reset"]),
                                          goal_state=np.asarray(root["placeholder_goal"])))
            for step in range(11):
                if step:
                    action = actions(root["raw_history"], 10)[step - 1]
                    _, _, stop, trunc, _ = env.step(action.copy())
                    require(not stop and not trunc and np.array_equal(body.latest_action, action), "warmup")
                if step in (0, 5, 10):
                    require(np.array_equal(env.render(), images[step // 5]), "warmup RGB")
            common.native_source.attach_goal(body, np.asarray(goal["goal_state"]), images[-1])

            def capture(step):
                state, pixels = np.asarray(body._get_obs()).copy(), np.asarray(env.render()).copy()
                require(state.shape == (7,) and np.isfinite(state).all(), "state invalid")
                path = output / f"{index:02}_{step:02}.png"
                Image.fromarray(pixels).save(path)
                native = common.metrics.qualification.native(goal["goal_state"], state)
                point = dict(step=step, state=state.tolist(),
                    metrics=dict(**common.metrics.metrics(state, goal["goal_state"]),
                        joint_xy=native["joint_xy_error"], native_success=native["initial_success"]),
                    png=str(path), file_sha256=sha(path),
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    state_sha256=common.panel.saved.spec.digest(state.tolist()))
                row["checkpoints"].append(point)
                if step == 5:
                    require(point["pixel_sha256"] == slot["cp5"]["pixel_sha256"]
                            and point["state_sha256"] == slot["cp5"]["state_sha256"], "actual cp5 mismatch")
                return state

            require(np.array_equal(capture(0), goal["root_state"]), "root mismatch")
            for step, command in enumerate(actions(slot["commands"], 20), 1):
                row["steps_attempted"] += 1
                obs, reward, stop, trunc, info = env.step(command.copy())
                row["steps_returned"] += 1
                row["executed_raw"].append(np.asarray(body.latest_action).tolist())
                require(np.array_equal(body.latest_action, command)
                        and np.array_equal(info["goal_state"], goal["goal_state"]), "action/goal mismatch")
                state = np.asarray(obs["state"])
                require(state.shape == (7,) and np.isfinite(state).all(), "invalid returned state")
                outside = not bool(np.all((state[:4] >= 0) & (state[:4] <= 512)))
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
        write_once(output / f"{index:02}.json", row)
        rows.append(row)
        print(json.dumps(dict(event="branch_end", index=index + 1, prefix=slot["prefix"],
                              tail=slot["tail"], status=row["status"], steps=row["steps"],
                              error=row.get("error"))), flush=True)
    verify_sources(sources)
    require(sha(output / "protocol.json") == protocol_sha, "protocol drift")
    write_once(output / "report.json", dict(schema=NAME + ".report.v1", rows=rows,
        protocol_sha256=protocol_sha, sources_verified=True, neural_calls=0,
        P1_admission=False, independent_confirmation=False))
    return all(r["status"] != "failed" for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorize-physical", action="store_true")
    args = parser.parse_args()
    if args.authorize_physical:
        require(args.output is not None, "physical output required")
        raise SystemExit(0 if execute(args.output) else 1)
    require(args.output is None, "output requires physical authorization")
    schedule, sources, _, _ = collect()
    print(json.dumps(dict(status="validated_no_physics", branches=len(schedule), sources=len(sources))))


if __name__ == "__main__":
    main()
