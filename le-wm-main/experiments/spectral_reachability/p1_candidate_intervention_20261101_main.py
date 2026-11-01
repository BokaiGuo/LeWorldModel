"""Execute all eight sealed prefixes with own/common/zero ten-step tails."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from . import codex_four_arm_prefix_intervention_20261030 as engine
from . import p1_equal_candidate_q_20261101_main as source
from . import p1_equal_candidate_q_saved_20261101_main as saved

NAME = "p1_candidate_intervention_20261101_main"
ROOT = source.ROOT
DOC = ROOT.parent / "docs/D2P_P1_CANDIDATE_INTERVENTION_PROTOCOL_20261101_MAIN.md"
DIRECTORY = ROOT / "results/p1_equal_candidate_q_20261101_main_structured_RUN_MAIN01"
PREPARED = ROOT / "results/p1_equal_candidate_q_20261101_main_structured_PREP_MAIN01.json"


def schedule(bank, selected):
    bank = np.asarray(bank, np.float32)
    engine.require(bank.shape == (8, 3, 5, 2) and np.isfinite(bank).all()
                   and (abs(bank) <= 1).all() and type(selected) is int and 0 <= selected < 8,
                   "full bounded bank")
    rows = []
    for index in range(8):
        for role, tail in (("own", bank[index, 1:]), ("common", bank[selected, 1:]),
                           ("zero", np.zeros((2, 5, 2), np.float32))):
            rows.append(dict(index=len(rows), candidate=index, suffix_role=role,
                             commands=engine.actions(np.concatenate((bank[index, :1], tail)).reshape(15, 2), 15)))
    return rows


def episode(env, ref, data, branch, row, meter, store, expected_root, prefix_checks):
    if not engine.moving.setup(env, ref, data["plan"], row, meter, store):
        row["status"] = "native_stopped_warmup"
        return
    cp0 = row["last_observation"]
    engine.require(all(cp0[k] == expected_root[k] for k in ("state", "snapshot", "pixel_sha256", "goal")),
                   "exact source physical root")
    row["observations"] = [deepcopy(cp0)]
    body = env.unwrapped
    for step, command in enumerate(np.asarray(branch["commands"], np.float32), 1):
        meter.region = "prefix" if step <= 5 else "suffix"
        store.emit("branch_dispatch", branch=branch["index"], step=step, command=command.tolist())
        obs, reward, terminated, truncated, info = env.step(command.copy())
        row["control_returned"] += 1
        actual = np.asarray(body.latest_action)
        engine.require(engine.old.kernel.moving.exact(actual, command)
                       and np.array_equal(info["goal_state"], ref["goal"])
                       and np.array_equal(info["goal"], ref["goal_rgb"]), "actual command/goal")
        state = np.asarray(obs["state"])
        engine.require(state.shape == (7,) and np.isfinite(state).all()
                       and engine.old.kernel.moving.exact(state, np.asarray(body._get_obs())),
                       "native state")
        contact = info.get("n_contacts")
        engine.require(type(contact) is int and contact >= 0, "contact count")
        engine.old.native_transition(body, ref["goal"], obs, reward, terminated, truncated, row)
        event = dict(branch=branch["index"], step=step, command=actual.tolist(), state=state.tolist(),
                     n_contacts=contact, terminated=bool(terminated), truncated=bool(truncated))
        row["executed_raw"].append(actual.tolist())
        row["transitions"].append(event)
        store.emit("branch_returned", **event)
        try:
            _, receipt = engine.old.capture(env, obs, row, store, "plan", step)
        except engine.old.kernel.moving.Rejected as exc:
            engine.require(str(exc) == "fresh RGB/canvas guard" and engine.guard_origin(exc),
                           "original domain guard")
            proof = engine.validate_censor(branch, event, row["last_observation"])
            row.update(status="censored_out_of_domain", censor=proof, endpoint=None)
            return
        row["observations"].append(deepcopy(receipt))
        if step == 5:
            checkpoint = {k: receipt[k] for k in ("state", "snapshot", "pixel_sha256")}
            key = branch["candidate"]
            if key in prefix_checks:
                engine.require(checkpoint == prefix_checks[key], "same prefix reproduces exact state/RGB")
            else:
                prefix_checks[key] = checkpoint
            row["prefix_checkpoint"] = checkpoint
        if terminated or truncated:
            row.update(status="native_stopped", endpoint=None)
            return
    row.update(status="completed15", endpoint=deepcopy(row["last_observation"]))


def run(output):
    validation = saved.check(DIRECTORY)
    report_source = json.loads((DIRECTORY / "report.json").read_bytes())
    source_row = report_source["rows"][0]
    node = next(n for n in report_source["nodes"][source_row["episode_id"]] if n["boundary"] == 0)
    trace = Path(node["receipt"]["trace"])
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    payloads = {e["kind"]: e["payload"] for e in events}
    bank = payloads["optimized_candidates"]["raw"]
    selected = payloads["final_selection"]["index"]
    experiment = source.setup("structured")
    prepared = experiment.runner.verify(str(PREPARED), engine.sha(PREPARED))
    data = engine.p3.source_data(prepared["source"])
    experiment.rebind(prepared, data)
    ref = data["references"][0]
    branches = schedule(bank, selected)
    bindings = dict(prepared["sources"])
    for path in (PREPARED, DIRECTORY / "report.json", trace, DOC, Path(__file__).resolve(),
                 Path(engine.__file__).resolve(), Path(saved.__file__).resolve()):
        bindings[str(path)] = engine.sha(path)
    plan = dict(schedule=branches, bindings=bindings, selected_index=selected,
                source_validation=validation, predicted_scores=payloads["final_selection"]["scores"],
                physical_labels_used_for_selection=False, development_only=True,
                neural_calls=0, P1_admission=False)
    engine.NAME = NAME
    engine.CAPS = {key: 8 * value for key, value in engine.p3.contract()["physical_caps"].items()}
    store = engine.Store(output)
    store.put("prepared.json", plan)
    started = time.monotonic()

    def check():
        engine.require(time.monotonic() - started < engine.WALL, "wall cap")
        store.check()

    meter = engine.Meter(store, check)
    report = dict(status="partial_or_failed", rows=[], errors=[], planned_branches=24,
                  neural_calls=0, P1_admission=False, independent_confirmation=False)
    root = next(c for c in source_row["checkpoints"] if c["step"] == 0)
    selected_cp = next(c for c in source_row["checkpoints"] if c["step"] == 5)
    prefix_checks = {selected: {k: selected_cp[k] for k in ("state", "snapshot", "pixel_sha256")}}
    try:
        for branch in branches:
            row = deepcopy(engine.p3.roster()[0])
            row.update(episode_id=f"branch_{branch['index']:02}", branch=branch["index"],
                       candidate=branch["candidate"], suffix_role=branch["suffix_role"],
                       status="running", transitions=[], endpoint=None, observations=[])
            report["rows"].append(row)
            meter.episode, meter.region = row["episode_id"], "setup"
            env = None
            try:
                env = meter.call("constructor", lambda: engine.old.native_factory(meter))
                episode(env, ref, data, branch, row, meter, store, root, prefix_checks)
            finally:
                meter.region = "closing"
                if env is not None:
                    meter.call("close", env.close)
                store.put(row["episode_id"] + "_result.json", row)
            print(json.dumps(dict(branch=branch["index"], status=row["status"])), flush=True)
        engine.require(all(engine.sha(path) == value for path, value in bindings.items()), "source drift")
        engine.require(not meter.pending and not meter.secondary, "physical accounting")
        report["status"] = "recorded_not_independently_consumed"
    except BaseException as exc:
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        store.finalizing = True
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report.update(counts=meter.counts, journal_sha256=engine.sha(store.path / "journal.jsonl"),
                      journal_chain_tail=store.previous, wall_seconds=time.monotonic() - started)
        store.put("report.json", report, final=True)
    print(json.dumps(dict(output=str(output), status=report["status"], errors=report["errors"])), flush=True)
    return report["status"] == "recorded_not_independently_consumed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
