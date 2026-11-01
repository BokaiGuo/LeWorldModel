"""Bind root-1 winner feedback to the existing CUDA same-action evaluator."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_four_arm_root1_prediction_20261030_main as engine
from . import p1_four_arm_winner_root1_20261031 as experiment
from . import p1_four_arm_winner_root1_saved_20261031 as saved
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as gpu

ROOT = engine.ROOT
NAME = "p1_winner_root1_same_action_20261031"
RUN = ROOT / "results/p1_four_arm_winner_root1_20261031_RUN01"
PREP = ROOT / "results/p1_four_arm_winner_root1_20261031_prepare_RUN01.json"
REPORT_SHA = "5e58003df2d3d70eb87843789b2bb3b0994750ce5dcad2ee4825325a64f14838"
PREP_SHA = "c3dc32c5c55372af3c469aa1768d377443cef264e61cfc35f223a57f61450847"
PROTOCOL = ROOT.parent / "docs/D2P_FOUR_ARM_WINNER_ROOT1_SAME_ACTION_PROTOCOL_20261031.md"
need, sha, f32 = engine.need, engine.sha, engine.f32


def saved_inputs():
    need(sha(RUN / "report.json") == REPORT_SHA, "report identity")
    experiment.configure()
    p = experiment.runner.verify(str(PREP), PREP_SHA)
    saved.check(RUN, REPORT_SHA)
    r = engine.prior.signal.read(RUN / "report.json", 2*1024**2)
    need(r["prepare_sha256"] == PREP_SHA and r["model_before"] == r["model_after"]
         and r["model_before"]["original_state_dict"] == engine.MODEL_SHA, "report/model identity")
    events = [json.loads(line) for line in (RUN / "journal.jsonl").read_bytes().splitlines()]
    previous = None
    for index, event in enumerate(events):
        need(event["index"] == index and event["previous_sha256"] == previous and
             event["event_sha256"] == engine.prior.old.source.digest(
                 {k: v for k, v in event.items() if k != "event_sha256"}), "journal chain")
        previous = event["event_sha256"]
    need(previous == r["journal_chain_tail"] and events[-1]["kind"] == "run_finished", "journal terminal")
    base = p["source"]["tasks"][1]
    result = []
    for row in r["rows"]:
        arm, eid = row["arm"], row["episode_id"]
        need(eid == "s1_" + arm and row["source_slot"] == 1 and row["status"] == "completed20"
             and row["control_returned"] == 20, "physical trajectory")
        actions = f32(row["executed_raw"], (20, 2))
        returned = [e for e in events if e["kind"] == "control_return" and e["episode_id"] == eid]
        warm = [e for e in events if e["kind"] == "warmup_return" and e["episode_id"] == eid]
        need([e["step"] for e in warm] == list(range(1, 11)) and
             [e["step"] for e in returned] == list(range(1, 21)) and
             all(not e["terminated"] and not e["truncated"] for e in warm + returned) and
             f32([e["command"] for e in returned], (20, 2)).tobytes() == actions.tobytes(), "actions")
        warm_actions = f32([e["command"] for e in warm], (10, 2))
        checkpoints = row["checkpoints"]
        need([c["step"] for c in checkpoints] == [0, 5, 10, 15, 20], "checkpoint schedule")
        for cp in checkpoints:
            engine.png(cp)
        context = deepcopy(base["root_context"])
        for i, boundary in enumerate(engine.BOUNDARIES):
            stem = f"{eid}_b{boundary:02}"
            seal = engine.prior.signal.read(RUN / (stem + "_seal.json"), 4*1024**2)
            enc = engine.prior.signal.read(RUN / (stem + "_encoding.json"), 4*1024**2)
            identity, obs = seal["identity"], seal["identity"]["observation"]
            need(identity["episode_id"] == eid and identity["arm"] == arm and identity["boundary"] == boundary
                 and obs["png_sha256"] == checkpoints[i]["file_sha256"]
                 and obs["pixel_sha256"] == checkpoints[i]["pixel_sha256"]
                 and obs["encoding_sha256"] == engine.prior.old.source.digest(enc), "observation identity")
            response = enc["response"]
            ld = response["latent"]
            lp = Path(ld["path"])
            need(enc["actual_step"] == boundary and response["device_type"] == "cuda"
                 and response["model_hash_before"] == response["model_hash_after"] == engine.MODEL_SHA
                 and response["model_unchanged"] is True and lp.parent == RUN / (stem + "_encode")
                 and lp == lp.resolve() and not lp.is_symlink() and sha(lp) == ld["file_sha256"], "encoding identity")
            z = np.load(lp, allow_pickle=False)
            need(z.shape == (192,) and z.dtype == np.float32 and np.isfinite(z).all()
                 and hashlib.sha256(z.tobytes()).hexdigest() == ld["array_sha256"]
                 and engine.prior.old.packet_digest(z.tolist()) == obs["observed_latent_sha256"], "latent identity")
            history = f32(identity["raw10"], (10, 2))
            if boundary == 0:
                need(history.tobytes() == warm_actions.tobytes(), "warm history")
                context["latent_history"][-1] = z.tolist()
                context["action_history"] = engine.prior.rgb.moving.normalized(base, history)
            else:
                expected = np.concatenate((warm_actions, actions[:boundary]))[-10:]
                need(history.tobytes() == expected.tobytes(), "executed history")
                context = engine.prior.rgb.moving.roll(context, z, history, base)
            task = engine.prior.p3.local_task(base, context, identity)
            need(engine.prior.digest(task) == seal["task_sha256"], "node task")
            receipt = seal["receipt"]
            need(r["nodes"][eid][i]["receipt"] == receipt and r["nodes"][eid][i]["status"] == "sealed",
                 "node receipt")
            encoded_events = [e for e in events if e["kind"] == "boundary_encoded"
                              and e["episode_id"] == eid and e["actual_step"] == boundary]
            need(len(encoded_events) == 1 and encoded_events[0]["index"] == obs["encoded_index"]
                 and encoded_events[0]["encoding_sha256"] == obs["encoding_sha256"], "encoding event")
            commands = f32(receipt["commands"], (5, 2))
            trace = Path(receipt["trace"])
            need(commands.tobytes() == actions[boundary:boundary+5].tobytes()
                 and trace.parent == RUN / (stem + "_winner") and sha(trace) == receipt["trace_sha256"],
                 "executed packet/trace")
            result.append(dict(arm=arm, boundary=boundary, task=task, actions=commands,
                               target=checkpoints[i+1], source=checkpoints[i], task_sha256=seal["task_sha256"],
                               seal_sha256=sha(RUN / (stem+"_seal.json"))))
    need(len(result) == 16 and sha(RUN / "report.json") == REPORT_SHA, "complete/no drift")
    return p, r, result


def configure():
    engine.RUN, engine.PREP = RUN, PREP
    engine.REPORT_SHA, engine.PREP_SHA, engine.NAME = REPORT_SHA, PREP_SHA, NAME
    engine.saved_inputs = saved_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    execute = sub.add_parser("run")
    execute.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    before = {str(p): sha(p) for p in (Path(__file__).resolve(), PROTOCOL, Path(saved.__file__).resolve())}
    if args.command == "inspect":
        _, _, nodes = saved_inputs()
        result = {"nodes": len(nodes), "neural_execution": False}
    else:
        with gpu.locking.gpu_lock(gpu.LOCK):
            result = engine.run(args.output, engine.AUTH)
    need(all(sha(p) == digest for p, digest in before.items()), "wrapper/source drift")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
