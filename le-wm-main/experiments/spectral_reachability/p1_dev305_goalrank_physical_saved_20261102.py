"""Verify all fixed dev305 spectral/random physical executions and censoring."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from . import p1_dev305_goalrank_spectral_physical_20261102 as physical
from . import p1_dev305_goalrank_saved_20261102 as model_saved
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

base, previous = physical.base, physical.previous
ROOT = physical.ROOT / "results"
CONTRACT_SHA = "56bdc2ab5d6497735489d9498c08ec4322ef22c1ac3d6ae10ef583c32794cb7a"
REPORT_SHA = "ad3f961cbc823dc1ff76896d193db3e1d04e394784f2e20b5c8f2c6373a4f16f"
need = physical.need


def analyze():
    model_saved.analyze()
    spec = base.read(ROOT / f"{physical.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    need(spec == physical.protocol(), "contract")
    directory = ROOT / f"{physical.NAME}_RUN_DEV01"
    report = base.read(directory / "report.json", REPORT_SHA)
    need(report["contract_sha256"] == CONTRACT_SHA and report["status"] == "recorded"
         and not report["errors"] and base.sha(directory / "journal.jsonl") == report["journal_sha256"],
         "report")
    prior = base.read(previous.ROOT / f"{previous.NAME}_EXEC_DEV01/report.json", previous.EXEC_SHA)
    capture = base.read(previous.ROOT / f"{previous.NAME}_CAPTURE_DEV01/report.json", previous.CAPTURE_SHA)
    model = base.read(ROOT / f"{physical.model_run.NAME}_RUN_DEV01/report.json", physical.MODEL_SHA)
    expected = [(role, candidate, old, model_row)
                for role, old, model_row in zip(spec["roles"], prior["rows"], model["rows"], strict=True)
                for candidate in role["candidates"]]
    need(len(expected) == len(report["rows"]) == 18, "complete roster")
    rows, frame_count = [], 0
    for row, (role, candidate, old, model_row) in zip(report["rows"], expected, strict=True):
        name = f"{role['arm']}_{candidate['index']:02d}"
        need(row["arm"] == name and base.read(directory / (name + ".row.json")) == row
             and row["status"] in ("fixed_horizon_completed", "native_early_stop"), "row status")
        expected_actions = np.concatenate((np.asarray(role["prefix"], dtype="<f4"),
                                           np.asarray(candidate["actions"], dtype="<f4").reshape(10, 2)))
        transitions = row["transitions"]
        need(6 <= len(transitions) <= 15, "transition count")
        for index, transition in enumerate(transitions):
            need(transition["phase"] == ("prefix" if index < 5 else "tail")
                 and transition["step"] == (index + 1 if index < 5 else index - 4)
                 and np.array_equal(transition["action"], expected_actions[index]), "action binding")
            if index < 5:
                need(transition["state"] == old["transitions"][index]["state"], "prefix replay")
            if index < len(transitions)-1:
                need(not transition["terminated"] and not transition["truncated"], "no steps after stop")
        stopped = transitions[-1]["terminated"] or transitions[-1]["truncated"]
        need((row["status"] == "native_early_stop" and stopped)
             or (row["status"] == "fixed_horizon_completed" and len(transitions) == 15 and not stopped),
             "stop and censoring")
        observations = row["observations"]
        tail_count = len(transitions)-5
        need([(o["phase"], o["step"]) for o in observations]
             == [("warmup", x) for x in (0, 5, 10)] + [("prefix", 5)]
             + [("tail", x) for x in range(1, tail_count+1)], "frame roster")
        for observation in observations:
            previous.verify_frame(directory, observation)
            frame_count += 1
        for observation, expected_obs in zip(observations[:4],
                                              [*old["observations"][:3], old["observations"][7]], strict=True):
            need(all(observation[k] == expected_obs[k] for k in ("pixel_sha256", "state", "snapshot")),
                 "RGB and checkpoint replay")
        need(all(obs["state"] == transition["state"]
                 for obs, transition in zip(observations[4:], transitions[5:], strict=True)), "tail state")
        item = model_row["candidates"][candidate["index"]]
        need(hashlib.sha256(np.asarray(item["actions"], dtype="<f4").tobytes()).hexdigest()
             == candidate["actions_sha256"], "candidate SHA")
        rows.append({"prefix": role["arm"], "candidate": candidate["index"], "kind": candidate["kind"],
                     "selected": candidate["index"] == role["model_selected_index"],
                     "model_score": item["score"], "tail_steps": tail_count, "status": row["status"],
                     "terminated": transitions[-1]["terminated"], "truncated": transitions[-1]["truncated"],
                     "terminal": measures(transitions[-1]["state"], capture["goal_state_evaluation_only"]),
                     "reward": transitions[-1]["reward"]})
    return {"schema": physical.NAME + ".saved.v1", "frames_verified": frame_count, "rows": rows,
            "scope": "saved actions, replayed states, frames and stop status; no new neural inference",
            "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
