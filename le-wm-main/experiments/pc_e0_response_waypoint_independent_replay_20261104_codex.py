"""Replay every sealed waypoint-panel prediction with separate wall geometry."""
import json

import numpy as np

from experiments import pc_e0_response_waypoint_direction_dev_20261104_codex as source
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent


def main():
    output = source.OUT / "independent_geometry_replay.json"
    if output.exists():
        raise FileExistsError(output)
    doc = json.loads((source.OUT / "result.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    if audit["result_sha256"] != source.prior.previous.sha(source.OUT / "result.json"):
        raise ValueError("result not audited")
    samples, transitions, rows = 0, 0, []
    for item in doc["rows"]:
        full, crossed, selected_crossed = 0, 0, None
        with np.load(source.OUT / item["trace_file"], allow_pickle=False) as trace:
            for index in range(item["batches"]):
                actions = trace[f"actions_{index}"]
                path, blocked = independent(item["task"]["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], trace[f"positions_{index}"], atol=1e-12, rtol=0)
                samples += len(actions)
                transitions += len(actions) * actions.shape[1]
                if actions.shape[1] == 15:
                    full += len(actions)
                    crossed += int(np.any(path[:, :, 0] > 0, axis=1).sum())
            if item["selected"] is not None:
                path, _ = independent(item["task"]["start"], np.asarray(item["selected"], np.float32)[None])
                selected_crossed = bool(np.any(path[0, :, 0] > 0))
        rows.append({"root": item["root"], "arm": item["arm"], "full_chains": full,
                     "crossed_chains": crossed, "selected_crossed": selected_crossed})
    source.prior.previous.write(output, {"status": "passed", "result_sha256": source.prior.previous.sha(source.OUT / "result.json"),
        "samples": samples, "transitions": transitions, "rows": rows,
        "scope": "separately implemented geometric replay of the same abstract wall; not another environment"})
    print(json.dumps({"status": "passed", "samples": samples, "transitions": transitions, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
