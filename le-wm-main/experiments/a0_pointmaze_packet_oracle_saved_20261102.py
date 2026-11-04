"""Independently verify saved packet oracle choices and rerun selected endpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import a0_pointmaze_packet_oracle_20261102 as source
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

ROOT = source.ROOT
REPORT = ROOT / "results/a0_pointmaze_packet_oracle_20261102_DEV01.json"
REPORT_SHA = "3b38090507edc8f787cf0981e4d93a0d9d76636a593e2356e77386a4953a7b2e"


def analyze():
    need = source.require
    need(source.earlier.digest(REPORT) == REPORT_SHA, "report digest")
    report = json.loads(REPORT.read_bytes())
    need(report["source"]["code_sha256"] == source.earlier.digest(source.__file__)
         and report["source"]["strict_sha256"] == source.earlier.SOURCE_SHA
         and report["source"]["previous_sha256"] == source.PREVIOUS_SHA
         and report["source"]["position_backend_receipt"] == physical.position_backend_receipt(), "source")
    inputs = source.earlier.extract_inputs(json.loads(source.earlier.SOURCE.read_bytes()))
    previous = json.loads(source.PREVIOUS.read_bytes())
    need(len(inputs) == len(report["rows"]) == len(previous["rows"]) == 8, "roster")
    results = []
    for item, row, old in zip(inputs, report["rows"], previous["rows"], strict=True):
        bank = np.asarray(row["candidate_bank"], dtype=np.float64)
        target = np.asarray(row["target"], dtype=float)
        p = np.asarray(row["prefix_positions"], dtype=float)
        t = np.asarray(row["terminal_positions"], dtype=float)
        need(row["root"] == old["root"] == item["root"] and np.array_equal(bank, item["bank"])
             and np.array_equal(target, item["target"]) and p.shape == (32, 2)
             and t.shape == (32, 32, 2) and np.isfinite(p).all() and np.isfinite(t).all(), "row")
        pc = np.linalg.norm(p-target, axis=-1)
        tc = np.linalg.norm(t-target, axis=-1)
        prefix = int(np.argmin(pc))
        i, j = np.unravel_index(int(np.argmin(tc)), tc.shape)
        tail = int(np.argmin(tc[prefix]))
        need(row["prefix_selected_index"] == prefix and row["terminal_selected_indices"] == [int(i), int(j)]
             and row["prefix_selected_best_terminal_tail_index"] == tail
             and row["first_action_differs"] == bool(i != prefix)
             and np.isclose(row["terminal_cost_at_terminal_choice"], tc[i, j], atol=1e-12)
             and np.isclose(row["terminal_cost_with_prefix_first_best_tail"], tc[prefix, tail], atol=1e-12),
             "oracle argmins")
        seq1 = np.repeat(bank[prefix][None, :], 5, axis=0)[None]
        seq2 = np.concatenate((np.repeat(bank[i][None, :], 5, axis=0),
                               np.repeat(bank[j][None, :], 5, axis=0)))[None]
        state = np.asarray(row["source_state_privileged"], dtype=np.float64)[None]
        need(np.array_equal(state[0], item["source_state"]), "state binding")
        one = physical.replay_position_endpoints(state, seq1)[0, :2]
        two = physical.replay_position_endpoints(state, seq2)[0, :2]
        need(np.allclose(one, p[prefix], atol=1e-12, rtol=0)
             and np.allclose(two, t[i, j], atol=1e-12, rtol=0), "replay witness")
        results.append({"root": item["root"], "packet_first_differs": bool(i != prefix),
                        "old_first_differs": old["first_action_differs"],
                        "prefix_cost_gap": float(pc[i] - pc[prefix]),
                        "terminal_advantage": float(tc[prefix, tail] - tc[i, j])})
    return {"schema": source.NAME + ".analysis.v1", "rows": results,
            "packet_different_first": sum(x["packet_first_differs"] for x in results),
            "old_different_first": sum(x["old_first_differs"] for x in results),
            "scope": "8 previously opened roots; position-only privileged fixed-bank offline oracle",
            "p1_admission": False, "scientific_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source.require(args.output.is_absolute() and args.output.parent == ROOT / "results"
                   and args.output.name.startswith(source.NAME + "_analysis_"), "exclusive output")
    raw = (json.dumps(analyze(), sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(raw)
    print(hashlib.sha256(raw).hexdigest())
