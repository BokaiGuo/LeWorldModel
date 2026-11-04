"""Verify 1/2 archive-action clock and saved-state restoration on real data."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments import a0_pointmaze_prefix_terminal_oracle_20261102_codex_exclusive as previous
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

ROOT = Path(__file__).resolve().parents[1]
NAME = "a0_pointmaze_replay_audit_20261102_codex_exclusive"
DATA = ROOT / "data/point_maze_tail1800_1899"
BACKEND = ROOT / "experiments/a0_pointmaze_replay_backend_20261102_codex_exclusive.py"
PRIOR_RESULT = ROOT / "results/a0_pointmaze_prefix_terminal_oracle_20261102_codex_exclusive_RESULT01.json"
PRIOR_SHA = "002abb3fc1e431ea6273e5c466fb4aa91fdd8866684cee394ddd674292a472d7"
FRAMES = (0, 5, 10, 15, 20, 25, 30, 35, 40)
POSITION_TOLERANCE = 3e-6
STATE_TOLERANCE = 2e-6
RESET_TOLERANCE = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build():
    require(sha(previous.SOURCE) == previous.SOURCE_SHA and sha(PRIOR_RESULT) == PRIOR_SHA,
            "sealed source/result identity")
    datasets = {key: torch.load(DATA / (key + ".pth"), map_location="cpu",
                                 weights_only=True).numpy()
                for key in ("states", "actions", "seq_lengths")}
    states, actions, lengths = datasets["states"], datasets["actions"], datasets["seq_lengths"]
    require(states.shape == (2000, 100, 4) and actions.shape == (2000, 100, 2)
            and lengths.shape == (2000,), "archive shapes")
    roots = np.array(previous.ROOTS, dtype=np.int64)
    frames = np.array(FRAMES, dtype=np.int64)
    require(all(int(lengths[r]) > max(FRAMES) + 2 for r in roots), "valid trajectory extent")
    xml = _legacy_pointmaze_xml()
    contract = physical.position_backend_receipt()
    with tempfile.TemporaryDirectory(prefix="a0_pointmaze_replay_") as temporary:
        directory = Path(temporary)
        request, xmlfile, output = directory / "input.npz", directory / "model.xml", directory / "report.json"
        np.savez_compressed(request, states=states, actions=actions, lengths=lengths,
                            roots=roots, frames=frames)
        xmlfile.write_text(xml)
        completed = subprocess.run([str(physical.LEGACY_PYTHON), str(BACKEND),
                                    "--input", str(request), "--xml", str(xmlfile),
                                    "--output", str(output)], capture_output=True, text=True,
                                   timeout=120, check=False)
        require(completed.returncode == 0 and output.is_file(),
                "legacy replay backend failed: " + completed.stderr[-3000:])
        report = json.loads(output.read_bytes())
    require(report["backend"] == {"engine": "mujoco", "version": "2.2.0"}, "backend version")
    rows, resets = report["archive_rows"], report["reset_rows"]
    require(len(rows) == len(roots)*len(frames)*2 and len(resets) == len(roots)*2, "row counts")
    max_position = max(r["position_error"] for r in rows)
    max_state = max(r["state_error"] for r in rows)
    max_fresh = max(r["fresh_state_error"] for r in resets)
    max_reused = max(r["reused_state_error"] for r in resets)
    saved = json.loads(PRIOR_RESULT.read_bytes())
    saved_rows = saved["rows"]
    sources = np.array([r["source_state_privileged"] for r in saved_rows])
    banks = [np.asarray(r["candidate_bank"]) for r in saved_rows]
    prefixes = np.array([[b[r["prefix_selected_index"]]] for b, r in zip(banks, saved_rows, strict=True)])
    pairs = np.array([[b[i] for i in r["terminal_selected_indices"]]
                      for b, r in zip(banks, saved_rows, strict=True)])
    one = physical.replay_position_endpoints(sources, prefixes)
    two = physical.replay_position_endpoints(sources, pairs)
    differences = []
    for i, row in enumerate(saved_rows):
        a, b = row["terminal_selected_indices"]
        k = row["prefix_selected_index"]
        differences.append({"root": row["root"],
                            "saved_prefix_position_error": float(np.linalg.norm(one[i, :2] - row["prefix_positions"][k])),
                            "saved_terminal_position_error": float(np.linalg.norm(two[i, :2] - row["terminal_positions"][a][b]))})
    saved_error = max(max(x["saved_prefix_position_error"], x["saved_terminal_position_error"])
                      for x in differences)
    return {
        "schema": NAME + ".v1", "backend": report["backend"],
        "source_hashes": {"previous_result": PRIOR_SHA, "strict_report": previous.SOURCE_SHA,
                          "states": sha(DATA / "states.pth"), "actions": sha(DATA / "actions.pth"),
                          "lengths": sha(DATA / "seq_lengths.pth"), "backend_script": sha(BACKEND),
                          "xml": hashlib.sha256(xml.encode()).hexdigest()},
        "historical_position_receipt_sha256": contract["contract_sha256"],
        "protocol": {"roots": list(previous.ROOTS), "frames": list(FRAMES),
                     "archive_actions": [1, 2], "mujoco_ticks_per_archive_action": 5,
                     "velocity_clip_before_each_action": 5.0, "action_clipped": [-1, 1],
                     "action_repetition": "one candidate vector is one archive action, not five archive actions",
                     "planner_model_token_warning": "a model block may repeat five archive actions; this audit does not equate one action with one model block",
                     "position_tolerance": POSITION_TOLERANCE, "full_state_tolerance": STATE_TOLERANCE,
                     "reset_tolerance": RESET_TOLERANCE, "roots_previously_opened": True,
                     "state_restore": "qpos and qvel only; compare continuous to fresh and contaminated reused MjData",
                     "training": False, "gpu": False},
        "summary": {"archive_row_count": len(rows), "max_position_error": max_position,
                    "max_state_error": max_state, "position_pass": max_position <= POSITION_TOLERANCE,
                    "state_pass": max_state <= STATE_TOLERANCE,
                    "max_fresh_reset_state_error": max_fresh, "fresh_reset_pass": max_fresh <= RESET_TOLERANCE,
                    "max_reused_reset_state_error": max_reused, "reused_reset_pass": max_reused <= RESET_TOLERANCE,
                    "max_saved_oracle_endpoint_error": saved_error, "saved_endpoint_pass": saved_error <= RESET_TOLERANCE},
        "archive_rows": rows, "reset_rows": resets, "saved_endpoint_rows": differences,
        "interpretation": "bounded replay diagnostic only; no universal full-state or RGB planning certification",
        "P1_gate_changed": False, "scientific_admission": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent == ROOT / "results"
            and args.output.name.startswith(NAME + "_") and args.output.suffix == ".json",
            "exclusive output")
    print("Auditing recorded 1/2-action transitions, reset and saved endpoints; CPU only.")
    report = build()
    raw = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "summary": report["summary"]}))


if __name__ == "__main__":
    main()
