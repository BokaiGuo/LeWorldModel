"""Bounded root 1900 replay diagnosis; no modification of historical results."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments import a0_pointmaze_replay_audit_20261102_codex_exclusive as prior
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml

ROOT = Path(__file__).resolve().parents[1]
NAME = "a0_pointmaze_frame25_audit_20261102_codex_exclusive"
BACKEND = ROOT / "experiments/a0_pointmaze_frame25_backend_20261102_codex_exclusive.py"
REPLAY_RESULT = ROOT / "results/a0_pointmaze_replay_audit_20261102_codex_exclusive_RESULT01.json"
REPLAY_SHA = "35ccd1cab3ec67d64ae90d29d3bc3a90eb42ae0382e789514efb922902226984"


def build():
    prior.require(prior.sha(REPLAY_RESULT) == REPLAY_SHA, "prior audit identity")
    old = json.loads(REPLAY_RESULT.read_bytes())
    for key in ("states", "actions"):
        prior.require(prior.sha(prior.DATA / (key + ".pth")) == old["source_hashes"][key], "archive identity")
    states = torch.load(prior.DATA / "states.pth", map_location="cpu", weights_only=True).numpy()[1900]
    actions = torch.load(prior.DATA / "actions.pth", map_location="cpu", weights_only=True).numpy()[1900]
    xml = _legacy_pointmaze_xml()
    prior.require(hashlib.sha256(xml.encode()).hexdigest() == old["source_hashes"]["xml"], "XML identity")
    with tempfile.TemporaryDirectory(prefix="a0_frame25_") as tmp:
        directory = Path(tmp)
        request, model = directory / "input.npz", directory / "model.xml"
        np.savez_compressed(request, states=states, actions=actions)
        model.write_text(xml)
        versions = []
        for tag, python in (("historical_2_2", prior.physical.LEGACY_PYTHON), ("current_3_1", Path(sys.executable))):
            output = directory / (tag + ".json")
            result = subprocess.run([str(python), "-B", str(BACKEND), "--input", str(request),
                                     "--xml", str(model), "--output", str(output)],
                                    capture_output=True, text=True, timeout=120, check=False)
            prior.require(result.returncode == 0 and output.is_file(),
                          tag + " backend failed: " + result.stderr[-2000:])
            versions.append(json.loads(output.read_bytes()))
    prior.require([v["runtime"]["mujoco"] for v in versions] == ["2.2.0", "3.1.6"],
                  "declared runtime identity")
    prior.require(all(len(v["runs"]) == 5 for v in versions), "run count")
    return {"schema": NAME + ".v1", "historical_replay_sha256": REPLAY_SHA,
            "inputs": {"root": 1900, "frame": 25, "states_sha256": old["source_hashes"]["states"],
                       "actions_sha256": old["source_hashes"]["actions"], "xml_sha256": old["source_hashes"]["xml"],
                       "backend_sha256": prior.sha(BACKEND)},
            "protocol": {"continuous_starts": [0, 20, 23, 24, 25],
                         "branches": ["continuous", "full_clone", "qv_from_continuous", "qv_from_archive"],
                         "prediction_frames": [26, 27],
                         "precision_variants": "float32 round-trip to inspect sensitivity only; not an interface correction",
                         "archive_hidden_state": "unavailable; recorded archive has qpos/qvel only",
                         "thresholds_unchanged": True, "training": False, "gpu": False},
            "versions": versions, "P1_gate_changed": False,
            "interpretation": "comparison to archive and internal reset behavior; not proof of dataset generator runtime or hidden fields"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prior.require(args.output.is_absolute() and args.output.parent == ROOT / "results"
                  and args.output.name.startswith(NAME + "_") and args.output.suffix == ".json", "exclusive output")
    print("Diagnosing root 1900 frame 25 with MuJoCo 2.2 and 3.1; CPU only.")
    result = build()
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
