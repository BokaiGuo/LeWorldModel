"""Save two-version contact traces tied to the earlier frame-25 audit."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments import a0_pointmaze_frame25_audit_20261102_codex_exclusive as prior
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml

NAME = "a0_pointmaze_frame25_contact_audit_20261102_codex_exclusive"
RESULT = prior.ROOT / "results/a0_pointmaze_frame25_audit_20261102_codex_exclusive_RESULT01.json"
RESULT_SHA = "3da0d0916c05fd8db4113d41d07d748d388d28836702075d48fba870c3d3015e"
BACKEND = prior.ROOT / "experiments/a0_pointmaze_frame25_contact_20261102_codex_exclusive.py"


def build():
    prior.prior.require(prior.prior.sha(RESULT) == RESULT_SHA, "frame25 audit identity")
    old = json.loads(RESULT.read_bytes())
    for name in ("states", "actions"):
        prior.prior.require(prior.prior.sha(prior.prior.DATA / (name + ".pth")) ==
                            old["inputs"][name + "_sha256"], "source identity")
    prior.prior.require(prior.prior.sha(prior.BACKEND) == old["inputs"]["backend_sha256"],
                        "frame25 backend identity")
    states = torch.load(prior.prior.DATA / "states.pth", map_location="cpu", weights_only=True).numpy()[1900]
    actions = torch.load(prior.prior.DATA / "actions.pth", map_location="cpu", weights_only=True).numpy()[1900]
    with tempfile.TemporaryDirectory(prefix="a0_contact_") as tmp:
        directory = Path(tmp)
        request, xml = directory / "input.npz", directory / "model.xml"
        np.savez_compressed(request, states=states, actions=actions)
        xml.write_text(_legacy_pointmaze_xml())
        results = []
        for index, python in enumerate((prior.prior.physical.LEGACY_PYTHON, Path(sys.executable))):
            output = directory / f"{index}.json"
            proc = subprocess.run([str(python), "-B", str(BACKEND), "--input", str(request),
                                   "--xml", str(xml), "--output", str(output)],
                                  capture_output=True, text=True, timeout=120, check=False)
            prior.prior.require(proc.returncode == 0 and output.is_file(), "contact backend: " + proc.stderr[-2000:])
            results.append(json.loads(output.read_bytes()))
    prior.prior.require([x["mujoco"] for x in results] == ["2.2.0", "3.1.6"], "versions")
    return {"schema": NAME + ".v1", "source_audit_sha256": RESULT_SHA,
            "backend_sha256": prior.prior.sha(BACKEND), "results": results,
            "scope": "frame 25 continuous from archive frame 0 vs qpos/qvel reset; contact trace only",
            "archive_hidden_state": "not available", "thresholds_unchanged": True,
            "P1_gate_changed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior.prior.require(args.output.is_absolute() and args.output.parent == prior.ROOT / "results"
                        and args.output.name.startswith(NAME + "_") and args.output.suffix == ".json",
                        "exclusive output")
    report = build()
    raw = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as f:
        f.write(raw)
    print(json.dumps({"output": str(args.output), "sha256": prior.hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
