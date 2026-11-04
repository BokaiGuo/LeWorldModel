"""Export observation/action-only inputs for a second feedback planning node.

This is a development-root input export, not a planner or a closed-loop result.
The preceding saved-trace verifier must accept the entire fixed-horizon panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_residual_cp5_timing_saved_20261102 as saved


def node_input(role, row, image_dir):
    """Keep exactly the two observed RGB frames and two executed action packets."""
    if role["arm"] != row["arm"] or row["status"] != "fixed_horizon_completed":
        raise ValueError("arm binding or incomplete execution")
    transitions = row["transitions"]
    packets = []
    for phase, last_step in (("prefix", 5), ("tail", 5)):
        packet = [t for t in transitions if t["phase"] == phase and t["step"] <= last_step]
        if [t["step"] for t in packet] != list(range(1, 6)):
            raise ValueError("incomplete or unordered action packet")
        actions = np.asarray([t["action"] for t in packet], dtype="<f4")
        if (actions.shape != (5, 2) or not np.isfinite(actions).all()
                or (np.abs(actions) > 1).any()
                or any(t["terminated"] or t["truncated"] for t in packet)):
            raise ValueError("invalid or censored action packet")
        packets.append(actions)

    frames = []
    for phase in ("prefix", "tail"):
        matches = [o for o in row["observations"] if o["phase"] == phase and o["step"] == 5]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous RGB checkpoint")
        obs = matches[0]
        path = image_dir / obs["file"]
        if path.parent != image_dir or path.is_symlink() or not path.is_file():
            raise ValueError("unsafe RGB path")
        if saved.physical.sha(path) != obs["file_sha256"]:
            raise ValueError("RGB file changed")
        frames.append({"path": str(path.resolve()), "file_sha256": obs["file_sha256"],
                       "pixel_sha256": obs["pixel_sha256"]})

    actions = np.stack(packets)
    return {"arm": role["arm"], "label": role["label"], "previous_objective": role["metric"],
            "rgb_cp5": frames[0], "rgb_cp10": frames[1],
            "executed_raw_action_packets": actions.tolist(),
            "action_packets_sha256": hashlib.sha256(actions.tobytes()).hexdigest()}


def build():
    verification = saved.analyze()
    contract = saved.physical.read(saved.CONTRACT, saved.CONTRACT_SHA)
    report = saved.physical.read(saved.RUN / "report.json", saved.REPORT_SHA)
    if (verification["frame_references_verified"] != 144
            or len(contract["roles"]) != 8 or len(report["rows"]) != 8):
        raise ValueError("verified source panel is incomplete")
    rows = [node_input(role, row, saved.RUN)
            for role, row in zip(contract["roles"], report["rows"], strict=True)]
    return {"schema": "p1_timing_second_feedback_inputs.v1",
            "classification": "known_development_root_input_only",
            "source_contract_sha256": saved.CONTRACT_SHA,
            "source_report_sha256": saved.REPORT_SHA,
            "model_context": "warmup10 RGB latent, cp5 RGB latent, cp10 RGB latent; "
                             "executed prefix and first tail packets",
            "normalization": "apply checkpoint action mean/std once at model input",
            "rows": rows, "planning_performed": False, "physical_execution_performed": False,
            "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (json.dumps(build(), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
