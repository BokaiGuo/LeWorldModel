import json
import subprocess
import sys
from pathlib import Path

from experiments.run_frozen_protocol_queue import inputs_ready

ROOT = Path(__file__).resolve().parents[1]


def test_external_validity_manifest_is_frozen_and_contains_all_target_mechanisms(tmp_path):
    output = tmp_path / "protocol"
    subprocess.run(
        [sys.executable, "experiments/prepare_external_validity_protocol.py", "--output", str(output), "--device", "cpu"],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads((output / "manifest.json").read_text())
    command = payload["commands"]["cross_environment_prediction"]
    assert "swm/PushT-v1" in command
    assert "swm/TwoRoom-v1" in command
    assert "swm/SimplePointMaze-v0" in command
    assert len(payload["sha256_before_execution"]) == 64
    repeated = subprocess.run(
        [sys.executable, "experiments/prepare_external_validity_protocol.py", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite frozen manifest" in repeated.stderr


def test_downstream_queue_refuses_to_substitute_a_missing_checkpoint():
    values = ["python", "experiments/audit_causal_visual_checkpoint.py", "--checkpoint", "missing.pt", "--output", "unused"]
    assert inputs_ready(values, "selection") is False
