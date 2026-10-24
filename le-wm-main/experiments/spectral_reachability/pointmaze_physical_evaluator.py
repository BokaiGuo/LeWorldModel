"""Exact high-level PointMaze transition contract for SRA physical labels.

The recorded dataset stores one state per ``MazeEnv.step`` call, rather than
one state per MuJoCo integrator tick.  The source environment clips velocity
before each such action and uses ``frame_skip=5``.  Every SRA physical label
must use this helper so that a plausible-looking one-tick rollout cannot be
mistaken for an environment outcome.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


FRAME_SKIP = 5
VELOCITY_LIMIT = 5.0
ROOT = Path(__file__).resolve().parents[2]
# Fresh P2b panels use all still-unconsumed roots from 1864--1879 except
# 1877, which failed the fixed raw-replay threshold.  Older panels retain the
# immutable contract digest embedded in their own reports.
POSITION_CONTRACT = ROOT / "results/sra_pointmaze_replay_mujoco220_receipt_contract_v3_20261024_cuda/report.json"
LEGACY_PYTHON = ROOT / ".legacy_mujoco210_py310/bin/python"
REPLAY_BACKEND = Path(__file__).with_name("audit_sra_pointmaze_replay_backend.py")
ENDPOINT_BACKEND = Path(__file__).with_name("evaluate_sra_pointmaze_candidate_endpoints.py")


def build_model() -> Any:
    import mujoco

    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml

    return mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())


def replay_recorded_actions(model: Any, source: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Replay stored high-level actions under the vendored ``MazeEnv.step`` contract."""

    import mujoco

    source = np.asarray(source, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if source.shape != (4,):
        raise ValueError(f"PointMaze source state must have shape (4,), got {source.shape}")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"PointMaze actions must have shape (T, 2), got {actions.shape}")
    data = mujoco.MjData(model)
    data.qpos[:] = source[:2]
    data.qvel[:] = source[2:]
    mujoco.mj_forward(model, data)
    for action in actions:
        # Matches MazeEnv.step: clip_velocity() then do_simulation(action, 5).
        data.qvel[:] = np.clip(data.qvel, -VELOCITY_LIMIT, VELOCITY_LIMIT)
        mujoco.mj_forward(model, data)
        data.ctrl[:] = np.clip(action, -1.0, 1.0)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
    return np.r_[np.asarray(data.qpos[:2]), np.asarray(data.qvel[:2])]


def replay_position_endpoints(sources: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Use only the audited MuJoCo-2.2 position-label backend for SRA P1/P2."""

    sources, sequences = np.asarray(sources, dtype=np.float64), np.asarray(sequences, dtype=np.float64)
    if sources.ndim != 2 or sources.shape[1] != 4 or sequences.ndim != 3 or sequences.shape[0] != sources.shape[0] or sequences.shape[2] != 2:
        raise ValueError("sources must be [N,4] and sequences [N,T,2]")
    if not POSITION_CONTRACT.is_file() or not LEGACY_PYTHON.is_file():
        raise RuntimeError("audited MuJoCo-2.2 position backend is unavailable; do not substitute a different evaluator")
    contract = json.loads(POSITION_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("position_label_contract", {}).get("status") != "pass" or contract.get("backend", {}).get("backend", {}).get("mujoco") != "2.2.0":
        raise RuntimeError("MuJoCo backend lacks the audited position-only replay contract")
    with tempfile.TemporaryDirectory(prefix="sra_pointmaze_position_backend_") as temporary:
        folder = Path(temporary)
        request, xml, response = folder / "request.npz", folder / "pointmaze.xml", folder / "response.npz"
        np.savez_compressed(request, sources=sources, sequences=sequences)
        from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml

        xml.write_text(_legacy_pointmaze_xml(), encoding="utf-8")
        completed = subprocess.run([str(LEGACY_PYTHON), str(ENDPOINT_BACKEND), "--input", str(request), "--xml", str(xml), "--output", str(response)], capture_output=True, text=True, check=False)
        if completed.returncode or not response.is_file():
            raise RuntimeError(f"MuJoCo-2.2 endpoint backend failed: {completed.stderr[-1000:]}")
        expected_receipt = {
            "request_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
            "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
            "endpoint_script_sha256": hashlib.sha256(ENDPOINT_BACKEND.read_bytes()).hexdigest(),
            "replay_backend_script_sha256": hashlib.sha256(REPLAY_BACKEND.read_bytes()).hexdigest(),
        }
        with np.load(response, allow_pickle=False) as payload:
            actual_receipt = {key: str(payload[key].item()) for key in expected_receipt}
            endpoints = payload["endpoints"]
        if actual_receipt != expected_receipt:
            raise RuntimeError("endpoint response receipt does not match the requested physical rollout")
    if endpoints.shape != sources.shape or not np.isfinite(endpoints).all():
        raise RuntimeError("invalid MuJoCo-2.2 endpoint response")
    return endpoints


def position_backend_receipt() -> dict[str, object]:
    """Expose immutable MuJoCo-2.2 position-contract provenance to P1/P2."""

    if not POSITION_CONTRACT.is_file():
        raise RuntimeError("audited MuJoCo-2.2 position contract is unavailable")
    payload = json.loads(POSITION_CONTRACT.read_text(encoding="utf-8"))
    contract = payload.get("position_label_contract", {})
    if contract.get("status") != "pass":
        raise RuntimeError("position contract did not pass")
    inputs = payload.get("inputs", {})
    expected = {
        "replay_backend_script_sha256": hashlib.sha256(REPLAY_BACKEND.read_bytes()).hexdigest(),
        "endpoint_backend_script_sha256": hashlib.sha256(ENDPOINT_BACKEND.read_bytes()).hexdigest(),
    }
    if any(inputs.get(key) != value for key, value in expected.items()):
        raise RuntimeError("position contract does not bind the current endpoint evaluator scripts")
    return {"contract": str(POSITION_CONTRACT), "contract_sha256": hashlib.sha256(POSITION_CONTRACT.read_bytes()).hexdigest(), "backend": payload.get("backend", {}).get("backend", {}), "position_label_contract": contract, "endpoint_evaluator_scripts": expected}
