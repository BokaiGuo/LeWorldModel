"""CUDA-gated parent for a version-isolated PointMaze replay backend audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, require_cuda, sha256_file, write_json


def parse_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item]
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("integer lists must be nonempty and unique")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--roots", default="1880,1881,1882,1883,1884,1885,1886,1887,1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899")
    parser.add_argument("--frames", default="0,5,10,15,20,25,30,35,40")
    parser.add_argument("--horizons", default="5,10")
    parser.add_argument("--state-tolerance", type=float, default=2e-6)
    parser.add_argument("--position-tolerance", type=float, default=3e-6)
    parser.add_argument("--allow-position-label-contract", action="store_true")
    parser.add_argument("--engine", choices=("mujoco", "mujoco_py"), default="mujoco")
    parser.add_argument("--mujoco-py-path", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.legacy_python.is_file():
        raise FileExistsError("output must be new and legacy interpreter must exist")
    device = require_cuda()
    roots, frames, horizons = parse_ints(args.roots), parse_ints(args.frames), parse_ints(args.horizons)
    states_path, actions_path = args.dataset_root / "states.pth", args.dataset_root / "actions.pth"
    states = torch.load(states_path, map_location="cpu", weights_only=True).numpy().astype(np.float64)
    actions = torch.load(actions_path, map_location="cpu", weights_only=True).numpy().astype(np.float64)
    if max(frames) + max(horizons) >= states.shape[1]:
        raise ValueError("requested replay range exceeds dataset")
    args.output_dir.mkdir(parents=True)
    subset, xml_path, backend_report = args.output_dir / "raw_subset.npz", args.output_dir / "legacy_pointmaze.xml", args.output_dir / "backend_report.json"
    replay_backend_script = Path(__file__).with_name("audit_sra_pointmaze_replay_backend.py")
    endpoint_backend_script = Path(__file__).with_name("evaluate_sra_pointmaze_candidate_endpoints.py")
    np.savez_compressed(subset, states=states, actions=actions, roots=np.asarray(roots), frames=np.asarray(frames), horizons=np.asarray(horizons))
    xml = _legacy_pointmaze_xml()
    xml_path.write_text(xml, encoding="utf-8")
    command = [str(args.legacy_python), str(replay_backend_script), "--subset", str(subset), "--xml", str(xml_path), "--report", str(backend_report), "--state-tolerance", str(args.state_tolerance), "--engine", args.engine]
    environment = os.environ.copy()
    runtime_receipt: dict[str, str] | None = None
    if args.engine == "mujoco_py":
        if args.mujoco_py_path is None or not (args.mujoco_py_path / "bin/libmujoco210.so").is_file():
            raise FileNotFoundError("mujoco_py engine requires a validated MuJoCo 2.1 runtime path")
        runtime = args.mujoco_py_path.resolve()
        runtime_library = runtime / "lib/libmujoco.so.2.1.2"
        if not runtime_library.is_file():
            raise FileNotFoundError("mujoco_py runtime must expose libmujoco.so.2.1.2")
        environment["MUJOCO_PY_MUJOCO_PATH"] = str(runtime)
        environment["LD_LIBRARY_PATH"] = ":".join((str(runtime / "lib"), str(runtime / "bin"), "/usr/lib/nvidia", environment.get("LD_LIBRARY_PATH", "")))
        runtime_receipt = {"path": str(runtime), "libmujoco_sha256": sha256_file(runtime_library)}
    completed = subprocess.run(command, check=False, text=True, capture_output=True, env=environment)
    if not backend_report.is_file():
        raise RuntimeError(f"legacy backend emitted no report: {completed.stderr[-1000:]}")
    backend = json.loads(backend_report.read_text(encoding="utf-8"))
    position_pass = bool(backend["summary"]["max_position_l2_error"] <= args.position_tolerance)
    position_contract = {"status": "pass" if position_pass else "fail", "scope": "position-only labels used by SRA P1/P2; not a full-state simulator equivalence claim", "position_tolerance": args.position_tolerance, "max_position_l2_error": backend["summary"]["max_position_l2_error"]}
    status = backend["status"] if backend["status"] == "pass" else ("strict_state_fail_position_label_pass" if position_pass else "fail")
    report = {"schema": "sra_pointmaze_versioned_replay_contract.v3", "status": status, "scope": "host CUDA gate plus backend-version replay fidelity; no learned-model CPU fallback", "cuda_host_preflight": {"device": str(device), "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda}, "inputs": {"dataset": str(args.dataset_root), "states_sha256": sha256_file(states_path), "actions_sha256": sha256_file(actions_path), "legacy_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "backend_python": str(args.legacy_python), "replay_backend_script": str(replay_backend_script), "replay_backend_script_sha256": sha256_file(replay_backend_script), "endpoint_backend_script": str(endpoint_backend_script), "endpoint_backend_script_sha256": sha256_file(endpoint_backend_script), "mujoco_py_runtime": runtime_receipt, "backend_stdout": completed.stdout[-2000:]}, "position_label_contract": position_contract, "backend": backend}
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"output": str(args.output_dir), "status": report["status"], "backend": backend["backend"], "summary": backend["summary"]}, sort_keys=True))
    if completed.returncode or (backend["status"] != "pass" and not (args.allow_position_label_contract and position_pass)):
        raise RuntimeError("selected replay backend did not meet the exact raw-transition contract")


if __name__ == "__main__":
    main()
