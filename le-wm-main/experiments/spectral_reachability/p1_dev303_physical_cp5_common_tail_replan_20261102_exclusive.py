"""Exclusive dev303 physical/cp5 common-tail/replan contract and CUDA entrypoint.

The CPU phase consumes only the sealed dev303 selector preparation/report through
the sealed physical-prefix bridge.  It does not inspect development outcome
artifacts and it never selects or rewrites an action after the selector seal.
The CUDA phase is an explicit host preflight boundary; physical execution is
kept behind a separate authorization flag until the native executor is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_physical_prefix_bridge_20261101_exclusive as bridge


NAME = "p1_dev303_physical_cp5_common_tail_replan_20261102_exclusive"
ROOT = bridge.Path(__file__).resolve().parents[2]
ROLES = bridge.ROLES
PHYSICAL_HORIZON = 15
PREFIX_HORIZON = 5
TAIL_HORIZON = 10


def require(value, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def action_sha(actions: np.ndarray) -> str:
    return sha(np.asarray(actions, dtype="<f4").tobytes())


def _prefix(value, role: str) -> np.ndarray:
    array = np.asarray(value, dtype="<f4")
    require(array.shape == (PREFIX_HORIZON, 2), f"{role}: prefix shape")
    require(np.isfinite(array).all() and (np.abs(array) <= 1).all(),
            f"{role}: bounded finite prefix")
    return array


def build_contract(source: dict, selector_contract: dict) -> dict:
    """Build a CPU-only physical contract from the sealed bridge output."""
    require(source["seed"] == bridge.SEED and
            source["queue_sha256"] == bridge.QUEUE_SHA and
            source["source_receipt_sha256"] == bridge.RECEIPT_SHA,
            "dev303 source identity")
    require(selector_contract["schema"] == bridge.NAME + ".contract.v1",
            "sealed bridge contract schema")
    require(selector_contract["source"] == source, "bridge source binding")
    require(selector_contract["selection_input"] == "rgb_only" and
            selector_contract["feedback_boundary"] == PREFIX_HORIZON and
            selector_contract["physical_execution_authorized"] is False and
            selector_contract["independent_confirmation"] is False and
            selector_contract["P1_admission"] is False,
            "selector/bridge boundary")
    rows = []
    for row in selector_contract["roles"]:
        role = row["role"]
        require(role in ROLES, f"unknown role: {role}")
        prefix = _prefix(row["actions"], role)
        require(row["actions_sha256"] == action_sha(prefix),
                f"{role}: sealed prefix SHA")
        rows.append({
            "role": role,
            "candidate_index": int(row["candidate_index"]),
            "candidate_label": row["candidate_label"],
            "prefix_actions": prefix.tolist(),
            "prefix_sha256": action_sha(prefix),
            "tail_policy": "cp5_common_tail_replan_from_actual_rgb",
            "tail_horizon": TAIL_HORIZON,
        })
    require([row["role"] for row in rows] == list(ROLES), "complete role roster")
    return {
        "schema": NAME + ".contract.v1",
        "status": "cpu_contract_sealed_waiting_for_cuda",
        "seed": bridge.SEED,
        "roles": rows,
        "selection_input": "sealed_rgb_only_selector",
        "simulator_state_used_for_selection": False,
        "development_outcomes_used_for_selection": False,
        "feedback_boundary": PREFIX_HORIZON,
        "common_tail": {
            "mode": "pending_native_common_tail_and_replanner",
            "horizon": TAIL_HORIZON,
            "selection_after_cp5": "forbidden",
            "action_source": "sealed selector/bridge plus actual cp5 RGB runtime input",
        },
        "physical_horizon": PHYSICAL_HORIZON,
        "same_replanning_budget_required": True,
        "new_neural_calls_during_prefix": 0,
        "P1_admission": False,
        "independent_confirmation": False,
        "physical_execution_authorized": False,
        "bindings": {
            "bridge_schema": bridge.NAME + ".contract.v1",
            "source_queue_sha256": bridge.QUEUE_SHA,
            "source_receipt_sha256": bridge.RECEIPT_SHA,
        },
    }


def prepare_contract(prepare_path, prepare_sha: str, report_path, report_sha: str) -> dict:
    """Read only the explicitly named sealed selector/bridge inputs."""
    contract = bridge.prepare_contract(prepare_path, prepare_sha, report_path, report_sha)
    source = bridge.verified_source()
    return build_contract(source, contract)


def write_new(path: Path, value: dict) -> str:
    path = Path(path).absolute()
    if os.path.lexists(path):
        raise FileExistsError(path)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(payload)


def cuda_preflight() -> dict:
    """Run the host-CUDA check without opening physical outcomes."""
    import torch

    require(torch.cuda.is_available(), "host CUDA unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "cuda": torch.version.cuda,
        "status": "host_cuda_preflight_passed_no_physical_outcome",
        "P1_admission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-path", required=True)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--selector-report", required=True)
    parser.add_argument("--selector-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda-preflight", action="store_true")
    parser.add_argument("--authorize-cuda-and-physics", action="store_true")
    args = parser.parse_args()
    require(args.output.lower().find("outcome") < 0, "outcome output forbidden")
    contract = prepare_contract(
        args.prepare_path, args.prepare_sha256,
        args.selector_report, args.selector_report_sha256,
    )
    digest = write_new(Path(args.output), contract)
    result = {"contract": os.path.abspath(args.output), "sha256": digest,
              "P1_admission": False}
    if args.cuda_preflight:
        result["cuda"] = cuda_preflight()
    if args.authorize_cuda_and_physics:
        raise RuntimeError(
            "physical executor is intentionally not enabled by this CPU-contract step; "
            "use a separately authorized native executor"
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
