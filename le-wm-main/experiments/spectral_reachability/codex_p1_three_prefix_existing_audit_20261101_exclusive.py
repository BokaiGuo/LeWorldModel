"""Read-only audit of existing three-prefix CUDA/physical development runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_four_arm_consumer_20261030 as consumer

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_p1_three_prefix_existing_audit_20261101_exclusive"
BASE = "p1_three_prefix_own_host_20261101_main"
ROLES = ("selected", "zero", "greedy")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_one(root, role):
    directory = ROOT / "results" / f"{BASE}_root{root}_{role}_RUN_MAIN01"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    protocol, row = report["protocol"], report["rows"][0]
    require(report["status"] == "recorded_known_root_not_admitted"
            and protocol["prefix_role"] == role and protocol["root_slot"] == root
            and report["host"]["device"] == "cuda:0"
            and report["model_before"] == report["model_after"], "run identity")
    require(sha(report["prepared"]) == report["prepare_sha256"], "prepared drift")
    require(row["status"] == "completed20" and row["control_returned"] == 20
            and row["initial_exact"], "physical execution incomplete")
    nodes = report["nodes"][f"s{root}_structured"]
    require([n["boundary"] for n in nodes] == [0, 5, 10, 15]
            and all(n["status"] == "sealed" for n in nodes), "node roster")
    checkpoints = {point["step"]: point for point in row["checkpoints"]}
    accounts, configs, observations = [], [], []
    for node in nodes:
        boundary = node["boundary"]
        receipt = node["receipt"]
        seal = json.loads((directory / f"s{root}_structured_b{boundary:02}_seal.json").read_bytes())
        require(seal["receipt"] == receipt, "node receipt mismatch")
        obs = seal["identity"]["observation"]
        frame = checkpoints[boundary]
        require(obs["pixel_sha256"] == frame["pixel_sha256"]
                and obs["png_sha256"] == frame["file_sha256"]
                and sha(frame["path"]) == frame["file_sha256"], "observation PNG mismatch")
        with Image.open(frame["path"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        require(pixels.shape == (224, 224, 3)
                and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"], "pixel mismatch")
        require(obs["boundary"] == boundary, "observation boundary mismatch")
        if boundary:
            require(seal["identity"]["raw10"][-5:] == row["executed_raw"][boundary - 5:boundary],
                    "actual history mismatch")
        ledger = receipt["ledger"]
        account = receipt["account"]
        require(sha(receipt["trace"]) == receipt["trace_sha256"], "trace drift")
        observed = consumer.consume(ledger["journal"], ledger["sha256"], account["contract_sha256"],
                                    "structured", "node", account["run_id"])
        require(observed == account, "ledger account mismatch")
        with Path(ledger["journal"]).open() as stream:
            header = json.loads(next(stream))
        configs.append(header["payload"]["contract"]["arms"]["structured"])
        accounts.append(account["charged_upper_bounds"])
        observations.append(dict(boundary=boundary, pixel_sha256=frame["pixel_sha256"],
                                 png_sha256=frame["file_sha256"], ledger_sha256=ledger["sha256"],
                                 trace_sha256=receipt["trace_sha256"], account_status=account["status"]))
    require(all(config == configs[0] for config in configs), "optimizer changed by boundary")
    return dict(root=root, role=role, report=str(report_path), report_sha256=sha(report_path),
                prepare_sha256=report["prepare_sha256"], binding=report["cp10_goal_binding"],
                model=report["model_before"], initial_state=checkpoints[0]["state"],
                initial_pixel=checkpoints[0]["pixel_sha256"], goal_state=checkpoints[0]["goal"],
                prefix=row["executed_raw"][:5], optimizer=configs[0], charges=accounts,
                observations=observations, counts=report["counts"], native_success=row["native_success"],
                actual_usage_verified=False, coverage_verified=False)


def audit():
    rows = [audit_one(root, role) for root in (0, 1) for role in ROLES]
    require(all(row["optimizer"] == rows[0]["optimizer"] and row["charges"] == rows[0]["charges"]
                and row["model"] == rows[0]["model"] for row in rows), "optimizer/model/charge mismatch")
    pairs = []
    for root in (0, 1):
        group = [row for row in rows if row["root"] == root]
        require(all(row["binding"] == group[0]["binding"] and row["initial_state"] == group[0]["initial_state"]
                    and row["initial_pixel"] == group[0]["initial_pixel"]
                    and row["goal_state"] == group[0]["goal_state"] for row in group), "root/goal mismatch")
        require(all(x == [0.0, 0.0] for x in group[1]["prefix"]), "zero prefix mismatch")
        pairs.append(dict(root=root, selected_equals_greedy=group[0]["prefix"] == group[2]["prefix"],
                          selected_equals_zero=group[0]["prefix"] == group[1]["prefix"],
                          cp5_selected_equals_greedy=group[0]["observations"][1]["pixel_sha256"]
                          == group[2]["observations"][1]["pixel_sha256"]))
    return dict(schema=NAME + ".v1", rows=rows, prefix_collisions=pairs,
                interpretation="known-root feedback MPC, not open-loop own15 nor coherent S/G/Z",
                accounting="charged upper bounds only; full cost and fairness unverified",
                prospective_batch_sealed="unknown", independent_confirmation=False,
                P1_admission=False, neural_calls=0, physical_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + "_") and not path.exists(), "exclusive output")
    result = audit()
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(path), "sha256": sha(path),
                      "prefix_collisions": result["prefix_collisions"]}), flush=True)


if __name__ == "__main__":
    main()
