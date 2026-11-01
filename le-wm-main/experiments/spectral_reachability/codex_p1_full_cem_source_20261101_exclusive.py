"""Read-only mapping from saved first-feedback prepares to CEM input identity.

This verifies the pinned prepare bytes and extracts its task, not the thousands
of transitive sources or the runtime model/physical restoration.
"""
import hashlib
import json
from pathlib import Path

import torch

from .codex_p1_full_cem_controller_20261101_exclusive import tensor_sha
from .codex_four_arm_full_cost_20261030 import digest

ROOT = Path(__file__).resolve().parents[2]
PINNED = {
    0: "2de28d0ff18bf722fa9aef9a59b1902ec30c0cf46f7a450a5b66fa2582313d0a",
    1: "dfc577bf8d2d2e7e6f5f9b55c85c1e12082199a69c79fcb73377b427fbaa79bb",
}
REPORTS = {
    0: "7aaa901a28974c09bd0cb12fb4b7a7efe939a533dbe2649327d9c33802fb5d9d",
    1: "4d4f4f074a1b3cdc0b9999c09c71414d78cd56825c067c139e1446f73e19aab2",
}
TARGET_SHA = "a0a524ef256dc8737995ce2f0daac5c15ad7578e2234be42c19c19cfbc864aef"


def pinned_json(path, expected):
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("saved source digest mismatch")
    return json.loads(body)


def inspect(root):
    if type(root) is not int or root not in PINNED:
        raise ValueError("root must be 0 or 1")
    path = ROOT / "results" / f"p1_first_feedback_carry_20261101_main_root{root}_PREP_MAIN02.json"
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != PINNED[root]:
        raise ValueError("saved prepare digest mismatch")
    value = json.loads(body)
    expected = {"0": [3, False], "5": [2, True], "10": [1, True], "15": [1, False]}
    protocol, source = value["protocol"], value["source"]
    if (protocol["root_slot"] != root or protocol["boundaries"] != [0, 5, 10, 15]
            or protocol["feedback_horizons"] != expected or len(source["tasks"]) != 2):
        raise ValueError("source schedule or roster mismatch")
    task = source["tasks"][root]
    binding = task["binding"]
    if binding["model_sha256"] != source["expected_model_hash"]:
        raise ValueError("source model binding mismatch")
    target_path = ROOT / "results/e3_cp10_target_cuda_20261031_RUN01.json"
    target = pinned_json(target_path, TARGET_SHA)
    report_path = ROOT / "results" / f"p1_first_feedback_carry_20261101_main_root{root}_RUN_MAIN02/report.json"
    report = pinned_json(report_path, REPORTS[root])
    row = target["rows"][root]
    rebound = report["cp10_goal_binding"]
    if (target["status"] != "complete" or row["root"] != root
        or report["prepare_sha256"] != PINNED[root]
        or rebound["goal_pixel_sha256"] != row["pixel_sha256"]
        or rebound["goal_latent_sha256"] != digest(row["latent"])):
        raise ValueError("actual cp10 goal binding mismatch")
    goal = torch.tensor(row["latent"], dtype=torch.float32).unsqueeze(0)
    normalizer = torch.tensor([task["action_mean"], task["action_std"]], dtype=torch.float32)
    matrix = torch.tensor(task["projection"], dtype=torch.float64)
    name = f"e3_cp10_four_arm_root{root}_20261031"
    task_id = digest([name, binding["task_id"], row["pixel_sha256"]])
    identity = {"model_sha256": binding["model_sha256"],
                "root_sha256": rebound["root_sha256"],
                "task_sha256": digest([name, "task", task_id]),
                "goal_sha256": tensor_sha(goal),
                "normalizer_sha256": tensor_sha(normalizer),
                "metric_sha256": tensor_sha(matrix)}
    return {"root": root, "prepare": str(path), "prepare_sha256": PINNED[root],
            "identity": identity, "original_binding": binding,
            "rebound_goal_binding": rebound, "target_sha256": TARGET_SHA,
            "report_sha256": REPORTS[root],
            "transitive_sources_verified": False, "runtime_identity_verified": False,
            "physical_root_restoration_verified": False, "execution_authorized": False}


if __name__ == "__main__":
    print(json.dumps([inspect(0), inspect(1)], sort_keys=True))
