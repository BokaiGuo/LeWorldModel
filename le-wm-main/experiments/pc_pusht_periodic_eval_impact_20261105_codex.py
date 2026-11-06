"""Compare upstream and versioned PushT evaluators on bounded saved state banks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper
from experiments import pc_jepa_multiroot_action_effect_dev_20261105_codex as multi
from experiments.pc_pusht_periodic_eval_v1 import PeriodicPushTWrapper

ROOT = multi.ROOT
OUT = ROOT / "results/pc_pusht_periodic_eval_impact_20261105_codex"
SOURCE = ROOT / "results/pc_jepa_v22_contact_feedback_dev_20261104_codex"
FIXED = ROOT / "results/pc_jepa_v22_fixed_bank_posthoc_20261105_codex"
UNQUERIED = ROOT / "results/pc_jepa_v22_unqueried_suffix_dev_20261105_codex"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    multi.verify()
    assert read(multi.OUT / "audit.json")["status"] == "passed"
    fixed_result = read(FIXED / "result.json")
    unq_result = read(UNQUERIED / "physical.json")
    assert fixed_result["status"] == "complete" and unq_result["status"] == "completed"
    assert read(FIXED / "audit.json")["status"] == "passed"
    assert read(UNQUERIED / "audit.json")["status"] == "passed"
    assert sha(FIXED / "arrays.npz") == fixed_result["arrays_sha256"]
    assert sha(UNQUERIED / "physical_arrays.npz") == unq_result["arrays_sha256"]
    goal_path = SOURCE / "goal_state_evaluation_only.npy"
    assert sha(goal_path) == fixed_result["bindings"][str(goal_path)]
    goal = np.load(goal_path, allow_pickle=False)
    paths = [Path(__file__), ROOT / "experiments/pc_pusht_periodic_eval_v1.py",
             ROOT / "external/jepa-wms/evals/simu_env_planning/envs/pusht_gym_wrap.py",
             multi.OUT / "physical.json", multi.OUT / "audit.json",
             FIXED / "arrays.npz", FIXED / "result.json", FIXED / "audit.json",
             UNQUERIED / "physical_arrays.npz", UNQUERIED / "physical.json",
             UNQUERIED / "audit.json", goal_path]
    banks = []
    for number in range(4):
        path = multi.OUT / f"root_{number}/physical.npz"
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            banks.append((f"multiroot_{number}", data["states"].copy(), data["goal_state"].copy()))
    for label, path in (("fixed_bank", FIXED / "arrays.npz"),
                        ("unqueried_suffix", UNQUERIED / "physical_arrays.npz")):
        with np.load(path, allow_pickle=False) as data:
            banks.append((label, data["states"].copy(), goal))
    rows = []
    for label, states, target in banks:
        cases = []
        for index, state in enumerate(states):
            before = PushTWrapper.eval_state(None, target, state)
            after = PeriodicPushTWrapper.eval_state(None, target, state)
            assert before["state_dist"] == after["state_dist"]
            cases.append({"index": index, "raw_angle_difference": float(abs(target[4] - state[4])),
                          "upstream_success": bool(before["success"]), "periodic_success": after["success"]})
        rows.append({"name": label, "count": len(cases),
                     "upstream_success": sum(c["upstream_success"] for c in cases),
                     "periodic_success": sum(c["periodic_success"] for c in cases),
                     "disagreements": sum(c["upstream_success"] != c["periodic_success"] for c in cases),
                     "above_2pi": sum(c["raw_angle_difference"] > 2 * np.pi for c in cases),
                     "cases": cases})
    report = {"status": "completed", "rows": rows,
              "inputs_sha256": {str(path): sha(path) for path in paths},
              "scope": "Six selected saved banks only; no inference about other historical results."}
    OUT.mkdir()
    with (OUT / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "cases"} for row in rows], indent=2))


if __name__ == "__main__":
    main()
