"""Assessment-only geometry witnesses; never enters planner."""
import json
from pathlib import Path

import numpy as np
from experiments import pc_paired_new_task_costed_dev_20261105_codex as source

ROOT = source.ROOT
OUT = ROOT / "results/pc_e0_new_task_privileged_feasibility_diag_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_E0_NEW_TASK_PRIVILEGED_FEASIBILITY_DIAG_PROTOCOL_20261105_CODEX.md"


def witness(task, side):
    position = np.asarray(task["start"][:2], np.float64)
    heading = float(task["start"][2])
    steps = []
    for target in (np.array([0., side * 1.1]), np.asarray(task["goal"], np.float64)):
        vec = target - position
        distance = float(np.linalg.norm(vec))
        if distance > 1e-12:
            angle = float(np.arctan2(vec[1], vec[0]))
            turn = (angle-heading+np.pi) % (2*np.pi)-np.pi
            count = int(np.ceil(abs(turn) / (np.pi/2)))
            if count:
                steps.extend([[0., turn / count / (np.pi/2)]] * count)
            count = int(np.ceil(distance / .3))
            steps.extend([[distance / count / .3, 0.]] * count)
            heading = angle
        position = target
    required = len(steps)
    if required > 15:
        return required, None
    steps.extend([[0., 0.]] * (15-required))
    actions = np.asarray(steps, np.float32)
    if np.any(np.abs(actions) > 1):
        raise ValueError("action out of bounds")
    return required, actions


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = json.loads((source.OUT / "manifest.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    digest = source.prior.old.prior.digest
    if audit["status"] != "passed" or audit["summary_sha256"] != digest(source.OUT / "summary.json"):
        raise ValueError("source audit mismatch")
    base = source.fixed.prior.prior.prior
    seal = base.seal
    seal.verify(old["dependencies"])
    seal.verify(old["source_dependencies"])
    deps = seal.capture([Path(__file__), PROTOCOL, source.OUT / "manifest.json", source.OUT / "audit.json",
                         Path(base.exact.wall.__file__), Path(base.exact.wall.physics.__file__)])
    OUT.mkdir()
    rows = []
    for root, task in enumerate(old["tasks"]):
        attempts = []
        for side in (1, -1):
            n, chain = witness(task, side)
            row = {"side": side, "required_steps": n, "within_horizon": chain is not None}
            if chain is not None:
                padded = np.pad(chain[None], ((0, 0), (0, 1), (0, 0))) * base.exact.wall.HIGH
                trajectory, blocked = base.exact.wall.rollout(task["start"], padded)
                score = float(np.linalg.norm(trajectory[0, 15, :2]-task["goal"]))
                row.update(actions=chain.tolist(), trajectory=trajectory[0, :16].tolist(),
                           blocked=blocked[0, :15].tolist(), score=score, success=bool(score < .05))
            else:
                row.update(success=False)
            attempts.append(row)
        rows.append({"root": root, "task": task, "attempts": attempts})
        print(root, [r["success"] for r in attempts], flush=True)
    seal.verify(deps)
    seal.verify(old["source_dependencies"])
    report = {"status": "completed", "dependencies": deps, "source_dependencies": old["source_dependencies"],
              "rows": rows, "scope": "privileged geometry, same physics, not an oracle or planner"}
    with (OUT / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    print("Checking privileged fixed-waypoint feasibility; do not feed witnesses into the planner.", flush=True)
    main()
