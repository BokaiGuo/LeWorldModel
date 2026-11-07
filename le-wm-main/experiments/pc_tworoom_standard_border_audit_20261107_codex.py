"""Audit the saved seed-42 reference panel, without changing the evaluator."""
import hashlib
import inspect
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_standard_tworoom_reference_replay_20261105_codex import select_rows
from experiments.pc_tworoom_reset_boundary_contract_20261107_codex import border_contract

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5")
REFERENCE = ROOT / "results/pc_standard_tworoom_reference_replay_20261105_codex/report.json"
OUTPUT = ROOT / "results/pc_tworoom_standard_border_audit_20261107_codex.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference = json.loads(REFERENCE.read_text())
    records = reference["records"]
    rows = []
    selected_hash = hashlib.sha256()
    env = gym.make("swm/TwoRoom-v1")
    try:
        base = env.unwrapped
        env_path = Path(inspect.getfile(type(base)))
        with h5py.File(DATA, "r") as data:
            selected = select_rows(data)
            if selected.tolist() != [entry["row"] for entry in records]:
                raise ValueError("selected rows differ from sealed reference report")
            for index, ref in zip(selected, records, strict=True):
                index = int(index)
                if not ref["completed"]:
                    raise ValueError(f"incomplete reference row {index}")
                positions = np.asarray(data["proprio"][index:index + 26])
                selected_hash.update(np.asarray([index], dtype="<i8").tobytes())
                selected_hash.update(np.asarray(positions, dtype="<f4").tobytes())
                if not np.allclose(positions[0], ref["initial_position"], atol=1e-5, rtol=0):
                    raise ValueError(f"reference initial state mismatch at {index}")
                if not np.allclose(positions[-1], ref["goal_position"], atol=1e-5, rtol=0):
                    raise ValueError(f"reference goal mismatch at {index}")
                env.reset(seed=42)
                base._set_state(positions[0])
                base._set_goal_state(positions[-1])
                radius = float(base.variation_space["agent"]["radius"].value.item())
                initial = border_contract(positions[0], base.BORDER_SIZE, radius, base.IMG_SIZE)
                goal = border_contract(positions[-1], base.BORDER_SIZE, radius, base.IMG_SIZE)
                before = base.agent_position.numpy().copy()
                base.step(np.zeros(2, dtype=np.float32))
                after = base.agent_position.numpy().copy()
                rows.append({"row": index, "episode": ref["episode"], "start_step": ref["start_step"],
                             "initial_border": initial, "goal_border": goal,
                             "after_zero": after.tolist(), "zero_displacement": float(np.linalg.norm(after - before)),
                             "reference_ever_success": ref["ever_success"], "radius": radius})
    finally:
        env.close()
    stat = DATA.stat()
    result = {"dataset": str(DATA), "dataset_size": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns,
              "selected_rows_positions_sha256": selected_hash.hexdigest(), "reference_sha256": digest(REFERENCE),
              "environment_path": str(env_path), "environment_sha256": digest(env_path),
              "script_sha256": digest(Path(__file__)), "rows": rows,
              "scope": "seed-42 reference rows, default reset geometry, border-only and zero-action diagnostics; not full official evaluator equivalence"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(rows),
                      "invalid_initial": sum(not row["initial_border"]["valid"] for row in rows),
                      "invalid_goal": sum(not row["goal_border"]["valid"] for row in rows),
                      "nonzero_noop": sum(row["zero_displacement"] > 1e-5 for row in rows),
                      "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    run()
