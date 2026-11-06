"""Replay saved v17 intervention trajectories in alternate same-environment geometry."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v17_e0_same_chain_mechanism_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT / "interventions_geometry_audit.json"
    if out.exists():
        raise FileExistsError(out)
    inputs = [source.OUT / name for name in
              ("manifest.json", "plans.json", "plans_audit.json", "interventions.json", "interventions_audit.json")]
    manifest, plans, plan_audit, interventions, audit = [json.loads(path.read_text()) for path in inputs]
    seal.verify(manifest["dependencies"])
    if (plan_audit["plans_sha256"] != seal.digest(inputs[1])
            or audit["interventions_sha256"] != seal.digest(inputs[3])
            or interventions["plans_sha256"] != seal.digest(inputs[1])):
        raise ValueError("source binding mismatch")
    deps = seal.capture([Path(__file__), *inputs, Path(source.prior.previous.independent.__code__.co_filename),
                        *(source.OUT / row["arrays"] for row in interventions["rows"] if row["status"] == "completed")])
    names, trajectories = source.prior.intervention.NAMES, 0
    for plan, row in zip(plans["rows"], interventions["rows"], strict=True):
        if plan["root"] != row["root"]:
            raise ValueError("root mismatch")
        if row["status"] != "completed":
            continue
        if seal.digest(source.OUT / row["arrays"]) != row["arrays_sha256"]:
            raise ValueError("trace mismatch")
        data, goal = row["data"], np.asarray(plan["task"]["goal"])
        np.testing.assert_array_equal(np.asarray(data["selected_chain"], np.float32),
                                      np.asarray(plan["result"]["selected"], np.float32))
        with np.load(source.OUT / row["arrays"], allow_pickle=False) as arrays:
            for key in ("selection_actions", "own_actions", "common_actions"):
                value = arrays[key]
                flat = value.reshape(-1, 15, 2).astype(np.float32)
                if not np.isfinite(flat).all() or np.max(np.abs(flat)) > 1:
                    raise ValueError("invalid action")
                path, _ = source.prior.previous.independent(plan["task"]["start"], flat)
                positions = path[:, -1, :2].reshape(*value.shape[:-2], 2)
                scores = np.linalg.norm(positions - goal, axis=-1)
                if key == "own_actions":
                    np.testing.assert_allclose(positions, arrays["own_positions"], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, arrays["own_scores"], atol=1e-12, rtol=0)
                    for i, name in enumerate(names):
                        for repeat in range(3):
                            np.testing.assert_allclose(scores[i, repeat].min(),
                                data["controls"][name]["own"][repeat]["score"], atol=1e-12, rtol=0)
                elif key == "common_actions":
                    for i, name in enumerate(names):
                        np.testing.assert_allclose(scores[i], data["controls"][name]["common_distances"],
                                                   atol=1e-12, rtol=0)
                else:
                    for i, selection in enumerate(data["selection_runs"]):
                        np.testing.assert_allclose(scores[i].min(), selection["score"], atol=1e-12, rtol=0)
                trajectories += len(flat)
    seal.verify(deps)
    source.prior.writer(out, {"status": "passed", "dependencies": deps, "trajectories": trajectories,
        "scope": "alternate same-environment geometry; not independent replication"})
    print(f"verified {trajectories} stored intervention trajectories")


if __name__ == "__main__":
    main()
