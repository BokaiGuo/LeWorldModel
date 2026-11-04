"""Independently replay stored v13 intervention trajectories; no new search."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT / "interventions_geometry_audit.json"
    if out.exists():
        raise FileExistsError(out)
    inputs = [source.OUT / "manifest.json", source.OUT / "plans.json",
              source.OUT / "plans_audit.json", source.OUT / "interventions.json",
              source.OUT / "interventions_audit.json"]
    manifest, plans, plans_audit, interventions, audit = [json.loads(p.read_text()) for p in inputs]
    seal.verify(manifest["dependencies"])
    if (plans_audit["plans_sha256"] != seal.digest(inputs[1])
            or audit["interventions_sha256"] != seal.digest(inputs[3])
            or interventions["plans_sha256"] != seal.digest(inputs[1])):
        raise ValueError("invalid source binding")
    deps = seal.capture([Path(__file__), *inputs, Path(source.previous.independent.__code__.co_filename),
                         *(source.OUT / row["arrays"] for row in interventions["rows"]
                           if row["status"] == "completed")])
    trajectories = 0
    names = source.intervention.NAMES
    for plan, row in zip(plans["rows"], interventions["rows"], strict=True):
        if plan["root"] != row["root"]:
            raise ValueError("root mismatch")
        if row["status"] != "completed":
            continue
        if seal.digest(source.OUT / row["arrays"]) != row["arrays_sha256"]:
            raise ValueError("trace hash mismatch")
        data, goal, task = row["data"], np.asarray(plan["task"]["goal"]), plan["task"]
        np.testing.assert_array_equal(np.asarray(data["selected_chain"], np.float32),
                                      np.asarray(plan["result"]["selected"], np.float32))
        with np.load(source.OUT / row["arrays"], allow_pickle=False) as arrays:
            for key in ("selection_actions", "own_actions", "common_actions"):
                value = arrays[key]
                flat = value.reshape(-1, 15, 2).astype(np.float32)
                if np.max(np.abs(flat)) > 1 or not np.isfinite(flat).all():
                    raise ValueError("invalid action")
                path, _ = source.previous.independent(task["start"], flat)
                positions = path[:, -1, :2].reshape(*value.shape[:-2], 2)
                scores = np.linalg.norm(positions - goal, axis=-1)
                if key == "own_actions":
                    np.testing.assert_allclose(positions, arrays["own_positions"], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, arrays["own_scores"], atol=1e-12, rtol=0)
                    for i, name in enumerate(names):
                        for repeat in range(3):
                            np.testing.assert_allclose(scores[i, repeat].min(),
                                                       data["controls"][name]["own"][repeat]["score"],
                                                       atol=1e-12, rtol=0)
                elif key == "common_actions":
                    for i, name in enumerate(names):
                        np.testing.assert_allclose(scores[i], data["controls"][name]["common_distances"],
                                                   atol=1e-12, rtol=0)
                else:
                    for i, selection in enumerate(data["selection_runs"]):
                        np.testing.assert_allclose(scores[i].min(), selection["score"], atol=1e-12, rtol=0)
                trajectories += len(flat)
    seal.verify(deps)
    source.writer(out, {"status": "passed", "dependencies": deps, "trajectories": trajectories,
                        "scope": "alternate same-environment geometry; not independent replication"})
    print(f"independently checked {trajectories} stored intervention trajectories")


if __name__ == "__main__":
    main()
