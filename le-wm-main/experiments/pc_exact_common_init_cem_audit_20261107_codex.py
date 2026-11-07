"""Independently reconstruct common initial candidates and subsequent CEM draws."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_common_init_cem_20261107_codex as run
from experiments.pc_exact_primitive_cem_control_audit_20261107_codex import independent_decode


def main():
    if run.AUDIT.exists():
        raise FileExistsError(run.AUDIT)
    data = json.loads(run.OUTPUT.read_text())
    if data["dependencies"] != {p: run.prior.sha(p) for p in data["dependencies"]}:
        raise ValueError("dependency drift")
    assert len(data["rows"]) == 24
    results = []
    for row in data["rows"]:
        root, task = row["root"], row["task"]
        offsets = np.random.default_rng(2718000 + root).uniform(-1.9, 1.9, 384)
        np.testing.assert_array_equal(offsets, row["offsets"])
        geometry, gscore = np.asarray(row["geometry_actions"]), np.asarray(row["geometry_costs"])
        parameters = np.asarray(row["initial_parameters"])
        refined, rcost = np.asarray(row["refined_actions"]), np.asarray(row["refined_costs"])
        rparams = np.asarray(row["refined_parameters"])
        assert geometry.shape == (384, 15, 2) and refined.shape == (256, 15, 2)
        assert parameters.shape == (128, 4) and rparams.shape == (256, 4)
        assert np.max(np.abs(geometry)) <= 1 + 1e-12 and np.max(np.abs(refined)) <= 1 + 1e-12
        for i, offset in enumerate(offsets):
            chain, point, _ = run.bridge.make_chain(task["start"], task["goal"], offset)
            np.testing.assert_allclose(geometry[i], chain, rtol=0, atol=1e-12)
            if i < 128:
                reconstructed, _ = independent_decode(task["start"], task["goal"], parameters[i])
                np.testing.assert_allclose(chain, reconstructed, rtol=0, atol=1e-12)
        elite = parameters[np.argsort(gscore[:128], kind="stable")[:16]]
        mean, scale = elite.mean(0), np.maximum(elite.std(0), .08)
        rng = np.random.default_rng(2719000 + root)
        for j in range(4):
            sl = slice(j * 64, (j + 1) * 64)
            expected = np.clip(mean + rng.normal(size=(64, 4)) * scale, -1, 1)
            np.testing.assert_allclose(rparams[sl], expected, rtol=0, atol=0)
            for param, action in zip(expected, refined[sl], strict=True):
                reconstructed, _ = independent_decode(task["start"], task["goal"], param)
                np.testing.assert_allclose(action, reconstructed, rtol=0, atol=1e-12)
            elite = expected[np.argsort(rcost[sl], kind="stable")[:16]]
            mean, scale = elite.mean(0), np.maximum(elite.std(0), .08)
        np.testing.assert_allclose(gscore, run.score(task, geometry), rtol=0, atol=1e-12)
        np.testing.assert_allclose(rcost, run.score(task, refined), rtol=0, atol=1e-12)
        joint = np.r_[gscore[:128], rcost]
        assert row["geometry_best"] == int(np.argmin(gscore))
        assert row["joint_best"] == int(np.argmin(joint))
        assert row["counts"] == {"geometry": 384, "joint": 384}
        assert joint.min() <= min(gscore[:128]) + 1e-12
        results.append({"root": root, "initial": float(min(gscore[:128])),
                        "geometry": float(min(gscore)), "joint": float(min(joint)),
                        "joint_from_refinement": row["joint_best"] >= 128})
    with run.AUDIT.open("x") as f:
        json.dump({"source_sha256": run.prior.sha(run.OUTPUT),
                   "audit_sha256": run.prior.sha(Path(__file__)), "rows": results}, f, indent=2)
    print(f"audited {len(results)} roots, 9216 geometry and 6144 refinement chains")


if __name__ == "__main__":
    main()
