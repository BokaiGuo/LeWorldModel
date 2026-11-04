"""Audit static RGB, distances and local target-geometry diagnostics."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_reacher_goal_geometry_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = run.live.host.bridge.sha_file
    manifest, data = json.loads(run.MANIFEST.read_text()), json.loads(run.RESULT.read_text())
    prev = json.loads(run.prev.RESULT.read_text())
    seal.verify(manifest["seal"])
    if data["manifest_sha256"] != sha(run.MANIFEST) or data["arrays_sha256"] != sha(run.ARRAYS):
        raise ValueError("manifest or arrays mismatch")
    with np.load(run.ARRAYS, allow_pickle=False) as bundle:
        arrays = {k: bundle[k].copy() for k in bundle.files}
    if len(data["rows"]) != 16 or arrays["frames"].shape[0] != 16:
        raise ValueError("incomplete geometry")
    target = np.asarray(data["target_qpos"])
    if data["target_image_equal"] != bool(np.array_equal(arrays["frames"][0], arrays["goal_rgb"])):
        raise ValueError("target image equality mismatch")
    for i, row in enumerate(data["rows"]):
        if row["frame_sha256"] != run.live.context_lib.rgb_digest(arrays["frames"][i]):
            raise ValueError("frame digest mismatch")
        distance = float(run.live.p0._qpos_distance(np.asarray(row["qpos"]), target))
        latent = float(np.mean((arrays["latent"][i] - arrays["goal_latent"]) ** 2))
        pixel = float(np.mean((arrays["frames"][i].astype(np.float64) - arrays["goal_rgb"]) ** 2))
        np.testing.assert_allclose([distance, latent, pixel],
                                   [row["distance"], row["latent_goal_mse"], row["pixel_goal_mse"]],
                                   rtol=0, atol=1e-8)
        if row["branch"] is not None:
            j = next(j for j, branch in enumerate(prev["rows"]) if branch["name"] == row["branch"])
            endpoint = np.asarray(prev["rows"][j]["trajectory"][-1]["qpos"])
            delta = (endpoint - target + np.pi) % (2*np.pi) - np.pi
            np.testing.assert_allclose(row["qpos"], target + row["alpha"] * delta, rtol=0, atol=1e-10)
            np.testing.assert_allclose(distance, row["alpha"] * np.linalg.norm(delta[:2]), rtol=0, atol=1e-10)
            if row["alpha"] == 1:
                if row["matches_saved_frame"] != bool(np.array_equal(arrays["frames"][i], arrays["prior_frames"][j])):
                    raise ValueError("prior frame mismatch")
                np.testing.assert_allclose(row["prior_latent_max_abs"],
                    np.max(np.abs(arrays["latent"][i] - arrays["prior_latent"][j])), rtol=0, atol=1e-10)
    inversions, monotonic = {}, {}
    for field in ("latent_goal_mse", "pixel_goal_mse"):
        count = eligible = 0
        for i in range(16):
            for j in range(i+1, 16):
                d = data["rows"][i]["distance"] - data["rows"][j]["distance"]
                m = data["rows"][i][field] - data["rows"][j][field]
                if abs(d) <= 1e-10 or abs(m) <= 1e-10:
                    continue
                eligible += 1
                count += int(d*m < 0)
        inversions[field] = {"inverted": count, "non_tied_pairs": eligible}
        monotonic[field] = {}
        for branch in prev["rows"]:
            values = [data["rows"][0][field]] + [r[field] for r in data["rows"] if r["branch"] == branch["name"]]
            monotonic[field][branch["name"]] = {"values": values, "nondecreasing": bool(np.all(np.diff(values) >= -1e-10))}
    seal.verify(manifest["seal"])
    run.live.host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
                                 "arrays_sha256": sha(run.ARRAYS), "audit_script_sha256": sha(Path(__file__)),
                                 "inversions": inversions, "monotonic": monotonic,
                                 "scope": "numerical audit of saved RGB and latent arrays; no encoder rerun"})
    print(json.dumps({"inversions": inversions, "monotonic": monotonic}, indent=2))


if __name__ == "__main__":
    main()
