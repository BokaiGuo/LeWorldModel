"""Replay the sealed third-decision PointMaze response probes."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v7_pointmaze_rgb_live_dev_20261104_codex as configured
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = configured.ROOT
PREFIX = ROOT / "results/pc_v7_pointmaze_actual_fd_physical_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V7_POINTMAZE_ACTUAL_FD_PHYSICAL_DEV_PROTOCOL_20261104_CODEX.md"
MANIFEST, ARRAYS, RESULT = [PREFIX.with_name(PREFIX.name + suffix)
                            for suffix in ("_manifest.json", "_arrays.npz", "_result.json")]


def compare(model, observed):
    error = float(np.linalg.norm(model - observed))
    a, b = float(np.linalg.norm(model)), float(np.linalg.norm(observed))
    return {"absolute_frobenius_error": error, "model_norm": a, "observed_norm": b,
            "relative_to_observed": error / b if b > 1e-12 else None,
            "cosine": float(np.sum(model * observed) / (a * b)) if min(a, b) > 1e-12 else None}


def ordering(left, right):
    i, j = np.triu_indices(len(left), 1)
    a, b = left[i] - left[j], right[i] - right[j]
    valid = (a != 0) & (b != 0)
    return {"non_tie_pairs": int(valid.sum()),
            "agreement": float(np.mean(np.sign(a[valid]) == np.sign(b[valid]))) if valid.any() else None}


def main():
    if any(p.exists() for p in (MANIFEST, ARRAYS, RESULT)):
        raise FileExistsError("prior or partial output")
    if os.environ.get("DISPLAY") != ":0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("expected DISPLAY=:0 and physical CUDA device 0")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if uuid.lower().removeprefix("gpu-") != configured.GPU_UUID.lower().removeprefix("gpu-"):
        raise ValueError("unexpected GPU UUID")
    configured.configure()
    run = configured.live
    sha = run.old.prior.sha
    decisions, images, execution_path = run.paths("response")
    audit = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    replay = configured.PREFIX.with_name(configured.PREFIX.name + "_rgb_frame_replay.json")
    physical = configured.PREFIX.with_name(configured.PREFIX.name + "_physical.json")
    manifest = json.loads(run.MANIFEST.read_text())
    receipt = json.loads(execution_path.read_text())
    if (manifest["dependencies"] != run.dependencies()
            or receipt["manifest_sha256"] != sha(run.MANIFEST)
            or receipt["images_sha256"] != sha(images)
            or json.loads(audit.read_text())["execution_sha256"] != sha(execution_path)
            or json.loads(replay.read_text())["exact_frames"] != 31
            or json.loads(physical.read_text())["audit_sha256"] != sha(audit)):
        raise ValueError("source audit or dependencies do not match")
    decision = json.loads(decisions[2].read_text())
    if receipt["rows"][2]["decision_sha256"] != sha(decisions[2]):
        raise ValueError("decision not bound to dispatch")
    batches = []
    for info in decision["result"]["first_response"]:
        nominal = np.asarray(info["nominal"], dtype=float)
        axes = np.eye(10).reshape(10, 5, 2)
        expected = np.concatenate((nominal[None], nominal[None] + .05 * axes,
                                   nominal[None] - .05 * axes))
        matches = [entry for entry in decision["trace"]
                   if np.array_equal(np.asarray(entry["requested"]), expected)]
        if len(matches) != 1:
            raise ValueError("probe batch missing or ambiguous")
        batch = matches[0]
        positions = np.asarray(batch["positions"])
        np.testing.assert_allclose(((positions[1:11] - positions[11:21]) / .1).T,
                                   info["matrix"], atol=1e-12, rtol=0)
        batches.append(batch)
    if len(batches) != 2:
        raise ValueError("unexpected number of nominal probes")
    probes = np.asarray([b["raw"] for b in batches], dtype=np.float32)
    prefix = np.asarray(run.old.WARMUP.tolist() + receipt["executed_actions"][:10], dtype=np.float32)
    if probes.shape != (2, 21, 5, 2) or prefix.shape != (25, 2):
        raise ValueError("unexpected action shape")
    deps = seal.capture([Path(__file__), PROTOCOL, run.MANIFEST, *decisions, images,
                         execution_path, audit, replay, physical, backend.WORKER,
                         *map(Path, manifest["dependencies"])])
    run.old.write_json(MANIFEST, {"seal": deps, "gpu_uuid": uuid,
                                "scope": "42 sealed probes, one exposed boundary"})
    with np.load(images, allow_pickle=False) as data:
        boundary = data["rgb"][25].copy()
    camera = json.loads(run.h.CAMERA.read_text())["metadata"]
    offset = json.loads(run.h.OLD.read_text())["body_offset_from_static_scene"]
    frames = []
    print("Replaying 42 sealed probe packets; no replanning or physical feedback.", flush=True)
    with run.h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        with run.bridge.runner.ShadowRGBBridge(np.asarray([*run.old.STARTS[0], 0., 0.])) as env:
            for index, packet in enumerate(probes.reshape(42, 5, 2)):
                env.reset()
                for action in prefix:
                    image = env.step(action[None])
                if not np.array_equal(image, boundary):
                    raise ValueError(f"shadow boundary mismatch at probe {index}")
                frames.append(np.stack([env.step(action[None]) for action in packet]))
                if (index + 1) % 10 == 0:
                    print(f"shadow branches {index + 1}/42", flush=True)
    frames = np.asarray(frames).reshape(2, 21, 5, 224, 224, 3)
    estimated = np.asarray([run.h.causal_state(branch[-2], branch[-1], camera, offset, .05)
                            for branch in frames.reshape(42, 5, 224, 224, 3)]).reshape(2, 21, 4)
    with tempfile.TemporaryDirectory(prefix="v7_pointmaze_fd_native_") as directory:
        temp = Path(directory)
        request, xml, output = temp / "request.json", temp / "scene.xml", temp / "out.json"
        request.write_text(json.dumps({"rows": [
            {"start": list(run.old.STARTS[0]), "policy": str(i),
             "actions": np.concatenate((prefix, packet)).tolist()}
            for i, packet in enumerate(probes.reshape(42, 5, 2))]}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(request), str(xml), str(output)], env=env, check=True, timeout=180)
        native = json.loads(output.read_text())
    if native["binding"] != "2.1.2.14" or len(native["rows"]) != 42:
        raise ValueError("native replay mismatch")
    old_trace = json.loads(physical.read_text())["trace"]
    for i, row in enumerate(native["rows"]):
        if row["policy"] != str(i) or len(row["trace"]) != 31:
            raise ValueError("native action clock mismatch")
        np.testing.assert_allclose([t["state"] for t in row["trace"][:26]],
                                   [t["state"] for t in old_trace[:26]], atol=1e-12, rtol=0)
    states = np.asarray([[t["state"] for t in row["trace"]] for row in native["rows"]]).reshape(2, 21, 31, 4)
    seal.verify(deps)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, actions=probes, prefix=prefix, frames=frames,
                            estimated=estimated, native_states=states)
        stream.flush()
        os.fsync(stream.fileno())
    rows = []
    for i, batch in enumerate(batches):
        model_pos = np.asarray(batch["positions"])
        model_matrix = ((model_pos[1:11] - model_pos[11:21]) / .1).T
        rgb_matrix = ((estimated[i, 1:11, :2] - estimated[i, 11:21, :2]) / .1).T
        native_matrix = ((states[i, 1:11, -1, :2] - states[i, 11:21, -1, :2]) / .1).T
        scores = np.asarray(batch["scores"])
        native_distance = np.linalg.norm(states[i, :, -1, :2] - [1.8, 2.8], axis=1)
        rgb_distance = np.linalg.norm(estimated[i, :, :2] - [1.8, 2.8], axis=1)
        rows.append({"nominal": i, "model_matrix": model_matrix.tolist(),
                     "rgb_matrix": rgb_matrix.tolist(), "native_matrix": native_matrix.tolist(),
                     "model_vs_rgb": compare(model_matrix, rgb_matrix),
                     "model_vs_native": compare(model_matrix, native_matrix),
                     "rgb_vs_native": compare(rgb_matrix, native_matrix),
                     "model_scores": scores.tolist(), "native_distances": native_distance.tolist(),
                     "rgb_distances": rgb_distance.tolist(),
                     "model_vs_native_ordering": ordering(scores, native_distance),
                     "model_vs_rgb_ordering": ordering(scores, rgb_distance)})
    run.old.write_json(RESULT, {"manifest_sha256": sha(MANIFEST), "arrays_sha256": sha(ARRAYS),
                                "native_backend": native["binding"], "native_rows": native["rows"],
                                "scope": "one exposed root; cross-backend native physical replay", "rows": rows})
    print("Saved shadow RGB, causal estimates, native traces and response comparisons.", flush=True)


if __name__ == "__main__":
    main()
