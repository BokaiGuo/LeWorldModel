"""Independent process replay and shape/clock checks for native camera data."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_real_camera_effect_dev_20261106_codex as run


def validate(data):
    traces, images, times = data["traces"], data["images"], data["times"]
    if traces.shape != (8, 3, 2, 6, 4) or images.shape != (8, 3, 2, 6, 224, 224, 3):
        raise ValueError("wrong axes")
    if not np.isfinite(traces).all() or not np.isfinite(times).all():
        raise ValueError("nonfinite physical values")
    np.testing.assert_array_equal(traces[:, :, 0], traces[:, :, 1])
    np.testing.assert_array_equal(times[:, :, 0], times[:, :, 1])
    np.testing.assert_array_equal(data["contacts"][:, :, 0], data["contacts"][:, :, 1])
    initial = traces[:, :1, :1, :1]
    np.testing.assert_array_equal(traces[:, :, :, :1], np.broadcast_to(initial, (8, 3, 2, 1, 4)))
    increments = np.diff(times, axis=-1)
    if not np.all(increments > 0):
        raise ValueError("nonincreasing physical time")
    np.testing.assert_allclose(increments, increments.flat[0], atol=1e-12, rtol=0)
    return float(increments.flat[0])


def main():
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME

    output = run.OUT.with_name(run.OUT.name + "_audit.json")
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads(run.OUT.with_suffix(".json").read_text())
    if (meta["code_sha256"] != run.sha(run.__file__) or
            meta["protocol_sha256"] != run.sha(run.PROTOCOL) or
            meta["payload_sha256"] != run.sha(run.OUT.with_suffix(".npz")) or
            meta["source_sha256"] != run.sha(DATA / "states.pth")):
        raise ValueError("sealed input drift")
    ids = [(row["root"], row["frame"]) for row in meta["rows"]]
    archive = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True)
    states = np.stack([archive[root, frame].numpy() for root, frame in ids])
    with np.load(run.OUT.with_suffix(".npz"), allow_pickle=False) as sealed:
        dt = validate(sealed)
        np.testing.assert_allclose(sealed["traces"][:, 0, 0, 0], states, atol=1e-10, rtol=0)
        with tempfile.TemporaryDirectory(prefix="pc_camera_replay_") as temp:
            folder = Path(temp)
            request, xml, response = folder / "request.npz", folder / "scene.xml", folder / "replay.npz"
            np.savez_compressed(request, states=states)
            xml.write_text(official_xml())
            if run.sha(xml) != meta["xml_sha256"]:
                raise ValueError("XML drift")
            env = os.environ.copy()
            env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
            env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                              "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
            subprocess.run([str(LEGACY / "bin/python"), str(Path(run.__file__).resolve()), "--backend",
                            str(request), str(xml), str(response)], env=env, check=True, timeout=180)
            with np.load(response, allow_pickle=False) as replay:
                validate(replay)
                for key in sealed.files:
                    np.testing.assert_array_equal(replay[key], sealed[key])
        # Representative root chosen by protocol order, not by result.
        from PIL import Image
        tiles = sealed["images"][:2, 0, :, 0]
        canvas = np.concatenate([np.concatenate(list(row), axis=1) for row in tiles], axis=0)
        image_path = run.OUT.with_name(run.OUT.name + "_views.png")
        if image_path.exists():
            raise FileExistsError(image_path)
        Image.fromarray(canvas).save(image_path)
    result = {"status": "exact independent-process replay", "trajectories": 48,
              "physical_seconds_per_environment_step": dt,
              "total_seconds_per_chain": 5 * dt, "archive_initial_states_match": True,
              "report_sha256": run.sha(run.OUT.with_suffix(".json")),
              "audit_code_sha256": run.sha(__file__), "views_sha256": run.sha(image_path),
              "scope": "renderer and physical pairing only; not learned invariance or planning"}
    with output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
