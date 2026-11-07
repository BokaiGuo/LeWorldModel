"""Replay all sealed actions with native-precision archival physical states."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments import pc_pointmaze_archive_context_multistep_20261107_codex as source
from experiments.pc_input_permissions import STATE, file_hash

PROTOCOL = source.ROOT.parent / "docs/PC_POINTMAZE_ARCHIVE_NATIVE_STATE_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = source.ROOT / "results/pc_pointmaze_archive_native_state_replay_20261107_codex.json"


def main():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(source.PLAN.read_text())
    for key, path in (("protocol_sha256", source.PROTOCOL), ("runner_sha256", source.__file__),
                      ("input_sha256", source.INPUT), ("pred_sha256", source.PRED)):
        if plan[key] != file_hash(path):
            raise ValueError(f"sealed drift: {key}")
    states = torch.load(source.FOLDER / "states.pth", map_location="cpu", weights_only=True)
    rows = []
    with np.load(source.INPUT, allow_pickle=False) as inputs, tempfile.TemporaryDirectory(prefix="pc_native_state_") as temp:
        if str(inputs["states_hash"]) != file_hash(source.FOLDER / "states.pth"):
            raise ValueError("state archive drift")
        folder = Path(temp)
        xml_path = folder / "scene.xml"
        xml_path.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, root in enumerate(source.ROOTS):
            request, response = folder / f"req_{root}.npz", folder / f"res_{root}.npz"
            initial = states[root, 0].numpy()
            model_initial = inputs["states"][index]
            if not np.array_equal(initial.astype(np.float32), model_initial):
                raise ValueError("initial model state not archive cast")
            with request.open("xb") as f:
                np.savez_compressed(f, state=initial, actions=inputs["actions"][index][None])
            subprocess.run([str(LEGACY / "bin/python"), str(Path(source.physics.__file__).resolve()),
                            "--backend", str(request), str(xml_path), str(response)], env=env, check=True, timeout=120)
            with np.load(response, allow_pickle=False) as replay:
                trace = replay["trace"][0].copy()
            if trace.shape != (26, 5) or not np.isfinite(trace).all():
                raise ValueError("bad replay")
            np.testing.assert_allclose(trace[:, 4], np.arange(26) * .05, rtol=0, atol=1e-10)
            errors = np.max(np.abs(trace[:, :4] - states[root, :26].numpy()), axis=1)
            rows.append({"root": root, "max_error": float(errors.max()), "pass_1e4": bool(errors.max() <= 1e-4),
                         "initial_model_vs_physical_max": float(np.max(np.abs(initial - model_initial))),
                         "per_step_max_error": errors.tolist()})
            print(f"root={root} error={errors.max():.8g}", flush=True)
    with OUT.open("x") as f:
        json.dump({"track": STATE, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_plan_sha256": file_hash(source.PLAN), "rows": rows,
                   "pass_count": sum(row["pass_1e4"] for row in rows),
                   "scope": "posthoc precision repair; exact model inputs still float32"}, f, indent=2)
    print(json.dumps({"pass_count": sum(row["pass_1e4"] for row in rows),
                      "max_error": max(row["max_error"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
