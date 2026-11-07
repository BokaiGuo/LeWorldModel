"""2x2 archival precision replay for the prespecified failing root."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments import pc_pointmaze_archive_context_multistep_20261107_codex as source
from experiments.pc_input_permissions import STATE, file_hash

PROTOCOL = source.ROOT.parent / "docs/PC_POINTMAZE_ARCHIVE_PRECISION_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = source.ROOT / "results/pc_pointmaze_archive_precision_replay_20261107_codex.json"
AUDIT = source.ROOT / "results/pc_pointmaze_archive_context_physics_failure_audit_20261107_codex.json"


def main():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads(AUDIT.read_text())
    bad = [x for x in previous["rows"] if not x["physical_pass_1e4"]]
    if len(bad) != 1 or bad[0]["root"] != 1891:
        raise ValueError("failing root does not match protocol")
    states = torch.load(source.FOLDER / "states.pth", map_location="cpu", weights_only=True)[1891, :26].numpy()
    actions = torch.load(source.FOLDER / "actions.pth", map_location="cpu", weights_only=True)[1891, :25].numpy()
    if states.shape != (26, 4) or actions.shape != (25, 2):
        raise ValueError("bad archive shape")
    rows = []
    with tempfile.TemporaryDirectory(prefix="pc_precision_replay_") as temporary:
        folder = Path(temporary)
        xml_path = folder / "scene.xml"
        xml_path.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for name, initial in (("original", states[0]), ("float32", states[0].astype(np.float32))):
            request, response = folder / f"req_{name}.npz", folder / f"res_{name}.npz"
            action_pair = np.stack([actions, actions.astype(np.float32).astype(actions.dtype)])
            with request.open("xb") as f:
                np.savez_compressed(f, state=initial, actions=action_pair)
            subprocess.run([str(LEGACY / "bin/python"), str(Path(source.physics.__file__).resolve()),
                            "--backend", str(request), str(xml_path), str(response)], env=env, check=True, timeout=120)
            with np.load(response, allow_pickle=False) as data:
                traces = data["trace"].copy()
            if traces.shape != (2, 26, 5):
                raise ValueError("bad replay shape")
            np.testing.assert_allclose(traces[:, :, 4], np.broadcast_to(np.arange(26) * .05, (2, 26)), atol=1e-10, rtol=0)
            for index, action_name in enumerate(("original", "float32")):
                errors = np.max(np.abs(traces[index, :, :4] - states), axis=1)
                rows.append({"state": name, "action": action_name, "max_error": float(errors.max()),
                             "final_error": float(errors[-1]), "pass_1e4": bool(errors.max() <= 1e-4),
                             "per_step_max_error": errors.tolist()})
    with OUT.open("x") as f:
        json.dump({"track": STATE, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_audit_sha256": file_hash(AUDIT), "source_states_sha256": file_hash(source.FOLDER / "states.pth"),
                   "source_actions_sha256": file_hash(source.FOLDER / "actions.pth"),
                   "source_state_dtype": str(states.dtype), "source_action_dtype": str(actions.dtype),
                   "rows": rows, "scope": "single exposed root precision diagnostic"}, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
