"""Independently replay the dispatched feedback action sequences."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_pointmaze_packet_cem_feedback_20261107_codex as source
from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.pc_input_permissions import file_hash

OUTPUT = source.ROOT / "results/pc_pointmaze_packet_cem_feedback_replay_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(source.OUTPUT.read_text())
    if data["source_plan"] != file_hash(source.source.PLAN) or data["source_physical"] != file_hash(source.source.PHYSICAL):
        raise ValueError("source mismatch")
    rows = []
    with np.load(source.source.prior.PRIVATE, allow_pickle=False) as witness, tempfile.TemporaryDirectory(prefix="pc_feedback_replay_") as tmp:
        folder = Path(tmp)
        xml = folder / "scene.xml"
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for i, row in enumerate(data["rows"]):
            if row["root"] != 1884 + i:
                raise ValueError("root mismatch")
            actions = np.asarray(row["executed_actions"], dtype=np.float32)
            if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                raise ValueError("illegal action sequence")
            req, res = folder / f"req_{i}.npz", folder / f"res_{i}.npz"
            with req.open("xb") as stream:
                np.savez_compressed(stream, state=witness["initial"][i], actions=actions[None])
            subprocess.run([str(LEGACY / "bin/python"),
                            str(Path(source.source.prior.retention.archive.physics.__file__).resolve()),
                            "--backend", str(req), str(xml), str(res)], env=env, check=True, timeout=120)
            with np.load(res, allow_pickle=False) as values:
                trace = values["trace"].copy()
            if trace.shape != (1, 26, 5):
                raise ValueError("invalid replay")
            np.testing.assert_allclose(trace[0, :, 4], np.arange(26)*.05, atol=1e-10, rtol=0)
            error = float(np.max(np.abs(trace[0, -1, :4] - row["final_proprio"])))
            if error > 1e-6:
                raise ValueError(f"replay mismatch {row['root']}: {error}")
            rows.append({"root": row["root"], "final_state_max_abs_error": error})
    with OUTPUT.open("x") as stream:
        json.dump({"feedback": file_hash(source.OUTPUT), "runner": file_hash(__file__), "rows": rows}, stream, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
