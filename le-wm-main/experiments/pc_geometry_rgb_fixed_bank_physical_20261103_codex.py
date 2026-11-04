"""Post-seal physical replay of the entire model-scored bank."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_geometry_rgb_fixed_bank_select_20261103_codex as decision
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base

PROTOCOL = base.PROJECT / "docs/PC_GEOMETRY_RGB_FIXED_BANK_PHYSICAL_PROTOCOL_20261103_CODEX.md"
OUTPUT = base.ROOT / "results/pc_geometry_rgb_fixed_bank_physical_20261103_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    selected = json.loads(decision.OUTPUT.read_text())
    if (selected["arrays_sha256"] != base.digest(decision.ARRAYS)
            or selected["runner_sha256"] != base.digest(Path(decision.__file__))):
        raise ValueError("sealed decision changed")
    requests = []
    for row in selected["rows"]:
        for candidate in row["candidates"]:
            actions = np.concatenate((np.zeros((1, 2)), candidate["prefix_raw"], candidate["suffix_raw"]))
            if actions.shape != (16, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.000001:
                raise ValueError("invalid saved action chain")
            requests.append({"start": list(decision.STARTS[row["root_index"]]),
                             "policy": candidate["name"], "actions": np.clip(actions, -1, 1).tolist()})
    with tempfile.TemporaryDirectory(prefix="pc_model_physical_") as temporary:
        temp = Path(temporary)
        req, xml, out = temp / "request.json", temp / "official.xml", temp / "out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if base.digest(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("official XML drift")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=120)
        native = json.loads(out.read_text())
    rows = []
    for index, row in enumerate(native["rows"]):
        root = index // 12
        traces = row["trace"]
        if len(traces) != 17:
            raise ValueError("wrong action clock")
        states = np.asarray([t["state"] for t in traces])
        distances = np.linalg.norm(states[:, :2] - base.GOAL, axis=1)
        rows.append({"root_index": root, "candidate_index": index % 12, "name": row["policy"],
                     "selected": index % 12 == selected["rows"][root]["selected_index"],
                     "warmup_state_error": float(np.linalg.norm(states[1] - states[0])),
                     "initial_distance": float(distances[1]), "prefix_distance": float(distances[6]),
                     "terminal_distance": float(distances[-1]), "trace": traces})
    report = {"schema": "pc_geometry_rgb_fixed_bank_physical.v1", "rows": rows,
              "decision_sha256": base.digest(decision.OUTPUT), "protocol_sha256": base.digest(PROTOCOL),
              "worker_sha256": base.digest(backend.WORKER), "runner_sha256": base.digest(Path(__file__)),
              "mujoco_binding": native["binding"],
              "max_warmup_state_error": max(r["warmup_state_error"] for r in rows)}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, allow_nan=False)
        handle.write("\n")
    print("saved physical replay; warmup error", report["max_warmup_state_error"], flush=True)


if __name__ == "__main__":
    main()
