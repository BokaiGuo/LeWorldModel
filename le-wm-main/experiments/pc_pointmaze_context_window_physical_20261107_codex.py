"""Replay preselected context-window actions, without selecting from physical outcomes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_pointmaze_context_window_decision_20261107_codex as previous
from experiments import pc_fixed_prefix_state_physical_20261107_codex as physics
from experiments.pc_input_permissions import STATE, file_hash

PROTOCOL = previous.inputs.ROOT.parent / "docs/PC_POINTMAZE_CONTEXT_WINDOW_PHYSICAL_PROTOCOL_20261107_CODEX.md"
OUT = previous.inputs.ROOT / "results/pc_pointmaze_context_window_physical_20261107_codex.json"
METHODS = ("parent", "lookahead", "vector", "shuffled")


def main():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(previous.OUT.read_text())
    for key, path in (("protocol_sha256", previous.PROTOCOL), ("runner_sha256", previous.__file__),
                      ("config_sha256", previous.inputs.CONFIG), ("input_sha256", previous.inputs.PUBLIC)):
        if plan[key] != file_hash(path):
            raise ValueError(f"selection drift {key}")
    if plan["track"] != STATE or len(plan["rows"]) != 3:
        raise ValueError("track or panel mismatch")
    xml = official_xml()
    rows = []
    with (np.load(previous.inputs.PUBLIC, allow_pickle=False) as public,
          tempfile.TemporaryDirectory(prefix="pc_context_physics_") as temporary):
        folder = Path(temporary)
        xml_path = folder / "scene.xml"
        xml_path.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, row in enumerate(plan["rows"]):
            if row["root"] != 1884 + index:
                raise ValueError("unexpected root")
            state = public["proprio"][index, 0].copy()
            goal = public["goal_proprio"][index, 0].copy()
            actions = np.asarray([row["methods"][method]["windows"][window]["actions"]
                                  for method in METHODS for window in ("1", "2")], dtype=np.float32)
            if actions.shape != (8, 25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                raise ValueError("invalid sealed action")
            request, response = folder / f"request_{index}.npz", folder / f"response_{index}.npz"
            with request.open("xb") as f:
                np.savez_compressed(f, state=state, actions=actions)
            subprocess.run([str(LEGACY / "bin/python"), str(Path(physics.__file__).resolve()),
                            "--backend", str(request), str(xml_path), str(response)],
                           env=env, check=True, timeout=120)
            with np.load(response, allow_pickle=False) as result:
                traces = result["trace"].copy()
                versions = {"binding": str(result["binding"]), "runtime": str(result["runtime"])}
            if traces.shape != (8, 26, 5) or not np.isfinite(traces).all():
                raise ValueError("invalid traces")
            np.testing.assert_allclose(traces[:, 0, :4], np.broadcast_to(state, (8, 4)), atol=1e-7, rtol=0)
            np.testing.assert_allclose(traces[:, :, 4], np.broadcast_to(np.arange(26) * .05, (8, 26)),
                                       atol=1e-10, rtol=0)
            outcomes = {}
            for j, method in enumerate(METHODS):
                outcomes[method] = {}
                for k, window in enumerate(("1", "2")):
                    trace = traces[2*j + k]
                    distances = np.linalg.norm(trace[:, :2] - goal[:2], axis=1)
                    hit = np.flatnonzero(distances < .5)
                    first = int(hit[0]) if len(hit) else None
                    outcomes[method][window] = {"ever": first is not None, "first_hit": first,
                                                "prefix_hit": first is not None and first <= 10,
                                                "initial_distance": float(distances[0]),
                                                "final_distance": float(distances[-1]), "trace": trace.tolist()}
            rows.append({"root": row["root"], "outcomes": outcomes})
            print(f"root={row['root']} replayed", flush=True)
    summary = {}
    for method in METHODS:
        pairs = [(r["outcomes"][method]["1"], r["outcomes"][method]["2"]) for r in rows]
        summary[method] = {"context1_ever": sum(a["ever"] for a, b in pairs),
                           "context2_ever": sum(b["ever"] for a, b in pairs),
                           "context2_wins": sum(b["ever"] and not a["ever"] for a, b in pairs),
                           "context2_losses": sum(a["ever"] and not b["ever"] for a, b in pairs),
                           "context1_prefix_hits": sum(a["prefix_hit"] for a, b in pairs),
                           "context2_prefix_hits": sum(b["prefix_hit"] for a, b in pairs)}
    with OUT.open("x") as f:
        json.dump({"track": STATE, "scope": "exposed open-loop state-assisted counterfactual",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "plan_sha256": file_hash(previous.OUT), "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                   "versions": versions, "rows": rows, "summary": summary}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
