"""Fixed-action, stepwise shadow-physics versus modern-backend audit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base

WORKER = base.ROOT / "experiments/pc_geometry_backend_replay_worker_20261103_codex.py"
PROTOCOL = base.PROJECT / "docs/PC_GEOMETRY_BACKEND_STEPWISE_PROTOCOL_20261103_CODEX.md"
OUTPUT = base.ROOT / "results/pc_geometry_backend_stepwise_20261103_codex.json"
LEGACY = base.ROOT / ".legacy_mujoco210_py310"
RUNTIME = LEGACY / "vendor/mujoco-2.1.2"


def wall_contact(pairs):
    return any("particle_geom" in pair and any(name.startswith("wall_") for name in pair)
               for pair in pairs)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(base.OUTPUT.read_text())
    if source["xml_sha256"] != base.digest(base.XML):
        raise ValueError("source XML mismatch")
    request = {"rows": [{"start": row["start"], "policy": row["policy"],
                         "actions": [tick["action"] for tick in row["trace"]]}
                        for row in source["rows"]]}
    with tempfile.TemporaryDirectory(prefix="pc_geometry_backend_") as temporary:
        tmp = Path(temporary)
        req, official, modern = tmp / "request.json", tmp / "official.xml", tmp / "modern.xml"
        req.write_text(json.dumps(request, allow_nan=False))
        official.write_text(official_xml())
        modern.write_bytes(base.XML.read_bytes())
        import hashlib
        official_hash = hashlib.sha256(official.read_bytes()).hexdigest()
        if official_hash != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("unexpected official executor XML")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        results = {}
        for engine, python, xml in (("native", LEGACY / "bin/python", official),
                                    ("modern", Path(sys.executable), modern)):
            out = tmp / (engine + ".json")
            subprocess.run([str(python), str(WORKER), engine, str(req), str(xml), str(out)],
                           env=env, check=True, timeout=120)
            results[engine] = json.loads(out.read_text())
    native, modern = results["native"], results["modern"]
    if native["binding"] != "2.1.2.14" or modern["binding"] != "3.1.6":
        raise ValueError("unexpected simulator version")
    rows = []
    for old, left, right in zip(source["rows"], native["rows"], modern["rows"], strict=True):
        if (left["start"], left["policy"]) != (right["start"], right["policy"]):
            raise ValueError("replay mismatch")
        lstate = np.asarray([t["state"] for t in left["trace"]])
        rstate = np.asarray([t["state"] for t in right["trace"]])
        recorded = np.asarray([old["start"] + [0, 0]] +
                              [t["position"] + t["velocity"] for t in old["trace"]])
        if not np.array_equal(rstate, recorded):
            raise ValueError("modern replay disagrees with original physical receipt")
        pos = np.linalg.norm(lstate[:, :2] - rstate[:, :2], axis=1)
        vel = np.linalg.norm(lstate[:, 2:] - rstate[:, 2:], axis=1)
        clock = np.asarray([x["time"] for x in left["trace"]]) - np.asarray([x["time"] for x in right["trace"]])
        lw = [wall_contact(x["contacts"]) for x in left["trace"]]
        rw = [wall_contact(x["contacts"]) for x in right["trace"]]
        first = next((i for i, (a, b) in enumerate(zip(lw, rw)) if a or b), None)
        rows.append({"start": old["start"], "policy": old["policy"], "steps": len(old["trace"]),
                     "success": old["success"], "native": left["trace"], "modern": right["trace"],
                     "position_l2": pos.tolist(), "velocity_l2": vel.tolist(), "clock_delta": clock.tolist(),
                     "first_wall_contact_tick": first,
                     "wall_contact_disagreements": sum(a != b for a, b in zip(lw, rw)),
                     "contact_pair_disagreements": sum(a["contacts"] != b["contacts"]
                                                       for a, b in zip(left["trace"], right["trace"]))})
    max_pos = max(max(r["position_l2"]) for r in rows)
    max_vel = max(max(r["velocity_l2"]) for r in rows)
    max_clock = max(max(abs(x) for x in r["clock_delta"]) for r in rows)
    mismatch = sum(r["wall_contact_disagreements"] for r in rows)
    consistent = max_pos <= 1e-8 and max_vel <= 1e-8 and max_clock <= 1e-12 and mismatch == 0
    report = {"schema": "pc_geometry_backend_stepwise.v1", "rows": rows,
              "native": {k: native[k] for k in ("engine", "binding", "runtime", "timestep")},
              "modern": {k: modern[k] for k in ("engine", "binding", "runtime", "timestep")},
              "official_xml_sha256": official_hash, "modern_xml_sha256": base.digest(base.XML),
              "source_sha256": base.digest(base.OUTPUT), "protocol_sha256": base.digest(PROTOCOL),
              "worker_sha256": base.digest(WORKER), "max_position_l2": max_pos,
              "max_velocity_l2": max_vel, "max_clock_delta": max_clock,
              "wall_contact_disagreements": mismatch, "fixed_panel_consistent": consistent}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("fixed_panel_consistent", "max_position_l2",
                                             "max_velocity_l2", "max_clock_delta",
                                             "wall_contact_disagreements")}))


if __name__ == "__main__":
    main()
