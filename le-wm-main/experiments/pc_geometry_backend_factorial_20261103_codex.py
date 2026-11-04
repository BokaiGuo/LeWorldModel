"""Fixed-action 2x2 backend/XML diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as audit
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base

PROTOCOL = base.PROJECT / "docs/PC_GEOMETRY_BACKEND_FACTORIAL_PROTOCOL_20261103_CODEX.md"
OUTPUT = base.ROOT / "results/pc_geometry_backend_factorial_20261103_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(base.OUTPUT.read_text())
    request = {"rows": [{"start": row["start"], "policy": row["policy"],
                         "actions": [tick["action"] for tick in row["trace"]]}
                        for row in source["rows"]]}
    with tempfile.TemporaryDirectory(prefix="pc_geometry_factorial_") as temporary:
        tmp = Path(temporary)
        req = tmp / "request.json"
        req.write_text(json.dumps(request, allow_nan=False))
        xmls = {"official": tmp / "official.xml", "legacy": tmp / "legacy.xml"}
        xmls["official"].write_text(official_xml())
        xmls["legacy"].write_bytes(base.XML.read_bytes())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(audit.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(audit.RUNTIME / "lib"), str(audit.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        runs = {}
        for engine, python in (("native", audit.LEGACY / "bin/python"),
                               ("modern", Path(sys.executable))):
            for xml_name, xml in xmls.items():
                key = engine + "_" + xml_name
                out = tmp / (key + ".json")
                subprocess.run([str(python), str(audit.WORKER), engine, str(req), str(xml), str(out)],
                               env=env, check=True, timeout=120)
                runs[key] = json.loads(out.read_text())
        xml_hashes = {key: base.digest(path) for key, path in xmls.items()}
    pairs = (("native_official", "native_legacy"), ("modern_official", "modern_legacy"),
             ("native_official", "modern_official"), ("native_legacy", "modern_legacy"))
    comparisons = []
    for left, right in pairs:
        maxpos = maxvel = 0.0
        mismatch = 0
        for a, b in zip(runs[left]["rows"], runs[right]["rows"], strict=True):
            if a["start"] != b["start"] or a["policy"] != b["policy"]:
                raise ValueError("mismatched roots")
            for x, y in zip(a["trace"], b["trace"], strict=True):
                delta = np.asarray(x["state"]) - np.asarray(y["state"])
                maxpos = max(maxpos, float(np.linalg.norm(delta[:2])))
                maxvel = max(maxvel, float(np.linalg.norm(delta[2:])))
                mismatch += audit.wall_contact(x["contacts"]) != audit.wall_contact(y["contacts"])
        comparisons.append({"left": left, "right": right, "max_position_l2": maxpos,
                            "max_velocity_l2": maxvel, "wall_contact_disagreements": mismatch})
    report = {"schema": "pc_geometry_backend_factorial.v1", "comparisons": comparisons, "runs": runs,
              "xml_sha256": xml_hashes, "source_sha256": base.digest(base.OUTPUT),
              "protocol_sha256": base.digest(PROTOCOL), "worker_sha256": base.digest(audit.WORKER),
              "script_sha256": base.digest(Path(__file__))}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, allow_nan=False)
        handle.write("\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
