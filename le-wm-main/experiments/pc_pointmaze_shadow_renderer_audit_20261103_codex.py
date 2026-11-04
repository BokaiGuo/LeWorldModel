"""Run the preregistered post-outcome shadow renderer replay."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from experiments import pc_pointmaze_cp12_native_position_stepwise_20261103_codex as panel
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME

ROOT = panel.ROOT
WORKER = ROOT / "experiments/pc_pointmaze_shadow_renderer_worker_20261103_codex.py"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SHADOW_RENDERER_PROTOCOL_20261103_CODEX.md"
PREVIOUS = ROOT / "results/pc_pointmaze_rgb_forward_clock_differences_20261103_CODEX.npz"
OUT = ROOT / "results/pc_pointmaze_shadow_renderer_audit_20261103_CODEX.json"
ARRAYS = OUT.with_suffix(".npz")
JOURNAL = OUT.with_name(OUT.stem + "_events.jsonl")
EXPECTED = {**panel.EXPECTED,
            WORKER: "99d84f089bafaa8127552a169083ea2c092bf672e3bcea661f30739e83fefd7b",
            PROTOCOL: "55e2ed99fe17e459bda1ac75b09c74463f1639b48a827891238d7c8ee3567a3d",
            PREVIOUS: "91ea76ea2baa3309fadf507e5c1367844cccc658ec33af3829fcf40a3bad434a"}


def main():
    if any(p.exists() for p in (OUT, ARRAYS, JOURNAL)):
        raise FileExistsError("refuse overwrite")
    for path, digest in EXPECTED.items():
        if panel.prior.sha(path) != digest:
            raise ValueError(f"source changed: {path}")
    _, _, names = panel.prior.files()
    for path in (panel.prior.CONTEXT, panel.prior.DECISIONS, panel.prior.EXECUTION, panel.prior.EVALUATION):
        panel.prior.load_sealed(path)
    starts, actions, sealed_hashes = panel.build_inputs(names)
    xml = official_xml().encode()
    if panel.digest_bytes(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
        raise ValueError("XML changed")
    with JOURNAL.open("x") as journal, tempfile.TemporaryDirectory(prefix="pointmaze_shadow_") as tmp:
        tmp = Path(tmp)
        request, scene, result = tmp/"request.npz", tmp/"scene.xml", tmp/"output.npz"
        np.savez_compressed(request, starts=starts, actions=actions)
        scene.write_bytes(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME/"lib"), str(RUNTIME/"bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        journal.write(json.dumps({"event": "worker_started", "time_ns": time.time_ns()})+"\n")
        journal.flush()
        proc = subprocess.run([str(LEGACY/"bin/python"), str(WORKER), str(request), str(scene), str(result)],
                              env=env, capture_output=True, text=True, timeout=300)
        journal.write(json.dumps({"event": "worker_returned", "returncode": proc.returncode,
                                  "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-2000:]})+"\n")
        journal.flush()
        if proc.returncode or not result.exists():
            raise RuntimeError(f"worker failed; see {JOURNAL}")
        with np.load(result, allow_pickle=False) as src:
            data = {key: src[key].copy() for key in src.files}
        if (str(data["request_sha256"].item()) != panel.prior.sha(request)
                or str(data["xml_sha256"].item()) != panel.prior.sha(scene)):
            raise ValueError("request provenance mismatch")
        # Preserve full observations before applying gates, including on a negative result.
        with ARRAYS.open("xb") as f:
            np.savez_compressed(f, **data, sealed_hashes=sealed_hashes)
        with np.load(PREVIOUS, allow_pickle=False) as previous:
            original = previous["states"][0]
        state_matches = np.all(data["states"] == original, axis=2)
        raw_matches = data["raw_hashes"] == sealed_hashes
        residuals = data["residuals"]
        checks = {"all_sealed_raw_hashes": bool(raw_matches.all()),
                  "main_full_tick_exact": bool(state_matches.all()),
                  "state_copy_main_unchanged_shadow_state": bool(data["invariants"].all()),
                  "finite_geometry_below_1e_minus_12": bool(np.isfinite(residuals).all() and (residuals < 1e-12).all())}
        report = {"schema": "pointmaze_shadow_renderer_audit.v1", "passed": all(checks.values()),
                  "checks": checks, "sealed_raw_matches": int(raw_matches.sum()),
                  "state_tick_matches": int(state_matches.sum()),
                  "invariant_counts": data["invariants"].sum(axis=(0, 1)).tolist(),
                  "geometry_residual_maxima": residuals.max(axis=(0, 1)).tolist(),
                  "shadow_vs_raw_changed": int(np.count_nonzero(data["raw_hashes"] != data["shadow_hashes"])),
                  "metadata": json.loads(str(data["metadata"].item())),
                  "arrays_sha256": panel.prior.sha(ARRAYS),
                  "dependencies": {str(p): panel.prior.sha(p) for p in [*EXPECTED, Path(__file__)]},
                  "counts": {"trajectories": 64, "boundaries": 1152, "commands": 1088, "model_calls": 0, "planner_calls": 0}}
        with OUT.open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        journal.write(json.dumps({"event": "gates_recorded", "passed": report["passed"]})+"\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("shadow audit failed; negative result preserved")


if __name__ == "__main__":
    main()
