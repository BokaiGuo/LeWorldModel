"""Record all roots without weakening the original physical replay threshold."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments import pc_pointmaze_archive_context_multistep_20261107_codex as source
from experiments.pc_input_permissions import STATE, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

PROTOCOL = source.ROOT.parent / "docs/PC_POINTMAZE_ARCHIVE_CONTEXT_PHYSICS_FAILURE_AUDIT_PROTOCOL_20261107_CODEX.md"
OUT = source.ROOT / "results/pc_pointmaze_archive_context_physics_failure_audit_20261107_codex.json"


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
    model, _, audit1 = load_model()
    if model_fingerprint(model) != plan["model_sha256"] or audit1.fingerprint != plan["audit1"]:
        raise ValueError("model or input audit drift")
    if configure_context(model, audit1).fingerprint != plan["audit2"]:
        raise ValueError("context-2 audit drift")
    states = torch.load(source.FOLDER / "states.pth", map_location="cpu", weights_only=True)
    rows = []
    with (np.load(source.INPUT, allow_pickle=False) as inputs, np.load(source.PRED, allow_pickle=False) as pred,
          tempfile.TemporaryDirectory(prefix="pc_context_failure_audit_") as temporary, torch.inference_mode()):
        if str(inputs["states_hash"]) != file_hash(source.FOLDER / "states.pth") or str(inputs["actions_hash"]) != file_hash(source.FOLDER / "actions.pth"):
            raise ValueError("archive drift")
        folder = Path(temporary)
        xml_path = folder / "scene.xml"
        xml_path.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, root in enumerate(source.ROOTS):
            archive = source.prior.archive_path(root)
            if str(inputs["archive_hashes"][index]) != file_hash(archive):
                raise ValueError("image archive drift")
            request, response = folder / f"req_{root}.npz", folder / f"res_{root}.npz"
            with request.open("xb") as f:
                np.savez_compressed(f, state=inputs["states"][index], actions=inputs["actions"][index][None])
            subprocess.run([str(LEGACY / "bin/python"), str(Path(source.physics.__file__).resolve()),
                            "--backend", str(request), str(xml_path), str(response)], env=env, check=True, timeout=120)
            with np.load(response, allow_pickle=False) as replay:
                trace = replay["trace"][0].copy()
            if trace.shape != (26, 5) or not np.isfinite(trace).all():
                raise ValueError("bad replay")
            np.testing.assert_allclose(trace[:, 4], np.arange(26) * .05, rtol=0, atol=1e-10)
            per_step = np.max(np.abs(trace[:, :4] - states[root, :26].numpy()), axis=1)
            images = torch.load(archive, map_location="cpu", weights_only=True)
            metrics = []
            for step in range(1, 6):
                t = step * 5
                actual = model.encode({"visual": rgb_tensor(images[t].numpy()),
                                       "proprio": states[root, t].float().cuda()[None, None]})
                values = {}
                for key in ("visual", "proprio"):
                    truth = actual[key][0, 0].cpu().numpy()
                    values[key] = {f"window{w}": source.mse(pred[f"r{root}_w{w}_{key}"][step, 0], truth)
                                   for w in (1, 2)}
                    values[key]["persistence"] = source.mse(pred[f"r{root}_w1_{key}"][0, 0], truth)
                metrics.append({"environment_step": t, **values})
            rows.append({"root": root, "physical_max_error": float(per_step.max()),
                         "physical_pass_1e4": bool(np.max(per_step) <= 1e-4),
                         "failing_steps": np.flatnonzero(per_step > 1e-4).tolist(), "metrics": metrics})
            print(f"root={root} physical max={per_step.max():.8g}", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model weights changed")
    summary = []
    for step in range(5):
        entry = {"environment_step": (step + 1) * 5}
        for key in ("visual", "proprio"):
            values = [r["metrics"][step][key] for r in rows]
            entry[key] = {name: float(np.mean([v[name] for v in values]))
                          for name in ("window1", "window2", "persistence")}
            entry[key]["window2_lower_roots"] = sum(v["window2"] < v["window1"] for v in values)
        summary.append(entry)
    with OUT.open("x") as f:
        json.dump({"track": STATE, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_plan_sha256": file_hash(source.PLAN), "rows": rows, "summary": summary,
                   "physical_pass_count": sum(r["physical_pass_1e4"] for r in rows),
                   "scope": "exposed factual archive, physical mismatch retained, no decision claim"},
                  f, indent=2, allow_nan=False)
    print(json.dumps({"physical_pass": sum(r["physical_pass_1e4"] for r in rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
