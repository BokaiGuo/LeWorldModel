"""Evaluate the same sealed root3 plans in RGB and native physics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments import pc_equal_resource_dev_20261103_codex as base
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge

old, h, ROOT = base.old, base.h, base.ROOT
PROTOCOL = ROOT.parent/"docs/PC_ROOT3_FIXED_CHAIN_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT/"results/pc_root3_fixed_chain_diagnostic_20261103_codex.json"
IMAGES = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or IMAGES.exists():
        raise FileExistsError(OUTPUT)
    sha = old.prior.sha
    prior = json.loads(base.ROOT_PLAN.read_text())
    source = json.loads(old.DECISIONS.read_text())
    if (prior["runner_sha256"] != sha(Path(base.__file__)) or prior["protocol_sha256"] != sha(base.PROTOCOL)
            or prior["history_sha256"] != sha(old.HISTORY)):
        raise ValueError("prior plan mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    plans = [next(x for x in prior["rows"] if x["root"] == 3 and x["method"] == role)
             for role in ("spectral", "no_spectral")]
    requests = [{"start": list(old.STARTS[3]), "policy": row["method"],
                 "actions": old.WARMUP.tolist()+row["chosen"]["prefix"]+row["chosen"]["suffix"]} for row in plans]
    with tempfile.TemporaryDirectory(prefix="pc_root3_diagnostic_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp/"request.json", tmp/"scene.xml", tmp/"physical.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME/"lib"), str(backend.RUNTIME/"bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY/"bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=120)
        physical = json.loads(out.read_text())
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != 2:
        raise ValueError("physical replay mismatch")
    rows, images = [], {}
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT/"external/local_dinov2")
        model, _ = h.load_model_and_dataset(args, device)
        model.eval()
        if h.parameter_hash(model) != prior["model_hash"]:
            raise ValueError("model mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        goal_state, _ = h.estimate_state(goal, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
        for plan, physical_row in zip(plans, physical["rows"], strict=True):
            method, program = plan["method"], plan["chosen"]
            if physical_row["policy"] != method or len(physical_row["trace"]) != 31:
                raise ValueError("physical role/clock mismatch")
            with ShadowRGBBridge(np.asarray([*old.STARTS[3], 0., 0.])) as env:
                frames = [env.initial_rgb.copy()]
                warm, _ = old.do_actions(env, old.WARMUP)
                frames += warm
                if [old.digest_image(x) for x in frames] != source["rows"][3]["rgb_sha256"]:
                    raise ValueError("warmup RGB mismatch")
                observed, times = old.do_actions(env, program["prefix"]+program["suffix"])
            images[method] = np.stack(observed)
            state = h.causal_state(observed[-2], observed[-1], metadata, offset, .05)
            with torch.no_grad():
                encoded = h.encode_with_state(model, observed[-1], state, device)["proprio"]
                rgb_score = float(h.projected_cost(encoded.reshape(1, 256, 16), target, pinv, std)[0])
            distances = np.linalg.norm(np.asarray([x["state"][:2] for x in physical_row["trace"]])-[1.8, 2.8], axis=1)
            rows.append({"method": method, "name": program["name"], "actions": program["prefix"]+program["suffix"],
                         "model_terminal_score": program["terminal_score"], "real_rgb_terminal_score": rgb_score,
                         "initial_distance": float(distances[15]), "prefix_distance": float(distances[20]),
                         "physical_terminal_distance": float(distances[30]), "trace": physical_row["trace"],
                         "rgb_sha256": [old.digest_image(x) for x in observed], "times_ns": times,
                         "root_counts": plan["counts"], "svd_calls": plan["svd_calls"],
                         "samples_per_candidate": len(program["scores"])})
    with IMAGES.open("xb") as handle:
        np.savez_compressed(handle, **images)
    old.write_json(OUTPUT, {"rows": rows, "prior_sha256": sha(base.ROOT_PLAN), "protocol_sha256": sha(PROTOCOL),
                            "runner_sha256": sha(Path(__file__)), "worker_sha256": sha(backend.WORKER),
                            "images_sha256": sha(IMAGES)})
    print("saved two fixed-chain RGB and physics diagnostics", flush=True)


if __name__ == "__main__":
    main()
