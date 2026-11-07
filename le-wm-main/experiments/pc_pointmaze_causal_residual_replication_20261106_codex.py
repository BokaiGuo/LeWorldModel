"""Fixed extra-panel replay and causal-residual experiment, preserving original artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments import pc_pointmaze_causal_residual_dev_20261106_codex as base
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = base.ROOT
NAME = "pc_pointmaze_causal_residual_replication_20261106_codex"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CAUSAL_RESIDUAL_REPLICATION_PROTOCOL_20261106_CODEX.md"
REPLAY = ROOT / "results" / (NAME + "_replay.json")
BASE_RUNNER = Path(base.__file__)


def replay():
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    if REPLAY.exists():
        raise FileExistsError(REPLAY)
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True)[1892:1900, :16].numpy()
    actions = torch.load(DATA / "actions.pth", map_location="cpu", weights_only=True)[1892:1900, :15].numpy()
    lengths = torch.load(DATA / "seq_lengths.pth", map_location="cpu", weights_only=True)
    if states.shape != (8, 16, 4) or actions.shape != (8, 15, 2) or any(int(lengths[i]) < 16 for i in range(1892, 1900)):
        raise ValueError("incomplete replay data")
    if not np.isfinite(states).all() or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
        raise ValueError("invalid replay states or actions")
    with tempfile.TemporaryDirectory(prefix="pc_extra_replay_") as tmp:
        folder = Path(tmp)
        request, xml, response = folder / "request.npz", folder / "scene.xml", folder / "response.npz"
        np.savez_compressed(request, states=states[:, 0], actions=actions)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        replay_runner = ROOT / "experiments/pc_pointmaze_packet_clock_replay_20261106_codex.py"
        subprocess.run([str(LEGACY / "bin/python"), str(replay_runner), "backend",
                        str(request), str(xml), str(response)], env=env, check=True, timeout=180)
        with np.load(response, allow_pickle=False) as data:
            errors = np.abs(data["traces"] - states[:, 1:])
            clock_error = float(np.abs(data["clocks"] - np.arange(1, 16) * 5 * float(data["simulator_dt"])).max())
    output = {"roots": list(range(1892, 1900)), "max_abs_state_error": float(errors.max()),
              "per_root_max_abs_error": errors.max(axis=(1, 2)).tolist(), "max_clock_error": clock_error,
              "strict_replay_atol_1e_4": bool(errors.max() <= 1e-4),
              "source_states_sha256": file_hash(DATA / "states.pth"),
              "source_actions_sha256": file_hash(DATA / "actions.pth"),
              "protocol_sha256": file_hash(PROTOCOL), "replay_backend_sha256": file_hash(replay_runner),
              "base_runner_sha256": file_hash(BASE_RUNNER),
              "runner_sha256": file_hash(__file__)}
    with REPLAY.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(output, indent=2), flush=True)
    if not output["strict_replay_atol_1e_4"] or clock_error > 1e-9:
        raise ValueError("replay failed")


def configure():
    """Use a private copy of original functions with only fixed panel constants changed."""
    source = BASE_RUNNER.read_text()
    source = source.replace("1884:1892", "1892:1900").replace("range(1884, 1892)", "range(1892, 1900)")
    source = source.replace("1884+i", "1892+i").replace("-1884", "-1892")
    source = source.replace('NAME = "pc_pointmaze_causal_residual_dev_20261106_codex"', f'NAME = "{NAME}"')
    source = source.replace("PC_POINTMAZE_CAUSAL_RESIDUAL_DEV_PROTOCOL_20261106_CODEX.md", PROTOCOL.name)
    source = source.replace("pc_pointmaze_packet_clock_replay_20261106_codex.json", REPLAY.name)
    # __file__ must identify this actual wrapper for the existing dependency hash.
    namespace = {"__file__": __file__, "__name__": "causal_replication_frozen"}
    exec(compile(source, str(BASE_RUNNER), "exec"), namespace)
    record = json.loads(REPLAY.read_text())
    if (record["roots"] != list(range(1892, 1900)) or record["runner_sha256"] != file_hash(__file__)
            or record["base_runner_sha256"] != file_hash(BASE_RUNNER)
            or record["protocol_sha256"] != file_hash(PROTOCOL) or record["max_clock_error"] > 1e-9):
        raise ValueError("replay manifest mismatch")
    return namespace


def persistence(ns):
    out = ROOT / "results" / (NAME + "_persistence.json")
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads(ns["SELECTION"].read_text())
    audit = json.loads(ns["AUDIT"].read_text())
    if (selection["runner_sha256"] != file_hash(__file__) or selection["protocol_sha256"] != file_hash(PROTOCOL)
            or selection["public_sha256"] != file_hash(ns["PUBLIC"])
            or audit["selection_sha256"] != file_hash(ns["SELECTION"])
            or audit["physical_sha256"] != file_hash(ns["PRIVATE"])):
        raise ValueError("sealed artifacts differ")
    model, _, permission = load_model()
    if model_fingerprint(model) != selection["model_fingerprint"] or permission.fingerprint != selection["input_audit_hash"]:
        raise ValueError("model differs")
    rows = []
    with np.load(ns["PRIVATE"], allow_pickle=False) as physical, np.load(ns["PUBLIC"], allow_pickle=False) as inputs, torch.inference_mode():
        for i in range(8):
            for camera in range(2):
                current = model.encode(rgb_tensor(inputs["current"][i, camera]))[0]
                targets = [model.encode(rgb_tensor(physical["images"][3*i+1, a, camera, -1]))[0] for a in range(3)]
                previous = next(r for r in audit["rows"] if r["root"] == 1892+i and r["camera"] == camera)
                rows.append({"root": 1892+i, "camera": camera, "mse": {
                    **previous["latent_mse"],
                    "persistence": float(torch.stack([(current-z).square().mean() for z in targets]).mean())}})
    per_root = [{"root": root, "mse": {k: float(np.mean([r["mse"][k] for r in rows if r["root"] == root]))
                for k in ("raw", "corrected", "persistence")}} for root in range(1892, 1900)]
    output = {"rows": rows, "per_root": per_root,
              "mean_mse": {k: float(np.mean([r["mse"][k] for r in per_root]))
                           for k in ("raw", "corrected", "persistence")},
              "corrected_better_than_persistence_roots": sum(r["mse"]["corrected"] < r["mse"]["persistence"] for r in per_root),
              "selection_sha256": file_hash(ns["SELECTION"]), "audit_sha256": file_hash(ns["AUDIT"]),
              "scope": "state-assisted post-hoc development diagnostic"}
    with out.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("mean_mse", "corrected_better_than_persistence_roots")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("replay", "prepare", "score", "audit", "persistence"):
        raise SystemExit("expected replay|prepare|score|audit|persistence")
    if sys.argv[1] == "replay":
        replay()
    else:
        ns = configure()
        if sys.argv[1] == "persistence":
            persistence(ns)
        else:
            ns[sys.argv[1]]()
