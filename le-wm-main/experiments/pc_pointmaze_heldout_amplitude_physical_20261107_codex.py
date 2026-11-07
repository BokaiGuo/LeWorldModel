"""Seal existing state-assisted predictions, then replay unqueried amplitudes physically."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_state_effect_local_dev_20261106_codex as source

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_HELDOUT_AMPLITUDE_PHYSICAL_PROTOCOL_20261107_CODEX.md"
SELECT = ROOT / "results/pc_pointmaze_heldout_amplitude_physical_selection_20261107_codex.json"
PHYSICAL = ROOT / "results/pc_pointmaze_heldout_amplitude_physical_20261107_codex.npz"
RESULT = ROOT / "results/pc_pointmaze_heldout_amplitude_physical_result_20261107_codex.json"
VALUES = np.array([-.1, -.03, .03, .1, 0.])
RULES = ("model", "parent", "probe_min", "linear_ad", "quadratic")


def choices(row):
    scores = {"model": row["heldout_model_costs"], **row["predictions"]}
    if set(scores) != set(RULES):
        raise ValueError("prediction rules mismatch")
    values = {key: np.asarray(scores[key], dtype=float) for key in RULES}
    if any(value.shape != (4,) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("invalid sealed scores")
    return {key: int(np.argmin(value)) for key, value in values.items()}


def seal():
    if SELECT.exists():
        raise FileExistsError(SELECT)
    model = json.loads(source.PREFIX.with_suffix(".json").read_text())
    if model["track_id"] != "official_state_assisted":
        raise ValueError("wrong track")
    rows = []
    for row in model["rows"]:
        if row["heldout_raw_x"] != VALUES[:4].tolist():
            raise ValueError("action bank changed")
        rows.append({"root": row["root"], "camera": row["camera"], "choices": choices(row),
                     "model_scores": row["heldout_model_costs"]})
    if {(r["root"], r["camera"]) for r in rows} != {(r, c) for r in range(1880, 1884) for c in range(2)} or len(rows) != 8:
        raise ValueError("panel mismatch")
    with SELECT.open("x") as stream:
        json.dump({"rows": rows, "source_sha256": file_hash(source.PREFIX.with_suffix(".json")),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "track": "official_state_assisted"}, stream, indent=2, allow_nan=False)


def physical():
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    selection = json.loads(SELECT.read_text())
    if (selection["runner_sha256"] != file_hash(__file__)
            or selection["protocol_sha256"] != file_hash(PROTOCOL)
            or selection["source_sha256"] != file_hash(source.PREFIX.with_suffix(".json"))):
        raise ValueError("sealed decision drift")
    archive = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True)
    states = np.stack([archive[root, frame].numpy() for root in range(1880, 1884) for frame in (0, 5)])
    with tempfile.TemporaryDirectory(prefix="pc_heldout_amp_") as tmp:
        directory = Path(tmp)
        request, xml, result = directory / "request.npz", directory / "scene.xml", directory / "traces.npz"
        np.savez_compressed(request, states=states)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        worker = Path(__file__).with_name("pc_pointmaze_heldout_amplitude_backend_20261107_codex.py")
        subprocess.run([str(LEGACY / "bin/python"), str(worker),
                        str(request), str(xml), str(result)], env=env, check=True, timeout=180)
        with np.load(result, allow_pickle=False) as tmp_result, np.load(
                ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz", allow_pickle=False) as prior:
            traces = tmp_result["traces"]
            if traces.shape != (8, 5, 6, 4):
                raise ValueError("physical trace shape mismatch")
            # Existing reference order is zero, +0.1, -0.1.
            np.testing.assert_allclose(traces[:, [4, 3, 0]], prior["traces"][:, :, 0], atol=1e-9, rtol=0)
            with PHYSICAL.open("xb") as stream:
                np.savez_compressed(stream, traces=traces, states=states,
                                    selection_sha256=np.asarray(file_hash(SELECT)),
                                    prior_sha256=np.asarray(file_hash(ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz")),
                                    worker_sha256=np.asarray(file_hash(worker)))
    print("Replayed all held-out amplitudes and matched original zero/plus/minus trajectories.", flush=True)


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    selection = json.loads(SELECT.read_text())
    if (selection["runner_sha256"] != file_hash(__file__) or selection["protocol_sha256"] != file_hash(PROTOCOL)
            or selection["source_sha256"] != file_hash(source.PREFIX.with_suffix(".json"))):
        raise ValueError("selection dependency drift")
    with np.load(PHYSICAL, allow_pickle=False) as data:
        if str(data["selection_sha256"]) != file_hash(SELECT):
            raise ValueError("physical selection binding mismatch")
        traces, states = data["traces"], data["states"]
        rows = []
        for row in selection["rows"]:
            index = row["root"] - 1880
            distances = np.linalg.norm(traces[2 * index, :4, -1, :2] - states[2 * index + 1, :2], axis=1)
            pairwise = [(i, j) for i in range(4) for j in range(i + 1, 4)
                        if abs(distances[i] - distances[j]) > 1e-12
                        and abs(row["model_scores"][i] - row["model_scores"][j]) > 1e-12]
            concordant = sum((distances[i] - distances[j]) * (row["model_scores"][i] - row["model_scores"][j]) > 0
                             for i, j in pairwise)
            rows.append({"root": row["root"], "camera": row["camera"], "distances": distances.tolist(),
                         "regret": {rule: float(distances[row["choices"][rule]] - distances.min()) for rule in RULES},
                         "model_pairwise_correct": int(concordant), "model_pairwise_non_ties": len(pairwise)})
    roots = [{"root": root, "regret": {rule: float(np.mean([x["regret"][rule] for x in rows if x["root"] == root]))
                                      for rule in RULES}} for root in range(1880, 1884)]
    mean = {rule: float(np.mean([x["regret"][rule] for x in roots])) for rule in RULES}
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "roots": roots, "mean_regret": mean,
                   "selection_sha256": file_hash(SELECT), "physical_sha256": file_hash(PHYSICAL),
                   "scope": "state-assisted exposed roots; distance is not standard success"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("seal", "physical", "analyze"):
        {"seal": seal, "physical": physical, "analyze": analyze}[sys.argv[1]]()
    else:
        raise SystemExit("expected seal|physical|analyze")
