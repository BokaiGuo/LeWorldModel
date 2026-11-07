"""Re-score the sealed RGB histories without opening the physical trace or truth."""
import json
import sys

import numpy as np
import torch

from experiments import pc_rgb_true_history_alignment_20261107_codex as base
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

DIRECTORY = base.ROOT / "results"
MANIFEST = DIRECTORY / "pc_rgb_true_history_isolated_20261107_codex_manifest.json"
PREDICTIONS = DIRECTORY / "pc_rgb_true_history_isolated_20261107_codex_pred.npz"
RECORD = DIRECTORY / "pc_rgb_true_history_isolated_20261107_codex_record.json"
COMPARISON = DIRECTORY / "pc_rgb_true_history_isolated_20261107_codex_comparison.json"


def prepare():
    """Export public provenance only; this phase is not the scoring process."""
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    trace = json.loads(base.TRACE.read_text())
    previous = json.loads(base.RECORD.read_text())
    if trace["input_sha256"] != file_hash(base.INPUT):
        raise ValueError("input differs from collected trace")
    if trace["protocol_sha256"] != file_hash(base.PROTOCOL):
        raise ValueError("protocol differs from collected trace")
    manifest = {
        "input_sha256": trace["input_sha256"],
        "protocol_sha256": trace["protocol_sha256"],
        "model_sha256": previous["model_sha256"],
        "action_array_sha256": previous["action_array_sha256"],
        "roots": [0, 1, 8, 9],
        "sides": [-1, 1],
        "scope": "public provenance only; no successor images or physical states",
    }
    with MANIFEST.open("x") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)


def score():
    for path in (PREDICTIONS, RECORD):
        if path.exists():
            raise FileExistsError(path)
    manifest = json.loads(MANIFEST.read_text())
    if manifest["input_sha256"] != file_hash(base.INPUT):
        raise ValueError("public input changed")
    if manifest["protocol_sha256"] != file_hash(base.PROTOCOL):
        raise ValueError("protocol changed")
    model, scaler, transform, action_hash = base.model_and_scaler()
    fingerprint = model_fingerprint(model)
    if fingerprint != manifest["model_sha256"] or action_hash != manifest["action_array_sha256"]:
        raise ValueError("model or normalizer source changed")
    predictions = {}
    print("Predict sealed RGB histories without loading the physical trace or successor.", flush=True)
    with np.load(base.INPUT, allow_pickle=False) as data, torch.inference_mode():
        for root in manifest["roots"]:
            for side in manifest["sides"]:
                key = f"r{root}_s{side}"
                frames = torch.stack([transform(x) for x in data[key + "_frames"]]).cuda()
                goal = transform(data[key + "_goal"]).cuda()
                raw = data[key + "_actions"]
                flat = scaler.transform(raw.reshape(15, 2))
                if np.max(np.abs(scaler.inverse_transform(flat) - raw.reshape(15, 2))) > 1e-6:
                    raise ValueError("action normalization roundtrip")
                packets = torch.tensor(flat.reshape(1, 1, 3, 10), dtype=torch.float32, device="cuda")
                for name in ("true_history", "zero_history", "single"):
                    history = frames[-1:] if name == "single" else frames
                    selected = packets[:, :, -1:] if name == "single" else packets.clone()
                    if name == "zero_history":
                        selected[:, :, :2] = 0
                    context = {"pixels": history[None, None], "goal": goal[None, None, None],
                               "action": torch.zeros((1, 1, len(history), 10), device="cuda")}
                    model.get_cost(context, selected)
                    predictions[key + "_" + name] = context["predicted_emb"][0, 0, -1].cpu().numpy()
                predictions[key + "_current"] = model.encode({"pixels": frames[-1:][None]})["emb"][0, 0].cpu().numpy()
                predictions[key + "_goal"] = model.encode({"pixels": goal[None, None]})["emb"][0, 0].cpu().numpy()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weight drift")
    with PREDICTIONS.open("xb") as file:
        np.savez_compressed(file, **predictions)
    record = {"manifest_sha256": file_hash(MANIFEST), "input_sha256": file_hash(base.INPUT),
              "pred_sha256": file_hash(PREDICTIONS), "runner_sha256": file_hash(__file__),
              "model_sha256": fingerprint, "action_array_sha256": action_hash,
              "scope": "RGB-only isolated prediction; no truth or physical trace read"}
    with RECORD.open("x") as file:
        json.dump(record, file, indent=2, allow_nan=False)


def compare():
    if COMPARISON.exists():
        raise FileExistsError(COMPARISON)
    record = json.loads(RECORD.read_text())
    if record["pred_sha256"] != file_hash(PREDICTIONS) or record["manifest_sha256"] != file_hash(MANIFEST):
        raise ValueError("isolated prediction or manifest changed")
    original_record = json.loads(base.RECORD.read_text())
    if original_record["pred_sha256"] != file_hash(base.PRED):
        raise ValueError("original prediction changed")
    with np.load(base.PRED, allow_pickle=False) as old, np.load(PREDICTIONS, allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise ValueError("prediction keys differ")
        differences = {key: float(np.max(np.abs(old[key] - new[key]))) for key in old.files}
    output = {"max_abs_difference": max(differences.values()), "per_array": differences,
              "original_record_sha256": file_hash(base.RECORD), "isolated_record_sha256": file_hash(RECORD),
              "scope": "comparison after both predictions sealed"}
    with COMPARISON.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
    print(json.dumps({"max_abs_difference": output["max_abs_difference"]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "score", "compare"):
        raise SystemExit("expected prepare|score|compare")
    {"prepare": prepare, "score": score, "compare": compare}[sys.argv[1]]()
