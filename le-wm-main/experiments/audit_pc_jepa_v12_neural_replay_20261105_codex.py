"""Same-input rerun of the v12 neural workers, without planning or physics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from experiments import pc_jepa_v12_contact_score_bank_neural_posthoc_20261104_codex as old

ROOT, SOURCE = old.ROOT, old.OUT
OUT = ROOT / "results/pc_jepa_v12_neural_replay_audit_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V12_NEURAL_REPLAY_AUDIT_PROTOCOL_20261105_CODEX.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare(old_values, new_values):
    a, b = np.asarray(old_values, np.float64), np.asarray(new_values, np.float64)
    if a.shape != (64,) or b.shape != (64,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("expected 64 finite scores")
    delta = b - a
    return {"max_abs_difference": float(np.max(np.abs(delta))),
            "mean_abs_difference": float(np.mean(np.abs(delta))),
            "within_1e_6": int(np.count_nonzero(np.abs(delta) <= 1e-6)),
            "new_minus_original": delta.tolist()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    original_report = SOURCE / "neural/report.json"
    original_result = SOURCE / "result.json"
    prior_predict = SOURCE / "neural/predict/output/decision.json"
    prior_encode = SOURCE / "neural/encode/output/encoded.json"
    report = json.loads(original_report.read_text())
    pred = json.loads(prior_predict.read_text())
    enc = json.loads(prior_encode.read_text())
    result = json.loads(original_result.read_text())
    if (report["status"] != "complete" or result["status"] != "complete"
            or report["source_result_sha256"] != sha(original_result)
            or report["predict_sha256"] != sha(prior_predict)
            or report["encoder_sha256"] != sha(prior_encode)
            or result["arrays_sha256"] != sha(SOURCE / "arrays.npz")):
        raise ValueError("original results not bound")
    inputs = {
        "predict": ("context.npz", "config.json", "model_metadata.json", "bank.npy"),
        "encode": ("images.npz", "model_metadata.json"),
    }
    paths = [PROTOCOL, Path(__file__), original_report, original_result, prior_predict, prior_encode,
             old.PREDICT_WORKER, old.ENCODE_WORKER, SOURCE / "arrays.npz"]
    paths += [SOURCE / "neural" / group / "input" / name
              for group, names in inputs.items() for name in names]
    hashes = {str(p): sha(p) for p in paths}
    old_input = SOURCE / "neural/predict/input"
    old_image = SOURCE / "neural/encode/input/images.npz"
    if (pred["input_sha256"] != sha(old_input / "context.npz")
            or pred["bank_sha256"] != sha(old_input / "bank.npy")
            or pred["config_sha256"] != sha(old_input / "config.json")
            or pred["metadata_sha256"] != sha(old_input / "model_metadata.json")
            or pred["worker_sha256"] != sha(old.PREDICT_WORKER)
            or enc["image_sha256"] != sha(old_image)
            or enc["metadata_sha256"] != sha(SOURCE / "neural/encode/input/model_metadata.json")
            or enc["worker_sha256"] != sha(old.ENCODE_WORKER)):
        raise ValueError("worker inputs or source differ from original run")
    np.testing.assert_array_equal(np.asarray(pred["decision"]["scores"]), report["predicted"])
    np.testing.assert_array_equal(np.asarray(enc["scores"]), report["actual_encoded"])
    OUT.mkdir()
    try:
        for group, names in inputs.items():
            for sub in ("input", "output"):
                (OUT / group / sub).mkdir(parents=True)
            for name in names:
                source = SOURCE / "neural" / group / "input" / name
                target = OUT / group / "input" / name
                shutil.copyfile(source, target)
                if sha(source) != sha(target):
                    raise ValueError("copied input mismatch")
        print("Replaying 64 fixed predictions and terminal-image scores; no selection.", flush=True)
        old.launch(OUT / "predict", "experiments.pc_jepa_v12_contact_score_predict_worker_20261104_codex")
        old.launch(OUT / "encode", "experiments.pc_jepa_v11_same_bank_actual_encoding_worker_20261104_codex")
        p = json.loads((OUT / "predict/output/decision.json").read_text())
        e = json.loads((OUT / "encode/output/encoded.json").read_text())
        if any(sha(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError("source changed during replay")
        for obj, group, mapping in (
            (p, "predict", {"input_sha256": "context.npz", "bank_sha256": "bank.npy",
                            "config_sha256": "config.json", "metadata_sha256": "model_metadata.json"}),
            (e, "encode", {"image_sha256": "images.npz", "metadata_sha256": "model_metadata.json"}),
        ):
            if any(obj[key] != sha(OUT / group / "input" / name) for key, name in mapping.items()):
                raise ValueError("replay output input binding mismatch")
        if p["worker_sha256"] != sha(old.PREDICT_WORKER) or e["worker_sha256"] != sha(old.ENCODE_WORKER):
            raise ValueError("replay worker mismatch")
        output = {"status": "completed", "source_sha256": hashes,
                  "predicted": compare(report["predicted"], p["decision"]["scores"]),
                  "actual_encoded": compare(report["actual_encoded"], e["scores"]),
                  "original_meter": pred["meter"], "replay_meter": p["meter"],
                  "new_predict_sha256": sha(OUT / "predict/output/decision.json"),
                  "new_encode_sha256": sha(OUT / "encode/output/encoded.json"),
                  "scope": "same inputs, implementation and checkpoint; no new physics or independent replication"}
    except Exception:
        output = {"status": "failed", "source_sha256": hashes, "error": traceback.format_exc()}
    with (OUT / "audit.json").open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: output.get(k) for k in ("status", "error", "predicted", "actual_encoded")}), flush=True)
    if output["status"] != "completed":
        raise RuntimeError("neural replay failed; failure retained")


if __name__ == "__main__":
    main()
