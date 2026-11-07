"""Separate RGB-only inputs, sealed GPU scoring, and legacy parity audit."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit, manifest
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as source

ROOT = source.ROOT
NAME = "pc_rgb_contact_track_replay_20261107_codex"
INPUTS = ROOT / "results" / (NAME + "_inputs.npz")
SCORE = ROOT / "results" / (NAME + "_scores.json")
AUDIT = ROOT / "results" / (NAME + "_audit.json")
PROTOCOL = ROOT.parent / "docs/PC_RGB_CONTACT_TRACK_REPLAY_PROTOCOL_20261107_CODEX.md"
NAMES = ("zero", "plus", "minus")


def prepare():
    if INPUTS.exists():
        raise FileExistsError(INPUTS)
    record = json.loads(source.PHYSICAL.read_text())
    if record["images_sha256"] != file_hash(source.IMAGES):
        raise ValueError("image fingerprint mismatch")
    payload = {}
    with np.load(source.IMAGES, allow_pickle=False) as images:
        for row in record["rows"]:
            if not row["valid"]:
                raise ValueError("panel includes invalid root")
            root = row["root"]
            for condition in row["conditions"]:
                side, color = condition["side"], condition["appearance"]
                for action in NAMES:
                    item = condition["actions"][action]
                    key = f"r{root}_{color}_{side}_{action}"
                    image = images[key + "_initial"]
                    if source.array_sha(image) != item["initial_sha256"]:
                        raise ValueError("initial image hash mismatch")
                    payload[key + "_initial"] = image
                    payload[key + "_action"] = np.asarray(item["physical_action"], dtype=np.float32)
                    payload[f"r{root}_{color}_goal"] = images[f"r{root}_{color}_goal"]
    if len([k for k in payload if k.endswith("_action")]) != 192:
        raise ValueError("panel not complete")
    with INPUTS.open("xb") as file:
        np.savez_compressed(file, **payload)
    print(f"Sealed {len(payload)} permitted input arrays in {INPUTS}", flush=True)


def score():
    import h5py
    from sklearn.preprocessing import StandardScaler
    from eval import img_transform
    from experiments.checkpoint_compat import load_compatible_pretrained, _resolve_local
    if SCORE.exists():
        raise FileExistsError(SCORE)
    stats = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
    weights, config = _resolve_local("pusht/lewm")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = model_fingerprint(model)
    with h5py.File(stats) as file:
        actions = np.asarray(file["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    audit = freeze_audit(RGB, "lewm", [weights, config, stats, Path(__file__),
                         Path(inspect.getfile(RGBLeWM)), Path(inspect.getfile(type(model))),
                         Path(inspect.getfile(load_compatible_pretrained)), Path(inspect.getfile(img_transform))],
                         preprocessing="original PushT img_transform and finite train action scaler",
                         clock="one 5-step packet; explicit empty history placeholder")
    wrapper = RGBLeWM(model, audit)
    rows = []
    print("RGB-only scoring: sealed RGB, goal and action input only; no outcome files.", flush=True)
    with np.load(INPUTS, allow_pickle=False) as data, torch.inference_mode():
        for root in range(16):
            for color in ("white", "gray"):
                for side in (-1, 1):
                    for name in NAMES:
                        key = f"r{root}_{color}_{side}_{name}"
                        raw = data[key + "_action"]
                        if raw.shape != (5, 2) or not np.isfinite(raw).all():
                            raise ValueError("invalid action")
                        packet = torch.tensor(scaler.transform(raw).reshape(1, 1, 1, 10),
                                              dtype=torch.float32, device="cuda")
                        current = transform(data[key + "_initial"]).cuda()
                        goal = transform(data[f"r{root}_{color}_goal"]).cuda()
                        inputs = {"pixels": current[None, None, None], "goal": goal[None, None, None],
                                  "action": torch.zeros_like(packet)}
                        cost, _ = wrapper.score(inputs, packet)
                        rows.append({"root": root, "appearance": color, "side": side, "action": name,
                                     "score": float(cost[0, 0])})
            print(f"root {root}: 12 RGB-only candidate scores", flush=True)
    if before != model_fingerprint(model):
        raise ValueError("model parameter drift")
    record = manifest(audit, algorithm_version="contact_candidate_rgb_replay_v1", protocol_path=PROTOCOL,
                      task_manifest_path=INPUTS, receipts=wrapper.receipts)
    record.update(rows=rows, model_sha256=before, runner_sha256=file_hash(__file__),
                  input_sha256=file_hash(INPUTS),
                  scope="RGB-only input replay; exposed diagnostic panel; not closed-loop evidence")
    with SCORE.open("x") as file:
        json.dump(record, file, indent=2, allow_nan=False)
    print(f"Sealed {len(rows)} scores: {SCORE}", flush=True)


def compare(new_rows, old_rows):
    old = {}
    for root in old_rows:
        for condition in root["conditions"]:
            for name in NAMES:
                old[(root["root"], condition["appearance"], condition["side"], name)] = condition["actions"][name]["predicted_cost"]
    new = {(r["root"], r["appearance"], r["side"], r["action"]): r["score"] for r in new_rows}
    if len(new_rows) != 192 or len(new) != 192 or set(new) != set(old):
        raise ValueError("candidate identity or denominator mismatch")
    close = [bool(np.isclose(new[k], old[k], atol=1e-4, rtol=1e-5)) for k in old]
    choices = []
    for root in range(16):
        for color in ("white", "gray"):
            for side in (-1, 1):
                keys = [(root, color, side, name) for name in NAMES]
                choices.append({"root": root, "appearance": color, "side": side,
                                "old": int(np.argmin([old[k] for k in keys])),
                                "new": int(np.argmin([new[k] for k in keys]))})
    return {"candidate_count": len(new), "close_count": sum(close),
            "max_absolute_score_error": max(abs(new[k] - old[k]) for k in old),
            "choice_count": len(choices), "choice_matches": sum(c["old"] == c["new"] for c in choices),
            "choices": choices}


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    result, old = json.loads(SCORE.read_text()), json.loads(source.MODEL.read_text())
    if (result["input_sha256"] != file_hash(INPUTS) or result["runner_sha256"] != file_hash(__file__)
            or result["protocol_hash"] != file_hash(PROTOCOL)
            or old["physical_sha256"] != file_hash(source.PHYSICAL)
            or old["features_sha256"] != file_hash(source.FEATURES)
            or result["model_sha256"] != old["model_sha256"]):
        raise ValueError("sealed dependencies or model drift")
    physical = json.loads(source.PHYSICAL.read_text())
    if physical["images_sha256"] != file_hash(source.IMAGES):
        raise ValueError("physical image drift")
    with np.load(INPUTS, allow_pickle=False) as inputs, np.load(source.IMAGES, allow_pickle=False) as images:
        for root in physical["rows"]:
            for condition in root["conditions"]:
                color, side = condition["appearance"], condition["side"]
                goal_key = f"r{root['root']}_{color}_goal"
                np.testing.assert_array_equal(inputs[goal_key], images[goal_key])
                for name in NAMES:
                    key = f"r{root['root']}_{color}_{side}_{name}"
                    np.testing.assert_array_equal(inputs[key + "_initial"], images[key + "_initial"])
                    np.testing.assert_array_equal(inputs[key + "_action"],
                                                  np.asarray(condition["actions"][name]["physical_action"],
                                                             dtype=np.float32))
    comparison = compare(result["rows"], old["rows"])
    comparison.update({"parity_pass": comparison["close_count"] == 192 and comparison["choice_matches"] == 64,
                       "scores_sha256": file_hash(SCORE), "old_model_sha256": file_hash(source.MODEL),
                       "scope": "exposed RGB-only interface replay, not performance or gate evidence"})
    with AUDIT.open("x") as file:
        json.dump(comparison, file, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in comparison.items() if k != "choices"}, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "score", "audit"):
        raise SystemExit("expected prepare|score|audit")
    {"prepare": prepare, "score": score, "audit": audit}[sys.argv[1]]()
