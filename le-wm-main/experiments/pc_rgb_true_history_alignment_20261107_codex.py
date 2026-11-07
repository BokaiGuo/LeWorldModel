"""Sealed physical RGB histories, causal scoring, and post-hoc successor audit."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as source

ROOT = source.ROOT
NAME = "pc_rgb_true_history_alignment_20261107_codex"
INPUT = ROOT / "results" / (NAME + "_input.npz")
TRUTH = ROOT / "results" / (NAME + "_truth.npz")
TRACE = ROOT / "results" / (NAME + "_trace.json")
PRED = ROOT / "results" / (NAME + "_pred.npz")
RECORD = ROOT / "results" / (NAME + "_record.json")
AUDIT = ROOT / "results" / (NAME + "_audit.json")
PROTOCOL = ROOT.parent / "docs/PC_RGB_TRUE_HISTORY_ALIGNMENT_PROTOCOL_20261107_CODEX.md"


def collect():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
    for path in (INPUT, TRUTH, TRACE):
        if path.exists():
            raise FileExistsError(path)
    physical = json.loads(source.PHYSICAL.read_text())
    if physical["images_sha256"] != file_hash(source.IMAGES):
        raise ValueError("source images changed")
    inputs, truth, rows = {}, {}, []
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    print("Collect actual 15-step RGB/action histories for all eight fixed development conditions.", flush=True)
    try:
        with np.load(source.IMAGES, allow_pickle=False) as original:
            for root in (0, 1, 8, 9):
                row = physical["rows"][root]
                if row["root"] != root or not row["valid"]:
                    raise ValueError("source root invalid")
                for side in (-1, 1):
                    condition = next(c for c in row["conditions"] if c["side"] == side and c["appearance"] == "white")
                    env.reset(seed=881001 + root, options={"state": np.asarray(condition["requested_state"]),
                                                           "goal_state": np.asarray(row["goal"])})
                    actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                    first = snapshot(actor)
                    np.testing.assert_allclose(first, condition["actions"]["plus"]["initial_snapshot"], atol=1e-9, rtol=0)
                    key = f"r{root}_s{side}"
                    frames = [actor.render().copy()]
                    np.testing.assert_array_equal(frames[0], original[f"r{root}_white_{side}_plus_initial"])
                    packets = [np.asarray(condition["actions"][name]["physical_action"], np.float32)
                               for name in ("plus", "minus", "plus")]
                    if any(p.shape != (5, 2) or not np.isfinite(p).all() for p in packets):
                        raise ValueError("invalid source action")
                    steps = []
                    for packet in packets:
                        for action in packet:
                            obs, _, terminated, truncated, _ = actor.step(action.copy())
                            steps.append({"state": np.asarray(obs["state"]).tolist(),
                                          "terminated": bool(terminated), "truncated": bool(truncated),
                                          "snapshot": snapshot(actor).tolist()})
                            if truncated:
                                raise ValueError("unexpected truncation")
                        frames.append(actor.render().copy())
                    inputs[key+"_frames"] = np.stack(frames[:3])
                    inputs[key+"_goal"] = original[f"r{root}_white_goal"]
                    inputs[key+"_actions"] = np.stack(packets)
                    truth[key+"_future"] = frames[3]
                    rows.append({"root": root, "side": side, "steps": steps})
    finally:
        env.close()
    with INPUT.open("xb") as file:
        np.savez_compressed(file, **inputs)
    with TRUTH.open("xb") as file:
        np.savez_compressed(file, **truth)
    result = {"rows": rows, "input_sha256": file_hash(INPUT), "truth_sha256": file_hash(TRUTH),
              "source_physical_sha256": file_hash(source.PHYSICAL), "source_images_sha256": file_hash(source.IMAGES),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "scope": "exposed PushT development physical trajectories"}
    with TRACE.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)


def model_and_scaler():
    import h5py
    from sklearn.preprocessing import StandardScaler
    from eval import img_transform
    from experiments.checkpoint_compat import load_compatible_pretrained
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5") as file:
        actions = np.asarray(file["action"][:])
    actions = actions[~np.isnan(actions).any(axis=1)]
    scaler = StandardScaler().fit(actions)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    return model, scaler, transform, hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


def score():
    for path in (PRED, RECORD):
        if path.exists():
            raise FileExistsError(path)
    trace = json.loads(TRACE.read_text())
    if (trace["input_sha256"] != file_hash(INPUT) or trace["protocol_sha256"] != file_hash(PROTOCOL)
            or trace["runner_sha256"] != file_hash(__file__)):
        raise ValueError("sealed input/contract changed")
    model, scaler, transform, action_hash = model_and_scaler()
    fingerprint = model_fingerprint(model)
    prior = json.loads((ROOT / "results/pc_rgb_contact_track_replay_20261107_codex_scores.json").read_text())
    if fingerprint != prior["model_sha256"]:
        raise ValueError("checkpoint differs from prior RGB model")
    predictions, rows = {}, []
    print("Score only observed RGB and action histories; no successor file opened.", flush=True)
    with np.load(INPUT, allow_pickle=False) as data, torch.inference_mode():
        for root in (0, 1, 8, 9):
            for side in (-1, 1):
                key = f"r{root}_s{side}"
                frames = torch.stack([transform(x) for x in data[key+"_frames"]]).cuda()
                goal = transform(data[key+"_goal"]).cuda()
                raw = data[key+"_actions"]
                flat = scaler.transform(raw.reshape(15, 2))
                roundtrip = float(np.max(np.abs(scaler.inverse_transform(flat) - raw.reshape(15, 2))))
                packets = torch.tensor(flat.reshape(1, 1, 3, 10), dtype=torch.float32, device="cuda")
                if roundtrip > 1e-6:
                    raise ValueError("action normalization roundtrip")
                results = {}
                for name in ("true_history", "zero_history", "single"):
                    history = frames[-1:] if name == "single" else frames
                    selected = packets[:, :, -1:] if name == "single" else packets.clone()
                    if name == "zero_history":
                        selected[:, :, :2] = 0
                    context = {"pixels": history[None, None], "goal": goal[None, None, None],
                               "action": torch.zeros((1, 1, len(history), 10), device="cuda")}
                    cost = model.get_cost(context, selected)
                    output = context["predicted_emb"][0, 0, -1]
                    predictions[key+"_"+name] = output.cpu().numpy()
                    results[name] = {"model_goal_cost": float(cost.item()),
                                     "history_frames": int(len(history)),
                                     "packet_count": int(selected.shape[2])}
                current = model.encode({"pixels": frames[-1:][None]})["emb"][0, 0]
                target = model.encode({"pixels": goal[None, None]})["emb"][0, 0]
                predictions[key+"_current"] = current.cpu().numpy()
                predictions[key+"_goal"] = target.cpu().numpy()
                rows.append({"root": root, "side": side, "roundtrip": roundtrip, "predictions": results})
    if fingerprint != model_fingerprint(model):
        raise ValueError("model weight drift")
    with PRED.open("xb") as file:
        np.savez_compressed(file, **predictions)
    output = {"rows": rows, "model_sha256": fingerprint, "input_sha256": file_hash(INPUT),
              "trace_sha256": file_hash(TRACE), "pred_sha256": file_hash(PRED),
              "action_array_sha256": action_hash, "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "scope": "RGB-only, no future successor used"}
    with RECORD.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    record, trace = json.loads(RECORD.read_text()), json.loads(TRACE.read_text())
    if (record["input_sha256"] != file_hash(INPUT) or record["trace_sha256"] != file_hash(TRACE)
            or record["pred_sha256"] != file_hash(PRED) or trace["truth_sha256"] != file_hash(TRUTH)
            or record["protocol_sha256"] != file_hash(PROTOCOL) or record["runner_sha256"] != file_hash(__file__)):
        raise ValueError("sealed dependencies changed")
    model, _, transform, action_hash = model_and_scaler()
    if model_fingerprint(model) != record["model_sha256"] or action_hash != record["action_array_sha256"]:
        raise ValueError("model/action statistics changed")
    rows = []
    with np.load(TRUTH, allow_pickle=False) as truth, np.load(PRED, allow_pickle=False) as pred, torch.inference_mode():
        for r in record["rows"]:
            key = f"r{r['root']}_s{r['side']}"
            image = transform(truth[key+"_future"]).cuda()
            actual = model.encode({"pixels": image[None, None]})["emb"][0, 0].cpu().numpy()
            goal = pred[key+"_goal"]
            real_cost = float(np.square(actual-goal).sum())
            mse, cost_error = {}, {}
            for name in ("true_history", "zero_history", "single", "current"):
                value = pred[key+"_"+name]
                mse[name] = float(np.mean(np.square(value-actual)))
                cost_error[name] = float(abs(np.square(value-goal).sum() - real_cost))
            rows.append({"root": r["root"], "side": r["side"], "mse": mse, "goal_cost_abs_error": cost_error})
    names = ("true_history", "zero_history", "single", "current")
    per_root = [{"root": root, "mse": {n: float(np.mean([r["mse"][n] for r in rows if r["root"] == root]))
                                        for n in names}} for root in (0, 1, 8, 9)]
    output = {"rows": rows, "per_root": per_root,
              "mean_mse": {n: float(np.mean([r["mse"][n] for r in per_root])) for n in names},
              "mean_goal_cost_abs_error": {n: float(np.mean([r["goal_cost_abs_error"][n] for r in rows])) for n in names},
              "true_history_better_than_single_roots": sum(r["mse"]["true_history"] < r["mse"]["single"] for r in per_root),
              "record_sha256": file_hash(RECORD), "truth_sha256": file_hash(TRUTH),
              "scope": "RGB-only exposed development, post-seal successor diagnostic; not closed-loop"}
    with AUDIT.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("mean_mse", "mean_goal_cost_abs_error",
                                              "true_history_better_than_single_roots")}, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("collect", "score", "audit"):
        raise SystemExit("expected collect|score|audit")
    {"collect": collect, "score": score, "audit": audit}[sys.argv[1]]()
