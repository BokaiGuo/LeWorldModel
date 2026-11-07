"""State-assisted same-render-path frozen prediction and physical encoding audit."""
import json
from pathlib import Path
import sys

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_pointmaze_heldout_amplitude_encoding_20261107_codex as aligned
from experiments import pc_pointmaze_original_renderer_replay_20261107_codex as replay

ROOT = aligned.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SAME_RENDERER_THREE_WAY_PROTOCOL_20261107_CODEX.md"
INPUT = ROOT / "results/pc_pointmaze_same_renderer_three_way_20261107_codex_input.npz"
PREFIX = ROOT / "results/pc_pointmaze_same_renderer_three_way_v2_20261107_codex"
MODEL_METADATA = ROOT / "results/pc_camera_multiview_dev_20261106_codex_selection.json"
PRED = PREFIX.with_name(PREFIX.name + "_prediction.npz")
PLAN = PREFIX.with_name(PREFIX.name + "_plan.json")
ACTUAL = PREFIX.with_name(PREFIX.name + "_actual.npz")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ACTIONS = (-.1, -.03, .03, .1)


def prepare():
    if INPUT.exists():
        raise FileExistsError(INPUT)
    audit = json.loads(replay.OUTPUT.read_text())
    if (audit["replay_sha256"] != file_hash(replay.REPLAY)
            or audit["new_frames_sha256"] != file_hash(aligned.FRAMES)
            or audit["metrics"]["selected_new_frames"]["max"] != 0):
        raise ValueError("current renderer parity not established")
    with np.load(replay.REPLAY, allow_pickle=False) as data:
        images, traces = data["images"], data["traces"]
        if images.shape != (8, 3, 2, 6, 224, 224, 3) or traces.shape != (8, 3, 2, 6, 4):
            raise ValueError("unexpected source shape")
        current, goal = images[::2, 0, :, 0], images[1::2, 0, :, 0]
        proprio = traces[::2, 0, :, 0]
        with INPUT.open("xb") as stream:
            np.savez_compressed(stream, current=current, goal=goal, proprio=proprio,
                                renderer_replay_sha256=np.asarray(file_hash(replay.REPLAY)))
    print("Sealed RGB/goal/proprio initial inputs only.")


def predict():
    import torch
    from experiments.pc_input_permissions import StateAssistedPointMaze, manifest
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
    from experiments.pc_state_track_smoke_20261106_codex import load_model

    if PRED.exists() or PLAN.exists():
        raise FileExistsError("prediction output exists")
    model, dataset, permission = load_model()
    fingerprint = model_fingerprint(model)
    old = json.loads(MODEL_METADATA.read_text())
    if (old["track_id"] != "official_state_assisted" or fingerprint != old["model_fingerprint"]
            or permission.fingerprint != old["input_audit_hash"]):
        raise ValueError("model or input contract changed")
    wrapper = StateAssistedPointMaze(model, permission)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    arrays, rows = {}, []
    with np.load(INPUT, allow_pickle=False) as data, torch.inference_mode():
        if str(data["renderer_replay_sha256"]) != file_hash(replay.REPLAY):
            raise ValueError("input source drift")
        for root in range(4):
            for camera in range(2):
                key = f"r{root}_v{camera}"
                inputs = {"visual": rgb_tensor(data["current"][root, camera]),
                          "proprio": torch.tensor(data["proprio"][root, camera], dtype=torch.float32,
                                                   device="cuda")[None, None]}
                goal = model.encode(rgb_tensor(data["goal"][root, camera]))[0]
                features, scores = [], []
                for x in ACTIONS:
                    raw = torch.tensor([[x, 0.]] * 5, device="cuda")
                    action = ((raw - mean) / std).reshape(1, 1, 10)
                    predicted = wrapper.predict(inputs, action)["visual"][-1]
                    features.append(predicted.cpu().numpy())
                    scores.append(float((predicted - goal).square().mean()))
                arrays[key + "_pred"] = np.stack(features)
                arrays[key + "_goal"] = goal.cpu().numpy()
                rows.append({"root": 1880 + root, "camera": camera, "scores": scores,
                             "choice": int(np.argmin(scores))})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    with PRED.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    record = manifest(permission, algorithm_version="same_renderer_four_action_v1", protocol_path=PROTOCOL,
                      task_manifest_path=INPUT, receipts=wrapper.receipts)
    record.update(rows=rows, predictions_sha256=file_hash(PRED), runner_sha256=file_hash(__file__),
                  model_metadata_sha256=file_hash(MODEL_METADATA),
                  model_sha256=fingerprint, actions=list(ACTIONS),
                  scope="state-assisted exposed development; no candidate outcomes read")
    with PLAN.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(f"Sealed {len(rows)} decisions, {len(wrapper.receipts)} action predictions.")


def evaluate():
    import torch
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
    from experiments.pc_state_track_smoke_20261106_codex import load_model

    if ACTUAL.exists() or RESULT.exists():
        raise FileExistsError("evaluation output exists")
    plan = json.loads(PLAN.read_text())
    if (plan["predictions_sha256"] != file_hash(PRED) or plan["task_manifest_hash"] != file_hash(INPUT)
            or plan["protocol_hash"] != file_hash(PROTOCOL) or plan["runner_sha256"] != file_hash(__file__)
            or plan["model_metadata_sha256"] != file_hash(MODEL_METADATA)
            or plan["track_id"] != "official_state_assisted"):
        raise ValueError("sealed plan drift")
    model, _, permission = load_model()
    if model_fingerprint(model) != plan["model_sha256"] or permission.fingerprint != plan["input_audit_hash"]:
        raise ValueError("model contract drift")
    arrays, rows = {}, []
    with (np.load(aligned.FRAMES, allow_pickle=False) as frames, np.load(
            aligned.prior.PHYSICAL, allow_pickle=False) as physical,
          np.load(PRED, allow_pickle=False) as prediction, torch.inference_mode()):
        if str(frames["physical_sha256"]) != file_hash(aligned.prior.PHYSICAL):
            raise ValueError("physical frames drift")
        for item in plan["rows"]:
            root, camera = item["root"] - 1880, item["camera"]
            key = f"r{root}_v{camera}"
            actual = np.stack([model.encode(rgb_tensor(frames["frames"][root, i, camera]))[0].cpu().numpy()
                               for i in range(4)])
            target = physical["states"][2 * root + 1, :2]
            distances = np.linalg.norm(physical["traces"][2 * root, :4, -1, :2] - target, axis=1)
            result = aligned.comparisons(prediction[key + "_pred"], actual, prediction[key + "_goal"], distances)
            np.testing.assert_allclose(result["predicted_costs"], item["scores"], atol=1e-6, rtol=1e-5)
            if result["selected"]["predicted"] != item["choice"]:
                raise ValueError("predicted choice mismatch")
            result.update(root=item["root"], camera=camera)
            rows.append(result)
            arrays[key] = actual
    if len(rows) != 8 or len({(r["root"], r["camera"]) for r in rows}) != 8:
        raise ValueError("incomplete panel")
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    means = {name: float(np.mean([np.mean([r["regret"][name] for r in rows if r["root"] == root])
                                 for root in range(1880, 1884)]))
             for name in ("predicted", "encoded", "physical")}
    counts = {}
    for name in ("prediction_encoding", "encoding_physical", "prediction_physical"):
        pairs = [p[name] for r in rows for p in r["pairs"] if p[name] is not None]
        counts[name] = {"agree": sum(pairs), "non_ties": len(pairs)}
    with ACTUAL.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "mean_regret": means, "pair_counts": counts,
                   "plan_sha256": file_hash(PLAN), "prediction_sha256": file_hash(PRED),
                   "actual_sha256": file_hash(ACTUAL), "frames_sha256": file_hash(aligned.FRAMES),
                   "physical_sha256": file_hash(aligned.prior.PHYSICAL), "runner_sha256": file_hash(__file__),
                   "track": "official_state_assisted",
                   "scope": "exposed fixed panel; actual encoding evaluation only"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps({"mean_regret": means, "pair_counts": counts}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "predict", "evaluate"):
        raise SystemExit("expected prepare|predict|evaluate")
    {"prepare": prepare, "predict": predict, "evaluate": evaluate}[sys.argv[1]]()
