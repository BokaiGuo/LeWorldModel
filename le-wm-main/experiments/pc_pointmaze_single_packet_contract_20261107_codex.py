"""Compare the actual state-assisted one-packet call path with its primitives."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_same_renderer_three_way_20261107_codex import ACTIONS, INPUT, PRED, ACTUAL
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model, ROOT

OUTPUT = ROOT / "results/pc_pointmaze_single_packet_contract_v2_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SINGLE_PACKET_CONTRACT_PROTOCOL_20261107_CODEX.md"


def difference(left, right):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    delta = (left.detach().cpu().double() - right.detach().cpu().double()).abs()
    return {"max_abs": float(delta.max()), "mse": float(delta.square().mean())}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    model, dataset, audit = load_model()
    from app.plan_common.datasets import DATA_STATS
    initial_hash = model_fingerprint(model)
    wrapper = StateAssistedPointMaze(model, audit)
    stats = {}
    for name in ("action_mean", "action_std", "proprio_mean", "proprio_std"):
        actual = getattr(dataset, name)
        official = torch.tensor(DATA_STATS["pointmaze"][name])
        stats[name] = difference(actual, official)
        torch.testing.assert_close(actual, official, atol=0, rtol=0)
    rows = []
    with (np.load(INPUT, allow_pickle=False) as inputs_file,
          np.load(PRED, allow_pickle=False) as saved,
          np.load(ACTUAL, allow_pickle=False) as truth,
          torch.inference_mode()):
        for root in range(4):
            for camera in range(2):
                key = f"r{root}_v{camera}"
                inputs = {
                    "visual": rgb_tensor(inputs_file["current"][root, camera]),
                    "proprio": torch.tensor(inputs_file["proprio"][root, camera],
                                             device="cuda", dtype=torch.float32)[None, None],
                }
                encoded = model.encode(inputs)
                visual_only = model.encode(inputs["visual"])
                torch.testing.assert_close(encoded["visual"], visual_only, atol=1e-5, rtol=1e-5)
                expected_proprio = model.model.encode_proprio(
                    model.preprocessor.normalize_proprios(inputs["proprio"].cpu()).cuda())
                torch.testing.assert_close(encoded["proprio"], expected_proprio, atol=1e-5, rtol=1e-5)
                for index, action in enumerate(ACTIONS):
                    raw = torch.tensor([[action, 0.]] * 5, dtype=torch.float32)
                    manual = ((raw - dataset.action_mean) / dataset.action_std)
                    official = model.preprocessor.normalize_actions(raw)
                    torch.testing.assert_close(manual, official, atol=0, rtol=0)
                    packet = manual.reshape(1, 1, 10).cuda()
                    wrapped = wrapper.predict(inputs, packet)["visual"]
                    unrolled = model.unroll(encoded, act_suffix=packet)["visual"]
                    action_features = model.model.encode_act(packet.permute(1, 0, 2))
                    direct, _, _ = model.model.forward_pred(
                        encoded["visual"], action_features, encoded["proprio"])
                    if wrapped.shape[0] != 2:
                        raise ValueError("one context frame plus one predicted frame expected")
                    for left, right in ((wrapped, unrolled), (wrapped[0], encoded["visual"][:, 0]),
                                        (wrapped[1], direct[:, -1])):
                        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
                    old = torch.from_numpy(saved[key + "_pred"][index]).cuda()
                    torch.testing.assert_close(wrapped[1], old, atol=1e-5, rtol=1e-5)
                    real = torch.from_numpy(truth[key][index]).cuda()
                    rows.append({"root": 1880 + root, "camera": camera, "action": action,
                                 "wrapper_direct": difference(wrapped[1], direct[:, -1]),
                                 "sealed_prediction": difference(wrapped[1], old),
                                 "prediction_real": difference(wrapped[1], real),
                                 "context_real": difference(wrapped[0], real)})
    if model_fingerprint(model) != initial_hash:
        raise ValueError("model weights changed")
    result = {"track": audit.track_id, "scope": "exposed state-assisted development only",
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "input_sha256": file_hash(INPUT), "prediction_sha256": file_hash(PRED),
              "actual_sha256": file_hash(ACTUAL), "model_sha256": initial_hash,
              "statistics": stats, "n_conditions": len(rows), "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"n_conditions": len(rows),
                      "max_direct": max(r["wrapper_direct"]["max_abs"] for r in rows),
                      "max_replay": max(r["sealed_prediction"]["max_abs"] for r in rows),
                      "prediction_mse": float(np.mean([r["prediction_real"]["mse"] for r in rows])),
                      "context_mse": float(np.mean([r["context_real"]["mse"] for r in rows]))}, indent=2))


if __name__ == "__main__":
    main()
