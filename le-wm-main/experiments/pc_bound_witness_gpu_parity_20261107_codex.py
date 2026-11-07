"""Reproduce one sealed state-assisted descriptor with immutable witnesses."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_action_effect_descriptor import describe
from experiments.pc_action_witness_v2_20261107_codex import ActionWitnessV2, describe_bound
from experiments.pc_input_permissions import StateAssistedPointMaze, file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import ROOT, rgb_tensor, witness
from experiments.pc_state_track_smoke_20261106_codex import load_model

OUTPUT = ROOT / "results/pc_bound_witness_gpu_parity_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_BOUND_WITNESS_GPU_PARITY_PROTOCOL_20261107_CODEX.md"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior_path = ROOT / "results/pc_state_effect_local_dev_20261106_codex.json"
    source_path = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
    prior = json.loads(prior_path.read_text())
    source_meta = json.loads(source_path.with_suffix(".json").read_text())
    if source_meta["payload_sha256"] != file_hash(source_path):
        raise ValueError("sealed input differs")
    model, dataset, audit = load_model()
    wrapper = StateAssistedPointMaze(model, audit)
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["frozen_fingerprint"]:
        raise ValueError("model differs from sealed experiment")
    with np.load(source_path, allow_pickle=False) as data, torch.no_grad():
        inputs = {"visual": rgb_tensor(data["images"][0, 0, 0, 0]),
                  "proprio": torch.tensor(data["traces"][0, 0, 0, 0].copy(),
                                          device="cuda", dtype=torch.float32)[None, None]}
        goal = model.encode(rgb_tensor(data["images"][1, 0, 0, 0]))[0]
        mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()

        def score(array):
            raw = torch.as_tensor(array, dtype=torch.float32, device="cuda")
            pred = wrapper.predict(inputs, ((raw - mean) / std).reshape(1, 1, 10))["visual"][-1]
            return (pred - goal).square().mean()

        old = describe(score, witness("zero", 0.),
                       [witness("minus_probe", -.003), witness("plus_probe", .003)])
        new = [ActionWitnessV2(n, "1880", "5_environment_steps",
                               [[x, 0.] for _ in range(5)])
               for n, x in (("zero", 0.), ("minus_probe", -.003), ("plus_probe", .003))]
        current = describe_bound(score, new[0], new[1:])
        old_values = old["costs"].cpu().numpy()
        new_values = current["costs"].cpu().numpy()
    row = next(r for r in prior["rows"] if r["root"] == 1880 and r["camera"] == 0)
    np.testing.assert_allclose(new_values, old_values, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(new_values[0], row["predicted_costs_zero_plus_minus"][0], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose((new_values[0] - new_values[1:]), row["descriptor_effects"], atol=1e-6, rtol=1e-5)
    if model_fingerprint(model) != fingerprint or len(wrapper.receipts) != 6:
        raise ValueError("weights or query count mismatch")
    payload = {"track": "official_state_assisted", "root": 1880, "camera": 0,
               "model_sha256": fingerprint, "protocol_sha256": file_hash(PROTOCOL),
               "runner_sha256": file_hash(Path(__file__)), "source_sha256": file_hash(source_path),
               "prior_sha256": file_hash(prior_path), "old_costs": old_values.tolist(),
               "new_costs": new_values.tolist(), "bound_actions": current["actions"],
               "predict_calls": len(wrapper.receipts), "scope": "interface parity only; not gate evidence"}
    with OUTPUT.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"old": payload["old_costs"], "new": payload["new_costs"], "calls": 6}))


if __name__ == "__main__":
    main()
