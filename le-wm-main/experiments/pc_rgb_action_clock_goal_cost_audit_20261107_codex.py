"""Independently check terminal cost against goal embedding for clock sample."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_rgb_action_clock_internal_20261107_codex as clock

OUT = clock.ROOT / "results/pc_rgb_action_clock_goal_cost_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads(clock.OUT.read_text())
    if (prior["inputs_sha256"] != file_hash(clock.INPUTS)
            or prior["runner_sha256"] != file_hash(clock.__file__)):
        raise ValueError("clock inputs or code changed")
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5") as data:
        actions = np.asarray(data["action"][:])
    actions = actions[~np.isnan(actions).any(axis=1)]
    if hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest() != prior["finite_train_action_array_sha256"]:
        raise ValueError("action statistics changed")
    scaler = StandardScaler().fit(actions)
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    if model_fingerprint(model) != prior["model_fingerprint"]:
        raise ValueError("model differs")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows = []
    print("Recompute terminal visual cost for each clock panel condition.", flush=True)
    with np.load(clock.INPUTS, allow_pickle=False) as data, torch.inference_mode():
        for root in (0, 1, 8, 9):
            for side in (-1, 1):
                key = f"r{root}_white_{side}"
                raw = np.stack((data[key+"_plus_action"], data[key+"_minus_action"]))
                packets = torch.tensor(scaler.transform(raw.reshape(10, 2)).reshape(1, 1, 2, 10),
                                       dtype=torch.float32, device="cuda")
                pixels = transform(data[key+"_plus_initial"]).cuda()[None, None, None]
                goal = transform(data[f"r{root}_white_goal"]).cuda()[None, None, None]
                context = {"pixels": pixels, "goal": goal, "action": torch.zeros_like(packets[:, :, :1])}
                actual = model.get_cost(context, packets)
                direct = (context["predicted_emb"][:, :, -1] - context["goal_emb"][:, -1][:, None]).square().sum(-1)
                difference = float((actual-direct).abs().max())
                rows.append({"root": root, "side": side, "difference": difference})
    maximum = max(row["difference"] for row in rows)
    if maximum > 1e-4:
        raise ValueError("terminal cost mismatch")
    output = {"rows": rows, "max_abs_cost_error": maximum, "clock_sha256": file_hash(clock.OUT),
              "runner_sha256": file_hash(__file__), "model_fingerprint": prior["model_fingerprint"],
              "scope": "RGB-only exposed-panel numerical equivalence, not control"}
    with OUT.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
    print(json.dumps({"max_abs_cost_error": maximum}, indent=2))


if __name__ == "__main__":
    main()
