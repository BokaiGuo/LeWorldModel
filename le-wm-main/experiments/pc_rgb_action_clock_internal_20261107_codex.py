"""GPU verification of single-frame LeWM candidate packet semantics."""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import _resolve_local, load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "results/pc_rgb_contact_track_replay_20261107_codex_inputs.npz"
SCORES = ROOT / "results/pc_rgb_contact_track_replay_20261107_codex_scores.json"
PROTOCOL = ROOT.parent / "docs/PC_RGB_ACTION_CLOCK_INTERNAL_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_action_clock_internal_20261107_codex.json"


def manual_two_packets(model, pixels, packets):
    """Exact one-frame predictor recurrence; packets are [1, 1, 2, D]."""
    encoded = model.encode({"pixels": pixels[:, 0]})["emb"]
    one = model.predict(encoded, model.action_encoder(packets[:, 0, :1]))[:, -1:]
    two = model.predict(torch.cat((encoded, one), dim=1),
                        model.action_encoder(packets[:, 0]))[:, -1:]
    return torch.stack((encoded[:, 0], one[:, 0], two[:, 0]), dim=1)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads(SCORES.read_text())
    if prior["input_sha256"] != file_hash(INPUTS):
        raise ValueError("sealed inputs changed")
    stats = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
    weights, config = _resolve_local("pusht/lewm")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"]:
        raise ValueError("checkpoint model drift")
    with h5py.File(stats) as file:
        actions = np.asarray(file["action"][:])
    actions = actions[~np.isnan(actions).any(axis=1)]
    scaler = StandardScaler().fit(actions)
    stats_hash = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows = []
    print("Check single-frame context, 5/10-step packet recurrence and placeholder dependence.", flush=True)
    with np.load(INPUTS, allow_pickle=False) as data, torch.inference_mode():
        for root in (0, 1, 8, 9):
            for side in (-1, 1):
                prefix = f"r{root}_white_{side}"
                first = data[prefix+"_plus_action"]
                second = data[prefix+"_minus_action"]
                raw = np.stack((first, second))
                normalized = scaler.transform(raw.reshape(10, 2))
                roundtrip = float(np.abs(scaler.inverse_transform(normalized) - raw.reshape(10, 2)).max())
                packet = torch.tensor(normalized.reshape(1, 1, 2, 10), device="cuda", dtype=torch.float32)
                pixels = transform(data[prefix+"_plus_initial"]).cuda()[None, None, None]
                goal = transform(data[f"r{root}_white_goal"]).cuda()[None, None, None]
                def evaluate(placeholder, candidate):
                    context = {"pixels": pixels, "goal": goal, "action": placeholder}
                    cost = model.get_cost(context, candidate)
                    return cost, context["predicted_emb"]
                zero = torch.zeros((1, 1, 1, 10), device="cuda")
                cost, trajectory = evaluate(zero, packet)
                other_cost, other_trajectory = evaluate(torch.ones_like(zero), packet)
                manual = manual_two_packets(model, pixels, packet)
                modified = packet.clone()
                modified[:, :, 1] = packet[:, :, 0]
                modified_cost, modified_trajectory = evaluate(zero, modified)
                first_only_cost, first_only_trajectory = evaluate(zero, packet[:, :, :1])
                record = {"root": root, "side": side, "action_roundtrip_max_abs": roundtrip,
                          "placeholder_trajectory_max_abs": float((trajectory-other_trajectory).abs().max()),
                          "placeholder_cost_abs": float((cost-other_cost).abs().max()),
                          "manual_recurrence_max_abs": float((trajectory[:, 0]-manual).abs().max()),
                          "single_packet_prefix_max_abs": float((trajectory[:, :, :2]-first_only_trajectory).abs().max()),
                          "second_action_terminal_l2_change": float(torch.linalg.vector_norm(
                              trajectory[:, :, -1]-modified_trajectory[:, :, -1])),
                          "one_packet_cost": float(first_only_cost.item()), "two_packet_cost": float(cost.item()),
                          "changed_second_packet_cost": float(modified_cost.item()),
                          "trajectory_shape": list(trajectory.shape)}
                rows.append(record)
    limits = ("action_roundtrip_max_abs", "placeholder_trajectory_max_abs", "placeholder_cost_abs",
              "manual_recurrence_max_abs", "single_packet_prefix_max_abs")
    maxima = {key: max(row[key] for row in rows) for key in limits}
    if maxima["action_roundtrip_max_abs"] > 1e-6 or any(maxima[k] > 1e-4 for k in limits[1:]):
        raise ValueError(f"clock/input contract mismatch: {maxima}")
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    result = {"rows": rows, "maxima": maxima,
              "nonzero_second_action_responses": sum(r["second_action_terminal_l2_change"] > 1e-7 for r in rows),
              "model_fingerprint": fingerprint, "weights_sha256": file_hash(weights),
              "config_sha256": file_hash(config), "model_implementation_sha256": file_hash(inspect.getfile(type(model))),
              "finite_train_action_array_sha256": stats_hash, "inputs_sha256": file_hash(INPUTS),
              "source_scores_sha256": file_hash(SCORES), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "scope": "exposed RGB-only model-clock diagnostic"}
    with OUT.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps({"maxima": maxima, "nonzero_second_action_responses": result["nonzero_second_action_responses"]}, indent=2))


if __name__ == "__main__":
    main()
