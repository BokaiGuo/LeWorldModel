"""Controlled border-corrected replay of one sealed TwoRoom root."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_tworoom_reset_boundary_contract_20261107_codex import border_contract, require_legal_border
from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as base

ROOT = base.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_CORRECTED_BORDER_REPLAY_PROTOCOL_20261107_CODEX.md"
INPUT = ROOT / "results/pc_tworoom_corrected_border_input_20261107_codex.npz"
MANIFEST = ROOT / "results/pc_tworoom_corrected_border_manifest_20261107_codex.json"
SCORE = ROOT / "results/pc_tworoom_corrected_border_score_20261107_codex.json"
FRAMES = ROOT / "results/pc_tworoom_corrected_border_frames_20261107_codex.npz"
PHYS = ROOT / "results/pc_tworoom_corrected_border_physical_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_corrected_border_result_20261107_codex.json"
SEED = 2717103


def prepare():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if INPUT.exists() or MANIFEST.exists():
        raise FileExistsError("prepared output exists")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        env.reset(seed=SEED)
        actor = env.unwrapped
        initial = actor.agent_position.numpy().copy()
        radius = float(actor.variation_space["agent"]["radius"].value.item())
        info = border_contract(initial, actor.BORDER_SIZE, radius, actor.IMG_SIZE)
        if info["valid"]:
            raise ValueError("expected original out-of-border start")
        corrected = np.clip(initial, info["lower"], info["upper"])
        env.reset(seed=SEED, options={"state": corrected})
        np.testing.assert_array_equal(actor.agent_position.numpy(), corrected)
        require_legal_border(corrected, actor.BORDER_SIZE, radius, actor.IMG_SIZE)
        current, goal = actor.render().copy(), base.image_goal(actor)
        with np.load(base.INPUT, allow_pickle=False) as old:
            np.testing.assert_array_equal(goal, old[f"s{SEED}_goal"])
    finally:
        env.close()
    with INPUT.open("xb") as stream:
        np.savez_compressed(stream, current=current, goal=goal)
    with MANIFEST.open("x") as stream:
        json.dump({"seed": SEED, "original": initial.tolist(), "corrected": corrected.tolist(),
                   "input_sha256": file_hash(INPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__)}, stream, indent=2)


def score():
    if SCORE.exists():
        raise FileExistsError(SCORE)
    model, scaler, transform, normalizer_hash = base.model_setup()
    model_hash = model_fingerprint(model)
    prior = json.loads(base.PRED.read_text())
    if prior["model_sha256"] != model_hash or prior["normalizer_action_sha256"] != normalizer_hash:
        raise ValueError("model or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, INPUT,
                         base.CHECKPOINT / "weights.pt", base.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom dataset action scaler",
                         clock="two 5-step action packets")
    chains = np.stack([np.stack([np.tile(a, (5, 1)), np.tile(b, (5, 1))])
                       for a in base.ACTIONS for b in base.ACTIONS])
    normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, 25, 2, 10)
    with np.load(INPUT, allow_pickle=False) as images, torch.inference_mode():
        wrapper = RGBLeWM(model, audit)
        pixels = transform(images["current"])[None, None, None].cuda()
        goal = transform(images["goal"])[None, None, None].cuda()
        values, _ = wrapper.score({"pixels": pixels, "goal": goal,
                                   "action": torch.empty((1, 1, 0, 10), device="cuda")},
                                  torch.tensor(normal, dtype=torch.float32, device="cuda"))
    with SCORE.open("x") as stream:
        json.dump({"cost": values[0].cpu().reshape(5, 5).tolist(), "receipt": wrapper.receipts[0],
                   "input_sha256": file_hash(INPUT), "model_sha256": model_hash,
                   "normalizer_action_sha256": normalizer_hash, "runner_sha256": file_hash(__file__)}, stream)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYS.exists() or FRAMES.exists():
        raise FileExistsError("physical output exists")
    manifest, scored = json.loads(MANIFEST.read_text()), json.loads(SCORE.read_text())
    if (manifest["input_sha256"] != file_hash(INPUT) or scored["input_sha256"] != file_hash(INPUT)
            or manifest["runner_sha256"] != file_hash(__file__)):
        raise ValueError("source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    frames, rows = {}, []
    try:
        with np.load(INPUT, allow_pickle=False) as images:
            for i, a in enumerate(base.ACTIONS):
                for j, b in enumerate(base.ACTIONS):
                    env.reset(seed=SEED, options={"state": np.asarray(manifest["corrected"], np.float32)})
                    actor = env.unwrapped
                    radius = float(actor.variation_space["agent"]["radius"].value.item())
                    require_legal_border(actor.agent_position.numpy(), actor.BORDER_SIZE, radius, actor.IMG_SIZE)
                    np.testing.assert_array_equal(actor.render(), images["current"])
                    np.testing.assert_array_equal(base.image_goal(actor), images["goal"])
                    trace = []
                    for action in [a] * 5 + [b] * 5:
                        _, _, terminated, truncated, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(terminated), "truncated": bool(truncated)})
                    frames[f"{i}_{j}"] = actor.render().copy()
                    rows.append({"prefix": i, "tail": j, "trace": trace})
    finally:
        env.close()
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, **frames)
    with PHYS.open("x") as stream:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES), "score_sha256": file_hash(SCORE),
                   "manifest_sha256": file_hash(MANIFEST), "runner_sha256": file_hash(__file__)}, stream)


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    physical, score_data = json.loads(PHYS.read_text()), json.loads(SCORE.read_text())
    if physical["frames_sha256"] != file_hash(FRAMES) or physical["score_sha256"] != file_hash(SCORE):
        raise ValueError("source drift")
    model, _, transform, normalizer_hash = base.model_setup()
    if model_fingerprint(model) != score_data["model_sha256"] or normalizer_hash != score_data["normalizer_action_sha256"]:
        raise ValueError("model drift")
    with np.load(INPUT, allow_pickle=False) as images, np.load(FRAMES, allow_pickle=False) as frames, torch.inference_mode():
        goal = model.encode({"pixels": transform(images["goal"])[None, None].cuda()})["emb"][0, 0]
        batch = torch.stack([transform(frames[f"{i}_{j}"]) for i in range(5) for j in range(5)]).cuda()
        actual = model.encode({"pixels": batch[:, None]})["emb"][:, 0]
        encoded = (actual - goal).square().sum(-1).cpu().numpy().reshape(5, 5)
    pred = np.asarray(score_data["cost"])
    distance = np.asarray([r["trace"][-1]["distance"] for r in physical["rows"]]).reshape(5, 5)
    def effects(matrix):
        return matrix[:, :1] - matrix[:, 1:]
    result = {"predicted_vs_encoded": base.pair_agreement(effects(pred), effects(encoded)),
              "predicted_vs_distance": base.pair_agreement(effects(pred), effects(distance)),
              "encoded_vs_distance": base.pair_agreement(effects(encoded), effects(distance)),
              "predicted_vs_distance_ranking": base.ranking_agreement(pred, distance),
              "ever_success": sum(any(s["terminated"] for s in r["trace"]) for r in physical["rows"]),
              "physical_sha256": file_hash(PHYS), "score_sha256": file_hash(SCORE),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "scope": "single corrected initial state, not original official protocol"}
    with RESULT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "score", "physical", "analyze"):
        raise SystemExit("expected prepare|score|physical|analyze")
    {"prepare": prepare, "score": score, "physical": physical, "analyze": analyze}[sys.argv[1]]()
