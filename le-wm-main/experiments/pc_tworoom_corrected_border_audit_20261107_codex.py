"""Rescore and physically replay the controlled border-corrected experiment."""
import json

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash, tensor_receipt
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_corrected_border_replay_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_tworoom_corrected_border_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest, score, phys, result = (json.loads(p.read_text()) for p in (run.MANIFEST, run.SCORE, run.PHYS, run.RESULT))
    if (manifest["input_sha256"] != file_hash(run.INPUT) or phys["frames_sha256"] != file_hash(run.FRAMES)
            or phys["score_sha256"] != file_hash(run.SCORE) or result["physical_sha256"] != file_hash(run.PHYS)
            or manifest["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("source drift")
    model, scaler, transform, normalizer = run.base.model_setup()
    if model_fingerprint(model) != score["model_sha256"] or normalizer != score["normalizer_action_sha256"]:
        raise ValueError("model mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    errors = []
    try:
        with np.load(run.INPUT, allow_pickle=False) as images, np.load(run.FRAMES, allow_pickle=False) as frames, \
                np.load(run.base.INPUT, allow_pickle=False) as old, torch.inference_mode():
            np.testing.assert_array_equal(images["goal"], old[f"s{run.SEED}_goal"])
            env.reset(seed=run.SEED, options={"state": np.asarray(manifest["corrected"], np.float32)})
            before = env.unwrapped.agent_position.numpy().copy()
            env.unwrapped.step(np.zeros(2, np.float32))
            np.testing.assert_array_equal(env.unwrapped.agent_position.numpy(), before)
            for row in phys["rows"]:
                i, j = row["prefix"], row["tail"]
                env.reset(seed=run.SEED, options={"state": np.asarray(manifest["corrected"], np.float32)})
                np.testing.assert_array_equal(env.unwrapped.render(), images["current"])
                for action, saved in zip([run.base.ACTIONS[i]] * 5 + [run.base.ACTIONS[j]] * 5,
                                         row["trace"], strict=True):
                    _, _, terminated, truncated, info = env.unwrapped.step(action.copy())
                    errors.append(abs(float(info["distance_to_target"]) - saved["distance"]))
                    if (bool(terminated), bool(truncated)) != (saved["terminated"], saved["truncated"]):
                        raise ValueError("flags differ")
                np.testing.assert_array_equal(env.unwrapped.render(), frames[f"{i}_{j}"])
            pixel = transform(images["current"])[None, None, None].cuda()
            goal = transform(images["goal"])[None, None, None].cuda()
            chains = np.stack([np.stack([np.tile(a, (5, 1)), np.tile(b, (5, 1))])
                               for a in run.base.ACTIONS for b in run.base.ACTIONS])
            actions = torch.tensor(scaler.transform(chains.reshape(-1, 2)).reshape(1, 25, 2, 10),
                                   dtype=torch.float32, device="cuda")
            if (tensor_receipt(pixel) != score["receipt"]["inputs"]["pixels"]
                    or tensor_receipt(goal) != score["receipt"]["inputs"]["goal"]
                    or tensor_receipt(actions) != score["receipt"]["inputs"]["candidate_actions"]):
                raise ValueError("model receipt mismatch")
            context = {"pixels": pixel, "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
            forecast = model.get_cost(context, actions).cpu().numpy().reshape(5, 5)
            np.testing.assert_allclose(forecast, score["cost"], atol=1e-4, rtol=0)
            target = model.encode({"pixels": transform(images["goal"])[None, None].cuda()})["emb"][0, 0]
            batch = torch.stack([transform(frames[f"{i}_{j}"]) for i in range(5) for j in range(5)]).cuda()
            actual = model.encode({"pixels": batch[:, None]})["emb"][:, 0]
            encoded = (actual - target).square().sum(-1).cpu().numpy().reshape(5, 5)
            distance = np.asarray([r["trace"][-1]["distance"] for r in phys["rows"]]).reshape(5, 5)
            effect = lambda x: x[:, :1] - x[:, 1:]
            for key, x, y in (("predicted_vs_encoded", forecast, encoded),
                              ("predicted_vs_distance", forecast, distance),
                              ("encoded_vs_distance", encoded, distance)):
                if run.base.pair_agreement(effect(x), effect(y)) != result[key]:
                    raise ValueError("effect mismatch")
    finally:
        env.close()
    if len(errors) != 250 or max(errors) > 1e-6:
        raise ValueError("physical replay mismatch")
    with OUTPUT.open("x") as stream:
        json.dump({"physical_steps": len(errors), "max_distance_error": max(errors),
                   "result_sha256": file_hash(run.RESULT), "source_sha256": file_hash(__file__)}, stream, indent=2)
    print(f"audited {len(errors)} physical steps and 25 GPU predictions")


if __name__ == "__main__":
    audit()
