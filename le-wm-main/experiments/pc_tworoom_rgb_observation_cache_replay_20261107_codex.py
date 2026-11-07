"""Paired same-observation model replay with and without per-plan image cache."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_incumbent_rgb_feedback_20261107_codex as run
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

ROOT = run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_OBSERVATION_CACHE_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_rgb_observation_cache_replay_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(run.OUT.read_text())
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("planner_sha256", run.planner.__file__)):
        if source[key] != file_hash(Path(path)):
            raise ValueError(f"source drift {key}")
    common = run.prior.prior.source.block.prior.common
    input_path = run.prior.prior.source.block.prior.inputs.INPUT
    if source["input_sha256"] != file_hash(input_path):
        raise ValueError("input drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != source["model_sha256"] or normalizer != source["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, Path(run.planner.__file__),
                         input_path, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five steps per packet")
    encoder_calls = [0]
    handle = model.encoder.register_forward_hook(lambda *args: encoder_calls.__setitem__(0, encoder_calls[0] + 1))
    rows = []
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        with (np.load(input_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as data, torch.inference_mode()):
            for task in source["rows"]:
                root, row = task["root"], task["row"]
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                for arm, episode in task["arms"].items():
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    cursor, old_tail = 0, None
                    for decision in episode["decisions"]:
                        remaining = 25 - cursor
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}

                        def execute(cached):
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            encoder_before = encoder_calls[0]
                            boundary = (RGBObservationScoringSession if cached else RGBLeWM)(model, audit)

                            def score(actions):
                                norm = scaler.transform(actions.reshape(-1, 2)).reshape(-1, remaining // 5, 10)
                                result = []
                                for offset in range(0, len(actions), 32):
                                    chunk = torch.as_tensor(norm[offset:offset + 32][None], device="cuda",
                                                            dtype=torch.float32)
                                    value, _ = boundary.score(current, chunk)
                                    result.extend(value[0].cpu().tolist())
                                return np.asarray(result)

                            result = run.planner.search(score, remaining=remaining, old_tail=old_tail,
                                                        seed=decision["seed"], arm=arm)
                            torch.cuda.synchronize()
                            return result, {"seconds": time.perf_counter() - started,
                                            "encoder_calls": encoder_calls[0] - encoder_before,
                                            "forward_calls": len(boundary.receipts),
                                            "hits": boundary.cache_hits if cached else 0,
                                            "misses": boundary.cache_misses if cached else 0}

                        order = (False, True) if len(rows) % 2 == 0 else (True, False)
                        trials = {kind: execute(kind) for kind in order}
                        original, o_meta = trials[False]
                        cached, c_meta = trials[True]
                        old_scores = np.asarray(decision["all_scores"])
                        if (original["actions_sha256"] != decision["actions_sha256"]
                                or not np.allclose(original["all_scores"], old_scores, atol=1e-4, rtol=1e-5)):
                            raise ValueError("historical baseline not reproduced")
                        delta = np.abs(cached["all_scores"] - original["all_scores"])
                        equal = (cached["actions_sha256"] == original["actions_sha256"]
                                 and cached["winner"] == original["winner"]
                                 and cached["retained"] == original["retained"])
                        rows.append({"root": root, "arm": arm, "iteration": decision["iteration"],
                                     "order": ["cached" if x else "original" for x in order],
                                     "original": o_meta, "cached": c_meta,
                                     "max_score_abs_error": float(delta.max()),
                                     "scores_within_tolerance": bool(np.allclose(
                                         cached["all_scores"], original["all_scores"], atol=1e-4, rtol=1e-5)),
                                     "decision_equal": bool(equal), "winner_original": original["winner"],
                                     "winner_cached": cached["winner"]})
                        # The original trajectory is followed even if the cached decision differs.
                        n = decision["dispatched"]
                        for i in range(n):
                            _, _, term, trunc, info = actor.step(original["actions"][i].copy())
                            old = episode["trace"][cursor + i]
                            if (bool(term) != old["terminated"] or bool(trunc) != old["truncated"]
                                    or abs(float(info["distance_to_target"]) - old["distance"]) > 1e-6):
                                raise ValueError("historical physical trace mismatch")
                        cursor += n
                        old_tail = original["actions"][n:].copy()
                print(f"root={root} paired decisions completed", flush=True)
    finally:
        handle.remove()
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("frozen weights changed")
    result = {"source_sha256": file_hash(run.OUT), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "cache_sha256": file_hash(Path(
                  __file__).parent / "pc_rgb_observation_cache_20261107_codex.py"),
              "model_sha256": fingerprint, "rows": rows,
              "scope": "same-model same-observation paired development replay"}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
