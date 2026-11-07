"""Recompute and audit every endpoint-tie decision on its own physical feedback history."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_rgb_temporal_tiebreak_feedback_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
OUT = ROOT / "results/pc_rgb_temporal_tiebreak_audit_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    archive = json.loads(prior.OUT.read_text())
    if (archive["protocol_sha256"] != file_hash(prior.PROTOCOL)
            or archive["runner_sha256"] != file_hash(prior.__file__)
            or archive["selector_sha256"] != file_hash(prior.new.__file__)):
        raise ValueError("source drift")
    model, scaler, transform, normalizer = prior.source.prior.common.model_setup()
    if model_fingerprint(model) != archive["model_sha256"] or normalizer != archive["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), prior.PROTOCOL, prior.source.prior.INPUT,
                         prior.source.prior.common.CHECKPOINT / "weights.pt",
                         prior.source.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, fixed action normalizer",
                         clock="five environment steps per packet")
    boundary = RGBLeWM(model, audit)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    checks = []
    print("Recompute every tie-rule selection on the actual feedback observation.", flush=True)
    try:
        with np.load(prior.source.prior.INPUT, allow_pickle=False) as inputs, \
                h5py.File(prior.source.prior.common.DATA, "r") as data, torch.inference_mode():
            for index, row in enumerate(archive["rows"]):
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row["row"]])
                actor._set_goal_state(data["proprio"][row["row"] + 25])
                goal = transform(inputs["goal"][index])[None, None, None].cuda()
                tail, cursor = None, 0
                historic = row["arms"]["endpoint_tie"]["original"]
                for iteration, saved in enumerate(historic["decisions"]):
                    remaining = 25-cursor
                    context = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                               "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    args = dict(scaler=scaler, old_tail=tail, candidate_seed=27121000 + row["episode"] * 10 + iteration,
                                strategy_seed=27122000 + row["episode"] * 10 + iteration,
                                strategy="effect", remaining_steps=remaining)
                    base = prior.source.program.compile_rgb(boundary, context, final_rule="temporal", **args)
                    new = prior.new.compile_rgb(boundary, context, **args)
                    if remaining > 5:
                        times = np.concatenate([np.asarray(x) for x in base["temporal_scores"]])
                        ends = np.concatenate([np.ravel(base["parent_cost"]), np.ravel(base["probe_cost"]),
                                               np.ravel(base["tail_cost"])])
                        possible = [i for i, value in enumerate(times) if value <= min(times) + 0.001]
                        best_end = min(ends[i] for i in possible)
                        independent = next(i for i in possible if ends[i] <= best_end + 0.001)
                        if independent != new["tiebreak_index"]:
                            raise ValueError("independent tie ranking mismatch")
                    else:
                        independent = new["choice"].index
                    choice = {"source": new["choice"].source, "index": new["choice"].index}
                    if choice != saved["choice"] or new["model_queries"] != saved["queries"]:
                        raise ValueError("stored selection drift")
                    np.testing.assert_allclose(new["actions"], saved["actions"], atol=1e-6, rtol=0)
                    for action in saved["dispatched"]:
                        _, _, term, trunc, info = actor.step(np.asarray(action, np.float32))
                        actual = historic["trace"][cursor]
                        if (abs(float(info["distance_to_target"]) - actual["distance"]) > 1e-5
                                or bool(term) != actual["terminated"] or bool(trunc) != actual["truncated"]):
                            raise ValueError("physical trace mismatch")
                        cursor += 1
                    tail = np.asarray(saved["actions"], np.float32)[len(saved["dispatched"]):]
                    checks.append({"episode": row["episode"], "iteration": iteration,
                                   "remaining": remaining, "independent_index": independent})
                if cursor != len(historic["trace"]):
                    raise ValueError("missing physical steps")
    finally:
        env.close()
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(prior.OUT), "runner_sha256": file_hash(__file__),
                   "checks": checks}, stream, indent=2, allow_nan=False)
    print(f"audited {len(checks)} feedback decisions", flush=True)


if __name__ == "__main__":
    main()
