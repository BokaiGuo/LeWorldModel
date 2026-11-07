"""Independent identity and numerical replay of the sealed spectral development."""
import json
import sys

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_precise_spectral_development_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_precise_spectral_proposals_20261107_codex import jacobian_and_directions, make_candidates
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest
from experiments.pc_tworoom_search_region_multihorizon_derivative_20261107_codex import normalize_actions

OUTPUT = run.ROOT / "results/pc_tworoom_precise_spectral_development_audit_20261107_codex.json"


def audit_arrays(plan_row, physical_row):
    if plan_row["episode"] != physical_row["episode"]:
        raise ValueError("episode mismatch")
    parents = []
    for arm in ("spectral", "random"):
        p, q = plan_row["arms"][arm], physical_row["arms"][arm]
        a, scores = np.asarray(p["candidates"], np.float32), np.asarray(p["scores"])
        if a.shape != (5, 25, 2) or scores.shape != (5,) or not np.isfinite(scores).all():
            raise ValueError("candidate dimensions invalid")
        if p["selected_index"] != int(np.argmin(scores)) or q["selected_index"] != p["selected_index"]:
            raise ValueError("selected index mismatch")
        reconstructed = make_candidates(a[0], p["directions"])
        np.testing.assert_allclose(a, reconstructed, rtol=0, atol=1e-7)
        if len(q["candidates"]) != 5:
            raise ValueError("missing physical candidates")
        for item in q["candidates"]:
            trace = item["trace"]
            if item["ever_success"] != any(t["success"] for t in trace):
                raise ValueError("physical success mismatch")
            if item["complete"] != (len(trace) == 25):
                raise ValueError("physical duration mismatch")
        parents.append((a[0], q["candidates"][0]["trace"]))
    np.testing.assert_array_equal(parents[0][0], parents[1][0])
    if parents[0][1] != parents[1][1]:
        raise ValueError("shared parent physical traces differ")
    expected = np.random.default_rng(27127000 + plan_row["episode"]).normal(size=(2, 25, 2))
    np.testing.assert_array_equal(expected, np.asarray(plan_row["arms"]["random"]["directions"]))


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(run.PLAN.read_text())
    physical = json.loads(run.PHYSICAL.read_text())
    if physical["plan_sha256"] != file_hash(run.PLAN) or plan["dependencies"] != run.dependencies():
        raise ValueError("artifact drift")
    model, scaler, transform, normalizer = run.source.common.model_setup()
    digest = model_digest(model)
    if digest != plan["model_digest"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, run.PROTOCOL, run.PLAN, run.PHYSICAL, run.source.INPUT,
                         run.source.common.CHECKPOINT / "weights.pt", run.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="five steps per packet")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Audit sealed candidate identities, recompute RGB Jacobians and candidate scores.", flush=True)
    try:
        with np.load(run.source.INPUT, allow_pickle=False) as data, h5py.File(run.source.common.DATA, "r") as db:
            for p, q in zip(plan["rows"], physical["rows"], strict=True):
                audit_arrays(p, q)
                idx, = np.flatnonzero(data["episodes"] == p["episode"])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(db["proprio"][p["row"]])
                actor._set_goal_state(db["proprio"][p["row"] + 25])
                image = actor.render().copy()
                if run.img_hash(image) != p["current_rgb_sha256"] or run.img_hash(data["goal"][idx]) != p["goal_rgb_sha256"]:
                    raise ValueError("RGB mismatch")
                context = {"pixels": transform(image)[None, None, None].cuda(),
                           "goal": transform(data["goal"][idx])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                raw = torch.tensor(p["arms"]["spectral"]["candidates"][0], dtype=torch.float32, device="cuda")
                def rollout(x):
                    _, ctx = guard.score(context, normalize_actions(x, scaler.mean_, scaler.scale_))
                    return ctx["predicted_emb"][0, 0, -1]
                _, singular, directions = jacobian_and_directions(rollout, raw)
                singular_error = float(np.max(np.abs(singular.cpu().numpy() - p["singular_values"])))
                cosine = [float(abs(np.dot(a.flatten(), b.flatten()) /
                                    (np.linalg.norm(a) * np.linalg.norm(b))))
                          for a, b in zip(directions.cpu().numpy(),
                                          np.asarray(p["arms"]["spectral"]["directions"]), strict=True)]
                max_cost_error = 0.
                for arm in ("spectral", "random"):
                    actions = np.asarray(p["arms"][arm]["candidates"], dtype=np.float32)
                    normal = scaler.transform(actions.reshape(-1, 2)).astype(np.float32).reshape(1, 5, 5, 10)
                    with torch.no_grad():
                        costs, _ = guard.score(context, torch.tensor(normal, device="cuda"))
                    max_cost_error = max(max_cost_error, float(np.max(np.abs(costs[0].cpu().numpy() -
                                                                            np.asarray(p["arms"][arm]["scores"])))))
                if singular_error > 1e-3 or min(cosine) < .9999 or max_cost_error > 1e-3:
                    raise ValueError("numerical replay mismatch")
                rows.append({"episode": p["episode"], "singular_max_abs_error": singular_error,
                             "direction_abs_cosines": cosine, "max_score_abs_error": max_cost_error})
                print(f"episode={p['episode']} audit passed", flush=True)
    finally:
        env.close()
    guard.verify_unchanged()
    with OUTPUT.open("x") as f:
        json.dump({"plan_sha256": file_hash(run.PLAN), "physical_sha256": file_hash(run.PHYSICAL),
                   "runner_sha256": file_hash(__file__), "model_digest": digest, "rows": rows}, f,
                  indent=2, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("no arguments")
    main()
