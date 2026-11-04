"""Post-hoc same-state physical replay of five sealed final-packet actions."""
import json
import os
from pathlib import Path

import numpy as np

from experiments import pc_v4_reacher_rgb_live_20261104_codex as v4
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

live = v4.live
ROOT = v4.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_FINAL_PACKET_COUNTERFACTUAL_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_final_packet_counterfactual_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
DECISION = v4.PREFIX.with_name(v4.PREFIX.name + "_response_decision2.json")
CONTEXT0 = v4.PREFIX.with_name(v4.PREFIX.name + "_response_context0.npz")
CONTEXT2 = v4.PREFIX.with_name(v4.PREFIX.name + "_response_context2.npz")
EXEC = v4.PREFIX.with_name(v4.PREFIX.name + "_response_execution.json")
AUDIT = v4.PREFIX.with_name(v4.PREFIX.name + "_audit.json")
MODEL_REPLAY = v4.PREFIX.with_name(v4.PREFIX.name + "_prediction_replay.json")
PHYSICAL_REPLAY = v4.PREFIX.with_name(v4.PREFIX.name + "_physical_replay.json")


def physical(env):
    row = live.physical(env)
    row["qvel"] = np.asarray(env.unwrapped.env.physics.data.qvel).copy().tolist()
    return row


def actions_and_scores(decision):
    result = decision["result"]
    actions = [("selected", result["selected"]["chain"]), ("carry", decision["carry_in"]),
               ("zero", np.zeros((5, 2)).tolist())]
    actions += [(f"nominal{i}", row["nominal"]) for i, row in enumerate(result["first_response"])]
    output = []
    for name, action in actions:
        raw = np.asarray(action, np.float32)
        if raw.shape != (5, 2) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1):
            raise ValueError("invalid action")
        scores = [float(batch["scores"][i]) for batch in decision["trace"]
                  for i, candidate in enumerate(batch["raw"])
                  if np.array_equal(np.asarray(candidate, np.float32), raw)]
        if not scores or max(scores) - min(scores) > 2e-6:
            raise ValueError("missing or inconsistent sealed model scores")
        output.append({"name": name, "raw": raw.tolist(), "model_scores": scores})
    return output


def main():
    if MANIFEST.exists() or RESULT.exists():
        raise FileExistsError("existing or partial result")
    v4.configure()
    host = live.host
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    manifest = json.loads(v4.MANIFEST.read_text())
    source = json.loads(live.run.SOURCE.read_text())
    if manifest["dependencies"] != live.dependencies(source):
        raise ValueError("live source dependency mismatch")
    for p in (AUDIT, MODEL_REPLAY, PHYSICAL_REPLAY):
        if json.loads(p.read_text())["manifest_sha256"] != sha(v4.MANIFEST):
            raise ValueError("upstream audit binding mismatch")
    decision, execution = json.loads(DECISION.read_text()), json.loads(EXEC.read_text())
    if (decision["context_sha256"] != sha(CONTEXT2) or execution["manifest_sha256"] != sha(v4.MANIFEST)
            or execution["rows"][2]["decision_sha256"] != sha(DECISION)):
        raise ValueError("decision or context binding mismatch")
    candidates = actions_and_scores(decision)
    if candidates[0]["raw"] != execution["rows"][2]["dispatch"]:
        raise ValueError("selected/dispatch mismatch")
    dependencies = seal.capture([Path(__file__), PROTOCOL, v4.MANIFEST, DECISION, CONTEXT0,
                                 CONTEXT2, EXEC, AUDIT, MODEL_REPLAY, PHYSICAL_REPLAY])
    host.write_new(MANIFEST, {"seal": dependencies, "scope": "post-hoc physical candidate diagnostic"})
    print("Replay five saved alternatives from the same reconstructed physical state.", flush=True)
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        live.setup.install_missing_field_compat()
        env = live.setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                                 target_delta=(.65, -.65))
        try:
            with np.load(CONTEXT0, allow_pickle=False) as context:
                warmup = context["history"].reshape(-1, 2).astype(np.float32)
                goal_sha = live.context_lib.rgb_digest(context["goal"])
            with np.load(CONTEXT2, allow_pickle=False) as context:
                root_sha = live.context_lib.rgb_digest(context["root"])
            before_actions = [row["dispatch"] for row in execution["rows"][:2]]
            initial, rows = None, []

            def execute(actions):
                result = []
                for action in actions:
                    _, _, terminated, truncated, _ = env.step(np.asarray(action, np.float32))
                    if terminated or truncated:
                        raise ValueError("unexpected termination")
                    result.append(physical(env))
                return result

            for candidate in candidates:
                _, info = env.reset(seed=20263160)
                if live.context_lib.rgb_digest(info["goal"]) != goal_sha:
                    raise ValueError("goal mismatch")
                execute(warmup)
                for packet in before_actions:
                    execute(packet)
                if live.context_lib.rgb_digest(np.asarray(env.render(), dtype=np.uint8)) != root_sha:
                    raise ValueError("step-10 RGB mismatch")
                before = physical(env)
                expected = execution["initial_and_block_outcomes"][2]
                for field in ("qpos", "target_qpos"):
                    np.testing.assert_allclose(before[field], expected[field], rtol=0, atol=1e-8)
                if initial is None:
                    initial = before
                for field in ("qpos", "qvel", "target_qpos"):
                    np.testing.assert_allclose(before[field], initial[field], rtol=0, atol=1e-10)
                trajectory = execute(candidate["raw"])
                if candidate["name"] == "selected":
                    expected = execution["initial_and_block_outcomes"][3]
                    for field in ("qpos", "target_qpos"):
                        np.testing.assert_allclose(trajectory[-1][field], expected[field], rtol=0, atol=1e-8)
                rows.append({**candidate, "before": before, "trajectory": trajectory})
                print(f"{candidate['name']}: final distance={trajectory[-1]['distance']:.6f}", flush=True)
        finally:
            env.close()
    seal.verify(dependencies)
    host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": rows,
                            "scope": "same-backend counterfactual, physical state not used by planner"})


if __name__ == "__main__":
    main()
