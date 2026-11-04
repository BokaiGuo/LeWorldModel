"""Replay actual v7 third-round candidate bank on one exposed PushT boundary."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v7_pusht_rgb_feedback_dev_20261104_codex as configured
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = configured.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_PUSHT_SAME_BANK_PHYSICAL_RANK_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_pusht_same_bank_physical_rank_dev_20261104_codex"
MANIFEST, RESULT, ARRAYS = [PREFIX.with_name(PREFIX.name + x) for x in ("_manifest.json", "_result.json", "_arrays.npz")]


def main():
    if any(path.exists() for path in (MANIFEST, RESULT, ARRAYS)):
        raise FileExistsError("existing or partial diagnostic")
    configured.configure()
    run, host = configured.live, configured.live.HOST
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("GPU UUID required")
    sha = host.bridge.sha_file
    audit = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    decision_path = run.path("response", "decision2.json")
    execution_path = run.path("response", "execution.json")
    context_path = run.path("response", "context2.npz")
    audit_record, decision, execution = (json.loads(path.read_text()) for path in (audit, decision_path, execution_path))
    if (audit_record["manifest_sha256"] != sha(run.MANIFEST) or audit_record["result_sha256"] != sha(run.RESULT)
            or decision["context_sha256"] != sha(context_path)
            or execution["decisions"][2]["decision_sha256"] != sha(decision_path)):
        raise ValueError("v7 source audit mismatch")
    seal.verify(json.loads(run.MANIFEST.read_text())["seal"])
    result = decision["result"]
    bank = result["first_bank"]
    if len(bank) != len(result["prefix_scores"]):
        raise ValueError("bank and score length mismatch")
    candidates = [{"name": f"bank/{row['name']}", "actions": row["prefix"], "score": score}
                  for row, score in zip(bank, result["prefix_scores"], strict=True)]
    candidates.extend([
        {"name": "selected", "actions": result["selected"]["chain"], "score": result["selected"]["terminal_score"]},
        {"name": "carry", "actions": decision["carry_in"], "score": result["carry_score"]},
    ])
    if any(np.asarray(c["actions"]).shape != (5, 2) for c in candidates):
        raise ValueError("unexpected five-step candidate shape")
    deps = seal.capture([Path(__file__), PROTOCOL, run.MANIFEST, run.RESULT, audit, decision_path,
                         execution_path, context_path, *map(Path, json.loads(run.MANIFEST.read_text())["seal"]["files"])])
    host.write_new(MANIFEST, {"seal": deps, "candidate_names": [c["name"] for c in candidates],
                              "scope": "post-hoc candidate interventions; exposed root"})
    print(f"Replaying {len(candidates)} v7 bank, carry and selected packets from the same physical boundary.", flush=True)
    dataset, _ = run.phys.select_rows(run.phys.DEFAULT_DATA, (0,))
    root = dataset[0]
    with np.load(context_path, allow_pickle=False) as ctx:
        expected_frames, expected_history, expected_goal = (ctx[x].copy() for x in ("frames", "history", "goal"))
    rows, frames = [], []
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        for candidate in candidates:
            env = run.phys.official_env_factory()
            try:
                _, info = env.reset(seed=0, options={"state": root["states"][0].copy(),
                                                     "goal_state": run.phys.DIAGNOSTIC_GOAL.copy()})
                run.phys._goal_contract(env, info, root, run.phys.official_env_factory)
                np.testing.assert_array_equal(info["goal"], expected_goal)
                with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as journal:
                    guarded = run.Guarded(env, run.phys._array_hash(root["states"][0]), journal)
                    history = run.packets.BoundaryHistory(guarded, root_state=root["states"][0],
                                                         goal=run.phys.DIAGNOSTIC_GOAL)
                    for packet in root["actions"][:15].reshape(3, 5, 2):
                        history.execute(packet)
                    for prior in execution["decisions"][:2]:
                        history.execute(prior["dispatch"])
                    np.testing.assert_array_equal(np.stack(history.frames[-3:]), expected_frames)
                    np.testing.assert_array_equal(history.packets[-3:].reshape(15, 2).astype(np.float32), expected_history)
                    np.testing.assert_allclose(run.packets.state_of(env), execution["outcomes"][2]["state"], atol=1e-9, rtol=0)
                    try:
                        history.execute(np.asarray(candidate["actions"], dtype=np.float32))
                    except Exception as exc:
                        rows.append({"name": candidate["name"], "actions": candidate["actions"],
                                     "model_score": candidate["score"], "status": "stopped",
                                     "failure": {"type": type(exc).__name__, "message": str(exc)},
                                     "guard": guarded.guard.receipt()})
                        frames.append(np.zeros_like(expected_goal))
                    else:
                        state = run.packets.state_of(env)
                        frame = run.phys.render_without_advancing(env).copy()
                        rows.append({"name": candidate["name"], "actions": candidate["actions"],
                                     "model_score": candidate["score"], "status": "completed",
                                     "state": state.tolist(), "official": run.metrics.official_checked(env, state, run.phys.DIAGNOSTIC_GOAL),
                                     "pose": run.pose._pose_label(state, run.phys.DIAGNOSTIC_GOAL),
                                     "guard": guarded.guard.receipt()})
                        frames.append(frame)
                    journal.seek(0)
                    rows[-1]["events"] = [json.loads(line) for line in journal]
            finally:
                env.close()
        model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        model_hash = run.saved.checked.tensor_state_hash(model)
        original_result = json.loads(run.RESULT.read_text())
        if model_hash != original_result["model_before"] or model_hash != original_result["model_after"]:
            raise ValueError("frozen model mismatch")
        device, transform = torch.device("cuda:0"), run.saved.p0.image_transform()
        with torch.no_grad():
            goal = run.saved.p0._goal_latent(model, expected_goal, transform=transform, device=device)
            for row, frame in zip(rows, frames, strict=True):
                if row["status"] == "completed":
                    actual = run.saved.p0._goal_latent(model, frame, transform=transform, device=device)
                    row["observed_goal_latent_mse"] = float((actual-goal).square().mean().cpu())
            torch.cuda.synchronize()
        if run.saved.checked.tensor_state_hash(model) != model_hash:
            raise ValueError("model drift")
    seal.verify(deps)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, frames=np.stack(frames))
        stream.flush()
        os.fsync(stream.fileno())
    completed = [row for row in rows if row["status"] == "completed"]
    comparisons = {}
    for label, getter in (("observed_goal_latent_mse", lambda r: r["observed_goal_latent_mse"]),
                          ("object_xy_distance", lambda r: r["pose"]["xy_distance"]),
                          ("object_angle_distance", lambda r: r["pose"]["angle_distance"])):
        pairs = [(a, b) for i, a in enumerate(completed) for b in completed[i+1:]
                 if a["model_score"] != b["model_score"] and getter(a) != getter(b)]
        agree = sum(np.sign(a["model_score"]-b["model_score"]) == np.sign(getter(a)-getter(b)) for a, b in pairs)
        comparisons[label] = {"valid_pairs": len(pairs), "agree": int(agree),
                              "fraction": agree/len(pairs) if pairs else None}
    host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "arrays_sha256": sha(ARRAYS),
        "rows": rows, "ordering": comparisons, "cuda_allocation": allocation, "model_hash": model_hash,
        "scope": "post-hoc, same root and backend; physical metrics not used for selection"})
    print(json.dumps({"completed": len(completed), "ordering": comparisons}), flush=True)


if __name__ == "__main__":
    main()
