"""Post-hoc model versus actual encoded RGB for six dispatched PushT packets."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_pusht_rgb_feedback_dev_20261104_codex as run

ROOT = run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_PUSHT_EXECUTED_PACKET_PREDICTION_DIAG_PROTOCOL_20261104_CODEX.md"
AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
OUT = ROOT / "results/pc_v4_pusht_executed_packet_diag_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, sha = run.HOST, run.HOST.bridge.sha_file
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    manifest, result, audit = (json.loads(p.read_text()) for p in (run.MANIFEST, run.RESULT, AUDIT))
    run.seal.verify(manifest["seal"])
    if (result["manifest_sha256"] != sha(run.MANIFEST)
            or audit["manifest_sha256"] != sha(run.MANIFEST)
            or audit["result_sha256"] != sha(run.RESULT)):
        raise ValueError("upstream binding mismatch")
    files = [Path(__file__), PROTOCOL, run.MANIFEST, run.RESULT, AUDIT]
    rows = []
    for arm, summary, audit_row in zip(run.ARMS, result["rows"], audit["rows"], strict=True):
        execution_path = run.path(arm, "execution.json")
        events_path = run.path(arm, "events.jsonl")
        execution = json.loads(execution_path.read_text())
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        if (summary["arm"] != arm or audit_row["arm"] != arm or audit_row["physical_steps"] != 30
                or summary["execution_sha256"] != sha(execution_path)
                or execution["events_sha256"] != sha(events_path)
                or execution["status"] != "completed" or len(events) != 60):
            raise ValueError("upstream execution mismatch")
        files.extend((execution_path, events_path))
        decisions, contexts = [], []
        for index in range(4):
            context_path = run.path(arm, f"context{index}.npz")
            files.append(context_path)
            with np.load(context_path, allow_pickle=False) as ctx:
                context = {key: ctx[key].copy() for key in ("frames", "history", "goal")}
            if run.phys._array_hash(context["frames"][-1]) != events[2*(15+5*index)-1]["rgb_sha256"]:
                raise ValueError("context boundary frame mismatch")
            logged_history = np.asarray([events[2*j]["command"] for j in range(5*index, 15+5*index)], np.float32)
            np.testing.assert_array_equal(context["history"], logged_history)
            contexts.append(context)
            if index < 3:
                decision_path = run.path(arm, f"decision{index}.json")
                decision = json.loads(decision_path.read_text())
                if (sha(decision_path) != execution["decisions"][index]["decision_sha256"]
                        or decision["context_sha256"] != sha(context_path)
                        or decision["manifest_sha256"] != sha(run.MANIFEST)):
                    raise ValueError("decision binding mismatch")
                logged = [events[2*j]["command"] for j in range(15+5*index, 20+5*index)]
                np.testing.assert_array_equal(np.asarray(decision["dispatch"], np.float32), np.asarray(logged, np.float32))
                decisions.append(decision)
                files.append(decision_path)
        rows.append((arm, execution, contexts, decisions))
    deps = run.seal.capture(files)
    print("Comparing six executed packet predictions with observed RGB encodings.", flush=True)
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        before = run.saved.checked.tensor_state_hash(model)
        if before != result["model_before"] or before != result["model_after"]:
            raise ValueError("model hash mismatch")
        transform = run.saved.p0.image_transform()
        mean = np.tile(np.asarray(run.saved.p0.PUBLIC_ACTION_MEAN, np.float32), 5)
        std = np.tile(np.asarray(run.saved.p0.PUBLIC_ACTION_STD, np.float32), 5)
        lower, upper = (torch.as_tensor(x, device=device) for x in ((-1-mean)/std, (1-mean)/std))
        measurements = []
        with torch.no_grad():
            for arm, execution, contexts, decisions in rows:
                for step in range(3):
                    ctx, nxt = contexts[step], contexts[step+1]
                    np.testing.assert_array_equal(ctx["goal"], nxt["goal"])
                    frames = run.saved.p0._frames_tensor(list(ctx["frames"]), transform=transform, device=device)
                    next_frames = run.saved.p0._frames_tensor(list(nxt["frames"]), transform=transform, device=device)
                    encoded = model.encode({"pixels": frames})["emb"]
                    actual = model.encode({"pixels": next_frames})["emb"][0, -1]
                    current = encoded[0, -1]
                    goal = run.saved.p0._goal_latent(model, ctx["goal"], transform=transform, device=device)
                    history = run.saved.live.adapter.raw_to_packets(ctx["history"][None], mean, std, device=device)
                    future = run.saved.live.adapter.raw_to_packets(
                        np.asarray(decisions[step]["dispatch"], np.float32)[None], mean, std, device=device)
                    predicted = run.saved.live.adapter.cached_terminal(model, encoded, history, future, lower, upper)[0]

                    def mse(a, b):
                        return float((a-b).square().mean().cpu())

                    def norm(a):
                        return float(torch.linalg.vector_norm(a).cpu())

                    cur_goal, pred_goal, actual_goal = mse(current, goal), mse(predicted, goal), mse(actual, goal)
                    change, predicted_change = actual-current, predicted-current
                    denom = norm(change) * norm(predicted_change)
                    cosine = float(torch.dot(change, predicted_change).cpu()) / denom if denom > 1e-12 else None
                    old, new = execution["outcomes"][step], execution["outcomes"][step+1]
                    measurements.append({"arm": arm, "step": step,
                        "prediction_actual_mse": mse(predicted, actual),
                        "persistence_actual_mse": mse(current, actual),
                        "prediction_actual_l2": norm(predicted-actual),
                        "prediction_actual_max_abs": float((predicted-actual).abs().max().cpu()),
                        "actual_displacement_l2": norm(change), "predicted_displacement_l2": norm(predicted_change),
                        "displacement_cosine": cosine, "goal_mse_current": cur_goal,
                        "goal_mse_predicted_next": pred_goal, "goal_mse_actual_next": actual_goal,
                        "predicted_goal_mse_change": pred_goal-cur_goal, "actual_goal_mse_change": actual_goal-cur_goal,
                        "object_xy_change": new["pose"]["xy_distance"]-old["pose"]["xy_distance"],
                        "object_angle_change": new["pose"]["angle_distance"]-old["pose"]["angle_distance"],
                        "official_state_distance_change": new["official"]["official_state_distance"]-old["official"]["official_state_distance"]})
        if run.saved.checked.tensor_state_hash(model) != before:
            raise ValueError("model drift")
    run.seal.verify(deps)
    host.write_new(OUT, {"protocol_sha256": sha(PROTOCOL), "dependencies": deps, "model_hash": before,
        "cuda_allocation": allocation, "measurements": measurements,
        "scope": "post-hoc executed packet only; same checkpoint and exposed root; no causal attribution"})
    print(f"Saved {len(measurements)} observed packet comparisons.", flush=True)


if __name__ == "__main__":
    main()
