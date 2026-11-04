"""Independent same-backend replay and explicit model unroll for PushT feedback."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_pusht_rgb_feedback_dev_20261104_codex as run

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, sha = run.HOST, run.HOST.bridge.sha_file
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("GPU UUID required")
    manifest = json.loads(run.MANIFEST.read_text())
    result = json.loads(run.RESULT.read_text())
    run.seal.verify(manifest["seal"])
    if result["manifest_sha256"] != sha(run.MANIFEST):
        raise ValueError("manifest mismatch")
    rows, _ = run.phys.select_rows(run.phys.DEFAULT_DATA, (0,))
    root = rows[0]
    audits = []
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        initial_hash = run.saved.checked.tensor_state_hash(model)
        if initial_hash != result["model_before"] or initial_hash != result["model_after"]:
            raise ValueError("model mismatch")
        transform = run.saved.p0.image_transform()
        for arm, summary in zip(run.ARMS, result["rows"], strict=True):
            execution_file = run.path(arm, "execution.json")
            execution = json.loads(execution_file.read_text())
            if (summary["arm"] != arm or summary["execution_sha256"] != sha(execution_file)
                    or execution["manifest_sha256"] != sha(run.MANIFEST)
                    or execution["status"] != "completed" or len(execution["decisions"]) != 3):
                raise ValueError("execution binding or status")
            event_file = run.path(arm, "events.jsonl")
            if execution["events_sha256"] != sha(event_file):
                raise ValueError("event hash")
            events = [json.loads(line) for line in event_file.read_text().splitlines()]
            if len(events) != 60:
                raise ValueError("unexpected event count")
            env = run.phys.official_env_factory()
            try:
                _, info = env.reset(seed=0, options={"state": root["states"][0].copy(),
                    "goal_state": run.phys.DIAGNOSTIC_GOAL.copy()})
                run.phys._goal_contract(env, info, root, run.phys.official_env_factory)
                frames, packets = [], []
                decisions = []
                for step in range(3):
                    file = run.path(arm, f"decision{step}.json")
                    data = json.loads(file.read_text())
                    record = execution["decisions"][step]
                    if (sha(file) != record["decision_sha256"] or data["manifest_sha256"] != sha(run.MANIFEST)
                            or data["context_sha256"] != sha(run.path(arm, f"context{step}.npz"))
                            or record["sealed_ns"] >= record["executed_ns"]):
                        raise ValueError("decision binding")
                    np.testing.assert_array_equal(data["dispatch"], record["dispatch"])
                    np.testing.assert_array_equal(np.asarray(data["result"]["selected"]["chain"], np.float32)[:5],
                                                  np.asarray(data["dispatch"], np.float32))
                    decisions.append(data)
                commands = np.concatenate([np.asarray(root["actions"][:15], np.float64)] +
                    [np.asarray(d["dispatch"], np.float64) for d in decisions])
                for i, command in enumerate(commands):
                    attempt, returned = events[2*i:2*i+2]
                    if attempt["kind"] != "attempt" or returned["kind"] != "returned":
                        raise ValueError("event sequence")
                    np.testing.assert_array_equal(np.asarray(attempt["command"], np.float64), command)
                    transition = env.step(command)
                    state = run.packets.state_of(env)
                    np.testing.assert_allclose(state, returned["state"], atol=1e-9, rtol=0)
                    if (run.phys._array_hash(run.phys.render_without_advancing(env)) != returned["rgb_sha256"]
                            or bool(transition[2]) != returned["terminated"]
                            or bool(transition[3]) != returned["truncated"]):
                        raise ValueError("physical replay mismatch")
                    if (i + 1) % 5 == 0:
                        frames.append(run.phys.render_without_advancing(env).copy())
                        packets.append(commands[i-4:i+1])
                    if i + 1 >= 15 and (i + 1) % 5 == 0:
                        index = (i + 1 - 15) // 5
                        with np.load(run.path(arm, f"context{index}.npz")) as context:
                            np.testing.assert_array_equal(context["frames"], np.stack(frames[-3:]))
                            np.testing.assert_array_equal(context["history"], np.concatenate(packets[-3:]).astype(np.float32))
                            np.testing.assert_array_equal(context["goal"], info["goal"])
                        np.testing.assert_allclose(state, execution["outcomes"][index]["state"], atol=1e-9, rtol=0)
            finally:
                env.close()
            batch_count = sample_count = total_tokens = 0
            max_position = max_score = 0.0
            tokens_by_decision = []
            with torch.no_grad():
                for step, data in enumerate(decisions):
                    with np.load(run.path(arm, f"context{step}.npz")) as context:
                        images = run.saved.p0._frames_tensor(list(context["frames"]), transform=transform, device=device)
                        emb0 = model.encode({"pixels": images})["emb"]
                        goal = run.saved.p0._goal_latent(model, context["goal"], transform=transform, device=device)
                        hist = run.saved.live.adapter.raw_to_packets(context["history"][None],
                            np.tile(run.saved.p0.PUBLIC_ACTION_MEAN, 5),
                            np.tile(run.saved.p0.PUBLIC_ACTION_STD, 5), device=device)
                    chart = run.saved.live.adapter.goal_chart(emb0, goal, 2027111800 + step)
                    np.testing.assert_allclose(chart.cpu().numpy(), data["chart"], atol=2e-5, rtol=0)
                    scores_all, found, count, tokens = [], False, 0, 0
                    selected = np.asarray(data["result"]["selected"]["chain"])
                    value = data["result"]["selected"]["terminal_score"]
                    horizon = 15 - 5 * step
                    for batch in data["trace"]:
                        requested = np.asarray(batch["requested"])
                        raw = np.asarray(batch["raw"], np.float32)
                        np.testing.assert_array_equal(requested.astype(np.float32), raw)
                        action = torch.tensor(raw, device=device)
                        future = run.packets.normalize_live_packet(action.reshape(-1, 5, 2),
                            mean=run.saved.p0.PUBLIC_ACTION_MEAN, std=run.saved.p0.PUBLIC_ACTION_STD).reshape(len(raw), -1, 10)
                        emb = emb0.expand(len(raw), -1, -1)
                        between = hist[:, 1:, :].expand(len(raw), -1, -1)
                        for t in range(future.shape[1]):
                            outgoing = torch.cat((between, future[:, t:t+1]), dim=1)
                            latent = model.predict(emb, model.action_encoder(outgoing))[:, -1]
                            emb = torch.cat((emb[:, 1:], latent[:, None]), dim=1)
                            between = outgoing[:, 1:]
                        positions = (latent @ chart).cpu().numpy()
                        scores = (latent - goal).square().mean(dim=1).cpu().numpy()
                        max_position = max(max_position, float(np.max(np.abs(positions - batch["positions"]))))
                        max_score = max(max_score, float(np.max(np.abs(scores - batch["scores"]))))
                        if raw.shape[1] == horizon:
                            scores_all.extend(batch["scores"])
                            found |= any(batch["scores"][j] == value for j in np.flatnonzero(
                                np.all(requested == selected, axis=(1, 2))))
                        expected = len(raw) * raw.shape[1] // 5
                        if batch["tokens"] != expected:
                            raise ValueError("batch token count")
                        tokens += expected
                        count += len(raw)
                    if (not found or min(scores_all) != value or tokens > 1920
                            or tokens != data["counts"]["prediction_token_samples"]
                            or count != data["counts"]["trajectory_samples"]
                            or len(data["trace"]) != data["counts"]["prediction_calls"]):
                        raise ValueError("selection or budget mismatch")
                    tokens_by_decision.append(tokens)
                    total_tokens += tokens
                    sample_count += count
                    batch_count += len(data["trace"])
            if max_position > 2e-5 or max_score > 2e-6:
                raise ValueError("model replay mismatch")
            audits.append({"arm": arm, "physical_steps": 30, "decision_count": 3,
                "batches": batch_count, "samples": sample_count, "tokens": tokens_by_decision,
                "max_position_error": max_position, "max_score_error": max_score})
            print(f"{arm}: replayed 30 physical steps and {batch_count} model batches", flush=True)
        if run.saved.checked.tensor_state_hash(model) != initial_hash:
            raise ValueError("model drift")
    run.seal.verify(manifest["seal"])
    host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
        "audit_script_sha256": sha(Path(__file__)), "rows": audits, "cuda_allocation": allocation,
        "scope": "same checkpoint and backend; no training provenance or generalization claim"})


if __name__ == "__main__":
    main()
