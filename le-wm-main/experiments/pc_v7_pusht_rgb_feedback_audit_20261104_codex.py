"""Replay v7 PushT events and independently unroll saved model batches."""

import json
import os

import numpy as np
import torch

from experiments import pc_v7_pusht_rgb_feedback_dev_20261104_codex as configured


def main():
    configured.configure()
    run, host = configured.live, configured.live.HOST
    out = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    if out.exists():
        raise FileExistsError(out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    manifest, result = json.loads(run.MANIFEST.read_text()), json.loads(run.RESULT.read_text())
    run.seal.verify(manifest["seal"])
    if result["manifest_sha256"] != sha(run.MANIFEST):
        raise ValueError("manifest binding mismatch")
    receipt_path = run.path("response", "execution.json")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != "completed" or len(receipt["decisions"]) != 3
            or result["rows"][0]["execution_sha256"] != sha(receipt_path)
            or receipt["manifest_sha256"] != sha(run.MANIFEST)):
        raise ValueError("incomplete or mismatched execution")
    events_path = run.path("response", "events.jsonl")
    if receipt["events_sha256"] != sha(events_path):
        raise ValueError("event binding mismatch")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if len(events) != 60:
        raise ValueError("incorrect event count")
    rows, _ = run.phys.select_rows(run.phys.DEFAULT_DATA, (0,))
    root = rows[0]
    decisions, carry, tokens_by_step = [], None, []
    for step, entry in enumerate(receipt["decisions"]):
        decision = run.path("response", f"decision{step}.json")
        context = run.path("response", f"context{step}.npz")
        data = json.loads(decision.read_text())
        if (sha(decision) != entry["decision_sha256"] or data["context_sha256"] != sha(context)
                or data["manifest_sha256"] != sha(run.MANIFEST) or data["carry_in"] != carry
                or entry["sealed_ns"] >= entry["executed_ns"]):
            raise ValueError("decision, carry or execution order mismatch")
        selected = np.asarray(data["result"]["selected"]["chain"])
        if selected.shape != (15-5*step, 2):
            raise ValueError("horizon mismatch")
        np.testing.assert_array_equal(selected[:5].astype(np.float32), np.asarray(data["dispatch"], np.float32))
        np.testing.assert_array_equal(data["dispatch"], entry["dispatch"])
        result_row = data["result"]
        pair = data["trace"][-1]
        request, raw, scores = np.asarray(pair["requested"]), np.asarray(pair["raw"], np.float32), pair["scores"]
        np.testing.assert_array_equal(request, np.stack((result_row["precomparison"]["chain"], result_row["comparison_chain"])))
        same = np.array_equal(raw[0], raw[1])
        abstain = bool(same or scores[1] <= scores[0])
        selected_index = int(abstain)
        comparison = result_row["paired_comparison"]
        if (result_row["abstained"] != abstain or comparison["same_actual_float32_actions"] != same
                or comparison["incumbent_score"] != scores[0] or comparison["zero_prefix_score"] != scores[1]
                or comparison["reserved_token_samples"] != 2*(3-step)
                or result_row["selected"]["terminal_score"] != scores[selected_index]
                or not np.array_equal(selected, request[selected_index])):
            raise ValueError("paired abstention mismatch")
        if step and (data["trace"][0]["requested"] != [carry] or result_row["carry_score"] != data["trace"][0]["scores"][0]):
            raise ValueError("carry rescore mismatch")
        carry = selected[5:].astype(np.float32).tolist() if step < 2 else None
        decisions.append(data)
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        env = run.phys.official_env_factory()
        try:
            _, info = env.reset(seed=0, options={"state": root["states"][0].copy(),
                                                 "goal_state": run.phys.DIAGNOSTIC_GOAL.copy()})
            run.phys._goal_contract(env, info, root, run.phys.official_env_factory)
            commands = np.concatenate([np.asarray(root["actions"][:15], np.float64)] +
                                      [np.asarray(d["dispatch"], np.float64) for d in decisions])
            frames, packets = [], []
            for i, command in enumerate(commands):
                attempt, returned = events[2*i:2*i+2]
                if attempt["kind"] != "attempt" or returned["kind"] != "returned":
                    raise ValueError("event sequence mismatch")
                np.testing.assert_array_equal(np.asarray(attempt["command"], np.float64), command)
                transition = env.step(command)
                state = run.packets.state_of(env)
                np.testing.assert_allclose(state, returned["state"], atol=1e-9, rtol=0)
                if (run.phys._array_hash(run.phys.render_without_advancing(env)) != returned["rgb_sha256"]
                        or bool(transition[2]) != returned["terminated"]
                        or bool(transition[3]) != returned["truncated"]):
                    raise ValueError("physical replay mismatch")
                if (i+1) % 5 == 0:
                    frames.append(run.phys.render_without_advancing(env).copy())
                    packets.append(commands[i-4:i+1])
                if i+1 >= 15 and (i+1) % 5 == 0:
                    index = (i+1-15)//5
                    with np.load(run.path("response", f"context{index}.npz"), allow_pickle=False) as context:
                        np.testing.assert_array_equal(context["frames"], np.stack(frames[-3:]))
                        np.testing.assert_array_equal(context["history"], np.concatenate(packets[-3:]).astype(np.float32))
                        np.testing.assert_array_equal(context["goal"], info["goal"])
                    np.testing.assert_allclose(state, receipt["outcomes"][index]["state"], atol=1e-9, rtol=0)
        finally:
            env.close()
        model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        model_hash = run.saved.checked.tensor_state_hash(model)
        if model_hash != result["model_before"] or model_hash != result["model_after"]:
            raise ValueError("model parameter mismatch")
        device = torch.device("cuda:0")
        transform = run.saved.p0.image_transform()
        max_position = max_score = 0.
        sample_total = batches = 0
        with torch.no_grad():
            for step, data in enumerate(decisions):
                with np.load(run.path("response", f"context{step}.npz"), allow_pickle=False) as ctx:
                    images = run.saved.p0._frames_tensor(list(ctx["frames"]), transform=transform, device=device)
                    emb0 = model.encode({"pixels": images})["emb"]
                    goal = run.saved.p0._goal_latent(model, ctx["goal"], transform=transform, device=device)
                    history = run.saved.live.adapter.raw_to_packets(ctx["history"][None],
                        np.tile(run.saved.p0.PUBLIC_ACTION_MEAN, 5), np.tile(run.saved.p0.PUBLIC_ACTION_STD, 5), device=device)
                chart = run.saved.live.adapter.goal_chart(emb0, goal, 2027111800 + step)
                np.testing.assert_allclose(chart.cpu().numpy(), data["chart"], atol=2e-5, rtol=0)
                count, tokens = 0, 0
                for batch in data["trace"]:
                    requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
                    np.testing.assert_array_equal(requested.astype(np.float32), raw)
                    if np.any(np.abs(raw)>1):
                        raise ValueError("out-of-bounds action")
                    action = torch.tensor(raw, device=device)
                    future = run.packets.normalize_live_packet(action.reshape(-1, 5, 2),
                        mean=run.saved.p0.PUBLIC_ACTION_MEAN, std=run.saved.p0.PUBLIC_ACTION_STD).reshape(len(raw), -1, 10)
                    embeddings = emb0.expand(len(raw), -1, -1)
                    previous = history[:, 1:, :].expand(len(raw), -1, -1)
                    for t in range(future.shape[1]):
                        outgoing = torch.cat((previous, future[:, t:t+1]), dim=1)
                        latent = model.predict(embeddings, model.action_encoder(outgoing))[:, -1]
                        embeddings = torch.cat((embeddings[:, 1:], latent[:, None]), dim=1)
                        previous = outgoing[:, 1:]
                    position = (latent @ chart).cpu().numpy()
                    scores = (latent-goal).square().mean(dim=1).cpu().numpy()
                    max_position = max(max_position, float(np.max(np.abs(position-batch["positions"]))))
                    max_score = max(max_score, float(np.max(np.abs(scores-batch["scores"]))))
                    expected = len(raw)*raw.shape[1]//5
                    if batch["tokens"] != expected:
                        raise ValueError("token mismatch")
                    count, tokens = count+len(raw), tokens+expected
                if (tokens > 1920 or tokens != data["counts"]["prediction_token_samples"]
                        or count != data["counts"]["trajectory_samples"]
                        or len(data["trace"]) != data["counts"]["prediction_calls"]):
                    raise ValueError("resource mismatch")
                tokens_by_step.append(tokens)
                sample_total, batches = sample_total+count, batches+len(data["trace"])
        if max_position > 2e-5 or max_score > 2e-6:
            raise ValueError(f"model reference mismatch: {max_position}, {max_score}")
        if run.saved.checked.tensor_state_hash(model) != model_hash:
            raise ValueError("model drift")
    run.seal.verify(manifest["seal"])
    host.write_new(out, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
        "execution_sha256": sha(receipt_path), "physical_steps": 30, "batches": batches, "samples": sample_total,
        "tokens": tokens_by_step, "max_position_error": max_position, "max_score_error": max_score,
        "cuda_allocation": allocation, "scope": "same model and physical backend, no independent replication"})
    print(f"Audited 30 physical commands and {batches} model batches.", flush=True)


if __name__ == "__main__":
    main()
