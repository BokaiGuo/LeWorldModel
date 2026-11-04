"""Independently recompute every PushT v4 batch through explicit packet unroll."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_pusht_saved_context_adapter_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, sha = run.HOST, run.HOST.bridge.sha_file
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("GPU UUID required")
    manifest, result = json.loads(run.MANIFEST.read_text()), json.loads(run.RESULT.read_text())
    seal.verify(manifest["seal"])
    source, bindings, _ = run.checked.validate_inputs(run.SOURCE, run.POLICY)
    if result["manifest_sha256"] != sha(run.MANIFEST) or manifest["bindings"] != bindings:
        raise ValueError("source binding mismatch")
    rows = []
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = run.load_compatible_pretrained(run.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        initial_hash = run.checked.tensor_state_hash(model)
        if initial_hash != result["model_before"] or result["model_before"] != result["model_after"]:
            raise ValueError("checkpoint or buffer mismatch")
        for index, source_row in enumerate(source["rows"]):
            path = run.root_path(index)
            data = json.loads(path.read_text())
            if (result["rows"][index]["path_sha256"] != sha(path) or data["manifest_sha256"] != sha(run.MANIFEST)
                    or data["root_sha256"] != source_row["root_sha256"] or len(data["parity"]) != 9):
                raise ValueError("root binding mismatch")
            compiled = source_row["compiled"]
            emb0 = torch.tensor(compiled["input_context"]["emb"], dtype=torch.float32, device=device)
            history0 = torch.tensor(compiled["input_context"]["actions"], dtype=torch.float32, device=device)
            goal = torch.tensor(compiled["goal_vectors"][0], dtype=torch.float32, device=device)
            chart = run.live.adapter.goal_chart(emb0, goal, 2027111800 + index)
            np.testing.assert_allclose(chart.cpu().numpy(), data["chart"], atol=2e-5, rtol=0)
            selected = np.asarray(data["result"]["selected"]["chain"])
            score = data["result"]["selected"]["terminal_score"]
            count, tokens, errors, score_errors, full_scores, found = 0, 0, [], [], [], False
            with torch.no_grad():
                for batch in data["trace"]:
                    requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
                    np.testing.assert_array_equal(requested.astype(np.float32), raw)
                    actions = torch.tensor(raw, device=device)
                    packets = run.p0.packets.normalize_live_packet(actions.reshape(-1, 5, 2),
                        mean=run.p0.PUBLIC_ACTION_MEAN, std=run.p0.PUBLIC_ACTION_STD).reshape(len(raw), -1, 10)
                    emb = emb0.expand(len(raw), -1, -1)
                    between = history0[:, 1:, :].expand(len(raw), -1, -1)
                    for t in range(packets.shape[1]):
                        outgoing = torch.cat((between, packets[:, t:t+1]), dim=1)
                        latent = model.predict(emb, model.action_encoder(outgoing))[:, -1]
                        emb = torch.cat((emb[:, 1:], latent[:, None]), dim=1)
                        between = outgoing[:, 1:]
                    positions = (latent @ chart).cpu().numpy()
                    scores = (latent - goal).square().mean(dim=1).cpu().numpy()
                    errors.append(float(np.max(np.abs(positions - batch["positions"]))))
                    score_errors.append(float(np.max(np.abs(scores - batch["scores"]))))
                    if raw.shape[1] == 15:
                        full_scores.extend(batch["scores"])
                        found |= any(batch["scores"][j] == score for j in np.flatnonzero(
                            np.all(requested == selected, axis=(1, 2))))
                    count += len(raw)
                    if batch["tokens"] != len(raw) * raw.shape[1] // 5:
                        raise ValueError("bad token count")
                    tokens += batch["tokens"]
            if (not found or min(full_scores) != score or tokens > 1920
                    or tokens != data["counts"]["prediction_token_samples"]
                    or count != data["counts"]["trajectory_samples"]
                    or len(data["trace"]) != data["counts"]["prediction_calls"]
                    or max(errors) > 2e-5 or max(score_errors) > 2e-6):
                raise ValueError("prediction, selection or budget mismatch")
            rows.append({"root": index, "batches": len(data["trace"]), "samples": count, "tokens": tokens,
                         "max_position_error": max(errors), "max_score_error": max(score_errors),
                         "reference_parity_max": max(x["max_abs"] for x in data["parity"])})
            print(f"root={index}: independently recomputed {len(data['trace'])} batches", flush=True)
        if run.checked.tensor_state_hash(model) != initial_hash:
            raise ValueError("model drift")
    seal.verify(manifest["seal"])
    host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
                         "script_sha256": sha(Path(__file__)), "rows": rows, "cuda_allocation": allocation,
                         "scope": "same checkpoint, separately unrolled saved contexts; no physical execution"})


if __name__ == "__main__":
    main()
