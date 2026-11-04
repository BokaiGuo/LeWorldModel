"""Exposed-root frozen Cube v6 decision and separate packet replay audit."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v6_cube_packet_reference_dev_20261104_codex as prior
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_v3_dependency_seal as seal
from experiments.spectral_reachability import pc_response_temporal_v3_reacher_adapter as chart_adapter

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V6_CUBE_DECISION_DEV_PROTOCOL_20261104_CODEX.md"
OUTPUT = ROOT / "results/pc_v6_cube_decision_dev_20261104_codex.json"
TRACE = OUTPUT.with_suffix(".npz")


def image_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def main():
    if OUTPUT.exists() or TRACE.exists():
        raise FileExistsError("refusing to overwrite decision or trace")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if prior.digest(prior.POLICY / "weights.pt") != prior.EXPECTED_WEIGHTS:
        raise ValueError("checkpoint changed")
    if prior.digest(prior.RECEIPT) != prior.EXPECTED_RECEIPT:
        raise ValueError("receipt changed")
    receipt_source = json.loads(prior.RECEIPT.read_text())["source"]
    if prior.DATA.resolve() != Path(receipt_source["dataset"]).resolve():
        raise ValueError("dataset mismatch")
    receipt = prior.source.validate_receipt(prior.RECEIPT)
    contexts, provenance = prior.source.sample_source_native_contexts(
        prior.DATA, expected_sha256=receipt_source["dataset_sha256"],
        history_len=3, goal_offset_packets=1, count=4, seed=202610250,
    )
    ctx = contexts[0]
    deps = seal.capture([Path(__file__), PROTOCOL, Path(v6.__file__), Path(v4.__file__),
                         Path(local.__file__), Path(prior.adapter.__file__),
                         Path(prior.reference.__file__), Path(prior.__file__),
                         prior.POLICY / "config.json", prior.RECEIPT])
    mean = np.asarray(receipt["calibration_action_mean"], dtype=np.float32)
    std = np.asarray(receipt["calibration_action_std"], dtype=np.float32)
    print("Running v6 on one exposed Cube RGB source context; no physical execution.", flush=True)
    with prior.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        runtime = prior.require_cuda_allocation("cuda")
        device = torch.device("cuda")
        model = prior.load_compatible_pretrained(prior.POLICY, device="cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        prior._validate_cube_action_protocol(model, mean, std)
        model_hash = prior.reference._parameter_hash(model)
        transform = prior._image_transform()
        frames = [*ctx["frames"][1:], ctx["root_frame"]]
        pixels = prior.reference._frames_tensor(frames, transform=transform, device=device)
        history = prior.reference._normalize_raw_packets(
            torch.as_tensor(ctx["history_raw_packets"], device=device), mean, std,
            action_block=5, action_dim=5)[None]
        with torch.no_grad():
            encoded = model.encode({"pixels": pixels})["emb"]
            goal = prior.reference._goal_latent(model, ctx["goal"], transform=transform, device=device)
            raw_predict, target = prior.adapter.make_predictor(model, encoded, history, goal, mean, std, chart_seed=42)
            chart = chart_adapter.goal_chart(encoded, goal, 42)
        meter = Meter(1920)
        trace = {}

        def evaluate(actions):
            requested = np.asarray(actions, dtype=np.float64)
            raw = requested.astype(np.float32)
            position, score = raw_predict(raw)
            torch.cuda.synchronize()
            index = len(trace) // 4
            trace[f"requested_{index}"] = requested
            trace[f"raw_{index}"] = raw
            trace[f"position_{index}"] = position
            trace[f"score_{index}"] = score
            return position, score

        def predict(actions):
            return meter.predict(np.asarray(actions), evaluate)

        nominals = np.stack((np.zeros((5, 5)), np.full((5, 5), .2)))
        start = time.monotonic()
        result = v6.decide("response", "response", 0, 0, target, predict, meter, nominals)
        torch.cuda.synchronize()
        decision_seconds = time.monotonic() - start
        print(f"decision done: tokens={meter.token_samples}, audit starting", flush=True)
        # Audit is intentionally outside the decision meter and cannot affect selection.
        audit_start = time.monotonic()
        probes = trace["raw_0"]
        if probes.shape != (51, 5, 5):
            raise ValueError("unexpected first finite-difference batch")
        batch_positions, batch_scores = raw_predict(probes)
        singles = [raw_predict(probe[None]) for probe in probes]
        single_positions = np.concatenate([row[0] for row in singles])
        single_scores = np.concatenate([row[1] for row in singles])
        batch_jac = ((batch_positions[1:26] - batch_positions[26:]) / .1).T
        single_jac = ((single_positions[1:26] - single_positions[26:]) / .1).T
        jac_difference = float(np.linalg.norm(batch_jac - single_jac))
        jac_norm = float(np.linalg.norm(single_jac))
        selected = np.asarray(result["selected"]["chain"], dtype=np.float32)
        audit_actions = [probes[i] for i in (0, 1, 26)] + [selected]
        low, high = torch.full((5,), -1., device=device), torch.full((5,), 1., device=device)
        comparisons = []
        for action in audit_actions:
            pos, score = raw_predict(action[None])
            with torch.no_grad():
                expected = prior.reference._terminal_prediction(
                    model, ctx["frames"], ctx["root_frame"], ctx["history_raw_packets"],
                    torch.as_tensor(action.reshape(-1, 5, 5), device=device), mean=mean, std=std,
                    action_block=5, action_dim=5, transform=transform, device=device,
                    low=low, high=high, future_packets_are_raw=True)
                expected_pos = (expected @ chart).cpu().numpy()
                expected_score = float(((expected - goal) ** 2).mean().item())
            comparisons.append({"position_error": float(np.max(np.abs(pos[0] - expected_pos))),
                                "score_error": abs(float(score[0]) - expected_score)})
        torch.cuda.synchronize()
        audit_seconds = time.monotonic() - audit_start
        if model_hash != prior.reference._parameter_hash(model):
            raise ValueError("frozen model changed")
    seal.verify(deps)
    with TRACE.open("xb") as handle:
        np.savez_compressed(handle, **trace)
    audit = {
        "probe_samples": 51, "independent_reference_samples": 4,
        "max_batch_vs_single_position": float(np.max(np.abs(batch_positions-single_positions))),
        "max_batch_vs_single_score": float(np.max(np.abs(batch_scores-single_scores))),
        "fd_matrix_absolute_difference_frobenius": jac_difference,
        "fd_reference_frobenius": jac_norm,
        "fd_relative_difference": jac_difference / jac_norm if jac_norm > 1e-12 else None,
        "reference_comparisons": comparisons, "audit_seconds": audit_seconds,
    }
    output = {"scope": "one exposed source root; frozen model only; no physical execution",
              "protocol_sha256": prior.digest(PROTOCOL), "dependencies": deps,
              "dataset_sha256": receipt_source["dataset_sha256"],
              "checkpoint_sha256": prior.EXPECTED_WEIGHTS, "model_hash": model_hash,
              "source_rows": ctx["source_rows"], "source_provenance": provenance[0],
              "history_rgb_sha256": [image_hash(frame) for frame in frames],
              "goal_rgb_sha256": image_hash(ctx["goal"]),
              "history_actions_sha256": image_hash(ctx["history_raw_packets"]),
              "runtime": runtime, "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
              "target": target.tolist(), "nominals": nominals.tolist(),
              "decision": result, "counts": meter.as_dict(), "decision_seconds": decision_seconds,
              "trace_sha256": prior.digest(TRACE), "trace_batches": len(trace)//4, "audit": audit}
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(OUTPUT), "counts": meter.as_dict(), "audit": audit}), flush=True)


if __name__ == "__main__":
    main()
