"""Run the unchanged v4 compiler against saved PushT latent contexts."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_response_temporal_v3_reacher_rgb_live_20261104_codex as live
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as method
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_v3_dependency_seal as seal
from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as reference
from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
from experiments.checkpoint_compat import load_compatible_pretrained

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_V4_PUSHT_SAVED_CONTEXT_ADAPTER_PROTOCOL_20261104_CODEX.md"
SOURCE = ROOT / "results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json"
POLICY = Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")
PREFIX = ROOT / "results/pc_v4_pusht_saved_context_adapter_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
HOST = live.host


def root_path(index):
    return PREFIX.with_name(PREFIX.name + f"_root{index}.json")


def main():
    paths = [MANIFEST, RESULT] + [root_path(i) for i in range(4)]
    if any(p.exists() for p in paths):
        raise FileExistsError("existing or partial output")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != HOST.UUID:
        raise ValueError("cooperative GPU UUID required")
    report, bindings, config = checked.validate_inputs(SOURCE, POLICY)
    source_paths = checked.sources_before_allocate(report, config)
    deps = seal.capture([Path(__file__), PROTOCOL, Path(method.__file__), Path(local.__file__),
                         Path(reference.__file__), Path(p0.__file__), *source_paths,
                         *[Path(x) for x in bindings]])
    HOST.write_new(MANIFEST, {"seal": deps, "bindings": bindings, "roots": report["protocol"]["roots"],
                              "scope": "four exposed saved latent contexts; model only"})
    print("PushT saved contexts: reference packet parity, then unchanged v4 model-only plans.", flush=True)
    sha = HOST.bridge.sha_file
    mean = np.tile(np.asarray(p0.PUBLIC_ACTION_MEAN, np.float32), 5)
    std = np.tile(np.asarray(p0.PUBLIC_ACTION_STD, np.float32), 5)
    lo, hi = (-1 - mean) / std, (1 - mean) / std
    summaries = []
    with HOST.lock.gpu_lock(HOST.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if HOST.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != HOST.canonical_gpu_uuid(HOST.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = load_compatible_pretrained(POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        before = checked.tensor_state_hash(model)
        lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
        old_adapter = live.adapter
        replay = reference.CudaReplay(model, device)
        for index, row in enumerate(report["rows"]):
            compiled = row["compiled"]
            encoded = torch.tensor(compiled["input_context"]["emb"], dtype=torch.float32, device=device)
            history = torch.tensor(compiled["input_context"]["actions"], dtype=torch.float32, device=device)
            goal = torch.tensor(compiled["goal_vectors"][0], dtype=torch.float32, device=device)
            if encoded.shape != (1, 3, 192) or history.shape != (1, 3, 10) or goal.shape != (192,):
                raise ValueError("saved context shape mismatch")
            context = {"layout": "states_with_between_state_actions_v1", "context_length": 3,
                       "latent_dim": 192, "action_dim": 10,
                       "latent_history": encoded[0].cpu().tolist(), "action_history": history[0, 1:].cpu().tolist()}
            with torch.no_grad():
                chart = old_adapter.goal_chart(encoded, goal, 2027111800 + index)
                target = (goal @ chart).cpu().numpy()
            parity = []
            for horizon in (5, 10, 15):
                n = np.arange(horizon, dtype=np.float32)
                batch = np.stack((np.zeros((horizon, 2), np.float32),
                    np.tile(np.array([.25, -.3], np.float32), (horizon, 1)),
                    np.stack((.4*np.sin(n), .3*np.cos(n)), axis=1)))
                with torch.no_grad():
                    terminal = old_adapter.cached_terminal(model, encoded, history,
                        old_adapter.raw_to_packets(batch, mean, std, device=device), lower, upper).cpu().numpy()
                for j, action in enumerate(batch):
                    ref = np.asarray(replay(context, action)["latent_history"][-1])
                    err = float(np.max(np.abs(terminal[j] - ref)))
                    if err > 2e-5:
                        raise ValueError(f"reference parity failed {index}/{horizon}/{j}: {err}")
                    parity.append({"horizon": horizon, "action_index": j, "max_abs": err})
            meter, trace = Meter(1920), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)
                def evaluate(batch):
                    future = old_adapter.raw_to_packets(batch, mean, std, device=device)
                    with torch.no_grad():
                        terminal = old_adapter.cached_terminal(model, encoded, history, future, lower, upper)
                        projected = terminal @ chart
                        scores = (terminal - goal).square().mean(dim=1)
                        torch.cuda.synchronize()
                    return projected.cpu().numpy(), scores.cpu().numpy()
                positions, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(), "positions": positions.tolist(),
                              "scores": scores.tolist(), "tokens": len(raw) * raw.shape[1] // 5})
                return positions, scores

            started = time.monotonic()
            result = method.decide("response", index, 0, target, predict, meter)
            if meter.token_samples > 1920 or np.asarray(result["selected"]["chain"]).shape != (15, 2):
                raise ValueError("invalid decision")
            if checked.tensor_state_hash(model) != before:
                raise ValueError("model drift")
            seal.verify(deps)
            HOST.write_new(root_path(index), {"manifest_sha256": sha(MANIFEST), "root": index,
                "root_sha256": row["root_sha256"], "parity": parity, "chart": chart.cpu().tolist(),
                "target": target.tolist(), "result": result, "trace": trace, "counts": meter.as_dict(),
                "planning_seconds": time.monotonic() - started, "physical_execution": False})
            summaries.append({"root": index, "path_sha256": sha(root_path(index)),
                              "tokens": meter.token_samples, "terminal_score": result["selected"]["terminal_score"]})
            print(f"root={index}: parity and {meter.token_samples} tokens complete", flush=True)
        after = checked.tensor_state_hash(model)
    seal.verify(deps)
    checked.validate_inputs(SOURCE, POLICY)
    HOST.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": summaries, "model_before": before,
        "model_after": after, "cuda_allocation": allocation, "physical_execution": False,
        "scope": "saved exposed PushT contexts, not E3 contact evidence"})


if __name__ == "__main__":
    main()
