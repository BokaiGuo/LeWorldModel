"""Replay saved prefix and suffix actions on CUDA; no physical execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_response_common_suffix20_cuda_20261030_main as checked
from . import p1_response_common_suffix_cuda_20261030_main as source

NAME = "p1_response_common_suffix20_replay_20261030_main"
OUTCOME = source.base.ROOT / "results/p1_response_common_suffix20_cuda_20261030_main_RUN01.json"
OUTCOME_SHA = "9b29bfd041bd0ed73ba03c3fe2add8bc22522de6a319326a02ea68b9c0f6a053"


def validate_result(value, report, selection):
    if (value["status"] != "complete_model_only" or value["config"]["steps"] != 20
            or len(value["rows"]) != 8):
        raise ValueError("recorded model-only contract")
    for row, arm, selected in zip(value["rows"], report["rows"], selection["arms"], strict=True):
        sets = source.candidate_sets(arm, selected)
        if row["slot"] != arm["slot"] or row["arm"] != arm["arm"] or row["candidate_sets"] != sets:
            raise ValueError("candidate roster")
        union = set(sets["diverse"] + sets["immediate"])
        if {int(x) for x in row["candidates"]} != union:
            raise ValueError("candidate union")
        for index in union:
            candidate = arm["result"]["candidates"][index]
            saved = row["candidates"][str(index)]
            raw = bytes.fromhex(saved["suffix_float32_hex"])
            if (saved["prefix_float32_hex"] != candidate["raw_float32_bits"][:80]
                    or len(raw) != 40 or hashlib.sha256(raw).hexdigest() != saved["suffix_sha256"]
                    or not np.isfinite(np.frombuffer(raw, "<f4")).all()):
                raise ValueError("candidate action bytes")
        for name, indices in sets.items():
            winner = min(indices, key=lambda i: (row["candidates"][str(i)]["model_terminal_mse"], i))
            if row["winners"][name] != winner:
                raise ValueError("winner")


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    output = Path(output)
    if (not output.is_absolute() or output.parent != source.base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    report, prepared, selection = source.inputs()
    outcome = source.panel.source.saved.read_verified(OUTCOME, OUTCOME_SHA)
    validate_result(outcome, report, selection)
    bindings = source.base.bindings()
    for path in (Path(__file__).resolve(), OUTCOME, Path(source.__file__).resolve(),
                 Path(checked.__file__).resolve(), source.PANEL, source.panel.source.REPORT,
                 source.panel.source.RECEIPT, Path(report["prepared"])):
        bindings[str(path)] = source.base.sha(path)
    results = []
    with source.base.locking.gpu_lock(source.base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(source.base.MODEL), device=device).eval().requires_grad_(False)
        before = source.base.model_hash(model)
        if before != source.base.MODEL_HASH:
            raise ValueError("frozen checkpoint")
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, arm in zip(outcome["rows"], report["rows"], strict=True):
                task = prepared["requests"][row["slot"]]["task"]
                tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
                mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                for index in sorted(map(int, row["candidates"])):
                    candidate = arm["result"]["candidates"][index]
                    saved = row["candidates"][str(index)]
                    checked.check_context(task, candidate)
                    prefix = tensor(np.frombuffer(bytes.fromhex(saved["prefix_float32_hex"]), "<f4").copy().reshape(1, 5, 2))
                    suffix = tensor(np.frombuffer(bytes.fromhex(saved["suffix_float32_hex"]), "<f4").copy().reshape(1, 5, 2))
                    states = tensor(task["root_context"]["latent_history"])[None]
                    history = tensor(task["root_context"]["action_history"])[None]
                    context = torch.cat((history, ((prefix - mean) / std).reshape(1, 1, 10)), dim=1)
                    pred_mid = model.predict(states, model.action_encoder(context))
                    actual_mid = tensor(candidate["mid"]["latent_history"][-1])
                    mid_max = float((pred_mid[0, -1] - actual_mid).abs().max().item())
                    if mid_max > 1e-5:
                        raise ValueError(f"prefix replay mismatch: {row['slot']}/{row['arm']}/{index}: {mid_max}")
                    # Use saved mid for an independent score replay matching the optimizer input.
                    mid_states = tensor(candidate["mid"]["latent_history"])[None]
                    mid_history = tensor(candidate["mid"]["action_history"])[None]
                    suffix_context = torch.cat((mid_history, ((suffix - mean) / std).reshape(1, 1, 10)), dim=1)
                    terminal = model.predict(mid_states, model.action_encoder(suffix_context))
                    score = float((terminal[:, -1] - tensor(task["goal"])[None]).square().mean().item())
                    score_error = abs(score - saved["model_terminal_mse"])
                    if score_error > 1e-6:
                        raise ValueError("saved score replay mismatch")
                    results.append({"slot": row["slot"], "arm": row["arm"], "candidate": index,
                                    "prefix_latent_max_abs_error": mid_max,
                                    "terminal_score_abs_error": score_error})
        torch.cuda.synchronize(device)
        after = source.base.model_hash(model)
        if before != after or any(source.base.sha(Path(k)) != v for k, v in bindings.items()):
            raise ValueError("model/source drift")
        result = {"schema": NAME, "status": "complete_cuda_replay", "rows": results,
                  "source_outcome_sha256": OUTCOME_SHA, "sources": bindings, "host": host,
                  "model_before": before, "model_after": after,
                  "calls": {"action_encoder": 2 * len(results), "predict": 2 * len(results)},
                  "physical_execution": False, "independent_model_implementation": False,
                  "equal_budget_certified": False, "P1_admission": False}
        digest = source.base.write(output, result)
    print(json.dumps({"status": result["status"], "rows": len(results), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replay every saved prefix and suffix on the frozen host CUDA model.", flush=True)
    run(args.output)
