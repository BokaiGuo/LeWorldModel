"""Known-root model-only comparison of response-diverse and immediate prefix sets."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import p1_task_response_saved_panel_20261030_main as panel
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base

NAME = "p1_response_common_suffix_cuda_20261030_main"
PANEL = base.ROOT / "results/p1_task_response_saved_panel_20261030_main_MAIN01.json"
PANEL_SHA = "940c854036dc660cc3c0217f1486abf401c0148dd8fa3078ec90c301df16cc88"
STEPS = 5
LR = 0.05


def candidate_sets(arm, selection):
    candidates = arm["result"]["candidates"]
    if len(candidates) != 20 or arm["slot"] != selection["slot"] or arm["arm"] != selection["arm"]:
        raise ValueError("candidate panel identity")
    diverse = selection["selected_indices"]
    if len(diverse) != 4 or len(set(diverse)) != 4 or any(type(i) is not int or not 0 <= i < 20 for i in diverse):
        raise ValueError("diverse indices")
    immediate = sorted(range(20), key=lambda i: (candidates[i]["stage_Q"], i))[:4]
    return {"diverse": diverse, "immediate": immediate}


def inputs():
    src = panel.source
    report = src.saved.read_verified(src.REPORT, src.REPORT_SHA)
    receipt = src.saved.read_verified(src.RECEIPT, src.RECEIPT_SHA)
    if receipt.get("report_sha256") != src.REPORT_SHA:
        raise ValueError("source receipt")
    prepared = src.saved.read_verified(report["prepared"], report["prepare_sha256"])
    src.saved.analyze(report, receipt, prepared)
    src.host.verify(report["prepared"], report["prepare_sha256"])
    selected = src.saved.read_verified(PANEL, PANEL_SHA)
    if selected != panel.analyze(report, prepared):
        raise ValueError("selection reconstruction")
    return report, prepared, selected


def optimize(model, task, candidate, device):
    import torch
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    context = candidate["mid"]
    states = tensor(context["latent_history"])[None]
    history = tensor(context["action_history"])[None]
    goal = tensor(task["goal"])[None]
    mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
    if (states.shape != (1, 3, 192) or history.shape != (1, 2, 10)
            or goal.shape != (1, 192) or mean.shape != (2,) or std.shape != (2,)
            or not bool((std > 0).all())):
        raise ValueError("candidate model context")
    action = torch.zeros((1, 5, 2), dtype=torch.float32, device=device)

    def score(raw):
        normalized = ((raw - mean) / std).reshape(1, 1, 10)
        prediction = model.predict(states, model.action_encoder(torch.cat((history, normalized), dim=1)))
        if prediction.shape != states.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError("prediction")
        return (prediction[:, -1] - goal).square().mean()

    for _ in range(STEPS):
        action = action.detach().requires_grad_(True)
        loss = score(action)
        gradient, = torch.autograd.grad(loss, action)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("gradient")
        action = (action - LR * gradient).clamp(-1, 1).detach()
    with torch.no_grad():
        final_score = float(score(action).item())
    raw = np.asarray(action[0].cpu().numpy(), dtype="<f4")
    return {"suffix_float32_hex": raw.tobytes().hex(),
            "suffix_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "model_terminal_mse": final_score}


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    report, prepared, selection = inputs()
    sources = base.bindings()
    for path in (Path(__file__).resolve(), Path(panel.__file__).resolve(), PANEL,
                 panel.source.REPORT, panel.source.RECEIPT, Path(report["prepared"])):
        sources[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen checkpoint mismatch")
        start = time.monotonic()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for arm, selected in zip(report["rows"], selection["arms"], strict=True):
                sets = candidate_sets(arm, selected)
                task = prepared["requests"][arm["slot"]]["task"]
                candidate_indices = sorted(set(sets["diverse"] + sets["immediate"]))
                results = {}
                for index in candidate_indices:
                    candidate = arm["result"]["candidates"][index]
                    result = optimize(model, task, candidate, device)
                    result["prefix_float32_hex"] = candidate["raw_float32_bits"][:80]
                    results[str(index)] = result
                winners = {name: min(indices, key=lambda i: (results[str(i)]["model_terminal_mse"], i))
                           for name, indices in sets.items()}
                rows.append({"slot": arm["slot"], "arm": arm["arm"], "candidate_sets": sets,
                             "candidates": results, "winners": winners})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(Path(path)) != digest for path, digest in sources.items()):
            raise ValueError("model or source drift")
        count = sum(len(row["candidates"]) for row in rows)
        result = {"schema": NAME, "status": "complete_model_only", "rows": rows,
                  "sources": sources, "host": host, "model_before": before, "model_after": after,
                  "config": {"sets": 4, "steps": STEPS, "learning_rate": LR,
                             "initialization": "zero_raw", "suffix_horizon": 5},
                  "calls": {"unique_candidates": count, "action_encoder": count * (STEPS + 1),
                            "predict": count * (STEPS + 1), "backward": count * STEPS},
                  "elapsed_excluding_load": time.monotonic() - start,
                  "physical_execution": False, "equal_budget_certified": False, "P1_admission": False,
                  "limitations": "Known roots; predicted prefix successors, not real RGB feedback; "
                                 "overlapping sets share evaluated candidates; costs exclude upstream "
                                 "Jacobian/prefix generation and have no independent hooks."}
        digest = base.write(output, result)
    print(json.dumps({"status": result["status"], "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Optimize the same fixed suffix from model-predicted candidates; no physical execution.", flush=True)
    run(args.output)
