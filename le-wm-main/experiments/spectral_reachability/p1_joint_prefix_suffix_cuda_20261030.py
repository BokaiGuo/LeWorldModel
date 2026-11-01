"""Known-root full-action refinement on all saved P1 candidate witnesses."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import p1_regularized_suffix_host_20261030 as base
from . import p1_backtrack_action_inventory_20261030 as inventory

NAME = "p1_joint_prefix_suffix_cuda_20261030"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/D2P_P1_JOINT_PREFIX_SUFFIX_PROTOCOL_20261030_MAIN.md"
UPDATES = 4
RATE = 0.05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def optimize(initial, evaluate):
    """Projected fixed-step action optimization; evaluate returns scalar and endpoints."""
    require(initial.shape == (3, 5, 2) and initial.is_floating_point()
            and bool(torch.isfinite(initial).all().item())
            and bool((initial.abs() <= 1).all().item()), "initial action")
    action = initial.detach().clone()
    records = []
    for iteration in range(UPDATES + 1):
        action = action.detach().requires_grad_(iteration < UPDATES)
        loss, endpoints = evaluate(action)
        require(loss.ndim == 0 and bool(torch.isfinite(loss).item())
                and len(endpoints) == 3 and all(bool(torch.isfinite(x).all().item()) for x in endpoints),
                "finite rollout")
        records.append({"iteration": iteration, "score": float(loss.detach().item()),
                        "actions": action.detach().cpu().tolist(),
                        "prefix_latent": endpoints[0].detach().cpu().tolist(),
                        "terminal_latent": endpoints[-1].detach().cpu().tolist()})
        if iteration < UPDATES:
            gradient = torch.autograd.grad(loss, action)[0]
            require(gradient.shape == action.shape and bool(torch.isfinite(gradient).all().item()), "gradient")
            action = (action - RATE * gradient).clamp(-1, 1).detach()
    winner = min(range(len(records)), key=lambda i: (records[i]["score"], i))
    return {"trajectory": records, "best_iteration": winner, "best_score": records[winner]["score"],
            "best_actions": records[winner]["actions"]}


def run(output):
    source, saved, host = base.one.source, base.one.saved, base.one.host
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    require(receipt.get("report_sha256") == source.REPORT_SHA, "source receipt")
    prepared = saved.read_verified(report["prepared"], report["prepare_sha256"])
    saved.analyze(report, receipt, prepared)
    host.verify(report["prepared"], report["prepare_sha256"])
    backtrack = saved.read_verified(ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", inventory.SHA)
    inventory.verifier.verify(backtrack, report, prepared)
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results" and output.resolve() == output
            and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_info = require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    calls = {"predict": 0, "action_encoder": 0, "reverse": 0}
    rows = []
    with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared["policy"], device=device).eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        require(before == report["model_before"], "model hash")
        for module in model.modules():
            require(str(Path(inspect.getfile(type(module))).resolve()) in prepared["bindings"], "model class source")

        def counter(name):
            def count(_module, _args):
                calls[name] += 1
            return count
        hooks = [model.predictor.register_forward_pre_hook(counter("predict")),
                 model.action_encoder.register_forward_pre_hook(counter("action_encoder"))]
        try:
            for arm_index, arm in enumerate(report["rows"]):
                task = prepared["requests"][arm_index // 2]["task"]
                goal = torch.tensor(task["goal"], dtype=torch.float32, device=device)
                states = torch.tensor(task["root_context"]["latent_history"], dtype=torch.float32, device=device)[None]
                history = torch.tensor(task["root_context"]["action_history"], dtype=torch.float32, device=device)[None]
                mean = torch.tensor(np.tile(task["action_mean"], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(task["action_std"], 5), dtype=torch.float32, device=device)
                require(bool((std > 0).all().item()), "normalizer")
                for candidate_index, candidate in enumerate(arm["result"]["candidates"]):
                    continuation = backtrack["rows"][arm_index * 20 + candidate_index]
                    initial_np = np.concatenate((np.asarray(candidate["prefix"], np.float32),
                                                 np.asarray(continuation["final"]["raw"], np.float32).reshape(10, 2)))
                    initial = torch.tensor(initial_np.reshape(3, 5, 2), device=device)

                    def evaluate(action):
                        tokens = [(packet.reshape(10) - mean) / std for packet in action]
                        ends = base.advance(states, history, tokens,
                                            lambda z, h: model.predict(z, model.action_encoder(h)))
                        loss = (ends[-1] - goal).square().mean()
                        return loss, ends
                    result = optimize(initial, evaluate)
                    calls["reverse"] += UPDATES
                    rows.append({"slot": arm["slot"], "arm": arm["arm"], "candidate_index": candidate_index,
                                 "start": candidate["start"], "iteration": candidate["iteration"], **result})
        finally:
            for hook in hooks:
                hook.remove()
        torch.cuda.synchronize(device)
        after = host.signal.model_hash(model)
        require(before == after and calls == {"predict": 2400, "action_encoder": 2400, "reverse": 640},
                "model mutation/call denominator")
    host.F.check_bindings(prepared["bindings"])
    winners = []
    for i in range(8):
        group = rows[i * 20:(i + 1) * 20]
        chosen = min(group, key=lambda item: (item["best_score"], item["candidate_index"]))
        winners.append({"slot": chosen["slot"], "arm": chosen["arm"],
                        "candidate_index": chosen["candidate_index"],
                        "best_iteration": chosen["best_iteration"], "best_score": chosen["best_score"]})
    result = {"schema": NAME, "source_report_sha256": source.REPORT_SHA,
              "source_backtrack_sha256": inventory.SHA, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "device": device_info,
              "model_before": before, "model_after": after, "calls": calls, "rows": rows, "winners": winners,
              "training": False, "physical_release": False, "fairness_proven": False,
              "independent_confirmation": False, "P1_admission": False}
    return {"path": str(output), "sha256": saved.write(output, result), "winners": winners}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Jointly refine every fixed 15-step action witness on host CUDA; model-only exploration.", flush=True)
    print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
