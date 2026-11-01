"""Fixed-panel, fixed-cost two-packet backtracking; exploratory host CUDA only."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_regularized_suffix_host_20261030 as base

NAME = "p1_bounded_backtrack_host_20261030"
STEPS = (1.0, .5, .25, .125)
UPDATES = 8


def trial_logits(x, grad, raw, step):
    x = np.asarray(x, np.float64).reshape(2, 10)
    grad = np.asarray(grad, np.float64).reshape(2, 10)
    raw = np.asarray(raw, np.float32).astype(np.float64).reshape(2, 10)
    if not all(np.isfinite(v).all() for v in (x, grad, raw)) or step not in STEPS:
        raise ValueError("invalid trial input")
    direction = grad * (1 - raw * raw) + 2 * base.one.saved.DAMPING * x / 192
    return np.stack([base.one.projection.projected_logits(
        np.stack((x[k] - step * direction[k], np.zeros(10))), base.one.CAP)[0]
        for k in range(2)])


def choose(current, scores):
    if len(scores) != len(STEPS) or not np.isfinite([current, *scores]).all():
        raise ValueError("invalid trial scores")
    return next((i for i, score in enumerate(scores) if score < current), None)


def run(output):
    one = base.one
    source, saved, host = one.source, one.saved, one.host
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt.get("report_sha256") != source.REPORT_SHA:
        raise ValueError("receipt binding")
    prepared = saved.read_verified(report["prepared"], report["prepare_sha256"])
    saved.analyze(report, receipt, prepared)
    host.verify(report["prepared"], report["prepare_sha256"])
    output = Path(output)
    if (not output.is_absolute() or output.parent != source.ROOT / "results"
            or output.resolve() != output or not output.name.startswith(NAME + "_")
            or output.exists()):
        raise ValueError("exclusive output required")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_info = require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    calls = dict(predict=0, action_encoder=0, reverse=0)
    rows = []
    with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared["policy"], device=device).eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        if before != report["model_before"]:
            raise ValueError("model hash")
        for module in model.modules():
            if str(Path(inspect.getfile(type(module))).resolve()) not in prepared["bindings"]:
                raise ValueError("model class source unbound")

        def hook(name):
            def count(_module, _args):
                calls[name] += 1
            return count

        hooks = [model.predictor.register_forward_pre_hook(hook("predict")),
                 model.action_encoder.register_forward_pre_hook(hook("action_encoder"))]
        try:
            for arm_index, arm in enumerate(report["rows"]):
                task = prepared["requests"][arm_index // 2]["task"]
                goal = torch.tensor(task["goal"], dtype=torch.float32, device=device)
                goal64 = np.asarray(task["goal"], np.float64)
                mean = torch.tensor(np.tile(task["action_mean"], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(task["action_std"], 5), dtype=torch.float32, device=device)
                for candidate in arm["result"]["candidates"]:
                    z = torch.tensor(candidate["mid"]["latent_history"], dtype=torch.float32, device=device)[None]
                    h = torch.tensor(candidate["mid"]["action_history"], dtype=torch.float32, device=device)[None]
                    x = np.stack((np.asarray(candidate["suffix_solve"]["params"], np.float64), np.zeros(10)))

                    def evaluate(values, differentiable):
                        raw = np.tanh(np.asarray(values, np.float32)).reshape(2, 5, 2)
                        action = torch.tensor(raw, dtype=torch.float32, device=device, requires_grad=differentiable)
                        tokens = [(action[k].reshape(10) - mean) / std for k in range(2)]
                        ends = base.advance(z, h, tokens, lambda s, a: model.predict(s, model.action_encoder(a)))
                        endpoint = ends[-1].detach().cpu().numpy().astype(np.float64)
                        q = float(np.mean((endpoint - goal64) ** 2))
                        score = q + saved.DAMPING * float(np.sum(values * values)) / 192
                        grad = None
                        if differentiable:
                            loss = torch.mean((ends[-1] - goal) ** 2)
                            grad = torch.autograd.grad(loss, action)[0].detach().cpu().numpy()
                            calls["reverse"] += 1
                        return dict(logits=values.tolist(), raw=raw.tolist(), endpoint=endpoint.tolist(),
                                    first_endpoint=ends[0].detach().cpu().numpy().astype(np.float64).tolist(),
                                    terminal_q=q, regularized=score,
                                    raw_gradient=None if grad is None else grad.tolist())

                    trajectory = []
                    for iteration in range(UPDATES):
                        current = evaluate(x, True)
                        if iteration == 0 and not np.array_equal(
                                current["first_endpoint"], candidate["end"]["latent_history"][-1]):
                            raise ValueError("first packet saved replay mismatch")
                        with torch.no_grad():
                            trials = [evaluate(trial_logits(x, current["raw_gradient"], current["raw"], step), False)
                                      for step in STEPS]
                        selected = choose(current["regularized"], [t["regularized"] for t in trials])
                        trajectory.append(dict(iteration=iteration, current=current, trials=trials, selected=selected))
                        if selected is not None:
                            x = np.asarray(trials[selected]["logits"], np.float64)
                    with torch.no_grad():
                        final = evaluate(x, False)
                    rows.append(dict(slot=arm["slot"], arm=arm["arm"], start=candidate["start"],
                                     iteration=candidate["iteration"], stage_uphill=candidate["stage_uphill"],
                                     trajectory=trajectory, final=final,
                                     best_regularized=final["regularized"]))
        finally:
            for handle in hooks:
                handle.remove()
        torch.cuda.synchronize(device)
        after = host.signal.model_hash(model)
        if before != after or calls != dict(predict=13120, action_encoder=13120, reverse=1280):
            raise ValueError("model mutation or call denominator mismatch")
    host.F.check_bindings(prepared["bindings"])
    summary = base.summarize(report, rows)
    summary["arm_winners"] = [
        dict(slot=group[0]["slot"], arm=group[0]["arm"],
             winner_index=min(range(20), key=lambda j: (group[j]["best_regularized"], j)),
             stage_uphill=min(group, key=lambda r: r["best_regularized"])["stage_uphill"])
        for group in (rows[i:i + 20] for i in range(0, 160, 20))
    ]
    result = dict(schema=NAME, source_report_sha256=source.REPORT_SHA,
                  source_receipt_sha256=source.RECEIPT_SHA,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  device=device_info, model_before=before, model_after=after, calls=calls,
                  steps=STEPS, rows=rows, summary=summary, training=False,
                  physical_release=False, P1_admission=False)
    return dict(path=str(output), sha256=saved.write(output, result), summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Evaluating fixed-cost backtracking on all fixed P1 candidates; model-only exploration.", flush=True)
    print(json.dumps(run(parser.parse_args().output)))


if __name__ == "__main__":
    main()
