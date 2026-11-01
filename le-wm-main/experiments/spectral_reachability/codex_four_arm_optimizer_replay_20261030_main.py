"""Post-hoc, fixed 16-node CUDA replay of optimizer inputs and outputs."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import codex_four_arm_same_action_prediction_20261030_v1 as fixed
from . import codex_p1_fixed_prefix_suffix_cuda_20261030_main as locking_source

NAME = "codex_four_arm_optimizer_replay_20261030_main"
need = fixed.need


def trace_candidates(node, report):
    entry = report["nodes"]["s0_" + node["arm"]][node["boundary"] // 5]["receipt"]
    path = Path(entry["trace"])
    need(fixed.sha(path) == entry["trace_sha256"], "trace changed")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    def extract(kind):
        matches = [r["payload"]["raw"] for r in rows if r["kind"] == kind]
        need(len(matches) == 1, "candidate event cardinality")
        values = fixed.f32(matches[0], (8, 3, 5, 2))
        need((np.abs(values) <= 1).all(), "candidate bounds")
        return values
    return extract("initial_candidates"), extract("optimized_candidates"), entry["scores"]


def replay(model, device, task, actions):
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    z = t(task["root_context"]["latent_history"])[None]
    past = t(task["root_context"]["action_history"])[None]
    goal, mean, std = t(task["goal"]), t(task["action_mean"]), t(task["action_std"])
    need(bool((std > 0).all().item()), "normalizer")
    for packet in actions:
        token = ((t(packet) - mean) / std).reshape(1, 1, 10)
        context = torch.cat((past, token), dim=1)
        pred = model.predict(z, model.action_encoder(context))
        need(pred.shape == z.shape and bool(torch.isfinite(pred).all().item()), "prediction")
        z = torch.cat((z, pred[:, -1:]), dim=1)[:, -3:]
        past = context[:, -2:]
    value = float((z[0, -1] - goal).square().mean().item())
    need(np.isfinite(value), "score finite")
    return value


def run(output):
    start = time.monotonic()
    output = Path(output)
    need(output.is_absolute() and output.parent == fixed.ROOT / "results"
         and output.name.startswith(NAME + "_") and not os.path.lexists(output), "exclusive output")
    prepared, report, nodes = fixed.saved_inputs()
    banks = [trace_candidates(n, report) for n in nodes]
    code_sha = fixed.sha(Path(__file__))
    output.mkdir()
    result = {"schema": NAME, "status": "failed", "records": [],
              "source_report_sha256": fixed.REPORT_SHA, "code_sha256": code_sha,
              "P1_admission": False, "fairness_proven": False, "physical_execution": False}
    handles = []
    calls = {"action_encoder": 0, "predictor": 0}
    try:
        with locking_source.locking.gpu_lock(locking_source.LOCK):
            host = gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = checkpoint_compat.load_compatible_pretrained(
                prepared["source"]["policy"], device=device).eval().requires_grad_(False)
            before = fixed.prior.signal.model_hash(model)
            need(before == report["model_before"] and before["original_state_dict"] == fixed.MODEL_SHA,
                 "model identity")
            for key in calls:
                def count(_module, _args, name=key):
                    calls[name] += 1
                handles.append(getattr(model, key).register_forward_pre_hook(count))
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                for node, (initial, optimized, original) in zip(nodes, banks, strict=True):
                    need(time.monotonic() - start < 1800, "wall cap")
                    first = [replay(model, device, node["task"], action) for action in initial]
                    last = [replay(model, device, node["task"], action) for action in optimized]
                    need(np.allclose(last, original, rtol=1e-4, atol=1e-5), "saved scores mismatch")
                    result["records"].append(dict(arm=node["arm"], boundary=node["boundary"],
                        initial=first, optimized=last,
                        delta=(np.asarray(last)-np.asarray(first)).tolist(),
                        saved_scores=original, task_sha256=node["task_sha256"]))
            torch.cuda.synchronize(device)
            need(fixed.prior.signal.model_hash(model) == before and fixed.sha(Path(__file__)) == code_sha
                 and fixed.sha(fixed.RUN / "report.json") == fixed.REPORT_SHA, "identity drift")
            need(calls == {"action_encoder": 768, "predictor": 768}, "actual calls")
            result.update(status="complete", host=host, model_hash=before)
    except BaseException as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        for handle in handles:
            handle.remove()
        result.update(actual_module_calls=calls, wall_seconds=time.monotonic()-start)
        path = output / "report.json"
        with path.open("x") as f:
            json.dump(result, f, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({"status": result["status"], "sha256": fixed.sha(path),
                          "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args().output))
