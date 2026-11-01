"""Prespecified 32-step diagnostic continuation from the same eight starts."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from . import codex_p1_blind_uphill_bank_20261101_exclusive as blind

NAME = "codex_p1_bounded_joint_diag32_20261101_exclusive"
ROOT = blind.ROOT
UPDATES = 32
RATE = 0.05
PENALTY = 10.0
MAX_CALLS = 1584
OLD_REPORT = ROOT / "results/codex_p1_bounded_joint_20261101_exclusive_RUN_CODEXJOINT01/report.json"
OLD_SHA = "995aba2fa6a7618450395b4db7e1820cc0e295ff07cc483a9dbb7d69e71ff909"


def choose(root, rows):
    if not np.isfinite(root):
        raise ValueError("nonfinite root")
    eligible = []
    for row in rows:
        if not np.isfinite([row["q1"], row["q3"]]).all():
            raise ValueError("nonfinite score")
        if row["q1"] >= root and row["q3"] < root:
            eligible.append(row)
    return min(eligible, key=lambda x: (x["q3"], x["candidate"], x["iteration"])) if eligible else None


def optimize(initial, evaluate, reverse, record):
    import torch
    if initial.shape != (3, 5, 2) or not bool(torch.isfinite(initial).all()) or not bool((initial.abs() <= 1).all()):
        raise ValueError("invalid initial action")
    action = initial.detach().clone()
    for iteration in range(UPDATES + 1):
        action = action.detach().requires_grad_(iteration < UPDATES)
        root, q1, q3 = evaluate(action, iteration)
        loss = q3 + PENALTY * torch.relu(root - q1).square()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite loss")
        gradient = None
        if iteration < UPDATES:
            gradient = reverse(loss, action, iteration)
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("nonfinite gradient")
        record(iteration, action, q1, q3, loss, gradient)
        if iteration < UPDATES:
            action = (action - RATE * gradient).clamp(-1, 1).detach()


def protocol():
    previous = blind.protocol()
    dependencies = dict(previous["dependency_sha256"])
    dependencies[str(Path(blind.__file__).resolve())] = blind.host.sha(Path(blind.__file__).resolve())
    blind.host.require(blind.host.sha(OLD_REPORT) == OLD_SHA, "old report drift")
    return {
        "schema": NAME + ".protocol.v1", "inputs": previous["inputs"],
        "code_sha256": blind.host.sha(Path(__file__).resolve()), "dependency_sha256": dependencies,
        "algorithm": {"roots": [0, 1], "candidates_per_root": 8, "updates": UPDATES,
                      "rate": RATE, "penalty": PENALTY, "loss": "q3+10*relu(q0-q1)^2",
                      "projection": "clamp each action to [-1,1]", "precision": "float32; TF32 off; math SDPA"},
        "selection": "q1>=q0 and q3<q0; min(q3,candidate,iteration); otherwise no selection",
        "prior_report_sha256": OLD_SHA,
        "rationale": "fixed 4x step budget: old mean loss still decreases at step 8, with zero boundary saturation; not a convergence claim",
        "budget": {"predict_attempts": MAX_CALLS, "action_encoder_attempts": MAX_CALLS,
                   "reverse_attempts": 512, "wall_seconds": 1800, "event_bytes_max": 16000000},
        "scope": "known development roots and previously generated candidates; no physical future or training",
        "physical_rule": "no physical execution by this entrypoint; separate source-authenticated binding required",
        "cost_boundary": "new model API calls and reverse requests only; old search, historical candidate generation and RGB encoding excluded",
        "P1_admission": False,
    }


def run(prep, expected, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    host = blind.host
    host.require(host.sha(prep) == expected and json.loads(prep.read_bytes()) == protocol(), "protocol drift")
    output.mkdir()
    events = output / "events.jsonl"
    counts = {"action_encoder": 0, "predict": 0, "reverse": 0}
    returned = dict(counts)
    hooks = {"action_encoder": 0, "predict": 0}
    started = time.monotonic()

    def emit(obj):
        payload = (json.dumps(obj, sort_keys=True, allow_nan=False) + "\n").encode()
        host.require(time.monotonic() - started < 1800, "wall limit")
        host.require((events.stat().st_size if events.exists() else 0) + len(payload) <= 16000000, "log limit")
        with events.open("ab") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

    def call(kind, fn, context):
        limit = 512 if kind == "reverse" else MAX_CALLS
        host.require(counts[kind] < limit, "call limit")
        counts[kind] += 1
        emit({"event": "attempt", "kind": kind, **context})
        value = fn()
        returned[kind] += 1
        emit({"event": "return", "kind": kind, **context})
        return value

    with blind.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        gpu = require_cuda_allocation("cuda:0")
        data = [blind.load(i) for i in (0, 1)]
        host.require(data[0][3] == data[1][3], "policy mismatch")
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(data[0][3], device="cuda:0").eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        host.require(before["original_state_dict"] == "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546", "model hash")
        # Each module class must be among the authenticated source closure of both roots.
        bindings = []
        for slot in (0, 1):
            prepared = json.loads((ROOT / "results" / blind.ITEMS[slot][0]).read_bytes())
            bindings.append(prepared["sources"])
        for module in model.modules():
            filename = str(Path(inspect.getfile(type(module))).resolve())
            host.require(all(filename in b and host.sha(Path(filename)) == b[filename] for b in bindings),
                         "unauthenticated model class: " + filename)

        def hook(kind):
            def count(_module, _args):
                hooks[kind] += 1
            return count
        handles = [model.predictor.register_forward_pre_hook(hook("predict")),
                   model.action_encoder.register_forward_pre_hook(hook("action_encoder"))]
        rows = []
        try:
            with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                for slot, (task, bank, _, _) in enumerate(data):
                    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device="cuda:0")
                    states = tensor(task["root_context"]["latent_history"])[None]
                    history = tensor(task["root_context"]["action_history"])[None]
                    goal, mean, std = tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"])
                    host.require(bool((std > 0).all()), "normalizer")
                    root = (states[:, -1] - goal).square().mean()
                    root_value = float(root.item())
                    candidates = []
                    for candidate, plan in enumerate(bank):
                        def evaluate(action, iteration):
                            z, past = states, history
                            scores = []
                            for packet in range(3):
                                context = torch.cat((past, ((action[packet] - mean) / std).reshape(1, 1, 10)), 1)
                                loc = {"slot": slot, "candidate": candidate, "iteration": iteration, "packet": packet}
                                encoded = call("action_encoder", lambda: model.action_encoder(context), loc)
                                pred = call("predict", lambda: model.predict(z, encoded), loc)
                                host.require(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "model output")
                                z = torch.cat((z, pred[:, -1:]), 1)[:, -3:]
                                past = context[:, -2:]
                                scores.append((z[:, -1] - goal).square().mean())
                            return root, scores[0], scores[-1]
                        def reverse(loss, action, iteration):
                            return call("reverse", lambda: torch.autograd.grad(loss, action)[0],
                                        {"slot": slot, "candidate": candidate, "iteration": iteration})
                        def record(iteration, action, q1, q3, loss, gradient):
                            raw = action.detach()
                            stats = None
                            if gradient is not None:
                                g = gradient.detach()
                                proposal = raw - RATE * g
                                stats = {"l2": float(g.norm().item()), "max_abs": float(g.abs().max().item()),
                                         "packet_l2": [float(x.norm().item()) for x in g],
                                         "clipped_coordinates": int((proposal.abs() > 1).sum().item()),
                                         "projected_step_l2": float((proposal.clamp(-1, 1) - raw).norm().item())}
                            row = {"candidate": candidate, "iteration": iteration, "q1": float(q1.item()),
                                   "q3": float(q3.item()), "loss": float(loss.item()), "gradient": stats,
                                   "saturated_coordinates": int((raw.abs() >= 1).sum().item()),
                                   "actions": raw.cpu().tolist()}
                            candidates.append(row)
                            emit({"event": "score", "slot": slot, **row})
                        optimize(tensor(plan), evaluate, reverse, record)
                    selected = choose(root_value, candidates)
                    rows.append({"slot": slot, "root_mse": root_value, "evaluations": candidates, "selected": selected})
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.synchronize()
        after = host.signal.model_hash(model)
        host.require(before == after and counts == returned == {"predict": MAX_CALLS, "action_encoder": MAX_CALLS, "reverse": 512}
                     and hooks == {"predict": MAX_CALLS, "action_encoder": MAX_CALLS}, "incomplete calls or model mutation")
        host.require(host.sha(prep) == expected and json.loads(prep.read_bytes()) == protocol(), "post-run source drift")
        old = json.loads(OLD_REPORT.read_bytes())
        differences = []
        for newrow, oldrow in zip(rows, old["rows"], strict=True):
            for candidate in range(8):
                for iteration in range(9):
                    new = newrow["evaluations"][candidate * 33 + iteration]
                    prior = oldrow["evaluations"][candidate * 9 + iteration]
                    differences.append(max(abs(new["q1"] - prior["q1"]), abs(new["q3"] - prior["q3"]),
                                           float(np.max(np.abs(np.asarray(new["actions"]) - np.asarray(prior["actions"]))))))
        result = {"schema": NAME + ".report.v1", "protocol_sha256": expected, "rows": rows,
                  "attempts": counts, "returns": returned, "module_hooks": hooks, "gpu": gpu,
                  "model_before": before, "model_after": after, "events_sha256": host.sha(events),
                  "old_first_nine_max_abs_difference": max(differences),
                  "physical_execution": False, "physical_future_used": False, "training": False,
                  "historical_candidate_and_rgb_cost_included": False, "P1_admission": False}
        return blind.write_new(output / "report.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    blind.host.require(args.tag.isalnum(), "tag")
    prep = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    blind.host.require(not prep.exists() and not output.exists() and not prep.is_symlink() and not output.is_symlink(), "exclusive output")
    sha = blind.write_new(prep, protocol())
    print(json.dumps({"protocol": str(prep), "sha256": sha}), flush=True)
    if args.run:
        print(json.dumps({"report_sha256": run(prep, sha, output), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
