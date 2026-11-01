"""Unified model-only constrained CEM search on the two known P1 roots."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from . import codex_p1_blind_uphill_bank_20261101_exclusive as blind

ROOT = blind.ROOT
NAME = "p1_unified_constrained_cem_20261101_codex"
ITERATIONS = 5
SAMPLES = 64
ELITES = 8
HORIZON = 3
SEED = 20261101
INITIAL_STD = 0.45
MIN_STD = 0.05
MAX_CALLS = 2 * ITERATIONS * SAMPLES * HORIZON


def candidate_key(row):
    """Prefer non-greedy prefixes, then terminal goal cost."""
    values = [row.get("violation"), row.get("q3"), row.get("iteration"), row.get("candidate")]
    if not np.isfinite(values).all():
        raise ValueError("finite candidate scores required")
    feasible = row["violation"] == 0.0
    return (0 if feasible else 1, row["q3"] if feasible else row["violation"],
            row["q3"], row["iteration"], row["candidate"])


def choose(root, rows):
    if not np.isfinite(root):
        raise ValueError("finite root score required")
    eligible = [
        row for row in rows
        if row["q1"] >= root and row["q3"] < root
    ]
    return min(eligible, key=lambda row: (row["q3"], row["iteration"], row["candidate"])) if eligible else None


def make_bank(rng, mean, std, initial, iteration):
    if mean.shape != (HORIZON, 5, 2) or std.shape != mean.shape:
        raise ValueError("invalid CEM state")
    bank = np.clip(rng.normal(mean, std, (SAMPLES, *mean.shape)), -1, 1).astype(np.float32)
    bank[0] = mean
    if iteration == 0:
        if initial.shape != (8, HORIZON, 5, 2):
            raise ValueError("invalid initial DLS bank")
        bank[1:9] = initial
    return bank


def update(bank, rows):
    if bank.shape != (SAMPLES, HORIZON, 5, 2) or len(rows) != SAMPLES:
        raise ValueError("invalid CEM bank or rows")
    order = sorted(range(SAMPLES), key=lambda index: candidate_key(rows[index]))
    elites = bank[order[:ELITES]]
    mean = elites.mean(axis=0, dtype=np.float32)
    std = np.clip(elites.std(axis=0, dtype=np.float32), MIN_STD, 1.0).astype(np.float32)
    return mean, std, order[:ELITES]


def write_new(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def protocol():
    inputs = []
    for slot in (0, 1):
        task, bank, sources, policy = blind.load(slot)
        inputs.append({
            "slot": slot,
            "task_sha256": blind.host.digest(task),
            "initial_bank_sha256": hashlib.sha256(bank.astype("<f4").tobytes()).hexdigest(),
            "sources": sources,
            "policy": policy,
        })
    dependencies = (blind, blind.host, blind.chain, blind.locking, blind.host.p3,
                    blind.host.rgb.moving, blind.host.checkpoint_compat, blind.host.gpu_preflight)
    return {
        "schema": NAME + ".protocol.v1",
        "inputs": inputs,
        "code_sha256": blind.host.sha(Path(__file__).resolve()),
        "dependency_sha256": {
            str(Path(module.__file__).resolve()): blind.host.sha(Path(module.__file__).resolve())
            for module in dependencies
        },
        "algorithm": {
            "roots": [0, 1],
            "iterations": ITERATIONS,
            "samples": SAMPLES,
            "elites": ELITES,
            "horizon_packets": HORIZON,
            "seed_base": SEED,
            "initial_std": INITIAL_STD,
            "minimum_std": MIN_STD,
            "iteration0": "candidate0 mean; candidates1:9 authenticated DLS H3 bank; remainder Gaussian",
            "elite_order": "feasible q1>=q0 first by q3; otherwise prefix violation then q3",
            "selection": "among q1>=q0 and q3<q0 choose minimum q3, then iteration and candidate",
            "action_projection": "clip raw actions to [-1,1]",
        },
        "budget": {
            "predict_attempts": MAX_CALLS,
            "action_encoder_attempts": MAX_CALLS,
            "wall_seconds": 1800,
            "event_bytes_max": 32_000_000,
        },
        "scope": "known development roots; model-only; no physical future or training",
        "historical_candidate_generation_cost_included": False,
        "physical_execution": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }


def run(prepared, expected_sha, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    host = blind.host
    host.require(host.sha(prepared) == expected_sha and json.loads(prepared.read_bytes()) == protocol(),
                 "protocol drift")
    output.mkdir()
    events = output / "events.jsonl"
    attempts = {"predict": 0, "action_encoder": 0}
    returns = dict(attempts)
    hooks = dict(attempts)
    started = time.monotonic()

    def emit(event):
        payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        host.require(time.monotonic() - started < 1800, "wall limit")
        host.require((events.stat().st_size if events.exists() else 0) + len(payload) <= 32_000_000,
                     "event log limit")
        with events.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def call(kind, fn, context):
        host.require(attempts[kind] < MAX_CALLS, "call limit")
        attempts[kind] += 1
        emit({"event": "attempt", "kind": kind, **context})
        value = fn()
        returns[kind] += 1
        emit({"event": "return", "kind": kind, **context})
        return value

    with blind.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        gpu = require_cuda_allocation("cuda:0")
        data = [blind.load(slot) for slot in (0, 1)]
        host.require(data[0][3] == data[1][3], "policy mismatch")
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(data[0][3], device="cuda:0").eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        host.require(before["original_state_dict"]
                     == "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546",
                     "model hash")
        source_closures = [
            json.loads((ROOT / "results" / blind.ITEMS[slot][0]).read_bytes())["sources"]
            for slot in (0, 1)
        ]
        for module in model.modules():
            filename = str(Path(inspect.getfile(type(module))).resolve())
            host.require(all(filename in closure and host.sha(Path(filename)) == closure[filename]
                             for closure in source_closures),
                         "unauthenticated model class: " + filename)

        def hook(kind):
            def count(_module, _args):
                hooks[kind] += 1
            return count

        handles = [
            model.predictor.register_forward_pre_hook(hook("predict")),
            model.action_encoder.register_forward_pre_hook(hook("action_encoder")),
        ]
        result_rows = []
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                for slot, (task, initial, _, _) in enumerate(data):
                    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device="cuda:0")
                    states = tensor(task["root_context"]["latent_history"])[None]
                    history = tensor(task["root_context"]["action_history"])[None]
                    goal = tensor(task["goal"])[None]
                    mean_action = tensor(task["action_mean"])
                    std_action = tensor(task["action_std"])
                    host.require(bool((std_action > 0).all()), "normalizer")
                    root = float((states[:, -1] - goal).square().mean().item())
                    rng = np.random.default_rng(SEED + slot)
                    mean = np.zeros((HORIZON, 5, 2), np.float32)
                    std = np.full_like(mean, INITIAL_STD)
                    all_rows = []
                    iterations = []
                    for iteration in range(ITERATIONS):
                        bank = make_bank(rng, mean, std, initial, iteration)
                        rows = []
                        for candidate, raw in enumerate(bank):
                            z, past = states, history
                            scores = []
                            for packet in range(HORIZON):
                                context = torch.cat(
                                    (past, ((tensor(raw[packet]) - mean_action) / std_action).reshape(1, 1, 10)),
                                    dim=1,
                                )
                                identity = {
                                    "slot": slot,
                                    "iteration": iteration,
                                    "candidate": candidate,
                                    "packet": packet,
                                }
                                encoded = call("action_encoder", lambda: model.action_encoder(context), identity)
                                predicted = call("predict", lambda: model.predict(z, encoded), identity)
                                host.require(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()),
                                             "model output")
                                z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                                past = context[:, -2:]
                                scores.append(float((z[:, -1] - goal).square().mean().item()))
                            row = {
                                "iteration": iteration,
                                "candidate": candidate,
                                "q1": scores[0],
                                "q3": scores[-1],
                                "violation": max(0.0, root - scores[0]),
                                "actions": raw.tolist(),
                            }
                            rows.append(row)
                            all_rows.append(row)
                            emit({"event": "score", "slot": slot, **row})
                        mean, std, elite_indices = update(bank, rows)
                        iterations.append({
                            "iteration": iteration,
                            "elite_indices": elite_indices,
                            "mean": mean.tolist(),
                            "std": std.tolist(),
                        })
                    selected = choose(root, all_rows)
                    result_rows.append({
                        "slot": slot,
                        "root_mse": root,
                        "eligible_count": sum(row["q1"] >= root and row["q3"] < root for row in all_rows),
                        "selected": selected,
                        "iterations": iterations,
                    })
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.synchronize()
        after = host.signal.model_hash(model)
        expected_counts = {"predict": MAX_CALLS, "action_encoder": MAX_CALLS}
        host.require(before == after and attempts == returns == hooks == expected_counts,
                     "incomplete calls or model mutation")
        host.require(host.sha(prepared) == expected_sha and json.loads(prepared.read_bytes()) == protocol(),
                     "post-run source drift")
        report = {
            "schema": NAME + ".report.v1",
            "status": "model_only_complete",
            "protocol_sha256": expected_sha,
            "rows": result_rows,
            "attempts": attempts,
            "returns": returns,
            "module_hooks": hooks,
            "host": gpu,
            "model_before": before,
            "model_after": after,
            "events_sha256": host.sha(events),
            "physical_execution": False,
            "physical_future_used": False,
            "training": False,
            "historical_candidate_generation_cost_included": False,
            "independent_confirmation": False,
            "P1_admission": False,
        }
        return write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    blind.host.require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    blind.host.require(not prepared.exists() and not output.exists()
                       and not prepared.is_symlink() and not output.is_symlink(),
                       "exclusive outputs")
    digest = write_new(prepared, protocol())
    print(json.dumps({"protocol": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        print(json.dumps({"report_sha256": run(prepared, digest, output), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
