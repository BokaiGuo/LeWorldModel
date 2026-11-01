"""Known-root real-cp5 RGB H3 CEM candidate; not an official reproduction."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import p1_coherent_own_suffix_20261101_codex as own

NAME = "p1_cp5_cem_baseline_20261101_leibniz"
ROOT = own.ROOT
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
AUTH = "MAIN_AUTHORIZES_CP5_CEM_HOST_CUDA"
CONFIG = dict(prefix="structured", horizon=3, packet_steps=5, iterations=4,
              samples=16, elites=4, seed=20261101, rng="numpy.PCG64",
              initial_mean=0., initial_std=0.5, std_min=0.05, std_max=1.,
              inject_mean_at_index=0, bound=1., selection="global_cost_then_index",
              objective="existing_centered_quadratic_Q", fresh_winner_replay=1)


def costs():
    v = dict.fromkeys(own.spec.AXES, 0)
    v.update(encode_calls=1, encode_samples=1, action_encode_calls=195,
             predict_calls=195, predict_samples=195, replay_calls=65,
             selection_evaluations=65)
    return v


def optimize(score, emit):
    """4x16 causal CEM, immutable banks before scoring and stable global winner."""
    rng = np.random.default_rng(CONFIG["seed"])
    mean = np.zeros((3, 5, 2), np.float32)
    std = np.full_like(mean, CONFIG["initial_std"])
    candidates, scores, updates = [], [], []
    for iteration in range(CONFIG["iterations"]):
        bank = np.clip(rng.normal(mean, std, (16, 3, 5, 2)), -1, 1).astype(np.float32)
        bank[0] = mean
        emit("bank", dict(iteration=iteration, mean=mean.tolist(), std=std.tolist(), raw=bank.tolist()))
        values = []
        for raw in bank:
            value = float(score(raw))
            own.require(np.isfinite(value), "nonfinite CEM score")
            values.append(value)
            candidates.append(raw.copy())
            scores.append(value)
        elites = sorted(range(16), key=lambda i: (values[i], i))[:4]
        mean = bank[elites].mean(axis=0, dtype=np.float32)
        std = np.clip(bank[elites].std(axis=0, dtype=np.float32), .05, 1.).astype(np.float32)
        update = dict(iteration=iteration, scores=values, elite_indices=elites,
                      mean=mean.tolist(), std=std.tolist())
        updates.append(update)
        emit("update", update)
    index = min(range(64), key=lambda i: (scores[i], i))
    chosen = candidates[index]
    emit("selected_before_replay", dict(index=index, score=scores[index], raw=chosen.tolist()))
    repeated = float(score(chosen))
    own.require(np.isfinite(repeated) and np.isclose(repeated, scores[index], rtol=1e-6, atol=1e-8),
                "selected replay mismatch")
    return dict(selected_index=index, selected_raw=chosen.tolist(), scores=scores,
                replay_score=repeated, updates=updates)


def contract_for(task, model_sha, runner):
    _, contract, _ = own.node_contract(task, model_sha, runner)
    contract = deepcopy(contract)
    optimizer = own.sha(Path(__file__))
    config_hash = own.spec.digest(CONFIG)
    contract["identity"]["optimizer"] = optimizer
    contract["identity"]["optimizer_config"] = config_hash
    for arm in contract["arms"].values():
        arm["optimizer_sha256"] = optimizer
        arm["optimizer_config_sha256"] = config_hash
    cap = costs()
    contract["levels"] = [dict(id="node", caps=cap),
                          dict(id="unused_larger", caps={k: v + 1 for k, v in cap.items()})]
    own.spec.preflight(contract, own.spec.digest(contract))
    return contract


def build(runner):
    base = own.build(runner)
    own.require([r["prefix"] for r in base["rows"]] == list(own.PREFIXES), "actual source roster")
    rows = [r for r in base["rows"] if r["prefix"] == CONFIG["prefix"]]
    own.require(len(rows) == 1, "unique structured cp5")
    sources = dict(base["sources"])
    for path in (Path(__file__).resolve(), DOC, TEST):
        sources[str(path)] = own.sha(path)
    own.require(all(own.sha(p) == h for p, h in sources.items()), "source closure")
    return dict(schema=NAME + ".prepare.v1", config=CONFIG, source=base, row=rows[0],
                sources=sources, caps=costs(), physical_execution=False,
                official_reproduction=False, fairness_proven=False, P1_admission=False)


def exclusive(path):
    path = Path(path)
    own.require(path.is_absolute() and path.parent == ROOT / "results"
                and path.name.startswith(NAME + "_") and not path.exists()
                and not path.is_symlink(), "exclusive path")
    return path


def verify(path, digest, runner):
    path = Path(path)
    own.require(path.is_absolute() and path.parent == ROOT / "results" and not path.is_symlink()
                and own.spec.is_hash(digest) and own.sha(path) == digest, "prepare identity")
    value = json.loads(path.read_bytes())
    own.require(value == build(runner), "prepare/source drift")
    return value


def run(path, digest, output, authorization, experiment, runner):
    own.require(authorization == AUTH, "explicit host CUDA authorization")
    value = verify(path, digest, runner)
    output = exclusive(output)
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        output.mkdir()
        binding = own.write(output / "binding.json", value)
        model = driver = trace = None
        before = after = failure = result = ledger = account = host = task = latent = None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = own.gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                model = own.checkpoint_compat.load_compatible_pretrained(
                    value["source"]["model_policy"], device=device).eval().requires_grad_(False)
                before = runner.signal.model_hash(model)
                own.require(before == value["source"]["expected_model_hash"], "model identity")
                row, base = value["row"], value["source"]
                from PIL import Image
                image_path = Path(row["png"])
                own.require(own.sha(image_path) == row["png_sha256"], "cp5 file")
                with Image.open(image_path) as image:
                    pixels = np.asarray(image).copy()
                import hashlib
                own.require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"], "cp5 pixels")
                placeholder = own.make_task(base["base_task"], row, [0.] * 192, base["warmup_raw"], runner)
                contract = contract_for(placeholder, base["model_sha256"], runner)
                contract_sha = own.spec.digest(contract)
                driver = own.durable.Driver(output / "ledger", contract, contract_sha,
                                             "structured", "node", NAME)
                leaf = own.leafmod.LeafAdapter(model, driver, contract, contract_sha)
                trace = own.journal.Journal(output / "trace.jsonl", 16 * 1024**2)
                image = own.p0._image_tensor(pixels, transform=own.p0.image_transform(), device=device)
                with torch.inference_mode():
                    latent = leaf.encode({"pixels": image}, phase="setup")["emb"][0, 0].cpu().numpy()
                    task = own.make_task(base["base_task"], row, latent, base["warmup_raw"], runner)
                    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                    states = tensor(task["root_context"]["latent_history"])[None]
                    history = tensor(task["root_context"]["action_history"])[None]
                    goal = tensor(task["goal"])[None]
                    mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                    matrix = torch.as_tensor(own.coherent.centered.h4.reverse.Q.kernels.MATRIX,
                                             dtype=torch.float64, device=device)
                    def score(raw):
                        leaf._dispatch("verification", "replay", {"replay_calls": 1}, lambda: None)
                        z, past = states, history
                        for packet in tensor(raw):
                            token = ((packet - mean) / std).reshape(1, 1, 10)
                            context = torch.cat((past, token), 1)
                            pred = leaf.predict(z, leaf.action_encode(context, phase="verification"), phase="verification")
                            own.require(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "prediction")
                            z = torch.cat((z, pred[:, -1:]), 1)[:, -3:]
                            past = context[:, -2:]
                        def objective():
                            residual = (z[:, -1] - goal).double()
                            return float(((residual @ matrix.T).square().mean()).item())
                        return leaf._dispatch("verification", "selection", {"selection_evaluations": 1}, objective)
                    result = optimize(score, trace.emit)
                torch.cuda.synchronize(device)
                after = runner.signal.model_hash(model)
                own.require(before == after, "model drift")
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
        finally:
            try:
                if trace is not None:
                    trace.close()
                if driver is not None:
                    ledger = driver.finish()
                    account = own.consumer.consume(ledger["journal"], ledger["sha256"], contract_sha,
                                                    "structured", "node", NAME)
            except BaseException as exc:
                failure = failure or dict(type=type(exc).__name__, message="ledger finalization: " + str(exc)[:900])
            try:
                source_ok = all(own.sha(p) == h for p, h in value["sources"].items())
            except (OSError, ValueError):
                source_ok = False
            complete = (failure is None and source_ok and result is not None and before == after
                        and account is not None and account["status"] == "closed_unverified"
                        and account["charged_upper_bounds"] == costs())
            report = dict(schema=NAME + ".report.v1", status="complete" if complete else "partial",
                failure=failure, prepared=str(path), prepare_sha256=digest, binding_sha256=binding,
                result=result, task=task, cp5_latent=latent.tolist() if latent is not None else None,
                ledger=ledger, account=account, host=host, model_before=before,
                model_after=after, source_verified=source_ok, elapsed_seconds=time.monotonic()-started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                trace_sha256=own.sha(output / "trace.jsonl") if trace is not None else None,
                physical_execution=False, official_reproduction=False, fairness_proven=False, P1_admission=False)
            report_sha = own.write(output / "report.json", report)
        return dict(status=report["status"], report_sha256=report_sha, failure=failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    for arg in ("prepared", "sha256", "output", "authorization"):
        execute.add_argument("--" + arg, required=True)
    execute.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    experiment, runner = own.configure()
    if args.command == "dry-prepare":
        value = build(runner)
        task = own.make_task(value["source"]["base_task"], value["row"], [0.] * 192,
                             value["source"]["warmup_raw"], runner)
        contract = contract_for(task, value["source"]["model_sha256"], runner)
        result = dict(status="validated_no_model_or_physics", digest=own.spec.digest(value),
                      caps=value["caps"], contract_sha256=own.spec.digest(contract),
                      prefix=value["row"]["prefix"])
    elif args.command == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=own.write(path, build(runner)))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization, experiment, runner)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
