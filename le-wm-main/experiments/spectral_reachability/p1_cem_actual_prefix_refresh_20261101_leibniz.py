"""Fixed CEM actions: open-loop versus actual-prefix one-packet prediction."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import p1_cp5_cem_rgb_mismatch_20261101_leibniz as prev

NAME = "p1_cem_actual_prefix_refresh_20261101_leibniz"
ROOT = prev.ROOT
own, family = prev.own, prev.family
RUN = ROOT / "results/p1_cp5_cem_rgb_mismatch_20261101_leibniz_RUN_MAIN02"
REPORT_SHA = "9445dcb9664d6681d8754eeca742a9994a2b33e41abdcc598d0e342725040a72"
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
AUTH = "MAIN_AUTHORIZES_P1_CEM_ACTUAL_PREFIX_REFRESH_CUDA"


def costs():
    value = dict.fromkeys(own.spec.AXES, 0)
    value.update(predict_calls=6, predict_samples=6, action_encode_calls=6)
    return value


def simulate(leaf, states, history, raw, actual, mean, std):
    """Predict three packets for each arm with equal fixed actions and context."""
    family.require(states.shape == (1, 3, 192) and history.shape == (1, 2, 10)
                   and raw.shape == (3, 5, 2) and actual.shape == (4, 192)
                   and mean.shape == std.shape == (2,) and bool((std > 0).all())
                   and all(bool(torch.isfinite(t).all()) for t in (states, history, raw, actual, mean, std))
                   and bool((raw.abs() <= 1).all()), "input shape or bounds")
    out = {}
    for arm in ("open", "actual_prefix"):
        z, past = states.clone(), history.clone()
        predictions = []
        for index, packet in enumerate(raw):
            token = ((packet - mean) / std).reshape(1, 1, 10)
            context = torch.cat((past, token), 1)
            pred = leaf.predict(z, leaf.action_encode(context, phase="verification"), phase="verification")
            family.require(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "invalid prediction")
            predictions.append(pred[0, -1].detach().cpu().numpy().copy())
            next_state = pred[:, -1:] if arm == "open" else actual[index + 1].reshape(1, 1, 192)
            z = torch.cat((z, next_state), 1)[:, -3:]
            past = context[:, -2:]
        out[arm] = np.stack(predictions)
    return out


def contract():
    value = deepcopy(prev.contract())
    optimizer, cfg = family.sha(Path(__file__)), own.spec.digest({"name": NAME, "caps": costs()})
    value["identity"].update(optimizer=optimizer, optimizer_config=cfg)
    for arm in value["arms"].values():
        arm.update(optimizer_sha256=optimizer, optimizer_config_sha256=cfg)
    cap = costs()
    value["levels"] = [dict(id="node", caps=cap),
                       dict(id="unused_larger", caps={k: v + 1 for k, v in cap.items()})]
    own.spec.preflight(value, own.spec.digest(value))
    return value


def build():
    family.require(family.sha(RUN / "report.json") == REPORT_SHA, "prior diagnostic report hash")
    report = json.loads((RUN / "report.json").read_bytes())
    family.require(report["schema"] == prev.NAME + ".report.v1" and report["status"] == "complete"
                   and report["failure"] is None and report["source_verified"] is True
                   and report["physical_execution"] is False and report["P1_admission"] is False
                   and report["model_before"] == report["model_after"], "prior status")
    prepared = Path(report["prepared"])
    family.require(family.sha(prepared) == report["prepare_sha256"] == report["binding_sha256"]
                   and family.sha(RUN / "binding.json") == report["binding_sha256"], "prior prepare/binding")
    value = json.loads(prepared.read_bytes())
    family.require(value == json.loads((RUN / "binding.json").read_bytes()), "binding bytes")
    family.require(value == prev.build(), "prior source/preparation reconstruction")
    family.verify_sources(value["sources"])
    ledger = report["ledger"]
    journal = RUN / "ledger/events.jsonl"
    family.require(Path(ledger["journal"]) == journal and family.sha(journal) == ledger["sha256"], "prior ledger")
    prior_contract = json.loads(journal.read_text().splitlines()[0])["payload"]["contract"]
    prior_sha = own.spec.digest(prior_contract)
    account = own.consumer.consume(journal, ledger["sha256"], prior_sha, "structured", "node", prev.NAME)
    family.require(account == report["account"] and account["status"] == "closed_unverified"
                   and account["charged_upper_bounds"] == prev.costs(), "prior ledger accounting")
    family.require(report["model_before"] == value["model_hash"], "same frozen model")
    result = report["result"]
    actual = np.asarray(result["actual_latents"], np.float32)
    predicted = np.asarray(result["predicted_latents"], np.float32)
    family.require(actual.shape == (4, 192) and predicted.shape == (3, 192)
                   and np.isfinite(actual).all() and np.isfinite(predicted).all()
                   and np.allclose(actual[0], np.asarray(value["cp5_latent"], np.float32), rtol=1e-5, atol=1e-6)
                   and np.isclose(result["replay_terminal_q"], value["original_terminal_q"], rtol=1e-6, atol=1e-8),
                   "prior latent/terminal binding")
    task = value["task"]
    family.require(np.array_equal(np.asarray(task["root_context"]["latent_history"][-1], np.float32),
                                   np.asarray(value["cp5_latent"], np.float32)), "context cp5")
    family.actions(value["actual_actions"], 20)
    family.require([p["step"] for p in value["images"]] == [5, 10, 15, 20], "RGB time roster")
    sources = dict(value["sources"])
    for path in (RUN / "report.json", RUN / "binding.json", prepared, journal,
                 Path(__file__).resolve(), DOC, TEST):
        family.merge_sources(sources, {str(path): family.sha(path)})
    family.verify_sources(sources)
    return dict(schema=NAME + ".prepare.v1", prior_report_sha256=REPORT_SHA,
                prior_prepare_sha256=report["prepare_sha256"], task=task,
                actual_latents=actual.tolist(), prior_open_latents=predicted.tolist(),
                physical_metrics=[p["metrics"] for p in value["images"][1:]],
                actual_actions=value["actual_actions"], model_policy=value["model_policy"],
                model_hash=value["model_hash"], original_terminal_q=value["original_terminal_q"],
                sources=sources, costs=costs(), P1_admission=False, independent_confirmation=False,
                physical_execution=False, neural_execution=False)


def exclusive(path):
    path = Path(path)
    family.require(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_")
                   and not path.exists() and not path.is_symlink(), "exclusive path")
    return path


def run(prepared, expected, output, authorization):
    family.require(authorization == AUTH, "explicit CUDA authorization")
    path = Path(prepared)
    family.require(path.is_absolute() and path.parent == ROOT / "results" and not path.is_symlink()
                   and own.spec.is_hash(expected) and family.sha(path) == expected, "prepare identity")
    value = json.loads(path.read_bytes())
    family.require(value == build(), "prepare/source drift")
    output = exclusive(output)
    extra = prev.launcher.extra
    with extra.gpu.locking.gpu_lock(extra.gpu.LOCK):
        output.mkdir()
        binding_sha = own.write(output / "binding.json", value)
        model = driver = ledger = account = before = after = failure = result = host = None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = own.gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
                model = own.checkpoint_compat.load_compatible_pretrained(value["model_policy"], device=device).eval().requires_grad_(False)
                before = extra.runner.signal.model_hash(model)
                family.require(before == value["model_hash"], "model identity")
                c = contract()
                contract_sha = own.spec.digest(c)
                driver = own.durable.Driver(output / "ledger", c, contract_sha, "structured", "node", NAME)
                leaf = own.leafmod.LeafAdapter(model, driver, c, contract_sha)
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                task = value["task"]
                predictions = simulate(leaf, tensor(task["root_context"]["latent_history"])[None],
                    tensor(task["root_context"]["action_history"])[None],
                    tensor(np.asarray(value["actual_actions"], np.float32)[5:].reshape(3, 5, 2)),
                    tensor(value["actual_latents"]), tensor(task["action_mean"]), tensor(task["action_std"]))
                torch.cuda.synchronize(device)
                family.require(np.allclose(predictions["open"], np.asarray(value["prior_open_latents"]),
                                           rtol=1e-5, atol=1e-6), "prior open rollout mismatch")
                family.require(np.allclose(predictions["open"][0], predictions["actual_prefix"][0],
                                           rtol=1e-6, atol=1e-7), "identical cp5 first prediction")
                matrix = own.coherent.centered.h4.reverse.Q.kernels.MATRIX
                actual = value["actual_latents"][1:]
                rows = {arm: prev.analyze(pred, actual, task["goal"], matrix, value["physical_metrics"])
                        for arm, pred in predictions.items()}
                family.require(np.isclose(rows["open"][-1]["predicted_q"], value["original_terminal_q"],
                                          rtol=1e-6, atol=1e-8), "original terminal Q")
                result = dict(rows=rows, predicted_latents={k: v.tolist() for k, v in predictions.items()},
                              actual_latents=value["actual_latents"])
                after = extra.runner.signal.model_hash(model)
                family.require(before == after, "model drift")
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
        finally:
            try:
                if driver is not None:
                    ledger = driver.finish()
                    account = own.consumer.consume(ledger["journal"], ledger["sha256"], contract_sha,
                                                    "structured", "node", NAME)
            except BaseException as exc:
                failure = failure or dict(type=type(exc).__name__, message="ledger: " + str(exc)[:900])
            try:
                family.verify_sources(value["sources"])
                source_ok = True
            except (OSError, ValueError):
                source_ok = False
            complete = (failure is None and source_ok and before is not None and before == after
                        and result is not None and account is not None and account["status"] == "closed_unverified"
                        and account["charged_upper_bounds"] == costs())
            report = dict(schema=NAME + ".report.v1", status="complete" if complete else "partial",
                failure=failure, prepared=str(path), prepare_sha256=expected, binding_sha256=binding_sha,
                result=result, ledger=ledger, account=account, source_verified=source_ok, host=host,
                model_before=before, model_after=after, elapsed_seconds=time.monotonic()-started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                physical_execution=False, P1_admission=False, independent_confirmation=False)
            digest = own.write(output / "report.json", report)
        return dict(status=report["status"], report_sha256=digest, failure=failure)


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
    if args.command == "dry-prepare":
        value = build()
        result = dict(status="validated_no_neural_or_physical", digest=own.spec.digest(value),
                      contract_sha256=own.spec.digest(contract()), costs=costs())
    elif args.command == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=own.write(path, build()))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
