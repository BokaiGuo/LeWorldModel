"""Sealed same-action CEM rollout and actual RGB encoding; host CUDA only."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from . import p1_cp5_cem_physical_20261101_leibniz as physical
from . import p1_reverse_h4_parfit_20261101 as launcher

NAME = "p1_cp5_cem_rgb_mismatch_20261101_leibniz"
ROOT = physical.ROOT
family, cem, own = physical.family, physical.cem, physical.cem.own
RUN = ROOT / "results/p1_cp5_cem_physical_20261101_leibniz_RUN_20261101_01"
REPORT_SHA = "5225e645130e67699fe237e050c8bf0f1a5a0c756450b9cf22b479695eb418e9"
PROTOCOL_SHA = "997d301b13b8ddb4048164621bb0525fcb576f668c600389f05433f9ffacb322"
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
AUTH = "MAIN_AUTHORIZES_CP5_CEM_RGB_MISMATCH_HOST_CUDA"
STEPS = (5, 10, 15, 20)


def costs():
    value = dict.fromkeys(own.spec.AXES, 0)
    value.update(encode_calls=4, encode_samples=4, action_encode_calls=3,
                 predict_calls=3, predict_samples=3)
    return value


def validate_physical(report, protocol, sources):
    family.require(report["schema"] == physical.NAME + ".report.v1"
                   and report["protocol_sha256"] == PROTOCOL_SHA
                   and report["sources_verified"] is True and report["neural_calls"] == 0
                   and report["P1_admission"] is False, "physical report status")
    family.require([r["arm"] for r in report["rows"]] == list(physical.ARMS)
                   and [r["arm"] for r in protocol["schedule"]] == list(physical.ARMS),
                   "physical roster")
    row, slot = report["rows"][0], protocol["schedule"][0]
    raw = family.actions(slot["commands"], 20)
    family.require(row["status"] == "completed20" and row["steps_returned"] == 20
                   and row["steps_attempted"] == 20 and not row.get("error")
                   and np.array_equal(family.actions(row["executed_raw"], 20), raw),
                   "actual CEM actions")
    family.require([p["step"] for p in row["checkpoints"]] == [0, 5, 10, 15, 20],
                   "physical checkpoint roster")
    images = []
    for point in row["checkpoints"][1:]:
        path = Path(point["png"])
        family.require(path.parent == RUN and not path.is_symlink()
                       and family.sha(path) == point["file_sha256"], "physical PNG")
        with Image.open(path) as image:
            pixels = np.asarray(image)
        family.require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                       and hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"],
                       "physical pixels")
        family.require(own.spec.digest(point["state"]) == point["state_sha256"], "physical state")
        family.merge_sources(sources, {str(path): point["file_sha256"]})
        images.append(point)
    family.require(images[0]["pixel_sha256"] == protocol["cp5"]["pixel_sha256"]
                   and images[0]["state_sha256"] == protocol["cp5"]["state_sha256"], "actual cp5")
    return raw, images


def build():
    schedule, sources, _, _, _, cp5 = physical.build()
    family.require(family.sha(RUN / "report.json") == REPORT_SHA
                   and family.sha(RUN / "protocol.json") == PROTOCOL_SHA, "physical report/protocol SHA")
    report = json.loads((RUN / "report.json").read_bytes())
    protocol = json.loads((RUN / "protocol.json").read_bytes())
    family.require(protocol["schedule"] == schedule and protocol["cp5"] == cp5, "physical protocol rebuild")
    family.merge_sources(sources, protocol["sources"])
    raw, images = validate_physical(report, protocol, sources)
    original = json.loads(physical.REPORT.read_bytes())
    family.require(np.array_equal(raw[5:], np.asarray(original["result"]["selected_raw"], np.float32).reshape(15, 2)),
                   "actual same CEM suffix")
    for path in (RUN / "report.json", RUN / "protocol.json", Path(__file__).resolve(), DOC, TEST):
        family.merge_sources(sources, {str(path): family.sha(path)})
    family.verify_sources(sources)
    return dict(schema=NAME + ".prepare.v1", physical_report_sha256=REPORT_SHA,
        physical_protocol_sha256=PROTOCOL_SHA, cem_report_sha256=physical.REPORT_SHA,
        source_cem_report=str(physical.REPORT), actual_actions=raw.tolist(),
        images=images, sources=sources, model_hash=original["model_before"],
        model_policy=json.loads(Path(original["prepared"]).read_bytes())["source"]["model_policy"],
        task=original["task"], cp5_latent=original["cp5_latent"],
        original_terminal_q=original["result"]["replay_score"], costs=costs(),
        p1_admission=False, independent_confirmation=False, physical_execution=False)


def contract():
    ledger = json.loads((physical.REPORT.parent / "ledger/events.jsonl").read_text().splitlines()[0])
    value = deepcopy(ledger["payload"]["contract"])
    optimizer, cfg = family.sha(Path(__file__)), own.spec.digest({"name": NAME, "caps": costs()})
    value["identity"]["optimizer"], value["identity"]["optimizer_config"] = optimizer, cfg
    for arm in value["arms"].values():
        arm["optimizer_sha256"], arm["optimizer_config_sha256"] = optimizer, cfg
    cap = costs()
    value["levels"] = [dict(id="node", caps=cap), dict(id="unused_larger", caps={k: v + 1 for k, v in cap.items()})]
    own.spec.preflight(value, own.spec.digest(value))
    return value


def exclusive(path):
    path = Path(path)
    family.require(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_")
                   and not path.exists() and not path.is_symlink(), "exclusive path")
    return path


def analyze(predicted, actual, goal, matrix, metrics):
    predicted, actual = np.asarray(predicted, np.float64), np.asarray(actual, np.float64)
    goal, matrix = np.asarray(goal, np.float64), np.asarray(matrix, np.float64)
    family.require(predicted.shape == actual.shape == (3, 192) and goal.shape == (192,)
                   and matrix.shape == (192, 192) and np.isfinite(predicted).all()
                   and np.isfinite(actual).all() and np.isfinite(goal).all()
                   and np.isfinite(matrix).all() and len(metrics) == 3, "diagnostic shape/finiteness")
    result = []
    for index, (p, a) in enumerate(zip(predicted, actual, strict=True)):
        q = lambda z: float(np.mean(np.square(matrix @ (z - goal))))
        result.append(dict(step=STEPS[index + 1], predicted_q=q(p), actual_rgb_q=q(a),
                           latent_rmse=float(np.sqrt(np.mean((p-a)**2))),
                           projected_rmse=float(np.sqrt(np.mean(np.square(matrix @ (p-a))))),
                           physical_metrics=metrics[index]))
    return result


def run(prepared, expected, output, authorization):
    family.require(authorization == AUTH, "explicit CUDA authorization")
    path = Path(prepared)
    family.require(path.is_absolute() and path.parent == ROOT / "results" and not path.is_symlink()
                   and own.spec.is_hash(expected) and family.sha(path) == expected, "prepare identity")
    value = json.loads(path.read_bytes())
    family.require(value == build(), "prepare/source drift")
    output = exclusive(output)
    extra = launcher.extra
    runner = extra.runner
    with extra.gpu.locking.gpu_lock(extra.gpu.LOCK):
        output.mkdir()
        binding_sha = own.write(output / "binding.json", value)
        driver = model = None
        before = after = account = ledger = failure = None
        result = None
        host = None
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
                before = runner.signal.model_hash(model)
                family.require(before == value["model_hash"], "checkpoint tensor hash")
                c = contract()
                contract_sha = own.spec.digest(c)
                driver = own.durable.Driver(output / "ledger", c, contract_sha, "structured", "node", NAME)
                leaf = own.leafmod.LeafAdapter(model, driver, c, contract_sha)
                transform = own.p0.image_transform()
                actual = []
                for point in value["images"]:
                    with Image.open(point["png"]) as image:
                        pixels = np.asarray(image).copy()
                    family.require(family.sha(point["png"]) == point["file_sha256"]
                                   and hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"], "PNG drift")
                    img = own.p0._image_tensor(pixels, transform=transform, device=device)
                    latent = leaf.encode({"pixels": img}, phase="setup")["emb"][0, 0]
                    family.require(latent.shape == (192,) and bool(torch.isfinite(latent).all()), "encoded latent")
                    actual.append(latent.cpu().numpy().copy())
                cp5 = np.asarray(value["cp5_latent"], np.float32)
                family.require(np.allclose(actual[0], cp5, rtol=1e-5, atol=1e-6), "cp5 latent replay mismatch")
                task = value["task"]
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                states = tensor(task["root_context"]["latent_history"])[None]
                history = tensor(task["root_context"]["action_history"])[None]
                mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                predicted = []
                suffix = tensor(np.asarray(value["actual_actions"], np.float32)[5:].reshape(3, 5, 2))
                for packet in suffix:
                    token = ((packet - mean) / std).reshape(1, 1, 10)
                    context = torch.cat((history, token), 1)
                    pred = leaf.predict(states, leaf.action_encode(context, phase="verification"), phase="verification")
                    family.require(pred.shape == states.shape and bool(torch.isfinite(pred).all()), "prediction")
                    states = torch.cat((states, pred[:, -1:]), 1)[:, -3:]
                    history = context[:, -2:]
                    predicted.append(pred[0, -1].cpu().numpy().copy())
                torch.cuda.synchronize(device)
                matrix = own.coherent.centered.h4.reverse.Q.kernels.MATRIX
                rows = analyze(predicted, actual[1:], task["goal"], matrix,
                               [p["metrics"] for p in value["images"][1:]])
                family.require(np.isclose(rows[-1]["predicted_q"], value["original_terminal_q"],
                                          rtol=1e-6, atol=1e-8), "original CEM terminal Q mismatch")
                result = dict(rows=rows, predicted_latents=np.asarray(predicted).tolist(),
                              actual_latents=np.asarray(actual).tolist(),
                              cp5_latent_max_abs_error=float(np.max(np.abs(actual[0]-cp5))),
                              original_terminal_q=value["original_terminal_q"],
                              replay_terminal_q=rows[-1]["predicted_q"])
                after = runner.signal.model_hash(model)
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
                        and result is not None and account is not None
                        and account["status"] == "closed_unverified" and account["charged_upper_bounds"] == costs())
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
                      contract_sha256=own.spec.digest(contract()), costs=value["costs"])
    elif args.command == "prepare":
        output = exclusive(args.output)
        result = dict(path=str(output), sha256=own.write(output, build()))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
