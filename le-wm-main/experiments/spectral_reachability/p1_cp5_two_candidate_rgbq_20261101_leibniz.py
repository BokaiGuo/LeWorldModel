"""Same-root, same-actions paired terminal RGB-Q audit; host CUDA only."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from . import p1_cp5_cem_rgb_mismatch_20261101_leibniz as prev

NAME = "p1_cp5_two_candidate_rgbq_20261101_leibniz"
ROOT = prev.ROOT
own, family, physical = prev.own, prev.family, prev.physical
RUN = prev.RUN
REPORT_SHA, PROTOCOL_SHA = prev.REPORT_SHA, prev.PROTOCOL_SHA
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
ARMS = ("cem", "structured_own")
AUTH = "MAIN_AUTHORIZES_CP5_TWO_CANDIDATE_RGBQ_CUDA"


def costs():
    value = dict.fromkeys(own.spec.AXES, 0)
    value.update(encode_calls=2, encode_samples=2, action_encode_calls=6,
                 predict_calls=6, predict_samples=6)
    return value


def contract():
    value = deepcopy(prev.contract())
    optimizer, config = family.sha(Path(__file__)), own.spec.digest({"name": NAME, "caps": costs()})
    value["identity"].update(optimizer=optimizer, optimizer_config=config)
    for arm in value["arms"].values():
        arm.update(optimizer_sha256=optimizer, optimizer_config_sha256=config)
    cap = costs()
    value["levels"] = [dict(id="node", caps=cap),
                       dict(id="unused_larger", caps={k: v + 1 for k, v in cap.items()})]
    own.spec.preflight(value, own.spec.digest(value))
    return value


def validate_pair(report, protocol, schedule):
    family.require(report["schema"] == physical.NAME + ".report.v1"
                   and report["protocol_sha256"] == PROTOCOL_SHA
                   and report["sources_verified"] is True and report["neural_calls"] == 0
                   and report["P1_admission"] is False, "physical receipt")
    family.require(protocol["schedule"] == schedule
                   and [r["arm"] for r in report["rows"]] == list(physical.ARMS), "fixed roster")
    rows = []
    for index, arm in enumerate(ARMS):
        slot, row = schedule[index], report["rows"][index]
        family.require(slot["arm"] == row["arm"] == arm and row["status"] == "completed20"
                       and row["steps_attempted"] == row["steps_returned"] == 20
                       and not row.get("error")
                       and np.array_equal(family.actions(slot["commands"], 20),
                                          family.actions(row["executed_raw"], 20)), "actual action binding")
        points = row["checkpoints"]
        family.require([p["step"] for p in points] == [0, 5, 10, 15, 20], "checkpoint roster")
        for point in (points[1], points[-1]):
            path = Path(point["png"])
            family.require(path.parent == RUN and not path.is_symlink()
                           and family.sha(path) == point["file_sha256"], "PNG file binding")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            family.require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                           and hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"]
                           and own.spec.digest(point["state"]) == point["state_sha256"], "PNG/state binding")
        rows.append(dict(arm=arm, actions=slot["commands"], cp5=points[1], terminal=points[-1]))
    family.require(np.array_equal(np.asarray(rows[0]["actions"][:5]), np.asarray(rows[1]["actions"][:5]))
                   and rows[0]["cp5"]["pixel_sha256"] == rows[1]["cp5"]["pixel_sha256"] == protocol["cp5"]["pixel_sha256"]
                   and rows[0]["cp5"]["state_sha256"] == rows[1]["cp5"]["state_sha256"] == protocol["cp5"]["state_sha256"],
                   "shared cp5")
    family.require(not np.array_equal(np.asarray(rows[0]["actions"][5:]), np.asarray(rows[1]["actions"][5:])),
                   "distinct candidates required")
    return rows


def build():
    schedule, sources, _, _, _, cp5 = physical.build()
    family.require(family.sha(RUN / "report.json") == REPORT_SHA
                   and family.sha(RUN / "protocol.json") == PROTOCOL_SHA, "physical report hashes")
    report = json.loads((RUN / "report.json").read_bytes())
    protocol = json.loads((RUN / "protocol.json").read_bytes())
    family.require(protocol["cp5"] == cp5, "cp5 source")
    rows = validate_pair(report, protocol, schedule)
    family.merge_sources(sources, protocol["sources"])
    for row in rows:
        for point in (row["cp5"], row["terminal"]):
            family.merge_sources(sources, {point["png"]: point["file_sha256"]})
    cem_report = json.loads(physical.REPORT.read_bytes())
    own_report = json.loads((family.PLAN_FIVE / "report.json").read_bytes())
    other = own_report["rows"][0]
    family.require(other["prefix"] == "structured" and cem_report["model_before"] == own_report["model_before"]
                   == own_report["model_after"], "model source identity")
    task = cem_report["task"]
    family.require(all(task[k] == other["task"][k] for k in
                       ("goal", "projection", "action_mean", "action_std", "root_context"))
                   and cem_report["cp5_latent"] == other["cp5_latent"], "same cp5 task")
    prior_q = {"cem": cem_report["result"]["replay_score"],
               "structured_own": other["scores"][other["selected_index"]]}
    for row in rows:
        row["prior_predicted_q"] = prior_q[row["arm"]]
    for path in (RUN / "report.json", RUN / "protocol.json", Path(__file__).resolve(), DOC, TEST):
        family.merge_sources(sources, {str(path): family.sha(path)})
    family.verify_sources(sources)
    return dict(schema=NAME + ".prepare.v1", arms=rows, task=task,
        cp5_latent=cem_report["cp5_latent"], sources=sources, costs=costs(),
        model_hash=cem_report["model_before"],
        model_policy=json.loads(Path(cem_report["prepared"]).read_bytes())["source"]["model_policy"],
        physical_execution=False, P1_admission=False, independent_confirmation=False)


def predict(leaf, states, history, raw, mean, std):
    family.require(states.shape == (1, 3, 192) and history.shape == (1, 2, 10)
                   and raw.shape == (3, 5, 2) and mean.shape == std.shape == (2,)
                   and bool((std > 0).all())
                   and all(bool(torch.isfinite(x).all()) for x in (states, history, raw, mean, std))
                   and bool((raw.abs() <= 1).all()), "predict inputs")
    z, past = states.clone(), history.clone()
    for packet in raw:
        token = ((packet - mean) / std).reshape(1, 1, 10)
        context = torch.cat((past, token), 1)
        prediction = leaf.predict(z, leaf.action_encode(context, phase="verification"), phase="verification")
        family.require(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()), "prediction")
        z = torch.cat((z, prediction[:, -1:]), 1)[:, -3:]
        past = context[:, -2:]
    return z[0, -1]


def comparisons(rows):
    family.require([r["arm"] for r in rows] == list(ARMS), "comparison roster")
    a, b = rows
    return {field + "_structured_minus_cem": float(b[field] - a[field])
            for field in ("predicted_q", "actual_rgb_q", "physical_joint_xy")}


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
                states = tensor(task["root_context"]["latent_history"])[None]
                history = tensor(task["root_context"]["action_history"])[None]
                mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                goal = tensor(task["goal"])
                matrix = torch.as_tensor(own.coherent.centered.h4.reverse.Q.kernels.MATRIX,
                                         dtype=torch.float64, device=device)
                rows = []
                for slot in value["arms"]:
                    point = slot["terminal"]
                    with Image.open(point["png"]) as image:
                        pixels = np.asarray(image).copy()
                    family.require(family.sha(point["png"]) == point["file_sha256"]
                                   and hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"],
                                   "terminal RGB drift")
                    image = own.p0._image_tensor(pixels, transform=own.p0.image_transform(), device=device)
                    actual = leaf.encode({"pixels": image}, phase="setup")["emb"][0, 0]
                    raw = tensor(np.asarray(slot["actions"], np.float32)[5:].reshape(3, 5, 2))
                    predicted = predict(leaf, states, history, raw, mean, std)
                    family.require(actual.shape == (192,) and bool(torch.isfinite(actual).all()), "encoded latent")
                    def q(latent):
                        return float(((matrix @ (latent-goal).double()).square().mean()).item())
                    predicted_q, actual_q = q(predicted), q(actual)
                    family.require(np.isclose(predicted_q, slot["prior_predicted_q"], rtol=1e-6, atol=1e-8),
                                   "sealed model Q replay")
                    rows.append(dict(arm=slot["arm"], predicted_q=predicted_q, actual_rgb_q=actual_q,
                        physical_joint_xy=point["metrics"]["joint_xy"], native_success=point["metrics"]["native_success"],
                        physical_metrics=point["metrics"], predicted_latent=predicted.cpu().tolist(),
                        actual_rgb_latent=actual.cpu().tolist()))
                torch.cuda.synchronize(device)
                result = dict(rows=rows, differences=comparisons(rows))
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
            complete = (failure is None and source_ok and before is not None and before == after and result is not None
                        and account is not None and account["status"] == "closed_unverified"
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
