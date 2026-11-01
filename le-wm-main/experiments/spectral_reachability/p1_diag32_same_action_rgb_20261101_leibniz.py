"""Fixed diag32 candidate 4/13: saved physical RGB vs open-loop CUDA rollout."""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from . import codex_p1_diag32_fixed_physical_verify_20261101_exclusive as pv
from . import p1_cp5_cem_rgb_mismatch_20261101_leibniz as base

NAME = "p1_diag32_same_action_rgb_20261101_leibniz"
ROOT, physical, own, family = pv.ROOT, pv.physical, base.own, base.family
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
STEPS = (0, 5, 10, 15)
AUTH = "AUTHORIZE_DIAG32_SAME_ACTION_RGB_HOST_CUDA"
TEMPLATE = ROOT / "results/codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01/common_structured_b00/ledger/events.jsonl"


def costs():
    result = dict.fromkeys(own.spec.AXES, 0)
    result.update(encode_calls=4, encode_samples=4, action_encode_calls=3, predict_calls=3, predict_samples=3)
    return result


def metrics(actual, predicted, goal, physical_metrics):
    actual, predicted, goal = (np.asarray(x, np.float64) for x in (actual, predicted, goal))
    family.require(actual.shape == predicted.shape == (4, 192) and goal.shape == (192,)
                   and np.isfinite(actual).all() and np.isfinite(predicted).all()
                   and np.isfinite(goal).all() and len(physical_metrics) == 4, "metrics inputs")
    return [dict(step=step, predicted_goal_mse=float(np.mean((p-goal)**2)),
                 actual_rgb_goal_mse=float(np.mean((a-goal)**2)),
                 predicted_actual_latent_rmse=float(np.sqrt(np.mean((p-a)**2))),
                 physical_metrics=m)
            for step, p, a, m in zip(STEPS, predicted, actual, physical_metrics, strict=True)]


def build():
    verification = pv.verify()
    family.require(verification["all_actions_returned"] and verification["native_terminated_at"] == 15,
                   "fixed physical record")
    contract = json.loads(pv.PREP.read_bytes())
    report = json.loads((pv.DIRECTORY / "report.json").read_bytes())
    task, _, task_sources, policy = physical.blind.load(0)
    diag = json.loads((physical.DIAG_DIR / "report.json").read_bytes())
    selected = diag["rows"][0]["selected"]
    family.require(contract["selection"]["candidate"] == selected["candidate"] == 4
                   and contract["selection"]["iteration"] == selected["iteration"] == 13
                   and np.array_equal(np.asarray(contract["commands"], np.float32),
                                      np.asarray(selected["actions"], np.float32).reshape(15, 2)),
                   "fixed exact actions")
    row = report["row"]
    points = [row["observations"][step] for step in STEPS]
    family.require([p["step"] for p in points] == list(STEPS), "four actual frames")
    metrics_list = [row["initial_metrics"]] + [row["transitions"][step-1]["metrics"] for step in STEPS[1:]]
    sources = dict(contract["bindings"])
    for name, digest in task_sources.items():
        family.merge_sources(sources, {name: digest})
    for path in [Path(__file__).resolve(), DOC, TEST, Path(pv.__file__).resolve(), TEMPLATE,
                 pv.PREP, pv.DIRECTORY / "report.json", pv.DIRECTORY / "row.json",
                 pv.DIRECTORY / "journal.jsonl", *[Path(p["path"]) for p in points]]:
        family.merge_sources(sources, {str(path): family.sha(path)})
    family.verify_sources(sources)
    family.require(task["binding"]["root_sha256"] == contract["root_binding"]["root_sha256"]
                   and own.spec.digest(task["goal"]) == contract["root_binding"]["goal_latent_sha256"]
                   and np.array_equal(np.asarray(task["action_mean"], np.float32), np.asarray(contract["action_mean"], np.float32))
                   and np.array_equal(np.asarray(task["action_std"], np.float32), np.asarray(contract["action_std"], np.float32)),
                   "task/physical binding")
    return dict(schema=NAME + ".prepare.v1", candidate=4, iteration=13, task=task, policy=policy,
                commands=contract["commands"], actions_sha256=contract["commands_float32_le_sha256"],
                images=points, physical_metrics=metrics_list, selected_q1=selected["q1"],
                selected_q3=selected["q3"], selected_q0=diag["rows"][0]["root_mse"],
                model_hash=diag["model_before"], sources=sources, costs=costs(),
                training=False, physical_execution=False, P1_admission=False)


def ledger_contract(value):
    original = json.loads(TEMPLATE.read_text().splitlines()[0])["payload"]["contract"]
    result = deepcopy(original)
    optimizer, cfg = family.sha(Path(__file__)), own.spec.digest({"name": NAME, "caps": costs()})
    result["identity"].update(optimizer=optimizer, optimizer_config=cfg)
    for arm in result["arms"].values():
        arm.update(optimizer_sha256=optimizer, optimizer_config_sha256=cfg)
    cap = costs()
    result["levels"] = [dict(id="node", caps=cap),
                        dict(id="unused_larger", caps={k: v + 1 for k, v in cap.items()})]
    family.require(result["identity"]["root"] == value["task"]["binding"]["root_sha256"]
                   and result["identity"]["goal"] == own.spec.digest(value["task"]["goal"])
                   and result["identity"]["model"] == value["model_hash"]["original_state_dict"], "ledger identity")
    own.spec.preflight(result, own.spec.digest(result))
    return result


def exclusive(path):
    path = Path(path)
    family.require(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_")
                   and not path.exists() and not path.is_symlink(), "exclusive path")
    return path


def run(prep, expected, output):
    family.require(family.sha(prep) == expected, "prepare SHA")
    value = json.loads(Path(prep).read_bytes())
    family.require(value == build(), "prepare/source drift")
    output = exclusive(output)
    with physical.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        output.mkdir()
        binding = own.write(output / "binding.json", value)
        model = driver = account = ledger = before = after = host = result = failure = None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = own.gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                model = own.checkpoint_compat.load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
                before = physical.blind.host.signal.model_hash(model)
                family.require(before == value["model_hash"], "model hash")
                for module in model.modules():
                    path = str(Path(inspect.getfile(type(module))).resolve())
                    family.require(path in value["sources"] and family.sha(path) == value["sources"][path], "model source")
                c = ledger_contract(value)
                contract_sha = own.spec.digest(c)
                driver = own.durable.Driver(output / "ledger", c, contract_sha, "structured", "node", NAME)
                leaf = own.leafmod.LeafAdapter(model, driver, c, contract_sha)
                actual = []
                transform = own.p0.image_transform()
                for point in value["images"]:
                    with Image.open(point["path"]) as image:
                        pixels = np.asarray(image).copy()
                    family.require(family.sha(point["path"]) == point["file_sha256"]
                                   and hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"], "RGB drift")
                    latent = leaf.encode({"pixels": own.p0._image_tensor(pixels, transform=transform, device=device)},
                                         phase="setup")["emb"][0, 0]
                    family.require(latent.shape == (192,) and bool(torch.isfinite(latent).all()), "encoding")
                    actual.append(latent.cpu().numpy().copy())
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                task = value["task"]
                root = np.asarray(task["root_context"]["latent_history"][-1], np.float32)
                family.require(np.allclose(actual[0], root, rtol=1e-5, atol=1e-6), "root RGB encoding")
                z = tensor(task["root_context"]["latent_history"])[None]
                past = tensor(task["root_context"]["action_history"])[None]
                mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                predicted = [root.copy()]
                for packet in tensor(np.asarray(value["commands"], np.float32).reshape(3, 5, 2)):
                    context = torch.cat((past, ((packet - mean) / std).reshape(1, 1, 10)), 1)
                    pred = leaf.predict(z, leaf.action_encode(context, phase="verification"), phase="verification")
                    family.require(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "prediction")
                    predicted.append(pred[0, -1].cpu().numpy().copy())
                    z = torch.cat((z, pred[:, -1:]), 1)[:, -3:]
                    past = context[:, -2:]
                torch.cuda.synchronize(device)
                rows = metrics(actual, predicted, task["goal"], value["physical_metrics"])
                for row, key in zip((rows[0], rows[1], rows[3]), ("selected_q0", "selected_q1", "selected_q3"), strict=True):
                    family.require(np.isclose(row["predicted_goal_mse"], value[key], rtol=1e-5, atol=1e-6),
                                   "original model score mismatch " + key)
                result = dict(rows=rows, actual_latents=np.asarray(actual).tolist(),
                              predicted_latents=np.asarray(predicted).tolist(), goal_latent=task["goal"],
                              root_latent_max_abs_error=float(np.max(np.abs(actual[0]-root))))
                after = physical.blind.host.signal.model_hash(model)
                family.require(before == after, "model mutation")
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
                          failure=failure, prepared=str(prep), prepare_sha256=expected, binding_sha256=binding,
                          result=result, ledger=ledger, account=account, source_verified=source_ok,
                          model_before=before, model_after=after, host=host, elapsed_seconds=time.monotonic()-started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                          training=False, physical_execution=False, P1_admission=False, independent_confirmation=False)
            digest = own.write(output / "report.json", report)
        return dict(status=report["status"], report_sha256=digest, failure=failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry-prepare")
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p = sub.add_parser("run")
    for name in ("prepared", "sha256", "output", "authorization"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if args.cmd == "dry-prepare":
        value = build()
        result = dict(status="validated_no_neural_or_physical", digest=own.spec.digest(value),
                      contract_sha256=own.spec.digest(ledger_contract(value)), costs=costs())
    elif args.cmd == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=own.write(path, build()))
    else:
        family.require(args.authorization == AUTH, "explicit authorization")
        result = run(args.prepared, args.sha256, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.cmd == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
