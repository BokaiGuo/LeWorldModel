"""Five fixed cp5 RGB prefixes; independent coherent H3 model-only suffixes."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import p1_centered_h4_coherent_20261101_launcher as coherent
from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_durable_20261030 as durable
from . import codex_four_arm_leaf_suffix_20261030 as leafmod
from . import codex_four_arm_consumer_20261030 as consumer
from . import agent_p3_c3_k17_physical_transport_20261029_v1 as journal
from . import preflight_pusht_d2p_p0 as p0

NAME = "p1_coherent_own_suffix_20261101_codex"
ROOT = coherent.centered.adapter.ROOT
DIR = ROOT / "results/p1_coherent_intervention_20261101_main_RUN_MAIN01"
REPORT = DIR / "report.json"
REPORT_SHA = "9ce20f621310930b466c3e3c66df40bf8fde28b736c45ecaa896badba8935db6"
DOC = ROOT.parent / "docs/D2P_P1_COHERENT_OWN_SUFFIX_20261101_CODEX.md"
PREFIXES = ("structured", "native", "random", "immediate_progress", "zero")
AUTH = "MAIN_AUTHORIZES_COHERENT_OWN_SUFFIX_HOST_CUDA"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, text):
    if not ok:
        raise ValueError(text)


def read_prefixes():
    require(sha(REPORT) == REPORT_SHA, "sealed intervention report")
    report = json.loads(REPORT.read_bytes())
    protocol = DIR / "protocol.json"
    require(sha(protocol) == report["protocol_sha256"] and report["neural_calls"] == 0
            and report["P1_admission"] is False and report["independent_confirmation"] is False,
            "intervention protocol")
    require(all(sha(path) == digest for path, digest in report["sources"].items()),
            "intervention sources")
    result = []
    for name in PREFIXES:
        rows = [r for r in report["rows"] if r["prefix"] == name and r["suffix"] == "common_structured"]
        controls = [r for r in report["rows"] if r["prefix"] == name and r["suffix"] == "zero"]
        require(len(rows) == len(controls) == 1, "unique branch")
        row, control = rows[0], controls[0]
        require(row["steps"] >= 5 and control["steps"] >= 5, "actual cp5 required")
        action = np.asarray(row["executed_raw"][:5], np.float32)
        require(action.shape == (5, 2) and np.isfinite(action).all()
                and np.all(np.abs(action) <= 1)
                and np.array_equal(action, np.asarray(control["executed_raw"][:5], np.float32)),
                "identical actual prefix across branches")
        frames = [c for c in row["checkpoints"] if c["step"] == 5]
        other = [c for c in control["checkpoints"] if c["step"] == 5]
        require(len(frames) == len(other) == 1, "unique cp5")
        frame, peer = frames[0], other[0]
        require(frame["state"] == peer["state"] and frame["pixel_sha256"] == peer["pixel_sha256"],
                "matched cp5 state/RGB")
        path = Path(frame["png"])
        require(path.parent == DIR and not path.is_symlink() and sha(path) == frame["file_sha256"],
                "cp5 PNG source")
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"],
                "cp5 RGB hash")
        if name == "zero":
            require(np.count_nonzero(action) == 0, "zero prefix")
        result.append(dict(prefix=name, prefix_raw=action.tolist(), png=str(path),
                           png_sha256=frame["file_sha256"], pixel_sha256=frame["pixel_sha256"],
                           cp5_state_identity_sha256=spec.digest(frame["state"])))
    return result, dict(report["sources"], **{str(REPORT): REPORT_SHA, str(protocol): sha(protocol)})


def configure():
    experiment = coherent.setup("structured")
    return experiment, experiment.runner


def hold_plan(row):
    prefix = np.asarray(row["prefix_raw"], np.float32)
    require(prefix.shape == (5, 2) and np.isfinite(prefix).all()
            and (np.abs(prefix) <= 1).all(), "bounded actual prefix")
    return np.concatenate((prefix[None], np.zeros((3, 5, 2), np.float32)))


def make_task(base, row, latent, warmup, runner):
    task = deepcopy(base)
    z = np.asarray(latent, np.float32)
    require(z.shape == (192,) and np.isfinite(z).all(), "actual cp5 latent")
    prefix = np.asarray(row["prefix_raw"], np.float32)
    raw10 = np.concatenate((np.asarray(warmup, np.float32)[-5:], prefix))
    context = task["root_context"]
    context["latent_history"] = deepcopy(context["latent_history"][-2:]) + [z.tolist()]
    context["action_history"] = runner.rgb.moving.normalized(task, raw10)
    binding = task["binding"]
    binding["context_sha256"] = spec.digest(context)
    binding["root_sha256"] = spec.digest([NAME, "cp5", row["pixel_sha256"], row["prefix_raw"]])
    binding["task_id"] = spec.digest([NAME, binding["task_id"], binding["root_sha256"]])
    binding["task_sha256"] = spec.digest([NAME, "task", binding["task_id"]])
    binding["task_contract_sha256"] = spec.digest(
        [NAME, "contract", binding["task_id"], "predefined_hold", hold_plan(row).tolist()])
    runner.p3.core.validate_task(task)
    return task


def node_contract(task, model_sha, runner):
    h4 = coherent.centered.h4
    control = h4.reverse.Q.carry.control
    with patch.dict(control.base.CONFIG, h4.boundary_config(5)), \
            patch.object(control, "cost", lambda arm: h4.h4.node_cost(arm, 5)):
        policy, contract, config = control.contract_for(task, model_sha, spec.digest({"device_policy": "host_cuda_0"}))
    cost = h4.h4.node_cost("structured", 5)
    require(policy["horizon"] == 3 and config["steps"] == 6
            and all(contract["levels"][0]["caps"][axis] >= cost[axis] for axis in spec.AXES),
            "H3 coherent node contract")
    return policy, contract, config


def encode_contract(task, model_sha, runner):
    policy, contract, config = node_contract(task, model_sha, runner)
    contract = deepcopy(contract)
    caps = dict.fromkeys(spec.AXES, 0)
    caps.update(encode_calls=1, encode_samples=1)
    contract["levels"] = [dict(id="node", caps=caps),
                          dict(id="unused_larger", caps={k: v + 1 for k, v in caps.items()})]
    spec.preflight(contract, spec.digest(contract))
    return policy, contract, config


def build(runner):
    prepared = runner.build()
    data = runner.p3.source_data(prepared["source"])
    report, target, _, images, _ = coherent.centered.source_inputs()
    coherent.centered.rebind(prepared, data, runner, report, target, images)
    rows, sources = read_prefixes()
    sources.update(prepared["sources"])
    for path in (Path(__file__), DOC, Path(p0.__file__), Path(checkpoint_compat.__file__),
                 Path(gpu_preflight.__file__), Path(durable.__file__), Path(leafmod.__file__),
                 Path(consumer.__file__), Path(journal.__file__)):
        path = path.resolve()
        sources[str(path)] = sha(path)
    for row in rows:
        sources[row["png"]] = row["png_sha256"]
    require(all(sha(path) == expected for path, expected in sources.items()), "source closure")
    model_sha = prepared["source"]["expected_model_hash"]
    require(model_sha == json.loads(coherent.centered.adapter.TARGET.read_bytes())["model_before"]["original_state_dict"],
            "encoder/model source identity")
    kernels = coherent.centered.h4.reverse.Q.kernels
    require(prepared["source"]["tasks"][0]["goal"] == kernels.GOAL
            and kernels.ARTIFACT_SHA == coherent.centered.adapter.METRIC_SHA,
            "centered goal and metric identity")
    return dict(schema=NAME + ".prepare.v1", rows=rows,
                base_task=prepared["source"]["tasks"][0], model_sha256=model_sha,
                model_policy=prepared["source"]["policy"], sources=sources,
                warmup_raw=report["raw_history"][-5:],
                metric_sha256=coherent.centered.adapter.METRIC_SHA,
                target_sha256=coherent.centered.adapter.TARGET_SHA,
                expected_model_hash=json.loads(coherent.centered.adapter.TARGET.read_bytes())["model_before"],
                intervention_report_sha256=REPORT_SHA,
                node_cost=coherent.centered.h4.h4.node_cost("structured", 5),
                encode_cost={"encode_calls": 1, "encode_samples": 1},
                hold_reference="actual_prefix5_then_zero15_not_original_plan",
                protocol="known_root_openloop15_own_suffix_exploration",
                P1_admission=False, independent_confirmation=False, physical_execution=False)


def exclusive(value):
    path = Path(value)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
            "exclusive output")
    return path


def write(path, data):
    with path.open("xb") as stream:
        payload = spec.canonical(data) + b"\n"
        require(stream.write(payload) == len(payload), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def verify_prepared(path, expected, runner):
    path = Path(path)
    require(path.is_absolute() and path.parent == ROOT / "results" and path.is_file()
            and not path.is_symlink() and spec.is_hash(expected) and sha(path) == expected,
            "prepared identity")
    data = json.loads(path.read_bytes())
    require(data == build(runner), "prepared/source drift")
    return data


def dry(runner):
    value = build(runner)
    contracts = []
    for row in value["rows"]:
        task = make_task(value["base_task"], row, [0.] * 192, value["warmup_raw"], runner)
        _, contract, _ = node_contract(task, value["model_sha256"], runner)
        _, encoding, _ = encode_contract(task, value["model_sha256"], runner)
        contracts.append(dict(prefix=row["prefix"], node=spec.digest(contract),
                              encoding=spec.digest(encoding)))
    return dict(status="dry_prepared_not_written", build_digest=spec.digest(value),
                contracts=contracts, node_cost=value["node_cost"], encode_cost=value["encode_cost"],
                no_model_executed=True, cuda_used=False)


def run(prepared_path, expected, output, authorization, experiment, runner):
    require(authorization == AUTH, "explicit host CUDA authorization")
    value = verify_prepared(prepared_path, expected, runner)
    output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= 512 * 1024**2, "disk reserve")
    lock = experiment.extra.gpu.locking
    with lock.gpu_lock(experiment.extra.gpu.LOCK):
        output.mkdir()
        binding_sha = write(output / "binding.json", value)
        result_rows, failure, before, after = [], None, None, None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            def check_limits():
                require(time.monotonic() - started < 1800
                        and torch.cuda.memory_allocated(device) <= 4 * 1024**3
                        and torch.cuda.memory_reserved(device) <= 6 * 1024**3
                        and shutil.disk_usage(output).free >= 128 * 1024**2,
                        "wall, CUDA memory or disk limit")

            def limit_driver(driver):
                dispatch = driver.dispatch
                def checked(*args, **kwargs):
                    try:
                        check_limits()
                    except BaseException:
                        if not driver.failed and not driver.closed:
                            driver.failed = True
                            driver._emit("invalid", {})
                        raise
                    return dispatch(*args, **kwargs)
                driver.dispatch = checked

            with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                check_limits()
                model = checkpoint_compat.load_compatible_pretrained(value["model_policy"], device=device).eval().requires_grad_(False)
                before = runner.signal.model_hash(model)
                require(before == value["expected_model_hash"], "model identity")
                matrix = torch.as_tensor(coherent.centered.h4.reverse.Q.kernels.MATRIX, dtype=torch.float64, device=device)
                for row in value["rows"]:
                    name = row["prefix"]
                    image_path = Path(row["png"])
                    require(sha(image_path) == row["png_sha256"], "cp5 PNG drift")
                    with Image.open(image_path) as image:
                        pixels = np.asarray(image).copy()
                    require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"], "cp5 pixels")
                    placeholder = make_task(value["base_task"], row, [0.] * 192, value["warmup_raw"], runner)
                    _, ec, _ = encode_contract(placeholder, value["model_sha256"], runner)
                    encode_sha = spec.digest(ec)
                    folder = output / name
                    folder.mkdir()
                    ed = durable.Driver(folder / "encode_ledger", ec, encode_sha, "structured", "node", name + "_encode")
                    limit_driver(ed)
                    try:
                        leaf = leafmod.LeafAdapter(model, ed, ec, encode_sha)
                        image = p0._image_tensor(pixels, transform=p0.image_transform(), device=device)
                        with torch.inference_mode():
                            latent = leaf.encode({"pixels": image}, phase="setup")["emb"][0, 0].detach().cpu().numpy()
                    finally:
                        encode_ledger = ed.finish()
                    ea = consumer.consume(encode_ledger["journal"], encode_ledger["sha256"],
                                          encode_sha, "structured", "node", name + "_encode")
                    encoding_cost = dict.fromkeys(spec.AXES, 0)
                    encoding_cost.update(encode_calls=1, encode_samples=1)
                    require(ea["status"] == "closed_unverified" and ea["charged_upper_bounds"] == encoding_cost,
                            "separate encoding ledger")
                    task = make_task(value["base_task"], row, latent, value["warmup_raw"], runner)
                    policy, contract, config = node_contract(task, value["model_sha256"], runner)
                    contract_sha = spec.digest(contract)
                    trace = journal.Journal(folder / "trace.jsonl", 16 * 1024**2)
                    driver = durable.Driver(folder / "node_ledger", contract, contract_sha, "structured", "node", name + "_node")
                    limit_driver(driver)
                    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                    previous = tensor(hold_plan(row))
                    try:
                        leaf = leafmod.LeafAdapter(model, driver, contract, contract_sha)
                        result = coherent.node.run_node(
                            leaf, arm="structured", boundary=5,
                            states=tensor(task["root_context"]["latent_history"])[None],
                            history=tensor(task["root_context"]["action_history"])[None],
                            goal=tensor(task["goal"])[None], mean=tensor(task["action_mean"]),
                            std=tensor(task["action_std"]), matrix=matrix, previous=previous,
                            issued=tensor(row["prefix_raw"]), emit=trace.emit)
                        torch.cuda.synchronize(device)
                    finally:
                        trace.close()
                        ledger = driver.finish()
                    account = consumer.consume(ledger["journal"], ledger["sha256"], contract_sha,
                                               "structured", "node", name + "_node")
                    require(account["status"] == "closed_unverified"
                            and account["charged_upper_bounds"] == value["node_cost"] == result["charges"],
                            "node ledger")
                    result_rows.append(dict(prefix=name, pixel_sha256=row["pixel_sha256"],
                        cp5_state_identity_sha256=row["cp5_state_identity_sha256"],
                        prefix_raw=row["prefix_raw"], cp5_latent=latent.tolist(),
                        task=task, policy=policy, config=config, contract_sha256=contract_sha,
                        hold_reference=value["hold_reference"], hold_plan=hold_plan(row).tolist(),
                        selected_suffix=result["selected_raw"].cpu().tolist(),
                        selected_index=result["selected_index"], scores=result["scores"],
                        encoding_ledger=encode_ledger, encoding_account=ea, node_ledger=ledger,
                        node_account=account, trace=str(folder / "trace.jsonl"),
                        trace_sha256=sha(folder / "trace.jsonl")))
                    write(folder / "receipt.json", result_rows[-1])
                after = runner.signal.model_hash(model)
                require(before == after and all(sha(p) == h for p, h in value["sources"].items()), "model/source drift")
        except BaseException as exc:
            host = locals().get("host")
            failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
        summary = dict(schema=NAME + ".report.v1", status="complete" if failure is None and len(result_rows) == 5 else "partial",
                       error=failure, binding_sha256=binding_sha, prepared=str(prepared_path),
                       prepare_sha256=expected, rows=result_rows, host=host, model_before=before,
                       model_after=after, P1_admission=False, independent_confirmation=False,
                       fairness_proven=False, physical_execution=False,
                       limitations="known root, model-only open-loop15; no independent GPU hooks or physical outcome")
        report_sha = write(output / "report.json", summary)
        return dict(status=summary["status"], output=str(output), report_sha256=report_sha,
                    completed=len(result_rows), error=failure)


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
    experiment, runner = configure()
    if args.command == "dry-prepare":
        result = dry(runner)
    elif args.command == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=write(path, build(runner)))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization, experiment, runner)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
