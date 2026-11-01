"""One actual same-bank-greedy cp5, one coherent H3 own suffix."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import p1_coherent_own_suffix_20261101_codex as own

NAME = "p1_coherent_own_suffix_samebank_20261101_codex"
ROOT = own.ROOT
GREEDY = ROOT / "results/p1_coherent_same_bank_greedy_20261101_codex_RUN_MAIN01/report.json"
GREEDY_SHA = "df39207feda1a2239c2b2bcee120a78680e99a7c91b9393c7572b53731e9cc76"
PHYSICAL = ROOT / "results/p1_samebank_greedy_physical_20261101_main_RUN_MAIN01/report.json"
DOC = ROOT.parent / "docs/D2P_P1_SAMEBANK_GREEDY_OWN_SUFFIX_20261101_CODEX.md"
AUTH = "MAIN_AUTHORIZES_SAMEBANK_OWN_SUFFIX_HOST_CUDA"
PREFIX = "same_bank_greedy"
require, sha, spec = own.require, own.sha, own.spec


def greedy_source():
    require(sha(GREEDY) == GREEDY_SHA, "sealed greedy report")
    report = json.loads(GREEDY.read_bytes())
    require(report["schema"] == "p1_coherent_same_bank_greedy_20261101_codex.report.v1"
            and report["status"] == "complete" and report["physical_execution"] is False
            and report["P1_admission"] is False and report["model_before"] == report["model_after"],
            "greedy report status")
    result = report["result"]
    scores = np.asarray(result["scores"], np.float64)
    prefixes = np.asarray(result["prefixes"], np.float32)
    index = result["greedy_index"]
    require(scores.shape == (8,) and prefixes.shape == (8, 5, 2)
            and np.isfinite(scores).all() and np.isfinite(prefixes).all()
            and (np.abs(prefixes) <= 1).all() and type(index) is int
            and index == min(range(8), key=lambda i: (scores[i], i))
            and np.array_equal(prefixes[index], np.asarray(result["selected_prefix"], np.float32)),
            "greedy argmin/selected identity")
    receipt = result["ledger"]
    ledger = Path(receipt["journal"])
    require(ledger.resolve().is_relative_to(GREEDY.parent) and sha(ledger) == receipt["sha256"],
            "greedy ledger")
    account = own.consumer.consume(ledger, receipt["sha256"], result["contract_sha256"],
                                   "structured", "node", "same_bank")
    costs = dict.fromkeys(spec.AXES, 0)
    costs.update(predict_calls=8, predict_samples=8, action_encode_calls=8,
                 replay_calls=8, selection_evaluations=8)
    require(account == result["account"] and account["status"] == "closed_unverified"
            and account["charged_upper_bounds"] == costs, "greedy ledger accounting")
    binding_path = GREEDY.parent / "binding.json"
    require(sha(binding_path) == report["binding_sha256"], "greedy source binding")
    binding = json.loads(binding_path.read_bytes())
    require(binding["expected_model_hash"] == report["model_before"]
            and np.array_equal(np.asarray(binding["prefixes"], np.float32), prefixes)
            and all(sha(path) == digest for path, digest in binding["sources"].items()),
            "greedy model and source closure")
    sources = dict(binding["sources"])
    sources.update({str(GREEDY): GREEDY_SHA, str(ledger): receipt["sha256"],
                    str(binding_path): report["binding_sha256"]})
    return prefixes[index].copy(), report["model_before"], index, sources


def load_physical(path, expected_sha, selected_prefix, expected_root_state, expected_root_pixel):
    """Read only prefix actions and cp0/cp5 identity; no future outcomes."""
    path = Path(path)
    require(path.is_absolute() and path == PHYSICAL and not path.is_symlink()
            and spec.is_hash(expected_sha) and path.is_file() and sha(path) == expected_sha,
            "physical report SHA/path")
    report = json.loads(path.read_bytes())
    protocol = path.parent / "protocol.json"
    require(report["P1_admission"] is False and report["independent_confirmation"] is False
            and report["neural_calls"] == 0 and sha(protocol) == report["protocol_sha256"]
            and report["sources"].get(str(GREEDY)) == GREEDY_SHA
            and all(sha(source) == digest for source, digest in report["sources"].items()),
            "physical protocol and source closure")
    pairs = []
    sources = dict(report["sources"])
    sources.update({str(path): expected_sha, str(protocol): report["protocol_sha256"]})
    for suffix in ("common", "zero"):
        rows = [r for r in report["rows"] if r["prefix"] == PREFIX and r["suffix"] == suffix]
        require(len(rows) == 1, "unique actual branch")
        row = rows[0]
        require(not row.get("error") and len(row["executed_raw"]) >= 5, "actual prefix completed")
        raw = np.asarray(row["executed_raw"][:5], np.float32)
        require(raw.shape == (5, 2) and np.array_equal(raw, selected_prefix),
                "actual returned greedy prefix")
        cp0 = [c for c in row["checkpoints"] if c["step"] == 0]
        cp5 = [c for c in row["checkpoints"] if c["step"] == 5]
        require(len(cp0) == len(cp5) == 1
                and np.array_equal(np.asarray(cp0[0]["state"], np.float64),
                                   np.asarray(expected_root_state, np.float64))
                and cp0[0]["pixel_sha256"] == expected_root_pixel,
                "centered root and cp5")
        frame = cp5[0]
        image_path = Path(frame["png"])
        require(image_path.parent == path.parent and not image_path.is_symlink()
                and sha(image_path) == frame["file_sha256"], "physical cp5 PNG identity")
        with Image.open(image_path) as im:
            pixels = np.asarray(im).copy()
        require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"],
                "physical cp5 pixels")
        state = np.asarray(frame["state"], np.float64)
        require(state.shape == (7,) and np.isfinite(state).all(), "finite cp5 state identity")
        pairs.append((frame["pixel_sha256"], state, image_path, frame["file_sha256"]))
        sources[str(image_path)] = frame["file_sha256"]
    require(pairs[0][0] == pairs[1][0] and np.array_equal(pairs[0][1], pairs[1][1]),
            "common/zero cp5 identity")
    return dict(prefix=PREFIX, prefix_raw=selected_prefix.tolist(),
                png=str(pairs[0][2]), png_sha256=pairs[0][3], pixel_sha256=pairs[0][0],
                cp5_state_identity_sha256=spec.digest(pairs[0][1].tolist())), sources


def build(runner, physical_path, physical_sha):
    selected, model_hash, index, sources = greedy_source()
    base = runner.build()
    data = runner.p3.source_data(base["source"])
    report, target, _, images, _ = own.coherent.centered.source_inputs()
    own.coherent.centered.rebind(base, data, runner, report, target, images)
    require(model_hash == json.loads(own.coherent.centered.adapter.TARGET.read_bytes())["model_before"]
            and model_hash["original_state_dict"] == base["source"]["expected_model_hash"],
            "same centered frozen model")
    require(base["source"]["tasks"][0]["goal"] == own.coherent.centered.h4.reverse.Q.kernels.GOAL
            and own.coherent.centered.h4.reverse.Q.kernels.ARTIFACT_SHA ==
            own.coherent.centered.adapter.METRIC_SHA, "same centered goal/Q")
    row, physical_sources = load_physical(physical_path, physical_sha, selected,
                                          target["root_state"], target["history_pixel_sha256"][-1])
    sources.update(base["sources"])
    sources.update(physical_sources)
    for item in (Path(__file__), Path(own.__file__), DOC, own.DOC,
                 Path(own.p0.__file__), Path(checkpoint_compat.__file__), Path(gpu_preflight.__file__),
                 Path(own.durable.__file__), Path(own.leafmod.__file__),
                 Path(own.consumer.__file__), Path(own.journal.__file__)):
        item = item.resolve()
        sources[str(item)] = sha(item)
    require(all(sha(path) == digest for path, digest in sources.items()), "source closure")
    return dict(schema=NAME + ".prepare.v1", rows=[row], base_task=base["source"]["tasks"][0],
                model_sha256=model_hash["original_state_dict"], expected_model_hash=model_hash,
                model_policy=base["source"]["policy"], warmup_raw=report["raw_history"][-5:],
                sources=sources, target_sha256=own.coherent.centered.adapter.TARGET_SHA,
                metric_sha256=own.coherent.centered.adapter.METRIC_SHA, greedy_report_sha256=GREEDY_SHA,
                greedy_index=index, physical_report=str(physical_path), physical_report_sha256=physical_sha,
                node_cost=own.coherent.centered.h4.h4.node_cost("structured", 5),
                encode_cost={"encode_calls": 1, "encode_samples": 1},
                hold_reference="actual_prefix5_then_zero15_not_original_plan",
                protocol="known_root_openloop15_single_greedy_prefix_exploration",
                P1_admission=False, independent_confirmation=False, physical_execution=False)


def exclusive(path):
    path = Path(path)
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
            "exclusive output")
    return path


def dry(runner, physical_path, physical_sha):
    value = build(runner, physical_path, physical_sha)
    task = own.make_task(value["base_task"], value["rows"][0], [0.] * 192, value["warmup_raw"], runner)
    _, node, _ = own.node_contract(task, value["model_sha256"], runner)
    _, enc, _ = own.encode_contract(task, value["model_sha256"], runner)
    return dict(status="dry_prepared_not_written", prefix_count=1, prefix=PREFIX,
                build_digest=spec.digest(value), node_contract_sha256=spec.digest(node),
                encode_contract_sha256=spec.digest(enc), node_cost=value["node_cost"],
                encode_cost=value["encode_cost"], cuda_used=False, no_model_executed=True)


def run(prepared_path, expected, output, authorization, experiment, runner):
    require(authorization == AUTH, "explicit host CUDA authorization")
    prepared_path = Path(prepared_path)
    require(prepared_path.is_absolute() and prepared_path.parent == ROOT / "results"
            and prepared_path.is_file() and not prepared_path.is_symlink()
            and spec.is_hash(expected) and sha(prepared_path) == expected, "prepared identity")
    value = json.loads(prepared_path.read_bytes())
    require(value == build(runner, value["physical_report"], value["physical_report_sha256"]),
            "prepared/source drift")
    output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= 512 * 1024**2, "disk reserve")
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        output.mkdir()
        binding_sha = own.write(output / "binding.json", value)
        rows, failure, before, after, host = [], None, None, None, None
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
                        and shutil.disk_usage(output).free >= 128 * 1024**2, "resource limits")

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
                matrix = torch.as_tensor(own.coherent.centered.h4.reverse.Q.kernels.MATRIX, dtype=torch.float64, device=device)
                row = value["rows"][0]
                image_path = Path(row["png"])
                require(sha(image_path) == row["png_sha256"], "cp5 PNG drift")
                with Image.open(image_path) as im:
                    pixels = np.asarray(im).copy()
                require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"], "cp5 pixels")
                placeholder = own.make_task(value["base_task"], row, [0.] * 192, value["warmup_raw"], runner)
                _, ec, _ = own.encode_contract(placeholder, value["model_sha256"], runner)
                ec_sha = spec.digest(ec)
                folder = output / PREFIX
                folder.mkdir()
                ed = own.durable.Driver(folder / "encode_ledger", ec, ec_sha, "structured", "node", PREFIX + "_encode")
                limit_driver(ed)
                try:
                    leaf = own.leafmod.LeafAdapter(model, ed, ec, ec_sha)
                    image = own.p0._image_tensor(pixels, transform=own.p0.image_transform(), device=device)
                    with torch.inference_mode():
                        latent = leaf.encode({"pixels": image}, phase="setup")["emb"][0, 0].detach().cpu().numpy()
                finally:
                    enc_ledger = ed.finish()
                ea = own.consumer.consume(enc_ledger["journal"], enc_ledger["sha256"], ec_sha,
                                          "structured", "node", PREFIX + "_encode")
                encode_cost = dict.fromkeys(spec.AXES, 0)
                encode_cost.update(encode_calls=1, encode_samples=1)
                require(ea["status"] == "closed_unverified" and ea["charged_upper_bounds"] == encode_cost,
                        "encoding ledger")
                task = own.make_task(value["base_task"], row, latent, value["warmup_raw"], runner)
                policy, contract, config = own.node_contract(task, value["model_sha256"], runner)
                contract_sha = spec.digest(contract)
                trace = own.journal.Journal(folder / "trace.jsonl", 16 * 1024**2)
                driver = own.durable.Driver(folder / "node_ledger", contract, contract_sha, "structured", "node", PREFIX + "_node")
                limit_driver(driver)
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                try:
                    leaf = own.leafmod.LeafAdapter(model, driver, contract, contract_sha)
                    result = own.coherent.node.run_node(
                        leaf, arm="structured", boundary=5,
                        states=tensor(task["root_context"]["latent_history"])[None],
                        history=tensor(task["root_context"]["action_history"])[None],
                        goal=tensor(task["goal"])[None], mean=tensor(task["action_mean"]),
                        std=tensor(task["action_std"]), matrix=matrix,
                        previous=tensor(own.hold_plan(row)), issued=tensor(row["prefix_raw"]), emit=trace.emit)
                    torch.cuda.synchronize(device)
                finally:
                    trace.close()
                    ledger = driver.finish()
                account = own.consumer.consume(ledger["journal"], ledger["sha256"], contract_sha,
                                               "structured", "node", PREFIX + "_node")
                require(account["status"] == "closed_unverified"
                        and account["charged_upper_bounds"] == value["node_cost"] == result["charges"],
                        "node ledger")
                rows.append(dict(prefix=PREFIX, pixel_sha256=row["pixel_sha256"],
                    cp5_state_identity_sha256=row["cp5_state_identity_sha256"],
                    prefix_raw=row["prefix_raw"], cp5_latent=latent.tolist(), task=task,
                    policy=policy, config=config, contract_sha256=contract_sha,
                    hold_reference=value["hold_reference"], hold_plan=own.hold_plan(row).tolist(),
                    selected_suffix=result["selected_raw"].cpu().tolist(), selected_index=result["selected_index"],
                    scores=result["scores"], encoding_ledger=enc_ledger, encoding_account=ea,
                    node_ledger=ledger, node_account=account, trace=str(folder / "trace.jsonl"),
                    trace_sha256=sha(folder / "trace.jsonl")))
                own.write(folder / "receipt.json", rows[0])
                after = runner.signal.model_hash(model)
                require(before == after and all(sha(p) == h for p, h in value["sources"].items()),
                        "model/source drift")
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
        report = dict(schema=NAME + ".report.v1", status="complete" if failure is None and len(rows) == 1 else "partial",
                      error=failure, binding_sha256=binding_sha, prepared=str(prepared_path),
                      prepare_sha256=expected, rows=rows, host=host, model_before=before, model_after=after,
                      P1_admission=False, independent_confirmation=False, fairness_proven=False,
                      physical_execution=False,
                      limitations="known root, model-only open-loop15; no independent GPU hooks or physical outcome")
        report_sha = own.write(output / "report.json", report)
        return dict(status=report["status"], output=str(output), report_sha256=report_sha,
                    completed=len(rows), error=failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dp = sub.add_parser("dry-prepare")
    pp = sub.add_parser("prepare")
    for item in (dp, pp):
        item.add_argument("--physical", default=str(PHYSICAL))
        item.add_argument("--physical-sha256", required=True)
    pp.add_argument("--output", required=True)
    rp = sub.add_parser("run")
    for arg in ("prepared", "sha256", "output", "authorization"):
        rp.add_argument("--" + arg, required=True)
    rp.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    experiment, runner = own.configure()
    if args.command == "dry-prepare":
        result = dry(runner, args.physical, args.physical_sha256)
    elif args.command == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=own.write(path, build(runner, args.physical, args.physical_sha256)))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization, experiment, runner)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
