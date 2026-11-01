"""Known-root spectral8 H3 single-arm CUDA entry; no physics or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import codex_four_arm_host_cuda_20261030 as old
from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as spectral
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as suffix
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport

NAME = "codex_spectral8_host_cuda_20261030"
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/codex_four_arm_host_cuda_20261030_PREP01.json"
BASE_SHA = "b8044f55943b34c5a56a0b5db99cfc5105c5180cdf0f2139926fbb9c82d2b5ec"
AUTH = "MAIN_AUTHORIZES_SPECTRAL8_SINGLE_ARM_CUDA_ONLY"
TRACE_LIMIT = 16 * 1024**2
DISK_FLOOR = 256 * 1024**2
WALL_NS = 600 * 10**9
ALLOC, RESERVED = 4 * 1024**3, 6 * 1024**3


def assemble(base):
    """Construct immutable config/policy/contract from already checked base."""
    task = base["task"]
    config = {"steps": 2, "learning_rate": 0.05, "history_size": 3}
    policy = spectral.policy(task["action_mean"], task["action_std"], 3)
    identity = {"root": task["binding"]["root_sha256"], "task": spec.digest(task),
                "model": base["expected_model_state_sha256"],
                "action_domain": task["binding"]["action_domain_sha256"],
                "history": spec.digest(task["root_context"]), "goal": spec.digest(task["goal"]),
                "optimizer": spectral.file_sha(suffix.__file__), "optimizer_config": spec.digest(config),
                "meter": spectral.file_sha(durable.__file__),
                "hardware": spec.digest({"device_policy": "host_cuda_0"})}
    arms = {arm: {"initializer_sha256": spectral.file_sha(spectral.__file__) if arm == "structured"
                  else spec.digest({"unimplemented_arm": arm}),
                  "optimizer_sha256": identity["optimizer"],
                  "optimizer_config_sha256": identity["optimizer_config"]} for arm in spec.ARMS}
    caps = spectral.required_caps(config)
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "spectral8", "caps": caps},
                           {"id": "unused_larger", "caps": {k: v+1 for k,v in caps.items()}}]}
    spec.preflight(contract, spec.digest(contract))
    return {"schema": NAME + ".prepare.v1", "task": task, "config": config, "policy": policy,
            "contract": contract, "contract_sha256": spec.digest(contract),
            "checkpoint_directory": base.get("checkpoint_directory"),
            "checkpoint_weights_sha256": base.get("checkpoint_weights_sha256"),
            "checkpoint_config_sha256": base.get("checkpoint_config_sha256"),
            "expected_model_state_sha256": base["expected_model_state_sha256"],
            "known_request_slot": 0, "base_prepare_sha256": BASE_SHA,
            "label": spectral.LABEL, "executed_arms": ["structured"],
            "other_arms": "unimplemented_identity_placeholders_not_comparison",
            "wall_ns": WALL_NS, "allocated_cap": ALLOC, "reserved_cap": RESERVED,
            "trace_bytes": TRACE_LIMIT, "disk_floor": DISK_FLOOR,
            "fairness_proven": False, "physical_execution_authorized": False, "P1_admission": False}


def build():
    base = old.read_prepare(BASE, BASE_SHA)
    result = assemble(base)
    paths = {str(Path(__file__).resolve()), str(Path(spectral.__file__).resolve()),
             str(Path(suffix.__file__).resolve())}
    result["sources"] = dict(base["sources"])
    result["sources"].update({path: old.sha(path) for path in sorted(paths)})
    return result


def _output_path(path, suffix):
    path = Path(path)
    spec.require(path.is_absolute() and path.parent == ROOT / "results"
                 and path.parent.resolve() == path.parent and path.name.startswith(NAME + "_")
                 and path.name.endswith(suffix) and not path.exists() and not path.is_symlink(),
                 "exclusive output path")
    return path


def prepare(path):
    path = _output_path(path, ".json")
    result = build()
    spec.require(shutil.disk_usage(path.parent).free >= DISK_FLOOR + 16*1024**2, "disk floor")
    return {"prepared": str(path), "sha256": old._write_new(path, result)}


def read_prepare(path, expected):
    path = Path(path)
    spec.require(path.is_absolute() and path.parent == ROOT/"results" and path.is_file()
                 and not path.is_symlink() and spec.is_hash(expected) and path.stat().st_size < 8*1024**2,
                 "prepared path/SHA")
    data = path.read_bytes()
    spec.require(hashlib.sha256(data).hexdigest() == expected, "prepared SHA")
    value = json.loads(data, object_pairs_hook=consumer.unique)
    spec.require(value == build(), "prepared source/inputs drift")
    return value


def execute_engine(model, prepared, directory, check, *, device, analytic_fixture=False):
    """Durable structured execution seam. CPU allowed only with explicit analytic fixture."""
    task, config, policy = prepared["task"], prepared["config"], prepared["policy"]
    contract, digest = prepared["contract"], prepared["contract_sha256"]
    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    states = tensor(task["root_context"]["latent_history"])[None]
    history = tensor(task["root_context"]["action_history"])[None]
    goal, mean, std = tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"])
    projection = tensor(task["projection"])
    trace = transport.Journal(Path(directory)/"candidates.jsonl", TRACE_LIMIT)
    driver = None
    failure, result, receipt = None, None, None
    try:
        driver = durable.Driver(Path(directory)/"ledger", contract, digest, "structured", "spectral8", "structured")
        leaf = leafmod.LeafAdapter(model, driver, contract, digest, analytic_fixture=analytic_fixture)
        original_emit = driver._emit
        def checked_emit(kind, payload):
            if kind == "intent":
                try:
                    check()
                except BaseException:
                    original_emit("invalid", {})
                    raise
            return original_emit(kind, payload)
        driver._emit = checked_emit
        def record(kind, payload):
            check()
            spec.require(shutil.disk_usage(directory).free >= DISK_FLOOR, "disk floor")
            trace.emit(kind, payload)
        record("header", {"schema": NAME+".trace.v1", "contract_sha256": digest,
                           "candidate_ids": policy["candidate_ids"], "label": spectral.LABEL})
        result = spectral.run(leaf, states, history, goal, mean, std, projection, config, policy, contract, emit=record)
        record("terminal", {"status": "complete_adapter"})
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        if driver is not None and not driver.failed and not driver.closed:
            try:
                driver.dispatch("setup", "__abort__", {}, lambda: None)
            except BaseException:
                pass
        try:
            trace.emit("terminal", {"status": "failed_or_partial", "error": failure})
        except BaseException:
            pass
    finally:
        trace.close()
        if driver is not None:
            receipt = driver.finish()
    account = None if receipt is None else consumer.consume(receipt["journal"], receipt["sha256"], digest,
                                                             "structured", "spectral8", "structured")
    if failure is None:
        spec.require(account["status"] == "closed_unverified"
                     and account["charged_upper_bounds"] == spectral.required_caps(config), "account mismatch")
    entry = {"status": "failed_or_partial" if failure else "complete",
             "error": failure, "ledger": receipt, "account": account,
             "trace": str(Path(directory)/"candidates.jsonl"),
             "trace_sha256": old.sha(Path(directory)/"candidates.jsonl")}
    if result is not None and failure is None:
        entry.update(selected_index=result["selected_index"], selected_id=result["selected_id"],
                     selected_raw=result["optimized_raw"][result["selected_index"]].cpu().tolist(),
                     scores=result["scores"])
    return entry


def run_cuda(output, prepared, authorization):
    spec.require(authorization == AUTH, "explicit CUDA authorization required")
    output = _output_path(output, "")
    spec.require(prepared == build(), "prepared source/inputs drift")
    spec.require(shutil.disk_usage(output.parent).free >= DISK_FLOOR + TRACE_LIMIT + durable.MAX_BYTES,
                 "disk floor and output allowance")
    output.mkdir()
    report = {"schema": NAME+".report.v1", "status": "failed_or_partial",
              "prepared_canonical_sha256": spec.digest(prepared), "engine": None, "error": None,
              "label": spectral.LABEL, "four_arm_comparison": False,
              "fairness_proven": False, "physical_execution_authorized": False, "P1_admission": False}
    started = time.monotonic_ns()
    try:
        info = require_cuda_allocation("cuda:0")
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def memory():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        guard = old.Guard(WALL_NS, time.monotonic_ns, memory, ALLOC, RESERVED)
        with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            guard.check()
            model = load_compatible_pretrained(prepared["checkpoint_directory"], device=torch.device("cuda:0"))
            model.eval().requires_grad_(False)
            before = old.source.signal.model_hash(model)
            spec.require(before["original_state_dict"] == prepared["expected_model_state_sha256"], "model hash")
            guard.check()
            report["engine"] = execute_engine(model, prepared, output, guard.check, device="cuda:0")
            after = old.source.signal.model_hash(model)
            report.update(model_before=before, model_after=after, device=info)
            spec.require(before == after, "model drift")
            spec.require(prepared == build(), "source or inputs changed")
            guard.check()
            report["peak_allocated"], report["peak_reserved"] = memory()
            if report["engine"]["status"] == "complete":
                report["status"] = "complete_single_arm_engineering_not_admitted"
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    report["elapsed_ns"] = time.monotonic_ns()-started
    digest = old._write_new(output/"report.json", report)
    return {"status": report["status"], "report_sha256": digest, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p = sub.add_parser("run")
    p.add_argument("--prepared", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    else:
        result = run_cuda(args.output, read_prepare(args.prepared, args.sha256), args.authorization)
    print(json.dumps(result, sort_keys=True))
    return 0 if args.command == "prepare" or result["status"] == "complete_single_arm_engineering_not_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
