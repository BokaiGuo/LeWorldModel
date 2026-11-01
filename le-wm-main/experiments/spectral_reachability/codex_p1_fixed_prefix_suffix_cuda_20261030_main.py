"""Known-panel CUDA comparison of four fixed prefixes with common suffix optimizer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import p1_joint_suffix_fd_host_20261029 as source
from . import p1_task_response_saved_panel_20261030_main as selection
from . import codex_p1_fixed_prefix_common_suffix_20261030_main as kernel
from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_leaf_suffix_20261030 as leafmod
from . import codex_four_arm_durable_20261030 as durable
from . import codex_four_arm_consumer_20261030 as consumer
from . import codex_four_arm_final_replay_20261030 as replay
from . import agent_p3_c3_k17_physical_transport_20261029_v1 as journal
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

ROOT = source.ROOT
NAME = "codex_p1_fixed_prefix_suffix_cuda_20261030_main"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
CONFIG = {"steps": 2, "learning_rate": .05, "history_size": 3}


def contract(task, model_sha, mean, std):
    policy = {"schema": replay.POLICY_SCHEMA, "candidate_ids": [f"candidate_{i}" for i in range(4)],
              "horizon": 3, "history_size": 3, "metric": "terminal_latent_mean_squared_error",
              "tie": "first_declared_candidate", "replay_batch": 1,
              "action_chart": "raw_packet_5x2_box_minus1_to1", "mean": mean, "std": std}
    identity = {"root": task["binding"]["root_sha256"], "task": spec.digest(task),
                "model": model_sha, "action_domain": task["binding"]["action_domain_sha256"],
                "history": spec.digest(task["root_context"]), "goal": spec.digest(task["goal"]),
                "optimizer": source.host.F.sha(Path(kernel.__file__)),
                "optimizer_config": spec.digest(CONFIG), "meter": source.host.F.sha(Path(durable.__file__)),
                "hardware": spec.digest({"device_policy": "host_cuda_0"})}
    arms = {arm: {"initializer_sha256": spec.digest({"fixed_prefix_panel": source.REPORT_SHA}),
                  "optimizer_sha256": identity["optimizer"],
                  "optimizer_config_sha256": identity["optimizer_config"]} for arm in spec.ARMS}
    caps = kernel.required_cost(4, CONFIG["steps"])
    value = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
             "scope": "standalone_per_root_per_arm_including_attributable_setup",
             "phase_policy_sha256": spec.digest(policy),
             "levels": [{"id": "node", "caps": caps},
                        {"id": "unused_larger", "caps": {k: v + 1 for k, v in caps.items()}}]}
    spec.preflight(value, spec.digest(value))
    return policy, value


def inputs():
    report = source.saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = source.saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    spec.require(receipt["report_sha256"] == source.REPORT_SHA, "source receipt identity")
    prepared = source.saved.read_verified(report["prepared"], report["prepare_sha256"])
    source.saved.analyze(report, receipt, prepared)
    source.host.verify(report["prepared"], report["prepare_sha256"])
    picked = selection.analyze(report, prepared)
    rows = []
    for i, item in enumerate(picked["arms"]):
        original = report["rows"][i]
        task = prepared["requests"][item["slot"]]["task"]
        prefixes = [original["result"]["candidates"][index]["prefix"] for index in item["selected_indices"]]
        rows.append({"slot": item["slot"], "arm": item["arm"], "indices": item["selected_indices"],
                     "prefixes": prefixes, "task": task})
    spec.require(len(rows) == 8, "fixed eight arms")
    return report, prepared, rows


def write_new(path, payload):
    data = spec.canonical(payload) + b"\n"
    with path.open("xb") as stream:
        spec.require(stream.write(data) == len(data), "short output")
        stream.flush()
        os.fsync(stream.fileno())
    return source.host.F.sha(path)


def run(output):
    output = Path(output)
    spec.require(output.is_absolute() and output.parent == ROOT / "results" and
                 output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
                 "exclusive output")
    report, prepared, rows = inputs()
    started = time.monotonic()
    source_hashes = {str(Path(module.__file__).resolve()): source.host.F.sha(Path(module.__file__).resolve())
                     for module in (kernel, selection, spec, leafmod, durable, consumer, replay,
                                    checkpoint_compat, gpu_preflight, locking)}
    source_hashes[str(Path(__file__).resolve())] = source.host.F.sha(Path(__file__).resolve())
    spec.require(shutil.disk_usage(output.parent).free >= 512 * 1024**2, "disk reserve")
    with locking.gpu_lock(LOCK):
        output.mkdir()
        binding = {"source_report_sha256": source.REPORT_SHA, "source_receipt_sha256": source.RECEIPT_SHA,
                   "prepared_sha256": report["prepare_sha256"], "rows": rows, "config": CONFIG,
                   "sources": source_hashes, "fresh": False, "physical_execution": False}
        binding_sha = write_new(output / "binding.json", binding)
        results = []
        failure = None
        model_before = model_after = host = None
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                model = checkpoint_compat.load_compatible_pretrained(prepared["policy"], device=device)
                model.eval().requires_grad_(False)
                model_before = source.host.signal.model_hash(model)
                spec.require(model_before == report["model_before"] and
                             model_before["original_state_dict"] == prepared["expected_model_hash"],
                             "checkpoint state mismatch")
                for index, row in enumerate(rows):
                    spec.require(time.monotonic() - started < 600 and
                                 torch.cuda.memory_allocated(device) <= 4 * 1024**3 and
                                 torch.cuda.memory_reserved(device) <= 6 * 1024**3, "wall or memory cap")
                    task = row["task"]
                    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
                    mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                    policy, c = contract(task, prepared["expected_model_hash"], mean.cpu().tolist(), std.cpu().tolist())
                    digest = spec.digest(c)
                    folder = output / f"node_{index:02d}"
                    folder.mkdir()
                    trace = journal.Journal(folder / "trace.jsonl", 16 * 1024**2)
                    driver = durable.Driver(folder / "ledger", c, digest, "structured", "node", f"node_{index}")
                    leaf = leafmod.LeafAdapter(model, driver, c, digest)
                    try:
                        result = kernel.plan(leaf, tensor(task["root_context"]["latent_history"])[None],
                            tensor(task["root_context"]["action_history"])[None], tensor(row["prefixes"])[:, None],
                            torch.zeros((1, 2, 5, 2), device=device), tensor(task["goal"])[None], mean, std,
                            CONFIG, policy, c, emit=trace.emit)
                        torch.cuda.synchronize(device)
                    finally:
                        trace.close()
                        ledger = driver.finish()
                    account = consumer.consume(ledger["journal"], ledger["sha256"], digest,
                                               "structured", "node", f"node_{index}")
                    spec.require(account["status"] == "closed_unverified" and
                                 account["charged_upper_bounds"] == kernel.required_cost(4, CONFIG["steps"]),
                                 "ledger mismatch")
                    results.append({"slot": row["slot"], "arm": row["arm"], "indices": row["indices"],
                                    "scores": result["scores"], "selected_index": result["selected_index"],
                                    "ledger": ledger, "account": account,
                                    "trace_sha256": source.host.F.sha(folder / "trace.jsonl")})
                model_after = source.host.signal.model_hash(model)
                spec.require(model_before == model_after, "model changed")
                source.host.F.check_bindings(prepared["bindings"])
                spec.require(all(source.host.F.sha(Path(p)) == h for p, h in source_hashes.items()), "source drift")
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        summary = {"schema": NAME + ".v1", "status": "complete" if failure is None else "partial",
                   "error": failure, "binding_sha256": binding_sha, "host": host, "rows": results,
                   "model_before": model_before, "model_after": model_after,
                   "elapsed_seconds": time.monotonic() - started, "P1_admission": False,
                   "physical_execution": False, "independent_confirmation": False, "fairness_proven": False,
                   "limitations": "Post-outcome known panel; model-only; cooperative lock is not global isolation."}
        sha = write_new(output / "report.json", summary)
        print(json.dumps({"status": summary["status"], "rows": len(results), "sha256": sha,
                          "error": failure}, sort_keys=True))
        return 0 if failure is None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args().output))
