"""Preflight and explicit-opt-in CUDA evaluator for dev303 RGB geometry prefixes.

No CUDA/model is imported or initialized by prepare/verify. Existing latent
history is reused; the image encoder/projector are forbidden and instrumented.
P1 and physical admission remain false. No command is executed by prepare.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_dev303_v3_cuda_executor_20261102_exclusive"
PREP = ROOT / "results/p1_dev303_v3_model_evaluation_contract_20261102_exclusive_PREP_02.json"
PREP_SHA = "fafd1d2de1d211af9166c6ff4966a61ecf29b83c78fda6e2e0a3bffaa7e44345"
SOURCE_NAMES = (
    "experiments/spectral_reachability/p1_dev303_v3_model_evaluation_contract_20261102_exclusive.py",
    "experiments/spectral_reachability/p1_dev303_rgb_boundary_static_v3_20261102_exclusive.py",
    "experiments/spectral_reachability/p1_dev303_rgb_boundary_static_20261102_exclusive.py",
    "experiments/spectral_reachability/p1_rgb_boundary_contact_witness_v3_20261102_exclusive.py",
    "experiments/spectral_reachability/p1_rgb_boundary_contact_witness_20261102_exclusive.py",
    "experiments/spectral_reachability/p1_dev302_mixed_prefix_bank_20261101_codex.py",
    "experiments/spectral_reachability/p1_dev302_rgb_contact_feedback_20261101_codex.py",
    "experiments/spectral_reachability/codex_p1_rgb_task_score_20261030_v2.py",
    "experiments/spectral_reachability/codex_p1_blind_h2_host_cuda_20261030_v1.py",
    "experiments/spectral_reachability/d2p_planning_compute_ledger.py",
)
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
AUTHORITY = "EXPLICIT_DEV303_RGB_MODEL_ONLY_AUTHORIZATION"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_prepared():
    need(PREP.is_file() and not PREP.is_symlink() and sha_file(PREP) == PREP_SHA,
         "PREP_02 SHA")
    value = json.loads(PREP.read_bytes())
    need(value["schema"] == "p1_dev303_v3_model_evaluation_contract_20261102_exclusive.prepare.v1"
         and value["P1_admission"] is False
         and value["physical_execution_authorized"] is False
         and value["model_execution_authorized"] is False, "development boundary")
    return value


def verify_bindings(mapping):
    need(isinstance(mapping, dict) and bool(mapping), "nonempty bindings")
    for name, expected in mapping.items():
        path = Path(name)
        need(path.is_absolute() and path.is_file() and not path.is_symlink()
             and sha_file(path) == expected, f"source drift: {path}")


def validate_task_identity(envelope, task_slot, model_state_sha, model_bindings, policy):
    weight_path = str(Path(policy) / "weights.pt")
    need(weight_path in model_bindings, "checkpoint weight binding missing")
    need(envelope["task_id"] == task_slot["task_id"]
         and envelope["P1_admission"] is False
         and envelope["task"]["model_sha256"] == model_bindings[weight_path],
         "task/checkpoint file identity")
    need(isinstance(model_state_sha, str) and len(model_state_sha) == 64,
         "state-dict identity")


def prepare_contract():
    value = read_prepared()
    verify_bindings(value["model_source_bindings_declared_not_reverified"])
    verify_bindings(value["producer_sources"])
    need(sha_file(ROOT / "experiments/spectral_reachability/d2p_planning_compute_ledger.py")
         == value["ledger_source_sha256"], "ledger identity")
    sources = {str((ROOT / name).resolve()): sha_file(ROOT / name) for name in SOURCE_NAMES}
    sources[str(Path(__file__).resolve())] = sha_file(__file__)
    return {
        "schema": NAME + ".run_prepare.v1", "status": "prepared_not_authorized",
        "input_prepare_sha256": PREP_SHA,
        "model_sha256": value["model_sha256_expected_from_prepare"],
        "task": value["task"], "candidate_hashes": [r["actions_sha256"] for r in value["candidates"]],
        "source_bindings": sources,
        "model_source_bindings": value["model_source_bindings_declared_not_reverified"],
        "producer_bindings": value["producer_sources"],
        "measurement": {"image_encoder": 0, "image_projector": 0, "action_encoder": 7,
                        "predict_calls": 7, "predict_samples": 7,
                        "candidate_count": 7},
        "encoding_scope": "inherited latent task; earlier image encoding cost excluded, not zero",
        "failure": "abort whole run; retain journal and partial rows; no selection or retry",
        "model_execution_authorized": False, "physical_execution_authorized": False,
        "P1_admission": False, "independent_confirmation": False,
        "candidate_semantics": "RGB geometry approaches, not spectral programs",
        "dynamic_import_and_runtime_closure": "not fully verified by static source inventory",
    }


def verify_contract(path, expected_sha):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and sha_file(path) == expected_sha,
         "run contract SHA")
    saved = json.loads(path.read_bytes())
    need(saved == prepare_contract(), "run contract drift")
    return saved


class Meter:
    """Forward hooks count calls/samples, deny visual encoder before entry."""
    def __init__(self):
        self.counts = dict(image_encoder=0, image_projector=0, action_encoder=0,
                           predictor=0, predictor_samples=0)
        self.handles = []

    def _hook(self, kind):
        def call(_module, args):
            self.counts[kind] += 1
            if kind in ("image_encoder", "image_projector"):
                raise ValueError("unexpected image encoding")
            need(args and hasattr(args[0], "shape") and len(args[0].shape) > 0,
                 "hook batch unavailable")
            if kind == "predictor":
                self.counts["predictor_samples"] += int(args[0].shape[0])
        return call

    def attach(self, model):
        for name, module in (("image_encoder", model.encoder), ("image_projector", model.projector),
                             ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _record(stream, value):
    stream.write(canonical(value))
    stream.flush()
    os.fsync(stream.fileno())


def execute(contract_path, contract_sha, output, authority):
    """Explicit later invocation only; caller must supply separate authorization."""
    need(authority == AUTHORITY, "separate model-only execution authority required")
    plan = verify_contract(contract_path, contract_sha)
    prepared = read_prepared()
    output = Path(output).absolute()
    need(not os.path.lexists(output), "exclusive output directory")
    # Lock is shared with existing model runners; no physical or CPU fallback.
    with LOCK.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir()
        with (output / "events.jsonl").open("xb") as journal:
            _record(journal, {"kind": "start", "run_contract_sha256": contract_sha})
            meter, rows, attempts = Meter(), [], {"action_encoder": 0, "predict": 0}
            before = after = None
            try:
                os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
                import torch
                from torch.nn.attention import SDPBackend, sdpa_kernel
                from experiments.checkpoint_compat import load_compatible_pretrained
                from experiments.gpu_preflight import require_cuda_allocation
                from experiments.spectral_reachability.codex_p1_blind_h2_host_cuda_20261030_v1 import model_hash
                from experiments.spectral_reachability import p1_dev303_v3_model_evaluation_contract_20261102_exclusive as selector

                selector.select(prepared, [0.0] * 7)  # Validate all action bytes before model dispatch.
                require_cuda_allocation("cuda:0")
                device = torch.device("cuda:0")
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                model = load_compatible_pretrained(prepared["model_policy"], device=device).eval().requires_grad_(False)
                before = model_hash(model)
                need(before["original_state_dict"] == plan["model_sha256"], "model hash before")
                meter.attach(model)
                task_path = Path(plan["task"]["task_export"])
                need(sha_file(task_path) == plan["task"]["task_sha256"], "task SHA")
                with gzip.open(task_path, "rt") as f:
                    envelope = json.load(f)
                task = envelope["task"]
                validate_task_identity(envelope, plan["task"], plan["model_sha256"],
                                       plan["model_source_bindings"], prepared["model_policy"])
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                    z = tensor(task["root_context"]["latent_history"])[None]
                    past = tensor(task["root_context"]["action_history"])[None]
                    mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                    goal = tensor(task["goal"])[None]
                    need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
                    started = time.monotonic()
                    for row in prepared["candidates"]:
                        need(time.monotonic() - started < 1800, "wall cap")
                        raw = tensor(row["actions"])
                        token = ((raw - mean) / std).reshape(1, 1, 10)
                        context = torch.cat((past, token), dim=1)
                        _record(journal, {"kind": "candidate_start", "index": row["index"],
                                          "actions_sha256": row["actions_sha256"]})
                        attempts["action_encoder"] += 1
                        _record(journal, {"kind": "dispatch", "resource": "action_encoder",
                                          "index": row["index"]})
                        encoded = model.action_encoder(context)
                        attempts["predict"] += 1
                        _record(journal, {"kind": "dispatch", "resource": "predict",
                                          "index": row["index"], "batch": 1})
                        predicted = model.predict(z, encoded)
                        torch.cuda.synchronize(device)
                        score = float((predicted[:, -1] - goal).square().mean().cpu())
                        need(math.isfinite(score), "nonfinite score")
                        item = {"index": row["index"], "label": row["label"],
                                "actions_sha256": row["actions_sha256"], "score": score}
                        rows.append(item)
                        _record(journal, {"kind": "candidate_complete", **item})
                after = model_hash(model)
                need(before == after, "model drift")
                need(attempts == {"action_encoder": 7, "predict": 7}
                     and meter.counts == {"image_encoder": 0, "image_projector": 0,
                                          "action_encoder": 7, "predictor": 7,
                                          "predictor_samples": 7}, "hook accounting")
                verify_contract(contract_path, contract_sha)
                selection = selector.select(prepared, [item["score"] for item in rows])
                result = {"schema": NAME + ".report.v1", "status": "model_only_complete",
                          "contract_sha256": contract_sha, "rows": rows, "selection": selection,
                          "attempts": attempts, "hooks": meter.counts, "model_before": before,
                          "model_after": after, "P1_admission": False,
                          "physical_execution_authorized": False, "independent_confirmation": False,
                          "ledger_note": "observed hooks; historical encoding and total compute not measured"}
                _record(journal, {"kind": "complete", "report_sha256": hashlib.sha256(canonical(result)).hexdigest()})
                with (output / "report.json").open("xb") as stream:
                    _record(stream, result)
                return result
            except BaseException as exc:
                try:
                    _record(journal, {"kind": "failed", "error": str(exc), "attempts": attempts,
                                      "hooks": meter.counts, "completed_rows": rows,
                                      "selection_released": False})
                finally:
                    raise
            finally:
                meter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare_contract()
        with args.contract.open("xb") as stream:
            _record(stream, value)
        print(json.dumps({"contract": str(args.contract), "sha256": sha_file(args.contract)}))
    elif args.command == "verify":
        verify_contract(args.contract, args.contract_sha256)
        print(json.dumps({"status": "source_matched_no_execution"}))
    else:
        need(args.output is not None, "output required")
        execute(args.contract, args.contract_sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
