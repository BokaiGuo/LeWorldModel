"""Host-CUDA budget integration only: saved predictions, no physical/P3 claim."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from experiments.spectral_reachability.d2p_adaptive_budget_v2 import (
    Binding, PredictionBudget, PredictionBudgetExceeded,
)
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

INPUT_SHA = "d5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a"
OFFICIAL_SHA = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
CONFIG_SHA = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
ATOL = 1e-5
METHOD = "saved_suffix_budget_guard_cuda_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def array(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fi" or not np.isfinite(a).all():
        raise ValueError(f"finite real array of shape {shape} required")
    return a


def validate_rows(report):
    protocol = report["protocol"]
    roots = protocol["roots"]
    if (len(roots) != 4 or len(set(roots)) != 4 or
            any(not isinstance(r, str) or len(r) != 64 or any(c not in '0123456789abcdef' for c in r) for r in roots)
            or protocol["suffix_horizon"] != 1 or protocol["fixed_prefixes_per_root"] != 16):
        raise ValueError("four stable SHA roots and 16 horizon-one suffixes required")
    rows = report["rows"]
    if len(rows) != 4 or [r["root_sha256"] for r in rows] != roots:
        raise ValueError("root identity/order mismatch")
    for row in rows:
        receipts = row["compiled"]["receipts"]
        if len(receipts) != 16 or len({r["candidate_id"] for r in receipts}) != 16:
            raise ValueError("16 unique saved candidate receipts required")
        for receipt in receipts:
            array(receipt["context_emb"], (1, 3, 192))
            array(receipt["context_actions"], (1, 3, 10))
            array(receipt["executed_suffix_params"], (1, 2))
            array(receipt["predicted_terminal_vector"], (192,))
    return rows


def validate_inputs(compilation, policy):
    compilation, policy = Path(compilation).resolve(), Path(policy).resolve()
    if digest(compilation) != INPUT_SHA:
        raise ValueError("saved compilation SHA mismatch")
    report = json.loads(compilation.read_text())
    expected_gates = {"four_roots", "parameters_unchanged", "buffers_unchanged", "all_eval", "inputs_unchanged"}
    if (report.get("schema") != "pusht_d2p_metric_program_preflight.v1"
            or report.get("status") != "compiled_calibration"
            or set(report.get("gates", {})) != expected_gates
            or any(v is not True for v in report["gates"].values())):
        raise ValueError("complete passing compilation integrity required")
    validate_rows(report)
    seal = report["source_seal"]
    verify_seal(output_dir=seal["output_dir"], expected_manifest_sha256=seal["manifest_sha256"])
    bindings = dict(report["protocol"]["bindings"])
    for name, expected in (("weights.pt", OFFICIAL_SHA), ("config.json", CONFIG_SHA)):
        path = str((policy / name).resolve())
        if bindings.get(path) != expected:
            raise ValueError("official current checkpoint/config binding required")
    bindings[str(compilation)] = INPUT_SHA
    if not all(Path(p).is_absolute() and digest(p) == sha for p, sha in bindings.items()):
        raise ValueError("input binding changed")
    config = json.loads((policy / "config.json").read_text())
    if (config["_target_"] != "stable_worldmodel.wm.lewm.LeWM"
            or config["predictor"]["num_frames"] != 3
            or config["predictor"]["output_dim"] != 192
            or config["action_encoder"]["input_dim"] != 10):
        raise ValueError("official horizon/context/action config required")
    return report, bindings, config


def sources_before_allocate(report, config):
    """Seal upstream plus loaded experiment code and configured model constructors.

    Framework binaries are versioned in protocol, not copied as Python source.
    Imports here resolve constructors but never instantiate a model.
    """
    def visit(value):
        if isinstance(value, dict):
            target = value.get("_target_", "")
            if target and not target.startswith("torch."):
                importlib.import_module(target.rsplit(".", 1)[0])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(config)
    manifest = json.loads((Path(report["source_seal"]["output_dir"]) / "manifest.json").read_text())
    sources = {Path(entry["source"]).resolve() for entry in manifest["files"]}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("experiments.", "stable_worldmodel.wm.lewm", "stable_pretraining.backbone")):
            path = getattr(module, "__file__", None)
            if path and path.endswith(".py"):
                sources.add(Path(path).resolve())
    sources.update((Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "tests/test_run_pusht_d2p_budget_guard_cuda.py",
        Path(__file__).resolve().parents[3] / "docs/D2P_BUDGET_GUARD_CUDA_INTEGRATION_20261026.md"))
    return sorted(sources)


def verify_prediction(predicted, saved):
    predicted, saved = array(predicted, (192,)), array(saved, (192,))
    error = float(np.max(np.abs(predicted.astype(float) - saved.astype(float))))
    if error > ATOL:
        raise ValueError(f"saved terminal mismatch: max absolute error {error} > {ATOL}")
    return error


def probe_root(model, row, binding, predict_one):
    """Shared control flow tested with metadata fakes; production uses CUDA only."""
    original = model.predict
    had_override, override = "predict" in vars(model), vars(model).get("predict")
    entered = 0

    # Retain the original signature required by PredictInstrumentation.
    from functools import wraps
    @wraps(original)
    def entered_predict(*args, **kwargs):
        nonlocal entered
        entered += 1
        return original(*args, **kwargs)

    model.predict = entered_predict
    errors = []
    try:
        with PredictionBudget(model, binding=binding, method=METHOD, sample_limit=16) as budget:
            with budget.phase("planning", method=METHOD, horizon_tokens=1):
                for receipt in row["compiled"]["receipts"]:
                    value = predict_one(receipt)
                    errors.append({"candidate_id": receipt["candidate_id"],
                        "max_abs_error": verify_prediction(value, receipt["predicted_terminal_vector"])})
                before_block = entered
                try:
                    predict_one(row["compiled"]["receipts"][0])
                except PredictionBudgetExceeded:
                    blocked = True
                else:
                    raise RuntimeError("17th prediction reached model without budget rejection")
                if entered != before_block or entered != 16:
                    raise RuntimeError("budget did not block before actual predict entry")
        receipt = budget.receipt()
    finally:
        if had_override:
            model.predict = override
        else:
            delattr(model, "predict")
    phases = receipt["phases"]
    if (receipt["attempted_predict_samples"] != 16 or receipt["blocked_calls"] != 1
            or len(phases) != 1 or phases[0]["predict_samples"] != 16
            or phases[0]["predict_calls"] != 16 or phases[0]["failed_predict_calls"] != 0
            or not phases[0]["completed"]):
        raise RuntimeError("actual budget receipt mismatch")
    return {"root_sha256": row["root_sha256"], "predictions": errors, "budget": receipt,
            "actual_predict_entries": entered, "seventeenth_blocked_before_predict": blocked}


def tensor_state_hash(model):
    """Hash parameters AND all buffers including nonpersistent buffers, one at a time."""
    result = {}
    for kind, values in (("parameters", model.named_parameters()), ("buffers", model.named_buffers())):
        h = hashlib.sha256()
        for name, tensor in values:
            h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            h.update(tensor.detach().contiguous().reshape(-1).view(__import__('torch').uint8).cpu().numpy().tobytes())
        result[kind] = h.hexdigest()
    return result


def run(*, compilation, policy, output):
    report, bindings, config = validate_inputs(compilation, policy)
    # Imports resolve code only; source seal and protocol precede CUDA allocation.
    import torch
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
    sources = sources_before_allocate(report, config)
    protocol = {"schema": "pusht_d2p_budget_guard_cuda_protocol.v1", "bindings": bindings,
        "roots": report["protocol"]["roots"], "method": METHOD, "sample_limit_per_root": 16,
        "suffix_horizon": 1, "batch": 1, "absolute_tolerance": ATOL, "relative_tolerance": 0,
        "expected_actual_predict_calls": 64, "expected_blocked_calls": 4,
        "model_id": OFFICIAL_SHA + ":" + CONFIG_SHA, "torch_version": torch.__version__,
        "scope": "HOST CUDA integration only; no new physical execution or P3 admission",
        "P1_admission": False, "P3_admission": False, "compute_advantage": False}
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["protocol_id"] = protocol_id
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    seal = create_seal(files=sources, args_json=json.dumps(protocol), output_dir=output / "source_seal")
    write_new(output / "protocol.json", protocol)
    # Recheck after sealing and immediately before first CUDA allocation.
    validate_inputs(compilation, policy)
    verify_seal(output_dir=seal["output_dir"], expected_manifest_sha256=seal["manifest_sha256"])
    print("HOST CUDA integration: 64 saved horizon-one predictions, 4 blocked attempts; no physical/P3 test", flush=True)
    host = require_cuda_allocation("cuda")
    model = load_compatible_pretrained(str(Path(policy).resolve()), device="cuda").eval().requires_grad_(False)
    if any(t.device.type != "cuda" for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("all model parameters/buffers must reside on host CUDA")
    before = tensor_state_hash(model)
    rows = []
    failure = None
    try:
        with torch.no_grad():
            for row in report["rows"]:
                def predict_one(saved):
                    emb = torch.tensor(saved["context_emb"], dtype=torch.float32, device="cuda")
                    act = torch.tensor(saved["context_actions"], dtype=torch.float32, device="cuda")
                    suffix = torch.tensor(saved["executed_suffix_params"], dtype=torch.float32, device="cuda")[None]
                    value = p1._suffix_terminal(model, emb, act, suffix)[0]
                    torch.cuda.synchronize()
                    return value.cpu().numpy()
                binding = Binding(protocol_id, "pusht_saved_suffix_h1", row["root_sha256"], protocol["model_id"])
                result = probe_root(model, row, binding, predict_one)
                rows.append(result)
                write_new(output / f"root_{row['root_sha256']}.json", result)
                print(f"{row['root_sha256']}: 16 matched predictions, attempt17 blocked", flush=True)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    after = tensor_state_hash(model)
    gates = {"four_roots": len(rows) == 4, "all_predictions_matched": len(rows) == 4 and all(len(r["predictions"]) == 16 for r in rows),
        "all_blocked_before_predict": len(rows) == 4 and all(r["seventeenth_blocked_before_predict"] for r in rows),
        "parameters_unchanged": before["parameters"] == after["parameters"],
        "buffers_unchanged": before["buffers"] == after["buffers"],
        "all_frozen_eval": all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        "inputs_unchanged": all(digest(p) == sha for p, sha in bindings.items())}
    try:
        verified = verify_seal(output_dir=seal["output_dir"], expected_manifest_sha256=seal["manifest_sha256"])
        upstream = report["source_seal"]
        verify_seal(output_dir=upstream["output_dir"], expected_manifest_sha256=upstream["manifest_sha256"])
        gates["sources_unchanged"] = True
    except Exception as exc:
        verified = {"status": "failed", "message": str(exc)}
        gates["sources_unchanged"] = False
    passed = failure is None and all(gates.values())
    write_new(output / "report.json", {"schema": "pusht_d2p_budget_guard_cuda.v1",
        "status": "cuda_integration_pass" if passed else "integrity_fail", "protocol": protocol,
        "host": host, "gates": gates, "rows": rows, "failure": failure, "model_state_before": before,
        "model_state_after": after, "source_seal": seal, "source_verification": verified,
        "P1_admission": False, "P3_admission": False, "new_physical_execution": False, "compute_advantage": False})
    if not passed:
        raise RuntimeError("CUDA integration integrity failed; inspect report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compilation", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
