"""CPU-only saved FD-suffix report auditor; no model, simulator or GPU calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight_sources

ROOTS = [20263160, 20263161]
MARGIN = 1e-6
EXPECTED_COSTS = {"evaluation_calls": 29, "prefix_predict_samples": 29,
                  "suffix_token_predict_samples": 290, "prefix_gradient_vjp_calls": 0}
REQUIRED_SOURCE_NAMES = {
    "run_reacher_pc_p1_fd_suffix_cuda_20261102_codexexclusive.py",
    "reacher_pc_p1_fd_suffix_search_20261102_codexexclusive.py",
    "reacher_pc_p1_common_tail_tangent_20261102_codexexclusive.py",
    "reacher_pc_p1_constrained_common_tail_20261102_codexexclusive.py",
    "PC_REACHER_P1_FD_SUFFIX_DEV_20261102_CODEXEXCLUSIVE.md",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arr(value, shape, name, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    require(array.shape == shape and np.isfinite(array).all(), f"invalid {name}")
    return array


def close(a, b, name, *, atol=1e-10, rtol=1e-10):
    require(np.allclose(a, b, atol=atol, rtol=rtol), f"{name} mismatch")


def _expected_bank(probes, base_f):
    require(len(probes) == 10 and [p["axis"] for p in probes] == list(range(10)),
            "FD probe order mismatch")
    grad_i, grad_f = [], []
    for p in probes:
        ip = float(p["plus_immediate"])
        im = float(p["minus_immediate"])
        require(np.isfinite([ip, im]).all(), "nonfinite FD immediate")
        grad_i.append((ip - im) / .02)
        grad_f.append((arr(p["plus_final"], (5,), "FD plus scores")
                       - arr(p["minus_final"], (5,), "FD minus scores")) / .02)
    gi = np.asarray(grad_i)
    gf = np.asarray(grad_f).T
    norm = np.linalg.norm(gi)
    require(np.isfinite(norm) and norm > 1e-12, "zero FD immediate gradient")
    normal = gi / norm
    best = int(np.argmin(base_f))
    targets = (gf.mean(axis=0), gf[best], gf[0], (gf[1:] - gf[0]).mean(axis=0))
    bank = [np.zeros(10)]
    valid = []
    for gradient in targets:
        tangent = -gradient + (gradient @ normal) * normal
        magnitude = np.linalg.norm(tangent)
        ok = bool(magnitude > 1e-12)
        valid.append(ok)
        direction = tangent / magnitude + .25 * normal if ok else np.zeros(10)
        if ok:
            direction /= np.linalg.norm(direction)
        bank.extend([radius * direction for radius in (.05, .10)])
    return np.asarray(bank), valid, best


def audit_row(row, mean, std):
    require(row["status"] == "complete", "incomplete root")
    receipt = row["compiler_receipt"]
    rows = receipt["proposal_rows"]
    require(len(rows) == 9 and [r["proposal_index"] for r in rows] == list(range(9)),
            "candidate roster mismatch")
    require(receipt["logical_costs"] == EXPECTED_COSTS, "search budget mismatch")
    require(row["census_nominal_samples"] == {"prefix": 9, "suffix_token": 90},
            "census budget mismatch")
    bank = arr(row["bank_params"], (9, 1, 10), "bank")
    normalized = arr(row["prefix_normalized"], (9, 1, 10), "normalized prefix", np.float32)
    raw = arr(row["prefix_raw"], (9, 5, 2), "raw prefix", np.float32)
    tail_norm = arr(row["tail_normalized"], (5, 2, 10), "normalized tail", np.float32)
    tail_raw = arr(row["tail_raw"], (5, 10, 2), "raw tail", np.float32)
    arr(row["predicted_endpoints"], (9, 5, 192), "predicted endpoints")
    recalculated = arr(row["recalculated_final_costs"], (9, 5), "recalculated scores")
    scores = np.stack([arr(r["final_costs"], (5,), "final scores") for r in rows])
    immediate = arr([r["immediate_cost"] for r in rows], (9,), "immediate scores")
    close(scores, recalculated, "recalculated scores", atol=1e-7, rtol=1e-5)
    expected, valid, best_base = _expected_bank(receipt["fd_probes"], scores[0])
    close(bank[:, 0, :], expected, "FD proposal bank", atol=2e-6, rtol=2e-5)
    require(receipt["direction_valid"] == valid, "direction validity mismatch")
    require(receipt["baseline_best_tail_index"] == best_base, "baseline best tail mismatch")
    basis = np.asarray([[0, 0], [.3, 0], [-.3, 0], [0, .3], [0, -.3]], dtype=np.float32)
    tails = np.tile(basis, (1, 5))[:, None, :].repeat(2, axis=1)
    close(arr(receipt["common_suffix_parameters"], (5, 2, 10), "tail parameters"),
          tails, "fixed common tails", atol=1e-7, rtol=0)
    mean = arr(mean, (10,), "action mean", np.float32)
    std = arr(std, (10,), "action std", np.float32)
    require((std >= 1e-6).all(), "invalid action std")
    lo, hi = (-1.0 - mean) / std, (1.0 - mean) / std
    for params, actual in ((bank, normalized), (tails, tail_norm)):
        x = torch.as_tensor(params, dtype=torch.float32)
        low, high = torch.as_tensor(lo), torch.as_tensor(hi)
        predicted = ((low + high) / 2 + (high - low) / 2 * torch.tanh(x)).numpy()
        close(actual, predicted, "bounded parameter to normalized action", atol=3e-6, rtol=1e-6)
    for a, n in ((raw, normalized), (tail_raw, tail_norm)):
        require(np.all(np.abs(a) <= 1 + 1e-6), "raw action outside fixed box")
        for actual, chart in zip(a, n, strict=True):
            decoded = setup.p0._normalized_blocks_to_raw_actions(chart, mean, std, action_dim=2)
            require(np.array_equal(decoded, actual), "raw action binding mismatch")
    strict, dev = [], []
    for i, r in enumerate(rows):
        delta = scores[i] - scores[0]
        adjusted = delta[1:] - delta[0]
        close(delta, arr(r["per_tail_delta"], (5,), "delta"), "delta")
        close(adjusted, arr(r["active_tail_neutral_adjusted_delta"], (4,), "adjusted delta"),
              "adjusted delta")
        close(float(r["parameter_delta_l2"]), float(np.linalg.norm(bank[i, 0])), "parameter norm",
              atol=2e-6, rtol=1e-5)
        s = bool(i > 0 and immediate[i] >= immediate[0] + MARGIN
                 and delta.mean() <= -MARGIN and np.all(delta <= 0)
                 and adjusted.mean() <= -MARGIN and np.all(adjusted <= 0))
        require(type(r["eligible"]) is bool and r["eligible"] == s, "strict selector flag mismatch")
        if s:
            strict.append((float(sum(delta.tolist())), i))
        if i:
            tail = int(np.argmin(scores[i]))
            direction_ok = valid[(i - 1) // 2]
            d = bool(direction_ok and immediate[i] >= immediate[0] + MARGIN
                     and scores[i, tail] <= scores[0, best_base] - MARGIN)
            require(type(r["development_eligible"]) is bool and r["development_eligible"] == d
                    and r["direction_valid"] is direction_ok and r["best_tail_index"] == tail,
                    "development selector flag mismatch")
            if d:
                dev.append((float(scores[i, tail]), i, tail))
    sw = min(strict)[1] if strict else None
    dw = min(dev) if dev else None
    require(receipt["strict_selected_index"] == sw, "strict winner mismatch")
    require(receipt["development_selected_index"] == (dw[1] if dw else None)
            and receipt["development_selected_tail_index"] == (dw[2] if dw else None),
            "development winner mismatch")
    return {"seed": row["seed"], "candidates": 9, "strict_eligible": len(strict),
            "development_eligible": len(dev), "strict_selected_index": sw,
            "development_selected_index": dw[1] if dw else None,
            "development_selected_tail_index": dw[2] if dw else None,
            "goal_q_independent_endpoint_recomputation": "unavailable_goal_latent_not_saved"}


def audit_binding(report, plan, preflight, *, plan_hash, preflight_hash, check_files=True):
    require(plan["schema"] == "pc_fd_suffix_cuda_development_v1", "plan schema mismatch")
    require(report["plan_sha256"] == plan_hash and plan["preflight_sha256"] == preflight_hash,
            "plan/preflight hash mismatch")
    require(report["plan_postcheck"] is True and report["frozen_parameters_unchanged"] is True,
            "producer postcheck failed")
    require(report["parameter_hash_before"] == report["parameter_hash_after"]
            == plan["parameters_sha256"] == preflight["model_parameter_hash_expected"],
            "model parameter identity mismatch")
    for plan_key, preflight_key in (("q_sha256", "q_report_sha256"),
                                    ("root_source_sha256", "root_source_sha256"),
                                    ("weights_sha256", "weights_sha256"),
                                    ("stats_sha256", "action_stats_sha256")):
        require(plan[plan_key] == preflight[preflight_key], "preflight identity mismatch: " + plan_key)
    roots = plan["roots"]
    require(roots == preflight["roots"] and [r["seed"] for r in roots] == ROOTS,
            "plan root roster mismatch")
    require([r["seed"] for r in report["rows"]] == ROOTS, "report root roster mismatch")
    for root, row in zip(roots, report["rows"], strict=True):
        if row["status"] == "complete":
            require(all(row[k] == root[k] for k in ("history_index", "root_rgb_sha256", "goal_rgb_sha256")),
                    "root identity mismatch")
        else:
            require(row["status"] == "error" and row.get("failed_stage") and row.get("error_type")
                    and row.get("error"), "invalid failure receipt")
    sources = plan["sources"]
    names = [Path(path).name for path in sources]
    require(len(names) == len(set(names)) and REQUIRED_SOURCE_NAMES.issubset(names),
            "required plan sources missing or ambiguous")
    if check_files:
        project = Path(__file__).resolve().parents[2]
        for name in REQUIRED_SOURCE_NAMES:
            expected_path = (project.parent / "docs" / name if name.endswith(".md")
                             else Path(__file__).resolve().parent / name)
            require(str(expected_path) in sources, "required source path mismatch: " + name)
        for path, expected in sources.items():
            require(sha(path) == expected, "plan source hash mismatch: " + path)
        require(sha(preflight["weights_path"]) == plan["weights_sha256"], "checkpoint file mismatch")
        require(sha(preflight["action_stats_path"]) == plan["stats_sha256"], "action stats file mismatch")
        require(sha(preflight_sources.Q_PATH) == plan["q_sha256"], "goal Q report hash mismatch")
        require(sha(preflight_sources.SOURCE) == plan["root_source_sha256"], "root source hash mismatch")


def main():
    parser = argparse.ArgumentParser()
    for name in ("report", "plan", "preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    require(sha(args.report) == args.report_sha256 and sha(args.plan) == args.plan_sha256,
            "sealed report or plan hash mismatch")
    report, plan, preflight = [json.loads(p.read_text()) for p in (args.report, args.plan, args.preflight)]
    audit_binding(report, plan, preflight, plan_hash=args.plan_sha256, preflight_hash=sha(args.preflight))
    mean, std = setup._load_action_stats(Path(preflight["action_stats_path"]), "swm/ReacherDMControl-v0")
    results = []
    for row in report["rows"]:
        if row["status"] == "complete":
            results.append({"status": "verified_saved_consistency", **audit_row(row, mean, std)})
        else:
            results.append({key: row[key] for key in ("seed", "status", "failed_stage", "error_type", "error")})
    result = {"status": "complete_saved_consistency" if all(r["status"] == "verified_saved_consistency" for r in results)
              else "partial_failures_retained", "rows": results, "report_sha256": args.report_sha256,
              "plan_sha256": args.plan_sha256, "preflight_sha256": sha(args.preflight),
              "consumer_sha256": sha(__file__), "model_inference": False, "physical_execution": False,
              "P1_admission": False, "privileged_input_absence": "not_independently_certified",
              "endpoint_q": "not_independently_recomputed",
              "device_identity": "name_only_uuid_not_certified"}
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps({"status": result["status"], "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
