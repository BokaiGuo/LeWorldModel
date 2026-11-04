"""Fixed two-root joint/conditional model planning and physical handoff.

No physical execution. CUDA execution is explicit, host-only and cooperative.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import p2_joint_host_20261102_codexexclusive as host
from . import p2_joint_saved_20261102_codexexclusive as consumer
from . import p2_joint_vs_conditional_matched_20261102_codexexclusive as matched
from . import p2_common_objective_ledger_20261102_exclusive as core
from . import codex_p2_multistart_joint_physical_20261030_v1 as physical

NAME = "p2_multi_joint_development_20261102_codexexclusive"
ROOT, RESULTS = host.base.ROOT, host.RESULTS
PANEL = RESULTS / "codex_p2_cp15_symmetric_spectral_20261030_v1_PREP_01.json"
PANEL_SHA = "908e830d2a3354c39dce875479be20f6a2bfdcb62676945d64f37d3420cd6fbd"
PHYS = {
    0: ("codex_p2_multistart_joint_physical_20261030_v1_B0_PREP_01.json",
        "ceb38405762f7631d367a490e55d378c655dfe0463fcc648ee88ebade683a23c"),
    2: ("codex_p2_multistart_joint_physical_20261030_v1_B2_PREP_01.json",
        "a0a31ecedf3c4d13842eff989f593782b438e0e58d539ada8f46c46d698110b0")}
CASES = (0, 2)
AUTH = "P2_TWO_ROOT_FIXED_BANK_JOINT_CONDITIONAL_MODEL_ONLY"
canonical = host.native.bank.canonical


def need(value, message):
    if not value:
        raise ValueError(message)


def read_checked(path, digest):
    need(host.base.file_sha(path) == digest, "source SHA " + str(path))
    raw = Path(path).read_bytes()
    need(host.base.sha(raw) == digest, "source changed during read")
    return json.loads(raw)


def input_cases():
    panel = read_checked(PANEL, PANEL_SHA)
    fixed = host.native.verified_input()
    need([c["base"] for c in panel["cases"]] == [0, 1, 2, 3], "panel census")
    cases = []
    for index, root in zip(CASES, (0, 1), strict=True):
        case = panel["cases"][index]
        path, digest = PHYS[index]
        old = read_checked(RESULTS / path, digest)
        raw = np.asarray(case["prefix_raw"], dtype="<f4")
        need(case["base"] == index and case["root"] == root and
             case["parent_arm"] == "two_program_transfer" and old["job_index"] == index
             and case["context"] == old["observed_context"] and raw.shape == (15, 2)
             and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
             and host.base.sha(raw.tobytes()) == case["prefix_float32_sha256"], "root/context/prefix")
        root_sha = old["plans"][0]["native_root_sha256"]
        need(len(old["plans"]) == 2 and len(old["prefix_reference"]) == 106
             and all(p["task_id"] == case["task_id"] and p["root_slot"] == root
                 and p["native_root_sha256"] == root_sha and
                 np.asarray(p["commands"], dtype="<f4").tobytes()[:120] == raw.tobytes()
                 for p in old["plans"]), "physical task/prefix identity")
        context, goal = case["context"], case["goal"]
        need(np.asarray(context["latent_history"]).shape == (3, 192)
             and np.asarray(context["action_history"]).shape == (2, 10)
             and np.asarray(goal).shape == (192,)
             and all(np.isfinite(np.asarray(x, dtype=np.float64)).all()
                     for x in (context["latent_history"], context["action_history"], goal)), "finite model input")
        cases.append({"base": index, "root": root, "task_id": case["task_id"],
                      "native_root_sha256": root_sha,
                      "task_sha256": case["task_sha256"], "context": context, "goal": goal,
                      "context_sha256": host.base.sha(canonical(context)),
                      "goal_sha256": host.base.sha(canonical(goal)),
                      "inherited_feedback_context_sha256": case["feedback_context_sha256"],
                      "prefix_float32_hex": raw.tobytes().hex(), "prefix_sha256": case["prefix_float32_sha256"],
                      "prefix_reference_sha256": physical.digest(old["prefix_reference"]),
                      "old_physical_prep": str(RESULTS / path), "old_physical_prep_sha256": digest})
    need(len({c["task_id"] for c in cases}) == 2 and
         len({c["native_root_sha256"] for c in cases}) == 2 and len(fixed["bank"]) == 16, "distinct roots and bank")
    return cases, fixed["bank"]


def plan_value():
    cases, bank = input_cases()
    model_plan = host.build_plan()
    sources = dict(model_plan["sources"])
    for name in (NAME + ".py", "p2_multi_joint_saved_20261102_codexexclusive.py"):
        path = Path(__file__).parent / name
        sources[str(path)] = host.base.file_sha(path)
    doc = ROOT.parent / "docs/PC_P2_MULTI_JOINT_20261102_CODEXEXCLUSIVE.md"
    sources[str(doc)] = host.base.file_sha(doc)
    return {"schema": NAME + ".plan_v1", "panel_sha256": PANEL_SHA, "cases": cases,
            "bank_sha256": [host.base.sha(x) for x in bank], "bank_origin": "fixed B0 bank reused unchanged for both roots and both arms",
            "case_order": list(CASES), "arm_order": list(matched.ARMS),
            "caps_per_root": matched.CAPS, "max_scores_per_arm_root": 144, "max_gradients_per_arm_root": 128,
            "jacobian_shapes": [[192, 10], [192, 20]], "root_dispatch_deadline_seconds": 900,
            "model_session_deadline_seconds": 2100, "output_cap_bytes": 128 * 1024 * 1024,
            "physical_next_step": "separately authorize physical executor; fixed 10 warm + prefix15 + selected suffix10; stop censored",
            "failure_policy": "first engineering failure aborts remaining roots; retain partial and not_started; no retry or imputation",
            "device_uuid": host.UUID, "lock": str(host.native.LOCK), "authorization": AUTH,
            "checkpoint_files": model_plan["checkpoint_files"], "model_tensor_sha256": model_plan["model_tensor_sha256"],
            "normalizer": model_plan["normalizer"], "sources": sources,
            "source_limits": ["panel and old physical prep bytes checked; full historical provenance requires isolated original validators",
                              "B0 bank was previously exposed and reused out of distribution at second root",
                              "source manifest is not full transitive runtime closure", "physical execution requires a separate plan and authorization"],
            "historical_sra_authenticated": False, "equal_total_compute": False,
            "independent_confirmation": False, "P2_admission": False}


def path_new(path, prefix):
    path = Path(path)
    need(path.is_absolute() and path.parent == RESULTS and path.parent.resolve() == RESULTS
         and path.name.startswith(NAME + prefix) and not os.path.lexists(path), "exclusive output")
    return path


def verify_plan(path, sha):
    path = Path(path)
    need(path.is_absolute() and path.parent == RESULTS and path.name.startswith(NAME + "_PLAN_")
         and not path.is_symlink(), "plan path")
    plan = read_checked(path, sha)
    need(plan == plan_value() and host.base.file_sha(path) == sha, "plan/source drift")
    return plan


def run_root(case, bank, transition, meter, device):
    """Real matched initializer and common optimizer; outer host owns CUDA scope."""
    import torch
    need(torch.device(device).type == "cuda", "CUDA only; no CPU neural inference")
    return _run_root(case, bank, transition, meter, device)


def _run_root(case, bank, transition, meter, device):
    """Analytic test seam; never used for CPU neural inference."""
    fixed = {"context_json": canonical(case["context"]), "goal_json": canonical(case["goal"])}
    evaluate = core.objective(transition, case["context"], case["goal"], meter.ledger, device)
    rows = matched.optimize_schedule(
        bank, lambda arm, x, i: matched._initialize_with_transition(fixed, x, transition, meter, device, arm, i),
        evaluate, lambda e: meter.ledger.record("optimizer_" + e["kind"], e))
    winners = {}
    for arm in matched.ARMS:
        score, index, iteration, entry = min(
            ((e["model_score"], r["candidate_index"], e["winner_iteration"], e)
             for r in rows for e in r["arms"] if e["arm"] == arm), key=lambda x: x[:3])
        winners[arm] = {"candidate_index": index, "iteration": iteration, "model_score": score,
                        "raw_float32_hex": entry["raw_float32_hex"]}
    counts = meter.finish()
    return {"schema": NAME + ".root_v1", "base": case["base"], "root": case["root"],
            "task_id": case["task_id"], "context_sha256": case["context_sha256"],
            "goal_sha256": case["goal_sha256"], "rows": rows, "model_selected_winners": winners,
            "counts": counts, "control_method": "joint_gauss_newton_not_historical_sra",
            "equal_total_compute": False, "independent_confirmation": False, "P2_admission": False}


def physical_handoff(case, result):
    need(result["base"] == case["base"] and result["root"] == case["root"]
         and result["task_id"] == case["task_id"] and result["context_sha256"] == case["context_sha256"]
         and result["goal_sha256"] == case["goal_sha256"], "case/result binding")
    prefix = bytes.fromhex(case["prefix_float32_hex"])
    need(len(prefix) == 120 and host.base.sha(prefix) == case["prefix_sha256"], "prefix bytes")
    arms = []
    for arm in matched.ARMS:
        item = result["model_selected_winners"][arm]
        suffix = bytes.fromhex(item["raw_float32_hex"])
        host.native.optimizer.raw_actions(np.frombuffer(suffix, dtype="<f4").reshape(10, 2))
        arms.append({"arm": arm, "method": "conditional_suffix" if arm == matched.ARMS[0]
                     else "joint_gauss_newton_not_historical_sra", **item,
                     "suffix_sha256": host.base.sha(suffix), "full25_float32_hex": (prefix + suffix).hex(),
                     "full25_sha256": host.base.sha(prefix + suffix)})
    return {"base": case["base"], "root": case["root"], "task_id": case["task_id"],
            "native_root_sha256": case["native_root_sha256"], "prefix_reference_sha256": case["prefix_reference_sha256"],
            "context_sha256": case["context_sha256"], "goal_sha256": case["goal_sha256"],
            "physical_prep_sha256": case["old_physical_prep_sha256"], "arms": arms,
            "physical_ready": False, "reason": "requires separately sealed physical plan and historical root validation"}


class BoundedLedger(core.Ledger):
    def __init__(self, path, cap):
        self.cap = cap
        super().__init__(path)

    def record(self, kind, payload):
        estimate = len(json.dumps(payload, allow_nan=False).encode()) + len(kind) + 8192
        used = sum(p.stat().st_size for p in self.path.parent.iterdir() if p.is_file())
        need(used + estimate <= self.cap - 2 * 1024 * 1024, "output reserve")
        return super().record(kind, payload)


def execute(plan_path, digest, output, auth):
    need(auth == AUTH, "explicit model-only authorization")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "visible CUDA UUID")
    plan = verify_plan(plan_path, digest)
    output = path_new(output, "_RUN_")
    need(shutil.disk_usage(RESULTS).free >= host.base.MIN_FREE_BYTES + plan["output_cap_bytes"], "disk")
    output.mkdir(mode=0o700)
    host.base.write_new(output / "binding.json", {"plan": str(plan_path), "plan_sha256": digest})
    cases, bank = input_cases()
    rows = [{"base": c["base"], "root": c["root"], "status": "not_started"} for c in cases]
    report = {"schema": NAME + ".report_v1", "plan_sha256": digest, "rows": rows,
              "status": "partial_or_failed", "error": None, "device": None,
              "frozen_parameters_unchanged": False, "physical_execution": False,
              "historical_sra_authenticated": False, "equal_total_compute": False,
              "independent_confirmation": False, "P2_admission": False}
    started = time.monotonic()
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from . import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint
        from . import preflight_pusht_d2p_p0 as p0
        from . import pusht_full_packet_compiler_20261027 as compiler
        with host.native.locking.gpu_lock(host.native.LOCK):
            need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "single CUDA")
            props = torch.cuda.get_device_properties(0)
            need(str(getattr(props, "uuid", "")).lower().removeprefix("gpu-")
                 == host.UUID.lower().removeprefix("gpu-"), "device UUID")
            report["device"] = {"uuid": host.UUID, "name": props.name, "torch": str(torch.__version__),
                                "cuda_runtime": torch.version.cuda}
            host.native.check_normalizer(checkpoint.native_metadata(), p0.PUBLIC_ACTION_MEAN, p0.PUBLIC_ACTION_STD)
            for name, expected in checkpoint.MODELS.items():
                need(checkpoint.sha(name) == expected, "checkpoint drift")
            model = load_compatible_pretrained(str(checkpoint.POLICY), device="cuda:0").eval().requires_grad_(False)
            checkpoint.frozen(model)
            need(checkpoint.tensor_hash(model) == checkpoint.MODEL_TENSOR_SHA, "tensor identity")
            old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
                    for i, case in enumerate(cases):
                        need(time.monotonic() - started < plan["model_session_deadline_seconds"], "session wall")
                        jpath = output / f"root_{case['base']}_journal.jsonl"
                        ledger = BoundedLedger(jpath, plan["output_cap_bytes"])
                        rows[i]["status"] = "running"
                        try:
                            meter = host.TimedMeter(ledger, lambda: torch.cuda.synchronize(0),
                                                   min(time.monotonic() + 900, started + 2100))
                            transition = host.measured_transition(compiler.make_transition(model, torch.device("cuda:0")), meter)
                            with core.native_model_scope(model, meter):
                                result = run_root(case, bank, transition, meter, "cuda:0")
                        finally:
                            ledger.close()
                        audit = consumer.audit_complete(jpath, result, bank)
                        result_sha = host.base.write_new(output / f"root_{case['base']}_result.json", result)
                        handoff = physical_handoff(case, result)
                        handoff_sha = host.base.write_new(output / f"root_{case['base']}_handoff.json", handoff)
                        rows[i].update(status="complete", journal_sha256=host.base.file_sha(jpath),
                                       result_sha256=result_sha, handoff_sha256=handoff_sha, audit=audit)
                        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                             <= plan["output_cap_bytes"], "output cap")
                torch.cuda.synchronize(0)
                checkpoint.frozen(model)
                report["frozen_parameters_unchanged"] = checkpoint.tensor_hash(model) == checkpoint.MODEL_TENSOR_SHA
                need(report["frozen_parameters_unchanged"], "model drift")
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
        verify_plan(plan_path, digest)
        report["status"] = "complete_model_only_physical_blocked"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        for row in rows:
            if row["status"] == "running":
                row["status"] = "failed_partial"
    report["wall_seconds"] = time.monotonic() - started
    return report, host.base.write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    check = sub.add_parser("check")
    check.add_argument("--plan", required=True)
    check.add_argument("--plan-sha256", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = path_new(args.output, "_PLAN_")
        print(json.dumps({"plan": str(path), "sha256": host.base.write_new(path, plan_value())}))
    elif args.command == "check":
        verify_plan(args.plan, args.plan_sha256)
        print(json.dumps({"status": "verified_no_gpu_or_physics"}))
    else:
        report, sha = execute(args.plan, args.plan_sha256, args.output, args.authorization)
        print(json.dumps({"status": report["status"], "report_sha256": sha}))
        if report["error"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
