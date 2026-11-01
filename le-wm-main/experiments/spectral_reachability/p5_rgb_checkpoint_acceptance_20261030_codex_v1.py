"""Offline checkpoint baseline and future runtime/supervisor evidence acceptance."""
import argparse
import io
import os
import re
from pathlib import Path

from experiments.spectral_reachability import p5_rgb_cuda_supervisor_20261030_codex_v1 as sup

gate, runner, abi = sup.gate, sup.runner, sup.abi
NAME = "p5_rgb_checkpoint_acceptance_20261030_codex_v1"
SUP_SHA = "bb5063f84fe4c410839a458c3fcc36d5966649172b745e6509a0ccc529faba75"
STATE_FIELDS = {"modules", "parameters", "normalization", "precision"}


def pinned():
    abi.check(abi.bridge.digest(Path(sup.__file__).read_bytes()) == SUP_SHA, "supervisor SHA")
    sup.checked_gate()


def tensor_record(t):
    """Pure byte observation; no model forward. Device-to-host copy if called on CUDA."""
    a = t.detach().cpu().contiguous().numpy()
    abi.check(str(a.dtype) == "float32", "float32 tensor required")
    return {"shape": list(a.shape), "dtype": "float32",
            "sha256": abi.bridge.digest(a.astype("<f4", copy=False).tobytes(order="C"))}


def capture(model, torch):
    """Future caller captures actual module and precision state; does not execute the model."""
    modules = [{"name": name, "class": type(mod).__qualname__, "training": bool(mod.training)}
               for name, mod in model.named_modules()]
    params = {}
    for name, p in model.named_parameters():
        params[name] = {**tensor_record(p), "device": str(p.device), "requires_grad": bool(p.requires_grad)}
    norm = {name: {**tensor_record(getattr(model, name)),
                   "device": str(getattr(model, name).device)}
            for name in ("_state_mean", "_state_std")}
    precision = {"default_dtype": str(torch.get_default_dtype()),
                 "matmul_precision": torch.get_float32_matmul_precision(),
                 "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                 "cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda"))}
    return {"modules": modules, "parameters": params, "normalization": norm, "precision": precision}


def baseline():
    """Strict-load actual checkpoint on CPU only; no forward, training or CUDA."""
    abi.check(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only checkpoint audit requires CUDA_VISIBLE_DEVICES empty")
    pinned()
    import torch
    from experiments.run_causal_visual_pusht_world_model import Normalization, VisualCausalWorldModel
    from experiments.evaluate_causal_visual_checkpoint import checkpoint_adapter_architecture, checkpoint_utility_mode
    checkpoint = abi.bridge.read_fixed("checkpoint")
    payload = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
    args = payload["args"]
    flags = ("structured_readout", "contact_frame_dynamics", "mode_conditioned_dynamics",
             "current_only_readout", "torque_features", "temporal_dynamics",
             "previous_action_dynamics", "explicit_angle_effect", "explicit_angle_readout",
             "explicit_goal_angle_readout", "rich_temporal_dynamics", "invariant_angle_readout",
             "contact_interface_dynamics", "angle_readout_current_only")
    abi.check(set(payload) == {"model", "normalization", "args"} and args["latent_dim"] == 96 and
              args["vision_encoder"] == "mask_moments" and args["temporal_context"] is True and
              args["image_size"] == 64 and args["action_repeat"] == 1 and
              all(args.get(k, False) is False for k in flags) and
              checkpoint_utility_mode(payload) == "phase" and
              checkpoint_adapter_architecture(payload) == "legacy", "checkpoint ABI")
    with torch.device("cpu"):
        model = VisualCausalWorldModel(96, "mask_moments", True, "phase", "legacy")
        loaded = model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        norm = Normalization(**payload["normalization"])
        model.set_state_normalization(*norm.tensors("cpu"))
    abi.check(not loaded.missing_keys and not loaded.unexpected_keys and len(payload["model"]) == 26 and
              sum(p.numel() for p in model.parameters()) == 93842, "strict checkpoint load")
    state = capture(model, torch)
    abi.check(set(state["parameters"]) == set(payload["model"]) and
              all(not x["training"] for x in state["modules"]) and
              all(x["device"] == "cpu" and not x["requires_grad"] for x in state["parameters"].values()),
              "baseline frozen state")
    return {"schema": NAME + ".baseline", "checkpoint_sha256": abi.bridge.FILES["checkpoint"][1],
            "prepared_sha256": runner.PREP_SHA, "supervisor_sha256": SUP_SHA,
            "model_source_sha256": abi.bridge.FILES["model_source"][1],
            "source_sha256": {**runner.CODE_PINS, **gate.PINS,
                              "experiments/spectral_reachability/p5_rgb_cuda_review_gate_20261030_codex_v1.py": sup.GATE_SHA,
                              "experiments/spectral_reachability/p5_rgb_cuda_supervisor_20261030_codex_v1.py": SUP_SHA,
                              "experiments/run_causal_visual_pusht_world_model.py": abi.bridge.FILES["model_source"][1],
                              "acceptance_source": abi.bridge.digest(Path(__file__).read_bytes())},
            "strict_load": True, "state_dict_entries": 26, "parameter_count": 93842,
            "torch_version": torch.__version__, "state": state,
            "actual_neural_forward_calls": 0, "actual_cuda_calls": 0,
            "runtime_evidence_available": False, "P5_admission": False}


def check_snapshots(base, before, after):
    abi.check(set(before) == set(after) == STATE_FIELDS, "runtime snapshot fields")
    abi.check(before == after, "runtime state/precision changed")
    expected = base["state"]
    abi.check(before["modules"] == expected["modules"], "module names/classes/eval")
    abi.check(set(before["parameters"]) == set(expected["parameters"]), "parameter names")
    for name, record in before["parameters"].items():
        expected_record = dict(expected["parameters"][name], device="cuda:0")
        abi.check(record == expected_record, "parameter checkpoint/device/frozen mismatch: " + name)
    for name, record in before["normalization"].items():
        abi.check(name in expected["normalization"] and
                  record == dict(expected["normalization"][name], device="cuda:0"), "normalization mismatch")
    abi.check(set(before["normalization"]) == set(expected["normalization"]), "normalization names")
    precision = before["precision"]
    abi.check(set(precision) == {"default_dtype", "matmul_precision", "cuda_matmul_allow_tf32",
                                "cudnn_allow_tf32", "cuda_autocast_enabled"} and
              precision["default_dtype"] == "torch.float32" and
              precision["matmul_precision"] in ("highest", "high", "medium") and
              all(type(precision[k]) is bool for k in
                  ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cuda_autocast_enabled")) and
              not precision["cuda_autocast_enabled"], "runtime precision")


def accept(base, bundle, supervisor_bytes, journal_bytes):
    """Consistency check only; all runtime attestations remain self-reported."""
    abi.check(set(bundle) == {"schema", "baseline_sha256", "supervisor_sha256", "journal_sha256",
                              "capture_producer_sha256", "before", "after"} and
              bundle["schema"] == NAME + ".runtime_bundle", "bundle fields")
    abi.check(bundle["baseline_sha256"] == abi.bridge.digest(abi.bridge.canonical(base)) and
              bundle["supervisor_sha256"] == abi.bridge.digest(supervisor_bytes) and
              bundle["journal_sha256"] == abi.bridge.digest(journal_bytes) and
              re.fullmatch(r"[0-9a-f]{64}", bundle["capture_producer_sha256"]), "bundle binding")
    check_snapshots(base, bundle["before"], bundle["after"])
    abi.check(supervisor_bytes.endswith(b"\n"), "supervisor log truncated")
    events = [abi.bridge.read_json(line) for line in supervisor_bytes.splitlines()]
    abi.check([x["event"] for x in events] == ["created", "admitted", "launch_attempt", "launched", "finished"],
              "supervisor event sequence")
    created, admitted, launch, launched, finished = events
    run_name = created["run_name"]
    abi.check(isinstance(run_name, str) and re.fullmatch(
        re.escape(runner.NAME + "_") + r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name), "run name")
    expected_command = [str(abi.bridge.ROOT / ".venv/bin/python"), "-m",
                        "experiments.spectral_reachability.p5_rgb_cuda_runner_20261030_codex_v1",
                        "--authorization", runner.AUTH, "--device-index", "0",
                        "--max-seconds", str(created["runner_seconds"]), "--output-name", run_name]
    abi.check(created["prepared_sha256"] == runner.PREP_SHA and created["gate_sha256"] == sup.GATE_SHA and
              created["retry"] is False and finished["retry"] is False and
              admitted["snapshot"]["uuid"] == created["gpu_uuid"] and
              sup.UUID.fullmatch(created["gpu_uuid"]) is not None and launch["command"] == expected_command and
              type(created["hard_seconds"]) is int and 1 <= created["hard_seconds"] <= 7200 and
              type(created["runner_seconds"]) is int and 1 <= created["runner_seconds"] <= 3600,
              "supervisor binding")
    sup.admit(admitted["snapshot"], created["min_free_mib"], created["min_total_mib"])
    abi.check(admitted["snapshot"]["mig_mode"] in ("Disabled", "N/A", "[N/A]") and
              admitted["snapshot"]["compute_mode"] in ("Default", "Exclusive_Process") and
              finished["status"] == "complete_self_report_only" and
              finished["execution"]["status"] == "exited" and finished["execution"]["returncode"] == 0 and
              finished["execution"]["pid"] == launched["pid"] and
              finished["journal"]["sha256"] == bundle["journal_sha256"], "supervisor completion")
    reviewed = gate.strict_review(runner.load_prepared(), journal_bytes, bundle["journal_sha256"])
    abi.check(reviewed["status"] == "complete_journal_verified" and
              reviewed["condition_counts"] == {"off": {"denominator": 82, "complete": 82},
                                               "on": {"denominator": 82, "complete": 82}},
              "complete fixed journal")
    return {"schema": NAME + ".accept", "status": "self_report_consistency_only_not_scientific",
            "checkpoint_sha256": base["checkpoint_sha256"], "prepared_sha256": runner.PREP_SHA,
            "supervisor_sha256": bundle["supervisor_sha256"], "journal_sha256": bundle["journal_sha256"],
            "capture_producer_sha256": bundle["capture_producer_sha256"],
            "condition_counts": reviewed["condition_counts"], "P5_admission": False,
            "boundary": "snapshots and execution are self-reported; no proof of capture timing or actual hardware"}


def read_new_file(path, expected):
    abi.check(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024, "evidence file")
    data = path.read_bytes()
    abi.check(abi.bridge.digest(data) == expected, "evidence SHA")
    return data


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--output-name", required=True)
    c = sub.add_parser("accept")
    for arg in ("baseline-name", "baseline-sha", "bundle-name", "bundle-sha", "run-suffix",
                "supervisor-sha", "journal-sha", "output-name"):
        c.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    results = abi.bridge.ROOT / "results"
    def name(value, prefix):
        abi.check(Path(value).name == value and value.startswith(prefix) and value.endswith(".json"), "file name")
        return results / value
    output = name(args.output_name, NAME + "_")
    if args.cmd == "baseline":
        report = baseline()
    else:
        pinned()
        abi.check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_suffix), "run suffix")
        base_bytes = read_new_file(name(args.baseline_name, NAME + "_"), args.baseline_sha)
        base = abi.bridge.read_json(base_bytes)
        abi.check(base == baseline(), "actual checkpoint baseline mismatch")
        bundle_bytes = read_new_file(name(args.bundle_name, NAME + "_"), args.bundle_sha)
        bundle = abi.bridge.read_json(bundle_bytes)
        sup_dir = results / (sup.NAME + "_" + args.run_suffix)
        run_dir = results / (runner.NAME + "_" + args.run_suffix)
        abi.check(not sup_dir.is_symlink() and not run_dir.is_symlink(), "evidence directories")
        sb = read_new_file(sup_dir / "supervisor.jsonl", args.supervisor_sha)
        jb = read_new_file(run_dir / "events.jsonl", args.journal_sha)
        events = [abi.bridge.read_json(line) for line in sb.splitlines()]
        abi.check(events and events[0].get("run_name") == run_dir.name, "supervisor suffix binding")
        report = accept(base, bundle, sb, jb)
    print(gate.write_new(output, report))


if __name__ == "__main__":
    main()
