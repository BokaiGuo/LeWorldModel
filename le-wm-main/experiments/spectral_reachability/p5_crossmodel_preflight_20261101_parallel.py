"""P5 saved contract, lock-scoped CUDA strict-load preflight, and saved verification."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
NAME = "p5_crossmodel_preflight_20261101_parallel"
LOCK = RESULTS / "e3_p2_host_cuda_cooperative.lock"
PREPARE_OUTPUT = RESULTS / "p5_crossmodel_prepare_20261101_parallel.json"
CUDA_OUTPUT = RESULTS / "p5_crossmodel_cuda_20261101_parallel.json"
VERIFY_OUTPUT = RESULTS / "p5_crossmodel_verify_20261101_parallel.json"
DEFAULT_INPUTS = {
    "panel": {
        "path": RESULTS / "p5_crossmodel_panel_binding_20261030_codex_v1_MAIN01.json",
        "sha256": "80ed10d3b8f623d361cc25e2764d390bc9cc758178c89e7467e99da3d785ef98",
    },
    "jepa_abi": {
        "path": RESULTS / "p5_jepa_rgb_checkpoint_abi_20261030_codex_v1_MAIN01.json",
        "sha256": "e50df3f1f8c139dea00405adaff7f03df6b7d5b3d2d00101298dd6271866ee13",
    },
    "lewm_checkpoint": {
        "path": RESULTS / "causal_visual_pusht_phaseutility_formaltrain30_20261009_seed20261009"
                          / "causal_visual_world_model.pt",
        "sha256": "ca9b51ed2b98923bebc59941ba24612bc5ab8f59d7afdf6239334b550350d0dd",
    },
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read_fixed(ref, parse_json=True):
    path = Path(ref["path"])
    require(path.is_file() and not path.is_symlink(), f"regular pinned input: {path}")
    raw = path.read_bytes()
    require(digest(raw) == ref["sha256"], f"pinned SHA: {path}")
    return strict_json(raw) if parse_json else raw


def build_prepare(refs):
    require(set(refs) == {"panel", "jepa_abi", "lewm_checkpoint"}, "exact P5 inputs")
    panel = read_fixed(refs["panel"])
    jepa = read_fixed(refs["jepa_abi"])
    checkpoint = read_fixed(refs["lewm_checkpoint"], parse_json=False)
    require(panel["schema"] == "p5_crossmodel_panel_binding_20261030_codex_v1" and
            panel["status"] == "saved_input_physical_identity_bound_not_scientific_admission" and
            panel["branch_denominator"] == 164 and panel["pair_denominator"] == 80 and
            panel["known_roots"] == 2 and panel["fresh_roots_certified"] == 0,
            "fixed known-root panel contract")
    require(panel["P1_admission"] is panel["P5_admission"] is False, "panel admission")
    require("agent position and velocity" in panel["permissions"]["jepa"] and
            "RGB-only" in panel["permissions"]["lewm"], "saved permission difference")
    require(jepa["schema"] == "p5_jepa_rgb_checkpoint_abi_20261030_codex_v1" and
            jepa["status"] == "static_native_abi_and_rgb_only_gap_recorded" and
            jepa["P1_admission"] is jepa["P5_admission"] is False,
            "JEPA ABI contract")
    require(jepa["model_config_from_pinned_loader"]["use_proprio"] is True and
            "proprio" in jepa["native_h1_predictor_input_abi"] and
            jepa["analysis"]["same_permission_status"].startswith("blocked:"),
            "native JEPA proprio requirement")
    require(digest(checkpoint) == refs["lewm_checkpoint"]["sha256"], "LeWM checkpoint stable")
    return {
        "schema": NAME + ".prepare",
        "status": "p5_crossmodel_cuda_preflight_prepared",
        "inputs": {name: {"path": str(Path(ref["path"])), "sha256": ref["sha256"]}
                   for name, ref in refs.items()},
        "branch_denominator": 164,
        "pair_denominator": 80,
        "known_roots": 2,
        "fresh_roots_certified": 0,
        "lewm_checkpoint_sha256": refs["lewm_checkpoint"]["sha256"],
        "jepa_checkpoint_sha256": jepa["pinned_sha256"]["checkpoint"],
        "same_permission_comparison_ready": False,
        "cuda_execution_ready_for_bounded_preflight": True,
        "blockers": [
            "native_jepa_requires_proprio",
            "no_verified_rgb_only_native_jepa_checkpoint",
            "known_roots_only",
            "no_independent_confirmation",
            "upstream_stage_admission_false",
        ],
        "training_calls": 0,
        "physics_calls": 0,
        "P1_admission": False,
        "P5_admission": False,
        "independent_confirmation": False,
        "boundary": "Preparation permits only bounded CUDA strict-load checks; it does not authorize physics, efficacy comparison, or scientific admission.",
    }


def build_cuda_receipt(prepared, runtime):
    require(prepared["P5_admission"] is False and
            prepared["independent_confirmation"] is False, "prepared gate boundary")
    require(runtime["lock_held"] is True, "cooperative lock required")
    require(runtime["cuda_available"] is True, "CUDA required; no CPU fallback")
    require(runtime["checkpoint_before"] == runtime["checkpoint_after"] ==
            prepared["lewm_checkpoint_sha256"], "frozen checkpoint identity")
    require(runtime["strict_load"] is runtime["frozen"] is runtime["eval"] is True,
            "strict frozen eval model")
    require(runtime["neural_forward_calls"] == 0, "preflight does not run inference")
    return {
        "schema": NAME + ".cuda",
        "status": "p5_cuda_strict_load_preflight_passed",
        "prepare_sha256": prepared.get("_saved_sha256"),
        "lock_path": str(LOCK),
        "lock_held_during_cuda": True,
        "cuda_available": True,
        "device_index": runtime["device_index"],
        "device_name": runtime["device_name"],
        "torch_version": runtime.get("torch_version"),
        "cuda_runtime_version": runtime.get("cuda_runtime_version"),
        "driver_version": runtime.get("driver_version"),
        "memory_before": runtime.get("memory_before"),
        "memory_after_load": runtime.get("memory_after_load"),
        "lewm_checkpoint_sha256_before": runtime["checkpoint_before"],
        "lewm_checkpoint_sha256_after": runtime["checkpoint_after"],
        "lewm_strict_load": True,
        "lewm_frozen_eval": True,
        "jepa_live_strict_load": False,
        "jepa_live_strict_load_blocker": "native interface requires privileged proprio and is not same-permission RGB-only",
        "same_permission_comparison_ready": False,
        "neural_forward_calls": 0,
        "training_calls": 0,
        "physics_calls": 0,
        "P1_admission": False,
        "P5_admission": False,
        "independent_confirmation": False,
        "evidence_level": "host-CUDA model strict-load preflight",
    }


def verify_saved(prepared, cuda):
    require(prepared["schema"] == NAME + ".prepare" and
            prepared["status"] == "p5_crossmodel_cuda_preflight_prepared", "prepare receipt")
    require(cuda["schema"] == NAME + ".cuda" and
            cuda["status"] == "p5_cuda_strict_load_preflight_passed", "CUDA receipt")
    require(cuda["lock_held_during_cuda"] is True and cuda["cuda_available"] is True,
            "saved lock/CUDA evidence")
    require(cuda["lewm_checkpoint_sha256_before"] ==
            cuda["lewm_checkpoint_sha256_after"] ==
            prepared["lewm_checkpoint_sha256"], "saved checkpoint identity")
    require(cuda["lewm_strict_load"] is cuda["lewm_frozen_eval"] is True and
            cuda["jepa_live_strict_load"] is False and
            cuda["same_permission_comparison_ready"] is False, "saved model boundary")
    require(cuda["neural_forward_calls"] == cuda["training_calls"] ==
            cuda["physics_calls"] == 0, "preflight-only counters")
    require(prepared["P5_admission"] is cuda["P5_admission"] is False and
            prepared["independent_confirmation"] is
            cuda["independent_confirmation"] is False, "saved admissions")
    return {
        "schema": NAME + ".verify",
        "status": "p5_crossmodel_cuda_preflight_saved_validation_passed",
        "prepare_sha256": prepared.get("_saved_sha256"),
        "cuda_sha256": cuda.get("_saved_sha256"),
        "same_permission_comparison_ready": False,
        "fresh_roots_certified": 0,
        "P1_admission": False,
        "P5_admission": False,
        "independent_confirmation": False,
        "evidence_level": "independent saved-receipt reconstruction",
        "blockers": prepared["blockers"],
    }


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        body = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest(path.read_bytes())


def load_saved(path):
    raw = Path(path).read_bytes()
    value = strict_json(raw)
    value["_saved_sha256"] = digest(raw)
    return value


def run_cuda(prepared, device_index):
    require(LOCK.parent == RESULTS and LOCK.name == "e3_p2_host_cuda_cooperative.lock",
            "fixed cooperative lock")
    LOCK.touch(exist_ok=True)
    with LOCK.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("cooperative CUDA lock busy") from exc
        from experiments.spectral_reachability import p5_rgb_cuda_runner_20261030_codex_v1 as runner
        before = digest(Path(DEFAULT_INPUTS["lewm_checkpoint"]["path"]).read_bytes())
        contract = runner.load_prepared()
        checkpoint, _ = runner.verify_sources()
        backend = runner.CudaBackend(checkpoint, contract, device_index)
        torch = backend.torch
        device = backend.device
        torch.cuda.synchronize(device)
        runtime = {
            "lock_held": True,
            "cuda_available": torch.cuda.is_available(),
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "driver_version": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
            "memory_before": list(torch.cuda.mem_get_info(device)),
            "memory_after_load": list(torch.cuda.mem_get_info(device)),
            "checkpoint_before": before,
            "checkpoint_after": digest(Path(DEFAULT_INPUTS["lewm_checkpoint"]["path"]).read_bytes()),
            "strict_load": True,
            "frozen": not backend.model.training and all(not p.requires_grad for p in backend.model.parameters()),
            "eval": not backend.model.training,
            "neural_forward_calls": 0,
        }
        receipt = build_cuda_receipt(prepared, runtime)
        del backend
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "verify"))
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    if args.phase == "prepare":
        report = build_prepare(DEFAULT_INPUTS)
        output = PREPARE_OUTPUT
    elif args.phase == "run":
        prepared = load_saved(PREPARE_OUTPUT)
        report = run_cuda(prepared, args.device_index)
        output = CUDA_OUTPUT
    else:
        prepared, cuda = load_saved(PREPARE_OUTPUT), load_saved(CUDA_OUTPUT)
        report = verify_saved(prepared, cuda)
        output = VERIFY_OUTPUT
    sha256 = write_new(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": sha256},
                     sort_keys=True))


if __name__ == "__main__":
    main()
