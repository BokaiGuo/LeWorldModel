"""SHA-bound restricted meta inspection, and separately opt-in CUDA strict-load only.

No dataset access, unsafe allowlisting, encoder construction or model forward.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml
from torch._subclasses.fake_tensor import FakeTensorMode

NAME = "p5_jepa_strict_metadata_20261027_v3"
HASHES = {
    "pusht": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
    "pointmaze": "a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc",
    "wall": "8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e",
}
PROPRIO_DIMS = {"pusht": 4, "pointmaze": 4, "wall": 2}
CONFIGS = {
    "pusht": "pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml",
    "pointmaze": "mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml",
    "wall": "wall_sweep/wall_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def load_meta(path, expected):
    if Path(path).stat().st_size > 256 * 1024**2 or sha(path) != expected:
        raise ValueError("checkpoint size/SHA binding rejected")
    # mmap avoids eager storage reads; FakeTensorMode avoids real tensor storage.
    # Never change weights_only or add globals after any rejection.
    with FakeTensorMode():
        return torch.load(path, weights_only=True, map_location="meta", mmap=True)


def nested_metadata(value, path=()):
    if isinstance(value, torch.Tensor):
        if value.device.type != "meta":
            raise ValueError("non-meta tensor rejected")
        return [{"path": list(path), "kind": "tensor", "shape": list(value.shape),
                 "dtype": str(value.dtype), "device": str(value.device),
                 "stride": list(value.stride()), "storage_offset": value.storage_offset()}]
    if isinstance(value, (dict, list, tuple)):
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        entries = list(entries)
        out = [{"path": list(path), "kind": type(value).__name__, "keys": [k for k, _ in entries]}]
        for key, child in entries:
            if not isinstance(key, (str, int)):
                raise ValueError("unsupported metadata key")
            out.extend(nested_metadata(child, (*path, key)))
        return out
    if value is None or isinstance(value, (str, int, float, bool)):
        return [{"path": list(path), "kind": type(value).__name__, "value": value}]
    raise ValueError("unsupported restricted object: " + type(value).__name__)


def expected_schema(cfg, proprio_dim=4):
    model, data = cfg["model"], cfg["data"]
    assert model["predictor"]["pred_type"] == "AdaLN"
    assert model["visual_encoder"]["embed_dim"] == 384
    assert model["predictor"]["pred_embed_dim"] == 384
    assert model["predictor"]["pred_depth"] == 6
    assert model["predictor"]["pred_num_heads"] == 16
    assert model["predictor"]["use_SiLU"] is False
    assert model["predictor"]["use_rope"] is True
    assert model["predictor"]["tubelet_size"] == 1
    assert model["num_frames_pred"] == 4 and data["img_size"] == 224
    assert model["attn"]["local_window_time"] == 3
    assert model["attn"]["local_window_h"] == -1 and model["attn"]["local_window_w"] == -1
    assert model["action_conditioning"] == "token"
    assert model["proprio_encoding"] == "feature"
    assert model["proprio_encoder"]["proprio_emb_dim"] == 16
    assert model["proprio_encoder"]["proprio_encoder_inpred"] is False
    assert model["proprio_encoder"]["prop_mlp"] is False
    assert model["action_encoder"]["action_encoder_inpred"] is True
    assert model["tubelet_size_enc"] == 1
    assert data["custom"]["frameskip"] == 5 and data["custom"]["action_skip"] == 1
    # utils.py AdaLN branch + AdaLN_vit.py constructor; no model instantiated.
    # External task dimensions: action=2, proprio=4 (PushT/Maze) or 2 (Wall).
    predictor = {}
    def linear(prefix, out_dim, in_dim):
        predictor[prefix + ".weight"] = [out_dim, in_dim]
        predictor[prefix + ".bias"] = [out_dim]
    def norm(prefix):
        predictor[prefix + ".weight"] = [400]
        predictor[prefix + ".bias"] = [400]
    linear("predictor_embed", 384, 384)
    linear("action_encoder", 400, 10)
    for i in range(6):
        p = f"predictor_blocks.{i}."
        norm(p + "norm1")
        norm(p + "norm2")
        linear(p + "attn.qkv", 1200, 400)
        linear(p + "attn.proj", 400, 400)
        linear(p + "mlp.fc1", 1600, 400)
        linear(p + "mlp.fc2", 400, 1600)
        linear(p + "adaLN_modulation.1", 2400, 400)
    norm("predictor_norm")
    linear("predictor_proj", 384, 384)
    return {"predictor": predictor,
            "proprio_encoder": {"patch_embed.weight": [16, proprio_dim, 1], "patch_embed.bias": [16]}}


def normalize(state):
    result = {}
    for key, tensor in state.items():
        clean = key.removeprefix("module.")
        if clean in result:
            raise ValueError("key collision after single DDP-prefix removal")
        result[clean] = tensor
    return result


def compare(state, expected):
    actual = normalize(state)
    common = sorted(actual.keys() & expected.keys())
    result = {"missing_keys": sorted(expected.keys() - actual.keys()),
              "unexpected_keys": sorted(actual.keys() - expected.keys()),
              "shape_mismatches": [{"key": k, "actual": list(actual[k].shape), "expected": expected[k]}
                                   for k in common if list(actual[k].shape) != expected[k]],
              "dtype_mismatches": [{"key": k, "actual": str(actual[k].dtype), "expected": "torch.float32"}
                                   for k in common if actual[k].dtype != torch.float32],
              "expected_key_count": len(expected), "actual_key_count": len(actual)}
    result["exact_match"] = not any(result[k] for k in (
        "missing_keys", "unexpected_keys", "shape_mismatches", "dtype_mismatches"))
    return result


def source_records(root):
    base = root / "external/jepa-wms"
    paths = [base / p for p in (
        "app/plan_common/models/AdaLN_vit.py", "app/plan_common/models/prop_embedding.py",
        "app/vjepa_wm/utils.py", "src/models/utils/modules.py", "src/utils/tensors.py",
        "src/utils/logging.py")]
    paths += [base / "configs/vjepa_wm" / p for p in CONFIGS.values()]
    paths += [Path(__file__).resolve()]
    paths += [root / "experiments/theory_spectral_transfer" / f for f in (
        "run_jepa_wm_temporal_eval.py", "run_jepa_wm_pointmaze_temporal_eval.py",
        "run_jepa_wm_wall_temporal_eval.py")]
    return [{"path": str(p), "sha256": sha(p)} for p in paths]


def inspect(root):
    results = []
    for task, digest in HASHES.items():
        path = root / f"data/jepa_wm_checkpoints/jepa_wm_{task}.pth.tar"
        row = {"task": task, "path": str(path), "expected_sha256": digest}
        try:
            value = load_meta(path, digest)
            row["nested_metadata"] = nested_metadata(value)
            cfg_path = root / "external/jepa-wms/configs/vjepa_wm" / CONFIGS[task]
            cfg = yaml.safe_load(cfg_path.read_text())
            expected = expected_schema(cfg, PROPRIO_DIMS[task])
            row["config"] = cfg
            row["source_derived_expected_shapes"] = expected
            row["comparison"] = {k: compare(value[k], shapes) for k, shapes in expected.items()}
            row["cross_task_pusht_proprio_comparison"] = compare(
                value["proprio_encoder"], expected_schema(cfg, 4)["proprio_encoder"])
            row["native_interface"] = {"primitive_action_dim": 2, "packed_action_dim": 10,
                                       "proprio_dim": PROPRIO_DIMS[task]}
            row["top_level_keys"] = list(value)
            row["external_action_encoder"] = "absent as expected: embedded in predictor"
            row["status"] = "exact_source_schema_match" if all(v["exact_match"] for v in row["comparison"].values()) else "mismatch"
            del value
        except Exception as exc:
            row["status"] = "restricted_load_or_inspection_rejected"
            row["exception_type"] = type(exc).__name__
            row["exception"] = str(exc)
        results.append(row)
    return {"schema": NAME, "torch_version": torch.__version__, "checkpoints": results,
            "sources": source_records(root), "tensor_storage": "FakeTensorMode + mmap + meta, no real tensor storage allocated",
            "normalization": "remove exactly one leading module.; reject collisions; no other renames",
            "native_modules_instantiated": False, "model_forward": False, "cuda_executed": False,
            "encoder_runtime_verified": False, "D2P_ready": False, "P5_admission": False,
            "boundary": "Exact source-derived parameter schema, not actual native module construction or value/encoder/full-runtime certification."}


def validate_cuda(root, metadata_report, task):
    """Main-thread-only strict loading on CUDA; no forward, encoder or dataset."""
    report = json.loads(metadata_report.read_text())
    sources_before = source_records(root)
    if report["schema"] != NAME or report["sources"] != sources_before:
        raise ValueError("source/config seal drift")
    row = next(r for r in report["checkpoints"] if r["task"] == task)
    if row["status"] != "exact_source_schema_match":
        raise ValueError("no exact metadata match")
    path = root / f"data/jepa_wm_checkpoints/jepa_wm_{task}.pth.tar"
    checkpoint_sha_before = sha(path)
    if checkpoint_sha_before != HASHES[task]:
        raise ValueError("checkpoint SHA drift")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    sys.path.insert(0, str(root / "external/jepa-wms"))
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
    from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding
    # Only this explicitly requested mode materializes tensor storage, on CUDA.
    with torch.device("cuda"):
        predictor = vit_predictor_AdaLN(img_size=224, patch_size=14, num_frames=4,
            tubelet_size=1, embed_dim=384, predictor_embed_dim=384, depth=6,
            num_heads=16, use_silu=False, use_rope=True, local_window=(3, -1, -1),
            action_dim=10, proprio_dim=PROPRIO_DIMS[task], act_mlp=False, prop_mlp=False,
            proprio_encoder_inpred=False, action_encoder_inpred=True,
            proprio_encoding="feature", proprio_emb_dim=16, proprio_tokens=0)
        proprio = ProprioceptiveEmbedding(num_frames=4, tubelet_size=1, in_chans=PROPRIO_DIMS[task],
            tokens_per_step=1, embed_dim=16, shift_input=False, use_mlp=False)
    value = torch.load(path, weights_only=True, map_location="cuda")
    outcomes, receipts = {}, {}
    for key, module in (("predictor", predictor), ("proprio_encoder", proprio)):
        expected = {k: list(v.shape) for k, v in module.state_dict().items()}
        check = compare(value[key], expected)
        if not check["exact_match"]:
            raise ValueError(json.dumps({key: check}))
        module.load_state_dict(normalize(value[key]), strict=True)
        module.eval()
        module.requires_grad_(False)
        if any(v.device.type != "cuda" for v in module.state_dict().values()):
            raise ValueError("non-CUDA module tensor")
        if any(p.requires_grad for p in module.parameters()) or any(m.training for m in module.modules()):
            raise ValueError("module must be frozen/eval")
        outcomes[key] = check
        receipts[key] = {"strict_load": True, "actual_native_state_dict_compared_before_load": True,
                         "all_state_tensors_cuda": True, "all_parameters_frozen": True,
                         "all_submodules_eval": True,
                         "state_tensor_count": len(module.state_dict()),
                         "parameter_numel": sum(p.numel() for p in module.parameters()),
                         "devices": sorted({str(v.device) for v in module.state_dict().values()})}
    torch.cuda.synchronize()
    sources_after = source_records(root)
    checkpoint_sha_after = sha(path)
    if sources_after != sources_before or checkpoint_sha_after != checkpoint_sha_before:
        raise ValueError("source/config/checkpoint drift during CUDA strict load")
    return {"schema": NAME, "task": task, "status": "native_dynamics_strict_load_only",
            "comparison": outcomes, "receipts": receipts,
            "sources_before": sources_before, "sources_after": sources_after,
            "source_hashes_unchanged": True, "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after, "checkpoint_hash_unchanged": True,
            "cuda_executed": True, "model_forward": False,
            "encoder_runtime_verified": False, "D2P_ready": False, "P5_admission": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validate-cuda", action="store_true")
    p.add_argument("--metadata-report", type=Path)
    p.add_argument("--task", choices=HASHES, default="pusht")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.validate_cuda and args.metadata_report is None:
        p.error("--validate-cuda requires --metadata-report")
    result = validate_cuda(args.root, args.metadata_report, args.task) if args.validate_cuda else inspect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "cuda_executed": result["cuda_executed"]}))


if __name__ == "__main__":
    main()
