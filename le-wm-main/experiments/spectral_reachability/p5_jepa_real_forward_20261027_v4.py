"""Bounded PushT real-data native dynamics check; CUDA only, offline HF shim disclosed.

--prepare is non-neural. --validate-cuda is for the host scheduler only.
Never imports the unsafe native dataset loader or opens its pickle files.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

NAME = "p5_jepa_real_forward_20261027_v4"
FRAME_IDS = [0, 5, 10, 15]
ARRAY_HASHES = {
    "states.pth": "7ef3ab74983415802e342642e41bf6429b927d3fff7a54db15da74745dd2fcb0",
    "rel_actions.pth": "c5e04761f261db6f96b2d16db35e5f1d7e42b8d2c507994713c98343d0f685dc",
    "velocities.pth": "b1e4694152206f34931cc5015dfefb4456608c42d9f6a7f2688d20f2fc90d59b",
}
CKPT_SHA = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
ENCODER_SHA = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def restricted_array(path, expected_sha):
    if Path(path).stat().st_size > 2 * 1024**2 or sha(path) != expected_sha:
        raise ValueError("array SHA/size rejected")
    value = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("expected plain CPU data tensor")
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite data array")
    return value.float().numpy()


def source_stats(path):
    """Read literal source constants without importing dataset code."""
    wanted = {"ACTION_MEAN", "ACTION_STD", "PROPRIO_MEAN", "PROPRIO_STD"}
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in wanted and isinstance(node.value, ast.Call):
                found[key] = ast.literal_eval(node.value.args[0])
    if found.keys() != wanted:
        raise ValueError("native normalization constants unresolved")
    return found


def histories(states, velocities, actions, stats):
    if states.ndim != 3 or states.shape[2] != 5 or states.shape[0] != 21:
        raise ValueError("states shape mismatch")
    if actions.shape != (*states.shape[:2], 2) or velocities.shape != actions.shape:
        raise ValueError("action/velocity shape mismatch")
    if states.shape[1] < 16:
        raise ValueError("short trajectory arrays")
    # No padded frames are selected: video independently must contain frame 15.
    prop_raw = np.concatenate([states[0, FRAME_IDS, :2], velocities[0, FRAME_IDS]], axis=-1)
    prop = (prop_raw - np.array(stats["PROPRIO_MEAN"], np.float32)) / np.array(stats["PROPRIO_STD"], np.float32)
    raw_actions = actions[0, :15].copy()
    # Native training normalization, unlike old temporal diagnostic's no-mean ablation.
    normalized = (raw_actions / 100.0 - np.array(stats["ACTION_MEAN"], np.float32)) / np.array(stats["ACTION_STD"], np.float32)
    return {"proprio_raw": prop_raw, "proprio_normalized": prop,
            "actions_raw": raw_actions, "action_packets": normalized.reshape(3, 10)}


def decode_video(path):
    # Native preferred decoder and RGB order; no dataset class, lengths pickle or fallback.
    import decord
    reader = decord.VideoReader(str(path), num_threads=1, ctx=decord.cpu(0))
    if len(reader) < 16 or len(reader) > 10000:
        raise ValueError("video frame count out of bounds")
    frames = reader.get_batch(FRAME_IDS).asnumpy()
    if frames.dtype != np.uint8 or frames.shape != (4, 224, 224, 3):
        # Deterministic full-frame resize comparison below requires square input;
        # allow other bounded square source resolutions.
        if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != 4
                or frames.shape[-1] != 3 or frames.shape[1] != frames.shape[2]
                or not 32 <= frames.shape[1] <= 1024):
            raise ValueError("unexpected decoded RGB layout/resolution")
    return frames, len(reader)


def native_root(root):
    return root / "external/jepa-wms"


def data_root(root):
    return root / "data/jepa_wm_osf/extracted/pusht_noise/val"


def bindings(root):
    paths = list((native_root(root) / "app").rglob("*.py"))
    paths += list((native_root(root) / "src").rglob("*.py"))
    paths += [Path(__file__).resolve(), root / "external/local_dinov2/hubconf.py",
              root / ".cache/dinov2/dinov2-small/config.json",
              root / ".cache/dinov2/dinov2-small/model.safetensors",
              root / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar",
              root / "results/p5_jepa_strict_metadata_20261027_v3/report_frozen.json",
              root / "results/p5_jepa_strict_metadata_20261027_v3/pusht_native_strict_cuda_main_v1.json"]
    paths += [data_root(root) / name for name in ARRAY_HASHES]
    paths += [data_root(root) / "obses/episode_000.mp4"]
    frozen = json.loads((root / "results/p5_jepa_strict_metadata_20261027_v3/report_frozen.json").read_text())
    paths += [Path(s["path"]) for s in frozen["sources"]]
    return [{"path": str(p), "sha256": sha(p)} for p in sorted(set(paths))]


def check_v3(root):
    base = root / "results/p5_jepa_strict_metadata_20261027_v3"
    meta = json.loads((base / "report_frozen.json").read_text())
    receipt = json.loads((base / "pusht_native_strict_cuda_main_v1.json").read_text())
    if (receipt["task"] != "pusht" or not receipt["cuda_executed"] or receipt["model_forward"]
            or not receipt["source_hashes_unchanged"] or not receipt["checkpoint_hash_unchanged"]
            or receipt["sources_before"] != receipt["sources_after"]
            or receipt["sources_before"] != meta["sources"]
            or receipt["checkpoint_sha256_after"] != CKPT_SHA):
        raise ValueError("invalid native strict CUDA receipt")
    for record in meta["sources"]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("v3 source seal drift")
    for part in ("predictor", "proprio_encoder"):
        if not receipt["comparison"][part]["exact_match"]:
            raise ValueError("native strict comparison absent")
        if not all(receipt["receipts"][part][k] for k in (
            "strict_load", "all_state_tensors_cuda", "all_parameters_frozen", "all_submodules_eval")):
            raise ValueError("native CUDA/freeze receipt rejected")
    encoder_root = root / ".cache/dinov2/dinov2-small"
    cfg = json.loads((encoder_root / "config.json").read_text())
    if (cfg.get("model_type") != "dinov2" or cfg.get("hidden_size") != 384
            or cfg.get("patch_size") != 14 or cfg.get("num_hidden_layers") != 12
            or cfg.get("num_attention_heads") != 6
            or sha(encoder_root / "model.safetensors") != ENCODER_SHA):
        raise ValueError("offline encoder identity rejected")


def real_inputs(root):
    arrays = {n: restricted_array(data_root(root) / n, digest) for n, digest in ARRAY_HASHES.items()}
    stats = source_stats(native_root(root) / "app/plan_common/datasets/pusht_dset.py")
    h = histories(arrays["states.pth"], arrays["velocities.pth"], arrays["rel_actions.pth"], stats)
    rgb, video_length = decode_video(data_root(root) / "obses/episode_000.mp4")
    return h, rgb, stats, video_length


def prepare(root):
    check_v3(root)
    before = bindings(root)
    h, rgb, stats, length = real_inputs(root)
    if before != bindings(root):
        raise ValueError("input/source drift during preparation")
    return {"schema": NAME, "status": "prepared_non_neural", "bindings": before,
            "frame_ids": FRAME_IDS, "context_frames": FRAME_IDS[:3], "target_frame": 15,
            "primitive_action_slices": [[0, 5], [5, 10], [10, 15]], "episode": 0,
            "video_length": length, "decoded_rgb_shape": list(rgb.shape), "decoded_rgb_sha256": array_sha(rgb),
            "histories": {k: {"shape": list(v.shape), "sha256": array_sha(v), "values": v.tolist()} for k, v in h.items()},
            "native_stats": stats, "cuda_executed": False, "P5_admission": False,
            "encoder_equivalence_to_official_dinov2": "unverified; native source routes to HF offline shim"}


def verify_seal(root, manifest):
    if manifest["schema"] != NAME or manifest["status"] != "prepared_non_neural" or manifest["bindings"] != bindings(root):
        raise ValueError("v4 input/source seal drift")
    check_v3(root)


def model_receipt(module):
    h = hashlib.sha256()
    for key, v in module.state_dict().items():
        if v.device.type != "cuda" or not torch.isfinite(v).all():
            raise ValueError("model state must be finite CUDA")
        h.update(key.encode())
        h.update(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    if any(p.requires_grad for p in module.parameters()) or any(m.training for m in module.modules()):
        raise ValueError("model must remain frozen/eval")
    return {"state_sha256": h.hexdigest(), "all_cuda_finite": True, "frozen": True, "eval": True}


def encoder_header_comparison(root, model):
    from safetensors import safe_open
    with safe_open(root / ".cache/dinov2/dinov2-small/model.safetensors", framework="pt", device="cpu") as f:
        expected = {k: f.get_slice(k).get_shape() for k in f.keys()}
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    if actual != expected:
        raise ValueError(json.dumps({"encoder_missing": sorted(expected.keys()-actual.keys()),
            "encoder_unexpected": sorted(actual.keys()-expected.keys()),
            "encoder_shape_mismatch": [k for k in actual.keys() & expected.keys() if actual[k] != expected[k]]}))
    return {"cached_safetensor_keys_shapes_exact": True, "key_count": len(actual),
            "official_original_implementation_equivalence": False}


def validate_cuda(root, manifest):
    verify_seal(root, manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no neural CPU fallback")
    # Force all assets offline and override environment-dependent shim routes.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["JEPAWM_DINOV2_SOURCE"] = str(root / "external/local_dinov2")
    os.environ["LEWM_DINOV2_WEIGHTS"] = str(root / ".cache/dinov2/dinov2-small")
    sys.path.insert(0, str(native_root(root)))
    v3path = root / "experiments/spectral_reachability/p5_jepa_strict_metadata_20261027_v3.py"
    spec = importlib.util.spec_from_file_location("p5_v3_sealed", v3path)
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
    from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding
    from app.plan_common.models.dino import DinoEncoder
    from app.plan_common.datasets.transforms import make_transforms
    h, rgb, stats, length = real_inputs(root)
    if array_sha(rgb) != manifest["decoded_rgb_sha256"] or any(
            array_sha(v) != manifest["histories"][k]["sha256"] for k, v in h.items()):
        raise ValueError("real input replay differs from preparation")
    with torch.device("cuda"):
        predictor = vit_predictor_AdaLN(img_size=224, patch_size=14, num_frames=4, tubelet_size=1,
            embed_dim=384, predictor_embed_dim=384, depth=6, num_heads=16, use_silu=False,
            use_rope=True, local_window=(3, -1, -1), action_dim=10, proprio_dim=4,
            act_mlp=False, prop_mlp=False, proprio_encoder_inpred=False, action_encoder_inpred=True,
            proprio_encoding="feature", proprio_emb_dim=16, proprio_tokens=0)
        proprio = ProprioceptiveEmbedding(num_frames=4, tubelet_size=1, in_chans=4,
            tokens_per_step=1, embed_dim=16, shift_input=False, use_mlp=False)
        encoder = DinoEncoder("dinov2_vits14", "x_norm_patchtokens")
    # No neural CPU execution: .to only ensures construction-path device differences are removed.
    modules = {"predictor": predictor, "proprio_encoder": proprio, "encoder_native_HF_shim": encoder}
    for mod in modules.values():
        mod.to("cuda").eval().requires_grad_(False)
    encoder_header = encoder_header_comparison(root, encoder.base_model.model)
    checkpoint = root / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"
    if sha(checkpoint) != CKPT_SHA:
        raise ValueError("checkpoint SHA drift")
    weights = torch.load(checkpoint, weights_only=True, map_location="cuda")
    strict = {}
    for key, mod in (("predictor", predictor), ("proprio_encoder", proprio)):
        strict[key] = v3.compare(weights[key], {k: list(v.shape) for k, v in mod.state_dict().items()})
        if not strict[key]["exact_match"]:
            raise ValueError(json.dumps(strict[key]))
        mod.load_state_dict(v3.normalize(weights[key]), strict=True)
    del weights
    before = {k: model_receipt(v) for k, v in modules.items()}
    # Two independently expressed preprocessing paths on real decoded RGB.
    pixels = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.0
    native_transform = make_transforms(img_size=224, random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
        reprob=0., auto_augment=False, motion_shift=False)
    preprocessed = native_transform(pixels.clone())
    reference = torch.nn.functional.interpolate(pixels, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    reference = (reference - mean) / std
    preprocessing_delta = (preprocessed - reference).abs().max().item()
    if preprocessing_delta > 1e-6:
        raise ValueError("native/reference preprocessing mismatch")
    with torch.inference_mode():
        patches = encoder(preprocessed)
        paired = encoder.base_model.model(pixel_values=preprocessed).last_hidden_state[:, 1:]
        interface_delta = (patches - paired).abs().max().item()
        if patches.shape != (4, 256, 384) or interface_delta > 1e-6:
            raise ValueError("native-source/HF shim interface mismatch")
        visual = patches[:3].reshape(1, 3, 1, 16, 16, 384)
        prop = torch.from_numpy(h["proprio_normalized"][:3].copy()).unsqueeze(0).to("cuda")
        prop_features = proprio(prop).expand(1, 3, 256, 16)
        acts = torch.from_numpy(h["action_packets"].copy()).unsqueeze(0).to("cuda")
        alternative = acts.clone()
        alternative[:, -1] = 0.  # only new packet ablated; both history packets preserved
        if torch.equal(acts, alternative):
            raise ValueError("zero-new-packet intervention is identical; do not select another sample")
        observed = predictor(visual, acts, prop_features)
        repeated = predictor(visual, acts, prop_features)
        changed = predictor(visual, alternative, prop_features)
        for output in (observed, repeated, changed):
            if output[0].shape != (1, 3, 256, 384) or output[2].shape != (1, 3, 256, 16):
                raise ValueError("native successor shape mismatch")
            if any(not torch.isfinite(v).all() or v.device.type != "cuda" for v in (output[0], output[2])):
                raise ValueError("nonfinite/nonCUDA successor")
        successor = observed[0][:, -1]
        repeat_delta = (successor - repeated[0][:, -1]).abs().max().item()
        action_delta = (successor - changed[0][:, -1]).abs().max().item()
        target_mse = (successor - patches[3:4]).square().mean().item()
    after = {k: model_receipt(v) for k, v in modules.items()}
    if before != after:
        raise ValueError("model parameter/buffer drift")
    verify_seal(root, manifest)
    gates = {"finite_native_successors": True, "native_reference_preprocessing_match": True,
             "same_shim_source_interface_match": True, "frozen_model_bytes_unchanged": True,
             "repeat_max_abs_le_1e_6": repeat_delta <= 1e-6,
             "new_packet_action_delta_gt_1e_7": action_delta > 1e-7}
    imported = [{"module": n, "path": str(Path(m.__file__).resolve()), "sha256": sha(m.__file__)}
                for n, m in sorted(sys.modules.items()) if getattr(m, "__file__", None)
                and str(m.__file__).endswith(".py") and (n.startswith(("app.", "src.", "transformers.models.dinov2")) or "hubconf" in n)]
    return {"schema": NAME, "status": "bounded_shim_backed_native_forward_recorded" if all(gates.values()) else "validation_failed",
            "gates": gates, "cuda_executed": True, "strict_comparison": strict,
            "model_before": before, "model_after": after, "source_and_input_seals_unchanged": True,
            "encoder_cache_header_comparison": encoder_header,
            "imported_source_receipts": imported, "episode": 0, "frame_ids": FRAME_IDS,
            "action_packets": h["action_packets"].tolist(), "proprio_raw": h["proprio_raw"].tolist(),
            "preprocessing_max_abs": preprocessing_delta, "interface_max_abs": interface_delta,
            "repeat_successor_max_abs": repeat_delta, "new_packet_action_successor_max_abs": action_delta,
            "real_target_patch_mse_descriptive_only": target_mse,
            "encoder_pair_boundary": "native DinoEncoder -> local HF shim versus direct SAME HF model; NOT independent official DINOv2 equivalence",
            "official_dinov2_equivalence_proven": False, "P5_admission": False, "D2P_ready": False,
            "neural_forward_counts": {"encoder_native": 1, "encoder_direct_same_HF": 1, "proprio": 1, "predictor": 3},
            "no_dataset_pickle": True, "no_training": True, "cpu_neural_fallback": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--validate-cuda", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.validate_cuda and args.manifest is None:
        p.error("--validate-cuda requires --manifest")
    result = prepare(args.root) if args.prepare else validate_cuda(args.root, json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]}))


if __name__ == "__main__":
    main()
