"""Fixed v8 ridge delta: nonlinear native CUDA replay, never action execution."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as parent

NAME = "p5_jepa_native_delta_replay_20261027_v9"
PARENT = "results/p5_jepa_native_action_response_20261027_v8"
GATES = ("count_contract_matches", "frame_contract_matches", "phase_contract_matches",
         "source_data_seals_unchanged", "frozen_state_unchanged")


def delta_from_arrays(columns, residual):
    """Array-only preflight; same frozen .01 ridge, no candidate selection."""
    return np.asarray(parent.ridge_arrays(columns, residual)["delta"], np.float64)


def prepare(root):
    frozen_path = root / PARENT / "manifest_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    parent.verify_manifest(root, frozen)
    receipt_path = root / PARENT / "episode1_native_action_response_cuda_main_v1.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != "native_action_response_recorded"
            or not all(receipt[k] for k in GATES)
            or receipt["counts"] != parent.protocol()["counts"]
            or receipt["manifest_canonical_sha256"] != hashlib.sha256(
                json.dumps(frozen, sort_keys=True).encode()).hexdigest()
            or receipt["response_compile"]["applied"]
            or receipt["response_compile"]["nonlinear_candidate_evaluated"]):
        raise ValueError("complete immutable v8 response receipt required")
    arrays = {}
    for key, entry in receipt["audit_arrays"].items():
        path = Path(entry["path"])
        if parent.sha(path) != entry["sha256"]:
            raise ValueError("v8 numeric artifact hash rejected")
        arrays[key] = np.load(path, allow_pickle=False, mmap_mode="r")
        if list(arrays[key].shape) != entry["shape"] or str(arrays[key].dtype) != entry["dtype"]:
            raise ValueError("v8 numeric artifact schema rejected")
    j = arrays["response_columns"]
    r = arrays["target_minus_successor"].reshape(-1)
    if not np.array_equal(r, (arrays["official_target"].astype(np.float64)
                              - arrays["nominal_successor"].astype(np.float64)).reshape(-1)):
        raise ValueError("v8 target residual identity rejected")
    delta = np.asarray(receipt["response_compile"]["delta_normalized"], np.float64).reshape(10)
    if not np.allclose(delta, delta_from_arrays(j, r), rtol=1e-10, atol=1e-12):
        raise ValueError("frozen ridge delta numerical replay rejected")
    seals = dict(frozen["seals"])
    paths = [Path(__file__).resolve(), root / "tests" / ("test_" + NAME + ".py"),
             root.parent / "docs" / (NAME + ".md"), frozen_path, receipt_path]
    paths += [Path(entry["path"]) for entry in receipt["audit_arrays"].values()]
    seals.update({str(p): parent.sha(p) for p in paths})
    return dict(schema=NAME, status="prepared_non_neural", parent_manifest=frozen,
                seals=seals, parent_receipt_sha256=parent.sha(receipt_path),
                nominal_sha256=receipt["input_binding"]["successor_sha256"],
                nominal_mse=receipt["target_residual"]["global_patch_mse"],
                linear_mse=receipt["response_compile"]["linear_residual_mse"],
                delta_normalized=delta.tolist(),
                protocol=dict(episode=1, fixed_candidate="unchanged v8 ridge delta, float32 at model boundary",
                              counts=dict(official_encoder=1, proprio_encoder=1, predictor=3),
                              predictions=["nominal", "fixed_delta", "fixed_delta_exact_replay"],
                              selection=False, training=False, physical_execution=False,
                              target_role="already observed frame15 patch target; fitting diagnostic, not prospective planning evidence",
                              cpu_neural_fallback=False, P5_admission=False))


def run_cuda(root, manifest):
    if manifest != prepare(root):
        raise ValueError("frozen v9 manifest differs")
    parent.require_cuda()
    v7 = parent.helper(root)
    _, v5, _ = v7.dependencies(root)
    v4 = v5.import_v4(root)
    counts = dict(official_encoder=0, proprio_encoder=0, predictor=0)
    report = dict(schema=NAME, status="validation_failed", counts=counts,
                  protocol=manifest["protocol"], P5_admission=False,
                  physical_execution=False, cuda_executed=True)
    models = None
    handles = []
    phase = "load"
    start = time.perf_counter()
    try:
        models, strict, sources, before = parent.load_cuda(root, manifest["parent_manifest"])
        report.update(model_before=before, strict_comparison=strict, imported_sources=sources)
        def hook(key):
            def counted(module, args, output):
                counts[key] += 1
            return counted
        for key, module in models.items():
            handles.append(module.register_forward_hook(hook(key)))
        arrays = v5.read_arrays(v4, root)
        stats = v4.source_stats(v4.native_root(root) / "app/plan_common/datasets/pusht_dset.py")
        values, rgb, _ = v5.row_inputs(v4, root, arrays, stats, 1)
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import sdpa_kernel, SDPBackend
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            phase = "encode"
            pixels = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
            patches = models["official_encoder"](transform(pixels.clone()), is_training=True)["x_norm_patchtokens"]
            expected = manifest["parent_manifest"]["expected_v7_features"]
            if (v4.array_sha(patches.cpu().numpy()) != expected["official_features_sha256"]
                    or v4.array_sha(patches[3:4].cpu().numpy()) != expected["target_features_sha256"]):
                raise ValueError("official real feature identity rejected")
            visual = patches[:3].reshape(1, 3, 1, 16, 16, 384)
            target = patches[3:4]
            prop = torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
            features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            packets = torch.from_numpy(values["packets"].copy()).unsqueeze(0).to("cuda")
            delta = torch.tensor(manifest["delta_normalized"], device="cuda", dtype=torch.float32)
            def predict(offset):
                actions = torch.cat((packets[:, :2], packets[:, 2:3] + offset[None, None]), dim=1)
                return models["predictor"](visual, actions, features)[0][:, -1]
            phase = "nominal"
            nominal = predict(torch.zeros_like(delta))
            if v4.array_sha(nominal.cpu().numpy()) != manifest["nominal_sha256"]:
                raise ValueError("v8 nominal successor exact replay rejected")
            phase = "fixed_delta"
            candidate = predict(delta)
            phase = "fixed_delta_replay"
            replay = predict(delta)
            if not torch.isfinite(candidate).all() or not torch.equal(candidate, replay):
                raise ValueError("fixed delta finite/exact replay rejected")
            mse = (candidate.double() - target.double()).square().mean().item()
            nominal_mse = (nominal.double() - target.double()).square().mean().item()
            if abs(nominal_mse - manifest["nominal_mse"]) > 1e-6:
                raise ValueError("v8 nominal MSE replay rejected")
            report.update(status="native_delta_replay_recorded", nominal_mse=nominal_mse,
                candidate_mse=mse, candidate_minus_nominal_mse=mse - nominal_mse,
                frozen_linear_mse=manifest["linear_mse"],
                nonlinear_minus_linear_mse=mse - manifest["linear_mse"],
                delta_normalized=manifest["delta_normalized"],
                model_boundary_delta_sha256=v4.array_sha(delta.cpu().numpy()),
                candidate_sha256=v4.array_sha(candidate.cpu().numpy()), exact_replay=True)
    except Exception as exc:
        report.update(status="validation_failed", failed_phase=phase,
                      error=type(exc).__name__ + ": " + str(exc))
    finally:
        for handle in handles:
            handle.remove()
        if models is not None:
            report["model_after"] = {k: v4.model_receipt(m) for k, m in models.items()}
            report["model_unchanged"] = report["model_after"] == report.get("model_before")
        report["source_data_seals_unchanged"] = all(parent.sha(p) == h for p, h in manifest["seals"].items())
        report["counts_match"] = counts == manifest["protocol"]["counts"]
        report["elapsed_wall_seconds"] = time.perf_counter() - start
        if not all(report.get(k, False) for k in ("model_unchanged", "source_data_seals_unchanged", "counts_match")):
            report["status"] = "validation_failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--validate-cuda", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.validate_cuda and args.manifest is None:
        parser.error("--manifest required")
    result = run_cuda(args.root.resolve(), json.loads(args.manifest.read_text())) if args.validate_cuda else prepare(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(result["status"], flush=True)
    if result["status"] == "validation_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
