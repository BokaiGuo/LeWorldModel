"""Offline latent eigenspace audit for LeWorldModel checkpoints.

Example:
    python experiments/latent_eigenspace_audit.py \
        --env pusht --policy pusht/lewm --output results/pusht/lewm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.directional_metrics import (
    band_energy_profile,
    cluster_states,
    compute_eigenspace,
    gaussian_control_like,
    identity_basis,
    lift_curve,
    random_orthonormal_basis,
    shuffled_control,
    spectral_descriptors,
    summarize_action_conditioned,
    summarize_by_cluster,
    tangent_cosine,
)
from experiments.io_utils import (
    load_policy,
    prepare_resumable_chunk_dir,
    save_json,
    save_npz,
    to_numpy,
    write_run_status,
)


ENV_PRESETS = {
    "pusht": {
        "dataset": "pusht_expert_train.h5",
        "keys": ["pixels", "action", "proprio", "state"],
        "state_key": "state",
        "physical_keys": ["proprio", "state"],
    },
    "tworoom": {
        "dataset": "tworoom.h5",
        "keys": ["pixels", "action", "proprio"],
        "state_key": "proprio",
        "physical_keys": ["proprio"],
    },
    "cube": {
        "dataset": "ogbench/cube_single_expert.h5",
        "keys": ["pixels", "action"],
        "state_key": None,
        "physical_keys": [],
    },
    "reacher": {
        "dataset": "dmc/reacher_random.h5",
        "keys": ["pixels", "action", "observation"],
        "state_key": "observation",
        "physical_keys": ["observation"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(ENV_PRESETS), default="pusht")
    parser.add_argument("--policy", default=None, help="stable_worldmodel policy name or checkpoint path")
    parser.add_argument("--dataset", default=None, help="override dataset name")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--save-max-samples", type=int, default=200000, help="maximum rows to write to latents.npz; 0 saves all collected rows")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--chunk-batches", type=int, default=50)
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--resume", action="store_true", help="append after existing chunks in output/chunks instead of starting over")
    parser.add_argument("--resume-start-batch", type=int, default=0, help="override inferred resume batch index")
    parser.add_argument("--finalize-chunks-only", action="store_true", help="skip collection and finalize existing chunk_*.npz files")
    parser.add_argument("--max-finalize-chunks", type=int, default=0, help="optional cap for finalize-only chunk count")
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--action-bins", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=256)
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--rich-fields", action="store_true", help="save goals, success proxies, episode/phase labels, physical_* arrays, and counterfactual tangents when available")
    parser.add_argument("--rich-horizons", type=int, nargs="+", default=[1, 3, 5], help="multi-step horizons exported with --rich-fields")
    parser.add_argument("--goal-threshold", type=float, default=0.15, help="distance threshold for success proxy when --rich-fields is enabled")
    parser.add_argument("--counterfactual-roll", type=int, default=4, help="pixel roll shift used for counterfactual_impossible_roll when --rich-fields is enabled")
    return parser.parse_args()


def _infer_history_size(model) -> int:
    predictor = getattr(model, "predictor", None)
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if pos_embedding is None:
        return 1
    return max(1, int(pos_embedding.size(1)))


def build_dataset(args: argparse.Namespace):
    import h5py
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_column_normalizer, get_img_preprocessor

    preset = ENV_PRESETS[args.env]
    dataset_name = args.dataset or preset["dataset"]
    requested_steps = int(args.num_steps)
    keys_to_load = list(preset["keys"])
    if getattr(args, "rich_fields", False):
        dataset_path = Path(swm.data.utils.get_cache_dir(sub_folder="datasets"), f"{dataset_name}.h5")
        with h5py.File(dataset_path, "r", swmr=True) as f:
            available = set(f.keys())
        optional_keys = [
            "goal",
            "goal_pixels",
            "goal_state",
            "goal_proprio",
            "goal_observation",
            "episode_idx",
            "ep_idx",
            "step_idx",
            *preset.get("physical_keys", []),
        ]
        keys_to_load.extend(key for key in optional_keys if key in available and key not in keys_to_load)
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        num_steps=requested_steps,
        frameskip=args.frameskip,
        keys_to_load=keys_to_load,
    )
    transforms: list[Any] = [get_img_preprocessor("pixels", "pixels")]
    for key in keys_to_load:
        if key in {"pixels", "goal", "goal_pixels", "episode_idx", "ep_idx", "step_idx"}:
            continue
        try:
            transforms.append(get_column_normalizer(dataset, key, key))
        except Exception:
            pass
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def _flatten_time(value: torch.Tensor, rows: int, steps: int | None = None) -> np.ndarray:
    value = torch.nan_to_num(value, 0.0)
    if steps is None:
        steps = value.size(1) - 1
    return to_numpy(value[:, :steps].reshape(rows, -1))


def _append_if_present(buffers: dict[str, list[np.ndarray]], key: str, value: np.ndarray | None) -> None:
    if value is not None:
        buffers.setdefault(key, []).append(value)


def _normalized_horizons(horizons: list[int] | tuple[int, ...]) -> list[int]:
    values = sorted({int(h) for h in horizons})
    if not values or values[0] < 1:
        raise ValueError("rich horizons must contain positive integers")
    return values


@torch.no_grad()
def autoregressive_horizon_predictions(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    horizons: list[int] | tuple[int, ...],
    history_size: int | None = None,
) -> dict[int, torch.Tensor]:
    """Predict future embeddings from every valid observed prefix.

    The returned tensors have shape ``(B, T-max_horizon, D)`` and use the
    observed action at each future step.  This mirrors ``JEPA.rollout`` while
    keeping the starting point for every row in the batch explicit.
    """

    horizon_list = _normalized_horizons(horizons)
    max_horizon = max(horizon_list)
    if emb.ndim != 3 or act_emb.ndim != 3 or emb.shape[:2] != act_emb.shape[:2]:
        raise ValueError("emb and act_emb must be aligned 3D tensors")
    if emb.size(1) <= max_horizon:
        raise ValueError("sequence is too short for requested horizons")
    if history_size is None:
        positional = getattr(getattr(model, "predictor", None), "pos_embedding", None)
        history_size = int(positional.size(1)) if positional is not None else emb.size(1)
    history_size = max(1, int(history_size))

    valid_steps = emb.size(1) - max_horizon
    predictions: dict[int, list[torch.Tensor]] = {h: [] for h in horizon_list}
    for start in range(valid_steps):
        context_start = max(0, start - history_size + 1)
        context_emb = emb[:, context_start : start + 1]
        context_act = act_emb[:, context_start : start + 1]
        for step in range(1, max_horizon + 1):
            pred = model.predict(context_emb, context_act)[:, -1]
            if step in predictions:
                predictions[step].append(pred)
            if step < max_horizon:
                next_act = act_emb[:, start + step : start + step + 1]
                context_emb = torch.cat((context_emb, pred[:, None]), dim=1)[:, -history_size:]
                context_act = torch.cat((context_act, next_act), dim=1)[:, -history_size:]

    return {h: torch.stack(values, dim=1) for h, values in predictions.items()}


def fast_forward_loader(loader: DataLoader, start_batch: int) -> DataLoader | list:
    if start_batch <= 0:
        return loader
    batch_size = loader.batch_size or 1
    try:
        dataset_len = len(loader.dataset)
    except TypeError:
        return loader
    start_item = start_batch * batch_size
    if start_item >= dataset_len:
        return []
    subset = torch.utils.data.Subset(loader.dataset, range(start_item, dataset_len))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        drop_last=False,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
    )


@torch.no_grad()
def collect_latents(
    model,
    loader: DataLoader,
    args: argparse.Namespace,
    output: Path,
) -> dict[str, np.ndarray]:
    preset = ENV_PRESETS[args.env]
    state_key = preset["state_key"]
    latents = []
    dynamics = []
    predicted_dynamics = []
    prediction_residual = []
    planning = []
    actions = []
    states = []
    nuisance = []
    rich: dict[str, list[np.ndarray]] = {}
    chunk_dir, existing_chunk_paths, next_chunk_idx = prepare_resumable_chunk_dir(
        output / "chunks", resume=args.resume
    )
    chunk_paths: list[Path] = list(existing_chunk_paths)
    if args.resume and existing_chunk_paths:
        if args.chunk_batches <= 0 and args.resume_start_batch <= 0:
            raise ValueError("--resume with existing chunks requires --chunk-batches > 0 or --resume-start-batch")
        start_batch = args.resume_start_batch or len(existing_chunk_paths) * args.chunk_batches
    else:
        start_batch = 0
    progress_path = output / "progress.jsonl"
    if not args.resume:
        progress_path.write_text("")
    else:
        progress_path.touch()
    write_run_status(
        output,
        "phase1_latent_audit",
        "running",
        resume=bool(args.resume),
        resume_start_batch=int(start_batch),
        existing_chunks=len(existing_chunk_paths),
        chunks=len(chunk_paths),
    )
    if start_batch:
        print(
            f"[audit] resuming from {len(existing_chunk_paths)} chunks; skipping {start_batch} batches",
            flush=True,
        )

    def flush_chunk(chunk_idx: int) -> None:
        if not latents:
            return
        arrays = {
            "latents": np.concatenate(latents, axis=0),
            "dynamics": np.concatenate(dynamics, axis=0),
            "planning": np.concatenate(planning, axis=0),
            "nuisance": np.concatenate(nuisance, axis=0),
        }
        if predicted_dynamics:
            arrays["predicted_dynamics"] = np.concatenate(predicted_dynamics, axis=0)
        if prediction_residual:
            arrays["prediction_residual"] = np.concatenate(prediction_residual, axis=0)
        if actions:
            arrays["actions"] = np.concatenate(actions, axis=0)
        if states:
            arrays["states"] = np.concatenate(states, axis=0)
        for key, values in rich.items():
            if values:
                arrays[key] = np.concatenate(values, axis=0)
        path = chunk_dir / f"chunk_{chunk_idx:06d}.npz"
        save_npz(path, **arrays)
        chunk_paths.append(path)
        for buf in (latents, dynamics, predicted_dynamics, prediction_residual, planning, actions, states, nuisance):
            buf.clear()
        rich.clear()

    resume_loader = fast_forward_loader(loader, start_batch)
    rich_horizons = _normalized_horizons([1, *args.rich_horizons]) if args.rich_fields else [1]
    max_horizon = max(rich_horizons)
    for local_batch_idx, batch in enumerate(resume_loader):
        batch_idx = start_batch + local_batch_idx
        if args.max_batches and batch_idx >= args.max_batches:
            break
        batch = {
            k: (torch.nan_to_num(v, 0.0).to(args.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        out = model.encode(batch)
        emb = out["emb"]
        step_count = emb.size(1) - max_horizon if args.rich_fields else emb.size(1) - 1
        if step_count <= 0:
            raise ValueError(
                f"sequence length {emb.size(1)} is too short for max horizon {max_horizon}"
            )
        z_seq = emb[:, :step_count]
        z_t = z_seq.reshape(-1, emb.size(-1))
        delta_seq = emb[:, 1 : step_count + 1] - z_seq
        delta = delta_seq.reshape(-1, emb.size(-1))
        latents.append(to_numpy(z_t))
        dynamics.append(to_numpy(delta))
        horizon_predictions: dict[int, torch.Tensor] = {}
        if "act_emb" in out:
            if args.rich_fields:
                horizon_predictions = autoregressive_horizon_predictions(
                    model,
                    emb,
                    out["act_emb"],
                    rich_horizons,
                )
                pred = horizon_predictions[1]
            else:
                pred = model.predict(emb[:, :-1], out["act_emb"][:, :-1])
            pred_delta = pred.reshape(-1, emb.size(-1)) - z_t
            predicted_dynamics.append(to_numpy(pred_delta))
            prediction_residual.append(to_numpy(delta - pred_delta))
        planning.append(to_numpy((emb[:, -1:] - z_seq).reshape(-1, emb.size(-1))))

        if "action" in batch:
            actions.append(to_numpy(batch["action"][:, :step_count].reshape(z_t.size(0), -1)))
        if state_key and state_key in batch:
            states.append(to_numpy(batch[state_key][:, :step_count].reshape(z_t.size(0), -1)))
        if args.rich_fields:
            rows = z_t.size(0)
            for horizon in rich_horizons:
                if horizon == 1:
                    continue
                true_h = emb[:, horizon : horizon + step_count] - z_seq
                _append_if_present(rich, f"dynamics_h{horizon}", to_numpy(true_h.reshape(rows, -1)))
                pred_h = horizon_predictions.get(horizon)
                if pred_h is not None:
                    pred_h_delta = pred_h.reshape(rows, -1) - z_t
                    true_h_delta = true_h.reshape(rows, -1)
                    _append_if_present(rich, f"predicted_dynamics_h{horizon}", to_numpy(pred_h_delta))
                    _append_if_present(rich, f"prediction_residual_h{horizon}", to_numpy(true_h_delta - pred_h_delta))
            for key in ["episode_idx", "ep_idx", "step_idx"]:
                if key in batch and torch.is_tensor(batch[key]):
                    out_key = "episode_id" if key in {"episode_idx", "ep_idx"} else "timestep"
                    _append_if_present(rich, out_key, _flatten_time(batch[key], rows, steps=step_count))
            if state_key and state_key in batch:
                state_seq = torch.nan_to_num(batch[state_key], 0.0)
                goal_state = state_seq[:, -1:].expand_as(state_seq[:, :step_count])
                goal_flat = to_numpy(goal_state.reshape(rows, -1))
                _append_if_present(rich, "goals", goal_flat)
                dist = (state_seq[:, :step_count].reshape(rows, -1) - goal_flat)
                success = (dist.square().sum(dim=1).sqrt() <= args.goal_threshold).float()
                _append_if_present(rich, "success", to_numpy(success[:, None]))
            for key in preset.get("physical_keys", []):
                if key in batch and torch.is_tensor(batch[key]):
                    _append_if_present(rich, f"physical_{key}", _flatten_time(batch[key], rows, steps=step_count))
            if "act_emb" in out and out["act_emb"].size(0) > 1:
                wrong_action_emb = torch.roll(out["act_emb"], shifts=1, dims=0)
                wrong_predictions = autoregressive_horizon_predictions(
                    model,
                    emb,
                    wrong_action_emb,
                    rich_horizons,
                )
                for horizon, pred_h in wrong_predictions.items():
                    suffix = "" if horizon == 1 else f"_h{horizon}"
                    _append_if_present(
                        rich,
                        f"counterfactual_wrong_action{suffix}",
                        to_numpy(pred_h.reshape(rows, -1) - z_t),
                    )

        aug = dict(batch)
        aug["pixels"] = torch.roll(batch["pixels"], shifts=2, dims=-1)
        aug_emb = model.encode(aug)["emb"][:, :step_count].reshape(-1, emb.size(-1))
        nuisance.append(to_numpy(aug_emb - z_t))
        if args.rich_fields and args.counterfactual_roll:
            impossible = dict(batch)
            impossible["pixels"] = torch.roll(batch["pixels"], shifts=int(args.counterfactual_roll), dims=-2)
            impossible_emb = model.encode(impossible)["emb"][:, :step_count].reshape(-1, emb.size(-1))
            _append_if_present(rich, "counterfactual_impossible_roll", to_numpy(impossible_emb - z_t))

        if args.progress_every and (batch_idx + 1) % args.progress_every == 0:
            record = {
                "batch": batch_idx + 1,
                "samples_in_memory": int(sum(x.shape[0] for x in latents)),
                "chunks": len(chunk_paths),
            }
            with progress_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            write_run_status(
                output,
                "phase1_latent_audit",
                "running",
                resume=bool(args.resume),
                batch=batch_idx + 1,
                chunks=len(chunk_paths),
                samples_in_memory=record["samples_in_memory"],
            )
            print(f"[audit] processed {batch_idx + 1} batches, chunks={len(chunk_paths)}", flush=True)

        if args.chunk_batches and (batch_idx + 1) % args.chunk_batches == 0:
            flush_chunk(next_chunk_idx + len(chunk_paths) - len(existing_chunk_paths))

    flush_chunk(next_chunk_idx + len(chunk_paths) - len(existing_chunk_paths))
    if not chunk_paths:
        raise RuntimeError("no batches were collected; check dataset and loader settings")

    arrays = load_latent_chunks_from_paths(chunk_paths)

    if not args.keep_chunks:
        for path in chunk_paths:
            path.unlink()
    return arrays


def load_latent_chunks(output: Path, max_chunks: int = 0) -> dict[str, np.ndarray]:
    chunk_paths = sorted((output / "chunks").glob("chunk_*.npz"))
    if max_chunks:
        chunk_paths = chunk_paths[:max_chunks]
    if not chunk_paths:
        raise RuntimeError(f"no latent chunks found in {output / 'chunks'}")
    return load_latent_chunks_from_paths(chunk_paths)


def load_latent_chunks_from_paths(chunk_paths: list[Path]) -> dict[str, np.ndarray]:
    keys = set()
    for path in chunk_paths:
        with np.load(path) as chunk:
            keys.update(chunk.files)
    arrays: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        parts = []
        for path in chunk_paths:
            with np.load(path) as chunk:
                if key in chunk:
                    parts.append(chunk[key])
        arrays[key] = np.concatenate(parts, axis=0)
    if "states" not in arrays:
        arrays["states"] = arrays["latents"]
    if "prediction_residual" not in arrays and {"dynamics", "predicted_dynamics"}.issubset(arrays):
        arrays["prediction_residual"] = arrays["dynamics"] - arrays["predicted_dynamics"]
    return arrays


def select_saved_latent_arrays(
    arrays: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = int(arrays["latents"].shape[0])
    max_samples = int(args.save_max_samples)
    if max_samples > 0 and n > max_samples:
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(n, size=max_samples, replace=False))
        saved_arrays = {
            key: value[rows] if isinstance(value, np.ndarray) and value.shape[:1] == (n,) else value
            for key, value in arrays.items()
        }
    else:
        rows = None
        saved_arrays = arrays

    manifest = {
        "num_total_samples": n,
        "num_saved_samples": int(saved_arrays["latents"].shape[0]),
        "save_max_samples": max_samples,
        "sampled": rows is not None,
        "seed": int(args.seed),
    }
    return saved_arrays, manifest


def save_latent_outputs(arrays: dict[str, np.ndarray], args: argparse.Namespace, output: Path) -> None:
    saved_arrays, manifest = select_saved_latent_arrays(arrays, args)
    save_npz(output / "latents.npz", **saved_arrays)
    save_json(output / "latents_manifest.json", manifest)

    labels = cluster_states(arrays["states"], args.clusters, seed=args.seed)
    tangents = {
        "dynamics": arrays["dynamics"],
        "planning": arrays["planning"],
        "nuisance": arrays["nuisance"],
        "shuffled_dynamics": to_numpy(shuffled_control(arrays["dynamics"], seed=args.seed)),
        "gaussian_dynamics": to_numpy(gaussian_control_like(arrays["dynamics"], seed=args.seed)),
    }
    if "predicted_dynamics" in arrays:
        tangents["predicted_dynamics"] = arrays["predicted_dynamics"]
    if "prediction_residual" in arrays:
        tangents["prediction_residual"] = arrays["prediction_residual"]
    metrics = summarize_by_cluster(
        arrays["latents"],
        tangents,
        labels,
        args.ks,
        min_cluster_size=args.min_cluster_size,
    )

    d = arrays["latents"].shape[1]
    global_eig = compute_eigenspace(arrays["latents"])
    bands = [(0, min(8, d)), (min(8, d), min(32, d)), (min(32, d), min(128, d)), (min(128, d), d)]
    metrics["global_spectrum"] = spectral_descriptors(global_eig.eigenvalues)
    global_controls = {
        "identity_dynamics_lift": lift_curve(identity_basis(d), arrays["dynamics"], args.ks),
        "random_basis_dynamics_lift": lift_curve(
            random_orthonormal_basis(d, seed=args.seed), arrays["dynamics"], args.ks
        ),
    }
    metrics["global_controls"] = global_controls
    metrics["global_band_profiles"] = {
        name: band_energy_profile(global_eig.eigenvectors, tangent, bands)
        for name, tangent in tangents.items()
        if not name.startswith(("shuffled", "gaussian"))
    }
    if "predicted_dynamics" in arrays:
        dyn_profile = metrics["global_band_profiles"]["dynamics"]
        pred_profile = metrics["global_band_profiles"]["predicted_dynamics"]
        metrics["prediction_tangent_alignment"] = {
            "mean_cosine": tangent_cosine(arrays["dynamics"], arrays["predicted_dynamics"]),
            "mse": float(np.square(arrays["dynamics"] - arrays["predicted_dynamics"]).mean()),
            "band_profile_l1": float(
                sum(abs(dyn_profile[k] - pred_profile.get(k, 0.0)) for k in dyn_profile)
            ),
        }
    if "actions" in arrays:
        action_tangents = {
            k: v
            for k, v in tangents.items()
            if k in {"dynamics", "predicted_dynamics", "prediction_residual", "planning"}
        }
        metrics["action_conditioned"] = summarize_action_conditioned(
            arrays["latents"],
            action_tangents,
            arrays["actions"],
            args.ks,
            n_action_bins=args.action_bins,
            min_bin_size=args.min_cluster_size,
            seed=args.seed,
        )
    metrics["num_samples"] = int(arrays["latents"].shape[0])
    metrics["config"] = vars(args)
    save_json(output / "lift_metrics.json", metrics)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.finalize_chunks_only:
        arrays = load_latent_chunks(output, max_chunks=args.max_finalize_chunks)
    else:
        if not args.policy:
            raise ValueError("--policy is required unless --finalize-chunks-only is set")
        model = load_policy(args.policy, device=args.device)
        if args.rich_fields:
            history_size = _infer_history_size(model)
            args.num_steps = max(args.num_steps, history_size + max(_normalized_horizons(args.rich_horizons)))
        dataset = build_dataset(args)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        arrays = collect_latents(model, loader, args, output)
    save_latent_outputs(arrays, args, output)
    write_run_status(
        output,
        "phase1_latent_audit",
        "completed",
        num_samples=int(arrays["latents"].shape[0]),
        chunks=len(list((output / "chunks").glob("chunk_*.npz"))),
    )


if __name__ == "__main__":
    main()
