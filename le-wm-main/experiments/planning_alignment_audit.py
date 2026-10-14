"""Planning-cost alignment audit for LeWorldModel.

This script measures whether LeWM latent costs rank physically closer endpoints
above farther endpoints. It is an offline diagnostic: candidates are built from
dataset trajectory windows and optionally action rollouts. Online success/score
should still be measured with eval.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_metrics import compute_eigenspace, cost_success_alignment
from experiments.io_utils import (
    load_policy,
    prepare_resumable_chunk_dir,
    save_json,
    save_npz,
    to_numpy,
    write_run_status,
)
from experiments.latent_eigenspace_audit import ENV_PRESETS, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(ENV_PRESETS), default="pusht")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--chunk-batches", type=int, default=100)
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--resume", action="store_true", help="append after existing chunks in output/planning_chunks instead of starting over")
    parser.add_argument("--resume-start-batch", type=int, default=0, help="override inferred resume batch index")
    parser.add_argument("--success-threshold", type=float, default=0.15)
    parser.add_argument("--max-mismatch-examples", type=int, default=256)
    parser.add_argument("--spectral-basis", default=None, help="optional latents.npz from Phase 1")
    parser.add_argument("--spectral-ks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--spectral-max-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=3072)
    return parser.parse_args()


def endpoint_costs(emb: torch.Tensor) -> torch.Tensor:
    goal = emb[:, -1]
    endpoints = emb[:, 1:]
    return (endpoints - goal[:, None]).square().sum(dim=-1)


def endpoint_spectral_costs(emb: torch.Tensor, bases: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    goal = emb[:, -1]
    endpoints = emb[:, 1:]
    diff = (endpoints - goal[:, None]).reshape(-1, emb.size(-1))
    out = {}
    for name, basis in bases.items():
        proj = diff @ basis.to(device=emb.device, dtype=emb.dtype)
        out[name] = proj.square().sum(dim=-1).reshape(endpoints.shape[:2])
    return out


def load_spectral_bases(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if not args.spectral_basis:
        return {}
    with np.load(args.spectral_basis) as data:
        latents = data["latents"]
    if args.spectral_max_samples and latents.shape[0] > args.spectral_max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(latents.shape[0], size=args.spectral_max_samples, replace=False)
        latents = latents[idx]
    eig = compute_eigenspace(latents)
    return {f"spectral_top_{k}": eig.eigenvectors[:, : min(k, eig.eigenvectors.size(1))] for k in args.spectral_ks}


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
def collect_alignment(
    model,
    loader: DataLoader,
    args: argparse.Namespace,
    output: Path,
    spectral_bases: dict[str, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    preset = ENV_PRESETS[args.env]
    state_key = preset["state_key"]
    latent_costs = []
    state_distances = []
    successes = []
    goal_distances = []
    spectral_bases = spectral_bases or {}
    spectral_costs: dict[str, list[np.ndarray]] = {name: [] for name in spectral_bases}
    failure_rows = []
    chunk_dir, existing_chunk_paths, next_chunk_idx = prepare_resumable_chunk_dir(
        output / "planning_chunks", resume=args.resume
    )
    chunk_paths: list[Path] = list(existing_chunk_paths)
    if args.resume and existing_chunk_paths:
        if args.chunk_batches <= 0 and args.resume_start_batch <= 0:
            raise ValueError("--resume with existing planning chunks requires --chunk-batches > 0 or --resume-start-batch")
        start_batch = args.resume_start_batch or len(existing_chunk_paths) * args.chunk_batches
    else:
        start_batch = 0
    progress_path = output / "planning_progress.jsonl"
    if not args.resume:
        progress_path.write_text("")
    else:
        progress_path.touch()
    write_run_status(
        output,
        "phase2_planning_alignment",
        "running",
        resume=bool(args.resume),
        resume_start_batch=int(start_batch),
        existing_chunks=len(existing_chunk_paths),
        chunks=len(chunk_paths),
    )
    if start_batch:
        print(
            f"[planning] resuming from {len(existing_chunk_paths)} chunks; skipping {start_batch} batches",
            flush=True,
        )

    def flush_chunk(chunk_idx: int) -> None:
        if not latent_costs:
            return
        path = chunk_dir / f"chunk_{chunk_idx:06d}.npz"
        arrays = {
            "latent_cost": np.concatenate(latent_costs),
            "state_distance": np.concatenate(state_distances),
            "success": np.concatenate(successes),
            "goal_distance": np.concatenate(goal_distances),
        }
        for name, values in spectral_costs.items():
            if values:
                arrays[name] = np.concatenate(values)
        save_npz(path, **arrays)
        chunk_paths.append(path)
        for buf in (latent_costs, state_distances, successes, goal_distances):
            buf.clear()
        for values in spectral_costs.values():
            values.clear()

    resume_loader = fast_forward_loader(loader, start_batch)
    for local_batch_idx, batch in enumerate(resume_loader):
        batch_idx = start_batch + local_batch_idx
        if args.max_batches and batch_idx >= args.max_batches:
            break
        batch = {
            k: (torch.nan_to_num(v, 0.0).to(args.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        emb = model.encode(batch)["emb"]
        costs = endpoint_costs(emb)
        spectral = endpoint_spectral_costs(emb, spectral_bases) if spectral_bases else {}

        if state_key and state_key in batch:
            state = batch[state_key].reshape(batch[state_key].size(0), batch[state_key].size(1), -1)
        else:
            state = emb
        goal_state = state[:, -1]
        endpoint_state = state[:, 1:]
        dist = (endpoint_state - goal_state[:, None]).square().sum(dim=-1).sqrt()
        success = (dist <= args.success_threshold).float()
        initial_goal_dist = (state[:, :1] - goal_state[:, None]).square().sum(dim=-1).sqrt()
        initial_goal_dist = initial_goal_dist.expand_as(dist)

        latent_costs.append(to_numpy(costs.reshape(-1)))
        state_distances.append(to_numpy(dist.reshape(-1)))
        successes.append(to_numpy(success.reshape(-1)))
        goal_distances.append(to_numpy(initial_goal_dist.reshape(-1)))
        for name, value in spectral.items():
            spectral_costs[name].append(to_numpy(value.reshape(-1)))

        if len(failure_rows) < args.max_mismatch_examples:
            best_latent = costs.argmin(dim=1)
            best_true = dist.argmin(dim=1)
            mismatch = best_latent != best_true
            for row in torch.nonzero(mismatch, as_tuple=False).flatten():
                if len(failure_rows) >= args.max_mismatch_examples:
                    break
                failure_rows.append(
                    {
                        "batch": int(batch_idx),
                        "row": int(row.item()),
                        "latent_choice": int(best_latent[row].item()),
                        "true_choice": int(best_true[row].item()),
                        "latent_choice_state_distance": float(dist[row, best_latent[row]].item()),
                        "true_choice_state_distance": float(dist[row, best_true[row]].item()),
                        "latent_choice_cost": float(costs[row, best_latent[row]].item()),
                        "true_choice_cost": float(costs[row, best_true[row]].item()),
                    }
                )

        if args.progress_every and (batch_idx + 1) % args.progress_every == 0:
            record = {
                "batch": batch_idx + 1,
                "points_in_memory": int(sum(x.shape[0] for x in latent_costs)),
                "chunks": len(chunk_paths),
                "mismatch_examples": len(failure_rows),
            }
            with progress_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            write_run_status(
                output,
                "phase2_planning_alignment",
                "running",
                resume=bool(args.resume),
                batch=batch_idx + 1,
                chunks=len(chunk_paths),
                points_in_memory=record["points_in_memory"],
                mismatch_examples=record["mismatch_examples"],
            )
            print(f"[planning] processed {batch_idx + 1} batches, chunks={len(chunk_paths)}", flush=True)

        if args.chunk_batches and (batch_idx + 1) % args.chunk_batches == 0:
            flush_chunk(next_chunk_idx + len(chunk_paths) - len(existing_chunk_paths))

    flush_chunk(next_chunk_idx + len(chunk_paths) - len(existing_chunk_paths))
    if not chunk_paths:
        raise RuntimeError("no batches were collected; check dataset and loader settings")

    arrays = {}
    for key in ["latent_cost", "state_distance", "success", "goal_distance", *spectral_costs.keys()]:
        parts = []
        for path in chunk_paths:
            with np.load(path) as chunk:
                parts.append(chunk[key])
        arrays[key] = np.concatenate(parts)
    arrays["rank_mismatch_examples"] = np.asarray(failure_rows, dtype=object)

    if not args.keep_chunks:
        for path in chunk_paths:
            path.unlink()
    return arrays


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_policy(args.policy, device=args.device)
    spectral_bases = load_spectral_bases(args)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    arrays = collect_alignment(model, loader, args, output, spectral_bases=spectral_bases)
    raw_arrays = {
        key: value
        for key, value in arrays.items()
        if key != "rank_mismatch_examples"
    }
    save_npz(output / "planning_alignment_raw.npz", **raw_arrays)
    metrics = cost_success_alignment(
        arrays["latent_cost"],
        arrays["success"],
        arrays["state_distance"],
        arrays["goal_distance"],
    )
    metrics["num_points"] = int(arrays["latent_cost"].shape[0])
    if spectral_bases:
        metrics["spectral_planner_costs"] = {
            name: cost_success_alignment(
                arrays[name],
                arrays["success"],
                arrays["state_distance"],
                arrays["goal_distance"],
            )
            for name in spectral_bases
            if name in arrays
        }
    metrics["success_threshold"] = args.success_threshold
    metrics["rank_mismatch_examples"] = arrays["rank_mismatch_examples"].tolist()
    metrics["config"] = vars(args)
    save_json(output / "planning_alignment.json", metrics)
    write_run_status(
        output,
        "phase2_planning_alignment",
        "completed",
        num_points=int(arrays["latent_cost"].shape[0]),
        chunks=len(list((output / "planning_chunks").glob("chunk_*.npz"))),
    )


if __name__ == "__main__":
    main()

