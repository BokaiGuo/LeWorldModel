"""CUDA-only SRA P0: audit action-Jacobians of the published PointMaze LeWM.

P0 deliberately establishes *plumbing*, not a planning claim.  It uses the
published PointMaze checkpoint and its native RGB/proprio preprocessing, keeps
every model parameter frozen, and differentiates only through ``EncPredWM``'s
``unroll`` method with respect to a normalized action suffix.  It records:

* an action-to-terminal-latent Jacobian sketch and central-finite-difference
  agreement;
* parameter hashes before/after the run, so an accidental optimizer/model
  update fails the audit;
* matched CEM, vanilla action-GD, and a clearly-labelled gradient-and-restart
  (GRASP-style) optimizer on the same latent-goal objective.

The GRASP-style arm is an in-repository gradient-and-restart baseline, not a
claim of bit-for-bit reproduction of an external GRASP implementation.  This
file intentionally refuses CPU: a learned-model result obtained after a CUDA
failure is not a valid substitute for this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/point_maze_tail1800_1899"
DEFAULT_CHECKPOINT = ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DEFAULT_CONFIG = ROOT / "external/jepa-wms/configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    """Hash immutable parameter values with names/shapes, independent of device."""

    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("SRA P0 is CUDA-only: torch.cuda.is_available() is false")
    device = torch.device("cuda")
    # An allocation catches driver/runtime mismatches that availability alone
    # can conceal.  No CPU fallback exists anywhere in this runner.
    torch.empty(1, device=device)
    return device


def parse_roots(value: str) -> list[int]:
    roots = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(roots) < 2 or len(roots) != len(set(roots)):
        raise ValueError("--roots needs at least two distinct held-out episode ids")
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--roots", default="1896,1897,1898,1899")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--probe-dims", type=int, default=16)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--fd-pairs", type=int, default=8)
    parser.add_argument("--action-bound", type=float, default=3.0)
    parser.add_argument("--cem-population", type=int, default=24)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--cem-elite", type=int, default=6)
    parser.add_argument("--gd-steps", type=int, default=12)
    parser.add_argument("--gd-lr", type=float, default=0.15)
    parser.add_argument("--grasp-restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026102301)
    return parser.parse_args()


def _install_local_dinov2(source: Path) -> None:
    """Bind any DINO hub request to the checked-in local source, never network."""

    if not source.exists():
        raise FileNotFoundError(f"local DINOv2 source is required: {source}")
    original = torch.hub.load

    def local_hub(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir == "facebookresearch/dinov2":
            return original(str(source), model, source="local", *args, **kwargs)
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = local_hub


def load_model_and_dataset(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    if str(ROOT / "external/jepa-wms") not in sys.path:
        sys.path.insert(0, str(ROOT / "external/jepa-wms"))
    _install_local_dinov2(args.dinov2_source)
    from app.plan_common.datasets.point_maze_dset import PointMazeDataset
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    aug = config["data_aug"]
    transform = make_transforms(
        img_size=224,
        normalize=aug["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    dataset = PointMazeDataset(data_path=str(args.dataset_root), normalize_action=True)
    preprocessor = Preprocessor(
        dataset.action_mean, dataset.action_std,
        dataset.state_mean, dataset.state_std,
        dataset.proprio_mean, dataset.proprio_std, transform,
    )
    model = init_module(
        str(args.checkpoint.parent), args.checkpoint.name, config["model"], device,
        action_dim=2, proprio_dim=4, preprocessor=preprocessor,
        cfgs_data=config["data"], wrapper_kwargs={"ctxt_window": 1},
    ).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen-model contract failed: a model parameter still requires grad")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("frozen-model contract failed: a model parameter is not on CUDA")
    return model, dataset


def prepare_archive_observation_for_model(dataset: Any, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Undo the dataset-only normalization before ``EncPredWM.encode`` owns it.

    ``PointMazeDataset.get_frames`` serves training tensors: RGB is already in
    [0, 1] and proprio is already z-scored.  The released ``EncPredWM.encode``
    interface instead accepts raw [0, 255] RGB and raw proprio, then performs
    its own /255, visual transform, and proprio normalization.  Supplying the
    archive tensors directly silently applied both operations twice.  Actions
    are intentionally excluded: all SRA action packets are already normalized
    exactly once for the predictor's action encoder.
    """

    if set(observation) != {"visual", "proprio"}:
        raise ValueError(f"unexpected PointMaze observation keys: {sorted(observation)}")
    visual, proprio = observation["visual"].float(), observation["proprio"].float()
    if visual.ndim != 4 or proprio.ndim != 2:
        raise ValueError(f"unexpected archive observation shapes: visual={tuple(visual.shape)}, proprio={tuple(proprio.shape)}")
    if not torch.isfinite(visual).all() or float(visual.min()) < -1.0e-6 or float(visual.max()) > 1.0 + 1.0e-6:
        raise ValueError("PointMaze archive visual tensor must be dataset-scaled RGB in [0, 1]")
    raw_proprio = proprio * dataset.proprio_std.to(proprio) + dataset.proprio_mean.to(proprio)
    if not torch.isfinite(raw_proprio).all():
        raise ValueError("PointMaze archive proprio de-normalization produced non-finite values")
    return {"visual": visual * 255.0, "proprio": raw_proprio}


def make_root_context(model: Any, dataset: Any, root: int, start: int, horizon: int, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Use only recorded RGB/proprio context and its future RGB frame as goal."""

    from tensordict import TensorDict

    goal_frame = start + 5 * horizon
    sequence_length = int(dataset.get_seq_length(root))
    if goal_frame >= sequence_length:
        raise ValueError(f"root {root} is too short for frame {goal_frame}; length={sequence_length}")
    context_raw, _, _, _, _ = dataset.get_frames(root, [start])
    goal_raw, _, _, _, _ = dataset.get_frames(root, [goal_frame])
    context_obs = {key: value.unsqueeze(0).to(device) for key, value in prepare_archive_observation_for_model(dataset, context_raw).items()}
    goal_obs = {key: value.unsqueeze(0).to(device) for key, value in prepare_archive_observation_for_model(dataset, goal_raw).items()}
    # Encoding is intentionally no-grad in the published wrapper: SRA P0 only
    # needs action derivatives through the frozen predictor/unroll, not pixels.
    with torch.no_grad():
        encoded_context = model.encode(context_obs)
        encoded_goal = model.encode(goal_obs)
    context = TensorDict(
        {"visual": encoded_context["visual"], "proprio": encoded_context["proprio"]},
        batch_size=[1, 1],
        device=device,
    )
    goal = encoded_goal["visual"][0, 0].mean(dim=(0, 1, 2)).detach()
    observed = dataset.actions[root, start : start + 5 * horizon].reshape(horizon, -1).to(device)
    return context, goal, observed


def terminal_latent(model: Any, context: Any, action_suffix: torch.Tensor) -> torch.Tensor:
    """Return one pooled terminal visual latent per candidate without detaching actions."""

    rollout = model.unroll(context, act_suffix=action_suffix)
    terminal = rollout["visual"][-1]  # [B, views, grid_h, grid_w, latent]
    return terminal.mean(dim=(1, 2, 3))


def terminal_cost(model: Any, context: Any, suffix: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    latent = terminal_latent(model, context, suffix)
    return torch.mean((latent - goal.unsqueeze(0)) ** 2, dim=1)


def to_bounded_actions(raw: torch.Tensor, bound: float) -> torch.Tensor:
    return float(bound) * torch.tanh(raw)


def run_cem(model: Any, context: Any, goal: torch.Tensor, *, horizon: int, action_dim: int, bound: float, population: int, iterations: int, elite: int, seed: int, device: torch.device) -> dict[str, Any]:
    if elite < 2 or elite >= population:
        raise ValueError("CEM requires 2 <= elite < population")
    generator = torch.Generator(device=device).manual_seed(seed)
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones_like(mean)
    best_cost = math.inf
    best_suffix: torch.Tensor | None = None
    trace: list[float] = []
    for _ in range(iterations):
        raw = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(horizon, population, action_dim, generator=generator, device=device)
        suffix = to_bounded_actions(raw, bound)
        with torch.no_grad():
            costs = terminal_cost(model, context, suffix, goal)
        order = torch.argsort(costs)
        if float(costs[order[0]]) < best_cost:
            best_cost = float(costs[order[0]])
            best_suffix = suffix[:, order[0]].detach().clone()
        elites = raw[:, order[:elite]]
        mean = elites.mean(dim=1)
        std = elites.std(dim=1, unbiased=False).clamp_min(0.05)
        trace.append(float(costs[order[0]]))
    assert best_suffix is not None
    return {"best_cost": best_cost, "best_suffix": best_suffix, "trace": trace}


def run_vanilla_gd(model: Any, context: Any, goal: torch.Tensor, *, horizon: int, action_dim: int, bound: float, steps: int, lr: float, initial_raw: torch.Tensor | None = None) -> dict[str, Any]:
    raw = torch.zeros(horizon, 1, action_dim, device=goal.device) if initial_raw is None else initial_raw.detach().clone().unsqueeze(1)
    raw.requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=lr)
    trace: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        cost = terminal_cost(model, context, to_bounded_actions(raw, bound), goal).mean()
        cost.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all() or float(raw.grad.norm()) <= 1.0e-12:
            raise RuntimeError("vanilla action-GD has a missing, non-finite, or identically-zero action gradient")
        optimizer.step()
        trace.append(float(cost.detach()))
    suffix = to_bounded_actions(raw.detach(), bound)
    with torch.no_grad():
        best_cost = float(terminal_cost(model, context, suffix, goal)[0])
    return {"best_cost": best_cost, "best_suffix": suffix[:, 0], "trace": trace}


def run_grasp_style(model: Any, context: Any, goal: torch.Tensor, *, horizon: int, action_dim: int, bound: float, restarts: int, steps: int, lr: float, seed: int, device: torch.device) -> dict[str, Any]:
    """Gradient-and-restart comparator, explicitly not an external GRASP clone."""

    generator = torch.Generator(device=device).manual_seed(seed)
    candidates: list[dict[str, Any]] = []
    for index in range(restarts):
        initial = torch.randn(horizon, action_dim, generator=generator, device=device)
        result = run_vanilla_gd(model, context, goal, horizon=horizon, action_dim=action_dim, bound=bound, steps=steps, lr=lr, initial_raw=initial)
        candidates.append({"restart": index, **result})
    winner = min(candidates, key=lambda item: float(item["best_cost"]))
    return {"best_cost": float(winner["best_cost"]), "best_suffix": winner["best_suffix"], "trace": winner["trace"], "all_restart_costs": [float(item["best_cost"]) for item in candidates]}


def jacobian_and_fd(model: Any, context: Any, goal: torch.Tensor, *, horizon: int, action_dim: int, bound: float, probe_dims: int, epsilon: float, fd_pairs: int, seed: int, device: torch.device) -> dict[str, Any]:
    """Validate real action derivatives against central finite differences."""

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        base_latent = terminal_latent(model, context, torch.zeros(horizon, 1, action_dim, device=device))[0]
    latent_dim = int(base_latent.numel())
    if probe_dims > latent_dim:
        raise ValueError(f"probe dims {probe_dims} exceeds latent dimension {latent_dim}")
    random_basis = torch.randn(latent_dim, probe_dims, generator=generator, device=device)
    basis, _ = torch.linalg.qr(random_basis, mode="reduced")

    def projected(flat_actions: torch.Tensor) -> torch.Tensor:
        suffix = to_bounded_actions(flat_actions.reshape(horizon, 1, action_dim), bound)
        return terminal_latent(model, context, suffix)[0] @ basis

    origin = torch.zeros(horizon * action_dim, device=device, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(projected, origin, vectorize=True)
    if jacobian.shape != (probe_dims, horizon * action_dim):
        raise RuntimeError(f"unexpected Jacobian shape {tuple(jacobian.shape)}")
    if not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1.0e-10:
        raise RuntimeError("action-to-latent Jacobian is non-finite or numerically zero")
    pairs = min(fd_pairs, probe_dims, horizon * action_dim)
    fd_rows = []
    for index in range(pairs):
        output_index = index
        input_index = (3 * index + 1) % (horizon * action_dim)
        plus = origin.detach().clone()
        minus = origin.detach().clone()
        plus[input_index] += epsilon
        minus[input_index] -= epsilon
        with torch.no_grad():
            finite_difference = float((projected(plus)[output_index] - projected(minus)[output_index]) / (2.0 * epsilon))
        autograd_value = float(jacobian[output_index, input_index])
        abs_error = abs(autograd_value - finite_difference)
        relative_error = abs_error / max(abs(autograd_value), abs(finite_difference), 1.0e-6)
        fd_rows.append({"output_index": output_index, "action_index": input_index, "autograd": autograd_value, "central_fd": finite_difference, "absolute_error": abs_error, "relative_error": relative_error})
    singular_values = torch.linalg.svdvals(jacobian)
    return {
        "latent_dim": latent_dim,
        "jacobian_shape": list(jacobian.shape),
        "frobenius_norm": float(jacobian.norm()),
        "singular_values": [float(value) for value in singular_values],
        "effective_rank_1e-4": int((singular_values > singular_values[0] * 1.0e-4).sum()),
        "finite_difference": fd_rows,
        "median_relative_error": float(np.median([item["relative_error"] for item in fd_rows])),
        "max_relative_error": float(max(item["relative_error"] for item in fd_rows)),
        "goal_latent_norm": float(goal.norm()),
    }


def order_sensitivity(model: Any, context: Any, *, horizon: int, action_dim: int, bound: float, seed: int, device: torch.device) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    suffix = to_bounded_actions(torch.randn(horizon, 1, action_dim, generator=generator, device=device), bound)
    with torch.no_grad():
        forward = terminal_latent(model, context, suffix)[0]
        reversed_order = terminal_latent(model, context, suffix.flip(0))[0]
    return float(torch.linalg.vector_norm(forward - reversed_order))


def main() -> None:
    args = parse_args()
    roots = parse_roots(args.roots)
    if args.output_dir.exists():
        raise FileExistsError(f"P0 output is immutable; refusing to reuse {args.output_dir}")
    if args.horizon < 1 or args.start_frame < 0 or args.probe_dims < 2 or args.fd_pairs < 1:
        raise ValueError("horizon/probe dimensions/frame must be positive and probe_dims >= 2")
    for path in (args.dataset_root, args.checkpoint, args.config):
        if not path.exists():
            raise FileNotFoundError(path)
    device = require_cuda()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    args.output_dir.mkdir(parents=True)
    model, dataset = load_model_and_dataset(args, device)
    parameter_before = parameter_hash(model)
    action_dim = int(model.action_dim)
    expected_action_dim = int(dataset.actions.shape[-1]) * 5
    if action_dim != expected_action_dim:
        raise RuntimeError(f"published action contract mismatch: model expects {action_dim}, dataset blocks provide {expected_action_dim}")

    root_results = []
    for index, root in enumerate(roots):
        context, goal, observed = make_root_context(model, dataset, root, args.start_frame, args.horizon, device)
        if tuple(observed.shape) != (args.horizon, action_dim):
            raise RuntimeError(f"root {root}: observed action shape {tuple(observed.shape)} violates {(args.horizon, action_dim)}")
        jacobian = jacobian_and_fd(model, context, goal, horizon=args.horizon, action_dim=action_dim, bound=args.action_bound, probe_dims=args.probe_dims, epsilon=args.fd_epsilon, fd_pairs=args.fd_pairs, seed=args.seed + index, device=device)
        with torch.no_grad():
            observed_cost = float(terminal_cost(model, context, observed.unsqueeze(1), goal)[0])
            zero_cost = float(terminal_cost(model, context, torch.zeros_like(observed).unsqueeze(1), goal)[0])
        cem = run_cem(model, context, goal, horizon=args.horizon, action_dim=action_dim, bound=args.action_bound, population=args.cem_population, iterations=args.cem_iterations, elite=args.cem_elite, seed=args.seed + 1000 + index, device=device)
        gd = run_vanilla_gd(model, context, goal, horizon=args.horizon, action_dim=action_dim, bound=args.action_bound, steps=args.gd_steps, lr=args.gd_lr)
        grasp = run_grasp_style(model, context, goal, horizon=args.horizon, action_dim=action_dim, bound=args.action_bound, restarts=args.grasp_restarts, steps=args.gd_steps, lr=args.gd_lr, seed=args.seed + 2000 + index, device=device)
        root_results.append({
            "root": root,
            "jacobian": jacobian,
            "order_sensitivity_l2": order_sensitivity(model, context, horizon=args.horizon, action_dim=action_dim, bound=args.action_bound, seed=args.seed + 3000 + index, device=device),
            "objective": {"observed_suffix_cost": observed_cost, "zero_suffix_cost": zero_cost, "cem": {"best_cost": cem["best_cost"], "trace": cem["trace"]}, "vanilla_gd": {"best_cost": gd["best_cost"], "trace": gd["trace"]}, "grasp_style_gradient_restart": {"best_cost": grasp["best_cost"], "trace": grasp["trace"], "restart_costs": grasp["all_restart_costs"]}},
        })
        print(json.dumps({"root": root, "jacobian_norm": jacobian["frobenius_norm"], "fd_median_relative_error": jacobian["median_relative_error"], "cem": cem["best_cost"], "gd": gd["best_cost"], "grasp_style": grasp["best_cost"]}, sort_keys=True))

    parameter_after = parameter_hash(model)
    fd_errors = [row["jacobian"]["median_relative_error"] for row in root_results]
    jacobian_norms = [row["jacobian"]["frobenius_norm"] for row in root_results]
    order_effects = [row["order_sensitivity_l2"] for row in root_results]
    parameter_immutable = parameter_before == parameter_after
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": parameter_immutable,
        "every_root_nonzero_jacobian": bool(min(jacobian_norms) > 1.0e-10),
        "central_fd_median_relative_error_le_0_10_each_root": bool(max(fd_errors) <= 0.10),
        "action_order_effect_nonzero_each_root": bool(min(order_effects) > 1.0e-7),
    }
    report = {
        "schema": "sra_p0_pointmaze_action_jacobian.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "frozen PointMaze LeWM action-Jacobian and optimizer plumbing; not a physical-success or SRA efficacy result",
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "inputs": {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "config": str(args.config), "config_sha256": sha256_file(args.config), "dataset": str(args.dataset_root), "roots": roots, "start_frame": args.start_frame, "horizon": args.horizon, "model_action_dim": action_dim, "expected_dataset_block_dim": expected_action_dim},
        "frozen_contract": {"all_parameters_requires_grad_false": True, "parameter_hash_before": parameter_before, "parameter_hash_after": parameter_after},
        "optimizer_contract": {"cem": "native Gaussian cross-entropy method on the frozen latent-goal objective", "vanilla_gd": "Adam on action variables only", "grasp_style_gradient_restart": "self-contained stochastic-restart action-GD comparator; not a bit-for-bit external GRASP reproduction"},
        "gates": gates,
        "aggregate": {"jacobian_frobenius_norms": jacobian_norms, "fd_median_relative_errors": fd_errors, "order_sensitivity_l2": order_effects},
        "roots": root_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "README.md").write_text(
        "# SRA P0 PointMaze CUDA audit\n\n"
        "This is a frozen-checkpoint derivative/optimizer plumbing audit. It does not establish RGB closed-loop planning performance, physical success, or an SRA advantage. The GRASP-style baseline is a transparent in-repository gradient-and-restart comparator, not an external reproduction.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output_dir), "status": report["status"], "gates": gates}, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("SRA P0 failed one or more frozen-path/Jacobian gates; do not advance to P1")


if __name__ == "__main__":
    main()
