"""RGB-only receding-horizon planning on the legacy PointMaze dynamics.

This experiment is the online consequence of the support-corrected PointMaze
candidate-bank result.  It deliberately keeps the environment state on the
simulator side: the planner receives only current/goal RGB frames and a
calibration-only visual proprio proxy derived from the RGB marker.  Simulator
state is used for physics stepping and evaluation, never in the planner input.

The released ``swm/SimplePointMaze-v0`` package is not installed in this
checkout.  We therefore use the repository's already audited legacy XML and
the dataset-matched RGB renderer.  The report names this boundary explicitly;
it is a closed-loop same-dynamics RGB consequence, not evidence for a
universal world model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.decision_benchmark.fit_pointmaze_online_effect_policy import (
    OnlineEffectPolicy,
)

GOAL_STATE = np.asarray([1.2, 1.2], dtype=np.float64)
MARKER_RADIUS = 9
FRAME_STRIDE = 5
HORIZONS = (1, 2, 4)
GATE_TOLERANCE_GRID = (0.05, 0.10, 0.20, 0.30, 0.50)
PRIVILEGED_METHODS = frozenset({
    "ground_truth_candidate_oracle",
    "perfect_dynamics_mpc",
})


def require_cuda() -> torch.device:
    """Fail closed instead of silently running this experiment on CPU."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PointMaze live planning requires CUDA; torch.cuda.is_available() is false"
        )
    device = torch.device("cuda")
    probe = torch.ones(8, device=device)
    torch.cuda.synchronize(device)
    if float(probe.sum().item()) != 8.0:
        raise RuntimeError("CUDA tensor probe returned an invalid value")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT.parent)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--effect-policy-checkpoint",
        type=Path,
        default=None,
        help="frozen train/calibration-only action-effect ensemble for repaired policies",
    )
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-start", type=int, default=1800)
    parser.add_argument("--calibration-end", type=int, default=1850)
    parser.add_argument("--test-start", type=int, default=1850)
    parser.add_argument("--test-end", type=int, default=1860)
    parser.add_argument(
        "--calibration-episodes",
        default="",
        help="optional comma-separated explicit calibration episode IDs; overrides the range",
    )
    parser.add_argument(
        "--test-episodes",
        default="",
        help="optional comma-separated explicit test episode IDs; overrides the range",
    )
    parser.add_argument(
        "--gate-calibration-episodes",
        default="",
        help="optional disjoint comma-separated episodes used only to select gate tolerance",
    )
    parser.add_argument("--max-test-episodes", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=12)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--probe-blocks", type=int, default=1)
    parser.add_argument("--calibration-frames", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--success-radius", type=float, default=0.25)
    parser.add_argument(
        "--test-visual-source",
        choices=("stored_episode", "dataset_matched_renderer"),
        default="stored_episode",
        help=(
            "Use each test episode's stored RGB or one renderer fitted only "
            "from calibration RGB; the latter supports roots without stored RGB."
        ),
    )
    parser.add_argument("--planner-map", choices=("rgb_maze", "direct"), default="rgb_maze")
    parser.add_argument("--kp", type=float, default=0.80)
    parser.add_argument("--kv", type=float, default=0.20)
    parser.add_argument("--wm-dynamics-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--effect-regret-tolerance", type=float, default=0.25,
        help="maximum calibrated fixed-bank regret bound for exploratory HTSC release",
    )
    parser.add_argument(
        "--projection-min-alpha", type=float, default=0.25,
        help="minimum retained WM-action fraction for wm_rgb_projected_guarded",
    )
    parser.add_argument("--oracle-depth", type=int, default=4)
    parser.add_argument("--oracle-beam-width", type=int, default=8)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "random", "visual_greedy", "visual_dynamics", "wm_rgb",
            "wm_rgb_guarded", "wm_rgb_projected", "wm_rgb_projected_guarded",
            "effect_adapter_rgb", "effect_adapter_rgb_htsc",
            "ground_truth_candidate_oracle",
            "perfect_dynamics_mpc",
        ),
        default=[
            "random", "visual_greedy", "visual_dynamics", "wm_rgb",
            "wm_rgb_guarded", "wm_rgb_projected", "wm_rgb_projected_guarded",
        ],
    )
    parser.add_argument("--seed", type=int, default=20261016)
    return parser.parse_args()


def parse_episode_list(value: str, label: str) -> list[int] | None:
    """Parse an explicit root list without silently accepting overlap or duplicates."""

    if not str(value).strip():
        return None
    tokens = [item.strip() for item in str(value).split(",") if item.strip()]
    if not tokens:
        raise ValueError(f"{label} must contain at least one episode ID")
    try:
        episodes = [int(item) for item in tokens]
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integer episode IDs") from exc
    if any(episode < 0 for episode in episodes):
        raise ValueError(f"{label} cannot contain negative episode IDs")
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"{label} cannot contain duplicate episode IDs")
    return episodes


def _episode_path(root: Path, episode: int) -> Path:
    path = root / "obses" / f"episode_{episode:03d}.pth"
    if not path.exists():
        raise FileNotFoundError(f"missing PointMaze episode RGB file: {path}")
    return path


def load_episode(root: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    images = torch.load(_episode_path(root, episode), map_location="cpu", weights_only=True)
    states = torch.load(root / "states.pth", map_location="cpu", weights_only=True)[episode]
    return np.asarray(images.numpy(), dtype=np.uint8), np.asarray(states.numpy(), dtype=np.float64)


def load_episode_states(root: Path, episode: int) -> np.ndarray:
    """Load environment states without requiring a test RGB artifact."""

    states = torch.load(root / "states.pth", map_location="cpu", weights_only=True)
    if episode < 0 or episode >= int(states.shape[0]):
        raise ValueError(f"episode {episode} is outside the recorded state dataset")
    return np.asarray(states[episode].numpy(), dtype=np.float64)


def green_center(image: np.ndarray) -> tuple[float, float]:
    """Extract the moving-point marker from the dataset-matched RGB frame."""

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    mask = (
        (green > 100.0)
        & (green > red.astype(np.float32) * 1.25)
        & (green > blue.astype(np.float32) * 1.15)
        & (red < 100.0)
    )
    y, x = np.where(mask)
    if len(x) == 0:
        raise ValueError("PointMaze RGB frame has no detectable green marker")
    return float(x.mean()), float(y.mean())


def _ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float = 1.0e-4) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    gram = features.T @ features
    gram += np.eye(features.shape[1], dtype=np.float64) * ridge
    return np.linalg.solve(gram, features.T @ targets)


@dataclass(frozen=True)
class VisualCalibration:
    pixel_to_position: np.ndarray
    state_to_pixel: np.ndarray
    pixel_delta_to_velocity: np.ndarray
    position_rmse: np.ndarray
    velocity_rmse: np.ndarray
    calibration_episodes: tuple[int, ...]

    def position(self, center: tuple[float, float]) -> np.ndarray:
        feature = np.asarray([center[0], center[1], 1.0], dtype=np.float64)
        return feature @ self.pixel_to_position

    def velocity(self, delta: tuple[float, float]) -> np.ndarray:
        feature = np.asarray([delta[0], delta[1], 1.0], dtype=np.float64)
        return feature @ self.pixel_delta_to_velocity

    def pixel(self, position: np.ndarray) -> np.ndarray:
        feature = np.asarray([position[0], position[1], 1.0], dtype=np.float64)
        return feature @ self.state_to_pixel


@dataclass(frozen=True)
class VisualDynamicsCalibration:
    """Calibration-only coarse block dynamics used by the RGB controller."""

    position_delta_weights: np.ndarray
    velocity_weights: np.ndarray
    position_delta_rmse: np.ndarray
    velocity_rmse: np.ndarray
    calibration_episodes: tuple[int, ...]

    def predict_position_delta(self, velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
        feature = np.asarray([*np.asarray(velocity)[:2], *np.asarray(action)[:2], 1.0], dtype=np.float64)
        return feature @ self.position_delta_weights

    def predict_velocity(self, velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
        feature = np.asarray([*np.asarray(velocity)[:2], *np.asarray(action)[:2], 1.0], dtype=np.float64)
        return feature @ self.velocity_weights


def fit_visual_dynamics_calibration(
    dataset_root: Path,
    episodes: list[int],
    ridge: float = 1.0e-3,
) -> VisualDynamicsCalibration:
    """Fit block dynamics from calibration roots; no test state is consumed."""

    states = torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True)
    actions = torch.load(dataset_root / "actions.pth", map_location="cpu", weights_only=True)
    states_np = np.asarray(states.numpy(), dtype=np.float64)
    actions_np = np.asarray(actions.numpy(), dtype=np.float64)
    features: list[np.ndarray] = []
    position_targets: list[np.ndarray] = []
    velocity_targets: list[np.ndarray] = []
    for episode in episodes:
        if episode < 0 or episode >= states_np.shape[0]:
            raise ValueError(f"calibration episode {episode} is outside the dataset")
        for frame in range(0, states_np.shape[1] - FRAME_STRIDE, FRAME_STRIDE):
            source = states_np[episode, frame]
            action = actions_np[episode, frame : frame + FRAME_STRIDE].mean(axis=0)
            target = states_np[episode, frame + FRAME_STRIDE]
            features.append(np.asarray([*source[2:4], *action[:2], 1.0], dtype=np.float64))
            position_targets.append(target[:2] - source[:2])
            velocity_targets.append(target[2:4])
    if len(features) < 32:
        raise ValueError("too few calibration transitions for block dynamics fitting")
    matrix = np.asarray(features, dtype=np.float64)
    position = np.asarray(position_targets, dtype=np.float64)
    velocity = np.asarray(velocity_targets, dtype=np.float64)
    position_weights = _ridge_fit(matrix, position, ridge=ridge)
    velocity_weights = _ridge_fit(matrix, velocity, ridge=ridge)
    return VisualDynamicsCalibration(
        position_delta_weights=position_weights,
        velocity_weights=velocity_weights,
        position_delta_rmse=np.sqrt(np.mean((matrix @ position_weights - position) ** 2, axis=0)),
        velocity_rmse=np.sqrt(np.mean((matrix @ velocity_weights - velocity) ** 2, axis=0)),
        calibration_episodes=tuple(episodes),
    )


def fit_visual_calibration(
    dataset_root: Path,
    episodes: list[int],
    max_frames: int,
) -> VisualCalibration:
    """Fit the RGB->proprio mapping using calibration roots only."""

    position_features: list[list[float]] = []
    position_targets: list[list[float]] = []
    pixel_features: list[list[float]] = []
    pixel_targets: list[list[float]] = []
    velocity_features: list[list[float]] = []
    velocity_targets: list[list[float]] = []
    used = 0
    for episode in episodes:
        images, states = load_episode(dataset_root, episode)
        centers = [green_center(image) for image in images]
        for frame, center in enumerate(centers):
            position_features.append([center[0], center[1], 1.0])
            position_targets.append(states[frame, :2].tolist())
            pixel_features.append([states[frame, 0], states[frame, 1], 1.0])
            pixel_targets.append([center[0], center[1]])
            if frame >= FRAME_STRIDE:
                previous = centers[frame - FRAME_STRIDE]
                velocity_features.append(
                    [center[0] - previous[0], center[1] - previous[1], 1.0]
                )
                velocity_targets.append(states[frame, 2:4].tolist())
            used += 1
            if used >= max_frames:
                break
        if used >= max_frames:
            break
    if len(position_features) < 32 or len(velocity_features) < 16:
        raise ValueError("too few calibration frames for visual proprio fitting")
    pos_x = np.asarray(position_features, dtype=np.float64)
    pos_y = np.asarray(position_targets, dtype=np.float64)
    state_x = np.asarray(pixel_features, dtype=np.float64)
    state_y = np.asarray(pixel_targets, dtype=np.float64)
    vel_x = np.asarray(velocity_features, dtype=np.float64)
    vel_y = np.asarray(velocity_targets, dtype=np.float64)
    pixel_to_position = _ridge_fit(pos_x, pos_y)
    state_to_pixel = _ridge_fit(state_x, state_y)
    pixel_delta_to_velocity = _ridge_fit(vel_x, vel_y)
    return VisualCalibration(
        pixel_to_position=pixel_to_position,
        state_to_pixel=state_to_pixel,
        pixel_delta_to_velocity=pixel_delta_to_velocity,
        position_rmse=np.sqrt(np.mean((pos_x @ pixel_to_position - pos_y) ** 2, axis=0)),
        velocity_rmse=np.sqrt(np.mean((vel_x @ pixel_delta_to_velocity - vel_y) ** 2, axis=0)),
        calibration_episodes=tuple(episodes),
    )


@dataclass
class DatasetMatchedRenderer:
    base: np.ndarray
    residual: np.ndarray
    source_center: tuple[float, float]
    calibration: VisualCalibration

    @classmethod
    def from_episode(
        cls, images: np.ndarray, calibration: VisualCalibration
    ) -> "DatasetMatchedRenderer":
        base = np.median(images.astype(np.int16), axis=0).astype(np.int16)
        source_center = green_center(images[0])
        x0, y0 = np.rint(source_center).astype(int)
        radius = MARKER_RADIUS
        residual = (
            images[0, y0 - radius : y0 + radius + 1, x0 - radius : x0 + radius + 1].astype(np.int16)
            - base[y0 - radius : y0 + radius + 1, x0 - radius : x0 + radius + 1]
        )
        return cls(base=base, residual=residual, source_center=source_center, calibration=calibration)

    def render(self, state: np.ndarray) -> np.ndarray:
        output = self.base.copy()
        center = self.calibration.pixel(np.asarray(state, dtype=np.float64)[:2])
        x0, y0 = np.rint(center).astype(int)
        radius = MARKER_RADIUS
        height, width = output.shape[:2]
        y0_start, y0_stop = max(0, y0 - radius), min(height, y0 + radius + 1)
        x0_start, x0_stop = max(0, x0 - radius), min(width, x0 + radius + 1)
        dy0, dy1 = y0_start - (y0 - radius), y0_stop - (y0 - radius)
        dx0, dx1 = x0_start - (x0 - radius), x0_stop - (x0 - radius)
        output[y0_start:y0_stop, x0_start:x0_stop] += self.residual[dy0:dy1, dx0:dx1]
        return np.clip(output, 0, 255).astype(np.uint8)

    def source_reconstruction_error(self, state: np.ndarray, image: np.ndarray) -> float:
        rendered = self.render(state).astype(np.float64)
        return float(np.mean(np.abs(rendered - image.astype(np.float64))))


@dataclass(frozen=True)
class RGBMazeMap:
    """Static occupancy inferred from the calibration RGB background."""

    free_cells: tuple[tuple[int, int], ...]
    cell_positions: dict[tuple[int, int], np.ndarray]

    @classmethod
    def from_renderer(cls, renderer: DatasetMatchedRenderer) -> "RGBMazeMap":
        free: list[tuple[int, int]] = []
        positions: dict[tuple[int, int], np.ndarray] = {}
        base = np.asarray(renderer.base, dtype=np.float64)
        for grid_x in range(5):
            for grid_y in range(5):
                # The legacy XML places wall centers at world coordinates
                # (grid_x+1, grid_y+1), while the particle body is translated
                # by (1.2, 1.2).  This is a scene-coordinate conversion, not
                # simulator-state input to the policy.
                position = np.asarray([grid_x - 0.2, grid_y - 0.2], dtype=np.float64)
                pixel = np.rint(renderer.calibration.pixel(position)).astype(int)
                x0, x1 = max(0, pixel[0] - 2), min(base.shape[1], pixel[0] + 3)
                y0, y1 = max(0, pixel[1] - 2), min(base.shape[0], pixel[1] + 3)
                patch = base[y0:y1, x0:x1]
                if patch.size == 0:
                    raise ValueError("RGB maze occupancy probe fell outside the image")
                mean_rgb = patch.reshape(-1, 3).mean(axis=0)
                # Wall material is warm/brown; floor and goal are blue/gray.
                is_wall = bool(mean_rgb[0] > 140.0 and mean_rgb[0] > mean_rgb[2] * 1.35)
                positions[(grid_x, grid_y)] = position
                if not is_wall:
                    free.append((grid_x, grid_y))
        if len(free) < 4:
            raise ValueError(f"RGB occupancy inference found too few free cells: {free}")
        return cls(tuple(free), positions)

    def _nearest_free(self, position: np.ndarray) -> tuple[int, int]:
        return min(
            self.free_cells,
            key=lambda cell: float(np.linalg.norm(self.cell_positions[cell] - np.asarray(position)[:2])),
        )

    def path_waypoints(self, current: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Return a shortest 4-neighbor route in RGB-inferred free space."""

        start = self._nearest_free(current)
        target = self._nearest_free(goal)
        free = set(self.free_cells)
        queue: list[tuple[int, int]] = [start]
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            node = queue.pop(0)
            if node == target:
                break
            for neighbor in (
                (node[0] - 1, node[1]),
                (node[0] + 1, node[1]),
                (node[0], node[1] - 1),
                (node[0], node[1] + 1),
            ):
                if neighbor in free and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if target not in parent:
            raise ValueError(f"RGB-inferred maze has no route from {start} to {target}")
        cells: list[tuple[int, int]] = []
        node: tuple[int, int] | None = target
        while node is not None:
            cells.append(node)
            node = parent[node]
        cells.reverse()
        points = [self.cell_positions[cell].copy() for cell in cells]
        # Remove collinear intermediate grid centers while retaining turns.
        compact: list[np.ndarray] = []
        for point in points:
            if len(compact) < 2:
                compact.append(point)
                continue
            previous = compact[-1] - compact[-2]
            current_step = point - compact[-1]
            if np.allclose(previous, current_step):
                compact[-1] = point
            else:
                compact.append(point)
        # The nearest free cell is the cell the agent is already occupying;
        # targeting its center first can send a high-velocity agent into a
        # wall before it reaches the next route segment.  Drop that start
        # cell and control directly toward the next free-cell center.
        if compact:
            compact = compact[1:]
        compact.append(np.asarray(goal, dtype=np.float64)[:2].copy())
        return compact


class LegacyPointMazeRuntime:
    """Environment-side legacy dynamics; its state never enters planner calls."""

    def __init__(self, initial_state: np.ndarray):
        import mujoco

        from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
            _legacy_pointmaze_xml,
        )

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
        self.data = mujoco.MjData(self.model)
        self.reset(initial_state)

    def reset(self, state: np.ndarray) -> None:
        self.data.qpos[:] = np.asarray(state[:2], dtype=np.float64)
        self.data.qvel[:] = np.asarray(state[2:4], dtype=np.float64)
        self._mujoco.mj_forward(self.model, self.data)

    def step_block(self, action: np.ndarray, steps: int = FRAME_STRIDE) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        for _ in range(steps):
            self.data.ctrl[:] = action
            self._mujoco.mj_step(self.model, self.data)
        return self.state

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.data.qpos[:2]), np.asarray(self.data.qvel[:2])])


def action_bank(
    current: np.ndarray,
    goal: np.ndarray,
    directions: int,
    magnitudes: list[float],
) -> np.ndarray:
    """Build a fixed-size goal-relative action bank in world coordinates."""

    if directions < 4:
        raise ValueError("directions must be at least four")
    delta = np.asarray(goal, dtype=np.float64)[:2] - np.asarray(current, dtype=np.float64)[:2]
    norm = float(np.linalg.norm(delta))
    if norm < 1.0e-8:
        delta = np.asarray([1.0, 0.0], dtype=np.float64)
        norm = 1.0
    forward = delta / norm
    perpendicular = np.asarray([-forward[1], forward[0]])
    values = []
    for magnitude in magnitudes:
        for angle in np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False):
            values.append(
                float(magnitude)
                * (np.cos(angle) * forward + np.sin(angle) * perpendicular)
            )
    bank = np.asarray(values, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[1] != 2 or not np.isfinite(bank).all():
        raise ValueError(f"invalid action bank shape/content: {bank.shape}")
    return bank


def _pool_visual(value: torch.Tensor) -> torch.Tensor:
    """Pool [N,V,H,W,D] or [T,N,V,H,W,D] visual tokens to [N,D]."""

    if value.ndim == 6:
        value = value[-1]
    if value.ndim != 5:
        raise ValueError(f"unexpected visual tensor shape {tuple(value.shape)}")
    return value.mean(dim=(1, 2, 3))


@dataclass
class LatentReadout:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.feature_mean, device=features.device, dtype=features.dtype)
        std = torch.as_tensor(self.feature_std, device=features.device, dtype=features.dtype)
        weights = torch.as_tensor(self.weights, device=features.device, dtype=features.dtype)
        normalized = (features - mean) / std
        return torch.cat([normalized, torch.ones((len(features), 1), device=features.device, dtype=features.dtype)], dim=1) @ weights


def _load_jepa_model(
    repo_root: Path,
    dataset_root: Path,
    checkpoint: Path,
    dinov2_source: Path,
    device: torch.device,
) -> tuple[Any, Any, Any, np.ndarray, np.ndarray]:
    root = repo_root / "le-wm-main"
    if not root.exists():
        root = repo_root
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "external/jepa-wms"))
    from experiments.dino_checkpoint_compat.datasets.point_maze_dset import PointMazeDataset
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    if not dinov2_source.exists():
        raise FileNotFoundError(f"DINOv2 hub source does not exist: {dinov2_source}")
    original_hub_load = torch.hub.load

    def hub_load(repo_or_dir, model, *extra, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            return original_hub_load(str(dinov2_source), model, source="local", *extra, **kwargs)
        return original_hub_load(repo_or_dir, model, *extra, **kwargs)

    torch.hub.load = hub_load
    config_path = root / "external/jepa-wms/configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
    stats = PointMazeDataset(data_path=str(dataset_root), normalize_action=True)
    preprocessor = Preprocessor(
        stats.action_mean,
        stats.action_std,
        stats.state_mean,
        stats.state_std,
        stats.proprio_mean,
        stats.proprio_std,
        transform,
    )
    model = init_module(
        str(checkpoint.parent), checkpoint.name, config["model"], device,
        action_dim=2, proprio_dim=4, preprocessor=preprocessor,
        cfgs_data=config["data"], wrapper_kwargs={"ctxt_window": 1},
    )
    model.eval()
    return model, transform, stats, np.asarray(stats.action_mean), np.asarray(stats.action_std)


def _image_tensor(images: list[np.ndarray]) -> torch.Tensor:
    values = np.stack(images, axis=0).astype(np.float32)
    return torch.from_numpy(values).permute(0, 3, 1, 2).unsqueeze(0)


def fit_latent_readout(
    model: Any,
    calibration: VisualCalibration,
    dataset_root: Path,
    episodes: list[int],
    max_frames: int,
    batch_size: int,
) -> LatentReadout:
    """Fit a calibration-only latent->position readout with visual proprio."""

    images: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for episode in episodes:
        episode_images, states = load_episode(dataset_root, episode)
        centers = [green_center(image) for image in episode_images]
        for frame in range(0, len(episode_images), FRAME_STRIDE):
            previous = centers[frame - FRAME_STRIDE] if frame >= FRAME_STRIDE else centers[frame]
            delta = (centers[frame][0] - previous[0], centers[frame][1] - previous[1])
            images.append(episode_images[frame])
            proprios.append(np.concatenate([calibration.position(centers[frame]), calibration.velocity(delta)]))
            targets.append(states[frame, :2])
            if len(images) >= max_frames:
                break
        if len(images) >= max_frames:
            break
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_array = np.stack(images[start : start + batch_size]).astype(np.float32)
            batch_images = torch.from_numpy(batch_array).permute(0, 3, 1, 2).unsqueeze(1)
            batch_proprio = torch.from_numpy(np.asarray(proprios[start : start + batch_size], dtype=np.float32)).unsqueeze(1)
            encoded = model.encode({"visual": batch_images, "proprio": batch_proprio})
            features.append(_pool_visual(encoded["visual"][:, 0]).detach().cpu().numpy())
    matrix = np.concatenate(features, axis=0).astype(np.float64)
    target = np.asarray(targets, dtype=np.float64)
    feature_mean = matrix.mean(axis=0)
    feature_std = np.maximum(matrix.std(axis=0), 1.0e-5)
    normalized = (matrix - feature_mean) / feature_std
    design = np.concatenate([normalized, np.ones((len(matrix), 1))], axis=1)
    weights = _ridge_fit(design, target, ridge=1.0e-2)
    return LatentReadout(feature_mean, feature_std, weights)


def _make_context(
    images: list[np.ndarray],
    proprios: list[np.ndarray],
) -> dict[str, torch.Tensor]:
    if len(images) != len(proprios):
        raise ValueError("image/proprio context lengths differ")
    return {
        "visual": _image_tensor(images),
        "proprio": torch.from_numpy(np.asarray(proprios, dtype=np.float32)).unsqueeze(0),
    }


def model_candidate_scores(
    model: Any,
    readout: LatentReadout,
    context_images: list[np.ndarray],
    context_proprios: list[np.ndarray],
    actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    from tensordict import TensorDict

    context = _make_context(context_images, context_proprios)
    with torch.no_grad():
        encoded = model.encode(context)
        latent_context = TensorDict(
            {"visual": encoded["visual"], "proprio": encoded["proprio"]},
            batch_size=[1, len(context_images)],
        )
        normalized = (actions - action_mean.reshape(1, 2)) / np.maximum(action_std.reshape(1, 2), 1.0e-6)
        block = np.repeat(normalized[:, None, :], FRAME_STRIDE, axis=1).reshape(len(actions), -1)
        suffix = torch.from_numpy(np.repeat(block[None, :, :], max(HORIZONS), axis=0).astype(np.float32)).to(model.device)
        rollout = model.unroll(latent_context, act_suffix=suffix)
        predicted_positions = []
        for horizon in HORIZONS:
            pooled = _pool_visual(rollout["visual"][len(context_images) - 1 + horizon])
            predicted_positions.append(readout.predict(pooled).detach().cpu().numpy())
    predicted = np.stack(predicted_positions, axis=1)
    goal = np.asarray(goal, dtype=np.float64).reshape(1, 2)
    distances = np.linalg.norm(predicted - goal, axis=-1)
    return distances[:, 0] * 0.25 + distances[:, 1] * 0.25 + distances[:, 2] * 0.50


def effect_policy_candidate_scores(
    model: Any,
    policy: OnlineEffectPolicy,
    context_images: list[np.ndarray],
    context_proprios: list[np.ndarray],
    current: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
) -> dict[str, Any]:
    """Score repeated-pulse candidates from current RGB without state access."""

    context = _make_context(context_images, context_proprios)
    with torch.no_grad():
        encoded = model.encode(context)
        pooled = np.stack([
            _pool_visual(encoded["visual"][:, index]).detach().cpu().numpy()[0]
            for index in range(len(context_images))
        ])
    projection = policy.project_history(pooled)
    report = policy.score(projection, current, target, actions)
    report["source_projection"] = projection
    return report


def visual_dynamics_scores(
    current: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
    dynamics: VisualDynamicsCalibration,
    kp: float,
    kv: float,
) -> np.ndarray:
    """Score candidate actions with RGB-estimated velocity and calibration dynamics."""

    current = np.asarray(current, dtype=np.float64)[:2]
    velocity = np.asarray(velocity, dtype=np.float64)[:2]
    target = np.asarray(target, dtype=np.float64)[:2]
    desired = np.clip(
        kp * (target - current) / max(float(np.linalg.norm(target - current)), 1.0e-8)
        - kv * velocity,
        -1.0,
        1.0,
    )
    predicted_position = np.stack(
        [current + dynamics.predict_position_delta(velocity, action) for action in actions], axis=0
    )
    predicted_velocity = np.stack(
        [dynamics.predict_velocity(velocity, action) for action in actions], axis=0
    )
    distance = np.linalg.norm(predicted_position - target.reshape(1, 2), axis=1)
    velocity_penalty = 0.05 * np.linalg.norm(predicted_velocity, axis=1)
    action_penalty = 0.02 * np.linalg.norm(actions - desired.reshape(1, 2), axis=1)
    return distance + velocity_penalty + action_penalty


def ground_truth_candidate_scores(
    runtime: LegacyPointMazeRuntime,
    actions: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Score a fixed candidate bank with one exact simulator block.

    This is a privileged candidate-bank ceiling. Simulator state is used only
    for counterfactual evaluation and is restored before returning; it is not
    a deployable RGB policy input.
    """

    if runtime is None:
        raise ValueError("ground-truth candidate oracle requires a simulator runtime")
    candidates = np.asarray(actions, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or not np.isfinite(candidates).all():
        raise ValueError(f"actions must have shape [N, 2] and be finite, got {candidates.shape}")
    target_xy = np.asarray(target, dtype=np.float64).reshape(-1)[:2]
    base_state = runtime.state.copy()
    scores: list[float] = []
    try:
        for action in candidates:
            runtime.reset(base_state)
            runtime.step_block(action)
            scores.append(float(np.linalg.norm(runtime.state[:2] - target_xy)))
    finally:
        runtime.reset(base_state)
    return np.asarray(scores, dtype=np.float64)


def perfect_dynamics_mpc_action(
    runtime: LegacyPointMazeRuntime,
    target: np.ndarray,
    directions: int,
    magnitudes: list[float],
    *,
    root_current: np.ndarray | None = None,
    depth: int = 4,
    beam_width: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the first action from an exact-dynamics finite-horizon beam MPC.

    This is a privileged planner ceiling. The root candidate bank is generated
    from the same observed current position as the fixed-bank baselines, while
    banks at simulated successor states may use exact simulator positions. Only
    the first action is executed before the next observation. The learned WM
    score is never consulted.
    """

    if runtime is None:
        raise ValueError("perfect-dynamics MPC requires a simulator runtime")
    if depth < 1 or beam_width < 1:
        raise ValueError("oracle depth and beam width must be positive")
    target_xy = np.asarray(target, dtype=np.float64).reshape(-1)[:2]
    initial_state = runtime.state.copy()
    root_position = (
        initial_state[:2]
        if root_current is None
        else np.asarray(root_current, dtype=np.float64).reshape(-1)[:2]
    )
    if root_position.shape != (2,) or not np.isfinite(root_position).all():
        raise ValueError("root_current must contain two finite coordinates")
    beam: list[tuple[float, np.ndarray, list[np.ndarray]]] = [(0.0, initial_state, [])]
    try:
        for _ in range(int(depth)):
            expansions: list[tuple[float, np.ndarray, list[np.ndarray]]] = []
            for _cost, state, sequence in beam:
                bank_position = root_position if not sequence else state[:2]
                actions = action_bank(bank_position, target_xy, directions, magnitudes)
                for action in actions:
                    runtime.reset(state)
                    runtime.step_block(action)
                    next_state = runtime.state.copy()
                    terminal_distance = float(np.linalg.norm(next_state[:2] - target_xy))
                    expansions.append((terminal_distance, next_state, sequence + [action.copy()]))
            expansions.sort(key=lambda item: item[0])
            beam = expansions[: int(beam_width)]
        if not beam or not beam[0][2]:
            raise RuntimeError("perfect-dynamics MPC produced no candidate sequence")
        return np.asarray(beam[0][2][0], dtype=np.float64), {
            "depth": int(depth),
            "beam_width": int(beam_width),
            "predicted_terminal_distance": float(beam[0][0]),
            "root_action_bank_source": (
                "rgb_calibrated_observation" if root_current is not None
                else "simulator_state_fallback"
            ),
            "successor_action_bank_source": "exact_simulator_state",
            "privileged_state_used": True,
            "privileged_role": "exact finite-horizon planner ceiling only",
        }
    finally:
        runtime.reset(initial_state)


def project_action_to_dynamics_support(
    wm_action: np.ndarray,
    dynamics_action: np.ndarray,
    current: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    dynamics: VisualDynamicsCalibration,
    kp: float,
    kv: float,
    tolerance: float,
    grid_size: int = 101,
) -> tuple[np.ndarray, float, float, float]:
    """Project a WM action toward a calibration-supported action.

    The projection searches the segment from the visual-dynamics action
    (alpha=0) to the WM action (alpha=1), returning the largest alpha whose
    calibration-dynamics score is within the declared tolerance of the
    dynamics action.  It changes the executed action without inventing a new
    direction and remains a calibration-only support intervention.
    """

    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    wm = np.clip(np.asarray(wm_action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
    dyn = np.clip(np.asarray(dynamics_action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
    if not np.isfinite(wm).all() or not np.isfinite(dyn).all():
        raise ValueError("wm_action and dynamics_action must be finite")
    alphas = np.linspace(1.0, 0.0, int(grid_size), dtype=np.float64)
    candidates = dyn[None, :] + alphas[:, None] * (wm - dyn)[None, :]
    scores = visual_dynamics_scores(
        current, velocity, target, candidates, dynamics, kp, kv,
    )
    support_score = float(scores[-1])
    admissible = np.flatnonzero(scores <= support_score + float(tolerance) + 1.0e-12)
    selected = int(admissible[0]) if len(admissible) else len(alphas) - 1
    selected_alpha = float(alphas[selected])
    selected_action = np.clip(candidates[selected], -1.0, 1.0)
    return (
        selected_action.astype(np.float64),
        selected_alpha,
        float(scores[selected]),
        support_score,
    )


def choose_action(
    method: str,
    model: Any,
    readout: LatentReadout,
    current_rgb: np.ndarray,
    goal_rgb: np.ndarray,
    context_images: list[np.ndarray],
    context_proprios: list[np.ndarray],
    calibration: VisualCalibration,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    directions: int,
    magnitudes: list[float],
    dynamics: VisualDynamicsCalibration,
    maze_map: RGBMazeMap | None,
    planner_map: str,
    kp: float,
    kv: float,
    wm_dynamics_tolerance: float,
    rng: np.random.Generator,
    projection_min_alpha: float = 0.25,
    runtime: LegacyPointMazeRuntime | None = None,
    oracle_depth: int = 4,
    oracle_beam_width: int = 8,
    effect_policy: OnlineEffectPolicy | None = None,
    effect_regret_tolerance: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    current_center = green_center(current_rgb)
    goal_center = green_center(goal_rgb)
    current = calibration.position(current_center)
    goal = calibration.position(goal_center)
    velocity = np.asarray(context_proprios[-1], dtype=np.float64)[2:4]
    if planner_map == "rgb_maze":
        if maze_map is None:
            raise ValueError("rgb_maze planner requires an RGB-inferred occupancy map")
        waypoints = maze_map.path_waypoints(current, goal)
        target = waypoints[0]
    elif planner_map == "direct":
        waypoints = [goal.copy()]
        target = goal
    else:
        raise ValueError(f"unknown planner map {planner_map}")
    actions = action_bank(current, target, directions, magnitudes)
    gate_info: dict[str, Any] = {
        "wm_selected_index": None,
        "dynamics_selected_index": None,
        "fallback_used": False,
        "fallback_reason": None,
        "dynamics_regret_of_wm": None,
    }
    if method == "random":
        index = int(rng.integers(0, len(actions)))
        scores = None
    elif method == "visual_greedy":
        direction = target - current
        norm = max(float(np.linalg.norm(direction)), 1.0e-8)
        scores = -(actions @ (direction / norm))
        index = int(np.argmin(scores))
    elif method == "visual_dynamics":
        scores = visual_dynamics_scores(current, velocity, target, actions, dynamics, kp, kv)
        index = int(np.argmin(scores))
    elif method == "ground_truth_candidate_oracle":
        scores = ground_truth_candidate_scores(runtime, actions, target)
        index = int(np.argmin(scores))
        gate_info.update({
            "privileged_state_used": True,
            "privileged_role": "counterfactual candidate-bank ceiling only",
        })
    elif method == "perfect_dynamics_mpc":
        selected_action, planner_info = perfect_dynamics_mpc_action(
            runtime, target, directions, magnitudes,
            root_current=current,
            depth=oracle_depth, beam_width=oracle_beam_width,
        )
        gate_info.update(planner_info)
        return selected_action, {
            "selected_index": None,
            "candidate_count": None,
            "current_visual_position": current.tolist(),
            "goal_visual_position": goal.tolist(),
            "visual_velocity": velocity.tolist(),
            "route_waypoints": [np.asarray(point).tolist() for point in waypoints],
            "active_waypoint": np.asarray(target).tolist(),
            "selected_action": selected_action.tolist(),
            "candidate_scores": None,
            "action_transfer_gate": gate_info,
        }
    elif method == "wm_rgb":
        scores = model_candidate_scores(
            model, readout, context_images, context_proprios, actions,
            action_mean, action_std, target,
        )
        index = int(np.argmin(scores))
    elif method in {"effect_adapter_rgb", "effect_adapter_rgb_htsc"}:
        if effect_policy is None:
            raise ValueError(f"{method} requires --effect-policy-checkpoint")
        effect_report = effect_policy_candidate_scores(
            model, effect_policy, context_images, context_proprios,
            current, target, actions,
        )
        effect_scores = np.asarray(effect_report["costs"], dtype=np.float64)
        effect_index = int(np.argmin(effect_scores))
        latent_distance = float(effect_report["latent_support_distance"])
        action_distance = float(effect_report["action_support_distances"][effect_index])
        latent_threshold = float(effect_policy.latent_support["threshold"])
        action_threshold = float(effect_policy.action_support["threshold"])
        calibration = effect_policy.regret_calibration
        bound = calibration.get("bound")
        state_support_passed = latent_distance <= latent_threshold
        action_support_passed = action_distance <= action_threshold
        bound_passed = (
            calibration.get("status") == "ok"
            and bound is not None
            and float(bound) <= float(effect_regret_tolerance)
        )
        released = bool(state_support_passed and action_support_passed and bound_passed)
        gate_info.update({
            "effect_selected_index": effect_index,
            "effect_policy_released": released,
            "state_support_passed": bool(state_support_passed),
            "state_support_distance": latent_distance,
            "state_support_threshold": latent_threshold,
            "action_support_passed": bool(action_support_passed),
            "action_support_distance": action_distance,
            "action_support_threshold": action_threshold,
            "calibrated_regret_bound": None if bound is None else float(bound),
            "effect_regret_tolerance": float(effect_regret_tolerance),
            "calibration_status": calibration.get("status"),
            "candidate_bank_mode": "rgb_state_conditioned_repeated_pulse",
            "certificate_boundary": (
                "exploratory support-and-bound gate; the calibrated bound is for the "
                "development fixed bank and does not certify this state-conditioned bank"
            ),
            "predicted_effect": np.asarray(effect_report["effects"])[effect_index].tolist(),
            "predicted_terminal": np.asarray(effect_report["terminals"])[effect_index].tolist(),
        })
        index = effect_index
        if method == "effect_adapter_rgb_htsc" and not released:
            dynamics_scores = visual_dynamics_scores(
                current, velocity, target, actions, dynamics, kp, kv,
            )
            dynamics_index = int(np.argmin(dynamics_scores))
            index = dynamics_index
            gate_info.update({
                "dynamics_selected_index": dynamics_index,
                "fallback_used": True,
                "fallback_reason": (
                    "effect_policy_support_failed" if not (
                        state_support_passed and action_support_passed
                    ) else "effect_policy_regret_bound_failed"
                ),
            })
        scores = effect_scores
    elif method == "wm_rgb_guarded":
        wm_scores = model_candidate_scores(
            model, readout, context_images, context_proprios, actions,
            action_mean, action_std, target,
        )
        dynamics_scores = visual_dynamics_scores(
            current, velocity, target, actions, dynamics, kp, kv
        )
        wm_index = int(np.argmin(wm_scores))
        dynamics_index = int(np.argmin(dynamics_scores))
        dynamics_regret = float(dynamics_scores[wm_index] - dynamics_scores[dynamics_index])
        gate_info.update({
            "wm_selected_index": wm_index,
            "dynamics_selected_index": dynamics_index,
            "dynamics_regret_of_wm": dynamics_regret,
        })
        if dynamics_regret <= wm_dynamics_tolerance:
            index = wm_index
        else:
            index = dynamics_index
            gate_info["fallback_used"] = True
            gate_info["fallback_reason"] = "wm_action_outside_calibration_dynamics_tolerance"
        scores = wm_scores
    elif method in {"wm_rgb_projected", "wm_rgb_projected_guarded"}:
        wm_scores = model_candidate_scores(
            model, readout, context_images, context_proprios, actions,
            action_mean, action_std, target,
        )
        dynamics_scores = visual_dynamics_scores(
            current, velocity, target, actions, dynamics, kp, kv
        )
        wm_index = int(np.argmin(wm_scores))
        dynamics_index = int(np.argmin(dynamics_scores))
        projected_action, alpha, projected_score, support_score = project_action_to_dynamics_support(
            actions[wm_index], actions[dynamics_index], current, velocity, target,
            dynamics, kp, kv, wm_dynamics_tolerance,
        )
        gate_info.update({
            "wm_selected_index": wm_index,
            "dynamics_selected_index": dynamics_index,
            "dynamics_regret_of_wm": float(
                dynamics_scores[wm_index] - dynamics_scores[dynamics_index]
            ),
            "projection_used": bool(alpha < 1.0 - 1.0e-12),
            "projection_alpha": alpha,
            "projected_dynamics_score": projected_score,
            "dynamics_support_score": support_score,
        })
        executed_action = projected_action
        if method == "wm_rgb_projected_guarded" and alpha < projection_min_alpha:
            executed_action = actions[dynamics_index]
            gate_info.update({
                "fallback_used": True,
                "fallback_reason": "projection_alpha_below_minimum",
                "projection_min_alpha": float(projection_min_alpha),
            })
        return executed_action, {
            "selected_index": wm_index,
            "candidate_count": int(len(actions)),
            "current_visual_position": current.tolist(),
            "goal_visual_position": goal.tolist(),
            "visual_velocity": velocity.tolist(),
            "route_waypoints": [np.asarray(point).tolist() for point in waypoints],
            "active_waypoint": np.asarray(target).tolist(),
            "selected_action": executed_action.tolist(),
            "candidate_scores": np.asarray(wm_scores).tolist(),
            "action_transfer_gate": gate_info,
        }
    else:
        raise ValueError(f"unknown planning method {method}")
    return actions[index], {
        "selected_index": index,
        "candidate_count": int(len(actions)),
        "current_visual_position": current.tolist(),
        "goal_visual_position": goal.tolist(),
        "visual_velocity": velocity.tolist(),
        "route_waypoints": [np.asarray(point).tolist() for point in waypoints],
        "active_waypoint": np.asarray(target).tolist(),
        "selected_action": actions[index].tolist(),
        "candidate_scores": None if scores is None else np.asarray(scores).tolist(),
        "action_transfer_gate": gate_info,
    }


def run_episode(
    method: str,
    episode: int,
    dataset_root: Path,
    calibration: VisualCalibration,
    model: Any,
    readout: LatentReadout,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    dynamics: VisualDynamicsCalibration,
    maze_map: RGBMazeMap | None,
    planner_map: str,
    directions: int,
    magnitudes: list[float],
    probe_blocks: int,
    kp: float,
    kv: float,
    wm_dynamics_tolerance: float,
    max_blocks: int,
    success_radius: float,
    seed: int,
    projection_min_alpha: float = 0.25,
    oracle_depth: int = 4,
    oracle_beam_width: int = 8,
    fixed_renderer: DatasetMatchedRenderer | None = None,
    effect_policy: OnlineEffectPolicy | None = None,
    effect_regret_tolerance: float = 0.25,
) -> dict[str, Any]:
    if fixed_renderer is None:
        images, states = load_episode(dataset_root, episode)
        renderer = DatasetMatchedRenderer.from_episode(images, calibration)
        initial_reference = images[0]
        renderer_source = "stored_test_episode_rgb"
    else:
        states = load_episode_states(dataset_root, episode)
        renderer = fixed_renderer
        initial_reference = None
        renderer_source = "calibration_only_dataset_matched_renderer"
    runtime = LegacyPointMazeRuntime(states[0])
    current_rgb = renderer.render(runtime.state)
    goal_rgb = renderer.render(np.r_[GOAL_STATE, [0.0, 0.0]])
    initial_render_error = (
        None
        if initial_reference is None
        else renderer.source_reconstruction_error(runtime.state, initial_reference)
    )
    center = green_center(current_rgb)
    proprio = np.concatenate([calibration.position(center), calibration.velocity((0.0, 0.0))])
    reset_state = runtime.state.copy()
    if probe_blocks < 0:
        raise ValueError("probe_blocks must be non-negative")
    if probe_blocks:
        for _ in range(probe_blocks):
            runtime.step_block(np.zeros(2, dtype=np.float64))
        probe_rgb = renderer.render(runtime.state)
        probe_center = green_center(probe_rgb)
        probe_delta = (probe_center[0] - center[0], probe_center[1] - center[1])
        probe_proprio = np.concatenate([
            calibration.position(probe_center), calibration.velocity(probe_delta)
        ])
        current_rgb = probe_rgb
        center = probe_center
        context_images = [renderer.render(reset_state), probe_rgb]
        context_proprios = [proprio.copy(), probe_proprio]
    else:
        context_images = [current_rgb, current_rgb.copy()]
        context_proprios = [proprio.copy(), proprio.copy()]
    planning_start_state = runtime.state.copy()
    rng = np.random.default_rng(seed)
    trace = []
    detector_failures = 0
    for block_index in range(max_blocks):
        try:
            action, decision = choose_action(
                method, model, readout, current_rgb, goal_rgb, context_images,
                context_proprios, calibration, action_mean, action_std,
                directions, magnitudes, dynamics, maze_map, planner_map, kp, kv,
                wm_dynamics_tolerance, rng, projection_min_alpha, runtime=runtime,
                oracle_depth=oracle_depth, oracle_beam_width=oracle_beam_width,
                effect_policy=effect_policy,
                effect_regret_tolerance=effect_regret_tolerance,
            )
        except ValueError as exc:
            detector_failures += 1
            trace.append({"block": block_index, "error": str(exc)})
            break
        runtime.step_block(action)
        next_rgb = renderer.render(runtime.state)
        try:
            next_center = green_center(next_rgb)
            delta = (next_center[0] - center[0], next_center[1] - center[1])
            next_proprio = np.concatenate([calibration.position(next_center), calibration.velocity(delta)])
            center = next_center
        except ValueError:
            detector_failures += 1
            next_proprio = context_proprios[-1].copy()
        context_images = [context_images[-1], next_rgb]
        context_proprios = [context_proprios[-1], next_proprio]
        current_rgb = next_rgb
        trace.append({
            "block": block_index,
            "runtime_state_for_evaluation": runtime.state.tolist(),
            **decision,
        })
        if float(np.linalg.norm(runtime.state[:2] - GOAL_STATE)) <= success_radius:
            break
    final_distance = float(np.linalg.norm(runtime.state[:2] - GOAL_STATE))
    return {
        "episode_id": int(episode),
        "method": method,
        "initial_state_for_environment": reset_state.tolist(),
        "planning_start_state_for_evaluation": planning_start_state.tolist(),
        "final_state_for_evaluation": runtime.state.tolist(),
        "initial_distance": float(np.linalg.norm(reset_state[:2] - GOAL_STATE)),
        "planning_start_distance": float(np.linalg.norm(planning_start_state[:2] - GOAL_STATE)),
        "final_distance": final_distance,
        "probe_progress": float(
            np.linalg.norm(reset_state[:2] - GOAL_STATE)
            - np.linalg.norm(planning_start_state[:2] - GOAL_STATE)
        ),
        "progress": float(np.linalg.norm(reset_state[:2] - GOAL_STATE) - final_distance),
        "success": bool(final_distance <= success_radius),
        "blocks_executed": len(trace),
        "detector_failures": int(detector_failures),
        "renderer_initial_mean_abs_error": initial_render_error,
        "trace": trace,
        "policy_contract": {
            "rgb_current_input": True,
            "rgb_goal_available": True,
            "goal_used_by_action_chart": True,
            "simulator_state_in_policy_input": False,
            "simulator_state_used_for": "environment stepping and held-out evaluation only",
            "rgb_velocity_probe_blocks": int(probe_blocks),
            "rgb_static_map": planner_map == "rgb_maze",
            "renderer_source": renderer_source,
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    gate_decisions = [
        trace.get("action_transfer_gate", {})
        for row in rows
        for trace in row.get("trace", [])
        if "action_transfer_gate" in trace
    ]
    fallback_count = sum(bool(item.get("fallback_used")) for item in gate_decisions)
    effect_decisions = [
        item for item in gate_decisions if item.get("effect_selected_index") is not None
    ]
    effect_released = sum(bool(item.get("effect_policy_released")) for item in effect_decisions)
    render_errors = [
        float(row["renderer_initial_mean_abs_error"])
        for row in rows
        if row.get("renderer_initial_mean_abs_error") is not None
    ]
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
        "mean_progress": float(np.mean([row["progress"] for row in rows])),
        "mean_probe_progress": float(np.mean([row["probe_progress"] for row in rows])),
        "mean_planning_start_distance": float(np.mean([row["planning_start_distance"] for row in rows])),
        "mean_blocks": float(np.mean([row["blocks_executed"] for row in rows])),
        "detector_failure_rate": float(np.mean([row["detector_failures"] > 0 for row in rows])),
        "renderer_initial_mean_abs_error": (
            float(np.mean(render_errors)) if render_errors else None
        ),
        "gate_decisions": len(gate_decisions),
        "gate_fallback_rate": float(fallback_count / len(gate_decisions)) if gate_decisions else 0.0,
        "effect_gate_decisions": len(effect_decisions),
        "effect_release_rate": (
            float(effect_released / len(effect_decisions)) if effect_decisions else None
        ),
    }


def select_gate_tolerance(candidates: list[dict[str, Any]]) -> float:
    """Select a gate tolerance using calibration outcomes only.

    Success is primary, continuous final distance is the first tie-break, and
    fallback rate is the second tie-break.  The smallest tolerance is the final
    deterministic tie-break, so merely using the WM more often cannot win when
    calibration task performance is unchanged.
    """

    if not candidates:
        raise ValueError("gate tolerance candidates cannot be empty")
    ordered = sorted(candidates, key=lambda item: (
        -float(item["aggregate"]["success_rate"]),
        float(item["aggregate"]["mean_final_distance"]),
        float(item["aggregate"]["gate_fallback_rate"]),
        float(item["tolerance"]),
    ))
    return float(ordered[0]["tolerance"])


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.projection_min_alpha) <= 1.0:
        raise ValueError("projection-min-alpha must be in [0, 1]")
    if float(args.effect_regret_tolerance) < 0.0:
        raise ValueError("effect-regret-tolerance must be non-negative")
    device = require_cuda()
    dataset_root = args.dataset_root.resolve()
    calibration_episodes = parse_episode_list(args.calibration_episodes, "calibration episodes")
    test_episodes = parse_episode_list(args.test_episodes, "test episodes")
    gate_calibration_episodes = parse_episode_list(
        args.gate_calibration_episodes, "gate calibration episodes",
    )
    if calibration_episodes is None:
        calibration_episodes = list(range(args.calibration_start, args.calibration_end))
    if test_episodes is None:
        test_episodes = list(range(args.test_start, args.test_end))
    if args.max_test_episodes is not None:
        test_episodes = test_episodes[: args.max_test_episodes]
    if not calibration_episodes or not test_episodes:
        raise ValueError("calibration and test episode ranges must be non-empty")
    overlap = set(calibration_episodes) & set(test_episodes)
    if overlap:
        raise ValueError(f"calibration/test episode overlap: {sorted(overlap)}")
    if gate_calibration_episodes is not None:
        gate_overlap = (
            set(gate_calibration_episodes) & set(calibration_episodes)
        ) | (set(gate_calibration_episodes) & set(test_episodes))
        if gate_overlap:
            raise ValueError(f"gate calibration overlaps another split: {sorted(gate_overlap)}")
    for episode in calibration_episodes:
        _episode_path(dataset_root, episode)
    state_episode_count = int(torch.load(
        dataset_root / "states.pth", map_location="cpu", weights_only=True
    ).shape[0])
    for episode in test_episodes + (gate_calibration_episodes or []):
        if episode < 0 or episode >= state_episode_count:
            raise ValueError(f"episode {episode} is outside the recorded state dataset")
        if args.test_visual_source == "stored_episode":
            _episode_path(dataset_root, episode)
    print("[PointMaze live] fitting RGB->proprio calibration on calibration roots only")
    calibration = fit_visual_calibration(dataset_root, calibration_episodes, args.calibration_frames)
    print("[PointMaze live] fitting calibration-only block dynamics and RGB maze occupancy")
    dynamics = fit_visual_dynamics_calibration(dataset_root, calibration_episodes)
    calibration_images, _calibration_states = load_episode(dataset_root, calibration_episodes[0])
    calibration_renderer = DatasetMatchedRenderer.from_episode(calibration_images, calibration)
    test_renderer = (
        calibration_renderer
        if args.test_visual_source == "dataset_matched_renderer"
        else None
    )
    maze_map = RGBMazeMap.from_renderer(calibration_renderer) if args.planner_map == "rgb_maze" else None
    if maze_map is not None:
        print(f"[PointMaze live] RGB occupancy inferred {len(maze_map.free_cells)} free cells")
    print("[PointMaze live] loading frozen JEPA-WM on CUDA; policy input is RGB-derived only")
    model, _transform, _stats, action_mean, action_std = _load_jepa_model(
        args.repo_root.resolve(), dataset_root, args.checkpoint.resolve(), args.dinov2_source.resolve(), device
    )
    print("[PointMaze live] fitting latent->position readout on calibration roots only")
    readout = fit_latent_readout(
        model, calibration, dataset_root, calibration_episodes,
        args.calibration_frames, args.batch_size,
    )
    effect_methods = {"effect_adapter_rgb", "effect_adapter_rgb_htsc"}
    requested_effect_methods = effect_methods & set(args.methods)
    if requested_effect_methods and args.effect_policy_checkpoint is None:
        raise ValueError(
            f"methods {sorted(requested_effect_methods)} require --effect-policy-checkpoint"
        )
    effect_policy = None
    if args.effect_policy_checkpoint is not None:
        print("[PointMaze live] loading frozen action-effect ensemble on CUDA")
        effect_policy = OnlineEffectPolicy.load(
            args.effect_policy_checkpoint.resolve(), device,
        )
    selected_tolerance = float(args.wm_dynamics_tolerance)
    gate_calibration_report: dict[str, Any] | None = None
    if gate_calibration_episodes is not None:
        print(
            "[PointMaze live] selecting WM dynamics gate tolerance on "
            f"{len(gate_calibration_episodes)} calibration-only roots"
        )
        tolerance_candidates: list[dict[str, Any]] = []
        for tolerance in GATE_TOLERANCE_GRID:
            calibration_rows = [run_episode(
                "wm_rgb_guarded", episode, dataset_root, calibration, model, readout,
                action_mean, action_std, dynamics, maze_map, args.planner_map,
                args.directions, args.magnitudes, args.probe_blocks, args.kp, args.kv,
                float(tolerance), args.max_blocks, args.success_radius,
                args.seed + 100000 + index, args.projection_min_alpha,
                args.oracle_depth, args.oracle_beam_width,
                fixed_renderer=test_renderer,
                effect_policy=effect_policy,
                effect_regret_tolerance=args.effect_regret_tolerance,
            ) for index, episode in enumerate(gate_calibration_episodes)]
            tolerance_candidates.append({
                "tolerance": float(tolerance),
                "aggregate": _aggregate(calibration_rows),
            })
        selected_tolerance = select_gate_tolerance(tolerance_candidates)
        gate_calibration_report = {
            "episodes": list(gate_calibration_episodes),
            "tolerance_grid": list(GATE_TOLERANCE_GRID),
            "selection_rule": "maximize success, then minimize final distance, then fallback rate, then tolerance",
            "selected_tolerance": selected_tolerance,
            "candidates": tolerance_candidates,
        }
        print(f"[PointMaze live] selected gate tolerance={selected_tolerance:.2f}")
    methods = tuple(args.methods)
    rows: list[dict[str, Any]] = []
    for method in methods:
        print(f"[PointMaze live] running method={method} on {len(test_episodes)} held-out roots")
        for index, episode in enumerate(test_episodes):
            rows.append(run_episode(
                method, episode, dataset_root, calibration, model, readout,
                action_mean, action_std, dynamics, maze_map, args.planner_map,
                args.directions, args.magnitudes, args.probe_blocks, args.kp, args.kv,
                selected_tolerance, args.max_blocks, args.success_radius,
                args.seed + index, args.projection_min_alpha,
                args.oracle_depth, args.oracle_beam_width,
                fixed_renderer=test_renderer,
                effect_policy=effect_policy,
                effect_regret_tolerance=args.effect_regret_tolerance,
            ))
    aggregates = {method: _aggregate([row for row in rows if row["method"] == method]) for method in methods}
    privileged_methods = [method for method in methods if method in PRIVILEGED_METHODS]
    report = {
        "schema": "pointmaze_rgb_online_planning.v3",
        "protocol": {
            "dataset_root": str(dataset_root),
            "checkpoint": str(args.checkpoint.resolve()),
            "dinov2_source": str(args.dinov2_source.resolve()),
            "device": str(device),
            "calibration_episodes": calibration_episodes,
            "test_episodes": test_episodes,
            "episode_selection": (
                "explicit_episode_lists"
                if args.calibration_episodes or args.test_episodes
                else "contiguous_ranges"
            ),
            "episode_disjoint": not bool(set(calibration_episodes) & set(test_episodes)),
            "action_bank": {
                "directions": args.directions,
                "magnitudes": args.magnitudes,
                "blocks_per_prediction": max(HORIZONS),
                "executed_blocks_before_reobservation": 1,
                "rgb_velocity_probe_blocks": args.probe_blocks,
                "planner_map": args.planner_map,
                "kp": args.kp,
                "kv": args.kv,
                "wm_dynamics_tolerance": args.wm_dynamics_tolerance,
                "selected_wm_dynamics_tolerance": selected_tolerance,
                "projection_min_alpha": args.projection_min_alpha,
                "oracle_depth": args.oracle_depth,
                "oracle_beam_width": args.oracle_beam_width,
            },
            "methods": list(methods),
            "privileged_methods": privileged_methods,
            "runtime_state_access": bool(privileged_methods),
            "goal_state_for_environment_only": GOAL_STATE.tolist(),
            "success_radius": args.success_radius,
            "test_visual_source": args.test_visual_source,
            "test_rgb_files_required": args.test_visual_source == "stored_episode",
            "effect_policy_checkpoint": (
                None if args.effect_policy_checkpoint is None
                else str(args.effect_policy_checkpoint.resolve())
            ),
            "effect_regret_tolerance": float(args.effect_regret_tolerance),
            "effect_policy_gate_boundary": (
                None if effect_policy is None else
                "exploratory online support-and-bound gate; fixed-bank calibration does "
                "not provide coverage for the state-conditioned repeated-pulse bank"
            ),
            "gate_calibration": gate_calibration_report,
        },
        "calibration": {
            "episodes": list(calibration.calibration_episodes),
            "position_rmse_pixels_to_state": calibration.position_rmse.tolist(),
            "velocity_rmse": calibration.velocity_rmse.tolist(),
            "pixel_to_position": calibration.pixel_to_position.tolist(),
            "state_to_pixel": calibration.state_to_pixel.tolist(),
            "pixel_delta_to_velocity": calibration.pixel_delta_to_velocity.tolist(),
            "readout": "ridge latent pooled visual tokens -> qpos, calibration roots only",
            "block_dynamics": {
                "position_delta_weights": dynamics.position_delta_weights.tolist(),
                "velocity_weights": dynamics.velocity_weights.tolist(),
                "position_delta_rmse": dynamics.position_delta_rmse.tolist(),
                "velocity_rmse": dynamics.velocity_rmse.tolist(),
                "episodes": list(dynamics.calibration_episodes),
            },
        },
        "aggregates": aggregates,
        "rows": rows,
        "evidence_boundary": (
            "same-legacy-dynamics mixed RGB consequence and privileged fixed-bank "
            "ceiling; privileged methods are not deployable and do not prove "
            "universal transfer, physical causality, or the official swm package "
            "interface"
            if privileged_methods else
            "same-legacy-dynamics RGB-only receding-horizon consequence with a "
            "calibration-only visual readout; not proof of universal transfer, "
            "physical causality, or the official swm package interface"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": aggregates, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
