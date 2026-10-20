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
GDD_SUPPORT_PRESERVING_MIN_RELEASE_RATE = 0.20
# The GDD verified-local arm uses a smaller, fixed information-action pulse
# than the legacy effect-policy probe.  This is a safety budget declared in
# the protocol, not a test-outcome-tuned parameter.
GDD_VERIFIED_LOCAL_PROBE_MAGNITUDE = 0.25
# Fixed protocol threshold for the active-information arm.  It is not
# selected from held-out task outcomes.
GDD_DISAGREEMENT_COSINE_THRESHOLD = 0.80
# Fixed post-action RGB verification tolerance in calibrated position units.
# A failed candidate is quarantined and never used to update the local chart.
GDD_LOCAL_EXECUTION_RESIDUAL_TOLERANCE = 0.25
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


def validate_effect_policy_execution_contract(
    policy: OnlineEffectPolicy,
    *,
    executed_action_steps: int = FRAME_STRIDE,
) -> None:
    """Reject a teacher whose labeled effect horizon differs from deployment."""

    if int(policy.action_steps) != int(executed_action_steps):
        raise ValueError(
            "effect-policy action horizon does not match the live re-observation "
            f"contract: policy={policy.action_steps}, executed={executed_action_steps}"
        )
    if policy.candidate_family != "runtime_one_block_pulse_v1":
        raise ValueError(
            "live effect policy requires runtime_one_block_pulse_v1; refusing "
            f"legacy family {policy.candidate_family!r}"
        )


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
    parser.add_argument("--gdd-adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--gdd-projection", type=Path, default=None,
                        help="D2 projection report for gdd_wm_rgb")
    parser.add_argument("--gdd-feature-table", type=Path, default=None,
                        help="student feature table containing the frozen goal anchor")
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
    parser.add_argument(
        "--gdd-calibration-episodes",
        default="",
        help=(
            "optional disjoint comma-separated roots used to authorize GDD teacher "
            "deployment; rejection fails closed to visual dynamics"
        ),
    )
    parser.add_argument("--max-test-episodes", type=int, default=None)
    parser.add_argument(
        "--max-blocks", type=int, default=40,
        help="closed-loop execution budget; 40 is the calibrated PointMaze budget",
    )
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--probe-blocks", type=int, default=1)
    parser.add_argument(
        "--active-probe-directions", type=int, default=4,
        help="number of evenly spaced goal-relative active effect probes; default 4 preserves the frozen protocol",
    )
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
        "--effect-visual-tolerance", type=float, default=0.10,
        help="maximum visual-dynamics regret allowed for the consistency gate",
    )
    parser.add_argument(
        "--gdd-disagreement-cosine-threshold", type=float,
        default=GDD_DISAGREEMENT_COSINE_THRESHOLD,
        help=(
            "fixed cosine threshold for triggering GDD RGB probes; probes run "
            "only when frozen GDD and visual-dynamics proposals disagree"
        ),
    )
    parser.add_argument(
        "--projection-min-alpha", type=float, default=0.25,
        help="minimum retained WM-action fraction for wm_rgb_projected_guarded",
    )
    parser.add_argument(
        "--gdd-score-horizons", type=int, nargs="+", default=[4],
        help=(
            "prediction horizons used by the frozen GDD projection student; "
            "must be a subset of adapter horizons and match distillation when possible"
        ),
    )
    parser.add_argument(
        "--gdd-candidate-count", type=int, default=16,
        help="number of deterministic paired_order_v2 candidates scored by GDD",
    )
    parser.add_argument("--oracle-depth", type=int, default=4)
    parser.add_argument("--oracle-beam-width", type=int, default=8)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "random", "visual_greedy", "visual_dynamics", "wm_rgb",
            "wm_rgb_guarded", "wm_rgb_projected", "wm_rgb_projected_guarded",
            "gdd_wm_rgb", "gdd_wm_rgb_guarded", "gdd_wm_rgb_verified_local",
            "gdd_wm_rgb_disagreement_verified_local", "gdd_wm_rgb_projected_guarded",
            "gdd_wm_rgb_waypoint",
            "gdd_wm_rgb_waypoint_guarded",
            "gdd_wm_rgb_support_preserving_guarded",
            "gdd_wm_rgb_support_matched_guarded",
            "gdd_runtime_task_effect_rgb",
            "effect_adapter_rgb", "effect_adapter_rgb_htsc",
            "effect_adapter_rgb_htsc_consistency",
            "effect_adapter_rgb_online", "effect_adapter_rgb_online_htsc",
            "effect_adapter_rgb_online_probe", "effect_adapter_rgb_online_verified_local",
            "effect_adapter_rgb_online_local_jacobian", "effect_adapter_rgb_online_local_direct",
            "effect_adapter_rgb_online_local_phase",
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


def validate_output_path(path: Path) -> Path:
    """Reject a directory early; this runner writes one JSON report file."""

    path = Path(path)
    if path.exists() and path.is_dir():
        raise ValueError(
            f"--output must be a JSON file path, not an existing directory: {path}"
        )
    if not path.name:
        raise ValueError("--output must name a JSON report file")
    return path


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


@dataclass
class OnlineLocalEffectResidual:
    """RGB-observed local correction for a frozen action-effect predictor.

    The correction is fitted only from executed frozen-effect action pairs and
    their observed-minus-predicted displacement residuals in the current
    episode.  It never reads simulator state.  The
    constant-only fallback is intentionally zero until an observation exists,
    so the frozen effect policy remains the explicit baseline at episode start.
    """

    ridge: float = 1.0e-2
    min_samples: int = 4
    shrinkage_samples: float = 4.0
    max_correction_norm: float = 0.25
    actions: list[np.ndarray] | None = None
    velocities: list[np.ndarray] | None = None
    effects: list[np.ndarray] | None = None
    verification_errors: list[float] | None = None
    verification_threshold: float = 0.25
    min_verification_samples: int = 2
    verification_direction_threshold: float | None = None
    directional_scores: list[float] | None = None

    def __post_init__(self) -> None:
        if self.ridge <= 0.0:
            raise ValueError("online residual ridge must be positive")
        if (
            self.min_samples < 1 or self.shrinkage_samples <= 0.0
            or self.max_correction_norm <= 0.0
            or self.verification_threshold <= 0.0
            or self.min_verification_samples < 1
        ):
            raise ValueError("online residual stability parameters must be positive")
        self.actions = [] if self.actions is None else self.actions
        self.velocities = [] if self.velocities is None else self.velocities
        self.effects = [] if self.effects is None else self.effects
        self.verification_errors = [] if self.verification_errors is None else self.verification_errors
        self.directional_scores = [] if self.directional_scores is None else self.directional_scores

    @staticmethod
    def _features(actions: np.ndarray, velocities: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("online residual actions must have shape [N, 2]")
        if velocities is None:
            velocity_values = np.zeros_like(values)
        else:
            velocity_values = np.asarray(velocities, dtype=np.float64)
            if velocity_values.ndim == 1:
                velocity_values = velocity_values.reshape(1, -1)
            if velocity_values.shape != values.shape:
                raise ValueError("online residual velocities must match action shape")
        vx, vy = velocity_values[:, 0], velocity_values[:, 1]
        ax, ay = values[:, 0], values[:, 1]
        return np.stack([
            np.ones(len(values)), vx, vy, ax, ay,
            vx * ax, vx * ay, vy * ax, vy * ay,
            ax * ax, ax * ay, ay * ay,
        ], axis=1)

    def update(
        self,
        action: np.ndarray,
        observed_effect: np.ndarray,
        velocity: np.ndarray | None = None,
        predicted_effect: np.ndarray | None = None,
    ) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        effect = np.asarray(observed_effect, dtype=np.float64).reshape(-1)
        velocity = np.zeros(2, dtype=np.float64) if velocity is None else np.asarray(velocity, dtype=np.float64).reshape(-1)
        if (
            action.shape != (2,) or velocity.shape != (2,) or effect.shape != (2,)
            or not np.isfinite(action).all() or not np.isfinite(velocity).all()
            or not np.isfinite(effect).all()
        ):
            raise ValueError("online residual update requires finite 2D action/effect")
        if predicted_effect is not None:
            predicted = np.asarray(predicted_effect, dtype=np.float64).reshape(-1)
            if predicted.shape != (2,) or not np.isfinite(predicted).all():
                raise ValueError("predicted effect must be a finite 2D vector")
            effect = effect - predicted
        self.actions.append(action.copy())
        self.velocities.append(velocity.copy())
        self.effects.append(effect.copy())

    def predict(self, actions: np.ndarray, velocity: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("online residual query actions must have shape [N, 2]")
        if not self.trusted:
            return np.zeros_like(values)
        matrix = self._features(np.stack(self.actions), np.stack(self.velocities))
        targets = np.stack(self.effects)
        gram = matrix.T @ matrix + np.eye(matrix.shape[1], dtype=np.float64) * self.ridge
        weights = np.linalg.solve(gram, matrix.T @ targets)
        query_velocity = None if velocity is None else np.repeat(
            np.asarray(velocity, dtype=np.float64).reshape(1, 2), len(values), axis=0
        )
        correction = self._features(values, query_velocity) @ weights
        # Do not let a high-dimensional local fit dominate before it has
        # accumulated enough independent observations.  The final norm cap is
        # a trust region in task-effect space, not a test-set threshold.
        correction *= len(self.actions) / (len(self.actions) + self.shrinkage_samples)
        norms = np.linalg.norm(correction, axis=1, keepdims=True)
        correction *= np.minimum(1.0, self.max_correction_norm / np.maximum(norms, 1.0e-12))
        return correction

    def record_verification(self, predicted_effect: np.ndarray, observed_effect: np.ndarray) -> None:
        predicted = np.asarray(predicted_effect, dtype=np.float64).reshape(-1)
        observed = np.asarray(observed_effect, dtype=np.float64).reshape(-1)
        if predicted.shape != (2,) or observed.shape != (2,) or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
            raise ValueError("successor verification requires finite 2D effects")
        self.verification_errors.append(float(np.linalg.norm(predicted - observed)))
        predicted_norm = float(np.linalg.norm(predicted))
        observed_norm = float(np.linalg.norm(observed))
        if predicted_norm > 1.0e-8 and observed_norm > 1.0e-8:
            self.directional_scores.append(float(np.dot(predicted, observed) / (predicted_norm * observed_norm)))

    def update_verified_execution(
        self,
        action: np.ndarray,
        observed_effect: np.ndarray,
        velocity: np.ndarray | None = None,
        predicted_effect: np.ndarray | None = None,
        verification_baseline_effect: np.ndarray | None = None,
    ) -> bool:
        """Record one frozen-effect execution, or abstain on a fallback.

        A ``None`` prediction means that the executed action came from a
        fallback controller.  Such a transition is useful to the fallback's
        own model but is not a residual observation for this frozen effect
        policy, so it is deliberately ignored here.
        """

        if predicted_effect is None:
            return False
        verification_predicted = np.asarray(predicted_effect, dtype=np.float64).reshape(-1)
        verification_observed = np.asarray(observed_effect, dtype=np.float64).reshape(-1)
        if verification_baseline_effect is not None:
            baseline = np.asarray(verification_baseline_effect, dtype=np.float64).reshape(-1)
            if baseline.shape != (2,) or not np.isfinite(baseline).all():
                raise ValueError("verification baseline effect must be a finite 2D vector")
            verification_predicted = verification_predicted - baseline
            verification_observed = verification_observed - baseline
        self.record_verification(verification_predicted, verification_observed)
        self.update(
            action,
            observed_effect,
            velocity=velocity,
            predicted_effect=predicted_effect,
        )
        return True

    @property
    def trusted(self) -> bool:
        if len(self.actions) < self.min_samples or len(self.verification_errors) < self.min_verification_samples:
            return False
        recent = np.asarray(self.verification_errors[-self.min_verification_samples :], dtype=np.float64)
        norm_ok = bool(float(np.mean(recent)) <= self.verification_threshold)
        if self.verification_direction_threshold is None or len(self.directional_scores) < self.min_verification_samples:
            return norm_ok
        directional = np.asarray(self.directional_scores[-self.min_verification_samples :], dtype=np.float64)
        return bool(float(np.mean(directional)) >= self.verification_direction_threshold)

    @property
    def sample_count(self) -> int:
        return len(self.actions)


@dataclass
class VerifiedLocalActionEffectModel:
    """Low-variance RGB-observed local action-effect model.

    Unlike ``OnlineLocalEffectResidual``, this model does not fit a frozen
    teacher residual with a 12-term interaction basis.  It fits the local
    affine map ``effect = bias + action @ Jacobian`` from observed RGB
    successors only.  The deliberately small basis is identifiable from the
    four predeclared cardinal probes and is refit after every executed pulse.
    It is a recovery arm, not a certificate for the global teacher.
    """

    ridge: float = 5.0e-2
    min_samples: int = 4
    actions: list[np.ndarray] | None = None
    effects: list[np.ndarray] | None = None
    verification_errors: list[float] | None = None
    quarantined: bool = False

    def __post_init__(self) -> None:
        if self.ridge <= 0.0 or self.min_samples < 3:
            raise ValueError("local action-effect model parameters must be positive")
        self.actions = [] if self.actions is None else self.actions
        self.effects = [] if self.effects is None else self.effects
        self.verification_errors = [] if self.verification_errors is None else self.verification_errors

    @staticmethod
    def _features(actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("local action-effect actions must have shape [N, 2]")
        return np.concatenate([np.ones((len(values), 1)), values], axis=1)

    def update(self, action: np.ndarray, observed_effect: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        effect = np.asarray(observed_effect, dtype=np.float64).reshape(-1)
        if (
            action.shape != (2,) or effect.shape != (2,)
            or not np.isfinite(action).all() or not np.isfinite(effect).all()
        ):
            raise ValueError("local action-effect update requires finite 2D values")
        self.actions.append(action.copy())
        self.effects.append(effect.copy())

    @property
    def sample_count(self) -> int:
        return len(self.actions)

    @property
    def trusted(self) -> bool:
        if self.quarantined or self.sample_count < self.min_samples:
            return False
        matrix = self._features(np.stack(self.actions))
        return bool(np.linalg.matrix_rank(matrix) >= 3)

    def predict(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("local action-effect query actions must have shape [N, 2]")
        if not self.trusted:
            return np.zeros_like(values)
        matrix = self._features(np.stack(self.actions))
        targets = np.stack(self.effects)
        weights = np.linalg.solve(
            matrix.T @ matrix + np.eye(3, dtype=np.float64) * self.ridge,
            matrix.T @ targets,
        )
        return self._features(values) @ weights

    def verify_execution(
        self,
        predicted_effect: np.ndarray,
        observed_effect: np.ndarray,
        tolerance: float = GDD_LOCAL_EXECUTION_RESIDUAL_TOLERANCE,
    ) -> bool:
        """Verify one released local action using only its next RGB frame.

        A failed check quarantines the episode-local chart.  The failed
        transition is deliberately not added to the chart, preventing an
        erroneous local inverse from teaching itself its own mistake.
        """

        predicted = np.asarray(predicted_effect, dtype=np.float64).reshape(-1)
        observed = np.asarray(observed_effect, dtype=np.float64).reshape(-1)
        if (
            predicted.shape != (2,) or observed.shape != (2,)
            or not np.isfinite(predicted).all() or not np.isfinite(observed).all()
            or tolerance <= 0.0
        ):
            raise ValueError("local execution verification requires finite 2D effects")
        residual = float(np.linalg.norm(predicted - observed))
        self.verification_errors.append(residual)
        passed = residual <= float(tolerance)
        if not passed:
            self.quarantined = True
        return passed


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

    def segment_is_free(
        self,
        start: np.ndarray,
        end: np.ndarray,
        clearance: float = 0.55,
        samples: int = 17,
    ) -> bool:
        """Check a proposed RGB-space displacement against inferred occupancy.

        This is deliberately conservative and uses only the static map fitted
        from calibration RGB.  It is a support gate, not privileged collision
        checking: if any sampled point is too far from every inferred free-cell
        center, the candidate is rejected.
        """

        start_value = np.asarray(start, dtype=np.float64).reshape(-1)
        end_value = np.asarray(end, dtype=np.float64).reshape(-1)
        if start_value.shape != (2,) or end_value.shape != (2,):
            raise ValueError("RGB maze segment endpoints must be 2D")
        if not np.isfinite(start_value).all() or not np.isfinite(end_value).all():
            raise ValueError("RGB maze segment endpoints must be finite")
        if clearance <= 0.0 or samples < 2:
            raise ValueError("RGB maze segment gate parameters are invalid")
        free_positions = np.stack([self.cell_positions[cell] for cell in self.free_cells])
        points = np.linspace(start_value, end_value, int(samples))
        distances = np.linalg.norm(points[:, None, :] - free_positions[None, :, :], axis=2)
        return bool(np.all(np.min(distances, axis=1) <= float(clearance)))

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


def active_probe_actions(
    current: np.ndarray,
    goal: np.ndarray,
    directions: int = 4,
    magnitude: float = 0.5,
) -> np.ndarray:
    """Return evenly spaced goal-relative probes in the runtime chart."""

    current = np.asarray(current, dtype=np.float64)[:2]
    goal = np.asarray(goal, dtype=np.float64)[:2]
    displacement = goal - current
    norm = float(np.linalg.norm(displacement))
    if norm <= 1.0e-8:
        raise ValueError("active probes require a nonzero current-goal displacement")
    if not np.isfinite(magnitude) or magnitude <= 0.0 or magnitude > 1.0:
        raise ValueError("active probe magnitude must lie in (0, 1]")
    if int(directions) < 4 or int(directions) % 2:
        raise ValueError("active probe directions must be an even number >= 4")
    direct = displacement / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    if int(directions) == 4:
        return np.stack([
            magnitude * direct, -magnitude * direct,
            magnitude * perpendicular, -magnitude * perpendicular,
        ]).astype(np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, int(directions), endpoint=False)
    return np.stack([
        magnitude * (np.cos(angle) * direct + np.sin(angle) * perpendicular)
        for angle in angles
    ]).astype(np.float64)


def paired_gdd_action_sequences(
    current: np.ndarray,
    goal: np.ndarray,
    steps: int = 20,
) -> np.ndarray:
    """Return the frozen ``paired_order_v2`` action family used by GDD.

    The GDD adapter was trained on two ordered half-horizon pulses, not on a
    repeated single action.  Keeping this generator local makes the live
    planner's action chart explicit and prevents silently feeding a different
    candidate family to the frozen adapter.
    """

    if steps != 20:
        raise ValueError("paired_order_v2 requires the frozen 20-step horizon")
    displacement = np.asarray(goal, dtype=np.float64).reshape(-1)[:2] - np.asarray(current, dtype=np.float64).reshape(-1)[:2]
    norm = float(np.linalg.norm(displacement))
    if norm <= 1.0e-8:
        raise ValueError("paired GDD action bank requires nonzero current-goal displacement")
    direct = displacement / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    half = steps // 2
    quarter = steps // 4
    sequences: list[np.ndarray] = []

    def pair(first: np.ndarray, second: np.ndarray) -> None:
        left = np.zeros((steps, 2), dtype=np.float64)
        right = np.zeros((steps, 2), dtype=np.float64)
        left[:half] = first
        left[half:] = second
        right[:half] = second
        right[half:] = first
        sequences.extend((np.clip(left, -1.0, 1.0), np.clip(right, -1.0, 1.0)))

    pair(direct * 0.75, perpendicular * 0.5)
    pair(direct * 0.75, -perpendicular * 0.5)
    pair(direct * 0.9, -direct * 0.3)
    pulse_a = np.zeros((steps, 2), dtype=np.float64)
    pulse_b = np.zeros((steps, 2), dtype=np.float64)
    pulse_a[:quarter] = direct
    pulse_b[-quarter:] = direct
    sequences.extend((pulse_a, pulse_b))
    sequences.append(np.tile(direct * 0.35, (steps, 1)))
    sequences.append(np.tile(perpendicular * 0.75, (steps, 1)))
    sequences.append(np.zeros((steps, 2), dtype=np.float64))
    # ``generate_candidate_bank.py`` deterministically appends five
    # support-bounded random sequences to reach the frozen count of 16.
    # Reproduce that exact canonical-bank tail; it is part of the adapter's
    # training support, not an online randomization.
    rng = np.random.default_rng(7000)
    while len(sequences) < 16:
        sequence = rng.uniform(-0.75, 0.75, size=(steps, 2))
        norms = np.linalg.norm(sequence, axis=1, keepdims=True)
        sequence = sequence * np.minimum(1.0, 0.75 / np.maximum(norms, 1.0e-12))
        sequences.append(sequence[:, 0, None] * direct + sequence[:, 1, None] * perpendicular)
    return np.stack(sequences, axis=0)


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
    # Preserve the exact preprocessing contract used by the frozen RGB
    # successor export for the GDD source chart.
    model._gdd_source_contract_transform = transform
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


def _make_gdd_source_contract_context(
    context: dict[str, torch.Tensor],
    transform: Any,
) -> dict[str, torch.Tensor]:
    """Apply the frozen RGB successor export contract before ``model.encode``.

    The released RGB successor artifacts were produced with
    ``raw / 255 -> transform -> *255 -> EncPredWM.encode``.  ``encode`` then
    performs its own ``/255`` and transform pass.  This apparently redundant
    round-trip is part of the frozen artifact provenance; silently replacing it
    with ``encode(raw)`` changes the source chart seen by the PCA/adapter.
    Proprio is left unchanged because ``encode`` owns its normalization.
    """

    if not isinstance(context, dict) or "visual" not in context or "proprio" not in context:
        raise ValueError("GDD context must contain visual and proprio tensors")
    visual = context["visual"]
    if visual.ndim != 5:
        raise ValueError(f"GDD visual context must be [B,T,C,H,W], got {tuple(visual.shape)}")
    if not callable(transform):
        raise ValueError("GDD source-contract transform is not callable")
    raw = visual.to(dtype=torch.float32)
    transformed = transform(raw / 255.0) * 255.0
    if not torch.is_tensor(transformed) or transformed.shape != raw.shape:
        raise ValueError(
            "GDD source-contract transform changed visual shape: "
            f"{tuple(raw.shape)} -> {getattr(transformed, 'shape', None)}"
        )
    return {"visual": transformed, "proprio": context["proprio"]}


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
    local_residual: OnlineLocalEffectResidual | None = None,
    velocity: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score repeated-pulse candidates from current RGB without state access."""

    context = _make_context(context_images, context_proprios)
    # Effect policies are fitted on the same frozen RGB source chart as the
    # successor export.  Applying model.encode(raw) here bypasses the
    # export's declared transform round-trip and can move an otherwise
    # matched history-4 context dozens of support units away.  GDD projection
    # already uses this wrapper; keep the effect-policy path identical.
    source_transform = getattr(model, "_gdd_source_contract_transform", None)
    if callable(source_transform):
        context = _make_gdd_source_contract_context(context, source_transform)
    with torch.no_grad():
        encoded = model.encode(context)
        pooled = np.stack([
            _pool_visual(encoded["visual"][:, index]).detach().cpu().numpy()[0]
            for index in range(len(context_images))
        ])
    if policy.feature_mode == "state_kinematics_action_basis":
        if velocity is None:
            raise ValueError("state_kinematics_action_basis requires RGB-derived velocity")
        projection = np.concatenate([
            np.asarray(current, dtype=np.float64).reshape(2),
            np.asarray(velocity, dtype=np.float64).reshape(2),
        ])
    else:
        projection = policy.project_history(pooled)
    report = policy.score(projection, current, target, actions)
    if local_residual is not None:
        correction = local_residual.predict(actions, velocity=velocity)
        effects = np.asarray(report["effects"], dtype=np.float64) + correction
        terminals = np.asarray(current, dtype=np.float64).reshape(1, 2) + effects
        report["effects"] = effects
        report["terminals"] = terminals
        report["costs"] = np.linalg.norm(
            terminals - np.asarray(target, dtype=np.float64).reshape(1, 2), axis=1,
        )
        report["online_residual_used"] = True
        report["online_residual_sample_count"] = local_residual.sample_count
        report["online_residual_trusted"] = local_residual.trusted
    else:
        report["online_residual_used"] = False
        report["online_residual_sample_count"] = 0
        report["online_residual_trusted"] = False
    report["source_projection"] = projection
    return report


def deployment_context_windows(effect_policy: OnlineEffectPolicy | None) -> int:
    """Return the causal RGB window required by the loaded effect policy."""

    if effect_policy is None:
        return 2
    return max(2, int(effect_policy.history_windows))


def deployment_warmup_blocks(
    probe_blocks: int,
    required_context_windows: int,
) -> int:
    """Align live history-N contexts with the exported source-frame clock."""

    if probe_blocks < 0:
        raise ValueError("probe_blocks must be non-negative")
    if required_context_windows < 1:
        raise ValueError("required_context_windows must be positive")
    # A history-N export starts at frame 5 and ends at frame 5*N.  Starting
    # from frame 0 therefore requires N explicit no-op blocks.
    return max(int(probe_blocks), int(required_context_windows))


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


def nearest_supported_action(
    proposed_action: np.ndarray,
    support_actions: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    """Map a proposal to the nearest action in the declared deployment bank."""

    proposal = np.asarray(proposed_action, dtype=np.float64).reshape(-1)
    bank = np.asarray(support_actions, dtype=np.float64)
    if proposal.shape != (2,) or bank.ndim != 2 or bank.shape[1] != 2:
        raise ValueError("proposal and support bank must have shapes [2] and [N, 2]")
    if len(bank) == 0 or not np.isfinite(proposal).all() or not np.isfinite(bank).all():
        raise ValueError("proposal and support bank must be non-empty and finite")
    distances = np.linalg.norm(bank - proposal[None, :], axis=1)
    index = int(np.argmin(distances))
    return index, bank[index].copy(), float(distances[index])


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
    local_effect_residual: OnlineLocalEffectResidual | None = None,
    local_action_effect_model: VerifiedLocalActionEffectModel | None = None,
    effect_regret_tolerance: float = 0.25,
    effect_visual_tolerance: float = 0.10,
    gdd_adapter: tuple[Any, dict[str, Any]] | None = None,
    gdd_projection: np.ndarray | None = None,
    gdd_goal_anchor: np.ndarray | None = None,
    gdd_score_horizons: tuple[int, ...] = (4,),
    gdd_candidate_count: int = 16,
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
    elif method in {
        "gdd_wm_rgb", "gdd_wm_rgb_guarded", "gdd_wm_rgb_verified_local",
        "gdd_wm_rgb_disagreement_verified_local", "gdd_wm_rgb_projected_guarded",
        "gdd_wm_rgb_waypoint",
        "gdd_wm_rgb_waypoint_guarded",
        "gdd_wm_rgb_support_preserving_guarded",
        "gdd_wm_rgb_support_matched_guarded",
    }:
        if gdd_adapter is None or gdd_projection is None or gdd_goal_anchor is None:
            raise ValueError("gdd_wm_rgb requires a frozen adapter, projection, and goal anchor")
        from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _predict
        adapter, adapter_payload = gdd_adapter
        pca_mean = np.asarray(adapter_payload["pca_mean"], dtype=np.float64)
        pca_basis = np.asarray(adapter_payload["pca_basis"], dtype=np.float64)
        action_chart = str(adapter_payload.get("adapter_training", {}).get("action_feature_chart", "raw"))
        with torch.no_grad():
            context = _make_context(context_images, context_proprios)
            source_transform = getattr(model, "_gdd_source_contract_transform", None)
            encoded = model.encode(_make_gdd_source_contract_context(context, source_transform))
            pooled = _pool_visual(encoded["visual"])[0].detach().cpu().numpy().astype(np.float64)
        source_projection = (pooled - pca_mean) @ pca_basis.T
        # The frozen adapter was trained on the exact paired_order_v2 family:
        # two ordered ten-step pulses, represented as four five-step blocks.
        # The selected sequence's first five-step block is the only action
        # released to the environment before RGB re-observation.
        fallback_actions = actions.copy()
        waypoint_conditioned = method in {
            "gdd_wm_rgb_waypoint_guarded",
            "gdd_wm_rgb_support_matched_guarded",
        }
        gdd_conditioning_target = target if waypoint_conditioned else goal
        support_matched = method == "gdd_wm_rgb_support_matched_guarded"
        if support_matched:
            # This family is identical to the package builder: every visual
            # fixed-bank action is repeated for the frozen 20-step adapter
            # horizon, and the first five-step block is released online.
            gdd_sequences = np.tile(
                fallback_actions[:, None, :], (1, 20, 1),
            )
        else:
            gdd_sequences = paired_gdd_action_sequences(current, gdd_conditioning_target)
        if not 4 <= int(gdd_candidate_count) <= len(gdd_sequences):
            raise ValueError(
                f"GDD candidate count must lie in [4, {len(gdd_sequences)}], "
                f"got {gdd_candidate_count}"
            )
        gdd_sequences = gdd_sequences[:int(gdd_candidate_count)]
        actions = gdd_sequences[:, 0, :].copy()
        repeated = gdd_sequences.reshape(len(gdd_sequences), 4, 10)
        # The adapter/projection was trained with the declared global task
        # goal. The RGB maze waypoint only determines the local action bank;
        # feeding it as the student goal creates a train/deploy covariate
        # shift and invalidates the learned geometry.
        rows = [{"source_state_raw": np.r_[current, velocity], "goal_raw": gdd_conditioning_target} for _ in actions]
        if action_chart != "raw":
            from experiments.empirical_spectral_audit.pointmaze_action_charts import action_blocks_for_chart
            repeated = action_blocks_for_chart(rows, repeated, action_chart)
        source_matrix = np.repeat(source_projection[None, :], len(actions), axis=0)
        adapter_features = np.concatenate([source_matrix, repeated.reshape(len(actions), -1)], axis=1)
        prediction = _predict(
            adapter, adapter_features, int(pca_basis.shape[0]), str(model.device),
            adapter_payload["adapter_training"],
        )
        anchor = np.asarray(gdd_goal_anchor, dtype=np.float64).reshape(-1)
        projection = np.asarray(gdd_projection, dtype=np.float64)
        score_horizons = tuple(int(horizon) for horizon in gdd_score_horizons)
        if not score_horizons or any(horizon not in HORIZONS for horizon in score_horizons):
            raise ValueError(
                f"GDD score horizons must be a non-empty subset of {HORIZONS}, "
                f"got {score_horizons}"
            )
        candidate_scores = []
        for horizon in score_horizons:
            latent = np.asarray(prediction[horizon], dtype=np.float64)
            features = np.concatenate([latent, np.repeat(gdd_conditioning_target[None, :], len(actions), axis=0)], axis=1)
            centered = features - anchor.reshape(1, -1)
            candidate_scores.append(np.linalg.norm(centered @ projection.T, axis=1) ** 2 / projection.shape[0])
        scores = np.mean(np.stack(candidate_scores, axis=1), axis=1)
        index = int(np.argmin(scores))
        if method in {
            "gdd_wm_rgb_guarded", "gdd_wm_rgb_projected_guarded",
            "gdd_wm_rgb_waypoint_guarded",
            "gdd_wm_rgb_support_preserving_guarded",
            "gdd_wm_rgb_support_matched_guarded",
        }:
            # Compare the GDD proposal against the *actual* visual-dynamics
            # bank, not only against the GDD-restricted candidate family.
            # The old comparison could certify a bad GDD action because the
            # visual baseline action was absent from the reference set.
            gdd_dynamics_scores = visual_dynamics_scores(
                current, velocity, target, actions, dynamics, kp, kv,
            )
            baseline_scores = visual_dynamics_scores(
                current, velocity, target, fallback_actions, dynamics, kp, kv,
            )
            baseline_index = int(np.argmin(baseline_scores))
            gdd_selected_action = actions[index].copy()
            support_index = baseline_index
            support_distance = 0.0
            if method == "gdd_wm_rgb_support_preserving_guarded":
                # Score with the frozen GDD family, but release only an action
                # from the same bank used by the visual-dynamics reference.
                support_index, _support_action, support_distance = nearest_supported_action(
                    gdd_selected_action, fallback_actions,
                )
                gdd_dynamics_scores = visual_dynamics_scores(
                    current, velocity, target, fallback_actions, dynamics, kp, kv,
                )
                gdd_regret = float(
                    gdd_dynamics_scores[support_index] - baseline_scores[baseline_index]
                )
            else:
                gdd_regret = float(gdd_dynamics_scores[index] - baseline_scores[baseline_index])
            gate_info.update({
                "gdd_dynamics_selected_index": baseline_index,
                "gdd_dynamics_regret": gdd_regret,
                "gdd_reference_bank": "visual_dynamics_fixed_bank",
                "gdd_reference_best_score": float(baseline_scores[baseline_index]),
                "gdd_selected_score": float(gdd_dynamics_scores[index]),
                "gdd_guard_tolerance": float(wm_dynamics_tolerance),
            })
            if method == "gdd_wm_rgb_support_preserving_guarded":
                gate_info.update({
                    "gdd_support_preserving": True,
                    "gdd_support_selected_index": int(support_index),
                    "gdd_support_mapping_distance": float(support_distance),
                    "gdd_release_tolerance": 0.0,
                })
                actions = fallback_actions
                scores = baseline_scores
                index = support_index
            release_tolerance = (
                0.0
                if method in {
                    "gdd_wm_rgb_support_preserving_guarded",
                    "gdd_wm_rgb_support_matched_guarded",
                }
                else float(wm_dynamics_tolerance)
            )
            if gdd_regret > release_tolerance:
                actions = fallback_actions
                scores = baseline_scores
                index = baseline_index
                gate_info.update({
                    "fallback_used": True,
                    "fallback_reason": "gdd_action_effect_regret_exceeds_rgb_dynamics_support",
                    "fallback_action_family": "visual_dynamics_fixed_bank",
                })
        gate_info.update({
            "gdd_projection_used": True,
            "gdd_feature_source": (
                "RGB_encoder_PCA_plus_frozen_adapter_prediction_plus_active_waypoint_anchor"
                if waypoint_conditioned
                else "RGB_encoder_PCA_plus_frozen_adapter_prediction_plus_fixed_goal_anchor"
            ),
            "gdd_teacher_runtime_used": False,
            "gdd_candidate_id_input": False,
            "gdd_teacher_rank_input": False,
            "gdd_action_family": (
                "visual_fixed_bank_repeated_20_step_v1"
                if support_matched else "paired_order_v2"
            ),
            "gdd_action_sequence_count": int(len(gdd_sequences)),
            "gdd_goal_conditioning": "active_waypoint" if waypoint_conditioned else "global_goal",
        })
        if method == "gdd_wm_rgb_projected_guarded":
            # Preserve the GDD preference only along a calibration-supported
            # segment toward the actual visual-dynamics action.  This handles
            # route/action-support mismatch without inventing a new direction.
            if gdd_regret > float(wm_dynamics_tolerance):
                executed_action = fallback_actions[baseline_index]
                projection_alpha = 0.0
                projected_score = float(baseline_scores[baseline_index])
                support_score = projected_score
                projection_fallback = True
            else:
                (
                    executed_action,
                    projection_alpha,
                    projected_score,
                    support_score,
                ) = project_action_to_dynamics_support(
                    gdd_selected_action,
                    fallback_actions[baseline_index],
                    current,
                    velocity,
                    target,
                    dynamics,
                    kp,
                    kv,
                    float(wm_dynamics_tolerance),
                )
                projection_fallback = projection_alpha < float(projection_min_alpha)
                if projection_fallback:
                    executed_action = fallback_actions[baseline_index]
                    projection_alpha = 0.0
            gate_info.update({
                "projection_used": bool(projection_alpha < 1.0 - 1.0e-12),
                "projection_alpha": float(projection_alpha),
                "projected_dynamics_score": float(projected_score),
                "dynamics_support_score": float(support_score),
                "projection_min_alpha": float(projection_min_alpha),
                "fallback_used": bool(
                    projection_fallback
                    or gdd_regret > float(wm_dynamics_tolerance)
                ),
                "fallback_reason": (
                    "gdd_cross_bank_regret_exceeds_tolerance"
                    if gdd_regret > float(wm_dynamics_tolerance)
                    else "projection_alpha_below_minimum"
                    if projection_fallback else None
                ),
            })
            return executed_action, {
                "selected_index": int(index),
                "candidate_count": int(len(actions)),
                "current_visual_position": current.tolist(),
                "goal_visual_position": goal.tolist(),
                "visual_velocity": velocity.tolist(),
                "route_waypoints": [np.asarray(point).tolist() for point in waypoints],
                "active_waypoint": np.asarray(target).tolist(),
                "selected_action": executed_action.tolist(),
                "candidate_scores": np.asarray(scores).tolist(),
                "action_transfer_gate": gate_info,
            }
        if method in {
            "gdd_wm_rgb_verified_local",
            "gdd_wm_rgb_disagreement_verified_local",
        }:
            # The projection student proposes a candidate, but the proposal is
            # not released until an observed RGB action-effect chart has
            # enough rank-two support.  This is deliberately a recovery arm:
            # it does not feed teacher ranks or held-out outcomes into the
            # student, and it falls back to the ordinary visual-dynamics bank
            # when the local chart is not identifiable.
            if local_action_effect_model is None:
                raise RuntimeError("GDD verified-local method requires a local action-effect model")
            local_effects = local_action_effect_model.predict(actions)
            local_support_limit = float(GDD_VERIFIED_LOCAL_PROBE_MAGNITUDE * 1.5)
            local_supported = (
                np.linalg.norm(np.asarray(actions, dtype=np.float64), axis=1)
                <= local_support_limit + 1.0e-12
            )
            local_path_supported = np.ones(len(actions), dtype=bool)
            if (
                method == "gdd_wm_rgb_disagreement_verified_local"
                and maze_map is not None
            ):
                local_endpoints = np.asarray(current, dtype=np.float64).reshape(1, 2) + local_effects
                local_path_supported = np.asarray([
                    maze_map.segment_is_free(current, endpoint)
                    for endpoint in local_endpoints
                ], dtype=bool)
                local_supported &= local_path_supported
            if not np.any(local_supported):
                local_supported = np.zeros(len(actions), dtype=bool)
            local_costs = np.linalg.norm(
                np.asarray(current, dtype=np.float64).reshape(1, 2)
                + local_effects
                - np.asarray(target, dtype=np.float64).reshape(1, 2),
                axis=1,
            )
            local_costs = np.where(local_supported, local_costs, np.inf)
            local_index = int(np.argmin(local_costs))
            local_trusted = bool(local_action_effect_model.trusted)
            gdd_index = int(index)
            dynamics_scores = visual_dynamics_scores(
                current, velocity, target, fallback_actions, dynamics, kp, kv,
            )
            dynamics_index = int(np.argmin(dynamics_scores))
            local_fallback_effects = local_action_effect_model.predict(fallback_actions)
            local_fallback_costs = np.linalg.norm(
                np.asarray(current, dtype=np.float64).reshape(1, 2)
                + local_fallback_effects
                - np.asarray(target, dtype=np.float64).reshape(1, 2),
                axis=1,
            )
            local_visual_regret = float(
                local_costs[local_index] - local_fallback_costs[dynamics_index]
            )
            gate_info.update({
                "gdd_proposed_index": gdd_index,
                "gdd_local_selected_index": local_index,
                "gdd_local_chart_trusted": local_trusted,
                "gdd_local_chart_sample_count": int(local_action_effect_model.sample_count),
                "gdd_local_chart_basis": "observed_rgb_affine_bias_plus_action_jacobian",
                "gdd_local_action_support_limit": local_support_limit,
                "gdd_local_supported_candidate_count": int(np.sum(local_supported)),
                "gdd_local_path_supported_candidate_count": int(np.sum(local_path_supported)),
                "gdd_local_path_gate": bool(
                    method == "gdd_wm_rgb_disagreement_verified_local"
                ),
                "gdd_local_visual_dynamics_index": dynamics_index,
                "gdd_local_visual_regret": local_visual_regret,
                "gdd_local_visual_tolerance": 0.10,
                "effect_selected_index": local_index,
                "predicted_effect": local_effects[local_index].tolist(),
            })
            if local_trusted and np.isfinite(local_costs[local_index]) and local_visual_regret <= 0.10:
                index = local_index
                gate_info.update({
                    "fallback_used": False,
                    "fallback_reason": "gdd_proposal_verified_by_local_rgb_effect",
                })
            else:
                actions = fallback_actions
                scores = dynamics_scores
                index = dynamics_index
                gate_info.update({
                    "effect_selected_index": None,
                    "fallback_used": True,
                    "fallback_reason": (
                        "gdd_local_effect_support_insufficient" if not local_trusted
                        else "gdd_local_effect_visual_consistency_failed"
                    ),
                    "fallback_action_family": "visual_dynamics_fixed_bank",
                })
    elif method == "gdd_runtime_task_effect_rgb":
        if gdd_adapter is None:
            raise ValueError("gdd_runtime_task_effect_rgb requires --gdd-adapter-checkpoint")
        from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _predict
        adapter, adapter_payload = gdd_adapter
        pca_mean = np.asarray(adapter_payload["pca_mean"], dtype=np.float64)
        pca_basis = np.asarray(adapter_payload["pca_basis"], dtype=np.float64)
        action_chart = str(adapter_payload.get("adapter_training", {}).get("action_feature_chart", "raw"))
        # This checkpoint was trained on the exact 32-candidate
        # runtime_repeated_pulse_v1 family.  Its source chart is the ordinary
        # single-encode RGB export, so do not apply the GDD successor-export
        # double-transform contract here.
        with torch.no_grad():
            encoded = model.encode(_make_context(context_images, context_proprios))
            pooled = _pool_visual(encoded["visual"])[0].detach().cpu().numpy().astype(np.float64)
        source_projection = (pooled - pca_mean) @ pca_basis.T
        runtime_actions = action_bank(current, goal, directions, magnitudes)
        repeated = np.repeat(runtime_actions[:, None, :], 20, axis=1).reshape(len(runtime_actions), 4, 10)
        rows = [{"source_state_raw": np.r_[current, velocity], "goal_raw": goal} for _ in runtime_actions]
        if action_chart != "raw":
            from experiments.empirical_spectral_audit.pointmaze_action_charts import action_blocks_for_chart
            repeated = action_blocks_for_chart(rows, repeated, action_chart)
        source_matrix = np.repeat(source_projection[None, :], len(runtime_actions), axis=0)
        adapter_features = np.concatenate([source_matrix, repeated.reshape(len(runtime_actions), -1)], axis=1)
        prediction = _predict(
            adapter, adapter_features, int(pca_basis.shape[0]), str(model.device),
            adapter_payload["adapter_training"],
        )
        task_training = adapter_payload["adapter_training"]
        task_scores = []
        for horizon in HORIZONS:
            latent = np.asarray(prediction[horizon], dtype=np.float64)
            basis = np.asarray(task_training["task_bases"][str(horizon)], dtype=np.float64).reshape(1, -1)
            center = np.asarray(task_training["task_centers"][str(horizon)], dtype=np.float64).reshape(1, -1)
            target_center = float(task_training.get("task_target_centers", {}).get(str(horizon), 0.0))
            # Recover the scalar task-effect coefficient from the frozen
            # one-dimensional task subspace; higher predicted progress is
            # preferred, so the planner minimizes its negative.
            coefficient = np.sum((latent - center) * basis, axis=1) / max(float(np.sum(basis * basis)), 1.0e-12)
            task_scores.append(coefficient + target_center)
        scores = -np.mean(np.stack(task_scores, axis=1), axis=1)
        index = int(np.argmin(scores))
        actions = runtime_actions
        gate_info.update({
            "gdd_runtime_task_effect_used": True,
            "gdd_feature_source": "RGB_encoder_PCA_plus_frozen_runtime_repeated_pulse_task_effect_adapter",
            "gdd_task_effect_checkpoint": str(adapter_payload.get("protocol", {}).get("checkpoint", "unknown")),
            "gdd_candidate_id_input": False,
            "gdd_teacher_runtime_used": False,
            "gdd_action_family": "runtime_repeated_pulse_v1",
            "gdd_action_sequence_count": int(len(runtime_actions)),
        })
    elif method in {
        "effect_adapter_rgb_online_local_jacobian",
        "effect_adapter_rgb_online_local_direct",
        "effect_adapter_rgb_online_local_phase",
    }:
        if local_action_effect_model is None:
            raise ValueError("local-jacobian method requires an observed-effect model")
        local_effects = local_action_effect_model.predict(actions)
        local_costs = np.linalg.norm(
            np.asarray(current, dtype=np.float64).reshape(1, 2)
            + local_effects
            - np.asarray(target, dtype=np.float64).reshape(1, 2),
            axis=1,
        )
        index = int(np.argmin(local_costs))
        local_trusted = bool(local_action_effect_model.trusted)
        gate_info.update({
            "local_action_effect_trusted": local_trusted,
            "local_action_effect_sample_count": local_action_effect_model.sample_count,
            "local_action_effect_basis": "affine_bias_plus_action_jacobian",
            "effect_policy_released": local_trusted,
            "predicted_effect": local_effects[index].tolist(),
        })
        dynamics_scores = visual_dynamics_scores(
            current, velocity, target, actions, dynamics, kp, kv,
        )
        dynamics_index = int(np.argmin(dynamics_scores))
        local_regret = float(local_costs[index] - dynamics_scores[dynamics_index])
        local_consistent = bool(local_regret <= float(effect_visual_tolerance))
        direct_release = method in {
            "effect_adapter_rgb_online_local_direct",
            "effect_adapter_rgb_online_local_phase",
        }
        gate_info.update({
            "local_action_effect_dynamics_index": dynamics_index,
            "local_action_effect_regret_of_visual": local_regret,
            "local_action_effect_visual_tolerance": float(effect_visual_tolerance),
            "local_action_effect_consistent": local_consistent,
            "local_action_effect_release_rule": (
                "trusted_chart_direct_ranking" if direct_release
                else "trusted_chart_and_visual_consistency"
            ),
        })
        if not local_trusted or (not direct_release and not local_consistent):
            index = dynamics_index
            gate_info.update({
                "dynamics_selected_index": dynamics_index,
                "fallback_used": True,
                "fallback_reason": (
                    "local_action_effect_not_identifiable" if not local_trusted
                    else "local_action_effect_visual_consistency_failed"
                ),
            })
        scores = local_costs
    elif method in {
        "effect_adapter_rgb", "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_htsc_consistency", "effect_adapter_rgb_online",
        "effect_adapter_rgb_online_htsc", "effect_adapter_rgb_online_probe",
        "effect_adapter_rgb_online_verified_local", "effect_adapter_rgb_online_local_jacobian",
        "effect_adapter_rgb_online_verified_local",
    }:
        if effect_policy is None:
            raise ValueError(f"{method} requires --effect-policy-checkpoint")
        effect_report = effect_policy_candidate_scores(
            model, effect_policy, context_images, context_proprios,
            current, target, actions,
            local_residual=(
                local_effect_residual
                if method in {
                    "effect_adapter_rgb_online", "effect_adapter_rgb_online_htsc",
                    "effect_adapter_rgb_online_probe", "effect_adapter_rgb_online_verified_local",
                }
                else None
            ),
            velocity=velocity,
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
            "online_residual_used": bool(effect_report.get("online_residual_used", False)),
            "online_residual_sample_count": int(effect_report.get("online_residual_sample_count", 0)),
        })
        index = effect_index
        if method in {
            "effect_adapter_rgb_htsc", "effect_adapter_rgb_htsc_consistency",
            "effect_adapter_rgb_online_htsc",
            "effect_adapter_rgb_online_probe",
        } and not released:
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
        elif method == "effect_adapter_rgb_online_verified_local" and not released:
            # Probe-and-update recovery arm. It may use the local observed
            # effect model only after this episode has verified enough probes
            # and the current state/action remain inside support. The global
            # teacher bound remains reported as failed; this arm tests local
            # system identification separately.
            local_verified = bool(
                local_effect_residual is not None
                and local_effect_residual.trusted
                and state_support_passed
                and action_support_passed
            )
            gate_info.update({
                "local_verified_release": local_verified,
                "local_verified_sample_count": (
                    0 if local_effect_residual is None else local_effect_residual.sample_count
                ),
            })
            if not local_verified:
                dynamics_scores = visual_dynamics_scores(
                    current, velocity, target, actions, dynamics, kp, kv,
                )
                dynamics_index = int(np.argmin(dynamics_scores))
                index = dynamics_index
                gate_info.update({
                    "dynamics_selected_index": dynamics_index,
                    "fallback_used": True,
                    "fallback_reason": "local_effect_verification_or_support_failed",
                })
            else:
                gate_info.update({
                    "fallback_used": False,
                    "fallback_reason": "global_teacher_bound_failed_local_verified_override",
                })
        elif method == "effect_adapter_rgb_htsc_consistency":
            dynamics_scores = visual_dynamics_scores(
                current, velocity, target, actions, dynamics, kp, kv,
            )
            dynamics_index = int(np.argmin(dynamics_scores))
            dynamics_regret = float(dynamics_scores[effect_index] - dynamics_scores[dynamics_index])
            gate_info.update({
                "dynamics_selected_index": dynamics_index,
                "dynamics_regret_of_effect": dynamics_regret,
                "effect_visual_tolerance": float(effect_visual_tolerance),
            })
            if dynamics_regret > float(effect_visual_tolerance):
                index = dynamics_index
                gate_info.update({
                    "fallback_used": True,
                    "fallback_reason": "effect_visual_consistency_failed",
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
    effect_visual_tolerance: float = 0.10,
    gdd_adapter: tuple[Any, dict[str, Any]] | None = None,
    gdd_projection: np.ndarray | None = None,
    gdd_goal_anchor: np.ndarray | None = None,
    gdd_disagreement_cosine_threshold: float = GDD_DISAGREEMENT_COSINE_THRESHOLD,
    gdd_score_horizons: tuple[int, ...] = (4,),
    gdd_candidate_count: int = 16,
    active_probe_directions: int = 4,
    gdd_authorized: bool = True,
) -> dict[str, Any]:
    requested_method = method
    gdd_authorization_fallback = False
    if (
        not gdd_authorized
        and method in {
            "gdd_wm_rgb", "gdd_wm_rgb_guarded", "gdd_wm_rgb_verified_local",
            "gdd_wm_rgb_disagreement_verified_local", "gdd_wm_rgb_projected_guarded",
            "gdd_runtime_task_effect_rgb",
        }
    ):
        # Fail closed before probe setup: a rejected teacher must not spend
        # information-action budget before reverting to visual dynamics.
        method = "visual_dynamics"
        gdd_authorization_fallback = True
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
    required_context_windows = deployment_context_windows(effect_policy)
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
    # A history-aware effect policy must receive the same declared number of
    # causal RGB contexts at deployment.  The old live path always retained
    # exactly two frames, silently truncating history-4 policies to history-2.
    # Warm-up uses explicit no-op blocks and is recorded in the report; it is
    # not replaced by duplicated frames.  Runtime temporal exports define a
    # history-N source at frames 5, ..., 5*N (the final frame is the source
    # observation).  Starting from frame 0 therefore requires N no-op blocks;
    # using N-1 would silently deploy [0, 5, ..., 5*(N-1)] and move the live
    # latent chart one action block away from the fitted teacher.
    warmup_blocks = deployment_warmup_blocks(probe_blocks, required_context_windows)
    history_images = [current_rgb]
    history_proprios = [proprio.copy()]
    for _ in range(warmup_blocks):
        runtime.step_block(np.zeros(2, dtype=np.float64))
        probe_rgb = renderer.render(runtime.state)
        probe_center = green_center(probe_rgb)
        probe_delta = (probe_center[0] - center[0], probe_center[1] - center[1])
        probe_proprio = np.concatenate([
            calibration.position(probe_center), calibration.velocity(probe_delta)
        ])
        history_images.append(probe_rgb)
        history_proprios.append(probe_proprio)
        center = probe_center
        current_rgb = probe_rgb
    if warmup_blocks == 0:
        history_images.append(current_rgb.copy())
        history_proprios.append(proprio.copy())
    context_images = history_images[-required_context_windows:]
    context_proprios = history_proprios[-required_context_windows:]
    planning_start_state = runtime.state.copy()
    rng = np.random.default_rng(seed)
    # Non-online baselines must pass an explicit no-residual state through the
    # shared planner path.  Leaving this uninitialized made the four-arm
    # runner fail on the first visual-dynamics episode before any scientific
    # metric was produced.
    local_effect_residual: OnlineLocalEffectResidual | None = None
    local_action_effect_model: VerifiedLocalActionEffectModel | None = None
    if method in {
        "effect_adapter_rgb_online", "effect_adapter_rgb_online_htsc",
        "effect_adapter_rgb_online_probe", "effect_adapter_rgb_online_verified_local",
        "effect_adapter_rgb_online_local_jacobian", "effect_adapter_rgb_online_local_direct",
        "effect_adapter_rgb_online_local_phase",
        "gdd_wm_rgb_verified_local",
        "gdd_wm_rgb_disagreement_verified_local",
    }:
        verification_threshold = (
            float(effect_policy.verification_threshold)
            if effect_policy is not None else 0.25
        )
        local_effect_residual = OnlineLocalEffectResidual(
            verification_threshold=verification_threshold,
            verification_direction_threshold=(
                None if effect_policy is None else effect_policy.verification_direction_threshold
            ),
        )
    if method in {
        "effect_adapter_rgb_online_local_jacobian", "effect_adapter_rgb_online_local_direct",
        "effect_adapter_rgb_online_local_phase",
        "gdd_wm_rgb_verified_local",
        "gdd_wm_rgb_disagreement_verified_local",
    }:
        local_action_effect_model = VerifiedLocalActionEffectModel()
    active_probe_trace: list[dict[str, Any]] = []
    probe_triggered_flag: bool | None = None
    if method in {
        "effect_adapter_rgb_online_probe", "effect_adapter_rgb_online_verified_local",
        "effect_adapter_rgb_online_local_jacobian", "effect_adapter_rgb_online_local_direct",
        "effect_adapter_rgb_online_local_phase",
        "gdd_wm_rgb_verified_local",
        "gdd_wm_rgb_disagreement_verified_local",
    }:
        # Predeclared action-space coverage probes.  These are deliberately
        # independent of the task-cost ranking: each cardinal pulse is
        # executed, re-observed, and used only to update the local effect
        # residual before ordinary planning begins.
        # The runtime candidate chart is goal-relative: these are exactly its
        # four cardinal directions at magnitude 0.5, rotated into raw control
        # coordinates for the current RGB-estimated state.
        probe_position = calibration.position(center)
        probe_target = GOAL_STATE.copy()
        if planner_map == "rgb_maze" and maze_map is not None:
            probe_waypoints = maze_map.path_waypoints(probe_position, GOAL_STATE)
            if probe_waypoints:
                probe_target = np.asarray(probe_waypoints[0], dtype=np.float64)
        probe_actions = active_probe_actions(
            probe_position, probe_target,
            directions=active_probe_directions,
            magnitude=(
                GDD_VERIFIED_LOCAL_PROBE_MAGNITUDE
                if method in {
                    "gdd_wm_rgb_verified_local",
                    "gdd_wm_rgb_disagreement_verified_local",
                } else 0.5
            ),
        )
        run_probes = True
        disagreement_info: dict[str, Any] = {}
        if method == "gdd_wm_rgb_disagreement_verified_local":
            # Dry-run both frozen proposal families before executing any
            # information action.  This uses only current RGB, calibration
            # dynamics, and the frozen student; no simulator outcome is read.
            gdd_dry_action, gdd_dry_decision = choose_action(
                "gdd_wm_rgb", model, readout, current_rgb, goal_rgb,
                context_images, context_proprios, calibration, action_mean,
                action_std, directions, magnitudes, dynamics, maze_map,
                planner_map, kp, kv, wm_dynamics_tolerance, rng,
                projection_min_alpha, runtime=runtime,
                oracle_depth=oracle_depth, oracle_beam_width=oracle_beam_width,
                effect_policy=effect_policy,
                gdd_adapter=gdd_adapter, gdd_projection=gdd_projection,
                gdd_goal_anchor=gdd_goal_anchor,
                gdd_score_horizons=gdd_score_horizons,
                gdd_candidate_count=gdd_candidate_count,
            )
            visual_actions = action_bank(
                probe_position, probe_target, directions, magnitudes,
            )
            visual_scores = visual_dynamics_scores(
                probe_position,
                np.asarray(context_proprios[-1], dtype=np.float64)[2:4],
                probe_target, visual_actions, dynamics, kp, kv,
            )
            visual_action = visual_actions[int(np.argmin(visual_scores))]
            gdd_norm = float(np.linalg.norm(gdd_dry_action))
            visual_norm = float(np.linalg.norm(visual_action))
            cosine = float(
                np.dot(gdd_dry_action, visual_action) / max(gdd_norm * visual_norm, 1.0e-12)
            ) if gdd_norm > 1.0e-12 and visual_norm > 1.0e-12 else -1.0
            run_probes = cosine < float(gdd_disagreement_cosine_threshold)
            probe_triggered_flag = bool(run_probes)
            disagreement_info = {
                "gdd_dry_action": gdd_dry_action.tolist(),
                "visual_dry_action": visual_action.tolist(),
                "proposal_cosine": cosine,
                "cosine_threshold": float(gdd_disagreement_cosine_threshold),
                "probe_triggered": bool(run_probes),
                "gdd_dry_decision": gdd_dry_decision,
            }
        if not run_probes:
            active_probe_trace.append({
                "probe_skipped": True,
                "reason": "frozen_gdd_and_visual_proposals_agree",
                **disagreement_info,
            })
        for probe_index, probe_action in enumerate(probe_actions if run_probes else []):
            try:
                previous_position = calibration.position(center)
                previous_velocity = np.asarray(context_proprios[-1], dtype=np.float64)[2:4]
                runtime.step_block(probe_action)
                next_rgb = renderer.render(runtime.state)
                next_center = green_center(next_rgb)
                observed_position = calibration.position(next_center)
                predicted_effect = None
                if method not in {
                    "gdd_wm_rgb_verified_local",
                    "gdd_wm_rgb_disagreement_verified_local",
                    "effect_adapter_rgb_online_local_jacobian",
                    "effect_adapter_rgb_online_local_direct",
                    "effect_adapter_rgb_online_local_phase",
                }:
                    probe_report = effect_policy_candidate_scores(
                        model, effect_policy, context_images, context_proprios,
                        calibration.position(center), GOAL_STATE, probe_action[None, :],
                    )
                    predicted_effect = np.asarray(probe_report["effects"][0], dtype=np.float64)
                    local_effect_residual.update_verified_execution(
                        probe_action,
                        observed_position - previous_position,
                        velocity=previous_velocity,
                        predicted_effect=predicted_effect,
                        verification_baseline_effect=dynamics.predict_position_delta(
                            previous_velocity, np.zeros(2, dtype=np.float64),
                        ),
                    )
                if local_action_effect_model is not None:
                    local_action_effect_model.update(
                        probe_action,
                        observed_position - previous_position,
                    )
                delta = (next_center[0] - center[0], next_center[1] - center[1])
                next_proprio = np.concatenate([
                    calibration.position(next_center), calibration.velocity(delta),
                ])
                context_images = (context_images + [next_rgb])[-required_context_windows:]
                context_proprios = (context_proprios + [next_proprio])[-required_context_windows:]
                current_rgb = next_rgb
                center = next_center
                active_probe_trace.append({
                    "probe_index": int(probe_index),
                    "action": probe_action.tolist(),
                    "coordinate_frame": "goal_relative_runtime_chart",
                    "probe_target": probe_target.tolist(),
                    "predicted_effect": None if predicted_effect is None else predicted_effect.tolist(),
                    "observed_effect": (observed_position - previous_position).tolist(),
                    "verification_error": (
                        None if method in {
                            "gdd_wm_rgb_verified_local",
                            "gdd_wm_rgb_disagreement_verified_local",
                            "effect_adapter_rgb_online_local_jacobian",
                            "effect_adapter_rgb_online_local_direct",
                            "effect_adapter_rgb_online_local_phase",
                        }
                        else float(local_effect_residual.verification_errors[-1])
                    ),
                    "residual_trusted": bool(
                        local_action_effect_model.trusted
                        if method in {
                            "gdd_wm_rgb_verified_local",
                            "gdd_wm_rgb_disagreement_verified_local",
                        } and local_action_effect_model is not None
                        else local_effect_residual.trusted
                    ),
                })
            except ValueError as exc:
                active_probe_trace.append({
                    "probe_index": int(probe_index),
                    "action": probe_action.tolist(),
                    "coordinate_frame": "goal_relative_runtime_chart",
                    "probe_target": probe_target.tolist(),
                    "error": str(exc),
                })
                break
    phase_reprobe_blocks = 0

    def remeasure_phase_chart(phase_target: np.ndarray) -> int:
        """Re-identify the RGB local chart after a route-phase transition.

        The initial chart is not transported across a waypoint/obstacle phase.
        This intervention spends the same predeclared active-probe budget in
        the new phase and uses only observed RGB successors.  It is deliberately
        isolated to the phase-reset arm so the historical local-direct result
        remains reproducible.
        """

        nonlocal current_rgb, center, context_images, context_proprios
        nonlocal local_action_effect_model, phase_reprobe_blocks
        local_action_effect_model = VerifiedLocalActionEffectModel()
        probe_position = calibration.position(center)
        probe_actions = active_probe_actions(
            probe_position,
            np.asarray(phase_target, dtype=np.float64),
            directions=active_probe_directions,
            magnitude=0.5,
        )
        executed = 0
        for probe_index, probe_action in enumerate(probe_actions):
            try:
                previous_position = calibration.position(center)
                previous_velocity = np.asarray(context_proprios[-1], dtype=np.float64)[2:4]
                runtime.step_block(probe_action)
                next_rgb = renderer.render(runtime.state)
                next_center = green_center(next_rgb)
                observed_position = calibration.position(next_center)
                local_action_effect_model.update(
                    probe_action,
                    observed_position - previous_position,
                )
                delta = (next_center[0] - center[0], next_center[1] - center[1])
                next_proprio = np.concatenate([
                    calibration.position(next_center), calibration.velocity(delta),
                ])
                context_images = (context_images + [next_rgb])[-required_context_windows:]
                context_proprios = (context_proprios + [next_proprio])[-required_context_windows:]
                current_rgb = next_rgb
                center = next_center
                active_probe_trace.append({
                    "phase_reprobe": True,
                    "probe_index": int(probe_index),
                    "action": probe_action.tolist(),
                    "coordinate_frame": "goal_relative_runtime_chart",
                    "probe_target": np.asarray(phase_target).tolist(),
                    "observed_effect": (observed_position - previous_position).tolist(),
                    "previous_velocity": previous_velocity.tolist(),
                    "residual_trusted": bool(local_action_effect_model.trusted),
                })
                executed += 1
            except ValueError as exc:
                active_probe_trace.append({
                    "phase_reprobe": True,
                    "probe_index": int(probe_index),
                    "action": probe_action.tolist(),
                    "probe_target": np.asarray(phase_target).tolist(),
                    "error": str(exc),
                })
                break
        phase_reprobe_blocks += executed
        return executed

    planning_start_state = runtime.state.copy()
    trace = []
    detector_failures = 0
    previous_waypoint: np.ndarray | None = None
    for block_index in range(max_blocks):
        try:
            action, decision = choose_action(
                method, model, readout, current_rgb, goal_rgb, context_images,
                context_proprios, calibration, action_mean, action_std,
                directions, magnitudes, dynamics, maze_map, planner_map, kp, kv,
                wm_dynamics_tolerance, rng, projection_min_alpha, runtime=runtime,
                oracle_depth=oracle_depth, oracle_beam_width=oracle_beam_width,
                effect_policy=effect_policy,
                local_effect_residual=local_effect_residual,
                local_action_effect_model=local_action_effect_model,
                effect_regret_tolerance=effect_regret_tolerance,
                effect_visual_tolerance=effect_visual_tolerance,
                gdd_adapter=gdd_adapter,
                gdd_projection=gdd_projection,
                gdd_goal_anchor=gdd_goal_anchor,
                gdd_score_horizons=gdd_score_horizons,
                gdd_candidate_count=gdd_candidate_count,
            )
        except ValueError as exc:
            detector_failures += 1
            trace.append({"block": block_index, "error": str(exc)})
            break
        action_gate = decision.get("action_transfer_gate", {})
        active_waypoint = decision.get("active_waypoint")
        phase_changed = bool(
            method in {
                "gdd_wm_rgb_disagreement_verified_local",
                "effect_adapter_rgb_online_local_phase",
            }
            and previous_waypoint is not None
            and active_waypoint is not None
            and not np.allclose(
                np.asarray(active_waypoint, dtype=np.float64),
                previous_waypoint,
                atol=1.0e-8,
                rtol=0.0,
            )
        )
        if (
            phase_changed
            and method == "effect_adapter_rgb_online_local_phase"
            and local_action_effect_model is not None
        ):
            # Re-identify, rather than transport, the local chart into a new
            # route phase. This is the phase-reset intervention under test.
            reprobe_count = remeasure_phase_chart(np.asarray(active_waypoint))
            action, decision = choose_action(
                method, model, readout, current_rgb, goal_rgb,
                context_images, context_proprios, calibration, action_mean,
                action_std, directions, magnitudes, dynamics, maze_map,
                planner_map, kp, kv, wm_dynamics_tolerance, rng,
                projection_min_alpha, runtime=runtime,
                oracle_depth=oracle_depth, oracle_beam_width=oracle_beam_width,
                effect_policy=effect_policy,
                local_effect_residual=local_effect_residual,
                local_action_effect_model=local_action_effect_model,
                effect_regret_tolerance=effect_regret_tolerance,
                effect_visual_tolerance=effect_visual_tolerance,
                gdd_adapter=gdd_adapter, gdd_projection=gdd_projection,
                gdd_goal_anchor=gdd_goal_anchor,
                gdd_score_horizons=gdd_score_horizons,
                gdd_candidate_count=gdd_candidate_count,
            )
            action_gate = decision.get("action_transfer_gate", {})
            action_gate.update({
                "route_phase_changed": True,
                "phase_chart_reset": True,
                "phase_reprobe_blocks": int(reprobe_count),
            })
            decision["selected_action"] = action.tolist()
        elif phase_changed and local_action_effect_model is not None:
            # A local chart identified before a waypoint transition is not
            # assumed valid in the new route phase.  Quarantine it and use the
            # same RGB-only visual fallback for this transition block.
            local_action_effect_model.quarantined = True
            fallback_action, fallback_decision = choose_action(
                "visual_dynamics", model, readout, current_rgb, goal_rgb,
                context_images, context_proprios, calibration, action_mean,
                action_std, directions, magnitudes, dynamics, maze_map,
                planner_map, kp, kv, wm_dynamics_tolerance, rng,
                projection_min_alpha, runtime=runtime,
                oracle_depth=oracle_depth, oracle_beam_width=oracle_beam_width,
            )
            action = fallback_action
            action_gate.update({
                "fallback_used": True,
                "fallback_reason": "route_waypoint_changed_local_chart_quarantined",
                "fallback_action_family": "visual_dynamics_fixed_bank",
                "route_phase_changed": True,
                "route_phase_fallback_selected_index": fallback_decision.get("selected_index"),
            })
            decision["selected_action"] = action.tolist()
            decision["selected_index"] = fallback_decision.get("selected_index")
        if active_waypoint is not None:
            previous_waypoint = np.asarray(active_waypoint, dtype=np.float64).copy()
        runtime.step_block(action)
        next_rgb = renderer.render(runtime.state)
        try:
            next_center = green_center(next_rgb)
            previous_position = calibration.position(center)
            observed_position = calibration.position(next_center)
            if local_effect_residual is not None:
                gate = decision.get("action_transfer_gate", {})
                selected_effect = gate.get("predicted_effect")
                selected_is_effect = (
                    gate.get("effect_selected_index") == decision.get("selected_index")
                    and selected_effect is not None
                )
                if selected_is_effect:
                    local_effect_residual.update_verified_execution(
                        action,
                        observed_position - previous_position,
                        velocity=np.asarray(context_proprios[-1], dtype=np.float64)[2:4],
                        predicted_effect=np.asarray(selected_effect, dtype=np.float64),
                        verification_baseline_effect=dynamics.predict_position_delta(
                            np.asarray(context_proprios[-1], dtype=np.float64)[2:4],
                            np.zeros(2, dtype=np.float64),
                        ),
                    )
                # Only a pulse actually selected from the frozen effect policy
                # may update its residual model.  A visual-dynamics fallback
                # has no frozen-effect prediction to subtract and must not
                # contaminate the effect adapter with a full displacement.
                gate["online_residual_trusted_after_observation"] = local_effect_residual.trusted
            if (
                local_action_effect_model is not None
                    and not (
                    method in {
                        "gdd_wm_rgb_verified_local",
                        "gdd_wm_rgb_disagreement_verified_local",
                        "effect_adapter_rgb_online_local_direct",
                        "effect_adapter_rgb_online_local_phase",
                    }
                    and bool(decision.get("action_transfer_gate", {}).get("fallback_used"))
                )
            ):
                local_gate = decision.get("action_transfer_gate", {})
                local_prediction = local_gate.get("predicted_effect")
                local_is_released = (
                    method in {
                        "gdd_wm_rgb_disagreement_verified_local",
                        "effect_adapter_rgb_online_local_direct",
                        "effect_adapter_rgb_online_local_phase",
                    }
                    and not bool(local_gate.get("fallback_used"))
                    and local_prediction is not None
                )
                local_execution_verified = True
                if local_is_released:
                    local_execution_verified = local_action_effect_model.verify_execution(
                        np.asarray(local_prediction, dtype=np.float64),
                        observed_position - previous_position,
                    )
                    local_gate["post_action_effect_verified"] = bool(local_execution_verified)
                    local_gate["post_action_effect_residual"] = float(
                        local_action_effect_model.verification_errors[-1]
                    )
                    local_gate["local_chart_quarantined"] = bool(
                        local_action_effect_model.quarantined
                    )
                if local_execution_verified:
                    local_action_effect_model.update(
                        action,
                        observed_position - previous_position,
                    )
            delta = (next_center[0] - center[0], next_center[1] - center[1])
            next_proprio = np.concatenate([calibration.position(next_center), calibration.velocity(delta)])
            center = next_center
        except ValueError:
            detector_failures += 1
            next_proprio = context_proprios[-1].copy()
        context_images = (context_images + [next_rgb])[-required_context_windows:]
        context_proprios = (context_proprios + [next_proprio])[-required_context_windows:]
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
        "method": requested_method,
        "gdd_authorization_fallback": gdd_authorization_fallback,
        "initial_state_for_environment": reset_state.tolist(),
        "planning_start_state_for_evaluation": planning_start_state.tolist(),
        "context_windows": int(required_context_windows),
        "warmup_blocks": int(warmup_blocks),
        # A skipped-probe diagnostic is intentionally present in the trace,
        # but it is not an executed environment action and must not be charged
        # to the information-action budget.
        "active_probe_blocks": int(sum(
            "probe_index" in item for item in active_probe_trace
        )),
        "phase_reprobe_blocks": int(phase_reprobe_blocks),
        "probe_triggered": probe_triggered_flag,
        "probe_disagreement_cosine": next(
            (
                float(item["proposal_cosine"])
                for item in active_probe_trace
                if "proposal_cosine" in item
            ),
            None,
        ),
        "active_probe_trace": active_probe_trace,
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
    post_verification_rows = [
        row for row in rows
        if any(
            "post_action_effect_verified" in trace.get("action_transfer_gate", {})
            for trace in row.get("trace", [])
        )
    ]
    post_verification_checks = [
        trace.get("action_transfer_gate", {})
        for row in rows
        for trace in row.get("trace", [])
        if "post_action_effect_verified" in trace.get("action_transfer_gate", {})
    ]
    phase_fallback_calls = sum(
        bool(trace.get("action_transfer_gate", {}).get("route_phase_changed"))
        for row in rows
        for trace in row.get("trace", [])
    )
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
        "mean_progress": float(np.mean([row["progress"] for row in rows])),
        "mean_probe_progress": float(np.mean([row["probe_progress"] for row in rows])),
        "mean_active_probe_blocks": float(np.mean([row["active_probe_blocks"] for row in rows])),
        "mean_phase_reprobe_blocks": float(np.mean([row.get("phase_reprobe_blocks", 0) for row in rows])),
        "probe_trigger_rate": float(np.mean([
            bool(row.get("probe_triggered")) for row in rows
            if row.get("probe_triggered") is not None
        ])) if any(row.get("probe_triggered") is not None for row in rows) else None,
        "post_action_verification_rate": (
            float(np.mean([
                bool(item["post_action_effect_verified"])
                for item in post_verification_checks
            ]))
            if post_verification_checks else None
        ),
        "chart_quarantine_rate": (
            float(np.mean([
                any(
                    bool(trace.get("action_transfer_gate", {}).get("local_chart_quarantined"))
                    for trace in row.get("trace", [])
                )
                for row in post_verification_rows
            ]))
            if post_verification_rows else None
        ),
        "route_phase_fallback_calls": int(phase_fallback_calls),
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


def authorize_gdd_teacher(
    teacher_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    *,
    minimum_release_rate: float = 0.0,
) -> dict[str, Any]:
    """Freeze a conservative, outcome-free-at-test GDD deployment decision.

    Calibration outcomes are allowed here because these roots are explicitly
    withheld from the final consequence.  Requiring both no success loss and
    no final-distance degradation prevents a teacher from being authorized by
    a single coarse binary metric while worsening continuous control quality.
    """
    if not teacher_rows or not visual_rows:
        raise ValueError("GDD authorization requires non-empty calibration rows")
    if not 0.0 <= float(minimum_release_rate) <= 1.0:
        raise ValueError("minimum_release_rate must lie in [0, 1]")
    teacher_success = float(np.mean([bool(row["success"]) for row in teacher_rows]))
    visual_success = float(np.mean([bool(row["success"]) for row in visual_rows]))
    teacher_distance = float(np.mean([float(row["final_distance"]) for row in teacher_rows]))
    visual_distance = float(np.mean([float(row["final_distance"]) for row in visual_rows]))
    release_rates: list[float] = []
    for row in teacher_rows:
        aggregate = row.get("aggregate")
        if isinstance(aggregate, dict) and "gate_fallback_rate" in aggregate:
            release_rates.append(1.0 - float(aggregate["gate_fallback_rate"]))
            continue
        trace = row.get("trace", [])
        gates = [
            item.get("action_transfer_gate", {})
            for item in trace
            if isinstance(item, dict) and isinstance(item.get("action_transfer_gate"), dict)
        ]
        if gates:
            release_rates.append(float(np.mean([
                not bool(gate.get("fallback_used", False)) for gate in gates
            ])))
    teacher_release_rate = float(np.mean(release_rates)) if release_rates else None
    release_gate_passed = (
        True if minimum_release_rate == 0.0 else bool(
            teacher_release_rate is not None
            and teacher_release_rate + 1.0e-12 >= float(minimum_release_rate)
        )
    )
    authorized = bool(
        teacher_success >= visual_success
        and teacher_distance <= visual_distance + 1.0e-12
        and release_gate_passed
    )
    return {
        "authorized": authorized,
        "rule": (
            "teacher success >= visual success AND teacher final distance <= visual final distance"
            " AND teacher release rate >= minimum_release_rate"
        ),
        "teacher_success_rate": teacher_success,
        "visual_success_rate": visual_success,
        "teacher_mean_final_distance": teacher_distance,
        "visual_mean_final_distance": visual_distance,
        "success_delta_teacher_minus_visual": teacher_success - visual_success,
        "final_distance_delta_teacher_minus_visual": teacher_distance - visual_distance,
        "minimum_release_rate": float(minimum_release_rate),
        "teacher_release_rate": teacher_release_rate,
        "release_rate_gate_passed": release_gate_passed,
        "n_teacher": len(teacher_rows),
        "n_visual": len(visual_rows),
        "outcome_used_for_test_selection": False,
    }


def main() -> None:
    args = parse_args()
    args.output = validate_output_path(args.output)
    if not 0.0 <= float(args.projection_min_alpha) <= 1.0:
        raise ValueError("projection-min-alpha must be in [0, 1]")
    if float(args.effect_regret_tolerance) < 0.0:
        raise ValueError("effect-regret-tolerance must be non-negative")
    args.gdd_score_horizons = tuple(int(horizon) for horizon in args.gdd_score_horizons)
    if not args.gdd_score_horizons or any(horizon not in HORIZONS for horizon in args.gdd_score_horizons):
        raise ValueError(
            f"gdd-score-horizons must be a non-empty subset of {HORIZONS}, "
            f"got {args.gdd_score_horizons}"
        )
    max_candidate_count = 32 if "gdd_wm_rgb_support_matched_guarded" in args.methods else 16
    if not 4 <= int(args.gdd_candidate_count) <= max_candidate_count:
        raise ValueError(f"gdd-candidate-count must lie in [4, {max_candidate_count}]")
    device = require_cuda()
    dataset_root = args.dataset_root.resolve()
    calibration_episodes = parse_episode_list(args.calibration_episodes, "calibration episodes")
    test_episodes = parse_episode_list(args.test_episodes, "test episodes")
    gate_calibration_episodes = parse_episode_list(
        args.gate_calibration_episodes, "gate calibration episodes",
    )
    gdd_calibration_episodes = parse_episode_list(
        args.gdd_calibration_episodes, "GDD calibration episodes",
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
    if gdd_calibration_episodes is not None:
        gdd_overlap = (
            set(gdd_calibration_episodes) & set(calibration_episodes)
        ) | (set(gdd_calibration_episodes) & set(test_episodes))
        if gate_calibration_episodes is not None:
            gdd_overlap |= set(gdd_calibration_episodes) & set(gate_calibration_episodes)
        if gdd_overlap:
            raise ValueError(
                f"GDD calibration overlaps another split: {sorted(gdd_overlap)}"
            )
    for episode in calibration_episodes:
        _episode_path(dataset_root, episode)
    state_episode_count = int(torch.load(
        dataset_root / "states.pth", map_location="cpu", weights_only=True
    ).shape[0])
    for episode in test_episodes + (gate_calibration_episodes or []) + (gdd_calibration_episodes or []):
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
    effect_methods = {
        "effect_adapter_rgb", "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_htsc_consistency", "effect_adapter_rgb_online",
        "effect_adapter_rgb_online_htsc", "effect_adapter_rgb_online_probe",
    }
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
        if requested_effect_methods:
            validate_effect_policy_execution_contract(effect_policy)
    gdd_adapter: tuple[Any, dict[str, Any]] | None = None
    gdd_projection: np.ndarray | None = None
    gdd_goal_anchor: np.ndarray | None = None
    gdd_projection_methods = {
        "gdd_wm_rgb", "gdd_wm_rgb_guarded", "gdd_wm_rgb_verified_local",
        "gdd_wm_rgb_disagreement_verified_local", "gdd_wm_rgb_projected_guarded",
        "gdd_wm_rgb_waypoint",
        "gdd_wm_rgb_waypoint_guarded",
        "gdd_wm_rgb_support_preserving_guarded",
        "gdd_wm_rgb_support_matched_guarded",
    }
    runtime_task_effect_methods = {"gdd_runtime_task_effect_rgb"}
    if gdd_projection_methods & set(args.methods):
        if not (args.gdd_adapter_checkpoint and args.gdd_projection and args.gdd_feature_table):
            raise ValueError("GDD projection methods require --gdd-adapter-checkpoint, --gdd-projection, and --gdd-feature-table")
    if runtime_task_effect_methods & set(args.methods) and args.gdd_adapter_checkpoint is None:
        raise ValueError("gdd_runtime_task_effect_rgb requires --gdd-adapter-checkpoint")
    if (gdd_projection_methods | runtime_task_effect_methods) & set(args.methods):
        from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _load_adapter
        gdd_adapter = _load_adapter(args.gdd_adapter_checkpoint.resolve(), str(device))
        if gdd_projection_methods & set(args.methods):
            projection_payload = json.loads(args.gdd_projection.read_text(encoding="utf-8"))
            gdd_projection = np.asarray(projection_payload["projection"], dtype=np.float64)
            with np.load(args.gdd_feature_table, allow_pickle=False) as table:
                goal_features = np.asarray(table["goal_features"], dtype=np.float64)
            if goal_features.ndim != 2 or len(goal_features) == 0:
                raise ValueError("GDD feature table has no goal anchor")
            gdd_goal_anchor = goal_features[0]
            if gdd_projection.ndim != 2 or gdd_projection.shape[1] != len(gdd_goal_anchor):
                raise ValueError("GDD projection and goal anchor dimensions disagree")
            print("[PointMaze live] loaded frozen GDD projection student; runtime teacher/rank inputs disabled")
        else:
            print("[PointMaze live] loaded frozen runtime task-effect adapter; teacher/rank inputs disabled")
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
                effect_visual_tolerance=args.effect_visual_tolerance,
                gdd_adapter=gdd_adapter,
                gdd_projection=gdd_projection,
                gdd_goal_anchor=gdd_goal_anchor,
                gdd_score_horizons=args.gdd_score_horizons,
                gdd_candidate_count=args.gdd_candidate_count,
                active_probe_directions=args.active_probe_directions,
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
    gdd_authorization: dict[str, bool] = {
        method: True for method in args.methods
    }
    gdd_authorization_report: dict[str, Any] | None = None
    requested_gdd_methods = sorted(
        set(args.methods) & {
            "gdd_wm_rgb", "gdd_wm_rgb_guarded", "gdd_wm_rgb_verified_local",
            "gdd_wm_rgb_disagreement_verified_local", "gdd_wm_rgb_projected_guarded",
            "gdd_wm_rgb_waypoint",
            "gdd_wm_rgb_waypoint_guarded",
            "gdd_wm_rgb_support_preserving_guarded",
            "gdd_wm_rgb_support_matched_guarded",
            "gdd_runtime_task_effect_rgb",
        }
    )
    if gdd_calibration_episodes is not None and requested_gdd_methods:
        teacher_method = requested_gdd_methods[0]
        print(
            "[PointMaze live] authorizing GDD teacher on "
            f"{len(gdd_calibration_episodes)} disjoint calibration-only roots "
            f"using method={teacher_method}"
        )
        teacher_rows = [run_episode(
            teacher_method, episode, dataset_root, calibration, model, readout,
            action_mean, action_std, dynamics, maze_map, args.planner_map,
            args.directions, args.magnitudes, args.probe_blocks, args.kp, args.kv,
            selected_tolerance, args.max_blocks, args.success_radius,
            args.seed + 200000 + index, args.projection_min_alpha,
            args.oracle_depth, args.oracle_beam_width,
            fixed_renderer=test_renderer,
            effect_policy=effect_policy,
            effect_regret_tolerance=args.effect_regret_tolerance,
            effect_visual_tolerance=args.effect_visual_tolerance,
            gdd_adapter=gdd_adapter,
            gdd_projection=gdd_projection,
            gdd_goal_anchor=gdd_goal_anchor,
            gdd_disagreement_cosine_threshold=args.gdd_disagreement_cosine_threshold,
            gdd_score_horizons=args.gdd_score_horizons,
            gdd_candidate_count=args.gdd_candidate_count,
            active_probe_directions=args.active_probe_directions,
        ) for index, episode in enumerate(gdd_calibration_episodes)]
        visual_rows = [run_episode(
            "visual_dynamics", episode, dataset_root, calibration, model, readout,
            action_mean, action_std, dynamics, maze_map, args.planner_map,
            args.directions, args.magnitudes, args.probe_blocks, args.kp, args.kv,
            selected_tolerance, args.max_blocks, args.success_radius,
            args.seed + 200000 + index, args.projection_min_alpha,
            args.oracle_depth, args.oracle_beam_width,
            fixed_renderer=test_renderer,
            effect_policy=effect_policy,
            effect_regret_tolerance=args.effect_regret_tolerance,
            effect_visual_tolerance=args.effect_visual_tolerance,
            gdd_adapter=gdd_adapter,
            gdd_projection=gdd_projection,
            gdd_goal_anchor=gdd_goal_anchor,
            gdd_disagreement_cosine_threshold=args.gdd_disagreement_cosine_threshold,
            gdd_score_horizons=args.gdd_score_horizons,
            gdd_candidate_count=args.gdd_candidate_count,
            active_probe_directions=args.active_probe_directions,
        ) for index, episode in enumerate(gdd_calibration_episodes)]
        minimum_release_rate = (
            GDD_SUPPORT_PRESERVING_MIN_RELEASE_RATE
            if teacher_method in {
                "gdd_wm_rgb_support_preserving_guarded",
                "gdd_wm_rgb_support_matched_guarded",
            }
            else 0.0
        )
        decision = authorize_gdd_teacher(
            teacher_rows,
            visual_rows,
            minimum_release_rate=minimum_release_rate,
        )
        for method in requested_gdd_methods:
            gdd_authorization[method] = bool(decision["authorized"])
        gdd_authorization_report = {
            "episodes": list(gdd_calibration_episodes),
            "teacher_method": teacher_method,
            "decision": decision,
            "teacher_aggregate": _aggregate(teacher_rows),
            "visual_aggregate": _aggregate(visual_rows),
            "applied_to_methods": requested_gdd_methods,
        }
        print(
            "[PointMaze live] GDD authorization="
            f"{decision['authorized']} (fail-closed on rejection)"
        )
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
                effect_visual_tolerance=args.effect_visual_tolerance,
                gdd_adapter=gdd_adapter,
                gdd_projection=gdd_projection,
                gdd_goal_anchor=gdd_goal_anchor,
                gdd_disagreement_cosine_threshold=args.gdd_disagreement_cosine_threshold,
                gdd_score_horizons=args.gdd_score_horizons,
                gdd_candidate_count=args.gdd_candidate_count,
                active_probe_directions=args.active_probe_directions,
                gdd_authorized=gdd_authorization.get(method, True),
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
            "gdd_calibration_episodes": gdd_calibration_episodes,
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
                "executed_action_steps_before_reobservation": FRAME_STRIDE,
                "rgb_velocity_probe_blocks": args.probe_blocks,
                "active_probe_directions": int(args.active_probe_directions),
                "planner_map": args.planner_map,
                "kp": args.kp,
                "kv": args.kv,
                "wm_dynamics_tolerance": args.wm_dynamics_tolerance,
                "selected_wm_dynamics_tolerance": selected_tolerance,
                "projection_min_alpha": args.projection_min_alpha,
                "gdd_disagreement_cosine_threshold": args.gdd_disagreement_cosine_threshold,
                "gdd_score_horizons": list(args.gdd_score_horizons),
                "gdd_candidate_count": int(args.gdd_candidate_count),
                "oracle_depth": args.oracle_depth,
                "oracle_beam_width": args.oracle_beam_width,
            },
            "methods": list(methods),
            "gdd_student": {
                "adapter_checkpoint": None if args.gdd_adapter_checkpoint is None else str(args.gdd_adapter_checkpoint.resolve()),
                "projection": None if args.gdd_projection is None else str(args.gdd_projection.resolve()),
                "feature_table": None if args.gdd_feature_table is None else str(args.gdd_feature_table.resolve()),
                "runtime_teacher_forbidden": True,
                "candidate_id_input_forbidden": True,
                "teacher_rank_input_forbidden": True,
            },
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
            "effect_visual_tolerance": float(args.effect_visual_tolerance),
            "effect_policy_gate_boundary": (
                None if effect_policy is None else
                "exploratory online support-and-bound gate; fixed-bank calibration does "
                "not provide coverage for the state-conditioned repeated-pulse bank"
            ),
            "gate_calibration": gate_calibration_report,
            "gdd_calibration_authorization": gdd_authorization_report,
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
