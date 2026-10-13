"""Train and evaluate a causal RGB-only local-effect adapter for PushT.

The existing visual successor head predicts an absolute next pose.  Its error
therefore mixes three quantities that are different for control: the current
visual pose, passive motion at zero action, and the action-conditioned effect.
This experiment separates them with matched interventions from the same source
root:

    zero_delta(o)   = pose(o, zero) - pose(o)
    action_effect(o,a) = pose(o,a) - pose(o, zero)

The adapter is trained on frozen RGB mask-moment features and predicts these
two local quantities.  At runtime it receives only current/previous RGB frames,
the goal RGB frame, and candidate actions.  Simulator state is used only to
construct offline labels and to score the closed-loop evaluator; no successor
query is used for action selection.

This is a mechanism test, not a claim that the frozen LeWM checkpoint is a
universal world model.  It is deliberately paired with a geometry-servo
baseline and reports candidate-ranking ceilings separately from visual
closed-loop success.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import cv2
import numpy as np
import stable_worldmodel  # noqa: F401
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .run_causal_visual_pusht_world_model import (
        PUSHT_RENDER_SIZE,
        Normalization,
        VisualCausalWorldModel,
        balanced_source_cases,
        bootstrap_closed_loop_source_cases,
        observable_pose_vector,
        raw_from_normalized,
        resize_image,
        state_vector,
        tensor_images,
        terminal_pose_cases,
    )
    from .evaluate_causal_visual_checkpoint import (
        checkpoint_adapter_architecture,
        checkpoint_utility_mode,
    )
except ImportError:
    from run_causal_visual_pusht_world_model import (  # type: ignore
        PUSHT_RENDER_SIZE,
        Normalization,
        VisualCausalWorldModel,
        balanced_source_cases,
        bootstrap_closed_loop_source_cases,
        observable_pose_vector,
        raw_from_normalized,
        resize_image,
        state_vector,
        tensor_images,
        terminal_pose_cases,
    )
    from evaluate_causal_visual_checkpoint import (  # type: ignore
        checkpoint_adapter_architecture,
        checkpoint_utility_mode,
    )


POSE_DIM = 3  # block dx, block dy, wrapped block dtheta
# The renderer is 224x224, but PushT simulator coordinates span a 512x512
# workspace.  Moment means therefore map to simulator coordinates with 512,
# not with the rendered pixel width.  Confusing these two coordinate systems
# was the upstream reason the first RGB-only controller declared contact and
# target proximity at the wrong locations.
PUSHT_STATE_SIZE = 512.0
BLOCK_FEATURE_START = 10
BLOCK_FEATURE_STOP = 20
TARGET_FEATURE_START = 20
TARGET_FEATURE_STOP = 30
RECOVERY_WORKSPACE_MARGIN = 28.0
# The route midpoint is checked against the T collision envelope by the RGB
# motion guard.  A 64 px centerline clearance is preferred, but it is not
# always geometrically realizable near a workspace boundary: the outer point
# can miss the 28 px safety box by only a few pixels.  The anchor search may
# therefore relax to 48 px as a workspace-feasibility fallback.  The runtime
# motion guard remains active, and relaxed routes are recorded in the audit
# rather than being treated as ordinary routes.
RECOVERY_MIN_CLEARANCE = 48.0


def terminal_ranker_score_contract(
    target: str,
    action_horizon: int,
) -> str:
    """Return the only score mode valid for a terminal ranker protocol.

    The terminal ranker is an interface contract, not merely a training
    annotation.  In particular, a pose-labelled repeated-action ranker must
    never be replayed as an angle ranker: the two objectives can select
    actions with very different translation/rotation trade-offs.
    """
    normalized_target = str(target).strip().lower()
    horizon = int(action_horizon)
    if normalized_target not in {"angle", "pose"}:
        raise ValueError(
            f"terminal_ranker_target must be 'angle' or 'pose', got {target!r}"
        )
    if horizon < 1:
        raise ValueError("terminal_action_horizon must be positive")
    if horizon == 1:
        if normalized_target == "pose":
            raise ValueError(
                "pose terminal rankers require a repeated-action horizon; "
                "one-step pose ranking is not implemented by this protocol"
            )
        return "one_step_residual"
    return f"direct_multistep_{normalized_target}"


def validate_terminal_ranker_contract(
    training: dict[str, Any],
    args: argparse.Namespace,
    *,
    report_metrics: dict[str, Any] | None = None,
) -> str:
    """Fail closed when a serialized ranker and replay protocol disagree."""
    target = str(getattr(args, "terminal_ranker_target", "angle"))
    horizon = int(getattr(args, "terminal_action_horizon", 1))
    expected_mode = terminal_ranker_score_contract(target, horizon)
    observed_mode = str(training.get("score_mode", ""))
    observed_horizon = int(training.get("action_horizon", -1))
    if observed_mode != expected_mode or observed_horizon != horizon:
        raise RuntimeError(
            "terminal ranker contract mismatch: "
            f"protocol(target={target!r}, horizon={horizon}, "
            f"score_mode={expected_mode!r}) != "
            f"artifact(score_mode={observed_mode!r}, "
            f"action_horizon={observed_horizon})"
        )
    if report_metrics is not None:
        report_target = report_metrics.get("target")
        report_mode = report_metrics.get("score_mode")
        report_horizon = report_metrics.get("action_horizon")
        if report_target is not None and str(report_target) != target:
            raise RuntimeError(
                "terminal ranker report target disagrees with protocol: "
                f"{report_target!r} != {target!r}"
            )
        if report_mode is not None and str(report_mode) != expected_mode:
            raise RuntimeError(
                "terminal ranker report score_mode disagrees with protocol: "
                f"{report_mode!r} != {expected_mode!r}"
            )
        if report_horizon is not None and int(report_horizon) != horizon:
            raise RuntimeError(
                "terminal ranker report action_horizon disagrees with protocol: "
                f"{report_horizon!r} != {horizon!r}"
            )
    return expected_mode


def terminal_control_lease_active(
    terminal_candidate: bool,
    terminal_fan: bool,
    candidate_contact_band: bool,
    lease_gap_max: float,
    lease_valid: bool,
) -> bool:
    """Return whether the small terminal-control interface is still valid.

    A support/contact lease alone does not authorize a terminal ranker: the
    block must also remain inside the visual terminal fan.  Otherwise the
    candidate set has already switched back to broad translation actions and
    applying an angular endpoint ranker becomes an objective/interface
    mismatch.  Keeping this predicate pure makes the state-machine boundary
    directly testable.
    """
    return bool(
        terminal_candidate
        and terminal_fan
        and candidate_contact_band
        and (float(lease_gap_max) <= 0.0 or lease_valid)
    )


def terminal_action_commitment_eligible(
    phase: str,
    action: np.ndarray,
    info: dict[str, Any],
) -> bool:
    """Return whether a selected action belongs to the terminal torque lease.

    Commitment is a control-interface property, not a property of one
    particular score head.  The previous implementation armed it only for
    ``terminal_angular_ranker_used``.  That silently disabled commitment for
    the RGB torque-prior and contact-safety fallback paths, even though those
    paths return the same repeated-action terminal intervention.  Keep the
    eligibility predicate explicit and provenance-based so replay can audit
    which terminal interface actually received commitment.

    Safety/recovery decisions are intentionally excluded: the outer loop
    cancels a lease when they request damping or pulse freezing.  This helper
    never consults simulator state or a successor observation.
    """
    if phase not in {"torque", "hold"}:
        return False
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.ndim != 1 or action_array.size != 2:
        return False
    if not bool(np.isfinite(action_array).all()) or float(np.linalg.norm(action_array)) <= 1.0e-6:
        return False
    provenance_keys = (
        "terminal_angular_ranker_used",
        "terminal_torque_prior_used",
        "terminal_support_fallback_used",
        "terminal_contact_safety_used",
    )
    return any(float(info.get(key, 0.0)) > 0.0 for key in provenance_keys)


def runtime_current_xy_policy(args: argparse.Namespace) -> str:
    """Return the observer actually authorized for closed-loop execution.

    ``current_xy_policy`` is the requested experiment protocol.  A readout
    can nevertheless fail its root-disjoint validation gate, in which case
    the runtime observer must fail closed to the selected validated fallback.
    Keeping this decision in a separate field prevents an ablation from
    accidentally consuming an unselected observer while its artifact claims
    that the gate rejected it.
    """
    return str(
        getattr(
            args,
            "_runtime_current_xy_policy",
            getattr(args, "current_xy_policy", "validation"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--train-roots", type=int, default=480)
    parser.add_argument("--val-roots", type=int, default=120)
    parser.add_argument("--root-rollout-steps", type=int, default=8)
    parser.add_argument("--contact-root-fraction", type=float, default=0.60)
    parser.add_argument("--terminal-root-fraction", type=float, default=0.25)
    parser.add_argument(
        "--terminal-position-jitter", type=float, default=16.0,
        help="maximum XY offset of terminal roots from the goal; kept below the 20 px evaluator tube so threshold validation has guaranteed positive cases",
    )
    parser.add_argument(
        "--terminal-root-support-gap", type=float, default=0.0,
        help=(
            "optional RGB-only support-point gap threshold used while constructing "
            "terminal train/validation/planning roots; zero preserves the legacy "
            "block-goal radial root protocol"
        ),
    )
    parser.add_argument(
        "--terminal-root-support-angle-samples", type=int, default=32,
        help="number of RGB contact directions searched per terminal root",
    )
    parser.add_argument(
        "--bootstrap-checkpoint", type=Path, default=None,
        help="immutable RGB-only controller used only to collect closed-loop visitation roots for offline labels",
    )
    parser.add_argument(
        "--bootstrap-root-fraction", type=float, default=0.0,
        help="fraction of train/validation roots collected from frozen-controller closed-loop visitation",
    )
    parser.add_argument("--bootstrap-rollout-episodes", type=int, default=80)
    parser.add_argument("--bootstrap-rollout-steps", type=int, default=100)
    parser.add_argument("--bootstrap-contact-distance", type=float, default=70.0)
    parser.add_argument("--contact-approach-steps", type=int, default=48)
    parser.add_argument("--action-directions", type=int, default=16)
    parser.add_argument("--action-magnitudes", type=float, nargs="+", default=[0.35, 0.60, 0.85])
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--goal-epochs", type=int, default=200)
    parser.add_argument("--contact-loss-weight", type=float, default=3.0,
                        help="training weight for contact-root interventions; fixed before held-out evaluation")
    parser.add_argument(
        "--terminal-loss-weight", type=float, default=1.0,
        help=(
            "additional action-effect loss weight for root-disjoint terminal "
            "interventions; one preserves the legacy protocol"
        ),
    )
    parser.add_argument("--planning-episodes", type=int, default=32)
    parser.add_argument("--planning-steps", type=int, default=100)
    parser.add_argument(
        "--planning-start", choices=("random", "terminal"), default="random",
        help="evaluation start distribution; terminal uses root-disjoint near-goal pose roots for closure-only evaluation",
    )
    parser.add_argument("--planning-controller", choices=("adapter", "hybrid"), default="adapter",
                        help="adapter uses learned effects throughout; hybrid reserves learned effects for terminal torque/hold and uses RGB geometry for push/translation")
    parser.add_argument("--contact-distance", type=float, default=42.0)
    parser.add_argument("--contact-tolerance", type=float, default=24.0)
    parser.add_argument(
        "--contact-side-alignment-min", type=float, default=0.35,
        help="minimum RGB cosine alignment between block-to-agent contact normal and block-to-goal direction before learned push handoff",
    )
    parser.add_argument(
        "--contact-side-misalignment-patience", type=int, default=1,
        help="consecutive RGB observations below contact-side alignment required before recovery; one preserves the legacy one-frame gate",
    )
    parser.add_argument(
        "--contact-recovery-retreat-magnitude", type=float, default=0.45,
        help="RGB-only retreat action used for one-step contact-side recovery before re-approaching the desired pushing side",
    )
    parser.add_argument(
        "--recovery-action-max-magnitude", type=float, default=0.0,
        help=(
            "optional true L2 cap for ordinary RGB recovery actions; zero "
            "preserves the legacy component-wise recovery protocol"
        ),
    )
    parser.add_argument("--approach-action-scale", type=float, default=100.0,
                        help="PushT action_scale used to convert RGB approach displacement into an action")
    parser.add_argument(
        "--contact-recovery-distance", type=float, default=100.0,
        help="RGB distance above which a learned contact episode is forced into recovery",
    )
    parser.add_argument(
        "--reacquisition-clearance", type=float, default=100.0,
        help="minimum RGB agent-block distance reached during side reacquisition before approaching the declared target contact point",
    )
    parser.add_argument(
        "--reacquisition-action-max-magnitude", type=float, default=0.35,
        help="maximum per-decision RGB action magnitude during side reacquisition; limits unintended block pushes while routing around the object",
    )
    parser.add_argument(
        "--reacquisition-target-action-max-magnitude", type=float, default=0.10,
        help=(
            "smaller L2 cap for the final target-contact approach; the outer "
            "route may use the regular reacquisition cap, but the last contact "
            "segment must not drive through the block"
        ),
    )
    parser.add_argument(
        "--reacquisition-motion-limit", type=float, default=3.0,
        help="RGB block-motion threshold that aborts target-side reacquisition and re-clears contact",
    )
    parser.add_argument(
        "--reacquisition-motion-patience", type=int, default=2,
        help="number of consecutive RGB motion-limit violations required before aborting a reacquisition route",
    )
    parser.add_argument(
        "--reacquisition-target-motion-patience", type=int, default=1,
        help="motion-limit patience for the final target-contact segment; one is fail-closed",
    )
    parser.add_argument(
        "--reacquisition-anchor-drift-limit", type=float, default=12.0,
        help="maximum cumulative RGB block displacement from the frozen recovery anchor before aborting the route",
    )
    parser.add_argument(
        "--reacquisition-post-success-translation-cap", type=float, default=0.0,
        help="optional translation-action L2 cap for the short RGB stabilization lease after reacquisition success; zero disables it",
    )
    parser.add_argument(
        "--reacquisition-phase-budget", type=int, default=36,
        help="maximum decisions spent in one side-reacquisition phase before releasing the anchor",
    )
    parser.add_argument(
        "--reacquisition-total-budget", type=int, default=72,
        help="maximum decisions spent in all side-reacquisition phases per episode",
    )
    parser.add_argument(
        "--reacquisition-max-retries", type=int, default=1,
        help=(
            "maximum number of failed side-reacquisition routes allowed per "
            "episode before fail-closed hold; prevents recovery/reacquisition loops"
        ),
    )
    parser.add_argument(
        "--reacquisition-lockout-steps", type=int, default=12,
        help=(
            "audit lease length after a failed side-reacquisition route; "
            "during the lease no new route may be started"
        ),
    )
    parser.add_argument(
        "--reacquisition-failure-translation-cap", type=float, default=0.10,
        help=(
            "one-shot L2 cap for a local terminal approach after a failed "
            "route when the current RGB side is already valid; zero disables"
        ),
    )
    parser.add_argument(
        "--terminal-side-reacquisition", action=argparse.BooleanOptionalAction, default=False,
        help="enable the experimental clear-waypoint-arc-target side-reacquisition state machine; disabled preserves the legacy recovery protocol",
    )
    parser.add_argument(
        "--persistent-recovery-anchor", action=argparse.BooleanOptionalAction, default=True,
        help="freeze the RGB block/goal geometry at recovery onset until side reacquisition succeeds; disabling reproduces drifting per-frame recovery targets",
    )
    parser.add_argument(
        "--contact-lock-distance", type=float, default=100.0,
        help="hysteresis radius: after contact handoff, keep the learned candidate support capped until this RGB agent-block distance",
    )
    parser.add_argument("--contact-recovery-stagnation", type=float, default=3.0,
                        help="maximum observed RGB block displacement that permits contact recovery")
    parser.add_argument(
        "--side-recovery-mode", choices=("contact_target", "retreat"), default="contact_target",
        help="RGB-only wrong-side recovery primitive; contact_target servos toward the declared desired contact point, retreat reproduces the legacy radial retreat",
    )
    parser.add_argument("--terminal-xy", type=float, default=35.0)
    parser.add_argument(
        "--terminal-repair-xy", type=float, default=-1.0,
        help="optional wider XY radius for an already-active terminal torque lease; negative reuses terminal-xy",
    )
    parser.add_argument(
        "--terminal-torque-escape-limit", type=float, default=-1.0,
        help=(
            "fixed maximum observed XY error for an active terminal torque lease; "
            "negative reuses --terminal-repair-xy. This prevents a terminal lease "
            "from expanding with its own predicted drift."
        ),
    )
    parser.add_argument(
        "--terminal-threshold-policy", choices=("validation", "fixed"), default="validation",
        help="select the visual torque-entry XY threshold on validation roots or keep the declared --terminal-xy value",
    )
    parser.add_argument("--terminal-angle", type=float, default=0.28,
                        help="visual angle tube in which a zero-action hold is allowed")
    parser.add_argument("--terminal-hold-xy", type=float, default=18.0,
                        help="narrower visual XY tube required before zero-action hold")
    parser.add_argument(
        "--terminal-confirmation-steps", type=int, default=2,
        help="consecutive RGB observations required before entering the terminal/hold interface",
    )
    parser.add_argument(
        "--terminal-max-observer-jump", type=float, default=8.0,
        help="maximum one-step RGB block displacement accepted while confirming terminal proximity",
    )
    parser.add_argument(
        "--terminal-contact-gap-max", type=float, default=0.0,
        help=(
            "maximum RGB boundary-support to agent gap in pixels before terminal torque is allowed; "
            "zero disables this new contact-feasibility gate for legacy protocols"
        ),
    )
    parser.add_argument(
        "--terminal-contact-target", choices=("boundary", "centroid"), default="boundary",
        help=(
            "RGB terminal-contact acquisition target. 'boundary' preserves the "
            "support-point chart protocol; 'centroid' approaches the declared "
            "goal-facing centroid contact and uses the boundary chart only as "
            "a diagnostic. The latter is an explicit intervention for false "
            "support-point contacts on asymmetric T shapes."
        ),
    )
    parser.add_argument(
        "--terminal-agent-distance-max", type=float, default=0.0,
        help=(
            "maximum RGB agent-centroid to block-centroid distance before terminal torque is allowed; "
            "zero disables this independent center-distance gate"
        ),
    )
    parser.add_argument(
        "--terminal-approach-action-max-magnitude", type=float, default=-1.0,
        help=(
            "optional L2 cap for RGB terminal-contact approach actions; "
            "negative preserves the legacy component-wise clip"
        ),
    )
    parser.add_argument(
        "--terminal-stagnation-watchdog", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "after repeated terminal actions with no observed RGB block motion, "
            "leave torque and reacquire the contact manifold; disabled preserves legacy behavior"
        ),
    )
    parser.add_argument(
        "--terminal-stagnation-patience", type=int, default=2,
        help="number of consecutive low-motion terminal observations before the watchdog fires",
    )
    parser.add_argument(
        "--terminal-stagnation-motion", type=float, default=1.5,
        help="maximum observed RGB block displacement treated as terminal stagnation",
    )
    parser.add_argument(
        "--terminal-contact-lease-gap-max", type=float, default=0.0,
        help=(
            "maximum RGB boundary-support gap, in pixels, while a torque/hold phase is active; "
            "a positive value fail-closes to recovery when the terminal contact lease is lost; "
            "zero preserves the legacy phase protocol"
        ),
    )
    parser.add_argument(
        "--support-contact-side-alignment-min", type=float, default=None,
        help=(
            "minimum RGB support-lever alignment used by the early contact lease; "
            "None reuses --contact-side-alignment-min and keeps the lease fail-closed"
        ),
    )
    parser.add_argument(
        "--support-contact-side-agnostic-terminal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "for terminal-root closure only, accept a valid RGB support gap "
            "without using the occluded centroid-derived goal-side cosine; "
            "the support normal becomes the effective local contact direction"
        ),
    )
    parser.add_argument(
        "--terminal-proximity-lease-fallback", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "when terminal RGB centroid distance and side alignment certify near-contact, "
            "allow torque without chasing a transient boundary-gap estimate; disabled preserves "
            "the historical boundary-only lease"
        ),
    )
    parser.add_argument(
        "--translation-contact-gap-max", type=float, default=0.0,
        help=(
            "optional RGB boundary-support gap gate for non-terminal push/translation; "
            "positive values fail closed to recovery when visible support is farther "
            "than this many pixels"
        ),
    )
    parser.add_argument(
        "--reacquisition-progress-steps", type=int, default=8,
        help="after RGB-confirmed reacquisition, temporarily disallow the zero-action deadlock for this many decisions",
    )
    parser.add_argument("--terminal-hold", action=argparse.BooleanOptionalAction, default=True,
                        help="hold with zero action once the visual terminal tube is satisfied")
    parser.add_argument(
        "--terminal-xy-observer", choices=("runtime", "decoder"), default="runtime",
        help="observer used only for terminal XY gating; decoder keeps RGB contact control while avoiding occlusion-induced terminal false positives",
    )
    parser.add_argument("--current-xy-min-relative-gain", type=float, default=0.20,
                        help="minimum validation relative error reduction required before using the learned current-XY decoder; otherwise use the raw RGB moment chart")
    parser.add_argument(
        "--agent-xy-max-error", type=float, default=12.0,
        help="absolute held-out pixel-error ceiling for the paired agent XY decoder; the agent chart need not beat an already accurate moment readout",
    )
    parser.add_argument(
        "--rgb-agent-occlusion-discovery", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "when the RGB agent mask is absent, emit bounded away-from-block "
            "discovery pulses before allowing terminal torque; disabled preserves "
            "the legacy hidden-agent protocol"
        ),
    )
    parser.add_argument(
        "--rgb-agent-occlusion-discovery-magnitude", type=float, default=0.12,
        help="magnitude of each RGB-only agent-occlusion discovery pulse",
    )
    parser.add_argument(
        "--rgb-agent-occlusion-discovery-max-pulses", type=int, default=2,
        help="maximum discovery pulses per episode before fail-closed zero/recovery",
    )
    parser.add_argument(
        "--current-xy-policy", choices=("validation", "raw", "disagreement_gate", "rgb_mask", "rgb_mask_fused", "rgb_calibrated", "paired_rgb_calibrated"),
        default="validation",
        help="runtime current-block XY observer: validation-selected decoder, raw RGB moment, RGB mask, RGB mask fused with the learned decoder, root-disjoint calibrated RGB shape, or paired current/goal rigid-chart calibration",
    )
    parser.add_argument(
        "--rgb-current-xy-readout", choices=("linear", "knn"), default="linear",
        help="root-disjoint RGB current-XY calibrator; KNN is support-constrained and intended for dynamic contact occlusion",
    )
    parser.add_argument(
        "--goal-xy-policy", choices=("model", "rgb_mask", "rgb_calibrated"), default="model",
        help="goal block XY observer: frozen/learned latent readout or direct RGB color-mask centroid",
    )
    parser.add_argument(
        "--goal-observation", choices=("centered", "aligned_state_goal"), default="centered",
        help="goal-image protocol; aligned_state_goal renders the target at the evaluator goal pose for a scene-aligned visual interface",
    )
    parser.add_argument(
        "--current-angle-policy", choices=("model", "rgb_shape", "rgb_shape_gated"), default="model",
        help=(
            "current block angle observer: learned readout, legacy raw RGB shape chart, "
            "or RGB shape readout accepted only after the held-out gate"
        ),
    )
    parser.add_argument(
        "--goal-angle-policy", choices=("model", "rgb_shape"), default="model",
        help="goal angle observer: learned goal head or direct RGB target shape chart",
    )
    parser.add_argument(
        "--rgb-angle-features", choices=("moments", "raster"), default="raster",
        help="RGB current-angle readout interface; raster keeps a position-normalized high-resolution block mask to resolve 64x64 moment aliasing",
    )
    parser.add_argument(
        "--rgb-angle-readout", choices=("mlp", "knn"), default="knn",
        help="current-angle raster readout; KNN is support-constrained and MLP is retained as an ablation",
    )
    parser.add_argument(
        "--rgb-angle-temporal-jump", type=float, default=0.75,
        help="maximum one-decision RGB angle jump before using the previous-frame calibrated angle as an occlusion-safe fallback",
    )
    parser.add_argument(
        "--rgb-temporal-angle-max-delta", type=float, default=0.55,
        help="maximum signed orientation increment searched by the RGB temporal silhouette observer",
    )
    parser.add_argument(
        "--rgb-temporal-angle-min-confidence", type=float, default=0.004,
        help="minimum silhouette-registration peak margin required for a temporal angle update",
    )
    parser.add_argument(
        "--rgb-angle-disagreement-policy", choices=("none", "latent_fallback"), default="none",
        help=(
            "cross-observer guard for contact occlusion: when the frozen latent angle and "
            "the RGB angle disagree beyond the declared threshold, use the latent angle; "
            "none preserves the historical RGB-selected protocol"
        ),
    )
    parser.add_argument(
        "--rgb-angle-disagreement-threshold", type=float, default=1.20,
        help="absolute circular disagreement in radians that activates the latent-angle fallback",
    )
    parser.add_argument(
        "--rgb-angle-disagreement-terminal-xy", type=float, default=-1.0,
        help=(
            "if nonnegative, apply the latent fallback only when the RGB block is within "
            "this terminal XY radius; negative preserves the all-phase gate"
        ),
    )
    parser.add_argument(
        "--rgb-goal-angle-max-error", type=float, default=0.50,
        help="fail-closed validation error (radians) required before a target-only RGB shape angle can enter the ranker",
    )
    parser.add_argument(
        "--rgb-goal-xy-max-error", type=float, default=20.0,
        help="fail-closed validation error (pixels) required before calibrated scene-aligned RGB goal XY can enter runtime",
    )
    parser.add_argument(
        "--rgb-current-xy-max-error", type=float, default=35.0,
        help="fail-closed validation error (pixels) required before calibrated current RGB shape XY can enter runtime",
    )
    parser.add_argument(
        "--rgb-current-xy-p95-max-error", type=float, default=60.0,
        help="fail-closed held-out p95 error (pixels) required before calibrated current RGB XY can enter runtime",
    )
    parser.add_argument(
        "--paired-rgb-current-xy-max-error", type=float, default=30.0,
        help="fail-closed held-out mean error for the paired current/goal rigid-chart observer",
    )
    parser.add_argument(
        "--paired-rgb-current-xy-p95-max-error", type=float, default=80.0,
        help="fail-closed held-out p95 error for the paired current/goal rigid-chart observer",
    )
    parser.add_argument(
        "--rgb-current-xy-static-fallback", action=argparse.BooleanOptionalAction, default=False,
        help="allow a root-disjoint static RGB XY calibrator to enter runtime when dynamic augmentation is disabled and its held-out gate passes; opt-in because this is not a dynamic-occlusion qualification",
    )
    parser.add_argument(
        "--rgb-pose-ridge-alpha", type=float, default=1.0,
        help="ridge strength for RGB centroid/pose bias calibration; fixed before held-out gating",
    )
    parser.add_argument(
        "--current-xy-disagreement-threshold", type=float, default=60.0,
        help="pixel threshold for disagreement_gate before falling back to the raw RGB moment chart",
    )
    parser.add_argument(
        "--paired-reference-policy", choices=("raw", "decoder"), default="raw",
        help=(
            "object-identity reference for paired RGB current XY: raw keeps the "
            "legacy centroid-first protocol; decoder uses the held-out-gated learned "
            "current-XY decoder as the primary reference"
        ),
    )
    parser.add_argument(
        "--rgb-max-observed-block-step", type=float, default=48.0,
        help="RGB-only one-step displacement cap used to reject catastrophic contact-occlusion jumps",
    )
    parser.add_argument(
        "--readout-augmentation-steps", type=int, default=0,
        help="number of offline RGB rollout frames per root used to train/validate current pose readouts; zero preserves the static-root protocol",
    )
    parser.add_argument(
        "--rgb-angle-contact-augmentation", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "add a root-disjoint contact-biased RGB rollout to the current-angle "
            "readout; disabled preserves the random dynamic-readout protocol"
        ),
    )
    parser.add_argument(
        "--structured-contact-adapter", action=argparse.BooleanOptionalAction, default=False,
        help="fit a closed-form RGB contact-geometry ridge adapter as a specialist candidate; it enters runtime only after held-out ranking gain",
    )
    parser.add_argument("--small-action-magnitudes", type=float, nargs="+",
                        default=[0.08, 0.16, 0.24, 0.35],
                        help="small candidate magnitudes used only for terminal orientation repair")
    parser.add_argument("--contact-action-max-magnitude", type=float, default=0.35,
                        help="maximum candidate displacement while the RGB agent-block distance is inside the contact band")
    parser.add_argument(
        "--terminal-action-max-magnitude", type=float, default=-1.0,
        help="independent cap for terminal torque candidates; negative reuses contact-action-max-magnitude",
    )
    parser.add_argument(
        "--contact-inward-min", type=float, default=0.0,
        help="minimum RGB-estimated inward action component while in contact band; zero disables the contact-preservation filter",
    )
    parser.add_argument(
        "--contact-translation-inward-min", type=float, default=0.0,
        help="minimum inward component during the translation phase; keeps tangential torque from ejecting the pusher before the terminal manifold",
    )
    parser.add_argument(
        "--contact-translation-tangent-ratio", type=float, default=1.25,
        help="maximum absolute tangential/inward action ratio during translation-phase contact",
    )
    parser.add_argument(
        "--contact-translation-progress-min", type=float, default=0.0,
        help="minimum action projection onto the current RGB block-to-goal direction during translation-phase contact",
    )
    parser.add_argument(
        "--contact-action-projection", action=argparse.BooleanOptionalAction, default=False,
        help="project the final RGB-selected action onto the contact-preserving half-space; opt-in so legacy protocols remain reproducible",
    )
    parser.add_argument(
        "--contact-projection-min-inward", type=float, default=0.08,
        help="minimum RGB-estimated action component toward the block enforced by the optional final action projection",
    )
    parser.add_argument(
        "--learned-candidate-support", choices=("trained", "all"), default="trained",
        help="candidate actions available to the learned selector; trained restricts selection to the declared offline action bank",
    )
    parser.add_argument(
        "--torque-sign-guard", action=argparse.BooleanOptionalAction, default=False,
        help="in terminal contact, reject learned candidates whose RGB geometric tangential torque opposes the angle error",
    )
    parser.add_argument(
        "--terminal-contact-inward-min", type=float, default=0.0,
        help=(
            "minimum RGB-estimated inward action component during terminal torque; "
            "zero preserves the legacy torque-only protocol, while a positive value "
            "requires the selected signed torque to remain contact-feasible"
        ),
    )
    parser.add_argument(
        "--terminal-contact-tangent-ratio", type=float, default=0.0,
        help=(
            "optional terminal feasibility bound on abs(tangent)/inward; zero disables "
            "the ratio while --terminal-contact-inward-min remains active"
        ),
    )
    parser.add_argument(
        "--terminal-local-angle-weight", type=float, default=0.0,
        help=(
            "weight of the RGB-only online local angle-effect score during "
            "terminal action selection; zero preserves the fixed-prior protocol"
        ),
    )
    parser.add_argument(
        "--terminal-angle-progress-guard", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "after a terminal torque pulse worsens the RGB angle error, reverse "
            "its tangential component with a bounded RGB-only corrective pulse"
        ),
    )
    parser.add_argument(
        "--terminal-angle-progress-tolerance", type=float, default=0.02,
        help="minimum RGB angle-error worsening that activates the terminal angle-progress guard",
    )
    parser.add_argument(
        "--online-signed-torque-probe", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "use a bounded RGB-only signed-torque excitation in the terminal "
            "contact manifold; the next-frame angle error validates or flips "
            "the tangent direction before continuing"
        ),
    )
    parser.add_argument(
        "--signed-torque-probe-magnitude", type=float, default=0.16,
        help="tangential magnitude of the RGB signed-torque probe",
    )
    parser.add_argument(
        "--signed-torque-probe-inward", type=float, default=0.08,
        help="inward component of the RGB signed-torque probe",
    )
    parser.add_argument(
        "--signed-torque-probe-max-magnitude", type=float, default=None,
        help=(
            "independent norm cap for the online signed-torque probe; when unset, "
            "the probe inherits the terminal/contact action cap"
        ),
    )
    parser.add_argument(
        "--signed-torque-probe-min-gain", type=float, default=0.02,
        help="minimum observed angle-error reduction required to validate a probe direction",
    )
    parser.add_argument(
        "--signed-torque-probe-max-pulses", type=int, default=3,
        help="maximum exploratory probe pulses before fail-closed recovery",
    )
    parser.add_argument(
        "--signed-torque-probe-max-xy-regression", type=float, default=3.0,
        help="maximum RGB block-to-goal regression permitted while validating a torque probe",
    )
    parser.add_argument(
        "--signed-torque-probe-shrink-factor", type=float, default=0.75,
        help="multiplicative probe-magnitude reduction after an XY-unsafe pulse",
    )
    parser.add_argument(
        "--signed-torque-probe-max-cumulative-xy-drift", type=float, default=3.0,
        help="maximum cumulative RGB terminal-tube drift from the first probe observation",
    )
    parser.add_argument(
        "--signed-torque-probe-xy-repair", action=argparse.BooleanOptionalAction, default=False,
        help="insert a bounded goal-direction repair after a torque pulse causes RGB XY regression",
    )
    parser.add_argument(
        "--signed-torque-probe-xy-repair-threshold", type=float, default=1.0,
        help="minimum RGB one-step XY regression that triggers a probe translation repair",
    )
    parser.add_argument(
        "--signed-torque-probe-xy-repair-magnitude", type=float, default=0.16,
        help="bounded action magnitude of the RGB goal-direction probe repair",
    )
    parser.add_argument(
        "--geometric-torque-fallback", action=argparse.BooleanOptionalAction, default=False,
        help="in the visual terminal tube, use an RGB lever-arm torque primitive with a small inward contact component instead of learned candidate ranking",
    )
    parser.add_argument(
        "--geometric-torque-magnitude", type=float, default=0.24,
        help="tangential action magnitude of the RGB lever-arm torque fallback",
    )
    parser.add_argument(
        "--geometric-torque-inward-magnitude", type=float, default=0.08,
        help="inward contact component of the RGB lever-arm torque fallback",
    )
    parser.add_argument(
        "--geometric-torque-support-lever", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "derive the geometric torque tangent from the visible RGB T-boundary support lever "
            "instead of the agent-block centroid vector"
        ),
    )
    parser.add_argument(
        "--geometric-torque-raw-rgb-angle", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "use the current block angle from the direct RGB shape chart for "
            "the geometric torque sign; opt-in audit intervention"
        ),
    )
    parser.add_argument(
        "--geometric-torque-side-alignment-min", type=float, default=0.35,
        help=(
            "minimum RGB contact-side cosine for the geometric torque fallback; "
            "a negative opt-in value provides bounded near-contact hysteresis"
        ),
    )
    parser.add_argument(
        "--geometric-torque-contact-distance-max", type=float, default=0.0,
        help=(
            "optional maximum RGB agent-block distance for geometric torque; "
            "zero reuses contact-lock-distance"
        ),
    )
    parser.add_argument(
        "--geometric-torque-adaptive", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "adapt the RGB geometric torque magnitude from the previous observed block motion; "
            "the first contact pulse keeps the declared magnitude, while excessive motion "
            "causes a fail-closed reduction toward the declared minimum"
        ),
    )
    parser.add_argument(
        "--geometric-torque-min-magnitude", type=float, default=0.06,
        help="lower bound for adaptive geometric torque tangential magnitude",
    )
    parser.add_argument(
        "--geometric-torque-motion-target", type=float, default=3.0,
        help="RGB block displacement target in pixels for adaptive geometric torque",
    )
    parser.add_argument(
        "--geometric-torque-motion-feedback", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "subtract a small RGB-observed block-motion component from geometric torque while "
            "preserving the support-lever signed moment"
        ),
    )
    parser.add_argument(
        "--geometric-torque-motion-feedback-gain", type=float, default=0.08,
        help="maximum action-space magnitude of the RGB block-motion compensation",
    )
    parser.add_argument(
        "--geometric-torque-drift-abort", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "when geometric terminal torque causes a declared RGB XY drift or motion jump, "
            "release the torque lease and hand control to a bounded goal-directed translation"
        ),
    )
    parser.add_argument(
        "--geometric-torque-drift-xy", type=float, default=20.0,
        help="RGB terminal XY distance at which adaptive torque is considered to have drifted",
    )
    parser.add_argument(
        "--geometric-torque-drift-motion", type=float, default=10.0,
        help="single-step RGB block motion at which adaptive torque is considered to have drifted",
    )
    parser.add_argument(
        "--geometric-torque-drift-translation-magnitude", type=float, default=0.35,
        help="bounded RGB translation magnitude used after a geometric torque drift abort",
    )
    parser.add_argument(
        "--forbid-zero-before-terminal", action=argparse.BooleanOptionalAction, default=False,
        help="during learned push/translation, disallow the zero action until the visual terminal hold tube is reached",
    )
    parser.add_argument(
        "--terminal-motion-damping", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "insert one RGB-only zero settle pulse after a terminal action causes "
            "observable block motion with negative goal progress; disabled preserves "
            "the legacy continuous-torque protocol"
        ),
    )
    parser.add_argument(
        "--terminal-motion-damping-motion", type=float, default=2.0,
        help="minimum observed RGB block motion that activates terminal motion damping",
    )
    parser.add_argument(
        "--terminal-motion-damping-regression", type=float, default=0.5,
        help="minimum RGB goal-distance regression that activates terminal motion damping",
    )
    parser.add_argument(
        "--terminal-pulse-confirmation", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "confirm each nonzero terminal torque pulse from the next RGB frame; "
            "a failed pulse is frozen once and followed by an opposite-tangent "
            "contact-preserving fallback"
        ),
    )
    parser.add_argument(
        "--terminal-pulse-min-angle-gain", type=float, default=0.0,
        help="optional minimum next-frame absolute-angle improvement; zero treats a quantized but non-regressing RGB angle as inconclusive-safe",
    )
    parser.add_argument(
        "--terminal-pulse-max-angle-regression", type=float, default=0.08,
        help="maximum next-frame RGB absolute-angle regression tolerated while confirming a terminal pulse",
    )
    parser.add_argument(
        "--terminal-pulse-max-xy-regression", type=float, default=2.0,
        help="maximum RGB block-to-goal distance regression tolerated while confirming a terminal pulse",
    )
    parser.add_argument(
        "--terminal-pulse-fallback-magnitude", type=float, default=0.06,
        help="L2 cap for the opposite-tangent fallback after a failed terminal pulse",
    )
    parser.add_argument(
        "--terminal-pulse-freeze-steps", type=int, default=1,
        help="number of zero-action settle decisions after a failed terminal pulse",
    )
    parser.add_argument(
        "--terminal-pulse-max-failures", type=int, default=2,
        help="maximum failed pulse confirmations per episode before terminal control remains frozen",
    )
    parser.add_argument(
        "--learned-progress-guard", action=argparse.BooleanOptionalAction, default=False,
        help="replace a stalled learned push with an RGB receding-progress geometry push",
    )
    parser.add_argument(
        "--progress-guard-magnitude", type=float, default=0.75,
        help="action magnitude of the RGB geometry fallback used by learned-progress-guard",
    )
    parser.add_argument(
        "--contact-stagnation-fallback", action=argparse.BooleanOptionalAction, default=False,
        help="after a declared number of RGB-only stagnant contact observations, temporarily use the geometry push primitive",
    )
    parser.add_argument(
        "--contact-stagnation-window", type=int, default=3,
        help="number of recent RGB finite-difference effects used by the stagnation fallback",
    )
    parser.add_argument("--translation-angle-weight", type=float, default=0.35,
                        help="orientation penalty during translation-phase candidate selection")
    parser.add_argument(
        "--goal-progress-score-weight", type=float, default=0.0,
        help="optional RGB goal-direction prior added after learned candidate scoring; zero preserves the legacy learned-only selector",
    )
    parser.add_argument(
        "--geometry-translation-tangent", type=float, default=0.0,
        help="RGB geometry baseline tangential component while block XY is outside the terminal tube; zero gives a pure receding translation push",
    )
    parser.add_argument(
        "--hybrid-translation-action-max-magnitude", type=float, default=0.0,
        help=(
            "optional RGB action-norm cap applied to the action actually executed by the hybrid "
            "translation branch while the learned observer reports contact; zero preserves the legacy protocol"
        ),
    )
    parser.add_argument("--terminal-xy-weight", type=float, default=0.08,
                        help="translation penalty during terminal orientation repair")
    parser.add_argument("--online-local-weight", type=float, default=0.0,
                        help="fixed blend weight for RGB-observed local Jacobian; default is disabled after a negative ablation")
    parser.add_argument(
        "--online-local-angle-weight", type=float, default=0.0,
        help="independent fixed blend weight for the RGB-observed local angular effect; leaves XY effect unchanged",
    )
    parser.add_argument("--online-local-min-samples", type=int, default=4)
    parser.add_argument(
        "--online-local-window", type=int, default=4,
        help="number of most recent same-interface contact effects used by online local identification",
    )
    parser.add_argument(
        "--online-local-angle-jump", type=float, default=0.75,
        help="fail-closed angular jump threshold for online RGB local-effect measurements",
    )
    parser.add_argument("--ranker-ranking-weight", type=float, default=4.0,
                        help="weight of the within-root listwise ranking loss; fixed before held-out evaluation")
    parser.add_argument("--ranker-pairwise-weight", type=float, default=0.0,
                        help="weight of the within-root pairwise ordering loss; fixed before held-out evaluation")
    parser.add_argument(
        "--ranker-interface", choices=("latent_geometry", "rgb_geometry"),
        default="rgb_geometry",
        help="candidate-ranker input interface; rgb_geometry uses the same image-derived chart in training and runtime",
    )
    parser.add_argument(
        "--terminal-ranker-feature-mode", choices=("full", "invariant", "invariant_adapter"),
        default="full",
        help=(
            "RGB terminal-ranker feature chart. 'invariant' removes absolute canvas "
            "coordinates and uses a contact-frame/action-moment chart; invariant_adapter "
            "also exposes the frozen adapter's predicted terminal baseline score; full "
            "preserves the legacy protocol."
        ),
    )
    parser.add_argument("--ranker-min-contact-roots", type=int, default=32,
                        help="minimum held-out contact roots required before a ranker can enter runtime")
    parser.add_argument(
        "--terminal-angular-ranker", action=argparse.BooleanOptionalAction, default=False,
        help="fit a root-disjoint terminal-pose angular ranker and enable it only after held-out angle-regret gain",
    )
    parser.add_argument(
        "--terminal-torque-prior", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "fit no extra successor model: use an RGB support-lever-arm torque prior on the terminal "
            "manifold, and enable it only after a held-out ranking/XY gate"
        ),
    )
    parser.add_argument(
        "--terminal-torque-prior-weight", type=float, default=1.0,
        help="fixed multiplier for the RGB lever-arm torque prior; selected before held-out evaluation",
    )
    parser.add_argument(
        "--terminal-torque-prior-blend", type=float, default=0.25,
        help="fixed root-wise normalized blend of baseline angular score and the torque prior",
    )
    parser.add_argument(
        "--terminal-action-horizon", type=int, default=1,
        help=(
            "number of repeated physical action steps used to label the terminal angular ranker; "
            "1 preserves the legacy one-step residual protocol, while 2/4 train a short-horizon "
            "direct angular lookahead ranker"
        ),
    )
    parser.add_argument(
        "--terminal-action-commitment", type=int, default=1,
        help=(
            "number of consecutive terminal torque decisions that reuse the selected action; "
            "set this equal to terminal-action-horizon when evaluating a repeated-action "
            "lookahead ranker, while 1 preserves per-frame replanning"
        ),
    )
    parser.add_argument(
        "--terminal-ranker-target", choices=("angle", "pose"), default="angle",
        help=(
            "training target for the terminal short-horizon ranker; pose adds a "
            "fixed XY term to reduce angle-improving but translation-damaging actions"
        ),
    )
    parser.add_argument(
        "--terminal-pose-xy-weight", type=float, default=1.0,
        help="fixed XY coefficient in the terminal pose-ranker endpoint label",
    )
    parser.add_argument(
        "--terminal-trajectory-contact-penalty", type=float, default=0.0,
        help=(
            "offline penalty for intermediate agent-block separation during a "
            "repeated terminal action; zero preserves endpoint-only labels"
        ),
    )
    parser.add_argument(
        "--terminal-trajectory-xy-excursion-penalty", type=float, default=0.0,
        help=(
            "offline penalty for intermediate block-goal XY excursion during a "
            "repeated terminal action; zero preserves endpoint-only labels"
        ),
    )
    parser.add_argument(
        "--terminal-trajectory-angle-excursion-penalty", type=float, default=0.0,
        help=(
            "offline penalty for intermediate angle-error excursion during a "
            "repeated terminal action; zero preserves endpoint-only labels"
        ),
    )
    parser.add_argument(
        "--terminal-trajectory-contact-distance-max", type=float, default=80.0,
        help="offline contact-distance threshold used by the repeated-action safety label",
    )
    parser.add_argument(
        "--terminal-angular-min-roots", type=int, default=8,
        help="minimum held-out terminal roots required before promoting the angular ranker",
    )
    parser.add_argument(
        "--terminal-angular-xy-tolerance", type=float, default=1.0,
        help="declared pixel tolerance for material XY degradation when promoting the terminal angular residual ranker",
    )
    parser.add_argument(
        "--contact-mode-gate", action=argparse.BooleanOptionalAction, default=False,
        help="fit an RGB+action contact-mode head and enable it only when held-out contact ranking improves",
    )
    parser.add_argument(
        "--contact-canonical-knn", action=argparse.BooleanOptionalAction, default=False,
        help="fit a root-disjoint RGB/contact-frame KNN candidate and enable it only after validation regret improves",
    )
    parser.add_argument("--contact-knn-k", type=int, default=4)
    parser.add_argument("--contact-knn-context-weight", type=float, default=0.25)
    parser.add_argument("--contact-knn-action-weight", type=float, default=2.0)
    parser.add_argument(
        "--contact-mode-action-repeat", type=int, default=2,
        help="offline horizon used only to label impending contact separation for the RGB mode head",
    )
    parser.add_argument("--effect-frame", choices=("world", "contact"), default="world",
                        help="coordinate frame for learned action effects; contact uses RGB agent-block normal/tangent")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def wrap(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def side_route_action(
    contact_normal: np.ndarray,
    goal_direction: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Route around the block before re-entering a wrong-side contact.

    ``contact_normal`` points from the pusher to the block and
    ``goal_direction`` points from the block to the goal.  A wrong-side
    contact has a negative dot product.  The signed tangent is chosen to
    rotate the observed normal toward the goal direction while the outward
    component clears the contact manifold.  This is an RGB geometry
    primitive, not a simulator-state or successor query.
    """
    normal = np.asarray(contact_normal, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_direction, dtype=np.float64).reshape(2)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
    goal /= max(float(np.linalg.norm(goal)), 1.0e-8)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    rotation_sign = 1.0 if float(normal[0] * goal[1] - normal[1] * goal[0]) >= 0.0 else -1.0
    route = -0.70 * normal + rotation_sign * 0.95 * tangent
    route /= max(float(np.linalg.norm(route)), 1.0e-8)
    return np.clip(float(magnitude) * route, -1.0, 1.0).astype(np.float32)


REACQUISITION_PHASES = frozenset({
    "reacquire_clear", "reacquire", "reacquire_waypoint",
    "reacquire_arc", "reacquire_outer", "reacquire_target",
})


def reacquisition_retreat_magnitude(args: argparse.Namespace) -> float:
    """Return the scalar retreat magnitude permitted by the reacquisition gate.

    Recovery historically had its own 0.45-magnitude fallback while the
    side-reacquisition protocol exposed a separate action cap.  That made the
    protocol parameter ineffective on several recovery-to-reacquisition
    transitions.  Keep the old recovery magnitude as an upper bound, but make
    the declared reacquisition cap authoritative for every side-route action.
    """
    return max(0.0, min(
        float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
        float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
    ))


def bounded_reacquisition_action(
    action: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    """Enforce a true two-dimensional L2 cap on a reacquisition action.

    Component-wise clipping is insufficient for diagonal actions: two
    components at 0.20 have L2 norm 0.283.  This helper preserves direction
    and scales the full vector, while retaining a raw/final audit record.
    """
    raw = np.asarray(action, dtype=np.float32).reshape(2)
    cap = max(0.0, float(getattr(args, "reacquisition_action_max_magnitude", 0.35)))
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm > cap and raw_norm > 1.0e-8:
        bounded = raw * (cap / raw_norm)
    else:
        bounded = raw.copy()
    bounded = np.clip(bounded, -1.0, 1.0).astype(np.float32)
    final_norm = float(np.linalg.norm(bounded))
    return bounded, {
        "reacquisition_action_cap_used": cap,
        "reacquisition_raw_norm": raw_norm,
        "reacquisition_final_norm": final_norm,
        "reacquisition_cap_applied": float(raw_norm > cap + 1.0e-7),
    }


def bounded_l2_action(action: np.ndarray, cap: float) -> tuple[np.ndarray, float, float]:
    """Apply an optional two-dimensional L2 cap and return audit norms.

    Terminal approach is a distinct interface from side reacquisition: its
    target can be several pixels away while the pusher is already close enough
    to move the asymmetric block.  Component-wise clipping therefore permits
    a diagonal impulse that is much larger than intended.  A negative cap
    preserves the legacy protocol.
    """
    raw = np.asarray(action, dtype=np.float32).reshape(2)
    raw_norm = float(np.linalg.norm(raw))
    if cap > 0.0 and raw_norm > cap and raw_norm > 1.0e-8:
        bounded = raw * (cap / raw_norm)
    else:
        bounded = raw.copy()
    bounded = np.clip(bounded, -1.0, 1.0).astype(np.float32)
    return bounded, raw_norm, float(np.linalg.norm(bounded))


def terminal_pulse_fallback_action(
    previous_action: np.ndarray,
    agent: np.ndarray,
    block: np.ndarray,
    cap: float,
    inward_min: float = 0.02,
) -> np.ndarray:
    """Return a bounded opposite-tangent pulse after a failed torque pulse.

    The action is computed only from the previous executed action and the
    current RGB contact geometry.  It is deliberately not an effect-model
    prediction: the normal component keeps the pusher on the contact
    manifold, while the tangential component reverses the last pulse.  This
    makes the pulse-confirmation state machine fail closed when the learned
    ranker repeats a locally wrong sign.
    """
    previous = np.asarray(previous_action, dtype=np.float32).reshape(2)
    normal = np.asarray(block, dtype=np.float32) - np.asarray(agent, dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-6)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float32)
    previous_tangent = float(np.dot(previous, tangent))
    inward = max(float(np.dot(previous, normal)), float(inward_min), 0.0)
    tangent_size = min(abs(previous_tangent), max(float(cap), 0.0))
    correction = inward * normal
    if tangent_size > 1.0e-7:
        correction = correction - np.sign(previous_tangent) * tangent_size * tangent
    bounded, _, _ = bounded_l2_action(correction, max(float(cap), 0.0))
    return bounded


def terminal_support_fallback_action(
    agent: np.ndarray,
    block: np.ndarray,
    current_angle: float,
    goal_angle: float,
    cap: float,
    inward_min: float = 0.0,
    support_lever: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct a bounded RGB-only terminal action when learned support is empty.

    A terminal action cap can be smaller than every magnitude represented in
    the offline intervention bank.  In that case the learned selector has no
    calibrated non-zero candidate and must not silently choose the zero row or
    an extrapolative effect prediction.  This fallback uses only the current
    RGB contact geometry: a positive contact-normal component keeps the
    pusher on the manifold, while the lever-arm tangent has the declared
    angle-correction sign.  It is therefore an interface fallback, not a
    learned-dynamics result.
    """
    cap = max(float(cap), 0.0)
    if cap <= 1.0e-8:
        raise ValueError("terminal support fallback requires a positive cap")
    agent = np.asarray(agent, dtype=np.float32).reshape(2)
    block = np.asarray(block, dtype=np.float32).reshape(2)
    normal = block - agent
    normal /= max(float(np.linalg.norm(normal)), 1.0e-6)
    lever = agent - block
    if support_lever is not None:
        candidate = np.asarray(support_lever, dtype=np.float32).reshape(2)
        if float(np.linalg.norm(candidate)) > 1.0e-4:
            lever = candidate
    lever /= max(float(np.linalg.norm(lever)), 1.0e-6)
    # The support-point lever is useful for choosing the sign of the moment,
    # but its tangent is not necessarily orthogonal to the centroid contact
    # normal on an asymmetric T. Projecting that tangent directly can make
    # the supposedly inward component negative. Use the contact tangent for
    # the action direction and use the lever only for the signed-moment test.
    contact_tangent = np.asarray([-normal[1], normal[0]], dtype=np.float32)
    torque_sign = 1.0 if float(wrap(goal_angle - current_angle)) >= 0.0 else -1.0
    # Reserve a non-zero inward component, but leave most of the small cap to
    # the tangential torque component. Both terms are scaled together below.
    inward = min(cap * 0.25, max(float(inward_min), cap * 0.10))
    tangent = math.sqrt(max(cap * cap - inward * inward, 0.0))
    tangent_moment = float(lever[0] * contact_tangent[1] - lever[1] * contact_tangent[0])
    tangent_direction = contact_tangent * (
        1.0 if torque_sign * tangent_moment >= 0.0 else -1.0
    )
    raw = inward * normal + tangent * tangent_direction
    action, _, final_norm = bounded_l2_action(raw, cap)
    signed_torque = float(lever[0] * action[1] - lever[1] * action[0])
    return action, {
        "terminal_support_fallback_torque_sign": float(torque_sign),
        "terminal_support_fallback_inward_component": float(np.dot(action, normal)),
        "terminal_support_fallback_signed_torque": signed_torque,
        "terminal_support_fallback_norm": final_norm,
    }


def bounded_recovery_action(
    action: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bound ordinary recovery without changing the legacy protocol by default.

    Recovery is an RGB-only servo and has no successor-state guarantee.  A
    component-wise clip can still produce a large diagonal impulse and, when
    the visual target is stale, can drive the pusher across the block.  The
    new protocol therefore uses an explicit L2 cap; ``0`` keeps the historical
    behavior available for matched baseline runs.
    """
    cap = float(getattr(args, "recovery_action_max_magnitude", 0.0))
    raw = np.asarray(action, dtype=np.float32).reshape(2)
    bounded, raw_norm, final_norm = bounded_l2_action(raw, cap)
    return bounded, {
        "recovery_action_cap_used": cap,
        "recovery_action_raw_norm": raw_norm,
        "recovery_action_final_norm": final_norm,
        "recovery_action_cap_applied": float(cap > 0.0 and raw_norm > cap + 1.0e-7),
    }


def terminal_centroid_local_repair(
    agent: np.ndarray,
    desired_agent: np.ndarray,
    action_scale: float,
    action_cap: float,
) -> tuple[np.ndarray, float, float, float]:
    """Take a bounded local step toward the declared terminal contact point.

    This observation-only primitive is used when the pusher remains in the
    broad contact band but RGB side alignment dips mildly.  Routing such a
    frame through outer reacquisition can move the pusher far away and
    repeatedly disturb the block; this repair preserves the local manifold
    while retaining a finite action cap.
    """
    approach = np.asarray(desired_agent, dtype=np.float32) - np.asarray(
        agent, dtype=np.float32
    )
    normalized = approach / max(float(action_scale), 1.0e-6)
    action, raw_norm, final_norm = bounded_l2_action(normalized, float(action_cap))
    return action, raw_norm, final_norm, float(np.linalg.norm(approach))


def terminal_visual_hold_eligible(
    phase: str,
    observed_xy: float,
    observed_angle_error: float,
    hold_xy: float,
    hold_angle: float,
) -> bool:
    """Return whether an RGB pose is already safe to freeze.

    The check is intentionally independent of contact-side alignment.  Once
    the observed pose is inside the declared terminal tube, a zero action is
    safer than launching a side-reacquisition maneuver whose sole purpose is
    to repair an already-solved contact lease.
    """
    return bool(
        phase in {"torque", "hold"}
        and float(observed_xy) <= float(hold_xy)
        and float(observed_angle_error) <= float(hold_angle)
    )


def update_runtime_block_history(
    args: argparse.Namespace, block: np.ndarray, window: int = 5,
) -> np.ndarray:
    """Maintain a short robust RGB block history for recovery diagnostics.

    The paired RGB readout is accurate enough for ordinary control but can
    jump under contact occlusion.  Recovery-anchor drift should not abort a
    collision-safe route because of one such frame.  This history is used only
    for the drift certificate; the controller still uses the current readout
    everywhere else.
    """
    history = list(getattr(args, "_runtime_block_history", []))
    current = np.asarray(block, dtype=np.float32).reshape(2).copy()
    if not history or float(np.linalg.norm(current - history[-1])) > 1.0e-6:
        history.append(current)
    history = history[-max(1, int(window)):]
    args._runtime_block_history = history
    return np.median(np.asarray(history, dtype=np.float32), axis=0).astype(np.float32)


def reacquisition_phase_active(phase: str, info: dict[str, Any]) -> bool:
    """Identify every runtime decision governed by the reacquisition lease."""
    return (
        phase in REACQUISITION_PHASES
        or float(info.get("reacquisition_attempt", 0.0)) > 0.0
    )


def make_recovery_anchor(
    block: np.ndarray,
    goal_direction: np.ndarray,
    agent: np.ndarray,
    clearance: float,
) -> dict[str, Any]:
    """Freeze the geometric frame for one side-reacquisition attempt.

    Recovery is a hybrid geometric controller.  Recomputing its waypoint from
    the current RGB block centroid after every action makes the target move
    with perception noise and with incidental block motion, which creates a
    route/retreat limit cycle.  The anchor is intentionally RGB-only and is
    released after the pusher reaches the declared target side.
    """
    block = np.asarray(block, dtype=np.float32).reshape(2)
    direction = np.asarray(goal_direction, dtype=np.float32).reshape(2)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-8)
    agent = np.asarray(agent, dtype=np.float32).reshape(2)
    requested_clearance = float(clearance)
    perpendicular = np.asarray([direction[1], -direction[0]], dtype=np.float32)
    lower = float(RECOVERY_WORKSPACE_MARGIN)
    upper = float(PUSHT_STATE_SIZE - RECOVERY_WORKSPACE_MARGIN)

    # The old anchor selected the nearest lateral waypoint but never checked
    # the opposite-side outer target.  Near a wall this produced an
    # unreachable target (e.g. y=523 in a 512-wide workspace), after which
    # ``reacquire_arc`` consumed the entire episode.  Search a small declared
    # clearance ladder and choose the shortest feasible collision-safe route.
    clearance_candidates = np.linspace(
        max(RECOVERY_MIN_CLEARANCE, min(requested_clearance, 180.0)),
        RECOVERY_MIN_CLEARANCE,
        num=16,
        dtype=np.float32,
    )
    feasible: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for candidate_clearance in clearance_candidates:
        outer = block - candidate_clearance * direction
        for candidate_sign in (1.0, -1.0):
            waypoint = block + candidate_sign * candidate_clearance * perpendicular
            arc_midpoint = block - (candidate_clearance / np.sqrt(2.0)) * direction + (
                candidate_sign * candidate_clearance / np.sqrt(2.0)
            ) * perpendicular
            if (
                np.all(waypoint >= lower) and np.all(waypoint <= upper)
                and np.all(outer >= lower) and np.all(outer <= upper)
                and np.all(arc_midpoint >= lower) and np.all(arc_midpoint <= upper)
            ):
                path_length = (
                    float(np.linalg.norm(waypoint - agent))
                    + float(np.linalg.norm(arc_midpoint - waypoint))
                    + float(np.linalg.norm(outer - arc_midpoint))
                )
                feasible.append((path_length, candidate_sign, waypoint, outer))
        if feasible:
            break
    if feasible:
        _path_length, route_sign, _waypoint, _outer = min(feasible, key=lambda row: row[0])
        clearance = float(np.linalg.norm(_outer - block))
        workspace_safe = True
    else:
        # Extremely constrained boundary cases remain auditable and finite.
        # Clip the outer target and retain a conservative lateral retreat;
        # the runtime motion guard will abort rather than chase it forever.
        # Do not silently shrink below the T's effective collision radius.
        # This anchor is explicitly marked infeasible and the runtime uses a
        # direct contact-side fallback instead of pretending that the clipped
        # outer point is a safe arc target.
        # No complete clearance circle fits inside the workspace.  Use the
        # smallest declared route radius and let ``recovery_anchor_geometry``
        # clip individual waypoints to the RGB workspace box.  The caller
        # records this as a clipped route and the motion guard can still
        # abort on observed block displacement; freezing here made boundary
        # recovery impossible by construction.
        clearance = float(RECOVERY_MIN_CLEARANCE)
        candidates = np.asarray([
            block + clearance * perpendicular,
            block - clearance * perpendicular,
        ], dtype=np.float32)
        route_sign = 1.0 if float(np.linalg.norm(candidates[0] - agent)) <= float(
            np.linalg.norm(candidates[1] - agent)
        ) else -1.0
        workspace_safe = False
    # 64 px is the preferred clearance inherited from the original recovery
    # certificate.  A smaller feasible route is a boundary repair, not an
    # ordinary route; expose it so paired replays can separate this repair
    # from learned action effects.
    clearance_relaxed = bool(
        workspace_safe and clearance < min(requested_clearance, 64.0) - 1.0e-6
    )
    return {
        "block": block.copy(),
        "direction": direction.copy(),
        "route_sign": float(route_sign),
        "clearance": float(clearance),
        "workspace_safe": bool(workspace_safe),
        "clearance_relaxed": clearance_relaxed,
    }


def recovery_anchor_geometry(
    anchor: dict[str, Any], contact_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed waypoint, arc midpoint, outer target, and contact point."""
    block = np.asarray(anchor["block"], dtype=np.float32).reshape(2)
    direction = np.asarray(anchor["direction"], dtype=np.float32).reshape(2)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-8)
    clearance = float(anchor["clearance"])
    route_sign = float(anchor["route_sign"])
    perpendicular = np.asarray([direction[1], -direction[0]], dtype=np.float32)
    waypoint = block + route_sign * clearance * perpendicular
    arc_midpoint = block - (clearance / np.sqrt(2.0)) * direction + (
        route_sign * clearance / np.sqrt(2.0)
    ) * perpendicular
    outer_target = block - clearance * direction
    target_contact = block - float(contact_distance) * direction
    lower = float(RECOVERY_WORKSPACE_MARGIN)
    upper = float(PUSHT_STATE_SIZE - RECOVERY_WORKSPACE_MARGIN)
    waypoint = np.clip(waypoint, lower, upper).astype(np.float32)
    arc_midpoint = np.clip(arc_midpoint, lower, upper).astype(np.float32)
    outer_target = np.clip(outer_target, lower, upper).astype(np.float32)
    target_contact = np.clip(target_contact, lower, upper).astype(np.float32)
    return waypoint, arc_midpoint, outer_target, target_contact


def recovery_anchor_trace(anchor: dict[str, Any] | None) -> dict[str, Any]:
    """Serialize anchor state into the per-step audit trace."""
    if anchor is None:
        return {
            "recovery_anchor_active": 0.0,
            "reacquisition_anchor_reference": "median_runtime_block_history",
        }
    return {
        "recovery_anchor_active": 1.0,
        "reacquisition_anchor_reference": "median_runtime_block_history",
        "recovery_anchor_block": np.asarray(anchor["block"], dtype=np.float32).tolist(),
        "recovery_anchor_direction": np.asarray(anchor["direction"], dtype=np.float32).tolist(),
        "recovery_anchor_route_sign": float(anchor["route_sign"]),
        "recovery_anchor_clearance": float(anchor["clearance"]),
        "recovery_anchor_workspace_safe": float(bool(anchor.get("workspace_safe", True))),
        "recovery_anchor_clearance_relaxed": float(bool(anchor.get("clearance_relaxed", False))),
        "recovery_anchor_workspace_clipped": float(not bool(anchor.get("workspace_safe", True))),
    }


def theta_from_pose(pose: np.ndarray) -> float:
    return float(np.arctan2(float(pose[4]), float(pose[5])))


def block_pose(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    return np.asarray([state[2], state[3], theta_from_pose(state)], dtype=np.float32)


def goal_observation_image(
    env: gym.Env,
    goal_state: np.ndarray,
    centered_goal_image: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Return the declared visual goal observation for one root.

    The installed PushT environment stores ``goal_state`` and renders its
    target shape at a separate fixed ``goal_pose``.  The centered image is
    therefore not a visual encoding of the evaluator goal position.  The
    aligned protocol changes only the renderer target pose; the controller
    still receives pixels and never reads the state.  The centered protocol
    remains available as the original negative control.
    """
    if mode == "centered":
        return np.asarray(centered_goal_image, dtype=np.uint8).copy()
    if mode != "aligned_state_goal":
        raise ValueError(f"unknown goal observation mode: {mode}")
    goal_state = np.asarray(goal_state, dtype=np.float32)
    if goal_state.shape[0] < 5:
        raise ValueError("goal_state must contain block position and angle")
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "goal_pose"):
        raise AttributeError("PushT environment does not expose a render goal pose")
    # The aligned protocol is an image-only target interface.  The previous
    # implementation changed ``goal_pose`` but rendered the current agent and
    # block on top of the target.  That is harmless for distant roots, but it
    # systematically occludes the target on the terminal manifold: the RGB
    # target centroid then ceases to be a pose observation and the current and
    # goal charts no longer share a physical coordinate system.  Render the
    # target in isolation while preserving the environment state.  This does
    # not expose simulator state to the controller; it only removes nuisance
    # scene pixels from the declared goal image.
    original_goal_pose = np.asarray(getattr(unwrapped, "goal_pose"), dtype=np.float32).copy()
    agent = getattr(unwrapped, "agent", None)
    block = getattr(unwrapped, "block", None)
    saved_agent_position = None if agent is None else np.asarray(agent.position, dtype=np.float64).copy()
    saved_block_position = None if block is None else np.asarray(block.position, dtype=np.float64).copy()
    saved_agent_velocity = None if agent is None else np.asarray(agent.velocity, dtype=np.float64).copy()
    saved_block_velocity = None if block is None else np.asarray(block.velocity, dtype=np.float64).copy()
    unwrapped.goal_pose = np.asarray(
        [goal_state[2], goal_state[3], goal_state[4]], dtype=np.float32,
    )
    try:
        # The renderer operates on a 512x512 canvas.  A far-off position is
        # used instead of changing colors or variation-space values so the
        # target geometry and renderer configuration remain untouched.
        if agent is not None:
            agent.position = (-10000.0, -10000.0)
        if block is not None:
            block.position = (-10000.0, -10000.0)
        # Pymunk keeps a shape broad-phase cache.  Reindexing is required when
        # a body is moved only for rendering; otherwise debug_draw can still
        # paint the old scene at its cached location.
        space = getattr(unwrapped, "space", None)
        if space is not None:
            if agent is not None:
                space.reindex_shapes_for_body(agent)
            if block is not None:
                space.reindex_shapes_for_body(block)
        return np.asarray(env.render(), dtype=np.uint8).copy()
    finally:
        unwrapped.goal_pose = original_goal_pose
        if agent is not None and saved_agent_position is not None:
            agent.position = tuple(float(value) for value in saved_agent_position)
            if saved_agent_velocity is not None:
                agent.velocity = tuple(float(value) for value in saved_agent_velocity)
        if block is not None and saved_block_position is not None:
            block.position = tuple(float(value) for value in saved_block_position)
            if saved_block_velocity is not None:
                block.velocity = tuple(float(value) for value in saved_block_velocity)
        space = getattr(unwrapped, "space", None)
        if space is not None:
            if agent is not None:
                space.reindex_shapes_for_body(agent)
            if block is not None:
                space.reindex_shapes_for_body(block)


def pose_cost(block: np.ndarray, goal_pose: np.ndarray) -> np.ndarray:
    """Fixed task-space candidate cost used for every ranking comparison."""
    block = np.asarray(block, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)
    if block.ndim == 2:
        xy = np.linalg.norm(block[:, :2] - goal_pose[None, :2], axis=-1) / 20.0
        angle = np.asarray(wrap(block[:, 2] - goal_pose[None, 2]))
    else:
        xy = np.linalg.norm(block[..., :2] - goal_pose[..., :2], axis=-1) / 20.0
        angle = np.asarray(wrap(block[..., 2] - goal_pose[..., 2]))
    return xy + 10.0 * np.abs(angle)


def strict_success(state: np.ndarray, goal: np.ndarray) -> bool:
    current = state_vector(state)
    target = state_vector(goal)
    return bool(
        np.linalg.norm(current[2:4] - target[2:4]) <= 20.0
        and abs(float(wrap(theta_from_pose(current) - theta_from_pose(target)))) <= np.pi / 9.0
    )


def action_bank(directions: int, magnitudes: list[float]) -> np.ndarray:
    if directions < 4 or len(magnitudes) < 1:
        raise ValueError("at least four directions and one magnitude are required")
    if any(not 0.0 < float(value) <= 1.0 for value in magnitudes):
        raise ValueError("action magnitudes must lie in (0, 1]")
    actions = [np.zeros(2, dtype=np.float32)]
    for magnitude in magnitudes:
        for index in range(directions):
            angle = 2.0 * np.pi * index / directions
            actions.append(
                np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32) * float(magnitude)
            )
    return np.asarray(actions, dtype=np.float32)


def declared_action_magnitudes(args: argparse.Namespace) -> list[float]:
    """Return the complete offline support used by the RGB controller.

    The terminal controller has a deliberately small torque fan.  Those
    actions are part of the deployed interface and therefore must also be
    present in the matched offline interventions; otherwise the controller
    evaluates an adapter on an unseen action magnitude exactly where contact
    is most sensitive.  Preserve the broad bank order and append unique small
    terminal magnitudes that satisfy the declared contact cap.
    """
    values = [float(value) for value in args.action_magnitudes]
    cap = float(getattr(args, "contact_action_max_magnitude", 0.0))
    for value in getattr(args, "small_action_magnitudes", []):
        value = float(value)
        if cap > 0.0 and value > cap + 1.0e-6:
            continue
        if not any(abs(value - existing) <= 1.0e-7 for existing in values):
            values.append(value)
    return values


def load_base(path: Path, device: str) -> tuple[VisualCausalWorldModel, Normalization, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_args = dict(payload["args"])
    if str(saved_args["vision_encoder"]) != "mask_moments":
        raise ValueError("the local RGB effect adapter requires the mask_moments encoder")
    model = VisualCausalWorldModel(
        int(saved_args["latent_dim"]),
        str(saved_args["vision_encoder"]),
        bool(saved_args.get("temporal_context", False)),
        checkpoint_utility_mode(payload),
        checkpoint_adapter_architecture(payload),
        bool(saved_args.get("structured_readout", False)),
        bool(saved_args.get("contact_frame_dynamics", False)),
        bool(saved_args.get("mode_conditioned_dynamics", False)),
        bool(saved_args.get("current_only_readout", False)),
        bool(saved_args.get("torque_features", False)),
        bool(saved_args.get("temporal_dynamics", False)),
        bool(saved_args.get("previous_action_dynamics", False)),
        bool(saved_args.get("explicit_angle_effect", False)),
        bool(saved_args.get("explicit_angle_readout", False)),
    ).to(device)
    load_result = model.load_state_dict(payload["model"], strict=False)
    # Some older otherwise-compatible checkpoints predate the optional
    # explicit goal-angle readout.  The adapter trains its own root-disjoint
    # goal decoder below, so this head is not required for the declared RGB
    # runtime.  Allow exactly these four absent parameters, while retaining a
    # hard failure for every other architecture mismatch.
    allowed_missing = {
        "goal_angle_head.0.weight", "goal_angle_head.0.bias",
        "goal_angle_head.2.weight", "goal_angle_head.2.bias",
    }
    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)
    if unexpected or not missing.issubset(allowed_missing):
        raise RuntimeError(
            "incompatible base checkpoint: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(device)
    # Explicit-angle/contact-frame checkpoints require the same frozen state
    # normalization during goal encoding and successor-effect decoding as
    # during training.  The older adapter loader omitted this initialization,
    # making otherwise valid cross-seed checkpoints fail before evaluation.
    model.set_state_normalization(mean, std)
    return model, normalizer, saved_args


def make_local_effect_dataset(
    env: gym.Env,
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    image_size: int,
    actions: np.ndarray,
    action_repeat: int,
    goal_observation: str = "centered",
) -> dict[str, np.ndarray]:
    """Collect matched local interventions, retaining root-level identities."""
    if action_repeat < 1:
        raise ValueError("action_repeat must be positive")
    images: list[np.ndarray] = []
    previous_images: list[np.ndarray] = []
    goal_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    successors: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    previous_action_rows: list[np.ndarray] = []
    trajectory_rows: list[np.ndarray] = []
    root_rows: list[int] = []
    contact_rows: list[bool] = []
    for root_id, (state, goal, image, previous, goal_image, previous_action, contact) in enumerate(roots):
        root_previous_action = (
            np.zeros(2, dtype=np.float32)
            if previous_action is None else np.asarray(previous_action, dtype=np.float32)
        )
        if root_previous_action.shape != (2,) or not np.isfinite(root_previous_action).all():
            raise ValueError("root previous_action must be a finite two-dimensional action")
        if goal_observation == "aligned_state_goal":
            env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
        root_goal_image = goal_observation_image(
            env, goal, goal_image, goal_observation,
        )
        successors_for_root: list[np.ndarray] = []
        for action in actions:
            env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
            observation: dict[str, Any] | None = None
            trajectory: list[np.ndarray] = [observable_pose_vector(state)]
            for _ in range(action_repeat):
                observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
                if observation is not None:
                    trajectory.append(observable_pose_vector(
                        np.asarray(observation["state"], dtype=np.float32),
                    ))
                if terminated or truncated:
                    break
            if observation is None:
                raise RuntimeError("environment returned no observation")
            # The ranker may use a repeated-action horizon.  Preserve a fixed
            # [h+1, pose] interface even when the simulator terminates early;
            # padding with the last observed pose is conservative and keeps
            # termination from being mistaken for an unobserved improvement.
            while len(trajectory) < action_repeat + 1:
                trajectory.append(trajectory[-1].copy())
            if len(trajectory) > action_repeat + 1:
                trajectory = trajectory[:action_repeat + 1]
            successor = observable_pose_vector(np.asarray(observation["state"], dtype=np.float32))
            successors_for_root.append(successor)
            images.append(resize_image(image, image_size))
            previous_images.append(resize_image(previous, image_size))
            goal_images.append(resize_image(root_goal_image, image_size))
            states.append(observable_pose_vector(state))
            successors.append(successor)
            action_rows.append(np.asarray(action, dtype=np.float32))
            previous_action_rows.append(root_previous_action.copy())
            trajectory_rows.append(np.asarray(trajectory, dtype=np.float32))
            root_rows.append(root_id)
            contact_rows.append(bool(contact))
        # The first action is contractually zero.  Keeping the check here makes
        # accidental reordering of the intervention bank fail closed.
        if not np.allclose(actions[0], 0.0):
            raise RuntimeError("local-effect action bank must start with zero")
        if not np.isfinite(np.asarray(successors_for_root)).all():
            raise RuntimeError("non-finite simulator successor")
    return {
        "image": np.asarray(images, dtype=np.uint8),
        "previous_image": np.asarray(previous_images, dtype=np.uint8),
        "goal_image": np.asarray(goal_images, dtype=np.uint8),
        "state": np.asarray(states, dtype=np.float32),
        "next_state": np.asarray(successors, dtype=np.float32),
        "action": np.asarray(action_rows, dtype=np.float32),
        "previous_action": np.asarray(previous_action_rows, dtype=np.float32),
        "trajectory_pose": np.asarray(trajectory_rows, dtype=np.float32),
        "root_id": np.asarray(root_rows, dtype=np.int64),
        "contact_root": np.asarray(contact_rows, dtype=bool),
    }


def root_goal_observation_images(
    env: gym.Env,
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    mode: str,
) -> np.ndarray:
    """Render root-disjoint goal observations at the declared interface."""
    images: list[np.ndarray] = []
    for state, goal, _image, _previous, centered_goal, _previous_action, _contact in roots:
        env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
        images.append(goal_observation_image(env, goal, centered_goal, mode))
    return np.asarray(images, dtype=np.uint8)


@torch.inference_mode()
def frozen_features(
    model: VisualCausalWorldModel, data: dict[str, np.ndarray], image_size: int, device: str,
) -> np.ndarray:
    # Encode in chunks: a formal action bank can contain tens of thousands of
    # RGB rows, and materializing current+previous images on the accelerator
    # at once is unnecessary and makes the experiment depend on unrelated GPU
    # memory occupancy.
    chunk_size = 2048

    def encode_rows(values: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(values), chunk_size):
            batch = tensor_images(values[start:start + chunk_size], device)
            chunks.append(model.encoder.features(batch).cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    current_features = encode_rows(data["image"])
    previous_features = encode_rows(data["previous_image"])
    # Goal is intentionally not included: physical action effect must not be
    # allowed to depend on the selected task.  The goal remains runtime input
    # only for the task-space scoring stage.
    del image_size
    action_np = np.asarray(data["action"], dtype=np.float32)
    # Add a small, explicit contact chart.  The raw moment vector already
    # contains this information, but the derived normal/tangent coordinates
    # make the relevant local geometry identifiable without requiring the MLP
    # to rediscover a coordinate transform from a small contact subset.
    contact_chart = contact_chart_from_features(current_features, action_np)
    boundary_chart = contact_boundary_chart_from_images(data["image"], action_np)
    values = np.concatenate([
        current_features, current_features - previous_features, action_np, contact_chart,
        boundary_chart,
    ], axis=1).astype(np.float32)
    # Previous action is an observable runtime variable and disambiguates
    # identical RGB contact geometry with different hidden momentum.  Keep it
    # appended so the legacy 82-column interface remains byte-compatible when
    # callers intentionally omit the field.
    if "previous_action" in data:
        previous_action = np.asarray(data["previous_action"], dtype=np.float32)
        if previous_action.shape != (len(values), 2):
            raise ValueError("previous_action rows must have shape [N, 2]")
        values = np.concatenate([values, previous_action], axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite frozen RGB features")
    return values


def contact_chart_from_features(features: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Return the explicit RGB contact-normal/tangent chart used by training."""
    features = np.asarray(features, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if features.ndim == 1:
        features = features[None]
    if actions.ndim == 1:
        actions = actions[None]
    if features.shape[0] != actions.shape[0] or features.shape[1] < 13 or actions.shape[1] != 2:
        raise ValueError("feature/action batch shapes are incompatible")
    agent = features[:, 1:3] * (PUSHT_STATE_SIZE / 2.0)
    block = features[:, 11:13] * (PUSHT_STATE_SIZE / 2.0)
    relative = block - agent
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    normal = relative / np.maximum(distance, 1e-6)
    tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
    return np.concatenate([
        agent, block, relative, distance,
        np.sum(actions * normal, axis=1, keepdims=True),
        np.sum(actions * tangent, axis=1, keepdims=True),
        np.linalg.norm(actions, axis=1, keepdims=True),
    ], axis=1).astype(np.float32)


def contact_boundary_chart_from_images(images: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Extract an RGB-only support-point and lever-arm chart for the T block."""
    values = np.asarray(images)
    if values.ndim == 3:
        values = values[None]
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("images must have shape [N, 3, H, W]")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if len(values) != len(actions) or actions.shape[1] != 2:
        raise ValueError("image/action batch shapes are incompatible")
    rgb = values.astype(np.float32)
    if float(np.max(rgb, initial=0.0)) > 1.5:
        rgb = rgb / 255.0
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    agent_mask = (blue > 0.70) & (blue > green + 0.12) & (blue > red + 0.18)
    block_mask = (blue < 0.93) & (red < 0.82) & (blue > green + 0.025) & (green > red + 0.025)
    height, width = values.shape[2:]
    ys, xs = np.mgrid[0:height, 0:width]
    x_grid = 2.0 * xs / max(width - 1, 1) - 1.0
    y_grid = 2.0 * ys / max(height - 1, 1) - 1.0
    output = np.zeros((len(values), 10), dtype=np.float32)
    for row in range(len(values)):
        agent_y, agent_x = np.where(agent_mask[row])
        block_y, block_x = np.where(block_mask[row])
        if len(agent_x) == 0 or len(block_x) == 0:
            continue
        agent = np.asarray([x_grid[agent_y, agent_x].mean(), y_grid[agent_y, agent_x].mean()])
        block = np.asarray([x_grid[block_y, block_x].mean(), y_grid[block_y, block_x].mean()])
        relative = agent - block
        distance = max(float(np.linalg.norm(relative)), 1e-6)
        normal = relative / distance
        tangent = np.asarray([-normal[1], normal[0]])
        points = np.column_stack([x_grid[block_y, block_x], y_grid[block_y, block_x]])
        offsets = points - block[None]
        normal_projection = offsets @ normal
        tangent_projection = offsets @ tangent
        support_index = int(np.argmax(normal_projection))
        support_pos = float(normal_projection[support_index])
        support_neg = float(-np.min(normal_projection))
        tangent_pos = float(np.max(tangent_projection))
        tangent_neg = float(-np.min(tangent_projection))
        # Preserve the actual visible support point.  Projecting it onto the
        # contact normal erases the tangential lever arm of the asymmetric T
        # and can invert the sign of r x F used by the torque ranker.
        lever = offsets[support_index]
        contact = block + lever
        contact_gap = float(np.linalg.norm(agent - contact))
        torque = float(lever[0] * actions[row, 1] - lever[1] * actions[row, 0])
        output[row] = np.asarray([
            contact[0], contact[1], lever[0], lever[1],
            support_pos, support_neg, tangent_pos, tangent_neg,
            contact_gap, torque,
        ], dtype=np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite RGB contact-boundary chart")
    return output


def rgb_signed_torque_probe_action(
    image: np.ndarray,
    agent: np.ndarray,
    block: np.ndarray,
    signed_angle_error: float,
    args: argparse.Namespace,
    orientation_sign: float,
) -> tuple[np.ndarray, dict[str, float | list[float]]]:
    """Construct one bounded, RGB-only signed-torque excitation.

    The visible support lever determines the tangent *basis*, but not its
    effective signed gain.  The latter is deliberately identified online by
    comparing the next RGB angle error with the pre-pulse error.  This helper
    therefore never claims that the geometric sign is causal; it only emits a
    small contact-preserving excitation that can be falsified on the next
    frame.  ``orientation_sign`` is flipped by the caller after a worsening
    angle error.
    """
    rgb = resize_image(image, int(getattr(args, "image_size", 64)))
    chart = contact_boundary_chart_from_images(
        rgb[None], np.zeros((1, 2), dtype=np.float32),
    )[0]
    support_lever = np.asarray(chart[2:4], dtype=np.float32)
    if float(np.linalg.norm(support_lever)) > 1.0e-4:
        lever = support_lever
        support_lever_used = 1.0
        inward = -support_lever
    else:
        lever = np.asarray(agent, dtype=np.float32) - np.asarray(block, dtype=np.float32)
        support_lever_used = 0.0
        inward = np.asarray(block, dtype=np.float32) - np.asarray(agent, dtype=np.float32)
    lever = lever / max(float(np.linalg.norm(lever)), 1.0e-6)
    inward = inward / max(float(np.linalg.norm(inward)), 1.0e-6)
    tangent = np.asarray([-lever[1], lever[0]], dtype=np.float32)
    desired_sign = 1.0 if float(signed_angle_error) >= 0.0 else -1.0
    tangent_sign = desired_sign * (1.0 if float(orientation_sign) >= 0.0 else -1.0)
    magnitude_scale = float(np.clip(
        getattr(args, "_signed_torque_probe_magnitude_scale", 1.0), 0.05, 1.0,
    ))
    tangent_magnitude = float(np.clip(
        getattr(args, "signed_torque_probe_magnitude", 0.16) * magnitude_scale,
        0.0, 0.35,
    ))
    inward_magnitude = float(np.clip(
        getattr(args, "signed_torque_probe_inward", 0.08) * magnitude_scale,
        0.0, 0.35,
    ))
    action = (
        inward_magnitude * inward
        + tangent_sign * tangent_magnitude * tangent
    ).astype(np.float32)
    # The probe is an explicitly declared identification intervention.  It
    # may use a larger norm than the ordinary terminal ranker, but only when
    # that separate cap is present.  Missing the field preserves every older
    # artifact's protocol exactly: it falls back to terminal/contact cap.
    cap = float(getattr(args, "signed_torque_probe_max_magnitude", -1.0) or -1.0)
    if cap <= 0.0:
        cap = float(getattr(args, "terminal_action_max_magnitude", -1.0))
    if cap <= 0.0:
        cap = float(getattr(args, "contact_action_max_magnitude", 0.35))
    norm = float(np.linalg.norm(action))
    if cap > 0.0 and norm > cap:
        action *= cap / max(norm, 1.0e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32), {
        "signed_torque_probe_orientation_sign": float(orientation_sign),
        "signed_torque_probe_desired_sign": float(desired_sign),
        "signed_torque_probe_tangent_sign": float(tangent_sign),
        "signed_torque_probe_magnitude": tangent_magnitude,
        "signed_torque_probe_inward_magnitude": inward_magnitude,
        "signed_torque_probe_magnitude_scale": magnitude_scale,
        "signed_torque_probe_support_lever_used": support_lever_used,
        "signed_torque_probe_lever": lever.tolist(),
        "signed_torque_probe_tangent": tangent.tolist(),
        "signed_torque_probe_inward": inward.tolist(),
        "signed_torque_probe_action_norm": float(np.linalg.norm(action)),
    }


def terminal_contact_lease_from_boundary(
    boundary_chart: np.ndarray,
    gap_max_pixels: float,
    coordinate_scale: float = PUSHT_STATE_SIZE / 2.0,
) -> tuple[bool, float, bool]:
    """Validate a terminal contact lease from one RGB boundary chart.

    The lease is deliberately based on the distance from the agent centroid
    to the visible support point, not on the block centroid. ``gap_max=0`` is
    the compatibility mode used by historical protocols. The visibility flag
    prevents an all-zero chart (missing masks) from being treated as contact.
    """
    chart = np.asarray(boundary_chart, dtype=np.float32).reshape(-1)
    if chart.shape[0] != 10 or not np.isfinite(chart).all():
        return False, float("nan"), False
    visible = bool(float(np.linalg.norm(chart[:4])) > 1.0e-6)
    gap = float(chart[8] * float(coordinate_scale))
    valid = bool(
        float(gap_max_pixels) <= 0.0
        or (visible and np.isfinite(gap) and gap <= float(gap_max_pixels))
    )
    return valid, gap, visible


def terminal_probe_contact_lease_from_image(
    image: np.ndarray,
    gap_max_pixels: float,
    image_size: int,
) -> tuple[bool, float, bool]:
    """Apply the declared RGB support-gap lease before signed probing.

    Centroid proximity and side alignment are insufficient on PushT's
    asymmetric T: the pusher can be close to the block while still missing
    the visible support boundary.  Signed probing in that state turns RGB
    angle noise into a false causal feedback signal.  This helper keeps the
    historical compatibility mode (``gap_max_pixels <= 0``) unchanged while
    making the non-degenerate boundary lease available to every terminal
    excitation path.
    """
    if float(gap_max_pixels) <= 0.0:
        return True, float("nan"), False
    rgb = resize_image(np.asarray(image), int(image_size))
    chart = contact_boundary_chart_from_images(
        rgb[None], np.zeros((1, 2), dtype=np.float32),
    )[0]
    return terminal_contact_lease_from_boundary(chart, float(gap_max_pixels))


def rgb_mask_xy_from_image(image: np.ndarray) -> np.ndarray | None:
    """Read a block centroid from RGB only, in PushT simulator coordinates.

    The PushT renderer uses a distinct blue agent and green/blue block. This
    observer deliberately uses no latent head and no simulator state; it is a
    diagnostic intervention for separating visual localization error from
    action-effect error. ``None`` is returned when the block mask is absent.
    """
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("RGB mask observer expects an HWC image")
    rgb = values.astype(np.float32)
    if float(np.max(rgb, initial=0.0)) > 1.5:
        rgb /= 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    block_mask = (blue < 0.93) & (red < 0.82) & (blue > green + 0.025) & (green > red + 0.025)
    ys, xs = np.where(block_mask)
    if len(xs) == 0:
        return None
    height, width = values.shape[:2]
    return np.asarray([
        float(xs.mean()) / max(width - 1, 1) * PUSHT_STATE_SIZE,
        float(ys.mean()) / max(height - 1, 1) * PUSHT_STATE_SIZE,
    ], dtype=np.float32)


def _rgb_mask_chart(mask: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Return a finite RGB shape chart for one binary object mask.

    The chart is deliberately computed in image coordinates and converted to
    the declared PushT canvas coordinates only at the outer interface.  The
    directed principal axis uses a third central moment to resolve the
    otherwise unavoidable pi ambiguity of a covariance eigenvector.  This is
    an observation helper, not a simulator-state readout.
    """
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return np.zeros(13, dtype=np.float32)
    points = np.column_stack([x_grid[ys, xs], y_grid[ys, xs]]).astype(np.float64)
    center = points.mean(axis=0)
    offsets = points - center[None]
    cxx = float(np.mean(offsets[:, 0] ** 2))
    cxy = float(np.mean(offsets[:, 0] * offsets[:, 1]))
    cyy = float(np.mean(offsets[:, 1] ** 2))
    cxxx = float(np.mean(offsets[:, 0] ** 3))
    cxxy = float(np.mean(offsets[:, 0] ** 2 * offsets[:, 1]))
    cxyy = float(np.mean(offsets[:, 0] * offsets[:, 1] ** 2))
    cyyy = float(np.mean(offsets[:, 1] ** 3))
    axis_angle = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    axis = np.asarray([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float64)
    third_along_axis = (
        cxxx * axis[0] ** 3
        + 3.0 * cxxy * axis[0] ** 2 * axis[1]
        + 3.0 * cxyy * axis[0] * axis[1] ** 2
        + cyyy * axis[1] ** 3
    )
    if third_along_axis < 0.0:
        axis = -axis
        axis_angle = math.atan2(float(axis[1]), float(axis[0]))
    return np.asarray([
        float(len(xs)) / float(mask.size),
        float(center[0]), float(center[1]),
        cxx, cxy, cyy, cxxx, cxxy, cxyy, cyyy,
        math.sin(axis_angle), math.cos(axis_angle), 1.0,
    ], dtype=np.float32)


def rgb_geometry_pose_chart(images: np.ndarray) -> dict[str, np.ndarray]:
    """Extract the shared image-derived geometry chart used by the ranker.

    ``images`` are the CHW uint8 tensors already consumed by the frozen
    encoder.  Current frames use the gray-blue block mask; target-only goal
    frames use the green target mask when available and fall back to the
    block mask.  Missing masks are represented explicitly and are filled by
    the caller's declared readout fallback, never by simulator state.
    """
    values = np.asarray(images)
    if values.ndim == 3:
        values = values[None]
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("RGB geometry chart expects images with shape [N, 3, H, W]")
    rgb = values.astype(np.float32)
    if float(np.max(rgb, initial=0.0)) > 1.5:
        rgb /= 255.0
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    agent_mask = (blue > 0.70) & (blue > green + 0.12) & (blue > red + 0.18)
    block_mask = (blue < 0.93) & (red < 0.82) & (blue > green + 0.025) & (green > red + 0.025)
    target_mask = (green > red + 0.14) & (green > blue + 0.14)
    height, width = values.shape[2:]
    x_grid = np.broadcast_to(
        2.0 * np.arange(width, dtype=np.float32)[None, :] / max(width - 1, 1) - 1.0,
        (height, width),
    )
    y_grid = np.broadcast_to(
        2.0 * np.arange(height, dtype=np.float32)[:, None] / max(height - 1, 1) - 1.0,
        (height, width),
    )
    agent_rows: list[np.ndarray] = []
    block_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for row in range(len(values)):
        agent_rows.append(_rgb_mask_chart(agent_mask[row], x_grid, y_grid))
        block_rows.append(_rgb_mask_chart(block_mask[row], x_grid, y_grid))
        target_rows.append(_rgb_mask_chart(target_mask[row], x_grid, y_grid))
    result = {
        "agent": np.asarray(agent_rows, dtype=np.float32),
        "block": np.asarray(block_rows, dtype=np.float32),
        "target": np.asarray(target_rows, dtype=np.float32),
    }
    if not all(np.isfinite(item).all() for item in result.values()):
        raise RuntimeError("non-finite RGB geometry pose chart")
    return result


def rgb_geometry_xy_angle_from_image(
    image: np.ndarray, role: str,
) -> tuple[np.ndarray | None, float | None]:
    """Return an RGB-only centroid and directed shape angle for one HWC frame."""
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("RGB geometry observer expects an HWC image")
    chart = rgb_geometry_pose_chart(np.moveaxis(values, -1, 0)[None])
    if role not in {"agent", "block", "target"}:
        raise ValueError("RGB geometry observer role must be agent, block, or target")
    selected = chart[role][0]
    if selected[-1] < 0.5:
        if role == "target":
            selected = chart["block"][0]
        if selected[-1] < 0.5:
            return None, None
    xy = ((selected[1:3] + 1.0) * (PUSHT_STATE_SIZE / 2.0)).astype(np.float32)
    angle = float(np.arctan2(selected[10], selected[11]))
    return xy, angle


def _rgb_block_mask_hwc(image: np.ndarray) -> np.ndarray:
    """Return the declared RGB-only PushT block mask for one HWC frame."""
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("RGB block mask expects an HWC image")
    rgb = values.astype(np.float32)
    if float(np.max(rgb, initial=0.0)) > 1.5:
        rgb /= 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (
        (blue < 0.93) & (red < 0.82)
        & (blue > green + 0.025) & (green > red + 0.025)
    )


def _rgb_centered_shape_patch(
    image: np.ndarray, patch_size: int = 64, crop_size: int = 88,
) -> tuple[np.ndarray | None, float]:
    """Build a translation-normalized block silhouette patch.

    The patch is intentionally based on the visible mask rather than latent
    features.  Centering by the visible centroid removes most pusher
    translation while retaining the T orientation.  The returned area ratio
    is used as a confidence signal because a heavily occluded mask should not
    drive a temporal orientation update.
    """
    mask = _rgb_block_mask_hwc(image)
    ys, xs = np.where(mask)
    if len(xs) < 12:
        return None, 0.0
    height, width = mask.shape
    cx, cy = float(xs.mean()), float(ys.mean())
    half = max(8, int(round(float(crop_size) / 2.0)))
    pad = np.pad(mask.astype(np.float32), ((half, half), (half, half)))
    px, py = int(round(cx)) + half, int(round(cy)) + half
    crop = pad[py - half:py + half, px - half:px + half]
    if crop.shape != (2 * half, 2 * half):
        return None, 0.0
    patch = cv2.resize(crop, (int(patch_size), int(patch_size)), interpolation=cv2.INTER_AREA)
    # The area is invariant to the later rotation search and is a useful
    # proxy for mask visibility under contact occlusion.
    area_ratio = float(np.mean(patch > 0.20))
    return patch.astype(np.float32), area_ratio


def rgb_temporal_orientation_increment(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    max_delta: float = 0.55,
    grid_step: float = 0.025,
    patch_size: int = 64,
    crop_size: int = 88,
) -> dict[str, float]:
    """Estimate a signed RGB-only orientation increment by silhouette registration.

    This is a local observer, not a successor model: it compares only the two
    rendered RGB frames already available to the controller.  The previous
    centered mask is rotated through a fixed, pre-declared small-angle grid
    and matched to the current mask.  The sign convention follows the image
    chart used by :func:`rgb_geometry_xy_angle_from_image` (positive y points
    down in the renderer).  A confidence margin and visible-area gate are
    returned so the runtime can fail closed under contact occlusion.
    """
    previous_patch, previous_area = _rgb_centered_shape_patch(
        previous, patch_size=patch_size, crop_size=crop_size,
    )
    current_patch, current_area = _rgb_centered_shape_patch(
        current, patch_size=patch_size, crop_size=crop_size,
    )
    empty = {
        "delta": 0.0, "confidence": 0.0, "best_score": 0.0,
        "second_score": 0.0, "visible_area": 0.0, "valid": 0.0,
    }
    if previous_patch is None or current_patch is None:
        return empty
    if min(previous_area, current_area) < 0.012:
        return {**empty, "visible_area": float(min(previous_area, current_area))}
    current_binary = current_patch > 0.20
    center = (0.5 * (patch_size - 1), 0.5 * (patch_size - 1))
    deltas = np.arange(
        -float(max_delta), float(max_delta) + 0.5 * float(grid_step),
        float(grid_step), dtype=np.float32,
    )
    scores: list[float] = []
    for delta in deltas:
        # OpenCV's positive angle is counter-clockwise in image coordinates;
        # the declared chart's positive angle is clockwise because y grows
        # downward, hence the minus sign.
        matrix = cv2.getRotationMatrix2D(
            center, -float(np.degrees(float(delta))), 1.0,
        )
        rotated = cv2.warpAffine(
            previous_patch, matrix, (patch_size, patch_size),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ) > 0.20
        intersection = float(np.logical_and(rotated, current_binary).sum())
        union = float(np.logical_or(rotated, current_binary).sum())
        scores.append(intersection / max(union, 1.0))
    order = np.argsort(np.asarray(scores))[::-1]
    best_index = int(order[0])
    best_score = float(scores[best_index])
    second_score = float(scores[int(order[1])]) if len(order) > 1 else 0.0
    margin = max(0.0, best_score - second_score)
    visible_area = float(min(previous_area, current_area))
    # Both absolute overlap and local peak separation are required.  This
    # prevents an occluded fragment with a flat score surface from creating a
    # confident but arbitrary torque direction.
    valid = bool(best_score >= 0.34 and margin >= 0.004 and visible_area >= 0.012)
    return {
        "delta": float(deltas[best_index]) if valid else 0.0,
        "confidence": float(margin),
        "best_score": best_score,
        "second_score": second_score,
        "visible_area": visible_area,
        "valid": float(valid),
    }


def temporal_rgb_angle_observation(
    image: np.ndarray,
    previous: np.ndarray,
    calibrated_current_angle: float,
    calibrated_previous_angle: float,
    state: dict[str, Any],
    *,
    max_delta: float = 0.55,
    min_confidence: float = 0.004,
    correction_gain: float = 0.10,
) -> tuple[float, dict[str, float]]:
    """Track a calibrated RGB angle with conservative temporal increments."""
    current_angle = float(calibrated_current_angle)
    previous_angle = float(calibrated_previous_angle)
    if not np.isfinite(current_angle) or not np.isfinite(previous_angle):
        current_angle = float(state.get("angle", 0.0))
        previous_angle = current_angle
    registration = rgb_temporal_orientation_increment(
        previous, image, max_delta=max_delta,
    )
    initialized = bool(state.get("initialized", False))
    if not initialized:
        selected = current_angle
        state["initialized"] = True
        state["angle"] = selected
        used = 0.0
    elif float(registration["valid"]) > 0.0 and float(registration["confidence"]) >= float(min_confidence):
        selected = float(wrap(float(state.get("angle", previous_angle)) + registration["delta"]))
        # A small calibrated correction controls integration drift without
        # allowing a contact-occluded KNN branch flip to overwrite the track.
        residual = float(wrap(current_angle - selected))
        if abs(residual) <= 0.35:
            selected = float(wrap(selected + float(correction_gain) * residual))
        state["angle"] = selected
        used = 1.0
    else:
        selected = float(state.get("angle", previous_angle))
        state["angle"] = selected
        used = 0.0
    audit = {
        "temporal_angle": float(selected),
        "temporal_angle_delta": float(registration["delta"]),
        "temporal_angle_confidence": float(registration["confidence"]),
        "temporal_angle_best_score": float(registration["best_score"]),
        "temporal_angle_second_score": float(registration["second_score"]),
        "temporal_angle_visible_area": float(registration["visible_area"]),
        "temporal_angle_update_used": float(used),
        "temporal_angle_branch_flip": float(
            abs(float(wrap(current_angle - previous_angle))) > 1.20
        ),
    }
    state["audit"] = audit
    return selected, audit


def paired_rgb_current_xy_from_images(
    current_image: np.ndarray,
    goal_image: np.ndarray,
    goal_xy: np.ndarray,
    current_angle_override: float | None = None,
    target_angle_override: float | None = None,
) -> np.ndarray | None:
    """Estimate current block XY from a paired rigid-shape chart.

    Absolute centroid-to-body calibration is brittle across PushT shape and
    scale variations and under contact occlusion.  The target image is an
    isolated render of the same rigid object, so its centroid-to-body vector
    can be transported to the current image using the relative directed
    shape angle.  The unknown renderer angle offset cancels in the difference
    ``current_chart_angle - goal_chart_angle``.  This is a visual pairwise
    geometry readout; it never consumes simulator state or a successor.
    """
    current_xy, current_angle = rgb_geometry_xy_angle_from_image(current_image, "block")
    target_xy, target_angle = rgb_geometry_xy_angle_from_image(goal_image, "target")
    if current_xy is None or target_xy is None or current_angle is None or target_angle is None:
        return None
    goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
    if goal_xy.shape[0] < 2 or not np.isfinite(goal_xy[:2]).all():
        return None
    if current_angle_override is not None and target_angle_override is not None:
        angle_delta = float(wrap(float(current_angle_override) - float(target_angle_override)))
    else:
        angle_delta = float(wrap(float(current_angle) - float(target_angle)))
    offset = np.asarray(goal_xy[:2], dtype=np.float32) - np.asarray(target_xy, dtype=np.float32)
    cosine, sine = float(np.cos(angle_delta)), float(np.sin(angle_delta))
    transported = np.asarray([
        cosine * offset[0] - sine * offset[1],
        sine * offset[0] + cosine * offset[1],
    ], dtype=np.float32)
    result = np.asarray(current_xy, dtype=np.float32) + transported
    return result if np.isfinite(result).all() else None


def observed_block_xy(
    image: np.ndarray,
    features: np.ndarray,
    xy_decoder: OrientationDecoder | None,
    xy_decoder_training: dict[str, Any] | None,
    device: str,
    policy: str,
    disagreement_threshold: float,
) -> np.ndarray:
    """Read block XY with the same observer used for the current frame.

    A previous version mixed a current-frame RGB-mask readout with a
    previous-frame latent-moment readout.  The resulting finite difference was
    not a physical motion estimate: even identical frames could appear to
    move by tens of pixels, causing the state machine to skip approach and
    enter learned pushing.  This helper keeps current and previous reads in
    one declared coordinate observer.
    """
    features = np.asarray(features, dtype=np.float32)
    raw_xy = (features[BLOCK_FEATURE_START + 1:BLOCK_FEATURE_START + 3] + 1.0) * (
        PUSHT_STATE_SIZE / 2.0
    )
    if policy == "rgb_mask":
        mask_xy, _ = rgb_geometry_xy_angle_from_image(image, "block")
        return raw_xy if mask_xy is None else np.asarray(mask_xy, dtype=np.float32)
    if policy == "rgb_calibrated":
        if not isinstance(xy_decoder, (LinearRGBPoseCalibrator, RGBAngleKNN)) or xy_decoder_training is None:
            # A requested calibration that fails its held-out gate must not
            # crash the run or silently fabricate coordinates.  Fall back to
            # the direct RGB mask, which is the declared conservative
            # observer used by the earlier protocol.
            mask_xy, _ = rgb_geometry_xy_angle_from_image(image, "block")
            return raw_xy if mask_xy is None else np.asarray(mask_xy, dtype=np.float32)
        shape = rgb_geometry_shape_features(
            np.moveaxis(np.asarray(image), -1, 0)[None], "block",
        )
        calibrated_xy = predict_orientation(xy_decoder, shape, xy_decoder_training, device)[0]
        # A static root-disjoint calibrator can extrapolate badly after the
        # pusher occludes the T or leaves the rendered canvas.  Its mean
        # validation error does not expose these rare but catastrophic
        # jumps; using one would fabricate a large block displacement and
        # make the recovery state machine alternate actions.  Keep the
        # calibrated chart only when it agrees with the simultaneously
        # available raw RGB mask.
        mask_xy, _ = rgb_geometry_xy_angle_from_image(image, "block")
        if mask_xy is None or not np.isfinite(calibrated_xy).all():
            return raw_xy if mask_xy is None else np.asarray(mask_xy, dtype=np.float32)
        if float(np.linalg.norm(calibrated_xy - mask_xy)) > float(disagreement_threshold):
            return np.asarray(mask_xy, dtype=np.float32)
        return calibrated_xy.astype(np.float32)
    if policy == "paired_rgb_calibrated":
        # The paired observer requires the goal image and is assembled in
        # ``runtime_observation``.  For temporal finite differences and
        # online local-effect identification, use its fail-closed RGB
        # centroid fallback rather than silently switching to the frozen
        # latent chart.
        mask_xy, _ = rgb_geometry_xy_angle_from_image(image, "block")
        return raw_xy if mask_xy is None else np.asarray(mask_xy, dtype=np.float32)
    if xy_decoder is None or xy_decoder_training is None or policy == "raw":
        return raw_xy.astype(np.float32)
    decoded_xy = predict_orientation(
        xy_decoder, features, xy_decoder_training, device,
    )[0]
    if policy == "rgb_mask_fused":
        mask_xy, _ = rgb_geometry_xy_angle_from_image(image, "block")
        if mask_xy is None:
            return decoded_xy.astype(np.float32)
        disagreement = float(np.linalg.norm(decoded_xy - mask_xy))
        if disagreement > float(disagreement_threshold):
            # Large disagreement is treated as an occlusion/mask failure
            # candidate.  The runtime-safe fallback is the direct RGB mask,
            # not the decoder: dynamic-frame validation showed that the
            # decoder can have large outliers exactly during contact.  Close
            # estimates are averaged to suppress single-pixel centroid
            # jitter, while a large disagreement remains fail-closed.
            return np.asarray(mask_xy, dtype=np.float32)
        return (0.5 * (decoded_xy + np.asarray(mask_xy, dtype=np.float32))).astype(np.float32)
    if policy == "validation":
        return decoded_xy.astype(np.float32)
    if policy == "disagreement_gate":
        return (
            decoded_xy if np.linalg.norm(decoded_xy - raw_xy) <= disagreement_threshold
            else raw_xy
        ).astype(np.float32)
    raise ValueError(f"unknown current XY observer policy: {policy}")


def safe_agent_xy_observation(
    candidate: np.ndarray,
    raw_xy: np.ndarray,
    held_agent_xy: np.ndarray | None = None,
    max_step: float = 96.0,
    workspace_margin: float = 32.0,
) -> np.ndarray:
    """Fail closed on impossible RGB/decoder pusher observations.

    Dynamic contact frames can make a learned pusher readout extrapolate far
    outside the 512-pixel PushT workspace.  Feeding that value to the contact
    state machine turns an observation error into a recovery trajectory.  The
    guard is observation-only: it uses the previous accepted RGB estimate (if
    available), never simulator state or a predicted successor.
    """
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    raw = np.asarray(raw_xy, dtype=np.float32).reshape(-1)
    held = None if held_agent_xy is None else np.asarray(
        held_agent_xy, dtype=np.float32,
    ).reshape(-1)
    def valid(value: np.ndarray | None) -> bool:
        return bool(
            value is not None
            and value.shape[0] >= 2
            and np.isfinite(value[:2]).all()
            and np.all(value[:2] >= -float(workspace_margin))
            and np.all(value[:2] <= PUSHT_STATE_SIZE + float(workspace_margin))
        )
    if not valid(candidate):
        if valid(held):
            return held[:2].copy()
        return raw[:2].copy()
    if valid(held) and float(max_step) > 0.0:
        if float(np.linalg.norm(candidate[:2] - held[:2])) > float(max_step):
            return held[:2].copy()
    return candidate[:2].copy()


def observed_agent_xy(
    image: np.ndarray,
    features: np.ndarray,
    agent_xy_decoder: OrientationDecoder | None,
    agent_xy_decoder_training: dict[str, Any] | None,
    device: str,
    policy: str,
    disagreement_threshold: float,
    held_agent_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Read agent XY in the same declared chart as the block observer.

    The contact controller uses agent--block distance, contact normal, side
    alignment, reacquisition, and terminal confirmation together.  Letting
    the agent remain in the frozen latent-moment chart while the block is
    decoded by a learned chart creates a hidden affine gauge mismatch.  This
    helper deliberately mirrors :func:`observed_block_xy` for the agent
    coordinates, including conservative RGB fallbacks.
    """
    features = np.asarray(features, dtype=np.float32)
    raw_xy = (features[1:3] + 1.0) * (PUSHT_STATE_SIZE / 2.0)
    if policy in {"rgb_mask", "rgb_calibrated", "paired_rgb_calibrated"}:
        mask_xy, _ = rgb_geometry_xy_angle_from_image(image, "agent")
        if mask_xy is not None:
            return np.asarray(mask_xy, dtype=np.float32)
        # A fully occluded pusher used to fall back to the latent moment
        # sentinel [256, 256].  Recovery then interpreted that sentinel as a
        # real position and repeatedly drove the agent toward the workspace
        # boundary.  Holding the last accepted RGB position is a bounded
        # observation fallback; it does not query simulator state or predict
        # a successor.
        held = None if held_agent_xy is None else np.asarray(
            held_agent_xy, dtype=np.float32,
        ).reshape(-1)
        if held is not None and held.shape[0] >= 2 and np.isfinite(held[:2]).all():
            return held[:2].copy()
        return raw_xy.astype(np.float32)
    if agent_xy_decoder is None or agent_xy_decoder_training is None or policy == "raw":
        return safe_agent_xy_observation(raw_xy, raw_xy, held_agent_xy=held_agent_xy)
    decoded_xy = predict_orientation(
        agent_xy_decoder, features, agent_xy_decoder_training, device,
    )[0]
    if policy == "rgb_mask_fused":
        mask_xy, _ = rgb_geometry_xy_angle_from_image(image, "agent")
        if mask_xy is None:
            return decoded_xy.astype(np.float32)
        if float(np.linalg.norm(decoded_xy - mask_xy)) > float(disagreement_threshold):
            return safe_agent_xy_observation(mask_xy, raw_xy, held_agent_xy=held_agent_xy)
        return safe_agent_xy_observation(
            0.5 * (decoded_xy + np.asarray(mask_xy, dtype=np.float32)),
            raw_xy, held_agent_xy=held_agent_xy,
        )
    if policy == "validation":
        return safe_agent_xy_observation(decoded_xy, raw_xy, held_agent_xy=held_agent_xy)
    if policy == "disagreement_gate":
        selected = (
            decoded_xy if np.linalg.norm(decoded_xy - raw_xy) <= disagreement_threshold
            else raw_xy
        )
        return safe_agent_xy_observation(selected, raw_xy, held_agent_xy=held_agent_xy)
    raise ValueError(f"unknown current XY observer policy: {policy}")


def temporally_filtered_block_xy(
    image: np.ndarray,
    previous: np.ndarray,
    features: np.ndarray,
    previous_features: np.ndarray,
    xy_decoder: OrientationDecoder | None,
    xy_decoder_training: dict[str, Any] | None,
    device: str,
    policy: str,
    disagreement_threshold: float,
    max_step: float,
) -> tuple[np.ndarray, float]:
    """Apply a bounded-motion guard to the declared RGB block observer.

    The guard is intentionally not a tracker or a successor model: it only
    rejects a single frame-to-frame jump that is outside the declared visual
    observation budget.  This prevents contact occlusion from turning a
    centroid/calibrator failure into a false terminal or recovery transition.
    """
    current = observed_block_xy(
        image, features, xy_decoder, xy_decoder_training, device,
        policy, disagreement_threshold,
    )
    prior = observed_block_xy(
        previous, previous_features, xy_decoder, xy_decoder_training, device,
        policy, disagreement_threshold,
    )
    delta = np.asarray(current, dtype=np.float32) - np.asarray(prior, dtype=np.float32)
    magnitude = float(np.linalg.norm(delta))
    limit = float(max_step)
    if limit > 0.0 and np.isfinite(magnitude) and magnitude > limit:
        current = np.asarray(prior, dtype=np.float32) + (
            delta / max(magnitude, 1.0e-6) * limit
        ).astype(np.float32)
    return np.asarray(current, dtype=np.float32), magnitude


def rgb_geometry_shape_features(images: np.ndarray, role: str) -> np.ndarray:
    """Return the raw RGB shape chart used by the angle calibration readout."""
    chart = rgb_geometry_pose_chart(images)
    if role not in {"block", "target"}:
        raise ValueError("RGB shape feature role must be block or target")
    selected = chart[role].copy()
    if role == "target":
        missing = selected[:, -1] < 0.5
        if np.any(missing):
            selected[missing] = chart["block"][missing]
    return selected[:, :12].astype(np.float32)


def rgb_pose_calibration_features(shape_features: np.ndarray) -> np.ndarray:
    """Build the fixed chart used to undo the T-mask pose offset.

    The first two coordinates are the mask centroid, the next two encode the
    directed principal axis, and the remaining moments capture the
    orientation-dependent asymmetric offset.  The second/third harmonics
    retain the dominant periodic renderer bias.  We intentionally do not
    append all axis--moment products: a calibration split can be smaller than
    the raw moment chart, and an over-complete linear chart would make the
    declared root-disjoint readout non-identifiable.  This fixed 16-column
    chart is therefore a protocol invariant, not a post-hoc regularization
    choice; no simulator state or runtime successor is used.  The intercept
    is kept unstandardized by the fitting routine.
    """
    values = np.asarray(shape_features, dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] < 12:
        raise ValueError("RGB pose calibration expects the 12-dimensional shape chart")
    centroid = values[:, 1:3]
    axis = values[:, 10:12]
    sin_theta, cos_theta = axis[:, 0], axis[:, 1]
    harmonics = np.column_stack([
        2.0 * sin_theta * cos_theta,
        cos_theta * cos_theta - sin_theta * sin_theta,
        sin_theta * (3.0 * cos_theta * cos_theta - sin_theta * sin_theta),
        cos_theta * (cos_theta * cos_theta - 3.0 * sin_theta * sin_theta),
    ])
    moments = values[:, 3:10]
    result = np.column_stack([
        np.ones(len(values), dtype=np.float32),
        centroid,
        axis,
        harmonics,
        moments,
    ]).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite RGB pose calibration features")
    return result


def rgb_shape_raster_features(images: np.ndarray, role: str) -> np.ndarray:
    """Extract a translation-normalized high-resolution object-mask patch.

    The compact 64x64 moment chart aliases several T orientations.  This
    probe keeps a 16x16 crop from the original rendered RGB image, centered
    by the detected object centroid.  It is used only for the current-angle
    readout and consumes pixels only; simulator state is never an input.
    """
    values = np.asarray(images)
    if values.ndim == 3:
        if values.shape[-1] == 3:
            values = values[None]
        elif values.shape[0] == 3:
            values = np.moveaxis(values[None], 1, -1)
        else:
            raise ValueError("RGB raster images must be HWC or CHW")
    elif values.ndim == 4 and values.shape[1] == 3:
        values = np.moveaxis(values, 1, -1)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("RGB raster images must have shape [N,H,W,3] or [N,3,H,W]")
    if role not in {"block", "target"}:
        raise ValueError("RGB raster role must be block or target")
    output: list[np.ndarray] = []
    for image in values:
        rgb = image.astype(np.float32)
        if float(np.max(rgb, initial=0.0)) > 1.5:
            rgb /= 255.0
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        block_mask = (
            (blue < 0.93) & (red < 0.82)
            & (blue > green + 0.025) & (green > red + 0.025)
        )
        target_mask = (green > red + 0.14) & (green > blue + 0.14)
        mask = target_mask if role == "target" else block_mask
        if role == "target" and int(mask.sum()) < 3:
            mask = block_mask
        ys, xs = np.where(mask)
        if len(xs) < 3:
            output.append(np.zeros(16 * 16, dtype=np.float32))
            continue
        height, width = mask.shape
        crop_width = max(16, int(round(72.0 * min(height, width) / 224.0)))
        pad = np.pad(mask.astype(np.float32), ((crop_width, crop_width),) * 2)
        cx, cy = float(xs.mean()), float(ys.mean())
        px = int(round(cx)) + crop_width
        py = int(round(cy)) + crop_width
        half = crop_width // 2
        crop = pad[py - half:py + half, px - half:px + half]
        output.append(cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1))
    result = np.asarray(output, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite RGB raster angle features")
    return result


def runtime_rgb_angle_features(
    image: np.ndarray, role: str, training: dict[str, Any] | None,
    image_size: int,
) -> np.ndarray:
    """Build the exact RGB angle interface declared during calibration."""
    if training is not None and training.get("feature_interface") == "raster":
        return rgb_shape_raster_features(np.asarray(image)[None], role)
    return rgb_geometry_shape_features(resize_image(image, image_size)[None], role)


def rgb_geometry_ranker_features(
    current_images: np.ndarray,
    previous_images: np.ndarray,
    goal_images: np.ndarray,
    actions: np.ndarray,
    fallback_current_angles: np.ndarray,
    fallback_goal_poses: np.ndarray,
    feature_mode: str = "full",
) -> np.ndarray:
    """Build the ranker interface entirely from current/previous/goal RGB.

    This is intentionally separate from ``geometry_ranker_features``.  The
    latter is retained as the latent-geometry negative control; this function
    is the causal interface candidate and is called identically during
    offline training, held-out validation, and runtime selection.
    """
    current = rgb_geometry_pose_chart(current_images)
    previous = rgb_geometry_pose_chart(previous_images)
    goal = rgb_geometry_pose_chart(goal_images)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    count = len(actions)
    if not (len(current["agent"]) == len(previous["agent"]) == len(goal["target"]) == count):
        raise ValueError("RGB ranker inputs must have the same row count")
    fallback_current_angles = np.asarray(fallback_current_angles, dtype=np.float32).reshape(-1)
    fallback_goal_poses = np.asarray(fallback_goal_poses, dtype=np.float32)
    if fallback_goal_poses.ndim == 1:
        fallback_goal_poses = fallback_goal_poses[None]
    if len(fallback_current_angles) != count or len(fallback_goal_poses) != count:
        raise ValueError("RGB ranker fallback rows must match actions")

    scale = PUSHT_STATE_SIZE / 2.0
    agent = current["agent"][:, 1:3] * scale
    block = current["block"][:, 1:3] * scale
    previous_agent = previous["agent"][:, 1:3] * scale
    previous_block = previous["block"][:, 1:3] * scale
    # PushT's ``goal`` render is a centered target-shape image, not a scene
    # rendered at the target world coordinates.  Its target-mask centroid is
    # therefore not a valid world-space XY readout.  Use the declared,
    # root-disjoint RGB-derived goal pose for XY; retain the target chart as
    # shape context and as a separately audited angle fallback.
    goal_object = goal["target"].copy()
    target_missing = goal_object[:, -1] < 0.5
    if np.any(target_missing):
        goal_object[target_missing] = goal["block"][target_missing]
    goal_xy = fallback_goal_poses[:, :2].astype(np.float32)
    current_shape = current["block"]
    goal_shape = goal_object
    current_missing = current_shape[:, -1] < 0.5
    # The principal-axis angle is retained as an input feature, but the
    # declared fallback angles are preferred for the control chart.  They can
    # be a root-disjoint RGB-shape calibration readout, which resolves the
    # renderer-specific fixed offset and pi ambiguity without exposing state.
    current_angle = fallback_current_angles.copy()
    current_angle[current_missing] = np.arctan2(
        current_shape[current_missing, 10], current_shape[current_missing, 11],
    )
    goal_angle = np.arctan2(fallback_goal_poses[:, 2], fallback_goal_poses[:, 3]).astype(np.float32)
    goal_missing = goal_shape[:, -1] < 0.5
    goal_angle[goal_missing] = np.arctan2(
        goal_shape[goal_missing, 10], goal_shape[goal_missing, 11],
    )
    relative = block - agent
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    normal = relative / np.maximum(distance, 1.0e-6)
    tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
    angle_error = wrap(goal_angle - current_angle)
    action_normal = np.sum(actions * normal, axis=1, keepdims=True)
    action_tangent = np.sum(actions * tangent, axis=1, keepdims=True)
    action_magnitude = np.linalg.norm(actions, axis=1, keepdims=True)
    current_boundary = contact_boundary_chart_from_images(current_images, actions)
    current_boundary = current_boundary * np.asarray([scale, scale, scale, scale, scale, scale, scale, scale, 1.0, 1.0], dtype=np.float32)
    if feature_mode not in {"full", "invariant"}:
        raise ValueError(f"unsupported RGB ranker feature mode: {feature_mode!r}")
    if feature_mode == "invariant":
        # Contact-frame chart: remove absolute canvas coordinates while
        # retaining the quantities that determine a local planar intervention.
        # The visible support lever and its action moment are observation-only
        # RGB features; no simulator successor is used here.
        # ``current_boundary[:2]`` is the absolute contact-point location and
        # must not be included: retaining it would make this nominally
        # invariant chart depend on the canvas origin.  The remaining chart
        # is block-relative (lever/support extents, gap, and moment).
        boundary_relative = np.concatenate([
            current_boundary[:, 2:4], current_boundary[:, 4:]], axis=1,
        )
        relative_motion = (
            (block - previous_block) - (agent - previous_agent)
        ) / PUSHT_STATE_SIZE
        lever = current_boundary[:, 2:4]
        action_moment = (
            lever[:, 0] * actions[:, 1] - lever[:, 1] * actions[:, 0]
        )[:, None]
        angle_moment = np.concatenate([
            np.sin(angle_error)[:, None] * action_moment / PUSHT_STATE_SIZE,
            np.cos(angle_error)[:, None] * action_moment / PUSHT_STATE_SIZE,
        ], axis=1)
        invariant = np.concatenate([
            relative / PUSHT_STATE_SIZE,
            distance / PUSHT_STATE_SIZE,
            np.sin(angle_error)[:, None], np.cos(angle_error)[:, None],
            actions, action_normal, action_tangent, action_magnitude,
            relative_motion,
            current_shape[:, 3:], goal_shape[:, 3:],
            boundary_relative / PUSHT_STATE_SIZE,
            action_moment / PUSHT_STATE_SIZE, angle_moment,
            np.column_stack([
                actions[:, 0] * current_boundary[:, 2],
                actions[:, 1] * current_boundary[:, 2],
                actions[:, 0] * current_boundary[:, 3],
                actions[:, 1] * current_boundary[:, 3],
            ]),
        ], axis=1).astype(np.float32)
        if not np.isfinite(invariant).all():
            raise RuntimeError("non-finite invariant RGB ranker features")
        return invariant
    explicit = np.concatenate([
        agent / PUSHT_STATE_SIZE, block / PUSHT_STATE_SIZE, goal_xy / PUSHT_STATE_SIZE,
        (goal_xy - block) / PUSHT_STATE_SIZE, relative / PUSHT_STATE_SIZE,
        distance / PUSHT_STATE_SIZE,
        np.sin(current_angle)[:, None], np.cos(current_angle)[:, None],
        np.sin(goal_angle)[:, None], np.cos(goal_angle)[:, None],
        np.sin(angle_error)[:, None], np.cos(angle_error)[:, None],
        actions, action_normal, action_tangent, action_magnitude,
        current_shape, goal_shape,
        (block - previous_block) / PUSHT_STATE_SIZE,
        (agent - previous_agent) / PUSHT_STATE_SIZE,
        current_boundary / PUSHT_STATE_SIZE,
        np.column_stack([
            actions[:, 0] * current_boundary[:, 2],
            actions[:, 1] * current_boundary[:, 2],
            actions[:, 0] * current_boundary[:, 3],
            actions[:, 1] * current_boundary[:, 3],
        ]),
    ], axis=1).astype(np.float32)
    if not np.isfinite(explicit).all():
        raise RuntimeError("non-finite RGB geometry ranker features")
    return explicit


def contact_physics_features(features: np.ndarray) -> np.ndarray:
    """Build a low-dimensional, RGB-only contact mechanics chart.

    The frozen feature layout is [current moments, temporal difference, action,
    contact chart, boundary chart].  This map exposes the quantities a local
    planar contact model can actually identify: normal/tangent action,
    visible lever arm, contact point, and their bilinear torque terms.  It is
    used only for the optional ridge specialist; all geometry is obtained
    from images, never from simulator state.
    """
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] < 82:
        raise ValueError("contact physics features require the 82-dimensional RGB adapter chart")
    action = values[:, 60:62]
    contact = values[:, 62:72]
    boundary = values[:, 72:82]
    lever = boundary[:, 2:4]
    contact_point = boundary[:, 0:2]
    torque = (lever[:, 0] * action[:, 1] - lever[:, 1] * action[:, 0])[:, None]
    bilinear = np.column_stack([
        lever[:, 0] * action[:, 0], lever[:, 0] * action[:, 1],
        lever[:, 1] * action[:, 0], lever[:, 1] * action[:, 1],
        contact_point[:, 0] * action[:, 0], contact_point[:, 1] * action[:, 1],
    ])
    return np.concatenate([
        action, contact[:, 2:], boundary, bilinear, torque,
        np.square(action), np.linalg.norm(action, axis=1, keepdims=True),
    ], axis=1).astype(np.float32)


def attach_targets(data: dict[str, np.ndarray], actions: np.ndarray) -> dict[str, np.ndarray]:
    """Construct root-matched zero baseline and causal action-effect labels."""
    zero_index = int(np.flatnonzero(np.all(np.isclose(actions, 0.0), axis=1))[0])
    zero_successors = np.zeros((len(data["root_id"]), 6), dtype=np.float32)
    target = np.zeros((len(data["root_id"]), 2 * POSE_DIM), dtype=np.float32)
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        if len(indices) != len(actions):
            raise RuntimeError("unbalanced intervention bank")
        zero = data["next_state"][indices[zero_index]]
        current = data["state"][indices[zero_index]]
        zero_successors[indices] = zero
        zero_block = block_pose(zero)
        current_block = block_pose(current)
        zero_delta = zero_block - current_block
        zero_delta[2] = float(wrap(zero_delta[2]))
        for index in indices:
            action_block = block_pose(data["next_state"][index])
            effect = action_block - zero_block
            effect[2] = float(wrap(effect[2]))
            target[index, :POSE_DIM] = zero_delta
            target[index, POSE_DIM:] = effect
    out = dict(data)
    out["zero_successor"] = zero_successors
    out["local_target"] = target
    contact_target = target.copy()
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        source = data["state"][indices[0]]
        relative = np.asarray(source[2:4] - source[:2], dtype=np.float64)
        distance = max(float(np.linalg.norm(relative)), 1e-6)
        normal = relative / distance
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        world_effect = target[indices, POSE_DIM:POSE_DIM + 2].astype(np.float64)
        contact_target[indices, POSE_DIM] = world_effect @ normal
        contact_target[indices, POSE_DIM + 1] = world_effect @ tangent
    out["local_target_contact"] = contact_target
    out["action_zero_index"] = np.asarray([zero_index], dtype=np.int64)
    if not np.isfinite(target).all():
        raise RuntimeError("non-finite local-effect target")
    return out


class LocalEffectAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2 * POSE_DIM),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class StructuredContactAdapter(nn.Module):
    """Closed-form linear readout over the explicit contact mechanics chart."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2 * POSE_DIM)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class GoalPoseDecoder(nn.Module):
    """Independent decoder for the rendered target image."""

    def __init__(self, input_dim: int = 30, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class OrientationDecoder(nn.Module):
    def __init__(self, input_dim: int = 30, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LinearRGBPoseCalibrator:
    """Root-disjoint linear calibration for a rendered target pose.

    The target mask is a translated, asymmetric T shape.  Its centroid is
    therefore not the simulator pose, and its PCA angle is not the simulator
    angle.  A small declared linear chart is sufficient to undo this fixed
    rendering geometry, whereas the previous MLP was unnecessarily
    high-variance on the small calibration split.  This object deliberately
    has no simulator-state access; it is only a pixel-chart readout.
    """

    def __init__(self, coefficients: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray):
        coefficients = np.asarray(coefficients, dtype=np.float32)
        if coefficients.ndim != 2 or coefficients.shape[1] != 2:
            raise ValueError("RGB pose calibrator coefficients must have shape [features, 2]")
        self.coefficients = coefficients
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        design = rgb_pose_calibration_features(values)
        if design.shape[1] != len(self.feature_mean):
            raise ValueError("RGB pose calibrator feature width mismatch")
        normalized = design.copy()
        normalized[:, 1:] = (
            normalized[:, 1:] - self.feature_mean[1:]
        ) / self.feature_std[1:]
        result = normalized @ self.coefficients
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite RGB pose calibration output")
        return result.astype(np.float32)

    def eval(self) -> "LinearRGBPoseCalibrator":
        return self


class LinearFeatureXYDecoder:
    """Root-disjoint affine decoder for a frozen feature chart.

    The mask-moment encoder exposes agent/block centroids nearly linearly.
    A high-capacity MLP can overfit the small root split and then distort one
    member of the agent--block pair.  This low-capacity decoder is learned
    from the same frozen features and labels as the nonlinear readout, but
    has an explicit ridge solution and therefore preserves the coordinate
    chart across dynamic validation frames.
    """

    def __init__(
        self, coefficients: np.ndarray, feature_mean: np.ndarray,
        feature_std: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray,
    ):
        self.coefficients = np.asarray(coefficients, dtype=np.float32)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        if self.coefficients.ndim != 2 or self.coefficients.shape[1] != 2:
            raise ValueError("linear feature XY coefficients must have shape [features+1, 2]")

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        normalized = (values - self.feature_mean) / self.feature_std
        design = np.concatenate([
            np.ones((len(normalized), 1), dtype=np.float32), normalized,
        ], axis=1)
        result = design @ self.coefficients
        result = result * self.target_std + self.target_mean
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite linear feature XY decoder output")
        return result.astype(np.float32)

    def eval(self) -> "LinearFeatureXYDecoder":
        return self


def readout_state_dict(model: Any) -> dict[str, Any] | None:
    """Serialize torch and closed-form RGB readouts through one interface."""
    if model is None:
        return None
    if hasattr(model, "state_dict"):
        return model.state_dict()
    if isinstance(model, LinearRGBPoseCalibrator):
        return {
            "coefficients": model.coefficients,
            "feature_mean": model.feature_mean,
            "feature_std": model.feature_std,
        }
    if isinstance(model, LinearFeatureXYDecoder):
        return {
            "coefficients": model.coefficients,
            "feature_mean": model.feature_mean,
            "feature_std": model.feature_std,
            "target_mean": model.target_mean,
            "target_std": model.target_std,
        }
    raise TypeError(f"unsupported readout type: {type(model)!r}")


class RGBAngleKNN(nn.Module):
    """Support-constrained RGB angle readout for high-resolution shape masks."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, k: int = 3):
        super().__init__()
        if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
            raise ValueError("RGB angle KNN arrays have incompatible shapes")
        if k < 1 or k > len(features):
            raise ValueError("RGB angle KNN k must be in [1, n]")
        self.register_buffer("support", torch.as_tensor(features, dtype=torch.float32))
        self.register_buffer("values", torch.as_tensor(targets, dtype=torch.float32))
        self.k = int(k)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(features, self.support)
        nearest = torch.topk(distances, k=self.k, largest=False, dim=1).indices
        return self.values[nearest].mean(dim=1)


class ActionCostRanker(nn.Module):
    """RGB-only candidate cost readout trained on matched offline outcomes."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ContactModeClassifier(nn.Module):
    """Predict the offline contact mode from RGB-derived state/action features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def contact_mode_labels(data: dict[str, np.ndarray]) -> np.ndarray:
    """Create offline-only PushT contact-mode labels from matched outcomes."""
    source = np.asarray(data["state"], dtype=np.float32)
    successor = np.asarray(data["next_state"], dtype=np.float32)
    start_distance = np.linalg.norm(source[:, :2] - source[:, 2:4], axis=1)
    end_distance = np.linalg.norm(successor[:, :2] - successor[:, 2:4], axis=1)
    block_motion = np.linalg.norm(successor[:, 2:4] - source[:, 2:4], axis=1)
    labels = np.zeros(len(source), dtype=np.int64)
    near = start_distance <= 65.0
    labels[near & (end_distance <= 80.0) & (block_motion <= 1.0)] = 1
    labels[near & (end_distance <= 80.0) & (block_motion > 1.0)] = 2
    labels[near & (end_distance > 80.0)] = 3
    return labels


def train_contact_mode_classifier(
    train_features: np.ndarray, train_labels: np.ndarray,
    val_features: np.ndarray, val_labels: np.ndarray, args: argparse.Namespace,
) -> tuple[ContactModeClassifier, dict[str, Any]]:
    """Fit a mode head with class-balanced offline supervision."""
    torch.manual_seed(args.seed + 131)
    feature_mean, feature_std = standardize_fit(train_features)
    train_x = (train_features - feature_mean) / feature_std
    val_x = (val_features - feature_mean) / feature_std
    model = ContactModeClassifier(train_x.shape[1], args.hidden_dim).to(args.device)
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=4).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=args.device))
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(args.seed + 132)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_x))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            x = torch.as_tensor(train_x[index], dtype=torch.float32, device=args.device)
            y = torch.as_tensor(train_labels[index], dtype=torch.long, device=args.device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(
                model(torch.as_tensor(val_x, dtype=torch.float32, device=args.device)),
                torch.as_tensor(val_labels, dtype=torch.long, device=args.device),
            ).cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_value:
            best_value = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("contact-mode classifier produced no checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "best_val_loss": best_value, "class_counts": counts.tolist(),
        "class_weights": class_weights.tolist(), "history": history,
    }


@torch.inference_mode()
def predict_contact_mode(
    model: ContactModeClassifier, features: np.ndarray, training: dict[str, Any], device: str,
) -> np.ndarray:
    mean = np.asarray(training["feature_mean"], dtype=np.float32)
    std = np.asarray(training["feature_std"], dtype=np.float32)
    x = torch.as_tensor((np.asarray(features, dtype=np.float32) - mean) / std,
                        dtype=torch.float32, device=device)
    return torch.softmax(model(x), dim=-1).cpu().numpy().astype(np.float32)


def standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=np.float32).mean(axis=0)
    std = np.maximum(np.asarray(values, dtype=np.float32).std(axis=0), 1e-4)
    return mean, std


def train_adapter(
    train_features: np.ndarray, train_targets: np.ndarray,
    val_features: np.ndarray, val_targets: np.ndarray, train_sample_weights: np.ndarray,
    args: argparse.Namespace,
) -> tuple[LocalEffectAdapter, dict[str, Any]]:
    torch.manual_seed(args.seed)
    feature_mean, feature_std = standardize_fit(train_features)
    target_mean, target_std = standardize_fit(train_targets)
    train_x = (train_features - feature_mean) / feature_std
    val_x = (val_features - feature_mean) / feature_std
    train_y = (train_targets - target_mean) / target_std
    val_y = (val_targets - target_mean) / target_std
    model = LocalEffectAdapter(train_x.shape[1], args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 1)
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    weights = torch.as_tensor([1.0, 1.0, 2.0, 1.0, 1.0, 2.0], dtype=torch.float32, device=args.device)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_x))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            x = torch.as_tensor(train_x[index], dtype=torch.float32, device=args.device)
            y = torch.as_tensor(train_y[index], dtype=torch.float32, device=args.device)
            prediction = model(x)
            sample_weight = torch.as_tensor(train_sample_weights[index], dtype=torch.float32, device=args.device)
            loss = (F.smooth_l1_loss(prediction, y, reduction="none") * weights).mean(dim=1)
            loss = (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(val_x, dtype=torch.float32, device=args.device)
            y = torch.as_tensor(val_y, dtype=torch.float32, device=args.device)
            value = float((F.smooth_l1_loss(model(x), y, reduction="none") * weights).mean().cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": value})
        if value < best_value:
            best_value = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("adapter training produced no checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "best_val_loss": best_value, "history": history,
    }


def train_structured_contact_adapter(
    train_features: np.ndarray, train_targets: np.ndarray,
    val_features: np.ndarray, val_targets: np.ndarray, args: argparse.Namespace,
    ridge_alpha: float = 10.0,
) -> tuple[StructuredContactAdapter, dict[str, Any]]:
    """Fit a deterministic ridge specialist on the explicit contact chart."""
    train_physics = contact_physics_features(train_features)
    val_physics = contact_physics_features(val_features)
    feature_mean, feature_std = standardize_fit(train_physics)
    target_mean, target_std = standardize_fit(train_targets)
    train_x = (train_physics - feature_mean) / feature_std
    val_x = (val_physics - feature_mean) / feature_std
    train_y = (train_targets - target_mean) / target_std
    design = np.concatenate([train_x, np.ones((len(train_x), 1), dtype=np.float32)], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T.astype(np.float64) @ design.astype(np.float64) + regularizer,
        design.T.astype(np.float64) @ train_y.astype(np.float64),
    ).astype(np.float32)
    model = StructuredContactAdapter(train_x.shape[1]).to(args.device)
    with torch.no_grad():
        model.linear.weight.copy_(torch.as_tensor(coefficients[:-1].T, device=args.device))
        model.linear.bias.copy_(torch.as_tensor(coefficients[-1], device=args.device))
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.as_tensor(val_x, dtype=torch.float32, device=args.device)).cpu().numpy()
    value = float(np.mean(np.square(prediction - (val_targets - target_mean) / target_std)))
    return model, {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "best_val_loss": value, "ridge_alpha": float(ridge_alpha),
        "input_feature_mode": "contact_physics",
        "history": [{"method": "closed_form_ridge", "val_loss": value}],
    }


def train_goal_decoder(
    train_features: np.ndarray, train_targets: np.ndarray,
    val_features: np.ndarray, val_targets: np.ndarray, args: argparse.Namespace,
) -> tuple[GoalPoseDecoder, dict[str, Any]]:
    """Fit a held-out goal-image decoder independently of the dynamics adapter."""
    torch.manual_seed(args.seed + 17)
    feature_mean, feature_std = standardize_fit(train_features)
    target_mean, target_std = standardize_fit(train_targets)
    train_x = (train_features - feature_mean) / feature_std
    val_x = (val_features - feature_mean) / feature_std
    train_y = (train_targets - target_mean) / target_std
    val_y = (val_targets - target_mean) / target_std
    model = GoalPoseDecoder(train_x.shape[1], args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 19)
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.goal_epochs):
        model.train()
        order = rng.permutation(len(train_x))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            x = torch.as_tensor(train_x[index], dtype=torch.float32, device=args.device)
            y = torch.as_tensor(train_y[index], dtype=torch.float32, device=args.device)
            loss = F.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(val_x, dtype=torch.float32, device=args.device)
            y = torch.as_tensor(val_y, dtype=torch.float32, device=args.device)
            value = float(F.mse_loss(model(x), y).cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": value})
        if value < best_value:
            best_value = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("goal decoder training produced no checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "best_val_loss": best_value, "history": history,
    }


def train_orientation_decoder(
    train_features: np.ndarray, train_targets: np.ndarray,
    val_features: np.ndarray, val_targets: np.ndarray, args: argparse.Namespace,
) -> tuple[OrientationDecoder, dict[str, Any]]:
    torch.manual_seed(args.seed + 31)
    feature_mean, feature_std = standardize_fit(train_features)
    target_mean, target_std = standardize_fit(train_targets)
    train_x = (train_features - feature_mean) / feature_std
    val_x = (val_features - feature_mean) / feature_std
    train_y = (train_targets - target_mean) / target_std
    val_y = (val_targets - target_mean) / target_std
    model = OrientationDecoder(train_x.shape[1], args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 37)
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.goal_epochs):
        model.train()
        order = rng.permutation(len(train_x))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            x = torch.as_tensor(train_x[index], dtype=torch.float32, device=args.device)
            y = torch.as_tensor(train_y[index], dtype=torch.float32, device=args.device)
            loss = F.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(val_x, dtype=torch.float32, device=args.device)
            y = torch.as_tensor(val_y, dtype=torch.float32, device=args.device)
            value = float(F.mse_loss(model(x), y).cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": value})
        if value < best_value:
            best_value = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("orientation decoder training produced no checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "best_val_loss": best_value, "history": history,
    }


def train_rgb_pose_calibrator(
    train_shape_features: np.ndarray,
    train_targets: np.ndarray,
    val_shape_features: np.ndarray,
    val_targets: np.ndarray,
    ridge_alpha: float = 1.0,
) -> tuple[LinearRGBPoseCalibrator, dict[str, Any]]:
    """Fit a fixed, root-disjoint RGB target-pose calibration.

    This is intentionally closed-form and low-capacity.  It corrects the
    renderer's deterministic centroid/PCA offset without learning a task
    policy.  The caller still applies an independent validation gate before
    allowing either calibrated XY or calibrated angle into runtime.
    """
    train_design = rgb_pose_calibration_features(train_shape_features)
    val_design = rgb_pose_calibration_features(val_shape_features)
    train_targets = np.asarray(train_targets, dtype=np.float32)
    val_targets = np.asarray(val_targets, dtype=np.float32)
    if train_targets.ndim != 2 or train_targets.shape[1] != 2:
        raise ValueError("RGB pose calibration targets must have shape [N, 2]")
    if len(train_design) != len(train_targets) or len(val_design) != len(val_targets):
        raise ValueError("RGB pose calibration rows must be aligned")
    feature_mean = train_design.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(train_design.std(axis=0), 1.0e-4).astype(np.float32)
    feature_mean[0] = 0.0
    feature_std[0] = 1.0
    normalized_train = (train_design - feature_mean) / feature_std
    normalized_val = (val_design - feature_mean) / feature_std
    regularizer = np.eye(normalized_train.shape[1], dtype=np.float64) * float(ridge_alpha)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        normalized_train.T.astype(np.float64) @ normalized_train.astype(np.float64)
        + regularizer,
        normalized_train.T.astype(np.float64) @ train_targets.astype(np.float64),
    ).astype(np.float32)
    model = LinearRGBPoseCalibrator(coefficients, feature_mean, feature_std)
    prediction = model.predict(val_shape_features)
    value = float(np.mean(np.square(prediction - val_targets)))
    return model, {
        "feature_interface": "target_mask_shape_chart",
        "readout": "closed_form_ridge_pose_calibration",
        "ridge_alpha": float(ridge_alpha),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "coefficients": coefficients.tolist(),
        "best_val_loss": value,
        "history": [{"method": "closed_form_ridge", "val_loss": value}],
    }


def train_rgb_angle_knn(
    train_features: np.ndarray, train_targets: np.ndarray, k: int = 3,
) -> tuple[RGBAngleKNN, dict[str, Any]]:
    """Fit a deterministic, support-constrained RGB angle interpolator."""
    feature_mean, feature_std = standardize_fit(train_features)
    target_mean, target_std = standardize_fit(train_targets)
    normalized_features = (np.asarray(train_features) - feature_mean) / feature_std
    normalized_targets = (np.asarray(train_targets) - target_mean) / target_std
    model = RGBAngleKNN(normalized_features.astype(np.float32), normalized_targets.astype(np.float32), k)
    return model.eval(), {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "readout": "support_constrained_knn", "k": int(k),
    }


def train_action_cost_ranker(
    train_features: np.ndarray, train_targets: np.ndarray,
    val_features: np.ndarray, val_targets: np.ndarray, args: argparse.Namespace,
    ranking_train_targets: np.ndarray | None = None,
    ranking_val_targets: np.ndarray | None = None,
) -> tuple[ActionCostRanker, dict[str, Any]]:
    """Fit a direct RGB goal-conditioned candidate-cost model.

    Labels are generated offline from matched simulator interventions, while
    runtime inputs contain only RGB-derived features and the candidate action.
    The objective retains a robust cost regression term but gives the
    within-root listwise ranking term an explicit, preregistered weight.  A
    low cost RMSE is not sufficient for selection, so the caller still audits
    top-1 and regret on held-out roots.
    """
    torch.manual_seed(args.seed + 71)
    feature_mean, feature_std = standardize_fit(train_features)
    target_mean, target_std = standardize_fit(train_targets[:, None])
    train_x = (train_features - feature_mean) / feature_std
    val_x = (val_features - feature_mean) / feature_std
    train_y = (train_targets - target_mean[0]) / target_std[0]
    val_y = (val_targets - target_mean[0]) / target_std[0]
    ranking_train_targets = train_targets if ranking_train_targets is None else np.asarray(ranking_train_targets, dtype=np.float32)
    ranking_val_targets = val_targets if ranking_val_targets is None else np.asarray(ranking_val_targets, dtype=np.float32)
    if len(ranking_train_targets) != len(train_targets) or len(ranking_val_targets) != len(val_targets):
        raise ValueError("ranking targets must align with ranker regression targets")
    model = ActionCostRanker(train_x.shape[1], args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    group_size = int(getattr(args, "ranker_group_size", 25))
    for epoch in range(args.epochs):
        model.train()
        x = torch.as_tensor(train_x, dtype=torch.float32, device=args.device)
        y = torch.as_tensor(train_y, dtype=torch.float32, device=args.device)
        prediction = model(x)
        regression_loss = F.smooth_l1_loss(prediction, y)
        rank_terms = []
        pair_terms = []
        for start in range(0, len(train_x), group_size):
            stop = min(start + group_size, len(train_x))
            if stop - start >= 2:
                label = int(np.argmin(ranking_train_targets[start:stop]))
                rank_terms.append(F.cross_entropy(
                    (-prediction[start:stop]).reshape(1, -1),
                    torch.tensor([label], device=args.device),
                ))
                target_group = torch.as_tensor(
                    ranking_train_targets[start:stop], dtype=torch.float32, device=args.device,
                )
                prediction_group = prediction[start:stop]
                target_delta = target_group[:, None] - target_group[None, :]
                prediction_delta = prediction_group[:, None] - prediction_group[None, :]
                upper = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
                informative = upper & (target_delta.abs() > 0.02)
                if bool(informative.any()):
                    pair_terms.append(F.softplus(
                        -torch.sign(target_delta[informative]) * prediction_delta[informative],
                    ).mean())
        rank_loss = torch.stack(rank_terms).mean() if rank_terms else prediction.new_zeros(())
        pairwise_loss = torch.stack(pair_terms).mean() if pair_terms else prediction.new_zeros(())
        rank_weight = float(getattr(args, "ranker_ranking_weight", 4.0))
        pairwise_weight = float(getattr(args, "ranker_pairwise_weight", 0.0))
        loss = regression_loss + rank_weight * rank_loss + pairwise_weight * pairwise_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses = [float(loss.detach().cpu())]
        model.eval()
        with torch.no_grad():
            val_prediction = model(torch.as_tensor(val_x, dtype=torch.float32, device=args.device))
            val_regression = F.smooth_l1_loss(
                val_prediction, torch.as_tensor(val_y, dtype=torch.float32, device=args.device),
            )
            val_rank_terms = []
            val_pair_terms = []
            for start in range(0, len(val_x), group_size):
                stop = min(start + group_size, len(val_x))
                if stop - start >= 2:
                    label = int(np.argmin(ranking_val_targets[start:stop]))
                    val_rank_terms.append(F.cross_entropy(
                        (-val_prediction[start:stop]).reshape(1, -1),
                        torch.tensor([label], device=args.device),
                    ))
                    target_group = torch.as_tensor(
                        ranking_val_targets[start:stop], dtype=torch.float32, device=args.device,
                    )
                    prediction_group = val_prediction[start:stop]
                    target_delta = target_group[:, None] - target_group[None, :]
                    prediction_delta = prediction_group[:, None] - prediction_group[None, :]
                    upper = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
                    informative = upper & (target_delta.abs() > 0.02)
                    if bool(informative.any()):
                        val_pair_terms.append(F.softplus(
                            -torch.sign(target_delta[informative]) * prediction_delta[informative],
                        ).mean())
            val_rank_loss = torch.stack(val_rank_terms).mean() if val_rank_terms else val_regression.new_zeros(())
            val_pairwise_loss = torch.stack(val_pair_terms).mean() if val_pair_terms else val_regression.new_zeros(())
            value = float((
                val_regression + rank_weight * val_rank_loss + pairwise_weight * val_pairwise_loss
            ).cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": value})
        if value < best_value:
            best_value = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("action-cost ranker training produced no checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "target_mean": float(target_mean[0]), "target_std": float(target_std[0]),
        "best_val_loss": best_value,
        "rank_weight": float(getattr(args, "ranker_ranking_weight", 4.0)),
        "pairwise_weight": float(getattr(args, "ranker_pairwise_weight", 0.0)),
        "regression_target": "residual" if ranking_train_targets is not train_targets else "absolute",
        "history": history,
    }


@torch.inference_mode()
def predict_action_cost(
    model: ActionCostRanker, features: np.ndarray, training: dict[str, Any], device: str,
) -> np.ndarray:
    feature_mean = np.asarray(training["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(training["feature_std"], dtype=np.float32)
    x = torch.as_tensor((np.asarray(features, dtype=np.float32) - feature_mean) / feature_std,
                        dtype=torch.float32, device=device)
    return (model(x).cpu().numpy() * float(training["target_std"]) + float(training["target_mean"])).astype(np.float32)


@torch.inference_mode()
def predict_targets(
    model: LocalEffectAdapter, features: np.ndarray, training: dict[str, Any], device: str,
) -> np.ndarray:
    mean = np.asarray(training["feature_mean"], dtype=np.float32)
    std = np.asarray(training["feature_std"], dtype=np.float32)
    target_mean = np.asarray(training["target_mean"], dtype=np.float32)
    target_std = np.asarray(training["target_std"], dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[None]
    if training.get("input_feature_mode") == "contact_physics":
        features = contact_physics_features(features)
    x = torch.as_tensor((features - mean) / std, dtype=torch.float32, device=device)
    values = model(x).cpu().numpy() * target_std + target_mean
    values[:, 2] = wrap(values[:, 2])
    values[:, 5] = wrap(values[:, 5])
    return values.astype(np.float32)


def contact_effect_to_world(effect: np.ndarray, agent: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Rotate [normal, tangent] effect back to world XY using RGB geometry."""
    effect = np.asarray(effect, dtype=np.float32).copy()
    relative = np.asarray(block, dtype=np.float64) - np.asarray(agent, dtype=np.float64)
    if relative.ndim == 1:
        distance = max(float(np.linalg.norm(relative)), 1e-6)
        normal = relative / distance
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        normal_x, normal_y = normal[0], normal[1]
        tangent_x, tangent_y = tangent[0], tangent[1]
    elif relative.ndim == 2 and relative.shape[1] == 2:
        distance = np.linalg.norm(relative, axis=1, keepdims=True)
        normal = relative / np.maximum(distance, 1e-6)
        tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
        normal_x, normal_y = normal[:, 0], normal[:, 1]
        tangent_x, tangent_y = tangent[:, 0], tangent[:, 1]
    else:
        raise ValueError("agent and block must have shape [2] or [N, 2]")
    normal_effect = effect[:, POSE_DIM].astype(np.float64).copy()
    tangent_effect = effect[:, POSE_DIM + 1].astype(np.float64).copy()
    effect[:, POSE_DIM] = (normal_effect * normal_x + tangent_effect * tangent_x).astype(np.float32)
    effect[:, POSE_DIM + 1] = (normal_effect * normal_y + tangent_effect * tangent_y).astype(np.float32)
    return effect


def _canonical_knn_context(features: np.ndarray) -> np.ndarray:
    """RGB-only contact context invariant to absolute canvas translation."""
    values = np.asarray(features, dtype=np.float32)
    agent = values[:, 1:3]
    block = values[:, 11:13]
    relative = block - agent
    block_shape = values[:, 13:20]
    temporal_agent = values[:, 31:33] - values[:, 1:3]
    temporal_block = values[:, 41:43] - values[:, 11:13]
    return np.concatenate([
        relative, np.linalg.norm(relative, axis=1, keepdims=True),
        block_shape, temporal_agent, temporal_block,
    ], axis=1).astype(np.float32)


def _canonical_knn_action(features: np.ndarray) -> np.ndarray:
    """RGB-derived [normal action, tangent action, magnitude] chart."""
    values = np.asarray(features, dtype=np.float32)
    return values[:, 69:72].astype(np.float32)


def contact_canonical_knn_predict(
    training: dict[str, Any], query_features: np.ndarray,
) -> np.ndarray:
    """Predict [zero delta, contact-frame effect] from root-disjoint RGB rows."""
    train_features = np.asarray(training["features"], dtype=np.float32)
    train_targets = np.asarray(training["targets"], dtype=np.float32)
    query_features = np.asarray(query_features, dtype=np.float32)
    context_train = _canonical_knn_context(train_features)
    context_query = _canonical_knn_context(query_features)
    context_mean = np.asarray(training["context_mean"], dtype=np.float32)
    context_std = np.asarray(training["context_std"], dtype=np.float32)
    context_train = (context_train - context_mean) / context_std
    context_query = (context_query - context_mean) / context_std
    action_train = _canonical_knn_action(train_features)
    action_query = _canonical_knn_action(query_features)
    action_mean = np.asarray(training["action_mean"], dtype=np.float32)
    action_std = np.asarray(training["action_std"], dtype=np.float32)
    action_train = (action_train - action_mean) / action_std
    action_query = (action_query - action_mean) / action_std
    root_ids = np.asarray(training["root_ids"], dtype=np.int64)
    predictions = np.zeros((len(query_features), 2 * POSE_DIM), dtype=np.float32)
    context_weight = float(training["context_weight"])
    action_weight = float(training["action_weight"])
    k = max(1, int(training["k"]))
    for row in range(len(query_features)):
        distance = context_weight * np.mean(np.square(context_train - context_query[row]), axis=1)
        distance += action_weight * np.mean(np.square(action_train - action_query[row]), axis=1)
        selected: list[int] = []
        seen: set[int] = set()
        for index in np.argsort(distance, kind="stable"):
            root = int(root_ids[index])
            if root not in seen:
                selected.append(int(index))
                seen.add(root)
            if len(selected) >= k:
                break
        selected_array = np.asarray(selected, dtype=np.int64)
        weights = 1.0 / np.maximum(distance[selected_array], 1.0e-5)
        weights /= weights.sum()
        predictions[row] = np.sum(train_targets[selected_array] * weights[:, None], axis=0)
    return predictions


def fit_contact_canonical_knn(
    features: np.ndarray, targets: np.ndarray, root_ids: np.ndarray,
    k: int, context_weight: float, action_weight: float,
) -> dict[str, Any]:
    """Fit the immutable RGB-only KNN table used by the validation gate."""
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    root_ids = np.asarray(root_ids, dtype=np.int64)
    context = _canonical_knn_context(features)
    action = _canonical_knn_action(features)
    context_mean = context.mean(axis=0)
    context_std = np.maximum(context.std(axis=0), 1.0e-4)
    action_mean = action.mean(axis=0)
    action_std = np.maximum(action.std(axis=0), 1.0e-4)
    return {
        "features": features,
        "targets": targets,
        "root_ids": root_ids,
        "context_mean": context_mean,
        "context_std": context_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "k": int(k),
        "context_weight": float(context_weight),
        "action_weight": float(action_weight),
    }


def world_effect_to_contact(effect: np.ndarray, agent: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Rotate world-XY effect into the RGB contact normal/tangent chart."""
    effect = np.asarray(effect, dtype=np.float32).copy()
    relative = np.asarray(block, dtype=np.float64) - np.asarray(agent, dtype=np.float64)
    distance = max(float(np.linalg.norm(relative)), 1e-6)
    normal = relative / distance
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    world_x = effect[:, POSE_DIM].astype(np.float64).copy()
    world_y = effect[:, POSE_DIM + 1].astype(np.float64).copy()
    effect[:, POSE_DIM] = (world_x * normal[0] + world_y * normal[1]).astype(np.float32)
    effect[:, POSE_DIM + 1] = (world_x * tangent[0] + world_y * tangent[1]).astype(np.float32)
    return effect


@torch.inference_mode()
def predict_goal(
    model: GoalPoseDecoder, features: np.ndarray, training: dict[str, Any], device: str,
) -> np.ndarray:
    feature_mean = np.asarray(training["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(training["feature_std"], dtype=np.float32)
    target_mean = np.asarray(training["target_mean"], dtype=np.float32)
    target_std = np.asarray(training["target_std"], dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[None]
    x = torch.as_tensor((features - feature_mean) / feature_std, dtype=torch.float32, device=device)
    return (model(x).cpu().numpy() * target_std + target_mean).astype(np.float32)


@torch.inference_mode()
def predict_orientation(
    model: OrientationDecoder, features: np.ndarray, training: dict[str, Any], device: str,
) -> np.ndarray:
    if isinstance(model, (LinearRGBPoseCalibrator, LinearFeatureXYDecoder)):
        return model.predict(features)
    feature_mean = np.asarray(training["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(training["feature_std"], dtype=np.float32)
    target_mean = np.asarray(training["target_mean"], dtype=np.float32)
    target_std = np.asarray(training["target_std"], dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[None]
    x = torch.as_tensor((features - feature_mean) / feature_std, dtype=torch.float32, device=device)
    return (model(x).cpu().numpy() * target_std + target_mean).astype(np.float32)


def train_linear_feature_xy_decoder(
    train_features: np.ndarray, train_targets: np.ndarray, ridge_alpha: float = 1.0,
) -> tuple[LinearFeatureXYDecoder, dict[str, Any]]:
    """Fit a closed-form affine frozen-feature XY readout."""
    features = np.asarray(train_features, dtype=np.float32)
    targets = np.asarray(train_targets, dtype=np.float32)
    if features.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("linear feature XY decoder expects [N,D] features and [N,2] targets")
    if len(features) != len(targets) or len(features) < 2:
        raise ValueError("linear feature XY decoder requires aligned non-empty rows")
    feature_mean, feature_std = standardize_fit(features)
    target_mean, target_std = standardize_fit(targets)
    x = (features - feature_mean) / feature_std
    y = (targets - target_mean) / target_std
    design = np.concatenate([
        np.ones((len(x), 1), dtype=np.float32), x,
    ], axis=1).astype(np.float64)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, design.T @ y.astype(np.float64))
    decoder = LinearFeatureXYDecoder(
        coefficients.astype(np.float32), feature_mean, feature_std,
        target_mean, target_std,
    )
    return decoder, {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "ridge_alpha": float(ridge_alpha),
        "decoder": "root_disjoint_affine_frozen_feature_xy",
    }


@torch.inference_mode()
def goal_feature_rows(
    model: VisualCausalWorldModel, data: dict[str, np.ndarray], device: str,
) -> np.ndarray:
    roots = np.unique(data["root_id"])
    indices = np.asarray([np.flatnonzero(data["root_id"] == root)[0] for root in roots], dtype=np.int64)
    goal = tensor_images(data["goal_image"][indices], device)
    values = model.encoder.features(goal).cpu().numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite goal RGB features")
    return values


@torch.inference_mode()
def current_feature_rows(
    model: VisualCausalWorldModel, data: dict[str, np.ndarray], device: str,
) -> np.ndarray:
    roots = np.unique(data["root_id"])
    indices = np.asarray([np.flatnonzero(data["root_id"] == root)[0] for root in roots], dtype=np.int64)
    image = tensor_images(data["image"][indices], device)
    values = model.encoder.features(image).cpu().numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite current RGB features")
    return values


@torch.inference_mode()
def dynamic_readout_rows(
    model: VisualCausalWorldModel,
    env: gym.Env,
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    image_size: int,
    device: str,
    seed: int,
    rollout_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect RGB-only readout examples after offline root rollouts.

    The simulator state is used only to create labels for this offline
    calibration set.  At runtime the readouts consume the rendered image
    alone.  Train and validation callers pass disjoint root lists, so this
    augmentation tests robustness to dynamic occlusion without sharing
    source states across the selection boundary.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    agent_xy_targets: list[np.ndarray] = []
    xy_targets: list[np.ndarray] = []
    orientation_targets: list[np.ndarray] = []
    for root_index, (state, goal, _image, _previous, _goal_image, _previous_action, _contact) in enumerate(roots):
        observation, _info = env.reset(
            seed=seed + root_index,
            options={"state": state.copy(), "goal_state": goal.copy()},
        )
        for step in range(rollout_steps + 1):
            image = env.render().copy()
            current_state = np.asarray(observation["state"], dtype=np.float32)
            images.append(resize_image(image, image_size))
            agent_xy_targets.append(current_state[:2].copy())
            xy_targets.append(current_state[2:4].copy())
            orientation_targets.append(
                np.asarray([np.sin(current_state[4]), np.cos(current_state[4])], dtype=np.float32)
            )
            if step == rollout_steps:
                break
            relative = current_state[2:4] - current_state[:2]
            direction = relative / max(float(np.linalg.norm(relative)), 1e-6)
            if (step + root_index) % 3 == 0:
                action = rng.uniform(-0.85, 0.85, size=2)
            else:
                action = np.clip(
                    0.65 * direction + 0.35 * rng.normal(size=2), -1.0, 1.0,
                )
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.float32),
            )
            if terminated or truncated:
                break
    image_array = np.asarray(images, dtype=np.uint8)
    feature_chunks: list[np.ndarray] = []
    for start in range(0, len(image_array), 2048):
        batch = tensor_images(image_array[start:start + 2048], device)
        feature_chunks.append(model.encoder.features(batch).cpu().numpy().astype(np.float32))
    features = np.concatenate(feature_chunks, axis=0)
    xy = np.asarray(xy_targets, dtype=np.float32)
    agent_xy = np.asarray(agent_xy_targets, dtype=np.float32)
    orientation = np.asarray(orientation_targets, dtype=np.float32)
    if not (len(features) == len(agent_xy) == len(xy) == len(orientation)):
        raise RuntimeError("dynamic readout rows are unbalanced")
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite dynamic readout features")
    return features, agent_xy, xy, orientation


@torch.inference_mode()
def dynamic_rgb_shape_angle_rows(
    env: gym.Env,
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    image_size: int,
    seed: int,
    rollout_steps: int,
    feature_interface: str = "moments",
    contact_biased: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect root-disjoint dynamic RGB shape charts for angle calibration.

    The frozen encoder feature augmentation above is not sufficient for the
    RGB-shape decoder: its input is the 12-dimensional image chart.  This
    companion protocol therefore returns the same chart at post-action
    frames, with simulator state used only for offline angle labels.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    orientation_targets: list[np.ndarray] = []
    for root_index, (state, goal, _image, _previous, _goal_image, _previous_action, _contact) in enumerate(roots):
        observation, _info = env.reset(
            seed=seed + root_index,
            options={"state": state.copy(), "goal_state": goal.copy()},
        )
        for step in range(rollout_steps + 1):
            image = env.render().copy()
            current_state = np.asarray(observation["state"], dtype=np.float32)
            images.append(image)
            orientation_targets.append(
                np.asarray([np.sin(current_state[4]), np.cos(current_state[4])], dtype=np.float32)
            )
            if step == rollout_steps:
                break
            relative = current_state[2:4] - current_state[:2]
            distance = float(np.linalg.norm(relative))
            direction = relative / max(distance, 1e-6)
            if contact_biased and (bool(_contact) or distance <= 85.0):
                # The ordinary dynamic augmentation is deliberately broad,
                # but it under-samples the exact failure regime: the pusher
                # occludes the T while a small tangential motion changes its
                # visible raster without a large object rotation.  Use a
                # bounded, alternating in-contact fan to generate that
                # distribution.  Labels remain simulator-offline only.
                tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
                sign = 1.0 if (step + root_index) % 2 == 0 else -1.0
                action = 0.20 * direction + sign * 0.14 * tangent
                action = np.clip(action, -0.35, 0.35)
            elif contact_biased:
                action = np.clip(0.75 * direction, -0.75, 0.75)
            elif (step + root_index) % 3 == 0:
                action = rng.uniform(-0.85, 0.85, size=2)
            else:
                action = np.clip(
                    0.65 * direction + 0.35 * rng.normal(size=2), -1.0, 1.0,
                )
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.float32),
            )
            if terminated or truncated:
                break
    image_array = np.asarray(images, dtype=np.uint8)
    shape_features = (
        rgb_shape_raster_features(image_array, "block")
        if feature_interface == "raster"
        else rgb_geometry_shape_features(
            np.asarray([resize_image(item, image_size) for item in image_array]), "block",
        )
    )
    orientation = np.asarray(orientation_targets, dtype=np.float32)
    if len(shape_features) != len(orientation):
        raise RuntimeError("dynamic RGB angle rows are unbalanced")
    if not np.isfinite(shape_features).all() or not np.isfinite(orientation).all():
        raise RuntimeError("non-finite dynamic RGB angle rows")
    return shape_features, orientation


@torch.inference_mode()
def dynamic_rgb_shape_xy_rows(
    env: gym.Env,
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    seed: int,
    rollout_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect root-disjoint dynamic shape charts with block-XY labels.

    Static shape calibration can look accurate while contact occlusion changes
    the visible T centroid.  This companion split uses the exact raw RGB
    chart consumed by the runtime current-XY observer and therefore qualifies
    a calibration for closed-loop use only when its dynamic held-out error is
    also below the declared gate.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    xy_targets: list[np.ndarray] = []
    for root_index, (state, goal, _image, _previous, _goal_image, _previous_action, _contact) in enumerate(roots):
        observation, _info = env.reset(
            seed=seed + root_index,
            options={"state": state.copy(), "goal_state": goal.copy()},
        )
        for step in range(rollout_steps + 1):
            image = env.render().copy()
            current_state = np.asarray(observation["state"], dtype=np.float32)
            images.append(image)
            xy_targets.append(current_state[2:4].copy())
            if step == rollout_steps:
                break
            relative = current_state[2:4] - current_state[:2]
            direction = relative / max(float(np.linalg.norm(relative)), 1e-6)
            if (step + root_index) % 3 == 0:
                action = rng.uniform(-0.85, 0.85, size=2)
            else:
                action = np.clip(
                    0.65 * direction + 0.35 * rng.normal(size=2), -1.0, 1.0,
                )
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.float32),
            )
            if terminated or truncated:
                break
    image_array = np.asarray(images, dtype=np.uint8)
    shape_features = rgb_geometry_shape_features(
        np.moveaxis(image_array, -1, 1), "block",
    )
    xy = np.asarray(xy_targets, dtype=np.float32)
    if len(shape_features) != len(xy):
        raise RuntimeError("dynamic RGB XY rows are unbalanced")
    if not np.isfinite(shape_features).all() or not np.isfinite(xy).all():
        raise RuntimeError("non-finite dynamic RGB XY rows")
    return shape_features, xy


@torch.inference_mode()
def previous_feature_rows(
    model: VisualCausalWorldModel, data: dict[str, np.ndarray], device: str,
) -> np.ndarray:
    roots = np.unique(data["root_id"])
    indices = np.asarray([np.flatnonzero(data["root_id"] == root)[0] for root in roots], dtype=np.int64)
    image = tensor_images(data["previous_image"][indices], device)
    values = model.encoder.features(image).cpu().numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite previous RGB features")
    return values


def geometry_ranker_features(
    current_features: np.ndarray,
    previous_features: np.ndarray,
    goal_features: np.ndarray,
    actions: np.ndarray,
    current_angles: np.ndarray,
    goal_poses: np.ndarray,
) -> np.ndarray:
    """Build an RGB-only task-space context for within-root action ranking.

    The legacy ranker receives the frozen feature vectors and the candidate
    action, but must discover the contact chart, current-to-goal displacement,
    and angular error through a large MLP.  This helper exposes those
    quantities explicitly while keeping every quantity image-derived.  The
    goal poses are decoder outputs supplied by the caller, never simulator
    labels.  It is intentionally separate from ``frozen_features`` so the
    effect adapter remains a causal, goal-independent diagnostic.
    """
    current_features = np.asarray(current_features, dtype=np.float32)
    previous_features = np.asarray(previous_features, dtype=np.float32)
    goal_features = np.asarray(goal_features, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    current_angles = np.asarray(current_angles, dtype=np.float32).reshape(-1)
    goal_poses = np.asarray(goal_poses, dtype=np.float32)
    if current_features.ndim == 1:
        current_features = current_features[None]
    if previous_features.ndim == 1:
        previous_features = previous_features[None]
    if goal_features.ndim == 1:
        goal_features = goal_features[None]
    if actions.ndim == 1:
        actions = actions[None]
    if goal_poses.ndim == 1:
        goal_poses = goal_poses[None]
    count = len(actions)
    if not (
        len(current_features) == len(previous_features) == len(goal_features)
        == len(current_angles) == len(goal_poses) == count
    ):
        raise ValueError("geometry ranker inputs must have the same row count")
    if current_features.shape[1] < BLOCK_FEATURE_STOP or goal_features.shape[1] < TARGET_FEATURE_STOP:
        raise ValueError("geometry ranker requires mask-moment block features")

    scale = PUSHT_STATE_SIZE / 2.0
    agent = current_features[:, 1:3] * scale
    block = current_features[:, 11:13] * scale
    # The goal RGB frame contains both the gray block and the green target.
    # The gray block chart [10:20] is the pose target used by the checkpoint;
    # retain the green chart separately as scene-context evidence below.
    goal_block = goal_features[:, BLOCK_FEATURE_START + 1:BLOCK_FEATURE_START + 3] * scale
    relative = block - agent
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    normal = relative / np.maximum(distance, 1e-6)
    tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
    goal_xy = goal_poses[:, :2]
    goal_angle = np.arctan2(goal_poses[:, 2], goal_poses[:, 3])
    angle_error = wrap(goal_angle - current_angles)
    # The ten block-moment coordinates contain more than a centroid: the
    # second and third central moments encode the T's principal axis and its
    # directed asymmetry.  Expose a numerically stable shape chart explicitly;
    # this is important because torque depends on the contact point relative
    # to the oriented T, which a centroid-only chart cannot identify.
    block_moments = current_features[:, BLOCK_FEATURE_START:BLOCK_FEATURE_STOP]
    goal_block_moments = goal_features[:, BLOCK_FEATURE_START:BLOCK_FEATURE_STOP]
    goal_target_moments = goal_features[:, TARGET_FEATURE_START:TARGET_FEATURE_STOP]
    block_mxx, block_mxy, block_myy = block_moments[:, 3], block_moments[:, 4], block_moments[:, 5]
    goal_mxx, goal_mxy, goal_myy = goal_block_moments[:, 3], goal_block_moments[:, 4], goal_block_moments[:, 5]
    block_shape_angle = 0.5 * np.arctan2(2.0 * block_mxy, block_mxx - block_myy)
    goal_shape_angle = 0.5 * np.arctan2(2.0 * goal_mxy, goal_mxx - goal_myy)
    block_anisotropy = np.sqrt((block_mxx - block_myy) ** 2 + 4.0 * block_mxy ** 2)
    goal_anisotropy = np.sqrt((goal_mxx - goal_myy) ** 2 + 4.0 * goal_mxy ** 2)
    block_axis = np.column_stack([np.cos(block_shape_angle), np.sin(block_shape_angle)])
    block_perp = np.column_stack([-block_axis[:, 1], block_axis[:, 0]])
    action_normal = np.sum(actions * normal, axis=1, keepdims=True)
    action_tangent = np.sum(actions * tangent, axis=1, keepdims=True)
    action_magnitude = np.linalg.norm(actions, axis=1, keepdims=True)
    explicit = np.concatenate([
        agent / PUSHT_STATE_SIZE,
        block / PUSHT_STATE_SIZE,
        goal_block / PUSHT_STATE_SIZE,
        (goal_xy - block) / PUSHT_STATE_SIZE,
        relative / PUSHT_STATE_SIZE,
        distance / PUSHT_STATE_SIZE,
        np.sin(current_angles)[:, None], np.cos(current_angles)[:, None],
        np.sin(goal_angle)[:, None], np.cos(goal_angle)[:, None],
        np.sin(angle_error)[:, None], np.cos(angle_error)[:, None],
        actions, action_normal, action_tangent, action_magnitude,
        block_moments, goal_block_moments,
        goal_target_moments,
        np.sin(2.0 * block_shape_angle)[:, None], np.cos(2.0 * block_shape_angle)[:, None],
        np.sin(2.0 * goal_shape_angle)[:, None], np.cos(2.0 * goal_shape_angle)[:, None],
        block_anisotropy[:, None], goal_anisotropy[:, None],
        np.sum(actions * block_axis, axis=1, keepdims=True),
        np.sum(actions * block_perp, axis=1, keepdims=True),
    ], axis=1).astype(np.float32)
    legacy = np.concatenate([
        current_features, current_features - previous_features, actions,
        contact_chart_from_features(current_features, actions), goal_features,
    ], axis=1)
    # Give the learned ranker the same RGB-visible support-point chart as the
    # causal adapter. Centroid/normal features alone discard the tangential
    # lever arm that determines contact torque on the asymmetric T.
    boundary_chart = contact_boundary_chart_from_images(current_features, actions)
    values = np.concatenate([legacy, explicit, boundary_chart], axis=1)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite geometry ranker features")
    return values


def angle_error_batch(predicted: np.ndarray, target: np.ndarray) -> float:
    offset = 2 if predicted.shape[1] >= 4 else 0
    left = np.arctan2(predicted[:, offset], predicted[:, offset + 1])
    right = np.arctan2(target[:, offset], target[:, offset + 1])
    return float(np.mean(np.abs(wrap(left - right))))


def held_out_cost_ranking(
    data: dict[str, np.ndarray], scores: np.ndarray, true_cost: np.ndarray,
    candidate_mask: np.ndarray, contact_only: bool = False,
    root_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a direct cost ranker on a fixed held-out action support.

    The action-cost ranker is not an effect predictor, so it cannot be passed
    through ``ranking_from_goal_states``.  This helper applies the identical
    root-wise selection and true-regret definition to scalar cost scores.  It
    is deliberately kept separate from training and consumes no runtime
    simulator state.
    """
    scores = np.asarray(scores, dtype=np.float32)
    true_cost = np.asarray(true_cost, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if len(scores) != len(data["root_id"]) or len(true_cost) != len(scores):
        raise ValueError("cost-ranking arrays must match the held-out rows")
    hits: list[float] = []
    regrets: list[float] = []
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        if contact_only and not bool(data["contact_root"][indices[0]]):
            continue
        if root_mask is not None and not bool(np.asarray(root_mask, dtype=bool)[root_id]):
            continue
        indices = indices[candidate_mask[indices]]
        if len(indices) < 2:
            raise RuntimeError("cost-ranking support must retain at least two actions per root")
        selected = int(indices[np.argmin(scores[indices])])
        oracle = float(np.min(true_cost[indices]))
        hits.append(float(true_cost[selected] <= oracle + 1.0e-6))
        regrets.append(float(true_cost[selected] - oracle))
    return {
        "top1": float(np.mean(hits)) if hits else float("nan"),
        "mean_true_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "roots": int(len(hits)),
    }


def predicted_pose_costs(
    data: dict[str, np.ndarray], predicted: np.ndarray, goal_poses: np.ndarray,
    effect_frame: str = "world", xy_weight: float = 1.0,
) -> np.ndarray:
    """Return the adapter's per-candidate predicted endpoint cost.

    This is the scalar counterpart of ``ranking_from_goal_states``.  Keeping
    it explicit lets a specialized terminal ranker be gated against the
    actual frozen-adapter score on the same root/action support.
    """
    if not np.isfinite(float(xy_weight)) or float(xy_weight) <= 0.0:
        raise ValueError("xy_weight must be finite and positive")
    scores = np.full(len(data["root_id"]), np.inf, dtype=np.float32)
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        goal = np.asarray(goal_poses[indices[0]], dtype=np.float32)
        goal_block = np.asarray([
            goal[0], goal[1], np.arctan2(float(goal[2]), float(goal[3])),
        ], dtype=np.float32)
        current = block_pose(data["state"][indices[0]])
        predicted_zero = np.repeat(predicted[indices[0], :POSE_DIM][None], len(indices), axis=0)
        predicted_effect = predicted[indices, POSE_DIM:].copy()
        if effect_frame == "contact":
            source = data["state"][indices[0]]
            predicted_effect = contact_effect_to_world(
                np.column_stack([
                    np.zeros((len(indices), POSE_DIM), dtype=np.float32), predicted_effect,
                ]), source[:2], source[2:4],
            )[:, POSE_DIM:]
        next_pose = np.tile(current, (len(indices), 1)) + predicted_zero + predicted_effect
        scores[indices] = (
            float(xy_weight) * np.linalg.norm(
                next_pose[:, :2] - goal_block[None, :2], axis=1,
            ) / 20.0
            + 10.0 * np.abs(wrap(next_pose[:, 2] - goal_block[2]))
        ).astype(np.float32)
    return scores


def predicted_angle_costs(
    data: dict[str, np.ndarray], predicted: np.ndarray, goal_poses: np.ndarray,
    effect_frame: str = "world",
) -> np.ndarray:
    """Return the adapter-predicted angular endpoint error per candidate."""
    scores = np.full(len(data["root_id"]), np.inf, dtype=np.float32)
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        goal = np.asarray(goal_poses[indices[0]], dtype=np.float32)
        goal_angle = float(np.arctan2(goal[2], goal[3]))
        current = block_pose(data["state"][indices[0]])
        predicted_zero = np.repeat(predicted[indices[0], :POSE_DIM][None], len(indices), axis=0)
        predicted_effect = predicted[indices, POSE_DIM:].copy()
        if effect_frame == "contact":
            source = data["state"][indices[0]]
            predicted_effect = contact_effect_to_world(
                np.column_stack([
                    np.zeros((len(indices), POSE_DIM), dtype=np.float32), predicted_effect,
                ]), source[:2], source[2:4],
            )[:, POSE_DIM:]
        next_pose = np.tile(current, (len(indices), 1)) + predicted_zero + predicted_effect
        scores[indices] = np.abs(wrap(next_pose[:, 2] - goal_angle)).astype(np.float32)
    return scores


def angular_costs(
    data: dict[str, np.ndarray], goal_poses: np.ndarray,
) -> np.ndarray:
    """Ground-truth one-step angular endpoint error for a fixed action bank."""
    values = np.full(len(data["root_id"]), np.inf, dtype=np.float32)
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        goal = np.asarray(goal_poses[indices[0]], dtype=np.float32)
        goal_angle = float(np.arctan2(goal[2], goal[3]))
        # ``next_state`` in the intervention table is the compact six-vector
        # [agent_xy, block_xy, sin(theta), cos(theta)], not the full simulator
        # state consumed by ``state_vector``.
        next_angles = np.asarray([
            np.arctan2(float(row[4]), float(row[5]))
            for row in data["next_state"][indices]
        ], dtype=np.float32)
        values[indices] = np.abs(wrap(next_angles - goal_angle)).astype(np.float32)
    return values


def pose_endpoint_costs(
    data: dict[str, np.ndarray], goal_poses: np.ndarray,
    xy_weight: float = 1.0,
) -> np.ndarray:
    """Return a fixed translation-plus-angle endpoint cost for ranker labels."""
    if not np.isfinite(float(xy_weight)) or float(xy_weight) <= 0.0:
        raise ValueError("xy_weight must be finite and positive")
    values = np.full(len(data["root_id"]), np.inf, dtype=np.float32)
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        goal = np.asarray(goal_poses[indices[0]], dtype=np.float32)
        goal_xy = goal[:2]
        goal_angle = float(np.arctan2(goal[2], goal[3]))
        next_pose = np.asarray([block_pose(row) for row in data["next_state"][indices]])
        values[indices] = (
            float(xy_weight) * np.linalg.norm(next_pose[:, :2] - goal_xy[None], axis=1) / 20.0
            + 10.0 * np.abs(wrap(next_pose[:, 2] - goal_angle))
        ).astype(np.float32)
    return values


def constrained_trajectory_pose_costs(
    data: dict[str, np.ndarray], goal_poses: np.ndarray,
    xy_weight: float = 1.0,
    angle_weight: float = 10.0,
    contact_penalty: float = 0.0,
    xy_excursion_penalty: float = 0.0,
    angle_excursion_penalty: float = 0.0,
    contact_distance_max: float = 80.0,
) -> np.ndarray:
    """Score a repeated-action trajectory with endpoint and safety terms.

    ``pose_endpoint_costs`` only sees the final simulator frame.  That is
    inconsistent with the terminal runtime, where a selected action can be
    committed for several physical steps.  This cost keeps the endpoint task
    objective but adds predeclared penalties for intermediate contact loss and
    excursions.  It is an offline label/evaluation quantity: no trajectory
    state is exposed to the RGB ranker at runtime.

    The zero-action row is still a valid root-matched reference.  Therefore
    the caller may center these costs per root without changing the within-root
    argmin while retaining the causal comparison to passive motion.
    """
    for name, value in (
        ("xy_weight", xy_weight), ("angle_weight", angle_weight),
        ("contact_penalty", contact_penalty),
        ("xy_excursion_penalty", xy_excursion_penalty),
        ("angle_excursion_penalty", angle_excursion_penalty),
        ("contact_distance_max", contact_distance_max),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(xy_weight) <= 0.0 and float(angle_weight) <= 0.0:
        raise ValueError("at least one endpoint weight must be positive")
    trajectory = np.asarray(data.get("trajectory_pose"), dtype=np.float32)
    if trajectory.ndim != 3 or trajectory.shape[2] < 6:
        raise ValueError("trajectory_pose must have shape [N, horizon+1, >=6]")
    if len(trajectory) != len(data["root_id"]):
        raise ValueError("trajectory_pose rows must match root_id")
    goal_poses = np.asarray(goal_poses, dtype=np.float32)
    if len(goal_poses) != len(trajectory) or goal_poses.shape[1] < 4:
        raise ValueError("goal_poses must be root-matched [N, 4]")
    values = np.full(len(trajectory), np.inf, dtype=np.float32)
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        goal = goal_poses[indices[0]]
        goal_xy = goal[:2]
        goal_angle = float(np.arctan2(goal[2], goal[3]))
        path = trajectory[indices]
        block_xy = path[:, :, 2:4]
        block_angle = np.arctan2(path[:, :, 4], path[:, :, 5])
        agent_xy = path[:, :, :2]
        xy_error = np.linalg.norm(block_xy - goal_xy[None, None, :], axis=2)
        angle_error = np.abs(wrap(block_angle - goal_angle))
        contact_distance = np.linalg.norm(agent_xy - block_xy, axis=2)
        endpoint = (
            float(xy_weight) * xy_error[:, -1] / 20.0
            + float(angle_weight) * angle_error[:, -1]
        )
        separation = np.maximum(
            np.max(contact_distance, axis=1) - float(contact_distance_max), 0.0,
        ) / 20.0
        xy_excursion = np.maximum(
            np.max(xy_error, axis=1) - xy_error[:, 0], 0.0,
        ) / 20.0
        angle_excursion = np.maximum(
            np.max(angle_error, axis=1) - angle_error[:, 0], 0.0,
        )
        values[indices] = (
            endpoint
            + float(contact_penalty) * separation
            + float(xy_excursion_penalty) * xy_excursion
            + float(angle_excursion_penalty) * angle_excursion
        ).astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite constrained trajectory costs")
    return values


@torch.inference_mode()
def base_goal_rows(
    model: VisualCausalWorldModel, normalizer: Normalization,
    data: dict[str, np.ndarray], device: str,
) -> np.ndarray:
    roots = np.unique(data["root_id"])
    indices = np.asarray([np.flatnonzero(data["root_id"] == root)[0] for root in roots], dtype=np.int64)
    goal = tensor_images(data["goal_image"][indices], device)
    mean, std = normalizer.tensors(device)
    return (model.encode_goal_pose(goal) * std[2:6] + mean[2:6]).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def base_orientation_rows(
    model: VisualCausalWorldModel, normalizer: Normalization,
    data: dict[str, np.ndarray], device: str,
) -> np.ndarray:
    roots = np.unique(data["root_id"])
    indices = np.asarray([np.flatnonzero(data["root_id"] == root)[0] for root in roots], dtype=np.int64)
    image = tensor_images(data["image"][indices], device)
    mean, std = normalizer.tensors(device)
    values = raw_from_normalized(model.encode_state(image, image), mean, std).cpu().numpy()
    return values[:, 4:6].astype(np.float32)


def candidate_ranking(
    data: dict[str, np.ndarray], predicted: np.ndarray, actions: np.ndarray,
) -> dict[str, Any]:
    all_rows: list[dict[str, float]] = []
    contact_rows: list[dict[str, float]] = []
    per_root = []
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        current = block_pose(data["state"][indices[0]])
        goal_state = data["goal_image"]  # sentinel replaced below by state-independent target unavailable here
        del goal_state
        # The dataset does not need to store goal state in runtime features;
        # recover it from the environment-generated root only in the caller.
        per_root.append((root_id, indices, current))
    return {"roots": per_root, "actions": actions.tolist()}


def ranking_from_goal_states(
    data: dict[str, np.ndarray], predicted: np.ndarray, goal_poses: np.ndarray,
    effect_frame: str = "world", candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    hits_oracle: list[float] = []
    hits_visual: list[float] = []
    effect_errors: list[float] = []
    contact_hits_oracle: list[float] = []
    contact_hits_visual: list[float] = []
    true_regrets: list[float] = []
    contact_regrets: list[float] = []
    true_ranked: list[int] = []
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        if candidate_mask is not None:
            candidate_mask = np.asarray(candidate_mask, dtype=bool)
            indices = indices[candidate_mask[indices]]
            if len(indices) < 2:
                raise RuntimeError("candidate support must retain at least two actions per root")
        true = np.asarray([block_pose(row) for row in data["next_state"][indices]])
        goal = np.asarray(goal_poses[indices[0]], dtype=np.float32)
        goal_block = np.asarray([goal[0], goal[1], theta_from_pose(np.r_[0.0, 0.0, goal[0], goal[1], goal[2], goal[3], 0.0, 0.0])], dtype=np.float32)
        # goal_poses are [x, y, sin(theta), cos(theta)].
        goal_block[2] = float(np.arctan2(float(goal[2]), float(goal[3])))
        true_cost = pose_cost(true, goal_block)
        true_best = int(np.argmin(true_cost))
        current = block_pose(data["state"][indices[0]])
        # The passive baseline is a property of the source observation, not
        # of the candidate action.  Reuse the zero-action query for every
        # candidate; otherwise the decomposition leaks action dependence back
        # into the baseline and destroys the causal ranking interpretation.
        predicted_zero = np.repeat(predicted[indices[0], :POSE_DIM][None], len(indices), axis=0)
        predicted_effect = predicted[indices, POSE_DIM:]
        if effect_frame == "contact":
            source = data["state"][indices[0]]
            predicted_effect = contact_effect_to_world(
                np.column_stack([np.zeros((len(indices), POSE_DIM), dtype=np.float32), predicted_effect]),
                source[:2], source[2:4],
            )[:, POSE_DIM:]
        oracle_next = np.tile(current, (len(indices), 1)) + predicted_zero + predicted_effect
        visual_current = current.copy()
        # Current visual positions are encoded in the frozen feature chart and
        # are added by the caller as an optional replacement when available.
        visual_next = np.tile(visual_current, (len(indices), 1)) + predicted_zero + predicted_effect
        oracle_cost = pose_cost(oracle_next, goal_block)
        visual_cost = pose_cost(visual_next, goal_block)
        hits_oracle.append(float(np.argmin(oracle_cost) == true_best))
        hits_visual.append(float(np.argmin(visual_cost) == true_best))
        selected = int(np.argmin(visual_cost))
        true_regrets.append(float(true_cost[selected] - np.min(true_cost)))
        effect_errors.extend(np.linalg.norm(predicted_effect - (true - np.asarray([block_pose(row) for row in data["zero_successor"][indices]])), axis=1).tolist())
        if bool(data["contact_root"][indices[0]]):
            contact_hits_oracle.append(hits_oracle[-1])
            contact_hits_visual.append(hits_visual[-1])
            contact_regrets.append(true_regrets[-1])
        true_ranked.append(true_best)
    return {
        "roots": len(hits_oracle),
        "top1_oracle_current": float(np.mean(hits_oracle)) if hits_oracle else float("nan"),
        "top1_visual_current_proxy": float(np.mean(hits_visual)) if hits_visual else float("nan"),
        "contact_roots": len(contact_hits_oracle),
        "contact_top1_oracle_current": float(np.mean(contact_hits_oracle)) if contact_hits_oracle else float("nan"),
        "contact_top1_visual_current_proxy": float(np.mean(contact_hits_visual)) if contact_hits_visual else float("nan"),
        "mean_true_regret": float(np.mean(true_regrets)) if true_regrets else float("nan"),
        "contact_mean_true_regret": float(np.mean(contact_regrets)) if contact_regrets else float("nan"),
        "action_effect_rmse": float(np.sqrt(np.mean(np.square(effect_errors)))) if effect_errors else float("nan"),
    }


def contact_mode_gated_ranking(
    data: dict[str, np.ndarray], predicted: np.ndarray, goal_poses: np.ndarray,
    mode_probability: np.ndarray, penalty: float, effect_frame: str = "world",
    candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit a separation-probability penalty on held-out action support."""
    hits: list[float] = []
    contact_hits: list[float] = []
    for root_id in np.unique(data["root_id"]):
        indices = np.flatnonzero(data["root_id"] == root_id)
        if candidate_mask is not None:
            mask = np.asarray(candidate_mask, dtype=bool)
            indices = indices[mask[indices]]
        if len(indices) < 2:
            continue
        true = np.asarray([block_pose(row) for row in data["next_state"][indices]])
        goal = np.asarray(goal_poses[indices[0]], dtype=np.float32)
        goal_block = np.asarray([goal[0], goal[1], np.arctan2(float(goal[2]), float(goal[3]))], dtype=np.float32)
        true_cost = pose_cost(true, goal_block)
        current = block_pose(data["state"][indices[0]])
        predicted_zero = np.repeat(predicted[indices[0], :POSE_DIM][None], len(indices), axis=0)
        predicted_effect = predicted[indices, POSE_DIM:]
        if effect_frame == "contact":
            source = data["state"][indices[0]]
            predicted_effect = contact_effect_to_world(
                np.column_stack([np.zeros((len(indices), POSE_DIM), dtype=np.float32), predicted_effect]),
                source[:2], source[2:4],
            )[:, POSE_DIM:]
        predicted_next = np.tile(current, (len(indices), 1)) + predicted_zero + predicted_effect
        separation_risk = mode_probability[indices, 3] - mode_probability[indices[0], 3]
        score = pose_cost(predicted_next, goal_block) + float(penalty) * np.maximum(separation_risk, 0.0)
        hit = float(np.argmin(score) == np.argmin(true_cost))
        hits.append(hit)
        if bool(data["contact_root"][indices[0]]):
            contact_hits.append(hit)
    return {
        "penalty": float(penalty),
        "held_out_top1": float(np.mean(hits)) if hits else float("nan"),
        "held_out_contact_top1": float(np.mean(contact_hits)) if contact_hits else float("nan"),
        "held_out_roots": int(len(hits)),
        "held_out_contact_roots": int(len(contact_hits)),
    }


def visual_chart_pose(features: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    """Extract RGB-only XY and learned orientation for runtime control."""
    features = np.asarray(features, dtype=np.float32)
    # mask-moment layout: [mass, mean_x, mean_y, ...] per object.
    agent = (features[1:3] + 1.0) * 0.5 * PUSHT_STATE_SIZE
    block = (features[11:13] + 1.0) * 0.5 * PUSHT_STATE_SIZE
    return np.asarray([agent[0], agent[1], block[0], block[1], base_state[4], base_state[5]], dtype=np.float32)


@torch.inference_mode()
def runtime_observation(
    model: VisualCausalWorldModel, normalizer: Normalization, image: np.ndarray,
    previous: np.ndarray, goal_image: np.ndarray, image_size: int, device: str,
    goal_decoder: GoalPoseDecoder | None = None, goal_decoder_training: dict[str, Any] | None = None,
    orientation_decoder: OrientationDecoder | None = None,
    orientation_decoder_training: dict[str, Any] | None = None,
    goal_component_selection: dict[str, bool] | None = None,
    goal_xy_decoder: OrientationDecoder | None = None,
    goal_xy_decoder_training: dict[str, Any] | None = None,
    goal_angle_decoder: OrientationDecoder | None = None,
    goal_angle_decoder_training: dict[str, Any] | None = None,
    xy_decoder: OrientationDecoder | None = None,
    xy_decoder_training: dict[str, Any] | None = None,
    current_xy_policy: str = "validation",
    current_xy_disagreement_threshold: float = 60.0,
    paired_reference_policy: str = "raw",
    current_angle_policy: str = "model",
    rgb_current_angle_decoder: OrientationDecoder | None = None,
    rgb_current_angle_training: dict[str, Any] | None = None,
    rgb_goal_angle_decoder: OrientationDecoder | None = None,
    rgb_goal_angle_training: dict[str, Any] | None = None,
    rgb_goal_xy_decoder: OrientationDecoder | None = None,
    rgb_goal_xy_training: dict[str, Any] | None = None,
    rgb_angle_temporal_jump: float = 0.75,
    rgb_angle_disagreement_policy: str = "none",
    rgb_angle_disagreement_threshold: float = 1.20,
    rgb_angle_disagreement_terminal_xy: float = -1.0,
    rgb_temporal_angle_state: dict[str, Any] | None = None,
    rgb_temporal_angle_observer: bool = False,
    rgb_temporal_angle_max_delta: float = 0.55,
    rgb_temporal_angle_min_confidence: float = 0.004,
    max_observed_block_step: float = 48.0,
    held_block_xy: np.ndarray | None = None,
    held_agent_xy: np.ndarray | None = None,
    agent_xy_decoder: OrientationDecoder | None = None,
    agent_xy_decoder_training: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = tensor_images(resize_image(image, image_size)[None], device)
    previous_t = tensor_images(resize_image(previous, image_size)[None], device)
    goal = tensor_images(resize_image(goal_image, image_size)[None], device)
    features = model.encoder.features(current)[0].cpu().numpy()
    previous_features = model.encoder.features(previous_t)[0].cpu().numpy()
    goal_features = model.encoder.features(goal)[0].cpu().numpy()
    mean, std = normalizer.tensors(device)
    base_state = raw_from_normalized(model.encode_state(current, previous_t), mean, std)[0].cpu().numpy()
    if orientation_decoder is not None and orientation_decoder_training is not None:
        current_orientation = predict_orientation(
            orientation_decoder, features, orientation_decoder_training, device,
        )[0]
        base_state[4:6] = current_orientation
    latent_orientation = np.asarray(base_state[4:6], dtype=np.float32).copy()
    # ``rgb_shape`` is retained as a legacy negative-control protocol.  The
    # raw directed PCA angle of the rendered T is not a valid controller
    # interface (the renderer leaves a pi-scale ambiguity and contact
    # occlusion makes it discontinuous).  ``rgb_shape_gated`` deliberately
    # leaves the checkpoint angle untouched until the root-disjoint RGB
    # readout has passed its validation gate.
    if current_angle_policy == "rgb_shape":
        _current_xy, current_angle = rgb_geometry_xy_angle_from_image(image, "block")
        if current_angle is not None:
            base_state[4:6] = np.asarray([
                np.sin(current_angle), np.cos(current_angle),
            ], dtype=np.float32)
    # A root-disjoint RGB shape decoder is allowed to replace the frozen
    # current-angle head only after its held-out error gate has passed.  The
    # earlier implementation used this decoder for offline ranking features
    # but silently kept the frozen angle in the actual controller, creating a
    # train/runtime posture-interface mismatch.
    # ``rgb_current_angle_decoder`` is passed only after the root-disjoint
    # held-out gate has selected the RGB readout for runtime.  In that case
    # it must override the frozen angle head even when the user-facing policy
    # remains ``model``: the latter is the legacy default, whereas the
    # selected decoder is the declared deployment interface.  Before this
    # branch was widened, the offline ranker used the gated RGB angle but the
    # closed loop silently used the latent angle, creating a train/runtime
    # posture mismatch and making action ranking appear to fail under
    # otherwise accurate RGB orientation readout.
    if (
        (
            current_angle_policy in {"rgb_shape", "rgb_shape_gated"}
            or (
                current_angle_policy == "model"
                and rgb_current_angle_decoder is not None
                and rgb_current_angle_training is not None
            )
        )
        and rgb_current_angle_decoder is not None
        and rgb_current_angle_training is not None
    ):
        current_rgb_orientation = predict_orientation(
            rgb_current_angle_decoder,
            runtime_rgb_angle_features(
                image, "block", rgb_current_angle_training, image_size,
            ),
            rgb_current_angle_training,
            device,
        )[0]
        # Contact occlusion can flip the directed T-shape chart by nearly pi
        # for one frame even when the physical block rotates only slightly.
        # The previous rendered frame is part of the declared runtime input;
        # use it as a continuity guard rather than allowing a false visual
        # torque phase.  This is a readout safeguard, not a simulator-state
        # query.
        previous_rgb_orientation = predict_orientation(
            rgb_current_angle_decoder,
            runtime_rgb_angle_features(
                previous, "block", rgb_current_angle_training, image_size,
            ),
            rgb_current_angle_training,
            device,
        )[0]
        angle_jump = float(abs(wrap(
            np.arctan2(current_rgb_orientation[0], current_rgb_orientation[1])
            - np.arctan2(previous_rgb_orientation[0], previous_rgb_orientation[1])
        )))
        if angle_jump > float(rgb_angle_temporal_jump):
            current_rgb_orientation = previous_rgb_orientation
        selected_orientation = current_rgb_orientation
        temporal_audit: dict[str, float] = {}
        if bool(rgb_temporal_angle_observer) and rgb_temporal_angle_state is not None:
            image_key = id(image)
            if rgb_temporal_angle_state.get("image_key") == image_key:
                temporal_audit = dict(rgb_temporal_angle_state.get("audit", {}))
            else:
                registration = rgb_temporal_orientation_increment(
                    previous, image, max_delta=float(rgb_temporal_angle_max_delta),
                )
                selected_angle = float(np.arctan2(
                    current_rgb_orientation[0], current_rgb_orientation[1],
                ))
                temporal_audit = {
                    "temporal_angle": selected_angle,
                    "temporal_angle_delta": float(registration["delta"]),
                    "temporal_angle_confidence": float(registration["confidence"]),
                    "temporal_angle_best_score": float(registration["best_score"]),
                    "temporal_angle_second_score": float(registration["second_score"]),
                    "temporal_angle_visible_area": float(registration["visible_area"]),
                    "temporal_angle_update_used": float(
                        float(registration["valid"]) > 0.0
                        and float(registration["confidence"]) >= float(rgb_temporal_angle_min_confidence)
                    ),
                    "temporal_angle_branch_flip": float(abs(float(wrap(
                        selected_angle - float(np.arctan2(
                            previous_rgb_orientation[0], previous_rgb_orientation[1],
                        ))
                    ))) > 1.20),
                }
                # Do not integrate the increment into the absolute pose.  The
                # root-disjoint RGB KNN is the better absolute readout; the
                # temporal quantity is deliberately reserved for action
                # effect/confirmation below, where its local nature is useful.
                rgb_temporal_angle_state["angle"] = selected_angle
                rgb_temporal_angle_state["image_key"] = image_key
            rgb_temporal_angle_state["audit"] = temporal_audit
        disagreement = float(abs(wrap(
            np.arctan2(selected_orientation[0], selected_orientation[1])
            - np.arctan2(latent_orientation[0], latent_orientation[1])
        )))
        base_state[4:6] = selected_orientation
        if rgb_temporal_angle_state is not None:
            rgb_temporal_angle_state["last_selected_angle"] = float(
                np.arctan2(selected_orientation[0], selected_orientation[1])
            )
            rgb_temporal_angle_state["audit"] = temporal_audit
    base_goal_pose = (model.encode_goal_pose(goal) * std[2:6] + mean[2:6])[0].cpu().numpy()
    goal_pose = base_goal_pose.copy()
    selection = goal_component_selection or {"xy": True, "angle": True}
    if selection.get("xy_source") == "direct_moment":
        # In an aligned goal image the gray block remains the current scene
        # object; the green target mask encodes the evaluator goal.  Reading
        # BLOCK_FEATURE_START here silently set the goal to the current block
        # and produced an apparently perfect but useless validation distance.
        goal_pose[:2] = (goal_features[TARGET_FEATURE_START + 1:TARGET_FEATURE_START + 3] + 1.0) * (
            PUSHT_STATE_SIZE / 2.0
        )
    if goal_decoder is not None and goal_decoder_training is not None:
        decoded_goal_pose = predict_goal(goal_decoder, goal_features, goal_decoder_training, device)[0]
        if selection.get("xy", False):
            goal_pose[:2] = decoded_goal_pose[:2]
        if selection.get("angle", False):
            goal_pose[2:4] = decoded_goal_pose[2:4]
    if (
        selection.get("xy", False)
        and selection.get("xy_source") != "joint_decoder"
        and goal_xy_decoder is not None
        and goal_xy_decoder_training is not None
    ):
        goal_pose[:2] = predict_orientation(
            goal_xy_decoder,
            goal_features[TARGET_FEATURE_START:TARGET_FEATURE_STOP],
            goal_xy_decoder_training, device,
        )[0]
    if selection.get("angle", False) and goal_angle_decoder is not None and goal_angle_decoder_training is not None:
        decoded_goal_angle = predict_orientation(
            goal_angle_decoder, goal_features, goal_angle_decoder_training, device,
        )[0]
        angle_disagreement = float(abs(wrap(
            np.arctan2(decoded_goal_angle[0], decoded_goal_angle[1])
            - np.arctan2(goal_pose[2], goal_pose[3])
        )))
        disagreement_threshold = selection.get("angle_disagreement_threshold")
        if disagreement_threshold is not None and angle_disagreement > float(disagreement_threshold):
            # High-disagreement goal readouts are rejected at runtime.  The
            # fallback is the frozen checkpoint goal head already in
            # goal_pose; this gate is selected only from validation roots.
            pass
        else:
            angle_blend_weight = float(np.clip(selection.get("angle_blend_weight", 1.0), 0.0, 1.0))
            goal_pose[2:4] = (
                (1.0 - angle_blend_weight) * goal_pose[2:4]
                + angle_blend_weight * decoded_goal_angle
            )
    if rgb_goal_angle_decoder is not None and rgb_goal_angle_training is not None:
        goal_pose[2:4] = predict_orientation(
            rgb_goal_angle_decoder,
            runtime_rgb_angle_features(
                goal_image, "target", rgb_goal_angle_training, image_size,
            ),
            rgb_goal_angle_training,
            device,
        )[0]
    if selection.get("xy_source") == "rgb_mask":
        mask_xy, _mask_angle = rgb_geometry_xy_angle_from_image(goal_image, "target")
        if mask_xy is not None:
            goal_pose[:2] = mask_xy
    if selection.get("xy_source") == "rgb_calibrated":
        if rgb_goal_xy_decoder is not None and rgb_goal_xy_training is not None:
            goal_shape = rgb_geometry_shape_features(
                np.moveaxis(np.asarray(goal_image), -1, 0)[None], "target",
            )
            goal_pose[:2] = predict_orientation(
                rgb_goal_xy_decoder, goal_shape, rgb_goal_xy_training, device,
            )[0]
    if (
        current_xy_policy == "paired_rgb_calibrated"
        and rgb_goal_xy_decoder is not None
        and rgb_goal_xy_training is not None
    ):
        # The paired current/goal rigid chart was validated with the
        # root-disjoint RGB target readout.  Do not feed it the frozen latent
        # goal head at runtime: that creates two different coordinate origins
        # while the offline paired gate appears to pass.  This explicit branch
        # keeps the runtime interface identical to the validation protocol.
        paired_goal_shape = rgb_geometry_shape_features(
            np.moveaxis(np.asarray(goal_image), -1, 0)[None], "target",
        )
        goal_pose[:2] = predict_orientation(
            rgb_goal_xy_decoder, paired_goal_shape, rgb_goal_xy_training, device,
        )[0]
    # An uncalibrated directed target-mask PCA is not a valid goal-angle
    # interface on this renderer.  It used to overwrite a validated decoder
    # in the runtime path even though it was only intended as a ranker
    # fallback.  Goal angle now changes only through the base/learned decoder
    # above or through the explicitly gated RGB decoder.
    visual = visual_chart_pose(features, base_state)
    visual[:2] = observed_agent_xy(
        image, features, agent_xy_decoder, agent_xy_decoder_training, device,
        current_xy_policy, current_xy_disagreement_threshold,
        held_agent_xy=held_agent_xy,
    )
    if current_xy_policy == "paired_rgb_calibrated":
        previous_base_state = raw_from_normalized(
            model.encode_state(previous_t, previous_t), mean, std,
        )[0].cpu().numpy()
        if orientation_decoder is not None and orientation_decoder_training is not None:
            previous_base_state[4:6] = predict_orientation(
                orientation_decoder, previous_features, orientation_decoder_training, device,
            )[0]
        runtime_current_angle = float(np.arctan2(base_state[4], base_state[5]))
        runtime_previous_angle = float(np.arctan2(previous_base_state[4], previous_base_state[5]))
        runtime_goal_angle = float(np.arctan2(goal_pose[2], goal_pose[3]))
        paired_current = paired_rgb_current_xy_from_images(
            image, goal_image, goal_pose[:2],
            current_angle_override=runtime_current_angle,
            target_angle_override=runtime_goal_angle,
        )
        raw_current, _raw_current_angle = rgb_geometry_xy_angle_from_image(image, "block")
        calibrated_current = None
        if isinstance(xy_decoder, LinearRGBPoseCalibrator) and xy_decoder_training is not None:
            calibrated_shape = rgb_geometry_shape_features(
                np.moveaxis(np.asarray(image), -1, 0)[None], "block",
            )
            calibrated_current = predict_orientation(
                xy_decoder, calibrated_shape, xy_decoder_training, device,
            )[0]
        decoded_current = None
        if (
            xy_decoder is not None
            and xy_decoder_training is not None
            and not isinstance(xy_decoder, LinearRGBPoseCalibrator)
        ):
            decoded_current = predict_orientation(
                xy_decoder, features, xy_decoder_training, device,
            )[0]
        # The learned current-XY readout can be explicitly promoted as an
        # object-identity reference.  The default remains the legacy raw-first
        # protocol so old artifacts stay reproducible; the decoder policy is a
        # separate, auditable intervention for contact occlusion, where a raw
        # centroid may remain finite while tracking the pusher.
        decoder_reference_enabled = paired_reference_policy == "decoder"
        if (
            decoder_reference_enabled
            and decoded_current is not None
            and np.isfinite(decoded_current).all()
        ):
            current_reference = np.asarray(decoded_current, dtype=np.float32)
        else:
            current_reference = raw_current
        if (
            calibrated_current is not None
            and raw_current is not None
            and np.isfinite(calibrated_current).all()
            and float(np.linalg.norm(np.asarray(calibrated_current) - np.asarray(raw_current)))
            <= float(current_xy_disagreement_threshold)
        ):
            current_reference = np.asarray(calibrated_current, dtype=np.float32)
        if (
            paired_current is not None
            and current_reference is not None
            and float(np.linalg.norm(np.asarray(paired_current) - np.asarray(current_reference)))
            > float(current_xy_disagreement_threshold)
        ):
            # A directed PCA axis can differ by nearly pi between a current
            # blue T mask and an isolated green target mask.  Transporting a
            # large target centroid offset across that false relative angle
            # creates a catastrophic current-XY jump.  Keep the paired chart
            # only when it agrees with the simultaneously available raw
            # centroid; otherwise fail closed to the same RGB chart used by
            # local-effect finite differences.
            paired_current = np.asarray(current_reference, dtype=np.float32)
        elif paired_current is None and current_reference is not None:
            paired_current = np.asarray(current_reference, dtype=np.float32)
        # A visible mask can still be catastrophically biased by contact
        # occlusion (for example, the remaining T arm changes the centroid
        # without removing the mask).  Bound the innovation against the last
        # accepted RGB estimate as well as against the previous rendered
        # frame.  This is a causal observation filter: it uses no simulator
        # state and never predicts a successor.
        held = None if held_block_xy is None else np.asarray(held_block_xy, dtype=np.float32).reshape(-1)
        if (
            paired_current is not None
            and held is not None
            and held.shape[0] >= 2
            and np.isfinite(held[:2]).all()
            and np.isfinite(paired_current[:2]).all()
            and max_observed_block_step > 0.0
        ):
            innovation = np.asarray(paired_current[:2], dtype=np.float32) - held[:2]
            innovation_norm = float(np.linalg.norm(innovation))
            if innovation_norm > float(max_observed_block_step):
                paired_current = held[:2] + (
                    innovation / max(innovation_norm, 1.0e-6)
                    * float(max_observed_block_step)
                ).astype(np.float32)
        paired_previous = paired_rgb_current_xy_from_images(
            previous, goal_image, goal_pose[:2],
            current_angle_override=runtime_previous_angle,
            target_angle_override=runtime_goal_angle,
        )
        raw_previous, _raw_previous_angle = rgb_geometry_xy_angle_from_image(previous, "block")
        calibrated_previous = None
        if isinstance(xy_decoder, LinearRGBPoseCalibrator) and xy_decoder_training is not None:
            calibrated_previous_shape = rgb_geometry_shape_features(
                np.moveaxis(np.asarray(previous), -1, 0)[None], "block",
            )
            calibrated_previous = predict_orientation(
                xy_decoder, calibrated_previous_shape, xy_decoder_training, device,
            )[0]
        decoded_previous = None
        if (
            xy_decoder is not None
            and xy_decoder_training is not None
            and not isinstance(xy_decoder, LinearRGBPoseCalibrator)
        ):
            decoded_previous = predict_orientation(
                xy_decoder, previous_features, xy_decoder_training, device,
            )[0]
        # Apply the same declared reference policy to the previous frame.
        # Mixing decoder-first current with raw-centroid previous would recreate
        # an artificial finite-difference displacement.
        if (
            decoder_reference_enabled
            and decoded_previous is not None
            and np.isfinite(decoded_previous).all()
        ):
            previous_reference = np.asarray(decoded_previous, dtype=np.float32)
        else:
            previous_reference = raw_previous
        if (
            calibrated_previous is not None
            and raw_previous is not None
            and np.isfinite(calibrated_previous).all()
            and float(np.linalg.norm(np.asarray(calibrated_previous) - np.asarray(raw_previous)))
            <= float(current_xy_disagreement_threshold)
        ):
            previous_reference = np.asarray(calibrated_previous, dtype=np.float32)
        if (
            paired_previous is not None
            and previous_reference is not None
            and float(np.linalg.norm(np.asarray(paired_previous) - np.asarray(previous_reference)))
            > float(current_xy_disagreement_threshold)
        ):
            paired_previous = np.asarray(previous_reference, dtype=np.float32)
        elif paired_previous is None and previous_reference is not None:
            paired_previous = np.asarray(previous_reference, dtype=np.float32)
        if paired_current is None:
            # A fully occluded RGB block must not fall back to the frozen
            # moment sentinel (zero normalized coordinates -> [256, 256]).
            # The last accepted RGB estimate is a bounded observation hold,
            # not a simulator-state query or a successor prediction.
            held = None if held_block_xy is None else np.asarray(held_block_xy, dtype=np.float32).reshape(-1)
            if held is not None and held.shape[0] >= 2 and np.isfinite(held[:2]).all():
                visual[2:4] = held[:2]
                _observed_block_jump = 0.0
            else:
                visual[2:4], _observed_block_jump = temporally_filtered_block_xy(
                    image, previous, features, previous_features,
                    xy_decoder, xy_decoder_training, device,
                    "raw", current_xy_disagreement_threshold,
                    max_observed_block_step,
                )
        elif paired_previous is None:
            visual[2:4] = paired_current
            _observed_block_jump = 0.0
        else:
            delta = np.asarray(paired_current, dtype=np.float32) - np.asarray(paired_previous, dtype=np.float32)
            _observed_block_jump = float(np.linalg.norm(delta))
            if max_observed_block_step > 0.0 and np.isfinite(_observed_block_jump) and _observed_block_jump > max_observed_block_step:
                paired_current = np.asarray(paired_previous, dtype=np.float32) + (
                    delta / max(_observed_block_jump, 1.0e-6) * max_observed_block_step
                ).astype(np.float32)
            visual[2:4] = paired_current
    else:
        visual[2:4], _observed_block_jump = temporally_filtered_block_xy(
            image, previous, features, previous_features,
            xy_decoder, xy_decoder_training, device,
            current_xy_policy, current_xy_disagreement_threshold,
            max_observed_block_step,
        )
    # Apply the cross-observer gate after the declared current XY and goal
    # pose have been assembled.  This permits a terminal-only intervention:
    # disagreement during long translation is not allowed to replace a
    # useful RGB angle with a globally weaker latent angle, while a near-goal
    # disagreement can still block a false terminal torque caused by a pi
    # branch flip.  The gate remains RGB/latent observation-only.
    if (
        str(rgb_angle_disagreement_policy) == "latent_fallback"
        and rgb_current_angle_decoder is not None
        and rgb_current_angle_training is not None
        and disagreement > float(rgb_angle_disagreement_threshold)
    ):
        current_xy_for_gate = float(np.linalg.norm(goal_pose[:2] - visual[2:4]))
        terminal_limit = float(rgb_angle_disagreement_terminal_xy)
        if terminal_limit < 0.0 or current_xy_for_gate <= terminal_limit:
            base_state[4:6] = latent_orientation
            visual = visual_chart_pose(features, base_state)
            visual[:2] = observed_agent_xy(
                image, features, agent_xy_decoder, agent_xy_decoder_training, device,
                current_xy_policy, current_xy_disagreement_threshold,
                held_agent_xy=held_agent_xy,
            )
            if current_xy_policy in {"rgb_mask", "rgb_calibrated", "paired_rgb_calibrated"}:
                agent_xy, _agent_angle = rgb_geometry_xy_angle_from_image(image, "agent")
                if agent_xy is not None:
                    visual[:2] = agent_xy
    if current_xy_policy in {"rgb_mask", "rgb_calibrated", "paired_rgb_calibrated"}:
        # The contact/recovery interface must use the same RGB chart for the
        # pusher as for the block.  Leaving the agent in the latent moment
        # chart creates a second, hidden coordinate mismatch in the distance
        # and approach tests.
        agent_xy, _agent_angle = rgb_geometry_xy_angle_from_image(image, "agent")
        if agent_xy is not None:
            visual[:2] = agent_xy
        elif held_agent_xy is not None:
            held = np.asarray(held_agent_xy, dtype=np.float32).reshape(-1)
            if held.shape[0] >= 2 and np.isfinite(held[:2]).all():
                visual[:2] = held[:2]
    return visual, features, previous_features, goal_pose


@torch.inference_mode()
def observed_local_effect(
    model: VisualCausalWorldModel, normalizer: Normalization,
    image: np.ndarray, previous: np.ndarray, image_size: int, device: str,
    orientation_decoder: OrientationDecoder | None = None,
    orientation_decoder_training: dict[str, Any] | None = None,
    goal_component_selection: dict[str, bool] | None = None,
    xy_decoder: OrientationDecoder | None = None,
    xy_decoder_training: dict[str, Any] | None = None,
    current_xy_policy: str = "raw",
    current_xy_disagreement_threshold: float = 60.0,
    rgb_current_angle_decoder: OrientationDecoder | None = None,
    rgb_current_angle_training: dict[str, Any] | None = None,
    rgb_angle_temporal_jump: float = 0.75,
    rgb_temporal_angle_observer: bool = False,
    rgb_temporal_angle_max_delta: float = 0.55,
    rgb_temporal_angle_min_confidence: float = 0.004,
    max_observed_block_step: float = 48.0,
) -> np.ndarray:
    """Measure a previous action's local effect from two RGB frames only."""
    current = tensor_images(resize_image(image, image_size)[None], device)
    previous_t = tensor_images(resize_image(previous, image_size)[None], device)
    current_features = model.encoder.features(current)[0].cpu().numpy()
    previous_features = model.encoder.features(previous_t)[0].cpu().numpy()
    current_block, _observed_block_jump = temporally_filtered_block_xy(
        image, previous, current_features, previous_features,
        xy_decoder, xy_decoder_training, device,
        current_xy_policy, current_xy_disagreement_threshold,
        max_observed_block_step,
    )
    previous_block = observed_block_xy(
        previous, previous_features, xy_decoder, xy_decoder_training, device,
        current_xy_policy, current_xy_disagreement_threshold,
    )
    block_delta = current_block - previous_block
    mean, std = normalizer.tensors(device)
    current_state = raw_from_normalized(model.encode_state(current, current), mean, std)[0].cpu().numpy()
    previous_state = raw_from_normalized(model.encode_state(previous_t, previous_t), mean, std)[0].cpu().numpy()
    if orientation_decoder is not None and orientation_decoder_training is not None:
        current_state[4:6] = predict_orientation(
            orientation_decoder, current_features, orientation_decoder_training, device,
        )[0]
        previous_state[4:6] = predict_orientation(
            orientation_decoder, previous_features, orientation_decoder_training, device,
        )[0]
    # The controller may use a root-disjoint RGB angle readout while the
    # frozen checkpoint still exposes a latent angle head. Mixing those two
    # interfaces in the online finite difference makes the local Jacobian
    # estimate predict a different quantity from the one used by the
    # candidate objective. Prefer the exact runtime RGB angle interface when
    # it has passed its validation gate; otherwise retain the legacy latent
    # readout for backward-compatible protocols.
    if rgb_current_angle_decoder is not None and rgb_current_angle_training is not None:
        current_orientation = predict_orientation(
            rgb_current_angle_decoder,
            runtime_rgb_angle_features(
                image, "block", rgb_current_angle_training, image_size,
            ),
            rgb_current_angle_training,
            device,
        )[0]
        previous_orientation = predict_orientation(
            rgb_current_angle_decoder,
            runtime_rgb_angle_features(
                previous, "block", rgb_current_angle_training, image_size,
            ),
            rgb_current_angle_training,
            device,
        )[0]
        angle_delta = float(wrap(
            np.arctan2(current_orientation[0], current_orientation[1])
            - np.arctan2(previous_orientation[0], previous_orientation[1])
        ))
        # A one-frame directed-shape flip is an observation failure candidate,
        # not evidence for a large physical torque effect. Exclude it from
        # online identification instead of allowing it to dominate the fit.
        if abs(angle_delta) > float(rgb_angle_temporal_jump):
            angle_delta = 0.0
        if bool(rgb_temporal_angle_observer):
            registration = rgb_temporal_orientation_increment(
                previous, image, max_delta=float(rgb_temporal_angle_max_delta),
            )
            if (
                float(registration["valid"]) > 0.0
                and float(registration["confidence"]) >= float(rgb_temporal_angle_min_confidence)
            ):
                angle_delta = float(registration["delta"])
    else:
        angle_delta = float(wrap(theta_from_pose(current_state) - theta_from_pose(previous_state)))
    return np.asarray([block_delta[0], block_delta[1], angle_delta], dtype=np.float32)


def project_contact_action(
    action: np.ndarray,
    image: np.ndarray,
    args: argparse.Namespace,
    phase: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a final RGB-only contact-manifold projection to one action.

    The learned selector remains responsible for ranking.  This layer only
    repairs an immediately infeasible action after ranking, using the same
    color-mask geometry used by the controller.  It is deliberately separate
    from candidate filtering: filtering can leave an all-invalid support and
    silently trigger a bad fallback, whereas projection preserves the chosen
    tangential component and adds the smallest normal correction.
    """
    raw = np.asarray(action, dtype=np.float32).reshape(2)
    info = {
        "contact_action_projection_enabled": float(
            bool(getattr(args, "contact_action_projection", False))
        ),
        "contact_action_projection_used": 0.0,
        "contact_projection_raw_inward": float("nan"),
        "contact_projection_final_inward": float("nan"),
        "contact_projection_correction_norm": 0.0,
        "contact_projection_contact_band": 0.0,
    }
    if not bool(getattr(args, "contact_action_projection", False)):
        return raw, info
    if phase in {"approach", "recovery", "hold"}:
        return raw, info
    agent, _ = rgb_geometry_xy_angle_from_image(image, "agent")
    block, _ = rgb_geometry_xy_angle_from_image(image, "block")
    if agent is None or block is None:
        return raw, info
    separation = block - agent
    distance = float(np.linalg.norm(separation))
    contact_band = distance <= (
        float(args.contact_distance) + 2.0 * float(args.contact_tolerance)
    )
    contact_lock = distance <= float(getattr(args, "contact_lock_distance", 100.0))
    if not (contact_band or contact_lock):
        return raw, info
    normal = separation / max(distance, 1.0e-6)
    # Translation and terminal torque have different feasible manifolds.  A
    # positive inward margin is useful while repairing orientation, but
    # imposing the same margin during long translation steps converts a
    # tangential push into repeated inward pushes and can stall the block.
    # Reuse the separately declared translation margin (zero by default) so
    # the projection repairs only outward escape during translation.
    if phase == "translate":
        minimum = float(max(0.0, getattr(args, "contact_translation_inward_min", 0.0)))
    else:
        minimum = float(max(0.0, getattr(args, "contact_projection_min_inward", 0.08)))
    raw_inward = float(np.dot(raw, normal))
    projected = raw.copy()
    if raw_inward < minimum:
        projected = projected + (minimum - raw_inward) * normal
    projected = np.clip(projected, -1.0, 1.0).astype(np.float32)
    # Box clipping can reduce the normal component for a diagonal normal.
    # Use the smallest feasible pure-normal correction only in that rare case;
    # it keeps the promised half-space invariant without querying a successor.
    if float(np.dot(projected, normal)) < minimum:
        projected = (minimum * normal).astype(np.float32)
    final_inward = float(np.dot(projected, normal))
    correction = float(np.linalg.norm(projected - raw))
    info.update({
        "contact_projection_raw_inward": raw_inward,
        "contact_projection_final_inward": final_inward,
        "contact_projection_correction_norm": correction,
        "contact_projection_contact_band": float(contact_band or contact_lock),
        "contact_action_projection_used": float(correction > 1.0e-7),
    })
    return projected, info


def runtime_features(
    features: np.ndarray, previous: np.ndarray, action: np.ndarray,
    boundary_chart: np.ndarray | None = None,
    previous_action: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(1, 2)
    current = np.asarray(features, dtype=np.float32).reshape(1, -1)
    previous = np.asarray(previous, dtype=np.float32).reshape(1, -1)
    chart = contact_chart_from_features(current, action)
    if boundary_chart is None:
        boundary_chart = np.zeros((1, 10), dtype=np.float32)
    boundary_chart = np.asarray(boundary_chart, dtype=np.float32)
    if boundary_chart.ndim == 1:
        boundary_chart = boundary_chart[None]
    if boundary_chart.shape != (1, 10):
        raise ValueError("boundary_chart must have shape [10] or [1, 10]")
    values = np.concatenate([current, current - previous, action, chart, boundary_chart], axis=1)
    if previous_action is not None:
        previous_action = np.asarray(previous_action, dtype=np.float32).reshape(1, 2)
        values = np.concatenate([values, previous_action], axis=1)
    return values


def choose_adapter_action(
    model: VisualCausalWorldModel, normalizer: Normalization, adapter: LocalEffectAdapter,
    adapter_training: dict[str, Any], image: np.ndarray, previous: np.ndarray, goal_image: np.ndarray,
    args: argparse.Namespace, phase: str, local_history: list[tuple[np.ndarray, np.ndarray]] | None = None,
    goal_decoder: GoalPoseDecoder | None = None, goal_decoder_training: dict[str, Any] | None = None,
    orientation_decoder: OrientationDecoder | None = None,
    orientation_decoder_training: dict[str, Any] | None = None,
    goal_component_selection: dict[str, bool] | None = None,
    goal_xy_decoder: OrientationDecoder | None = None,
    goal_xy_decoder_training: dict[str, Any] | None = None,
    goal_angle_decoder: OrientationDecoder | None = None,
    goal_angle_decoder_training: dict[str, Any] | None = None,
    xy_decoder: OrientationDecoder | None = None,
    xy_decoder_training: dict[str, Any] | None = None,
    action_cost_ranker: ActionCostRanker | None = None,
    action_cost_ranker_training: dict[str, Any] | None = None,
    terminal_angular_ranker: ActionCostRanker | None = None,
    terminal_angular_ranker_training: dict[str, Any] | None = None,
    contact_mode_classifier: ContactModeClassifier | None = None,
    contact_mode_training: dict[str, Any] | None = None,
    contact_mode_penalty: float = 0.0,
    recovery_cooldown: int = 0,
    previous_action: np.ndarray | None = None,
    contact_knn_training: dict[str, Any] | None = None,
    rgb_current_angle_decoder: OrientationDecoder | None = None,
    rgb_current_angle_training: dict[str, Any] | None = None,
    rgb_goal_angle_decoder: OrientationDecoder | None = None,
    rgb_goal_angle_training: dict[str, Any] | None = None,
    rgb_goal_xy_decoder: OrientationDecoder | None = None,
    rgb_goal_xy_training: dict[str, Any] | None = None,
    contact_side_misalignment_count: int = 0,
    recovery_anchor: dict[str, Any] | None = None,
    terminal_confirmation_count: int = 0,
    terminal_stagnation_count: int = 0,
    reacquisition_progress_count: int = 0,
    reacquisition_phase_steps: int = 0,
    reacquisition_total_steps: int = 0,
    reacquisition_motion_excess_count: int = 0,
    reacquisition_route_abort_count: int = 0,
    reacquisition_lockout_steps: int = 0,
    reacquisition_failure_repair_used: bool = False,
    agent_xy_decoder: OrientationDecoder | None = None,
    agent_xy_decoder_training: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str, dict[str, float]]:
    # Some terminal angle-progress branches are reached after a contact-side
    # transition that does not construct a recovery audit.  Keep the optional
    # audit payload total so enabling the guard cannot crash an otherwise
    # valid recovery/torque rollout.
    recovery_audit: dict[str, Any] = {}
    visual, features, previous_features, goal_pose_raw = runtime_observation(
        model, normalizer, image, previous, goal_image, args.image_size, args.device,
        goal_decoder, goal_decoder_training,
        orientation_decoder, orientation_decoder_training,
        goal_component_selection,
        goal_xy_decoder, goal_xy_decoder_training,
        goal_angle_decoder, goal_angle_decoder_training,
        xy_decoder, xy_decoder_training,
        current_xy_policy=runtime_current_xy_policy(args),
        current_xy_disagreement_threshold=getattr(args, "current_xy_disagreement_threshold", 60.0),
        paired_reference_policy=getattr(args, "paired_reference_policy", "raw"),
        current_angle_policy=getattr(args, "current_angle_policy", "model"),
        rgb_current_angle_decoder=(
            rgb_current_angle_decoder
            if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
        ),
        rgb_current_angle_training=(
            rgb_current_angle_training
            if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
        ),
        rgb_goal_angle_decoder=rgb_goal_angle_decoder,
        rgb_goal_angle_training=rgb_goal_angle_training,
        rgb_goal_xy_decoder=rgb_goal_xy_decoder,
        rgb_goal_xy_training=rgb_goal_xy_training,
        rgb_angle_temporal_jump=getattr(args, "rgb_angle_temporal_jump", 0.75),
        rgb_angle_disagreement_policy=getattr(args, "rgb_angle_disagreement_policy", "none"),
        rgb_angle_disagreement_threshold=getattr(args, "rgb_angle_disagreement_threshold", 1.20),
        rgb_angle_disagreement_terminal_xy=getattr(args, "rgb_angle_disagreement_terminal_xy", -1.0),
        rgb_temporal_angle_state=getattr(args, "_runtime_temporal_angle_state", None),
        rgb_temporal_angle_observer=bool(getattr(args, "_use_rgb_temporal_angle_runtime", False)),
        rgb_temporal_angle_max_delta=getattr(args, "rgb_temporal_angle_max_delta", 0.55),
        rgb_temporal_angle_min_confidence=getattr(args, "rgb_temporal_angle_min_confidence", 0.004),
        max_observed_block_step=getattr(args, "rgb_max_observed_block_step", 48.0),
        held_block_xy=getattr(args, "_runtime_last_block_xy", None),
        held_agent_xy=getattr(args, "_runtime_last_agent_xy", None),
        agent_xy_decoder=agent_xy_decoder,
        agent_xy_decoder_training=agent_xy_decoder_training,
    )
    if np.isfinite(visual[2:4]).all():
        args._runtime_last_block_xy = np.asarray(visual[2:4], dtype=np.float32).copy()
    # Do not retain the latent-moment occlusion sentinel as a usable RGB
    # position.  A later fully occluded frame can then hold the last reliable
    # agent chart instead of steering from [256, 256].
    if (
        np.isfinite(visual[:2]).all()
        and float(np.linalg.norm(np.asarray(visual[:2], dtype=np.float32) - 256.0)) > 1.0e-3
    ):
        args._runtime_last_agent_xy = np.asarray(visual[:2], dtype=np.float32).copy()
    agent, block = visual[:2], visual[2:4]
    robust_runtime_block = update_runtime_block_history(args, block)
    goal_xy = goal_pose_raw[:2]
    preterminal_xy = float(np.linalg.norm(goal_xy - block))
    preterminal_goal_angle = float(np.arctan2(goal_pose_raw[2], goal_pose_raw[3]))
    preterminal_current_angle = float(np.arctan2(visual[4], visual[5]))
    temporal_audit = getattr(args, "_runtime_temporal_angle_state", {}).get("audit", {})
    temporal_delta = float(temporal_audit.get("temporal_angle_delta", 0.0))
    temporal_delta_valid = bool(
        bool(getattr(args, "_use_rgb_temporal_angle_pulse", False))
        and float(temporal_audit.get("temporal_angle_update_used", 0.0)) > 0.0
    )
    # Keep the absolute RGB KNN angle for control/ranking, but expose the
    # local temporal increment to the pulse confirmation interface.  This is
    # the only place where a framewise-frozen absolute readout is insufficient.
    pulse_observed_angle = float(wrap(
        preterminal_current_angle + (temporal_delta if temporal_delta_valid else 0.0)
    ))
    preterminal_signed_angle_error = float(wrap(
        preterminal_goal_angle - preterminal_current_angle
    ))
    preterminal_angle_error = float(abs(preterminal_signed_angle_error))
    pulse_observed_angle_error = float(abs(wrap(
        preterminal_goal_angle - pulse_observed_angle
    )))
    pulse_guard_enabled = bool(getattr(args, "terminal_pulse_confirmation", False))
    pulse_guard_state = getattr(args, "_terminal_pulse_state", None)
    if not isinstance(pulse_guard_state, dict):
        pulse_guard_state = {}
        args._terminal_pulse_state = pulse_guard_state
    if pulse_guard_enabled and bool(pulse_guard_state.get("active", False)):
        # The state was armed immediately before the previous action was
        # executed.  The current RGB frame is the only confirmation signal;
        # no simulator successor or hidden state enters this decision.
        angle_gain = float(
            pulse_guard_state.get("angle_before", pulse_observed_angle_error)
            - pulse_observed_angle_error
        )
        xy_regression = float(
            preterminal_xy - pulse_guard_state.get("xy_before", preterminal_xy)
        )
        pulse_kind = str(pulse_guard_state.get("kind", "learned"))
        pulse_guard_state["active"] = False
        pulse_guard_state["last_angle_gain"] = angle_gain
        pulse_guard_state["last_xy_regression"] = xy_regression
        pulse_guard_state["last_kind"] = pulse_kind
        angle_ok = angle_gain >= -float(
            getattr(args, "terminal_pulse_max_angle_regression", 0.08)
        )
        minimum_gain = float(getattr(args, "terminal_pulse_min_angle_gain", 0.0))
        if minimum_gain > 0.0:
            angle_ok = angle_ok and angle_gain >= minimum_gain
        pulse_guard_state["last_confirmed"] = bool(
            angle_ok
            and xy_regression <= float(getattr(args, "terminal_pulse_max_xy_regression", 2.0))
        )
        if pulse_guard_state["last_confirmed"]:
            pulse_guard_state["last_outcome"] = "confirmed"
        else:
            failures = int(pulse_guard_state.get("failures", 0)) + 1
            pulse_guard_state["failures"] = failures
            pulse_guard_state["last_outcome"] = "failed"
            if pulse_kind == "learned" and failures <= int(
                max(0, getattr(args, "terminal_pulse_max_failures", 2))
            ) and phase in {"torque", "hold"}:
                # Do not give the same learned ranking another immediate
                # chance.  Reverse only its tangential component and retain a
                # small inward component so the fallback stays on contact.
                fallback_cap = float(getattr(args, "terminal_pulse_fallback_magnitude", 0.06))
                fallback = terminal_pulse_fallback_action(
                    previous_action,
                    agent,
                    block,
                    fallback_cap,
                    inward_min=float(getattr(args, "terminal_contact_inward_min", 0.02)),
                )
                pulse_guard_state["last_outcome"] = "fallback"
                return fallback, "torque", {
                    "approach_distance": float("nan"),
                    "observed_agent_block_distance": float("nan"),
                    "observed_block_motion": float("nan"),
                    "angle_error_visual": preterminal_angle_error,
                    "block_xy_visual": preterminal_xy,
                    "terminal_pulse_confirmation_used": 1.0,
                    "terminal_pulse_failed": 1.0,
                    "terminal_pulse_fallback_used": 1.0,
                    "terminal_pulse_angle_gain": angle_gain,
                    "terminal_pulse_xy_regression": xy_regression,
                    "terminal_pulse_failure_count": float(failures),
                    "terminal_pulse_kind": "fallback",
                    "predicted_best_cost": float("nan"),
                    "action_cost_ranker_used": 0.0,
                }
            pulse_guard_state["disabled"] = True
            pulse_guard_state["freeze_remaining"] = max(
                1, int(getattr(args, "terminal_pulse_freeze_steps", 1))
            )
    if pulse_guard_enabled and int(pulse_guard_state.get("freeze_remaining", 0)) > 0 and phase in {"torque", "hold"}:
        remaining = int(pulse_guard_state.get("freeze_remaining", 0))
        if remaining > 0:
            pulse_guard_state["freeze_remaining"] = remaining - 1
        return np.zeros(2, dtype=np.float32), "torque", {
            "approach_distance": float("nan"),
            "observed_agent_block_distance": float("nan"),
            "observed_block_motion": float("nan"),
            "angle_error_visual": preterminal_angle_error,
            "block_xy_visual": preterminal_xy,
            "terminal_pulse_confirmation_used": 1.0,
            "terminal_pulse_frozen": 1.0,
            "terminal_pulse_failure_count": float(pulse_guard_state.get("failures", 0)),
            "terminal_pulse_freeze_remaining": float(pulse_guard_state.get("freeze_remaining", 0)),
            "predicted_best_cost": float("nan"),
            "action_cost_ranker_used": 0.0,
        }
    # The learned effect score is intentionally not trusted to determine the
    # signed terminal torque.  When enabled, the controller first emits a
    # small RGB-only excitation and uses the next-frame angle error as an
    # online falsification test.  State lives on `args` because the public
    # controller signature is already used by several audited protocols.
    signed_probe_state = getattr(args, "_signed_torque_probe_state", None)
    if not isinstance(signed_probe_state, dict):
        signed_probe_state = {}
        args._signed_torque_probe_state = signed_probe_state
    signed_probe_feedback_gain = float("nan")
    signed_probe_feedback_used = False
    signed_probe_settle = False
    agent_mask_xy, _agent_mask_angle = rgb_geometry_xy_angle_from_image(image, "agent")
    agent_mask_visible = bool(
        agent_mask_xy is not None and np.isfinite(np.asarray(agent_mask_xy, dtype=np.float32)).all()
    )
    if (
        bool(getattr(args, "rgb_agent_occlusion_discovery", False))
        and not agent_mask_visible
        and not bool(signed_probe_state.get("agent_discovery_visible", False))
        and phase in {"approach", "push", "translate", "torque", "hold", "terminal_approach", "recovery"}
    ):
        discovery_pulses = int(signed_probe_state.get("agent_discovery_pulses", 0))
        discovery_max = max(
            1, int(getattr(args, "rgb_agent_occlusion_discovery_max_pulses", 2))
        )
        if discovery_pulses < discovery_max:
            discovery_direction = goal_xy - block
            discovery_direction /= max(float(np.linalg.norm(discovery_direction)), 1.0e-6)
            discovery_magnitude = float(np.clip(
                getattr(args, "rgb_agent_occlusion_discovery_magnitude", 0.12),
                0.0, 0.35,
            ))
            discovery_action = (-discovery_magnitude * discovery_direction).astype(np.float32)
            signed_probe_state["agent_discovery_pulses"] = discovery_pulses + 1
            return discovery_action, "recovery", {
                "rgb_agent_occlusion_discovery_used": 1.0,
                "rgb_agent_occlusion_discovery_pulses": float(discovery_pulses + 1),
                "rgb_agent_occlusion_discovery_max_pulses": float(discovery_max),
                "rgb_agent_occlusion_discovery_agent_visible": 0.0,
                "contact_lost_recovery": 1.0,
                "predicted_best_cost": float("nan"),
                "action_cost_ranker_used": 0.0,
            }
        # Do not let a permanently occluded agent silently fall through to
        # the torque/ranker interface.  A zero action is the only safe
        # continuation once the declared discovery budget is exhausted.
        return np.zeros(2, dtype=np.float32), "recovery", {
            "rgb_agent_occlusion_discovery_used": 0.0,
            "rgb_agent_occlusion_discovery_pulses": float(discovery_pulses),
            "rgb_agent_occlusion_discovery_max_pulses": float(discovery_max),
            "rgb_agent_occlusion_discovery_agent_visible": 0.0,
            "rgb_agent_occlusion_discovery_exhausted": 1.0,
            "contact_lost_recovery": 1.0,
            "predicted_best_cost": float("nan"),
            "action_cost_ranker_used": 0.0,
        }
    if agent_mask_visible:
        signed_probe_state["agent_discovery_visible"] = True
    if (
        bool(getattr(args, "online_signed_torque_probe", False))
        and bool(signed_probe_state.get("active", False))
        and previous_action is not None
        and np.isfinite(preterminal_angle_error)
    ):
        signed_probe_feedback_gain = float(
            float(signed_probe_state.get("error_before", preterminal_angle_error))
            - preterminal_angle_error
        )
        signed_probe_feedback_used = True
        signed_probe_state["active"] = False
        signed_probe_state["last_gain"] = signed_probe_feedback_gain
        signed_probe_xy_regression = float(
            preterminal_xy
            - float(signed_probe_state.get("xy_before", preterminal_xy))
        )
        signed_probe_state["last_xy_regression"] = signed_probe_xy_regression
        if "anchor_xy" in signed_probe_state:
            signed_probe_cumulative_xy = float(
                preterminal_xy - float(signed_probe_state["anchor_xy"])
            )
        else:
            signed_probe_cumulative_xy = 0.0
        signed_probe_state["last_cumulative_xy_drift"] = signed_probe_cumulative_xy
        min_gain = float(max(
            0.0, getattr(args, "signed_torque_probe_min_gain", 0.02),
        ))
        max_xy_regression = float(max(
            0.0, getattr(args, "signed_torque_probe_max_xy_regression", 3.0),
        ))
        xy_safe = signed_probe_xy_regression <= max_xy_regression
        max_cumulative_xy = float(max(
            0.0, getattr(args, "signed_torque_probe_max_cumulative_xy_drift", 3.0),
        ))
        cumulative_xy_safe = signed_probe_cumulative_xy <= max_cumulative_xy
        if signed_probe_feedback_gain >= min_gain and xy_safe:
            signed_probe_state["validated"] = True
            signed_probe_state["last_feedback"] = "accepted"
            if signed_probe_feedback_gain >= float(
                signed_probe_state.get("best_gain", -np.inf)
            ):
                signed_probe_state["best_gain"] = signed_probe_feedback_gain
                signed_probe_state["best_orientation_sign"] = float(
                    signed_probe_state.get("orientation_sign", 1.0)
                )
            signed_probe_settle = False
        elif not bool(signed_probe_state.get("validated", False)):
            if "best_orientation_sign" in signed_probe_state:
                # Preserve the best previously observed causal direction.  A
                # later pulse can be angle-positive yet XY-unsafe, so blindly
                # flipping after that pulse would discard useful local
                # identification and create a deterministic sign oscillation.
                signed_probe_state["orientation_sign"] = float(
                    signed_probe_state["best_orientation_sign"]
                )
                signed_probe_state["validated"] = True
                signed_probe_state["last_feedback"] = (
                    "revert_best_xy_unsafe" if not xy_safe else "revert_best_uninformative"
                )
            else:
                # A worsening or uninformative pulse is evidence that the
                # current signed tangent hypothesis is wrong or too weak.
                # Flip once and retest, with a hard exploratory-pulse budget.
                signed_probe_state["orientation_sign"] = -float(
                    signed_probe_state.get("orientation_sign", 1.0)
                )
                signed_probe_state["flips"] = int(signed_probe_state.get("flips", 0)) + 1
                signed_probe_state["last_feedback"] = (
                    "flip_xy_unsafe" if not xy_safe else "flip"
                )
            if not xy_safe:
                args._signed_torque_probe_magnitude_scale = float(np.clip(
                    float(getattr(args, "_signed_torque_probe_magnitude_scale", 1.0))
                    * float(np.clip(
                        getattr(args, "signed_torque_probe_shrink_factor", 0.75),
                        0.25, 1.0,
                    )),
                    0.05, 1.0,
                ))
            signed_probe_settle = True
        else:
            # Once a direction was validated, a later adverse pulse triggers
            # one online sign reversal rather than silently returning to the
            # uncalibrated learned ranker.
            if "best_orientation_sign" in signed_probe_state:
                signed_probe_state["orientation_sign"] = float(
                    signed_probe_state["best_orientation_sign"]
                )
            else:
                signed_probe_state["orientation_sign"] = -float(
                    signed_probe_state.get("orientation_sign", 1.0)
                )
                signed_probe_state["flips"] = int(signed_probe_state.get("flips", 0)) + 1
            signed_probe_state["validated"] = True
            signed_probe_state["last_feedback"] = (
                "validated_revert_best_xy_unsafe"
                if not xy_safe else "validated_revert_best"
            )
            if not xy_safe:
                args._signed_torque_probe_magnitude_scale = float(np.clip(
                    float(getattr(args, "_signed_torque_probe_magnitude_scale", 1.0))
                    * float(np.clip(
                        getattr(args, "signed_torque_probe_shrink_factor", 0.75),
                        0.25, 1.0,
                    )),
                    0.05, 1.0,
                ))
            signed_probe_settle = True
        if not cumulative_xy_safe:
            signed_probe_state["failed"] = True
            signed_probe_state["last_feedback"] = "cumulative_xy_drift_fail_closed"
            signed_probe_settle = False
        elif (
            bool(getattr(args, "signed_torque_probe_xy_repair", False))
            and signed_probe_xy_regression
            >= float(getattr(args, "signed_torque_probe_xy_repair_threshold", 1.0))
        ):
            # The angle pulse may be locally correct yet task-unsafe.  Mark a
            # separate one-step XY repair instead of asking the same terminal
            # torque primitive to undo its own translation.
            signed_probe_state["xy_repair_pending"] = True
            signed_probe_state["last_feedback"] = "xy_repair_pending"
        signed_probe_state["settle_pending"] = bool(signed_probe_settle)
    probe_agent_block_distance = float(np.linalg.norm(agent - block))
    probe_direction = goal_xy - block
    probe_direction /= max(float(np.linalg.norm(probe_direction)), 1.0e-6)
    probe_contact_normal = block - agent
    probe_contact_normal /= max(float(np.linalg.norm(probe_contact_normal)), 1.0e-6)
    probe_contact_alignment = float(np.dot(probe_contact_normal, probe_direction))
    probe_contact_ready = bool(
        preterminal_xy <= float(getattr(args, "terminal_xy", 35.0))
        and probe_agent_block_distance <= float(args.contact_distance) + float(args.contact_tolerance)
        and probe_contact_alignment >= float(getattr(args, "contact_side_alignment_min", 0.35))
    )
    probe_gap_max = float(getattr(args, "terminal_contact_gap_max", 0.0))
    probe_lease_valid, probe_gap_px, probe_support_visible = (
        terminal_probe_contact_lease_from_image(
            image, probe_gap_max, int(args.image_size),
        )
    )
    if probe_gap_max > 0.0:
        # Do not let a centroid/side-only contact estimate launch a signed
        # excitation.  Without this gate, a missed support point can produce
        # an apparently improving RGB angle while the physical block remains
        # stationary, poisoning the online sign identification.
        probe_contact_ready = bool(probe_contact_ready and probe_lease_valid)
    probe_contact_audit = {
        "signed_torque_probe_contact_gap_gate_used": float(probe_gap_max > 0.0),
        "signed_torque_probe_contact_gap_px": probe_gap_px,
        "signed_torque_probe_boundary_support_visible": float(probe_support_visible),
        "signed_torque_probe_contact_lease_valid": float(probe_lease_valid),
    }
    if (
        bool(getattr(args, "online_signed_torque_probe", False))
        and bool(getattr(args, "signed_torque_probe_xy_repair", False))
        and bool(signed_probe_state.pop("xy_repair_pending", False))
        and probe_contact_ready
    ):
        repair_magnitude = float(np.clip(
            getattr(args, "signed_torque_probe_xy_repair_magnitude", 0.16),
            0.0, 0.35,
        ))
        repair_action = (repair_magnitude * probe_direction).astype(np.float32)
        return repair_action, "signed_probe_xy_repair", {
            "signed_torque_probe_used": 0.0,
            "signed_torque_probe_xy_repair_used": 1.0,
            "signed_torque_probe_xy_repair_magnitude": repair_magnitude,
            "signed_torque_probe_feedback_used": float(signed_probe_feedback_used),
            "signed_torque_probe_feedback_gain": signed_probe_feedback_gain,
            "signed_torque_probe_xy_regression": float(
                signed_probe_state.get("last_xy_regression", float("nan"))
            ),
            "signed_torque_probe_cumulative_xy_drift": float(
                signed_probe_state.get("last_cumulative_xy_drift", float("nan"))
            ),
            "signed_torque_probe_contact_alignment": probe_contact_alignment,
            "signed_torque_probe_agent_block_distance": probe_agent_block_distance,
            "contact_lost_recovery": 0.0,
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            **probe_contact_audit,
        }
    if (
        bool(getattr(args, "online_signed_torque_probe", False))
        and bool(signed_probe_state.pop("settle_pending", False))
        and probe_contact_ready
    ):
        # A failed or XY-unsafe excitation gets one zero-action settle before
        # another probe.  This separates the measured response from the
        # controller's own corrective pulse and prevents a rapid probe/reverse
        # oscillation from ejecting the pusher.
        return np.zeros(2, dtype=np.float32), "torque", {
            "signed_torque_probe_used": 0.0,
            "signed_torque_probe_settle_used": 1.0,
            "signed_torque_probe_feedback_used": float(signed_probe_feedback_used),
            "signed_torque_probe_feedback_gain": signed_probe_feedback_gain,
            "signed_torque_probe_xy_regression": float(
                signed_probe_state.get("last_xy_regression", float("nan"))
            ),
            "signed_torque_probe_cumulative_xy_drift": float(
                signed_probe_state.get("last_cumulative_xy_drift", float("nan"))
            ),
            "signed_torque_probe_pulses": float(signed_probe_state.get("pulses", 0)),
            "signed_torque_probe_flips": float(signed_probe_state.get("flips", 0)),
            "signed_torque_probe_contact_alignment": probe_contact_alignment,
            "signed_torque_probe_agent_block_distance": probe_agent_block_distance,
            "contact_lost_recovery": 0.0,
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            **probe_contact_audit,
        }
    if (
        bool(getattr(args, "online_signed_torque_probe", False))
        and bool(signed_probe_state.get("failed", False))
        and probe_contact_ready
        and not bool(getattr(args, "geometric_torque_fallback", False))
    ):
        signed_probe_state["disabled"] = True
        signed_probe_state["failed"] = False
        recovery_action, recovery_audit = bounded_recovery_action(
            -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45))
            * probe_contact_normal,
            args,
        )
        return recovery_action, "recovery", {
            "signed_torque_probe_used": 0.0,
            "signed_torque_probe_failed": 1.0,
            "signed_torque_probe_feedback_used": float(signed_probe_feedback_used),
            "signed_torque_probe_feedback_gain": signed_probe_feedback_gain,
            "signed_torque_probe_xy_regression": float(
                signed_probe_state.get("last_xy_regression", float("nan"))
            ),
            "signed_torque_probe_cumulative_xy_drift": float(
                signed_probe_state.get("last_cumulative_xy_drift", float("nan"))
            ),
            "signed_torque_probe_pulses": float(signed_probe_state.get("pulses", 0)),
            "signed_torque_probe_flips": float(signed_probe_state.get("flips", 0)),
            "signed_torque_probe_contact_alignment": probe_contact_alignment,
            "signed_torque_probe_agent_block_distance": probe_agent_block_distance,
            "contact_lost_recovery": 1.0,
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            **recovery_audit,
        }
    if (
        bool(getattr(args, "online_signed_torque_probe", False))
        and probe_contact_ready
        and preterminal_angle_error > float(getattr(args, "terminal_angle", 0.28))
        and not bool(signed_probe_state.get("failed", False))
        and not bool(signed_probe_state.get("disabled", False))
        and not bool(getattr(args, "_signed_probe_blocked", False))
    ):
        max_pulses = max(1, int(getattr(args, "signed_torque_probe_max_pulses", 3)))
        exploratory_pulses = int(signed_probe_state.get("pulses", 0))
        validated = bool(signed_probe_state.get("validated", False))
        # Once a direction has passed an RGB feedback gate, it is no longer
        # exploratory.  It may continue at the shrunk magnitude while contact
        # remains safe; the pulse budget only limits unvalidated probing.
        can_probe = validated or exploratory_pulses < max_pulses
        if can_probe:
            orientation_sign = float(signed_probe_state.get("orientation_sign", 1.0))
            if "anchor_xy" not in signed_probe_state:
                signed_probe_state["anchor_xy"] = preterminal_xy
            probe_action, probe_info = rgb_signed_torque_probe_action(
                image, agent, block, preterminal_signed_angle_error, args, orientation_sign,
            )
            if not validated:
                signed_probe_state["pulses"] = exploratory_pulses + 1
            signed_probe_state["active"] = True
            signed_probe_state["error_before"] = preterminal_angle_error
            signed_probe_state["xy_before"] = preterminal_xy
            signed_probe_state["last_action"] = probe_action.tolist()
            probe_info.update({
                "signed_torque_probe_used": 1.0,
                "signed_torque_probe_feedback_used": float(signed_probe_feedback_used),
                "signed_torque_probe_feedback_gain": signed_probe_feedback_gain,
                "signed_torque_probe_xy_regression": float(
                    signed_probe_state.get("last_xy_regression", float("nan"))
                ),
                "signed_torque_probe_cumulative_xy_drift": float(
                    preterminal_xy
                    - float(signed_probe_state.get("anchor_xy", preterminal_xy))
                ),
                "signed_torque_probe_xy_safe": float(
                    float(signed_probe_state.get("last_xy_regression", 0.0))
                    <= float(getattr(args, "signed_torque_probe_max_xy_regression", 3.0))
                ),
                "signed_torque_probe_pulses": float(signed_probe_state.get("pulses", 0)),
                "signed_torque_probe_flips": float(signed_probe_state.get("flips", 0)),
                "signed_torque_probe_validated": float(validated),
                "signed_torque_probe_contact_alignment": probe_contact_alignment,
                "signed_torque_probe_agent_block_distance": probe_agent_block_distance,
                "angle_error_visual": preterminal_angle_error,
                "block_xy_visual": preterminal_xy,
                "terminal_control_active": 1.0,
                "predicted_best_cost": float("nan"),
                "online_calibration_samples": float(len(local_history) if local_history else 0),
                "online_calibrated": 0.0,
                "action_cost_ranker_used": 0.0,
            })
            return probe_action, "torque", probe_info
        signed_probe_state["disabled"] = True
        signed_probe_state["failed"] = False
        signed_probe_state["active"] = False
        # Fail closed: after the declared number of uninformative pulses, do
        # not hand control back to the same learned angular ranker that caused
        # the negative result.  The ordinary RGB recovery interface will
        # rebuild a contact lease on a later observation.
        recovery_action, recovery_audit = bounded_recovery_action(
            -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45))
            * probe_contact_normal,
            args,
        )
        return recovery_action, "recovery", {
            "signed_torque_probe_used": 0.0,
            "signed_torque_probe_failed": 1.0,
            "signed_torque_probe_feedback_used": float(signed_probe_feedback_used),
            "signed_torque_probe_feedback_gain": signed_probe_feedback_gain,
            "signed_torque_probe_pulses": float(signed_probe_state.get("pulses", 0)),
            "signed_torque_probe_flips": float(signed_probe_state.get("flips", 0)),
            "signed_torque_probe_contact_alignment": probe_contact_alignment,
            "signed_torque_probe_agent_block_distance": probe_agent_block_distance,
            "contact_lost_recovery": 1.0,
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            **recovery_audit,
        }
    # Once the RGB pose is already inside the declared terminal tube, freeze
    # before any side-reacquisition branch can reinterpret a one-frame contact
    # cosine drop as a lost lease.  This closes the failure mode in which
    # orientation had converged but recovery subsequently moved the block out
    # of the solved XY tube.
    if terminal_visual_hold_eligible(
        phase,
        preterminal_xy,
        preterminal_angle_error,
        float(getattr(args, "terminal_hold_xy", 18.0)),
        float(getattr(args, "terminal_angle", 0.28)),
    ):
        separation = block - agent
        distance = max(float(np.linalg.norm(separation)), 1.0e-6)
        goal_direction = goal_xy - block
        goal_distance = max(float(np.linalg.norm(goal_direction)), 1.0e-6)
        return np.zeros(2, dtype=np.float32), "hold", {
            "approach_distance": float("nan"),
            "observed_agent_block_distance": distance,
            "observed_block_motion": 0.0,
            "contact_side_alignment": float(np.dot(
                separation / distance, goal_direction / goal_distance,
            )),
            "contact_lost_recovery": 0.0,
            "terminal_visual_hold_guard": 1.0,
            "terminal_visual_xy": preterminal_xy,
            "terminal_visual_angle_error": preterminal_angle_error,
            "terminal_hold_xy": float(getattr(args, "terminal_hold_xy", 18.0)),
            "terminal_angle": float(getattr(args, "terminal_angle", 0.28)),
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
        }
    # Contact is a runtime-observable interface condition.  If a push or
    # torque action separates the RGB agent from the block, continuing to
    # optimize the pose endpoint applies actions on an invalid contact
    # manifold and was the dominant source of runaway traces.  Re-enter the
    # approach phase using only the current RGB geometry; no simulator state
    # or successor query is involved.
    observed_agent_block_distance = float(np.linalg.norm(agent - block))
    if previous_action is None:
        # At episode start the reference frame is the current frame.  Do not
        # infer a motion delta from a temporal latent head before any action
        # has been applied.
        previous_block = block.copy()
    else:
        if runtime_current_xy_policy(args) == "paired_rgb_calibrated":
            # ``block`` is the exact paired/calibrated runtime observer, but
            # the old previous-frame path silently used the latent/raw
            # observer.  Their fixed body-centroid offsets created a false
            # ~40 px motion on a physically stationary block and repeatedly
            # triggered contact recovery.  Motion is a temporal quantity, so
            # measure both frames in the same direct RGB chart here; the
            # absolute position remains the declared paired observer above.
            current_motion_block, _ = rgb_geometry_xy_angle_from_image(image, "block")
            previous_motion_block, _ = rgb_geometry_xy_angle_from_image(previous, "block")
            if current_motion_block is not None and previous_motion_block is not None:
                motion_delta = np.asarray(current_motion_block, dtype=np.float32) - np.asarray(
                    previous_motion_block, dtype=np.float32,
                )
                motion_magnitude = float(np.linalg.norm(motion_delta))
                if (
                    float(getattr(args, "rgb_max_observed_block_step", 48.0)) > 0.0
                    and motion_magnitude > float(getattr(args, "rgb_max_observed_block_step", 48.0))
                ):
                    motion_delta = np.zeros(2, dtype=np.float32)
                    motion_magnitude = 0.0
                previous_block = block - motion_delta
                observed_block_motion = motion_magnitude
                observed_block_delta = motion_delta
            else:
                previous_block = block.copy()
                observed_block_motion = 0.0
                observed_block_delta = np.zeros(2, dtype=np.float32)
        else:
            previous_block = observed_block_xy(
                previous, previous_features, xy_decoder, xy_decoder_training,
                args.device, runtime_current_xy_policy(args),
                getattr(args, "current_xy_disagreement_threshold", 60.0),
            )
            observed_block_motion = float(np.linalg.norm(block - previous_block))
            observed_block_delta = block - previous_block
    if previous_action is None:
        observed_block_motion = 0.0
        observed_block_delta = np.zeros(2, dtype=np.float32)
    direction = goal_xy - block
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    contact_normal = block - agent
    contact_normal /= max(float(np.linalg.norm(contact_normal)), 1e-6)
    contact_side_alignment = float(np.dot(contact_normal, direction))
    # The centroid chart is not a valid contact frame for an asymmetric T:
    # occlusion can move the observed agent/block centroids across the goal
    # direction even while the pusher remains on the same physical support.
    # Construct the support lease before any side-recovery branch so a valid
    # contact cannot be destroyed by a centroid-only ``reacquire_clear``.
    support_lease_gap_max = max(
        float(getattr(args, "terminal_contact_lease_gap_max", 0.0)),
        float(getattr(args, "terminal_contact_gap_max", 0.0)),
    )
    support_side_alignment_min = getattr(args, "support_contact_side_alignment_min", None)
    if support_side_alignment_min is None:
        support_side_alignment_min = getattr(args, "contact_side_alignment_min", 0.35)
    early_support_chart = contact_boundary_chart_from_images(
        resize_image(image, args.image_size)[None],
        np.zeros((1, 2), dtype=np.float32),
    )[0]
    early_support_lease_valid, early_support_gap_px, early_support_visible = (
        terminal_contact_lease_from_boundary(
            early_support_chart, support_lease_gap_max,
        )
    )
    early_support_lever = np.asarray(early_support_chart[2:4], dtype=np.float32)
    early_support_normal = -early_support_lever / max(
        float(np.linalg.norm(early_support_lever)), 1.0e-6,
    )
    early_support_side_alignment = float(np.dot(early_support_normal, direction))
    support_contact_goal_side_valid = bool(
        support_lease_gap_max > 0.0
        and early_support_lease_valid
        and early_support_visible
        and np.isfinite(early_support_side_alignment)
        and early_support_side_alignment >= float(support_side_alignment_min)
    )
    support_side_agnostic_terminal = bool(
        getattr(args, "support_contact_side_agnostic_terminal", False)
        and str(getattr(args, "planning_start", "random")) == "terminal"
    )
    support_contact_lease_valid = bool(
        support_contact_goal_side_valid
        or (
            support_side_agnostic_terminal
            and support_lease_gap_max > 0.0
            and early_support_lease_valid
            and early_support_visible
        )
    )
    args._support_lease_audit = {
        "support_contact_lease_valid": float(support_contact_lease_valid),
        "support_contact_goal_side_valid": float(support_contact_goal_side_valid),
        "support_contact_gap_px": float(early_support_gap_px),
        "support_contact_gap_max_px": float(support_lease_gap_max),
        "support_contact_visible": float(early_support_visible),
        "support_contact_side_alignment": float(early_support_side_alignment),
        "support_contact_side_alignment_min": float(support_side_alignment_min),
        "support_contact_side_agnostic_terminal_used": float(support_side_agnostic_terminal),
        "centroid_contact_side_alignment": float(contact_side_alignment),
        "support_contact_override_used": float(support_contact_lease_valid),
    }
    if support_contact_lease_valid:
        # Use the visible support lever as the declared contact frame for all
        # subsequent side, inward, and torque feasibility checks.  The goal
        # direction remains the task-space direction; only the contact frame
        # is replaced.  This is still RGB-only and does not query a successor.
        contact_normal = early_support_normal.astype(np.float32)
        if support_side_agnostic_terminal:
            # The RGB centroid-to-goal vector is not identifiable under the
            # contact occlusion present in these roots.  For this explicitly
            # declared terminal closure protocol, use the measured support
            # normal as the local pushing direction and keep the raw cosine in
            # the audit fields above rather than relabeling it as valid.
            direction = early_support_normal.astype(np.float32)
            contact_side_alignment = 1.0
    terminal_stagnant = bool(
        bool(getattr(args, "terminal_stagnation_watchdog", False))
        and phase in {"torque", "hold"}
        and previous_action is not None
        and observed_block_motion <= float(getattr(args, "terminal_stagnation_motion", 1.5))
        and int(terminal_stagnation_count) + 1 >= max(
            1, int(getattr(args, "terminal_stagnation_patience", 2))
        )
    )
    if terminal_stagnant:
        # A valid-looking RGB support point is not sufficient evidence that
        # the terminal action is coupled to the block.  First use one
        # in-contact counter-torque probe: preserve the inward component but
        # reflect the tangential component, which tests the opposite signed
        # moment without immediately spending the whole reacquisition budget.
        # If that probe also produces no response, the next call falls
        # through to the ordinary side-reacquisition path.  This is an
        # observation-only causal probe; it never reads a successor state or
        # simulator truth.
        if int(terminal_stagnation_count) == max(
            0, int(getattr(args, "terminal_stagnation_patience", 2)) - 1
        ):
            tangent = np.asarray([-contact_normal[1], contact_normal[0]], dtype=np.float32)
            inward = max(float(np.dot(np.asarray(previous_action, dtype=np.float32), contact_normal)), 0.08)
            tangential = float(np.dot(np.asarray(previous_action, dtype=np.float32), tangent))
            probe = inward * contact_normal - tangential * tangent
            probe = np.clip(probe, -float(args.contact_action_max_magnitude), float(args.contact_action_max_magnitude)).astype(np.float32)
            return probe, "torque", {
                "approach_distance": float(np.linalg.norm(
                    (block - float(args.contact_distance) * direction) - agent
                )),
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": 0.0,
                "terminal_stagnation_watchdog_used": 1.0,
                "terminal_stagnation_probe_used": 1.0,
                "terminal_stagnation_count": float(terminal_stagnation_count),
                "terminal_stagnation_probe_inward": inward,
                "terminal_stagnation_probe_tangent": -tangential,
                "reacquisition_attempt": 0.0,
                "reacquisition_success": 0.0,
            }
        if bool(getattr(args, "terminal_side_reacquisition", False)):
            return side_route_action(
                contact_normal, direction,
                reacquisition_retreat_magnitude(args),
            ), "recovery", {
                "approach_distance": float(np.linalg.norm(
                    (block - float(args.contact_distance) * direction) - agent
                )),
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": 1.0,
                "terminal_stagnation_watchdog_used": 1.0,
                "terminal_stagnation_count": float(terminal_stagnation_count),
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
            }
    # Contact is a hysteretic interface, not a one-frame distance test.  Once
    # the controller has handed off from approach, a moving block can leave
    # the narrow contact band by several pixels without actually losing the
    # controllable manifold.  The old rule reopened the 0.60/0.85 action fan
    # in exactly that situation; one extrapolative action then ejected the
    # pusher and made all subsequent local ranking evidence irrelevant.  Keep
    # the learned support capped until a larger RGB distance forces recovery.
    contact_lost = (
        phase not in {
            "approach", "recovery", "reacquire_clear", "reacquire",
            "reacquire_waypoint", "reacquire_arc", "reacquire_outer", "reacquire_target",
        }
        and int(recovery_cooldown) <= 0
        and not support_contact_lease_valid
        and observed_agent_block_distance > float(args.contact_recovery_distance)
        and observed_block_motion <= float(args.contact_recovery_stagnation)
    )
    if contact_lost:
        phase = "recovery"
    direction = goal_xy - block
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    contact_normal = block - agent
    contact_normal /= max(float(np.linalg.norm(contact_normal)), 1e-6)
    contact_side_alignment = float(np.dot(contact_normal, direction))
    # Reapply the support-aware frame after the legacy geometry refresh above;
    # otherwise later candidate and recovery branches silently revert to the
    # occluded centroid chart.
    if support_contact_lease_valid:
        contact_normal = early_support_normal.astype(np.float32)
        if support_side_agnostic_terminal:
            direction = early_support_normal.astype(np.float32)
            contact_side_alignment = 1.0
    side_alignment_threshold = float(getattr(args, "contact_side_alignment_min", 0.35))
    side_misalignment_patience = max(
        1, int(getattr(args, "contact_side_misalignment_patience", 1))
    )
    current_goal_distance = float(np.linalg.norm(goal_xy - block))
    previous_goal_distance = float(np.linalg.norm(goal_xy - previous_block))
    goal_progress = previous_goal_distance - current_goal_distance
    motion_direction_alignment = float(
        np.dot(observed_block_delta, direction)
        / max(float(np.linalg.norm(observed_block_delta)), 1.0e-6)
    ) if observed_block_motion > 1.0e-6 else 0.0
    motion_limit = float(getattr(args, "reacquisition_motion_limit", 3.0))
    motion_excess_count = (
        int(reacquisition_motion_excess_count) + 1
        if observed_block_motion > motion_limit else 0
    )
    motion_excess_patience = max(
        1, int(getattr(args, "reacquisition_motion_patience", 2))
    )
    if phase == "reacquire_target":
        # The final target-contact segment is the only route phase whose
        # action intentionally points toward the block.  One material RGB
        # block displacement is therefore sufficient evidence of a collision
        # or stale target; waiting for the ordinary two-frame patience caused
        # 16 px-scale pushes before the abort could fire.
        motion_excess_patience = max(
            1,
            int(getattr(args, "reacquisition_target_motion_patience", 1)),
        )
    support_near_terminal = bool(
        support_contact_lease_valid
        and preterminal_xy <= float(getattr(args, "terminal_xy", 35.0))
        + float(args.contact_tolerance)
    )
    near_terminal_tube = bool(
        current_goal_distance <= float(getattr(args, "terminal_xy", 35.0))
        or support_near_terminal
    )
    terminal_proximity_lease_candidate = bool(
        (
            getattr(args, "terminal_proximity_lease_fallback", False)
            and near_terminal_tube
            and observed_agent_block_distance <= float(args.contact_distance) + float(args.contact_tolerance)
            and contact_side_alignment >= side_alignment_threshold
        )
        or (support_contact_lease_valid and near_terminal_tube)
    )
    progressing_contact = bool(
        observed_block_motion > float(args.contact_recovery_stagnation)
        and goal_progress > 0.0
        and not near_terminal_tube
        # A wrong-side contact can move the block, so motion alone is not
        # sufficient. Conversely, RGB occlusion can make the instantaneous
        # centroid-to-goal contact alignment negative even when the observed
        # block displacement is toward the goal. Accept small positive
        # temporal progress as evidence of a valid contact mode.
        and (
            contact_side_alignment >= side_alignment_threshold
            or motion_direction_alignment >= -0.15
        )
    )
    side_misaligned = (
        not support_contact_lease_valid
        and
        contact_side_alignment < side_alignment_threshold
        and int(contact_side_misalignment_count) + 1 >= side_misalignment_patience
        # Under the explicit side-reacquisition protocol, terminal torque
        # still requires the declared positive-side lease.  Translation is a
        # slightly different interface: a small positive-side cosine can be
        # transient RGB occlusion while the object is demonstrably moving
        # toward the goal.  Preserve that useful contact, but never permit a
        # negative cosine (the pusher is genuinely on the wrong side) to pass
        # through the block.
        and (
            (
                bool(getattr(args, "terminal_side_reacquisition", False))
                and (
                    phase in {"terminal_approach", "torque", "hold"}
                    or contact_side_alignment < 0.0
                    or not progressing_contact
                )
            )
            or (
                not bool(getattr(args, "terminal_side_reacquisition", False))
                and not progressing_contact
            )
        )
        # In the legacy protocol, a side flip inside the terminal XY tube is
        # tolerated while orientation is being repaired.  The explicit
        # side-reacquisition protocol is different: an erroneous contact
        # side invalidates the terminal torque interface before the first
        # torque action is applied.  Otherwise the controller spends several
        # terminal steps pushing from the wrong side and only then enters
        # recovery, which was the direct cause of the large post-terminal
        # reversals in the audit traces.
        and (
            not near_terminal_tube
            or bool(getattr(args, "terminal_side_reacquisition", False))
        )
    )
    desired_agent = block - args.contact_distance * direction
    approach = desired_agent - agent
    approach_distance = float(np.linalg.norm(approach))
    terminal_contact_target_mode = str(
        getattr(args, "terminal_contact_target", "boundary")
    )
    if (
        phase == "terminal_approach"
        and float(getattr(args, "terminal_contact_gap_max", 0.0)) > 0.0
        and not terminal_proximity_lease_candidate
    ):
        # This support-point branch precedes the generic terminal-candidate
        # branch below.  It must still reject a wrong-side terminal contact:
        # a small boundary gap does not imply a valid torque lease for the
        # asymmetric T.  Return to recovery so the next frame can construct a
        # fresh RGB route anchor instead of handing the learned ranker an
        # invalid contact mode.
        if (
            bool(getattr(args, "terminal_side_reacquisition", False))
            and contact_side_alignment < side_alignment_threshold
        ):
            # Keep a mildly misaligned pusher on the local terminal manifold.
            # The previous ordering sent every terminal_approach frame to the
            # outer reacquisition route, even when the agent was only a few
            # pixels from the centroid-facing contact.  That route is a
            # recovery primitive, not a terminal servo, and its large arc can
            # create the observed XY drift/limit cycle.
            terminal_contact_band = float(args.contact_distance) + 2.0 * float(
                args.contact_tolerance
            )
            terminal_local_repair = bool(
                terminal_contact_target_mode == "centroid"
                and observed_agent_block_distance <= terminal_contact_band
                and contact_side_alignment >= -0.10
                and near_terminal_tube
            )
            if terminal_local_repair:
                (
                    local_action,
                    local_raw_norm,
                    local_final_norm,
                    local_target_distance,
                ) = terminal_centroid_local_repair(
                    agent,
                    desired_agent,
                    float(args.approach_action_scale),
                    float(getattr(args, "terminal_approach_action_max_magnitude", -1.0)),
                )
                return local_action, "terminal_approach", {
                    "approach_distance": approach_distance,
                    "observed_agent_block_distance": observed_agent_block_distance,
                    "observed_block_motion": observed_block_motion,
                    "contact_side_alignment": contact_side_alignment,
                    "contact_lost_recovery": 0.0,
                    "terminal_side_gate_local_repair": 1.0,
                    "terminal_side_gate_before_support_gap": 1.0,
                    "terminal_contact_target_mode": terminal_contact_target_mode,
                    "terminal_contact_target_distance": local_target_distance,
                    "terminal_approach_raw_norm": local_raw_norm,
                    "terminal_approach_final_norm": local_final_norm,
                    "reacquisition_attempt": 0.0,
                    "reacquisition_success": 0.0,
                }
            return side_route_action(
                contact_normal, direction,
                reacquisition_retreat_magnitude(args),
            ), "recovery", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": 1.0,
                "terminal_side_gate_before_support_gap": 1.0,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
            }
        # The support-point target is retained for the historical boundary
        # protocol.  It is not a universally valid acquisition target: on an
        # asymmetric T the farthest point along the centroid normal can be a
        # thin corner while the pusher is already correctly placed on the
        # goal-facing side.  The centroid intervention below deliberately
        # approaches the declared contact point instead and records the
        # boundary chart only as an audit quantity.
        terminal_rgb = resize_image(image, args.image_size)
        terminal_chart = contact_boundary_chart_from_images(
            terminal_rgb[None], np.zeros((1, 2), dtype=np.float32),
        )[0]
        terminal_support_visible = bool(float(np.linalg.norm(terminal_chart[:4])) > 1.0e-6)
        terminal_gap = float(terminal_chart[8]) * (PUSHT_STATE_SIZE / 2.0)
        if terminal_support_visible and np.isfinite(terminal_gap):
            if terminal_contact_target_mode == "centroid":
                terminal_target = np.asarray(desired_agent, dtype=np.float32)
                terminal_approach = terminal_target - np.asarray(agent, dtype=np.float32)
                terminal_target_distance = float(np.linalg.norm(terminal_approach))
                contact_reached = terminal_target_distance <= float(args.contact_tolerance)
            else:
                terminal_target = (
                    (np.asarray(terminal_chart[:2], dtype=np.float32) + 1.0)
                    * (PUSHT_STATE_SIZE / 2.0)
                )
                rgb_agent, _ = rgb_geometry_xy_angle_from_image(image, "agent")
                approach_agent = agent if rgb_agent is None else np.asarray(rgb_agent, dtype=np.float32)
                terminal_approach = terminal_target - approach_agent
                terminal_target_distance = float(np.linalg.norm(terminal_approach))
                contact_reached = terminal_gap <= float(args.terminal_contact_gap_max)
            if contact_reached:
                return np.zeros(2, dtype=np.float32), "torque", {
                    "approach_distance": approach_distance,
                    "observed_agent_block_distance": observed_agent_block_distance,
                    "observed_block_motion": observed_block_motion,
                    "contact_side_alignment": contact_side_alignment,
                    "contact_lost_recovery": 0.0,
                    "terminal_contact_approach_complete": 1.0,
                    "terminal_contact_boundary_gap_px": terminal_gap,
                    "terminal_contact_gap_max": float(args.terminal_contact_gap_max),
                    "terminal_contact_target_mode": terminal_contact_target_mode,
                    "terminal_contact_target_distance": terminal_target_distance,
                    "terminal_boundary_support_visible": 1.0,
                    "reacquisition_attempt": 0.0,
                    "reacquisition_success": 0.0,
                }
            terminal_approach_action, terminal_approach_raw_norm, terminal_approach_final_norm = bounded_l2_action(
                terminal_approach / float(args.approach_action_scale),
                float(getattr(args, "terminal_approach_action_max_magnitude", -1.0)),
            )
            return terminal_approach_action, "terminal_approach", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": 0.0,
                "terminal_contact_approach_complete": 0.0,
                "terminal_contact_boundary_gap_px": terminal_gap,
                "terminal_contact_gap_max": float(args.terminal_contact_gap_max),
                "terminal_contact_target_mode": terminal_contact_target_mode,
                "terminal_contact_target_distance": terminal_target_distance,
                "terminal_boundary_support_visible": 1.0,
                "terminal_approach_action_cap": float(
                    getattr(args, "terminal_approach_action_max_magnitude", -1.0)
                ),
                "terminal_approach_raw_norm": terminal_approach_raw_norm,
                "terminal_approach_final_norm": terminal_approach_final_norm,
                "reacquisition_attempt": 0.0,
                "reacquisition_success": 0.0,
            }
    # The desired contact point is only a noisy RGB reconstruction.  Once the
    # observed agent--block distance is inside the broad contact band, the
    # block may already be moving even when the reconstructed point error is
    # larger than the narrow servo tolerance.  Staying in approach in that
    # situation makes the state machine drive past the object and never gives
    # the learned selector control.  Use the contact band as the phase handoff
    # and reserve the narrow target error for genuinely remote approach.
    approach_contact_band = float(args.contact_distance) + 2.0 * float(args.contact_tolerance)
    recovery_handoff_distance = float(args.contact_distance) + 1.25 * float(args.contact_tolerance)
    reacquisition_clearance = max(
        recovery_handoff_distance,
        float(getattr(args, "reacquisition_clearance", args.contact_recovery_distance)),
    )
    side_reacquisition_enabled = bool(
        getattr(args, "terminal_side_reacquisition", False)
    )
    persistent_recovery_anchor = bool(
        getattr(args, "persistent_recovery_anchor", True)
    )
    active_recovery_anchor = recovery_anchor if persistent_recovery_anchor else None
    if (
        persistent_recovery_anchor
        and active_recovery_anchor is None
        and side_reacquisition_enabled
        and side_misaligned
        and phase not in {"approach", "reacquire_clear", "reacquire", "reacquire_waypoint", "reacquire_arc", "reacquire_outer", "reacquire_target"}
    ):
        active_recovery_anchor = make_recovery_anchor(
            block, direction, agent, reacquisition_clearance,
        )
    if active_recovery_anchor is not None:
        anchored_waypoint, anchored_arc_midpoint, anchored_outer_target, anchored_contact_target = recovery_anchor_geometry(
            active_recovery_anchor, float(args.contact_distance),
        )
    else:
        anchored_waypoint = anchored_arc_midpoint = anchored_outer_target = anchored_contact_target = None
    reacquisition_anchor_drift = 0.0
    if active_recovery_anchor is not None:
        reacquisition_anchor_drift = float(np.linalg.norm(
            np.asarray(robust_runtime_block, dtype=np.float32)
            - np.asarray(active_recovery_anchor["block"], dtype=np.float32)
        ))
    anchor_drift_limit = float(
        getattr(args, "reacquisition_anchor_drift_limit", 12.0)
    )
    # A block can move while the pusher is completing the final target-side
    # approach.  In that narrow case, the current contact geometry is more
    # informative than displacement from the stale route anchor: aborting on
    # the historical drift first creates a loop one frame before handoff.
    # This exemption is deliberately restricted to the declared handoff tube
    # and does not weaken drift protection during clear/arc/outer routing.
    reacquisition_handoff_ready = bool(
        phase == "reacquire_target"
        and observed_agent_block_distance <= recovery_handoff_distance
        and approach_distance <= float(args.contact_tolerance)
        and contact_side_alignment >= side_alignment_threshold
    )
    anchor_drift_exceeded = bool(
        active_recovery_anchor is not None
        and anchor_drift_limit > 0.0
        and reacquisition_anchor_drift > anchor_drift_limit
        and phase in {"reacquire_arc", "reacquire_outer", "reacquire_target"}
        and not reacquisition_handoff_ready
    )
    active_reacquisition_clearance = float(
        active_recovery_anchor.get("clearance", reacquisition_clearance)
        if active_recovery_anchor is not None else reacquisition_clearance
    )
    reacquisition_phases = {
        "reacquire_clear", "reacquire", "reacquire_waypoint",
        "reacquire_arc", "reacquire_outer", "reacquire_target",
    }
    route_retry_limit = max(
        1, int(getattr(args, "reacquisition_max_retries", 1))
    )
    # ``reacquisition_route_abort_count`` counts consecutive failed routes,
    # but it must not become a permanent lockout.  A route abort installs a
    # bounded RGB-only cooldown in ``reacquisition_lockout_steps``.  Once that
    # cooldown expires, the next observation is a fresh-anchor opportunity;
    # keeping ``route_latched`` true after the cooldown made every later
    # decision return ``hold`` forever, even though the pusher could have
    # re-established contact.  This was visible as long tails of
    # ``reacquisition_route_lockout=1`` with zero actions in the paired
    # replay.  The abort count remains in the audit and is reset only by a
    # confirmed reacquisition, while an active cooldown still fail-closes the
    # interface.
    route_latched = bool(
        side_reacquisition_enabled
        and int(reacquisition_route_abort_count) >= route_retry_limit
        and int(reacquisition_lockout_steps) > 0
    )
    if (
        side_reacquisition_enabled
        and (route_latched or int(reacquisition_lockout_steps) > 0)
        and phase not in {"approach", "push", "translate", "torque"}
    ):
        # A failed route is a terminal diagnosis, not a license to re-enter
        # the same recovery branch.  If the pusher has already returned to a
        # valid side and remains close enough, spend at most one explicitly
        # capped local approach action; otherwise freeze the interface.
        failure_repair_cap = max(
            0.0,
            float(getattr(args, "reacquisition_failure_translation_cap", 0.10)),
        )
        if (
            not bool(reacquisition_failure_repair_used)
            and failure_repair_cap > 0.0
            and contact_side_alignment >= side_alignment_threshold
            and observed_agent_block_distance <= recovery_handoff_distance
        ):
            (
                repair_action,
                repair_raw_norm,
                repair_final_norm,
                repair_target_distance,
            ) = terminal_centroid_local_repair(
                agent,
                desired_agent,
                float(args.approach_action_scale),
                failure_repair_cap,
            )
            return repair_action, "terminal_approach", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_route_lockout": 1.0,
                "reacquisition_bounded_translation_repair": 1.0,
                "reacquisition_failure_translation_cap": failure_repair_cap,
                "reacquisition_failure_repair_used": 1.0,
                "terminal_approach_raw_norm": repair_raw_norm,
                "terminal_approach_final_norm": repair_final_norm,
                "terminal_contact_target_distance": repair_target_distance,
                "reacquisition_attempt": 0.0,
                "reacquisition_success": 0.0,
                "recovery_anchor_release": 1.0,
            }
        return np.zeros(2, dtype=np.float32), "hold", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_side_alignment": contact_side_alignment,
            "reacquisition_route_lockout": 1.0,
            "reacquisition_route_lockout_steps": float(reacquisition_lockout_steps),
            "reacquisition_route_abort_count": float(reacquisition_route_abort_count),
            "reacquisition_failure_repair_used": float(reacquisition_failure_repair_used),
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            "recovery_anchor_release": 1.0,
        }
    if (
        phase in reacquisition_phases
        and (
            int(reacquisition_phase_steps) >= int(getattr(args, "reacquisition_phase_budget", 36))
            or int(reacquisition_total_steps) >= int(getattr(args, "reacquisition_total_budget", 72))
        )
    ):
        # Fail closed to the ordinary RGB recovery servo.  A bounded abort is
        # preferable to spending the remaining horizon on an unreachable
        # anchor; the next call can establish a fresh anchor from current RGB.
        return np.zeros(2, dtype=np.float32), "hold", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": 1.0,
            "contact_side_alignment": contact_side_alignment,
            "reacquisition_attempt": 1.0,
            "reacquisition_success": 0.0,
            "reacquisition_budget_exhausted": 1.0,
            "reacquisition_route_abort": 1.0,
            "reacquisition_lockout_steps": float(
                getattr(args, "reacquisition_lockout_steps", 12)
            ),
            "recovery_anchor_release": 1.0,
                    **recovery_anchor_trace(active_recovery_anchor),
        }
    # A workspace-infeasible anchor is still a finite RGB route after its
    # waypoint/arc/outer points are clipped by ``recovery_anchor_geometry``.
    # Do not convert this geometric warning into an unconditional hold: that
    # made every near-wall failure unrecoverable before an action was tried.
    # The clipped status is carried in the per-step trace, and the existing
    # motion/anchor-drift guards remain responsible for aborting unsafe routes.

    # Side recovery is a two-stage maneuver.  A single retreat followed by
    # the ordinary recovery branch was not enough: once the pusher returned
    # inside the handoff radius while still on the wrong side, that branch
    # retreated again and produced a limit cycle.  ``reacquire_clear`` clears
    # the contact manifold once; ``reacquire_waypoint`` routes around the
    # object at a safe radius; ``reacquire_arc`` moves to the target-side
    # outer point; ``reacquire_target`` then approaches the
    # declared contact point and does not use the stale current-side normal.
    if anchor_drift_exceeded:
        return np.zeros(2, dtype=np.float32), "hold", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": 1.0,
            "contact_side_alignment": contact_side_alignment,
            "reacquisition_attempt": 1.0,
            "reacquisition_success": 0.0,
            "reacquisition_anchor_drift": reacquisition_anchor_drift,
            "reacquisition_anchor_drift_limit": anchor_drift_limit,
            "reacquisition_anchor_drift_abort": 1.0,
            "reacquisition_route_abort": 1.0,
            "reacquisition_lockout_steps": float(
                getattr(args, "reacquisition_lockout_steps", 12)
            ),
            "recovery_anchor_release": 1.0,
            **recovery_anchor_trace(active_recovery_anchor),
        }
    if phase == "reacquire_clear":
        if observed_agent_block_distance < active_reacquisition_clearance:
            # Pure normal retreat can remain trapped against the asymmetric T:
            # the pusher moves outward, but its footprint continues to graze
            # the same protruding branch and the block follows it.  Use the
            # bounded side-route primitive so the clear phase has a tangential
            # escape component under the same RGB-only action cap.
            action = side_route_action(
                contact_normal,
                direction,
                reacquisition_retreat_magnitude(args),
            )
            return action, "reacquire_clear", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 0.0,
                "contact_side_alignment": contact_side_alignment,
                "motion_direction_alignment": motion_direction_alignment,
                "progressing_contact": float(progressing_contact),
                "reacquisition_clear_step": 1.0,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                "reacquisition_clear_side_route": 1.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        phase = "reacquire_waypoint"
    if phase in {"reacquire", "reacquire_waypoint"}:
        if anchored_waypoint is not None:
            waypoint = anchored_waypoint
        else:
            target_radial = -direction
            perpendicular = np.asarray([-target_radial[1], target_radial[0]], dtype=np.float32)
            waypoint_radius = reacquisition_clearance
            waypoint_candidates = np.asarray([
                block + waypoint_radius * perpendicular,
                block - waypoint_radius * perpendicular,
            ], dtype=np.float32)
            waypoint = waypoint_candidates[int(np.argmin(np.linalg.norm(
                waypoint_candidates - agent[None], axis=1,
            )))]
        waypoint_distance = float(np.linalg.norm(waypoint - agent))
        if waypoint_distance <= float(args.contact_tolerance):
            phase = "reacquire_arc"
        else:
            action = np.clip(
                (waypoint - agent) / float(args.approach_action_scale), -1.0, 1.0,
            ).astype(np.float32)
            action = np.clip(
                action,
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            )
            return action, "reacquire_waypoint", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 0.0,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_clear_step": 0.0,
                "reacquisition_waypoint_distance": waypoint_distance,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
    if phase == "reacquire_arc":
        # Reacquisition is supposed to move only the pusher.  Previously this
        # abort was conditional on the pusher still looking wrong-side; once
        # the RGB side cosine became positive, a collision during the arc was
        # misclassified as safe and the controller kept pushing the block.
        # Treat any material block motion as a broken no-contact lease,
        # release the stale anchor, and rebuild from the current RGB chart.
        if motion_excess_count >= motion_excess_patience:
            return np.zeros(2, dtype=np.float32), "hold", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 1.0,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                "reacquisition_arc_motion_abort": 1.0,
                "reacquisition_route_abort": 1.0,
                "reacquisition_lockout_steps": float(
                    getattr(args, "reacquisition_lockout_steps", 12)
                ),
                "reacquisition_motion_excess_count": float(motion_excess_count),
                "reacquisition_motion_patience": float(motion_excess_patience),
                "recovery_anchor_release": 1.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        outer_target = (
            anchored_outer_target
            if anchored_outer_target is not None
            else block + reacquisition_clearance * (-direction)
        )
        if anchored_arc_midpoint is not None:
            arc_target = anchored_arc_midpoint
        else:
            route_normal = np.asarray(
                [-direction[1], direction[0]], dtype=np.float32,
            )
            route_sign = 1.0 if float(np.dot(agent - block, route_normal)) >= 0.0 else -1.0
            arc_target = block - (reacquisition_clearance / np.sqrt(2.0)) * direction + (
                route_sign * reacquisition_clearance / np.sqrt(2.0)
            ) * route_normal
        midpoint_distance = float(np.linalg.norm(arc_target - agent))
        if midpoint_distance <= float(args.contact_tolerance):
            phase = "reacquire_outer"
        else:
            outer_distance = midpoint_distance
            action = np.clip(
                (arc_target - agent) / float(args.approach_action_scale), -1.0, 1.0,
            ).astype(np.float32)
            action = np.clip(
                action,
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            )
            return action, "reacquire_arc", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 0.0,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_clear_step": 0.0,
                "reacquisition_waypoint_distance": outer_distance,
                "reacquisition_arc_midpoint_active": 1.0,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
    if phase == "reacquire_outer":
        outer_target = (
            anchored_outer_target
            if anchored_outer_target is not None
            else block + reacquisition_clearance * (-direction)
        )
        outer_distance = float(np.linalg.norm(outer_target - agent))
        if outer_distance > float(args.contact_tolerance):
            action = np.clip(
                (outer_target - agent) / float(args.approach_action_scale), -1.0, 1.0,
            ).astype(np.float32)
            action = np.clip(
                action,
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            )
            return action, "reacquire_outer", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 0.0,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_clear_step": 0.0,
                "reacquisition_waypoint_distance": outer_distance,
                "reacquisition_arc_midpoint_active": 0.0,
                "reacquisition_outer_phase": 1.0,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        phase = "reacquire_target"
    if phase == "reacquire_target":
        anchored_direction = (
            np.asarray(active_recovery_anchor["direction"], dtype=np.float32)
            if active_recovery_anchor is not None else direction
        )
        anchored_normal = block - agent
        anchored_normal /= max(float(np.linalg.norm(anchored_normal)), 1.0e-6)
        anchored_alignment = float(np.dot(anchored_normal, anchored_direction))
        if anchored_alignment < side_alignment_threshold:
            # The target-contact segment crosses the block when the pusher is
            # still on the wrong side.  Following it is not reacquisition: it
            # is an unguarded push that explains the repeated XY explosions
            # after otherwise successful clear/arc phases.  Release the old
            # anchor because the block may have moved during the failed arc;
            # the next frame rebuilds the route from the current RGB chart.
            return np.zeros(2, dtype=np.float32), "hold", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 1.0,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_clear_step": 1.0,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                "reacquisition_target_side_invalid": 1.0,
                "reacquisition_route_abort": 1.0,
                "reacquisition_lockout_steps": float(
                    getattr(args, "reacquisition_lockout_steps", 12)
                ),
                "recovery_anchor_release": 1.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        # Keep the route frame fixed, but express the final contact point
        # relative to the current RGB block.  The block can legitimately
        # translate while the pusher clears the wrong side; freezing this
        # last point to the onset centroid creates an unreachable target and
        # a near-zero-action reacquire loop.
        target_contact = block - float(args.contact_distance) * anchored_direction
        target_approach = target_contact - agent
        motion_confirmed_contact = bool(
            observed_block_motion > float(args.contact_recovery_stagnation)
            and anchored_alignment >= side_alignment_threshold
            # Motion alone can be caused by a grazing collision while the
            # pusher is still too far from a controllable support point.  The
            # broad recovery handoff radius is intentionally permissive for
            # approach routing, but it is not a sufficient terminal-contact
            # certificate.  Require the tighter contact-plus-half-tolerance
            # radius for the motion shortcut; direct target-contact handoff
            # below retains the declared broad tolerance path.
            and observed_agent_block_distance
            <= float(args.contact_distance) + 0.5 * float(args.contact_tolerance)
            and float(np.linalg.norm(target_approach)) <= approach_contact_band
        )
        reacquired = bool(
            (
                observed_agent_block_distance <= recovery_handoff_distance
                and float(np.linalg.norm(target_approach)) <= float(args.contact_tolerance)
                and anchored_alignment >= side_alignment_threshold
            )
            or (
                # The centroid-derived contact point is not exact for the
                # asymmetric T.  Once the pusher is on the declared goal
                # side and inside the broad handoff tube, continuing the
                # final approach can create the very block motion that the
                # route guard later mislabels as a failed reacquisition.
                observed_agent_block_distance <= recovery_handoff_distance
                and float(np.linalg.norm(target_approach)) <= approach_contact_band
                and anchored_alignment >= side_alignment_threshold
            )
            or motion_confirmed_contact
        )
        if reacquired:
            return np.zeros(2, dtype=np.float32), "push", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 0.0,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_clear_step": 0.0,
            "reacquisition_waypoint_distance": 0.0,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 1.0,
                "reacquisition_motion_handoff": float(motion_confirmed_contact),
                "recovery_anchor_release": 1.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        # A route may accumulate motion evidence while the pusher is still
        # traversing the outer arc.  Once the current frame satisfies the
        # explicit handoff geometry above, that evidence is no longer a
        # reason to reject the contact lease: doing so creates a deterministic
        # target->recovery->target loop exactly at the final approach.  Abort
        # only when the route is still outside the declared handoff tube.
        if motion_excess_count >= motion_excess_patience:
            action = np.clip(
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)) * contact_normal,
                -1.0, 1.0,
            ).astype(np.float32)
            return np.zeros(2, dtype=np.float32), "hold", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 0.0,
                "contact_side_alignment": contact_side_alignment,
                "reacquisition_clear_step": 1.0,
                "reacquisition_waypoint_distance": 0.0,
                "reacquisition_motion_abort": 1.0,
                "reacquisition_route_abort": 1.0,
                "reacquisition_lockout_steps": float(
                    getattr(args, "reacquisition_lockout_steps", 12)
                ),
                "reacquisition_motion_excess_count": float(motion_excess_count),
                "reacquisition_motion_patience": float(motion_excess_patience),
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                "recovery_anchor_release": 1.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        action = np.clip(
            target_approach / float(args.approach_action_scale), -1.0, 1.0,
        ).astype(np.float32)
        target_action_cap = max(
            0.0,
            float(getattr(
                args, "reacquisition_target_action_max_magnitude", 0.10,
            )),
        )
        action = np.clip(
            action,
            -target_action_cap,
            target_action_cap,
        )
        return action, "reacquire_target", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": 0.0,
            "contact_side_alignment": contact_side_alignment,
            "reacquisition_clear_step": 0.0,
            "reacquisition_waypoint_distance": 0.0,
            "reacquisition_target_action_cap": target_action_cap,
            "reacquisition_attempt": 1.0,
            "reacquisition_success": 0.0,
            **recovery_anchor_trace(active_recovery_anchor),
        }
    # A torque/hold phase is a contact lease, not merely a memory of the
    # previous XY tube entry.  Reacquisition can report motion-based success
    # while the pusher is still outside the broad handoff radius; retaining
    # the old phase in that case exposed the learned 0.60/0.85 fan without a
    # valid terminal contact.  Re-enter the ordinary RGB recovery interface
    # before any candidate score is evaluated.
    terminal_lease_lost_recovery = bool(
        phase in {"torque", "hold"}
        and not support_contact_lease_valid
        and observed_agent_block_distance > recovery_handoff_distance
    )
    if terminal_lease_lost_recovery:
        phase = "recovery"
    recovery_contact_confirmed = (
        observed_block_motion > float(args.contact_recovery_stagnation)
        and approach_distance <= approach_contact_band
        and contact_side_alignment >= float(getattr(args, "contact_side_alignment_min", 0.35))
    )
    # This guard must precede the generic recovery->reacquisition handoff.
    # A terminal pusher can remain physically close to the block while its
    # one-frame RGB side cosine falls below the strict torque threshold.  It
    # is still locally repairable when the cosine is only mildly negative or
    # positive; sending it through the outer route here was the control-flow
    # bug that made the later ``terminal_wrong_side`` repair unreachable.
    recovery_terminal_local_repair = bool(
        phase == "recovery"
        and bool(getattr(args, "terminal_side_reacquisition", False))
        and terminal_contact_target_mode == "centroid"
        and near_terminal_tube
        and observed_agent_block_distance <= float(args.contact_distance) + 2.0 * float(
            args.contact_tolerance
        )
        and contact_side_alignment >= -0.10
    )
    if recovery_terminal_local_repair:
        (
            local_action,
            local_raw_norm,
            local_final_norm,
            local_target_distance,
        ) = terminal_centroid_local_repair(
            agent,
            desired_agent,
            float(args.approach_action_scale),
            float(getattr(args, "terminal_approach_action_max_magnitude", -1.0)),
        )
        return local_action, "terminal_approach", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_side_alignment": contact_side_alignment,
            "contact_lost_recovery": 0.0,
            "terminal_side_gate_local_repair": 1.0,
            "terminal_side_gate_before_recovery": 1.0,
            "terminal_contact_target_mode": terminal_contact_target_mode,
            "terminal_contact_target_distance": local_target_distance,
            "terminal_approach_raw_norm": local_raw_norm,
            "terminal_approach_final_norm": local_final_norm,
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            **recovery_anchor_trace(active_recovery_anchor),
        }
    if phase == "recovery" and (
        observed_agent_block_distance > recovery_handoff_distance
        or (
            approach_distance > float(args.contact_tolerance)
            and not recovery_contact_confirmed
        )
        or side_misaligned
    ):
        if (
            side_reacquisition_enabled
            and side_misaligned
        ):
            # A wrong-side pusher must never be sent through the block by the
            # ordinary contact-target approach, even when it is already far
            # outside the handoff radius.  Near contact, clear first; when
            # already clear, route directly to the frozen lateral waypoint.
            if observed_agent_block_distance <= recovery_handoff_distance:
                action = side_route_action(
                    contact_normal, direction,
                    reacquisition_retreat_magnitude(args),
                )
                recovery_phase = "reacquire_clear"
                clear_step = 1.0
            else:
                waypoint = anchored_waypoint
                if waypoint is None:
                    waypoint = block + float(reacquisition_clearance) * np.asarray(
                        [direction[1], -direction[0]], dtype=np.float32,
                    )
                action = np.clip(
                    (waypoint - agent) / float(args.approach_action_scale),
                    -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                    float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                ).astype(np.float32)
                recovery_phase = "reacquire_waypoint"
                clear_step = 0.0
            return action, recovery_phase, {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": 1.0,
                "recovery_handoff_distance": recovery_handoff_distance,
                "contact_side_alignment": contact_side_alignment,
                "contact_side_recovery_retreat": 1.0,
                "reacquisition_clear_step": clear_step,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                "reacquisition_wrong_side_route": 1.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        elif side_misaligned and observed_agent_block_distance <= recovery_handoff_distance:
            if getattr(args, "side_recovery_mode", "contact_target") == "contact_target":
                action, recovery_audit = bounded_recovery_action(
                    approach / float(args.approach_action_scale), args,
                )
                if float(getattr(args, "recovery_action_max_magnitude", 0.0)) <= 0.0:
                    action = np.clip(
                        action,
                        -reacquisition_retreat_magnitude(args),
                        reacquisition_retreat_magnitude(args),
                    ).astype(np.float32)
                side_retreat = 0.0
            else:
                action, recovery_audit = bounded_recovery_action(
                    -reacquisition_retreat_magnitude(args) * contact_normal,
                    args,
                )
                side_retreat = 1.0
        else:
            action, recovery_audit = bounded_recovery_action(
                approach / float(args.approach_action_scale), args,
            )
            side_retreat = 0.0
        return action, "recovery", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": 1.0,
            "recovery_handoff_distance": recovery_handoff_distance,
            "contact_side_alignment": contact_side_alignment,
            "terminal_lease_lost_recovery": float(terminal_lease_lost_recovery),
            "contact_side_recovery_retreat": side_retreat,
            "reacquisition_clear_step": 0.0,
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            "contact_side_misalignment_count": float(contact_side_misalignment_count),
            "contact_side_gate_suppressed_by_progress": float(progressing_contact),
            **recovery_audit,
        }
    # A broad agent--block distance is not sufficient to establish the
    # correct pushing side.  In particular, an episode can begin with the
    # pusher close to the block but on the wrong side of the desired contact
    # point; handing off there makes the learned ranker optimize an invalid
    # contact mode and was the dominant source of traces that never reduced
    # the block-to-goal distance.  Handoff is allowed only after the desired
    # contact point is reached, or after RGB evidence shows that the block is
    # already moving (which preserves the receding-contact behavior once a
    # valid push has begun).
    approach_contact_confirmed = (
        support_contact_lease_valid
        or (
            observed_block_motion > float(args.contact_recovery_stagnation)
            and approach_distance <= approach_contact_band
            and contact_side_alignment >= float(getattr(args, "contact_side_alignment_min", 0.35))
        )
    )
    if (
        phase not in {"approach", "recovery"}
        and phase not in reacquisition_phases
        and side_misaligned
    ):
        if side_reacquisition_enabled:
            action = side_route_action(
                contact_normal, direction,
                reacquisition_retreat_magnitude(args),
            )
        elif getattr(args, "side_recovery_mode", "contact_target") == "contact_target":
            action, recovery_audit = bounded_recovery_action(
                approach / float(args.approach_action_scale), args,
            )
            if float(getattr(args, "recovery_action_max_magnitude", 0.0)) <= 0.0:
                action = np.clip(
                    action,
                    -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
                    float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
                ).astype(np.float32)
        else:
            action, recovery_audit = bounded_recovery_action(
                -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)) * contact_normal,
                args,
            )
        return action, "recovery", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": 0.0,
            "contact_side_alignment": contact_side_alignment,
            "contact_side_recovery": 1.0,
            "contact_side_recovery_retreat": float(
                getattr(args, "side_recovery_mode", "contact_target") == "retreat"
            ),
            "contact_side_target_recovery": float(
                getattr(args, "side_recovery_mode", "contact_target") == "contact_target"
            ),
            "contact_side_misalignment_count": float(contact_side_misalignment_count),
            "contact_side_gate_suppressed_by_progress": float(progressing_contact),
            "reacquisition_clear_step": float(side_reacquisition_enabled),
            "reacquisition_attempt": float(side_reacquisition_enabled),
            "reacquisition_success": 0.0,
            **recovery_anchor_trace(active_recovery_anchor),
        }
    if (
        phase == "approach"
        and side_reacquisition_enabled
        and side_misaligned
        and not progressing_contact
    ):
        # A terminal root can start close to the block but on the wrong side
        # of the goal-facing contact manifold.  The direct desired-contact
        # vector then crosses the asymmetric T and pushes it away before the
        # controller has a valid contact lease.  The old branch only handled
        # this case after ``approach_distance`` became small, which was too
        # late: the block had already drifted tens of pixels.  Route around
        # the contact normal as soon as the RGB side test is negative, using
        # the same bounded side primitive as post-contact recovery.
        precontact_anchor = make_recovery_anchor(
            block, direction, agent, reacquisition_clearance,
        )
        precontact_waypoint, _precontact_arc, _precontact_outer, _precontact_target = (
            recovery_anchor_geometry(precontact_anchor, float(args.contact_distance))
        )
        # Start with the bounded outward side-route primitive, then persist
        # the anchor so the next frame enters the fixed waypoint/arc state
        # machine instead of recomputing a fresh pulse from a moving RGB
        # centroid.
        precontact_clear_action = side_route_action(
            contact_normal, direction,
            reacquisition_retreat_magnitude(args),
        )
        if observed_agent_block_distance >= float(precontact_anchor["clearance"]):
            precontact_clear_action = np.clip(
                (precontact_waypoint - agent) / float(args.approach_action_scale),
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            ).astype(np.float32)
        return precontact_clear_action, "reacquire_clear", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": 1.0,
            "contact_side_alignment": contact_side_alignment,
            "precontact_wrong_side_route": 1.0,
            "reacquisition_attempt": 1.0,
            "reacquisition_success": 0.0,
            **recovery_anchor_trace(precontact_anchor),
        }
    if (
        phase == "approach"
        and side_misaligned
        and approach_distance <= float(args.contact_tolerance)
    ):
        action = np.clip(
            -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)) * contact_normal,
            -1.0, 1.0,
        ).astype(np.float32)
        return action, "approach", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": float(contact_lost),
            "contact_side_alignment": contact_side_alignment,
            "handoff_blocked_by_contact_side": 1.0,
            "contact_side_recovery_retreat": 1.0,
            "contact_side_misalignment_count": float(contact_side_misalignment_count),
            "contact_side_gate_suppressed_by_progress": float(progressing_contact),
        }
    if (
        phase == "approach"
        and observed_agent_block_distance > recovery_handoff_distance
        and not support_contact_lease_valid
    ):
        # A transient block-motion estimate cannot establish contact while
        # the pusher is still far from the block.  This explicit absolute
        # distance gate prevents dynamic RGB calibration noise from turning
        # the approach servo into an early learned handoff.
        action = np.clip(approach / float(args.approach_action_scale), -1.0, 1.0).astype(np.float32)
        return action, "approach", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": float(contact_lost),
            "handoff_blocked_by_absolute_contact_distance": 1.0,
        }
    if (
        phase == "approach"
        and approach_distance > args.contact_tolerance
        and not approach_contact_confirmed
    ):
        action = np.clip(approach / float(args.approach_action_scale), -1.0, 1.0).astype(np.float32)
        return action, "approach", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": float(contact_lost),
            "handoff_blocked_by_contact_geometry": float(
                observed_block_motion > args.contact_recovery_stagnation
            ),
        }
    approach_handoff = phase in {"approach", "recovery"} and approach_distance > args.contact_tolerance
    if phase in {"approach", "recovery"}:
        phase = "push"
    current_angle = float(np.arctan2(visual[4], visual[5]))
    goal_angle = float(np.arctan2(goal_pose_raw[2], goal_pose_raw[3]))
    angle_error = float(abs(wrap(goal_angle - current_angle)))
    visual_xy = float(np.linalg.norm(goal_xy - block))
    terminal_block = block
    if (
        getattr(args, "terminal_xy_observer", "runtime") == "decoder"
        and xy_decoder is not None
        and xy_decoder_training is not None
    ):
        terminal_block = predict_orientation(
            xy_decoder, features, xy_decoder_training, args.device,
        )[0].astype(np.float32)
    terminal_xy = float(np.linalg.norm(goal_xy - terminal_block))
    tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    if (
        bool(getattr(args, "terminal_angle_progress_guard", False))
        and phase in {"torque", "hold"}
        and previous_action is not None
        and contact_side_alignment >= side_alignment_threshold
    ):
        if (
            rgb_current_angle_decoder is not None
            and rgb_current_angle_training is not None
            and bool(getattr(args, "_use_rgb_current_angle_runtime", False))
        ):
            previous_rgb_orientation = predict_orientation(
                rgb_current_angle_decoder,
                runtime_rgb_angle_features(
                    previous, "block", rgb_current_angle_training, args.image_size,
                ),
                rgb_current_angle_training, args.device,
            )[0]
            current_rgb_orientation = predict_orientation(
                rgb_current_angle_decoder,
                runtime_rgb_angle_features(
                    image, "block", rgb_current_angle_training, args.image_size,
                ),
                rgb_current_angle_training, args.device,
            )[0]
            previous_rgb_angle = float(np.arctan2(
                previous_rgb_orientation[0], previous_rgb_orientation[1],
            ))
            current_rgb_angle = float(np.arctan2(
                current_rgb_orientation[0], current_rgb_orientation[1],
            ))
        else:
            _previous_rgb_xy, previous_rgb_angle = rgb_geometry_xy_angle_from_image(
                previous, "block",
            )
            _current_rgb_xy, current_rgb_angle = rgb_geometry_xy_angle_from_image(
                image, "block",
            )
        if previous_rgb_angle is not None and current_rgb_angle is not None:
            previous_rgb_error = abs(float(wrap(goal_angle - previous_rgb_angle)))
            current_rgb_error = abs(float(wrap(goal_angle - current_rgb_angle)))
            if current_rgb_error > previous_rgb_error + float(
                getattr(args, "terminal_angle_progress_tolerance", 0.02)
            ):
                # Reverse only the tangential component of the last pulse and
                # retain a small inward component.  This is a one-step RGB
                # feedback repair for multi-step ranking drift; it does not
                # infer a simulator successor or use hidden state.
                current_contact_normal = np.asarray(contact_normal, dtype=np.float32)
                previous_action_array = np.asarray(previous_action, dtype=np.float32)
                previous_tangent = float(np.dot(previous_action_array, tangent))
                inward = max(
                    float(np.dot(previous_action_array, current_contact_normal)),
                    float(getattr(args, "terminal_contact_inward_min", 0.0)),
                    0.02,
                )
                tangent_magnitude = min(
                    abs(previous_tangent),
                    float(getattr(args, "terminal_action_max_magnitude", 0.08)),
                )
                corrective = inward * current_contact_normal
                if tangent_magnitude > 1.0e-6:
                    corrective = corrective - np.sign(previous_tangent) * tangent_magnitude * tangent
                cap = float(getattr(args, "terminal_action_max_magnitude", 0.08))
                corrective_norm = float(np.linalg.norm(corrective))
                if cap > 0.0 and corrective_norm > cap:
                    corrective = corrective * (cap / corrective_norm)
                return np.asarray(corrective, dtype=np.float32), "torque", {
                    "approach_distance": approach_distance,
                    "observed_agent_block_distance": observed_agent_block_distance,
                    "observed_block_motion": observed_block_motion,
                    "contact_side_alignment": contact_side_alignment,
                    "contact_lost_recovery": float(contact_lost),
                    "angle_error_visual": angle_error,
                    "terminal_angle_progress_guard_used": 1.0,
                    "terminal_angle_previous_rgb_error": previous_rgb_error,
                    "terminal_angle_current_rgb_error": current_rgb_error,
                    "terminal_angle_progress_tolerance": float(
                        getattr(args, "terminal_angle_progress_tolerance", 0.02)
                    ),
                    "terminal_angle_corrective_action": np.asarray(
                        corrective, dtype=np.float32,
                    ).tolist(),
                    "reacquisition_attempt": 0.0,
                    "reacquisition_success": 0.0,
            **recovery_audit,
                }
    offsets = [0.0, -np.pi / 6.0, np.pi / 6.0, -np.pi / 3.0, np.pi / 3.0, -np.pi / 2.0, np.pi / 2.0]
    # Compute the zero-action boundary chart before constructing the candidate
    # fan.  The support gap is an observation-only contact certificate: unlike
    # agent--block centroid distance, it remains meaningful for the
    # asymmetric T whose valid support point can be far from its centroid.
    current_rgb = resize_image(image, args.image_size)
    zero_boundary_chart = contact_boundary_chart_from_images(
        current_rgb[None], np.zeros((1, 2), dtype=np.float32),
    )[0]
    local_calibrated = False
    local_sample_count = len(local_history) if local_history else 0
    lease_gap_max = float(getattr(args, "terminal_contact_lease_gap_max", 0.0))
    terminal_contact_lease_valid, runtime_contact_boundary_gap_px, runtime_boundary_support_visible = (
        terminal_contact_lease_from_boundary(zero_boundary_chart, lease_gap_max)
    )
    # Contact-boundary support can be occluded even when the RGB centroids
    # place the agent inside the declared contact band on the goal-facing
    # side.  In that specific case, trust the jointly observed proximity and
    # side alignment as a conservative fallback; it cannot activate for a
    # distant or wrong-side pusher.  The old gate sent these valid terminal
    # roots into recovery, whose approach action then pushed the block away.
    terminal_proximity_fallback = bool(
        lease_gap_max <= 0.0
        and
        observed_agent_block_distance
        <= float(args.contact_distance) + float(args.contact_tolerance)
        and contact_side_alignment >= side_alignment_threshold
    )
    terminal_contact_lease_for_control = bool(
        terminal_contact_lease_valid
        or support_contact_lease_valid
        or terminal_proximity_fallback
        or terminal_proximity_lease_candidate
    )
    candidates: list[np.ndarray] = []
    # The original fan had no action below 0.35.  That makes orientation
    # closure impossible to separate from translation: every torque repair
    # is also a large push.  Keep the broad fan for translation, and add a
    # declared small fan only on the terminal manifold.
    # Keep the runtime candidate support inside the intervention distribution
    # used to fit the adapter.  The previous fan introduced 0.95 actions and
    # arbitrary goal-relative rotations although the default training bank is
    # {0.35, 0.60, 0.85}; those candidates forced extrapolation exactly where
    # contact was most fragile.
    broad_magnitudes = tuple(float(value) for value in args.action_magnitudes)
    contact_band = observed_agent_block_distance <= (
        float(args.contact_distance) + 2.0 * float(args.contact_tolerance)
    )
    contact_lock_active = (
        phase not in {"approach", "recovery"}
        and not contact_lost
        and (
            observed_agent_block_distance <= float(args.contact_lock_distance)
            or (
                lease_gap_max > 0.0
                and runtime_boundary_support_visible
                and runtime_contact_boundary_gap_px <= lease_gap_max
            )
        )
    )
    # ``candidate_contact_band`` is deliberately wider than the phase
    # handoff band.  It protects the learned controller during the transient
    # after a valid push while preserving the explicit recovery branch above.
    candidate_contact_band = bool(contact_band or contact_lock_active)
    terminal_repair_xy = float(getattr(args, "terminal_repair_xy", -1.0))
    if terminal_repair_xy <= 0.0:
        terminal_repair_xy = float(args.terminal_xy)
    terminal_torque_escape_limit = float(
        getattr(args, "terminal_torque_escape_limit", -1.0)
    )
    if terminal_torque_escape_limit <= 0.0:
        terminal_torque_escape_limit = terminal_repair_xy
    terminal_fan = terminal_xy <= float(args.terminal_xy) or (
        phase in {"torque", "hold"}
        and terminal_xy <= terminal_torque_escape_limit
    )
    terminal_action_cap = float(getattr(args, "terminal_action_max_magnitude", -1.0))
    if terminal_action_cap <= 0.0:
        terminal_action_cap = float(args.contact_action_max_magnitude)
    candidate_magnitude_cap = terminal_action_cap if terminal_fan else float(args.contact_action_max_magnitude)
    if candidate_contact_band:
        magnitudes = tuple(sorted(set(
            float(value) for value in args.small_action_magnitudes
            if float(value) <= candidate_magnitude_cap
        ) | {candidate_magnitude_cap}))
    elif visual_xy <= float(args.terminal_xy):
        magnitudes = tuple(float(value) for value in args.small_action_magnitudes) + broad_magnitudes
    else:
        magnitudes = broad_magnitudes
    for magnitude in magnitudes:
        for offset in offsets:
            c, s = float(np.cos(offset)), float(np.sin(offset))
            rotated = np.asarray([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]], dtype=np.float32)
            candidates.append(magnitude * rotated)
    # Add the exact absolute action bank used for offline intervention labels.
    # The relative fan remains useful for local geometry, while these actions
    # provide an in-distribution support on which the learned adapter was
    # actually trained.
    for magnitude in tuple(float(value) for value in args.action_magnitudes) + tuple(
        float(value) for value in args.small_action_magnitudes
        if float(value) <= float(args.contact_action_max_magnitude)
    ):
        if candidate_contact_band and magnitude > candidate_magnitude_cap:
            continue
        for index in range(int(args.action_directions)):
            angle = 2.0 * np.pi * index / float(args.action_directions)
            candidates.append(np.asarray([
                magnitude * np.cos(angle), magnitude * np.sin(angle),
            ], dtype=np.float32))
    candidates.append(np.zeros(2, dtype=np.float32))
    actions = np.asarray(candidates, dtype=np.float32)
    boundary_charts = contact_boundary_chart_from_images(
        np.repeat(current_rgb[None], len(actions), axis=0), actions,
    )
    # A terminal phase is a temporary contact lease, not a license to keep
    # applying the broad action fan after the pusher has separated.  The old
    # implementation let ``phase == torque`` survive a failed contact check
    # whenever the block was still moving, which converted motion after
    # separation into false evidence of a valid contact mode.  Fail closed to
    # the existing RGB contact-target recovery servo; no simulator state or
    # successor query is used here.
    if (
        lease_gap_max > 0.0
        and (
            phase in {"torque", "hold"}
            or terminal_xy <= float(args.terminal_xy)
        )
        and not terminal_contact_lease_for_control
    ):
        recovery_action, recovery_audit = bounded_recovery_action(
            approach / float(args.approach_action_scale), args,
        )
        return recovery_action, "recovery", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_side_alignment": contact_side_alignment,
            "contact_lost_recovery": 1.0,
            "terminal_contact_lease_valid": 0.0,
            "terminal_contact_proximity_fallback": 0.0,
            "terminal_contact_lease_recovery": 1.0,
            "terminal_contact_boundary_gap_px": runtime_contact_boundary_gap_px,
            "terminal_contact_lease_gap_max": lease_gap_max,
            "terminal_boundary_support_visible": float(runtime_boundary_support_visible),
            "terminal_locked": 0.0,
            "predicted_best_cost": float("nan"),
            "online_calibration_samples": float(local_sample_count),
            "online_calibrated": float(local_calibrated),
            "action_cost_ranker_used": 0.0,
            **recovery_audit,
        }
    runtime_previous_action = (
        np.zeros(2, dtype=np.float32)
        if previous_action is None else np.asarray(previous_action, dtype=np.float32)
    )
    feature_rows = np.concatenate(
        [runtime_features(
            features, previous_features, action, boundary_charts[index], runtime_previous_action,
        )
         for index, action in enumerate(actions)], axis=0,
    )
    predicted = predict_targets(adapter, feature_rows, adapter_training, args.device)
    # Keep an explicit frame flag.  The canonical KNN already returns a
    # contact-frame prediction rotated into world coordinates below; the
    # generic adapter path is rotated later according to its training frame.
    # Without this flag a selected KNN was rotated twice whenever the base
    # adapter had been trained in contact coordinates, corrupting both XY and
    # angular candidate ranking.
    predicted_effect_in_world = adapter_training.get("effect_frame", "world") != "contact"
    contact_knn_used = False
    if contact_knn_training is not None and candidate_contact_band:
        # The table predicts in the contact frame and is rotated by the
        # current RGB geometry.  It is never queried outside the contact
        # manifold, where the held-out validation gate was measured.
        knn_prediction = contact_canonical_knn_predict(contact_knn_training, feature_rows)
        predicted = contact_effect_to_world(knn_prediction, agent, block)
        predicted_effect_in_world = True
        contact_knn_used = True
    zero_index = int(np.flatnonzero(np.all(np.isclose(actions, 0.0), axis=1))[0])
    zero_features = runtime_features(
        features, previous_features, np.zeros(2, dtype=np.float32), boundary_charts[zero_index],
        runtime_previous_action,
    )
    predicted_zero = predict_targets(adapter, zero_features, adapter_training, args.device)[0, :POSE_DIM]
    local_calibrated = False
    local_sample_count = 0
    local_angle_prediction: np.ndarray | None = None
    if local_history:
        local_sample_count = len(local_history)
        action_history = np.asarray([item[0] for item in local_history], dtype=np.float64)
        effect_history = np.asarray([item[1] for item in local_history], dtype=np.float64)
        design = np.column_stack([action_history, np.ones(len(action_history))])
        if (
            len(local_history) >= int(args.online_local_min_samples)
            and np.linalg.matrix_rank(action_history) >= 2
        ):
            # Ridge is solved explicitly to keep the runtime dependency small.
            # The intercept captures passive drift; candidate effects are
            # always differenced against the fitted zero-action intercept.
            regularizer = np.diag([0.05, 0.05, 1.0])
            coefficients = np.linalg.solve(
                design.T @ design + regularizer,
                design.T @ effect_history,
            )
            local_prediction = np.column_stack([actions, np.ones(len(actions))]) @ coefficients
            local_zero = coefficients[-1]
            local_effect = local_prediction - local_zero[None]
            blend = float(np.clip(args.online_local_weight, 0.0, 1.0))
            if adapter_training.get("effect_frame", "world") == "contact" and blend > 0.0:
                local_effect = world_effect_to_contact(
                    np.column_stack([np.zeros((len(actions), POSE_DIM), dtype=np.float32), local_effect]),
                    agent, block,
                )[:, POSE_DIM:]
            predicted[:, POSE_DIM:] = (1.0 - blend) * predicted[:, POSE_DIM:] + blend * local_effect
            local_calibrated = blend > 0.0
            angle_blend = float(np.clip(getattr(args, "online_local_angle_weight", 0.0), 0.0, 1.0))
            if angle_blend > 0.0:
                predicted[:, 5] = (
                    (1.0 - angle_blend) * predicted[:, 5]
                    + angle_blend * local_effect[:, 2]
                )
                local_calibrated = True
            local_angle_prediction = np.asarray(local_effect[:, 2], dtype=np.float32)
    if not predicted_effect_in_world:
        predicted = contact_effect_to_world(predicted, agent, block)
    predicted_block = np.tile(block, (len(actions), 1))
    predicted_block = np.column_stack([
        predicted_block[:, 0] + predicted_zero[0] + predicted[:, 3],
        predicted_block[:, 1] + predicted_zero[1] + predicted[:, 4],
        current_angle + predicted_zero[2] + predicted[:, 5],
    ])
    goal_block = np.asarray([goal_xy[0], goal_xy[1], goal_angle], dtype=np.float32)
    terminal_candidate = terminal_xy <= float(args.terminal_xy) or support_near_terminal or (
        phase in {"torque", "hold"}
        and terminal_xy <= terminal_torque_escape_limit
    )
    terminal_locked = bool(
        terminal_candidate
        and int(terminal_confirmation_count) + 1
        >= max(1, int(getattr(args, "terminal_confirmation_steps", 2)))
        and observed_block_motion <= float(getattr(args, "terminal_max_observer_jump", 8.0))
    )
    # Use the one-frame terminal candidate interface for specialized torque
    # selection. The stricter hysteretic lock can remain false under RGB
    # occlusion even when a valid terminal observation is available.
    terminal_candidate = (
        terminal_xy <= float(getattr(args, "terminal_xy", 35.0))
        or support_near_terminal
        or (
        phase in {"torque", "hold"}
        and terminal_xy <= terminal_torque_escape_limit
        )
    )
    # A proximity lease can certify that the block is near its goal, but it
    # cannot certify that the pusher is still close enough to transmit force.
    # In particular, a failed probe may trigger recovery and leave the agent
    # 50+ pixels away while the old proximity path continues issuing the
    # small terminal action fan.  That creates a false terminal-contact loop:
    # the block is stationary, the agent cannot reacquire it, and every later
    # action is interpreted as another failed torque attempt.  Apply the
    # declared RGB agent-distance gate even when the support-gap/proximity
    # fallback is active.  This is observation-only and preserves the legacy
    # protocol when the max is zero.
    terminal_agent_distance_max = float(
        getattr(args, "terminal_agent_distance_max", 0.0)
    )
    if (
        terminal_candidate
        and terminal_agent_distance_max > 0.0
        and observed_agent_block_distance > terminal_agent_distance_max
    ):
        # This gate is defined in centroid distance, so its servo target must
        # have the same semantics.  Reusing the asymmetric boundary-support
        # target here can point to a corner that is tens of pixels away from
        # the centroid-facing contact and causes lateral oscillation after a
        # failed probe.  The boundary target remains available to the legacy
        # support-gap protocol below; this recovery gate is centroid-only.
        contact_target = np.asarray(desired_agent, dtype=np.float32)
        contact_target_mode = "centroid_distance_gate"
        rgb_agent, _ = rgb_geometry_xy_angle_from_image(image, "agent")
        approach_agent = (
            np.asarray(agent, dtype=np.float32)
            if rgb_agent is None
            else np.asarray(rgb_agent, dtype=np.float32)
        )
        contact_approach = contact_target - approach_agent
        approach_action, approach_raw_norm, approach_final_norm = bounded_l2_action(
            contact_approach / float(args.approach_action_scale),
            float(getattr(args, "terminal_approach_action_max_magnitude", -1.0)),
        )
        return approach_action, "terminal_approach", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_side_alignment": contact_side_alignment,
            "contact_lost_recovery": 0.0,
            "terminal_agent_distance_gate_used": 1.0,
            "terminal_agent_distance_px": observed_agent_block_distance,
            "terminal_agent_distance_max": terminal_agent_distance_max,
            "terminal_contact_target": contact_target.tolist(),
            "terminal_contact_target_mode": contact_target_mode,
            "terminal_contact_target_distance": float(np.linalg.norm(contact_approach)),
            "terminal_approach_raw_norm": approach_raw_norm,
            "terminal_approach_final_norm": approach_final_norm,
            "reacquisition_attempt": 0.0,
            "reacquisition_success": 0.0,
            "predicted_best_cost": float("nan"),
            "action_cost_ranker_used": 0.0,
        }
    translation_gap_max = float(getattr(args, "translation_contact_gap_max", 0.0))
    near_aligned_contact = bool(
        observed_agent_block_distance
        <= float(args.contact_distance) + 0.5 * float(args.contact_tolerance)
        and contact_side_alignment >= side_alignment_threshold
    )
    translation_contact_gap_recovery = bool(
        translation_gap_max > 0.0
        and phase in {"push", "translate"}
        and not terminal_candidate
        and runtime_boundary_support_visible
        and runtime_contact_boundary_gap_px > translation_gap_max
        and not near_aligned_contact
    )
    if translation_contact_gap_recovery:
        recovery_action, recovery_audit = bounded_recovery_action(
            approach / float(args.approach_action_scale), args,
        )
        return recovery_action, "recovery", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_side_alignment": contact_side_alignment,
            "contact_lost_recovery": 1.0,
            "translation_contact_gap_recovery": 1.0,
            "translation_contact_gap_px": runtime_contact_boundary_gap_px,
            "translation_contact_gap_max": translation_gap_max,
            "translation_contact_boundary_support_visible": 1.0,
            "predicted_best_cost": float("nan"),
            "online_calibration_samples": float(local_sample_count),
            "online_calibrated": float(local_calibrated),
            "action_cost_ranker_used": 0.0,
            **recovery_audit,
        }
    terminal_contact_gap_px = float("nan")
    terminal_contact_gap_gate_used = False
    terminal_agent_distance_gate_used = False
    if (
        terminal_candidate
        and float(getattr(args, "terminal_contact_gap_max", 0.0)) > 0.0
        and not terminal_proximity_lease_candidate
    ):
        # A support-point gap is necessary but not sufficient for terminal
        # torque.  On the asymmetric T, the same visible boundary can be
        # approached from the wrong side; handing off there sends the learned
        # angular selector across the block and can reverse the pose.  Apply
        # the side gate before the gap gate so this transition is owned by the
        # declared RGB-only reacquisition state machine.
        terminal_wrong_side = (
            bool(getattr(args, "terminal_side_reacquisition", False))
            and contact_side_alignment < side_alignment_threshold
        )
        if terminal_wrong_side:
            # A small cosine drop while the pusher is still inside the broad
            # contact band is not evidence that the contact manifold is
            # unreachable.  Clearing to the 100--120 px outer route in this
            # case was the dominant terminal limit cycle: the pusher was only
            # 15--20 px from the block, but the route deliberately drove it
            # away and repeatedly reintroduced block drift.  First repair the
            # local centroid-facing contact with a bounded terminal approach;
            # reserve side reacquisition for genuinely remote or strongly
            # wrong-side configurations.
            contact_band_distance = float(args.contact_distance) + 2.0 * float(
                args.contact_tolerance
            )
            mild_wrong_side = bool(
                observed_agent_block_distance <= contact_band_distance
                and contact_side_alignment >= -0.10
                and terminal_contact_target_mode == "centroid"
            )
            if mild_wrong_side:
                terminal_contact_approach = np.asarray(
                    desired_agent - agent, dtype=np.float32,
                )
                approach_action, approach_raw_norm, approach_final_norm = bounded_l2_action(
                    terminal_contact_approach / float(args.approach_action_scale),
                    float(getattr(args, "terminal_approach_action_max_magnitude", -1.0)),
                )
                return approach_action, "terminal_approach", {
                    "approach_distance": approach_distance,
                    "observed_agent_block_distance": observed_agent_block_distance,
                    "observed_block_motion": observed_block_motion,
                    "contact_side_alignment": contact_side_alignment,
                    "contact_lost_recovery": 0.0,
                    "terminal_side_gate_local_repair": 1.0,
                    "terminal_side_gate_before_gap": 1.0,
                    "terminal_contact_target_mode": terminal_contact_target_mode,
                    "terminal_contact_target_distance": float(
                        np.linalg.norm(terminal_contact_approach)
                    ),
                    "terminal_approach_raw_norm": approach_raw_norm,
                    "terminal_approach_final_norm": approach_final_norm,
                    "reacquisition_attempt": 0.0,
                    "reacquisition_success": 0.0,
                }
            return side_route_action(
                contact_normal, direction,
                reacquisition_retreat_magnitude(args),
            ), "reacquire_clear", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": 1.0,
                "terminal_side_gate_before_gap": 1.0,
                "reacquisition_attempt": 1.0,
                "reacquisition_success": 0.0,
                **recovery_anchor_trace(active_recovery_anchor),
            }
        # A small block-to-goal distance does not imply that the pusher is at
        # a controllable support point.  The old terminal interface used only
        # the former and consequently applied torque while the agent was
        # visibly separated from the T boundary.  Compute the gap from the
        # same RGB boundary chart used by the torque-sign guard.  This is an
        # observation-only gate and does not inspect a simulator successor.
        zero_boundary = contact_boundary_chart_from_images(
            current_rgb[None], np.zeros((1, 2), dtype=np.float32),
        )[0]
        terminal_contact_gap_px = float(zero_boundary[8]) * (PUSHT_STATE_SIZE / 2.0)
        terminal_agent_distance_max = float(getattr(args, "terminal_agent_distance_max", 0.0))
        center_distance_too_large = (
            terminal_agent_distance_max > 0.0
            and observed_agent_block_distance > terminal_agent_distance_max
        )
        centroid_target_distance = float(np.linalg.norm(approach))
        # The centroid target is a preferred acquisition point, not a reason
        # to break an already valid contact manifold.  With asymmetric T
        # geometry and a learned agent readout, the exact target can remain
        # several pixels away even while the pusher is close enough and on
        # the correct side to transmit torque.  Chasing that residual gap
        # turns a valid terminal contact into an outward approach pulse and
        # was the direct cause of the torque -> terminal_approach -> recovery
        # loop.  Preserve contact whenever the independent distance band and
        # side-alignment checks already certify a usable support state.
        centroid_contact_not_reached = bool(
            centroid_target_distance > float(args.contact_tolerance)
            and not near_aligned_contact
        )
        use_boundary_target = terminal_contact_target_mode == "boundary"
        if (
            (use_boundary_target and terminal_contact_gap_px > float(args.terminal_contact_gap_max))
            or center_distance_too_large
            or (not use_boundary_target and centroid_contact_not_reached)
        ):
            if use_boundary_target:
                contact_point = (zero_boundary[:2] + 1.0) * (PUSHT_STATE_SIZE / 2.0)
                contact_target = np.asarray(contact_point, dtype=np.float32)
                # The gap was measured from the raw RGB boundary chart.  Use
                # the same raw RGB agent centroid for the approach displacement;
                # the learned agent decoder can have a small absolute bias
                # that is harmless for pose scoring but large enough to miss a
                # point contact when the two coordinate interfaces are mixed.
                rgb_agent_xy, _ = rgb_geometry_xy_angle_from_image(image, "agent")
                approach_agent = agent if rgb_agent_xy is None else rgb_agent_xy
                contact_approach = contact_target - np.asarray(approach_agent, dtype=np.float32)
            else:
                # Centroid mode is intentionally based on the same paired RGB
                # pose chart used by the state machine.  It avoids turning a
                # noisy/asymmetric boundary support point into a false motion
                # command; the boundary gap remains in the trace for diagnosis.
                contact_target = np.asarray(desired_agent, dtype=np.float32)
                contact_approach = contact_target - np.asarray(agent, dtype=np.float32)
            terminal_contact_gap_gate_used = True
            terminal_agent_distance_gate_used = bool(center_distance_too_large)
            terminal_approach_action, terminal_approach_raw_norm, terminal_approach_final_norm = bounded_l2_action(
                contact_approach / float(args.approach_action_scale),
                float(getattr(args, "terminal_approach_action_max_magnitude", -1.0)),
            )
            return terminal_approach_action, "terminal_approach", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": float(contact_lost),
                "angle_error_visual": angle_error,
                "block_xy_visual": visual_xy,
                "block_xy_terminal_observer": terminal_xy,
                "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
                "terminal_locked": 0.0,
                "terminal_contact_gap_px": terminal_contact_gap_px,
                "terminal_contact_gap_gate_used": 1.0,
                "terminal_contact_approach_complete": 0.0,
                "terminal_agent_distance_px": observed_agent_block_distance,
                "terminal_agent_distance_gate_used": float(terminal_agent_distance_gate_used),
                "terminal_agent_distance_max": terminal_agent_distance_max,
                "terminal_contact_target": contact_target.tolist(),
                "terminal_contact_target_mode": terminal_contact_target_mode,
                "terminal_contact_target_distance": float(np.linalg.norm(contact_approach)),
                "terminal_centroid_contact_relaxed": float(
                    bool(not use_boundary_target and near_aligned_contact)
                ),
                "terminal_approach_action_cap": float(
                    getattr(args, "terminal_approach_action_max_magnitude", -1.0)
                ),
                "terminal_approach_raw_norm": terminal_approach_raw_norm,
                "terminal_approach_final_norm": terminal_approach_final_norm,
                "predicted_best_cost": float("nan"),
                "online_calibration_samples": float(local_sample_count),
                "online_calibrated": float(local_calibrated),
                "action_cost_ranker_used": 0.0,
            }
    # ``terminal_locked`` is intentionally hysteretic and can be cleared by
    # one large but otherwise valid RGB observer jump.  The controller may
    # nevertheless remain in the torque phase while the block is still in the
    # terminal tube.  Use a wider control-active predicate for all terminal
    # torque feasibility checks; otherwise a single jump silently reopens the
    # unconstrained learned fan and recreates contact ejection.
    # ``terminal_candidate`` may remain true through the support/contact lease
    # while the block has already left the visual terminal tube.  In that
    # state ``magnitudes`` has deliberately switched back to the broad
    # translation fan.  The pure predicate prevents the angular specialist
    # from ranking that incompatible action set.
    terminal_control_active = terminal_control_lease_active(
        terminal_candidate,
        terminal_fan,
        candidate_contact_band,
        lease_gap_max,
        terminal_contact_lease_for_control,
    )
    if (
        bool(getattr(args, "terminal_motion_damping", False))
        and terminal_control_active
        and phase in {"torque", "hold"}
        and previous_action is not None
        and float(np.linalg.norm(previous_action)) > 1.0e-6
        and abs(angle_error) > float(args.terminal_angle)
        and observed_block_motion >= float(
            getattr(args, "terminal_motion_damping_motion", 2.0)
        )
        and goal_progress <= -float(
            getattr(args, "terminal_motion_damping_regression", 0.5)
        )
    ):
        # A terminal torque pulse is allowed to rotate the block, but it is
        # not allowed to accumulate unobserved translation.  This RGB-only
        # settle pulse is triggered by the *next* frame's measured motion and
        # goal-distance regression; it never consumes simulator state or a
        # successor query.  The following decision returns to the same
        # learned terminal selector with a fresh contact observation.
        # A settle pulse is a terminal safety exit, not a successful torque
        # decision.  Returning ``torque`` here lets the next receding step
        # immediately re-enter the same unstable terminal ranker.  Returning
        # ``push`` hands hybrid control back to the RGB translation servo,
        # which can reacquire a safe contact manifold before any new torque
        # candidate is considered.
        return actions[zero_index], "push", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_side_alignment": contact_side_alignment,
            "contact_lost_recovery": float(contact_lost),
            "angle_error_visual": angle_error,
            "block_xy_visual": visual_xy,
            "block_xy_terminal_observer": terminal_xy,
            "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
            "terminal_locked": float(terminal_locked),
            "terminal_control_active": 1.0,
            "terminal_motion_damping_used": 1.0,
            "force_geometry_translation": 1.0,
            "terminal_motion_damping_motion": observed_block_motion,
            "terminal_motion_damping_regression": goal_progress,
            "predicted_best_cost": float("nan"),
            "online_calibration_samples": float(local_sample_count),
            "online_calibrated": float(local_calibrated),
            "action_cost_ranker_used": 0.0,
        }
    if phase in {"torque", "hold"} and not terminal_control_active:
        # A torque phase is a temporary terminal-manifold lease.  Once the
        # small terminal fan is inactive, continuing in the torque/hold state
        # is a state-machine error even when the wider escape limit has not
        # yet fired: the candidate set has already switched to translation.
        # Exit the phase immediately so commitment, angular ranking, and
        # torque-specific fallbacks cannot leak into the broad action fan.
        phase = "push"
    if terminal_control_active:
        # Use an explicitly phase-specific objective.  A single weighted sum
        # lets a noisy angle readout override translation while far from the
        # target; conversely, on the terminal manifold it lets a large action
        # buy a small angle improvement by ejecting the block.
        scores = (
            np.abs(wrap(predicted_block[:, 2] - goal_angle))
            + float(args.terminal_xy_weight)
            * np.linalg.norm(predicted_block[:, :2] - goal_xy[None], axis=1) / 20.0
        )
    else:
        scores = (
            np.linalg.norm(predicted_block[:, :2] - goal_xy[None], axis=1) / 20.0
            + float(args.translation_angle_weight)
            * np.abs(wrap(predicted_block[:, 2] - goal_angle))
        )
    if terminal_control_active:
        # Once the RGB-measured block is in the terminal XY tube, orientation
        # repair must be tube-preserving.  A pure angle objective can select a
        # torque action that improves theta while ejecting the block from the
        # already-solved translation region.  This is a predicted-state
        # constraint; no simulator successor is queried.
        xy_limit = max(float(args.terminal_xy), terminal_xy + 5.0)
        predicted_xy = np.linalg.norm(predicted_block[:, :2] - goal_xy[None], axis=1)
        violation = np.maximum(predicted_xy - xy_limit, 0.0)
        scores = scores + 10.0 * violation
        terminal_baseline_scores = np.asarray(scores, dtype=np.float32).copy()
        current_angle_error = float(abs(wrap(goal_angle - current_angle)))
        if (
            bool(args.terminal_hold)
            and terminal_xy <= float(args.terminal_hold_xy)
            and current_angle_error <= float(args.terminal_angle)
        ):
            # Zero is the only action that cannot destroy an already-solved
            # pose under this interface.  The environment/evaluator still
            # decides success; this is only a visual terminal interface.
            zero_index = int(np.flatnonzero(np.all(np.isclose(actions, 0.0), axis=1))[0])
            return actions[zero_index], "hold", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": float(contact_lost),
                "angle_error_visual": angle_error,
                "block_xy_visual": visual_xy,
                "block_xy_terminal_observer": terminal_xy,
                "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
                "terminal_locked": 1.0,
                "terminal_hold": 1.0,
                "terminal_contact_gap_px": terminal_contact_gap_px,
                "terminal_contact_gap_gate_used": float(terminal_contact_gap_gate_used),
                "terminal_agent_distance_px": observed_agent_block_distance,
                "terminal_agent_distance_gate_used": float(terminal_agent_distance_gate_used),
                "terminal_agent_distance_max": float(getattr(args, "terminal_agent_distance_max", 0.0)),
                "predicted_best_cost": float(scores[zero_index]),
                "online_calibration_samples": float(local_sample_count),
                "online_calibrated": float(local_calibrated),
                "action_cost_ranker_used": 0.0,
            }
        geometric_torque_side_threshold = float(
            getattr(args, "geometric_torque_side_alignment_min", side_alignment_threshold)
        )
        geometric_torque_distance_max = float(
            getattr(args, "geometric_torque_contact_distance_max", 0.0)
        )
        if (
            bool(getattr(args, "geometric_torque_fallback", False))
            and contact_side_alignment >= geometric_torque_side_threshold
            and observed_agent_block_distance <= (
                geometric_torque_distance_max
                if geometric_torque_distance_max > 0.0
                else float(args.contact_lock_distance)
            )
        ):
            # For a point contact, the sign of the planar moment is determined
            # by r x F, where r is the RGB-estimated block-to-agent lever arm.
            # The older goal-tangent primitive used the block-to-goal direction
            # instead of the actual lever arm, so it could have the correct
            # visual angle sign but the wrong physical torque sign.  Keep a
            # small inward component to avoid immediately leaving contact.
            support_lever_used = False
            lever = np.asarray(agent - block, dtype=np.float32)
            if bool(getattr(args, "geometric_torque_support_lever", False)):
                support_chart = contact_boundary_chart_from_images(
                    resize_image(image, args.image_size)[None],
                    np.zeros((1, 2), dtype=np.float32),
                )[0]
                support_lever = np.asarray(support_chart[2:4], dtype=np.float32)
                if float(np.linalg.norm(support_lever)) > 1.0e-4:
                    lever = support_lever
                    support_lever_used = True
            lever /= max(float(np.linalg.norm(lever)), 1e-6)
            lever_tangent = np.asarray([-lever[1], lever[0]], dtype=np.float32)
            torque_inward_direction = contact_normal
            if support_lever_used:
                # The force that keeps the pusher on the visible support is
                # normal to the support point, not necessarily normal to the
                # centroid-to-centroid segment.  Use the inward direction
                # opposite the boundary lever when the support chart is
                # available; retain the centroid fallback for occluded RGB.
                torque_inward_direction = -lever
            geometric_torque_angle = float(current_angle)
            geometric_torque_raw_angle_used = False
            if bool(getattr(args, "geometric_torque_raw_rgb_angle", False)):
                _raw_rgb_xy, raw_rgb_angle = rgb_geometry_xy_angle_from_image(
                    image, "block",
                )
                if raw_rgb_angle is not None and np.isfinite(float(raw_rgb_angle)):
                    geometric_torque_angle = float(raw_rgb_angle)
                    geometric_torque_raw_angle_used = True
            torque_sign = 1.0 if wrap(goal_angle - geometric_torque_angle) >= 0.0 else -1.0
            geometric_torque_magnitude = float(
                getattr(args, "geometric_torque_magnitude", 0.24)
            )
            geometric_torque_min_magnitude = float(
                np.clip(
                    getattr(args, "geometric_torque_min_magnitude", 0.06),
                    0.0,
                    geometric_torque_magnitude,
                )
            )
            geometric_torque_adaptive_used = bool(
                getattr(args, "geometric_torque_adaptive", False)
            )
            geometric_torque_motion_target = float(
                max(getattr(args, "geometric_torque_motion_target", 3.0), 1.0e-6)
            )
            if geometric_torque_adaptive_used and observed_block_motion > 0.0:
                # Use the previous RGB finite difference as a conservative
                # local gain estimate.  The first pulse is intentionally not
                # attenuated: a too-small pulse is indistinguishable from a
                # failed contact on PushT.  After motion is observed, the
                # inverse-motion rule prevents repeated tangential pushes from
                # accumulating translation and ejecting the pusher.
                geometric_torque_magnitude = float(
                    np.clip(
                        geometric_torque_magnitude
                        * geometric_torque_motion_target
                        / max(float(observed_block_motion), geometric_torque_motion_target),
                        geometric_torque_min_magnitude,
                        geometric_torque_magnitude,
                    )
                )
            geometric_torque_inward = float(
                getattr(args, "geometric_torque_inward_magnitude", 0.08)
            )
            if geometric_torque_adaptive_used and observed_block_motion > geometric_torque_motion_target:
                # Keep a smaller inward component with the attenuated tangent;
                # otherwise the controller can still drive the block forward
                # even after the torque component has been reduced.
                geometric_torque_inward = float(
                    np.clip(
                        geometric_torque_inward
                        * geometric_torque_magnitude
                        / max(float(getattr(args, "geometric_torque_magnitude", 0.24)), 1.0e-6),
                        0.02,
                        geometric_torque_inward,
                    )
                )
            geometric_torque_drift_abort = bool(
                getattr(args, "geometric_torque_drift_abort", False)
                and geometric_torque_adaptive_used
                and (
                    visual_xy > float(getattr(args, "geometric_torque_drift_xy", 20.0))
                    or observed_block_motion > float(
                        getattr(args, "geometric_torque_drift_motion", 10.0)
                    )
                )
            )
            torque_action = (
                geometric_torque_inward * torque_inward_direction
                + torque_sign * geometric_torque_magnitude * lever_tangent
            )
            motion_feedback_used = False
            motion_feedback = np.zeros(2, dtype=np.float32)
            if (
                bool(getattr(args, "geometric_torque_motion_feedback", False))
                and observed_block_motion > geometric_torque_motion_target
                and float(np.linalg.norm(observed_block_delta)) > 1.0e-6
            ):
                feedback_gain = float(
                    np.clip(
                        getattr(args, "geometric_torque_motion_feedback_gain", 0.08),
                        0.0,
                        0.25,
                    )
                )
                feedback_scale = min(
                    1.0,
                    float(observed_block_motion) / geometric_torque_motion_target,
                )
                motion_feedback = (
                    -feedback_gain * feedback_scale
                    * np.asarray(observed_block_delta, dtype=np.float32)
                    / max(float(np.linalg.norm(observed_block_delta)), 1.0e-6)
                )
                torque_action = torque_action + motion_feedback
                motion_feedback_used = True
            torque_action = np.clip(torque_action, -1.0, 1.0).astype(np.float32)
            return torque_action, "translate" if geometric_torque_drift_abort else "torque", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_side_alignment": contact_side_alignment,
                "contact_lost_recovery": float(contact_lost),
                "angle_error_visual": angle_error,
                "block_xy_visual": visual_xy,
                "block_xy_terminal_observer": terminal_xy,
                "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
                "terminal_locked": 1.0,
                "geometric_torque_fallback_used": 1.0,
                "terminal_contact_gap_px": terminal_contact_gap_px,
                "terminal_contact_gap_gate_used": float(terminal_contact_gap_gate_used),
                "geometric_torque_sign": float(torque_sign),
                "geometric_torque_angle_used": float(geometric_torque_angle),
                "geometric_torque_raw_rgb_angle_used": float(geometric_torque_raw_angle_used),
                "geometric_torque_lever_tangent": lever_tangent.tolist(),
                "geometric_torque_support_lever_used": float(support_lever_used),
                "geometric_torque_side_alignment_min": geometric_torque_side_threshold,
                "geometric_torque_contact_distance_max": (
                    geometric_torque_distance_max
                    if geometric_torque_distance_max > 0.0
                    else float(args.contact_lock_distance)
                ),
                "geometric_torque_support_lever": lever.tolist(),
                "geometric_torque_inward_direction": torque_inward_direction.tolist(),
                "geometric_torque_adaptive_used": float(geometric_torque_adaptive_used),
                "geometric_torque_magnitude_used": float(geometric_torque_magnitude),
                "geometric_torque_inward_magnitude_used": float(geometric_torque_inward),
                "geometric_torque_observed_motion": float(observed_block_motion),
                "geometric_torque_motion_target": float(geometric_torque_motion_target),
                "geometric_torque_motion_feedback_used": float(motion_feedback_used),
                "geometric_torque_motion_feedback": motion_feedback.tolist(),
                "geometric_torque_drift_abort": float(geometric_torque_drift_abort),
                "geometric_torque_drift_xy": float(
                    getattr(args, "geometric_torque_drift_xy", 20.0)
                ),
                "geometric_torque_drift_motion": float(
                    getattr(args, "geometric_torque_drift_motion", 10.0)
                ),
                "force_geometry_translation": float(geometric_torque_drift_abort),
                "predicted_best_cost": float("nan"),
                "online_calibration_samples": float(local_sample_count),
                "online_calibrated": float(local_calibrated),
                "action_cost_ranker_used": 0.0,
            }
    local_calibrated = False
    local_sample_count = len(local_history) if local_history else 0
    stagnant_contact = False
    if (
        bool(getattr(args, "contact_stagnation_fallback", False))
        and candidate_contact_band
        and local_history
        and len(local_history) >= int(getattr(args, "contact_stagnation_window", 3))
    ):
        recent_effects = np.asarray(
            [item[1][:2] for item in local_history[-int(args.contact_stagnation_window):]],
            dtype=np.float32,
        )
        robust_motion = float(np.linalg.norm(np.median(recent_effects, axis=0)))
        stagnant_contact = robust_motion <= float(args.contact_recovery_stagnation)
        if stagnant_contact:
            # This is an RGB-only recovery primitive.  It is used only after
            # several consecutive local finite differences fail to show
            # object motion; it does not query a simulator successor.
            torque_sign = 1.0 if angle_error >= 0.0 else -1.0
            fallback = direction + torque_sign * 0.75 * tangent
            fallback /= max(float(np.linalg.norm(fallback)), 1e-6)
            fallback *= min(float(args.contact_action_max_magnitude), 0.35)
            return fallback.astype(np.float32), "torque" if visual_xy <= float(args.terminal_xy) else "push", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": float(contact_lost),
                "angle_error_visual": angle_error,
                "block_xy_visual": visual_xy,
                "block_xy_terminal_observer": terminal_xy,
                "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
                "terminal_locked": float(terminal_locked),
                "contact_band": float(contact_band),
                "contact_stagnation_fallback": 1.0,
                "contact_stagnation_motion": robust_motion,
                "online_calibration_samples": float(local_sample_count),
                "online_calibrated": float(local_calibrated),
                "action_cost_ranker_used": 0.0,
            }
    # Keep candidate selection local to a contact manifold.  The score is
    # learned effect prediction; the small action regularizer breaks ties only.
    scores = scores + 0.04 * np.sum(np.square(actions), axis=1)
    trained_actions = action_bank(
        int(args.action_directions), declared_action_magnitudes(args),
    )
    in_distribution = np.asarray([
        bool(np.any(np.all(np.isclose(trained_actions, action[None], atol=1e-6), axis=1)))
        for action in actions
    ])
    ranker_used = False
    terminal_angular_ranker_used = False
    terminal_ranker_blend_used = False
    terminal_torque_prior_used = False
    if (
        (action_cost_ranker is not None and action_cost_ranker_training is not None)
        or (terminal_angular_ranker is not None and terminal_angular_ranker_training is not None)
    ):
        goal_tensor = tensor_images(resize_image(goal_image, args.image_size)[None], args.device)
        goal_features = model.encoder.features(goal_tensor)[0].cpu().numpy()
        if getattr(args, "ranker_interface", "rgb_geometry") == "rgb_geometry":
            ranker_current_angle = current_angle
            ranker_goal_pose = goal_pose_raw.copy()
            if (
                rgb_current_angle_decoder is not None
                and rgb_current_angle_training is not None
                and str(getattr(args, "rgb_angle_disagreement_policy", "none")) == "none"
            ):
                ranker_current_orientation = predict_orientation(
                    rgb_current_angle_decoder,
                    runtime_rgb_angle_features(
                        image, "block", rgb_current_angle_training, args.image_size,
                    ),
                    rgb_current_angle_training, args.device,
                )[0]
                ranker_current_angle = float(np.arctan2(
                    ranker_current_orientation[0], ranker_current_orientation[1],
                ))
            if rgb_goal_angle_decoder is not None and rgb_goal_angle_training is not None:
                ranker_goal_pose[2:4] = predict_orientation(
                    rgb_goal_angle_decoder,
                    runtime_rgb_angle_features(
                        goal_image, "target", rgb_goal_angle_training, args.image_size,
                    ),
                    rgb_goal_angle_training, args.device,
                )[0]
            ranker_features = rgb_geometry_ranker_features(
                np.repeat(resize_image(image, args.image_size)[None], len(actions), axis=0),
                np.repeat(resize_image(previous, args.image_size)[None], len(actions), axis=0),
                np.repeat(resize_image(goal_image, args.image_size)[None], len(actions), axis=0),
                actions,
                np.full(len(actions), ranker_current_angle, dtype=np.float32),
                np.tile(ranker_goal_pose, (len(actions), 1)),
                feature_mode="full",
            )
            terminal_ranker_features = ranker_features
            if str(getattr(args, "terminal_ranker_feature_mode", "full")) != "full":
                terminal_ranker_features = rgb_geometry_ranker_features(
                    np.repeat(resize_image(image, args.image_size)[None], len(actions), axis=0),
                    np.repeat(resize_image(previous, args.image_size)[None], len(actions), axis=0),
                    np.repeat(resize_image(goal_image, args.image_size)[None], len(actions), axis=0),
                    actions,
                    np.full(len(actions), ranker_current_angle, dtype=np.float32),
                    np.tile(ranker_goal_pose, (len(actions), 1)),
                    feature_mode=(
                        "invariant"
                        if str(args.terminal_ranker_feature_mode) == "invariant_adapter"
                        else str(args.terminal_ranker_feature_mode)
                    ),
                )
            if str(getattr(args, "terminal_ranker_feature_mode", "full")) == "invariant_adapter":
                terminal_ranker_features = np.concatenate([
                    terminal_ranker_features,
                    np.zeros((len(actions), 1), dtype=np.float32),
                ], axis=1)
        else:
            ranker_features = geometry_ranker_features(
                np.tile(features, (len(actions), 1)),
                np.tile(previous_features, (len(actions), 1)),
                np.tile(goal_features, (len(actions), 1)),
                actions,
                np.full(len(actions), current_angle, dtype=np.float32),
                np.tile(goal_pose_raw, (len(actions), 1)),
            )
            terminal_ranker_features = ranker_features
        ranker_scores = None
        if action_cost_ranker is not None and action_cost_ranker_training is not None:
            ranker_scores = predict_action_cost(
                action_cost_ranker, ranker_features, action_cost_ranker_training, args.device,
            )
        # A ranker is only calibrated on the declared absolute intervention
        # bank. Do not let it select goal-relative candidates on which it was
        # never trained.  The terminal small fan is eligible only when the
        # caller explicitly included those magnitudes in that bank; this
        # makes action-support coverage a recorded protocol choice rather
        # than a silent extrapolation.
        if ranker_scores is not None and np.any(in_distribution):
            ranker_scores = np.asarray(ranker_scores, dtype=np.float32)
            ranker_scores[~in_distribution] = np.max(ranker_scores[in_distribution]) + 1.0e6
        # Keep the learned cost model as the declared selector only when the
        # candidate set contains calibrated actions. The effect scores remain
        # available in the trace for an auditable paired ablation.
        if ranker_scores is not None and np.any(in_distribution):
            scores = ranker_scores
            ranker_used = True
        # A terminal pose is a different control problem from translation:
        # the relevant label is the signed one-step angular endpoint error,
        # not the coarse joint endpoint cost.  The model is promoted only by
        # an independent held-out gate in ``main``; this branch never falls
        # back to a learned angular predictor that failed that gate.
        if (
            terminal_control_active
            and candidate_contact_band
            and terminal_angular_ranker is not None
            and terminal_angular_ranker_training is not None
        ):
            # A serialized specialist is allowed into the controller only
            # when its objective is identical to the declared runtime
            # protocol.  This catches stale artifacts before they can turn a
            # pose-trained model into an angle-only action selector.
            validate_terminal_ranker_contract(
                terminal_angular_ranker_training, args,
            )
            angular_scores = predict_action_cost(
                terminal_angular_ranker,
                terminal_ranker_features
                if str(getattr(args, "terminal_ranker_feature_mode", "full")) != "invariant_adapter"
                else np.concatenate([
                    terminal_ranker_features[:, :-1],
                    np.asarray(terminal_baseline_scores, dtype=np.float32)[:, None],
                ], axis=1),
                terminal_angular_ranker_training, args.device,
            )
            if terminal_angular_ranker_training.get("score_mode") in {
                "direct_multistep_angle", "direct_multistep_pose",
            }:
                # Multi-step specialists are trained directly on the declared
                # endpoint objective after the repeated-action horizon;
                # adding the one-step adapter score would mix incompatible
                # objectives and recreate the residual-ranking failure mode.
                angular_scores = angular_scores.astype(np.float32)
            else:
                # The legacy one-step specialist predicts a correction to the
                # adapter's angular endpoint score.  Retaining the baseline
                # prevents a low-capacity residual model from erasing useful
                # action-effect structure.
                angular_scores = (
                    np.abs(wrap(predicted_block[:, 2] - goal_angle)).astype(np.float32)
                    + angular_scores
                )
            if np.any(in_distribution):
                angular_scores = np.asarray(angular_scores, dtype=np.float32)
                angular_scores[~in_distribution] = np.max(angular_scores[in_distribution]) + 1.0e6
                ranker_blend = float(np.clip(
                    getattr(args, "terminal_ranker_blend", 1.0), 0.0, 1.0,
                ))
                if ranker_blend < 1.0 and terminal_control_active:
                    def _root_standardize(values: np.ndarray) -> np.ndarray:
                        finite = np.isfinite(values)
                        if not bool(np.any(finite)):
                            return np.zeros_like(values, dtype=np.float32)
                        center = float(np.mean(values[finite]))
                        scale = max(float(np.std(values[finite])), 1.0e-6)
                        return ((values - center) / scale).astype(np.float32)
                    scores = (
                        ranker_blend * _root_standardize(angular_scores)
                        + (1.0 - ranker_blend) * _root_standardize(terminal_baseline_scores)
                    ).astype(np.float32)
                    terminal_ranker_blend_used = True
                else:
                    scores = angular_scores
                ranker_used = True
                terminal_angular_ranker_used = True
    if (
        terminal_control_active
        and candidate_contact_band
        and str(getattr(args, "_terminal_torque_prior_runtime_mode", "none")) != "none"
    ):
        # The prior is a local, observation-only mechanics chart: the visible
        # support lever arm determines the sign of r x F, while the observed
        # angle error determines which signed torque reduces the error.  It
        # is selected by a root-disjoint held-out gate in main; it is not a
        # simulator successor query and is not silently used when the gate
        # fails.
        torque = boundary_charts[:, 9].astype(np.float32)
        prior_weight = float(getattr(args, "terminal_torque_prior_weight", 1.0))
        prior_scores = (
            abs(float(angle_error))
            - prior_weight * float(wrap(goal_angle - current_angle)) * torque
        ).astype(np.float32)
        prior_runtime_mode = str(
            getattr(args, "_terminal_torque_prior_runtime_mode", "none")
        )
        if prior_runtime_mode in {"combined", "ranker_prior"}:
            baseline_scores = np.abs(
                wrap(predicted_block[:, 2] - goal_angle),
            ).astype(np.float32)
            if prior_runtime_mode == "ranker_prior":
                # Replay-only audit mode: preserve the learned terminal
                # ranker as the coarse candidate score and add the visible
                # signed-moment chart as a directional correction.
                baseline_scores = np.asarray(scores, dtype=np.float32).copy()
            def _root_normalize(values: np.ndarray) -> np.ndarray:
                spread = max(float(np.std(values)), 1.0e-6)
                return (values - float(np.min(values))) / spread
            scores = (
                _root_normalize(baseline_scores)
                + float(getattr(args, "terminal_torque_prior_blend", 0.25))
                * _root_normalize(prior_scores)
            ).astype(np.float32)
        else:
            scores = prior_scores
        terminal_torque_prior_used = True
    terminal_local_angle_used = False
    terminal_local_angle_weight = float(
        max(0.0, getattr(args, "terminal_local_angle_weight", 0.0))
    )
    if (
        terminal_control_active
        and local_angle_prediction is not None
        and terminal_local_angle_weight > 0.0
        and len(local_angle_prediction) == len(actions)
    ):
        # The fixed RGB lever chart supplies a sign prior, but the effective
        # angular gain depends on the current support point and contact mode.
        # Once the controller has observed its own recent RGB angle changes,
        # use the local signed angular effect as an additional score.  This is
        # causal online identification: it uses only past frames and never a
        # simulator successor or hidden state.
        desired_sign = 1.0 if wrap(goal_angle - current_angle) >= 0.0 else -1.0
        signed_local_gain = desired_sign * local_angle_prediction
        spread = max(float(np.std(signed_local_gain)), 1.0e-6)
        normalized_gain = (signed_local_gain - float(np.min(signed_local_gain))) / spread
        scores = np.asarray(scores, dtype=np.float32) - (
            terminal_local_angle_weight * normalized_gain.astype(np.float32)
        )
        terminal_local_angle_used = True
    if terminal_control_active and candidate_contact_band:
        # Re-apply the terminal tube constraint after specialist ranking.
        # The original ordering computed this penalty before the terminal
        # ranker, whose ``scores = angular_scores`` assignment erased it.
        # A pose ranker may improve angle while selecting an action whose
        # predicted XY endpoint ejects the block; that is precisely the
        # terminal-interface failure this guard is meant to prevent.
        xy_limit = max(float(args.terminal_xy), terminal_xy + 5.0)
        predicted_xy = np.linalg.norm(predicted_block[:, :2] - goal_xy[None], axis=1)
        violation = np.maximum(predicted_xy - xy_limit, 0.0)
        scores = np.asarray(scores, dtype=np.float32) + 10.0 * violation.astype(np.float32)
    if getattr(args, "learned_candidate_support", "trained") == "trained":
        # The effect adapter is an intervention model, not a continuous
        # action interpolator.  Relative fans and magnitudes absent from the
        # offline bank are extrapolations and were a direct source of contact
        # ejection in the previous controller.  Keep them available only for
        # the explicit geometry progress/recovery fallbacks.
        scores = scores.copy()
        scores[~in_distribution] = np.inf
    if (
        not terminal_locked
        and float(getattr(args, "goal_progress_score_weight", 0.0)) != 0.0
    ):
        # The effect adapter estimates local physical consequences; it does
        # not by itself guarantee that the selected action makes progress on
        # the current task.  This explicitly declared RGB task-direction
        # prior separates that interface failure from action-effect ranking.
        # It is applied after the learned/ranker score and before feasibility
        # guards, so its incremental value is auditable.
        scores = scores - float(args.goal_progress_score_weight) * (actions @ direction)
    mode_gate_used = False
    separation_probability = np.zeros(len(actions), dtype=np.float32)
    separation_risk = np.zeros(len(actions), dtype=np.float32)
    if (
        contact_mode_classifier is not None
        and contact_mode_training is not None
        and float(contact_mode_penalty) > 0.0
    ):
        mode_features = feature_rows
        separation_probability = predict_contact_mode(
            contact_mode_classifier, mode_features, contact_mode_training, args.device,
        )[:, 3]
        separation_risk = np.maximum(separation_probability - separation_probability[zero_index], 0.0)
        scores = scores + float(contact_mode_penalty) * separation_risk
        mode_gate_used = True
    if candidate_contact_band and float(args.contact_inward_min) > 0.0:
        # A learned one-step effect can rank a tangential or outward action
        # highly even when the RGB geometry says the pusher is on the contact
        # manifold.  Such an action is not a valid local control primitive:
        # it leaves the object before the next observation can be used.  The
        # filter is purely current-frame geometry and does not inspect a
        # simulator successor.
        contact_normal = block - agent
        contact_normal /= max(float(np.linalg.norm(contact_normal)), 1e-6)
        inward_component = actions @ contact_normal
        scores = scores.copy()
        scores[inward_component < float(args.contact_inward_min)] = 1.0e6
    torque_sign_guard_used = False
    if (
        bool(getattr(args, "torque_sign_guard", False))
        and terminal_control_active
        and candidate_contact_band
        and abs(angle_error) > float(args.terminal_angle)
    ):
        # The RGB normal/tangent chart provides a necessary local condition
        # for orientation repair: the tangential component must have the same
        # signed direction as the desired angle correction.  This is a
        # feasibility guard, not a simulator-success label or a successor
        # query.  It is applied after optional learned ranking so a learned
        # selector cannot choose an action with the wrong torque sign.
        # Use the rendered T boundary lever arm rather than the agent--block
        # tangent.  The latter is only a contact-frame direction and does not
        # determine torque when the visible support point is offset from the
        # centroid.  ``boundary_charts[..., 9]`` is the image-derived
        # lever-arm cross action in the same canvas convention as the RGB
        # shape readout.
        signed_torque = boundary_charts[:, 9] * (
            1.0 if wrap(goal_angle - current_angle) >= 0.0 else -1.0
        )
        scores = scores.copy()
        scores[signed_torque <= 0.0] = 1.0e6
        torque_sign_guard_used = True
    terminal_contact_safety_used = False
    terminal_contact_safety_fallback = False
    terminal_contact_inward = np.full(len(actions), np.nan, dtype=np.float32)
    terminal_contact_tangent = np.full(len(actions), np.nan, dtype=np.float32)
    terminal_contact_inward_min = float(
        max(0.0, getattr(args, "terminal_contact_inward_min", 0.0))
    )
    if (
        terminal_control_active
        and candidate_contact_band
        and abs(angle_error) > float(args.terminal_angle)
        and terminal_contact_inward_min > 0.0
    ):
        # Torque-sign compatibility is necessary but not sufficient for a
        # valid terminal intervention.  A signed tangential action can have
        # the correct r x F sign while moving the pusher away from the
        # contact manifold.  Enforce the independent RGB half-space constraint
        # before the final argmin, so the learned score cannot reintroduce
        # contact ejection after the torque-sign guard has passed.
        contact_normal = np.asarray(block - agent, dtype=np.float32)
        contact_normal /= max(float(np.linalg.norm(contact_normal)), 1.0e-6)
        contact_tangent = np.asarray([-contact_normal[1], contact_normal[0]], dtype=np.float32)
        terminal_contact_inward = (actions @ contact_normal).astype(np.float32)
        terminal_contact_tangent = np.abs(actions @ contact_tangent).astype(np.float32)
        feasible = in_distribution & (terminal_contact_inward >= terminal_contact_inward_min)
        tangent_ratio = float(getattr(args, "terminal_contact_tangent_ratio", 0.0))
        if tangent_ratio > 0.0:
            feasible &= terminal_contact_tangent <= tangent_ratio * np.maximum(
                terminal_contact_inward, terminal_contact_inward_min,
            )
        terminal_contact_safety_used = True
        scores = scores.copy()
        scores[~feasible] = np.inf
        if not bool(np.any(feasible)):
            # Keep the gate fail-closed with respect to unsupported actions,
            # but avoid an all-infinite argmin selecting zero or an arbitrary
            # extrapolative fan element.  The least unsafe calibrated action
            # maximizes inward support and then minimizes the already computed
            # score; this branch is explicitly auditable in the trace.
            fallback = in_distribution & (np.linalg.norm(actions, axis=1) > 1.0e-6)
            if bool(np.any(fallback)):
                fallback_scores = np.where(
                    fallback,
                    -terminal_contact_inward + 1.0e-3 * np.nan_to_num(scores, nan=0.0),
                    np.inf,
                )
                selected_fallback = int(np.argmin(fallback_scores))
                scores[:] = np.inf
                scores[selected_fallback] = fallback_scores[selected_fallback]
                terminal_contact_safety_fallback = True
    contact_translation_safety_used = False
    contact_translation_safety_fallback = False
    if (
        candidate_contact_band
        and not terminal_locked
        and float(getattr(args, "contact_translation_inward_min", 0.0)) > 0.0
    ):
        # Before the terminal manifold, a tangential action is useful only if
        # it still contains enough inward motion to keep the pusher against
        # the block.  The learned one-step effect model does not observe the
        # future contact normal, so it can otherwise repeat a locally good
        # torque direction until the pusher separates.  This constraint is
        # RGB-only and is applied to the declared intervention support.
        contact_normal = block - agent
        contact_normal /= max(float(np.linalg.norm(contact_normal)), 1e-6)
        inward_component = actions @ contact_normal
        tangent_component = np.abs(actions @ tangent)
        progress_component = actions @ direction
        inward_min = float(args.contact_translation_inward_min)
        tangent_ratio = float(max(args.contact_translation_tangent_ratio, 0.0))
        progress_min = float(getattr(args, "contact_translation_progress_min", 0.0))
        safe = (
            in_distribution
            & (inward_component >= inward_min)
            & (tangent_component <= tangent_ratio * np.maximum(inward_component, 1e-6))
            & (progress_component >= progress_min)
        )
        scores = scores.copy()
        scores[~safe] = np.inf
        contact_translation_safety_used = True
        if not bool(np.any(safe)):
            # Never let an all-invalid mask fall through to argmin, whose
            # deterministic first element is often the zero action.  Select
            # the most inward in-distribution nonzero action as a conservative
            # contact-preserving recovery primitive.
            fallback = in_distribution & (np.linalg.norm(actions, axis=1) > 1e-6)
            if bool(np.any(fallback)):
                fallback_score = (
                    -inward_component
                    - 0.50 * progress_component
                    + 0.15 * tangent_component
                )
                fallback_score[~fallback] = np.inf
                selected_fallback = int(np.argmin(fallback_score))
                contact_translation_safety_fallback = True
                return actions[selected_fallback], "translate", {
                    "approach_distance": approach_distance,
                    "observed_agent_block_distance": observed_agent_block_distance,
                    "observed_block_motion": observed_block_motion,
                    "contact_lost_recovery": float(contact_lost),
                    "approach_handoff": float(approach_handoff),
                    "angle_error_visual": angle_error,
                    "block_xy_visual": visual_xy,
                    "block_xy_terminal_observer": terminal_xy,
                    "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
                    "terminal_locked": float(terminal_locked),
                    "contact_band": float(contact_band),
                    "candidate_contact_band": float(candidate_contact_band),
                    "contact_translation_safety_used": 1.0,
                    "contact_translation_safety_fallback": 1.0,
                    "contact_inward_component": float(inward_component[selected_fallback]),
                    "contact_progress_component": float(progress_component[selected_fallback]),
                    "predicted_best_cost": float(fallback_score[selected_fallback]),
                    "predicted_effect_selected": np.asarray(
                        predicted[selected_fallback, POSE_DIM:], dtype=np.float32,
                    ).tolist(),
                    "predicted_next_block_selected": np.asarray(
                        predicted_block[selected_fallback], dtype=np.float32,
                    ).tolist(),
                    "online_calibration_samples": float(local_sample_count),
                    "online_calibrated": float(local_calibrated),
                    "action_cost_ranker_used": float(ranker_used),
                    "contact_mode_gate_used": float(mode_gate_used),
                    "contact_canonical_knn_used": float(contact_knn_used),
                }
    zero_action_guard_used = False
    zero_guard_after_reacquisition = int(reacquisition_progress_count) > 0
    if (
        (bool(getattr(args, "forbid_zero_before_terminal", False)) or zero_guard_after_reacquisition)
        and (
            visual_xy > float(args.terminal_hold_xy)
            or (
                terminal_control_active
                and angle_error > float(args.terminal_angle)
            )
        )
    ):
        # Zero is a valid terminal hold, but it is not a progress action. A
        # learned cost model that systematically underestimates action effect
        # otherwise turns its passive baseline into a deadlock before the
        # terminal manifold is reached.
        scores = scores.copy()
        scores[zero_index] = 1.0e6
        zero_action_guard_used = True

    # A learned action selector is only valid on its declared intervention
    # support. With a terminal cap below the smallest trained magnitude, the
    # support mask can contain zero but no finite non-zero action. The old
    # ``scores[zero_index] = 1e6`` guard then still won against ``inf`` and
    # silently produced a no-op. Fail closed with an explicit, auditable RGB
    # geometry action instead of treating zero as a calibrated prediction.
    terminal_support_count = int(np.sum(in_distribution & (
        np.linalg.norm(actions, axis=1) > 1.0e-6
    )))
    terminal_finite_nonzero_support_count = int(np.sum(
        in_distribution
        & (np.linalg.norm(actions, axis=1) > 1.0e-6)
        & np.isfinite(scores)
    ))
    if (
        terminal_control_active
        and candidate_contact_band
        and getattr(args, "learned_candidate_support", "trained") == "trained"
        and terminal_finite_nonzero_support_count == 0
    ):
        support_lever = np.asarray(zero_boundary_chart[2:4], dtype=np.float32)
        fallback, fallback_audit = terminal_support_fallback_action(
            agent,
            block,
            current_angle,
            goal_angle,
            candidate_magnitude_cap,
            float(getattr(args, "terminal_contact_inward_min", 0.0)),
            support_lever=support_lever,
        )
        return fallback, "torque", {
            "approach_distance": approach_distance,
            "observed_agent_block_distance": observed_agent_block_distance,
            "observed_block_motion": observed_block_motion,
            "contact_lost_recovery": float(contact_lost),
            "angle_error_visual": angle_error,
            "block_xy_visual": visual_xy,
            "block_xy_terminal_observer": terminal_xy,
            "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
            "terminal_locked": float(terminal_locked),
            "terminal_control_active": 1.0,
            "terminal_support_fallback_used": 1.0,
            "terminal_support_fallback_reason": "no_finite_nonzero_in_distribution_action",
            "terminal_support_count": float(terminal_support_count),
            "terminal_finite_nonzero_support_count": float(terminal_finite_nonzero_support_count),
            "terminal_support_cap": float(candidate_magnitude_cap),
            "terminal_contact_gap_px": terminal_contact_gap_px,
            "terminal_contact_lease_valid": float(terminal_contact_lease_valid),
            "terminal_boundary_support_visible": float(runtime_boundary_support_visible),
            "predicted_best_cost": float("nan"),
            "online_calibration_samples": float(local_sample_count),
            "online_calibrated": float(local_calibrated),
            "action_cost_ranker_used": 0.0,
            **fallback_audit,
        }
    selected = int(np.argmin(scores))
    progress_guard_used = False
    if bool(getattr(args, "learned_progress_guard", False)) and terminal_xy > float(args.terminal_hold_xy):
        previous_visual_xy = float("nan")
        if previous_action is not None and phase not in {"approach", "recovery"}:
            previous_visual_xy = float(np.linalg.norm(goal_xy - previous_block))
        selected_is_stalled = selected == zero_index
        # RGB centroid noise is routinely sub-pixel to a few pixels after
        # resizing and mask thresholding.  Treating a 0.28 px improvement as
        # regression made this guard replace the learned candidate with a
        # large 0.75-magnitude tangential action, which was itself a source of
        # contact ejection.  Require a declared, materially observable
        # regression before overriding the learned selector.
        previous_step_regressed = (
            np.isfinite(previous_visual_xy)
            and visual_xy >= previous_visual_xy + 3.0
        )
        if selected_is_stalled or previous_step_regressed:
            # Use the same image-derived normal/tangent geometry as the
            # transparent baseline.  This is a fail-safe for a learned
            # selector, not a simulator-success oracle: it only reacts to
            # observed RGB progress and preserves the learned path whenever
            # the previous step made measurable progress.
            torque_sign = 1.0 if wrap(goal_angle - current_angle) >= 0.0 else -1.0
            fallback = direction + torque_sign * 0.75 * tangent
            fallback /= max(float(np.linalg.norm(fallback)), 1e-6)
            fallback_magnitude = float(args.progress_guard_magnitude)
            if candidate_contact_band:
                fallback_magnitude = min(
                    fallback_magnitude, float(args.contact_action_max_magnitude),
                )
            fallback *= fallback_magnitude
            progress_guard_used = True
            return fallback.astype(np.float32), "torque" if terminal_locked else "push", {
                "approach_distance": approach_distance,
                "observed_agent_block_distance": observed_agent_block_distance,
                "observed_block_motion": observed_block_motion,
                "contact_lost_recovery": float(contact_lost),
                "approach_handoff": float(approach_handoff),
                "angle_error_visual": angle_error,
                "block_xy_visual": visual_xy,
                "block_xy_terminal_observer": terminal_xy,
                "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
                "terminal_locked": float(terminal_locked),
                "contact_band": float(contact_band),
                "contact_lock_active": float(contact_lock_active),
                "candidate_contact_band": float(candidate_contact_band),
                "learned_progress_guard_used": 1.0,
                "learned_progress_previous_xy": previous_visual_xy,
                "learned_progress_selected_stalled": float(selected_is_stalled),
                "predicted_best_cost": float(scores[selected]),
                "predicted_effect_selected": np.asarray(
                    predicted[selected, POSE_DIM:], dtype=np.float32,
                ).tolist(),
                "predicted_next_block_selected": np.asarray(
                    predicted_block[selected], dtype=np.float32,
                ).tolist(),
                "online_calibration_samples": float(local_sample_count),
                "online_calibrated": float(local_calibrated),
                "action_cost_ranker_used": float(ranker_used),
                "contact_mode_gate_used": float(mode_gate_used),
                "contact_canonical_knn_used": float(contact_knn_used),
            }
    if angle_error <= float(args.terminal_angle) and terminal_xy > args.terminal_xy:
        phase = "translate"
    elif terminal_control_active:
        # A terminal XY observation alone does not grant a torque lease.  The
        # reacquisition trace can legitimately return a near-goal block while
        # the pusher is still far away; promoting that frame to ``torque``
        # makes hybrid planning bypass its geometry translation/recovery
        # interface and can expose an unsupported 0.85 action.  Require the
        # independently observed contact band before entering the terminal
        # state.  This is an RGB state-machine invariant, not a successor
        # query or a simulator-state condition.
        phase = "torque"
    elif terminal_xy <= args.terminal_xy:
        # The visual block is near the goal, but the RGB support-point chart
        # says that the pusher is not on a valid terminal contact manifold.
        # Keep the next decision in translation/recovery rather than handing
        # an invalid terminal lease to the torque state.
        phase = "translate"
    return actions[selected], phase, {
        "approach_distance": approach_distance,
        "observed_agent_block_distance": observed_agent_block_distance,
        "observed_block_motion": observed_block_motion,
        "contact_side_alignment": contact_side_alignment,
        "contact_lost_recovery": float(contact_lost),
        "approach_handoff": float(approach_handoff),
        "angle_error_visual": angle_error,
        "block_xy_visual": float(np.linalg.norm(goal_xy - block)),
        "block_xy_terminal_observer": terminal_xy,
        "terminal_xy_observer": getattr(args, "terminal_xy_observer", "runtime"),
        "terminal_locked": float(terminal_locked),
        "terminal_control_active": float(terminal_control_active),
        "terminal_contact_gap_px": terminal_contact_gap_px,
        "terminal_contact_boundary_gap_px": runtime_contact_boundary_gap_px,
        "terminal_contact_lease_valid": float(terminal_contact_lease_valid),
        "terminal_contact_proximity_fallback": float(terminal_proximity_fallback),
        "terminal_contact_lease_gap_max": lease_gap_max,
        "terminal_boundary_support_visible": float(runtime_boundary_support_visible),
        "terminal_contact_gap_gate_used": float(terminal_contact_gap_gate_used),
        "terminal_agent_distance_px": observed_agent_block_distance,
        "terminal_agent_distance_gate_used": float(terminal_agent_distance_gate_used),
        "terminal_agent_distance_max": float(getattr(args, "terminal_agent_distance_max", 0.0)),
        "contact_band": float(contact_band),
        "contact_lock_active": float(contact_lock_active),
        "candidate_contact_band": float(candidate_contact_band),
        "contact_action_cap": float(args.contact_action_max_magnitude if candidate_contact_band else max(broad_magnitudes)),
        "contact_inward_component": float(np.dot(actions[selected], (block - agent) / max(float(np.linalg.norm(block - agent)), 1e-6))),
        "contact_inward_filter": float(contact_band and float(args.contact_inward_min) > 0.0),
        "contact_translation_safety_used": float(contact_translation_safety_used),
        "contact_translation_safety_fallback": float(contact_translation_safety_fallback),
        "contact_progress_component": float(np.dot(actions[selected], direction)),
        "torque_sign_guard_used": float(torque_sign_guard_used),
        "terminal_contact_safety_used": float(terminal_contact_safety_used),
        "terminal_contact_safety_fallback": float(terminal_contact_safety_fallback),
        "terminal_contact_inward_min": float(terminal_contact_inward_min),
        "terminal_contact_inward_selected": float(terminal_contact_inward[selected]),
        "terminal_contact_tangent_selected": float(terminal_contact_tangent[selected]),
        "terminal_support_fallback_used": 0.0,
        "terminal_support_count": float(terminal_support_count),
        "terminal_finite_nonzero_support_count": float(terminal_finite_nonzero_support_count),
        "terminal_angular_ranker_used": float(terminal_angular_ranker_used),
        "terminal_ranker_blend_used": float(terminal_ranker_blend_used),
        "terminal_torque_prior_used": float(terminal_torque_prior_used),
        "terminal_local_angle_used": float(terminal_local_angle_used),
        "terminal_local_angle_weight": float(terminal_local_angle_weight),
        "zero_action_guard_used": float(zero_action_guard_used),
        "zero_action_guard_after_reacquisition": float(zero_guard_after_reacquisition),
        "learned_progress_guard_used": float(progress_guard_used),
        "contact_stagnation_fallback": float(stagnant_contact),
        "predicted_best_cost": float(scores[selected]),
        "online_calibration_samples": float(local_sample_count),
        "online_calibrated": float(local_calibrated),
        "action_cost_ranker_used": float(ranker_used),
        "contact_mode_gate_used": float(mode_gate_used),
        "contact_canonical_knn_used": float(contact_knn_used),
        "predicted_separation_probability": float(separation_probability[selected]),
        "predicted_separation_risk": float(separation_risk[selected]),
    }


def geometry_baseline_action(
    model: VisualCausalWorldModel, normalizer: Normalization, image: np.ndarray,
    previous: np.ndarray, goal_image: np.ndarray, args: argparse.Namespace, phase: str,
    goal_decoder: GoalPoseDecoder | None = None, goal_decoder_training: dict[str, Any] | None = None,
    orientation_decoder: OrientationDecoder | None = None,
    orientation_decoder_training: dict[str, Any] | None = None,
    goal_component_selection: dict[str, bool] | None = None,
    goal_xy_decoder: OrientationDecoder | None = None,
    goal_xy_decoder_training: dict[str, Any] | None = None,
    goal_angle_decoder: OrientationDecoder | None = None,
    goal_angle_decoder_training: dict[str, Any] | None = None,
    xy_decoder: OrientationDecoder | None = None,
    xy_decoder_training: dict[str, Any] | None = None,
    rgb_current_angle_decoder: OrientationDecoder | None = None,
    rgb_current_angle_training: dict[str, Any] | None = None,
    rgb_goal_angle_decoder: OrientationDecoder | None = None,
    rgb_goal_angle_training: dict[str, Any] | None = None,
    rgb_goal_xy_decoder: OrientationDecoder | None = None,
    rgb_goal_xy_training: dict[str, Any] | None = None,
    contact_side_misalignment_count: int = 0,
    agent_xy_decoder: OrientationDecoder | None = None,
    agent_xy_decoder_training: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str]:
    visual, _features, _previous, goal_pose = runtime_observation(
        model, normalizer, image, previous, goal_image, args.image_size, args.device,
        goal_decoder, goal_decoder_training,
        orientation_decoder, orientation_decoder_training,
        goal_component_selection,
        goal_xy_decoder, goal_xy_decoder_training,
        goal_angle_decoder, goal_angle_decoder_training,
        xy_decoder, xy_decoder_training,
        current_xy_policy=runtime_current_xy_policy(args),
        current_xy_disagreement_threshold=getattr(args, "current_xy_disagreement_threshold", 60.0),
        paired_reference_policy=getattr(args, "paired_reference_policy", "raw"),
        current_angle_policy=getattr(args, "current_angle_policy", "model"),
        rgb_current_angle_decoder=(
            rgb_current_angle_decoder
            if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
        ),
        rgb_current_angle_training=(
            rgb_current_angle_training
            if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
        ),
        rgb_goal_angle_decoder=rgb_goal_angle_decoder,
        rgb_goal_angle_training=rgb_goal_angle_training,
        rgb_goal_xy_decoder=rgb_goal_xy_decoder,
        rgb_goal_xy_training=rgb_goal_xy_training,
        rgb_angle_temporal_jump=getattr(args, "rgb_angle_temporal_jump", 0.75),
        rgb_angle_disagreement_policy=getattr(args, "rgb_angle_disagreement_policy", "none"),
        rgb_angle_disagreement_threshold=getattr(args, "rgb_angle_disagreement_threshold", 1.20),
        rgb_angle_disagreement_terminal_xy=getattr(args, "rgb_angle_disagreement_terminal_xy", -1.0),
        rgb_temporal_angle_state=getattr(args, "_runtime_temporal_angle_state", None),
        rgb_temporal_angle_observer=bool(getattr(args, "_use_rgb_temporal_angle_runtime", False)),
        rgb_temporal_angle_max_delta=getattr(args, "rgb_temporal_angle_max_delta", 0.55),
        rgb_temporal_angle_min_confidence=getattr(args, "rgb_temporal_angle_min_confidence", 0.004),
        max_observed_block_step=getattr(args, "rgb_max_observed_block_step", 48.0),
        held_block_xy=getattr(args, "_runtime_last_block_xy", None),
        held_agent_xy=getattr(args, "_runtime_last_agent_xy", None),
        agent_xy_decoder=agent_xy_decoder,
        agent_xy_decoder_training=agent_xy_decoder_training,
    )
    if np.isfinite(visual[2:4]).all():
        args._runtime_last_block_xy = np.asarray(visual[2:4], dtype=np.float32).copy()
    agent, block = visual[:2], visual[2:4]
    robust_runtime_block = update_runtime_block_history(args, block)
    direction = goal_pose[:2] - block
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    contact_normal = block - agent
    contact_normal /= max(float(np.linalg.norm(contact_normal)), 1e-6)
    contact_side_alignment = float(np.dot(contact_normal, direction))
    desired = block - args.contact_distance * direction
    approach = desired - agent
    observed_agent_block_distance = float(np.linalg.norm(agent - block))
    if runtime_current_xy_policy(args) == "paired_rgb_calibrated":
        current_motion_block, _ = rgb_geometry_xy_angle_from_image(image, "block")
        previous_motion_block, _ = rgb_geometry_xy_angle_from_image(previous, "block")
        if current_motion_block is not None and previous_motion_block is not None:
            motion_delta = np.asarray(current_motion_block, dtype=np.float32) - np.asarray(
                previous_motion_block, dtype=np.float32,
            )
            observed_block_motion = float(np.linalg.norm(motion_delta))
            if (
                float(getattr(args, "rgb_max_observed_block_step", 48.0)) > 0.0
                and observed_block_motion > float(getattr(args, "rgb_max_observed_block_step", 48.0))
            ):
                observed_block_motion = 0.0
            previous_block = block - motion_delta if observed_block_motion > 0.0 else block.copy()
        else:
            previous_block = block.copy()
            observed_block_motion = 0.0
    else:
        previous_block = observed_block_xy(
            previous, _previous, xy_decoder, xy_decoder_training, args.device,
            runtime_current_xy_policy(args),
            getattr(args, "current_xy_disagreement_threshold", 60.0),
        )
        observed_block_motion = float(np.linalg.norm(block - previous_block))
    observed_block_delta = block - previous_block
    motion_direction_alignment = float(
        np.dot(observed_block_delta, direction)
        / max(float(np.linalg.norm(observed_block_delta)), 1.0e-6)
    ) if observed_block_motion > 1.0e-6 else 0.0
    side_alignment_threshold = float(getattr(args, "contact_side_alignment_min", 0.35))
    side_misalignment_patience = max(
        1, int(getattr(args, "contact_side_misalignment_patience", 1))
    )
    goal_progress = float(
        np.linalg.norm(goal_pose[:2] - previous_block)
        - np.linalg.norm(goal_pose[:2] - block)
    )
    near_terminal_tube = float(np.linalg.norm(goal_pose[:2] - block)) <= float(
        getattr(args, "terminal_xy", 35.0)
    )
    progressing_contact = bool(
        observed_block_motion > float(args.contact_recovery_stagnation)
        and goal_progress > 0.0
        and not near_terminal_tube
        and (
            contact_side_alignment >= side_alignment_threshold
            or motion_direction_alignment >= -0.15
        )
    )
    side_misaligned = (
        contact_side_alignment < side_alignment_threshold
        and int(contact_side_misalignment_count) + 1 >= side_misalignment_patience
        and not progressing_contact
        and not near_terminal_tube
    )
    # Keep the geometry baseline's terminal lease definition aligned with the
    # learned controller: a near-goal block is not a torque-contact state when
    # the pusher is outside the broad RGB handoff band.
    candidate_contact_band = bool(
        observed_agent_block_distance
        <= float(args.contact_distance) + 2.0 * float(args.contact_tolerance)
    )
    contact_lost = (
        phase not in {
            "approach", "recovery", "reacquire_clear", "reacquire",
            "reacquire_waypoint", "reacquire_arc", "reacquire_outer", "reacquire_target",
        }
        and observed_agent_block_distance > float(args.contact_recovery_distance)
        and observed_block_motion <= float(args.contact_recovery_stagnation)
    )
    if contact_lost:
        phase = "recovery"
    recovery_handoff_distance = float(args.contact_distance) + 1.25 * float(args.contact_tolerance)
    reacquisition_clearance = max(
        recovery_handoff_distance,
        float(getattr(args, "reacquisition_clearance", args.contact_recovery_distance)),
    )
    side_reacquisition_enabled = bool(
        getattr(args, "terminal_side_reacquisition", False)
    )
    if phase == "reacquire_clear":
        if observed_agent_block_distance < reacquisition_clearance:
            return np.clip(
                -reacquisition_retreat_magnitude(args) * contact_normal,
                -1.0, 1.0,
            ).astype(np.float32), "reacquire_clear"
        phase = "reacquire_waypoint"
    if phase in {"reacquire", "reacquire_waypoint"}:
        target_radial = -direction
        perpendicular = np.asarray([-target_radial[1], target_radial[0]], dtype=np.float32)
        waypoint_radius = reacquisition_clearance
        waypoint_candidates = np.asarray([
            block + waypoint_radius * perpendicular,
            block - waypoint_radius * perpendicular,
        ], dtype=np.float32)
        waypoint = waypoint_candidates[int(np.argmin(np.linalg.norm(
            waypoint_candidates - agent[None], axis=1,
        )))]
        waypoint_distance = float(np.linalg.norm(waypoint - agent))
        if waypoint_distance > float(args.contact_tolerance):
            action = np.clip(
                (waypoint - agent) / float(args.approach_action_scale), -1.0, 1.0,
            ).astype(np.float32)
            action = np.clip(
                action,
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            )
            return action, "reacquire_waypoint"
        phase = "reacquire_arc"
    if phase == "reacquire_arc":
        outer_target = block + reacquisition_clearance * (-direction)
        route_normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        route_sign = 1.0 if float(np.dot(agent - block, route_normal)) >= 0.0 else -1.0
        arc_midpoint = block - (reacquisition_clearance / np.sqrt(2.0)) * direction + (
            route_sign * reacquisition_clearance / np.sqrt(2.0)
        ) * route_normal
        midpoint_distance = float(np.linalg.norm(arc_midpoint - agent))
        if midpoint_distance > float(args.contact_tolerance):
            action = np.clip(
                (arc_midpoint - agent) / float(args.approach_action_scale), -1.0, 1.0,
            ).astype(np.float32)
            action = np.clip(
                action,
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            )
            return action, "reacquire_arc"
        phase = "reacquire_outer"
    if phase == "reacquire_outer":
        outer_target = block + reacquisition_clearance * (-direction)
        outer_distance = float(np.linalg.norm(outer_target - agent))
        if outer_distance > float(args.contact_tolerance):
            action = np.clip(
                (outer_target - agent) / float(args.approach_action_scale), -1.0, 1.0,
            ).astype(np.float32)
            action = np.clip(
                action,
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
                float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            )
            return action, "reacquire_outer"
        phase = "reacquire_target"
    if phase == "reacquire_target":
        if (
            observed_block_motion > float(getattr(args, "reacquisition_motion_limit", 3.0))
            and contact_side_alignment < side_alignment_threshold
        ):
            return np.clip(
                -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)) * contact_normal,
                -1.0, 1.0,
            ).astype(np.float32), "reacquire_clear"
        reacquired = bool(
            observed_agent_block_distance <= recovery_handoff_distance
            and np.linalg.norm(approach) <= float(args.contact_tolerance)
            and contact_side_alignment >= side_alignment_threshold
        )
        if reacquired:
            return np.zeros(2, dtype=np.float32), "push"
        action = np.clip(
            approach / float(args.approach_action_scale), -1.0, 1.0,
        ).astype(np.float32)
        action = np.clip(
            action,
            -float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
            float(getattr(args, "reacquisition_action_max_magnitude", 0.35)),
        )
        return action, "reacquire_target"
    if phase == "recovery" and observed_agent_block_distance > recovery_handoff_distance:
        recovery_action, _recovery_audit = bounded_recovery_action(
            approach / float(args.approach_action_scale), args,
        )
        return recovery_action, "recovery"
    if (
        side_reacquisition_enabled
        and
        phase == "recovery"
        and side_misaligned
    ):
        return side_route_action(
            contact_normal, direction,
            reacquisition_retreat_magnitude(args),
        ), "recovery"
    if phase == "recovery" and side_misaligned:
        if getattr(args, "side_recovery_mode", "contact_target") == "contact_target":
            recovery_action, _recovery_audit = bounded_recovery_action(
                approach / float(args.approach_action_scale), args,
            )
            if float(getattr(args, "recovery_action_max_magnitude", 0.0)) <= 0.0:
                recovery_action = np.clip(
                    recovery_action,
                    -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
                    float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
                ).astype(np.float32)
            return recovery_action, "recovery"
        recovery_action, _recovery_audit = bounded_recovery_action(
            -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)) * contact_normal,
            args,
        )
        return recovery_action, "recovery"
    # Do not hand off merely because the pusher is close to the object.  A
    # stationary block is evidence that the reconstructed contact side is
    # still wrong; continue the RGB approach servo until the contact point
    # converges or motion confirms a valid push.
    if (
        phase == "approach"
        and observed_agent_block_distance > recovery_handoff_distance
    ):
        return np.clip(
            approach / float(args.approach_action_scale), -1.0, 1.0,
        ).astype(np.float32), "approach"
    if (
        phase == "approach"
        and np.linalg.norm(approach) > float(args.contact_tolerance)
        and observed_block_motion <= float(args.contact_recovery_stagnation)
    ):
        return np.clip(approach / float(args.approach_action_scale), -1.0, 1.0).astype(np.float32), "approach"
    if (
        phase == "approach"
        and side_misaligned
    ):
        if getattr(args, "side_recovery_mode", "contact_target") == "contact_target":
            return np.clip(
                approach / float(args.approach_action_scale),
                -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
                float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
            ).astype(np.float32), "approach"
        return np.clip(
            -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)) * contact_normal,
            -1.0, 1.0,
        ).astype(np.float32), "approach"
    if (
        phase not in {"approach", "recovery"}
        and phase not in REACQUISITION_PHASES
        and side_misaligned
    ):
        if side_reacquisition_enabled:
            return side_route_action(
                contact_normal, direction,
                reacquisition_retreat_magnitude(args),
            ), "recovery"
        if getattr(args, "side_recovery_mode", "contact_target") == "contact_target":
            recovery_action, _recovery_audit = bounded_recovery_action(
                approach / float(args.approach_action_scale), args,
            )
            if float(getattr(args, "recovery_action_max_magnitude", 0.0)) <= 0.0:
                recovery_action = np.clip(
                    recovery_action,
                    -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
                    float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)),
                ).astype(np.float32)
            return recovery_action, "recovery"
        recovery_action, _recovery_audit = bounded_recovery_action(
            -float(getattr(args, "contact_recovery_retreat_magnitude", 0.45)) * contact_normal,
            args,
        )
        return recovery_action, "recovery"
    if phase in {"approach", "recovery"}:
        phase = "push"
    angle_error = float(wrap(np.arctan2(goal_pose[2], goal_pose[3]) - np.arctan2(visual[4], visual[5])))
    tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    visual_xy = float(np.linalg.norm(goal_pose[:2] - block))
    tangent_gain = float(getattr(args, "geometry_translation_tangent", 0.0))
    force_translation = bool(getattr(args, "_force_geometry_translation", False))
    if force_translation:
        # This branch is only enabled by the explicit geometric-torque drift
        # ablation.  It is a bounded task-space repair, not a hidden state or
        # successor query: the direction comes from the current RGB goal/block
        # chart and the magnitude is fixed before evaluation.
        push = direction
        mode = "translate"
        translation_magnitude = float(
            np.clip(
                getattr(args, "geometric_torque_drift_translation_magnitude", 0.35),
                0.0,
                1.0,
            )
        )
    elif visual_xy > float(getattr(args, "terminal_xy", 35.0)):
        # Translation and orientation are intentionally decoupled.  Applying
        # a large tangent component while far from the target creates the
        # same coupled endpoint failure as the original joint objective: the
        # pusher rotates/ejects the block before translation is solved.
        push = direction + tangent_gain * tangent
        mode = "translate"
    else:
        push = direction + math.copysign(0.75, angle_error if abs(angle_error) > 1e-6 else 1.0) * tangent
        # Geometry must obey the same terminal-contact lease as the learned
        # branch.  Near-goal XY with a distant pusher is not a torque state:
        # labeling it ``torque`` makes hybrid planning preserve a terminal
        # phase after reacquisition has actually lost contact.
        mode = "torque" if (
            abs(angle_error) > 0.28 and candidate_contact_band
        ) else "translate"
    push /= max(float(np.linalg.norm(push)), 1e-6)
    if not force_translation:
        translation_magnitude = 0.75
    return np.clip(translation_magnitude * push, -1.0, 1.0).astype(np.float32), mode


def run_planning(
    env: gym.Env, model: VisualCausalWorldModel, normalizer: Normalization,
    adapter: LocalEffectAdapter, adapter_training: dict[str, Any], args: argparse.Namespace,
    controller: str, goal_decoder: GoalPoseDecoder | None = None,
    goal_decoder_training: dict[str, Any] | None = None,
    orientation_decoder: OrientationDecoder | None = None,
    orientation_decoder_training: dict[str, Any] | None = None,
    goal_component_selection: dict[str, bool] | None = None,
    goal_xy_decoder: OrientationDecoder | None = None,
    goal_xy_decoder_training: dict[str, Any] | None = None,
    goal_angle_decoder: OrientationDecoder | None = None,
    goal_angle_decoder_training: dict[str, Any] | None = None,
    xy_decoder: OrientationDecoder | None = None,
    xy_decoder_training: dict[str, Any] | None = None,
    action_cost_ranker: ActionCostRanker | None = None,
    action_cost_ranker_training: dict[str, Any] | None = None,
    terminal_angular_ranker: ActionCostRanker | None = None,
    terminal_angular_ranker_training: dict[str, Any] | None = None,
    contact_mode_classifier: ContactModeClassifier | None = None,
    contact_mode_training: dict[str, Any] | None = None,
    contact_mode_penalty: float = 0.0,
    contact_knn_training: dict[str, Any] | None = None,
    rgb_current_angle_decoder: OrientationDecoder | None = None,
    rgb_current_angle_training: dict[str, Any] | None = None,
    rgb_goal_angle_decoder: OrientationDecoder | None = None,
    rgb_goal_angle_training: dict[str, Any] | None = None,
    rgb_goal_xy_decoder: OrientationDecoder | None = None,
    rgb_goal_xy_training: dict[str, Any] | None = None,
    agent_xy_decoder: OrientationDecoder | None = None,
    agent_xy_decoder_training: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    terminal_planning_roots = None
    if getattr(args, "planning_start", "random") == "terminal":
        terminal_planning_roots = terminal_pose_cases(
            env, args.planning_episodes, args.seed + 700_000,
            position_jitter=float(args.terminal_position_jitter),
            support_gap_max=float(getattr(args, "terminal_root_support_gap", 0.0)),
            support_angle_samples=int(getattr(args, "terminal_root_support_angle_samples", 32)),
        )
    for episode in range(args.planning_episodes):
        if terminal_planning_roots is None:
            observation, info = env.reset(seed=args.seed + 100_000 + episode)
            goal = np.asarray(info["goal_state"], dtype=np.float32).copy()
            previous = env.render().copy()
            goal_image = goal_observation_image(
                env, goal, np.asarray(info["goal"], dtype=np.uint8),
                getattr(args, "goal_observation", "centered"),
            )
        else:
            source_state, source_goal, _image, _previous, _goal_image, _previous_action, _contact = (
                terminal_planning_roots[episode]
            )
            observation, info = env.reset(
                seed=0,
                options={"state": source_state.copy(), "goal_state": source_goal.copy()},
            )
            goal = np.asarray(source_goal, dtype=np.float32).copy()
            previous = env.render().copy()
            goal_image = goal_observation_image(
                env, goal, np.asarray(info["goal"], dtype=np.uint8),
                getattr(args, "goal_observation", "centered"),
            )
        # ``planning_start=terminal`` constrains the *block pose* and the
        # visible support-gap chart; it does not guarantee that the pusher is
        # close enough to transmit force.  Keep the historical push shortcut
        # as the default for exact replay, but allow an explicit
        # ``terminal_start_phase=approach`` intervention to establish the
        # agent-side contact lease before the terminal ranker is queried.
        terminal_start_phase = str(
            getattr(args, "terminal_start_phase", "push")
        )
        if terminal_start_phase not in {"push", "approach"}:
            raise ValueError(
                f"unsupported terminal_start_phase={terminal_start_phase!r}"
            )
        phase = (
            terminal_start_phase
            if getattr(args, "planning_start", "random") == "terminal"
            else "approach"
        )
        previous_action: np.ndarray | None = None
        previous_phase = "approach"
        local_history: list[tuple[np.ndarray, np.ndarray]] = []
        recovery_cooldown = 0
        contact_side_misalignment_count = 0
        recovery_anchor: dict[str, Any] | None = None
        terminal_commitment_action: np.ndarray | None = None
        terminal_commitment_remaining = 0
        terminal_confirmation_count = 0
        terminal_stagnation_count = 0
        reacquisition_progress_count = 0
        reacquisition_phase_steps = 0
        reacquisition_total_steps = 0
        reacquisition_motion_excess_count = 0
        reacquisition_route_abort_count = 0
        reacquisition_lockout_steps = 0
        reacquisition_failure_repair_used = False
        args._runtime_last_block_xy = None
        args._runtime_last_agent_xy = None
        args._runtime_temporal_angle_state = {}
        args._runtime_block_history = []
        args._signed_torque_probe_state = {}
        args._signed_torque_probe_magnitude_scale = 1.0
        args._terminal_pulse_state = {}
        trace: list[dict[str, Any]] = []
        for step in range(args.planning_steps):
            recovery_cooldown = max(0, recovery_cooldown - 1)
            if phase in {
                "reacquire_clear", "reacquire", "reacquire_waypoint",
                "reacquire_arc", "reacquire_outer", "reacquire_target",
            }:
                reacquisition_phase_steps += 1
                reacquisition_total_steps += 1
            else:
                reacquisition_phase_steps = 0
            image = env.render().copy()
            if previous_action is not None and previous_phase != "approach":
                measured_effect = observed_local_effect(
                    model, normalizer, image, previous, args.image_size, args.device,
                    orientation_decoder, orientation_decoder_training,
                    current_xy_policy=runtime_current_xy_policy(args),
                    current_xy_disagreement_threshold=getattr(
                        args, "current_xy_disagreement_threshold", 60.0,
                    ),
                    xy_decoder=xy_decoder,
                    xy_decoder_training=xy_decoder_training,
                    rgb_current_angle_decoder=(
                        rgb_current_angle_decoder
                        if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
                    ),
                    rgb_current_angle_training=(
                        rgb_current_angle_training
                        if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
                    ),
                    rgb_angle_temporal_jump=getattr(args, "online_local_angle_jump", 0.75),
                    rgb_temporal_angle_observer=bool(getattr(args, "_use_rgb_temporal_angle_local_effect", False)),
                    rgb_temporal_angle_max_delta=getattr(args, "rgb_temporal_angle_max_delta", 0.55),
                    rgb_temporal_angle_min_confidence=getattr(args, "rgb_temporal_angle_min_confidence", 0.004),
                    max_observed_block_step=getattr(args, "rgb_max_observed_block_step", 48.0),
                )
                local_history.append((previous_action.copy(), measured_effect))
                # Contact point and lever arm change during a PushT episode.
                # A long history violates the local-stationarity assumption
                # of the Ridge probe, so retain only the declared recent
                # window. Setting --online-local-window 12 reproduces the
                # previous protocol exactly.
                local_history = local_history[-int(max(1, getattr(args, "online_local_window", 4))):]
            phase_before_action = phase
            if controller in {"adapter", "hybrid"}:
                learned_action, learned_phase, learned_info = choose_adapter_action(
                    model, normalizer, adapter, adapter_training, image, previous, goal_image, args, phase, local_history,
                    goal_decoder, goal_decoder_training,
                    orientation_decoder, orientation_decoder_training,
                    goal_component_selection,
                    goal_xy_decoder, goal_xy_decoder_training,
                    goal_angle_decoder, goal_angle_decoder_training,
                    xy_decoder, xy_decoder_training,
                    action_cost_ranker, action_cost_ranker_training,
                    terminal_angular_ranker, terminal_angular_ranker_training,
                    contact_mode_classifier, contact_mode_training, contact_mode_penalty,
                    recovery_cooldown=recovery_cooldown,
                    previous_action=previous_action,
                    contact_knn_training=contact_knn_training,
                    rgb_current_angle_decoder=rgb_current_angle_decoder,
                    rgb_current_angle_training=rgb_current_angle_training,
                    rgb_goal_angle_decoder=rgb_goal_angle_decoder,
                    rgb_goal_angle_training=rgb_goal_angle_training,
                    rgb_goal_xy_decoder=rgb_goal_xy_decoder,
                    rgb_goal_xy_training=rgb_goal_xy_training,
                    contact_side_misalignment_count=contact_side_misalignment_count,
                    recovery_anchor=recovery_anchor,
                    terminal_confirmation_count=terminal_confirmation_count,
                    terminal_stagnation_count=terminal_stagnation_count,
                    reacquisition_progress_count=reacquisition_progress_count,
                    reacquisition_phase_steps=reacquisition_phase_steps,
                    reacquisition_total_steps=reacquisition_total_steps,
                    reacquisition_motion_excess_count=reacquisition_motion_excess_count,
                    reacquisition_route_abort_count=reacquisition_route_abort_count,
                    reacquisition_lockout_steps=reacquisition_lockout_steps,
                    reacquisition_failure_repair_used=reacquisition_failure_repair_used,
                    agent_xy_decoder=agent_xy_decoder,
                    agent_xy_decoder_training=agent_xy_decoder_training,
                )
                if controller == "adapter":
                    action, phase, info_row = learned_action, learned_phase, learned_info
                else:
                    args._force_geometry_translation = bool(
                        float(learned_info.get("force_geometry_translation", 0.0)) > 0.0
                    )
                    geometry_action, geometry_phase = geometry_baseline_action(
                        model, normalizer, image, previous, goal_image, args, phase,
                        goal_decoder, goal_decoder_training,
                        orientation_decoder, orientation_decoder_training,
                        goal_component_selection,
                        goal_xy_decoder, goal_xy_decoder_training,
                        goal_angle_decoder, goal_angle_decoder_training,
                        xy_decoder, xy_decoder_training,
                        rgb_current_angle_decoder, rgb_current_angle_training,
                        rgb_goal_angle_decoder, rgb_goal_angle_training,
                        rgb_goal_xy_decoder, rgb_goal_xy_training,
                        contact_side_misalignment_count,
                        agent_xy_decoder, agent_xy_decoder_training,
                    )
                    args._force_geometry_translation = False
                    hybrid_translation_cap = float(
                        getattr(args, "hybrid_translation_action_max_magnitude", 0.0)
                    )
                    if (
                        hybrid_translation_cap > 0.0
                        and learned_phase in {"translate", "push"}
                        and float(learned_info.get("candidate_contact_band", 0.0)) > 0.0
                    ):
                        geometry_norm = float(np.linalg.norm(geometry_action))
                        if geometry_norm > hybrid_translation_cap:
                            geometry_action = (
                                np.asarray(geometry_action, dtype=np.float32)
                                / max(geometry_norm, 1.0e-6)
                                * hybrid_translation_cap
                            ).astype(np.float32)
                            learned_info["hybrid_translation_cap_used"] = 1.0
                        else:
                            learned_info["hybrid_translation_cap_used"] = 0.0
                        learned_info["hybrid_translation_cap"] = hybrid_translation_cap
                    # The learned controller owns the approach/contact
                    # interface until it has explicitly established a valid
                    # contact manifold.  Previously the hybrid wrapper only
                    # recognized the returned phase name; when the learned
                    # branch returned a conservative ``approach`` action
                    # together with a handoff-blocking audit flag, the
                    # geometry baseline was allowed to replace it with a
                    # large terminal push.  On terminal roots this caused a
                    # single 0.75-scale geometry action before contact and
                    # produced the large XY explosions seen in the replay.
                    # Preserve the learned safety action for these explicit
                    # no-handoff conditions.  This is a control-flow guard,
                    # not a simulator-state or successor-state query.
                    learned_contact_handoff_blocked = any(
                        float(learned_info.get(flag, 0.0)) > 0.0
                        for flag in (
                            "handoff_blocked_by_contact_geometry",
                            "handoff_blocked_by_absolute_contact_distance",
                            "handoff_blocked_by_contact_side",
                            "terminal_side_gate_before_support_gap",
                            "terminal_side_gate_before_recovery",
                        )
                    )
                    post_reacq_cap = float(getattr(
                        args, "reacquisition_post_success_translation_cap", 0.0
                    ))
                    if (
                        post_reacq_cap > 0.0
                        and int(reacquisition_progress_count) > 0
                        and learned_phase in {"translate", "push"}
                    ):
                        geometry_norm = float(np.linalg.norm(geometry_action))
                        if geometry_norm > post_reacq_cap:
                            geometry_action = (
                                np.asarray(geometry_action, dtype=np.float32)
                                / max(geometry_norm, 1.0e-6)
                                * post_reacq_cap
                            ).astype(np.float32)
                            learned_info["reacquisition_post_success_cap_used"] = 1.0
                        else:
                            learned_info["reacquisition_post_success_cap_used"] = 0.0
                        learned_info["reacquisition_post_success_cap"] = post_reacq_cap
                    if learned_phase in {
                        "recovery", "terminal_approach", "torque", "hold", "signed_probe_xy_repair",
                        "reacquire_clear", "reacquire",
                        "reacquire_waypoint", "reacquire_arc", "reacquire_outer", "reacquire_target",
                    } or learned_contact_handoff_blocked:
                        action, phase, info_row = learned_action, learned_phase, learned_info
                        info_row["hybrid_source"] = 1.0
                        info_row["hybrid_contact_handoff_guard_used"] = float(
                            learned_contact_handoff_blocked
                        )
                    else:
                        # Translation is deliberately kept on the RGB
                        # geometry servo.  In particular, a successful
                        # reacquisition returns the learned interface to a
                        # push/translation phase but must not hand the first
                        # post-recovery action to the unstable learned zero
                        # baseline or torque ranker.
                        geometry_handoff_phase = geometry_phase
                        if (
                            learned_phase == "push"
                            and float(learned_info.get("reacquisition_success", 0.0)) > 0.0
                        ):
                            geometry_handoff_phase = "push"
                        action, phase, info_row = geometry_action, geometry_handoff_phase, {
                            **learned_info, "hybrid_source": 0.0,
                        }
            else:
                action, phase = geometry_baseline_action(
                    model, normalizer, image, previous, goal_image, args, phase,
                    goal_decoder, goal_decoder_training,
                    orientation_decoder, orientation_decoder_training,
                    goal_component_selection,
                    goal_xy_decoder, goal_xy_decoder_training,
                    goal_angle_decoder, goal_angle_decoder_training,
                    xy_decoder, xy_decoder_training,
                    rgb_current_angle_decoder, rgb_current_angle_training,
                    rgb_goal_angle_decoder, rgb_goal_angle_training,
                    rgb_goal_xy_decoder, rgb_goal_xy_training,
                    contact_side_misalignment_count,
                    agent_xy_decoder, agent_xy_decoder_training,
                )
                info_row = {}
            # ``choose_adapter_action`` computes the support lease before any
            # early state-machine return.  Merge that audit after branch
            # selection so every replay trace records both the centroid chart
            # and the support-aware override decision, including approach and
            # reacquisition rows that return before the learned candidate fan.
            support_lease_audit = getattr(args, "_support_lease_audit", None)
            if isinstance(support_lease_audit, dict):
                info_row = {**info_row, **support_lease_audit}
            # Approach is a distinct RGB servo phase.  It should not inherit
            # the much larger translation/torque action scale merely because
            # a terminal root is already close to the goal.  A replay may
            # opt into this cap without changing the stored training or
            # ranking protocol; the default remains disabled for historical
            # reproducibility.  Applying it at the final controller boundary
            # covers both learned and geometry sources and prevents a hybrid
            # wrapper from reintroducing an unsafe approach impulse.
            approach_cap = float(
                getattr(args, "approach_action_max_magnitude", -1.0)
            )
            if phase == "approach" and approach_cap > 0.0:
                action_before_approach_cap = np.asarray(action, dtype=np.float32).copy()
                approach_norm = float(np.linalg.norm(action_before_approach_cap))
                if approach_norm > approach_cap:
                    action = (
                        action_before_approach_cap
                        * (approach_cap / max(approach_norm, 1.0e-6))
                    ).astype(np.float32)
                    info_row["approach_action_cap_used"] = 1.0
                else:
                    info_row["approach_action_cap_used"] = 0.0
                info_row["approach_action_cap"] = approach_cap
                info_row["approach_action_raw_norm"] = approach_norm
                info_row["approach_action_final_norm"] = float(np.linalg.norm(action))
            # A short-horizon terminal ranker is trained on a repeated-action
            # intervention.  Replanning a fresh action every frame changes the
            # deployed control problem and was a direct source of the earlier
            # h=4 ranker's negative closed-loop result.  The commitment is
            # deliberately restricted to the terminal torque lease; recovery,
            # approach, and translation remain fully receding-horizon.
            commitment = max(
                1, int(getattr(args, "terminal_action_commitment", 1))
            )
            # A safety observer that requests a settle/reversal must also
            # cancel any repeated-action lease.  Previously
            # ``terminal_motion_damping`` returned a zero action from the
            # selector, but this outer commitment wrapper immediately
            # replaced it with the stale ranker action.  The trace therefore
            # reported damping while the environment still received the
            # unsafe pulse, recreating terminal contact ejection.
            if (
                float(info_row.get("terminal_motion_damping_used", 0.0)) > 0.0
                or float(info_row.get("terminal_pulse_frozen", 0.0)) > 0.0
                or float(info_row.get("terminal_pulse_failed", 0.0)) > 0.0
                or float(info_row.get("terminal_pulse_fallback_used", 0.0)) > 0.0
            ):
                terminal_commitment_action = None
                terminal_commitment_remaining = 0
                info_row["terminal_action_commitment_cancelled"] = 1.0
            if terminal_commitment_remaining > 0:
                if (
                    phase in {"torque", "hold"}
                    and phase_before_action in {"torque", "hold"}
                    and terminal_commitment_action is not None
                ):
                    action = terminal_commitment_action.copy()
                    info_row = {
                        **info_row,
                        "terminal_action_commitment_used": 1.0,
                        "terminal_action_commitment_remaining_before": float(
                            terminal_commitment_remaining
                        ),
                        "terminal_action_commitment": float(commitment),
                    }
                    terminal_commitment_remaining -= 1
                else:
                    terminal_commitment_action = None
                    terminal_commitment_remaining = 0
            commitment_eligible = terminal_action_commitment_eligible(
                phase, action, info_row,
            )
            info_row["terminal_action_commitment_eligible"] = float(commitment_eligible)
            if (
                terminal_commitment_remaining <= 0
                and commitment > 1
                and commitment_eligible
            ):
                terminal_commitment_action = np.asarray(action, dtype=np.float32).copy()
                terminal_commitment_remaining = commitment - 1
                info_row = {
                    **info_row,
                    "terminal_action_commitment_started": 1.0,
                    "terminal_action_commitment": float(commitment),
                }
            info_row.setdefault("terminal_action_commitment_used", 0.0)
            info_row.setdefault("terminal_action_commitment_started", 0.0)
            info_row.setdefault(
                "terminal_action_commitment_remaining_after",
                float(terminal_commitment_remaining),
            )
            # Arm the one-step RGB confirmation only after the final action
            # interface has been selected.  This keeps the confirmation
            # protocol invariant across adapter and hybrid controllers and
            # prevents a zero/hold action from being treated as a pulse.
            if bool(getattr(args, "terminal_pulse_confirmation", False)):
                pulse_norm = float(np.linalg.norm(np.asarray(action, dtype=np.float32)))
                pulse_angle_before = float(
                    info_row.get(
                        "audit_temporal_angle_error",
                        info_row.get("angle_error_visual", float("nan")),
                    )
                )
                pulse_xy_before = float(info_row.get("block_xy_visual", float("nan")))
                if (
                    phase in {"torque", "hold"}
                    and pulse_norm > 1.0e-6
                    and np.isfinite(pulse_angle_before)
                    and np.isfinite(pulse_xy_before)
                    and not bool(getattr(args, "_terminal_pulse_state", {}).get("disabled", False))
                    and float(info_row.get("terminal_pulse_frozen", 0.0)) <= 0.0
                ):
                    args._terminal_pulse_state = {
                        "active": True,
                        "kind": str(info_row.get("terminal_pulse_kind", "learned")),
                        "angle_before": pulse_angle_before,
                        "xy_before": pulse_xy_before,
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "armed_step": int(step),
                        "failures": int(getattr(args, "_terminal_pulse_state", {}).get("failures", 0)),
                    }
                info_row["terminal_pulse_confirmation_used"] = 1.0
                info_row["terminal_pulse_armed"] = float(
                    bool(getattr(args, "_terminal_pulse_state", {}).get("active", False))
                )
                info_row["terminal_pulse_failure_count"] = float(
                    getattr(args, "_terminal_pulse_state", {}).get("failures", 0)
                )
            # Every side-reacquisition implementation (learned, hybrid, and
            # geometry baseline) must share the same final action lease.  A
            # few historical recovery branches returned phase="recovery"
            # with an empty info dict, so the explicit protocol flag is also
            # part of the gate.  This is intentionally applied at the final
            # controller boundary, after branch selection, to make an omitted
            # local cap fail safe rather than silently change the protocol.
            reacquisition_active = reacquisition_phase_active(phase, info_row) or (
                phase == "recovery"
                and bool(getattr(args, "terminal_side_reacquisition", False))
            )
            if reacquisition_active:
                pre_cap_action = np.asarray(action, dtype=np.float32).copy()
                action, reacquisition_cap_info = bounded_reacquisition_action(action, args)
                info_row.update({
                    **reacquisition_cap_info,
                    "reacquisition_action_before_cap": pre_cap_action.tolist(),
                })
            raw_action = np.asarray(action, dtype=np.float32).copy()
            if bool(getattr(args, "persistent_recovery_anchor", True)):
                if float(info_row.get("recovery_anchor_release", 0.0)) > 0.0:
                    recovery_anchor = None
                elif float(info_row.get("recovery_anchor_active", 0.0)) > 0.0:
                    recovery_anchor = {
                        "block": np.asarray(info_row["recovery_anchor_block"], dtype=np.float32),
                        "direction": np.asarray(info_row["recovery_anchor_direction"], dtype=np.float32),
                        "route_sign": float(info_row["recovery_anchor_route_sign"]),
                        "clearance": float(info_row["recovery_anchor_clearance"]),
                        "workspace_safe": bool(float(info_row.get("recovery_anchor_workspace_safe", 1.0)) > 0.0),
                        "clearance_relaxed": bool(float(info_row.get("recovery_anchor_clearance_relaxed", 0.0)) > 0.0),
                    }
                elif phase_before_action in {
                    "recovery", "reacquire_clear", "reacquire", "reacquire_waypoint",
                    "reacquire_arc", "reacquire_outer", "reacquire_target",
                } and phase not in {
                    "recovery", "reacquire_clear", "reacquire", "reacquire_waypoint",
                    "reacquire_arc", "reacquire_outer", "reacquire_target",
                }:
                    recovery_anchor = None
            observed_terminal_xy = float(info_row.get(
                "block_xy_terminal_observer",
                info_row.get("block_xy_visual", float("inf")),
            ))
            if (
                observed_terminal_xy <= float(getattr(args, "terminal_xy", 35.0))
                and float(info_row.get("observed_block_motion", float("inf")))
                <= float(getattr(args, "terminal_max_observer_jump", 8.0))
                and phase not in {
                    "recovery", "reacquire_clear", "reacquire", "reacquire_waypoint",
                    "reacquire_arc", "reacquire_outer", "reacquire_target",
                }
            ):
                terminal_confirmation_count += 1
            else:
                terminal_confirmation_count = 0
            info_row["terminal_confirmation_count"] = float(terminal_confirmation_count)
            if float(info_row.get("reacquisition_success", 0.0)) > 0.0:
                reacquisition_progress_count = max(
                    0, int(getattr(args, "reacquisition_progress_steps", 8)),
                )
            elif reacquisition_progress_count > 0:
                reacquisition_progress_count -= 1
            info_row["reacquisition_progress_count"] = float(reacquisition_progress_count)
            info_row["reacquisition_phase_steps"] = float(reacquisition_phase_steps)
            info_row["reacquisition_total_steps"] = float(reacquisition_total_steps)
            if phase in {"reacquire_arc", "reacquire_target"}:
                reacquisition_motion_excess_count = int(
                    info_row.get("reacquisition_motion_excess_count", 0.0)
                )
            else:
                reacquisition_motion_excess_count = 0
            info_row["reacquisition_motion_excess_count"] = float(
                reacquisition_motion_excess_count
            )
            if float(info_row.get("reacquisition_route_abort", 0.0)) > 0.0:
                reacquisition_route_abort_count += 1
                reacquisition_lockout_steps = max(
                    reacquisition_lockout_steps,
                    int(info_row.get(
                        "reacquisition_lockout_steps",
                        getattr(args, "reacquisition_lockout_steps", 12),
                    )),
                )
                reacquisition_failure_repair_used = False
            elif float(info_row.get("reacquisition_bounded_translation_repair", 0.0)) > 0.0:
                reacquisition_failure_repair_used = True
            elif float(info_row.get("reacquisition_success", 0.0)) > 0.0:
                reacquisition_route_abort_count = 0
                reacquisition_lockout_steps = 0
                reacquisition_failure_repair_used = False
            elif reacquisition_lockout_steps > 0:
                reacquisition_lockout_steps -= 1
            info_row["reacquisition_route_abort_count_after"] = float(
                reacquisition_route_abort_count
            )
            info_row["reacquisition_lockout_steps_after"] = float(
                reacquisition_lockout_steps
            )
            info_row["reacquisition_failure_repair_used_after"] = float(
                reacquisition_failure_repair_used
            )
            if (
                bool(getattr(args, "terminal_stagnation_watchdog", False))
                and phase_before_action in {"torque", "hold"}
                and float(info_row.get("observed_block_motion", float("inf")))
                <= float(getattr(args, "terminal_stagnation_motion", 1.5))
            ):
                terminal_stagnation_count += 1
            else:
                terminal_stagnation_count = 0
            info_row["terminal_stagnation_count_after"] = float(terminal_stagnation_count)
            action, projection_info = project_contact_action(
                action, image, args, phase,
            )
            if reacquisition_active:
                post_projection_action = np.asarray(action, dtype=np.float32).copy()
                action, post_cap_info = bounded_reacquisition_action(action, args)
                info_row.update({
                    "reacquisition_post_projection_raw_norm": post_cap_info[
                        "reacquisition_raw_norm"
                    ],
                    "reacquisition_post_projection_final_norm": post_cap_info[
                        "reacquisition_final_norm"
                    ],
                    "reacquisition_post_projection_cap_applied": post_cap_info[
                        "reacquisition_cap_applied"
                    ],
                    "reacquisition_action_after_projection_before_cap": post_projection_action.tolist(),
                })
            info_row = {
                **info_row,
                **projection_info,
                "action_before_projection": raw_action.tolist(),
                "action_after_projection": np.asarray(action, dtype=np.float32).tolist(),
            }
            observed_alignment = info_row.get("contact_side_alignment")
            if observed_alignment is not None and np.isfinite(float(observed_alignment)):
                if float(observed_alignment) < float(args.contact_side_alignment_min):
                    contact_side_misalignment_count += 1
                else:
                    contact_side_misalignment_count = 0
            else:
                contact_side_misalignment_count = 0
            info_row["contact_side_misalignment_count_after"] = float(
                contact_side_misalignment_count
            )
            if float(info_row.get("contact_lost_recovery", 0.0)) > 0.0:
                recovery_cooldown = 4
            # Runtime observer audit.  This is deliberately computed before
            # stepping the environment and is never passed back to the
            # controller.  It records the exact RGB interface used by the
            # controller alongside simulator truth so a failed rollout can
            # be attributed to observer drift versus action-effect ranking.
            audit_visual, _audit_features, _audit_previous_features, audit_goal_pose = runtime_observation(
                model, normalizer, image, previous, goal_image, args.image_size, args.device,
                goal_decoder=goal_decoder,
                goal_decoder_training=goal_decoder_training,
                orientation_decoder=orientation_decoder,
                orientation_decoder_training=orientation_decoder_training,
                goal_component_selection=goal_component_selection,
                goal_xy_decoder=goal_xy_decoder,
                goal_xy_decoder_training=goal_xy_decoder_training,
                goal_angle_decoder=goal_angle_decoder,
                goal_angle_decoder_training=goal_angle_decoder_training,
                xy_decoder=xy_decoder,
                xy_decoder_training=xy_decoder_training,
                current_xy_policy=runtime_current_xy_policy(args),
                current_xy_disagreement_threshold=getattr(
                    args, "current_xy_disagreement_threshold", 60.0,
                ),
                paired_reference_policy=getattr(args, "paired_reference_policy", "raw"),
                current_angle_policy=getattr(args, "current_angle_policy", "model"),
                rgb_current_angle_decoder=(
                    rgb_current_angle_decoder
                    if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
                ),
                rgb_current_angle_training=(
                    rgb_current_angle_training
                    if bool(getattr(args, "_use_rgb_current_angle_runtime", False)) else None
                ),
                rgb_goal_angle_decoder=rgb_goal_angle_decoder,
                rgb_goal_angle_training=rgb_goal_angle_training,
                rgb_goal_xy_decoder=rgb_goal_xy_decoder,
                rgb_goal_xy_training=rgb_goal_xy_training,
                rgb_angle_temporal_jump=getattr(args, "rgb_angle_temporal_jump", 0.75),
                rgb_angle_disagreement_policy=getattr(args, "rgb_angle_disagreement_policy", "none"),
                rgb_angle_disagreement_threshold=getattr(args, "rgb_angle_disagreement_threshold", 1.20),
                rgb_angle_disagreement_terminal_xy=getattr(args, "rgb_angle_disagreement_terminal_xy", -1.0),
                rgb_temporal_angle_state=getattr(args, "_runtime_temporal_angle_state", None),
                rgb_temporal_angle_observer=bool(getattr(args, "_use_rgb_temporal_angle_runtime", False)),
                rgb_temporal_angle_max_delta=getattr(args, "rgb_temporal_angle_max_delta", 0.55),
                rgb_temporal_angle_min_confidence=getattr(args, "rgb_temporal_angle_min_confidence", 0.004),
                max_observed_block_step=getattr(args, "rgb_max_observed_block_step", 48.0),
                held_block_xy=getattr(args, "_runtime_last_block_xy", None),
                held_agent_xy=getattr(args, "_runtime_last_agent_xy", None),
                agent_xy_decoder=agent_xy_decoder,
                agent_xy_decoder_training=agent_xy_decoder_training,
            )
            if np.isfinite(audit_visual[2:4]).all():
                args._runtime_last_block_xy = np.asarray(audit_visual[2:4], dtype=np.float32).copy()
            state_before = np.asarray(observation["state"], dtype=np.float32).copy()
            audit_goal_xy = np.asarray(audit_goal_pose[:2], dtype=np.float32)
            audit_true_current_angle = theta_from_pose(state_vector(state_before))
            audit_observed_current_angle = float(np.arctan2(audit_visual[4], audit_visual[5]))
            audit_goal_angle = float(np.arctan2(audit_goal_pose[2], audit_goal_pose[3]))
            info_row.update({
                "audit_observed_agent_xy": np.asarray(audit_visual[:2], dtype=np.float32).tolist(),
                "audit_observed_block_xy": np.asarray(audit_visual[2:4], dtype=np.float32).tolist(),
                "audit_observed_goal_xy": audit_goal_xy.tolist(),
                "audit_observer_current_xy_error": float(
                    np.linalg.norm(np.asarray(audit_visual[2:4]) - state_before[2:4])
                ),
                "audit_observer_agent_xy_error": float(
                    np.linalg.norm(np.asarray(audit_visual[:2]) - state_before[:2])
                ),
                "audit_observer_goal_xy_error": float(
                    np.linalg.norm(audit_goal_xy - goal[2:4])
                ),
                "audit_observed_goal_distance": float(
                    np.linalg.norm(audit_goal_xy - np.asarray(audit_visual[2:4]))
                ),
                "audit_physical_goal_distance_before": float(
                    np.linalg.norm(state_before[2:4] - goal[2:4])
                ),
                "audit_observed_angle": audit_observed_current_angle,
                "audit_observed_angle_error": float(abs(wrap(audit_goal_angle - audit_observed_current_angle))),
                "audit_physical_angle_error_before": float(abs(wrap(audit_goal_angle - audit_true_current_angle))),
                "audit_physical_agent_block_distance_before": float(
                    np.linalg.norm(state_before[:2] - state_before[2:4])
                ),
                "audit_observer_interface": runtime_current_xy_policy(args),
            })
            temporal_audit = getattr(args, "_runtime_temporal_angle_state", {}).get("audit", {})
            if temporal_audit:
                info_row.update({
                    f"audit_{key}": float(value)
                    for key, value in temporal_audit.items()
                    if np.isscalar(value) and np.isfinite(float(value))
                })
                temporal_delta = float(temporal_audit.get("temporal_angle_delta", 0.0))
                temporal_valid = float(temporal_audit.get("temporal_angle_update_used", 0.0)) > 0.0
                temporal_angle_for_confirmation = float(wrap(
                    audit_observed_current_angle + (temporal_delta if temporal_valid else 0.0)
                ))
                info_row["audit_temporal_angle_error"] = float(abs(wrap(
                    audit_goal_angle - temporal_angle_for_confirmation
                )))
            terminated = False
            truncated = False
            for _physical_step in range(args.action_repeat):
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            # The learned adapter is trained for ``action_repeat`` physical
            # steps per decision.  Keep the pre-action image as the temporal
            # reference so the next RGB finite difference has the same
            # horizon as the offline intervention label.
            previous = image
            previous_action = action.copy()
            previous_phase = phase
            state = np.asarray(observation["state"], dtype=np.float32)
            info_row.update({
                "audit_physical_block_motion": float(
                    np.linalg.norm(state[2:4] - state_before[2:4])
                ),
                "audit_physical_goal_distance_after": float(
                    np.linalg.norm(state[2:4] - goal[2:4])
                ),
                "audit_physical_angle_error_after": float(
                    abs(wrap(theta_from_pose(state_vector(state)) - theta_from_pose(state_vector(goal))))
                ),
                "audit_physical_agent_block_distance_after": float(
                    np.linalg.norm(state[:2] - state[2:4])
                ),
                "audit_goal_progress_physical": float(
                    np.linalg.norm(state_before[2:4] - goal[2:4])
                    - np.linalg.norm(state[2:4] - goal[2:4])
                ),
            })
            trace.append({
                "step": step, "phase": phase, "action": action.tolist(),
                "block_xy": float(np.linalg.norm(state[2:4] - goal[2:4])),
                "angle": float(abs(wrap(theta_from_pose(state_vector(state)) - theta_from_pose(state_vector(goal))))),
                "agent_block_distance": float(np.linalg.norm(state[:2] - state[2:4])),
                **info_row,
            })
            if strict_success(state, goal) or terminated or truncated:
                break
        rows.append({
            "episode": episode,
            "strict_success": strict_success(state, goal),
            "steps": len(trace),
            "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "angle_error": float(abs(wrap(theta_from_pose(state_vector(state)) - theta_from_pose(state_vector(goal))))),
            "trace": trace,
        })
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.action_repeat < 1:
        raise ValueError("action_repeat must be positive")
    if args.contact_mode_action_repeat < 1:
        raise ValueError("contact_mode_action_repeat must be positive")
    if args.terminal_action_horizon < 1:
        raise ValueError("terminal_action_horizon must be positive")
    if args.terminal_action_commitment < 1:
        raise ValueError("terminal_action_commitment must be positive")
    if not 0.0 <= args.terminal_root_fraction <= 1.0:
        raise ValueError("terminal_root_fraction must lie in [0, 1]")
    if not 0.0 <= args.bootstrap_root_fraction <= 1.0:
        raise ValueError("bootstrap_root_fraction must lie in [0, 1]")
    if args.terminal_root_fraction + args.bootstrap_root_fraction > 1.0:
        raise ValueError("terminal_root_fraction + bootstrap_root_fraction must not exceed 1")
    if args.bootstrap_root_fraction > 0.0 and args.bootstrap_checkpoint is None:
        raise ValueError("bootstrap_root_fraction requires --bootstrap-checkpoint")
    if args.bootstrap_checkpoint is not None and not args.bootstrap_checkpoint.is_file():
        raise FileNotFoundError(f"bootstrap checkpoint not found: {args.bootstrap_checkpoint}")
    data_seed = args.seed if args.data_seed is None else int(args.data_seed)
    model, normalizer, base_args = load_base(Path(args.base_checkpoint).resolve(), args.device)
    actions = action_bank(args.action_directions, declared_action_magnitudes(args))
    env = gym.make(
        "swm/PushT-v1", render_mode="rgb_array",
        max_episode_steps=args.planning_steps * args.action_repeat,
    )
    try:
        train_terminal = int(round(args.train_roots * args.terminal_root_fraction))
        val_terminal = int(round(args.val_roots * args.terminal_root_fraction))
        train_bootstrap = int(round(args.train_roots * args.bootstrap_root_fraction))
        val_bootstrap = int(round(args.val_roots * args.bootstrap_root_fraction))
        train_roots = balanced_source_cases(
            env, args.train_roots - train_terminal - train_bootstrap, args.root_rollout_steps, data_seed,
            args.contact_root_fraction, args.contact_approach_steps,
        )
        train_roots.extend(
            terminal_pose_cases(
                env, train_terminal, data_seed + 400_000,
                position_jitter=float(args.terminal_position_jitter),
                support_gap_max=float(getattr(args, "terminal_root_support_gap", 0.0)),
                support_angle_samples=int(getattr(args, "terminal_root_support_angle_samples", 32)),
            )
        )
        train_bootstrap_roots = []
        if train_bootstrap:
            train_bootstrap_roots = bootstrap_closed_loop_source_cases(
                env, args.bootstrap_checkpoint, train_bootstrap, data_seed + 500_000,
                args.bootstrap_rollout_episodes, args.bootstrap_rollout_steps,
                args.bootstrap_contact_distance, args.device,
            )
            train_roots.extend(train_bootstrap_roots)
        val_roots = balanced_source_cases(
            env, args.val_roots - val_terminal - val_bootstrap, args.root_rollout_steps, data_seed + 10_000,
            args.contact_root_fraction, args.contact_approach_steps,
        )
        val_roots.extend(
            terminal_pose_cases(
                env, val_terminal, data_seed + 410_000,
                position_jitter=float(args.terminal_position_jitter),
                support_gap_max=float(getattr(args, "terminal_root_support_gap", 0.0)),
                support_angle_samples=int(getattr(args, "terminal_root_support_angle_samples", 32)),
            )
        )
        val_bootstrap_roots = []
        if val_bootstrap:
            val_bootstrap_roots = bootstrap_closed_loop_source_cases(
                env, args.bootstrap_checkpoint, val_bootstrap, data_seed + 510_000,
                args.bootstrap_rollout_episodes, args.bootstrap_rollout_steps,
                args.bootstrap_contact_distance, args.device,
            )
            val_roots.extend(val_bootstrap_roots)
        train = attach_targets(make_local_effect_dataset(
            env, train_roots, args.image_size, actions, args.action_repeat,
            args.goal_observation,
        ), actions)
        val = attach_targets(make_local_effect_dataset(
            env, val_roots, args.image_size, actions, args.action_repeat,
            args.goal_observation,
        ), actions)
        # Root split check uses source state hashes, not row counts.
        train_hashes = {tuple(np.asarray(root[0]).round(6)) for root in train_roots}
        val_hashes = {tuple(np.asarray(root[0]).round(6)) for root in val_roots}
        if train_hashes & val_hashes:
            raise RuntimeError("train/validation source-root overlap")
        print(json.dumps({
            "phase": "dataset", "train_roots": len(train_roots), "val_roots": len(val_roots),
            "actions_per_root": len(actions), "train_bootstrap_roots": len(train_bootstrap_roots),
            "val_bootstrap_roots": len(val_bootstrap_roots),
        }), flush=True)
        train_features = frozen_features(model, train, args.image_size, args.device)
        val_features = frozen_features(model, val, args.image_size, args.device)
        mode_train = None
        mode_val = None
        if args.contact_mode_gate:
            # The effect adapter keeps the declared one-step intervention
            # protocol.  The mode head uses a separate, fixed two-step label
            # window so impending contact separation is represented even when
            # it is not visible after a single simulator step.
            mode_train = make_local_effect_dataset(
                env, train_roots, args.image_size, actions, args.contact_mode_action_repeat,
                args.goal_observation,
            )
            mode_val = make_local_effect_dataset(
                env, val_roots, args.image_size, actions, args.contact_mode_action_repeat,
                args.goal_observation,
            )
        train_goal_features = goal_feature_rows(model, train, args.device)
        val_goal_features = goal_feature_rows(model, val, args.device)
        train_current_features = current_feature_rows(model, train, args.device)
        val_current_features = current_feature_rows(model, val, args.device)
        train_goal_targets = np.asarray([observable_pose_vector(root[1])[2:6] for root in train_roots], dtype=np.float32)
        val_goal_targets = np.asarray([observable_pose_vector(root[1])[2:6] for root in val_roots], dtype=np.float32)
        train_orientation_targets = np.asarray(
            [[np.sin(root[0][4]), np.cos(root[0][4])] for root in train_roots], dtype=np.float32,
        )
        val_orientation_targets = np.asarray(
            [[np.sin(root[0][4]), np.cos(root[0][4])] for root in val_roots], dtype=np.float32,
        )
        train_xy_targets = np.asarray([root[0][2:4] for root in train_roots], dtype=np.float32)
        val_xy_targets = np.asarray([root[0][2:4] for root in val_roots], dtype=np.float32)
        train_agent_xy_targets = np.asarray([root[0][:2] for root in train_roots], dtype=np.float32)
        val_agent_xy_targets = np.asarray([root[0][:2] for root in val_roots], dtype=np.float32)
        readout_train_features = train_current_features
        readout_val_features = val_current_features
        readout_train_orientation_targets = train_orientation_targets
        readout_val_orientation_targets = val_orientation_targets
        readout_train_xy_targets = train_xy_targets
        readout_val_xy_targets = val_xy_targets
        readout_train_agent_xy_targets = train_agent_xy_targets
        readout_val_agent_xy_targets = val_agent_xy_targets
        dynamic_readout_metrics: dict[str, Any] = {
            "enabled": bool(args.readout_augmentation_steps > 0),
            "steps_per_root": int(args.readout_augmentation_steps),
            "train_rows": 0,
            "validation_rows": 0,
        }
        if args.readout_augmentation_steps > 0:
            dynamic_train_features, dynamic_train_agent_xy, dynamic_train_xy, dynamic_train_orientation = dynamic_readout_rows(
                model, env, train_roots, args.image_size, args.device,
                data_seed + 600_000, args.readout_augmentation_steps,
            )
            dynamic_val_features, dynamic_val_agent_xy, dynamic_val_xy, dynamic_val_orientation = dynamic_readout_rows(
                model, env, val_roots, args.image_size, args.device,
                data_seed + 610_000, args.readout_augmentation_steps,
            )
            readout_train_features = np.concatenate(
                [train_current_features, dynamic_train_features], axis=0,
            )
            readout_val_features = np.concatenate(
                [val_current_features, dynamic_val_features], axis=0,
            )
            readout_train_orientation_targets = np.concatenate(
                [train_orientation_targets, dynamic_train_orientation], axis=0,
            )
            readout_val_orientation_targets = np.concatenate(
                [val_orientation_targets, dynamic_val_orientation], axis=0,
            )
            readout_train_xy_targets = np.concatenate(
                [train_xy_targets, dynamic_train_xy], axis=0,
            )
            readout_val_xy_targets = np.concatenate(
                [val_xy_targets, dynamic_val_xy], axis=0,
            )
            readout_train_agent_xy_targets = np.concatenate(
                [train_agent_xy_targets, dynamic_train_agent_xy], axis=0,
            )
            readout_val_agent_xy_targets = np.concatenate(
                [val_agent_xy_targets, dynamic_val_agent_xy], axis=0,
            )
            dynamic_readout_metrics.update({
                "train_rows": int(len(dynamic_train_features)),
                "validation_rows": int(len(dynamic_val_features)),
                "selection_validation_distribution": "static roots plus root-disjoint offline rollout frames",
            })
        goal_decoder, goal_decoder_training = train_goal_decoder(
            train_goal_features, train_goal_targets, val_goal_features, val_goal_targets, args,
        )
        decoded_goals = predict_goal(goal_decoder, val_goal_features, goal_decoder_training, args.device)
        goal_readout_metrics = {
            "xy_rmse": float(np.sqrt(np.mean(np.square(decoded_goals[:, :2] - val_goal_targets[:, :2])))),
            "xy_mean_error": float(np.mean(np.linalg.norm(decoded_goals[:, :2] - val_goal_targets[:, :2], axis=1))),
            "angle_mean_error": angle_error_batch(decoded_goals, val_goal_targets),
            "roots": int(len(val_goal_targets)),
        }
        base_goals = base_goal_rows(model, normalizer, val, args.device)
        base_goal_readout_metrics = {
            "xy_rmse": float(np.sqrt(np.mean(np.square(base_goals[:, :2] - val_goal_targets[:, :2])))),
            "xy_mean_error": float(np.mean(np.linalg.norm(base_goals[:, :2] - val_goal_targets[:, :2], axis=1))),
            "angle_mean_error": angle_error_batch(base_goals, val_goal_targets),
            "roots": int(len(val_goal_targets)),
        }
        goal_decoder_score = goal_readout_metrics["xy_mean_error"] + 20.0 * goal_readout_metrics["angle_mean_error"]
        base_goal_score = base_goal_readout_metrics["xy_mean_error"] + 20.0 * base_goal_readout_metrics["angle_mean_error"]
        selected_goal_decoder = goal_decoder if goal_decoder_score <= base_goal_score else None
        selected_goal_training = goal_decoder_training if selected_goal_decoder is not None else None
        goal_selection = {
            "selected": "independent_decoder" if selected_goal_decoder is not None else "base_checkpoint_goal_head",
            "validation_task_score_decoder": float(goal_decoder_score),
            "validation_task_score_base": float(base_goal_score),
            "decoder_metrics": goal_readout_metrics,
            "base_metrics": base_goal_readout_metrics,
        }
        goal_interface_selection: dict[str, Any] = {
            "selection_status": "pending_exact_runtime_interface_audit",
        }
        # The joint four-output goal decoder is retained as an audit baseline,
        # but translation and orientation have different conditioning and
        # error geometry.  Fit separate two-output readouts and select each
        # component against the frozen checkpoint head on validation roots.
        goal_xy_decoder, goal_xy_decoder_training = train_orientation_decoder(
            train_goal_features[:, TARGET_FEATURE_START:TARGET_FEATURE_STOP], train_goal_targets[:, :2],
            val_goal_features[:, TARGET_FEATURE_START:TARGET_FEATURE_STOP], val_goal_targets[:, :2], args,
        )
        goal_angle_decoder, goal_angle_decoder_training = train_orientation_decoder(
            train_goal_features, train_goal_targets[:, 2:4],
            val_goal_features, val_goal_targets[:, 2:4], args,
        )
        decoded_goal_xy = predict_orientation(
            goal_xy_decoder,
            val_goal_features[:, TARGET_FEATURE_START:TARGET_FEATURE_STOP],
            goal_xy_decoder_training, args.device,
        )
        decoded_goal_angle = predict_orientation(
            goal_angle_decoder, val_goal_features, goal_angle_decoder_training, args.device,
        )
        goal_xy_decoder_error = float(np.mean(np.linalg.norm(decoded_goal_xy - val_goal_targets[:, :2], axis=1)))
        goal_angle_decoder_error = angle_error_batch(decoded_goal_angle, val_goal_targets[:, 2:4])
        direct_goal_xy = (
            val_goal_features[:, TARGET_FEATURE_START + 1:TARGET_FEATURE_START + 3] + 1.0
        ) * (PUSHT_STATE_SIZE / 2.0)
        direct_goal_xy_error = float(np.mean(
            np.linalg.norm(direct_goal_xy - val_goal_targets[:, :2], axis=1)
        ))
        goal_component_selection = {
            # The joint decoder is already an independently validated
            # candidate.  A component decoder may replace it only when it
            # beats *all* available references, including the joint decoder;
            # comparing only against the frozen head previously allowed a
            # 105px component readout to overwrite a 54px joint readout.
            "xy": (
                selected_goal_decoder is not None
                and goal_readout_metrics["xy_mean_error"] < base_goal_readout_metrics["xy_mean_error"]
            ) or goal_xy_decoder_error < min(
                base_goal_readout_metrics["xy_mean_error"],
                direct_goal_xy_error,
                goal_readout_metrics["xy_mean_error"],
            ),
            "xy_source": "joint_decoder" if (
                selected_goal_decoder is not None
                and goal_readout_metrics["xy_mean_error"] < base_goal_readout_metrics["xy_mean_error"]
                and goal_readout_metrics["xy_mean_error"] <= goal_xy_decoder_error
                and goal_readout_metrics["xy_mean_error"] <= direct_goal_xy_error
            ) else "direct_moment" if direct_goal_xy_error < min(
                base_goal_readout_metrics["xy_mean_error"],
                goal_xy_decoder_error,
                goal_readout_metrics["xy_mean_error"],
            ) else "learned",
            "angle": (
                selected_goal_decoder is not None
                and goal_readout_metrics["angle_mean_error"] < base_goal_readout_metrics["angle_mean_error"]
            ) or goal_angle_decoder_error < min(
                base_goal_readout_metrics["angle_mean_error"],
                goal_readout_metrics["angle_mean_error"],
            ),
            "angle_blend_weight": 1.0,
        }
        rgb_goal_xy_decoder: OrientationDecoder | None = None
        rgb_goal_xy_training: dict[str, Any] | None = None
        rgb_goal_xy_metrics: dict[str, Any] = {
            "enabled": bool(args.goal_observation == "aligned_state_goal"),
            "selected_for_runtime": False,
            "selection_max_error": float(args.rgb_goal_xy_max_error),
        }
        aligned_goal_train_images: np.ndarray | None = None
        aligned_goal_val_images: np.ndarray | None = None
        aligned_goal_train_shapes: np.ndarray | None = None
        aligned_goal_val_shapes: np.ndarray | None = None
        if args.goal_observation == "aligned_state_goal":
            train_root_indices = np.asarray([
                np.flatnonzero(train["root_id"] == root)[0]
                for root in range(len(train_roots))
            ], dtype=np.int64)
            val_root_indices = np.asarray([
                np.flatnonzero(val["root_id"] == root)[0]
                for root in range(len(val_roots))
            ], dtype=np.int64)
            # Keep the original 224x224 render for target-pose calibration.
            # The 64x64 tensor used by the frozen encoder aliases the small
            # centroid/PCA offsets of the T mask and was the source of the
            # apparently good average gate but poor per-episode target pose.
            aligned_goal_train_images = root_goal_observation_images(
                env, train_roots, args.goal_observation,
            )
            aligned_goal_val_images = root_goal_observation_images(
                env, val_roots, args.goal_observation,
            )
            aligned_goal_train_shapes = rgb_geometry_shape_features(
                np.moveaxis(aligned_goal_train_images, -1, 1), "target",
            )
            aligned_goal_val_shapes = rgb_geometry_shape_features(
                np.moveaxis(aligned_goal_val_images, -1, 1), "target",
            )
            rgb_goal_xy_decoder, rgb_goal_xy_training = train_rgb_pose_calibrator(
                aligned_goal_train_shapes,
                train_goal_targets[:, :2],
                aligned_goal_val_shapes,
                val_goal_targets[:, :2],
                ridge_alpha=float(args.rgb_pose_ridge_alpha),
            )
            rgb_goal_xy_prediction = predict_orientation(
                rgb_goal_xy_decoder,
                aligned_goal_val_shapes,
                rgb_goal_xy_training,
                args.device,
            )
            rgb_goal_xy_error = float(np.mean(
                np.linalg.norm(rgb_goal_xy_prediction - val_goal_targets[:, :2], axis=1),
            ))
            rgb_goal_xy_metrics.update({
                "validation_mean_error": rgb_goal_xy_error,
                "validation_roots": int(len(val_goal_targets)),
                "interface": "scene_aligned_target_rgb_shape_calibration",
            })
            rgb_goal_xy_selected = bool(
                args.goal_xy_policy == "rgb_calibrated"
                and rgb_goal_xy_error <= float(args.rgb_goal_xy_max_error)
            )
            if rgb_goal_xy_selected:
                goal_component_selection["xy"] = False
                goal_component_selection["xy_source"] = "rgb_calibrated"
                rgb_goal_xy_metrics["selected_for_runtime"] = True
        if args.goal_xy_policy == "rgb_calibrated" and not rgb_goal_xy_metrics["selected_for_runtime"]:
            # Keep the requested calibrated interface fail-closed if the
            # root-disjoint error gate is not met.  When the runtime current
            # observer is already an RGB chart, however, falling back to the
            # latent goal head creates a hidden coordinate-system mismatch:
            # current XY is measured in the rendered 512x512 canvas while the
            # latent goal head is calibrated for the checkpoint's goal-image
            # convention.  The mismatch can make a physically solved block
            # appear more than 100 px from the target and suppress the
            # terminal interface.  Pair the failed calibrated goal readout
            # with the same raw RGB target chart instead; this is conservative
            # (and separately auditable) rather than silently mixing gauges.
            if args.current_xy_policy in {"rgb_mask", "rgb_calibrated"}:
                goal_component_selection["xy"] = False
                goal_component_selection["xy_source"] = "rgb_mask"
                rgb_goal_xy_metrics["fallback_pairing"] = "raw_rgb_target_with_rgb_current_observer"
            else:
                goal_component_selection["xy_source"] = "learned"
        if args.goal_xy_policy == "rgb_mask":
            goal_component_selection["xy"] = False
            goal_component_selection["xy_source"] = "rgb_mask"
        # The raw directed target-mask PCA is not a calibrated control
        # interface.  Keep the base/learned goal readout unless the separate
        # root-disjoint RGB angle decoder below passes its gate.
        goal_component_selection["angle_source"] = "model"
        if goal_component_selection["xy_source"] == "direct_moment":
            goal_component_selection["xy"] = False
        goal_selection["component_decoders"] = {
            "xy_mean_error": goal_xy_decoder_error,
            "direct_moment_xy_mean_error": direct_goal_xy_error,
            "angle_mean_error": goal_angle_decoder_error,
            "selected_components": {
                "xy": (
                    "joint_goal_decoder"
                    if goal_component_selection["xy_source"] == "joint_decoder"
                    else
                    "direct_rgb_block_moment_chart"
                    if goal_component_selection["xy_source"] == "direct_moment"
                    else "scene_aligned_rgb_target_calibration"
                    if goal_component_selection["xy_source"] == "rgb_calibrated"
                    else "block_moment_two_output_decoder"
                    if goal_component_selection["xy"]
                    else "base_checkpoint_goal_head"
                ),
                "angle": (
                    "joint_goal_decoder"
                    if selected_goal_decoder is not None
                    and goal_component_selection["angle"]
                    and goal_readout_metrics["angle_mean_error"] <= goal_angle_decoder_error
                    else "independent_two_output_decoder"
                    if goal_component_selection["angle"]
                    else "base_checkpoint_goal_head"
                ),
            },
        }
        print(json.dumps({"phase": "goal_decoder", **goal_readout_metrics}), flush=True)
        orientation_decoder, orientation_decoder_training = train_orientation_decoder(
            readout_train_features, readout_train_orientation_targets,
            readout_val_features, readout_val_orientation_targets, args,
        )
        decoded_orientation = predict_orientation(
            orientation_decoder, readout_val_features, orientation_decoder_training, args.device,
        )
        orientation_readout_metrics = {
            "angle_mean_error": angle_error_batch(decoded_orientation, readout_val_orientation_targets),
            "roots": int(len(readout_val_orientation_targets)),
            "static_validation_roots": int(len(val_orientation_targets)),
            "dynamic_readout_augmentation": dynamic_readout_metrics,
        }
        base_orientations = base_orientation_rows(model, normalizer, val, args.device)
        base_orientation_metrics = {
            "angle_mean_error": angle_error_batch(base_orientations, val_orientation_targets),
            "roots": int(len(val_orientation_targets)),
        }
        selected_orientation_decoder = (
            orientation_decoder
            if orientation_readout_metrics["angle_mean_error"] <= base_orientation_metrics["angle_mean_error"]
            else None
        )
        selected_orientation_training = (
            orientation_decoder_training if selected_orientation_decoder is not None else None
        )
        orientation_selection = {
            "selected": "independent_decoder" if selected_orientation_decoder is not None else "base_checkpoint_state_head",
            "decoder_metrics": orientation_readout_metrics,
            "base_metrics": base_orientation_metrics,
        }
        # The same RGB block-orientation readout is also a candidate for the
        # rendered goal image.  This is a transfer candidate, not an oracle:
        # it is trained only on current-image labels and is accepted only if
        # its goal-image error is lower on the held-out validation roots.
        transferred_goal_angle_error = float("inf")
        if selected_orientation_decoder is not None and selected_orientation_training is not None:
            transferred_goal_angle = predict_orientation(
                selected_orientation_decoder, val_goal_features,
                selected_orientation_training, args.device,
            )
            base_goal_angle = base_goals[:, 2:4]
            blend_candidates = []
            for weight in np.linspace(0.0, 1.0, 11):
                blended = (1.0 - weight) * base_goal_angle + weight * transferred_goal_angle
                blend_candidates.append((
                    angle_error_batch(blended, val_goal_targets[:, 2:4]), float(weight),
                ))
            transferred_goal_angle_error = angle_error_batch(
                transferred_goal_angle, val_goal_targets[:, 2:4],
            )
            blend_error, blend_weight = min(blend_candidates, key=lambda item: item[0])
            current_best = base_goal_readout_metrics["angle_mean_error"]
            if goal_component_selection["angle"]:
                current_best = min(current_best, goal_angle_decoder_error)
            disagreement = np.abs(wrap(
                np.arctan2(transferred_goal_angle[:, 0], transferred_goal_angle[:, 1])
                - np.arctan2(base_goal_angle[:, 0], base_goal_angle[:, 1])
            ))
            gated_candidates = []
            for threshold in np.linspace(0.0, math.pi, 33):
                gated = np.where(
                    (disagreement <= threshold)[:, None],
                    transferred_goal_angle,
                    base_goal_angle,
                )
                gated_candidates.append((
                    angle_error_batch(gated, val_goal_targets[:, 2:4]), float(threshold),
                ))
            gated_error, gated_threshold = min(gated_candidates, key=lambda item: item[0])
            if gated_error < min(current_best, blend_error):
                goal_angle_decoder = selected_orientation_decoder
                goal_angle_decoder_training = selected_orientation_training
                goal_component_selection["angle"] = True
                goal_component_selection["angle_blend_weight"] = 1.0
                goal_component_selection["angle_disagreement_threshold"] = gated_threshold
                goal_selection["component_decoders"]["angle_mean_error"] = gated_error
                goal_selection["component_decoders"]["angle_source"] = "current_orientation_decoder_disagreement_gate"
                goal_selection["component_decoders"]["angle_disagreement_threshold"] = gated_threshold
            elif blend_error < current_best:
                goal_angle_decoder = selected_orientation_decoder
                goal_angle_decoder_training = selected_orientation_training
                goal_component_selection["angle"] = True
                goal_component_selection["angle_blend_weight"] = blend_weight
                goal_selection["component_decoders"]["angle_mean_error"] = blend_error
                goal_selection["component_decoders"]["angle_source"] = "current_orientation_decoder_validation_blend"
                goal_selection["component_decoders"]["angle_blend_weight"] = blend_weight
        orientation_selection["goal_transfer_angle_mean_error"] = transferred_goal_angle_error
        goal_selection["selected_components"] = {
            "xy": goal_selection["component_decoders"]["selected_components"]["xy"],
            "angle": goal_selection["component_decoders"].get(
                "angle_source",
                goal_selection["component_decoders"]["selected_components"]["angle"],
            ),
        }
        print(json.dumps({"phase": "orientation_decoder", **orientation_readout_metrics}), flush=True)
        train_xy_targets = np.asarray([root[0][2:4] for root in train_roots], dtype=np.float32)
        val_xy_targets = np.asarray([root[0][2:4] for root in val_roots], dtype=np.float32)
        current_xy_decoder, current_xy_decoder_training = train_orientation_decoder(
            readout_train_features, readout_train_xy_targets,
            readout_val_features, readout_val_xy_targets, args,
        )
        decoded_xy = predict_orientation(
            current_xy_decoder, readout_val_features, current_xy_decoder_training, args.device,
        )
        decoded_xy_error = np.linalg.norm(decoded_xy - readout_val_xy_targets, axis=1)
        moment_xy = (readout_val_features[:, 11:13] + 1.0) * (PUSHT_STATE_SIZE / 2.0)
        moment_xy_error = np.linalg.norm(moment_xy - readout_val_xy_targets, axis=1)
        current_xy_metrics = {
            "decoder_mean_error": float(np.mean(decoded_xy_error)),
            "moment_mean_error": float(np.mean(moment_xy_error)),
            "decoder_rmse": float(np.sqrt(np.mean(np.square(decoded_xy - readout_val_xy_targets)))),
            "roots": int(len(readout_val_xy_targets)),
            "static_validation_roots": int(len(val_xy_targets)),
            "dynamic_readout_augmentation": dynamic_readout_metrics,
        }
        # A decoder selected by a tiny static-root gain can be catastrophically
        # non-robust after the agent moves.  The runtime interface is therefore
        # fail-closed: the learned readout must beat the raw RGB chart by a
        # declared relative margin, rather than merely by one validation pixel.
        required_decoder_error = (
            (1.0 - float(args.current_xy_min_relative_gain))
            * current_xy_metrics["moment_mean_error"]
        )
        use_current_xy_decoder = current_xy_metrics["decoder_mean_error"] <= required_decoder_error
        current_xy_metrics["selection_rule"] = "decoder_error <= (1 - min_relative_gain) * moment_error"
        current_xy_metrics["min_relative_gain"] = float(args.current_xy_min_relative_gain)
        current_xy_metrics["required_decoder_error"] = float(required_decoder_error)
        print(json.dumps({"phase": "current_xy_decoder", **current_xy_metrics, "selected": use_current_xy_decoder}), flush=True)
        # The agent and block must be decoded in one declared chart.  A block
        # decoder alone is insufficient: contact distance, normal, side
        # alignment, reacquisition, and terminal confirmation all compare the
        # two points.  Fit a root-disjoint agent decoder with the identical
        # feature interface and dynamic augmentation protocol, then gate it
        # against the frozen agent moment readout using the same conservative
        # relative-improvement rule.
        agent_xy_decoder, agent_xy_decoder_training = train_linear_feature_xy_decoder(
            readout_train_features, readout_train_agent_xy_targets,
            ridge_alpha=float(args.rgb_pose_ridge_alpha),
        )
        decoded_agent_xy = predict_orientation(
            agent_xy_decoder, readout_val_features,
            agent_xy_decoder_training, args.device,
        )
        agent_moment_xy = (readout_val_features[:, 1:3] + 1.0) * (PUSHT_STATE_SIZE / 2.0)
        agent_xy_errors = np.linalg.norm(
            decoded_agent_xy - readout_val_agent_xy_targets, axis=1,
        )
        agent_moment_errors = np.linalg.norm(
            agent_moment_xy - readout_val_agent_xy_targets, axis=1,
        )
        agent_xy_metrics = {
            "decoder_mean_error": float(np.mean(agent_xy_errors)),
            "moment_mean_error": float(np.mean(agent_moment_errors)),
            "decoder_rmse": float(np.sqrt(np.mean(np.square(
                decoded_agent_xy - readout_val_agent_xy_targets,
            )))),
            "roots": int(len(readout_val_agent_xy_targets)),
            "static_validation_roots": int(len(val_agent_xy_targets)),
            "dynamic_readout_augmentation": dynamic_readout_metrics,
        }
        required_agent_decoder_error = max(
            float(args.agent_xy_max_error),
            (1.0 - float(args.current_xy_min_relative_gain))
            * agent_xy_metrics["moment_mean_error"],
        )
        use_agent_xy_decoder = (
            agent_xy_metrics["decoder_mean_error"] <= required_agent_decoder_error
        )
        agent_xy_metrics.update({
            "selection_rule": "decoder_error <= max(agent_absolute_error_ceiling, (1 - min_relative_gain) * moment_error)",
            "min_relative_gain": float(args.current_xy_min_relative_gain),
            "absolute_error_ceiling": float(args.agent_xy_max_error),
            "required_decoder_error": float(required_agent_decoder_error),
        })
        # Both outputs are supervised in the same simulator-pixel chart.  The
        # agent moment is already a high-quality member of that chart; forcing
        # a learned affine replacement when it is worse would create a new
        # interface error.  Therefore a pair is accepted when the block
        # decoder passes and the agent is either the learned affine candidate
        # or the independently validated raw moment chart.
        agent_raw_chart_safe = bool(
            agent_xy_metrics["moment_mean_error"] <= float(args.agent_xy_max_error)
        )
        use_joint_xy_decoder = bool(
            use_current_xy_decoder and (use_agent_xy_decoder or agent_raw_chart_safe)
        )
        agent_xy_metrics["selected"] = bool(use_agent_xy_decoder)
        agent_xy_metrics["raw_moment_chart_safe"] = agent_raw_chart_safe
        agent_xy_metrics["selected_source"] = (
            "learned_affine_agent_decoder" if use_agent_xy_decoder
            else "raw_agent_moment_same_physical_xy_chart" if agent_raw_chart_safe
            else "none"
        )
        agent_xy_metrics["joint_pair_selected"] = use_joint_xy_decoder
        print(json.dumps({"phase": "agent_xy_decoder", **agent_xy_metrics}), flush=True)
        rgb_current_xy_decoder: LinearRGBPoseCalibrator | None = None
        rgb_current_xy_training: dict[str, Any] | None = None
        rgb_current_xy_metrics: dict[str, Any] = {
            "enabled": True,
            "selected_for_runtime": False,
            "selection_max_error": float(args.rgb_current_xy_max_error),
            "interface": "root_disjoint_current_rgb_shape_calibration",
        }
        current_train_images = np.asarray([root[2] for root in train_roots], dtype=np.uint8)
        current_val_images = np.asarray([root[2] for root in val_roots], dtype=np.uint8)
        current_train_shapes = rgb_geometry_shape_features(
            np.moveaxis(current_train_images, -1, 1), "block",
        )
        current_val_shapes = rgb_geometry_shape_features(
            np.moveaxis(current_val_images, -1, 1), "block",
        )
        dynamic_current_xy_metrics: dict[str, Any] = {
            "enabled": bool(args.readout_augmentation_steps > 0),
            "steps_per_root": int(args.readout_augmentation_steps),
            "train_rows": 0,
            "validation_rows": 0,
        }
        calibration_train_shapes = current_train_shapes
        calibration_train_targets = train_xy_targets
        calibration_val_shapes = current_val_shapes
        calibration_val_targets = val_xy_targets
        if args.readout_augmentation_steps > 0:
            dynamic_train_shapes, dynamic_train_xy = dynamic_rgb_shape_xy_rows(
                env, train_roots, data_seed + 640_000,
                args.readout_augmentation_steps,
            )
            dynamic_val_shapes, dynamic_val_xy = dynamic_rgb_shape_xy_rows(
                env, val_roots, data_seed + 650_000,
                args.readout_augmentation_steps,
            )
            calibration_train_shapes = np.concatenate(
                [current_train_shapes, dynamic_train_shapes], axis=0,
            )
            calibration_train_targets = np.concatenate(
                [train_xy_targets, dynamic_train_xy], axis=0,
            )
            calibration_val_shapes = np.concatenate(
                [current_val_shapes, dynamic_val_shapes], axis=0,
            )
            calibration_val_targets = np.concatenate(
                [val_xy_targets, dynamic_val_xy], axis=0,
            )
            dynamic_current_xy_metrics.update({
                "train_rows": int(len(dynamic_train_shapes)),
                "validation_rows": int(len(dynamic_val_shapes)),
                "selection_validation_distribution": "static roots plus root-disjoint offline rollout frames",
            })
        if args.rgb_current_xy_readout == "knn":
            rgb_current_xy_decoder, rgb_current_xy_training = train_rgb_angle_knn(
                calibration_train_shapes, calibration_train_targets, k=4,
            )
            rgb_current_xy_decoder = rgb_current_xy_decoder.to(args.device)
            rgb_current_xy_training["feature_interface"] = "moments"
            rgb_current_xy_training["feature_role"] = "block"
            rgb_current_xy_training["readout"] = "support_constrained_knn_xy"
        else:
            rgb_current_xy_decoder, rgb_current_xy_training = train_rgb_pose_calibrator(
                calibration_train_shapes, calibration_train_targets,
                calibration_val_shapes, calibration_val_targets,
                ridge_alpha=float(args.rgb_pose_ridge_alpha),
            )
        rgb_current_xy_prediction = predict_orientation(
            rgb_current_xy_decoder, calibration_val_shapes,
            rgb_current_xy_training, args.device,
        )
        rgb_current_xy_error = float(np.mean(
            np.linalg.norm(rgb_current_xy_prediction - calibration_val_targets, axis=1),
        ))
        rgb_current_xy_errors = np.linalg.norm(
            rgb_current_xy_prediction - calibration_val_targets, axis=1,
        )
        rgb_current_xy_metrics.update({
            "validation_mean_error": rgb_current_xy_error,
            "validation_p95_error": float(np.quantile(rgb_current_xy_errors, 0.95)),
            "validation_max_error": float(np.max(rgb_current_xy_errors, initial=0.0)),
            "validation_roots": int(len(calibration_val_targets)),
            "readout": str(args.rgb_current_xy_readout),
            "dynamic_validation": dynamic_current_xy_metrics,
        })
        paired_rgb_current_xy_metrics: dict[str, Any] = {
            "enabled": bool(args.current_xy_policy == "paired_rgb_calibrated"),
            "selected_for_runtime": False,
            "selection_protocol": "root_disjoint_paired_current_goal_rigid_chart",
        }
        paired_current_xy_validation: np.ndarray | None = None
        if (
            args.current_xy_policy == "paired_rgb_calibrated"
            and aligned_goal_val_images is not None
            and rgb_goal_xy_decoder is not None
            and rgb_goal_xy_training is not None
        ):
            paired_goal_xy = predict_orientation(
                rgb_goal_xy_decoder,
                aligned_goal_val_shapes
                if aligned_goal_val_shapes is not None
                else rgb_geometry_shape_features(
                    np.moveaxis(aligned_goal_val_images, -1, 1), "target",
                ),
                rgb_goal_xy_training,
                args.device,
            )
            paired_predictions: list[np.ndarray] = []
            for index, goal_xy in zip(range(len(val_roots)), paired_goal_xy):
                estimate = paired_rgb_current_xy_from_images(
                    np.asarray(val_roots[index][2], dtype=np.uint8),
                    np.asarray(aligned_goal_val_images[index], dtype=np.uint8),
                    goal_xy,
                )
                if estimate is None:
                    estimate = (val_current_features[index, BLOCK_FEATURE_START + 1:BLOCK_FEATURE_START + 3] + 1.0) * (
                        PUSHT_STATE_SIZE / 2.0
                    )
                paired_predictions.append(np.asarray(estimate, dtype=np.float32))
            paired_predictions_array = np.asarray(paired_predictions, dtype=np.float32)
            paired_current_xy_validation = paired_predictions_array
            paired_errors = np.linalg.norm(paired_predictions_array - val_xy_targets, axis=1)
            paired_rgb_current_xy_metrics.update({
                "interface": "paired_current_block_and_isolated_goal_target_rigid_chart",
                "validation_mean_error": float(np.mean(paired_errors)),
                "validation_p95_error": float(np.quantile(paired_errors, 0.95)),
                "validation_max_error": float(np.max(paired_errors, initial=0.0)),
                "validation_roots": int(len(paired_errors)),
                "selection_max_error": float(args.paired_rgb_current_xy_max_error),
                "selection_p95_max_error": float(args.paired_rgb_current_xy_p95_max_error),
            })
            paired_rgb_current_xy_metrics["selected_for_runtime"] = bool(
                paired_rgb_current_xy_metrics["validation_mean_error"]
                <= float(args.paired_rgb_current_xy_max_error)
                and paired_rgb_current_xy_metrics["validation_p95_error"]
                <= float(args.paired_rgb_current_xy_p95_max_error)
            )
        static_fallback_requested = bool(
            getattr(args, "rgb_current_xy_static_fallback", False)
            and args.readout_augmentation_steps == 0
        )
        use_rgb_current_xy = bool(
            args.current_xy_policy == "rgb_calibrated"
            and rgb_current_xy_error <= float(rgb_current_xy_metrics["selection_max_error"])
            and float(rgb_current_xy_metrics["validation_p95_error"])
            <= float(args.rgb_current_xy_p95_max_error)
            and (args.readout_augmentation_steps > 0 or static_fallback_requested)
        )
        rgb_current_xy_metrics["selection_protocol"] = (
            "dynamic_root_disjoint_gate"
            if args.readout_augmentation_steps > 0
            else "static_root_disjoint_opt_in_fallback"
            if static_fallback_requested
            else "dynamic_required_fail_closed"
        )
        rgb_current_xy_metrics["selected_for_runtime"] = use_rgb_current_xy
        rgb_current_xy_metrics["selection_p95_max_error"] = float(
            args.rgb_current_xy_p95_max_error
        )
        print(json.dumps({"phase": "rgb_current_xy_calibration", **rgb_current_xy_metrics}), flush=True)
        print(json.dumps({"phase": "paired_rgb_current_xy_calibration", **paired_rgb_current_xy_metrics}), flush=True)
        # Select the visual terminal-phase threshold on validation roots only.
        # False terminal positives are more damaging than false negatives: a
        # false positive switches the controller to torque repair while the
        # physical block is still far from the target.  The strict evaluator
        # uses a 20 px XY gate, but the RGB readout is noisy, so the runtime
        # threshold is an explicitly audited interface parameter rather than
        # an unexamined copy of the evaluator threshold.
        # Terminal-threshold selection must use the exact observer that will
        # be called online.  The earlier code always used the latent decoder
        # here and fell back to ``base_goals`` even when runtime used the
        # scene-aligned RGB goal calibration, creating a train/validation vs.
        # runtime interface mismatch and false terminal positives.
        val_root_indices = np.asarray([
            np.flatnonzero(val["root_id"] == root)[0]
            for root in range(len(val_roots))
        ], dtype=np.int64)
        if args.terminal_xy_observer == "decoder" and current_xy_decoder is not None and current_xy_decoder_training is not None:
            validation_current_xy = predict_orientation(
                current_xy_decoder, val_current_features, current_xy_decoder_training, args.device,
            )
        elif args.current_xy_policy == "paired_rgb_calibrated" and paired_current_xy_validation is not None:
            validation_current_xy = paired_current_xy_validation
        elif args.current_xy_policy == "rgb_mask":
            validation_current_xy = np.asarray([
                rgb_geometry_xy_angle_from_image(
                    np.moveaxis(val["image"][index], 0, 2), "block",
                )[0]
                for index in val_root_indices
            ], dtype=np.float32)
        elif args.current_xy_policy == "rgb_mask_fused":
            validation_current_xy = np.asarray([
                observed_block_xy(
                    np.moveaxis(val["image"][index], 0, 2),
                    # ``val_current_features`` is root-level (one row per
                    # validation root), whereas ``index`` indexes the full
                    # intervention table.  Mixing these two index spaces
                    # silently corrupted the fused observer audit and could
                    # even index past the root-level feature array.
                    val_current_features[root_position],
                    current_xy_decoder,
                    current_xy_decoder_training,
                    args.device,
                    "rgb_mask_fused",
                    args.current_xy_disagreement_threshold,
                )
                for root_position, index in enumerate(val_root_indices)
            ], dtype=np.float32)
        elif use_rgb_current_xy:
            validation_current_xy = predict_orientation(
                rgb_current_xy_decoder, current_val_shapes,
                rgb_current_xy_training, args.device,
            )
        elif use_current_xy_decoder:
            validation_current_xy = predict_orientation(
                current_xy_decoder, val_current_features, current_xy_decoder_training, args.device,
            )
        else:
            validation_current_xy = (val_current_features[:, 11:13] + 1.0) * (PUSHT_STATE_SIZE / 2.0)
        # Select the goal XY readout by held-out goal-pose error.  Runtime
        # current-to-goal distance is retained as an audit quantity, but is
        # not allowed to hide a shared chart offset by subtracting two wrong
        # readouts.  The earlier selector did exactly that and could promote
        # a visually close yet physically wrong goal interface.
        if args.goal_xy_policy == "model":
            goal_interface_candidates: dict[str, np.ndarray] = {
                "base_checkpoint_goal_head": base_goals[:, :2],
                "block_goal_decoder": decoded_goal_xy,
                "direct_goal_moment": direct_goal_xy,
            }
            if selected_goal_decoder is not None:
                goal_interface_candidates["joint_goal_decoder"] = decoded_goals[:, :2]
            goal_pose_errors = {
                name: float(np.mean(np.linalg.norm(candidate - val_goal_targets[:, :2], axis=1)))
                for name, candidate in goal_interface_candidates.items()
            }
            runtime_geometry_errors = {
                name: float(np.mean(np.linalg.norm(candidate - validation_current_xy, axis=1)))
                for name, candidate in goal_interface_candidates.items()
            }
            selected_goal_source = min(goal_pose_errors, key=goal_pose_errors.get)
            goal_interface_selection = {
                "selection_status": "held_out_goal_pose_error_with_runtime_geometry_audit",
                "candidate_mean_goal_pose_error": goal_pose_errors,
                "candidate_mean_current_goal_distance": runtime_geometry_errors,
                "selected_source": selected_goal_source,
                "selection_uses_training_only": True,
            }
            if selected_goal_source == "joint_goal_decoder":
                goal_component_selection.update({"xy": True, "xy_source": "joint_decoder"})
            elif selected_goal_source == "block_goal_decoder":
                goal_component_selection.update({"xy": True, "xy_source": "learned"})
            elif selected_goal_source == "direct_goal_moment":
                goal_component_selection.update({"xy": False, "xy_source": "direct_moment"})
            else:
                goal_component_selection.update({"xy": False, "xy_source": "learned"})
        if goal_component_selection.get("xy_source") == "rgb_calibrated":
            if rgb_goal_xy_decoder is None or rgb_goal_xy_training is None:
                raise RuntimeError("rgb_calibrated goal observer was selected without a decoder")
            if aligned_goal_val_shapes is not None:
                validation_goal_xy = predict_orientation(
                    rgb_goal_xy_decoder,
                    aligned_goal_val_shapes,
                    rgb_goal_xy_training,
                    args.device,
                )
            else:
                validation_goal_xy = predict_orientation(
                    rgb_goal_xy_decoder,
                    rgb_geometry_shape_features(
                        val["goal_image"][val_root_indices], "target",
                    ),
                    rgb_goal_xy_training,
                    args.device,
                )
        elif goal_component_selection.get("xy_source") == "rgb_mask":
            validation_goal_xy = np.asarray([
                rgb_geometry_xy_angle_from_image(
                    np.moveaxis(val["goal_image"][index], 0, 2), "target",
                )[0]
                for index in val_root_indices
            ], dtype=np.float32)
        elif goal_component_selection.get("xy_source") == "joint_decoder":
            validation_goal_xy = decoded_goals[:, :2]
        elif goal_component_selection["xy"]:
            validation_goal_xy = predict_orientation(
                goal_xy_decoder,
                val_goal_features[:, TARGET_FEATURE_START:TARGET_FEATURE_STOP],
                goal_xy_decoder_training, args.device,
            )
        else:
            validation_goal_xy = base_goals[:, :2]
        validation_visual_xy = np.linalg.norm(validation_goal_xy - validation_current_xy, axis=1)
        validation_true_xy = np.asarray(
            [np.linalg.norm(root[0][2:4] - root[1][2:4]) for root in val_roots], dtype=np.float32,
        )
        terminal_threshold_rows = []
        terminal_truth = validation_true_xy <= 20.0
        for threshold in (16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 35.0):
            predicted_terminal = validation_visual_xy <= threshold
            false_positive = int(np.sum(predicted_terminal & ~terminal_truth))
            false_negative = int(np.sum(~predicted_terminal & terminal_truth))
            terminal_threshold_rows.append({
                "threshold": float(threshold),
                "false_positive": false_positive,
                "false_negative": false_negative,
                "weighted_error": float(4 * false_positive + false_negative),
                "predicted_terminal": int(np.sum(predicted_terminal)),
                "true_terminal": int(np.sum(terminal_truth)),
            })
        if args.terminal_threshold_policy == "fixed":
            selected_terminal_threshold = {
                "threshold": float(args.terminal_xy),
                "false_positive": int(np.sum((validation_visual_xy <= args.terminal_xy) & ~terminal_truth)),
                "false_negative": int(np.sum((validation_visual_xy > args.terminal_xy) & terminal_truth)),
                "weighted_error": float(
                    4 * np.sum((validation_visual_xy <= args.terminal_xy) & ~terminal_truth)
                    + np.sum((validation_visual_xy > args.terminal_xy) & terminal_truth)
                ),
                "predicted_terminal": int(np.sum(validation_visual_xy <= args.terminal_xy)),
                "true_terminal": int(np.sum(terminal_truth)),
            }
            terminal_selection_status = "fixed_declared_threshold"
        elif bool(np.any(terminal_truth)):
            selected_terminal_threshold = min(
                terminal_threshold_rows,
                key=lambda row: (row["weighted_error"], row["threshold"]),
            )
            terminal_selection_status = "validation_selected"
        else:
            # There is no information in a validation split with zero strict
            # terminal positives.  Do not let every threshold tie and silently
            # select the smallest tube; retain the evaluator's 20 px target
            # and mark the interface selection as underpowered.
            selected_terminal_threshold = next(
                row for row in terminal_threshold_rows if row["threshold"] == 20.0
            )
            terminal_selection_status = "no_positive_validation_cases_defaulted_to_strict_target"
        args.terminal_xy = float(selected_terminal_threshold["threshold"])
        terminal_threshold_selection = {
            "selected_threshold": float(args.terminal_xy),
            "selection_status": terminal_selection_status,
            "strict_xy_target": 20.0,
            "false_positive_weight": 4.0,
            "validation_rows": terminal_threshold_rows,
            "validation_visual_xy_mean": float(np.mean(validation_visual_xy)),
            "validation_true_xy_mean": float(np.mean(validation_true_xy)),
        }
        print(json.dumps({"phase": "terminal_threshold", **terminal_threshold_selection}), flush=True)
        train_sample_weights = np.where(
            train["contact_root"], float(args.contact_loss_weight), 1.0,
        ).astype(np.float32)
        terminal_train_start = len(train_roots) - train_terminal - train_bootstrap
        terminal_root_mask = np.zeros(len(train_roots), dtype=bool)
        if train_terminal > 0:
            terminal_root_mask[terminal_train_start:terminal_train_start + train_terminal] = True
        terminal_rows = terminal_root_mask[train["root_id"]]
        train_sample_weights *= np.where(
            terminal_rows,
            max(1.0, float(args.terminal_loss_weight)),
            1.0,
        ).astype(np.float32)
        adapter_targets = train["local_target_contact"] if args.effect_frame == "contact" else train["local_target"]
        val_adapter_targets = val["local_target_contact"] if args.effect_frame == "contact" else val["local_target"]
        adapter, adapter_training = train_adapter(
            train_features, adapter_targets, val_features, val_adapter_targets,
            train_sample_weights, args,
        )
        adapter_training["effect_frame"] = args.effect_frame
        adapter_training["terminal_loss_weight"] = float(args.terminal_loss_weight)
        adapter_training["terminal_training_rows"] = int(np.sum(terminal_rows))
        # Contact and non-contact interventions have different local maps. A
        # single regressor can fit the dominant passive regime while still
        # mis-ranking the actions used after approach. Fit a contact-only
        # specialist and use it only for the closed-loop adapter controller.
        contact_train_mask = np.asarray(train["contact_root"], dtype=bool)
        contact_val_mask = np.asarray(val["contact_root"], dtype=bool)
        contact_adapter = None
        contact_adapter_training = None
        contact_adapter_metrics: dict[str, Any] = {
            "selected_for_runtime": False,
            "train_rows": int(np.sum(contact_train_mask)),
            "val_rows": int(np.sum(contact_val_mask)),
        }
        contact_knn_training: dict[str, Any] | None = None
        contact_knn_metrics: dict[str, Any] = {
            "enabled": bool(args.contact_canonical_knn),
            "selected_for_runtime": False,
            "selection_status": "disabled" if not args.contact_canonical_knn else "pending_validation_regret",
        }
        if args.contact_canonical_knn and int(np.sum(contact_train_mask)) >= len(actions):
            contact_knn_training = fit_contact_canonical_knn(
                train_features[contact_train_mask],
                train["local_target_contact"][contact_train_mask],
                train["root_id"][contact_train_mask],
                args.contact_knn_k, args.contact_knn_context_weight,
                args.contact_knn_action_weight,
            )
            contact_knn_metrics.update({
                "train_rows": int(np.sum(contact_train_mask)),
                "train_roots": int(len(np.unique(train["root_id"][contact_train_mask]))),
                "k": int(args.contact_knn_k),
                "context_weight": float(args.contact_knn_context_weight),
                "action_weight": float(args.contact_knn_action_weight),
            })
        contact_val_prediction: np.ndarray | None = None
        if int(np.sum(contact_train_mask)) >= len(actions) and int(np.sum(contact_val_mask)) >= len(actions):
            if args.structured_contact_adapter:
                contact_adapter, contact_adapter_training = train_structured_contact_adapter(
                    train_features[contact_train_mask], adapter_targets[contact_train_mask],
                    val_features[contact_val_mask], val_adapter_targets[contact_val_mask], args,
                )
            else:
                contact_adapter, contact_adapter_training = train_adapter(
                    train_features[contact_train_mask], adapter_targets[contact_train_mask],
                    val_features[contact_val_mask], val_adapter_targets[contact_val_mask],
                    np.ones(int(np.sum(contact_train_mask)), dtype=np.float32), args,
                )
            contact_adapter_training["effect_frame"] = args.effect_frame
            contact_prediction = predict_targets(
                contact_adapter, val_features[contact_val_mask],
                contact_adapter_training, args.device,
            )
            contact_val_prediction = contact_prediction
            contact_truth = val_adapter_targets[contact_val_mask]
            contact_adapter_metrics.update({
                "effect_rmse": float(np.sqrt(np.mean(np.square(
                    contact_prediction[:, POSE_DIM:] - contact_truth[:, POSE_DIM:]
                )))),
                "selection_status": "pending_held_out_contact_ranking",
            })
        print(json.dumps({"phase": "contact_adapter", **contact_adapter_metrics}), flush=True)
        train_prediction = predict_targets(adapter, train_features, adapter_training, args.device)
        val_prediction = predict_targets(adapter, val_features, adapter_training, args.device)
        contact_mode_classifier = None
        contact_mode_training = None
        val_mode_probability = np.zeros((len(val_features), 4), dtype=np.float32)
        contact_mode_metrics: dict[str, Any] = {
            "enabled": bool(args.contact_mode_gate),
            "selected_for_runtime": False,
        }
        if args.contact_mode_gate:
            if mode_train is None or mode_val is None:
                raise RuntimeError("contact-mode label data were not constructed")
            train_mode_labels = contact_mode_labels(mode_train)
            val_mode_labels = contact_mode_labels(mode_val)
            contact_mode_classifier, contact_mode_training = train_contact_mode_classifier(
                train_features, train_mode_labels, val_features, val_mode_labels, args,
            )
            val_mode_probability = predict_contact_mode(
                contact_mode_classifier, val_features, contact_mode_training, args.device,
            )
            mode_prediction = np.argmax(val_mode_probability, axis=1)
            contact_mode_metrics.update({
                "accuracy": float(np.mean(mode_prediction == val_mode_labels)),
                "separation_recall": float(
                    np.mean(mode_prediction[val_mode_labels == 3] == 3)
                ) if np.any(val_mode_labels == 3) else float("nan"),
                "validation_rows": int(len(val_mode_labels)),
                "class_counts": np.bincount(val_mode_labels, minlength=4).tolist(),
            })
            print(json.dumps({"phase": "contact_mode", **contact_mode_metrics}), flush=True)
        # Direct goal-conditioned ranker: train on the same matched
        # interventions, but predict the task-space successor cost directly.
        # We retain the legacy feature interface as a negative control and
        # add an explicit geometry-aware interface.  Both see exactly the
        # same roots, action bank, labels, optimizer, and held-out split.
        train_goal_root_features = goal_feature_rows(model, train, args.device)
        val_goal_root_features = goal_feature_rows(model, val, args.device)
        train_previous_features = previous_feature_rows(model, train, args.device)
        val_previous_features = previous_feature_rows(model, val, args.device)
        legacy_train_rank_features = np.concatenate([
            train_features, train_goal_root_features[train["root_id"]],
        ], axis=1)
        legacy_val_rank_features = np.concatenate([
            val_features, val_goal_root_features[val["root_id"]],
        ], axis=1)
        # Construct task context from RGB-derived readouts only.  In
        # particular, the goal poses below are decoder predictions rather
        # than the simulator goal labels used to form the ranking target.
        train_base_current_orientation = base_orientation_rows(model, normalizer, train, args.device)
        val_base_current_orientation = base_orientation_rows(model, normalizer, val, args.device)
        if selected_orientation_decoder is not None and selected_orientation_training is not None:
            train_current_orientation = predict_orientation(
                selected_orientation_decoder, train_current_features,
                selected_orientation_training, args.device,
            )
            val_current_orientation = predict_orientation(
                selected_orientation_decoder, val_current_features,
                selected_orientation_training, args.device,
            )
        else:
            train_current_orientation = train_base_current_orientation
            val_current_orientation = val_base_current_orientation
        train_current_angles = np.arctan2(
            train_current_orientation[:, 0], train_current_orientation[:, 1],
        ).astype(np.float32)
        val_current_angles = np.arctan2(
            val_current_orientation[:, 0], val_current_orientation[:, 1],
        ).astype(np.float32)
        train_goal_estimates = base_goal_rows(model, normalizer, train, args.device)
        val_goal_estimates = base_goal_rows(model, normalizer, val, args.device)
        if selected_goal_decoder is not None and selected_goal_training is not None:
            train_goal_estimates = predict_goal(
                selected_goal_decoder, train_goal_root_features,
                selected_goal_training, args.device,
            )
            val_goal_estimates = predict_goal(
                selected_goal_decoder, val_goal_root_features,
                selected_goal_training, args.device,
            )
        if goal_component_selection.get("xy_source") == "joint_decoder" and selected_goal_decoder is not None and selected_goal_training is not None:
            train_goal_estimates[:, :2] = predict_goal(
                selected_goal_decoder, train_goal_root_features,
                selected_goal_training, args.device,
            )[:, :2]
            val_goal_estimates[:, :2] = predict_goal(
                selected_goal_decoder, val_goal_root_features,
                selected_goal_training, args.device,
            )[:, :2]
        elif goal_component_selection["xy"]:
            train_goal_estimates[:, :2] = predict_orientation(
                goal_xy_decoder, train_goal_root_features[:, TARGET_FEATURE_START:TARGET_FEATURE_STOP],
                goal_xy_decoder_training, args.device,
            )
            val_goal_estimates[:, :2] = predict_orientation(
                goal_xy_decoder, val_goal_root_features[:, TARGET_FEATURE_START:TARGET_FEATURE_STOP],
                goal_xy_decoder_training, args.device,
            )
        elif goal_component_selection.get("xy_source") == "direct_moment":
            train_goal_estimates[:, :2] = (
                train_goal_root_features[:, TARGET_FEATURE_START + 1:TARGET_FEATURE_START + 3] + 1.0
            ) * (PUSHT_STATE_SIZE / 2.0)
            val_goal_estimates[:, :2] = (
                val_goal_root_features[:, TARGET_FEATURE_START + 1:TARGET_FEATURE_START + 3] + 1.0
            ) * (PUSHT_STATE_SIZE / 2.0)
        elif (
            goal_component_selection.get("xy_source") == "rgb_calibrated"
            and rgb_goal_xy_decoder is not None
            and rgb_goal_xy_training is not None
        ):
            train_goal_estimates[:, :2] = predict_orientation(
                rgb_goal_xy_decoder,
                aligned_goal_train_shapes
                if aligned_goal_train_shapes is not None
                else rgb_geometry_shape_features(train["goal_image"][train_root_indices], "target"),
                rgb_goal_xy_training,
                args.device,
            )
            val_goal_estimates[:, :2] = predict_orientation(
                rgb_goal_xy_decoder,
                aligned_goal_val_shapes
                if aligned_goal_val_shapes is not None
                else rgb_geometry_shape_features(val["goal_image"][val_root_indices], "target"),
                rgb_goal_xy_training,
                args.device,
            )
        if goal_component_selection["angle"]:
            train_goal_estimates[:, 2:4] = predict_orientation(
                goal_angle_decoder, train_goal_root_features,
                goal_angle_decoder_training, args.device,
            )
            val_goal_estimates[:, 2:4] = predict_orientation(
                goal_angle_decoder, val_goal_root_features,
                goal_angle_decoder_training, args.device,
            )
        rgb_current_angle_decoder: OrientationDecoder | None = None
        rgb_current_angle_training: dict[str, Any] | None = None
        rgb_goal_angle_decoder: OrientationDecoder | None = None
        rgb_goal_angle_training: dict[str, Any] | None = None
        ranker_train_current_angles = train_current_angles
        ranker_val_current_angles = val_current_angles
        ranker_train_goal_estimates = train_goal_estimates
        ranker_val_goal_estimates = val_goal_estimates
        rgb_angle_readout_metrics: dict[str, Any] = {"enabled": False}
        if args.ranker_interface == "rgb_geometry":
            train_root_indices = np.asarray([
                np.flatnonzero(train["root_id"] == root)[0] for root in range(len(train_roots))
            ], dtype=np.int64)
            val_root_indices = np.asarray([
                np.flatnonzero(val["root_id"] == root)[0] for root in range(len(val_roots))
            ], dtype=np.int64)
            if args.rgb_angle_features == "raster":
                rgb_current_train_shape = rgb_shape_raster_features(
                    np.asarray([root[2] for root in train_roots]), "block",
                )
                rgb_current_val_shape = rgb_shape_raster_features(
                    np.asarray([root[2] for root in val_roots]), "block",
                )
            else:
                rgb_current_train_shape = rgb_geometry_shape_features(
                    train["image"][train_root_indices], "block",
                )
                rgb_current_val_shape = rgb_geometry_shape_features(
                    val["image"][val_root_indices], "block",
                )
            raw_train_goal_images = aligned_goal_train_images
            raw_val_goal_images = aligned_goal_val_images
            rgb_current_train_targets = train_orientation_targets
            rgb_current_val_targets = val_orientation_targets
            rgb_dynamic_metrics: dict[str, Any] = {
                "enabled": bool(args.readout_augmentation_steps > 0),
                "steps_per_root": int(args.readout_augmentation_steps),
                "contact_biased_enabled": bool(
                    getattr(args, "rgb_angle_contact_augmentation", False)
                ),
                "train_rows": 0,
                "validation_rows": 0,
                "contact_train_rows": 0,
                "contact_validation_rows": 0,
            }
            if args.readout_augmentation_steps > 0:
                dynamic_rgb_train_shape, dynamic_rgb_train_orientation = dynamic_rgb_shape_angle_rows(
                    env, train_roots, args.image_size, data_seed + 620_000,
                    args.readout_augmentation_steps, args.rgb_angle_features,
                )
                dynamic_rgb_val_shape, dynamic_rgb_val_orientation = dynamic_rgb_shape_angle_rows(
                    env, val_roots, args.image_size, data_seed + 630_000,
                    args.readout_augmentation_steps, args.rgb_angle_features,
                )
                rgb_current_train_shape = np.concatenate(
                    [rgb_current_train_shape, dynamic_rgb_train_shape], axis=0,
                )
                rgb_current_val_shape = np.concatenate(
                    [rgb_current_val_shape, dynamic_rgb_val_shape], axis=0,
                )
                rgb_current_train_targets = np.concatenate(
                    [rgb_current_train_targets, dynamic_rgb_train_orientation], axis=0,
                )
                rgb_current_val_targets = np.concatenate(
                    [rgb_current_val_targets, dynamic_rgb_val_orientation], axis=0,
                )
                rgb_dynamic_metrics.update({
                    "train_rows": int(len(dynamic_rgb_train_shape)),
                    "validation_rows": int(len(dynamic_rgb_val_shape)),
                    "selection_validation_distribution": "static roots plus root-disjoint offline rollout frames",
                })
                if bool(getattr(args, "rgb_angle_contact_augmentation", False)):
                    contact_rgb_train_shape, contact_rgb_train_orientation = (
                        dynamic_rgb_shape_angle_rows(
                            env, train_roots, args.image_size, data_seed + 625_000,
                            args.readout_augmentation_steps, args.rgb_angle_features,
                            contact_biased=True,
                        )
                    )
                    contact_rgb_val_shape, contact_rgb_val_orientation = (
                        dynamic_rgb_shape_angle_rows(
                            env, val_roots, args.image_size, data_seed + 635_000,
                            args.readout_augmentation_steps, args.rgb_angle_features,
                            contact_biased=True,
                        )
                    )
                    rgb_current_train_shape = np.concatenate(
                        [rgb_current_train_shape, contact_rgb_train_shape], axis=0,
                    )
                    rgb_current_val_shape = np.concatenate(
                        [rgb_current_val_shape, contact_rgb_val_shape], axis=0,
                    )
                    rgb_current_train_targets = np.concatenate(
                        [rgb_current_train_targets, contact_rgb_train_orientation], axis=0,
                    )
                    rgb_current_val_targets = np.concatenate(
                        [rgb_current_val_targets, contact_rgb_val_orientation], axis=0,
                    )
                    rgb_dynamic_metrics.update({
                        "contact_train_rows": int(len(contact_rgb_train_shape)),
                        "contact_validation_rows": int(len(contact_rgb_val_shape)),
                        "contact_augmentation_protocol": (
                            "bounded alternating inward/tangential RGB rollout on "
                            "root-disjoint contact-biased states"
                        ),
                    })
            if args.rgb_angle_features == "raster" and args.rgb_angle_readout == "knn":
                rgb_current_angle_decoder, rgb_current_angle_training = train_rgb_angle_knn(
                    rgb_current_train_shape, rgb_current_train_targets,
                )
                rgb_current_angle_decoder = rgb_current_angle_decoder.to(args.device)
            else:
                rgb_current_angle_decoder, rgb_current_angle_training = train_orientation_decoder(
                    rgb_current_train_shape,
                    rgb_current_train_targets,
                    rgb_current_val_shape,
                    rgb_current_val_targets,
                    args,
                )
            rgb_current_angle_training["feature_interface"] = args.rgb_angle_features
            rgb_current_angle_training["feature_role"] = "block"
            rgb_current_angle_training["readout"] = (
                "support_constrained_knn"
                if args.rgb_angle_features == "raster" and args.rgb_angle_readout == "knn"
                else "mlp"
            )
            # Target pose calibration uses the compact geometric chart rather
            # than the current-angle raster.  The raster is translation
            # normalized and is therefore appropriate for orientation
            # classification, but it intentionally discards the absolute
            # centroid needed to recover the scene-aligned target position.
            goal_angle_train_features = (
                aligned_goal_train_shapes
                if aligned_goal_train_shapes is not None
                else rgb_geometry_shape_features(train["goal_image"][train_root_indices], "target")
            )
            goal_angle_val_features = (
                aligned_goal_val_shapes
                if aligned_goal_val_shapes is not None
                else rgb_geometry_shape_features(val["goal_image"][val_root_indices], "target")
            )
            rgb_goal_angle_decoder, rgb_goal_angle_training = train_rgb_pose_calibrator(
                goal_angle_train_features, train_goal_targets[:, 2:4],
                goal_angle_val_features, val_goal_targets[:, 2:4], ridge_alpha=1.0,
            )
            rgb_goal_angle_training["feature_interface"] = "moments"
            rgb_goal_angle_training["feature_role"] = "target"
            train_rgb_current_orientation = predict_orientation(
                rgb_current_angle_decoder,
                rgb_current_train_shape[:len(train_roots)],
                rgb_current_angle_training,
                args.device,
            )
            val_rgb_current_orientation = predict_orientation(
                rgb_current_angle_decoder,
                rgb_current_val_shape[:len(val_roots)],
                rgb_current_angle_training,
                args.device,
            )
            rgb_current_validation_error = angle_error_batch(
                predict_orientation(
                    rgb_current_angle_decoder,
                    rgb_current_val_shape,
                    rgb_current_angle_training,
                    args.device,
                ),
                rgb_current_val_targets,
            )
            train_rgb_goal_orientation = predict_orientation(
                rgb_goal_angle_decoder,
                goal_angle_train_features,
                rgb_goal_angle_training,
                args.device,
            )
            val_rgb_goal_orientation = predict_orientation(
                rgb_goal_angle_decoder,
                goal_angle_val_features,
                rgb_goal_angle_training,
                args.device,
            )
            # Keep the offline ranker interface identical to runtime.  The
            # old path always trained on RGB angle predictions, while a
            # failed validation gate made runtime fall back to the frozen
            # angle head (or, for ``rgb_shape``, to an uncalibrated raw chart).
            # That mismatch corrupted contact/torque ranking even when the
            # action-effect model itself was unchanged.
            rgb_angle_selected = bool(
                rgb_current_validation_error <= base_orientation_metrics["angle_mean_error"]
            )
            args._use_rgb_current_angle_runtime = rgb_angle_selected
            if rgb_angle_selected or args.current_angle_policy == "rgb_shape":
                ranker_train_current_angles = np.arctan2(
                    train_rgb_current_orientation[:, 0], train_rgb_current_orientation[:, 1],
                ).astype(np.float32)
                ranker_val_current_angles = np.arctan2(
                    val_rgb_current_orientation[:, 0], val_rgb_current_orientation[:, 1],
                ).astype(np.float32)
            else:
                ranker_train_current_angles = train_current_angles
                ranker_val_current_angles = val_current_angles
            rgb_angle_readout_metrics = {
                "enabled": True,
                "current_angle_mean_error": float(rgb_current_validation_error),
                "current_angle_static_mean_error": angle_error_batch(
                    val_rgb_current_orientation, val_orientation_targets,
                ),
                "current_angle_dynamic_readout": rgb_dynamic_metrics,
                "goal_angle_mean_error": angle_error_batch(
                    val_rgb_goal_orientation, val_goal_targets[:, 2:4],
                ),
                "interface": f"root_disjoint_rgb_shape_{args.rgb_angle_features}_current_readout_and_ridge_target_calibration",
            }
            # Static-root accuracy alone is not a deployment qualification for
            # a contact controller.  Runtime replacement is enabled only
            # after the optional root-disjoint dynamic validation protocol
            # also beats the frozen current-angle head.
            args._use_rgb_current_angle_runtime = bool(
                rgb_current_validation_error <= base_orientation_metrics["angle_mean_error"]
            )
            # The target-only green render is not a reliable directed pose
            # chart on this renderer: retain the independently selected goal
            # readout unless its held-out angle error beats the base goal
            # head.  The failed RGB target calibration remains reported below
            # as an explicit negative diagnostic and is never consumed.
            goal_rgb_angle_selected = bool(
                rgb_angle_readout_metrics["goal_angle_mean_error"]
                <= float(args.rgb_goal_angle_max_error)
                and rgb_angle_readout_metrics["goal_angle_mean_error"]
                < base_goal_readout_metrics["angle_mean_error"]
            )
            if goal_rgb_angle_selected:
                ranker_train_goal_estimates[:, 2:4] = train_rgb_goal_orientation
                ranker_val_goal_estimates[:, 2:4] = val_rgb_goal_orientation
                goal_component_selection["angle_source"] = "rgb_angle_decoder"
            else:
                rgb_goal_angle_decoder = None
                rgb_goal_angle_training = None
            rgb_angle_readout_metrics["goal_angle_selected_for_ranker"] = goal_rgb_angle_selected
            rgb_angle_readout_metrics["goal_angle_selection_max_error"] = float(
                args.rgb_goal_angle_max_error
            )
            print(json.dumps({"phase": "rgb_shape_angle_readout", **rgb_angle_readout_metrics}), flush=True)
            geometry_train_rank_features = rgb_geometry_ranker_features(
                train["image"], train["previous_image"], train["goal_image"],
                train["action"], ranker_train_current_angles[train["root_id"]],
                ranker_train_goal_estimates[train["root_id"]],
                feature_mode="full",
            )
            geometry_val_rank_features = rgb_geometry_ranker_features(
                val["image"], val["previous_image"], val["goal_image"],
                val["action"], ranker_val_current_angles[val["root_id"]],
                ranker_val_goal_estimates[val["root_id"]],
                feature_mode="full",
            )
        else:
            geometry_train_rank_features = geometry_ranker_features(
                train_current_features[train["root_id"]],
                train_previous_features[train["root_id"]],
                train_goal_root_features[train["root_id"]],
                train["action"],
                train_current_angles[train["root_id"]],
                train_goal_estimates[train["root_id"]],
            )
            geometry_val_rank_features = geometry_ranker_features(
                val_current_features[val["root_id"]],
                val_previous_features[val["root_id"]],
                val_goal_root_features[val["root_id"]],
                val["action"],
                val_current_angles[val["root_id"]],
                val_goal_estimates[val["root_id"]],
            )
        terminal_geometry_train_rank_features = geometry_train_rank_features
        terminal_geometry_val_rank_features = geometry_val_rank_features
        terminal_feature_mode = str(getattr(args, "terminal_ranker_feature_mode", "full"))
        if (
            args.ranker_interface == "rgb_geometry"
            and terminal_feature_mode != "full"
        ):
            terminal_geometry_train_rank_features = rgb_geometry_ranker_features(
                train["image"], train["previous_image"], train["goal_image"],
                train["action"], ranker_train_current_angles[train["root_id"]],
                ranker_train_goal_estimates[train["root_id"]],
                feature_mode=(
                    "invariant"
                    if terminal_feature_mode == "invariant_adapter"
                    else terminal_feature_mode
                ),
            )
            terminal_geometry_val_rank_features = rgb_geometry_ranker_features(
                val["image"], val["previous_image"], val["goal_image"],
                val["action"], ranker_val_current_angles[val["root_id"]],
                ranker_val_goal_estimates[val["root_id"]],
                feature_mode=(
                    "invariant"
                    if terminal_feature_mode == "invariant_adapter"
                    else terminal_feature_mode
                ),
            )
        train_goal_states = np.asarray([observable_pose_vector(root[1]) for root in train_roots], dtype=np.float32)
        val_goal_states = np.asarray([observable_pose_vector(root[1]) for root in val_roots], dtype=np.float32)
        train_goal_block = np.column_stack([
            train_goal_states[train["root_id"], 2:4],
            np.arctan2(train_goal_states[train["root_id"], 4], train_goal_states[train["root_id"], 5]),
        ])
        val_goal_block = np.column_stack([
            val_goal_states[val["root_id"], 2:4],
            np.arctan2(val_goal_states[val["root_id"], 4], val_goal_states[val["root_id"], 5]),
        ])
        train_next_block = np.column_stack([
            train["next_state"][:, 2:4],
            np.arctan2(train["next_state"][:, 4], train["next_state"][:, 5]),
        ])
        val_next_block = np.column_stack([
            val["next_state"][:, 2:4],
            np.arctan2(val["next_state"][:, 4], val["next_state"][:, 5]),
        ])
        train_rank_targets = (
            np.linalg.norm(train_next_block[:, :2] - train_goal_block[:, :2], axis=1) / 20.0
            + 10.0 * np.abs(wrap(train_next_block[:, 2] - train_goal_block[:, 2]))
        ).astype(np.float32)
        val_rank_targets = (
            np.linalg.norm(val_next_block[:, :2] - val_goal_block[:, :2], axis=1) / 20.0
            + 10.0 * np.abs(wrap(val_next_block[:, 2] - val_goal_block[:, 2]))
        ).astype(np.float32)
        # The selector only needs within-root action improvement.  Centering
        # by the matched zero-action cost removes the nuisance difficulty of
        # the source root while preserving the argmin exactly.
        for root_id in np.unique(train["root_id"]):
            indices = np.flatnonzero(train["root_id"] == root_id)
            train_rank_targets[indices] -= train_rank_targets[indices[0]]
        for root_id in np.unique(val["root_id"]):
            indices = np.flatnonzero(val["root_id"] == root_id)
            val_rank_targets[indices] -= val_rank_targets[indices[0]]
        args.ranker_group_size = len(actions)
        terminal_angular_ranker = None
        terminal_angular_ranker_training = None
        terminal_angular_ranker_metrics: dict[str, Any] = {
            "enabled": bool(args.terminal_angular_ranker),
            "selected_for_runtime": False,
            "selection_rule": "held_out terminal-root angular regret gain with no material XY degradation",
        }
        # Terminal roots are appended as one contiguous, root-disjoint block
        # before optional bootstrap roots.  The angular specialist sees only
        # this declared manifold; ordinary translation/contact roots cannot
        # dominate its loss or its promotion gate.
        train_terminal_start = len(train_roots) - train_terminal - train_bootstrap
        val_terminal_start = len(val_roots) - val_terminal - val_bootstrap
        train_terminal_root_mask = np.zeros(len(train_roots), dtype=bool)
        val_terminal_root_mask = np.zeros(len(val_roots), dtype=bool)
        if train_terminal > 0:
            train_terminal_root_mask[train_terminal_start:train_terminal_start + train_terminal] = True
        if val_terminal > 0:
            val_terminal_root_mask[val_terminal_start:val_terminal_start + val_terminal] = True
        train_terminal_rows = train_terminal_root_mask[train["root_id"]]
        val_terminal_rows = val_terminal_root_mask[val["root_id"]]
        train_goal_pose_rows = train_goal_states[train["root_id"], 2:6]
        val_goal_pose_rows = val_goal_states[val["root_id"], 2:6]
        train_angular_true_cost = angular_costs(train, train_goal_pose_rows)
        val_angular_true_cost = angular_costs(val, val_goal_pose_rows)
        train_baseline_angular_cost = predicted_angle_costs(
            train, train_prediction, train_goal_pose_rows, args.effect_frame,
        )
        val_baseline_angular_cost = predicted_angle_costs(
            val, val_prediction, val_goal_pose_rows, args.effect_frame,
        )
        if terminal_feature_mode == "invariant_adapter":
            train_terminal_baseline = (
                predicted_pose_costs(
                    train, train_prediction, train_goal_pose_rows, args.effect_frame,
                    xy_weight=float(args.terminal_pose_xy_weight),
                )
                if str(getattr(args, "terminal_ranker_target", "angle")) == "pose"
                else predicted_angle_costs(
                    train, train_prediction, train_goal_pose_rows, args.effect_frame,
                )
            )
            val_terminal_baseline = (
                predicted_pose_costs(
                    val, val_prediction, val_goal_pose_rows, args.effect_frame,
                    xy_weight=float(args.terminal_pose_xy_weight),
                )
                if str(getattr(args, "terminal_ranker_target", "angle")) == "pose"
                else predicted_angle_costs(
                    val, val_prediction, val_goal_pose_rows, args.effect_frame,
                )
            )
            terminal_geometry_train_rank_features = np.concatenate([
                terminal_geometry_train_rank_features,
                train_terminal_baseline[:, None].astype(np.float32),
            ], axis=1)
            terminal_geometry_val_rank_features = np.concatenate([
                terminal_geometry_val_rank_features,
                val_terminal_baseline[:, None].astype(np.float32),
            ], axis=1)
        terminal_train_angular_labels = np.abs(
            wrap(train_next_block[:, 2] - train_goal_block[:, 2])
        ).astype(np.float32)
        terminal_val_angular_labels = np.abs(
            wrap(val_next_block[:, 2] - val_goal_block[:, 2])
        ).astype(np.float32)
        for root_id in np.unique(train["root_id"]):
            indices = np.flatnonzero(train["root_id"] == root_id)
            terminal_train_angular_labels[indices] -= terminal_train_angular_labels[indices[0]]
        for root_id in np.unique(val["root_id"]):
            indices = np.flatnonzero(val["root_id"] == root_id)
            terminal_val_angular_labels[indices] -= terminal_val_angular_labels[indices[0]]
        # Residualize against the already trained adapter.  A direct angular
        # ranker discarded a useful frozen-effect baseline and regressed from
        # scratch; this specialist learns only the correction while its
        # listwise supervision remains the true held-out angular ordering.
        terminal_train_angular_targets = train_angular_true_cost - train_baseline_angular_cost
        terminal_val_angular_targets = val_angular_true_cost - val_baseline_angular_cost
        terminal_ranker_score_mode = "one_step_residual"
        terminal_ranker_eval_data = val
        terminal_ranker_eval_root_mask = val_terminal_root_mask
        terminal_ranker_eval_true_cost = val_angular_true_cost
        terminal_ranker_eval_baseline_scores = val_baseline_angular_cost
        terminal_ranker_eval_goal_pose_rows = val_goal_pose_rows
        # Promotion must be evaluated on the exact terminal candidate
        # support used at runtime.  The terminal controller may use a
        # much smaller cap than the broad contact fan (e.g. 0.04 versus
        # 0.35); evaluating the gate on the broad fan certifies a
        # different action-selection problem and can produce a positive
        # offline result followed by a negative closed loop.  Preserve
        # the broad cap only when no terminal-specific cap is declared.
        terminal_eval_cap = float(getattr(args, "terminal_action_max_magnitude", -1.0))
        if terminal_eval_cap <= 0.0:
            terminal_eval_cap = float(args.contact_action_max_magnitude)
        terminal_ranker_eval_action_mask = np.linalg.norm(val["action"], axis=1) <= (
            terminal_eval_cap + 1.0e-6
        )
        terminal_multistep_protocol: dict[str, Any] = {
            "enabled": bool(args.terminal_action_horizon > 1),
            "horizon": int(args.terminal_action_horizon),
            "training_rows": 0,
            "validation_rows": 0,
        }
        if bool(args.terminal_angular_ranker):
            if int(np.sum(train_terminal_rows)) < len(actions) or int(np.sum(val_terminal_rows)) < len(actions):
                terminal_angular_ranker_metrics.update({
                    "selection_status": "underpowered_terminal_root_support",
                    "train_terminal_roots": int(np.sum(train_terminal_root_mask)),
                    "validation_terminal_roots": int(np.sum(val_terminal_root_mask)),
                })
            else:
                if int(args.terminal_action_horizon) > 1:
                    # The runtime controller replans after every observation,
                    # but a one-step angular label is too noisy to identify a
                    # useful torque ordering.  Train the specialist on a
                    # declared short lookahead: each candidate is repeated
                    # for h physical simulator steps from the same source
                    # root.  The RGB ranker input is unchanged; only the
                    # offline ordering label is multi-step.
                    train_terminal_roots_only = [
                        train_roots[index]
                        for index in np.flatnonzero(train_terminal_root_mask)
                    ]
                    val_terminal_roots_only = [
                        val_roots[index]
                        for index in np.flatnonzero(val_terminal_root_mask)
                    ]
                    terminal_train_multistep = make_local_effect_dataset(
                        env, train_terminal_roots_only, args.image_size, actions,
                        int(args.terminal_action_horizon), args.goal_observation,
                    )
                    terminal_val_multistep = make_local_effect_dataset(
                        env, val_terminal_roots_only, args.image_size, actions,
                        int(args.terminal_action_horizon), args.goal_observation,
                    )
                    train_terminal_goals = np.asarray([
                        observable_pose_vector(root[1])[2:6]
                        for root in train_terminal_roots_only
                    ], dtype=np.float32)
                    val_terminal_goals = np.asarray([
                        observable_pose_vector(root[1])[2:6]
                        for root in val_terminal_roots_only
                    ], dtype=np.float32)
                    terminal_train_goal_rows = np.repeat(train_terminal_goals, len(actions), axis=0)
                    terminal_val_goal_rows = np.repeat(val_terminal_goals, len(actions), axis=0)
                    constrained_terminal_labels = any(
                        float(getattr(args, name, 0.0)) > 0.0
                        for name in (
                            "terminal_trajectory_contact_penalty",
                            "terminal_trajectory_xy_excursion_penalty",
                            "terminal_trajectory_angle_excursion_penalty",
                        )
                    )
                    terminal_target = str(getattr(args, "terminal_ranker_target", "angle"))
                    if constrained_terminal_labels:
                        terminal_cost_fn = lambda data, goals: constrained_trajectory_pose_costs(
                            data,
                            goals,
                            xy_weight=(
                                float(args.terminal_pose_xy_weight)
                                if terminal_target == "pose" else 0.0
                            ),
                            angle_weight=(10.0 if terminal_target == "pose" else 1.0),
                            contact_penalty=float(args.terminal_trajectory_contact_penalty),
                            xy_excursion_penalty=float(args.terminal_trajectory_xy_excursion_penalty),
                            angle_excursion_penalty=float(args.terminal_trajectory_angle_excursion_penalty),
                            contact_distance_max=float(args.terminal_trajectory_contact_distance_max),
                        )
                    else:
                        terminal_cost_fn = (
                            lambda data, goals: pose_endpoint_costs(
                                data, goals, float(args.terminal_pose_xy_weight),
                            )
                            if terminal_target == "pose"
                            else angular_costs(data, goals)
                        )
                    terminal_train_multistep_cost = terminal_cost_fn(
                        terminal_train_multistep, terminal_train_goal_rows,
                    )
                    terminal_val_multistep_cost = terminal_cost_fn(
                        terminal_val_multistep, terminal_val_goal_rows,
                    )
                    # The terminal specialist must be trained on the same
                    # action support on which it is promoted and executed.
                    # Training on the broad contact fan and validating only
                    # the small terminal fan makes the ranker extrapolate
                    # precisely where the closed-loop controller relies on
                    # it.  Keep the full arrays for evaluation/audit, but
                    # fit the model only on support-matched rows.
                    terminal_train_support_mask = np.linalg.norm(
                        terminal_train_multistep["action"], axis=1,
                    ) <= (terminal_eval_cap + 1.0e-6)
                    terminal_val_support_mask = np.linalg.norm(
                        terminal_val_multistep["action"], axis=1,
                    ) <= (terminal_eval_cap + 1.0e-6)
                    for root_id in np.unique(terminal_train_multistep["root_id"]):
                        indices = np.flatnonzero(terminal_train_multistep["root_id"] == root_id)
                        terminal_train_multistep_cost[indices] -= terminal_train_multistep_cost[indices[0]]
                    for root_id in np.unique(terminal_val_multistep["root_id"]):
                        indices = np.flatnonzero(terminal_val_multistep["root_id"] == root_id)
                        terminal_val_multistep_cost[indices] -= terminal_val_multistep_cost[indices[0]]
                    # Support filtering changes the number of candidates per
                    # root (for example, a 0.35 terminal cap retains roughly
                    # 65 of the 97 broad-fan actions).  The ranker loss is
                    # root-wise; retaining the broad-fan group size here would
                    # silently concatenate rows from adjacent roots and train
                    # against invalid within-group minima.  This failure mode
                    # can increase apparent top-1 hits while producing
                    # catastrophic held-out regret.
                    train_support_counts = [
                        int(np.sum(
                            terminal_train_support_mask
                            & (terminal_train_multistep["root_id"] == root_id)
                        ))
                        for root_id in np.unique(terminal_train_multistep["root_id"])
                    ]
                    val_support_counts = [
                        int(np.sum(
                            terminal_val_support_mask
                            & (terminal_val_multistep["root_id"] == root_id)
                        ))
                        for root_id in np.unique(terminal_val_multistep["root_id"])
                    ]
                    if (
                        not train_support_counts
                        or len(set(train_support_counts)) != 1
                        or len(set(val_support_counts)) != 1
                        or train_support_counts[0] != val_support_counts[0]
                        or train_support_counts[0] < 2
                    ):
                        raise RuntimeError(
                            "terminal support must retain the same >=2 candidate count per root"
                        )
                    saved_ranker_group_size = int(args.ranker_group_size)
                    args.ranker_group_size = int(train_support_counts[0])
                    try:
                        terminal_angular_ranker, terminal_angular_ranker_training = train_action_cost_ranker(
                            terminal_geometry_train_rank_features[train_terminal_rows][terminal_train_support_mask],
                            terminal_train_multistep_cost[terminal_train_support_mask],
                            terminal_geometry_val_rank_features[val_terminal_rows][terminal_val_support_mask],
                            terminal_val_multistep_cost[terminal_val_support_mask],
                            args,
                            ranking_train_targets=terminal_train_multistep_cost[terminal_train_support_mask],
                            ranking_val_targets=terminal_val_multistep_cost[terminal_val_support_mask],
                        )
                    finally:
                        args.ranker_group_size = saved_ranker_group_size
                    terminal_ranker_score_mode = (
                        "direct_multistep_pose"
                        if str(getattr(args, "terminal_ranker_target", "angle")) == "pose"
                        else "direct_multistep_angle"
                    )
                    terminal_ranker_eval_data = terminal_val_multistep
                    terminal_ranker_eval_root_mask = np.ones(
                        len(val_terminal_roots_only), dtype=bool,
                    )
                    if constrained_terminal_labels:
                        terminal_ranker_eval_true_cost = constrained_trajectory_pose_costs(
                            terminal_val_multistep,
                            terminal_val_goal_rows,
                            xy_weight=(
                                float(args.terminal_pose_xy_weight)
                                if terminal_target == "pose" else 0.0
                            ),
                            angle_weight=(10.0 if terminal_target == "pose" else 1.0),
                            contact_penalty=float(args.terminal_trajectory_contact_penalty),
                            xy_excursion_penalty=float(args.terminal_trajectory_xy_excursion_penalty),
                            angle_excursion_penalty=float(args.terminal_trajectory_angle_excursion_penalty),
                            contact_distance_max=float(args.terminal_trajectory_contact_distance_max),
                        )
                    else:
                        terminal_ranker_eval_true_cost = (
                            pose_endpoint_costs(
                                terminal_val_multistep,
                                terminal_val_goal_rows,
                                float(args.terminal_pose_xy_weight),
                            )
                            if terminal_target == "pose"
                            else angular_costs(
                                terminal_val_multistep, terminal_val_goal_rows,
                            )
                        )
                    if str(getattr(args, "terminal_ranker_target", "angle")) == "pose":
                        # Compare the specialist with the frozen adapter under
                        # the same task-space objective.  The old gate used
                        # one-step angle scores even when the specialist was
                        # trained on repeated-action pose labels, which could
                        # reward a nominal top-1 angle hit while worsening the
                        # actual pose regret.
                        terminal_ranker_eval_baseline_scores = predicted_pose_costs(
                            terminal_val_multistep,
                            val_prediction[val_terminal_rows],
                            terminal_val_goal_rows,
                            args.effect_frame,
                            xy_weight=float(args.terminal_pose_xy_weight),
                        )
                    else:
                        terminal_ranker_eval_baseline_scores = val_baseline_angular_cost[val_terminal_rows]
                    terminal_ranker_eval_goal_pose_rows = terminal_val_goal_rows
                    terminal_ranker_eval_action_mask = np.linalg.norm(
                        terminal_val_multistep["action"], axis=1,
                    ) <= (terminal_eval_cap + 1.0e-6)
                    terminal_multistep_protocol.update({
                        "training_rows": int(len(terminal_train_multistep["root_id"])),
                        "validation_rows": int(len(terminal_val_multistep["root_id"])),
                        "training_roots": int(len(train_terminal_roots_only)),
                        "validation_roots": int(len(val_terminal_roots_only)),
                        "action_cap": float(terminal_eval_cap),
                        "training_support_rows": int(np.sum(terminal_train_support_mask)),
                        "validation_support_rows": int(np.sum(terminal_val_support_mask)),
                        "support_candidates_per_root": int(train_support_counts[0]),
                        "support_matched_training": True,
                        "label": (
                            "pose cost after repeated candidate action"
                            if str(getattr(args, "terminal_ranker_target", "angle")) == "pose"
                            else "angle error after repeated candidate action"
                        ),
                        "target": str(getattr(args, "terminal_ranker_target", "angle")),
                        "trajectory_label": bool(constrained_terminal_labels),
                        "trajectory_contact_penalty": float(
                            getattr(args, "terminal_trajectory_contact_penalty", 0.0)
                        ),
                        "trajectory_xy_excursion_penalty": float(
                            getattr(args, "terminal_trajectory_xy_excursion_penalty", 0.0)
                        ),
                        "trajectory_angle_excursion_penalty": float(
                            getattr(args, "terminal_trajectory_angle_excursion_penalty", 0.0)
                        ),
                        "trajectory_contact_distance_max": float(
                            getattr(args, "terminal_trajectory_contact_distance_max", 80.0)
                        ),
                    })
                else:
                    terminal_angular_ranker, terminal_angular_ranker_training = train_action_cost_ranker(
                        terminal_geometry_train_rank_features[train_terminal_rows],
                        terminal_train_angular_targets[train_terminal_rows],
                        terminal_geometry_val_rank_features[val_terminal_rows],
                        terminal_val_angular_targets[val_terminal_rows],
                        args,
                        ranking_train_targets=terminal_train_angular_labels[train_terminal_rows],
                        ranking_val_targets=terminal_val_angular_labels[val_terminal_rows],
                    )
                terminal_angular_ranker_training["score_mode"] = terminal_ranker_score_mode
                terminal_angular_ranker_training["action_horizon"] = int(args.terminal_action_horizon)
                terminal_angular_ranker_training["target"] = str(
                    getattr(args, "terminal_ranker_target", "angle")
                )
                terminal_angular_ranker_training["feature_mode"] = str(
                    getattr(args, "terminal_ranker_feature_mode", "full")
                )
                validate_terminal_ranker_contract(
                    terminal_angular_ranker_training, args,
                )
                terminal_val_prediction = predict_action_cost(
                    terminal_angular_ranker,
                    terminal_geometry_val_rank_features[val_terminal_rows],
                    terminal_angular_ranker_training,
                    args.device,
                )
                terminal_angular_ranker_metrics.update({
                    "train_terminal_roots": int(np.sum(train_terminal_root_mask)),
                    "validation_terminal_roots": int(np.sum(val_terminal_root_mask)),
                    "held_out_cost_rmse": float(np.sqrt(np.mean(np.square(
                        terminal_val_prediction - (
                            terminal_val_multistep_cost
                            if int(args.terminal_action_horizon) > 1
                            else terminal_val_angular_targets[val_terminal_rows]
                        )
                    )))),
                    "support_actions_per_root": int(len(actions)),
                    "score_mode": terminal_ranker_score_mode,
                    "action_horizon": int(args.terminal_action_horizon),
                    "target": str(getattr(args, "terminal_ranker_target", "angle")),
                    "contract_version": 1,
                    "multistep_protocol": terminal_multistep_protocol,
                })
        else:
            terminal_angular_ranker_metrics["selection_status"] = "disabled"
        legacy_action_cost_ranker, legacy_action_cost_ranker_training = train_action_cost_ranker(
            legacy_train_rank_features, train_rank_targets,
            legacy_val_rank_features, val_rank_targets, args,
        )
        legacy_val_rank_prediction = predict_action_cost(
            legacy_action_cost_ranker, legacy_val_rank_features,
            legacy_action_cost_ranker_training, args.device,
        )
        legacy_ranker_hits: list[float] = []
        for root_id in np.unique(val["root_id"]):
            indices = np.flatnonzero(val["root_id"] == root_id)
            legacy_ranker_hits.append(float(
                np.argmin(legacy_val_rank_prediction[indices]) == np.argmin(val_rank_targets[indices])
            ))
        legacy_ranker_metrics = {
            "held_out_cost_rmse": float(np.sqrt(np.mean(np.square(legacy_val_rank_prediction - val_rank_targets)))),
            "held_out_top1": float(np.mean(legacy_ranker_hits)) if legacy_ranker_hits else float("nan"),
            "held_out_roots": int(len(legacy_ranker_hits)),
            "feature_interface": "legacy_frozen_plus_goal",
        }
        print(json.dumps({"phase": "legacy_action_cost_ranker", **legacy_ranker_metrics}), flush=True)
        action_cost_ranker, action_cost_ranker_training = train_action_cost_ranker(
            geometry_train_rank_features, train_rank_targets,
            geometry_val_rank_features, val_rank_targets, args,
        )
        val_rank_prediction = predict_action_cost(
            action_cost_ranker, geometry_val_rank_features,
            action_cost_ranker_training, args.device,
        )
        ranker_hits: list[float] = []
        ranker_contact_hits: list[float] = []
        for root_id in np.unique(val["root_id"]):
            indices = np.flatnonzero(val["root_id"] == root_id)
            hit = float(
                np.argmin(val_rank_prediction[indices]) == np.argmin(val_rank_targets[indices])
            )
            ranker_hits.append(hit)
            if bool(val["contact_root"][indices[0]]):
                ranker_contact_hits.append(hit)
        ranker_metrics = {
            "held_out_cost_rmse": float(np.sqrt(np.mean(np.square(val_rank_prediction - val_rank_targets)))),
            "held_out_top1": float(np.mean(ranker_hits)) if ranker_hits else float("nan"),
            "held_out_roots": int(len(ranker_hits)),
            "held_out_contact_top1": float(np.mean(ranker_contact_hits)) if ranker_contact_hits else float("nan"),
            "held_out_contact_roots": int(len(ranker_contact_hits)),
            "feature_interface": args.ranker_interface,
            "terminal_feature_mode": str(getattr(args, "terminal_ranker_feature_mode", "full")),
            "feature_mode": "full",
            "terminal_feature_mode": str(getattr(args, "terminal_ranker_feature_mode", "full")),
            "legacy_comparison": legacy_ranker_metrics,
        }
        print(json.dumps({"phase": "geometry_action_cost_ranker", **ranker_metrics}), flush=True)
        contact_ranker = None
        contact_ranker_training = None
        contact_ranker_metrics: dict[str, Any] = {
            "held_out_top1": float("nan"),
            "held_out_roots": 0,
            "selected_for_runtime": False,
        }
        if int(np.sum(contact_train_mask)) >= len(actions) and int(np.sum(contact_val_mask)) >= len(actions):
            contact_ranker, contact_ranker_training = train_action_cost_ranker(
                geometry_train_rank_features[contact_train_mask], train_rank_targets[contact_train_mask],
                geometry_val_rank_features[contact_val_mask], val_rank_targets[contact_val_mask], args,
            )
            contact_rank_prediction = predict_action_cost(
                contact_ranker, geometry_val_rank_features[contact_val_mask],
                contact_ranker_training, args.device,
            )
            contact_rank_hits: list[float] = []
            contact_rank_offset = 0
            for root_id in np.unique(val["root_id"]):
                indices = np.flatnonzero(val["root_id"] == root_id)
                if bool(val["contact_root"][indices[0]]):
                    count = len(indices)
                    local_prediction = contact_rank_prediction[contact_rank_offset:contact_rank_offset + count]
                    local_targets = val_rank_targets[indices]
                    contact_rank_hits.append(float(
                        np.argmin(local_prediction) == np.argmin(local_targets)
                    ))
                    contact_rank_offset += count
            contact_ranker_metrics.update({
                "held_out_cost_rmse": float(np.sqrt(np.mean(np.square(
                    contact_rank_prediction - val_rank_targets[contact_val_mask]
                )))),
                "held_out_top1": float(np.mean(contact_rank_hits)) if contact_rank_hits else float("nan"),
                "held_out_roots": int(len(contact_rank_hits)),
            })
        print(json.dumps({"phase": "contact_action_cost_ranker", **contact_ranker_metrics}), flush=True)
        goal_pose_by_row = np.zeros((len(val["root_id"]), 4), dtype=np.float32)
        # Goal pose is an offline evaluation label, never an adapter input.
        for root_id, root in enumerate(val_roots):
            goal_pose_by_row[val["root_id"] == root_id] = observable_pose_vector(root[1])[2:6]
        ranking = ranking_from_goal_states(val, val_prediction, goal_pose_by_row, args.effect_frame)
        if (
            bool(args.terminal_angular_ranker)
            and terminal_angular_ranker is not None
            and terminal_angular_ranker_training is not None
            and int(np.sum(val_terminal_root_mask)) >= int(args.terminal_angular_min_roots)
        ):
            terminal_angular_prediction = predict_action_cost(
                terminal_angular_ranker,
                terminal_geometry_val_rank_features[val_terminal_rows],
                terminal_angular_ranker_training,
                args.device,
            )
            eval_data = terminal_ranker_eval_data
            eval_root_mask = terminal_ranker_eval_root_mask
            eval_true_cost = terminal_ranker_eval_true_cost
            eval_baseline_scores = terminal_ranker_eval_baseline_scores
            eval_goal_pose_rows = terminal_ranker_eval_goal_pose_rows
            eval_action_mask = terminal_ranker_eval_action_mask
            direct_multistep = terminal_ranker_score_mode in {
                "direct_multistep_angle", "direct_multistep_pose",
            }
            if direct_multistep:
                candidate_angular_scores = terminal_angular_prediction.copy()
            else:
                candidate_angular_scores = np.full(
                    len(val["root_id"]), np.inf, dtype=np.float32,
                )
                candidate_angular_scores[val_terminal_rows] = (
                    val_baseline_angular_cost[val_terminal_rows]
                    + terminal_angular_prediction
                )
            baseline_metrics = held_out_cost_ranking(
                eval_data, eval_baseline_scores, eval_true_cost,
                eval_action_mask, root_mask=eval_root_mask,
            )
            candidate_metrics = held_out_cost_ranking(
                eval_data, candidate_angular_scores, eval_true_cost,
                eval_action_mask, root_mask=eval_root_mask,
            )
            def selected_true_xy(scores: np.ndarray) -> float:
                values: list[float] = []
                for root_id in np.flatnonzero(eval_root_mask):
                    indices = np.flatnonzero(eval_data["root_id"] == root_id)
                    indices = indices[eval_action_mask[indices]]
                    selected = int(indices[np.argmin(scores[indices])])
                    goal = eval_goal_pose_rows[indices[0]]
                    values.append(float(np.linalg.norm(
                        block_pose(eval_data["next_state"][selected])[:2] - goal[:2]
                    )))
                return float(np.mean(values)) if values else float("inf")
            baseline_xy = selected_true_xy(eval_baseline_scores)
            candidate_xy = selected_true_xy(candidate_angular_scores)
            angular_improves = bool(
                candidate_metrics["roots"] >= int(args.terminal_angular_min_roots)
                and np.isfinite(candidate_metrics["top1"])
                and np.isfinite(candidate_metrics["mean_true_regret"])
                and candidate_metrics["top1"] >= baseline_metrics["top1"]
                and candidate_metrics["mean_true_regret"] < baseline_metrics["mean_true_regret"]
                and candidate_xy <= baseline_xy + float(args.terminal_angular_xy_tolerance)
            )
            terminal_angular_ranker_metrics.update({
                "held_out_baseline": baseline_metrics,
                "held_out_candidate": candidate_metrics,
                "baseline_selected_true_xy": baseline_xy,
                "candidate_selected_true_xy": candidate_xy,
                "xy_tolerance": float(args.terminal_angular_xy_tolerance),
                "minimum_terminal_roots": int(args.terminal_angular_min_roots),
                "candidate_improves": angular_improves,
                "selected_for_runtime": angular_improves,
                "evaluation_horizon": int(args.terminal_action_horizon),
                "evaluation_protocol": (
                    "repeated-action terminal lookahead"
                    if direct_multistep else "one-step residual terminal ranking"
                ),
                "selection_status": "held_out_terminal_angle_regret_and_xy_gate"
                if angular_improves else "fail_closed_baseline_retained",
            })
            torque_prior_metrics: dict[str, Any] = {
                "enabled": bool(args.terminal_torque_prior),
                "selected_for_runtime": False,
                "selection_status": "disabled",
            }
            args._terminal_torque_prior_runtime_mode = "none"
            if bool(args.terminal_torque_prior):
                if direct_multistep:
                    terminal_ids = np.flatnonzero(val_terminal_root_mask)
                    prior_current_angles = np.repeat(
                        ranker_val_current_angles[terminal_ids], len(actions),
                    )
                    prior_goal_angles = np.repeat(
                        np.arctan2(
                            ranker_val_goal_estimates[terminal_ids, 2],
                            ranker_val_goal_estimates[terminal_ids, 3],
                        ),
                        len(actions),
                    )
                else:
                    prior_current_angles = ranker_val_current_angles[val["root_id"]]
                    prior_goal_angles = np.arctan2(
                        ranker_val_goal_estimates[val["root_id"], 2],
                        ranker_val_goal_estimates[val["root_id"], 3],
                    )
                prior_errors = wrap(prior_goal_angles - prior_current_angles).astype(np.float32)
                prior_boundary = contact_boundary_chart_from_images(
                    eval_data["image"], eval_data["action"],
                )
                prior_scores = (
                    np.abs(prior_errors)
                    - float(args.terminal_torque_prior_weight) * prior_errors * prior_boundary[:, 9]
                ).astype(np.float32)
                prior_baseline = held_out_cost_ranking(
                    eval_data, eval_baseline_scores, eval_true_cost,
                    eval_action_mask, root_mask=eval_root_mask,
                )
                prior_candidate = held_out_cost_ranking(
                    eval_data, prior_scores, eval_true_cost,
                    eval_action_mask, root_mask=eval_root_mask,
                )
                prior_xy = selected_true_xy(prior_scores)
                def passes_terminal_gate(metrics: dict[str, Any], xy: float) -> bool:
                    return bool(
                        metrics["roots"] >= int(args.terminal_angular_min_roots)
                        and np.isfinite(metrics["top1"])
                        and np.isfinite(metrics["mean_true_regret"])
                        and metrics["top1"] >= prior_baseline["top1"]
                        and metrics["mean_true_regret"] < prior_baseline["mean_true_regret"]
                        and xy <= baseline_xy + float(args.terminal_angular_xy_tolerance)
                    )
                prior_improves = passes_terminal_gate(prior_candidate, prior_xy)
                combined_scores = np.zeros_like(prior_scores)
                for root_id in np.unique(eval_data["root_id"]):
                    indices = np.flatnonzero(eval_data["root_id"] == root_id)
                    def root_normalize(values: np.ndarray) -> np.ndarray:
                        spread = max(float(np.std(values[indices])), 1.0e-6)
                        return (values[indices] - float(np.min(values[indices]))) / spread
                    combined_scores[indices] = (
                        root_normalize(eval_baseline_scores)
                        + float(args.terminal_torque_prior_blend)
                        * root_normalize(prior_scores)
                    )
                combined_candidate = held_out_cost_ranking(
                    eval_data, combined_scores, eval_true_cost,
                    eval_action_mask, root_mask=eval_root_mask,
                )
                combined_xy = selected_true_xy(combined_scores)
                combined_improves = passes_terminal_gate(combined_candidate, combined_xy)
                runtime_mode = "combined" if combined_improves else ("prior" if prior_improves else "none")
                torque_prior_metrics.update({
                    "weight": float(args.terminal_torque_prior_weight),
                    "blend": float(args.terminal_torque_prior_blend),
                    "held_out_baseline": prior_baseline,
                    "held_out_prior": prior_candidate,
                    "prior_selected_true_xy": prior_xy,
                    "prior_improves": prior_improves,
                    "held_out_combined": combined_candidate,
                    "combined_selected_true_xy": combined_xy,
                    "combined_improves": combined_improves,
                    "evaluation_horizon": int(args.terminal_action_horizon),
                    "selection_status": "held_out_combined_torque_prior_gain"
                    if runtime_mode == "combined" else (
                        "held_out_torque_prior_gain"
                        if runtime_mode == "prior" else "fail_closed_baseline_retained"
                    ),
                    "selected_for_runtime": runtime_mode != "none",
                    "runtime_mode": runtime_mode,
                })
                args._terminal_torque_prior_runtime_mode = runtime_mode
            terminal_angular_ranker_metrics["torque_prior"] = torque_prior_metrics
        elif bool(args.terminal_angular_ranker):
            terminal_angular_ranker_metrics["selection_status"] = "underpowered_terminal_validation"
        else:
            terminal_angular_ranker_metrics["selection_status"] = "disabled"
        # Audit the exact support used by the runtime contact selector.  In
        # the contact band, candidates above the declared cap are rejected
        # before ranking, so a full-bank top-1 is not a valid selection
        # statistic for posture closure.
        runtime_contact_mask = np.linalg.norm(val["action"], axis=1) <= (
            float(args.contact_action_max_magnitude) + 1.0e-6
        )
        runtime_contact_ranking = ranking_from_goal_states(
            val, val_prediction, goal_pose_by_row, args.effect_frame,
            candidate_mask=runtime_contact_mask,
        )
        selected_contact_knn_training: dict[str, Any] | None = None
        contact_knn_selection: dict[str, Any] = {
            "enabled": bool(args.contact_canonical_knn),
            "selected_for_runtime": False,
            "selection_rule": "fail_closed_when_validation_contact_true_regret_does_not_strictly_improve",
        }
        if contact_knn_training is not None:
            knn_val_prediction = val_prediction.copy()
            knn_contact_prediction = contact_canonical_knn_predict(
                contact_knn_training, val_features[contact_val_mask],
            )
            knn_agent = val_features[contact_val_mask, 1:3] * (PUSHT_STATE_SIZE / 2.0)
            knn_block = val_features[contact_val_mask, 11:13] * (PUSHT_STATE_SIZE / 2.0)
            knn_contact_prediction = contact_effect_to_world(
                knn_contact_prediction, knn_agent, knn_block,
            )
            knn_val_prediction[contact_val_mask] = knn_contact_prediction
            knn_runtime_ranking = ranking_from_goal_states(
                val, knn_val_prediction, goal_pose_by_row, "world",
                candidate_mask=runtime_contact_mask,
            )
            contact_knn_selection.update({
                "validation_contact_top1": knn_runtime_ranking["contact_top1_visual_current_proxy"],
                "validation_contact_mean_true_regret": knn_runtime_ranking["contact_mean_true_regret"],
                "validation_contact_roots": knn_runtime_ranking["contact_roots"],
                "reference_contact_top1": runtime_contact_ranking["contact_top1_visual_current_proxy"],
                "reference_contact_mean_true_regret": runtime_contact_ranking["contact_mean_true_regret"],
                "reference": "global_adapter_on_same_held_out_contact_support",
            })
            enough_knn_roots = int(knn_runtime_ranking["contact_roots"]) >= int(args.ranker_min_contact_roots)
            improves_regret = (
                np.isfinite(knn_runtime_ranking["contact_mean_true_regret"])
                and np.isfinite(runtime_contact_ranking["contact_mean_true_regret"])
                and float(knn_runtime_ranking["contact_mean_true_regret"])
                < float(runtime_contact_ranking["contact_mean_true_regret"])
            )
            if enough_knn_roots and improves_regret:
                selected_contact_knn_training = contact_knn_training
                contact_knn_selection.update({
                    "selected_for_runtime": True,
                    "selection_status": "held_out_contact_true_regret_gain",
                })
            else:
                contact_knn_selection["selection_status"] = "fail_closed_global_adapter_retained"
            # Keep the compact candidate summary and the detailed selection
            # record synchronized.  This is reporting state only: the
            # fail-closed decision above remains the sole runtime authority.
            contact_knn_metrics["selected_for_runtime"] = bool(
                contact_knn_selection["selected_for_runtime"]
            )
            contact_knn_metrics["selection_status"] = contact_knn_selection["selection_status"]
        runtime_ranker_hits: list[float] = []
        runtime_ranker_contact_hits: list[float] = []
        for root_id in np.unique(val["root_id"]):
            indices = np.flatnonzero(val["root_id"] == root_id)
            indices = indices[runtime_contact_mask[indices]]
            hit = float(np.argmin(val_rank_prediction[indices]) == np.argmin(val_rank_targets[indices]))
            runtime_ranker_hits.append(hit)
            if bool(val["contact_root"][indices[0]]):
                runtime_ranker_contact_hits.append(hit)
        ranker_metrics["held_out_runtime_support_top1"] = (
            float(np.mean(runtime_ranker_hits)) if runtime_ranker_hits else float("nan")
        )
        ranker_metrics["held_out_contact_runtime_support_top1"] = (
            float(np.mean(runtime_ranker_contact_hits))
            if runtime_ranker_contact_hits else float("nan")
        )
        ranker_metrics["runtime_support_actions_per_root"] = int(np.sum(runtime_contact_mask) // len(val_roots))
        ranker_metrics["runtime_support_definition"] = "zero plus absolute actions with magnitude <= contact_action_max_magnitude"
        mode_gate_selection: dict[str, Any] = {
            "enabled": bool(args.contact_mode_gate),
            "selected_for_runtime": False,
            "selected_penalty": 0.0,
            "candidates": [],
        }
        if args.contact_mode_gate and contact_mode_classifier is not None and contact_mode_training is not None:
            mode_candidates = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
            mode_rows = [contact_mode_gated_ranking(
                val, val_prediction, goal_pose_by_row, val_mode_probability,
                penalty, args.effect_frame, candidate_mask=runtime_contact_mask,
            ) for penalty in mode_candidates]
            mode_gate_selection["candidates"] = mode_rows
            baseline_contact = float(runtime_contact_ranking["contact_top1_visual_current_proxy"])
            eligible = [row for row in mode_rows if (
                int(row["held_out_contact_roots"]) >= int(args.ranker_min_contact_roots)
                and np.isfinite(row["held_out_contact_top1"])
            )]
            best_mode = max(eligible, key=lambda row: (row["held_out_contact_top1"], -row["penalty"])) if eligible else None
            if best_mode is not None and best_mode["held_out_contact_top1"] > baseline_contact:
                mode_gate_selection.update({
                    "selected_for_runtime": True,
                    "selected_penalty": float(best_mode["penalty"]),
                    "selection_reference": baseline_contact,
                    "selection_rule": "held_out_contact_top1 strictly exceeds effect baseline",
                })
                contact_mode_metrics["selected_for_runtime"] = True
            else:
                mode_gate_selection.update({
                    "selection_reference": baseline_contact,
                    "selection_rule": "fail_closed_when_no_held_out_contact_gain",
                })
            print(json.dumps({"phase": "contact_mode_selection", **mode_gate_selection}), flush=True)
        if contact_ranker is not None:
            contact_runtime_hits: list[float] = []
            contact_runtime_offset = 0
            for root_id in np.unique(val["root_id"]):
                indices = np.flatnonzero(val["root_id"] == root_id)
                if bool(val["contact_root"][indices[0]]):
                    supported = indices[runtime_contact_mask[indices]]
                    count = len(indices)
                    local_prediction = contact_rank_prediction[contact_runtime_offset:contact_runtime_offset + count]
                    contact_runtime_offset += count
                    contact_runtime_hits.append(float(
                        np.argmin(local_prediction[runtime_contact_mask[indices]])
                        == np.argmin(val_rank_targets[supported])
                    ))
            contact_ranker_metrics["held_out_runtime_support_top1"] = (
                float(np.mean(contact_runtime_hits)) if contact_runtime_hits else float("nan")
            )
            contact_ranker_metrics["runtime_support_actions_per_root"] = int(np.sum(runtime_contact_mask) // len(val_roots))
        # The ranker is consumed after RGB contact handoff, so a gain on
        # non-contact/terminal roots is not evidence for runtime selection.
        # Require agreement of *both* top-1 and true regret on exactly the
        # same held-out contact support.  A top-1-only gate can promote a
        # ranker after one lucky root while increasing the task cost on the
        # remaining roots; that failure mode caused the earlier false
        # promotion of the contact ranker.
        contact_reference = runtime_contact_ranking["contact_top1_visual_current_proxy"]
        contact_reference_regret = runtime_contact_ranking["contact_mean_true_regret"]
        global_ranker_support_metrics = held_out_cost_ranking(
            val, val_rank_prediction, val_rank_targets,
            runtime_contact_mask, contact_only=True,
        )
        contact_ranker_support_metrics: dict[str, Any] | None = None
        if contact_ranker is not None and contact_rank_prediction is not None:
            contact_ranker_full_scores = np.full(
                len(val["root_id"]), np.inf, dtype=np.float32,
            )
            contact_ranker_full_scores[contact_val_mask] = contact_rank_prediction
            contact_ranker_support_metrics = held_out_cost_ranking(
                val, contact_ranker_full_scores, val_rank_targets,
                runtime_contact_mask, contact_only=True,
            )
        ranker_metrics["held_out_contact_support_metrics"] = global_ranker_support_metrics
        if contact_ranker_support_metrics is not None:
            contact_ranker_metrics["held_out_contact_support_metrics"] = contact_ranker_support_metrics
        # The contact specialist is the model actually used on the runtime
        # contact manifold.  Its effect RMSE is not a sufficient deployment
        # criterion: ranking can change even when average regression error
        # improves.  Evaluate the specialist on exactly the same held-out
        # contact roots and runtime action support as the global adapter.
        contact_runtime_ranking: dict[str, Any] = {
            "contact_top1_visual_current_proxy": float("nan"),
            "contact_roots": 0,
            "action_effect_rmse": float("nan"),
        }
        if contact_val_prediction is not None:
            specialist_prediction = val_prediction.copy()
            specialist_prediction[contact_val_mask] = contact_val_prediction
            contact_runtime_ranking = ranking_from_goal_states(
                val, specialist_prediction, goal_pose_by_row, args.effect_frame,
                candidate_mask=runtime_contact_mask,
            )
            contact_adapter_metrics.update({
                "held_out_runtime_contact_top1": contact_runtime_ranking["contact_top1_visual_current_proxy"],
                "held_out_runtime_contact_roots": contact_runtime_ranking["contact_roots"],
                "selection_reference_global": contact_reference,
                "selection_rule": "strict held-out contact-support ranking improvement over global adapter",
            })
        global_contact_top1 = ranker_metrics["held_out_contact_runtime_support_top1"]
        enough_contact_roots = (
            int(ranker_metrics["held_out_contact_roots"])
            >= int(args.ranker_min_contact_roots)
        )
        global_ranker_improves = bool(
            enough_contact_roots
            and np.isfinite(global_ranker_support_metrics["top1"])
            and np.isfinite(global_ranker_support_metrics["mean_true_regret"])
            and global_ranker_support_metrics["top1"] >= contact_reference
            and global_ranker_support_metrics["mean_true_regret"] < contact_reference_regret
            and (
                global_ranker_support_metrics["top1"] > contact_reference
                or global_ranker_support_metrics["mean_true_regret"] < contact_reference_regret
            )
        )
        use_action_cost_ranker = global_ranker_improves
        selected_variant = "global" if use_action_cost_ranker else "none"
        contact_ranker_runtime_top1 = contact_ranker_metrics.get(
            "held_out_runtime_support_top1", float("nan")
        )
        contact_ranker_has_power = bool(
            contact_ranker is not None
            and int(contact_ranker_metrics.get("held_out_roots", 0))
            >= int(args.ranker_min_contact_roots)
        )
        contact_ranker_improves = bool(
            contact_ranker_has_power
            and contact_ranker_support_metrics is not None
            and np.isfinite(contact_ranker_support_metrics["top1"])
            and np.isfinite(contact_ranker_support_metrics["mean_true_regret"])
            and contact_ranker_support_metrics["top1"] >= contact_reference
            and contact_ranker_support_metrics["mean_true_regret"] < contact_reference_regret
            and (
                contact_ranker_support_metrics["top1"] > contact_reference
                or contact_ranker_support_metrics["mean_true_regret"] < contact_reference_regret
            )
        )
        if contact_ranker_improves:
            if not use_action_cost_ranker or contact_ranker_support_metrics["mean_true_regret"] < global_ranker_support_metrics["mean_true_regret"]:
                use_action_cost_ranker = True
                selected_variant = "contact_only"
        ranker_metrics["selected_for_runtime"] = use_action_cost_ranker
        ranker_metrics["selection_reference"] = contact_reference
        ranker_metrics["selection_reference_regret"] = contact_reference_regret
        ranker_metrics["selection_reference_scope"] = "held_out_contact_roots"
        ranker_metrics["minimum_contact_roots"] = int(args.ranker_min_contact_roots)
        ranker_metrics["selection_power_satisfied"] = bool(enough_contact_roots)
        ranker_metrics["selection_rule"] = "held_out contact top1 nondecrease and strict true-regret improvement"
        ranker_metrics["candidate_improves"] = bool(global_ranker_improves)
        ranker_metrics["selected_variant"] = selected_variant
        contact_ranker_metrics["candidate_improves"] = bool(contact_ranker_improves)
        contact_ranker_metrics["selected_for_runtime"] = bool(
            contact_ranker_improves and selected_variant == "contact_only"
        )
        contact_adapter_metrics["selected_for_runtime"] = bool(
            contact_val_prediction is not None
            and int(contact_runtime_ranking["contact_roots"]) >= int(args.ranker_min_contact_roots)
            and np.isfinite(contact_runtime_ranking["contact_top1_visual_current_proxy"])
            and np.isfinite(contact_runtime_ranking["contact_mean_true_regret"])
            and float(contact_runtime_ranking["contact_top1_visual_current_proxy"]) >= float(contact_reference)
            and float(contact_runtime_ranking["contact_mean_true_regret"]) < float(contact_reference_regret)
        )
        if not contact_adapter_metrics["selected_for_runtime"]:
            contact_adapter_metrics["selection_status"] = "fail_closed_global_adapter_retained"
        else:
            contact_adapter_metrics["selection_status"] = "held_out_contact_ranking_gain"
        print(json.dumps({"phase": "adapter", **ranking}), flush=True)

        # The requested observer and the observer authorized for closed-loop
        # execution are intentionally separate.  In particular, a paired
        # current/goal chart is an experiment candidate, not an automatic
        # runtime interface.  Earlier runs recorded
        # ``selected_for_runtime=False`` but still passed
        # ``paired_rgb_calibrated`` through the controller, which allowed a
        # ~40 px chart bias to turn a physically near-contact state into an
        # approach command.  Fail closed to the best already-gated observer.
        requested_current_xy_policy = str(args.current_xy_policy)
        runtime_observer_fallback_reason = "requested_protocol"
        if requested_current_xy_policy == "paired_rgb_calibrated":
            paired_selected = bool(
                paired_rgb_current_xy_metrics.get("selected_for_runtime", False)
            )
            if paired_selected:
                effective_current_xy_policy = "paired_rgb_calibrated"
                runtime_observer_fallback_reason = "paired_gate_passed"
            elif use_current_xy_decoder:
                effective_current_xy_policy = "validation"
                runtime_observer_fallback_reason = "paired_gate_failed_use_validated_current_decoder"
            elif use_rgb_current_xy:
                effective_current_xy_policy = "rgb_calibrated"
                runtime_observer_fallback_reason = "paired_gate_failed_use_validated_rgb_calibrator"
            else:
                effective_current_xy_policy = "rgb_mask"
                runtime_observer_fallback_reason = "paired_gate_failed_use_raw_rgb_mask"
        else:
            effective_current_xy_policy = requested_current_xy_policy
        args._runtime_current_xy_policy = effective_current_xy_policy
        paired_rgb_current_xy_metrics.update({
            "requested_for_runtime": requested_current_xy_policy,
            "effective_runtime_policy": effective_current_xy_policy,
            "runtime_fallback_reason": runtime_observer_fallback_reason,
        })

        # The paired RGB control uses the validated learned current-XY decoder
        # as an object-identity reference when available.  For an explicitly
        # selected paired chart this remains useful inside the paired branch;
        # for a fail-closed fallback it is the direct ``validation`` observer.
        if effective_current_xy_policy in {"paired_rgb_calibrated", "validation"} and use_current_xy_decoder:
            current_xy_runtime_decoder = current_xy_decoder
            current_xy_runtime_training = current_xy_decoder_training
        else:
            current_xy_runtime_decoder = (
                rgb_current_xy_decoder if use_rgb_current_xy
                or effective_current_xy_policy == "paired_rgb_calibrated"
                else (current_xy_decoder if use_current_xy_decoder else None)
            )
            current_xy_runtime_training = (
                rgb_current_xy_training if use_rgb_current_xy
                or effective_current_xy_policy == "paired_rgb_calibrated"
                else (current_xy_decoder_training if use_current_xy_decoder else None)
            )
        agent_xy_runtime_decoder = (
            agent_xy_decoder if use_joint_xy_decoder else None
        )
        agent_xy_runtime_training = (
            agent_xy_decoder_training if use_joint_xy_decoder else None
        )
        planning_adapter = run_planning(
            env, model, normalizer,
            contact_adapter if contact_adapter_metrics["selected_for_runtime"] else adapter,
            contact_adapter_training if contact_adapter_metrics["selected_for_runtime"] else adapter_training,
            args, args.planning_controller,
            selected_goal_decoder, selected_goal_training,
            selected_orientation_decoder, selected_orientation_training,
            goal_component_selection,
            goal_xy_decoder if goal_component_selection["xy"] else None,
            goal_xy_decoder_training if goal_component_selection["xy"] else None,
            goal_angle_decoder if goal_component_selection["angle"] else None,
            goal_angle_decoder_training if goal_component_selection["angle"] else None,
            current_xy_runtime_decoder,
            current_xy_runtime_training,
            contact_ranker if contact_ranker is not None and ranker_metrics["selected_variant"] == "contact_only" else (action_cost_ranker if use_action_cost_ranker else None),
            contact_ranker_training if contact_ranker is not None and ranker_metrics["selected_variant"] == "contact_only" else (action_cost_ranker_training if use_action_cost_ranker else None),
            terminal_angular_ranker if terminal_angular_ranker_metrics["selected_for_runtime"] else None,
            terminal_angular_ranker_training if terminal_angular_ranker_metrics["selected_for_runtime"] else None,
            contact_mode_classifier if mode_gate_selection["selected_for_runtime"] else None,
            contact_mode_training if mode_gate_selection["selected_for_runtime"] else None,
            float(mode_gate_selection["selected_penalty"]),
            selected_contact_knn_training,
            rgb_current_angle_decoder,
            rgb_current_angle_training,
            rgb_goal_angle_decoder,
            rgb_goal_angle_training,
            rgb_goal_xy_decoder if rgb_goal_xy_metrics["selected_for_runtime"] else None,
            rgb_goal_xy_training if rgb_goal_xy_metrics["selected_for_runtime"] else None,
            agent_xy_runtime_decoder,
            agent_xy_runtime_training,
        )
        planning_baseline = run_planning(
            env, model, normalizer, adapter, adapter_training, args, "geometry",
            selected_goal_decoder, selected_goal_training,
            selected_orientation_decoder, selected_orientation_training,
            goal_component_selection,
            goal_xy_decoder if goal_component_selection["xy"] else None,
            goal_xy_decoder_training if goal_component_selection["xy"] else None,
            goal_angle_decoder if goal_component_selection["angle"] else None,
            goal_angle_decoder_training if goal_component_selection["angle"] else None,
            current_xy_runtime_decoder,
            current_xy_runtime_training,
            None,
            None,
            None,
            None,
            None,
            None,
            0.0,
            None,
            rgb_current_angle_decoder,
            rgb_current_angle_training,
            rgb_goal_angle_decoder,
            rgb_goal_angle_training,
            rgb_goal_xy_decoder if rgb_goal_xy_metrics["selected_for_runtime"] else None,
            rgb_goal_xy_training if rgb_goal_xy_metrics["selected_for_runtime"] else None,
            agent_xy_runtime_decoder,
            agent_xy_runtime_training,
        )
    finally:
        env.close()
    def planning_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        traces = [item for row in rows for item in row["trace"]]
        torque_entries = [
            row for row in rows
            if any(item.get("phase") in {"torque", "hold"} for item in row["trace"])
        ]
        terminal_entries = [
            row for row in rows
            if any(float(item.get("block_xy", float("inf"))) <= float(args.terminal_xy)
                   for item in row["trace"])
        ]
        projection_steps = [
            item for item in traces
            if float(item.get("contact_action_projection_used", 0.0)) > 0.0
        ]
        reacquisition_attempts = [
            item for item in traces
            if float(item.get("reacquisition_attempt", 0.0)) > 0.0
        ]
        reacquisition_successes = [
            item for item in traces
            if float(item.get("reacquisition_success", 0.0)) > 0.0
        ]
        reacquisition_final_norms = [
            float(item["reacquisition_final_norm"])
            for item in traces
            if np.isfinite(float(item.get("reacquisition_final_norm", float("nan"))))
        ]
        reacquisition_raw_norms = [
            float(item["reacquisition_raw_norm"])
            for item in traces
            if np.isfinite(float(item.get("reacquisition_raw_norm", float("nan"))))
        ]
        reacquisition_post_projection_norms = [
            float(item["reacquisition_post_projection_final_norm"])
            for item in traces
            if np.isfinite(float(item.get("reacquisition_post_projection_final_norm", float("nan"))))
        ]
        contact_steps = [
            item for item in traces
            if float(item.get("contact_projection_contact_band", 0.0)) > 0.0
        ]
        raw_inward = [
            float(item["contact_projection_raw_inward"])
            for item in contact_steps
            if np.isfinite(float(item.get("contact_projection_raw_inward", float("nan"))))
        ]
        final_inward = [
            float(item["contact_projection_final_inward"])
            for item in contact_steps
            if np.isfinite(float(item.get("contact_projection_final_inward", float("nan"))))
        ]
        return {
            "episodes": rows,
            "strict_success_count": int(sum(bool(row["strict_success"]) for row in rows)),
            "strict_success_rate": float(np.mean([row["strict_success"] for row in rows])) if rows else None,
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])) if rows else None,
            "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])) if rows else None,
            "contact_retention_rate": float(np.mean([
                float(item.get("agent_block_distance", float("inf")))
                <= float(args.contact_recovery_distance)
                for item in traces
            ])) if traces else None,
            "terminal_tube_entry_rate": float(len(terminal_entries) / len(rows)) if rows else None,
            "torque_phase_entry_rate": float(len(torque_entries) / len(rows)) if rows else None,
            "reacquisition_attempt_count": int(len(reacquisition_attempts)),
            "reacquisition_success_count": int(len(reacquisition_successes)),
            "reacquisition_success_rate": float(
                len(reacquisition_successes) / len(reacquisition_attempts)
            ) if reacquisition_attempts else None,
            "reacquisition_budget_exhausted_count": int(sum(
                float(item.get("reacquisition_budget_exhausted", 0.0)) > 0.0
                for item in traces
            )),
            "reacquisition_arc_motion_abort_count": int(sum(
                float(item.get("reacquisition_arc_motion_abort", 0.0)) > 0.0
                for item in traces
            )),
            "reacquisition_route_abort_count": int(sum(
                float(item.get("reacquisition_route_abort", 0.0)) > 0.0
                for item in traces
            )),
            "reacquisition_route_lockout_count": int(sum(
                float(item.get("reacquisition_route_lockout", 0.0)) > 0.0
                for item in traces
            )),
            "reacquisition_bounded_translation_repair_count": int(sum(
                float(item.get("reacquisition_bounded_translation_repair", 0.0)) > 0.0
                for item in traces
            )),
            "reacquisition_hold_action_count": int(sum(
                item.get("phase") == "hold" for item in traces
            )),
            "agent_coordinate_sentinel_count": int(sum(
                np.isfinite(np.asarray(item.get("audit_observed_agent_xy", [np.nan, np.nan]), dtype=np.float32)).all()
                and float(np.linalg.norm(
                    np.asarray(item.get("audit_observed_agent_xy", [np.nan, np.nan]), dtype=np.float32) - 256.0
                )) <= 1.0e-3
                for item in traces
            )),
            "reacquisition_max_raw_norm": float(max(reacquisition_raw_norms))
            if reacquisition_raw_norms else None,
            "reacquisition_max_final_norm": float(max(reacquisition_final_norms))
            if reacquisition_final_norms else None,
            "reacquisition_max_post_projection_final_norm": float(max(reacquisition_post_projection_norms))
            if reacquisition_post_projection_norms else None,
            "reacquisition_cap_violation_count": int(sum(
                norm > float(getattr(args, "reacquisition_action_max_magnitude", 0.35)) + 1.0e-6
                for norm in reacquisition_final_norms
            )),
            "projection_use_rate_over_contact_steps": float(
                len(projection_steps) / len(contact_steps)
            ) if contact_steps else None,
            "projection_mean_correction_norm": float(np.mean([
                float(item.get("contact_projection_correction_norm", 0.0))
                for item in projection_steps
            ])) if projection_steps else 0.0,
            "projection_mean_raw_inward": float(np.mean(raw_inward)) if raw_inward else None,
            "projection_mean_final_inward": float(np.mean(final_inward)) if final_inward else None,
        }
    adapter_summary = planning_summary(planning_adapter)
    baseline_summary = planning_summary(planning_baseline)
    if bool(terminal_angular_ranker_metrics.get("selected_for_runtime", False)) and (
        terminal_angular_ranker is None or terminal_angular_ranker_training is None
    ):
        raise RuntimeError(
            "terminal ranker passed the held-out gate but was not available for serialization"
        )
    protocol_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_runtime_")
    }
    torch.save({
        "adapter": adapter.state_dict(), "adapter_training": adapter_training,
        "contact_adapter": contact_adapter.state_dict() if contact_adapter is not None else None,
        "contact_adapter_training": contact_adapter_training,
        "goal_decoder": goal_decoder.state_dict(), "goal_decoder_training": goal_decoder_training,
        "goal_xy_decoder": goal_xy_decoder.state_dict(),
        "goal_xy_decoder_training": goal_xy_decoder_training,
        "goal_angle_decoder": goal_angle_decoder.state_dict(),
        "goal_angle_decoder_training": goal_angle_decoder_training,
        "rgb_goal_xy_decoder": readout_state_dict(rgb_goal_xy_decoder),
        "rgb_goal_xy_training": rgb_goal_xy_training,
        "rgb_current_xy_decoder": readout_state_dict(rgb_current_xy_decoder),
        "rgb_current_xy_training": rgb_current_xy_training,
        "rgb_goal_angle_decoder": readout_state_dict(rgb_goal_angle_decoder),
        "rgb_goal_angle_training": rgb_goal_angle_training,
        "orientation_decoder": orientation_decoder.state_dict(),
        "orientation_decoder_training": orientation_decoder_training,
        "current_xy_decoder": current_xy_decoder.state_dict(),
        "current_xy_decoder_training": current_xy_decoder_training,
        "agent_xy_decoder": readout_state_dict(agent_xy_decoder),
        "agent_xy_decoder_training": agent_xy_decoder_training,
        "action_cost_ranker": action_cost_ranker.state_dict(),
        "action_cost_ranker_training": action_cost_ranker_training,
        "legacy_action_cost_ranker": legacy_action_cost_ranker.state_dict(),
        "legacy_action_cost_ranker_training": legacy_action_cost_ranker_training,
        "contact_ranker": contact_ranker.state_dict() if contact_ranker is not None else None,
        "contact_ranker_training": contact_ranker_training,
        "terminal_angular_ranker": (
            terminal_angular_ranker.state_dict()
            if terminal_angular_ranker is not None else None
        ),
        "terminal_angular_ranker_training": terminal_angular_ranker_training,
        "terminal_angular_ranker_metrics": terminal_angular_ranker_metrics,
        "contact_mode_classifier": contact_mode_classifier.state_dict() if contact_mode_classifier is not None else None,
        "contact_mode_training": contact_mode_training,
        "contact_mode_selection": mode_gate_selection,
        "contact_knn_training": contact_knn_training,
        "contact_knn_selection": contact_knn_selection,
        "input_dim": int(train_features.shape[1]), "hidden_dim": int(args.hidden_dim),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
    }, output / "rgb_local_effect_adapter.pt")
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            "adapter": "frozen mask-moment RGB features -> zero baseline + causal action effect",
            "action_bank": actions.tolist(),
            "train_roots": len(train_roots), "validation_roots": len(val_roots),
            "train_bootstrap_roots": len(train_bootstrap_roots),
            "validation_bootstrap_roots": len(val_bootstrap_roots),
            "root_sampling": {
                "balanced_roots": "randomized balanced/contact roots",
                "terminal_roots": "offline terminal pose manifold",
                "bootstrap_roots": "frozen RGB controller visitation roots; labels generated offline",
                "bootstrap_root_fraction": float(args.bootstrap_root_fraction),
                "bootstrap_checkpoint": str(args.bootstrap_checkpoint.resolve()) if args.bootstrap_checkpoint else None,
            },
            "train_validation_root_disjoint": True,
            "runtime_input": "current/previous RGB, goal RGB, candidate action",
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "current_xy_coordinate_protocol": (
                "joint_physical_xy_chart_block_decoder_plus_validated_agent_moment"
                if bool(use_joint_xy_decoder)
                else "legacy_single_block_decoder_or_declared_fallback"
            ),
            "goal_observation": args.goal_observation,
            "online_local_identification": "RGB block centroid/orientation finite differences; fixed ridge blend after minimum non-collinear history",
            "effect_frame": args.effect_frame,
            "offline_label_role": "matched simulator zero/action interventions only",
            "physical_steps_per_action": int(args.action_repeat),
            "wall_seconds": time.perf_counter() - started,
            **protocol_args,
        },
        "training": adapter_training,
        "goal_decoder": goal_decoder_training,
        "held_out_goal_readout": goal_readout_metrics,
        "goal_readout_selection": goal_selection,
        "goal_interface_selection": goal_interface_selection,
        "goal_xy_decoder": goal_xy_decoder_training,
        "rgb_goal_xy_decoder": rgb_goal_xy_metrics,
        "goal_angle_decoder": goal_angle_decoder_training,
        "goal_component_selection": goal_component_selection,
        "orientation_decoder": orientation_decoder_training,
        "held_out_orientation_readout": orientation_readout_metrics,
        "orientation_readout_selection": orientation_selection,
        "rgb_angle_readout": rgb_angle_readout_metrics,
        "current_xy_decoder": current_xy_decoder_training,
        "agent_xy_decoder": agent_xy_metrics,
        "joint_xy_decoder_selected": bool(use_joint_xy_decoder),
        "held_out_current_xy_readout": current_xy_metrics,
        "rgb_current_xy_readout": rgb_current_xy_metrics,
        "paired_rgb_current_xy_readout": paired_rgb_current_xy_metrics,
        "terminal_threshold_selection": terminal_threshold_selection,
        "held_out_ranking": ranking,
        "held_out_runtime_contact_ranking": runtime_contact_ranking,
        "held_out_legacy_action_cost_ranker": legacy_ranker_metrics,
        "held_out_action_cost_ranker": ranker_metrics,
        "held_out_contact_action_cost_ranker": contact_ranker_metrics,
        "held_out_terminal_angular_ranker": terminal_angular_ranker_metrics,
        "contact_adapter": contact_adapter_metrics,
        "contact_canonical_knn": contact_knn_metrics,
        "contact_canonical_knn_selection": contact_knn_selection,
        "contact_mode": contact_mode_metrics,
        "contact_mode_selection": mode_gate_selection,
            "planning": {"adapter": adapter_summary, "geometry_baseline": baseline_summary},
        "sanity_checks": {
            "finite_features": bool(np.isfinite(train_features).all() and np.isfinite(val_features).all()),
            "finite_predictions": bool(np.isfinite(val_prediction).all()),
            "finite_geometry_ranker_features": bool(
                np.isfinite(geometry_train_rank_features).all()
                and np.isfinite(geometry_val_rank_features).all()
            ),
            "finite_terminal_geometry_ranker_features": bool(
                np.isfinite(terminal_geometry_train_rank_features).all()
                and np.isfinite(terminal_geometry_val_rank_features).all()
            ),
            "zero_first_action": bool(np.allclose(actions[0], 0.0)),
            "source_root_disjoint": True,
            "finite_contact_mode_probability": bool(np.isfinite(val_mode_probability).all()),
            "finite_contact_knn": bool(
                contact_knn_training is None
                or np.isfinite(contact_knn_training["features"]).all()
                and np.isfinite(contact_knn_training["targets"]).all()
            ),
        },
        "claim_boundary": "RGB-only local-effect adapter diagnostic. A positive result would support local action-effect observability on PushT, not universal causal dynamics or universal world-model capability.",
    }
    (output / "rgb_local_effect_adapter.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "rgb_local_effect_adapter.md").write_text(
        "# RGB-only local-effect adapter\n\n"
        f"- Held-out action-effect RMSE: {ranking['action_effect_rmse']:.6g}.\n"
        f"- Held-out candidate top-1 with oracle current pose: {ranking['top1_oracle_current']:.3f}.\n"
        f"- Adapter strict visual success: {adapter_summary['strict_success_count']}/{len(planning_adapter)}.\n"
        f"- Geometry baseline strict visual success: {baseline_summary['strict_success_count']}/{len(planning_baseline)}.\n\n"
        f"- Adapter reacquisition attempts/successes: {adapter_summary['reacquisition_attempt_count']}/{adapter_summary['reacquisition_success_count']}.\n"
        f"- Terminal angular ranker: {terminal_angular_ranker_metrics.get('selection_status', 'not_run')}.\n"
        "The runtime controller consumes RGB frames and candidate actions only; simulator state is offline label/evaluator-only.\n"
    )
    print(json.dumps({"output": str(output), "ranking": ranking, "adapter": planning_summary(planning_adapter), "baseline": planning_summary(planning_baseline)}, indent=2))


if __name__ == "__main__":
    main()
