"""Evaluate a PlanWAM-style planning-shaped control on a sealed PointMaze panel.

This is an *execution-supervised control*, not a claim that the frozen LeWM
already supplies a planning-sufficient representation.  It separates three
training objectives using the exact same decision-time feature contract:

``cost_regression``
    predicts terminal task cost only;
``ranking_augmented``
    adds within-root candidate-pair ranking supervision;
``planning_shaped``
    additionally predicts the realized terminal effect.

The last arm is the minimal analogue of planning-shaped foresight: a shared
history/action representation is constrained both by a real future consequence
and by the task ranking that consequence must support.  Training sees executed
train-root effects and costs only.  Calibration labels choose one frozen arm;
outcome-root terminal states are executed only after that selection.  Thus a
positive result is a narrowly scoped, visual-feature execution control, not a
D-JEPA teacher, GDD distillation, or deployment result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_planning_shaped_execution_control.v0.4"
CONTROL_LABEL = "train_only_visual_history_action_planning_shaped_control"
V8_PREFLIGHT_SCHEMA = "pointmaze_shared_future_v8_static_preflight.v1"
V8_SPLIT_COUNTS = {"train": 128, "calibration": 192, "outcome": 64}

# This declaration is shared by the direct-cost, planning-shaped, and
# endpoint-readout controls.  Keeping it in one place prevents a benign
# looking metadata difference from turning into a representation side channel
# (or, worse, from stopping a CUDA panel only after all models have trained).
SUMMARY_ACTION_REPRESENTATIONS = frozenset({
    "summary_mean_std_first_last",
    "canonical_summary_mean_std_first_last",
})
ORDERED_ACTION_REPRESENTATIONS = frozenset({"full_ordered_20x2"})


def action_representation_input_fields(action_representation: str) -> list[str]:
    """Return the exact decision-time schema for one declared action basis.

    An unknown label is a contract error, not an implicit ordered feature.
    This matters for the V6 factorial comparison, whose primary estimand holds
    the full ordered action sequence fixed while changing only the readout.
    """

    representation = str(action_representation)
    if representation in SUMMARY_ACTION_REPRESENTATIONS:
        action_field = "action_sequence_summary"
    elif representation in ORDERED_ACTION_REPRESENTATIONS:
        action_field = "action_sequence"
    else:
        raise ValueError(f"unknown action representation: {representation}")
    return ["source_history_model", "goal_raw", action_field]


@dataclass(frozen=True)
class Objective:
    name: str
    effect_weight: float
    rank_weight: float


OBJECTIVES = (
    Objective("cost_regression", effect_weight=0.0, rank_weight=0.0),
    Objective("ranking_augmented", effect_weight=0.0, rank_weight=0.25),
    Objective("planning_shaped", effect_weight=0.25, rank_weight=0.25),
)


def _shared_future_feature(row: dict[str, Any]) -> np.ndarray:
    """Action-independent input to a PlanWAM-inspired shared-future context.

    This is intentionally distinct from ``_feature``: candidate actions are
    forbidden here and enter only the subsequent scoring head.  It is a
    lightweight scientific control inspired by shared-future planning, not a
    reproduction or evaluation of PlanWAM itself.
    """

    history = np.asarray(row.get("source_history_model"), dtype=np.float64).reshape(-1)
    goal = np.asarray(row.get("goal_raw"), dtype=np.float64).reshape(-1)
    if history.size < 1 or not np.isfinite(history).all():
        raise ValueError(f"{row.get('candidate_id')}: missing finite source_history_model")
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError(f"{row.get('candidate_id')}: goal_raw must be finite [2]")
    return np.concatenate((history, goal))


def _shared_future_action_feature(
    row: dict[str, Any], action_representation: str = "summary_mean_std_first_last",
) -> np.ndarray:
    """Candidate-only score input paired with a shared-future context."""

    if action_representation in SUMMARY_ACTION_REPRESENTATIONS:
        return _action_summary(row)
    if action_representation in ORDERED_ACTION_REPRESENTATIONS:
        action = np.asarray(row.get("action_sequence"), dtype=np.float64)
        if action.shape != (20, 2) or not np.isfinite(action).all():
            raise ValueError("full_ordered_20x2 requires finite action_sequence [20,2]")
        return action.reshape(-1)
    raise ValueError(f"unknown action representation: {action_representation}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _canonical_action_sequence(value: Any) -> str:
    """Return an exact, finite V8 action identity without a summary shortcut."""

    action = np.asarray(value, dtype=np.float64)
    if action.shape != (20, 2) or not np.isfinite(action).all():
        raise ValueError("V8 candidates require finite full_ordered_20x2 actions")
    return json.dumps(action.round(12).tolist(), separators=(",", ":"))


def _validate_v8_shared_future_contract(source: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Refuse a V8 fit unless its live panel still matches the frozen ledger.

    The builder validates the ledger when it creates a panel, but that is not
    enough: a later direct invocation of this runner could otherwise point at
    a different source/context pair and still produce a plausible report.
    This execution-boundary check binds the exact action identity and the
    train/calibration/outcome root identities before any model is fitted.
    """

    metadata = context.get("metadata")
    source_rows = source.get("records")
    context_rows = context.get("records")
    if not isinstance(metadata, dict) or not isinstance(source_rows, list) or not isinstance(context_rows, list):
        raise ValueError("V8 requires source/context metadata and record arrays")
    if metadata.get("candidate_generalization_regime") != "shared_candidate_in_support":
        raise ValueError("V8 requires candidate_generalization_regime=shared_candidate_in_support")
    if metadata.get("candidate_design") != "goal_balanced_interaction_pairs":
        raise ValueError("V8 requires candidate_design=goal_balanced_interaction_pairs")
    if metadata.get("scorer_candidate_bank_protocol") != "held_exact_shared_bank":
        raise ValueError("V8 requires scorer_candidate_bank_protocol=held_exact_shared_bank")
    if metadata.get("outcome_labels_materialized_during_build") is not False:
        raise ValueError("V8 context must reserve outcome labels until after calibration selection")
    receipt = metadata.get("frozen_preflight_ledger")
    if not isinstance(receipt, dict) or receipt.get("schema") != V8_PREFLIGHT_SCHEMA or receipt.get("status") != "passed_static_preflight":
        raise ValueError("V8 requires a passed static-preflight provenance receipt")
    ledger_path = receipt.get("path")
    if not isinstance(ledger_path, str) or not Path(ledger_path).is_file():
        raise ValueError("V8 static-preflight receipt lacks a readable ledger path")
    if receipt.get("sha256") != _sha256(Path(ledger_path)):
        raise ValueError("V8 static-preflight ledger bytes differ from the panel receipt")
    ledger = _load(Path(ledger_path))
    checks = ledger.get("checks")
    bank = ledger.get("candidate_bank")
    if (
        ledger.get("schema") != V8_PREFLIGHT_SCHEMA
        or ledger.get("status") != "passed_static_preflight"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or ledger.get("root_counts") != V8_SPLIT_COUNTS
        or not isinstance(bank, dict)
        or bank.get("shared_candidate_in_support") is not True
        or bank.get("scorer_candidate_bank_protocol") != "held_exact_shared_bank"
        or bank.get("shared_future_action_representation") != "full_ordered_20x2"
    ):
        raise ValueError("V8 static ledger lacks passed root/support/shared-future evidence")

    train_rows = [row for row in source_rows if isinstance(row, dict) and str(row.get("split")) == "train"]
    if not train_rows or len(train_rows) % V8_SPLIT_COUNTS["train"]:
        raise ValueError("V8 source records do not encode an exact train-root panel")
    held_by_split: dict[str, list[dict[str, Any]]] = {"calibration": [], "outcome": []}
    for row in context_rows:
        if not isinstance(row, dict):
            raise ValueError("V8 context rows must be objects")
        split = str(row.get("split"))
        if split in held_by_split:
            held_by_split[split].append(row)
    if not held_by_split["calibration"] or not held_by_split["outcome"]:
        raise ValueError("V8 context requires calibration and outcome rows")

    def action_map(rows: list[dict[str, Any]], *, label: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in rows:
            candidate_id = str(row.get("candidate_id"))
            identity = _canonical_action_sequence(row.get("action_sequence"))
            existing = result.setdefault(candidate_id, identity)
            if existing != identity:
                raise ValueError(f"{label}: action sequence changes for candidate {candidate_id!r}")
        return result

    train_actions = action_map(train_rows, label="V8 train")
    held_actions = action_map(held_by_split["calibration"] + held_by_split["outcome"], label="V8 held")
    if len(train_actions) < 2 or train_actions != held_actions:
        raise ValueError("V8 scorer source and held panels must have the same exact fixed candidate bank")
    expected_roots = ledger.get("split_roots")
    if not isinstance(expected_roots, dict):
        raise ValueError("V8 static ledger lacks split root identities")
    observed_roots = {
        "train": {int(row.get("start_episode_id")) for row in train_rows},
        "calibration": {int(row.get("start_episode_id")) for row in held_by_split["calibration"]},
        "outcome": {int(row.get("start_episode_id")) for row in held_by_split["outcome"]},
    }
    for split, count in V8_SPLIT_COUNTS.items():
        expected = {int(value) for value in expected_roots.get(split, [])}
        if observed_roots[split] != expected or len(observed_roots[split]) != count:
            raise ValueError(f"V8 {split} roots differ from the frozen static ledger")
    overlaps = {
        "train_calibration": observed_roots["train"] & observed_roots["calibration"],
        "train_outcome": observed_roots["train"] & observed_roots["outcome"],
        "calibration_outcome": observed_roots["calibration"] & observed_roots["outcome"],
    }
    if any(overlaps.values()):
        raise ValueError("V8 train/calibration/outcome roots must be pairwise disjoint")
    return {
        "verified": True,
        "ledger_path": str(Path(ledger_path).resolve()),
        "ledger_sha256": str(receipt["sha256"]),
        "root_counts": {name: len(values) for name, values in observed_roots.items()},
        "candidate_count": len(train_actions),
        "action_representation": "full_ordered_20x2",
    }


def _action_summary(row: dict[str, Any]) -> np.ndarray:
    action = np.asarray(row.get("action_sequence"), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 2 or not len(action) or not np.isfinite(action).all():
        raise ValueError(f"{row.get('candidate_id')}: action_sequence must be finite [steps, 2]")
    return np.concatenate((action.mean(axis=0), action.std(axis=0), action[0], action[-1], [float(len(action))]))


def _feature(row: dict[str, Any]) -> np.ndarray:
    """Only decision-time visual history, declared goal, and action enter here."""

    history = np.asarray(row.get("source_history_model"), dtype=np.float64).reshape(-1)
    goal = np.asarray(row.get("goal_raw"), dtype=np.float64).reshape(-1)
    if history.size < 1 or not np.isfinite(history).all():
        raise ValueError(f"{row.get('candidate_id')}: missing finite source_history_model")
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError(f"{row.get('candidate_id')}: goal_raw must be finite [2]")
    return np.concatenate((history, goal, _action_summary(row)))


def _canonical_action_summary(row: dict[str, Any]) -> np.ndarray:
    """Return an exactly order-invariant version of the legacy action summary.

    The historical encoder is preserved above for V1--V5 reproducibility.
    This helper is only an information-ablation control: sorting the action
    rows before reducing removes floating-point summation-order residue while
    retaining the legacy summary's mean, standard deviation, first action,
    last action, and length fields.
    """

    action = np.asarray(row.get("action_sequence"), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 2 or not len(action) or not np.isfinite(action).all():
        raise ValueError(f"{row.get('candidate_id')}: action_sequence must be finite [steps, 2]")
    canonical = action[np.lexsort((action[:, 1], action[:, 0]))]
    return np.concatenate((canonical.mean(axis=0), canonical.std(axis=0), action[0], action[-1], [float(len(action))]))


def _feature_canonical_action_summary(row: dict[str, Any]) -> np.ndarray:
    """Decision-time feature for a strict order-information ablation."""

    history = np.asarray(row.get("source_history_model"), dtype=np.float64).reshape(-1)
    goal = np.asarray(row.get("goal_raw"), dtype=np.float64).reshape(-1)
    if history.size < 1 or not np.isfinite(history).all():
        raise ValueError(f"{row.get('candidate_id')}: missing finite source_history_model")
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError(f"{row.get('candidate_id')}: goal_raw must be finite [2]")
    return np.concatenate((history, goal, _canonical_action_summary(row)))


def _feature_ordered_action_sequence(row: dict[str, Any]) -> np.ndarray:
    """Encode the complete declared 20-step candidate pulse in execution order.

    This is intentionally a V6-only representation.  The legacy summary
    encoder above remains the historical V1--V5 interface, so subsequent V6
    evidence cannot silently revise those controls.  Like ``_feature``, this
    function may read only decision-time visual history, the declared goal,
    and the candidate action; simulator state is forbidden.
    """

    history = np.asarray(row.get("source_history_model"), dtype=np.float64).reshape(-1)
    goal = np.asarray(row.get("goal_raw"), dtype=np.float64).reshape(-1)
    action = np.asarray(row.get("action_sequence"), dtype=np.float64)
    if history.size < 1 or not np.isfinite(history).all():
        raise ValueError(f"{row.get('candidate_id')}: missing finite source_history_model")
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError(f"{row.get('candidate_id')}: goal_raw must be finite [2]")
    if action.shape != (20, 2) or not np.isfinite(action).all():
        raise ValueError(f"{row.get('candidate_id')}: V6 action_sequence must be finite [20, 2]")
    return np.concatenate((history, goal, action.reshape(-1)))


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        root = str(row.get("instance_id", ""))
        if not root:
            raise ValueError("every record needs instance_id")
        result.setdefault(root, []).append(index)
    if any(len(indices) < 2 for indices in result.values()):
        raise ValueError("each root needs at least two candidates")
    return result


def _terminal_from_ref(row: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    reference = str(row.get("realized_terminal_state_ref", ""))
    if "arrays.npz::" not in reference:
        raise ValueError(f"{row.get('candidate_id')}: missing realized_terminal_state_ref")
    key = reference.split("arrays.npz::", 1)[1]
    if key not in arrays:
        raise ValueError(f"arrays.npz: missing terminal state {key}")
    state = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("terminal state must be finite [4]")
    return state


def _fresh_terminal(rows: list[dict[str, Any]]) -> np.ndarray:
    """Execute sealed outcome candidates only after configuration is frozen."""

    import mujoco
    from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _legacy_pointmaze_xml

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    values: list[np.ndarray] = []
    for row in rows:
        state = np.asarray(row.get("source_state_raw"), dtype=np.float64).reshape(-1)
        action = np.asarray(row.get("action_sequence"), dtype=np.float64)
        if state.shape != (4,) or action.ndim != 2 or action.shape[1] != 2:
            raise ValueError("outcome execution requires source_state_raw[4] and action_sequence[T,2]")
        data = mujoco.MjData(model)
        data.qpos[:] = state[:2]
        data.qvel[:] = state[2:]
        mujoco.mj_forward(model, data)
        for pulse in action:
            data.ctrl[:] = np.clip(pulse, -1.0, 1.0)
            mujoco.mj_step(model, data)
        values.append(np.r_[data.qpos[:2], data.qvel[:2]].astype(np.float64))
    return np.stack(values)


def _cost(rows: list[dict[str, Any]], terminal: np.ndarray) -> np.ndarray:
    return np.asarray([
        float(np.linalg.norm(value[:2] - np.asarray(row["goal_raw"], dtype=np.float64)))
        for row, value in zip(rows, terminal)
    ], dtype=np.float64)


def _pair_indices(groups: dict[str, list[int]], cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preferred: list[int] = []
    rejected: list[int] = []
    for indices in groups.values():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1:]:
                if np.isclose(cost[left], cost[right], rtol=0.0, atol=1.0e-10):
                    continue
                if cost[left] < cost[right]:
                    preferred.append(left)
                    rejected.append(right)
                else:
                    preferred.append(right)
                    rejected.append(left)
    if not preferred:
        raise ValueError("train roots contain no non-tied candidate pairs")
    return np.asarray(preferred, dtype=np.int64), np.asarray(rejected, dtype=np.int64)


def _fit(
    features: np.ndarray,
    effect: np.ndarray,
    cost: np.ndarray,
    groups: dict[str, list[int]],
    objective: Objective,
    *,
    width: int,
    steps: int,
    device: str,
    seed: int,
    action_representation: str = "summary_mean_std_first_last",
) -> tuple[dict[str, Any], Any]:
    """Fit a shared prior with task and (optionally) future-effect supervision."""

    import torch

    if width < 2 or steps < 1:
        raise ValueError("width >= 2 and steps >= 1 are required")
    torch.manual_seed(int(seed))
    x_mean, x_scale = features.mean(axis=0), features.std(axis=0)
    x_scale[x_scale < 1.0e-8] = 1.0
    effect_mean, effect_scale = effect.mean(axis=0), effect.std(axis=0)
    effect_scale[effect_scale < 1.0e-8] = 1.0
    cost_mean, cost_scale = float(cost.mean()), max(float(cost.std()), 1.0e-6)
    x = torch.as_tensor((features - x_mean) / x_scale, dtype=torch.float32, device=device)
    y_effect = torch.as_tensor((effect - effect_mean) / effect_scale, dtype=torch.float32, device=device)
    y_cost = torch.as_tensor((cost - cost_mean) / cost_scale, dtype=torch.float32, device=device).reshape(-1, 1)
    preferred, rejected = _pair_indices(groups, cost)
    p = torch.as_tensor(preferred, dtype=torch.long, device=device)
    r = torch.as_tensor(rejected, dtype=torch.long, device=device)

    class _Prior(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim), torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.GELU(),
            )
            self.cost = torch.nn.Linear(hidden_dim, 1)
            self.effect = torch.nn.Linear(hidden_dim, 4)

        def forward(self, values: Any) -> tuple[Any, Any]:
            hidden = self.trunk(values)
            return self.cost(hidden), self.effect(hidden)

    model = _Prior(x.shape[1], int(width)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1.0e-4)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_cost, predicted_effect = model(x)
        cost_loss = torch.nn.functional.mse_loss(predicted_cost, y_cost)
        effect_loss = torch.nn.functional.mse_loss(predicted_effect, y_effect)
        # A lower cost should receive a lower predicted cost.  This is the
        # candidate-ranking objective absent from an effect-only predictor.
        rank_loss = torch.nn.functional.softplus(predicted_cost[p] - predicted_cost[r]).mean()
        loss = cost_loss + objective.effect_weight * effect_loss + objective.rank_weight * rank_loss
        loss.backward()
        optimizer.step()

    def predict(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            tensor = torch.as_tensor((values - x_mean) / x_scale, dtype=torch.float32, device=device)
            predicted_cost, predicted_effect = model(tensor)
        return (
            predicted_cost.reshape(-1).detach().cpu().numpy().astype(np.float64) * cost_scale + cost_mean,
            predicted_effect.detach().cpu().numpy().astype(np.float64) * effect_scale + effect_mean,
        )

    contract = {
        "objective": objective.name,
        "effect_weight": objective.effect_weight,
        "rank_weight": objective.rank_weight,
        "width": int(width),
        "steps": int(steps),
        "initialization_seed": int(seed),
        "feature_dimension": int(features.shape[1]),
        "train_pair_count": int(len(preferred)),
        "input_fields": action_representation_input_fields(action_representation),
        "action_representation": str(action_representation),
        "target_fields": ["executed_terminal_effect", "executed_task_cost"],
        "outcome_root_input_forbidden": True,
    }
    return contract, predict


def _fit_shared_future_scorer(
    shared_features: np.ndarray,
    action_features: np.ndarray,
    effect: np.ndarray,
    cost: np.ndarray,
    groups: dict[str, list[int]],
    *, width: int, steps: int, device: str, seed: int,
    action_representation: str = "summary_mean_std_first_last",
) -> tuple[dict[str, Any], Any]:
    """Fit an action-independent future context plus candidate score head.

    The context encoder receives only history/goal.  Its auxiliary target is
    the root-level mean realized effect, which is shared by construction over
    that root's candidate set.  The score head may read the declared action,
    as a candidate evaluator normally would, but cannot route action detail
    through the predicted future context itself.
    """

    import torch

    shared = np.asarray(shared_features, dtype=np.float64)
    action = np.asarray(action_features, dtype=np.float64)
    targets = np.asarray(effect, dtype=np.float64)
    costs = np.asarray(cost, dtype=np.float64).reshape(-1)
    if shared.ndim != 2 or action.ndim != 2 or targets.shape != (len(shared), 4) or len(action) != len(shared):
        raise ValueError("shared/action/effect arrays have incompatible shapes")
    if width < 2 or steps < 1:
        raise ValueError("width >= 2 and steps >= 1 are required")
    # Every row for a root obtains the same target.  This prevents a hidden
    # candidate-action path from entering the shared-future supervision.
    root_effect = np.empty_like(targets)
    for indices in groups.values():
        root_effect[indices] = targets[indices].mean(axis=0, keepdims=True)
    torch.manual_seed(int(seed))
    shared_mean, shared_scale = shared.mean(axis=0), shared.std(axis=0)
    action_mean, action_scale = action.mean(axis=0), action.std(axis=0)
    shared_scale[shared_scale < 1.0e-8] = 1.0
    action_scale[action_scale < 1.0e-8] = 1.0
    effect_mean, effect_scale = root_effect.mean(axis=0), root_effect.std(axis=0)
    effect_scale[effect_scale < 1.0e-8] = 1.0
    cost_mean, cost_scale = float(costs.mean()), max(float(costs.std()), 1.0e-6)
    x_shared = torch.as_tensor((shared - shared_mean) / shared_scale, dtype=torch.float32, device=device)
    x_action = torch.as_tensor((action - action_mean) / action_scale, dtype=torch.float32, device=device)
    y_effect = torch.as_tensor((root_effect - effect_mean) / effect_scale, dtype=torch.float32, device=device)
    y_cost = torch.as_tensor((costs - cost_mean) / cost_scale, dtype=torch.float32, device=device).reshape(-1, 1)
    preferred, rejected = _pair_indices(groups, costs)
    p = torch.as_tensor(preferred, dtype=torch.long, device=device)
    r = torch.as_tensor(rejected, dtype=torch.long, device=device)

    class _SharedFutureScorer(torch.nn.Module):
        def __init__(self, shared_dim: int, action_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.context = torch.nn.Sequential(
                torch.nn.Linear(shared_dim, hidden_dim), torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.GELU(),
            )
            self.future_effect = torch.nn.Linear(hidden_dim, 4)
            self.score = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim + action_dim, hidden_dim), torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, history_goal: Any, candidate_action: Any) -> tuple[Any, Any]:
            context = self.context(history_goal)
            return self.score(torch.cat((context, candidate_action), dim=1)), self.future_effect(context)

    model = _SharedFutureScorer(x_shared.shape[1], x_action.shape[1], int(width)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1.0e-4)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_cost, predicted_effect = model(x_shared, x_action)
        cost_loss = torch.nn.functional.mse_loss(predicted_cost, y_cost)
        effect_loss = torch.nn.functional.mse_loss(predicted_effect, y_effect)
        rank_loss = torch.nn.functional.softplus(predicted_cost[p] - predicted_cost[r]).mean()
        (cost_loss + 0.25 * effect_loss + 0.25 * rank_loss).backward()
        optimizer.step()

    def predict(query_shared: np.ndarray, query_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query_shared = np.asarray(query_shared, dtype=np.float64)
        query_action = np.asarray(query_action, dtype=np.float64)
        if query_shared.ndim != 2 or query_shared.shape[1] != shared.shape[1]:
            raise ValueError("shared-future prediction has incompatible history/goal features")
        if query_action.shape != (len(query_shared), action.shape[1]):
            raise ValueError("shared-future prediction has incompatible action features")
        with torch.no_grad():
            score, predicted_effect = model(
                torch.as_tensor((query_shared - shared_mean) / shared_scale, dtype=torch.float32, device=device),
                torch.as_tensor((query_action - action_mean) / action_scale, dtype=torch.float32, device=device),
            )
        return (
            score.reshape(-1).detach().cpu().numpy().astype(np.float64) * cost_scale + cost_mean,
            predicted_effect.detach().cpu().numpy().astype(np.float64) * effect_scale + effect_mean,
        )

    contract = {
        "objective": "shared_future_planning",
        "shared_future_input_fields": ["source_history_model", "goal_raw"],
        "candidate_scorer_input_fields": ["shared_future_context", action_representation_input_fields(action_representation)[-1]],
        "action_representation": str(action_representation),
        "shared_future_action_independent": True,
        "shared_future_target": "within-root_mean_executed_terminal_effect",
        "effect_weight": 0.25, "rank_weight": 0.25,
        "width": int(width), "steps": int(steps), "initialization_seed": int(seed),
        "shared_feature_dimension": int(shared.shape[1]), "action_feature_dimension": int(action.shape[1]),
        "train_pair_count": int(len(preferred)),
        "target_fields": ["root_mean_executed_terminal_effect", "executed_task_cost"],
        "outcome_root_input_forbidden": True,
    }
    return contract, predict


def _mean_predictor(predictors: list[Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Average independently optimized copies of one fixed objective.

    The ensemble is a variance-reduction intervention, not another objective:
    every arm receives exactly the same initialization-seed set and the
    calibration selector still sees only one prediction per candidate.
    """

    if not predictors:
        raise ValueError("at least one predictor is required")

    def predict(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = [member(values) for member in predictors]
        return (
            np.mean(np.stack([item[0] for item in outputs]), axis=0),
            np.mean(np.stack([item[1] for item in outputs]), axis=0),
        )

    return predict


def endpoint_task_cost(endpoint: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Read a PointMaze task score directly from a predicted terminal state.

    This is intentionally distinct from the legacy auxiliary-effect arm.  That
    arm predicts a future effect, but a separate cost head still makes the
    deployed decision.  Here the candidate score is *defined* by the predicted
    endpoint and the declared goal, so endpoint supervision must influence the
    action ranking through the deployed readout.
    """

    value = np.asarray(endpoint, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all():
        raise ValueError("endpoint must be finite [rows,4]")
    if target.shape != (len(value), 2) or not np.isfinite(target).all():
        raise ValueError("goal must be finite [rows,2]")
    return np.linalg.norm(value[:, :2] - target, axis=1)


def _fit_counterfactual_endpoint(
    features: np.ndarray,
    terminal: np.ndarray,
    goal: np.ndarray,
    groups: dict[str, list[int]],
    *,
    width: int,
    steps: int,
    device: str,
    seed: int,
    action_representation: str = "summary_mean_std_first_last",
) -> tuple[dict[str, Any], Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    """Fit an action-conditioned terminal-state model with a coupled score.

    The supplied visual-history/action feature contract determines which
    action representation is used.  Unlike the planning-shaped auxiliary arm,
    neither raw source state nor a separately optimized value head is used
    when the returned predictor ranks candidates.  The training loss combines
    terminal prediction, the task score derived from that terminal prediction,
    and within-root contrast ranking.
    """

    import torch

    values = np.asarray(features, dtype=np.float64)
    target_terminal = np.asarray(terminal, dtype=np.float64)
    target_goal = np.asarray(goal, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("features must be non-empty finite [rows, dimensions]")
    if target_terminal.shape != (len(values), 4) or not np.isfinite(target_terminal).all():
        raise ValueError("terminal must be finite [rows,4]")
    if target_goal.shape != (len(values), 2) or not np.isfinite(target_goal).all():
        raise ValueError("goal must be finite [rows,2]")
    if width < 2 or steps < 1:
        raise ValueError("width >= 2 and steps >= 1 are required")
    torch.manual_seed(int(seed))
    x_mean, x_scale = values.mean(axis=0), values.std(axis=0)
    x_scale[x_scale < 1.0e-8] = 1.0
    terminal_mean, terminal_scale = target_terminal.mean(axis=0), target_terminal.std(axis=0)
    terminal_scale[terminal_scale < 1.0e-8] = 1.0
    x = torch.as_tensor((values - x_mean) / x_scale, dtype=torch.float32, device=device)
    y_terminal = torch.as_tensor((target_terminal - terminal_mean) / terminal_scale, dtype=torch.float32, device=device)
    physical_target_terminal = torch.as_tensor(target_terminal, dtype=torch.float32, device=device)
    y_goal = torch.as_tensor(target_goal, dtype=torch.float32, device=device)
    physical_scale = torch.as_tensor(terminal_scale, dtype=torch.float32, device=device)
    physical_mean = torch.as_tensor(terminal_mean, dtype=torch.float32, device=device)
    preferred, rejected = _pair_indices(groups, endpoint_task_cost(target_terminal, target_goal))
    p = torch.as_tensor(preferred, dtype=torch.long, device=device)
    r = torch.as_tensor(rejected, dtype=torch.long, device=device)

    class _EndpointPrior(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim), torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.GELU(),
            )
            self.endpoint = torch.nn.Linear(hidden_dim, 4)

        def forward(self, input_value: Any) -> Any:
            return self.endpoint(self.trunk(input_value))

    model = _EndpointPrior(x.shape[1], int(width)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1.0e-4)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_normalized = model(x)
        predicted_terminal = predicted_normalized * physical_scale + physical_mean
        predicted_cost = torch.linalg.vector_norm(predicted_terminal[:, :2] - y_goal, dim=1)
        true_cost = torch.linalg.vector_norm(physical_target_terminal[:, :2] - y_goal, dim=1)
        endpoint_loss = torch.nn.functional.mse_loss(predicted_normalized, y_terminal)
        task_loss = torch.nn.functional.mse_loss(predicted_cost, true_cost)
        rank_loss = torch.nn.functional.softplus(predicted_cost[p] - predicted_cost[r]).mean()
        (endpoint_loss + 0.25 * task_loss + 0.25 * rank_loss).backward()
        optimizer.step()

    def predict(input_values: np.ndarray, input_goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidate_features = np.asarray(input_values, dtype=np.float64)
        candidate_goal = np.asarray(input_goal, dtype=np.float64)
        if candidate_features.ndim != 2 or candidate_features.shape[1] != values.shape[1]:
            raise ValueError("prediction features have incompatible shape")
        if candidate_goal.shape != (len(candidate_features), 2) or not np.isfinite(candidate_goal).all():
            raise ValueError("prediction goals must be finite [rows,2]")
        with torch.no_grad():
            tensor = torch.as_tensor((candidate_features - x_mean) / x_scale, dtype=torch.float32, device=device)
            endpoint = model(tensor).detach().cpu().numpy().astype(np.float64) * terminal_scale + terminal_mean
        return endpoint_task_cost(endpoint, candidate_goal), endpoint

    contract = {
        "objective": "counterfactual_endpoint_readout",
        "endpoint_weight": 1.0,
        "task_readout_weight": 0.25,
        "rank_weight": 0.25,
        "score_readout": "norm(predicted_terminal_xy - declared_goal)",
        "width": int(width), "steps": int(steps), "initialization_seed": int(seed),
        "feature_dimension": int(values.shape[1]), "train_pair_count": int(len(preferred)),
        "input_fields": action_representation_input_fields(action_representation),
        "action_representation": str(action_representation),
        "target_fields": ["executed_terminal_state", "executed_task_cost"],
        "raw_source_state_live_input_forbidden": True,
    }
    return contract, predict


def _summary(rows: list[dict[str, Any]], cost: np.ndarray, predicted_cost: np.ndarray, predicted_effect: np.ndarray, terminal: np.ndarray) -> dict[str, Any]:
    groups = _group_rows(rows)
    pair_total = pair_correct = top1 = 0
    regrets: list[float] = []
    for indices in groups.values():
        oracle = min(indices, key=lambda index: (float(cost[index]), str(rows[index]["candidate_id"])))
        selected = min(indices, key=lambda index: (float(predicted_cost[index]), str(rows[index]["candidate_id"])))
        regrets.append(float(cost[selected] - cost[oracle]))
        top1 += int(selected == oracle)
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1:]:
                if np.isclose(cost[left], cost[right], rtol=0.0, atol=1.0e-10):
                    continue
                pair_total += 1
                pair_correct += int((predicted_cost[left] - predicted_cost[right]) * (cost[left] - cost[right]) > 0.0)
    effect = terminal - np.asarray([row["source_state_raw"] for row in rows], dtype=np.float64)
    return {
        "root_count": len(groups),
        "candidate_count": len(rows),
        "cost_rmse": float(np.sqrt(np.mean((predicted_cost - cost) ** 2))),
        "effect_rmse": float(np.sqrt(np.mean((predicted_effect - effect) ** 2))),
        "pairwise_ranking_accuracy": None if pair_total == 0 else float(pair_correct / pair_total),
        "comparable_pair_count": int(pair_total),
        "top1_oracle_rate": float(top1 / len(groups)),
        "mean_candidate_regret": float(np.mean(regrets)),
        "zero_regret_rate": float(np.mean(np.asarray(regrets) <= 1.0e-10)),
    }


def _per_root_decisions(rows: list[dict[str, Any]], cost: np.ndarray, predicted_cost: np.ndarray) -> list[dict[str, Any]]:
    """Emit paired, candidate-set decisions for post-selection diagnosis.

    These rows are observational reports of a frozen arm; they are deliberately
    not consumed by the calibration selector or any later optimization step.
    """

    grouped = _group_rows(rows)
    decisions: list[dict[str, Any]] = []
    for root in sorted(grouped):
        indices = grouped[root]
        oracle = min(indices, key=lambda index: (float(cost[index]), str(rows[index]["candidate_id"])))
        selected = min(indices, key=lambda index: (float(predicted_cost[index]), str(rows[index]["candidate_id"])))
        decisions.append({
            "instance_id": root,
            "candidate_count": int(len(indices)),
            "oracle_candidate_id": str(rows[oracle]["candidate_id"]),
            "selected_candidate_id": str(rows[selected]["candidate_id"]),
            "candidate_regret": float(cost[selected] - cost[oracle]),
            "zero_regret": bool(selected == oracle),
            "selected_predicted_cost": float(predicted_cost[selected]),
            "oracle_realized_cost": float(cost[oracle]),
        })
    return decisions


def _per_root_candidate_costs(
    rows: list[dict[str, Any]], cost: np.ndarray, predicted_cost: np.ndarray,
) -> list[dict[str, Any]]:
    """Export a fixed-bank candidate table for a post-fit safety audit.

    The regular per-root summary deliberately contains only the selected and
    oracle candidates.  A conformal fixed-bank certificate, however, needs
    every candidate score under a root.  This export is made only after an
    arm is frozen; consumers must never feed it back into objective selection.
    It preserves the action sequence so an independent evaluator can reject a
    report whose candidate identity changes across roots.
    """

    grouped = _group_rows(rows)
    exported: list[dict[str, Any]] = []
    for root in sorted(grouped):
        indices = sorted(grouped[root], key=lambda index: str(rows[index]["candidate_id"]))
        candidates: list[dict[str, Any]] = []
        for index in indices:
            action = np.asarray(rows[index]["action_sequence"], dtype=np.float64)
            if action.shape != (20, 2) or not np.isfinite(action).all():
                raise ValueError(f"{root}: candidate action must be finite [20,2]")
            candidates.append({
                "candidate_id": str(rows[index]["candidate_id"]),
                "candidate_action_hash": rows[index].get("candidate_action_hash"),
                "action_sequence": action.tolist(),
                "predicted_cost": float(predicted_cost[index]),
                "realized_cost": float(cost[index]),
            })
        exported.append({"instance_id": str(root), "candidates": candidates})
    return exported


def _calibration_terminal(rows: list[dict[str, Any]], labels: dict[str, Any]) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in rows:
        key = f"{row['instance_id']}/{row['candidate_id']}"
        label = labels.get(key)
        if not isinstance(label, dict):
            raise ValueError(f"calibration label missing {key}")
        terminal = np.asarray(label.get("realized_terminal_state"), dtype=np.float64).reshape(-1)
        if terminal.shape != (4,) or not np.isfinite(terminal).all():
            raise ValueError(f"invalid calibration terminal for {key}")
        values.append(terminal)
    return np.stack(values)


def run(
    *, source_dataset: Path, source_arrays: Path, context: Path, calibration_labels: Path,
    output: Path, device: str = "cuda", seed: int = 202610281, width: int = 32, steps: int = 400,
    ensemble_size: int = 1, include_shared_future_control: bool = False,
    action_representation: str = "summary_mean_std_first_last",
    require_v8_preflight: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source = _load(source_dataset)
    context_payload = _load(context)
    calibration = _load(calibration_labels)
    v8_contract = _validate_v8_shared_future_contract(source, context_payload) if require_v8_preflight else None
    if require_v8_preflight and (not include_shared_future_control or action_representation != "full_ordered_20x2"):
        raise ValueError("V8 preflight requires the shared-future arm and full_ordered_20x2 action representation")
    source_rows = source.get("records")
    context_rows = context_payload.get("records")
    if not isinstance(source_rows, list) or not isinstance(context_rows, list):
        raise ValueError("source/context require records lists")
    train_rows = [dict(row) for row in source_rows if str(row.get("split")) == "train"]
    calibration_rows = [dict(row) for row in context_rows if str(row.get("split")) == "calibration"]
    outcome_rows = [dict(row) for row in context_rows if str(row.get("split")) == "outcome"]
    if not train_rows or not calibration_rows or not outcome_rows:
        raise ValueError("source/context require non-empty train, calibration, and outcome roots")
    if {str(row["instance_id"]) for row in train_rows} & {str(row["instance_id"]) for row in calibration_rows + outcome_rows}:
        raise ValueError("train roots overlap calibration/outcome roots")
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")
    if action_representation in SUMMARY_ACTION_REPRESENTATIONS:
        feature_fn = _feature
    elif action_representation in ORDERED_ACTION_REPRESENTATIONS:
        feature_fn = _feature_ordered_action_sequence
    else:
        raise ValueError(f"unknown action representation: {action_representation}")
    cuda = require_cuda_allocation(device)
    with np.load(source_arrays, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    train_terminal = np.stack([_terminal_from_ref(row, arrays) for row in train_rows])
    train_effect = train_terminal - np.asarray([row["source_state_raw"] for row in train_rows], dtype=np.float64)
    train_cost = _cost(train_rows, train_terminal)
    train_features = np.stack([feature_fn(row) for row in train_rows])
    train_shared_features = np.stack([_shared_future_feature(row) for row in train_rows])
    train_action_features = np.stack([_shared_future_action_feature(row, action_representation) for row in train_rows])
    calibration_terminal = _calibration_terminal(calibration_rows, calibration)
    calibration_cost = _cost(calibration_rows, calibration_terminal)
    calibration_features = np.stack([feature_fn(row) for row in calibration_rows])
    calibration_shared_features = np.stack([_shared_future_feature(row) for row in calibration_rows])
    calibration_action_features = np.stack([_shared_future_action_feature(row, action_representation) for row in calibration_rows])
    train_groups = _group_rows(train_rows)
    candidates: list[dict[str, Any]] = []
    ensemble_seeds = tuple(int(seed) + offset for offset in range(int(ensemble_size)))
    for objective in OBJECTIVES:
        # The arms must begin from the identical parameter initialization.
        # Otherwise an arm-specific seed is a hidden intervention and a held-out
        # difference cannot be attributed to cost/ranking/effect supervision.
        members = [
            _fit(
                train_features, train_effect, train_cost, train_groups, objective,
                width=width, steps=steps, device=device, seed=member_seed,
                action_representation=action_representation,
            )
            for member_seed in ensemble_seeds
        ]
        contract, predict = members[0][0], _mean_predictor([member[1] for member in members])
        contract = {
            **contract,
            "initialization_seeds": list(ensemble_seeds),
            "ensemble_size": int(ensemble_size),
        }
        cal_cost, cal_effect = predict(calibration_features)
        candidates.append({
            "objective": objective, "contract": contract, "predict": predict,
            "calibration_predicted_cost": cal_cost,
            "calibration": _summary(calibration_rows, calibration_cost, cal_cost, cal_effect, calibration_terminal),
            "calibration_per_root": _per_root_decisions(calibration_rows, calibration_cost, cal_cost),
        })
    if include_shared_future_control:
        shared_contract, shared_predict = _fit_shared_future_scorer(
            train_shared_features, train_action_features, train_effect, train_cost,
            train_groups, width=width, steps=steps, device=device, seed=int(seed),
            action_representation=action_representation,
        )
        cal_cost, cal_effect = shared_predict(calibration_shared_features, calibration_action_features)
        candidates.append({
            "objective": Objective("shared_future_planning", effect_weight=0.25, rank_weight=0.25),
            "contract": shared_contract,
            "predict": lambda values: shared_predict(
                np.asarray(values, dtype=np.float64)[:, :train_shared_features.shape[1]],
                np.asarray(values, dtype=np.float64)[:, train_shared_features.shape[1]:],
            ),
            "calibration_predicted_cost": cal_cost,
            "calibration": _summary(calibration_rows, calibration_cost, cal_cost, cal_effect, calibration_terminal),
            "calibration_per_root": _per_root_decisions(calibration_rows, calibration_cost, cal_cost),
        })
    # This is the sole selector: calibration ranking regret, then simpler
    # objective order.  Outcome states are not yet materialized or read.
    chosen = min(candidates, key=lambda item: (item["calibration"]["mean_candidate_regret"], item["objective"].effect_weight, item["objective"].rank_weight))
    # All objectives are now frozen.  Materialize the outcome once, then score
    # every frozen arm on precisely the same candidates.  This makes the
    # PlanWAM-style arm a falsifiable baseline rather than merely the arm that
    # happened to win calibration.  These post-selection comparisons are never
    # fed back into objective selection.
    outcome_features = np.stack([feature_fn(row) for row in outcome_rows])
    outcome_terminal = _fresh_terminal(outcome_rows)
    outcome_cost = _cost(outcome_rows, outcome_terminal)
    for item in candidates:
        predicted_cost, predicted_effect = item["predict"](outcome_features)
        item["outcome"] = _summary(
            outcome_rows,
            outcome_cost,
            predicted_cost,
            predicted_effect,
            outcome_terminal,
        )
        item["outcome_per_root"] = _per_root_decisions(outcome_rows, outcome_cost, predicted_cost)
        item["outcome_predicted_cost"] = predicted_cost
    outcome = chosen["outcome"]
    report = {
        "schema": SCHEMA,
        "status": "completed_execution_grounded_control",
        "claim_boundary": (
            "A root-disjoint PointMaze control for planning-shaped visual history/action training. "
            "It is not a D-JEPA teacher, frozen-world-model modification, GDD distillation result, online deployment result, or universal claim."
        ),
        "control_contract": {
            "label": CONTROL_LABEL,
            "input_is_visual_history_goal_action_only": True,
            "action_representation": str(action_representation),
            "train_future_targets": ["executed_terminal_effect", "executed_task_cost"],
            "outcome_execution_deferred_until_after_calibration_selection": True,
            "outcome_cost_used_for_model_or_objective_selection": False,
            "all_frozen_objectives_scored_on_outcome_after_selection": True,
            "objective_initialization": "shared_seed_per_run; only supervision weights differ across arms",
            "objective_ensemble": "shared initialization-seed ensemble per arm; only supervision weights differ across arms",
            "calibration_selects": "minimum mean candidate regret; objective-complexity tie break",
            "shared_future_control_included": bool(include_shared_future_control),
            "shared_future_control_note": (
                "when enabled, shared future context receives history/goal only and candidate action enters only the scorer"
            ),
            "v8_static_preflight": v8_contract,
        },
        "split_isolation": {
            "train_root_count": len(_group_rows(train_rows)), "calibration_root_count": len(_group_rows(calibration_rows)),
            "outcome_root_count": len(_group_rows(outcome_rows)), "root_disjoint": True,
        },
        "candidate_objectives": [
            {
                "name": item["objective"].name,
                "contract": item["contract"],
                "calibration": item["calibration"],
                "calibration_per_root": item["calibration_per_root"],
                "calibration_candidate_costs": _per_root_candidate_costs(
                    calibration_rows, calibration_cost, item["calibration_predicted_cost"],
                ),
                "outcome": item["outcome"],
                "outcome_per_root": item["outcome_per_root"],
                "outcome_candidate_costs": _per_root_candidate_costs(
                    outcome_rows, outcome_cost, item["outcome_predicted_cost"],
                ),
                "outcome_role": "post-selection diagnostic only; never used to select an objective",
            }
            for item in candidates
        ],
        "selected_objective": chosen["objective"].name,
        "calibration": chosen["calibration"],
        "outcome": outcome,
        "cuda_preflight": cuda,
        "provenance": {
            "source_dataset": str(source_dataset.resolve()), "source_dataset_sha256": _sha256(source_dataset),
            "source_arrays": str(source_arrays.resolve()), "source_arrays_sha256": _sha256(source_arrays),
            "context": str(context.resolve()), "context_sha256": _sha256(context),
            "calibration_labels": str(calibration_labels.resolve()), "calibration_labels_sha256": _sha256(calibration_labels),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--calibration-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=202610281)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--include-shared-future-control", action="store_true")
    parser.add_argument("--require-v8-preflight", action="store_true")
    parser.add_argument("--action-representation", choices=tuple(sorted(SUMMARY_ACTION_REPRESENTATIONS | ORDERED_ACTION_REPRESENTATIONS)), default="summary_mean_std_first_last")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"status": result["status"], "selected_objective": result["selected_objective"], "outcome": result["outcome"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
