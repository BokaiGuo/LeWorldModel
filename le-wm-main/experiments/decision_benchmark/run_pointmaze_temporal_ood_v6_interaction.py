"""Run V6 GRU plus pulse-FiLM on matched PointMaze builder artifacts.

Calibrate trains only source-train rows, deterministically partitions the
builder calibration roots into selection and authorization halves, then writes
a hash-bound frozen selection.  Score requires that artifact and only then
generates held PointMaze labels with _simulate_sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _simulate_sequence
from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.gpu_preflight import require_cuda_recurrent_cell

SOURCE_SCHEMA = "pointmaze_temporal_ood_v6_matched_source.v1"
CONTEXT_SCHEMA = "pointmaze_temporal_ood_v6_matched_panel.v3"
FROZEN_SCHEMA = "pointmaze_temporal_ood_v6_interaction_frozen_selection.v7"
LEGACY_FROZEN_SCHEMA = "pointmaze_temporal_ood_v6_interaction_frozen_selection.v2"
OBSERVABILITY_FROZEN_SCHEMA = "pointmaze_temporal_ood_v6_observability_frozen_selection.v5"
FROZEN_MODEL_SCHEMA = "pointmaze_temporal_ood_v6_interaction_frozen_model.v6"
LEGACY_FROZEN_MODEL_SCHEMA = "pointmaze_temporal_ood_v6_interaction_frozen_model.v1"
SCORE_SCHEMA = "pointmaze_temporal_ood_v6_interaction_score.v6"
OUTCOME_SCHEMA = "pointmaze_temporal_ood_v6_generated_outcomes.v2"
MANIFEST_SCHEMA = "pointmaze_temporal_ood_v6_matched_manifest.v1"
HISTORY_FRAMES, FRAME_STRIDE, SOURCE_FRAME = 8, 5, 40
HIDDEN_WIDTHS, TEMPERATURES, CONTRAST_WEIGHTS = (16, 32), (0.75, 1.0), (0.0, 0.25, 1.0)
HISTORY_SOURCES = (
    "source_history_model",
    "source_history_visual_state",
    "hybrid_latent_visual_state",
)
TRAIN_SEED = 2026102601
CALIBRATION_SPLIT_SALT = "pointmaze-v6-selection-auth-v1"
SPLIT_COUNTS = {"train": 64, "calibration": 48, "outcome": 32}
TRAIN_CANDIDATE_COUNT, HELD_CANDIDATE_COUNT = 64, 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_panel(path: Path, schema: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata, records = payload.get("metadata"), payload.get("records")
    if not isinstance(metadata, dict) or metadata.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema!r}")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected non-empty records")
    return metadata, [dict(row) for row in records]


def _key(row: dict[str, Any]) -> str:
    return f"{row['instance_id']}/{int(row['candidate_id'])}"


def _array_key(reference: str) -> str:
    items = str(reference).split("::", 1)
    if len(items) != 2 or not items[1]:
        raise ValueError("invalid realized_terminal_state_ref")
    return items[1]


def _group(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["instance_id"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["candidate_id"]))
    return dict(sorted(groups.items()))


def _validated_root_counts(value: Any) -> dict[str, int]:
    """Validate a formal panel's declared root budget.

    V7 fixed the original 64/48/32 budget.  That made the calibration model
    selection grid much larger than its independent authorization sample.  A
    later panel may deliberately use a larger, immutable root budget, but it
    must be bound by the manifest instead of silently disabling strict checks.
    """
    if not isinstance(value, dict) or set(value) != set(SPLIT_COUNTS):
        raise ValueError("V6 root counts must have exactly train, calibration, and outcome")
    counts = {name: int(value[name]) for name in SPLIT_COUNTS}
    if any(not isinstance(value[name], int) or isinstance(value[name], bool) or counts[name] <= 0 for name in SPLIT_COUNTS):
        raise ValueError("V6 root counts must be positive integers")
    if counts["calibration"] < 4 or counts["calibration"] % 2:
        raise ValueError("V6 calibration root count must be even and at least four")
    return counts


def _validate_builder_contract(
    source: list[dict[str, Any]], context: list[dict[str, Any]], context_metadata: dict[str, Any] | None = None,
    *, strict_counts: bool = True, expected_root_counts: dict[str, int] | None = None,
) -> None:
    expected_counts = SPLIT_COUNTS if expected_root_counts is None else _validated_root_counts(expected_root_counts)
    source_groups, context_groups = _group(source), _group(context)
    if any({str(row.get("split")) for row in rows} != {"train"} for rows in source_groups.values()):
        raise ValueError("V6 source artifact must contain train roots only")
    if strict_counts and len(source_groups) != expected_counts["train"]:
        raise ValueError(f"V6 source artifact requires {expected_counts['train']} train roots")
    scorer_protocol = "" if context_metadata is None else str(context_metadata.get("scorer_candidate_bank_protocol", ""))
    exact_shared_source = scorer_protocol == "held_exact_shared_bank"
    source_generator = (
        "world_fixed_random_five_pulse_matched_held_v6"
        if exact_shared_source else "world_fixed_random_five_pulse_matched_train_v6"
    )
    source_candidate_count = HELD_CANDIDATE_COUNT if exact_shared_source else TRAIN_CANDIDATE_COUNT
    if strict_counts and scorer_protocol not in {"train_bank", "held_exact_shared_bank"}:
        raise ValueError("V6 context has an unsupported scorer candidate-bank protocol")
    if strict_counts and any(str(row.get("candidate_generator")) != source_generator for row in source):
        raise ValueError("V6 source artifact has an unexpected candidate generator")
    context_train = {root for root, rows in context_groups.items() if {str(row.get("split")) for row in rows} == {"train"}}
    context_eval = set(context_groups) - context_train
    if set(source_groups) != context_train:
        raise ValueError("builder source train roots must exactly bind context train roots")
    if set(source_groups) & context_eval:
        raise ValueError("builder train roots overlap calibration/outcome roots")
    observed: set[str] = set()
    for root, rows in context_groups.items():
        splits = {str(row.get("split")) for row in rows}
        if len(splits) != 1 or not splits <= {"train", "calibration", "outcome"}:
            raise ValueError(f"invalid context split for root {root!r}")
        observed |= splits
        if any("real_task_cost" in row or "realized_terminal_state" in row for row in rows):
            raise ValueError("builder context must not embed realised outcomes")
    if observed != {"train", "calibration", "outcome"}:
        raise ValueError("V6 context requires train, calibration, and outcome roots")
    context_counts = {split: len({str(row["instance_id"]) for row in context if str(row["split"]) == split}) for split in expected_counts}
    if strict_counts and context_counts != expected_counts:
        raise ValueError(f"V6 context root counts do not match protocol: {context_counts}")
    if strict_counts and any(str(row.get("candidate_generator")) != "world_fixed_random_five_pulse_matched_held_v6" for row in context):
        raise ValueError("V6 context artifact has an unexpected candidate generator")
    instance_to_root: dict[str, str] = {}
    root_to_instance: dict[str, str] = {}
    for row in [*source, *context]:
        instance, root = str(row["instance_id"]), str(row["start_episode_id"])
        if instance in instance_to_root and instance_to_root[instance] != root:
            raise ValueError("V6 instance_id maps to multiple start_episode_id values")
        if root in root_to_instance and root_to_instance[root] != instance:
            raise ValueError("V6 start_episode_id is duplicated under multiple instance_id values")
        instance_to_root[instance], root_to_instance[root] = root, instance
    _validate_candidate_identity(source, label="source", expected_count=source_candidate_count if strict_counts else None)
    for split in ("train", "calibration", "outcome"):
        _validate_candidate_identity(
            [row for row in context if str(row["split"]) == split], label=f"context/{split}",
            expected_count=HELD_CANDIDATE_COUNT if strict_counts else None,
        )
    # V5 and the initial V6 panels confused history value with candidate-side
    # extrapolation. New panels must bind the declared support regime to the
    # executable sequence identities; this is not an optional metadata hint.
    if not context_metadata or context_metadata.get("calibration_label_identity_bound") is not True:
        raise ValueError("V6 v2 context requires identity-bound calibration labels")
    if strict_counts:
        required_metadata = {
            "candidate_family": "world_fixed_random_five_pulse_matched_held_v6",
            "history_frames": HISTORY_FRAMES,
            "history_frame_stride": FRAME_STRIDE,
            "history_is_causal": True,
            "history_actions_are_strictly_pre_candidate": True,
            "outcome_labels_materialized_during_build": False,
        }
        for key, expected in required_metadata.items():
            if context_metadata.get(key) != expected:
                raise ValueError(f"V6 context metadata {key!r} does not match the current protocol")
    regime = str(context_metadata.get("candidate_generalization_regime", ""))
    if regime not in {"numeric_ood", "shared_candidate_in_support"}:
        raise ValueError("V6 v2 context requires a supported candidate generalization regime")
    source_hashes = {str(row.get("candidate_action_hash", "")) for row in source}
    held_hashes = {str(row.get("candidate_action_hash", "")) for row in context}
    if not source_hashes or not held_hashes or "" in source_hashes or "" in held_hashes:
        raise ValueError("support-regime panel lacks candidate action hashes")
    overlap = source_hashes & held_hashes
    if regime == "numeric_ood" and overlap:
        raise ValueError("numeric-OOD panel shares source and held candidate actions")
    if regime == "shared_candidate_in_support" and not held_hashes <= source_hashes:
        raise ValueError("shared-candidate panel has unsupported held actions")
    if strict_counts and exact_shared_source:
        declared_scorer_hashes = context_metadata.get("scorer_training_action_hashes", [])
        if not isinstance(declared_scorer_hashes, list) or set(map(str, declared_scorer_hashes)) != source_hashes:
            raise ValueError("held-exact scorer source hashes do not match context metadata")
        if source_hashes != held_hashes:
            raise ValueError("held-exact scorer source must exactly equal the held action bank")
    goal_qpos = np.asarray(context_metadata.get("goal_qpos", []), dtype=np.float64).reshape(-1)
    if goal_qpos.shape != (2,) or not np.isfinite(goal_qpos).all():
        raise ValueError("V6 v3 context requires an explicit finite goal_qpos")
    if not isinstance(context_metadata.get("legacy_goal_reproduction"), bool):
        raise ValueError("V6 v3 context requires legacy_goal_reproduction provenance")
    actual_legacy = bool(np.allclose(goal_qpos, np.asarray([1.2, 1.2]), rtol=0.0, atol=1.0e-12))
    if bool(context_metadata["legacy_goal_reproduction"]) != actual_legacy:
        raise ValueError("V6 legacy_goal_reproduction flag does not match goal_qpos")
    for row in [*source, *context]:
        row_goal = np.asarray(row.get("goal_raw", []), dtype=np.float64).reshape(-1)
        if row_goal.shape < (2,) or not np.allclose(row_goal[:2], goal_qpos, rtol=0.0, atol=1.0e-12):
            raise ValueError("V6 goal_qpos does not bind every source/context row")
    _validate_temporal_history_contract(source, label="source")
    _validate_temporal_history_contract(context, label="context")


def _validate_temporal_history_contract(records: list[dict[str, Any]], *, label: str) -> None:
    """Bind history tensors to their declared pre-candidate action clock.

    A `[35, 2]` action tensor alone cannot rule out a one-block temporal
    shift.  V6 fixes its source at frame 40, so both sampled RGB frames and
    the half-open raw-action interval are protocol fields.
    """

    expected_frames = list(range(
        SOURCE_FRAME - (HISTORY_FRAMES - 1) * FRAME_STRIDE,
        SOURCE_FRAME + 1,
        FRAME_STRIDE,
    ))
    expected_interval = [expected_frames[0], SOURCE_FRAME]
    expected_actions = ((HISTORY_FRAMES - 1) * FRAME_STRIDE, 2)
    for row in records:
        root = str(row.get("instance_id", "<unknown>"))
        if int(row.get("source_frame", -1)) != SOURCE_FRAME:
            raise ValueError(f"{label}: root {root!r} has an unexpected source frame")
        if list(row.get("history_frame_indices", [])) != expected_frames:
            raise ValueError(f"{label}: root {root!r} has a non-causal history-frame ledger")
        if list(row.get("history_action_interval", [])) != expected_interval:
            raise ValueError(f"{label}: root {root!r} has a non-causal history-action interval")
        actions = np.asarray(row.get("history_actions_raw"), dtype=np.float64)
        if actions.shape != expected_actions or not np.isfinite(actions).all():
            raise ValueError(f"{label}: root {root!r} has invalid pre-candidate actions")


def _validate_candidate_identity(records: list[dict[str, Any]], *, label: str, expected_count: int | None = None) -> None:
    """Bind declared action identities to executable candidate tensors.

    A selection report is meaningless if a row may claim a hash from one bank
    while executing another action tensor.  Validate the bank independently
    within every source/context split before either calibration labels or held
    outcomes are consumed.
    """
    groups = _group(records)
    if not groups:
        raise ValueError(f"{label}: empty candidate groups")
    expected_ids: list[int] | None = None
    reference: dict[int, tuple[str, np.ndarray]] = {}
    for root, rows in groups.items():
        ids = [int(row["candidate_id"]) for row in rows]
        if ids != list(range(len(ids))):
            raise ValueError(f"{label}: root {root!r} has non-contiguous candidate ids")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"{label}: candidate-id support differs across roots")
        hashes: list[str] = []
        for row in rows:
            candidate = np.asarray(row["action_sequence"], dtype=np.float64)
            if candidate.shape != (20, 2) or not np.isfinite(candidate).all():
                raise ValueError(f"{label}: candidate action must be finite [20, 2]")
            action_hash = sequence_hash(candidate)
            if str(row.get("candidate_action_hash", "")) != action_hash:
                raise ValueError(f"{label}: declared candidate action hash does not match sequence")
            candidate_id = int(row["candidate_id"])
            prior = reference.get(candidate_id)
            if prior is None:
                reference[candidate_id] = (action_hash, candidate.copy())
            elif prior[0] != action_hash or not np.array_equal(prior[1], candidate):
                raise ValueError(f"{label}: candidate id {candidate_id} differs across roots")
            hashes.append(action_hash)
        if expected_count is not None and len(rows) != expected_count:
            raise ValueError(f"{label}: root {root!r} has {len(rows)} candidates, expected {expected_count}")
        expected_bank = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        # Check every row, not merely the first one. Otherwise a malformed
        # context can bind the executed candidate sequence and its per-row
        # hash while silently changing the declared bank on a later row.
        # That would make root-level candidate support non-auditable.
        if any(str(row.get("candidate_bank_hash", "")) != expected_bank for row in rows):
            raise ValueError(f"{label}: candidate bank hash does not match candidate rows")


def deterministic_calibration_split(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    roots = sorted(_group([row for row in records if str(row["split"]) == "calibration"]))
    if len(roots) < 4 or len(roots) % 2:
        raise ValueError("V6 needs an even number of at least four calibration roots")
    ordered = sorted(roots, key=lambda root: hashlib.sha256(f"{CALIBRATION_SPLIT_SALT}|{root}".encode()).hexdigest())
    middle = len(ordered) // 2
    return {"selection": sorted(ordered[:middle]), "authorization": sorted(ordered[middle:])}


def _history_features(row: dict[str, Any], history_source: str) -> np.ndarray:
    """Return one declared causal history family without reading raw state.

    ``source_history_visual_state`` is a renderer-derived RGB geometry ledger
    captured by the panel builder, not ``source_state_raw``.  Keeping the
    families explicit lets a future panel test whether a negative latent-only
    result is an observability failure or an action-effect failure, while the
    selected family remains hash-bound before outcome labels are generated.
    """
    if history_source not in HISTORY_SOURCES:
        raise ValueError(f"unsupported history source: {history_source!r}")
    latent = np.asarray(row["source_history_model"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)
    if history_source == "source_history_model":
        result = latent
    else:
        if "source_history_visual_state" not in row:
            raise ValueError("selected visual-geometry history is absent from this panel row")
        visual = np.asarray(row["source_history_visual_state"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)
        if visual.shape[1] < 2:
            raise ValueError("visual-geometry history must contain at least XY coordinates per frame")
        result = visual if history_source == "source_history_visual_state" else np.concatenate((latent, visual), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("selected history source must be finite")
    return result


def _validate_history_sources(value: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    sources = ("source_history_model",) if value is None else tuple(str(item) for item in value)
    if not sources or any(item not in HISTORY_SOURCES for item in sources) or len(set(sources)) != len(sources):
        raise ValueError(f"history sources must be a non-empty unique subset of {HISTORY_SOURCES}")
    return sources


def _tensors(
    records: list[dict[str, Any]], *, history_source: str = "source_history_model",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histories, candidates, states, goals = [], [], [], []
    for row in records:
        history = _history_features(row, history_source)
        candidate = np.asarray(row["action_sequence"], dtype=np.float32)
        if candidate.ndim != 2 or candidate.shape[0] < 2 or candidate.shape[1] != 2:
            raise ValueError("candidate action must be finite [steps, 2]")
        prior = np.asarray(row["history_actions_raw"], dtype=np.float32)
        if prior.shape != ((HISTORY_FRAMES - 1) * FRAME_STRIDE, 2):
            raise ValueError("V6 expects 35 strictly pre-candidate action steps")
        if not np.isfinite(candidate).all():
            raise ValueError("history and candidate tensors must be finite")
        blocks = prior.reshape(HISTORY_FRAMES - 1, FRAME_STRIDE, 2).mean(axis=1)
        histories.append(np.c_[history, np.vstack([np.zeros((1, 2), dtype=np.float32), blocks])])
        candidates.append(candidate)
        state, goal = np.asarray(row["source_state_raw"], dtype=np.float32).reshape(-1), np.asarray(row["goal_raw"], dtype=np.float32).reshape(-1)
        if state.size < 4 or goal.size < 2:
            raise ValueError("source_state_raw needs [x, y, vx, vy] and goal_raw needs two coordinates")
        states.append(state[:4]); goals.append(goal[:2])
    return np.asarray(histories), np.asarray(candidates), np.asarray(states), np.asarray(goals)


def _pulse_features(candidate: torch.Tensor) -> torch.Tensor:
    delta = torch.cat([torch.zeros_like(candidate[:, :1]), candidate[:, 1:] - candidate[:, :-1]], dim=1)
    phase = torch.linspace(-1.0, 1.0, candidate.shape[1], dtype=candidate.dtype, device=candidate.device)
    return torch.cat([candidate, delta, phase.view(1, -1, 1).expand(candidate.shape[0], -1, -1)], dim=-1)


class PulseAwareCandidateEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(action_dim * 2 + 1, hidden)

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        features = _pulse_features(candidate)
        state = torch.zeros((candidate.shape[0], self.cell.hidden_size), dtype=candidate.dtype, device=candidate.device)
        for index in range(features.shape[1]):
            state = self.cell(features[:, index], state)
        return state


class RecurrentPulseFiLMEffect(nn.Module):
    """Causal history GRU whose state FiLM-modulates pulse-aware actions."""

    def __init__(self, history_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.history_encoder = nn.Sequential(nn.Linear(history_dim, hidden), nn.SiLU())
        self.history_cell = nn.GRUCell(hidden, hidden)
        self.candidate_encoder = PulseAwareCandidateEncoder(action_dim, hidden)
        self.memory_effect = nn.Linear(hidden, 2)
        self.film, self.effect_head = nn.Linear(hidden, hidden * 2), nn.Linear(hidden, 2)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, history: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        encoded_history = self.history_encoder(history)
        memory = torch.zeros((history.shape[0], self.history_cell.hidden_size), dtype=history.dtype, device=history.device)
        for index in range(encoded_history.shape[1]):
            memory = self.history_cell(encoded_history[:, index], memory)
        scale, shift = self.film(memory).chunk(2, dim=-1)
        action = self.candidate_encoder(candidate)
        return self.memory_effect(memory) + self.effect_head((1.0 + scale) * action + shift)


class ActionOnlyPulseEffect(nn.Module):
    def __init__(self, action_dim: int, hidden: int) -> None:
        super().__init__(); self.encoder = PulseAwareCandidateEncoder(action_dim, hidden); self.head = nn.Linear(hidden, 2)

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(candidate))


class PrivilegedStatePulseEffect(nn.Module):
    """Diagnostic-only state/action effect upper bound.

    It is deliberately never used by the active policy.  If it cannot beat
    action-only on a fresh fixed bank, the bank/horizon is not informative
    enough for a visual-history method to rescue; if it can, the gap localizes
    to the visual interface rather than candidate support.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU())
        self.candidate_encoder = PulseAwareCandidateEncoder(action_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, state: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.state_encoder(state), self.candidate_encoder(candidate)), dim=-1))


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _normalizer(values: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=axes), np.maximum(values.std(axis=axes), 1e-3)


def _root_candidate_index_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    """Return a root-by-candidate index matrix for a within-root contrast loss.

    Candidate-ranking supervision is meaningful only when every row compared
    shares its source state and goal.  This checks that invariant before
    building any pairwise losses, rather than treating arbitrary minibatch
    pairs as counterfactual alternatives.
    """

    grouped = _group(records)
    positions: dict[str, dict[int, int]] = defaultdict(dict)
    for index, row in enumerate(records):
        root, candidate_id = str(row["instance_id"]), int(row["candidate_id"])
        if candidate_id in positions[root]:
            raise ValueError(f"duplicate candidate id {candidate_id} for source root {root!r}")
        positions[root][candidate_id] = index
    candidate_ids: list[int] | None = None
    rows_by_root: list[list[int]] = []
    for root, rows in grouped.items():
        ids = [int(row["candidate_id"]) for row in rows]
        if candidate_ids is None:
            candidate_ids = ids
        elif ids != candidate_ids:
            raise ValueError("counterfactual contrast requires one shared candidate-id bank across source roots")
        states = np.asarray([row["source_state_raw"][:2] for row in rows], dtype=np.float64)
        goals = np.asarray([row["goal_raw"][:2] for row in rows], dtype=np.float64)
        if not np.allclose(states, states[:1], rtol=0.0, atol=0.0):
            raise ValueError(f"counterfactual contrast source root {root!r} has candidate-varying state")
        if not np.allclose(goals, goals[:1], rtol=0.0, atol=0.0):
            raise ValueError(f"counterfactual contrast source root {root!r} has candidate-varying goal")
        rows_by_root.append([positions[root][candidate_id] for candidate_id in ids])
    if candidate_ids is None or len(candidate_ids) < 2:
        raise ValueError("counterfactual contrast requires at least two candidates per source root")
    return np.asarray(rows_by_root, dtype=np.int64)


def rootwise_task_contrast_loss(
    predicted_effect: torch.Tensor,
    state: torch.Tensor,
    goal: torch.Tensor,
    observed_cost: torch.Tensor,
    root_candidate_indices: torch.Tensor,
    contrast_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Huber loss on counterfactual task-cost contrasts inside each root.

    It is intentionally a complement to effect supervision: predicted effects
    still have to match real successors, while this term prevents their
    candidate ordering from being an accidental by-product of average MSE.
    """

    if predicted_effect.ndim != 2 or predicted_effect.shape[-1] != 2:
        raise ValueError("predicted effect must have shape [rows, 2]")
    if state.shape != predicted_effect.shape or goal.shape != predicted_effect.shape:
        raise ValueError("counterfactual contrast state/goal must align with predicted effects")
    if observed_cost.shape != (len(predicted_effect),):
        raise ValueError("counterfactual contrast costs must align with predicted effects")
    if root_candidate_indices.ndim != 2 or root_candidate_indices.shape[1] < 2:
        raise ValueError("counterfactual contrast needs a root-by-candidate index matrix")
    predicted_cost = torch.linalg.vector_norm(state + predicted_effect - goal, dim=-1)
    predicted = predicted_cost[root_candidate_indices]
    observed = observed_cost[root_candidate_indices]
    mask = torch.triu(
        torch.ones((root_candidate_indices.shape[1], root_candidate_indices.shape[1]), dtype=torch.bool, device=predicted.device),
        diagonal=1,
    )
    scale = torch.as_tensor(contrast_scale, dtype=predicted.dtype, device=predicted.device).reshape(())
    if not bool(torch.isfinite(scale)) or float(scale.item()) <= 0.0:
        raise ValueError("counterfactual contrast scale must be finite and positive")
    return torch.nn.functional.smooth_l1_loss(
        (predicted[:, :, None] - predicted[:, None, :])[:, mask] / scale,
        (observed[:, :, None] - observed[:, None, :])[:, mask] / scale,
    )


def rootwise_task_contrast_scale(
    observed_cost: torch.Tensor, root_candidate_indices: torch.Tensor,
) -> torch.Tensor:
    """Return a train-source-only scale for dimensionless contrast training."""

    if observed_cost.ndim != 1 or root_candidate_indices.ndim != 2:
        raise ValueError("counterfactual contrast scale requires costs and a root-by-candidate index matrix")
    if root_candidate_indices.shape[1] < 2:
        raise ValueError("counterfactual contrast scale needs at least two candidates per root")
    observed = observed_cost[root_candidate_indices]
    mask = torch.triu(
        torch.ones((root_candidate_indices.shape[1], root_candidate_indices.shape[1]), dtype=torch.bool, device=observed.device),
        diagonal=1,
    )
    deltas = (observed[:, :, None] - observed[:, None, :])[:, mask]
    return torch.sqrt(torch.mean(deltas.square())).clamp_min(1.0e-3)


def _cyclic_root_history(records: list[dict[str, Any]], history: np.ndarray) -> np.ndarray:
    """Mismatches each root with another root's full causal history.

    The permutation is root-level, deterministic, and preserves the candidate
    rows and history marginal.  It is a negative control for whether correct
    history-to-current-state alignment, rather than extra input capacity,
    changes action ranking.
    """

    if len(records) != len(history):
        raise ValueError("records/history length mismatch")
    groups = _group(records)
    roots = sorted(groups)
    if len(roots) < 2:
        raise ValueError("cyclic history control needs at least two roots")
    positions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        positions[str(row["instance_id"])].append(index)
    result = np.empty_like(history)
    for index, root in enumerate(roots):
        donor = roots[(index + 1) % len(roots)]
        recipient_indices, donor_indices = positions[root], positions[donor]
        donor_history = history[donor_indices[0]]
        if not all(np.allclose(history[value], donor_history, rtol=0.0, atol=0.0) for value in donor_indices):
            raise ValueError("candidate rows of a root must share one source history")
        result[recipient_indices] = donor_history
    return result


def _frozen_model_path(selection_path: Path) -> Path:
    return selection_path.with_name(f"{selection_path.stem}_model.pt")


def _resolve_frozen_checkpoint(
    frozen: dict[str, Any],
    selection_path: Path,
    *,
    expected_schema: str,
) -> tuple[Path, str]:
    """Resolve a hash-bound checkpoint without silently retraining a legacy fit."""

    schema = frozen.get("schema")
    if schema != expected_schema:
        raise ValueError("frozen selection schema is not current and cannot be replayed")
    recorded_path, expected_sha256 = frozen.get("model_checkpoint_path"), frozen.get("model_checkpoint_sha256")
    if not isinstance(recorded_path, str) or not recorded_path or not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError("frozen selection lacks a hash-bound model checkpoint")

    candidates = [Path(recorded_path), _frozen_model_path(selection_path)]
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == expected_sha256:
            return candidate, expected_sha256
    raise FileNotFoundError(
        "frozen selection model checkpoint is missing or does not match its recorded hash"
    )


def _write_frozen_model(
    path: Path,
    recurrent: RecurrentPulseFiLMEffect,
    action: ActionOnlyPulseEffect,
    fit: dict[str, np.ndarray],
    *,
    hidden: int,
    contrast_weight: float,
    history_source: str = "source_history_model",
    privileged_state: PrivilegedStatePulseEffect | None = None,
    shuffled_history: RecurrentPulseFiLMEffect | None = None,
    state_mean: np.ndarray | None = None,
    state_scale: np.ndarray | None = None,
) -> str:
    """Persist the exact selected fit so score cannot silently retrain it."""
    _validate_history_sources((history_source,))
    payload = {
        "schema": FROZEN_MODEL_SCHEMA,
        "hidden": int(hidden),
        "contrast_weight": float(contrast_weight),
        "history_source": history_source,
        "history_dim": int(fit["history_mean"].shape[-1]),
        "action_dim": int(fit["candidate_mean"].shape[-1]),
        "recurrent_state_dict": {key: value.detach().cpu() for key, value in recurrent.state_dict().items()},
        "action_state_dict": {key: value.detach().cpu() for key, value in action.state_dict().items()},
        "normalizer": {key: torch.as_tensor(value, dtype=torch.float32) for key, value in fit.items()},
        "runner_source_sha256": _sha256(Path(__file__)),
    }
    observability_values = (privileged_state, shuffled_history, state_mean, state_scale)
    if any(value is not None for value in observability_values):
        if any(value is None for value in observability_values):
            raise ValueError("observability checkpoint requires all diagnostic arms and state normalizer")
        payload["observability"] = {
            "state_state_dict": {key: value.detach().cpu() for key, value in privileged_state.state_dict().items()},
            "shuffled_history_state_dict": {key: value.detach().cpu() for key, value in shuffled_history.state_dict().items()},
            "state_mean": torch.as_tensor(state_mean, dtype=torch.float32),
            "state_scale": torch.as_tensor(state_scale, dtype=torch.float32),
        }
    torch.save(payload, path)
    return _sha256(path)


def _load_frozen_model(
    path: Path, expected_sha256: str, *, device: str, expected_contrast_weight: float,
    expected_history_source: str = "source_history_model",
) -> tuple[RecurrentPulseFiLMEffect, ActionOnlyPulseEffect, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError("frozen selection model checkpoint is missing")
    if _sha256(path) != expected_sha256:
        raise ValueError("frozen selection model checkpoint hash mismatch")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != FROZEN_MODEL_SCHEMA:
        raise ValueError("frozen model schema is not current and cannot be replayed")
    if float(payload.get("contrast_weight", float("nan"))) != float(expected_contrast_weight):
        raise ValueError("frozen model contrast weight does not match frozen selection")
    _validate_history_sources((expected_history_source,))
    if str(payload.get("history_source", "source_history_model")) != expected_history_source:
        raise ValueError("frozen model history source does not match frozen selection")
    if payload.get("runner_source_sha256") != _sha256(Path(__file__)):
        raise ValueError("frozen model was produced by a different runner source")
    recurrent = RecurrentPulseFiLMEffect(int(payload["history_dim"]), int(payload["action_dim"]), int(payload["hidden"])).to(device)
    action = ActionOnlyPulseEffect(int(payload["action_dim"]), int(payload["hidden"])).to(device)
    recurrent.load_state_dict(payload["recurrent_state_dict"], strict=True)
    action.load_state_dict(payload["action_state_dict"], strict=True)
    normalizer = payload.get("normalizer")
    required = {
        "history_mean", "history_scale", "candidate_mean", "candidate_scale", "target_mean", "target_scale",
        "contrast_scale",
    }
    if not isinstance(normalizer, dict) or set(normalizer) != required:
        raise ValueError("frozen selection model has an invalid normalizer")
    fit = {key: value.detach().cpu().numpy().astype(np.float32, copy=False) for key, value in normalizer.items()}
    if fit["contrast_scale"].shape != (1,) or not np.isfinite(fit["contrast_scale"]).all() or float(fit["contrast_scale"][0]) <= 0.0:
        raise ValueError("frozen model has an invalid source-root contrast scale")
    return recurrent.eval(), action.eval(), fit


def _load_observability_models(
    path: Path, expected_sha256: str, *, device: str, expected_contrast_weight: float,
    expected_history_source: str = "source_history_model",
) -> tuple[RecurrentPulseFiLMEffect, ActionOnlyPulseEffect, PrivilegedStatePulseEffect, RecurrentPulseFiLMEffect, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load diagnostic-only arms while retaining ordinary V6 compatibility."""
    recurrent, action, fit = _load_frozen_model(
        path, expected_sha256, device=device, expected_contrast_weight=expected_contrast_weight,
        expected_history_source=expected_history_source,
    )
    payload = torch.load(path, map_location=device, weights_only=True)
    bundle = payload.get("observability")
    required = {"state_state_dict", "shuffled_history_state_dict", "state_mean", "state_scale"}
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise ValueError("frozen selection lacks required observability models")
    hidden, action_dim = int(payload["hidden"]), int(payload["action_dim"])
    state = PrivilegedStatePulseEffect(4, action_dim, hidden).to(device)
    shuffled = RecurrentPulseFiLMEffect(int(payload["history_dim"]), action_dim, hidden).to(device)
    state.load_state_dict(bundle["state_state_dict"], strict=True)
    shuffled.load_state_dict(bundle["shuffled_history_state_dict"], strict=True)
    state_fit = {
        "state_mean": bundle["state_mean"].detach().cpu().numpy().astype(np.float32, copy=False),
        "state_scale": bundle["state_scale"].detach().cpu().numpy().astype(np.float32, copy=False),
    }
    if state_fit["state_mean"].shape != (4,) or state_fit["state_scale"].shape != (4,):
        raise ValueError("invalid privileged state normalizer")
    return recurrent, action, state.eval(), shuffled.eval(), fit, state_fit


def _fit(
    source: list[dict[str, Any]], arrays: Any, *, hidden: int, epochs: int, device: str,
    contrast_weight: float = 0.0, history_source: str = "source_history_model",
) -> tuple[RecurrentPulseFiLMEffect, ActionOnlyPulseEffect, dict[str, np.ndarray]]:
    if float(contrast_weight) not in CONTRAST_WEIGHTS:
        raise ValueError(f"contrast weight must be one of {CONTRAST_WEIGHTS}")
    _validate_history_sources((history_source,))
    history, candidate, _state, _goal = _tensors(source, history_source=history_source)
    terminal = np.asarray([arrays[_array_key(row["realized_terminal_state_ref"])] for row in source], dtype=np.float32)
    state = np.asarray([row["source_state_raw"][:2] for row in source], dtype=np.float32)
    goal = np.asarray([row["goal_raw"][:2] for row in source], dtype=np.float32)
    target = terminal[:, :2] - state
    observed_cost = np.linalg.norm(terminal[:, :2] - goal, axis=1).astype(np.float32)
    root_indices = _root_candidate_index_matrix(source)
    _seed(TRAIN_SEED)
    hm, hs = _normalizer(history, (0, 1)); cm, cs = _normalizer(candidate, (0, 1)); ym, ys = _normalizer(target, (0,))
    recurrent, action = RecurrentPulseFiLMEffect(history.shape[-1], candidate.shape[-1], hidden).to(device), ActionOnlyPulseEffect(candidate.shape[-1], hidden).to(device)
    ropt, aopt = torch.optim.AdamW(recurrent.parameters(), lr=2e-3, weight_decay=1e-4), torch.optim.AdamW(action.parameters(), lr=2e-3, weight_decay=1e-4)
    th, tc, ty = torch.tensor((history - hm) / hs, device=device), torch.tensor((candidate - cm) / cs, device=device), torch.tensor((target - ym) / ys, device=device)
    ts = torch.tensor(state, device=device)
    tg = torch.tensor(goal, device=device)
    to = torch.tensor(observed_cost, device=device)
    ti = torch.tensor(root_indices, dtype=torch.long, device=device)
    contrast_scale = rootwise_task_contrast_scale(to, ti)
    y_mean, y_scale = torch.tensor(ym, device=device), torch.tensor(ys, device=device)
    for _ in range(int(epochs)):
        ropt.zero_grad(set_to_none=True)
        recurrent_effect = recurrent(th, tc)
        loss = (recurrent_effect - ty).square().mean()
        if contrast_weight:
            loss = loss + float(contrast_weight) * rootwise_task_contrast_loss(
                recurrent_effect * y_scale + y_mean, ts, tg, to, ti,
                contrast_scale,
            )
        loss.backward(); ropt.step()
        aopt.zero_grad(set_to_none=True)
        action_effect = action(tc)
        loss = (action_effect - ty).square().mean()
        if contrast_weight:
            loss = loss + float(contrast_weight) * rootwise_task_contrast_loss(
                action_effect * y_scale + y_mean, ts, tg, to, ti,
                contrast_scale,
            )
        loss.backward(); aopt.step()
    return recurrent, action, {
        "history_mean": hm, "history_scale": hs, "candidate_mean": cm, "candidate_scale": cs,
        "target_mean": ym, "target_scale": ys,
        "contrast_scale": np.asarray([float(contrast_scale.item())], dtype=np.float32),
    }


def _fit_observability(
    source: list[dict[str, Any]], arrays: Any, *, hidden: int, epochs: int, device: str,
    contrast_weight: float = 0.0, history_source: str = "source_history_model",
) -> tuple[RecurrentPulseFiLMEffect, ActionOnlyPulseEffect, PrivilegedStatePulseEffect, RecurrentPulseFiLMEffect, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit same-bank diagnostic arms without accessing held outcomes.

    The state arm is an upper-bound diagnostic only.  The shuffled-history arm
    has the identical visual architecture and parameter budget as recurrent,
    but is trained and evaluated with a deterministic root-level mismatch.
    """
    recurrent, action, fit = _fit(
        source, arrays, hidden=hidden, epochs=epochs, device=device, contrast_weight=contrast_weight,
        history_source=history_source,
    )
    history, candidate, state, _goal = _tensors(source, history_source=history_source)
    terminal = np.asarray([arrays[_array_key(row["realized_terminal_state_ref"])] for row in source], dtype=np.float32)
    target = terminal[:, :2] - state[:, :2]
    observed_cost = np.linalg.norm(terminal[:, :2] - _goal[:, :2], axis=1).astype(np.float32)
    root_indices = _root_candidate_index_matrix(source)
    sm, ss = _normalizer(state, (0,))
    th = torch.tensor((history - fit["history_mean"]) / fit["history_scale"], device=device)
    tc = torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device)
    ts = torch.tensor((state - sm) / ss, device=device)
    ty = torch.tensor((target - fit["target_mean"]) / fit["target_scale"], device=device)
    tstate = torch.tensor(state[:, :2], device=device)
    tgoal = torch.tensor(_goal[:, :2], device=device)
    tobserved = torch.tensor(observed_cost, device=device)
    tindex = torch.tensor(root_indices, dtype=torch.long, device=device)
    contrast_scale = torch.tensor(fit["contrast_scale"], device=device).reshape(())
    y_mean = torch.tensor(fit["target_mean"], device=device)
    y_scale = torch.tensor(fit["target_scale"], device=device)
    _seed(TRAIN_SEED + 1)
    privileged = PrivilegedStatePulseEffect(state.shape[-1], candidate.shape[-1], hidden).to(device)
    shuffled = RecurrentPulseFiLMEffect(history.shape[-1], candidate.shape[-1], hidden).to(device)
    state_opt = torch.optim.AdamW(privileged.parameters(), lr=2e-3, weight_decay=1e-4)
    shuffle_opt = torch.optim.AdamW(shuffled.parameters(), lr=2e-3, weight_decay=1e-4)
    shuffled_history = torch.tensor(
        (_cyclic_root_history(source, history) - fit["history_mean"]) / fit["history_scale"], device=device,
    )
    for _ in range(int(epochs)):
        state_opt.zero_grad(set_to_none=True)
        state_effect = privileged(ts, tc)
        state_loss = (state_effect - ty).square().mean()
        if contrast_weight:
            state_loss = state_loss + float(contrast_weight) * rootwise_task_contrast_loss(
                state_effect * y_scale + y_mean, tstate, tgoal, tobserved, tindex,
                contrast_scale,
            )
        state_loss.backward(); state_opt.step()
        shuffle_opt.zero_grad(set_to_none=True)
        shuffle_effect = shuffled(shuffled_history, tc)
        shuffle_loss = (shuffle_effect - ty).square().mean()
        if contrast_weight:
            shuffle_loss = shuffle_loss + float(contrast_weight) * rootwise_task_contrast_loss(
                shuffle_effect * y_scale + y_mean, tstate, tgoal, tobserved, tindex,
                contrast_scale,
            )
        shuffle_loss.backward(); shuffle_opt.step()
    return recurrent, action, privileged, shuffled, fit, {"state_mean": sm, "state_scale": ss}


def _predict(
    recurrent: RecurrentPulseFiLMEffect, action: ActionOnlyPulseEffect, fit: dict[str, np.ndarray],
    records: list[dict[str, Any]], *, history_source: str = "source_history_model",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history, candidate, state, goal = _tensors(records, history_source=history_source); device = next(recurrent.parameters()).device
    with torch.inference_mode():
        th = torch.tensor((history - fit["history_mean"]) / fit["history_scale"], device=device)
        tc = torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device)
        rec = recurrent(th, tc).cpu().numpy() * fit["target_scale"] + fit["target_mean"]
        act = action(tc).cpu().numpy() * fit["target_scale"] + fit["target_mean"]
    return rec, act, state, goal


def _predict_observability(
    recurrent: RecurrentPulseFiLMEffect,
    action: ActionOnlyPulseEffect,
    privileged: PrivilegedStatePulseEffect,
    shuffled: RecurrentPulseFiLMEffect,
    fit: dict[str, np.ndarray],
    state_fit: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    *,
    history_source: str = "source_history_model",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history, candidate, state, goal = _tensors(records, history_source=history_source)
    device = next(recurrent.parameters()).device
    with torch.inference_mode():
        th = torch.tensor((history - fit["history_mean"]) / fit["history_scale"], device=device)
        tc = torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device)
        ts = torch.tensor((state - state_fit["state_mean"]) / state_fit["state_scale"], device=device)
        shuffled_history = torch.tensor(
            (_cyclic_root_history(records, history) - fit["history_mean"]) / fit["history_scale"], device=device,
        )
        scale, offset = fit["target_scale"], fit["target_mean"]
        rec = recurrent(th, tc).cpu().numpy() * scale + offset
        act = action(tc).cpu().numpy() * scale + offset
        state_effect = privileged(ts, tc).cpu().numpy() * scale + offset
        shuffled_effect = shuffled(shuffled_history, tc).cpu().numpy() * scale + offset
    return rec, act, state_effect, shuffled_effect, state, goal


def interaction_diagnostics(
    recurrent: RecurrentPulseFiLMEffect,
    fit: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    *,
    history_source: str = "source_history_model",
) -> dict[str, float | bool]:
    """Measure whether a trained FiLM branch changes relative action effects.

    The V5 failure was partly architectural: an additive history branch cannot
    change the ordering of candidate effects.  A nonzero FiLM parameter alone
    is insufficient evidence that this repaired architecture used the
    interaction.  This diagnostic evaluates a mixed finite difference on two
    train-root histories and two candidate sequences without opening any
    calibration or outcome labels.
    """
    grouped = _group(records)
    if len(grouped) < 2:
        raise ValueError("interaction diagnostic needs at least two source roots")
    first_root, second_root = list(grouped)[:2]
    left, right = grouped[first_root], grouped[second_root]
    if len(left) < 2 or len(right) < 2:
        raise ValueError("interaction diagnostic needs at least two candidates per source root")
    history, candidate, _state, _goal = _tensors(
        [left[0], left[1], right[0], right[1]], history_source=history_source,
    )
    device = next(recurrent.parameters()).device
    with torch.inference_mode():
        history_tensor = torch.tensor((history - fit["history_mean"]) / fit["history_scale"], device=device)
        candidate_tensor = torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device)
        # f(h0, a0) - f(h0, a1) - f(h1, a0) + f(h1, a1)
        values = torch.stack((
            recurrent(history_tensor[0:1], candidate_tensor[0:1])[0],
            recurrent(history_tensor[0:1], candidate_tensor[1:2])[0],
            recurrent(history_tensor[2:3], candidate_tensor[0:1])[0],
            recurrent(history_tensor[2:3], candidate_tensor[1:2])[0],
        ))
        mixed = values[0] - values[1] - values[2] + values[3]
    # A two-history/two-action mixed difference is necessary but weak: it can
    # be non-zero while the deployment-relevant full-bank argmin is invariant.
    # Evaluate every common candidate under every train-root history against a
    # label-free canonical effect target. Current state/goal are removed from
    # the target, so changed ranks cannot be attributed to root geometry.
    roots = sorted(grouped)
    reference_rows = sorted(grouped[roots[0]], key=lambda row: int(row["candidate_id"]))
    candidate_ids = [int(row["candidate_id"]) for row in reference_rows]
    if candidate_ids != list(range(len(candidate_ids))):
        raise ValueError("interaction diagnostic requires contiguous full-bank candidate ids")
    reference_candidates = np.stack([
        np.asarray(row["action_sequence"], dtype=np.float32) for row in reference_rows
    ])
    root_rows = [sorted(grouped[root], key=lambda row: int(row["candidate_id"])) for root in roots]
    if any([int(row["candidate_id"]) for row in rows] != candidate_ids for rows in root_rows):
        raise ValueError("interaction diagnostic requires one shared candidate bank")
    # Reuse `_tensors` so this full-bank diagnostic has the identical
    # visual-plus-pre-candidate-action representation as the fitted model.
    root_history = np.stack([
        _tensors([rows[0]], history_source=history_source)[0][0] for rows in root_rows
    ])
    root_state = np.stack([
        np.asarray(rows[0]["source_state_raw"], dtype=np.float32).reshape(-1)[:2]
        for rows in root_rows
    ])
    root_goal = np.stack([
        np.asarray(rows[0]["goal_raw"], dtype=np.float32).reshape(-1)[:2]
        for rows in root_rows
    ])
    canonical_effect_target = np.median(root_goal - root_state, axis=0)
    full_history = np.repeat(root_history, len(reference_candidates), axis=0)
    full_candidates = np.tile(reference_candidates, (len(root_history), 1, 1))
    with torch.inference_mode():
        full_effects = recurrent(
            torch.tensor((full_history - fit["history_mean"]) / fit["history_scale"], device=device),
            torch.tensor((full_candidates - fit["candidate_mean"]) / fit["candidate_scale"], device=device),
        ).cpu().numpy() * fit["target_scale"] + fit["target_mean"]
    full_costs = np.linalg.norm(
        full_effects.reshape(len(roots), len(reference_candidates), -1)
        - canonical_effect_target.reshape(1, 1, -1),
        axis=-1,
    )
    full_argmin = np.argmin(full_costs, axis=1)
    unique_top1 = int(len(set(full_argmin.tolist())))
    return {
        "film_weight_frobenius": float(torch.linalg.vector_norm(recurrent.film.weight).item()),
        "film_bias_l2": float(torch.linalg.vector_norm(recurrent.film.bias).item()),
        "history_candidate_mixed_difference_l2": float(torch.linalg.vector_norm(mixed).item()),
        "interaction_active_at_1e-6": bool(torch.linalg.vector_norm(mixed).item() > 1.0e-6),
        "full_bank_candidate_count": int(len(reference_candidates)),
        "full_bank_history_count": int(len(roots)),
        "canonical_effect_target": canonical_effect_target.astype(float).tolist(),
        "full_bank_unique_top1_count": unique_top1,
        "history_changes_full_bank_canonical_ranking": bool(unique_top1 > 1),
    }


def _cost(state: np.ndarray, goal: np.ndarray, effect: np.ndarray, temperature: float) -> np.ndarray:
    return np.linalg.norm(state[:, :2] + float(temperature) * effect - goal[:, :2], axis=1)


def _selection(records: list[dict[str, Any]], predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if len(predicted) != len(records) or len(actual) != len(records):
        raise ValueError("predicted and actual costs must align one-to-one with records")
    groups, positions = _group(records), defaultdict(dict)
    for index, row in enumerate(records):
        root, candidate_id = str(row["instance_id"]), int(row["candidate_id"])
        if candidate_id in positions[root]:
            raise ValueError(f"duplicate candidate_id {candidate_id} for root {root!r}")
        positions[root][candidate_id] = index
    result: dict[str, Any] = {}
    for root, rows in groups.items():
        candidate_ids = [int(row["candidate_id"]) for row in rows]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"duplicate candidate ids after grouping root {root!r}")
        indices = [positions[root][candidate_id] for candidate_id in candidate_ids]
        winner, truth = int(np.argmin(predicted[indices])), actual[indices]
        result[root] = {"winner_candidate_id": candidate_ids[winner], "regret": float(truth[winner] - np.min(truth)), "ranking_correct": bool(winner == int(np.argmin(truth)))}
    regrets = np.asarray([row["regret"] for row in result.values()])
    return {"root_count": len(result), "mean_regret": float(regrets.mean()), "roots": result}


def _paired(method: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    roots = sorted(method["roots"])
    if roots != sorted(baseline["roots"]): raise ValueError("paired roots mismatch")
    values = np.asarray([baseline["roots"][root]["regret"] - method["roots"][root]["regret"] for root in roots])
    rng = np.random.default_rng(2026102602)
    draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)])
    return {"mean_gain": float(values.mean()), "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975))}


def pairwise_counterfactual_contrast_metrics(
    records: list[dict[str, Any]], predicted_cost: np.ndarray, actual_cost: np.ndarray, *, margin: float = 0.003,
) -> dict[str, Any]:
    """Score adjacent same-root action pairs without a full-bank argmin.

    This is a post-outcome C1 diagnostic, not a planner or safety certificate.
    Realised differences inside the predeclared margin are excluded from sign
    accuracy rather than misreported as identifiable preferences.
    """

    predicted, actual = np.asarray(predicted_cost, dtype=np.float64), np.asarray(actual_cost, dtype=np.float64)
    if predicted.shape != (len(records),) or actual.shape != (len(records),):
        raise ValueError("pairwise contrast costs must align one-to-one with records")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all() or float(margin) <= 0.0:
        raise ValueError("pairwise contrast costs and margin must be finite")
    positions: dict[str, dict[int, int]] = defaultdict(dict)
    for index, row in enumerate(records):
        root, candidate_id = str(row["instance_id"]), int(row["candidate_id"])
        if candidate_id in positions[root]:
            raise ValueError("pairwise contrast encountered duplicate candidate ID")
        positions[root][candidate_id] = index
    rows: list[dict[str, Any]] = []
    for root, by_id in sorted(positions.items()):
        candidate_ids = sorted(by_id)
        if candidate_ids != list(range(len(candidate_ids))) or len(candidate_ids) < 2 or len(candidate_ids) % 2:
            # This score is attached to the general V6 outcome report as an
            # optional diagnostic.  Older (and deliberately non-pairwise)
            # panels may have an odd candidate bank, so they must remain
            # scoreable.  Formal V11 admission is fail-closed earlier, in the
            # frozen-preflight/builder contract, where its exact pair bank is
            # checked before any labels are opened.
            return {
                "status": "not_applicable",
                "margin": float(margin),
                "pair_count": 0,
                "valid_pair_count": 0,
                "valid_pair_coverage": None,
                "sign_accuracy": None,
                "contrast_mae": None,
                "reason": "candidate_ids_are_not_contiguous_even_pairs",
                "incompatible_root": root,
                "candidate_ids": candidate_ids,
                "claim_boundary": "Pairwise contrast was not declared by this candidate bank; no pairwise inference is reported.",
            }
        for left in range(0, len(candidate_ids), 2):
            right = left + 1
            actual_delta = float(actual[by_id[left]] - actual[by_id[right]])
            predicted_delta = float(predicted[by_id[left]] - predicted[by_id[right]])
            valid = bool(abs(actual_delta) >= float(margin))
            rows.append({"root": root, "pair_index": left // 2, "actual_delta": actual_delta,
                         "predicted_delta": predicted_delta, "valid_margin": valid,
                         "sign_correct": None if not valid else bool(np.sign(actual_delta) == np.sign(predicted_delta))})
    valid_rows = [row for row in rows if row["valid_margin"]]
    if not valid_rows:
        return {"status": "evaluated", "margin": float(margin), "pair_count": len(rows), "valid_pair_count": 0,
                "valid_pair_coverage": 0.0, "sign_accuracy": None, "contrast_mae": None,
                "claim_boundary": "No realised pair clears the frozen margin; sign accuracy is undefined."}
    error = np.asarray([row["predicted_delta"] - row["actual_delta"] for row in valid_rows], dtype=np.float64)
    return {"status": "evaluated", "margin": float(margin), "pair_count": len(rows), "valid_pair_count": len(valid_rows),
            "valid_pair_coverage": float(len(valid_rows) / len(rows)),
            "sign_accuracy": float(np.mean([bool(row["sign_correct"]) for row in valid_rows])),
            "contrast_mae": float(np.mean(np.abs(error))),
            "claim_boundary": "Matched-pair post-outcome diagnostic only; not a full-bank planner certificate."}


def _pairwise_root_sign_correctness(
    records: list[dict[str, Any]], predicted_cost: np.ndarray, actual_cost: np.ndarray, *, margin: float = 0.003,
) -> dict[str, float]:
    """Return one fixed-margin pairwise-sign score per root for C1 calibration.

    Root-level aggregation is deliberate: candidate pairs belonging to the
    same history are dependent, so authorization must not treat 8 pairs from
    one root as eight independent deployment episodes.
    """
    predicted, actual = np.asarray(predicted_cost, dtype=np.float64), np.asarray(actual_cost, dtype=np.float64)
    if predicted.shape != (len(records),) or actual.shape != (len(records),):
        raise ValueError("pairwise root scores must align one-to-one with records")
    positions: dict[str, dict[int, int]] = defaultdict(dict)
    for index, row in enumerate(records):
        root, candidate_id = str(row["instance_id"]), int(row["candidate_id"])
        if candidate_id in positions[root]:
            raise ValueError("pairwise root scores encountered duplicate candidate ID")
        positions[root][candidate_id] = index
    result: dict[str, float] = {}
    for root, by_id in sorted(positions.items()):
        candidate_ids = sorted(by_id)
        if candidate_ids != list(range(len(candidate_ids))) or len(candidate_ids) < 2 or len(candidate_ids) % 2:
            raise ValueError("C1 pairwise calibration requires contiguous, even candidate IDs per root")
        correct: list[float] = []
        for left in range(0, len(candidate_ids), 2):
            right = left + 1
            actual_delta = float(actual[by_id[left]] - actual[by_id[right]])
            if abs(actual_delta) < float(margin):
                continue
            predicted_delta = float(predicted[by_id[left]] - predicted[by_id[right]])
            correct.append(float(np.sign(actual_delta) == np.sign(predicted_delta)))
        if not correct:
            raise ValueError(f"C1 pairwise calibration root {root!r} has no realised contrast above the frozen margin")
        result[root] = float(np.mean(correct))
    return result


def _paired_pairwise_sign_gain(
    records: list[dict[str, Any]], method_cost: np.ndarray, baseline_cost: np.ndarray, actual_cost: np.ndarray,
    *, margin: float = 0.003,
) -> dict[str, float | int]:
    """Root-cluster bootstrap gain of pairwise sign accuracy over a baseline."""
    method = _pairwise_root_sign_correctness(records, method_cost, actual_cost, margin=margin)
    baseline = _pairwise_root_sign_correctness(records, baseline_cost, actual_cost, margin=margin)
    if sorted(method) != sorted(baseline):
        raise ValueError("C1 pairwise authorization roots mismatch")
    values = np.asarray([method[root] - baseline[root] for root in sorted(method)], dtype=np.float64)
    rng = np.random.default_rng(2026102611)
    draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)])
    return {
        "root_count": int(len(values)), "mean_gain": float(values.mean()),
        "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975)),
    }


def _visual_history_motion_energy(
    records: list[dict[str, Any]], *, motion_source: str = "source_history_model",
) -> np.ndarray:
    """Return one causal visual-motion statistic per candidate row.

    This is intentionally computed from the frozen visual history only: it
    excludes the appended action blocks and source simulator state.  Values
    must be candidate-invariant within a root, otherwise a root gate could
    become a hidden candidate-specific selector.
    """
    if motion_source not in {"source_history_model", "source_history_visual_state"}:
        raise ValueError("motion source must be source_history_model or source_history_visual_state")
    values = []
    grouped = _group(records)
    for root, rows in grouped.items():
        root_values = []
        for row in rows:
            if motion_source == "source_history_visual_state":
                if "source_history_visual_state" not in row:
                    raise ValueError("visual-state motion gate requires renderer-derived visual history")
                visual = np.asarray(row["source_history_visual_state"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)[:, :2]
            else:
                visual = np.asarray(row["source_history_model"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)
            if not np.isfinite(visual).all():
                raise ValueError("visual history energy requires finite source_history_model")
            root_values.append(float(np.linalg.vector_norm(visual[-1] - visual[0])))
        if not np.allclose(root_values, root_values[0], rtol=0.0, atol=0.0):
            raise ValueError(f"visual history energy varies across candidates for root {root!r}")
        values.extend(root_values)
    # _group sorts roots, whereas the caller retains original row order.
    def energy(row: dict[str, Any]) -> float:
        if motion_source == "source_history_visual_state":
            value = np.asarray(row["source_history_visual_state"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)[:, :2]
        else:
            value = np.asarray(row["source_history_model"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)
        return float(np.linalg.vector_norm(value[-1] - value[0]))
    by_root = {str(rows[0]["instance_id"]): energy(rows[0]) for rows in grouped.values()}
    return np.asarray([by_root[str(row["instance_id"])] for row in records], dtype=np.float64)


def _history_gate_specs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predeclare a small input-only root-routing family for calibration.

    Selection can choose among the two global arms and six quantile gates, but
    authorization happens on disjoint roots.  No realised successor/cost is
    needed to compute a route at score time.
    """
    energy = _visual_history_motion_energy(records)
    root_energy = np.asarray([energy[index] for index, row in enumerate(records) if int(row["candidate_id"]) == 0])
    if root_energy.size < 4:
        raise ValueError("history gate selection needs at least four roots")
    specs = [
        {"kind": "action_only"},
        {"kind": "recurrent"},
    ]
    for quantile in (0.25, 0.5, 0.75):
        threshold = float(np.quantile(root_energy, quantile))
        specs.extend((
            {"kind": "history_energy_gate", "direction": "recurrent_when_low", "quantile": quantile, "threshold": threshold},
            {"kind": "history_energy_gate", "direction": "recurrent_when_high", "quantile": quantile, "threshold": threshold},
        ))
    return specs


def _history_gate_mask(records: list[dict[str, Any]], spec: dict[str, Any]) -> np.ndarray:
    kind = str(spec.get("kind"))
    if kind == "action_only":
        return np.zeros(len(records), dtype=bool)
    if kind == "recurrent":
        return np.ones(len(records), dtype=bool)
    if kind != "history_energy_gate":
        raise ValueError("unknown selected history policy")
    threshold, direction = float(spec["threshold"]), str(spec["direction"])
    motion_source = str(spec.get("motion_source", "source_history_model"))
    energy = _visual_history_motion_energy(records, motion_source=motion_source)
    if direction == "recurrent_when_low":
        return energy <= threshold
    if direction == "recurrent_when_high":
        return energy >= threshold
    raise ValueError("unknown history gate direction")


def _policy_cost(
    records: list[dict[str, Any]], recurrent_cost: np.ndarray, action_cost: np.ndarray, policy: dict[str, Any],
) -> np.ndarray:
    if len(recurrent_cost) != len(records) or len(action_cost) != len(records):
        raise ValueError("policy costs must align with records")
    return np.where(_history_gate_mask(records, policy), recurrent_cost, action_cost)


def _calibration_costs(path: Path, records: list[dict[str, Any]], *, require_identity: bool = False) -> np.ndarray:
    labels = json.loads(path.read_text(encoding="utf-8"))
    if set(labels) != {_key(row) for row in records}:
        raise ValueError("builder calibration labels must exactly match all builder calibration records")
    identity_keys = {"instance_id", "candidate_id", "candidate_action_hash", "candidate_bank_hash"}
    for row in records:
        label = labels[_key(row)]
        if not isinstance(label, dict):
            raise ValueError("calibration label must be an object")
        present = identity_keys & set(label)
        if require_identity and present != identity_keys:
            raise ValueError("identity-bound calibration labels are missing candidate provenance")
        if present and present != identity_keys:
            raise ValueError("calibration label has incomplete candidate provenance")
        if present and (
            str(label["instance_id"]) != str(row["instance_id"])
            or int(label["candidate_id"]) != int(row["candidate_id"])
            or str(label["candidate_action_hash"]) != str(row["candidate_action_hash"])
            or str(label["candidate_bank_hash"]) != str(row["candidate_bank_hash"])
        ):
            raise ValueError("calibration label candidate provenance does not match its record")
    return np.asarray([float(labels[_key(row)]["real_task_cost"]) for row in records], dtype=np.float64)


def _validate_manifest(
    manifest: Path, source: Path, source_arrays: Path, context: Path, calibration_labels: Path,
) -> dict[str, Any]:
    """Verify builder-level provenance before any fitting or outcome access."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("V6 manifest schema mismatch")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("V6 manifest lacks artifact bindings")
    expected = {"source_dataset": source, "source_arrays": source_arrays, "context": context, "calibration_outcomes": calibration_labels}
    for role, path in expected.items():
        entry = artifacts.get(role)
        if not isinstance(entry, dict) or entry.get("path") != str(path.resolve()) or entry.get("sha256") != _sha256(path):
            raise ValueError(f"V6 manifest {role} binding mismatch")
    outcome = artifacts.get("outcome_labels")
    if not isinstance(outcome, dict) or outcome.get("materialized") is not False or outcome.get("sha256") is not None:
        raise ValueError("V6 manifest does not keep outcome labels sealed")
    root_counts = payload.get("root_counts")
    formal_protocol = bool(payload.get("formal_protocol", True))
    if formal_protocol:
        try:
            _validated_root_counts(root_counts)
        except ValueError as error:
            raise ValueError(f"V6 manifest has invalid root counts: {error}") from error
    return payload


def _validate_frozen_selection_contract(
    frozen: dict[str, Any], context_rows: list[dict[str, Any]], *, expected_schema: str, history_gate: bool,
    expected_root_counts: dict[str, int],
) -> dict[str, Any]:
    if frozen.get("schema") != expected_schema:
        raise ValueError("frozen selection schema is not current and cannot be replayed")
    if frozen.get("history_gate_enabled") is not history_gate:
        raise ValueError("frozen selection history-gate mode mismatch")
    if frozen.get("root_counts") != _validated_root_counts(expected_root_counts):
        raise ValueError("frozen selection root budget does not match the bound panel")
    if int(frozen.get("selected_hidden", -1)) not in HIDDEN_WIDTHS:
        raise ValueError("frozen selection has an invalid hidden width")
    if float(frozen.get("selected_contrast_weight", float("nan"))) not in CONTRAST_WEIGHTS:
        raise ValueError("frozen selection has an invalid contrast weight")
    if float(frozen.get("selected_temperature", float("nan"))) not in TEMPERATURES:
        raise ValueError("frozen selection has an invalid temperature")
    policy = frozen.get("selected_policy")
    if not isinstance(policy, dict) or policy.get("kind") not in {
        "action_only", "recurrent", "history_energy_gate",
    }:
        raise ValueError("frozen selection lacks an explicit selected policy")
    if policy["kind"] == "history_energy_gate":
        if str(policy.get("direction")) not in {"recurrent_when_low", "recurrent_when_high"}:
            raise ValueError("frozen history-energy policy has an invalid direction")
        threshold = float(policy.get("threshold", float("nan")))
        if not np.isfinite(threshold):
            raise ValueError("frozen history-energy policy has an invalid threshold")
        if policy.get("fixed_threshold") is True:
            motion_source = str(policy.get("motion_source", "source_history_model"))
            if motion_source not in {"source_history_model", "source_history_visual_state"}:
                raise ValueError("fixed history-energy policy has an invalid motion source")
        else:
            quantile = float(policy.get("quantile", float("nan")))
            if quantile not in {0.25, 0.5, 0.75}:
                raise ValueError("frozen history-energy policy has an invalid quantile")
    authorization = frozen.get("authorization")
    if not isinstance(authorization, dict) or not isinstance(authorization.get("permitted"), bool):
        raise ValueError("frozen selection lacks an explicit authorization contract")
    partition = frozen.get("calibration_partition")
    calibration_roots = sorted(_group([row for row in context_rows if str(row["split"]) == "calibration"]))
    expected_partition = (
        {"selection": [], "authorization": calibration_roots}
        if frozen.get("fixed_policy_authorization_all_calibration") is True
        else deterministic_calibration_split(context_rows)
    )
    if not isinstance(partition, dict) or partition.get("selection") != expected_partition["selection"] or partition.get("authorization") != expected_partition["authorization"]:
        raise ValueError("frozen selection calibration partition does not bind current context")
    if set(expected_partition["selection"]) & set(expected_partition["authorization"]):
        raise ValueError("calibration selection and authorization roots overlap")
    candidates = frozen.get("history_source_candidates", ["source_history_model"])
    declared = _validate_history_sources(candidates if isinstance(candidates, list) else None)
    selected_history = str(frozen.get("selected_history_source", "source_history_model"))
    if selected_history not in declared:
        raise ValueError("frozen selection history source is not in its predeclared candidate family")
    return policy


def calibrate(
    source: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path, *,
    epochs: int = 100, device: str = "cuda", observability_suite: bool = False, history_gate: bool = False,
    manifest: Path | None = None, history_sources: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    manifest_payload = _validate_manifest(manifest, source, source_arrays, context, calibration_labels)
    expected_counts = _validated_root_counts(manifest_payload["root_counts"])
    cuda_runtime = require_cuda_recurrent_cell(device) if str(device) == "cuda" else None
    source_meta, source_rows = _read_panel(source, SOURCE_SCHEMA); context_meta, context_rows = _read_panel(context, CONTEXT_SCHEMA)
    if source_meta.get("history_frames") != HISTORY_FRAMES:
        raise ValueError("V6 source metadata history_frames mismatch")
    _validate_builder_contract(
        source_rows, context_rows, context_meta,
        strict_counts=bool(manifest_payload.get("formal_protocol", True)), expected_root_counts=expected_counts,
    ); arrays = np.load(source_arrays, allow_pickle=False)
    calibration_rows = [row for row in context_rows if str(row["split"]) == "calibration"]
    costs = {
        _key(row): value for row, value in zip(
            calibration_rows,
            _calibration_costs(
                calibration_labels, calibration_rows,
                require_identity=True,
            ),
        )
    }
    partition = deterministic_calibration_split(context_rows); selection_roots, auth_roots = set(partition["selection"]), set(partition["authorization"])
    select = [row for row in calibration_rows if str(row["instance_id"]) in selection_roots]
    auth = [row for row in calibration_rows if str(row["instance_id"]) in auth_roots]
    declared_history_sources = _validate_history_sources(history_sources)
    choices = []
    for history_source in declared_history_sources:
        for hidden in HIDDEN_WIDTHS:
            for contrast_weight in CONTRAST_WEIGHTS:
                recurrent, action, fit = _fit(
                    source_rows, arrays, hidden=hidden, epochs=epochs, device=device,
                    contrast_weight=float(contrast_weight), history_source=history_source,
                )
                prediction, action_prediction, state, goal = _predict(
                    recurrent, action, fit, select, history_source=history_source,
                )
                actual = np.asarray([costs[_key(row)] for row in select])
                for temperature in TEMPERATURES:
                    recurrent_cost = _cost(state, goal, prediction, temperature)
                    action_cost = _cost(state, goal, action_prediction, 1.0)
                    policies = _history_gate_specs(select) if history_gate else [{"kind": "recurrent"}]
                    for policy in policies:
                        selected = _selection(select, _policy_cost(select, recurrent_cost, action_cost, policy), actual)
                        choices.append({
                            "history_source": history_source, "hidden": hidden,
                            "contrast_weight": float(contrast_weight), "temperature": temperature,
                            "policy": policy, "mean_regret": selected["mean_regret"],
                        })
    chosen = min(
        choices,
        key=lambda row: (
            row["mean_regret"], row["hidden"], row["contrast_weight"], abs(row["temperature"] - 1.0),
            0 if row["policy"]["kind"] == "action_only" else 1,
            declared_history_sources.index(str(row["history_source"])),
        ),
    )
    selected_history_source = str(chosen["history_source"])
    if observability_suite:
        recurrent, action, privileged, shuffled, fit, state_fit = _fit_observability(
            source_rows, arrays, hidden=int(chosen["hidden"]), epochs=epochs, device=device,
            contrast_weight=float(chosen["contrast_weight"]),
            history_source=selected_history_source,
        )
    else:
        recurrent, action, fit = _fit(
            source_rows, arrays, hidden=int(chosen["hidden"]), epochs=epochs, device=device,
            contrast_weight=float(chosen["contrast_weight"]),
            history_source=selected_history_source,
        )
        privileged = shuffled = state_fit = None
    prediction, action_prediction, state, goal = _predict(
        recurrent, action, fit, auth, history_source=selected_history_source,
    )
    actual = np.asarray([costs[_key(row)] for row in auth])
    recurrent_cost = _cost(state, goal, prediction, float(chosen["temperature"]))
    action_cost = _cost(state, goal, action_prediction, 1.0)
    selected_policy = dict(chosen["policy"])
    gain = _paired(
        _selection(auth, _policy_cost(auth, recurrent_cost, action_cost, selected_policy), actual),
        _selection(auth, action_cost, actual),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = _frozen_model_path(output)
    model_sha256 = _write_frozen_model(
        model_path, recurrent, action, fit, hidden=int(chosen["hidden"]),
        contrast_weight=float(chosen["contrast_weight"]), history_source=selected_history_source,
        privileged_state=privileged,
        shuffled_history=shuffled, state_mean=None if state_fit is None else state_fit["state_mean"],
        state_scale=None if state_fit is None else state_fit["state_scale"],
    )
    interaction = interaction_diagnostics(
        recurrent, fit, source_rows, history_source=selected_history_source,
    )
    frozen = {
        "schema": OBSERVABILITY_FROZEN_SCHEMA if observability_suite else FROZEN_SCHEMA,
        "source_sha256": _sha256(source), "source_arrays_sha256": _sha256(source_arrays),
        "context_sha256": _sha256(context), "calibration_labels_sha256": _sha256(calibration_labels),
        "calibration_labels_path": str(calibration_labels.resolve()),
        "manifest_sha256": _sha256(manifest),
        "source_schema": source_meta["schema"], "context_schema": context_meta["schema"],
        "calibration_partition": {"method": f"sha256({CALIBRATION_SPLIT_SALT}|root), sorted half split", **partition, "overlap": []},
        "root_counts": expected_counts,
        "history_source_candidates": list(declared_history_sources),
        "selected_history_source": selected_history_source,
        "choice_grid": choices, "selected_hidden": int(chosen["hidden"]),
        "selected_contrast_weight": float(chosen["contrast_weight"]),
        "selected_temperature": float(chosen["temperature"]), "selected_policy": selected_policy,
        "history_gate_enabled": bool(history_gate),
        "model_checkpoint_path": str(model_path.resolve()), "model_checkpoint_sha256": model_sha256,
        "trained_interaction_diagnostics": interaction,
        "authorization": {"selected_policy_over_action_only": gain, "permitted": bool(gain["lower"] > 0.0), "criterion": "disjoint authorization paired bootstrap lower CI > 0"},
        "epochs": int(epochs), "device": str(device), "cuda_preflight": cuda_runtime,
        "observability_suite": bool(observability_suite),
        "outcome_labels_opened": False, "outcome_labels_generated": False,
        "claim_boundary": "Matched V6 calibration roots were deterministically partitioned before selection; the effect-plus-rootwise-task-contrast objective, model width, temperature, and routing policy are frozen before PointMaze outcome simulation.",
    }
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def calibrate_pairwise_contrast(
    source: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path, *,
    epochs: int = 100, device: str = "cuda", manifest: Path | None = None,
    history_sources: tuple[str, ...] | list[str] | None = None, margin: float = 0.003,
) -> dict[str, Any]:
    """Freeze a C1 model by pairwise contrast, then authorize it by root.

    Unlike ``calibrate``, hyperparameters are selected by the declared V11
    counterfactual endpoint.  Full-bank regret is retained only as a secondary
    diagnostic so an unrelated 16-way argmin cannot choose the C1 model.
    """
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    manifest_payload = _validate_manifest(manifest, source, source_arrays, context, calibration_labels)
    expected_counts = _validated_root_counts(manifest_payload["root_counts"])
    cuda_runtime = require_cuda_recurrent_cell(device) if str(device) == "cuda" else None
    source_meta, source_rows = _read_panel(source, SOURCE_SCHEMA)
    context_meta, context_rows = _read_panel(context, CONTEXT_SCHEMA)
    _validate_builder_contract(
        source_rows, context_rows, context_meta,
        strict_counts=bool(manifest_payload.get("formal_protocol", True)), expected_root_counts=expected_counts,
    )
    calibration_rows = [row for row in context_rows if str(row["split"]) == "calibration"]
    costs = {
        _key(row): value for row, value in zip(
            calibration_rows, _calibration_costs(calibration_labels, calibration_rows, require_identity=True),
        )
    }
    partition = deterministic_calibration_split(context_rows)
    selection_roots, authorization_roots = set(partition["selection"]), set(partition["authorization"])
    select = [row for row in calibration_rows if str(row["instance_id"]) in selection_roots]
    authorize = [row for row in calibration_rows if str(row["instance_id"]) in authorization_roots]
    declared_history_sources = _validate_history_sources(history_sources)
    choices: list[dict[str, Any]] = []
    arrays = np.load(source_arrays, allow_pickle=False)
    for history_source in declared_history_sources:
        for hidden in HIDDEN_WIDTHS:
            for contrast_weight in CONTRAST_WEIGHTS:
                recurrent, action, fit = _fit(
                    source_rows, arrays, hidden=hidden, epochs=epochs, device=device,
                    contrast_weight=float(contrast_weight), history_source=history_source,
                )
                prediction, action_prediction, state, goal = _predict(
                    recurrent, action, fit, select, history_source=history_source,
                )
                actual = np.asarray([costs[_key(row)] for row in select])
                for temperature in TEMPERATURES:
                    recurrent_cost = _cost(state, goal, prediction, temperature)
                    pairwise = pairwise_counterfactual_contrast_metrics(select, recurrent_cost, actual, margin=margin)
                    if pairwise.get("status") != "evaluated" or pairwise.get("sign_accuracy") is None:
                        raise RuntimeError("C1 selection panel has no pairwise contrast above its frozen margin")
                    choices.append({
                        "history_source": history_source, "hidden": int(hidden),
                        "contrast_weight": float(contrast_weight), "temperature": float(temperature),
                        "policy": {"kind": "recurrent"},
                        "pairwise_sign_accuracy": float(pairwise["sign_accuracy"]),
                        "pairwise_contrast_mae": float(pairwise["contrast_mae"]),
                        "pairwise_valid_pair_count": int(pairwise["valid_pair_count"]),
                        "full_bank_mean_regret_diagnostic": float(_selection(select, recurrent_cost, actual)["mean_regret"]),
                    })
    chosen = min(
        choices,
        key=lambda row: (
            -float(row["pairwise_sign_accuracy"]), float(row["pairwise_contrast_mae"]),
            int(row["hidden"]), float(row["contrast_weight"]), abs(float(row["temperature"]) - 1.0),
            declared_history_sources.index(str(row["history_source"])),
        ),
    )
    selected_history_source = str(chosen["history_source"])
    recurrent, action, fit = _fit(
        source_rows, arrays, hidden=int(chosen["hidden"]), epochs=epochs, device=device,
        contrast_weight=float(chosen["contrast_weight"]), history_source=selected_history_source,
    )
    prediction, action_prediction, state, goal = _predict(
        recurrent, action, fit, authorize, history_source=selected_history_source,
    )
    actual = np.asarray([costs[_key(row)] for row in authorize])
    recurrent_cost = _cost(state, goal, prediction, float(chosen["temperature"]))
    action_cost = _cost(state, goal, action_prediction, 1.0)
    pairwise_authorization = pairwise_counterfactual_contrast_metrics(authorize, recurrent_cost, actual, margin=margin)
    action_pairwise_authorization = pairwise_counterfactual_contrast_metrics(authorize, action_cost, actual, margin=margin)
    gain = _paired_pairwise_sign_gain(authorize, recurrent_cost, action_cost, actual, margin=margin)
    model_path = _frozen_model_path(output)
    model_sha256 = _write_frozen_model(
        model_path, recurrent, action, fit, hidden=int(chosen["hidden"]),
        contrast_weight=float(chosen["contrast_weight"]), history_source=selected_history_source,
    )
    frozen = {
        "schema": FROZEN_SCHEMA,
        "source_sha256": _sha256(source), "source_arrays_sha256": _sha256(source_arrays),
        "context_sha256": _sha256(context), "calibration_labels_sha256": _sha256(calibration_labels),
        "calibration_labels_path": str(calibration_labels.resolve()), "manifest_sha256": _sha256(manifest),
        "source_schema": source_meta["schema"], "context_schema": context_meta["schema"],
        "root_counts": expected_counts,
        "calibration_partition": {"method": f"sha256({CALIBRATION_SPLIT_SALT}|root), sorted half split", **partition, "overlap": []},
        "history_source_candidates": list(declared_history_sources), "selected_history_source": selected_history_source,
        "choice_grid": choices, "selected_hidden": int(chosen["hidden"]),
        "selected_contrast_weight": float(chosen["contrast_weight"]),
        "selected_temperature": float(chosen["temperature"]), "selected_policy": {"kind": "recurrent"},
        "history_gate_enabled": False,
        "model_checkpoint_path": str(model_path.resolve()), "model_checkpoint_sha256": model_sha256,
        "trained_interaction_diagnostics": interaction_diagnostics(
            recurrent, fit, source_rows, history_source=selected_history_source,
        ),
        "c1_pairwise_calibration": {
            "margin": float(margin), "selection_endpoint": "max pairwise sign accuracy, then min contrast MAE",
            "selection_choice": chosen, "authorization_recurrent": pairwise_authorization,
            "authorization_action_only": action_pairwise_authorization,
            "authorization_root_cluster_gain": gain,
        },
        "authorization": {
            "selected_policy_over_action_only": gain, "permitted": bool(float(gain["lower"]) > 0.0),
            "criterion": "disjoint root-cluster bootstrap lower CI > 0 for pairwise sign-accuracy gain over action-only",
        },
        "epochs": int(epochs), "device": str(device), "cuda_preflight": cuda_runtime,
        "observability_suite": False, "outcome_labels_opened": False, "outcome_labels_generated": False,
        "claim_boundary": "C1 pairwise counterfactual-identifiability calibration. Hyperparameters and authorization use only separated calibration roots; full-bank planner regret is diagnostic only.",
    }
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def calibrate_fixed_policy(
    source: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path, *,
    hidden: int, contrast_weight: float, temperature: float, history_source: str,
    epochs: int = 100, device: str = "cuda", manifest: Path | None = None,
    protocol_reference: str, policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a previously declared policy and use all fresh calibration roots.

    This route is intentionally narrower than ``calibrate``: it contains no
    new feature, width, contrast, temperature, or gate selection.  It is for
    a fresh confirmation of an already pre-outcome-selected protocol where a
    larger independent calibration set is needed to decide permit versus
    fallback.  The full calibration panel is authorization-only.
    """
    if int(hidden) not in HIDDEN_WIDTHS or float(contrast_weight) not in CONTRAST_WEIGHTS:
        raise ValueError("fixed policy has an unsupported hidden width or contrast weight")
    if float(temperature) not in TEMPERATURES:
        raise ValueError("fixed policy has an unsupported temperature")
    _validate_history_sources((history_source,))
    if not protocol_reference:
        raise ValueError("fixed policy must bind its pre-outcome protocol reference")
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    manifest_payload = _validate_manifest(manifest, source, source_arrays, context, calibration_labels)
    expected_counts = _validated_root_counts(manifest_payload["root_counts"])
    cuda_runtime = require_cuda_recurrent_cell(device) if str(device) == "cuda" else None
    source_meta, source_rows = _read_panel(source, SOURCE_SCHEMA)
    context_meta, context_rows = _read_panel(context, CONTEXT_SCHEMA)
    _validate_builder_contract(
        source_rows, context_rows, context_meta,
        strict_counts=bool(manifest_payload.get("formal_protocol", True)), expected_root_counts=expected_counts,
    )
    arrays = np.load(source_arrays, allow_pickle=False)
    calibration_rows = [row for row in context_rows if str(row["split"]) == "calibration"]
    costs = {
        _key(row): value for row, value in zip(
            calibration_rows,
            _calibration_costs(calibration_labels, calibration_rows, require_identity=True),
        )
    }
    recurrent, action, fit = _fit(
        source_rows, arrays, hidden=int(hidden), epochs=epochs, device=device,
        contrast_weight=float(contrast_weight), history_source=history_source,
    )
    prediction, action_prediction, state, goal = _predict(
        recurrent, action, fit, calibration_rows, history_source=history_source,
    )
    actual = np.asarray([costs[_key(row)] for row in calibration_rows])
    recurrent_cost = _cost(state, goal, prediction, float(temperature))
    action_cost = _cost(state, goal, action_prediction, 1.0)
    selected_policy = {"kind": "recurrent"} if policy is None else dict(policy)
    if selected_policy.get("kind") not in {"recurrent", "action_only", "history_energy_gate"}:
        raise ValueError("fixed policy must be recurrent, action_only, or history_energy_gate")
    if selected_policy.get("kind") == "history_energy_gate":
        if str(selected_policy.get("direction")) not in {"recurrent_when_low", "recurrent_when_high"}:
            raise ValueError("fixed history gate needs a valid direction")
        if not np.isfinite(float(selected_policy.get("threshold", float("nan")))):
            raise ValueError("fixed history gate needs a finite threshold")
        motion_source = str(selected_policy.get("motion_source", "source_history_model"))
        if motion_source not in {"source_history_model", "source_history_visual_state"}:
            raise ValueError("fixed history gate has an invalid motion source")
        selected_policy["motion_source"] = motion_source
        selected_policy["fixed_threshold"] = True
    selected = _selection(calibration_rows, _policy_cost(calibration_rows, recurrent_cost, action_cost, selected_policy), actual)
    action_only = _selection(calibration_rows, action_cost, actual)
    gain = _paired(selected, action_only)
    model_path = _frozen_model_path(output)
    model_sha256 = _write_frozen_model(
        model_path, recurrent, action, fit, hidden=int(hidden), contrast_weight=float(contrast_weight),
        history_source=history_source,
    )
    authorization_roots = sorted(_group(calibration_rows))
    frozen = {
        "schema": FROZEN_SCHEMA,
        "source_sha256": _sha256(source), "source_arrays_sha256": _sha256(source_arrays),
        "context_sha256": _sha256(context), "calibration_labels_sha256": _sha256(calibration_labels),
        "calibration_labels_path": str(calibration_labels.resolve()), "manifest_sha256": _sha256(manifest),
        "source_schema": source_meta["schema"], "context_schema": context_meta["schema"],
        "root_counts": expected_counts,
        "fixed_policy_authorization_all_calibration": True,
        "protocol_reference": protocol_reference,
        "calibration_partition": {
            "method": "fixed pre-outcome policy; all fresh calibration roots authorize only",
            "selection": [], "authorization": authorization_roots, "overlap": [],
        },
        "history_source_candidates": [history_source], "selected_history_source": history_source,
        "choice_grid": [{
            "kind": "fixed_pre_outcome_confirmation", "hidden": int(hidden),
            "contrast_weight": float(contrast_weight), "temperature": float(temperature),
            "history_source": history_source, "policy": selected_policy,
        }],
        "selected_hidden": int(hidden), "selected_contrast_weight": float(contrast_weight),
        "selected_temperature": float(temperature), "selected_policy": selected_policy,
        "history_gate_enabled": bool(selected_policy["kind"] == "history_energy_gate"),
        "model_checkpoint_path": str(model_path.resolve()), "model_checkpoint_sha256": model_sha256,
        "trained_interaction_diagnostics": interaction_diagnostics(
            recurrent, fit, source_rows, history_source=history_source,
        ),
        "authorization": {
            "selected_policy_over_action_only": gain, "permitted": bool(gain["lower"] > 0.0),
            "criterion": "all-fresh-calibration paired bootstrap lower CI > 0 for fixed pre-outcome policy",
        },
        "epochs": int(epochs), "device": str(device), "cuda_preflight": cuda_runtime,
        "observability_suite": False, "outcome_labels_opened": False, "outcome_labels_generated": False,
        "claim_boundary": "Fixed pre-outcome policy confirmation. All fresh calibration roots authorize one already-declared policy; no outcome labels or V10 policy search are used before scoring.",
    }
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def score(
    source: Path, source_arrays: Path, context: Path, frozen_selection: Path, output: Path, *,
    device: str = "cuda", observability_suite: bool = False, history_gate: bool = False, manifest: Path | None = None,
) -> dict[str, Any]:
    cuda_runtime = require_cuda_recurrent_cell(device) if str(device) == "cuda" else None
    if not frozen_selection.is_file(): raise FileNotFoundError("score requires a persisted frozen selection artifact")
    frozen = json.loads(frozen_selection.read_text(encoding="utf-8"))
    expected_schema = OBSERVABILITY_FROZEN_SCHEMA if observability_suite else FROZEN_SCHEMA
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    calibration_path = Path(str(frozen.get("calibration_labels_path", "")))
    if not calibration_path.is_file():
        raise ValueError("frozen selection lacks a valid calibration-label path")
    manifest_payload = _validate_manifest(manifest, source, source_arrays, context, calibration_path)
    expected_counts = _validated_root_counts(manifest_payload["root_counts"])
    if frozen.get("manifest_sha256") != _sha256(manifest):
        raise ValueError("frozen selection manifest provenance does not match current manifest")
    model_path, model_sha256 = _resolve_frozen_checkpoint(
        frozen, frozen_selection, expected_schema=expected_schema,
    )
    current = (_sha256(source), _sha256(source_arrays), _sha256(context))
    if (frozen.get("source_sha256"), frozen.get("source_arrays_sha256"), frozen.get("context_sha256")) != current:
        raise ValueError("frozen selection provenance does not match builder artifacts")
    _source_meta, source_rows = _read_panel(source, SOURCE_SCHEMA); _context_meta, context_rows = _read_panel(context, CONTEXT_SCHEMA)
    _validate_builder_contract(
        source_rows, context_rows, _context_meta,
        strict_counts=bool(manifest_payload.get("formal_protocol", True)), expected_root_counts=expected_counts,
    )
    selected_policy = _validate_frozen_selection_contract(
        frozen, context_rows, expected_schema=expected_schema, history_gate=history_gate,
        expected_root_counts=expected_counts,
    )
    contrast_weight = float(frozen["selected_contrast_weight"])
    selected_history_source = str(frozen.get("selected_history_source", "source_history_model"))
    if observability_suite:
        recurrent, action, privileged, shuffled, fit, state_fit = _load_observability_models(
            model_path, model_sha256, device=device, expected_contrast_weight=contrast_weight,
            expected_history_source=selected_history_source,
        )
    else:
        recurrent, action, fit = _load_frozen_model(
            model_path, model_sha256, device=device, expected_contrast_weight=contrast_weight,
            expected_history_source=selected_history_source,
        )
    output.mkdir(parents=True, exist_ok=True); label_path = output / "temporal_ood_v6_outcome_labels.json"
    if label_path.exists(): raise FileExistsError("refusing to overwrite V6 generated outcome labels")
    outcome_rows, labels, frozen_sha = [row for row in context_rows if str(row["split"]) == "outcome"], {}, _sha256(frozen_selection)
    # This is the first PointMaze outcome replay, after frozen selection validation.
    for row in outcome_rows:
        terminal = _simulate_sequence(np.asarray(row["source_state_raw"], dtype=np.float64), np.asarray(row["action_sequence"], dtype=np.float64))
        goal = np.asarray(row["goal_raw"], dtype=np.float64)
        labels[_key(row)] = {"real_task_cost": float(np.linalg.norm(terminal[:2] - goal[:2])), "realized_terminal_state": np.asarray(terminal, dtype=np.float64).tolist()}
    label_path.write_text(json.dumps({"schema": OUTCOME_SCHEMA, "generated_after_frozen_selection_sha256": frozen_sha, "executor": "experiments.decision_benchmark.run_unseen_action_bank_pointmaze._simulate_sequence", "labels": labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outcome_labels_sha256 = _sha256(label_path)
    if observability_suite:
        prediction, action_prediction, state_prediction, shuffled_prediction, state, goal = _predict_observability(
            recurrent, action, privileged, shuffled, fit, state_fit, outcome_rows,
            history_source=selected_history_source,
        )
    else:
        prediction, action_prediction, state, goal = _predict(
            recurrent, action, fit, outcome_rows, history_source=selected_history_source,
        )
    actual = np.asarray([labels[_key(row)]["real_task_cost"] for row in outcome_rows])
    recurrent_cost = _cost(state, goal, prediction, float(frozen["selected_temperature"]))
    action_cost = _cost(state, goal, action_prediction, 1.0)
    rec, act = _selection(outcome_rows, recurrent_cost, actual), _selection(outcome_rows, action_cost, actual)
    selected_policy = dict(selected_policy)
    selected = _selection(outcome_rows, _policy_cost(outcome_rows, recurrent_cost, action_cost, selected_policy), actual)
    selected_cost = _policy_cost(outcome_rows, recurrent_cost, action_cost, selected_policy)
    report = {"schema": SCORE_SCHEMA, "source_sha256": current[0], "source_arrays_sha256": current[1], "context_sha256": current[2], "frozen_selection_sha256": frozen_sha, "outcome_label_path": str(label_path.resolve()), "outcome_labels_sha256": outcome_labels_sha256, "outcome_labels_generated_after_frozen_selection": True, "outcome_labels_preexisted": False, "outcome_root_ids": sorted(_group(outcome_rows)), "recurrent": rec, "action_only": act, "selected_contrast_weight": contrast_weight, "selected_history_source": selected_history_source, "selected_policy": selected_policy, "selected": selected, "active_policy": selected_policy["kind"] if frozen["authorization"]["permitted"] else "action_only_fallback", "active": selected if frozen["authorization"]["permitted"] else act, "pairwise_counterfactual_contrast": {"recurrent": pairwise_counterfactual_contrast_metrics(outcome_rows, recurrent_cost, actual), "action_only": pairwise_counterfactual_contrast_metrics(outcome_rows, action_cost, actual), "selected": pairwise_counterfactual_contrast_metrics(outcome_rows, selected_cost, actual)}, "device": str(device), "cuda_preflight": cuda_runtime, "claim_boundary": "Offline renderer-backed PointMaze matched-support result with privileged source position only for paired task-cost readout; the contrast objective is fit on source roots and selected without opening outcome labels. It is not live visual deployment or a universal world-model claim."}
    if observability_suite:
        state_arm = _selection(outcome_rows, _cost(state, goal, state_prediction, 1.0), actual)
        shuffle_arm = _selection(outcome_rows, _cost(state, goal, shuffled_prediction, float(frozen["selected_temperature"])), actual)
        oracle = _selection(outcome_rows, actual, actual)
        report["observability_suite"] = {
            "candidate_oracle": oracle,
            "privileged_state_effect": state_arm,
            "history_cyclic_shuffle": shuffle_arm,
            "paired_gain_state_over_action_only": _paired(state_arm, act),
            "paired_gain_visual_over_cyclic_shuffle": _paired(rec, shuffle_arm),
            "paired_gain_visual_over_action_only": _paired(rec, act),
            "state_arm_diagnostic_only": True,
        }
    (output / "temporal_ood_v6_score_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--source", type=Path, required=True); calibration.add_argument("--source-arrays", type=Path, required=True); calibration.add_argument("--context", type=Path, required=True); calibration.add_argument("--calibration-labels", type=Path, required=True); calibration.add_argument("--output", type=Path, required=True); calibration.add_argument("--epochs", type=int, default=100); calibration.add_argument("--device", choices=("cuda",), default="cuda"); calibration.add_argument("--manifest", type=Path, default=None); calibration.add_argument("--observability-suite", action="store_true"); calibration.add_argument("--history-gate", action="store_true"); calibration.add_argument("--history-sources", choices=HISTORY_SOURCES, nargs="+", default=None)
    scoring = sub.add_parser("score")
    scoring.add_argument("--source", type=Path, required=True); scoring.add_argument("--source-arrays", type=Path, required=True); scoring.add_argument("--context", type=Path, required=True); scoring.add_argument("--frozen-selection", type=Path, required=True); scoring.add_argument("--output", type=Path, required=True); scoring.add_argument("--device", choices=("cuda",), default="cuda"); scoring.add_argument("--manifest", type=Path, default=None); scoring.add_argument("--observability-suite", action="store_true"); scoring.add_argument("--history-gate", action="store_true")
    args = vars(parser.parse_args()); command = args.pop("command")
    result = calibrate(**args) if command == "calibrate" else score(**args)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
