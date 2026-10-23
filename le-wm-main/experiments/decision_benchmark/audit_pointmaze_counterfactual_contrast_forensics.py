"""Diagnose why a sealed counterfactual-contrast panel did not transfer.

This is a post-outcome audit only.  It never selects a model, modifies a
frozen decision, or opens labels.  Its purpose is to distinguish three common
negative-result mechanisms: a fixed-bank action prior, absent visual headroom
relative to a privileged state arm, and a calibration/authorization failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pointmaze_counterfactual_contrast_forensic_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _winner_summary(arm: dict[str, Any], *, candidate_count: int | None) -> dict[str, Any]:
    roots = arm.get("roots")
    if not isinstance(roots, dict) or not roots:
        raise ValueError("score arm lacks root-level winner records")
    winners: list[int] = []
    regrets: list[float] = []
    for root, record in roots.items():
        if not isinstance(root, str) or not isinstance(record, dict):
            raise ValueError("malformed root-level score record")
        winner, regret = record.get("winner_candidate_id"), record.get("regret")
        if not isinstance(winner, int) or not np.isfinite(float(regret)):
            raise ValueError("root-level score record lacks finite winner/regret")
        winners.append(winner); regrets.append(float(regret))
    counts = Counter(winners)
    total_candidates = max(winners) + 1 if candidate_count is None else int(candidate_count)
    if total_candidates < len(counts):
        raise ValueError("declared candidate-bank size is smaller than observed winners")
    values = np.asarray(list(counts.values()), dtype=np.float64)
    probability = values / float(len(winners))
    entropy = float(-(probability * np.log(probability)).sum()) if len(probability) else 0.0
    normalized = entropy / float(np.log(max(2, total_candidates)))
    dominant_id, dominant_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "root_count": int(len(winners)), "candidate_bank_count": int(total_candidates),
        "unique_winner_count": int(len(counts)), "winner_support_fraction": float(len(counts) / total_candidates),
        "winner_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "dominant_candidate_id": int(dominant_id), "dominant_winner_fraction": float(dominant_count / len(winners)),
        "winner_entropy_normalized_by_candidate_bank": normalized,
        "mean_regret": float(np.mean(regrets)), "nonzero_regret_fraction": float(np.mean(np.asarray(regrets) > 0.0)),
    }


def _gain(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    suite = payload.get("observability_suite")
    if not isinstance(suite, dict):
        return None
    value = suite.get(name)
    return dict(value) if isinstance(value, dict) else None


def _paired_recurrent_gain(recurrent: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Summarize the root-paired effect of overriding the action-only arm.

    This stays available for ordinary V6 scores, which intentionally omit the
    optional observability suite.  The signed gain is action-only regret minus
    recurrent regret, so a negative value is direct evidence that the learned
    interaction override made the selected action worse on that root.
    """
    recurrent_roots, action_roots = recurrent["roots"], action["roots"]
    roots = sorted(recurrent_roots)
    if roots != sorted(action_roots):
        raise ValueError("recurrent and action-only root bindings differ")
    values = np.asarray([
        float(action_roots[root]["regret"]) - float(recurrent_roots[root]["regret"])
        for root in roots
    ], dtype=np.float64)
    generator = np.random.default_rng(2026102602)
    draws = np.asarray([
        generator.choice(values, size=len(values), replace=True).mean()
        for _ in range(2000)
    ], dtype=np.float64)
    return {
        "root_count": int(len(values)), "mean_gain": float(values.mean()),
        "lower": float(np.quantile(draws, 0.025)), "upper": float(np.quantile(draws, 0.975)),
        "improved_root_count": int(np.sum(values > 0.0)),
        "harmed_root_count": int(np.sum(values < 0.0)),
        "tied_root_count": int(np.sum(values == 0.0)),
    }


def _ranking_intervention(recurrent: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Separate no-op roots from roots whose selected candidate changes."""
    recurrent_roots, action_roots = recurrent["roots"], action["roots"]
    roots = sorted(recurrent_roots)
    changed = np.asarray([
        int(recurrent_roots[root]["winner_candidate_id"])
        != int(action_roots[root]["winner_candidate_id"])
        for root in roots
    ], dtype=bool)
    values = np.asarray([
        float(action_roots[root]["regret"]) - float(recurrent_roots[root]["regret"])
        for root in roots
    ], dtype=np.float64)
    changed_values = values[changed]
    return {
        "root_count": int(len(roots)), "changed_winner_root_count": int(changed.sum()),
        "changed_winner_fraction": float(changed.mean()),
        "unchanged_winner_root_count": int((~changed).sum()),
        "changed_winner_mean_gain": None if not len(changed_values) else float(changed_values.mean()),
        "changed_winner_improved_root_count": int(np.sum(changed_values > 0.0)),
        "changed_winner_harmed_root_count": int(np.sum(changed_values < 0.0)),
        "changed_winner_gain_q05": None if not len(changed_values) else float(np.quantile(changed_values, 0.05)),
        "changed_winner_gain_q95": None if not len(changed_values) else float(np.quantile(changed_values, 0.95)),
    }


def audit(build: Path, frozen: Path, score: Path, output: Path) -> dict[str, Any]:
    """Write a mechanism audit from existing sealed JSON artifacts only."""
    for path in (build, frozen, score):
        if not path.is_file():
            raise FileNotFoundError(path)
    build_payload, frozen_payload, score_payload = _read_object(build), _read_object(frozen), _read_object(score)
    for key in ("action_only", "recurrent", "selected"):
        if not isinstance(score_payload.get(key), dict):
            raise ValueError(f"score report lacks {key!r} arm")
    design = build_payload.get("interaction_pair_design")
    hashes = design.get("held_pair_hashes") if isinstance(design, dict) else None
    candidate_count = len(hashes) if isinstance(hashes, list) and hashes else None
    action, recurrent, selected = (
        _winner_summary(score_payload[key], candidate_count=candidate_count)
        for key in ("action_only", "recurrent", "selected")
    )
    paired_recurrent_gain = _paired_recurrent_gain(score_payload["recurrent"], score_payload["action_only"])
    ranking_intervention = _ranking_intervention(score_payload["recurrent"], score_payload["action_only"])
    authorization = frozen_payload.get("authorization")
    if not isinstance(authorization, dict) or not isinstance(authorization.get("permitted"), bool):
        raise ValueError("frozen selection lacks authorization status")
    visual_gain = _gain(score_payload, "paired_gain_visual_over_action_only")
    state_gain = _gain(score_payload, "paired_gain_state_over_action_only")
    shuffle_gain = _gain(score_payload, "paired_gain_visual_over_cyclic_shuffle")
    interaction = frozen_payload.get("trained_interaction_diagnostics")
    mechanisms = {
        "authorization_rejected": not bool(authorization["permitted"]),
        "action_prior_concentrated": bool(
            action["dominant_winner_fraction"] >= 0.50 or action["winner_support_fraction"] <= 0.25
        ),
        "history_interaction_active": bool(isinstance(interaction, dict) and interaction.get("interaction_active_at_1e-6") is True),
        "visual_beats_action_with_positive_lower_ci": bool(visual_gain and float(visual_gain.get("lower", float("nan"))) > 0.0),
        "privileged_state_beats_action_with_positive_lower_ci": bool(state_gain and float(state_gain.get("lower", float("nan"))) > 0.0),
        "visual_beats_history_shuffle_with_positive_lower_ci": bool(shuffle_gain and float(shuffle_gain.get("lower", float("nan"))) > 0.0),
        "recurrent_override_has_positive_lower_ci": bool(paired_recurrent_gain["lower"] > 0.0),
        "recurrent_override_harms_changed_roots_on_average": bool(
            ranking_intervention["changed_winner_mean_gain"] is not None
            and float(ranking_intervention["changed_winner_mean_gain"]) < 0.0
        ),
    }
    result = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "input_sha256": {"build": _sha256(build), "frozen": _sha256(frozen), "score": _sha256(score)},
        "candidate_design": None if not isinstance(design, dict) else design.get("candidate_design"),
        "action_only": action, "recurrent": recurrent, "selected": selected,
        "selected_contrast_weight": score_payload.get("selected_contrast_weight"),
        "selected_policy": score_payload.get("selected_policy"),
        "authorization": authorization,
        "paired_gains": {"recurrent_over_action_only": paired_recurrent_gain, "visual_over_action_only": visual_gain, "state_over_action_only": state_gain, "visual_over_cyclic_shuffle": shuffle_gain},
        "ranking_intervention": ranking_intervention,
        "trained_interaction": interaction,
        "mechanism_flags": mechanisms,
        "interpretation": {
            "action_prior": "A concentrated action-only winner distribution means a candidate-bank prior can mask root-specific ranking information; it does not prove visual history is irrelevant.",
            "state_ceiling": "A non-positive lower interval for state over action-only means this panel lacks validated evidence that privileged current state alone supplies transferable ranking headroom.",
            "next_decision": "Use the V8 two-fold robust-bank pre-fit gate and test a calibration-only route for the roots on which a learned scorer changes the candidate; do not tune model weights or routing thresholds on these outcome roots.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite forensic audit: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
