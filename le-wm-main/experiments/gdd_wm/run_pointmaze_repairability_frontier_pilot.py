"""Run an execution-grounded, root-disjoint PointMaze repairability pilot.

This is deliberately *not* a GDD teacher claim.  It asks a narrower question:
given a fixed candidate bank, can evidence frozen before outcome selection tell
us when a learned execution-supervised repair will beat a model-based
predicted-state-distance (PSD) selector?  All methods use simulator state and
executed transitions and are therefore explicitly privileged controls.

The split contract is strict:

* train roots fit PSD dynamics and the candidate scorers;
* GDD roots choose the ordinary MLP configuration and fit the repairability
  router labels;
* calibration roots freeze the router threshold; and
* outcome roots are executed only after every selector and routing decision is
  frozen.

No field named ``real_task_cost`` is consulted.  Costs are freshly calculated
from legacy PointMaze MuJoCo terminal states after a selector is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_execution_grounded_repairability_frontier.v0.1"
CONTROL_LABEL = "privileged_execution_supervised_state_action_control"
ROLES = ("train", "gdd", "calibration", "outcome")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_summary(row: dict[str, Any]) -> np.ndarray:
    action = np.asarray(row["action_sequence"], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 2 or not len(action) or not np.isfinite(action).all():
        raise ValueError(f"{row.get('candidate_id')}: action_sequence must be finite [steps,2]")
    return np.concatenate((action.mean(axis=0), action.std(axis=0), action[0], action[-1]))


def _features(row: dict[str, Any]) -> np.ndarray:
    state = np.asarray(row["source_state_raw"], dtype=np.float64).reshape(-1)
    goal = np.asarray(row["goal_raw"], dtype=np.float64).reshape(-1)
    latent = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1)
    if state.size != 4 or goal.size != 2 or latent.size < 1:
        raise ValueError("source_state_raw[4], goal_raw[2], and source_latent_model are required")
    return np.concatenate((state, goal, latent, _action_summary(row)))


def _state_action_features(row: dict[str, Any]) -> np.ndarray:
    """PSD deliberately excludes goal and latent features from dynamics fit."""

    state = np.asarray(row["source_state_raw"], dtype=np.float64).reshape(-1)
    if state.size != 4:
        raise ValueError("source_state_raw must have length 4")
    return np.concatenate((state, _action_summary(row)))


def _load_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata, records = payload.get("metadata"), payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list) or not records:
        raise ValueError("candidate dataset must contain non-empty metadata and records")
    rows = [dict(row) for row in records]
    for row in rows:
        for field in ("instance_id", "split", "candidate_id", "source_state_raw", "goal_raw", "action_sequence"):
            if field not in row:
                raise ValueError(f"candidate record is missing {field}")
    return metadata, rows


def _assign_roles(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Split original train roots deterministically into train/GDD, untouched held splits.

    The source protocol already supplies root-disjoint ``train``, ``calibration``
    and ``outcome`` roots.  We split only its train roots before reading any
    execution outcomes, using a root-ID hash; this creates an independent GDD
    model-selection role without contaminating calibration or outcome roots.
    """

    root_source: dict[str, str] = {}
    for row in rows:
        root, source = str(row["instance_id"]), str(row["split"])
        if source not in {"train", "calibration", "outcome"}:
            raise ValueError(f"unsupported source split {source}")
        if root_source.setdefault(root, source) != source:
            raise ValueError(f"root {root} appears in multiple source splits")
    source_train = sorted((root for root, split in root_source.items() if split == "train"), key=lambda x: _sha256_bytes(x.encode()))
    if len(source_train) < 8:
        raise ValueError("need at least eight source-train roots to create train/GDD roles")
    train_count = max(4, int(math.floor(len(source_train) * 2.0 / 3.0)))
    assignments = {root: ("train" if index < train_count else "gdd") for index, root in enumerate(source_train)}
    assignments.update({root: split for root, split in root_source.items() if split != "train"})
    if set(assignments.values()) != set(ROLES):
        raise ValueError("all train/GDD/calibration/outcome roles must be non-empty")
    return assignments


def _groups(rows: list[dict[str, Any]], roles: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        root = str(row["instance_id"])
        identity = (root, str(row["candidate_id"]))
        if identity in seen:
            raise ValueError(f"duplicate root/candidate identity {identity}")
        seen.add(identity)
        grouped[root].append(row)
    for root, group in grouped.items():
        if root not in roles or len(group) < 2:
            raise ValueError(f"{root}: each root requires at least two same-bank candidates")
        shapes = {tuple(np.asarray(row["source_state_raw"]).shape) for row in group}
        goals = {tuple(np.asarray(row["goal_raw"], dtype=np.float64)) for row in group}
        if shapes != {(4,)} or len(goals) != 1:
            raise ValueError(f"{root}: candidates must share a valid source-state shape and goal")
    return dict(grouped)


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float = 1.0e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate((np.ones((len(features), 1)), standardized), axis=1)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target), mean, scale


def _ridge_predict(features: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    beta, mean, scale = model
    design = np.concatenate((np.ones((len(features), 1)), (features - mean) / scale), axis=1)
    return design @ beta


def _fresh_execution(rows: list[dict[str, Any]]) -> np.ndarray:
    """Execute declared sequences afresh; outcome values are never dataset labels."""

    import mujoco
    from experiments.decision_benchmark.verify_rgb_successor import _simulate_blocks
    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    terminal = []
    for row in rows:
        blocks = list(_simulate_blocks(model, np.asarray(row["source_state_raw"], dtype=np.float64), np.asarray(row["action_sequence"], dtype=np.float64)))
        if not blocks:
            raise ValueError(f"{row['candidate_id']}: declared action had no terminal block")
        terminal.append(np.asarray(blocks[-1], dtype=np.float64))
    return np.stack(terminal)


def _terminal_cost(rows: list[dict[str, Any]], terminal: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.linalg.norm(value[:2] - np.asarray(row["goal_raw"], dtype=np.float64))
        for row, value in zip(rows, terminal)
    ], dtype=np.float64)


def _torch_mlp_fit(features: np.ndarray, target: np.ndarray, *, width: int, steps: int, device: str, seed: int) -> tuple[dict[str, Any], Callable[[np.ndarray], np.ndarray]]:
    import torch

    torch.manual_seed(seed)
    x_mean, x_scale = features.mean(axis=0), features.std(axis=0)
    x_scale[x_scale < 1.0e-8] = 1.0
    y_mean, y_scale = float(np.mean(target)), float(np.std(target))
    y_scale = max(y_scale, 1.0e-6)
    x = torch.as_tensor((features - x_mean) / x_scale, dtype=torch.float32, device=device)
    y = torch.as_tensor((target - y_mean) / y_scale, dtype=torch.float32, device=device).reshape(-1, 1)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], width), torch.nn.ReLU(), torch.nn.Linear(width, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-2, weight_decay=1.0e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}

    def predict(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            value = torch.as_tensor((values - x_mean) / x_scale, dtype=torch.float32, device=device)
            return (model(value).reshape(-1).cpu().numpy() * y_scale + y_mean).astype(np.float64)

    return {"width": width, "steps": steps, "x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale, "state": state}, predict


def _fit_pairwise(features: np.ndarray, costs: np.ndarray, root_keys: list[str], *, device: str, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Naive direct preference learner; pair labels are train execution outcomes."""

    import torch

    pairs, labels = [], []
    by_root: dict[str, list[int]] = defaultdict(list)
    for index, root in enumerate(root_keys):
        by_root[root].append(index)
    for indices in by_root.values():
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1:]:
                if abs(float(costs[left] - costs[right])) <= 1.0e-10:
                    continue
                pairs.append(features[left] - features[right])
                labels.append(float(costs[left] < costs[right]))
                pairs.append(features[right] - features[left])
                labels.append(float(costs[right] < costs[left]))
    if len(pairs) < 4:
        raise ValueError("insufficient non-tied train preference pairs")
    x_mean, x_scale = features.mean(axis=0), features.std(axis=0)
    x_scale[x_scale < 1.0e-8] = 1.0
    x = torch.as_tensor((np.asarray(pairs) / x_scale), dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], 1, bias=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-2, weight_decay=1.0e-3)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()

    def score(values: np.ndarray) -> np.ndarray:
        standardized = (values - x_mean) / x_scale
        with torch.no_grad():
            logits = model(torch.as_tensor(standardized, dtype=torch.float32, device=device)).reshape(-1).cpu().numpy()
        # Larger is better, matching all other selector score contracts.
        return np.asarray(logits, dtype=np.float64)

    return score


def _choose(group: list[dict[str, Any]], scores: np.ndarray) -> int:
    return min(range(len(group)), key=lambda index: (-float(scores[index]), str(group[index]["candidate_id"])))


def _router_features(group: list[dict[str, Any]], psd: np.ndarray, mlp: np.ndarray) -> np.ndarray:
    psd_pick, mlp_pick = _choose(group, psd), _choose(group, mlp)
    state = np.asarray(group[0]["source_state_raw"], dtype=np.float64)
    goal = np.asarray(group[0]["goal_raw"], dtype=np.float64)
    return np.asarray([
        float(psd[psd_pick] - psd[mlp_pick]),
        float(mlp[mlp_pick] - mlp[psd_pick]),
        float(np.std(psd)),
        float(np.std(mlp)),
        float(np.linalg.norm(state[:2] - goal)),
        float(np.linalg.norm(state[2:])),
        float(psd_pick != mlp_pick),
    ], dtype=np.float64)


def _fit_router(features: np.ndarray, labels: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Small logistic router fitted only on GDD roots; constant fallback is valid."""

    if len(np.unique(labels)) < 2:
        probability = float(np.mean(labels))
        return lambda values: np.full(len(values), probability, dtype=np.float64)
    design = np.concatenate((np.ones((len(features), 1)), features), axis=1)
    beta = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(400):
        logits = np.clip(design @ beta, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = design.T @ (prob - labels) / len(labels) + 1.0e-3 * np.r_[0.0, beta[1:]]
        beta -= 0.25 * grad
    return lambda values: 1.0 / (1.0 + np.exp(-np.clip(np.concatenate((np.ones((len(values), 1)), values), axis=1) @ beta, -30.0, 30.0)))


def _condition_outcome(groups: dict[str, list[dict[str, Any]]], scores: dict[str, dict[str, np.ndarray]], costs: dict[str, np.ndarray], router_probability: dict[str, float], threshold: float) -> dict[str, Any]:
    """Apply fixed selectors and report execution outcome after, never during, selection."""

    output: dict[str, Any] = {"rows": {}, "outcome_cost_used_for_selection": False}
    for root, group in groups.items():
        psd_pick = _choose(group, scores[root]["psd"])
        mlp_pick = _choose(group, scores[root]["mlp"])
        naive_pick = _choose(group, scores[root]["naive"])
        routed_pick = mlp_pick if router_probability[root] >= threshold else psd_pick
        oracle = int(np.argmin(costs[root]))
        output["rows"][root] = {
            "psd_pick": psd_pick,
            "mlp_pick": mlp_pick,
            "naive_pick": naive_pick,
            "routed_pick": routed_pick,
            "oracle_pick": oracle,
            "router_probability_pre_outcome": float(router_probability[root]),
            "router_selected_mlp_pre_outcome": bool(router_probability[root] >= threshold),
            "psd_regret": float(costs[root][psd_pick] - costs[root][oracle]),
            "mlp_regret": float(costs[root][mlp_pick] - costs[root][oracle]),
            "naive_regret": float(costs[root][naive_pick] - costs[root][oracle]),
            "routed_regret": float(costs[root][routed_pick] - costs[root][oracle]),
            "repair_outcome_mlp_beats_psd": bool(costs[root][mlp_pick] + 1.0e-10 < costs[root][psd_pick]),
        }
    return output


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = list(result["rows"].values())
    def average(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))
    repair = np.asarray([row["repair_outcome_mlp_beats_psd"] for row in rows], dtype=bool)
    routed = np.asarray([row["router_selected_mlp_pre_outcome"] for row in rows], dtype=bool)
    summary = {
        "root_count": len(rows),
        "mean_regret": {name: average(f"{name}_regret") for name in ("psd", "mlp", "naive", "routed")},
        "router_coverage": float(routed.mean()),
        "repair_outcome_prevalence": float(repair.mean()),
        "preselection_repair_precision": None if not routed.any() else float(repair[routed].mean()),
        "preselection_repair_recall": None if not repair.any() else float(routed[repair].mean()),
        "outcome_cost_used_for_selection": result["outcome_cost_used_for_selection"],
    }
    precision, prevalence = summary["preselection_repair_precision"], summary["repair_outcome_prevalence"]
    routed_regret, psd_regret = summary["mean_regret"]["routed"], summary["mean_regret"]["psd"]
    confirmed = bool(
        precision is not None
        and float(precision) > float(prevalence) + 0.05
        and routed_regret < psd_regret - 1.0e-10
    )
    summary["preselection_prediction_verdict"] = {
        "status": "confirmed_in_this_pilot" if confirmed else "not_confirmed_in_this_pilot",
        "criterion": "routing precision exceeds unconditioned repair prevalence by >0.05 and routed mean regret improves over PSD",
        "answer": (
            "The frozen routing condition predicts repair outcome before held execution under the declared pilot criterion."
            if confirmed else
            "The frozen routing condition does not establish pre-selection prediction of repair outcome: it fails the declared routing-benefit criterion."
        ),
    }
    return summary


def run(candidate_dataset: Path, output: Path, *, device: str = "cuda", seed: int = 8412) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata, rows = _load_records(candidate_dataset)
    roles = _assign_roles(rows)
    grouped = _groups(rows, roles)
    cuda = require_cuda_allocation(device)

    # Freshly execute every candidate only after root roles are locked.  Values
    # from GDD/calibration/outcome become labels only within their own phase.
    all_rows = [row for root in sorted(grouped) for row in grouped[root]]
    fresh_terminal = _fresh_execution(all_rows)
    fresh_cost = _terminal_cost(all_rows, fresh_terminal)
    costs_by_root: dict[str, np.ndarray] = {}
    cursor = 0
    for root in sorted(grouped):
        count = len(grouped[root])
        costs_by_root[root] = fresh_cost[cursor:cursor + count]
        cursor += count

    root_by_role = {role: sorted(root for root, assigned in roles.items() if assigned == role) for role in ROLES}
    train_roots = root_by_role["train"]
    train_rows = [row for root in train_roots for row in grouped[root]]
    train_cost = np.concatenate([costs_by_root[root] for root in train_roots])
    terminal_map = {id(row): terminal for row, terminal in zip(all_rows, fresh_terminal)}
    train_terminal = np.stack([terminal_map[id(row)] for row in train_rows])

    psd_model = _fit_ridge(np.stack([_state_action_features(row) for row in train_rows]), train_terminal)
    candidate_widths = (8, 16)
    gdd_roots = root_by_role["gdd"]
    gdd_rows = [row for root in gdd_roots for row in grouped[root]]
    gdd_cost = np.concatenate([costs_by_root[root] for root in gdd_roots])
    feature_train, feature_gdd = np.stack([_features(row) for row in train_rows]), np.stack([_features(row) for row in gdd_rows])
    candidates: list[tuple[float, int, dict[str, Any], Callable[[np.ndarray], np.ndarray]]] = []
    for width in candidate_widths:
        model_data, predictor = _torch_mlp_fit(feature_train, train_cost, width=width, steps=300, device=device, seed=seed + width)
        candidates.append((float(np.mean((predictor(feature_gdd) - gdd_cost) ** 2)), width, model_data, predictor))
    _, chosen_width, mlp_state, mlp_predict = min(candidates, key=lambda item: (item[0], item[1]))
    naive_score = _fit_pairwise(feature_train, train_cost, [str(row["instance_id"]) for row in train_rows], device=device, seed=seed + 31)

    scores: dict[str, dict[str, np.ndarray]] = {}
    router_feature: dict[str, np.ndarray] = {}
    for root, group in grouped.items():
        psd_terminal = _ridge_predict(np.stack([_state_action_features(row) for row in group]), psd_model)
        goals = np.stack([np.asarray(row["goal_raw"], dtype=np.float64) for row in group])
        psd_score = -np.linalg.norm(psd_terminal[:, :2] - goals, axis=1)
        full_features = np.stack([_features(row) for row in group])
        mlp_score = -mlp_predict(full_features)  # lower predicted cost is better
        naive = naive_score(full_features)
        scores[root] = {"psd": psd_score, "mlp": mlp_score, "naive": naive}
        router_feature[root] = _router_features(group, psd_score, mlp_score)

    gdd_features = np.stack([router_feature[root] for root in gdd_roots])
    gdd_labels = []
    for root in gdd_roots:
        group, cost = grouped[root], costs_by_root[root]
        psd_pick, mlp_pick = _choose(group, scores[root]["psd"]), _choose(group, scores[root]["mlp"])
        gdd_labels.append(float(cost[mlp_pick] + 1.0e-10 < cost[psd_pick]))
    router = _fit_router(gdd_features, np.asarray(gdd_labels, dtype=np.float64))
    probabilities = {root: float(router(router_feature[root].reshape(1, -1))[0]) for root in grouped}

    # Calibration outcomes tune one scalar release threshold; candidate/model
    # selection was already fixed by train/GDD.  Tie-break toward abstention.
    calibration_roots = root_by_role["calibration"]
    candidate_thresholds = sorted({0.0, 0.25, 0.5, 0.75, 1.0, *[probabilities[root] for root in calibration_roots]})
    best_threshold, best_regret = 1.0, float("inf")
    for threshold in candidate_thresholds:
        result = _condition_outcome({root: grouped[root] for root in calibration_roots}, scores, costs_by_root, probabilities, threshold)
        regret = _summary(result)["mean_regret"]["routed"]
        if regret < best_regret - 1.0e-12 or (np.isclose(regret, best_regret) and threshold > best_threshold):
            best_threshold, best_regret = float(threshold), float(regret)
    outcome_roots = root_by_role["outcome"]
    outcome = _condition_outcome({root: grouped[root] for root in outcome_roots}, scores, costs_by_root, probabilities, best_threshold)
    outcome_summary = _summary(outcome)
    calibration = _condition_outcome({root: grouped[root] for root in calibration_roots}, scores, costs_by_root, probabilities, best_threshold)

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SCHEMA,
        "status": "completed_execution_grounded_privileged_control_pilot",
        "claim_boundary": (
            "A PointMaze simulator-state, execution-supervised privileged control pilot. It tests whether a routing decision "
            "frozen before held outcome execution predicts whether an MLP repair beats PSD on a fixed bank. It is not a "
            "D-JEPA teacher result, frozen-visual result, GDD distillation result, deployment result, or universal claim."
        ),
        "control_contract": {
            "label": CONTROL_LABEL,
            "privileged_inputs": ["source_state_raw", "goal_raw"],
            "execution_supervision": "fresh_legacy_pointmaze_mujoco_terminal_state_and_goal_distance",
            "source_dataset_real_task_cost_used": False,
            "prediction_preservation": {
                "applicable": False,
                "reason": "The pilot does not alter the frozen encoder or world-model predictor; it compares privileged execution-supervised selectors.",
            },
        },
        "split_isolation": {
            "roles": {role: roots for role, roots in root_by_role.items()},
            "root_disjoint": all(not(set(root_by_role[left]) & set(root_by_role[right])) for left in ROLES for right in ROLES if left != right),
            "train_use": "fit PSD dynamics, MLP scorer, and naive pairwise learner",
            "gdd_use": "select ordinary MLP width and fit repairability router",
            "calibration_use": "freeze only router release threshold",
            "outcome_use": "fresh post-freeze execution evaluation only",
        },
        "preselection_contract": {
            "condition": "route to MLP iff train/GDD/calibration-frozen probability >= threshold; otherwise PSD",
            "outcome_cost_used_for_selection": False,
            "threshold_frozen_before_outcome": best_threshold,
            "core_question": "Does pre-selection predict repair outcome, rather than select after held outcome?",
        },
        "methods": {
            "psd_metric": "train-fit privileged state/action ridge predicts terminal state; score is predicted terminal XY distance to privileged goal",
            "ordinary_mlp_scorer": {"selected_width_on_gdd": chosen_width, "candidate_gdd_mse": {str(width): mse for mse, width, _state, _predictor in candidates}},
            "naive_preference_learner": "train-only pairwise execution-cost logistic learner",
            "repairability_routing": "GDD-fit logistic predictor of MLP-beats-PSD; calibration-frozen threshold",
        },
        "calibration": {"mean_routed_regret": _summary(calibration)["mean_regret"]["routed"], "threshold": best_threshold},
        "outcome": outcome_summary,
        "outcome_per_root": outcome["rows"],
        "cuda_preflight": cuda,
        "provenance": {"candidate_dataset": str(candidate_dataset.resolve()), "candidate_dataset_sha256": _file_hash(candidate_dataset), "world_model_sha256": metadata.get("world_model_sha256")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=8412)
    args = parser.parse_args()
    report = run(args.candidate_dataset, args.output, device=args.device, seed=args.seed)
    print(json.dumps({"status": report["status"], "outcome": report["outcome"], "claim_boundary": report["claim_boundary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
