"""Deploy a validated frozen-latent Reacher joint-angle observation readout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


READOUT_TASK = "reacher_absolute_qpos_sincos_ridge_v1"
PREDICTION_READOUT_TASK = "reacher_one_step_predicted_qpos_sincos_ridge_v1"


def load_qpos_readout(path: Path, *, expected_task: str = READOUT_TASK) -> dict[str, torch.Tensor]:
    """Load the closed-form calibration artifact with an explicit contract."""

    with np.load(Path(path), allow_pickle=False) as data:
        required = {"feature_mean", "feature_scale", "coefficient", "task"}
        if required.difference(data.files) or str(np.asarray(data["task"]).reshape(-1)[0]) != str(expected_task):
            raise ValueError("qpos readout does not satisfy the frozen Reacher angle-readout contract")
        values = {key: torch.as_tensor(np.asarray(data[key]).copy(), dtype=torch.float32) for key in ("feature_mean", "feature_scale", "coefficient")}
    width = int(values["feature_mean"].numel())
    if tuple(values["feature_scale"].shape) != (width,) or tuple(values["coefficient"].shape) != (width + 1, 4) or torch.any(values["feature_scale"] <= 0.0):
        raise ValueError("qpos readout arrays have incompatible shapes")
    return values


def qpos_readout_binding(
    encoder_artifact: Path | None,
    prediction_artifact: Path | None,
    *,
    policy_weights_sha256: str | None = None,
    adapter_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and fingerprint the optional source-calibrated P1 chart.

    Pairing the two readouts is necessary because an encoded RGB observation
    and a predictor output inhabit different frozen-latent distributions.
    A receipt cannot silently mix calibration artifacts or fall back to raw
    latent distance when only one artifact was supplied.
    """

    if (encoder_artifact is None) != (prediction_artifact is None):
        raise ValueError("Reacher P1 requires both encoder and predicted qpos-readout artifacts, or neither")
    if encoder_artifact is None:
        return {"enabled": False}
    encoder_path, prediction_path = Path(encoder_artifact).resolve(), Path(prediction_artifact).resolve()
    load_qpos_readout(encoder_path, expected_task=READOUT_TASK)
    load_qpos_readout(prediction_path, expected_task=PREDICTION_READOUT_TASK)
    binding = {
        "enabled": True,
        "encoder_artifact": str(encoder_path),
        "encoder_artifact_sha256": hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "encoder_task": READOUT_TASK,
        "prediction_task": PREDICTION_READOUT_TASK,
    }
    if policy_weights_sha256 is None and adapter_sha256 is None:
        # Kept only for lightweight tensor-contract tests. Runtime callers
        # must supply both hashes and therefore prove source provenance.
        return binding
    if not isinstance(policy_weights_sha256, str) or not isinstance(adapter_sha256, str):
        raise ValueError("Reacher P1 qpos readouts require both policy and action-adapter hashes")
    expected = (
        (encoder_path, "reacher_d2p_frozen_latent_qpos_readout.v1", READOUT_TASK),
        (prediction_path, "reacher_d2p_frozen_rollout_qpos_readout.v1", PREDICTION_READOUT_TASK),
    )
    reports: dict[str, dict[str, str]] = {}
    for artifact_path, schema, task in expected:
        report_path = artifact_path.parent / "report.json"
        if not report_path.is_file():
            raise ValueError("Reacher P1 qpos readout is missing its immutable calibration report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact = report.get("artifact", {})
        inputs = report.get("inputs", {})
        if (
            report.get("schema") != schema
            or report.get("status") != "completed"
            or not bool(report.get("validation", {}).get("gate_passed", False))
            or Path(str(artifact.get("path", ""))).resolve() != artifact_path
            or artifact.get("sha256") != hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            or inputs.get("weights_sha256") != policy_weights_sha256
            or inputs.get("adapter_sha256") != adapter_sha256
        ):
            raise ValueError("Reacher P1 qpos readout provenance does not match this frozen policy/action adapter")
        reports[task] = {"path": str(report_path.resolve()), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}
    binding["calibration_reports"] = reports
    return binding


def with_qpos_readout(base: torch.nn.Module, binding: dict[str, Any]) -> torch.nn.Module:
    """Wrap frozen dynamics in exactly the qpos chart named by a receipt."""

    if not bool(binding.get("enabled", False)):
        return base
    return ReacherQposReadoutCostModel(
        base,
        load_qpos_readout(Path(str(binding["encoder_artifact"])), expected_task=READOUT_TASK),
        load_qpos_readout(Path(str(binding["prediction_artifact"])), expected_task=PREDICTION_READOUT_TASK),
    )


class ReacherQposReadoutCostModel(torch.nn.Module):
    """Keep frozen dynamics while scoring terminals in an explicit qpos chart.

    An encoded RGB observation and an autoregressively predicted terminal are
    produced by different maps.  A single readout is valid only if that is
    established empirically.  ``prediction_artifact`` therefore allows a
    source-calibrated predictor readout while retaining the RGB-encoder
    artifact for observed roots and goals.  Omitting it preserves the original
    one-readout contract for historical receipts.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        artifact: dict[str, torch.Tensor],
        prediction_artifact: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.base = base
        if prediction_artifact is None:
            prediction_artifact = artifact
        for key in ("feature_mean", "feature_scale", "coefficient"):
            value = artifact[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"qpos readout field {key!r} is not a tensor")
            self.register_buffer(key, value.detach().clone())
            predicted = prediction_artifact[key]
            if not isinstance(predicted, torch.Tensor) or tuple(predicted.shape) != tuple(value.shape):
                raise ValueError("prediction qpos readout is incompatible with the encoder readout")
            self.register_buffer(f"prediction_{key}", predicted.detach().clone())

    @property
    def action_encoder(self):
        return self.base.action_encoder

    def encode(self, info_dict: dict):
        return self.base.encode(info_dict)

    def predict(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.base.predict(embeddings, actions)

    @staticmethod
    def _apply_readout(
        value: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        coefficient: torch.Tensor,
    ) -> torch.Tensor:
        scaled = (value - feature_mean.to(value)) / feature_scale.to(value)
        ones = torch.ones(*scaled.shape[:-1], 1, device=value.device, dtype=value.dtype)
        return torch.cat((scaled, ones), dim=-1) @ coefficient.to(value)

    def _readout(self, value: torch.Tensor) -> torch.Tensor:
        return self._apply_readout(value, self.feature_mean, self.feature_scale, self.coefficient)

    def _prediction_readout(self, value: torch.Tensor) -> torch.Tensor:
        return self._apply_readout(
            value,
            self.prediction_feature_mean,
            self.prediction_feature_scale,
            self.prediction_coefficient,
        )

    @staticmethod
    def _unit_coordinate(qpos_sincos: torch.Tensor) -> torch.Tensor:
        sine, cosine = qpos_sincos[..., :2], qpos_sincos[..., 2:]
        norm = torch.sqrt(sine.square() + cosine.square()).clamp_min(1.0e-8)
        return torch.cat((sine / norm, cosine / norm), dim=-1)

    def observed_coordinate(self, value: torch.Tensor) -> torch.Tensor:
        """Coordinate for an actually rendered successor RGB embedding."""

        return self._unit_coordinate(self._readout(value))

    def goal_coordinate(self, value: torch.Tensor) -> torch.Tensor:
        """Coordinate for an actually rendered RGB goal embedding."""

        return self.observed_coordinate(value)

    def predicted_coordinate(self, value: torch.Tensor) -> torch.Tensor:
        """Coordinate for a latent emitted by the frozen rollout predictor."""

        return self._unit_coordinate(self._prediction_readout(value))

    def reachability_coordinate(self, value: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for the planner's predicted-terminal chart."""

        return self.predicted_coordinate(value)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        # The base call both retains the unmodified frozen rollout and exposes
        # the predicted/goal embeddings under its established cache keys.
        self.base.get_cost(info_dict, action_candidates)
        predicted = info_dict["predicted_emb"][:, :, -1, :]
        goal = info_dict["goal_emb"][..., -1, :]
        if goal.ndim == 2:
            goal = goal.unsqueeze(1)
        predicted_unit = self.predicted_coordinate(predicted)
        goal_unit = self.goal_coordinate(goal)
        return (predicted_unit - goal_unit).square().sum(dim=-1)
