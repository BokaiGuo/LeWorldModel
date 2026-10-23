from pathlib import Path

import pytest

from experiments.decision_benchmark.materialize_pointmaze_rgb_calibration_extension import (
    parse_roots,
)


def test_parse_roots_accepts_distinct_nonnegative_ids() -> None:
    assert parse_roots("1900, 1901", "render roots") == [1900, 1901]


@pytest.mark.parametrize("value", ("", "1900,1900", "-1,1900", "root"))
def test_parse_roots_rejects_invalid_ids(value: str) -> None:
    with pytest.raises(ValueError):
        parse_roots(value, "render roots")


def test_v8_launcher_requires_renderer_calibration_extension() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_pointmaze_htsc_prior_anchored_contrast_probe_v8_runtime_safety_cuda.sh"
    )
    text = launcher.read_text(encoding="utf-8")
    assert "materialize_pointmaze_rgb_calibration_extension" in text
