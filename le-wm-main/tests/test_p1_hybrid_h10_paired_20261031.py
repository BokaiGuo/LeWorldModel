import pytest

from experiments.spectral_reachability.p1_hybrid_h10_paired_20261031 import (
    KEYS, csv_metrics, difference,
)


def test_missing_endpoint_never_imputed():
    values = dict.fromkeys(KEYS, 1.0)
    assert difference(values, None) is None
    assert difference(None, values) is None
    assert difference(values, values) == dict.fromkeys(KEYS, 0.0)


def test_partial_or_nonfinite_metrics_rejected():
    row = {"x_" + key: "" for key in KEYS}
    assert csv_metrics(row, "x_") is None
    row["x_" + KEYS[0]] = "1"
    with pytest.raises(ValueError):
        csv_metrics(row, "x_")
    row = {"x_" + key: "nan" for key in KEYS}
    with pytest.raises(ValueError):
        csv_metrics(row, "x_")
