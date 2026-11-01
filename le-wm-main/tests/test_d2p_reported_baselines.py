"""All numeric values below are test fixtures, not literature results."""
from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import d2p_reported_baselines as b


def record():
    return dict(paper="fixture-id", title="Synthetic test citation", version="v1",
                source_url="https://example.org/paper", table_or_page="Table 1",
                metric="success", value=0.5, unit="fraction", direction="higher_is_better",
                env="fixture-env-v1", task="fixture-task", dataset_split="test",
                checkpoint="fixture-checkpoint", observations={"input": "RGB"},
                action_space={"dim": 2, "units": "normalized"}, horizon=4,
                budget={"predict_samples": 32}, evaluation_protocol="fixture-protocol-v1",
                n_seeds=3, uncertainty={"kind": "standard_deviation", "value": 0.1,
                                         "unit": "fraction"}, evidence_kind="paper_reported")


def reference():
    row = record()
    return {key: deepcopy(row[key]) for key in b.PROTOCOL_FIELDS}


def test_empty_registry_and_roundtrip_no_aliasing():
    registry = b.ReportedBaselineRegistry()
    with pytest.raises(KeyError):
        registry.get("not-prefilled")
    row = record()
    registry.register("fixture", row)
    row["budget"]["predict_samples"] = 999
    fetched = registry.get("fixture")
    assert fetched["budget"]["predict_samples"] == 32
    fetched["budget"]["predict_samples"] = 999
    result = registry.compare("fixture", reference=reference())
    assert result["classification"] == "direct_comparable"
    assert result["local_reproduction"] is False
    assert result["paired_inference_allowed"] is False
    json.dumps(registry.get("fixture"), allow_nan=False)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("fixture", record())


@pytest.mark.parametrize("field", b.PROTOCOL_FIELDS)
def test_unknown_in_each_protocol_field_precludes_direct(field):
    row = record()
    row[field] = "unknown"
    result = b.classify(row, reference=reference())
    assert result["classification"] == "insufficient_metadata"
    assert field in result["unknown_fields"]
    ref = reference()
    ref[field] = "unknown"
    assert b.classify(record(), reference=ref)["classification"] == "insufficient_metadata"


@pytest.mark.parametrize("field", ["paper", "title", "version", "source_url", "table_or_page", "value"])
def test_unknown_citation_or_value_is_explicit(field):
    row = record()
    row[field] = "unknown"
    assert b.validate_record(row)[field] == "unknown"
    assert b.classify(row, reference=reference())["classification"] == "insufficient_metadata"


@pytest.mark.parametrize("field,value", [
    ("env", "fixture-env-v2"), ("checkpoint", "different"),
    ("dataset_split", "validation"), ("horizon", 8), ("n_seeds", 4),
    ("budget", {"predict_samples": 64}), ("unit", "percent"),
    ("direction", "lower_is_better"), ("evaluation_protocol", "other"),
    ("observations", {"input": "state"}), ("action_space", {"dim": 3}),
    ("uncertainty", {"kind": "standard_error", "value": 0.1}),
])
def test_known_protocol_mismatch_context_only(field, value):
    row = record()
    row[field] = value
    result = b.classify(row, reference=reference())
    assert result["classification"] == "context_only"
    assert field in result["mismatched_fields"]


def test_nested_unknown_and_known_mismatch_precedence():
    row = record()
    row["budget"] = {"predict_samples": "unknown"}
    assert b.classify(row, reference=reference())["classification"] == "insufficient_metadata"
    row["env"] = "different"
    result = b.classify(row, reference=reference())
    assert result["classification"] == "context_only"
    assert result["unknown_fields"] == ["budget"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, None, "0.5"])
def test_invalid_value(value):
    row = record()
    row["value"] = value
    with pytest.raises(ValueError):
        b.validate_record(row)


@pytest.mark.parametrize("field,value", [
    ("source_url", "file:///paper.pdf"), ("source_url", "https:///paper"),
    ("source_url", "https://user:password@example.org/paper"),
    ("source_url", "https://example.org/a b"), ("table_or_page", ""),
    ("evidence_kind", "local_reproduced"), ("evidence_kind", "unknown"),
    ("n_seeds", 0), ("n_seeds", 2.5), ("horizon", True),
    ("direction", "maximize"), ("budget", {"calls": float("nan")}),
    ("uncertainty", {"value": float("inf")}), ("checkpoint", None),
])
def test_invalid_metadata(field, value):
    row = record()
    row[field] = value
    with pytest.raises(ValueError):
        b.validate_record(row)


def test_missing_fields_reference_and_extra_fields_rejected():
    row = record()
    del row["version"]
    with pytest.raises(ValueError):
        b.validate_record(row)
    with pytest.raises(ValueError):
        b.validate_record({**record(), "local_reproduction": True})
    with pytest.raises(ValueError, match="reference"):
        b.classify(record(), reference={"env": "fixture-env-v1"})
    with pytest.raises(TypeError):
        b.classify(record())


def test_no_cross_protocol_means_or_paired_intervals_even_if_matched():
    registry = b.ReportedBaselineRegistry()
    registry.register("a", record())
    row = record()
    row["horizon"] = 8
    registry.register("b", row)
    for ids in (["a"], ["a", "b"]):
        for operation in ("mean", "paired_ci", "pool", "local_reproduction"):
            with pytest.raises(ValueError, match="cannot be pooled"):
                registry.require_operation(ids, reference=reference(), operation=operation)
    results = registry.require_operation(["a", "b"], reference=reference(),
                                         operation="report_individually")
    assert results["a"]["classification"] == "direct_comparable"
    assert results["b"]["classification"] == "context_only"
