"""Paper-reported result registry, never local reproduction evidence.

No literature values are bundled. Unknown metadata must be the literal
``unknown``. Protocol equality is literal JSON equality, not an inferred
equivalence or a unit conversion. No aggregate statistics are computed.
"""
from __future__ import annotations

from copy import deepcopy
import math
from urllib.parse import urlsplit

UNKNOWN = "unknown"
PROTOCOL_FIELDS = (
    "metric", "unit", "direction", "env", "task", "dataset_split", "checkpoint",
    "observations", "action_space", "horizon", "budget", "evaluation_protocol",
    "n_seeds", "uncertainty",
)
FIELDS = ("paper", "title", "version", "source_url", "table_or_page", "value",
          "evidence_kind", *PROTOCOL_FIELDS)


def _json_value(value):
    if isinstance(value, str):
        if not value.strip() or value != value.strip():
            raise ValueError("empty or whitespace-padded metadata forbidden; use unknown")
    elif isinstance(value, bool) or value is None:
        raise ValueError("null/boolean metadata forbidden; use unknown")
    elif isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError("finite metadata required")
    elif isinstance(value, dict):
        if not value or any(not isinstance(k, str) or not k.strip() for k in value):
            raise ValueError("nonempty string-key JSON objects required")
        for item in value.values():
            _json_value(item)
    elif isinstance(value, list):
        if not value:
            raise ValueError("empty metadata lists forbidden; use unknown")
        for item in value:
            _json_value(item)
    else:
        raise ValueError("JSON metadata required")


def _unknown(value):
    if isinstance(value, dict):
        return any(_unknown(v) for v in value.values())
    if isinstance(value, list):
        return any(_unknown(v) for v in value)
    return value == UNKNOWN


def _validate_protocol(protocol):
    if not isinstance(protocol, dict) or set(protocol) != set(PROTOCOL_FIELDS):
        raise ValueError("explicit reference must contain exactly all protocol fields")
    _json_value(protocol)
    if protocol["direction"] not in (UNKNOWN, "higher_is_better", "lower_is_better"):
        raise ValueError("invalid metric direction")
    seeds = protocol["n_seeds"]
    if seeds != UNKNOWN and (type(seeds) is not int or seeds < 1):
        raise ValueError("n_seeds must be a positive integer or unknown")
    horizon = protocol["horizon"]
    if horizon != UNKNOWN and (type(horizon) is not int or horizon < 1):
        raise ValueError("horizon must be a positive integer or unknown")


def validate_record(record):
    """Validate and return a detached JSON-compatible paper result record.

    Missing citation fields may be explicitly unknown, but then comparability
    cannot be direct. Known URLs must be absolute HTTP(S) URLs.
    """
    if not isinstance(record, dict) or set(record) != set(FIELDS):
        raise ValueError("record must contain exactly all required metadata fields")
    _json_value(record)
    if record["evidence_kind"] != "paper_reported":
        raise ValueError("evidence_kind must be paper_reported, never local reproduction")
    for key in ("paper", "title", "version", "source_url", "table_or_page"):
        if not isinstance(record[key], str):
            raise ValueError(f"{key} must be a string or unknown")
    url = record["source_url"]
    if url != UNKNOWN:
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or any(c.isspace() for c in url)):
            raise ValueError("absolute HTTP(S) source_url required")
    value = record["value"]
    if value != UNKNOWN and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError("value must be finite numeric or unknown")
    _validate_protocol({k: record[k] for k in PROTOCOL_FIELDS})
    return deepcopy(record)


def classify(record, *, reference):
    """Compare against an explicit complete reference protocol, without pooling.

    Known protocol mismatches take precedence over unknown fields: a proven
    mismatch already excludes direct comparison. Unknowns otherwise preclude
    direct comparison. Seed count and uncertainty definitions are conservative
    equality requirements, not claims of statistical pairing.
    """
    record = validate_record(record)
    _validate_protocol(reference)
    mismatches, unknowns = [], []
    for key in PROTOCOL_FIELDS:
        if _unknown(record[key]) or _unknown(reference[key]):
            unknowns.append(key)
        elif record[key] != reference[key]:
            mismatches.append(key)
    for key in ("paper", "title", "version", "source_url", "table_or_page", "value"):
        if _unknown(record[key]):
            unknowns.append(key)
    category = ("context_only" if mismatches else
                "insufficient_metadata" if unknowns else "direct_comparable")
    return {"classification": category, "mismatched_fields": mismatches,
            "unknown_fields": unknowns, "evidence_kind": "paper_reported",
            "local_reproduction": False, "paired_inference_allowed": False,
            "scope": "metadata comparability only; no empirical reproduction or statistical pooling"}


class ReportedBaselineRegistry:
    """In-memory, explicit-ID registry; no overwrite, inferred values, or I/O."""

    def __init__(self):
        self._records = {}

    def register(self, record_id, record):
        if not isinstance(record_id, str) or not record_id.strip() or record_id != record_id.strip():
            raise ValueError("explicit nonempty record ID required")
        if record_id in self._records:
            raise ValueError("duplicate record ID; overwrite forbidden")
        self._records[record_id] = validate_record(record)

    def get(self, record_id):
        return deepcopy(self._records[record_id])

    def compare(self, record_id, *, reference):
        return classify(self.get(record_id), reference=reference)

    def require_operation(self, record_ids, *, reference, operation):
        """Only individual reporting is allowed; no pooled means/paired CIs.

        Even identical protocol summaries lack shared trial/seed identities.
        The registry therefore forbids all pooling rather than manufacture
        independent or paired uncertainty from paper-level summary values.
        """
        if operation != "report_individually":
            raise ValueError("paper-reported summaries cannot be pooled or used for paired intervals")
        ids = list(record_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("nonempty unique record IDs required")
        return {key: self.compare(key, reference=reference) for key in ids}
