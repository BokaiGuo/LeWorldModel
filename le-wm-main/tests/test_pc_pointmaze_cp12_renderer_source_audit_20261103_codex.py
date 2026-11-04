import json

import numpy as np
import pytest

from experiments import pc_pointmaze_cp12_renderer_source_audit_20261103_codex as audit


def test_pixel_stats_counts_and_bounds():
    a = np.zeros((224, 224, 3), dtype=np.uint8)
    b = a.copy()
    b[3, 5, 1:] = [2, 7]
    result = audit.pixel_stats(a, b)
    assert not result["equal"]
    assert result["different_pixels"] == 1 and result["different_channels"] == 2
    assert result["bbox_xyxy"] == [5, 3, 5, 3]
    assert result["max_abs"] == 7
    assert audit.pixel_stats(a, a)["bbox_xyxy"] is None


def test_saved_cross_environment_audit():
    paths = [audit.ROOT / f"results/pc_pointmaze_cp12_renderer_source_audit_{label}_20261103_CODEX.json"
             for label in ("sandbox", "host")]
    if not all(x.exists() for x in paths):
        pytest.skip("audit receipts unavailable")
    sandbox, host = [json.loads(p.read_text()) for p in paths]
    assert sandbox["source_npz_sha256"] == host["source_npz_sha256"]
    assert sandbox["bridge_sha256"] == host["bridge_sha256"]
    assert sandbox["gl"]["cymj_sha256"] == host["gl"]["cymj_sha256"]
    assert sandbox["gl"]["runtime_library_sha256"] == host["gl"]["runtime_library_sha256"]
    assert sandbox["gl"]["gl_vendor"] == "Mesa"
    assert host["gl"]["gl_vendor"] == "NVIDIA Corporation"
    for doc, expected in ((sandbox, False), (host, True)):
        assert [r["root"] for r in doc["rows"]] == [8343, 8345, 8349, 8351]
        assert all(row["current_initial_repeat_exact"] for row in doc["rows"])
        assert all(f["equal"] == expected for row in doc["rows"] for f in row["frames"])
