"""Synthetic physical source fixtures; no neural model or physical experiment."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import p1_coherent_own_suffix_samebank_20261101_codex as single


def test_actual_greedy_source_and_synthetic_cp5_fail_closed(tmp_path, monkeypatch):
    prefix, model_hash, index, _ = single.greedy_source()
    assert prefix.shape == (5, 2) and index == 3
    assert model_hash["original_state_dict"]
    target = json.loads(single.own.coherent.centered.adapter.TARGET.read_bytes())["rows"][0]
    pixels = np.zeros((224, 224, 3), np.uint8)
    pixels[20, 30] = (1, 2, 3)
    pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    rows = []
    for suffix in ("common", "zero"):
        path = tmp_path / (suffix + ".png")
        Image.fromarray(pixels).save(path)
        rows.append(dict(prefix=single.PREFIX, suffix=suffix, executed_raw=prefix.tolist(),
                         checkpoints=[dict(step=0, state=target["root_state"],
                                           pixel_sha256=target["history_pixel_sha256"][-1]),
                                      dict(step=5, state=[1., 2., 3., 4., 5., 6., 7.],
                                           png=str(path), file_sha256=single.sha(path),
                                           pixel_sha256=pixel_sha)]))
    path = tmp_path / "report.json"
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    report = dict(rows=rows, P1_admission=False, independent_confirmation=False,
                  neural_calls=0, protocol_sha256=single.sha(protocol),
                  sources={str(single.GREEDY): single.GREEDY_SHA})
    path.write_text(json.dumps(report))
    monkeypatch.setattr(single, "PHYSICAL", path)
    row, sources = single.load_physical(path, single.sha(path), prefix, target["root_state"],
                                        target["history_pixel_sha256"][-1])
    assert row["prefix"] == single.PREFIX and row["pixel_sha256"] == pixel_sha
    assert len(sources) == 5 and np.array_equal(row["prefix_raw"], prefix)
    rows[1]["executed_raw"][0][0] += .2
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="actual returned greedy prefix"):
        single.load_physical(path, single.sha(path), prefix, target["root_state"],
                             target["history_pixel_sha256"][-1])
    with pytest.raises(ValueError, match="physical report SHA"):
        single.load_physical(path, "0" * 64, prefix, target["root_state"],
                             target["history_pixel_sha256"][-1])


def test_missing_physical_report_fails_before_cuda(tmp_path, monkeypatch):
    missing = tmp_path / "missing" / "report.json"
    monkeypatch.setattr(single, "PHYSICAL", missing)
    with pytest.raises(ValueError, match="physical report SHA"):
        single.load_physical(missing, "0" * 64, np.zeros((5, 2), np.float32),
                             [0] * 7, "0" * 64)
