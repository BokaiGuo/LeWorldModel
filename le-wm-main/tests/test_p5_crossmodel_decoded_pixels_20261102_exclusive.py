import hashlib
import io

import pytest
from PIL import Image

from experiments.spectral_reachability import p5_crossmodel_decoded_pixels_20261102_exclusive as audit


def fixture(tmp_path):
    rows = []
    for i in range(2):
        records = []
        for j, (phase, step) in enumerate((("warmup", 5), ("plan", 0), ("plan", 5), ("plan", 10))):
            image = Image.new("RGB", (224, 224), (i, j, 17))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            raw = buf.getvalue()
            path = tmp_path / f"{i}_{j}.png"
            path.write_bytes(raw)
            records.append({"path": str(path), "file_sha256": hashlib.sha256(raw).hexdigest(),
                            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "phase": phase, "step": step, "condition": "off", "overlay_index": 0})
        rows.append({"index": i, "condition": "off", "history_rgb": records[:3],
                     "successor_rgb_identity_only": records[3]})
    return {"schema": "p5_crossmodel_panel_binding_20261030_codex_v1",
            "status": "saved_input_physical_identity_bound_not_scientific_admission",
            "P1_admission": False, "P5_admission": False, "rows": rows}


def test_valid_and_boundary(tmp_path):
    result = audit.audit(fixture(tmp_path), tmp_path, expected_rows=2)
    assert result["frame_references_checked"] == 8
    assert result["successor_frame_references_checked"] == 2
    assert result["P5_admission"] is False and result["new_neural_calls"] == 0


@pytest.mark.parametrize("mutation", ["pixel", "file", "order", "condition", "admission", "index", "outside"])
def test_fail_closed(tmp_path, mutation):
    binding = fixture(tmp_path)
    row = binding["rows"][0]
    if mutation == "pixel":
        row["successor_rgb_identity_only"]["pixel_sha256"] = "0" * 64
    elif mutation == "file":
        row["history_rgb"][0]["file_sha256"] = "0" * 64
    elif mutation == "order":
        row["history_rgb"].reverse()
    elif mutation == "condition":
        row["history_rgb"][0]["overlay_index"] = 1
    elif mutation == "admission":
        binding["P5_admission"] = True
    elif mutation == "index":
        row["index"] = 1
    else:
        row["history_rgb"][0]["path"] = str(tmp_path.parent / "outside.png")
    with pytest.raises(ValueError):
        audit.audit(binding, tmp_path, expected_rows=2)
