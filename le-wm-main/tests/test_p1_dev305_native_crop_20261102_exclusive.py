import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev305_native_crop_20261102_exclusive as crop


def frame():
    pixels = np.full((512, 512, 3), 255, np.uint8)
    pixels[100:140, 100:140] = [120, 140, 160]
    pixels[200:220, 200:220] = [50, 100, 230]
    return pixels


def test_complete_roi_can_pass():
    _, result = crop.prepare_crop(frame())
    assert result["eligible"]
    assert all(x["contained"] for x in result["coverage"].values())


def test_mask_outside_crop_blocks():
    pixels = frame()
    pixels[100:140, 100:140] = 255
    pixels[40:80, 100:140] = [120, 140, 160]
    _, result = crop.prepare_crop(pixels)
    assert not result["eligible"]
    assert result["coverage"]["object"]["outside"] > 0


def test_empty_mask_and_bad_input_block():
    pixels = frame()
    pixels[100:140, 100:140] = 255
    _, result = crop.prepare_crop(pixels)
    assert not result["eligible"]
    with pytest.raises(ValueError):
        crop.prepare_crop(pixels.astype(float))


def test_failure_creates_report_and_no_overwrite(tmp_path):
    path = tmp_path / "new"
    def broken(_):
        raise RuntimeError("renderer unavailable")
    report = crop.run(path, broken, "", "")
    assert report["status"] == "engineering_failure"
    assert report["action"] is None and not report["selector_called"]
    assert (path / "report.json").exists()
    with pytest.raises(FileExistsError):
        crop.run(path, broken, "", "")
