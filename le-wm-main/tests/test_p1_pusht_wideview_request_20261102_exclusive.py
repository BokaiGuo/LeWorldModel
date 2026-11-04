import numpy as np
import pytest

from experiments.spectral_reachability import p1_pusht_wideview_request_20261102_exclusive as target


def frame(clipped=False):
    a = np.full((224, 224, 3), 255, np.uint8)
    a[210:224 if clipped else 224, 30:50] = [120, 140, 160] if clipped else 255
    if not clipped:
        a[40:60, 30:50] = [120, 140, 160]
    a[100:110, 100:110] = [50, 100, 230]
    return a


def pair():
    meta = {"camera_id": "cam", "session_id": "session", "target_id": "target", "view_id": "view"}
    return {"goal": {**meta, "pixels": frame(), "capture_id": "1"},
            "current": {**meta, "pixels": frame(), "capture_id": "2"}}


def test_valid_alternate_does_not_authorize_or_clear_old_failure():
    result = target.inspect_alternate(frame(True), frame(), lambda _: pair())
    assert result["status"] == "blocked"
    assert result["reason"] == "calibration_and_identity_verification_required"
    assert not result["original_goal_guard"]["eligible"]
    assert result["alternate_guard"]["goal"]["eligible"]
    assert result["action"] is None and not result["selector_called"]


@pytest.mark.parametrize("kind", ["missing", "camera", "duplicate", "bad_frame", "exception"])
def test_unavailable_mismatch_and_error_closed(kind):
    def acquire(_):
        if kind == "missing":
            return None
        if kind == "exception":
            raise OSError("camera offline")
        p = pair()
        if kind == "camera":
            p["current"]["camera_id"] = "other"
        if kind == "duplicate":
            p["current"]["capture_id"] = "1"
        if kind == "bad_frame":
            p["goal"]["pixels"] = frame(True)
        return p

    result = target.inspect_alternate(frame(True), frame(), acquire)
    assert result["status"] == "blocked" and result["action"] is None
    assert result["reason"] != "calibration_and_identity_verification_required"


def test_existing_good_goal_does_not_acquire():
    result = target.inspect_alternate(frame(), frame(), lambda _: pytest.fail("unexpected request"))
    assert result["adapter_calls"] == 0


def test_known_development_source():
    report = target.dev305()
    assert report["decision"]["reason"] == "wider_view_unavailable"
    assert report["decision"]["adapter_calls"] == 1
    assert report["physical_steps"] == report["neural_calls"] == 0
    assert len(report["requests"]) == 1
