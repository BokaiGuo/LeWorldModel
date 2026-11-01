import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_rgb_chunk_approach_interface_20261102_exclusive as subject,
)


def image(pusher=(20, 100), object_x=80):
    result = np.zeros((224, 224, 3), dtype=np.uint8)
    x, y = pusher
    result[y - 2:y + 3, x - 2:x + 3] = subject.rgb.BLUE
    result[98:103, object_x:object_x + 5] = subject.rgb.BLOCK
    return result


def controller():
    return subject.RGBChunkApproach(initial_previous_action_is_zero=True)


def test_one_action_then_actual_frame_receipt_before_next_action():
    control = controller()
    first = control.observe(image(), frame_id=0)
    assert first["status"] == "proposal_only_wait_for_real_rgb"
    assert first["physical_execution_authorized"] is False
    assert first["P1_admission"] is False
    second = control.observe(image((23, 100)), frame_id=1,
                             executed_action_sha256=first["action_sha256"])
    assert second["chunk"] == 2
    assert second["receipt_required_before_next_proposal"] is True


@pytest.mark.parametrize("frame,receipt", [(0, "correct"), (1, None), (1, "wrong")])
def test_missing_or_stale_receipt_permanently_closes(frame, receipt):
    control = controller()
    first = control.observe(image(), frame_id=0)
    actual = first["action_sha256"] if receipt == "correct" else receipt
    with pytest.raises(ValueError):
        control.observe(image(), frame_id=frame, executed_action_sha256=actual)
    with pytest.raises(ValueError, match="closed"):
        control.observe(image(), frame_id=2, executed_action_sha256=first["action_sha256"])


def test_clipped_object_fails_closed():
    control = controller()
    pixels = image()
    pixels[100, 0] = subject.rgb.BLOCK
    with pytest.raises(ValueError, match="clipped"):
        control.observe(pixels, frame_id=0)
    with pytest.raises(ValueError, match="closed"):
        control.observe(image(), frame_id=1)


def test_fragmented_object_fails_closed():
    control = controller()
    pixels = image()
    pixels[120:123, 100:103] = subject.rgb.BLOCK
    with pytest.raises(ValueError, match="fragmented object"):
        control.observe(pixels, frame_id=0)
    with pytest.raises(ValueError, match="closed"):
        control.observe(image(), frame_id=1)


def test_near_object_stops_without_claiming_contact():
    control = controller()
    result = control.observe(image(pusher=(20, 100), object_x=27), frame_id=0)
    assert result["status"] == "rgb_near_object_no_contact_claim"
    assert "action" not in result


def test_initial_disk_outside_image_rejects():
    control = controller()
    with pytest.raises(ValueError, match="disk outside"):
        control.observe(image(pusher=(5, 100)), frame_id=0)


def test_budget_stops_without_sixth_action():
    control = controller()
    result = control.observe(image(), frame_id=0)
    for frame in range(1, subject.MAX_CHUNKS + 1):
        result = control.observe(image(), frame_id=frame,
                                 executed_action_sha256=result["action_sha256"])
    assert result["status"] == "chunk_budget_exhausted"
    assert "action" not in result


def test_prior_action_must_be_asserted():
    with pytest.raises(ValueError, match="zero prior"):
        subject.RGBChunkApproach(initial_previous_action_is_zero=False)


def test_predicted_segment_distance_catches_intermediate_collision():
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    pixels = np.array([[5.0, 1.0]])
    assert subject._distance_segment(start, end, pixels) == 1.0
