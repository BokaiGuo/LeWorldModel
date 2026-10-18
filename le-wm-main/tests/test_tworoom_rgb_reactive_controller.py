import numpy as np

from experiments.audit_tworoom_rgb_reactive_controller import _action, _door_point, _red_centroid


def _frame(source=(40, 112), target=(180, 112)):
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    image[:, 109:115] = 0
    image[100:124, 109:115] = 255
    sx, sy = source
    tx, ty = target
    image[sy - 5:sy + 6, sx - 5:sx + 6] = [255, 0, 0]
    goal = np.full((224, 224, 3), 255, dtype=np.uint8)
    goal[:, 109:115] = 0
    goal[100:124, 109:115] = 255
    goal[ty - 5:ty + 6, tx - 5:tx + 6] = [255, 0, 0]
    return image, goal


def test_rgb_reactive_detects_marker_and_door() -> None:
    current, goal = _frame()
    assert np.allclose(_red_centroid(current), [40, 112], atol=1.0)
    door, vertical = _door_point(current)
    assert vertical
    assert np.allclose(door, [112, 112], atol=2.0)
    action, phase = _action(current, goal)
    assert phase == "door"
    assert action[0] > 0.8 and abs(float(action[1])) < 0.2
