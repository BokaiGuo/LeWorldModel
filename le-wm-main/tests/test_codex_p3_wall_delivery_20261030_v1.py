import json
import sys
import time

import pytest

from experiments.spectral_reachability import codex_p3_wall_delivery_20261030_v1 as wall


def test_deadline_success_and_timeout():
    completed = wall.run_deadline([sys.executable, "-c", "print('ok')"], 3)
    assert completed["exit_code"] == 0
    assert not completed["deadline_exceeded"]
    assert completed["stdout"].strip() == "ok"
    start = time.monotonic()
    expired = wall.run_deadline([sys.executable, "-c", "import time; time.sleep(10)"], .15)
    assert expired["deadline_exceeded"]
    assert expired["exit_code"] != 0
    assert time.monotonic() - start < 5


def test_fixed_slots_and_new_paths(tmp_path):
    assert wall.SECONDS == 30
    assert wall.SLOTS == ((20, "uniform"), (20, "signal"), (20, "reverse"),
                          (30, "uniform"), (30, "signal"), (30, "reverse"))
    with pytest.raises(ValueError):
        wall.exclusive(tmp_path / (wall.NAME + "_RUN_01"), "_RUN_")


def test_prepare_hash_rejects_tampering(tmp_path):
    plan = tmp_path / "plan"
    plan.write_text(json.dumps({"x": 1}))
    with pytest.raises(ValueError):
        wall.material(plan, "0" * 64)
