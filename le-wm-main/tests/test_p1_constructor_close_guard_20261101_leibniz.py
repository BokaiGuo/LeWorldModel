import hashlib
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_constructor_close_guard_20261101_leibniz as a

p, g = a.physical, a.physical.total


class Store:
    def check(self):
        pass

    def emit(self, *args, **kwargs):
        pass


def make_guards(tmp_path, close_cap=2):
    paths = [Path(m.__file__).resolve() for m in (a, p, g, p.legacy, p.moving)]
    contract = {"schema": g.NAME + ".contract.v1", "run_id": "close-test", "arm": "cem",
                "caps": dict.fromkeys(g.AXES, 20), "identity": dict.fromkeys(g.IDENTITY, "a" * 64),
                "source_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                "shared_initialization": "full_standalone_charge_each_arm", "hook_modules": g.HOOKS,
                "scope": "per_root_per_arm_start_before_load"}
    total_guard = g.Guard(tmp_path / "total.jsonl", contract, g.digest(contract))
    caps = dict.fromkeys(p.RESOURCES, 20)
    caps["close"] = close_cap
    physical_contract = {"schema": p.NAME + ".contract.v1", "total_contract_sha256": g.digest(contract), "caps": caps}
    guard = p.PhysicalGuard(tmp_path / "physical.jsonl", total_guard, caps, g.digest(physical_contract))
    return total_guard, guard


@pytest.mark.parametrize("close_fails", [False, True])
def test_actual_constructor_failure_invokes_counted_close(tmp_path, close_fails):
    total, guard = make_guards(tmp_path)
    meter, closed = p.moving.Meter(Store()), []
    class Base:
        def __init__(self):
            raise RuntimeError("constructor failed")
        def close(self):
            closed.append(True)
            if close_fails:
                raise OSError("close failed")
    with a.guarded_class_factory(guard, meter) as factory:
        cls = factory(Base)
        with pytest.raises(RuntimeError, match="constructor failed"):
            meter.call("constructor", cls)
    assert closed == [True]
    assert guard.charged["constructor"] == guard.charged["close"] == 1
    assert bool(meter.secondary) == close_fails
    receipt = guard.close()
    assert p.verify_journal(receipt["path"], receipt["sha256"], g.digest(guard.contract))["status"] == "failed_or_blocked"
    assert total.close()["status"] == "failed_or_blocked"


def test_explicit_host_close_and_direct_close_each_charged_once(tmp_path):
    total, guard = make_guards(tmp_path)
    meter, closed = p.moving.Meter(Store()), []
    class Base:
        def close(self):
            closed.append(True)
    with a.guarded_class_factory(guard, meter) as factory:
        cls = factory(Base)
        first = meter.call("constructor", cls)
        second = meter.call("constructor", cls)
        meter.call("close", first.close)
        second.close()
    assert closed == [True, True] and guard.charged["close"] == 2
    assert guard.close()["status"] == total.close()["status"] == "closed_unverified"


def test_cleanup_cap_blocks_and_preserves_constructor_error(tmp_path):
    total, guard = make_guards(tmp_path, close_cap=0)
    meter, closed = p.moving.Meter(Store()), []
    class Base:
        def __init__(self):
            raise RuntimeError("constructor failed")
        def close(self):
            closed.append(True)
    with a.guarded_class_factory(guard, meter) as factory:
        with pytest.raises(RuntimeError, match="constructor failed"):
            meter.call("constructor", factory(Base))
    assert not closed and guard.charged["close"] == 0
    assert meter.secondary and "cap" in meter.secondary[0]["message"]
    assert guard.close()["status"] == total.close()["status"] == "failed_or_blocked"
