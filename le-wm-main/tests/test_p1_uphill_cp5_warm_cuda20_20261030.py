import hashlib
import json

from experiments.spectral_reachability import p1_uphill_cp5_warm_cuda20_20261030 as warm


def test_all_three_roles_share_the_exact_fixed_tail():
    starts = warm.initializers()
    prepared = json.loads(warm.bundle.PREP.read_bytes())
    assert len(starts) == 24
    for group in range(8):
        tail = next(p for p in prepared["probe_banks"][group] if p["name"] == "original_selected_tail")
        assert starts[3 * group] == starts[3 * group + 1] == starts[3 * group + 2]
        assert hashlib.sha256(starts[3 * group]).hexdigest() == tail["raw10_sha256"]


def test_contract_names_actual_initializer():
    warm.configure()
    row = json.loads(warm.bundle.OUT.read_bytes())["rows"][0]
    _, contract = warm.planner.make_contract(row, warm.bindings())
    expected = warm.planner.spec.digest({"raw10_float32_hex": warm.initializers()[0].hex()})
    assert {arm["initializer_sha256"] for arm in contract["arms"].values()} == {expected}
