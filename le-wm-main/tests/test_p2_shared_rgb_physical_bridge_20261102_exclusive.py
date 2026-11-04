from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p2_shared_rgb_physical_bridge_20261102_exclusive as bridge


@pytest.fixture(scope="module")
def real_inputs():
    bundle, _, _, actions = bridge.launch.wire.load_capture(
        bridge.launch.CAPTURE / "capture.json", bridge.launch.CAPTURE_SHA)
    pair = json.loads((bridge.RUN / "pair.json").read_bytes())
    receipt = json.loads(bridge.capture.RECEIPT.read_bytes())
    events = [bridge.physical.native.decode(x) for x in
              (bridge.capture.RUN / "journal.jsonl").read_bytes().splitlines()]
    return bundle, actions, pair, receipt, events


def test_real_bridge_and_byte_mutations(real_inputs):
    bundle, actions, pair, receipt, events = real_inputs
    rows = bridge.action_and_source_binding(bundle, actions, pair, receipt, events)
    assert [row["arm"] for row in rows] == ["joint_multistart", "prefix_chain"]
    changed = deepcopy(pair)
    changed["branches"][0]["winner"]["raw_float32_hex"] = "00" * 80
    with pytest.raises(ValueError, match="winner action"):
        bridge.action_and_source_binding(bundle, actions, changed, receipt, events)
    changed = deepcopy(bundle)
    changed["frames"][0]["pixel_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="feedback RGB"):
        bridge.action_and_source_binding(changed, actions, pair, receipt, events)
    changed = deepcopy(receipt)
    changed["rows"][1]["feedback_guard"]["no_state_restore"] = False
    with pytest.raises(ValueError, match="prior guard"):
        bridge.action_and_source_binding(bundle, actions, pair, changed, events)


def test_contract_sha_and_old_result_audit():
    path = bridge.ROOT / "results/p2_shared_rgb_physical_bridge_20261102_exclusive_CONTRACT_B0_01.json"
    sha = "c0397aeed4754a29e93beba303a8af8157c71f525c414847a66f40da0b6dbdc2"
    contract = bridge.verify_contract(path, sha)
    receipt = bridge.audit_output(contract, bridge.capture.RUN,
                                  bridge.capture.REPORT_SHA, bridge.capture.COMPLETION_SHA)
    assert receipt["status"] == "verified_known_root_physical_not_admitted"
    assert receipt["P2_admission"] is False and receipt["independent_confirmation"] is False
    with pytest.raises(ValueError, match="bridge SHA"):
        bridge.verify_contract(path, "0" * 64)
