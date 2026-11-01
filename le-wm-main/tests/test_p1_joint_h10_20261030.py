from experiments.spectral_reachability import p1_joint_h10_20261030 as experiment


def test_sealed_roster_matches_joint_action_and_three_roles():
    prepared = experiment.build()
    physical = experiment.joint.runner.native.read(
        experiment.PHYS / "report.json", cap=64 * 1024**2)
    binding = physical["execution_binding"]
    plan, _ = experiment.joint.verify(binding["prepared"], binding["prepare_sha256"], experiment.JOIN_SHA)
    assert len(prepared["anchors"]) == 24
    assert len(prepared["probe_banks"]) == 8
    for i in range(8):
        anchors = prepared["anchors"][3*i:3*i+3]
        assert [a["role"] for a in anchors] == ["selected", "zero", "greedy"]
        raw = bytes.fromhex(plan["plans"][2*i+1]["raw15_float32_hex"])
        assert anchors[0]["prefix5_float32_hex"] == raw[:40].hex()
        assert anchors[1]["prefix5_float32_hex"] == bytes(40).hex()
        bank = prepared["probe_banks"][i]
        assert len(bank) == 42
        assert bank[0]["raw10_float32_hex"] == bytes(80).hex()
        assert bank[-1]["raw10_float32_hex"] == raw[40:].hex()
