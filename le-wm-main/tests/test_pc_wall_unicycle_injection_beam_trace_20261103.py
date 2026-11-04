import json

from experiments.spectral_reachability import pc_wall_unicycle_injection_beam_trace_20261103 as trace


def test_saved_injected_node_is_discarded_at_actual_cut():
    path = trace.ROOT / "results/pc_wall_unicycle_injection_beam_trace_20261103_DEV01.json"
    report = json.loads(path.read_text())
    for row in report["rows"]:
        node = row["exact_two_packet_node"]
        assert node["names"] == ["injected_0", "injected_1"]
        assert node["rank"] == 6 and not node["retained"]
        assert [snapshot["depth"] for snapshot in row["snapshots"]] == [2, 3, 4]
        assert sum(item["retained"] for item in row["snapshots"][0]["expanded"]) == 2


def test_one_recorded_trace_reproduces():
    saved = json.loads((trace.ROOT / "results/pc_wall_unicycle_injection_beam_trace_20261103_DEV01.json").read_text())
    source = json.loads(trace.inj.SOURCE.read_text())
    row = trace.run(source["rows"][0]["seed"], source["rows"][0]["selected"]["actions"])
    assert row["snapshots"] == saved["rows"][0]["snapshots"]
    assert row["selected_actions"] == saved["rows"][0]["selected_actions"]
