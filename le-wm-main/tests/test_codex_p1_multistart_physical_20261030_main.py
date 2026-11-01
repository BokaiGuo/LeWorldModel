import hashlib

from experiments.spectral_reachability import codex_p1_multistart_physical_20261030_main as physical
from experiments.spectral_reachability import codex_p1_multistart_physical_saved_20261030_main as saved


def test_fixed_winner_roster_hashes():
    prepared = physical.build()
    assert len(prepared["roster"]) == 24
    for row in prepared["roster"]:
        assert hashlib.sha256(bytes.fromhex(row["raw10_float32_hex"])).hexdigest() == row["raw10_sha256"]


def test_consumed_stops_are_not_filled():
    result = saved.consume()
    assert len(result["rows"]) == 24
    stopped = [row for row in result["rows"] if row["status"] == "native_stopped"]
    assert len(stopped) == 1
    assert stopped[0]["completed_endpoint"] is None
    assert stopped[0]["endpoint_metrics"] is None
