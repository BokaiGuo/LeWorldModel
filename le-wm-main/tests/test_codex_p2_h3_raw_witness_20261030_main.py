import copy
import hashlib

import pytest

from experiments.spectral_reachability import codex_p2_h3_raw_witness_20261030_main as witness


def fixture():
    events = []
    candidates = []
    for i in range(11):
        candidate = {"tag": str(i), "cost": float(i)}
        candidates.append(candidate)
        for packet in range(3):
            raw = bytes(40)
            item = {"kind": "actual_cuda_raw_packet_before_transition",
                    "candidate_ordinal": i, "packet": packet, "dtype": "<f4", "shape": [5, 2],
                    "raw_hex": raw.hex(), "raw_sha256": hashlib.sha256(raw).hexdigest()}
            events.extend(({"root": 0, "arm": "joint", "event": item},
                           {"root": 0, "arm": "joint",
                            "event": {"kind": "attempt", "resource": "packet"}}))
        events.append({"root": 0, "arm": "joint",
                       "event": {"kind": "candidate", "witness": candidate}})
    return events, [{"root": 0, "arm": "joint", "winner": candidates[4]}]


def test_winner_bytes_and_order():
    events, rows = fixture()
    result = witness.check_raw_events(events, rows)
    assert result[0]["winner_tag"] == "4"
    assert result[0]["raw_float32_hex"] == bytes(120).hex()


@pytest.mark.parametrize("failure", ("hash", "ordinal", "transition", "missing"))
def test_invalid_raw_witness_is_rejected(failure):
    events, rows = fixture()
    events = copy.deepcopy(events)
    if failure == "hash":
        events[0]["event"]["raw_sha256"] = "0" * 64
    elif failure == "ordinal":
        events[0]["event"]["candidate_ordinal"] = 1
    elif failure == "transition":
        events[1]["event"]["resource"] = "predict"
    else:
        events.pop(0)
    with pytest.raises(Exception):
        witness.check_raw_events(events, rows)
