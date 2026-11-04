"""Independent CPU audit of the sealed same-boundary PushT diagnostic."""
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_v3_dependency_seal as seal
from experiments.spectral_reachability import audit_pusht_d2p_packet_contract as phys

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/pc_v7_pusht_same_bank_physical_rank_dev_20261104_codex"
SOURCE = ROOT / "results/pc_v7_pusht_rgb_feedback_dev_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ARRAYS = PREFIX.with_name(PREFIX.name + "_arrays.npz")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def main():
    manifest, result = (json.loads(p.read_text()) for p in (MANIFEST, RESULT))
    seal.verify(manifest["seal"])
    assert result["manifest_sha256"] == seal.digest(MANIFEST)
    assert result["arrays_sha256"] == seal.digest(ARRAYS)
    decision = json.loads(SOURCE.with_name(SOURCE.name + "_response_decision2.json").read_text())
    execution = json.loads(SOURCE.with_name(SOURCE.name + "_response_execution.json").read_text())
    original = decision["result"]
    expected = [(f"bank/{x['name']}", x["prefix"], s)
                for x, s in zip(original["first_bank"], original["prefix_scores"], strict=True)]
    expected += [("selected", original["selected"]["chain"], original["selected"]["terminal_score"]),
                 ("carry", decision["carry_in"], original["carry_score"])]
    with np.load(ARRAYS, allow_pickle=False) as data:
        frames = data["frames"].copy()
    assert len(expected) == len(result["rows"]) == len(frames) == 11
    for row, frame, (name, actions, score) in zip(result["rows"], frames, expected, strict=True):
        assert (row["name"], row["model_score"], row["status"]) == (name, score, "completed")
        np.testing.assert_array_equal(np.asarray(row["actions"], np.float32), np.asarray(actions, np.float32))
        events = row["events"]
        assert len(events) == 60
        for i in range(30):
            attempt, returned = events[2*i:2*i+2]
            assert attempt["kind"] == "attempt" and returned["kind"] == "returned"
            assert attempt["step"] == returned["step"] == i + 1
            assert not returned["terminated"] and not returned["truncated"]
        np.testing.assert_array_equal(np.asarray([x["command"] for x in events[-10::2]], np.float32),
                                      np.asarray(actions, np.float32))
        np.testing.assert_array_equal(events[-1]["state"], row["state"])
        assert events[-1]["rgb_sha256"] == phys._array_hash(frame)
    selected = result["rows"][-2]
    np.testing.assert_allclose(selected["state"], execution["outcomes"][3]["state"], rtol=0, atol=1e-9)
    with np.load(SOURCE.with_name(SOURCE.name + "_response_context3.npz"), allow_pickle=False) as original_ctx:
        np.testing.assert_array_equal(frames[-2], original_ctx["frames"][-1])
    for field, key in (("observed_goal_latent_mse", "observed_goal_latent_mse"),
                       ("object_xy_distance", "xy_distance"), ("object_angle_distance", "angle_distance")):
        def val(row):
            return row[key] if field == "observed_goal_latent_mse" else row["pose"][key]
        pairs = [(a, b) for i, a in enumerate(result["rows"]) for b in result["rows"][i+1:]
                 if a["model_score"] != b["model_score"] and val(a) != val(b)]
        agree = sum(np.sign(a["model_score"] - b["model_score"]) == np.sign(val(a) - val(b))
                    for a, b in pairs)
        observed = result["ordering"][field]
        assert (len(pairs), agree) == (observed["valid_pairs"], observed["agree"])
        assert observed["fraction"] == agree / len(pairs)
    record = {"result_sha256": seal.digest(RESULT), "manifest_sha256": seal.digest(MANIFEST),
              "arrays_sha256": seal.digest(ARRAYS), "candidate_bindings": 11, "events_checked": 660,
              "selected_state_and_rgb_match_original": True, "ranking_recomputed": True,
              "limitation": "CPU record audit; no independent model re-encoding or physical replay"}
    with AUDIT.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
