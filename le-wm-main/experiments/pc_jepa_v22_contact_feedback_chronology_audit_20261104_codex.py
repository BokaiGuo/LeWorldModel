"""Check isolation-input schema, common proposal bank and journal ordering."""
import json

import numpy as np

from experiments.pc_jepa_v22_contact_feedback_audit_20261104_codex import OUT, journal_events, sha


def main():
    audit = json.loads((OUT / "audit.json").read_text())
    replay = json.loads((OUT / "physical_replay.json").read_text())
    if (audit["status"] != "saved_data_audit_passed" or replay["status"] != "exact"
            or replay["audit_sha256"] != sha(OUT / "audit.json")):
        raise ValueError("source audits not bound")
    events = journal_events(OUT / "journal.jsonl")
    rows, banks = [], {}
    allowed = {"rgb_history", "goal_rgb", "normalized_proprio", "normalized_history_packets",
               "action_mean", "action_std", "carry"}
    for arm in ("response", "cem"):
        for step in range(3):
            path = OUT / f"{arm}_{step}"
            decision = json.loads((path / "output/decision.json").read_text())
            with np.load(path / "input/context.npz", allow_pickle=False) as context:
                if set(context.files) != allowed:
                    raise ValueError("unexpected planner input")
                carry = context["carry"].copy()
            batch = np.asarray(decision["evaluations"][0]["actions"], np.float32)
            if step and not np.array_equal(batch[1], carry):
                raise ValueError("carry was not re-evaluated")
            if step == 0:
                banks[arm] = batch
            intent = [i for i, event in enumerate(events) if event.get("phase") == "plan_intent"
                      and event.get("arm") == arm and event.get("step") == step]
            sealed = [i for i, event in enumerate(events) if event.get("phase") == "decision_sealed"
                      and event.get("arm") == arm and event.get("step") == step]
            physical = [i for i, event in enumerate(events) if event.get("phase") == "intent"
                        and event.get("role") == arm and event.get("operation") == "step"
                        and event.get("details", {}).get("frame") in range(11 + 5 * step, 16 + 5 * step)]
            if (len(intent) != 1 or len(sealed) != 1 or len(physical) != 5
                    or not intent[0] < sealed[0] < min(physical)):
                raise ValueError("planning/dispatch ordering mismatch")
            rows.append({"arm": arm, "step": step, "plan_intent": intent[0],
                         "decision_sealed": sealed[0], "first_dispatch": min(physical)})
    np.testing.assert_array_equal(banks["response"], banks["cem"])
    result = {"status": "passed", "audit_sha256": sha(OUT / "audit.json"),
              "physical_replay_sha256": sha(OUT / "physical_replay.json"), "rows": rows,
              "scope": "saved-data chronology and declared input schema; not full information-flow proof"}
    with (OUT / "chronology_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
