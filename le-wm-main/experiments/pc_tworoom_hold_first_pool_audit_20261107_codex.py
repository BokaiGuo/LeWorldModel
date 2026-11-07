"""Independent arithmetic and identity checks on sealed physical decomposition."""
import json

import numpy as np

from experiments import pc_tworoom_hold_first_pool_physical_20261107_codex as physical
from experiments.pc_input_permissions import file_hash

OUT = physical.run.ROOT / "results/pc_tworoom_hold_first_pool_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(physical.OUT.read_text())
    run = json.loads(physical.run.OUT.read_text())
    if (data["source_sha256"] != file_hash(physical.run.OUT)
            or data["protocol_sha256"] != file_hash(physical.PROTOCOL)
            or data["runner_sha256"] != file_hash(physical.__file__)):
        raise ValueError("source changed")
    matches, records = 0, 0
    for row, original in zip(data["rows"], run["rows"], strict=True):
        if row["root"] != original["root"] or row["row"] != original["row"]:
            raise ValueError("row mismatch")
        decisions = physical.common_first(original)
        parents, probes, tails = physical.run.search.timed.build(decisions["effect"]["candidate_seed"], None, 10)
        bank = np.concatenate((np.zeros_like(probes[:1]), probes, tails))
        distances = np.array([[entry["distance"][-1] for entry in group] for group in row["physical"]])
        expected = distances[:, :1] - distances[:, 1:3]
        np.testing.assert_allclose(expected, row["physical_distance_effects"], rtol=0, atol=0)
        if physical.sign_report(row["predicted_effects"], expected) != row["sign_agreement"]:
            raise ValueError("sign summary mismatch")
        useful = []
        for i, group in enumerate(row["physical"]):
            if len(group) != 5:
                raise ValueError("candidate count mismatch")
            ref_distances = group[0]["distance"][:10]
            ref_success = group[0]["success"][:10]
            if any(x["distance"][:10] != ref_distances or x["success"][:10] != ref_success for x in group):
                raise ValueError("prefix replay mismatch")
            if row["prefix_success"][i] != any(ref_success):
                raise ValueError("prefix success mismatch")
            if row["zero_max_drifts"][i] != group[0]["zero_drift"]:
                raise ValueError("reference drift mismatch")
            useful.append(not row["initial_success"] and not any(ref_success) and
                          any(any(group[j]["success"][10:]) for j in (3, 4)))
            for j, entry in enumerate(group):
                if len(entry["distance"]) != 25 or len(entry["success"]) != 25:
                    raise ValueError("trace length mismatch")
                if any(bool(s) != (float(d) < 16.) for d, s in zip(entry["distance"], entry["success"], strict=True)):
                    raise ValueError("physical success predicate mismatch")
                chain = np.concatenate((parents[i, :10], bank[j]))
                chain_hash = physical.run.prior.action_hash(chain)
                for arm, d in decisions.items():
                    if chain_hash == d["actions_sha256"]:
                        n = len(d["dispatched"])
                        np.testing.assert_array_equal(chain[:n], np.asarray(d["dispatched"], np.float32))
                        expected_trace = original["arms"][arm]["trace"][:n]
                        np.testing.assert_allclose(entry["distance"][:n], [x["distance"] for x in expected_trace],
                                                   rtol=0, atol=0)
                        matches += 1
                records += 1
        if useful != row["useful_prefixes_independent_tails"]:
            raise ValueError("useful label mismatch")
        for arm, d in decisions.items():
            expected = {"useful_retained": sum(useful[i] for i in d["retained"]),
                        "useful_total": sum(useful),
                        "retained_any_useful": any(useful[i] for i in d["retained"])}
            if row["recall"][arm] != expected:
                raise ValueError("recall mismatch")
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(physical.OUT), "records": records,
                   "original_chosen_chain_matches": matches,
                   "scope": "arithmetic/identity and shared-backend trace comparison; no independent physical replay"},
                  stream, indent=2)
    print(f"validated {records} candidate traces and {matches} original chosen-chain matches")


if __name__ == "__main__":
    main()
