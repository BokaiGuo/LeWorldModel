"""Read-only candidate coverage/ranking diagnostic on sealed prior results."""
import json

import numpy as np

from experiments import pc_spectral_bank_rgb_dev_20261103_codex as base
from experiments import pc_spectral_bank_rgb_physical_20261103_codex as physical
from experiments import pc_spectral_followup_20261103_codex as followup

OUTPUT = followup.PREFIX.with_name(followup.PREFIX.name+"_bank_analysis.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = base.old.prior.sha
    source = json.loads(base.OUTPUT.read_text())
    outcome = json.loads(physical.OUTPUT.read_text())
    if outcome["decision_sha256"] != sha(base.OUTPUT):
        raise ValueError("physical result is not bound to source")
    rows = []
    for root in source["rows"]:
        programs = root["plan"]["programs"]
        candidates = [x for x in outcome["rows"] if x["root"] == root["root"] and x["kind"] == "candidate"]
        jac = np.asarray(root["spectrum"]["jacobian"])
        for item in programs:
            pred = np.asarray(item["prefix"]).reshape(-1)
            others = [(x["name"], np.linalg.norm(pred-np.asarray(x["prefix"]).reshape(-1)),
                       np.linalg.norm(jac @ (pred-np.asarray(x["prefix"]).reshape(-1))))
                      for x in programs if x["source"] != "rgb_model" and x["name"] != item["name"]]
            nearest = min(others, key=lambda x: x[1])
            physical_item = next(x for x in candidates if x["name"] == item["name"])
            rows.append({"root": root["root"], "name": item["name"], "source": item["source"],
                         "nearest_nonspectral_name": nearest[0], "action_l2": float(nearest[1]),
                         "linear_response_l2": float(nearest[2]), "model_terminal": item["terminal_score"],
                         "model_rank": 1+sum(x["terminal_score"] < item["terminal_score"] for x in programs),
                         "physical_terminal": physical_item["terminal_distance"],
                         "physical_rank": 1+sum(x["terminal_distance"] < physical_item["terminal_distance"] for x in candidates),
                         "initial_distance": physical_item["initial_distance"],
                         "prefix_distance": physical_item["prefix_distance"]})
    base.old.write_json(OUTPUT, {"rows": rows, "source_sha256": sha(base.OUTPUT),
                                "physical_sha256": sha(physical.OUTPUT), "protocol_sha256": sha(followup.PROTOCOL),
                                "sample_budget_evidence": "source loop; individual samples not retained"})
    print(f"saved {len(rows)} candidate comparisons", flush=True)


if __name__ == "__main__":
    main()
