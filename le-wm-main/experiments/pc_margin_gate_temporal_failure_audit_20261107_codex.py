"""Audit actual success timing where the frozen margin gate changes decisions."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_effect_margin_gate_20261107_codex as source
from experiments.pc_input_permissions import file_hash

PROTOCOL = source.source.src.prior.ROOT.parent / "docs/PC_MARGIN_GATE_TEMPORAL_FAILURE_AUDIT_PROTOCOL_20261107_CODEX.md"
OUT = source.source.src.prior.ROOT / "results/pc_margin_gate_temporal_failure_audit_20261107_codex.json"


def classify(direct, gated, direct_term, gated_term):
    direct, gated = np.asarray(direct), np.asarray(gated)
    direct_term, gated_term = np.asarray(direct_term), np.asarray(gated_term)
    if direct.shape != (8, 8) or gated.shape != (8, 8) or direct_term.shape != (8, 8, 25) or gated_term.shape != (8, 8, 25):
        raise ValueError("shape mismatch")
    if any(x.dtype != np.bool_ for x in (direct, gated, direct_term, gated_term)):
        raise ValueError("boolean arrays required")
    changed = direct != gated
    direct_ever, gated_ever = direct_term.any(-1), gated_term.any(-1)
    if np.any(~changed & (direct_ever != gated_ever)):
        raise ValueError("unchanged choice differs in success")
    result = {"changed": int(changed.sum()), "direct_only": int((changed & direct_ever & ~gated_ever).sum()),
              "gated_only": int((changed & gated_ever & ~direct_ever).sum()),
              "both": int((changed & direct_ever & gated_ever).sum()),
              "neither": int((changed & ~direct_ever & ~gated_ever).sum())}
    direct_wins = changed & direct_ever & ~gated_ever
    gate_wins = changed & gated_ever & ~direct_ever
    for label, mask, term in (("direct_only", direct_wins, direct_term),
                              ("gated_only", gate_wins, gated_term)):
        selected = term[mask]
        first = np.argmax(selected, axis=-1) + 1
        result[label + "_first_steps"] = first.tolist()
        result[label + "_success_lost_by_end"] = int((~selected[:, -1]).sum())
        result[label + "_first_in_prefix"] = int((first <= 10).sum())
    return result


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, old = json.loads(source.PLAN.read_text()), json.loads(source.RESULT.read_text())
    for actual, path in ((old["plan_sha256"], source.PLAN),
                         (old["physical_sha256"], source.source.src.PHYS),
                         (old["runner_sha256"], source.__file__),
                         (plan["prediction_sha256"], source.source.PRED),
                         (plan["protocol_sha256"], source.PROTOCOL),
                         (plan["runner_sha256"], source.__file__)):
        if actual != file_hash(path):
            raise ValueError(f"source mismatch {path}")
    rows = []
    with np.load(source.source.src.PHYS, allow_pickle=False) as physical:
        for p, r in zip(plan["rows"], old["rows"], strict=True):
            root = p["root"]
            if root != r["root"]:
                raise ValueError("root mismatch")
            choices = {name: np.asarray(p["choose_probe"][name], dtype=bool) for name in ("direct", "gated")}
            terms = physical[f"r{root}_term"]
            distance = physical[f"r{root}_distance"][:, :, -1]
            selected_terms = {}
            for name in choices:
                mask = choices[name]
                selected_terms[name] = np.where(mask[..., None], terms[:, 1:], terms[:, :1])
                actual = source.outcome(distance, terms, mask)
                if (not np.array_equal(actual["ever_success"], np.asarray(r["arms"][name]["success_per_pair"]))
                        or not np.array_equal(actual["distance"], np.asarray(r["arms"][name]["distance_per_pair"]))):
                    raise ValueError("prior outcome does not reconstruct")
            result = classify(choices["direct"], choices["gated"], selected_terms["direct"], selected_terms["gated"])
            result["root"] = root
            rows.append(result)
    fields = ("changed", "direct_only", "gated_only", "both", "neither",
              "direct_only_success_lost_by_end", "gated_only_success_lost_by_end",
              "direct_only_first_in_prefix", "gated_only_first_in_prefix")
    summary = {field: sum(r[field] for r in rows) for field in fields}
    summary["direct_only_first_steps"] = [x for r in rows for x in r["direct_only_first_steps"]]
    summary["gated_only_first_steps"] = [x for r in rows for x in r["gated_only_first_steps"]]
    if sum(summary[k] for k in ("direct_only", "gated_only", "both", "neither")) != summary["changed"]:
        raise ValueError("partition mismatch")
    expected = (summary["gated_only"] - summary["direct_only"]) / (len(rows) * 64)
    if not np.isclose(expected, old["summary"]["success_difference"], rtol=0, atol=1e-12):
        raise ValueError("net success mismatch")
    with OUT.open("x") as stream:
        json.dump({"summary": summary, "rows": rows, "result_sha256": file_hash(source.RESULT),
                   "plan_sha256": file_hash(source.PLAN), "physical_sha256": file_hash(source.source.src.PHYS),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "post-hoc same-backend evaluator-only audit"}, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
