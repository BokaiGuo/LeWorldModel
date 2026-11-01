"""Bind new CUDA winner action witnesses to earlier saved physical dispatches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p2_h3_raw_witness_20261030_main as witness
from . import codex_p2_h3_physical_saved_20261030_v1 as saved

ROOT = witness.base.ROOT
RUN = ROOT / "results/codex_p2_h3_raw_witness_20261030_main_RUN01"
RUN_SHA = "05271deca38756fef3af141e9ecc3e1f3dc4d665dd56b44090ee3235e4ece581"
PREP = ROOT / "results/codex_p2_h3_raw_witness_20261030_main_PREP01.json"
PREP_SHA = "15120a07716763f18deb2c8caa2b58d4fe088096ead39211c7488ff9af1c0638"
PHYS = ROOT / "results/codex_p2_h3_rechart_physical_20261030_v1_RUN01"
PHYS_SHA = "858f76701f8dcff60bb52a764e916685ba8e3919f3dd25032d5138e6ba7e57c6"
need = saved.need


def pair(winners, cases, binding, report, events):
    by_winner = {(row["root"], row["arm"]): row for row in winners}
    by_case = {case["root_index"]: case for case in cases}
    need(len(by_winner) == len(winners) == 8 and len(by_case) == len(cases) == 4,
         "unique winner/case roster")
    need(len(binding["plans"]) == len(report["rows"]) == 8, "physical roster")
    selected = {}
    for event in events:
        if event.get("kind") == "env_step" and event.get("phase") == "selected" \
                and event.get("completed") is False:
            key = (event["root_sha256"], event["method"])
            selected.setdefault(key, []).append(event["command"])
    paired = []
    seen = set()
    for plan, row in zip(binding["plans"], report["rows"], strict=True):
        key = (plan["root_index"], plan["method"])
        need(key not in seen and key in by_winner and key[0] in by_case, "unique physical key")
        seen.add(key)
        case = by_case[key[0]]
        need(plan["root_sha256"] == row["root_sha256"] == case["root_sha256"]
             and row["root_index"] == key[0] and row["method"] == key[1]
             and plan["predictor_context"] == case["context"]
             and plan["goal"] == case["goal"], "root/goal/context binding")
        commands = selected.get((plan["root_sha256"], key[1]), [])
        need(row["completed_commands"] == 15 and len(commands) == 15
             and commands == plan["commands"], "complete physical dispatch")
        values = np.asarray(commands, dtype="<f4")
        need(values.shape == (15, 2) and np.isfinite(values).all()
             and np.all(np.abs(values) <= 1), "physical command domain")
        raw = values.tobytes().hex()
        need(raw == plan["raw_float32_hex"] == by_winner[key]["raw_float32_hex"],
             "CUDA witness / dispatched action bytes")
        paired.append({"root": key[0], "arm": key[1], "root_sha256": plan["root_sha256"],
                       "winner_tag": by_winner[key]["winner_tag"],
                       "action_sha256": hashlib.sha256(bytes.fromhex(raw)).hexdigest(),
                       "physical_steps": row["completed_commands"],
                       "native_success": row["official"]["official_success"]})
    need(seen == set(by_winner) and len(selected) == 8, "unconsumed slots")
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    path = Path(args.receipt)
    need(path.is_absolute() and path.parent == ROOT / "results"
         and path.name.startswith("codex_p2_h3_witness_physical_pair_")
         and not path.exists() and not path.is_symlink(), "exclusive receipt")
    old = saved.consume(PHYS, PHYS_SHA)
    binding = saved.producer.build()
    report = saved.producer.native.io.decode((PHYS / "report.json").read_bytes())
    events = [json.loads(line) for line in (PHYS / "physical_budget.events.jsonl").read_text().splitlines()]
    # configure() intentionally mutates the shared legacy runner contract.
    # Verify all legacy sources before switching it to the new witness contract.
    witness.configure()
    _, cases = witness.base.verify_prepare(PREP, PREP_SHA)
    new = witness.audit(RUN, RUN_SHA)
    rows = pair(new["winner_raw_actions"], cases, binding, report, events)
    need(witness.base.digest_file(PHYS / "report.json") == PHYS_SHA, "report drift")
    result = {"schema": "codex_p2_h3_witness_physical_pair_20261030_main",
              "rows": rows, "new_cuda_report_sha256": RUN_SHA, "new_prepare_sha256": PREP_SHA,
              "physical_report_sha256": PHYS_SHA, "physical_saved_status": old["status"],
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "retrospective": True, "physical_rerun": False, "new_neural_calls": 0,
              "historical_raw_bits_proved": False, "equal_budget_certified": False,
              "P1_admission": False, "P2_admission": False}
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"receipt": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "slots": len(rows)}))


if __name__ == "__main__":
    main()
