"""Compare saved joint actions with their fixed prefix/zero-suffix intervention."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_p2_h3_prefix5_control_20261030_v1 as control
from experiments.spectral_reachability import codex_p2_h3_physical_saved_20261030_v1 as saved

NAME = "codex_p2_h3_prefix5_pair_20261030_v1"
ROOT = control.ROOT
OUT = ROOT / "results/codex_p2_h3_prefix5_control_20261030_v1_RUN01"
PREP = ROOT / "results/codex_p2_h3_prefix5_control_20261030_v1_PREP01.json"
PARENT = ROOT / "results/codex_p2_h3_rechart_physical_20261030_v1_RUN01"
SHA = "3bbff3a319b512bf6b777692baa51fb197202011e181a943a7cce99230be848a"
PREP_SHA = "4a409129986ce4334880205973d43ea217459cfe648927d63af4e7392599e192"
PARENT_SHA = "858f76701f8dcff60bb52a764e916685ba8e3919f3dd25032d5138e6ba7e57c6"


def pair_rows(current, earlier, plans, old_plans):
    need = control.need
    need(len(current) == 4 and len(earlier) == 8 and len(plans) == 4, "row census")
    results = []
    for row, plan in zip(current, plans, strict=True):
        old = [r for r in earlier if r["root_index"] == row["root_index"] and r["method"] == "joint"]
        op = [p for p in old_plans if p["root_index"] == row["root_index"] and p["method"] == "joint"]
        need(len(old) == len(op) == 1, "unique joint source")
        old, op = old[0], op[0]
        need(row["root_index"] == plan["root_index"] and row["root_sha256"] == old["root_sha256"] and
             row["context_identity"] == old["context_identity"], "same task context")
        need(row["status"] == old["status"] == "completed" and
             row["completed_commands"] == old["completed_commands"] == 15, "full 15 steps")
        need(plan["commands"][:5] == op["commands"][:5] and
             plan["commands"][5:] == [[0., 0.]] * 10, "fixed intervention")
        need(row["trace"][:5] == old["trace"][:5], "first five observed transitions")
        delta = {k: row["pose"][k] - old["pose"][k]
                 for k in ("xy_distance", "angle_distance", "pose_cost")}
        results.append(dict(root=row["root_index"], joint=old["pose"], prefix5_zero10=row["pose"],
                            prefix_minus_joint=delta,
                            native_success_joint=old["official"]["official_success"],
                            native_success_prefix=row["official"]["official_success"]))
    need({r["root"] for r in results} == {0, 1, 2, 3}, "four distinct roots")
    return results


def compare():
    need = control.need
    need(control.chart.prior.checks.run.digest_file(PREP) == PREP_SHA, "prepared SHA")
    binding = control.build()
    need(json.loads(PREP.read_text()) == binding, "prepared binding")
    receipt = control.consume(OUT, SHA)
    prior = saved.consume(PARENT, PARENT_SHA)
    current = json.loads((OUT / "report.json").read_text())
    earlier = json.loads((PARENT / "report.json").read_text())
    events = [json.loads(line) for line in (OUT / "physical_budget.events.jsonl").read_text().splitlines()]
    counts = saved.check_budget(events, current, binding)
    return dict(schema=NAME, status="descriptive_not_admitted",
                roots=pair_rows(current["rows"], earlier["rows"], binding["plans"], control.original.build()["plans"]),
                control_report_sha256=SHA, prepared_sha256=PREP_SHA, parent_report_sha256=PARENT_SHA,
                event_counts=counts, control_saved_check=receipt, parent_saved_check=prior,
                fixed_after_prior_results=True, preregistered=False, new_neural_calls=0,
                total_resource_matched=False, independent_confirmation=False, P2_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    control.need(path.is_absolute() and path.name.startswith(NAME + "_") and not os.path.lexists(path), "exclusive output")
    control.native.write(path, compare())
    print(json.dumps({"status": "descriptive_not_admitted", "sha256": control.chart.prior.checks.run.digest_file(path)}))


if __name__ == "__main__":
    main()
