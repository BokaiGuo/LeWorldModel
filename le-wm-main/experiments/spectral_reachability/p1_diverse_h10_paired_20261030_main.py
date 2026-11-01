"""Pair new selected ten-step probes with the fixed prior zero/greedy controls."""
import argparse
import json
from pathlib import Path

from . import p1_diverse_h10_20261030_main as h10
from . import codex_p1_h10_paired_table_20261030_main as metrics

NAME = "p1_diverse_h10_paired_20261030_main"
OLD = h10.ROOT / "results/codex_p1_h10_paired_table_20261030_main_FULL01.json"
OLD_SHA = "4406169f802bf40912f9ff384d6d704b645d2d6a29ddf6f19936af32d4f3ef24"
OLD_RUN_PREP = h10.ROOT / "results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json"
OLD_RUN_PREP_SHA = "ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30"


def build(p, run_dir, run_report, old):
    h10.need(old["prepare_sha256"] == OLD_RUN_PREP_SHA and len(old["rows"]) == 1008, "old table source")
    rows = []
    for anchor, bank, shard in zip(p["anchors"], p["probe_banks"], run_report["shards"], strict=True):
        h10.need(anchor["index"] == shard["anchor"], "shard identity")
        directory = run_dir / f"anchor_{anchor['index']:02d}"
        checked = h10.consumer.check_shard(directory, anchor, bank, shard["sha256"])
        raw = h10.native.read(directory / "report.json", cap=64*1024**2)
        own_zero = metrics.physical_metrics(checked["endpoints"][0], anchor["goal"])
        for j, (probe, state, status, branch) in enumerate(zip(
                bank, checked["endpoints"], checked["statuses"], raw["rows"], strict=True)):
            endpoint = metrics.physical_metrics(state, anchor["goal"])
            selected = dict(root_slot=anchor["root_slot"], condition=anchor["condition"],
                            arm=anchor["arm"], logical_pair=anchor["logical_pair"], role="selected",
                            probe=probe["name"], probe_sha256=probe["raw10_sha256"], status=status,
                            accepted_steps=len(branch["responses"]), censor=branch.get("censor"),
                            last_observed_state=branch["responses"][-1]["state"] if branch["responses"] else None,
                            root_metrics=metrics.physical_metrics(anchor["root_state"], anchor["goal"]),
                            prefix_metrics=metrics.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"]),
                            endpoint=endpoint, endpoint_minus_own_zero=metrics.paired_difference(endpoint, own_zero),
                            independent_suffix_result=None)
            controls = {}
            if j < 41:
                for role in ("zero", "greedy"):
                    matches = [r for r in old["rows"] if r["logical_pair"] == anchor["logical_pair"]
                               and r["role"] == role and r["probe"] == probe["name"]]
                    h10.need(len(matches) == 1, "old control roster")
                    control = matches[0]
                    h10.need(control["probe_sha256"] == probe["raw10_sha256"]
                             and control["root_slot"] == anchor["root_slot"]
                             and control["condition"] == anchor["condition"] and control["arm"] == anchor["arm"]
                             and control["root_metrics"] == selected["root_metrics"], "paired control identity")
                    old_end = control["common_suffix_endpoint_native_metrics"]
                    controls[role] = dict(status=control["status"], observation=control["observation"],
                                          endpoint=old_end,
                                          endpoint_minus_own_zero=control["same_prefix_endpoint_minus_own_zero_suffix"],
                                          selected_minus_control=metrics.paired_difference(endpoint, old_end),
                                          prefix_metrics=control["prefix_metrics"])
            rows.append(dict(selected=selected, controls=controls))
    h10.need(len(rows) == 126, "fixed row denominator")
    return dict(schema=NAME, rows=rows, P1_admission=False, independent_confirmation=False,
                old_control_runtime_provenance="old prepare did not bind actual resume entry",
                limitations=["known roots and duplicate prefixes", "same suffix tests action response, not plannability",
                             "independent suffix unmeasured", "original selected tail lacks matched controls"])


def main():
    parser = argparse.ArgumentParser()
    for name in ("directory", "report-sha", "prepared", "prepare-sha", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    h10.need(output.is_absolute() and output.parent == h10.ROOT / "results"
             and output.name.startswith(NAME+"_") and not output.exists(), "exclusive output")
    p, _ = h10.verify(args.prepared, args.prepare_sha)
    directory = Path(args.directory)
    report_path = directory / "run_report.json"
    h10.need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink()
             and h10.sha(report_path) == args.report_sha and h10.sha(OLD) == OLD_SHA
             and h10.sha(OLD_RUN_PREP) == OLD_RUN_PREP_SHA
             and h10.native.read(OLD_RUN_PREP, cap=64*1024**2)["original"] ==
             h10.native.read(h10.OLD, cap=64*1024**2), "pinned inputs")
    report = h10.native.read(report_path, cap=4*1024**2)
    binding = h10.native.read(directory / "run_binding.json", cap=64*1024**2)
    h10.need(report["status"] == "recorded_not_consumed" and report["fixed_branches"] == 126
             and len(report["shards"]) == 3
             and h10.sha(directory / "run_binding.json") == report["run_binding_sha256"]
             and binding["prepared"] == args.prepared and binding["prepare_sha256"] == args.prepare_sha
             and binding["bindings"] == dict(p["bindings"], **{args.prepared: args.prepare_sha})
             and all(h10.sha(f) == d for f, d in binding["bindings"].items()), "run binding")
    value = build(p, directory, report, h10.native.read(OLD, cap=32*1024**2))
    value.update(source_report_sha256=args.report_sha, source_prepare_sha256=args.prepare_sha,
                 old_paired_table_sha256=OLD_SHA, code_sha256=h10.sha(Path(__file__)))
    h10.need(h10.sha(report_path) == args.report_sha and h10.sha(OLD) == OLD_SHA, "final source drift")
    print(json.dumps({"sha256": h10.runner._write_new(output, value), "rows": len(value["rows"])}))


if __name__ == "__main__":
    main()
