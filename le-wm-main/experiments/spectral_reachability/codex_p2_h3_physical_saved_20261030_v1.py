"""Independent saved budget-event reconstruction, no neural/physics execution."""
import argparse
import json
from pathlib import Path
from experiments.spectral_reachability import codex_p2_h3_rechart_physical_20261030_v1 as producer

NAME = "codex_p2_h3_physical_saved_20261030_v1"
need = producer.need


def check_budget(events, report, binding):
    counts = {phase: {kind: {"attempts": 0, "returns": 0} for kind in ("step","reset")}
              for phase in ("root_collection","warmup","selected")}
    pending = None
    selected = {}
    for event in events:
        kind = event.get("kind")
        if kind not in ("env_step", "env_reset"): continue
        phase, op = event["phase"], kind[4:]
        need(phase in counts and type(event["completed"]) is bool, "typed event")
        count = counts[phase][op]
        if not event["completed"]:
            need(pending is None and event["index"] == count["attempts"], "attempt chronology")
            pending = event
            count["attempts"] += 1
            if phase == "selected" and op == "step":
                key = (event["root_sha256"], event["method"])
                selected.setdefault(key, []).append(event["command"])
        else:
            need(pending is not None and {k:v for k,v in event.items() if k != "completed"} ==
                 {k:v for k,v in pending.items() if k != "completed"}, "return matches intent")
            count["returns"] += 1
            pending = None
    need(pending is None, "unclosed operation")
    recorded = report["physical_call_receipt"]
    for phase in counts:
        need(counts[phase]["step"] == recorded["steps"][phase] and
             counts[phase]["reset"] == recorded["resets"][phase], "budget receipt mismatch")
    need(counts["root_collection"]["step"]["attempts"] == 0 and
         counts["warmup"]["step"]["attempts"] <= 120 and counts["selected"]["step"]["attempts"] <= 120,
         "phase limits")
    for plan, row in zip(binding["plans"], report["rows"], strict=True):
        cmds = selected.get((plan["root_sha256"],plan["method"]), [])
        need(len(cmds) == row["physical_env_step_attempts"] <= 15 and
             cmds == plan["commands"][:len(cmds)], "meter/plan exact prefix")
    return counts


def consume(directory, sha):
    directory = Path(directory)
    basic = producer.consume(directory, sha)
    report = producer.native.io.decode((directory/"report.json").read_bytes())
    binding = producer.build()
    events = [json.loads(x) for x in (directory/"physical_budget.events.jsonl").read_text().splitlines()]
    counts = check_budget(events, report, binding)
    return dict(schema=NAME+".receipt", status="saved_budget_and_branches_checked",
                report_sha256=sha, producer_check=basic, event_counts=counts,
                slots=[dict(root=r["root_index"], arm=r["method"], status=r["status"],
                            steps=r["completed_commands"], full=r["full_horizon_completed"],
                            native_success=r.get("official",{}).get("official_success"))
                       for r in report["rows"]],
                action_provenance="new_cuda_chart_reconstruction", original_raw_bits_proved=False,
                physics_rerun=False, neural_calls=0, total_resource_matched=False, P2_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True); p.add_argument("--sha", required=True)
    p.add_argument("--receipt", required=True)
    a = p.parse_args()
    path = Path(a.receipt)
    need(path.is_absolute() and path.parent.resolve() == path.parent and path.name.startswith(NAME+"_")
         and not path.exists() and not path.is_symlink(), "exclusive receipt")
    result = consume(a.output, a.sha)
    producer.native.write(path, result)
    print(json.dumps(dict(status=result["status"], sha256=producer.chart.prior.checks.run.digest_file(path))))


if __name__ == "__main__": main()
