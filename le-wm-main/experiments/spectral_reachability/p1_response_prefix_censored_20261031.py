"""Re-run fixed prefix roster with strictly verified per-branch canvas censoring."""
import argparse
import json
import time
from pathlib import Path

from . import p1_response_prefix_pilot_20261031 as pilot
from . import codex_p1_fixed_suffix_censored_runner_20261030 as censored
from . import codex_p1_fixed_suffix_censored_consume_20261030 as saved

ROOT, prior = pilot.ROOT, pilot.runner
NAME = "p1_response_prefix_censored_20261031"
JOIN_SHA = "16c86645bffcc01737e259810e749064d36e9d6a56b94d373e2725efbce71224"
PREP = ROOT / "results/p1_response_prefix_pilot_20261031_prepare_RUN01.json"
PREP_SHA = "9a6b0b55a647e1c0e43872008aa59bbe49a20c5ecd53b9745e740fb91e8bf27f"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_PREFIX_PILOT_CENSORED_PROTOCOL_20261031.md"
need, sha = prior.require, prior.sha


def build():
    need(sha(PREP) == PREP_SHA, "original prepared hash")
    original, data = pilot.material(JOIN_SHA)
    need(original == prior.native.read(PREP, cap=32 * 1024**2), "original prepare/source drift")
    bindings = dict(original["bindings"], **{str(PREP): PREP_SHA})
    for path in (Path(__file__).resolve(), Path(censored.__file__).resolve(),
                 Path(censored.episode.__file__).resolve(), Path(saved.__file__).resolve(),
                 Path(__file__).with_name("p1_response_prefix_censored_consume_20261031.py").resolve(),
                 PROTOCOL):
        bindings[str(path)] = sha(path)
    need(all(sha(p) == d for p, d in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "plans": original["plans"], "bindings": bindings,
            "fixed_rows": 16, "P1_admission": False, "independent_confirmation": False,
            "policy": "original guard and finite canvas violation only; no endpoint padding"}, data


def write(path, value):
    return pilot.replay.source.base.write(path, value)


def run(prepared_path, digest, output):
    need(sha(prepared_path) == digest, "prepared hash")
    fixed, data = build()
    need(fixed == prior.native.read(prepared_path, cap=32 * 1024**2), "prepared rebuild")
    prior.NAME = NAME
    prior.native.constructor_contract()
    from gymnasium import make
    import stable_worldmodel  # noqa: F401
    store = prior.Store(output)
    meter = prior.Meter(store, time.monotonic())
    rows = [prior.make_row(plan) for plan in fixed["plans"]]
    errors = []
    try:
        store.put("execution_binding.json", {"prepared": str(prepared_path), "prepare_sha256": digest})
        store.emit("run_started", fixed_rows=16, caps=prior.contract.contract()["caps"])
        censored.run_branches(fixed, data, store, meter, rows, errors, make)
    except BaseException as exc:
        errors.append(prior.native.base.error("execution", exc))
    errors.extend(meter.secondary)
    try:
        need(all(sha(p) == h for p, h in fixed["bindings"].items()), "final source drift")
        need(not any(meter.active.values()), "active calls")
        if not errors:
            prior.census(rows, meter.counts)
    except BaseException as exc:
        errors.append(prior.native.base.error("final_validation", exc))
    complete = not errors and all(r["status"] in
                                   ("horizon_completed", "native_stopped", "censored_out_of_domain")
                                   for r in rows)
    status = "physical_recorded_not_independently_consumed" if complete else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = {"schema": NAME + ".report", "status": status, "rows": rows, "errors": errors,
              "counts": meter.counts, "receipts": meter.receipts, "prepared": str(prepared_path),
              "prepare_sha256": digest, "journal_chain_tail": store.previous,
              "journal_sha256": sha(store.path / "journal.jsonl"), "fixed_rows": 16,
              "P1_admission": False, "independent_confirmation": False, "new_neural_calls": 0}
    store.put("report.json", report, final=True)
    return {"status": status, "report_sha256": sha(store.path / "report.json"),
            "statuses": [r["status"] for r in rows], "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    if args.command == "prepare":
        fixed, _ = build()
        result = {"sha256": write(output, fixed)}
    else:
        need(args.authorize_physics and args.prepared and args.sha256, "authorization and prepare required")
        print("Execute all 16 fixed prefixes, preserving verified canvas censoring.", flush=True)
        result = run(args.prepared, args.sha256, output)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
