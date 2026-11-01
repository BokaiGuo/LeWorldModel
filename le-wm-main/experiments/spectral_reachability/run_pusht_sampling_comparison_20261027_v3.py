"""Physics-only replay of sealed V2 plans; robust finalize, no CUDA/model load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027_v2 as v2

TAG = "pusht_sampling_comparison_20261027_v3"
write = v2.write


def error(exc, phase):
    return {"phase": phase, "type": type(exc).__name__, "message": str(exc)}


def finalize(*, output, rows, protocol, bindings, digest, verify_sources,
             failures=None, comparisons=None):
    """Always write a receipt for integrity-check failures (including str keys).

    Does not raise for a failed check. Caller raises only AFTER report.json is
    durably closed. Unwritable/full storage itself cannot be papered over.
    """
    failures = list(failures or [])
    expected = {(i,m) for i in range(4) for m in v2.METHODS}
    matrix = [(r.get("root_index"),r.get("method")) for r in rows]
    complete = len(matrix) == 8 and set(matrix) == expected
    gates = {"all_eight_retained": complete,
        "no_execution_errors": complete and all(r.get("status") in
            ("completed","observation_unavailable","environment_terminated") for r in rows),
        "trace_matches_sealed_prefix": complete and all(r.get("trace_matches_sealed_prefix") is True for r in rows),
        "official_and_pose_saved": complete and all("official" in r and "pose" in r for r in rows)}
    checks = []
    # Explicit loop, not all(generator): retain every failure; always Path-adapt.
    for path, expected_sha in bindings.items():
        try:
            actual = digest(Path(path))
            check = {"path": str(path), "expected": expected_sha, "actual": actual, "matched": actual == expected_sha}
        except Exception as exc:
            failures.append(error(exc, "binding_hash"))
            check = {"path": str(path), "expected": expected_sha, "matched": False, "error": str(exc)}
        checks.append(check)
    gates["bound_bytes_unchanged"] = bool(checks) and all(c["matched"] for c in checks)
    source_verification = None
    try:
        source_verification = verify_sources()
        gates["sources_inputs_unchanged"] = True
    except Exception as exc:
        failures.append(error(exc, "source_verification"))
        gates["sources_inputs_unchanged"] = False
    if comparisons is not None:
        gates["old_v2_all_eight_exact"] = (complete and len(comparisons) == 8
            and {(c["root_index"],c["method"]) for c in comparisons} == expected
            and all(c["full_trace_exact"] and c["whole_branch_exact"] for c in comparisons))
    passed = not failures and all(gates.values())
    report = {"schema": TAG+".physical.v3", "status": "guarded_calibration_recorded" if passed else "integrity_fail",
        "rows": rows, "gates": gates, "failures": failures, "protocol": protocol,
        "binding_verification": checks, "source_verification": source_verification,
        "v2_comparisons": comparisons,
        "completed_full_horizon_branches": sum(r.get("full_horizon_completed") is True for r in rows),
        "aborted_or_failed_branches": sum(r.get("full_horizon_completed") is not True for r in rows),
        "P1_admission": False, "P2_admission": False, "task_efficacy_repaired": False,
        "new_predict_calls": 0, "new_encoder_calls": 0, "new_optimization_calls": 0}
    write(Path(output)/"report.json", report)
    return report


def comparison(row, old):
    key = (row["root_index"], row["root_sha256"], row["method"])
    if key != (old.get("root_index"),old.get("root_sha256"),old.get("method")):
        raise ValueError("old V2 branch identity differs")
    # Empty/missing traces must not become a vacuous replay success.
    trace_exact = ("trace" in row and "trace" in old and row["trace"] == old["trace"]
        and len(row["trace"]) == row.get("completed_commands") == old.get("completed_commands"))
    return {"root_index": key[0], "method": key[2],
        "full_trace_exact": trace_exact, "whole_branch_exact": row == old}


def run(*, run, output, compare_v2=None):
    _, base, _, _, _, create_seal, verify_seal = v2.v1.dependencies()
    run, p, source, prepared = v2.load_prepared(run)
    plans_report = json.loads((run/"plans_report.json").read_text())
    witnesses = v2.validate_plans(run, p, source, prepared, plans_report, base.digest)
    bindings = dict(p["bindings"])
    bindings[str(run/"plans_report.json")] = base.digest(run/"plans_report.json")
    bindings.update({e["plan"]:e["plan_sha256"] for e in plans_report["rows"]})
    for w, _ in witnesses:
        bindings[str(run/f"root_{w['root_index']}.{w['method']}.candidates.jsonl")] = w["candidate_journal_sha256"]
    old_rows = {}
    if compare_v2 is not None:
        compare_v2 = Path(compare_v2).resolve()
        old_protocol_path = compare_v2/"protocol.json"
        old_protocol = json.loads(old_protocol_path.read_text())
        if (old_protocol["protocol_sha256"] != prepared["protocol_sha256"]
                or old_protocol["plans_bindings"] != {str(run/"plans_report.json"):base.digest(run/"plans_report.json"),
                    **{e["plan"]:e["plan_sha256"] for e in plans_report["rows"]}}):
            raise ValueError("old V2 physical protocol not bound to same plans")
        bindings[str(old_protocol_path)] = base.digest(old_protocol_path)
        for w, _ in witnesses:
            path = compare_v2/f"root_{w['root_index']}.{w['method']}.physical.json"
            old = json.loads(path.read_text())
            if (old["root_index"],old["root_sha256"],old["method"]) != (w["root_index"],w["root_sha256"],w["method"]):
                raise ValueError("old physical branch identity differs")
            old_rows[w["root_index"],w["method"]] = old
            bindings[str(path)] = base.digest(path)
    candidates = [json.loads(Path(path).read_text()) for path in p["bindings"] if Path(path).name == "report.json"]
    interventions = [r for r in candidates if "bank_path" in r.get("protocol",{})]
    if len(interventions) != 1:
        raise ValueError("unique upstream root source required")
    bank = json.loads(Path(interventions[0]["protocol"]["bank_path"]).read_text())
    upstream = base.physical._load_preflight(Path(bank["upstream_path"]))
    references = {r["root_sha256"]:r for r in upstream["rows"]}
    output = Path(output).resolve()
    if TAG not in output.name or output.parent != Path("results").resolve():
        raise ValueError("new tagged results directory required")
    output.mkdir(exist_ok=False)
    protocol = {"bindings": bindings, "v2_run": str(run), "v2_protocol_sha256": prepared["protocol_sha256"],
        "physical": p["physical"], "compare_v2": None if compare_v2 is None else str(compare_v2),
        "reoptimization": False, "model_loads": 0, "P1_admission": False, "P2_admission": False}
    rows, failures, seals = [], [], []
    comparisons = [] if compare_v2 is not None else None
    env = None
    try:
        env = base.gym.make("swm/PushT-v1", render_mode="rgb_array")
        evaluator = Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        files = [Path(__file__).resolve(), Path(v2.__file__).resolve(), Path(v2.v1.__file__).resolve(),
            Path(v2.visible.__file__).resolve(), Path(v2.guard.__file__).resolve(),
            Path(v2.labels.__code__.co_filename).resolve(), evaluator,
            Path(base.__file__).resolve(), Path(base.physical.__file__).resolve(), Path(base.p0.__file__).resolve(),
            Path(base.p0.packets.__file__).resolve(),
            Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename).resolve(),
            Path(__file__).resolve().parents[2]/f"tests/test_{TAG}.py",
            Path(__file__).resolve().parents[3]/f"docs/{TAG}.md"]
        seals.append(create_seal(files=sorted(set(files)), args_json=json.dumps(protocol), output_dir=output/"execution_source_seal"))
        write(output/"protocol.json", protocol)
        if tuple(env.unwrapped.variation_space["agent"]["color"].value.tolist()) != (65,105,225):
            raise ValueError("unsupported agent renderer")
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream["protocol"]["root_seed"]), oversample_factor=4)
        if len(roots) != 4 or [base.root_fingerprint(r) for r in roots] != p["roots"]:
            raise ValueError("physical root cohort/order differs")
        for w, commands in witnesses:
            index, key = w["root_index"], w["root_sha256"]
            row = {"root_index":index,"root_sha256":key,"method":w["method"]}
            try:
                root = roots[index]
                frames, history, goal_rgb = base.physical._reset_and_warmup(env,root,history_len=3)
                identity = v2.verify_rgb_context(frames, goal_rgb, references[key])
                identity.update(history=np.asarray(history).tolist(),state=base.p0.packets.state_of(env).tolist())
                peers = [r for r in rows if r["root_index"]==index and "context_identity" in r]
                if peers and identity != peers[0]["context_identity"]:
                    raise ValueError("method reset context differs")
                row["context_identity"] = identity
                row.update(v2.guarded_branch(env,commands,root_id=key,goal=root["goal"],
                    state_of=base.p0.packets.state_of,pose_label=base.physical._pose_label))
            except Exception as exc:
                row.update(status="branch_failed",failure=error(exc,"physical_branch"))
            # Persist physical outcomes BEFORE comparisons or final checks.
            write(output/f"root_{index}.{w['method']}.physical.json",row)
            rows.append(row)
            if comparisons is not None:
                try:
                    comparisons.append(comparison(row,old_rows[index,w["method"]]))
                except Exception as exc:
                    failures.append(error(exc,"v2_replay_comparison"))
            print(f"V3 root {index} {w['method']}: {row['status']}",flush=True)
    except Exception as exc:
        failures.append(error(exc,"physical_setup_or_execution"))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failures.append(error(exc,"env_close"))
    def verify_sources():
        v2.load_prepared(run)
        if not seals:
            raise ValueError("execution source seal was not created")
        return [verify_seal(output_dir=s["output_dir"],expected_manifest_sha256=s["manifest_sha256"]) for s in seals]
    report = finalize(output=output,rows=rows,protocol=protocol,bindings=bindings,
        digest=base.digest,verify_sources=verify_sources,failures=failures,comparisons=comparisons)
    if report["status"] != "guarded_calibration_recorded":
        raise RuntimeError("V3 integrity failure; report.json and all available branches retained")
    print(f"V3 report saved: {output/'report.json'}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--compare-v2",type=Path)
    run(**vars(parser.parse_args()))
