"""Read-only S/G/Z saved-plan audit; never authorizes prospective execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_saved_h4_parfit_20261101 as saved
from . import p1_coherent_samebank_own_physical_20261101_codex as own_selection
from . import p1_coherent_same_bank_greedy_20261101_codex as greedy_impl
from . import p1_prospective_mechanism_physical_20261101_codex as physical

NAME = "p1_prospective_upstream_consumer_20261101_codex"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_bound(path, expected):
    path = Path(path).resolve(strict=True)
    need(saved.spec.is_hash(expected) and file_sha(path) == expected, "external SHA mismatch: " + str(path))
    return json.loads(path.read_bytes())


def source_drift(mappings):
    """All referenced current source bytes, including missing files, fail closed."""
    wanted = {}
    for sources in mappings:
        need(isinstance(sources, dict) and sources, "empty source closure")
        for name, digest in sources.items():
            need(Path(name).is_absolute() and saved.spec.is_hash(digest), "source path/hash")
            if name in wanted:
                need(wanted[name] == digest, "conflicting expected source hashes")
            wanted[name] = digest
    mismatches = []
    for name, expected in wanted.items():
        path = Path(name)
        actual = file_sha(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(dict(path=name, expected=expected, actual=actual))
    return mismatches


def payload(events, name):
    matches = [e["payload"] for e in events if e["kind"] == name]
    need(len(matches) == 1, "unique event required: " + name)
    return matches[0]


def root_selection(events, receipt):
    initial = np.asarray(payload(events, "initial_candidates")["raw"], np.float32)
    opt = np.asarray(payload(events, "optimized_candidates_before_exchange")["raw"], np.float32)
    need(initial.shape == opt.shape == (8, 4, 5, 2) and np.isfinite(initial).all()
         and np.isfinite(opt).all() and np.max(np.abs(initial)) <= 1 and np.max(np.abs(opt)) <= 1
         and np.array_equal(initial[:, 0], opt[:, 0]), "initializer/protected-prefix binding")
    exchange = payload(events, "suffix_exchange")
    scores = np.asarray(exchange["scores"], np.float64)
    need(scores.shape == (64,) and np.isfinite(scores).all(), "64 exchange scores")
    tails = [min(range(8), key=lambda j: (scores[i * 8 + j], j != i, j)) for i in range(8)]
    need(tails == exchange["tail_indices"], "exchange tie/argmin")
    bank = np.stack([np.concatenate((opt[i, :1], opt[j, 1:])) for i, j in enumerate(tails)])
    selected = payload(events, "selected_full_plan")
    final_scores = np.asarray(selected["scores"], np.float64)
    need(final_scores.shape == (8,) and np.isfinite(final_scores).all()
         and selected["scores"] == [exchange["scores"][i * 8 + j] for i, j in enumerate(tails)]
         and selected["scores"] == receipt["scores"], "root selection score binding")
    index = min(range(8), key=lambda i: (final_scores[i], i))
    chosen = np.asarray(selected["raw"], np.float32)
    need(index == selected["index"] == receipt["selected_index"]
         and np.array_equal(chosen, bank[index])
         and np.array_equal(chosen, np.asarray(receipt["selected_full_plan"], np.float32))
         and np.array_equal(chosen[0], np.asarray(receipt["commands"], np.float32)),
         "root selected plan identity")
    return initial[:, 0], chosen[0], index


def greedy_selection(result, binding, initial):
    bank = np.asarray(result["prefixes"], np.float32)
    scores = np.asarray(result["scores"], np.float64)
    need(bank.shape == (8, 5, 2) and np.array_equal(bank, initial)
         and np.array_equal(bank, np.asarray(binding["prefixes"], np.float32))
         and scores.shape == (8,) and np.isfinite(scores).all()
         and binding["tie"] == "first_index", "same-bank inputs/scores")
    index = min(range(8), key=lambda i: (scores[i], i))
    need(index == result["greedy_index"]
         and np.array_equal(bank[index], np.asarray(result["selected_prefix"], np.float32)),
         "greedy first-argmin")
    return bank[index], index


def audit_account(ledger, expected_account, contract_sha, run_id, expected_cost):
    account = saved.consumer.consume(ledger["journal"], ledger["sha256"], contract_sha,
                                     "structured", "node", run_id)
    need(account == expected_account and account["status"] == "closed_unverified"
         and account["charged_upper_bounds"] == expected_cost, "durable account/cost")
    return account


def binding_report(path, sha):
    path = Path(path).resolve(strict=True)
    report = read_bound(path, sha)
    directory = path.parent
    need(report["status"] == "complete" and report["error"] is None
         and report["model_before"] is not None and report["model_before"] == report["model_after"]
         and report["physical_execution"] is False, "complete model-only report")
    binding = read_bound(directory / "binding.json", report["binding_sha256"])
    prepared = read_bound(report["prepared"], report["prepare_sha256"])
    need(binding == prepared and binding["expected_model_hash"] == report["model_before"], "prepared/model binding")
    return report, binding, directory


def inspect_own(row, inp, binding, directory):
    need(all(row[k] == inp[k] for k in ("prefix", "prefix_raw", "pixel_sha256", "cp5_state_identity_sha256")),
         "cp5 input binding")
    task, base = row["task"], binding["base_task"]
    need(all(task[k] == base[k] for k in ("goal", "action_mean", "action_std")), "goal/normalizer drift")
    prefix = np.asarray(row["prefix_raw"], np.float32)
    warmup = np.asarray(binding["warmup_raw"], np.float32)
    mean, std = np.asarray(base["action_mean"], np.float32), np.asarray(base["action_std"], np.float32)
    need(prefix.shape == (5, 2) and warmup.shape == (5, 2) and mean.shape == std.shape == (2,)
         and np.isfinite(prefix).all() and np.isfinite(warmup).all() and np.isfinite(std).all()
         and np.isfinite(mean).all() and (std > 0).all(), "own action history shape")
    history = ((np.concatenate((warmup, prefix)) - mean) / std).reshape(2, 10).tolist()
    need(task["root_context"]["action_history"] == history
         and task["root_context"]["latent_history"] == base["root_context"]["latent_history"][-2:] + [row["cp5_latent"]],
         "own actual history binding")
    hold = np.asarray(row["hold_plan"], np.float32)
    need(hold.shape == (4, 5, 2) and np.array_equal(hold[0], prefix)
         and np.count_nonzero(hold[1:]) == 0, "common zero hold reference")
    for kind, suffix in (("node", "_node"), ("encoding", "_encode")):
        ledger = row[kind + "_ledger"]
        path = Path(ledger["journal"]).resolve(strict=True)
        need(path.is_relative_to(directory), "own ledger outside report")
        contract = json.loads(path.read_bytes().splitlines()[0])["payload"]["contract"]
        contract_sha = saved.spec.digest(contract)
        if kind == "node":
            need(contract_sha == row["contract_sha256"], "node contract")
        expected = binding["node_cost"].copy() if kind == "node" else dict.fromkeys(saved.h4.AXES, 0)
        if kind == "encoding":
            expected.update(encode_calls=1, encode_samples=1)
        audit_account(ledger, row[kind + "_account"], contract_sha, row["prefix"] + suffix, expected)
    trace = Path(row["trace"]).resolve(strict=True)
    need(trace.is_relative_to(directory) and file_sha(trace) == row["trace_sha256"], "own trace hash/path")
    chosen = own_selection.validate_selection(saved.chain(trace), row)
    return prefix, chosen


def inspect(structured, greedy, own, samebank):
    structured_path, structured_sha = structured
    sr = read_bound(structured_path, structured_sha)
    need(sr["model_before"] is not None and sr["model_before"] == sr["model_after"], "structured model")
    sp = read_bound(sr["prepared"], sr["prepare_sha256"])
    need(sp["protocol"]["initializer_variant"] == "packet_constant_coherent", "coherent H4 protocol")
    row = sr["rows"][0]
    node = sr["nodes"][row["episode_id"]][0]
    need(row["arm"] == "structured" and node["boundary"] == 0 and node["status"] == "sealed",
         "structured root node")
    receipt = node["receipt"]
    trace = Path(receipt["trace"]).resolve(strict=True)
    need(trace.is_relative_to(Path(structured_path).resolve().parent)
         and file_sha(trace) == receipt["trace_sha256"], "root trace")
    events = saved.chain(trace)
    contract = payload(events, "h4_contract_binding")
    need(contract["boundary"] == 0 and contract["policy"]["horizon"] == 4
         and contract["contract_sha256"] == receipt["contract_sha256"], "root contract")
    initial, s_prefix, s_index = root_selection(events, receipt)
    audit_account(receipt["ledger"], receipt["account"], receipt["contract_sha256"], "structured",
                  saved.h4.node_cost("structured", 0))
    gr, gb, gd = binding_report(*greedy)
    need(gr["source_report_sha256"] == structured_sha
         and gr["source_trace_sha256"] == receipt["trace_sha256"]
         and gb["source_report_sha256"] == structured_sha
         and gb["source_trace_sha256"] == receipt["trace_sha256"]
         and gb["source_prepare_sha256"] == sr["prepare_sha256"]
         and gr["model_before"] == sr["model_before"], "greedy root/model provenance")
    g_result = gr["result"]
    need(Path(g_result["ledger"]["journal"]).resolve(strict=True).is_relative_to(gd), "greedy ledger path")
    audit_account(g_result["ledger"], g_result["account"], g_result["contract_sha256"],
                  "same_bank", greedy_impl.COST)
    g_prefix, g_index = greedy_selection(g_result, gb, initial)
    own_report, own_binding, own_dir = binding_report(*own)
    single_report, single_binding, single_dir = binding_report(*samebank)
    need(own_report["model_before"] == single_report["model_before"] == sr["model_before"], "model mismatch")
    def find(report, binding, role, directory):
        rows = [x for x in report["rows"] if x["prefix"] == role]
        inputs = [x for x in binding["rows"] if x["prefix"] == role]
        need(len(rows) == len(inputs) == 1, "unique own role")
        prefix, tail = inspect_own(rows[0], inputs[0], binding, directory)
        return prefix, tail, rows[0]
    s, st, srow = find(own_report, own_binding, "structured", own_dir)
    z, zt, zrow = find(own_report, own_binding, "zero", own_dir)
    g, gt, grow = find(single_report, single_binding, "same_bank_greedy", single_dir)
    need(np.array_equal(s, s_prefix) and np.array_equal(g, g_prefix)
         and np.count_nonzero(z) == 0, "upstream role identity")
    drift = source_drift([sp["sources"], gb["sources"], own_binding["sources"], single_binding["sources"]])
    return dict(schema=NAME + ".audit.v1",
        status="blocked_source_drift" if drift else "saved_consistency_only",
        source_drift=drift, model_before=sr["model_before"],
        role_results={name: dict(prefix_raw=p.tolist(), own_suffix_raw=t.reshape(15, 2).tolist(),
                                 cp5_pixel_sha256=r["pixel_sha256"],
                                 cp5_state_identity_sha256=r["cp5_state_identity_sha256"])
                      for name, p, t, r in (("S", s, st, srow), ("G", g, gt, grow), ("Z", z, zt, zrow))},
        root_selected_index=s_index, greedy_index=g_index,
        checks=dict(saved_root_argmin=True, saved_greedy_first_argmin=True,
                    saved_own_argmin_and_ledgers=True, source_closure=not drift),
        no_physical_outcome_leakage="unknown", source_independence="unknown",
        chronological_precommit="unknown", confirmation_ready=False, P1_admission=False,
        limitations=["structural audit does not recompute neural scores or prove their origin",
                     "historical inputs include physical outcome reports; hash chains do not prove blind access",
                     "cp5 exported dynamics and independent root authority are not certified"])


def bind_physical_input(audit, manifest, source_reports, root_checkpoint):
    """Check actual physical input values; this does not authorize execution."""
    for role in ("S", "G", "Z"):
        actual, expected = manifest["roles"][role], audit["role_results"][role]
        for key in ("prefix_raw", "own_suffix_raw"):
            need(np.array_equal(np.asarray(actual[key], np.float32), np.asarray(expected[key], np.float32)),
                 "physical manifest role " + role + " " + key)
        for key in ("cp5_pixel_sha256", "cp5_state_identity_sha256"):
            need(actual[key] == expected[key], "physical manifest role " + role + " " + key)
    root = manifest["root"]
    need(root["root_state"] == root_checkpoint["state"] and root["goal_state"] == root_checkpoint["goal"]
         and root["warmup_rgb"][-1]["pixel_sha256"] == root_checkpoint["pixel_sha256"],
         "physical manifest root/goal/RGB")
    for report_path, report_sha in source_reports:
        absolute = str(Path(report_path).resolve(strict=True))
        need(manifest["sources"].get(absolute) == report_sha
             and absolute in manifest["planning_evidence"], "physical manifest report source binding")
    return dict(status="input_values_match_saved_not_authorized",
                root_dynamics_binding="not_independently_verified",
                cp5_dynamics_binding="not_independently_verified",
                no_physical_outcome_leakage="unknown", confirmation_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("structured", "greedy", "own", "samebank"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-manifest")
    parser.add_argument("--physical-manifest-sha256")
    parser.add_argument("--require-confirmation-ready", action="store_true")
    args = parser.parse_args()
    pairs = [(getattr(args, k), getattr(args, k + "_sha256"))
             for k in ("structured", "greedy", "own", "samebank")]
    report = inspect(*pairs)
    need(bool(args.physical_manifest) == bool(args.physical_manifest_sha256), "physical manifest and SHA must both be given")
    if args.physical_manifest:
        manifest, _ = physical.load_input(args.physical_manifest, args.physical_manifest_sha256)
        source = read_bound(*pairs[0])
        checkpoints = [p for p in source["rows"][0]["checkpoints"] if p["step"] == 0]
        need(len(checkpoints) == 1, "unique root checkpoint")
        report["physical_manifest_check"] = bind_physical_input(report, manifest, pairs, checkpoints[0])
    if args.output:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if report["source_drift"] or (args.require_confirmation_ready and not report["confirmation_ready"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
