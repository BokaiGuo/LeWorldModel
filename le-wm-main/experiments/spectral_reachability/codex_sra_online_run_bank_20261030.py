"""Online SRA joint-H2 selection, with a saved, receipt-bound audit.

No model loading, environment, GPU allocation or physical dispatch. The caller
must supply the frozen model and replay. The fixture hook is explicitly marked
in every audit and is never evidence of a native neural execution.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_online_model_action_20261030 as online
from experiments.spectral_reachability import codex_native_runtime_meter_20261030 as native

HERE = Path(__file__).resolve()
DIR = HERE.parent
REQUIRED = (
    "codex_sra_online_run_bank_20261030.py",
    "codex_online_model_action_20261030.py",
    "codex_native_runtime_meter_20261030.py",
    "codex_paired_durable_journal_20261030.py",
    "codex_paired_physical_interface_20261030.py",
    "p2_no_composition_sra_20261027_v2.py",
    "p2_no_composition_sra_20261027_v1.py",
    "p2_full20_sampling_20261027_v2.py",
    "p2_matched_composition_20261027.py",
    "pusht_full_packet_compiler_20261027.py",
    "pusht_full_packet_chart_20261027.py",
    "pusht_composition_cuda_replay_v2.py",
    "pusht_d2p_packet_contract.py",
    "preflight_pusht_d2p_p0.py",
    "p4_feedback_matched_20261027_v2.py",
    "d2p_composition_witness_v2.py",
    "d2p_prerun_source_seal.py",
    "d2p_kinodynamic_program_search.py",
    "run_pusht_program_search_cuda.py",
)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=online.load_journal().no_duplicates,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_manifest(path, expected):
    need(file_sha(path) == expected, "manifest drift")
    manifest = read_json(path)
    need(set(manifest) == {"schema", "files", "action_units"} and
         manifest["schema"] == "codex_sra_online_sources_v1" and
         isinstance(manifest["action_units"], str) and manifest["action_units"].strip(),
         "invalid manifest")
    files = manifest["files"]
    need(isinstance(files, dict) and set(REQUIRED) <= set(files), "required sources missing")
    for name, ref in files.items():
        need(isinstance(name, str) and name and isinstance(ref, dict) and
             set(ref) == {"path", "sha256"} and file_sha(ref["path"]) == ref["sha256"],
             "dependency drift: " + str(name))
        if name in REQUIRED:
            need(Path(ref["path"]).resolve() == (DIR / name).resolve(), "required source path mismatch")
    need(files[SRA_NAME]["sha256"] == native.SRA_SHA, "native SRA source mismatch")
    return manifest


SRA_NAME = "p2_no_composition_sra_20261027_v2.py"


def checked_target(path, expected):
    need(file_sha(path) == expected, "goal source drift")
    target = read_json(path)
    need(set(target) == {"goal_rgb_sha256", "goal_vectors", "scoring"}, "target schema")
    digest = target["goal_rgb_sha256"]
    need(isinstance(digest, str) and len(digest) == 64 and
         all(c in "0123456789abcdef" for c in digest), "goal RGB identity")
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    compiled = {"goal_vectors": target["goal_vectors"], "scoring": target["scoring"]}
    metric_for(compiled)  # Check finite dimensions before any model work.
    return target, compiled


def audit_candidates(result, bank):
    attempts = result["attempts"]
    need(result["candidate_and_replay_complete"] is True and result["status"] == "search_complete"
         and len(attempts) == 40 and result["planning_predict_samples"] == 80
         and result["replay_predict_samples"] == 4 and len(result["ledger"]) == 84,
         "incomplete native SRA search or replay")
    need(all(x["completed"] is True and x["metric_attempted"] is True for x in attempts),
         "candidate failure")
    candidates = []
    for i, a in enumerate(attempts):
        need(a["candidate_id"] == f"joint_h2_no_composition_sra_{i:02d}"
             and a["commands"] == bank["commands"][i]
             and a["actions"] == a["commands"][0] + a["commands"][1], "candidate/bank mismatch")
        candidates.append({"id": a["candidate_id"], "cost": a["cost"], "commands": a["commands"][0]})
    best = min(candidates, key=lambda c: c["cost"])
    need(result["selected_candidate_id"] == best["id"] and result["selected_model_cost"] == best["cost"]
         and result["witness"]["independent_full_replay_exact"] is True
         and result["witness"]["receipt"]["actions"] ==
             attempts[candidates.index(best)]["actions"], "winner or replay mismatch")
    return candidates, best["id"]


def validate_model_entries(bank_result, entries, metered):
    need(entries[:8] == bank_result["native_entries"] and len(entries) == 176 and len(metered) == 176,
         "model entry phase census")
    for entry, event in zip(entries, metered):
        kind, batch = entry["kind"], entry["batch"]
        need(kind in ("predict", "action_encoder") and batch == 1 and
             entry["context_tokens"] == 3 and entry["returned"] is True,
             "native model entry")
        amounts = ({"predict_calls": 1, "predict_samples": 1} if kind == "predict"
                   else {"action_encoder": 1})
        need(event["label"] == "native_" + kind and event["amounts"] == amounts and
             event["returned"] is True, "native entry/journal meter mismatch")
    need([e["kind"] for e in entries] == ["action_encoder", "predict"] * 88,
         "native model entry order")


def _save_exclusive(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with Path(path).open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    online.load_journal().sync_dir(Path(path).parent)
    return online.sha(raw)


def planner(*, model, device, replay, manifest_path, manifest_sha256, target_path,
            audit_path, fixture_compile=None, fixture_replay=False):
    """Build a planner callback for online_select; fixture hooks are test-only."""
    manifest_path, target_path = str(Path(manifest_path).resolve()), str(Path(target_path).resolve())
    need((fixture_compile is None) == (fixture_replay is False), "fixture hooks must be paired")
    def choose(request, context, meter):
        need(request["arm"] == "sra_no_composition", "wrong arm")
        manifest = checked_manifest(manifest_path, manifest_sha256)
        target, compiled = checked_target(target_path, request["identity"]["goal_sha256"])
        need(meter.decision_id == online.sha(online.encode(
            {"request": request, "method_sha256": manifest_sha256})), "manifest not bound to decision")
        online.validate_context(context)
        if not fixture_replay:
            from experiments.spectral_reachability.p2_matched_composition_20261027 import RootAnchoredCudaReplay
            need(type(replay) is RootAnchoredCudaReplay and replay.model is model and replay.device == device,
                 "replay must use the same model and device")
        before = len(meter.events)
        bank_result = native.sra_online_bank(model, device, context, compiled, replay, meter,
                                              native=fixture_compile, source_inventory=manifest["files"])
        bank = bank_result["bank"]
        from experiments.spectral_reachability import p2_no_composition_sra_20261027_v2 as sra
        from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
        root = {"root_id": request["identity"]["root_sha256"], "root_rgb_sha256": request["frames"][-1],
                "goal_rgb_sha256": target["goal_rgb_sha256"], "context": deepcopy(context)}
        provenance = {"checkpoint_sha256": request["identity"]["model_sha256"],
                      "encoder_sha256": request["identity"]["model_sha256"],
                      "source_sha256": manifest_sha256, "protocol_sha256": manifest_sha256,
                      "metric_sha256": request["identity"]["goal_sha256"],
                      "candidate_bank_sha256": sra.digest(bank)}
        domain = {"name": "raw", "units": manifest["action_units"], "low": [-1., -1.], "high": [1., 1.],
                  "action_block": 5, "normalization_sha256": request["identity"]["normalizer_sha256"]}
        with native.metered_native_model(model, meter) as entries:
            result = sra.run_bank(bank=bank, root=root, provenance=provenance, action_domain=domain,
                                  target_metric=metric_for(compiled), replay=replay, fixture=fixture_replay)
        candidates, selected = audit_candidates(result, bank)
        all_entries = bank_result["native_entries"] + entries
        validate_model_entries(bank_result, all_entries, meter.events[before:])
        checked_manifest(manifest_path, manifest_sha256)
        checked_target(target_path, request["identity"]["goal_sha256"])
        audit = {"schema": "codex_sra_online_audit_v1", "request": request, "context": context,
                 "manifest_path": manifest_path, "manifest_sha256": manifest_sha256,
                 "target_path": target_path, "target_sha256": request["identity"]["goal_sha256"],
                 "bank_result": bank_result, "search": result, "model_entries": all_entries,
                 "metered_events": deepcopy(meter.events[before:]),
                 "fixture": fixture_compile is not None or fixture_replay,
                 "claims": {"equal_total_compute": False, "flops": None, "physical_evidence": False}}
        audit_sha = _save_exclusive(audit_path, audit)
        for c in candidates:
            c["id"] += "@" + audit_sha
        return {"mode": "online_callback", "request_sha256": online.sha(online.encode(request)),
                "context_sha256": online.sha(online.encode(context)), "candidates": candidates,
                "selected_id": selected + "@" + audit_sha}
    return choose


def consume_sra(journal_path, journal_sha256, receipts, audit_path, audit_sha256):
    """Read-only: validate journal, online receipt, full bank, selection and source hashes."""
    rebuilt = online.consume_online(journal_path, journal_sha256, receipts)
    need(file_sha(audit_path) == audit_sha256, "audit hash mismatch")
    audit = read_json(audit_path)
    need(audit["schema"] == "codex_sra_online_audit_v1" and
         audit["claims"] == {"equal_total_compute": False, "flops": None, "physical_evidence": False},
         "audit schema or claims")
    checked_manifest(audit["manifest_path"], audit["manifest_sha256"])
    checked_target(audit["target_path"], audit["target_sha256"])
    matches = []
    for path, _ in receipts:
        receipt = read_json(path)
        if receipt["request"] == audit["request"] and receipt["context"] == audit["context"]:
            matches.append(receipt)
    need(len(matches) == 1, "audit receipt identity")
    receipt = matches[0]
    need(receipt["method_source"]["sha256"] == audit["manifest_sha256"] and
         receipt["request"]["identity"]["goal_sha256"] == audit["target_sha256"],
         "audit manifest or target binding")
    candidates, selected = audit_candidates(audit["search"], audit["bank_result"]["bank"])
    for c in candidates:
        c["id"] += "@" + audit_sha256
    need(receipt["plan"]["candidates"] == candidates and
         receipt["plan"]["selected_id"] == selected + "@" + audit_sha256,
         "audit candidate binding")
    need(receipt["metered_events"][1:] == audit["metered_events"], "audit resource binding")
    validate_model_entries(audit["bank_result"], audit["model_entries"], audit["metered_events"])
    begins = [e["payload"] for e in (json.loads(line) for line in Path(journal_path).read_bytes().splitlines())
              if e["kind"] == "begin"]
    operation = receipt["decision_id"] + ":native-stage:sra"
    attempt = [i for i, e in enumerate(begins) if e["operation"] == "record_resource_attempt" and
               e["args"] == [audit["request"]["arm"], operation,
                             {"jacobian": 1, "vjp_directions": 192, "svd": 2, "solve": 1}]]
    returns = [i for i, e in enumerate(begins) if e["operation"] == "record_resource_return" and
               e["args"] == [audit["request"]["arm"], operation]]
    selections = [i for i, e in enumerate(begins) if e["operation"] == "select" and
                  e["args"][1] == audit["request"]]
    model_attempts = []
    for event in audit["metered_events"]:
        positions = [i for i, e in enumerate(begins) if e["operation"] == "record_resource_attempt"
                     and e["args"] == [audit["request"]["arm"], event["operation_id"], event["amounts"]]]
        need(len(positions) == 1, "native operation identity")
        model_attempts.append(positions[0])
    need(len(attempt) == len(returns) == len(selections) == 1 and
         attempt[0] < model_attempts[0] < model_attempts[7] < returns[0] < model_attempts[8] < selections[0],
         "compiler stage budget binding")
    return {"journal": rebuilt, "audit_sha256": audit_sha256, "fixture": audit["fixture"],
            "selected_id": selected, "new_model_calls": 0, "new_physical_steps": 0,
            "equal_total_compute": False, "flops": None}
