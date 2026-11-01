"""Saved-only join for guarded host records; no live execution or attestation."""
from pathlib import Path

from experiments.spectral_reachability import p3_six_guarded_host_20261030_v1 as host
from experiments.spectral_reachability import p3_six_host_driver_saved_20261030_v1 as old_saved

need, base = host.need, host.base


def consume(directory, *, begin_sha256, end_sha256, plan_sha256, source_sha256, producer_sha256,
            bridge_sha256, driver_sha256, native_sha256, host_sha256, report_sha256, completion_sha256):
    root = Path(directory)
    need(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink() and
         {p.name for p in root.iterdir()} == {"begin.json", "end.json", "run"} and
         (root / "run").is_dir() and not (root / "run").is_symlink(), "exclusive complete outer inventory")
    begin = old_saved._pinned(root / "begin.json", begin_sha256)
    end = old_saved._pinned(root / "end.json", end_sha256)
    need(type(begin) is dict and set(begin) == {"schema", "plan", "plan_sha256", "source_sha256",
         "producer_sha256", "bridge_sha256", "driver_sha256", "native_sha256", "host_sha256", "identity"} and
         begin["schema"] == host.NAME + ".begin", "guarded begin schema")
    plan = host.old.validate_plan(begin["plan"])
    pins = dict(plan_sha256=plan_sha256, source_sha256=source_sha256, producer_sha256=producer_sha256,
                bridge_sha256=bridge_sha256, driver_sha256=driver_sha256, native_sha256=native_sha256,
                host_sha256=host_sha256)
    need(all(begin[k] == v for k, v in pins.items()) and base.digest(plan) == plan_sha256 and
         host.producer.own_sha() == producer_sha256 and host.identity.own_sha() == bridge_sha256 and
         host.old.own_sha() == driver_sha256 and host.file_sha(host.native) == native_sha256 and
         host.file_sha(host) == host_sha256, "guarded source and plan pins")
    need(type(end) is dict and set(end) == {"schema", "begin_sha256", "receipt", "guard_counts", "peaks",
         "scientific_pass", "physical_certified", "full_saved_strict"} and
         end["schema"] == host.NAME + ".end" and end["begin_sha256"] == begin_sha256 and
         end["scientific_pass"] is end["physical_certified"] is end["full_saved_strict"] is False,
         "guarded end binding")
    receipt = end["receipt"]
    need(type(receipt) is dict and receipt.get("schema") == host.identity.NAME + ".receipt" and
         receipt.get("before") == receipt.get("after") == begin["identity"] and
         receipt.get("source_contract_sha256") == source_sha256 and
         receipt.get("producer_sha256") == producer_sha256 and receipt.get("bridge_sha256") == bridge_sha256 and
         receipt.get("identity_is_attestation") is False and receipt.get("physical_certified") is False and
         receipt.get("full_saved_strict") is False and receipt.get("scientific_pass") is False,
         "identity receipt binding")
    terminal = receipt.get("terminal")
    need(type(terminal) is dict and terminal.get("report_sha256") == report_sha256 and
         terminal.get("completion_sha256") == completion_sha256, "terminal pins")
    counts = end["guard_counts"]
    need(type(counts) is dict and all(type(k) is str and type(v) is int and v > 0
         for k, v in counts.items()) and counts.get("physical_constructor_before") == 6 and
         counts.get("physical_constructor_after") == 6 and counts.get("encoding_before") == 24 and
         counts.get("encoding_after") == 24 and counts.get("numerical_call", 0) > 0,
         "required call-boundary guards")
    peaks = end["peaks"]
    need(type(peaks) is dict and set(peaks) == {"allocated", "reserved"} and
         all(type(peaks[k]) is int and peaks[k] >= 0 for k in peaks) and
         peaks["allocated"] <= plan["cuda_allocated_bytes"] and
         peaks["reserved"] <= plan["cuda_reserved_bytes"], "guarded peak ceilings")
    contract, _, _ = host.producer.strict.source_material()
    snap = begin["identity"]
    need(type(snap) is dict and snap.get("model_hash", {}).get("original_state_dict") ==
         contract["expected_model_hash"] and snap.get("precision") == contract["precision"] and
         type(snap.get("device")) is dict and bool(snap["device"].get("uuid")), "identity source binding")
    report = host.producer.strict.native_saved._pinned(root / "run" / host.producer.terminal.REPORT, report_sha256)
    need(all(report[k] == plan[k] for k in ("limits", "cap", "reserve", "floor", "wall")) and
         sum((root / x).stat().st_size for x in ("begin.json", "end.json")) <= plan["outer_cap"],
         "saved resource profile")
    result = host.producer.consume(root / "run", report_sha256, completion_sha256,
                                   source_sha256, producer_sha256)
    return dict(schema="p3_six_guarded_host_saved_20261030_v1.consume", source_math=result,
                guarded_identity_joined=True, identity_is_attestation=False, physical_certified=False,
                full_saved_strict=False, scientific_pass=False)
