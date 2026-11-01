"""Independent saved join of outer host identity and six-slot source/math audit."""
from pathlib import Path

from experiments.spectral_reachability import p3_six_host_driver_20261030_v1 as driver

base, need = driver.base, driver.need
NAME = "p3_six_host_driver_saved_20261030_v1"


def _pinned(path, digest):
    need(type(digest) is str and len(digest) == 64 and
         all(c in "0123456789abcdef" for c in digest), "external record SHA")
    raw = base.read_bytes(path, driver.MAX_RECORD)
    need(base.fsha(raw) == digest, "outer record SHA mismatch")
    return base.decode(raw)


def consume(directory, *, begin_sha256, end_sha256, plan_sha256, source_sha256,
            producer_sha256, bridge_sha256, driver_sha256, report_sha256, completion_sha256):
    root = Path(directory)
    need(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(),
         "canonical outer directory")
    need({p.name for p in root.iterdir()} == {"begin.json", "end.json", "run"} and
         (root / "run").is_dir() and not (root / "run").is_symlink(), "complete exclusive outer inventory")
    begin = _pinned(root / "begin.json", begin_sha256)
    end = _pinned(root / "end.json", end_sha256)
    need(type(begin) is dict and set(begin) == {"schema", "plan", "plan_sha256", "source_sha256",
         "producer_sha256", "bridge_sha256", "driver_sha256", "identity"} and
         begin["schema"] == driver.NAME + ".begin", "begin schema")
    plan = driver.validate_plan(begin["plan"])
    need(base.digest(plan) == plan_sha256 == begin["plan_sha256"] and
         begin["source_sha256"] == source_sha256 and begin["producer_sha256"] == producer_sha256 and
         begin["bridge_sha256"] == bridge_sha256 and begin["driver_sha256"] == driver_sha256 and
         driver.own_sha() == driver_sha256 and driver.identity.own_sha() == bridge_sha256 and
         driver.producer.own_sha() == producer_sha256, "outer source and plan pins")
    need(type(end) is dict and set(end) == {"schema", "begin_sha256", "receipt", "guard_calls", "peaks",
         "physical_certified", "full_saved_strict", "scientific_pass"} and
         end["schema"] == driver.NAME + ".end" and end["begin_sha256"] == begin_sha256 and
         end["physical_certified"] is end["full_saved_strict"] is end["scientific_pass"] is False,
         "end schema and binding")
    receipt = end["receipt"]
    need(type(receipt) is dict and receipt.get("schema") == driver.identity.NAME + ".receipt" and
         receipt.get("before") == receipt.get("after") == begin["identity"] and
         receipt.get("source_contract_sha256") == source_sha256 and
         receipt.get("producer_sha256") == producer_sha256 and receipt.get("bridge_sha256") == bridge_sha256 and
         receipt.get("identity_is_attestation") is False and receipt.get("physical_certified") is False and
         receipt.get("full_saved_strict") is False and receipt.get("scientific_pass") is False,
         "identity receipt binding")
    terminal = receipt.get("terminal")
    need(type(terminal) is dict and terminal.get("report_sha256") == report_sha256 and
         terminal.get("completion_sha256") == completion_sha256, "terminal pins")
    peaks = end["peaks"]
    need(type(end["guard_calls"]) is int and end["guard_calls"] > 0 and type(peaks) is dict and
         set(peaks) == {"allocated", "reserved"} and
         all(type(peaks[k]) is int and peaks[k] >= 0 for k in peaks) and
         peaks["allocated"] <= plan["cuda_allocated_bytes"] and
         peaks["reserved"] <= plan["cuda_reserved_bytes"], "guard and peak limits")
    contract, _, _ = driver.producer.strict.source_material()
    snap = begin["identity"]
    need(type(snap) is dict and snap.get("model_hash", {}).get("original_state_dict") ==
         contract["expected_model_hash"] and snap.get("precision") == contract["precision"] and
         type(snap.get("device")) is dict and bool(snap["device"].get("uuid")),
         "saved identity and source contract")
    report = driver.producer.strict.native_saved._pinned(root / "run" / driver.producer.terminal.REPORT,
                                                          report_sha256)
    need(all(report[k] == plan[k] for k in ("limits", "cap", "reserve", "floor", "wall")),
         "saved resource profile")
    need(sum(p.stat().st_size for p in (root / "begin.json", root / "end.json")) <= plan["outer_cap"],
         "outer total cap")
    result = driver.producer.consume(root / "run", report_sha256, completion_sha256,
                                     source_sha256, producer_sha256)
    return dict(schema=NAME + ".consume", source_math=result, outer_identity_joined=True,
                identity_is_attestation=False, full_saved_strict=False,
                physical_certified=False, scientific_pass=False)
