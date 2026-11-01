"""Independent saved native journal audit; never a full HOST strict receipt."""
import io
from pathlib import Path
import numpy as np

from experiments.spectral_reachability import p3_six_terminal_20261030_v1 as terminal
from experiments.spectral_reachability import p3_six_durable_transport_20261030_v1 as transport
from experiments.spectral_reachability import p3_six_saved_consumer_20261030_v1 as callbacks
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_saved_codex_20261029 as encoder
from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as meter

base = transport.base
need = base.need
TERMINAL_CAP = 4 * 1024**2


def _pinned(path, digest):
    raw = base.read_bytes(path, TERMINAL_CAP)
    need(base.fsha(raw) == digest, "terminal hash")
    value = base.decode(raw)
    need(type(value) is dict, "terminal object")
    return value


def _artifacts(directory, events, inventory):
    """Check referenced PNG/NPY bytes and retain named orphan inventory."""
    names = {item["name"]: item for item in inventory}
    refs = {}
    def visit(obj):
        if type(obj) is dict:
            if set(obj) in ({"file", "sha256", "pixel_sha256", "shape", "dtype"},
                            {"file", "sha256", "shape", "dtype", "origin", "role"}):
                name = obj["file"]
                need(type(name) is str and Path(name).name == name and name in names and
                     name not in (terminal.REPORT, terminal.COMPLETION, "events.jsonl"), "artifact name")
                need(name not in refs or refs[name] == obj, "descriptor identity")
                refs[name] = obj
            else:
                for value in obj.values():
                    visit(value)
        elif type(obj) is list:
            for value in obj:
                visit(value)
    for e in events:
        visit(e["detail"])
    for name, d in refs.items():
        path = directory / name
        if "pixel_sha256" in d:
            need(name.endswith(".png") and d["shape"] == [224, 224, 3] and d["dtype"] == "uint8", "PNG ABI")
            raw = base.read_bytes(path, transport.pub.PNG_CAP)
            need(base.fsha(raw) == d["sha256"] and
                 transport.pub.png_data(raw) == d["pixel_sha256"], "PNG content")
        else:
            shape, dtype, origin, role = d["shape"], d["dtype"], d["origin"], d["role"]
            need(name.endswith(".npy") and type(shape) is list and
                 all(type(n) is int and n >= 0 for n in shape) and
                 origin in ("numerical", "encoding") and dtype in ("<f4", "|u1"), "array ABI")
            if origin == "encoding":
                need((role, tuple(shape), dtype) in base.ENC_SHAPES, "encoding shape")
            else:
                need(dtype == "<f4" and np.prod(shape, dtype=object) <= 768, "numeric shape")
            size = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
            raw = base.read_bytes(path, size + 4096)
            need(base.fsha(raw) == d["sha256"], "array hash")
            stream = io.BytesIO(raw)
            version = np.lib.format.read_magic(stream)
            need(version in ((1, 0), (2, 0)), "array version")
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            actual, fortran, dt = read(stream)
            need(list(actual) == shape and not fortran and dt.str == dtype and
                 len(raw) - stream.tell() == size, "array header")
            stream.seek(0)
            array = np.load(stream, allow_pickle=False)
            need(stream.tell() == len(raw) and np.isfinite(array).all(), "array finite EOF")
    orphans = sorted(set(names) - set(refs) - {"events.jsonl"})
    return dict(referenced=len(refs), orphans=orphans)


def _native_slot(events, slot):
    """Independent, fail-closed meter nesting and stop accounting."""
    episode = f"matched_slot_{slot:02}"
    counts = {k: dict(attempted=0, returned=0, failed=0) for k in meter.CAPS}
    stack = []
    stopped = closed = False
    dispatches, returns, observations, encodings = [], [], [], []
    for e in events:
        r, d, kind = e["resource"], e["detail"], e["kind"]
        if e["origin"] == "physical" and r in ("native_control_return", "native_warmup_return"):
            need(d.get("episode_id") == episode and type(d.get("terminated")) is bool and
                 type(d.get("truncated")) is bool, "native stop ABI")
            stopped |= d["terminated"] or d["truncated"]
        if e["origin"] == "physical" and r == "native_initial_native_evaluation":
            need(d.get("episode_id") == episode and type(d.get("success")) is bool, "initial success ABI")
            stopped |= d["success"]
        if e["origin"] == "physical" and r == "native_authorized_dispatch":
            need(not stopped and not closed and d.get("episode_id") == episode and
                 type(d.get("step")) is int and d["step"] == len(dispatches) + 1 and
                 type(d.get("raw_bits")) is str and len(d["raw_bits"]) == 16 and
                 all(c in "0123456789abcdef" for c in d["raw_bits"]) and
                 np.isfinite(np.frombuffer(bytes.fromhex(d["raw_bits"]), dtype="<f4")).all(),
                 "ordered native dispatch")
            dispatches.append(e)
        if e["origin"] == "physical" and r == "native_control_return":
            need(d.get("step") == len(returns) + 1 and len(dispatches) > len(returns),
                 "ordered native return")
            before = dispatches[len(returns)]
            need(before["detail"].get("raw_bits") == d.get("raw_bits") and
                 before["index"] < e["index"], "native dispatch/return raw binding")
            returns.append(e)
        if e["origin"] == "physical" and r == "native_fresh_observation":
            need(d.get("episode_id") == episode, "observation episode")
            observations.append(e)
        if e["origin"] == "encoding" and r == "encoding_complete":
            encodings.append(e)
        if e["origin"] != "physical" or not r.startswith("native_meter/"):
            continue
        name = r.removeprefix("native_meter/")
        need(name in meter.CAPS and kind in ("intent", "return", "failure") and
             set(d) == ({"episode_id", "region", "units", "error"} if kind == "failure" else
                        {"episode_id", "region", "units"}) and
             d["episode_id"] == episode and type(d["units"]) is int and d["units"] == 1 and
             d["region"] in meter.REGIONS[name], "native meter ABI")
        ident = (e["token"], name, d["region"])
        c = counts[name]
        if kind == "intent":
            need(not closed and not (stopped and name == "public_step"), "no dispatch after stop/close")
            c["attempted"] += 1
            need(c["attempted"] <= meter.CAPS[name], "native cap")
            stack.append(ident)
        else:
            need(stack and stack.pop() == ident, "nested native meter")
            c["returned" if kind == "return" else "failed"] += 1
            if name == "close":
                closed = True
    need(len(dispatches) <= 20 and len(returns) <= len(dispatches) and len(observations) <= 32 and
         len(encodings) <= 4, "native slot bounds")
    return dict(slot=slot, native_counts=counts, pending_meter=[list(x) for x in stack],
                native_stopped=stopped, closed=closed, dispatches=len(dispatches),
                returned_actions=len(returns), observations=len(observations),
                encodings=len(encodings), full_saved_strict=False)


def _complete_slot(events, slot, result, directory):
    """Bind completed callback packets to recorded native actions and frames."""
    native = _native_slot(events, slot)
    counts = native["native_counts"]
    need(not native["pending_meter"] and not native["native_stopped"] and native["closed"] and
         all(x["failed"] == 0 and x["attempted"] == x["returned"] for x in counts.values()) and
         counts["constructor"]["returned"] == counts["close"]["returned"] == 1 and
         counts["reset"]["returned"] == 2 and counts["goal_bind"]["returned"] == 1 and
         counts["public_step"]["returned"] == 30, "complete native meter")
    observations = [e for e in events if e["origin"] == "physical" and
                    e["resource"] == "native_fresh_observation"]
    warm = [e for e in observations if e["detail"]["receipt"]["phase"] == "warmup"]
    plan = [e for e in observations if e["detail"]["receipt"]["phase"] == "plan"]
    need(len(observations) == 32 and [e["detail"]["receipt"]["step"] for e in warm] == list(range(11)) and
         [e["detail"]["receipt"]["step"] for e in plan] == list(range(21)), "complete native observations")
    dispatch = [e for e in events if e["origin"] == "physical" and e["resource"] == "native_authorized_dispatch"]
    returned = [e for e in events if e["origin"] == "physical" and e["resource"] == "native_control_return"]
    need(len(dispatch) == len(returned) == 20 and len(result["packets"]) == 4, "complete native actions")
    for j, (send, ret) in enumerate(zip(dispatch, returned, strict=True)):
        raw = np.asarray(result["packets"][j // 5]["raw"][j % 5], dtype="<f4")
        need(raw.shape == (2,) and np.isfinite(raw).all() and
             raw.tobytes().hex() == send["detail"]["raw_bits"] == ret["detail"]["raw_bits"] and
             send["index"] < plan[j + 1]["index"] < ret["index"], "packet/native action binding")
    encoded = [e for e in events if e["origin"] == "encoding" and e["resource"] == "encoding_complete"]
    need([e["detail"]["boundary"] for e in encoded] == [5, 10, 15, 20], "encoding boundaries")
    for e in encoded:
        d = e["detail"]
        boundary = d["boundary"]
        operations = [x for x in events if x["origin"] == "encoding" and
                      x["detail"].get("boundary") == boundary]
        need([(x["resource"], x["kind"]) for x in operations] ==
             [("encoder", "intent"), ("encoder", "return"), ("projector", "intent"),
              ("projector", "return"), ("encoding_complete", "record")], "encoding lifecycle")
        frame = d["frame"]
        observation = plan[boundary]["detail"]["receipt"]
        need(Path(observation["path"]).name == frame["file"] and
             observation["file_sha256"] == frame["sha256"] and
             observation["pixel_sha256"] == frame["pixel_sha256"] and
             returned[boundary - 1]["index"] < e["index"], "native frame/encoding binding")
        encoder.inspect(directory, {k: frame[k] for k in ("file", "sha256", "pixel_sha256")},
                        d["arrays"], boundary)
    return dict(slot=slot, native_action_packet_binding=True, encoded_frames=4,
                source_root_independently_verified=False, full_saved_strict=False)


def consume(directory, report_sha, completion_sha):
    """Consume pinned snapshot, verify inventory and audit per-slot native records."""
    directory = Path(directory).absolute()
    need(directory.is_dir() and not directory.is_symlink() and directory.resolve() == directory, "output directory")
    report = _pinned(directory / terminal.REPORT, report_sha)
    completion = _pinned(directory / terminal.COMPLETION, completion_sha)
    need(report.get("schema") == terminal.NAME + ".report" and
         completion == dict(schema=terminal.NAME + ".completion", report_sha256=report_sha,
                            boundary="after_report_fsync", scientific_pass=False,
                            full_saved_strict=False, complete=False) and
         report.get("scientific_pass") is False and report.get("full_saved_strict") is False,
         "terminal binding and no certification")
    need(type(report["cap"]) is int and 0 < report["cap"] <= 16 * 1024**3, "bounded inventory cap")
    all_files = terminal.inventory(directory, report["cap"])
    observed = [x for x in all_files if x["name"] not in (terminal.REPORT, terminal.COMPLETION)]
    need(observed == report["inventory"] and
         {x["name"] for x in all_files} == {x["name"] for x in observed} |
         {terminal.REPORT, terminal.COMPLETION}, "exact snapshot inventory")
    path = directory / "events.jsonl"
    if report["journal_sha256"] is None:
        need(not path.exists(), "missing journal identity")
        j = dict(events=[], tail=None, pending={}, trailing_bytes=0, trailing_sha256=None,
                 transport_complete=False)
    else:
        j = transport.inspect(path, report["journal_sha256"], report["cap"])
    need(len(j["events"]) == report["event_count"] and j["tail"] == report["journal_tail"] and
         j["trailing_bytes"] == report["trailing_bytes"] and j["trailing_sha256"] == report["trailing_sha256"] and
         [dict(origin=k[0], token=k[1], resource=v) for k, v in sorted(j["pending"].items())] == report["pending"] and
         terminal.roster(j["events"]) == report["rows"], "independent terminal reconstruction")
    artifacts = _artifacts(directory, j["events"], observed)
    spans = [[] for _ in range(6)]
    active = None
    for event in j["events"]:
        if event["origin"] == "physical" and event["resource"] == "six_slot_attempt":
            active = event["detail"]["slot"]
        elif event["origin"] == "physical" and event["resource"] == "six_slot_return":
            active = None
        elif (event["origin"] == "physical" and event["resource"].startswith("native_")) or event["origin"] == "encoding":
            need(active is not None, "native/encoding outside slot")
        if active is not None:
            spans[active].append(event)
    slots = [_native_slot(span, i) if span else None for i, span in enumerate(spans)]
    callback_audit = None
    if len([e for e in j["events"] if e["resource"] == "six_slot_return" and e["origin"] == "physical"]) == 6:
        callback_audit = callbacks.consume_transport(j["events"])
        for row in callback_audit["rows"]:
            if row["status"] == "callback_feedback_complete_not_certified" and row["cleanup"] == "returned":
                slots[row["slot"]]["complete_checks"] = _complete_slot(
                    spans[row["slot"]], row["slot"], row["result"], directory)
    return dict(status="native_records_checked_not_full_strict", rows=report["rows"], slots=slots,
                journal_transport_complete=j["transport_complete"], trailing_bytes=j["trailing_bytes"],
                artifacts=artifacts, callback_audit=callback_audit, source_authenticated=False,
                physical_certified=False, full_saved_strict=False, scientific_pass=False)
