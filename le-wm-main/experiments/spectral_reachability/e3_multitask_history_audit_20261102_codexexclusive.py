"""Read-only reconstruction of saved E3 feedback histories; no model or physics."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import e3_cross_task_identity_join_codex_20261030_v1 as cross
from . import e3_cell03_encoder_parity_inputs_codex_20261030_v1 as arrays
from . import agent_e3_actual_feedback_join_20261029_v2 as join
from . import agent_e3_multinode_saved_20261029_v1 as nodes


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_packet(controls, boundary, action):
    require(len(controls) >= boundary and
            [x["step"] for x in controls[boundary-5:boundary]] == list(range(boundary-4, boundary+1)),
            "packet control steps")
    require(all(not x["terminated"] and not x["truncated"]
                for x in controls[boundary-5:boundary]), "stopped packet")
    physical = bytes.fromhex("".join(x["raw_bits"] for x in controls[boundary-5:boundary]))
    require(physical == np.asarray(action, dtype="<f4").tobytes(), "executed action packet")
    return physical


def reconstruct(index, spec):
    directory, report_sha, completion_sha, authority, authority_sha = spec
    path = cross.RESULTS / directory
    host = cross.host
    inspected = host.inspect(str(path), report_sha, completion_sha,
                             str(cross.RESULTS / authority), authority_sha)
    cross.check_inspection(index, inspected)
    binding = host.authority(str(cross.RESULTS / authority), authority_sha)
    fixed = host.launch.episode.old.authority.load(*binding["args"])
    setup = host.launch.episode.old.setup_authority.resolve(
        binding["resolver"]["path"], binding["resolver"]["sha256"], index, fixed["body"])
    raw = (path / "events.jsonl").read_bytes()
    report = json.loads((path / "report.json").read_bytes())
    require(hashlib.sha256(raw).hexdigest() == report["journal_sha256"], "journal drift")
    events = [json.loads(line) for line in raw.splitlines()]

    def one(resource, boundary):
        found = [e for e in events if e["resource"] == resource
                 and e["detail"].get("boundary") == boundary]
        require(len(found) == 1, f"unique {resource} at {boundary}")
        return found[0]

    controls = [e["detail"] for e in events if e["resource"] == "native_control_return"]
    require([x["step"] for x in controls] == list(range(1, 21)), "control steps")
    require(all(not x["terminated"] and not x["truncated"] for x in controls),
            "unexpected stop in completed input")
    body = deepcopy(fixed["body"])
    prior = np.asarray(setup["prefix5"], np.float32)
    action = np.asarray(fixed["first5"], np.float32)
    rows = []
    for boundary in (10, 15):
        encoded = one("encoding_complete", boundary)
        frame = encoded["detail"]["frame"]
        require(Path(frame["file"]).name == frame["file"], "frame path")
        frame_path = path / frame["file"]
        require(not frame_path.is_symlink() and frame_path.is_file(), "frame file")
        png = frame_path.read_bytes()
        require(hashlib.sha256(png).hexdigest() == frame["sha256"], "frame hash")
        latent = arrays.checked_array(path, encoded["detail"]["arrays"]["latent"])
        ei = one("native_e3_encode_intent", boundary)
        er = one("native_e3_encode_return", boundary)
        ni = one("native_e3_node_intent", boundary)
        nr = one("native_e3_node_return", boundary)
        require(ei["index"] < encoded["index"] < er["index"] < ni["index"] < nr["index"],
                "node chronology")
        require(ei["detail"]["observation"]["file_sha256"] == frame["sha256"],
                "physical observation frame")
        require(np.asarray(er["detail"]["latent"], np.float32).tobytes() == latent.tobytes(),
                "returned latent")
        physical = check_packet(controls, boundary, action)
        context = deepcopy(body["root_context"])
        for key in ("latent_history", "action_history"):
            context[key] = np.asarray(context[key], np.float32)
        context = join.joined_context(
            context, prior, action, action, latent,
            np.asarray(body["action_mean"], np.float32), np.asarray(body["action_std"], np.float32),
            expected_rgb_sha=frame["sha256"], rgb_bytes=png, encoding_input_sha=frame["sha256"],
            episode_id=f"cell{index:02d}", physical_episode_id=f"cell{index:02d}",
            encoder_episode_id=f"cell{index:02d}", start=boundary-5, end=boundary,
            physical_end=boundary, encoding_boundary=boundary, stopped=False)
        body["root_context"] = dict(context, latent_history=context["latent_history"].tolist(),
                                    action_history=context["action_history"].tolist())
        body["feedback_boundary"] = boundary
        digest = nodes.bridge.host.digest(body)
        require(digest == ni["detail"]["body_sha256"] == nr["detail"]["body_sha256"],
                "reconstructed node body")
        next_bits = nr["detail"]["raw_bits"]
        next_physical = "".join(x["raw_bits"] for x in controls[boundary:boundary+5])
        require(next_bits == next_physical, "next node action execution")
        rows.append(dict(boundary=boundary, rgb_sha256=frame["sha256"],
                         latent_sha256=encoded["detail"]["arrays"]["latent"]["sha256"],
                         executed_packet_sha256=hashlib.sha256(physical).hexdigest(),
                         next_action_sha256=hashlib.sha256(bytes.fromhex(next_bits)).hexdigest(),
                         reconstructed_body_sha256=digest))
        prior = action.copy()
        action = np.frombuffer(bytes.fromhex(next_bits), dtype="<f4").copy().reshape(5, 2)
    costs = Counter(e["resource"] for e in events if e["resource"].startswith("native_meter/"))
    return dict(cell=index, state="saved_history_bound", nodes=rows,
                source=dict(report_sha256=report_sha, completion_sha256=completion_sha,
                            authority_sha256=authority_sha, journal_sha256=report["journal_sha256"]),
                cost_event_counts=dict(sorted(costs.items())),
                cost_note="event counts only; not wall time, total cost, or additive across nested meters")


def build():
    rows = []
    for index, spec in cross.SPECS.items():
        try:
            rows.append(reconstruct(index, spec))
        except Exception as exc:
            rows.append(dict(cell=index, state="audit_failed", error_type=type(exc).__name__,
                             error=str(exc)[:2000], nodes=None, cost_event_counts=None))
    return dict(schema="e3_multitask_history_audit_20261102_codexexclusive",
                fixed_denominator=16, inspected=list(cross.SPECS), rows=rows,
                uninspected=[i for i in range(16) if i not in cross.SPECS],
                complete=all(r["state"] == "saved_history_bound" for r in rows),
                stop_censor_policy="uninspected cells unknown; never impute later nodes",
                full_saved_strict=False, scientific_admission=False,
                scope="Saved bytes only; no new physical replay, encoder or planner recomputation.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    out = Path(p.parse_args().output)
    require(out.is_absolute() and out.parent == cross.RESULTS and
            out.name.startswith("e3_multitask_history_audit_20261102_codexexclusive_"),
            "exclusive output path")
    result = build()
    data = (json.dumps(result, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(complete=result["complete"], sha256=hashlib.sha256(data).hexdigest())))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
