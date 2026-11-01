"""Independent saved-data verification and descriptive factorial outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measures(state, goal):
    s, g = np.asarray(state, dtype=float), np.asarray(goal, dtype=float)
    if s.shape != (7,) or g.shape != (7,) or not np.isfinite([s, g]).all():
        raise ValueError("finite seven-coordinate evaluator state")
    return dict(object_position=float(np.linalg.norm(s[2:4]-g[2:4])),
                object_angle=float(abs(np.arctan2(np.sin(s[4]-g[4]), np.cos(s[4]-g[4])))),
                pusher_position=float(np.linalg.norm(s[:2]-g[:2])),
                native_position=float(np.linalg.norm(s[:4]-g[:4])),
                velocity_squared=float(np.sum((s[5:]-g[5:])**2)))


def analyze(directory):
    directory = Path(directory).resolve()
    read = lambda p: json.loads(p.read_text())
    protocol = read(directory / "protocol.json")
    plans = read(directory / "plans.json")
    physics = directory / "physics"
    report = read(physics / "report.json")
    binding = read(physics / "binding.json")
    if (plans["protocol_sha256"] != sha(directory / "protocol.json")
            or binding["protocol_sha256"] != plans["protocol_sha256"]
            or report["protocol_sha256"] != plans["protocol_sha256"]):
        raise ValueError("report binding")
    if report["plans_sha256"] != sha(directory / "plans.json") or binding["plans_sha256"] != sha(directory / "plans.json"):
        raise ValueError("plan bytes changed")
    expected = {f"{role['index']:02d}_{objective}_{search}" for role in protocol["source"]["roles"]
                for objective in protocol["objectives"] for search in protocol["searches"]}
    for roster in (plans["rows"], report["rows"]):
        if len(roster) != len(expected) or {r["arm"] for r in roster} != expected:
            raise ValueError("full fixed factorial roster")
    planned = {r["arm"]: r for r in plans["rows"]}
    rows = []
    for r in report["rows"]:
        if read(physics / (r["arm"] + ".json")) != r or r["error"] or r["status"] != "recorded":
            raise ValueError("incomplete branch")
        spec = planned[r["arm"]]
        actions = ([("warmup", i+1, [0., 0.]) for i in range(10)]
                   + [("prefix", i+1, a) for i, a in enumerate(spec["prefix"])]
                   + [("tail", i+1, a) for i, a in enumerate(np.asarray(spec["selected"]["actions"]).reshape(10, 2).tolist())])
        transitions = r["transitions"]
        if len(transitions) > len(actions) or len(transitions) <= 15:
            raise ValueError("invalid execution length")
        for t, (phase, step, a) in zip(transitions, actions):
            if t["phase"] != phase or t["step"] != step or not np.array_equal(t["action"], a):
                raise ValueError("action trace mismatch")
        if any(t["terminated"] or t["truncated"] for t in transitions[:-1]):
            raise ValueError("continued after stop")
        end = transitions[-1]
        if len(transitions) < 25 and not (end["terminated"] or end["truncated"]):
            raise ValueError("unexplained censoring")
        for t in transitions:
            if not np.isclose(t["reward"], -np.linalg.norm(np.asarray(t["state"])-report["goal_state_evaluation_only"]), rtol=0, atol=1e-9):
                raise ValueError("native reward inconsistent with saved state")
        root, prefix, terminal = [measures(transitions[i]["state"], report["goal_state_evaluation_only"]) for i in (9, 14, -1)]
        rows.append(dict(arm=r["arm"], root=root, prefix=prefix, terminal=terminal,
                         prefix_change={k: prefix[k]-root[k] for k in root},
                         tail_change={k: terminal[k]-prefix[k] for k in root},
                         tail_steps=len(transitions)-15, native_success=end["terminated"],
                         truncated=end["truncated"], native_reward=end["reward"],
                         object_displacement=float(np.linalg.norm(np.asarray(end["state"])[2:4]
                                                          -np.asarray(transitions[14]["state"])[2:4]))))
    images = set()
    for line in (physics / "journal.jsonl").read_text().splitlines():
        event = json.loads(line)
        if "file" not in event or "pixel_sha256" not in event:
            continue
        path = physics / event["file"]
        if path.parent != physics or path.is_symlink() or sha(path) != event["file_sha256"]:
            raise ValueError("PNG bytes")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        if hashlib.sha256(pixels.tobytes()).hexdigest() != event["pixel_sha256"]:
            raise ValueError("decoded RGB")
        images.add(event["file"])
    required = {f"{r['arm']}_{phase}_{step:02d}.png" for r in rows
                for phase, n in (("prefix", 5), ("tail", r["tail_steps"])) for step in range(1, n+1)}
    if images != required:
        raise ValueError("complete image roster")
    return dict(rows=rows, unique_images_verified=len(images),
                report_sha256=sha(physics / "report.json"), plans_sha256=sha(directory / "plans.json"),
                protocol_sha256=sha(directory / "protocol.json"),
                P1_admission=False, independent_roots=0,
                scope="one known development root, dependent arms, no inferential statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    with (args.directory / "analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    for row in result["rows"]:
        print(row["arm"], round(row["terminal"]["object_position"], 3),
              round(row["terminal"]["object_angle"], 3), round(row["native_reward"], 3),
              row["native_success"])
