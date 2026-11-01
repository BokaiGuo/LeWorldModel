"""CPU-only audit of recorded same-step actions, not a new physical experiment."""
import argparse
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path

from experiments.spectral_reachability import p2_h15_fourway_pairing_20261029_v1 as four

NAME = "p2_saved_action_fairness_20261030_codex"
METHODS = four.METHODS
old = four.three.old
DIRS = {"h15": old.H_DIR, "cem": old.C_DIR, "gn": four.three.G_DIR,
        "gradient": four.GRAD_DIR}


def action_table(events, root, expected):
    rows = [e for e in events if e.get("kind") == "returned_transition"
            and e.get("phase") == "plan" and e.get("branch") == root]
    if [e.get("step") for e in rows] != list(range(1, expected + 1)):
        raise ValueError("missing, duplicate or unordered returned action")
    result = {}
    for e in rows:
        bits = e.get("actual_bits")
        if (e.get("actual_dtype") != "<f4" or not isinstance(bits, str)
                or len(bits) != 16 or bits != e.get("requested_bits")):
            raise ValueError("action dtype, shape or requested/actual mismatch")
        try:
            raw = bytes.fromhex(bits)
            values = struct.unpack("<2f", raw)
        except (ValueError, struct.error) as exc:
            raise ValueError("malformed action bits") from exc
        if not all(math.isfinite(x) for x in values):
            raise ValueError("non-finite action")
        result[e["step"]] = bits
    return result


def compare_actions(tables, length=15):
    if set(tables) != set(METHODS):
        raise ValueError("method roster")
    pairs = {}
    for a, b in itertools.combinations(METHODS, 2):
        steps = []
        for step in range(1, length + 1):
            x, y = tables[a].get(step), tables[b].get(step)
            steps.append({"step": step, "same_action_bits": None if x is None or y is None else x == y,
                          a + "_bits": x, b + "_bits": y})
        pairs[a + "__" + b] = {
            "available_steps": sum(x["same_action_bits"] is not None for x in steps),
            "identical_steps": sum(x["same_action_bits"] is True for x in steps),
            "steps": steps}
    return pairs


def build():
    # Reconstruct source consumers first: their hashes and journal chains bind
    # the actions, goals, warm-up histories and physical outcomes.
    prior = four.build()
    events = {m: [json.loads(line) for line in (d / "journal.jsonl").read_text().splitlines()]
              for m, d in DIRS.items()}
    roots = []
    for p, resources in zip(prior["pairs"], prior["resources"], strict=True):
        root = p["root_slot"]
        if root != resources["root_slot"]:
            raise ValueError("resource root mismatch")
        tables = {m: action_table(events[m], root, p["last_step"][m]) for m in METHODS}
        roots.append({"root_slot": root, "root_sha256": p["root_sha256"],
                      "same_returned_steps": p["common_steps"],
                      "pairs": compare_actions(tables),
                      "neural_completed": resources["neural_completed"],
                      "neural_unknowns": resources["neural_unknowns"]})
    return {"schema": NAME + ".report", "source_hashes": prior["sources"], "roots": roots,
            "meaning": "bitwise equal float32 executed actions at the same returned control step; "
                       "not equal prior trajectories, state, or a causal action intervention",
            "budget_boundary": "recorded completed counts only; incomplete CPU accounting and unknown FLOPs; "
                               "do not sum nested calls or infer equal total compute",
            "equal_total_compute": False, "P1_admission": False, "P2_admission": False,
            "new_neural_calls": 0, "new_physical_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or not args.output.name.startswith(NAME + "_"):
        raise ValueError("absolute exclusive output namespace required")
    result = build()
    data = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(data).hexdigest(),
                      "identical_steps": {str(r["root_slot"]): {
                          k: v["identical_steps"] for k, v in r["pairs"].items()} for r in result["roots"]}}))


if __name__ == "__main__":
    main()
