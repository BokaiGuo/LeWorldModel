"""Post-hoc physical two-metric tradeoffs from a pinned saved E1 panel."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/agent_e1_cross_panel_physical_pair_20261029_v1_MAIN01.json"
SHA = "e9024bb6a0f9c604e65dd04488e81fed05cfe940ed85a2b5d76e1d25f1602c08"
OUTPUT = ROOT / "results/codex_e1_cross_panel_tradeoff_20261030_LOCAL01.json"
METRICS = ("object_xy", "wrapped_object_angle_error")


def classify(a, b):
    """Compare forced minus subgoal; exact ties remain ties."""
    if a is None or b is None:
        return "missing"
    if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (*a, *b)):
        raise ValueError("invalid physical metric")
    signs = [(y > x) - (y < x) for x, y in zip(a, b, strict=True)]
    if signs == [0, 0]:
        return "tie"
    if all(x <= 0 for x in signs):
        return "forced_dominates"
    if all(x >= 0 for x in signs):
        return "subgoal_dominates"
    return "tradeoff"


def analyze(data):
    result = data["result"]
    pairs = result["pairs"]
    if result["denominator"] != 54 or len(pairs) != 54:
        raise ValueError("fixed denominator mismatch")
    if len(result["joins"]) != 2 or [x["root"] for x in result["joins"]] != [0, 1]:
        raise ValueError("root joins mismatch")
    joins = {j["root"]: j["initial_exact"] for j in result["joins"]}
    keys = set()
    rows = []
    for p in pairs:
        key = (p["root"], p["step"], p["subgoal_arm"], p["forced_arm"])
        if key in keys or p["root"] not in (0, 1) or p["step"] not in (0, 5, 10):
            raise ValueError("duplicate or invalid pair")
        keys.add(key)
        if p["status"] not in ("observed", "missing"):
            raise ValueError("invalid status")
        observed = p["status"] == "observed"
        if observed:
            if not joins[p["root"]] or p["missing"] or p["subgoal"] is None or p["forced"] is None:
                raise ValueError("invalid observed join")
            if p["subgoal"]["goal"] != p["forced"]["goal"]:
                raise ValueError("goal mismatch")
            a = [p["subgoal"][m] for m in METRICS]
            b = [p["forced"][m] for m in METRICS]
            category = classify(a, b)
        else:
            if not p["missing"]:
                raise ValueError("missing reason required")
            a = b = None
            category = "missing"
        rows.append(dict(root=key[0], step=key[1], subgoal_arm=key[2], forced_arm=key[3],
                         subgoal=a, forced=b, category=category,
                         missing=p["missing"], same_raw10=p["same_raw10"]))
    for root in (0, 1):
        for step in (0, 5, 10):
            group = [r for r in rows if r["root"] == root and r["step"] == step]
            if len(group) != 9 or len({r["subgoal_arm"] for r in group}) != 3 or len({r["forced_arm"] for r in group}) != 3:
                raise ValueError("incomplete 3x3 grid")
    if sum(r["category"] != "missing" for r in rows) != result["observed"]:
        raise ValueError("observed count mismatch")
    counts = {}
    for root in (0, 1):
        for step in (0, 5, 10):
            group = [r for r in rows if r["root"] == root and r["step"] == step]
            counts[f"root{root}_step{step}"] = {
                k: sum(r["category"] == k for r in group)
                for k in ("forced_dominates", "subgoal_dominates", "tradeoff", "tie", "missing")
            }
    return {"source_sha256": SHA, "metrics": list(METRICS), "rows": rows, "counts": counts,
            "scope": "post-hoc known-root same-task same-time physical diagnostics; not equal-budget or independent confirmation",
            "trust": "inherits pinned cross-panel saved analysis and its upstream authentication; no reconstruction of upstream receipt",
            "fresh_roots": 0, "new_physics": 0, "new_neural_calls": 0}


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError("source SHA mismatch")
    value = analyze(json.loads(raw))
    blob = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as f:
        f.write(blob)
    print(OUTPUT, hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main()
