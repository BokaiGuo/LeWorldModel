"""Saved-only fixed-80 descriptive cross-tab; no neural or physical execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPORAL = ROOT / "results/agent_p5_temporal_independent_scalars_20261029_v1_MAIN01.json"
NATIVE = ROOT / "results/agent_p5_fixed80_error_decomposition_20261029_v1_MAIN01.json"
PINS = ("9552571dc77ff819d54143dcdf402b39c08e684945325d3448ee39c3d01768cb",
        "b74240a0c9a10f7c9b938f799b91be090d61a2f027f522c33c1743047582ddfc")
CELLS = ("T_AA", "T_AR", "T_RA", "T_RR")
NATIVE_CLASSES = ("same", "opposite", "both_zero", "one_zero", "missing")
JEPA_CLASSES = ("positive", "negative", "zero", "null")
KEYS = ("slot", "root_slot", "condition", "coord", "h", "minus_index", "plus_index")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"SHA mismatch: {path}")
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("duplicate JSON key")
            d[k] = v
        return d
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sign(value):
    if value is None:
        return "null"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid cosine")
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def agreement(a, b):
    sa, sb = sign(a), sign(b)
    if "null" in (sa, sb):
        return "missing"
    if sa == sb == "zero":
        return "both_zero"
    if "zero" in (sa, sb):
        return "one_zero"
    return "same" if sa == sb else "opposite"


def analyze(temporal, native):
    if temporal["status"] != "all164_temporal_fourcell_and80_pair_independent_scalars_verified":
        raise ValueError("temporal status")
    if temporal["pair_count"] != 80 or temporal["branches"] != 164:
        raise ValueError("temporal denominator")
    if native["fixed_denominator"] != 80 or native["branch_denominator"] != 164:
        raise ValueError("native denominator")
    if any(x.get(k) is not False for x in (temporal, native) for k in ("P1_admission", "P5_admission")):
        raise ValueError("admission boundary")
    tp, np, rows = temporal["pairs"], native["pairs"], temporal["rows"]
    if len(tp) != 80 or len(np) != 80 or len(rows) != 164:
        raise ValueError("incomplete panel")
    out_rows = []
    for i, (t, n) in enumerate(zip(tp, np)):
        slot, within = divmod(i, 20)
        half, coord = divmod(within, 10)
        minus = slot * 41 + 1 + 2 * within
        expected = (slot, slot // 2, ("off", "on")[slot % 2], coord,
                    (0.01, 0.001)[half], minus, minus + 1)
        if tuple(t[k] for k in KEYS) != expected or tuple(n[k] for k in KEYS) != expected:
            raise ValueError(f"pair identity or schedule mismatch {i}")
        for index in (minus, minus + 1):
            r = rows[index]
            if (r["index"], r["root_slot"], r["condition"], r["task_id"]) != (
                index, expected[1], expected[2], n["task_id"]
            ):
                raise ValueError(f"branch identity mismatch {index}")
        category = n["direction_agreement"]["native_distance"]
        if category not in NATIVE_CLASSES or category != agreement(
            n["native_plus_minus"]["native_distance"], n["lewm_actual_Q_plus_minus"]
        ):
            raise ValueError("native category")
        signs = {}
        for cell in CELLS:
            c = t["cells"][cell]
            s = sign(c["cosine"])
            if s != c["cosine_sign"]:
                raise ValueError("cosine sign mismatch")
            signs[cell] = s
        out_rows.append({**{k: t[k] for k in KEYS}, "task_id": n["task_id"],
                         "native_distance_vs_actual_Q": category, "jepa_cosine_sign": signs})
    tables = {}
    for group in ("all", "slot0", "slot1", "slot2", "slot3"):
        subset = out_rows if group == "all" else [r for r in out_rows if r["slot"] == int(group[-1])]
        tables[group] = {}
        for cell in CELLS:
            table = {a: {b: 0 for b in JEPA_CLASSES} for a in NATIVE_CLASSES}
            for row in subset:
                table[row["native_distance_vs_actual_Q"]][row["jepa_cosine_sign"][cell]] += 1
            tables[group][cell] = {"denominator": len(subset), "counts": table}
    return {"schema": "p5_temporal_native_cross_tab_20261030_codex_v1",
            "status": "saved_descriptive_cross_tab_only",
            "input_sha256": list(PINS), "branch_denominator": 164, "pair_denominator": 80,
            "known_roots": 2, "new_neural_calls": 0, "new_physics_calls": 0,
            "P1_admission": False, "P5_admission": False,
            "boundary": "Post-hoc; inherited upstream strict; no independent root, model inference, causal or control-benefit claim. JEPA cosine compares predicted and observed feature responses; native category compares physical distance with LeWM actual-encoded Q, not LeWM predicted response.",
            "pairs": out_rows, "tables": tables}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.name.startswith("p5_temporal_native_cross_tab_20261030_codex_v1_"):
        raise ValueError("output must use exclusive new stem")
    t, n = load(TEMPORAL, PINS[0]), load(NATIVE, PINS[1])
    result = analyze(t, n)
    # Recheck pinned source bytes after analysis; never overwrite a prior artifact.
    load(TEMPORAL, PINS[0])
    load(NATIVE, PINS[1])
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps({"path": str(args.output), "sha256": digest(data), "pairs": 80}))


if __name__ == "__main__":
    main()
