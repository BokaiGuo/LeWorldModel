"""CPU-only paired audit of pinned, previously saved physical records."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEAM = "matched_domain_H2_beam20x6"
MPPI = "kernel_faithful_mppi_nominal"
TIMES = (0, 5, 10)
SOURCES = {
    "comparison": ("results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json", "eb661bfd89b75b4865b37ba517de4412f015472928a9ed3c2b76ec15367ed128"),
    "beam": ("results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01/report.json", "01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7"),
    "mppi": ("results/moving_prediction_matched_sampling_physical_20261027_v1_MAIN01/report.json", "8ce8634eddcd3e2e5b25d5b55572d80d06da1fd5d3513ee06aecec89001efa41"),
}
OUTPUT = ROOT / "results/codex_beam_mppi_physical_pair_20261030_LOCAL01.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value):
    require(isinstance(value, list) and len(value) == 7 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        for x in value), "invalid state or goal vector")
    return value


def load_sources():
    result = {}
    for name, (path, digest) in SOURCES.items():
        data = (ROOT / path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, "source hash mismatch: " + name)
        result[name] = json.loads(data)
    return result


def index(rows, arm):
    selected = [r for r in rows if r.get("arm") == arm]
    require(len(selected) == 2 and {r.get("root_slot") for r in selected} == {0, 1},
            "missing or duplicate root for " + arm)
    return {r["root_slot"]: r for r in selected}


def checkpoints(row):
    returned = row["own_returned"]
    require(type(returned) is int and 0 <= returned <= 10, "invalid returned count")
    status = row["status"]
    require(status in ("horizon_completed", "native_success", "native_stopped", "failed", "not_started"), "unknown status")
    if status == "horizon_completed":
        require(returned == 10 and row["first_error"] is None and not row["terminated"] and not row["truncated"], "inconsistent completion")
    if status == "not_started":
        require(returned == 0, "not started but returned steps")
    result = {}
    for cp in row["checkpoints"]:
        step = cp["step"]
        require(type(step) is int and step in TIMES and step not in result, "invalid or duplicate checkpoint")
        require(step <= returned and cp["phase"] == "plan", "checkpoint outside returned plan")
        state = vector(cp["state"])
        require(state == vector(cp["observation_state"]) == vector(cp["snapshot"]["state"]), "checkpoint state mismatch")
        result[step] = cp
    return result


def metrics(state, goal):
    d = [a - b for a, b in zip(state, goal)]
    return {"agent_xy_error": math.hypot(*d[:2]),
            "object_xy_error": math.hypot(*d[2:4]),
            "all_xy_error": math.sqrt(sum(x*x for x in d[:4])),
            "wrapped_angle_error": abs((d[4] + math.pi) % (2 * math.pi) - math.pi),
            "velocity_error": math.hypot(*d[5:]),
            "state7_euclidean_raw": math.sqrt(sum(x*x for x in d))}


def audit(sources):
    summary = sources["comparison"]
    require(summary["all_expected_same_identity"] is True, "upstream identity not admitted")
    s = {a: index(summary["rows"], a) for a in (BEAM, MPPI)}
    p = {BEAM: index(sources["beam"]["rows"], BEAM), MPPI: index(sources["mppi"]["rows"], MPPI)}
    pairs = []
    for root in (0, 1):
        b, m = s[BEAM][root], s[MPPI][root]
        ident = b["identity"]
        require(ident == m["identity"] and b["identity_sha256"] == m["identity_sha256"], "identity mismatch")
        require(ident["source_slot"] == root and ident["packet_steps"] == 5 and ident["horizon"] == 2, "clock mismatch")
        goal, root_state = vector(ident["native_goal"]), vector(ident["native_root"])
        checks = {}
        for arm in (BEAM, MPPI):
            row, sr = p[arm][root], s[arm][root]
            require(vector(row["goal"]) == goal and vector(row["root_state"]) == root_state, "goal or root mismatch")
            require(row["selection_digest"] == sr["selection_digest"], "selection mismatch")
            require(row["status"] == sr["physical_status"] and row["own_returned"] == sr["own_returned"], "status mismatch")
            checks[arm] = checkpoints(row)
            # Each available source checkpoint must correspond to the pinned summary.
            sc = {c["step"]: c for c in sr["checkpoints"]}
            require(len(sc) == len(sr["checkpoints"]), "duplicate summary checkpoint")
            for step, cp in checks[arm].items():
                require(step in sc and cp["file_sha256"] == sc[step]["PNG_sha256"]
                        and cp["pixel_sha256"] == sc[step]["pixel_sha256"], "checkpoint provenance mismatch")
            if 0 in checks[arm]:
                require(checks[arm][0]["state"] == root_state, "checkpoint zero is not root")
        for step in TIMES:
            values, reasons = {}, {}
            for arm in (BEAM, MPPI):
                cp = checks[arm].get(step)
                if cp is None:
                    values[arm] = None
                    reasons[arm] = {"status": p[arm][root]["status"], "own_returned": p[arm][root]["own_returned"], "reason": "checkpoint_not_saved"}
                else:
                    values[arm] = metrics(cp["state"], goal)
            delta = None if reasons else {k: values[MPPI][k] - values[BEAM][k] for k in values[BEAM]}
            pairs.append({"root_slot": root, "identity_sha256": b["identity_sha256"], "step": step,
                          "values": values, "mppi_minus_beam": delta, "missing": reasons})
    return {"schema": "codex_beam_mppi_physical_pair_20261030_v1",
            "sources_sha256": {k: v[1] for k, v in SOURCES.items()},
            "fixed_pair_denominator": 6, "observed_pairs": sum(x["mppi_minus_beam"] is not None for x in pairs),
            "pairs": pairs, "clock": "returned own physical plan steps",
            "metric_note": "state7 raw Euclidean mixes units; wrapped angle is diagnostic, not native success",
            "boundaries": ["two known roots; not independent confirmation", "no new physical or neural runs",
                           "equal predict calls are not equal total compute; search/replay budgets and support differ",
                           "per-arm wall time and FLOPs unknown", "missing checkpoint is null, not failure or success",
                           "identity and RGB provenance inherit pinned upstream audit; no fullstrict rerun"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    encoded = (json.dumps(audit(load_sources()), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if args.verify:
        require(OUTPUT.read_bytes() == encoded, "output mismatch")
        print("verified", OUTPUT)
    else:
        with OUTPUT.open("xb") as f:
            f.write(encoded)
        print("created", OUTPUT)


if __name__ == "__main__":
    main()
