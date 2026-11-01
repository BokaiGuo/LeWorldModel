"""Describe fixed H10 same-suffix differences; not a planning-capability test."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_h10_multimetric_pair_20261030"
SOURCE = ROOT / "results/p1_diverse_h10_adam_join_20261030_main_RUN02.json"
SOURCE_SHA = "1bd0307e5677bf10619a3835600cd828b2d683b5bc8cab61719c828b8199aa04"
METRICS = ("object_position_error", "object_angle_error_wrapped",
           "pusher_position_error", "native_joint_xy_error")
TOL = 1e-6


def require(ok, message):
    if not ok:
        raise ValueError(message)


def summarize(values):
    require(values and all(math.isfinite(v) for v in values), "finite paired values")
    return {"lower": sum(v < -TOL for v in values),
            "equal_within_tolerance": sum(abs(v) <= TOL for v in values),
            "higher": sum(v > TOL for v in values),
            "mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def analyze(payload):
    rows = payload["rows"]
    require(len(rows) == 9 and payload["P1_admission"] is False, "source roster")
    out = []
    for logical_pair in (2, 6, 7):
        selected = [x for x in rows if x["logical_pair"] == logical_pair and x["role"] == "selected"]
        require(len(selected) == 1, "selected roster")
        selected = selected[0]
        for reference in ("zero", "greedy"):
            controls = [x for x in rows if x["logical_pair"] == logical_pair and x["role"] == reference]
            require(len(controls) == 1, "reference roster")
            control = controls[0]
            require(all(selected[k] == control[k] for k in ("logical_pair", "root_slot", "condition", "arm")),
                    "pair identity")
            a = {item["probe"]: item["item"] for item in selected["common_suffixes"]}
            b = {item["probe"]: item["item"] for item in control["common_suffixes"]}
            require(len(a) == 42 and len(b) == 41 and set(b) < set(a)
                    and set(a) - set(b) == {"original_selected_tail"} and "zero" in b, "probe roster")
            probes = sorted(set(b) - {"zero"})
            require(len(probes) == 40, "nonzero probe denominator")
            absolute = {key: [] for key in METRICS}
            response = {key: [] for key in METRICS}
            successes = {"selected_only": 0, "reference_only": 0, "both": 0, "neither": 0}
            for probe in probes:
                x, y = a[probe], b[probe]
                observation = y["observation"]
                require(x["status"] == y["status"] == observation["status"] == "probe_completed"
                        and x["censor"] is None and observation["censor"] is None
                        and x["accepted_steps"] == observation["accepted_probe_steps"] == 10,
                        "probe completion")
                for metric in METRICS:
                    difference = x["endpoint"][metric] - y["endpoint"][metric]
                    require(math.isclose(difference, y["selected_minus_control"][metric],
                                         rel_tol=1e-12, abs_tol=1e-12), "paired difference binding")
                    absolute[metric].append(difference)
                    response[metric].append(x["endpoint_minus_own_zero"][metric]
                                            - y["endpoint_minus_own_zero"][metric])
                sx, sy = x["endpoint"]["native_success"], y["endpoint"]["native_success"]
                require(type(sx) is bool and type(sy) is bool, "native success")
                successes["both" if sx and sy else "selected_only" if sx else
                          "reference_only" if sy else "neither"] += 1
            out.append({"logical_pair": logical_pair, "root_slot": selected["root_slot"],
                        "condition": selected["condition"], "arm": selected["arm"],
                        "reference": reference, "matched_nonzero_suffixes": len(probes),
                        "selected_minus_reference_absolute": {k: summarize(v) for k, v in absolute.items()},
                        "selected_minus_reference_own_zero_response": {k: summarize(v) for k, v in response.items()},
                        "native_success_pairs": successes,
                        "selected_prefix_minus_root": selected["prefix_minus_root"],
                        "reference_prefix_minus_root": control["prefix_minus_root"]})
    return {"schema": NAME, "source_sha256": SOURCE_SHA, "rows": out, "tolerance": TOL,
            "limitations": ["two known roots, shared trajectories and duplicate logical arms",
                            "same suffix effect is not independent planned capability",
                            "source controls inherited incomplete runtime source binding",
                            "post-outcome descriptive analysis"],
            "independent_confirmation": False, "fairness_proven": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    require(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
            and not out.exists(), "exclusive output")
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, "source hash")
    result = analyze(json.loads(raw))
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with out.open("xb") as stream:
        require(stream.write(content) == len(content), "short write")
    print(json.dumps({"pairs": len(result["rows"]), "sha256": hashlib.sha256(content).hexdigest()}))


if __name__ == "__main__":
    main()
