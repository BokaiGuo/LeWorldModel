"""Descriptive same-plan model/physical rankings, retaining censored candidates."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_fixed_suffix_censored_consume_20261030 as consumer
from . import codex_p1_fixed_suffix_censored_runner_20261030 as runner

REPORT_SHA = "c40d47780ee866489a8d96aa439060bd05149aef76ac7458bb1fe4fa3be739d7"
DIRECTORY = runner.ROOT / "results/codex_p1_fixed_suffix_censored_runner_20261030_run_MAIN01"
MODEL = runner.checks.DIRECTORY / "report.json"


def analyze(physical, model):
    runner.need(len(physical["rows"]) == 32 and len(model["rows"]) == 8, "fixed roster")
    groups = []
    for i, model_row in enumerate(model["rows"]):
        rows = physical["rows"][4*i:4*i+4]
        runner.need(len(model_row["scores"]) == 4 and
                    all(r["plan"]["slot"] == model_row["slot"] and
                        r["arm"] == model_row["arm"] and r["role"] == f"candidate_{j}" and
                        r["plan"]["candidate_index"] == model_row["indices"][j]
                        for j, r in enumerate(rows)), "model/physical exact roster")
        candidates = []
        for j, row in enumerate(rows):
            result = dict(candidate=j, candidate_index=row["plan"]["candidate_index"],
                          model_score=model_row["scores"][j], status=row["status"],
                          endpoint=None)
            if row["status"] == "horizon_completed":
                response = row["responses"][-1]
                runner.need(response["step"] == 15, "actual step15 endpoint")
                metric = response["native_metrics"]
                result["endpoint"] = dict(object_position_error=metric["object_position_error"],
                                          wrapped_object_angle_error=metric["wrapped_object_angle_error"],
                                          native_success=response["native_success"])
            candidates.append(result)
        comparisons = {}
        for key in ("object_position_error", "wrapped_object_angle_error"):
            concordant = discordant = ties = 0
            for a in range(4):
                for b in range(a+1, 4):
                    if candidates[a]["endpoint"] is None or candidates[b]["endpoint"] is None:
                        continue
                    score_delta = candidates[a]["model_score"] - candidates[b]["model_score"]
                    physical_delta = candidates[a]["endpoint"][key] - candidates[b]["endpoint"][key]
                    if score_delta == 0 or physical_delta == 0:
                        ties += 1
                    elif score_delta * physical_delta > 0:
                        concordant += 1
                    else:
                        discordant += 1
            comparisons[key] = dict(concordant=concordant, discordant=discordant, ties=ties)
        groups.append(dict(slot=model_row["slot"], arm=model_row["arm"],
                           selected_index=model_row["selected_index"],
                           candidates=candidates, comparisons=comparisons))
    return dict(schema="p1_fixed_suffix_rank_analysis.v1", groups=groups, known_roots=2,
                independence_claim=False, causal_claim=False, P1_admission=False,
                note="8 dependent model groups; no censor imputation; exploratory, not fair-budget evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    runner.need(out.is_absolute() and out.parent == runner.ROOT / "results" and
                out.name.startswith("codex_p1_fixed_suffix_rank_analysis_20261030_") and
                not out.exists() and not out.is_symlink(), "exclusive output")
    report_path = DIRECTORY / "report.json"
    runner.need(runner.sha(report_path) == REPORT_SHA and runner.sha(MODEL) == runner.checks.REPORT_SHA,
                "external report hashes")
    report = runner.native.read(report_path, cap=64 * 1024**2)
    binding = runner.native.read(DIRECTORY / "execution_binding.json")
    prepared = runner.native.read(binding["prepared"], cap=64 * 1024**2)
    runner.need(runner.sha(binding["prepared"]) == binding["prepare_sha256"], "prepare hash")
    consumer.check(DIRECTORY, report, prepared)
    model = runner.native.read(MODEL, cap=64 * 1024**2)
    result = analyze(report, model)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with out.open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(output=str(out), sha256=hashlib.sha256(data).hexdigest())))


if __name__ == "__main__":
    main()
