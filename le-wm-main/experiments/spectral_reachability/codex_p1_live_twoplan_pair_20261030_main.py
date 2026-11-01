"""Join verified live two-stage outcomes to the fixed H10 intervention panel."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_adam_paired_join_20261030_main as common
from . import codex_p1_live_twoplan_saved_20261030_main as consumer

ROOT = common.ROOT
NAME = "codex_p1_live_twoplan_pair_20261030_main"
VIEW = ROOT / common.SOURCES["view"][0]
VIEW_SHA = common.SOURCES["view"][1]
PREP = ROOT / "results/codex_p1_live_twoplan_20261030_main_prepare_RUN01.json"
PREP_SHA = "012d7658e24c34c343bb2a8d22a175ea7df4e12944e6d4035eed9dad946347f4"
RUN = ROOT / "results/codex_p1_live_twoplan_20261030_main_run_RUN01"
RUN_SHA = "ebab08ed6f82764733cbeda16284b11a0883a62437256dbd9757241aad46e8d3"
SAVED = ROOT / "results/codex_p1_live_twoplan_saved_20261030_main_RUN01.json"
SAVED_SHA = "778bc41ccaee74cc82ce455d8e1f8d42bbd4b58dfa2358b3d7a79bc7789c7500"
need = common.need


def read(path, expected):
    need(path.is_file() and not path.is_symlink(), "source missing or symlink")
    data = path.read_bytes()
    need(hashlib.sha256(data).hexdigest() == expected, f"source hash: {path}")
    return json.loads(data)


def join(view, prepared, saved):
    need(len(view["rows"]) == len(prepared["roster"]) == len(saved["rows"]) == 24,
         "fixed denominator")
    need(saved["status"] == "saved_checks_passed" and saved["P1_admission"] is False,
         "saved status")
    rows = []
    for v, p, s in zip(view["rows"], prepared["roster"], saved["rows"], strict=True):
        ident = v["identity"]
        need(all(ident[k] == p[k] == s[k] for k in common.IDS), "row identity")
        need(all(ident[k] == p[k] for k in ("task_id", "physical_root_sha256", "prefix5_float32_hex")),
             "task, root or prefix identity")
        cp5 = p["checkpoint"]
        need(ident["cp5_file_sha256"] == cp5["file_sha256"]
             and ident["cp5_pixel_sha256"] == cp5["pixel_sha256"]
             and ident["cp5_state"] == cp5["state"]
             and ident["cp5_snapshot_sha256"] == consumer.saved.run.base.spec.digest(cp5["snapshot"]),
             "checkpoint identity")
        need(s["status"] in ("probe_completed", "native_stopped")
             and (s["endpoint_metrics"] is not None) == (s["status"] == "probe_completed"),
             "stop censor policy")
        zero = [x for x in v["common_suffixes"] if x["probe"] == "zero"]
        need(len(v["common_suffixes"]) == 42 and len(zero) == 1, "common probe denominator")
        rows.append({**v, "live_twoplan": {
            "status": s["status"], "accepted_suffix_steps": s["accepted_suffix_steps"],
            "early_stop": s["early_stop"], "native_stop_at_horizon": s["native_stop_at_horizon"],
            "endpoint_metrics": s["endpoint_metrics"], "last_observed_metrics": s["last_observed_metrics"],
            "endpoint_minus_own_zero_suffix": common.delta(
                s["endpoint_metrics"], zero[0]["common_suffix_endpoint_native_metrics"]),
            "stage_resources_code_path": s["stage_resources"], "physical_resources": s["physical_resources"]}})
    zeros = {}
    for row in rows:
        ident = row["identity"]
        if ident["role"] == "zero":
            need(ident["logical_pair"] not in zeros, "duplicate zero prefix")
            zeros[ident["logical_pair"]] = row
    need(len(zeros) == 8, "zero prefix denominator")
    for row in rows:
        ident = row["identity"]
        zero = zeros[ident["logical_pair"]]
        need(all(ident[k] == zero["identity"][k] for k in ("root_slot", "condition", "arm", "task_id")),
             "paired zero identity")
        row["live_twoplan"]["endpoint_minus_zero_prefix_live_endpoint"] = common.delta(
            row["live_twoplan"]["endpoint_metrics"], zero["live_twoplan"]["endpoint_metrics"])
    return rows


def build():
    view, prepared, saved = read(VIEW, VIEW_SHA), read(PREP, PREP_SHA), read(SAVED, SAVED_SHA)
    verified = consumer.consume(RUN, RUN_SHA, PREP, PREP_SHA)
    need(saved == verified, "saved consumer reconstruction")
    rows = join(view, prepared, saved)
    return {"schema": NAME, "status": "descriptive_source_bound_join", "rows": rows,
            "sources": {"view_sha256": VIEW_SHA, "prepare_sha256": PREP_SHA,
                        "run_sha256": RUN_SHA, "saved_sha256": SAVED_SHA},
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": ["known roots and repeated trajectories", "H10 run provenance has known gap",
                            "not a new physical experiment", "declared stage costs not independently hooked",
                            "common suffix effects are not independently optimized planning ability",
                            "no complete total budget or independent confirmation"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    out = Path(ap.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_"),
         "output scope")
    result = build()
    with out.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"rows": len(result["rows"]), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
