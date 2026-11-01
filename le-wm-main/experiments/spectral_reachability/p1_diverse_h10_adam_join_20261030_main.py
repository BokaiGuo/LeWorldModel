"""Join verified H10 probes to separately optimized Adam suffixes, without imputation."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_diverse_h10_20261030_main as h10
from . import p1_diverse_h10_paired_20261030_main as paired
from . import codex_p1_h10_paired_table_20261030_main as metrics

NAME = "p1_diverse_h10_adam_join_20261030_main"
ROOT = h10.ROOT
SOURCES = {
    "old": ("results/codex_p1_adam_paired_join_20261030_main_RUN01.json",
            "7fd04ee76f6a5bfafdb9f6249db961bf6be6fb26829df19fc642438f88a9daa8"),
    "plan": ("results/p1_diverse_adam_cp5_20261030_main_RUN01.json",
             "f4b297e53dd82fac41089eb2751cd7636b6c9500710d3af02af2630623e6f9aa"),
    "physical": ("results/p1_diverse_adam_physical_20261030_main_RUN01/report.json",
                 "f6d98341e9bdd4fe86901c9864f3543fcc8a5ea37d2b234b9c4b588d2c6df3ef"),
    "old_prepare": ("results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json",
                    "ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30"),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def delta(a, b):
    return metrics.paired_difference(a, b)


def build(prepared, table, old, plan, physical, old_prepare):
    need(len(prepared["anchors"]) == 3 and len(table["rows"]) == 126
         and len(old["rows"]) == 24 and len(plan["rows"]) == 3
         and len(physical["rows"]) == 16, "fixed roster")
    need(not old["equal_total_budget_certified"] and not plan["equal_budget_certified"],
         "budget claim")
    rows = []
    for anchor, prediction, physical_index in zip(prepared["anchors"], plan["rows"], (5, 13, 15), strict=True):
        p = physical["rows"][physical_index]
        ident = ("root_slot", "condition", "arm", "task_id")
        need(all(anchor[k] == p[k] for k in ident)
             and anchor["physical_root_sha256"] == p["root_sha256"]
             and anchor["goal"] == p["goal"] and anchor["root_state"] == p["root_state"],
             "selected root or task identity")
        need(prediction["index"] == anchor["index"]
             and prediction["logical_pair"] == anchor["logical_pair"]
             and prediction["role"] == "selected", "plan identity")
        cp = [x for x in p["checkpoints"] if x["step"] == 5]
        need(len(cp) == 1 and cp[0]["state"] == anchor["checkpoint"]["state"]
             and cp[0]["snapshot"] == anchor["checkpoint"]["snapshot"]
             and cp[0]["pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"]
             == prediction["cp5_pixel_sha256"], "selected checkpoint")
        raw = bytes.fromhex(p["plan"]["raw15_float32_hex"])
        need(len(raw) == 120 and raw[:40].hex() == anchor["prefix5_float32_hex"]
             == prediction["prefix_raw_f32_hex"]
             and hashlib.sha256(raw[40:]).hexdigest() == prediction["selected_action_sha256"],
             "selected action identity")
        probes = [x for x in table["rows"] if x["selected"]["logical_pair"] == anchor["logical_pair"]]
        need(len(probes) == 42 and probes[0]["selected"]["probe"] == "zero"
             and all(x["selected"]["prefix_metrics"] ==
             metrics.physical_metrics(cp[0]["state"], anchor["goal"]) for x in probes), "probe identity")
        observed = metrics.physical_metrics(p["last_observation"]["state"], anchor["goal"])
        need(p["status"] in ("native_stopped", "horizon_completed")
             and p["own_returned"] == 15, "independent execution status")
        independent = {
            "status": p["status"], "accepted_suffix_steps": p["own_returned"] - 5,
            "endpoint": observed if p["status"] == "horizon_completed" else None,
            "last_observed": observed, "last_observed_step": p["last_observation"]["step"],
            "suffix_sha256": prediction["selected_action_sha256"],
            "planning_wall_seconds": prediction["planning_wall_seconds"],
            "planning_calls": "aggregate only; see source plan",
        }
        group = []
        for role in ("selected", "zero", "greedy"):
            common = []
            for probe in probes:
                item = probe["selected"] if role == "selected" else probe["controls"].get(role)
                if item is None:
                    continue
                common.append({"probe": probe["selected"]["probe"], "item": item})
            if role == "selected":
                prefix = probes[0]["selected"]["prefix_metrics"]
                ind = independent
            else:
                candidates = [r for r in old["rows"] if r["identity"]["logical_pair"] == anchor["logical_pair"]
                              and r["identity"]["role"] == role]
                need(len(candidates) == 1, "control roster")
                prior = candidates[0]
                identity = prior["identity"]
                old_anchor = old_prepare["original"]["anchors"][identity["index"]]
                need(all(identity[k] == anchor[k] for k in ident)
                     and identity["physical_root_sha256"] == anchor["physical_root_sha256"]
                     and old_anchor["role"] == role
                     and old_anchor["task_id"] == identity["task_id"]
                     and old_anchor["physical_root_sha256"] == identity["physical_root_sha256"]
                     and old_anchor["root_state"] == anchor["root_state"]
                     and old_anchor["goal"] == anchor["goal"]
                     and old_anchor["prefix5_float32_hex"] == identity["prefix5_float32_hex"]
                     and old_anchor["checkpoint"]["state"] == identity["cp5_state"]
                     and old_anchor["checkpoint"]["pixel_sha256"] == identity["cp5_pixel_sha256"]
                     and prior["root_metrics"] == probes[0]["selected"]["root_metrics"]
                     and prior["prefix_metrics"] == probes[0]["controls"][role]["prefix_metrics"],
                     "control identity")
                prefix = prior["prefix_metrics"]
                ind = prior["adam_independent_suffix"]
            endpoint = ind["endpoint"] if role == "selected" else ind["horizon_metrics"]
            zero_probe = common[0]["item"]["endpoint"]
            group.append({"root_slot": anchor["root_slot"], "condition": anchor["condition"],
                          "arm": anchor["arm"], "logical_pair": anchor["logical_pair"], "role": role,
                          "root_metrics": probes[0]["selected"]["root_metrics"],
                          "prefix_metrics": prefix,
                          "prefix_minus_root": delta(prefix, probes[0]["selected"]["root_metrics"]),
                          "common_suffixes": common, "independent_adam": ind,
                          "independent_endpoint": endpoint,
                          "independent_last_observed": (
                              ind["last_observed"] if role == "selected" else ind["last_observed_metrics"]),
                          "independent_minus_own_zero_suffix": delta(endpoint, zero_probe)})
        zero_endpoint = group[1]["independent_endpoint"]
        for row in group:
            row["independent_minus_zero_prefix"] = delta(row["independent_endpoint"], zero_endpoint)
        rows.extend(group)
    return {"schema": NAME, "rows": rows, "P1_admission": False,
            "independent_confirmation": False, "equal_total_budget_certified": False,
            "limitations": ["known roots and duplicate actions", "old control runtime provenance incomplete",
                            "common suffix tests action response, not optimized capability",
                            "Adam is exploratory and complete method budget not certified",
                            "native-stop fixed-horizon endpoint remains null"]}


def main():
    parser = argparse.ArgumentParser()
    for key in ("prepared", "prepare-sha", "table", "table-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    prepared, _ = h10.verify(args.prepared, args.prepare_sha)
    table_path = Path(args.table)
    need(table_path.is_absolute() and h10.sha(table_path) == args.table_sha, "table hash")
    table = h10.native.read(table_path, cap=64 * 1024**2)
    need(table["source_prepare_sha256"] == args.prepare_sha and not table["P1_admission"], "table binding")
    inputs = {}
    for name, (path, digest) in SOURCES.items():
        source = ROOT / path
        need(h10.sha(source) == digest, "source hash " + name)
        inputs[name] = h10.native.read(source, cap=128 * 1024**2)
    result = build(prepared, table, **inputs)
    result.update(table_sha256=args.table_sha, prepare_sha256=args.prepare_sha,
                  sources=SOURCES, code_sha256=h10.sha(Path(__file__)))
    need(h10.sha(table_path) == args.table_sha and
         all(h10.sha(ROOT / path) == digest for path, digest in SOURCES.values()), "source drift")
    print(json.dumps({"rows": len(result["rows"]), "sha256": h10.runner._write_new(output, result)}))


if __name__ == "__main__":
    main()
