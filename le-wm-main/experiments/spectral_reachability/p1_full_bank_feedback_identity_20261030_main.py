"""Bind full-bank physical prefixes to previously executed actual-RGB Adam suffixes."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_h10_paired_table_20261030_main as metrics

NAME = "p1_full_bank_feedback_identity_20261030_main"
ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "physical": ("results/p1_full_bank_physical_runner_20261030_main_RUN01/report.json",
                 "a5b81efb67a61c6bb5e96ac0dc5af5fb2b76e06a4d8982b40f73303d360c39b0"),
    "joined": ("results/codex_p1_adam_paired_join_20261030_main_RUN01.json",
               "7fd04ee76f6a5bfafdb9f6249db961bf6be6fb26829df19fc642438f88a9daa8"),
    "prepare": ("results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json",
                "ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30"),
}


def need(value, message):
    if not value:
        raise ValueError(message)


def build(physical, joined, prepare):
    need(len(physical["rows"]) == 16 and len(joined["rows"]) == 24
         and len(prepare["original"]["anchors"]) == 24, "fixed denominator")
    rows = []
    for index in range(8):
        item = physical["rows"][2 * index + 1]
        candidates = [x for x in joined["rows"]
                      if x["identity"]["logical_pair"] == index and x["identity"]["role"] == "selected"]
        need(len(candidates) == 1, "selected old roster")
        prior = candidates[0]
        ident = prior["identity"]
        anchor = prepare["original"]["anchors"][ident["index"]]
        cp = [x for x in item["checkpoints"] if x["step"] == 5]
        need(len(cp) == 1 and item["role"] == "model_selected"
             and item["root_slot"] == ident["root_slot"]
             and item["condition"] == ident["condition"]
             and item["arm"] == ident["arm"]
             and item["task_id"] == ident["task_id"] == anchor["task_id"]
             and item["root_sha256"] == ident["physical_root_sha256"] == anchor["physical_root_sha256"]
             and item["root_state"] == anchor["root_state"] and item["goal"] == anchor["goal"]
             and item["plan"]["raw15_float32_hex"][:80] == ident["prefix5_float32_hex"]
             == anchor["prefix5_float32_hex"]
             and cp[0]["state"] == ident["cp5_state"] == anchor["checkpoint"]["state"]
             and cp[0]["snapshot"] == anchor["checkpoint"]["snapshot"]
             and cp[0]["pixel_sha256"] == ident["cp5_pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"],
             "full prefix, task or checkpoint identity")
        controls = {}
        for role in ("selected", "zero", "greedy"):
            matches = [r for r in joined["rows"] if r["identity"]["logical_pair"] == index
                       and r["identity"]["role"] == role]
            need(len(matches) == 1, "control roster")
            control = matches[0]
            control_anchor = prepare["original"]["anchors"][control["identity"]["index"]]
            need(control_anchor["task_id"] == item["task_id"]
                 and control_anchor["root_state"] == item["root_state"]
                 and control_anchor["goal"] == item["goal"]
                 and control["identity"]["prefix5_float32_hex"] == control_anchor["prefix5_float32_hex"]
                 and control["identity"]["cp5_state"] == control_anchor["checkpoint"]["state"],
                 "control identity")
            controls[role] = {"prefix_metrics": control["prefix_metrics"],
                              "independent_adam": control["adam_independent_suffix"]}
        rows.append({"logical_pair": index, "physical_row": item["index"], "root_slot": item["root_slot"],
                     "condition": item["condition"], "arm": item["arm"],
                     "selected_candidate_index": item["plan"]["candidate_index"],
                     "prefix_identical_to_prior_adam": True,
                     "root_metrics": metrics.physical_metrics(item["root_state"], item["goal"]),
                     "prefix_metrics": metrics.physical_metrics(cp[0]["state"], item["goal"]),
                     "full_bank_open_loop_status": item["status"],
                     "full_bank_open_loop_last": metrics.physical_metrics(item["last_observation"]["state"], item["goal"]),
                     "controls": controls})
    return {"schema": NAME, "rows": rows, "P1_admission": False, "equal_budget_certified": False,
            "independent_confirmation": False,
            "limitations": ["post-outcome development; known roots; repeated prefixes",
                            "old independent-Adam wrapper runtime was not fully bound",
                            "prior Adam results have no certified total budget or planning isolation",
                            "native-stop fixed horizon endpoint stays null"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    data = {}
    for key, (path, digest) in SOURCES.items():
        raw = (ROOT / path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == digest, "source hash " + key)
        data[key] = json.loads(raw)
    result = build(**data)
    result.update(sources=SOURCES, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    need(all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
             for path, digest in SOURCES.values()), "source drift")
    raw = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with out.open("xb") as stream:
        need(stream.write(raw) == len(raw), "short write")
    print(json.dumps({"rows": len(result["rows"]), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
