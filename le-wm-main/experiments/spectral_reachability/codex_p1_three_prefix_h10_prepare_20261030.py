"""Saved-only fixed three-prefix ten-step design; no executor or inference."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import codex_p1_prefix_null_matched_20261030 as null
from . import p1_backtrack_capability_prepare_20261030 as historical

ROOT = null.ROOT
NAME = "codex_p1_three_prefix_h10_prepare_20261030"
NULL_PREP = ROOT / "results/codex_p1_prefix_null_matched_20261030_prepare_CODEX02.json"
NULL_SHA = "b87a6459420395c438816d7af7a3828ee70db9f42a15548e9b57320f55dfcabc"
ANALYSIS = ROOT / "results/analyze_mixedmode_MAIN_20261028_v1_MAIN01.json"
ANALYSIS_SHA = "0788833b58af36e49c4794be90def317688c3590c2d873f81cfc631f2226c830"
MIXED_SHA = "d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2"
MIXED_REPORT = ROOT / "results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json"
DESIGN = ROOT.parent / "docs/D2P_P1_THREE_PREFIX_H10_CODEX_20261030.md"
TEST = ROOT / "tests/test_codex_p1_three_prefix_h10_prepare_20261030.py"
need, sha, native = null.need, null.sha, null.native


def bank(selected_tail):
    need(type(selected_tail) is bytes and len(selected_tail) == 80, "ten-step selected tail")
    actions = [("zero", np.zeros(20, dtype="<f4").tobytes())]
    for coordinate in range(20):
        for sign in (1, -1):
            logits = np.zeros(20, dtype=np.float32)
            logits[coordinate] = np.float32(sign * .25)
            actions.append((f"{sign:+d}_{coordinate}", np.tanh(logits).astype("<f4").tobytes()))
    actions.append(("original_selected_tail", selected_tail))
    result = []
    for i, (name, raw) in enumerate(actions):
        vector = np.frombuffer(raw, dtype="<f4")
        need(len(raw) == 80 and np.isfinite(vector).all() and (np.abs(vector) <= 1).all(), "probe bounds")
        result.append(dict(index=i, name=name, raw10_float32_hex=raw.hex(),
                           raw10_sha256=hashlib.sha256(raw).hexdigest()))
    return result


def assemble(null_p, old_p, analysis):
    need(analysis["schema"] == "MAIN.mixedmode222_analysis.v1" and
         len(null_p["anchors"]) == len(old_p["anchors"]) == 16 and
         len(analysis["rows"]) == len(analysis["strict_receipt"]["plans"]) == 4 and
         analysis["input_sha256"][str(MIXED_REPORT)] == MIXED_SHA
         and analysis["strict_receipt"]["status"] == "all4_mixedmode_commands_verified_not_admitted"
         and analysis["strict_receipt"]["errors"] == [], "greedy analysis source")
    plans = native.read(null.PREP, cap=32*1024**2)["plans"]
    null.source_saved.runner.check_roster(native.read(null.PREP, cap=32*1024**2))
    anchors, banks, groups = [], [], {}
    for logical in range(8):
        selected, zero = null_p["anchors"][2*logical:2*logical+2]
        inc, old_selected = old_p["anchors"][2*logical:2*logical+2]
        selected_plan = plans[2*logical]
        slot = logical // 2
        row, original = analysis["rows"][slot], analysis["strict_receipt"]["plans"][slot]
        need((selected["index"], zero["index"], inc["index"], old_selected["index"]) ==
             (2*logical, 2*logical+1, 2*logical, 2*logical+1), "anchor order")
        need((selected["role"], zero["role"], inc["role"], old_selected["role"]) ==
             ("selected_prefix", "zero_prefix", "incumbent", "model_selected"), "anchor roles")
        identity = ("root_slot", "condition", "arm", "task_id", "physical_root_sha256")
        need(all(selected[k] == zero[k] == inc[k] == old_selected[k] for k in identity)
             and row["root_slot"] == original["root_slot"] == selected["root_slot"]
             and row["condition"] == original["condition"] == selected["condition"]
             and original["task_id"] == selected["task_id"], "same root task and arm")
        need(row["selected_is_greedy"] is True and row["selected_index"] == row["greedy_index"] ==
             original["selected_index"] and original["whole_run_report_sha256"] == MIXED_SHA
             and original["selection_uses_physics"] is False, "stage greedy identity missing")
        greedy_raw = np.asarray(original["prefix_commands"], dtype="<f4").tobytes()
        need(len(greedy_raw) == 40 and inc["prefix5_float32_hex"] == greedy_raw.hex() and
             selected["prefix5_float32_hex"] == old_selected["prefix5_float32_hex"] and
             zero["prefix5_float32_hex"] == bytes(40).hex(), "literal prefix identity")
        raw = bytes.fromhex(selected_plan["raw15_float32_hex"])
        need(raw[:40].hex() == selected["prefix5_float32_hex"], "selected tail source")
        probe_bank = bank(raw[40:])
        banks.append(probe_bank)
        for role, src in (("selected", selected), ("zero", zero), ("greedy", inc)):
            a = copy.deepcopy(src)
            a["source_anchor_index"], a["source_role"] = src["index"], src["role"]
            a["index"], a["role"], a["logical_pair"] = len(anchors), role, logical
            key = (a["task_id"], a["physical_root_sha256"], a["condition"], a["prefix5_float32_hex"])
            a["duplicate_prefix_task_group"] = groups.setdefault(key, len(groups))
            a["source_family"] = "MAIN02_incumbent" if role == "greedy" else "v2_MAIN01"
            a["greedy_bank_index"] = row["greedy_index"] if role == "greedy" else None
            anchors.append(a)
    need(len(anchors) == 24 and len(banks) == 8 and all(len(b) == 42 for b in banks), "full roster")
    schedule = [dict(index=len_ , logical_pair=i, anchor=3*i+r, probe=j)
                for len_, (i, r, j) in enumerate((i, r, j) for i in range(8)
                                                  for r in range(3) for j in range(42))]
    return dict(schema=NAME+".prepare.v1", anchors=anchors, probe_banks=banks, schedule=schedule,
                fixed_pairs=8, fixed_anchors=24, probes_per_anchor=42, fixed_branches=1008,
                greedy_definition="original 222-bank minimum predicted stage cost, stable index tie break",
                selected_tail_source="fixed selected plan steps 6-15, post-outcome exploratory",
                caps_if_complete_no_early_stop=dict(constructor=1008, close=1008, reset=2016,
                    set_state=4032, goal_setter=3024, goal_render=2016, goal_bind=1008,
                    public_step=25200, physics_step=256032, explicit_render=81648),
                cost_scope="design arithmetic under inherited adapter; requires actual attempted/completed ledger",
                historical_proposal_and_shared_cost="unknown_not_zero", execution_implemented=False,
                saved_consumer_implemented=False, P1_admission=False, independent_confirmation=False,
                freshness=False, equal_budget_certified=False, new_neural_calls=0, new_physical_calls=0)


def build():
    null_p, _ = null.verify(NULL_PREP, NULL_SHA)
    need(sha(ANALYSIS) == ANALYSIS_SHA, "analysis SHA")
    analysis = native.read(ANALYSIS, cap=16*1024**2)
    old_p = historical.build()
    result = assemble(null_p, old_p, analysis)
    bindings = copy.deepcopy(null_p["bindings"])
    for name, digest in old_p["bindings"].items():
        need(name not in bindings or bindings[name] == digest, "source collision")
        bindings[name] = digest
    for path in (NULL_PREP, ANALYSIS, MIXED_REPORT, Path(__file__).resolve(), DESIGN, TEST):
        need(path.is_file() and not path.is_symlink(), "bound regular file")
        bindings[str(path)] = sha(path)
    need(bindings[str(NULL_PREP)] == NULL_SHA and bindings[str(ANALYSIS)] == ANALYSIS_SHA and
         bindings[str(MIXED_REPORT)] == MIXED_SHA and
         all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    result["bindings"] = bindings
    return result


def prepare(output):
    path = Path(output)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME+"_") and not path.is_symlink(), "exclusive prepare")
    data = native.raw(build())
    with path.open("xb") as stream:
        need(stream.write(data) == len(data), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    native.paired.fsync_dir(path.parent)
    return sha(path)


def verify(path, digest):
    path = Path(path)
    native.actual_hash(digest)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME+"_") and path.is_file() and not path.is_symlink()
         and sha(path) == digest, "prepare identity")
    value = native.read(path, cap=64*1024**2)
    need(value == build() and sha(path) == digest, "prepare/source drift")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        need(args.sha256 is None, "prepare arguments")
        digest = prepare(args.output)
    else:
        need(args.sha256 is not None, "verify digest")
        verify(args.output, args.sha256)
        digest = args.sha256
    print(json.dumps(dict(sha256=digest, status="prepared" if args.command == "prepare" else "verified",
                          fixed_branches=1008, execution_implemented=False, P1_admission=False)))


if __name__ == "__main__":
    main()
