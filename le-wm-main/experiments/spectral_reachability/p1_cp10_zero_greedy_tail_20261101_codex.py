"""Known-root zero/greedy by previously sealed common/zero tail; no neural calls."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_cp10_pair_common_tail_20261101_codex as old

ROOT = old.ROOT
NAME = "p1_cp10_zero_greedy_tail_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P1_CP10_ZERO_GREEDY_TAIL_20261101_CODEX.md"
PRIOR = ROOT / "results/p1_cp10_pair_common_tail_20261101_codex_run_CODEX01/prepared.json"
GREEDY = ROOT / "results/p1_three_prefix_own_host_20261101_main_root0_greedy_RUN_MAIN01/report.json"
ZERO = ROOT / "results/p1_three_prefix_own_host_20261101_main_root0_zero_RUN_MAIN01/report.json"
HASHES = {
    str(PRIOR): "db7c3dbeb9b03dfe050627163311b9193fe2c277d39bb2a67c90d87a32bd636a",
    str(GREEDY): "d63ce785be18aca3d89370cb86c7b4902ea1f1b99fac29cf160bdf9241ed3975",
    str(ZERO): "43cc73eb263e018f94501c5712f15affe486ccd7262d0530aaa9c6e76bd9020e",
}
PRIOR_BUILD = old.build


def schedule(zero, greedy, tail):
    result = []
    for candidate, (role, prefix) in enumerate((("zero", zero), ("greedy", greedy))):
        prefix = old.engine.actions(prefix, 5)
        for suffix_role, suffix in (("common", tail), ("zero", np.zeros((10, 2)))):
            suffix = old.engine.actions(suffix, 10)
            result.append(dict(index=len(result), candidate=candidate, prefix_role=role,
                               suffix_role=suffix_role, prefix=prefix, suffix=suffix,
                               commands=old.engine.actions(prefix + suffix, 15)))
    return result


def build():
    require = old.engine.require
    require(all(old.engine.sha(p) == h for p, h in HASHES.items()), "frozen inputs")
    previous = json.loads(PRIOR.read_bytes())
    # Re-run the previous source/selected-tail validation, not merely trust the schedule.
    prior_rebuilt = PRIOR_BUILD()
    require(all(prior_rebuilt[k] == v for k, v in previous.items() if k != "schema"),
            "prior prepared source and tail")
    old.EXPECTED.update({str(GREEDY): HASHES[str(GREEDY)], str(ZERO): HASHES[str(ZERO)]})
    gr, grow, gp, gc, ge = old.read_report(GREEDY)
    zr, zrow, zp, zc, _ = old.read_report(ZERO)
    require(np.array_equal(np.asarray(zp), np.zeros((5, 2))), "zero prefix")
    require(all(gc[0][k] == zc[0][k] == previous["root"][k]
                for k in ("state", "snapshot", "pixel_sha256", "goal")), "exact root")
    require(gr["cp10_goal_binding"] == zr["cp10_goal_binding"] == previous["goal_binding"],
            "goal binding")
    node = next(n for n in gr["nodes"][grow["episode_id"]] if n["boundary"] == 0)
    receipt = node["receipt"]
    trace = Path(receipt["trace"])
    require(node["status"] == "sealed" and old.engine.sha(trace) == receipt["trace_sha256"],
            "greedy trace")
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    matches = [e["payload"] for e in events if e["kind"] == "prefix_intervention"]
    require(len(matches) == 1, "unique selection")
    selection = matches[0]
    scores = np.asarray(selection["stage_scores"], dtype=np.float64)
    bank = np.asarray(selection["bank"], dtype=np.float32)
    require(selection["role"] == "greedy" and scores.shape == (8,) and np.isfinite(scores).all()
            and bank.shape == (8, 3, 5, 2), "scored bank")
    index = int(np.argmin(scores))
    require(np.array_equal(bank[index], np.asarray(selection["full_selected"], np.float32))
            and np.array_equal(bank[index, 0], np.asarray(gp, np.float32))
            and receipt["commands"] == gp, "first argmin and dispatched action")
    seals = [e for e in ge if e.get("episode_id") == grow["episode_id"]
             and e.get("kind") == "boundary_sealed" and e.get("boundary") == 0]
    dispatch = [e for e in ge if e.get("episode_id") == grow["episode_id"]
                and e.get("kind") == "authorized_dispatch" and e.get("step") == 1]
    require(len(seals) == len(dispatch) == 1 and seals[0]["index"] < dispatch[0]["index"]
            and seals[0]["plan_sha256"] == old.engine.digest(receipt), "pre-action node seal")
    ledger = receipt["ledger"]
    account = old.consumer.consume(ledger["journal"], ledger["sha256"],
                                    receipt["account"]["contract_sha256"],
                                    "structured", "node", "structured")
    require(account == receipt["account"], "greedy ledger")
    source = dict(previous["sources"])
    for path in (PRIOR, GREEDY, ZERO, GREEDY.parent / "journal.jsonl",
                 ZERO.parent / "journal.jsonl", trace, Path(ledger["journal"]),
                 DOC, Path(__file__).resolve()):
        source[str(path)] = old.engine.sha(path)
    require(all(old.engine.sha(p) == h for p, h in source.items()), "source closure")
    tail = previous["schedule"][0]["suffix"]
    return dict(schema=NAME + ".prepare.v1", schedule=schedule(zp, gp, tail), sources=source,
                goal_binding=previous["goal_binding"], source_prepared=previous["source_prepared"],
                source_prepare_sha256=previous["source_prepare_sha256"], root=previous["root"],
                prefixes=[{k: c[k] for k in ("state", "snapshot", "pixel_sha256")}
                          for c in (zc[5], gc[5])], greedy_index=index,
                greedy_stage_scores=scores.tolist(), tail_source=str(PRIOR),
                physical_outcomes_known_before_design=True, outcome_selected_for_this_run=False,
                independent_confirmation=False, P1_admission=False, neural_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    # Runtime rebinding is confined to this process; old sealed files are untouched.
    old.build = build
    old.NAME = NAME
    with old.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = old.run(args.output)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    if report["status"] != "recorded_not_independently_consumed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
