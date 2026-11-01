"""Read-only numeric checks of saved H3 candidates; no neural execution."""
import argparse
import json
from pathlib import Path
import numpy as np

from experiments.spectral_reachability import codex_p2_symmetric_h3_run_20261030_v2 as run

NAME = "codex_p2_symmetric_h3_saved_checks_20261030_v1"
need = run.bridge.host.need


def check_candidates(case, row, candidates):
    tags = ["nominal"] + [f"i{i}_a{alpha}" for i in range(2) for alpha in run.ALPHAS] + ["replay0", "replay1"]
    need([x["tag"] for x in candidates] == tags, "fixed candidate order")
    q = np.asarray(case["projection"], dtype=np.float64)
    target = q @ np.asarray(case["goal"], dtype=np.float64)
    for x in candidates:
        run.bridge.validate_logits(x["logits"], run.CAP)
        need(len(x["contexts"]) == 3, "three saved contexts")
        z = np.asarray(x["contexts"][-1]["latent_history"][-1], dtype=np.float64)
        projected = q @ z
        need(np.array_equal(projected, np.asarray(x["projected_terminal"], np.float64)),
             "saved projected terminal")
        cost = float(np.mean((projected-target)**2))
        need(cost == x["cost"], "saved projected cost")
    winner = candidates[0]
    for i in range(2):
        base = winner
        for x in candidates[1+4*i:5+4*i]:
            if row["arm"] == "suffix":
                cut = 10 if i == 0 else 20
                need(x["logits"][:cut] == base["logits"][:cut], "suffix prefix changed")
            if x["cost"] < winner["cost"]:
                winner = x
    need(winner == row["winner"], "strict stable winner")
    for x in candidates[-2:]:
        need(x["logits"] == winner["logits"] and x["contexts"] == winner["contexts"] and
             x["projected_terminal"] == winner["projected_terminal"] and x["cost"] == winner["cost"],
             "full replay mismatch")
    return dict(root=row["root"], arm=row["arm"], winner_tag=winner["tag"],
                projected_cost=winner["cost"], candidates=9, replays=2)


def check(directory, report_sha, prepared, prepare_sha):
    directory = Path(directory)
    producer_audit = run.audit(directory, report_sha)
    _, cases = run.verify_prepare(prepared, prepare_sha)
    report = run.bridge.host.frontier.read(directory/"report.json")
    need(report["prepare_sha256"] == prepare_sha, "report prepare identity")
    groups = {}
    with (directory/"events.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)["event"]
            if "root" in item and "arm" in item and item["event"]["kind"] == "candidate":
                groups.setdefault((item["root"], item["arm"]), []).append(item["event"]["witness"])
    rows = [check_candidates(cases[row["root"]], row, groups[(row["root"], row["arm"])])
            for row in report["rows"]]
    return dict(schema=NAME+".receipt", status="saved_numeric_checks_only", report_sha256=report_sha,
                prepare_sha256=prepare_sha, producer_audit=producer_audit, rows=rows,
                neural_reexecution=False, jacobian_reconstruction=False, proposal_reconstruction=False,
                equal_total_compute=False, P1_admission=False, P2_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--report-sha", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--prepare-sha", required=True)
    p.add_argument("--receipt", required=True)
    a = p.parse_args()
    path = Path(a.receipt)
    need(path.is_absolute() and path.name.startswith(NAME+"_") and
         path.parent.resolve() == path.parent and not path.exists() and not path.is_symlink(),
         "exclusive receipt")
    receipt = check(a.directory, a.report_sha, a.prepared, a.prepare_sha)
    run.bridge.host.bank._write(path, receipt)
    print(json.dumps(dict(status=receipt["status"], sha256=run.digest_file(path))))


if __name__ == "__main__":
    main()
