"""Evaluate a non-privileged symmetric-FD curvature proxy against branch labels."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v8_v7_e0_fd_branch_stability_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V8_V7_E0_FD_CURVATURE_PROXY_DEV_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v8_v7_e0_fd_curvature_proxy_dev_20261104_codex.json"


def summarize(rows):
    scores = np.asarray([r["proxy"] for r in rows])
    labels = np.asarray([r["branch_changed"] for r in rows], dtype=bool)
    absolute = np.asarray([r["absolute_midpoint"] for r in rows])
    denominator = np.asarray([r["difference_norm"] for r in rows])
    pos, neg = scores[labels], scores[~labels]
    auc = None
    if len(pos) and len(neg):
        auc = float(np.mean((pos[:, None] > neg[None, :]) + .5*(pos[:, None] == neg[None, :])))
    predicted = scores > .5
    median = lambda values: float(np.median(values)) if len(values) else None
    return {"coordinates": len(rows), "positive": int(labels.sum()), "negative": int((~labels).sum()),
            "proxy_median_positive": median(pos), "proxy_median_negative": median(neg),
            "abs_median_positive": median(absolute[labels]), "abs_median_negative": median(absolute[~labels]),
            "difference_median_positive": median(denominator[labels]),
            "difference_median_negative": median(denominator[~labels]),
            "difference_below_1e-8": int((denominator < 1e-8).sum()), "auc": auc,
            "threshold_gt_0_5": {"tp": int(np.sum(predicted & labels)), "fp": int(np.sum(predicted & ~labels)),
                                 "tn": int(np.sum(~predicted & ~labels)), "fn": int(np.sum(~predicted & labels))}}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    labels = json.loads(source.OUT.read_text())
    seal.verify(labels["dependencies"])
    deps = seal.capture([Path(__file__), PROTOCOL, source.OUT, Path(source.__file__),
                         source.geometry.v8.RESULT, source.geometry.panel.OUT])
    plans = {"v8": {r["root"]: r for r in json.loads(source.geometry.v8.RESULT.read_text())["rows"]},
             "v7": {r["root"]: r for r in json.loads(source.geometry.panel.OUT.read_text())["rows"]}}
    rows = []
    for item in labels["rows"]:
        batch = plans[item["arm"]][item["root"]]["trace"][item["trace_index"]]
        positions = np.asarray(batch["positions"])
        if positions.shape != (21, 2):
            raise ValueError("not a response probe")
        middle = np.linalg.norm(positions[1:11]+positions[11:21]-2*positions[0], axis=1)
        difference = np.linalg.norm(positions[1:11]-positions[11:21], axis=1)
        unstable = set(item["unstable_coordinates"])
        for i in range(10):
            rows.append({"arm": item["arm"], "root": item["root"], "layer": item["layer"],
                         "batch": item["index"], "coordinate": i, "branch_changed": i in unstable,
                         "absolute_midpoint": float(middle[i]), "difference_norm": float(difference[i]),
                         "proxy": float(middle[i]/(difference[i]+1e-8))})
    summary = {arm+"_"+layer: summarize([r for r in rows if r["arm"] == arm and r["layer"] == layer])
               for arm in plans for layer in ("first", "successor")}
    by_root = [{**{"arm": arm, "root": root},
                **summarize([r for r in rows if r["arm"] == arm and r["root"] == root
                             and r["layer"] == "successor"])}
               for arm in plans for root in plans[arm]]
    seal.verify(deps)
    source.geometry.panel.source.write(OUT, {"dependencies": deps, "rows": rows, "summary": summary,
                                             "successor_by_root": by_root,
                                             "scope": "posthoc exposed tasks; proxy not used by planner"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
