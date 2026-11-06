"""Give every sealed recursive candidate the same limited suffix opportunity."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_discovery_attribution_dev as panel
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem

OUT = panel.OUT / "candidate_suffix_quality_posthoc"
PROTOCOL = panel.ROOT.parent / "docs/PC_CANDIDATE_SUFFIX_QUALITY_POSTHOC_PROTOCOL_20261104_CODEX.md"
ARMS = ("spectral", "random_direction", "diverse")
COUNT = 120


def deps():
    return panel.seal.capture([Path(__file__), PROTOCOL, Path(panel.__file__),
        Path(panel.prior.__file__), Path(cem.__file__), Path(panel.journal.__file__),
        Path(panel.journal.write_json.__code__.co_filename), Path(panel.prior.journal_audit.__file__),
        Path(panel.prior.exact.__file__), Path(panel.prior.exact.wall.__file__),
        Path(panel.prior.exact.wall.physics.__file__),
        Path(panel.prior.old.OLD.previous.independent.__code__.co_filename)])


def verify_generation_dependencies(manifest):
    """Verify the sealed generation environment, recording this audit repair.

    The original run sealed this script too.  The only post-run edit is the
    audit-field fix; excluding exactly this file keeps every other dependency
    fail-closed while making the repair explicit in the audit receipt.
    """
    sealed = manifest["dependencies"]
    script_path = str(Path(__file__).resolve())
    expected = sealed.get("files", {}).get(script_path)
    if expected is None:
        panel.seal.verify(sealed)
        return None
    current = panel.seal.digest(script_path)
    if current == expected:
        panel.seal.verify(sealed)
        return None
    repaired = dict(sealed)
    repaired["files"] = dict(sealed["files"])
    del repaired["files"][script_path]
    panel.seal.verify(repaired)
    return {"path": script_path, "sealed_sha256": expected,
            "audit_sha256": current,
            "reason": "post-run audit comparison fix: calls list compared by len"}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    dependency_seal = deps()
    panel.journal.write_json(OUT / "manifest.json", {"dependencies": dependency_seal,
        "arms": ARMS, "count": COUNT, "scope": "posthoc candidate-quality diagnosis"})
    source_manifest = json.loads((panel.OUT / "manifest.json").read_text())
    for root, task in enumerate(panel.prior.tasks()):
        for arm in ARMS:
            source = json.loads((panel.OUT / f"plan_{root:02d}_{arm}.json").read_text())
            row = {"root": root, "arm": arm, "status": "completed", "candidates": []}
            if source["status"] != "completed":
                row.update(status="source_failed")
                panel.journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
                continue
            for depth in (0, 1):
                layer = source["result"]["child"]["layers"][depth]
                for index, candidate in enumerate(layer["candidates"]):
                    prefix = np.asarray(candidate["chain"], np.float32)
                    tail = np.zeros((15 - len(prefix), 2), np.float32)
                    observed = []
                    directory = OUT / f"root{root:02d}_{arm}_d{depth}_c{index:03d}_calls"
                    recorder = panel.journal.PredictionJournal(
                        directory, panel.prior.old.previous.fixed.predictor(task),
                        action_dim=2, packet_length=5, token_cap=100000)
                    def predict(actions):
                        positions, scores = recorder(actions)
                        observed.extend(np.asarray(scores).tolist())
                        return positions, scores
                    try:
                        initial = np.concatenate((prefix, tail))[None]
                        initial_score = float(predict(initial)[1][0])
                        chain, score, generations = cem.cem(
                            tail, prefix, COUNT, np.random.default_rng(
                                2032800000 + root * 1000 + depth * 100 + index),
                            predict, initial_score, 4)
                        row["candidates"].append({
                            "depth": depth, "index": index, "prefix": prefix.tolist(),
                            "retained": index in layer["retained"],
                            "source_path": candidate["path"], "immediate_score": candidate["score"],
                            "initial_score": initial_score, "best_score": score,
                            "improvement": initial_score - score, "success": score < .05,
                            "chain": chain.tolist(), "generations": generations,
                            "tokens": recorder.tokens, "calls": recorder.calls})
                    except Exception:
                        row["candidates"].append({"depth": depth, "index": index,
                            "retained": index in layer["retained"], "status": "failed",
                            "error": traceback.format_exc(), "tokens": recorder.tokens,
                            "calls": recorder.calls})
            panel.journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"quality root={root} arm={arm} candidates={len(row['candidates'])}", flush=True)
    panel.seal.verify(dependency_seal)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    repair = verify_generation_dependencies(manifest)
    summary = []
    for root, task in enumerate(panel.prior.tasks()):
        for arm in ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            source = json.loads((panel.OUT / f"plan_{root:02d}_{arm}.json").read_text())
            if row["status"] != "completed":
                summary.append({"root": root, "arm": arm, "status": row["status"]})
                continue
            for item in row["candidates"]:
                if item.get("status") == "failed":
                    continue
                directory = OUT / f"root{root:02d}_{arm}_d{item['depth']}_c{item['index']:03d}_calls"
                info = panel.prior.journal_audit.audit_journal(directory)
                if info["tokens"] != item["tokens"] or len(info["calls"]) != item["calls"]:
                    raise ValueError("candidate journal mismatch")
                calls = list(panel.prior.old.previous.fixed.read_calls(directory))
                replay = iter(calls)
                def predict(actions):
                    saved, features, scores = next(replay)
                    np.testing.assert_array_equal(np.asarray(actions, np.float32), saved)
                    return features, scores
                prefix = np.asarray(item["prefix"], np.float32)
                tail = np.zeros((15 - len(prefix), 2), np.float32)
                initial = float(predict(np.concatenate((prefix, tail))[None])[1][0])
                chain, score, generations = cem.cem(
                    tail, prefix, COUNT, np.random.default_rng(
                        2032800000 + root * 1000 + item["depth"] * 100 + item["index"]),
                    predict, initial, 4)
                if (next(replay, None) is not None or initial != item["initial_score"] or
                        score != item["best_score"] or chain.tolist() != item["chain"] or
                        generations != item["generations"]):
                    raise ValueError("candidate replay mismatch")
                source_candidate = source["result"]["child"]["layers"][item["depth"]]["candidates"][item["index"]]
                if bool(item["retained"]) != (item["index"] in source["result"]["child"]["layers"][item["depth"]]["retained"]):
                    raise ValueError("retention binding mismatch")
                if item["source_path"] != source_candidate["path"]:
                    raise ValueError("candidate path mismatch")
                summary.append({"root": root, "arm": arm, "depth": item["depth"],
                    "index": item["index"], "retained": item["retained"],
                    "initial_score": item["initial_score"], "best_score": item["best_score"],
                    "success": item["success"]})
            print(f"audited quality root={root} arm={arm}", flush=True)
    repair = repair or verify_generation_dependencies(manifest)
    panel.journal.write_json(output, {"status": "passed", "rows": summary,
        "dependency_audit_repair": repair,
        "scope": "same-environment deterministic candidate replay; not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
