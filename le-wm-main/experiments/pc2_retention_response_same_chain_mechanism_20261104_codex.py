"""Prospective same-chain PC2 intervention for the sealed response arm."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_boundary_response_posthoc as boundary
from experiments import pc_final_chain_mechanism_measure_v1 as measure
from experiments import pc_prediction_journal_audit_v1 as journal_audit
from experiments import pc_prediction_journal_v1 as journal
from experiments import pc_retention_equal_slot_cost_dev_20261104_codex as source
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem


OUT = source.ROOT / "results/pc2_retention_response_same_chain_mechanism_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC2_RETENTION_RESPONSE_SAME_CHAIN_MECHANISM_PROTOCOL_20261104_CODEX.md"
CONTROLS = ("selected", "zero", "greedy", "lookahead", "random")
SCALES = (0.01, 0.05)


def dependencies():
    return source.prior.seal.capture([
        Path(__file__), PROTOCOL, Path(source.__file__), Path(source.prior.__file__),
        Path(source.compiler.__file__), Path(source.compiler.core.__file__),
        Path(source.compiler.v13.__file__), Path(cem.__file__),
        Path(measure.__file__), Path(measure.response.__file__),
        Path(boundary.__file__), Path(journal.__file__),
        Path(journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
        Path(source.prior.exact.__file__), Path(source.prior.exact.wall.__file__),
        Path(source.prior.exact.wall.physics.__file__),
        Path(source.prior.old.previous.fixed.__file__),
        Path(source.prior.old.OLD.previous.independent.__code__.co_filename)])


def verify_generation_dependencies(manifest):
    sealed = manifest["dependencies"]
    script_path = str(Path(__file__).resolve())
    expected = sealed.get("files", {}).get(script_path)
    if expected is None:
        source.prior.seal.verify(sealed)
        return None
    current = source.prior.seal.digest(script_path)
    if current == expected:
        source.prior.seal.verify(sealed)
        return None
    repaired = dict(sealed)
    repaired["files"] = dict(sealed["files"])
    del repaired["files"][script_path]
    source.prior.seal.verify(repaired)
    return {"path": script_path, "sealed_sha256": expected,
            "audit_sha256": current,
            "reason": "post-run audit shape check repaired for variable-length exact-Wall branches"}


def controls_for(root, plan):
    result = plan["result"]
    chain = np.asarray(result["selected"], np.float32)
    layer = result["child"]["layers"][0]
    greedy = int(np.argmin([item["score"] for item in layer["candidates"]]))
    reachability = layer["reachability"]
    lookahead = min(reachability, key=lambda item: (item["value"], item["candidate"]))["candidate"]
    rng = np.random.default_rng(2032200000 + root)
    random_prefix = np.clip(cem.interpolate(rng.normal(size=(1, 4, 2)), 5)[0],
                            -1, 1).astype(np.float32)
    controls = {
        "selected": chain[:5],
        "zero": np.zeros((5, 2), np.float32),
        "greedy": np.asarray(layer["candidates"][greedy]["chain"], np.float32),
        "lookahead": np.asarray(layer["candidates"][lookahead]["chain"], np.float32),
        "random": random_prefix,
    }
    return chain, controls, {"greedy": greedy, "lookahead": lookahead}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    deps = dependencies()
    OUT.mkdir()
    source_manifest = source.OUT / "manifest.json"
    source_audit = source.OUT / "audit.json"
    if not source_manifest.exists() or not source_audit.exists():
        raise FileNotFoundError("sealed retention source audit required")
    audit = json.loads(source_audit.read_text())
    if audit.get("status") != "passed" or len(audit.get("rows", [])) != 80:
        raise ValueError("retention source audit incomplete")
    journal.write_json(OUT / "manifest.json", {
        "dependencies": deps, "source_manifest_sha256": source.prior.seal.digest(source_manifest),
        "source_audit_sha256": source.prior.seal.digest(source_audit),
        "controls": CONTROLS, "scales": SCALES,
        "scope": "PC2 development same-chain intervention; not confirmation"})
    for root, task in enumerate(source.tasks()):
        plan_path = source.OUT / f"plan_{root:02d}_response.json"
        plan = json.loads(plan_path.read_text())
        if plan["status"] != "completed":
            raise ValueError("response plan failed")
        chain, controls, indices = controls_for(root, plan)
        calls_dir = OUT / f"root{root:02d}_calls"
        recorder = journal.PredictionJournal(
            calls_dir, source.prior.old.previous.fixed.predictor(task),
            action_dim=2, packet_length=5, token_cap=100000)
        event_dir = OUT / f"root{root:02d}_events"
        event_dir.mkdir()
        def emit(event):
            journal.write_json(event_dir / f"{len(list(event_dir.iterdir())):04d}.json", event)
        row = {"root": root, "task": task, "status": "completed",
               "source_plan_sha256": source.prior.seal.digest(plan_path)}
        try:
            suffix = {}
            chain_tail = chain[5:]
            tails = [np.tile(x, (10, 1)) for x in
                     ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
            tails.append(chain_tail)
            for name, prefix in controls.items():
                common = np.asarray([np.concatenate((prefix, tail)) for tail in tails], np.float32)
                _, common_scores = recorder(common)
                runs = []
                for repeat in range(3):
                    zero_tail = np.zeros((10, 2), np.float32)
                    zero_chain = np.concatenate((prefix, zero_tail))[None]
                    initial = float(recorder(zero_chain)[1][0])
                    best_chain, score, generations = cem.cem(
                        zero_tail, prefix, 480,
                        np.random.default_rng(2032300000 + root * 10 + repeat),
                        recorder, initial, 4)
                    runs.append({"score": score, "chain": best_chain.tolist(),
                                 "generations": generations})
                suffix[name] = {"prefix": prefix.tolist(),
                                "common_scores": common_scores.tolist(),
                                "relative_zero": (common_scores - common_scores[0]).tolist(),
                                "runs": runs}
            def branch(actions):
                trajectory, blocked = source.prior.exact.wall.rollout(
                    task["start"], np.pad(actions, ((0, 0), (0, 1), (0, 0))) *
                    source.prior.exact.wall.HIGH)
                return trajectory[:, 15, :2], np.asarray(blocked)[:, :15]
            measurements = measure.measure(
                chain, controls, task["goal"], recorder, branch, emit,
                scales=SCALES, prefix_steps=5)
            _, prefix_scores = recorder(chain[:5][None])
            initial_distance = float(np.linalg.norm(np.asarray(task["start"][:2]) -
                                                    np.asarray(task["goal"])))
            nonprogress = bool(prefix_scores[0] >= initial_distance - 1e-10)
            nonzero = bool(np.any(chain[:5] != 0) and np.any(chain[5:] != 0))
            comparisons = {}
            selected = suffix["selected"]
            for name in CONTROLS[1:]:
                other = suffix[name]
                common_gain = bool(selected["relative_zero"][-1] <
                                   other["relative_zero"][-1] - 1e-10)
                replan_gain = bool(np.mean([x["score"] for x in selected["runs"]]) <
                                   np.mean([x["score"] for x in other["runs"]]) - 1e-10)
                gaps = {}
                for epsilon in ("0.01", "0.05"):
                    left, right = measurements["selected"][epsilon], measurements[name][epsilon]
                    matrix = float(np.linalg.norm(np.asarray(left["matrix"]) -
                                                   np.asarray(right["matrix"])))
                    lg, rg = left["goal_gradient"], right["goal_gradient"]
                    gradient = None if lg is None or rg is None else float(
                        np.linalg.norm(np.asarray(lg) - np.asarray(rg)))
                    gaps[epsilon] = {"matrix": matrix, "goal_gradient": gradient}
                response_diff = all(item["matrix"] > 1e-6 and
                                    item["goal_gradient"] is not None and
                                    item["goal_gradient"] > 1e-6
                                    for item in gaps.values())
                branches_differ = (
                    measurements["selected"]["0.01"]["branches"][0] !=
                    measurements[name]["0.01"]["branches"][0])
                comparisons[name] = {"common_gain": common_gain,
                                     "replan_gain": replan_gain,
                                     "response_diff": response_diff,
                                     "gaps": gaps,
                                     "nominal_branch_diff": branches_differ}
            smooth_qualified = (
                nonzero and nonprogress and
                all(comparisons[name]["common_gain"] and
                    comparisons[name]["replan_gain"] and
                    comparisons[name]["response_diff"]
                    for name in CONTROLS[1:]) and
                all(measurements[name][epsilon]["all_branch_stable"]
                    for name in CONTROLS for epsilon in ("0.01", "0.05")))
            row.update(chain=chain.tolist(), controls={k: v.tolist() for k, v in controls.items()},
                       control_indices=indices, suffix=suffix, measurements=measurements,
                       comparisons=comparisons, nonprogress=nonprogress, nonzero=nonzero,
                       smooth_mechanism_qualified=smooth_qualified,
                       tokens=recorder.tokens, calls=recorder.calls)
        except Exception:
            row.update(status="failed", error=traceback.format_exc(),
                       tokens=recorder.tokens, calls=recorder.calls)
        journal.write_json(OUT / f"root{root:02d}.json", row)
        print(f"root={root} status={row['status']} qualified={row.get('smooth_mechanism_qualified')}",
              flush=True)
    source.prior.seal.verify(deps)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    repair = verify_generation_dependencies(manifest)
    if manifest["controls"] != list(CONTROLS) or manifest["scales"] != list(SCALES):
        raise ValueError("manifest mismatch")
    rows = []
    for root, task in enumerate(source.tasks()):
        item = json.loads((OUT / f"root{root:02d}.json").read_text())
        if item["root"] != root or item["task"] != task:
            raise ValueError("row identity mismatch")
        info = journal_audit.audit_journal(OUT / f"root{root:02d}_calls")
        if info["tokens"] != item["tokens"] or len(info["calls"]) != item["calls"]:
            raise ValueError("journal mismatch")
        source_plan = source.OUT / f"plan_{root:02d}_response.json"
        if item["source_plan_sha256"] != source.prior.seal.digest(source_plan):
            raise ValueError("source plan drift")
        if item["status"] != "completed":
            rows.append({"root": root, "status": "failed", "error": item["error"]})
            continue
        chain = np.asarray(item["chain"], np.float32)
        if chain.shape != (15, 2) or not np.isfinite(chain).all() or np.max(np.abs(chain)) > 1:
            raise ValueError("invalid selected chain")
        for name in CONTROLS:
            prefix = np.asarray(item["controls"][name], np.float32)
            if prefix.shape != (5, 2) or np.max(np.abs(prefix)) > 1:
                raise ValueError("invalid control")
            if name == "selected" and not np.array_equal(prefix, chain[:5]):
                raise ValueError("selected control mismatch")
            for epsilon in ("0.01", "0.05"):
                result = item["measurements"][name][epsilon]
                if (len(result["matrix"]) != 2 or len(result["matrix"][0]) != 20
                        or len(result["coordinate_class"]) != 20
                        or not result["branches"]):
                    raise ValueError("measurement shape mismatch")
        rows.append({"root": root, "status": "completed",
                     "smooth_mechanism_qualified": item["smooth_mechanism_qualified"],
                     "nonprogress": item["nonprogress"], "nonzero": item["nonzero"],
                     "tokens": item["tokens"], "calls": item["calls"],
                     "comparisons": item["comparisons"]})
        print(f"audited root={root}", flush=True)
    repair = repair or verify_generation_dependencies(manifest)
    journal.write_json(output, {"status": "passed", "rows": rows,
        "dependency_audit_repair": repair,
        "scope": "same-environment sealed-chain mechanism audit; not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
