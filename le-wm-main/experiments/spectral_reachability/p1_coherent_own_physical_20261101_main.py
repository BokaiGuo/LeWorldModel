"""Verify five sealed own-suffix plans before any physical control."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_centered_program_replay_20261101_main as physical

saved = physical.common.panel.saved


def collect(directory, report_sha):
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    if saved.sha(report_path) != report_sha:
        raise ValueError("external report SHA")
    report = json.loads(report_path.read_bytes())
    binding_path = directory / "binding.json"
    if saved.sha(binding_path) != report["binding_sha256"]:
        raise ValueError("planning binding")
    binding = json.loads(binding_path.read_bytes())
    if saved.sha(report["prepared"]) != report["prepare_sha256"] or json.loads(Path(report["prepared"]).read_bytes()) != binding:
        raise ValueError("prepared/binding identity")
    if (report["status"] != "complete" or report["error"] or report["model_before"] is None
            or report["model_before"] != report["model_after"]):
        raise ValueError("complete frozen-model planning")
    if report["model_before"] != binding["expected_model_hash"]:
        raise ValueError("model identity")
    prefixes = [*physical.common.panel.ARMS, "zero"]
    if [r["prefix"] for r in report["rows"]] != prefixes or [r["prefix"] for r in binding["rows"]] != prefixes:
        raise ValueError("all five planned prefixes")
    sources = dict(binding["sources"])
    if any(saved.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    schedule = []
    for row, plan_input in zip(report["rows"], binding["rows"], strict=True):
        if any(row[k] != plan_input[k] for k in ("prefix", "prefix_raw", "pixel_sha256", "cp5_state_identity_sha256")):
            raise ValueError("planning prefix identity")
        task, base = row["task"], binding["base_task"]
        if any(task[k] != base[k] for k in ("goal", "action_mean", "action_std")):
            raise ValueError("unchanged task goal and normalizer")
        raw10 = np.concatenate((np.asarray(binding["warmup_raw"], np.float32),
                                np.asarray(row["prefix_raw"], np.float32)))
        normalized = ((raw10-np.asarray(base["action_mean"], np.float32)) /
                      np.asarray(base["action_std"], np.float32)).reshape(2, 10).tolist()
        if (task["root_context"]["action_history"] != normalized or
                task["root_context"]["latent_history"] != base["root_context"]["latent_history"][-2:] + [row["cp5_latent"]]):
            raise ValueError("actual cp5 history/action alignment")
        for kind, suffix in (("node", "_node"), ("encoding", "_encode")):
            ledger = row[kind + "_ledger"]
            account = row[kind + "_account"]
            header = json.loads(Path(ledger["journal"]).read_bytes().splitlines()[0])["payload"]
            digest = saved.spec.digest(header["contract"])
            actual = saved.consumer.consume(ledger["journal"], ledger["sha256"], digest,
                                            "structured", "node", row["prefix"] + suffix)
            if actual != account or actual["status"] != "closed_unverified":
                raise ValueError("actual durable account")
            expected = binding["node_cost"] if kind == "node" else dict.fromkeys(saved.h4.AXES, 0)
            if kind == "encoding":
                expected.update(encode_calls=1, encode_samples=1)
            if actual["charged_upper_bounds"] != expected:
                raise ValueError("same node/encoding budget")
            sources[ledger["journal"]] = ledger["sha256"]
        trace = Path(row["trace"])
        if not trace.resolve().is_relative_to(directory) or saved.sha(trace) != row["trace_sha256"]:
            raise ValueError("node trace")
        events = saved.chain(trace)
        def one(kind):
            found = [e["payload"] for e in events if e["kind"] == kind]
            if len(found) != 1:
                raise ValueError("unique node event")
            return found[0]
        optimized = np.asarray(one("optimized_candidates_before_exchange")["raw"], np.float32)
        exchange = one("suffix_exchange")
        scores = np.asarray(exchange["scores"], float)
        if optimized.shape != (8, 4, 5, 2) or scores.shape != (64,) or not np.isfinite(scores).all():
            raise ValueError("candidate bank")
        tails = [min(range(8), key=lambda j: (scores[8*i+j], j != i, j)) for i in range(8)]
        if tails != exchange["tail_indices"]:
            raise ValueError("exchange argmin")
        bank = np.stack([np.concatenate((optimized[i, :1], optimized[j, 1:]))[:3]
                         for i, j in enumerate(tails)])
        bank = np.concatenate((bank, np.zeros((1, 3, 5, 2), np.float32)))
        selected = one("selected_full_plan")
        score = np.asarray(row["scores"], float)
        if score.shape != (9,) or not np.isfinite(score).all() or selected["scores"] != row["scores"]:
            raise ValueError("selection scores")
        index = min(range(9), key=lambda i: (score[i], i != 8, i))
        chosen = np.asarray(row["selected_suffix"], np.float32)
        if (index != row["selected_index"] or index != selected["index"]
                or not np.array_equal(chosen, bank[index])
                or not np.array_equal(chosen, np.asarray(selected["raw"], np.float32))):
            raise ValueError("selected suffix identity")
        sources[str(trace)] = row["trace_sha256"]
        schedule.append(dict(prefix=row["prefix"],
            commands=np.concatenate((np.asarray(row["prefix_raw"], np.float32), chosen.reshape(15, 2))).tolist(),
            cp5_pixel_sha256=row["pixel_sha256"], cp5_state_identity_sha256=row["cp5_state_identity_sha256"]))
    for path in (report_path, binding_path, Path(report["prepared"]), Path(__file__).resolve()):
        sources[str(path)] = saved.sha(path)
    return schedule, sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", required=True, action="store_true")
    args = parser.parse_args()
    schedule, bindings = collect(args.plans, args.report_sha256)
    raise SystemExit(0 if physical.execute(schedule, bindings, args.output) else 1)
