"""Independent saved-array, action-lineage and accounting check for feedback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_six_prefix_feedback_20261101_codex as runner

require, sha, spec = runner.require, runner.sha, runner.spec


def selected_from_trace(node, directory):
    path = Path(node["trace"])
    require(path.resolve().is_relative_to(directory) and sha(path) == node["trace_sha256"],
            "trace identity")
    events = runner.five.saved.chain(path)
    def one(kind):
        found = [e["payload"] for e in events if e["kind"] == kind]
        require(len(found) == 1, "unique " + kind)
        return found[0]
    horizon = runner.h4.h4.dimensions(node["boundary"])[0]
    bank = np.asarray(one("optimized_candidates_before_exchange")["raw"], np.float32)
    require(bank.shape == (8, 4, 5, 2) and np.isfinite(bank).all()
            and (np.abs(bank) <= 1).all(), "finite bounded bank")
    if horizon > 1:
        exchange = one("suffix_exchange")
        scores = np.asarray(exchange["scores"], float)
        require(scores.shape == (64,) and np.isfinite(scores).all(), "exchange scores")
        tails = [min(range(8), key=lambda j: (scores[8*i+j], j != i, j)) for i in range(8)]
        require(tails == exchange["tail_indices"], "exchange winner")
        bank = np.stack([np.concatenate((bank[i, :1], bank[j, 1:])) for i, j in enumerate(tails)])
    else:
        require(not any(e["kind"] == "suffix_exchange" for e in events), "no H1 exchange")
    previous = np.asarray(node["previous_plan"], np.float32)
    candidates = np.concatenate((bank[:, :horizon], previous[None, 1:]))
    final = one("selected_full_plan")
    scores = np.asarray(node["scores"], float)
    require(scores.shape == (9,) and np.isfinite(scores).all()
            and final["scores"] == node["scores"], "current scores")
    index = min(range(9), key=lambda i: (scores[i], i != 8, i))
    chosen = np.asarray(node["selected_full_plan"], np.float32)
    require(index == node["selected_index"] == final["index"]
            and np.array_equal(chosen, candidates[index])
            and np.array_equal(chosen, np.asarray(final["raw"], np.float32)), "selected actual argmin")
    return chosen


def run(directory, expected_sha, output):
    directory, output = Path(directory).resolve(), Path(output)
    report_path = directory / "report.json"
    require(sha(report_path) == expected_sha, "report SHA")
    report = json.loads(report_path.read_bytes())
    binding_path = directory / "binding.json"
    require(sha(binding_path) == report["binding_sha256"]
            and sha(report["prepared"]) == report["prepare_sha256"], "binding/prepare SHA")
    binding = json.loads(binding_path.read_bytes())
    require(binding == json.loads(Path(report["prepared"]).read_bytes()), "bound input")
    runner.check_sources(binding["sources"])
    require(report["error"] is None and report["model_before"] is not None
            and report["model_before"] == report["model_after"] == binding["expected_model_hash"],
            "no global failure and frozen model")
    require([r["prefix"] for r in report["rows"]] == list(runner.PREFIXES), "all six rows")
    require([r["prefix"] for r in binding["rows"]] == list(runner.PREFIXES), "all six inputs")
    _, runtime = runner.own.configure()
    goal, _, sources = runner.common.target.collect()
    aggregate = dict.fromkeys(spec.AXES, 0)
    summary = []
    for row, original in zip(report["rows"], binding["rows"], strict=True):
        require(row.get("error") is None and row["status"] in
                ("completed20", "native_stopped", "canvas_censored"), "valid physical branch")
        raw = np.asarray(row["executed_raw"], np.float32)
        require(raw.shape == (row["steps"], 2) and np.isfinite(raw).all()
                and (np.abs(raw) <= 1).all(), "actual action array")
        require(np.array_equal(raw[:10], np.asarray(original["first10"], np.float32)[:len(raw[:10])]),
                "original first ten actions")
        points = {c["step"]: c for c in row["checkpoints"]}
        require(len(points) == len(row["checkpoints"]) and row["steps"] in points, "checkpoint uniqueness")
        for frame in row["checkpoints"]:
            require(sha(frame["png"]) == frame["file_sha256"], "physical PNG")
            with Image.open(frame["png"]) as image:
                pixels = np.asarray(image)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"], "physical pixels")
            native = runner.common.metrics.qualification.native(goal["goal_state"], frame["state"])
            metrics = dict(**runner.common.metrics.metrics(frame["state"], goal["goal_state"]),
                           joint_xy=native["joint_xy_error"], native_success=native["initial_success"])
            require(metrics == frame["metrics"], "physical metrics")
            sources[frame["png"]] = frame["file_sha256"]
        for old in original["checkpoints"]:
            if old["step"] in points:
                runner.check_checkpoint(points[old["step"]], old)
        previous = np.asarray(original["previous_cp5_plan"], np.float32)
        history = original["cp5_task"]["root_context"]["latent_history"]
        require([n["boundary"] for n in row["nodes"]] == [10, 15][:len(row["nodes"])], "node sequence")
        for node in row["nodes"]:
            boundary = node["boundary"]
            require(boundary in points and node["current_pixel_sha256"] == points[boundary]["pixel_sha256"]
                    and np.array_equal(previous, np.asarray(node["previous_plan"], np.float32)),
                    "current observed boundary and prior plan")
            task = runner.make_task(original["cp5_task"], history[-2:] + [node["latent"]],
                                    raw[boundary-10:boundary], points[boundary]["pixel_sha256"],
                                    previous, boundary, runtime)
            require(task == node["task"], "actual history task")
            bound = runner.carry.BoundPlan.seal(row["prefix"], boundary-5, previous)
            feedback = bound.bind_feedback(episode=row["prefix"], boundary=boundary,
                        issued_packet=raw[boundary-5:boundary], rgb_sha256=points[boundary]["pixel_sha256"])
            require({k: v for k, v in feedback.items() if k != "tail"} == node["feedback"], "carry identity")
            policy, contract, config, _ = runner.contracts(task, binding["model_sha256"], boundary)
            require(policy == node["policy"] and config == node["config"], "node configuration")
            placeholder = runner.make_task(original["cp5_task"], history[-2:] + [[0.]*192],
                            raw[boundary-10:boundary], points[boundary]["pixel_sha256"],
                            previous, boundary, runtime)
            _, _, _, encoding = runner.contracts(placeholder, binding["model_sha256"], boundary)
            for kind, c, suffix in (("node", contract, "node"), ("encoding", encoding, "encode")):
                ledger = node[kind + "_ledger"]
                journal = Path(ledger["journal"])
                require(journal.resolve().is_relative_to(directory), "ledger path")
                account = runner.own.consumer.consume(journal, ledger["sha256"], spec.digest(c),
                            "structured", "node", f"{row['prefix']}_b{boundary}_{suffix}")
                cost = runner.h4.h4.node_cost("structured", boundary) if kind == "node" else dict.fromkeys(spec.AXES, 0)
                if kind == "encoding":
                    cost.update(encode_calls=1, encode_samples=1)
                require(account == node[kind + "_account"] and account["status"] == "closed_unverified"
                        and account["charged_upper_bounds"] == cost, "actual durable charge")
                for key in aggregate:
                    aggregate[key] += cost[key]
                sources[str(journal)] = ledger["sha256"]
            chosen = selected_from_trace(node, directory)
            require(runner.carry.BoundPlan.seal(row["prefix"], boundary, chosen).plan_sha256 ==
                    node["selected_plan_sha256"], "selected plan bytes")
            issued = raw[boundary:boundary+5]
            require(np.array_equal(issued, chosen[0, :len(issued)]), "new selected packet actually executed")
            sources[node["trace"]] = node["trace_sha256"]
            history = history[-2:] + [node["latent"]]
            previous = chosen
        if row["status"] == "completed20":
            require(row["steps"] == 20 and len(row["nodes"]) == 2 and sorted(points) == [0, 5, 10, 15, 20],
                    "full feedback sequence")
        else:
            require(row["stop_reason"]["step"] == row["steps"], "actual stopping boundary")
        summary.append(dict(prefix=row["prefix"], status=row["status"], steps=row["steps"],
                            selected_indices=[n["selected_index"] for n in row["nodes"]],
                            terminal=points[row["steps"]]["metrics"]))
    sources.update(binding["sources"])
    for path in (report_path, binding_path, Path(__file__).resolve()):
        sources[str(path)] = sha(path)
    runner.check_sources(sources)
    require(output.is_absolute() and output.parent == runner.ROOT / "results" and not output.exists(),
            "exclusive output")
    result = dict(rows=summary, actual_dispatch_charges=aggregate, sources=sources,
                  verified=True, neural_calls=0, P1_admission=False, independent_confirmation=False,
                  scope="saved artifact and action lineage check; not independent model rerun or kernel metering")
    runner.own.write(output, result)
    print(json.dumps(dict(rows=summary, actual_dispatch_charges=aggregate, output=str(output), sha256=sha(output))),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.directory, args.sha256, args.output)
