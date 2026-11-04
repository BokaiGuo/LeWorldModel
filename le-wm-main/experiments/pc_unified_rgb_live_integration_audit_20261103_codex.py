"""Audit sealed live RGB integration inputs, plans, execution and accounting."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_unified_rgb_live_integration_20261103_codex as run

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(run.MANIFEST.read_text())
    receipt = json.loads(run.RECEIPT.read_text())
    require(receipt["manifest_sha256"] == sha(run.MANIFEST), "manifest binding")
    require(all(sha(Path(path)) == digest for path, digest in manifest["dependencies"].items()),
            "dependency drift")
    require(receipt["images_sha256"] == sha(run.IMAGES), "image archive binding")
    require(receipt["root"] == 0 and len(receipt["rows"]) == 3, "root or panel")
    with np.load(run.IMAGES, allow_pickle=False) as archive:
        frames = archive["rgb"]
        require(len(frames) == 31, "frame clock")
        previous_tail, dispatched, totals = None, [], {"calls": 0, "samples": 0, "tokens": 0}
        actions = run.old.WARMUP.tolist()
        records = []
        for step, (row, path) in enumerate(zip(receipt["rows"], run.DECISIONS, strict=True)):
            decision = json.loads(path.read_text())
            binding, plan = decision["binding"], decision["compiled"]
            require(row["step"] == step and row["decision_sha256"] == sha(path)
                    and decision["manifest_sha256"] == sha(run.MANIFEST), "decision binding")
            ticks = binding["ticks"]
            require(ticks == list(run.runner.previous.prior.context_ticks(16 + 5 * step)), "ticks")
            require(binding["rgb_sha256"] == [run.old.digest_image(frames[t]) for t in ticks]
                    and binding["previous_rgb_sha256"] == [run.old.digest_image(frames[t-1]) for t in ticks],
                    "RGB history mismatch")
            require(binding["history_actions"] == actions[-10:] and binding["carry"] == previous_tail,
                    "action history or carry mismatch")
            require(binding["model_hash"] == receipt["model_hash"]
                    and plan["context_id"] == run.digest_context(binding)
                    and plan["root"] == 0 and plan["step"] == step, "context binding")
            chain = plan["core_result"]["selected"]["chain"]
            require(len(chain) == 15 - 5 * step and plan["dispatched_actions"] == chain[:5]
                    and plan["remaining_tail"] == chain[5:] and row["actions"] == chain[:5],
                    "selected chain / dispatch mismatch")
            require(len(row["times_ns"]) == 5 and row["sealed_ns"] <= row["times_ns"][0][0]
                    and all(a <= b for a, b in row["times_ns"])
                    and all(row["times_ns"][i][1] <= row["times_ns"][i+1][0] for i in range(4)),
                    "seal or execution order")
            require(row["rgb_sha256"] == [run.old.digest_image(frame)
                    for frame in frames[16 + 5*step:21 + 5*step]], "feedback frames")
            trace = plan["prediction_trace"]
            calls, samples, tokens = len(trace), 0, 0
            for batch in trace:
                requested = np.asarray(batch["requested_actions"])
                raw = np.asarray(batch["model_raw_float32_actions"], dtype=np.float32)
                require(requested.ndim == 3 and requested.shape[1] in (5, 10, 15)
                        and requested.shape[2] == 2 and np.isfinite(requested).all()
                        and np.max(np.abs(requested)) <= 1
                        and np.array_equal(requested.astype(np.float32), raw), "model action mismatch")
                batch_tokens = len(raw) * raw.shape[1] // 5
                require(batch["tokens"] == batch_tokens and len(batch["positions"]) == len(raw)
                        and len(batch["scores"]) == len(raw), "batch accounting")
                samples += len(raw)
                tokens += batch_tokens
            counts = plan["counts"]
            require((calls, samples, tokens) == (counts["prediction_calls"], counts["trajectory_samples"],
                    counts["prediction_token_samples"]) and tokens <= 480, "compiler accounting")
            require(all(decision["outer_counts"][key] == counts[key]
                    for key in ("prediction_calls", "trajectory_samples", "prediction_token_samples")),
                    "outer accounting")
            for key, value in (("calls", calls), ("samples", samples), ("tokens", tokens)):
                totals[key] += value
            actions.extend(chain[:5])
            dispatched.extend(chain[:5])
            previous_tail = chain[5:] if step < 2 else None
            records.append({"step": step, "horizon": len(chain), "calls": calls, "samples": samples,
                            "tokens": tokens, "abstained": plan["core_result"]["abstained"],
                            "planning_seconds": decision["planning_seconds"]})
        require(dispatched == receipt["executed_actions"] and len(dispatched) == 15, "execution record")
    result = {"receipt_sha256": sha(run.RECEIPT), "manifest_sha256": sha(run.MANIFEST),
              "audit_script_sha256": sha(Path(__file__)), "dependency_count": len(manifest["dependencies"]),
              "records": records, "totals": totals,
              "limitations": ["Verifies saved clocks and records, not external attestation of runtime ordering.",
                              "No independent rerun of model or physical backend in this audit.",
                              "Source hashes do not certify complete dependency closure or independent training data."]}
    run.old.write_json(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
