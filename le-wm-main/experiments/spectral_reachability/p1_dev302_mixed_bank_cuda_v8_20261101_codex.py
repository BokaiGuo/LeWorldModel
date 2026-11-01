"""V8 CUDA selector for large endpoint-preserving support paths."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from . import p1_dev302_mixed_bank_cuda_v7_20261101_codex as v7r
from . import p1_dev302_mixed_prefix_bank_v8_20261101_codex as bank

runner, locking = v7r.runner, v7r.locking
NAME, ROOT = "p1_dev302_mixed_bank_cuda_v8_20261101_codex", runner.ROOT


def protocol():
    value = v7r.v6_runner.v5_runner.BASE_PROTOCOL()
    _a, _b, current, goal = runner._images()
    plans = runner._plans()
    value["schema"] = NAME + ".protocol.v1"
    value["prefix_bank"] = bank.build(
        current, goal, plans["full10_conditional_D2P"][:5], plans["line_search_GN"][:5]
    )
    value["algorithm"]["contact_filter"] = value["prefix_bank"]["contact_guard"]
    value["algorithm"]["tail_randomization"] = {"kind": "common_random_numbers", "seed": runner.SEED}
    n = len(value["prefix_bank"]["candidates"])
    value["algorithm"]["prefix_candidates"] = n
    value["budget"].update(
        action_encoder_calls=n * (1 + runner.ITERATIONS * runner.HORIZON),
        predict_calls=n * (1 + runner.ITERATIONS * runner.HORIZON),
        predict_samples=n * (1 + runner.ITERATIONS * runner.HORIZON * runner.SAMPLES * runner.VARIANTS),
        candidate_evaluations=n,
    )
    for path in (Path(__file__).resolve(), Path(bank.__file__).resolve()):
        value["bindings"][str(path)] = runner.sha(path)
    return value


def run(prepared, digest, output):
    old_name, old_protocol, original = runner.NAME, runner.protocol, runner.np.random.default_rng
    runner.NAME, runner.protocol = NAME, protocol
    runner.np.random.default_rng = lambda seed=None: (
        original(seed) if seed == bank.v7.v6.geometry.RANDOM_SEED else original(runner.SEED)
    )
    try:
        return runner.run(prepared, digest, output)
    finally:
        runner.np.random.default_rng = original
        runner.NAME, runner.protocol = old_name, old_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    runner.require(args.tag.isalnum() and not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive")
    digest = runner.write_new(prepared, protocol())
    print(json.dumps({"protocol": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            result = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": result}), flush=True)


if __name__ == "__main__":
    main()
