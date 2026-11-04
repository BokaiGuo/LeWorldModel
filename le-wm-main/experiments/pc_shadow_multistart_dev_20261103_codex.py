"""Two-start CEM baseline, using the durable RGB shadow executor."""
import argparse
import hashlib
from pathlib import Path
import threading

import numpy as np
import torch

from experiments import pc_shadow_spectral_closed_loop_dev_20261103_codex as base
from experiments.pc_pointmaze_durable_dispatch_20261103_codex import DurableJournal, _sync_dir
from experiments.spectral_reachability.pc_spectral_seeded_feedback_cem_20261103_codex import seeded_cem

BASE_SHA = "e0bc75a72f49af42e3b0d67d523d6d67b45a9df413e9ea33bec43eae533c2990"
DOMAIN = b"pointmaze-cem-multistart-v1\0"
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/PC_POINTMAZE_SHADOW_MULTISTART_DEV_PROTOCOL_20261103_CODEX.md"


def second_seed(noise):
    if noise.shape != (3, 16, 10) or noise.dtype != torch.float32 or not bool(torch.isfinite(noise).all()):
        raise ValueError("invalid CEM noise")
    raw = noise.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    return int.from_bytes(hashlib.sha256(DOMAIN + raw).digest()[:8], "little") & ((1 << 63) - 1)


def decide(arm, evaluate, response, mean, std, noise, synchronize=lambda: None, event=lambda *a, **k: None):
    if arm != "baseline":
        raise ValueError("baseline arm required")
    prior, lower, upper = ((-mean)/std).repeat(5), ((-1-mean)/std).repeat(5), ((1-mean)/std).repeat(5)
    seed = second_seed(noise)
    extra = torch.randn(noise.shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                        dtype=torch.float32).to(noise.device)
    runs = [seeded_cem(evaluate, prior, lower, upper, candidate_noise)
            for candidate_noise in (noise, extra)]
    selected = int(float(runs[1]["final_scores"].min()) < float(runs[0]["final_scores"].min()))
    result = dict(runs[selected])
    result["cem_counts"] = {"samples": 100, "batches": 8}
    event("multistart", extra_seed=seed, extra_noise=extra.cpu().tolist(), selected_start=selected,
          starts=[{"chosen": item["chosen"].cpu().tolist(), "final": item["final"].cpu().tolist(),
                   "final_scores": item["final_scores"].cpu().tolist(),
                   "trace": [[tensor.cpu().tolist() for tensor in iteration] for iteration in item["trace"]]}
                  for item in runs])
    return result, None, {"response_forward_calls": 0, "reverse_vector_calls": 0,
                          "svd_calls": 0, "fd_model_forwards": 0}


def physical_run(output_dir):
    if threading.active_count() != 1:
        raise RuntimeError("dedicated single-thread Python process required")
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("base executor hash changed")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    _sync_dir(output_dir.parent)
    original = base.decide
    with DurableJournal(output_dir / "comparison_events.jsonl") as journal:
        try:
            journal.record("comparison_started", actual_arm="two_start_cem",
                           base_label="baseline", protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                           executor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           base_sha256=BASE_SHA, equal_total_compute=False)
            journal.record("arm_attempt", actual_arm="two_start_cem")
            base.decide = decide
            try:
                base.physical_run(("baseline",), output_dir / "multistart")
            finally:
                base.decide = original
            journal.record("arm_returned", actual_arm="two_start_cem")
            journal.record("comparison_completed", independent_confirmation=False)
        except BaseException as exc:
            journal.record("comparison_failed", error_type=type(exc).__name__, message=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-physical", action="store_true")
    args = parser.parse_args()
    if not args.execute_physical:
        parser.error("physical execution requires explicit --execute-physical")
    physical_run(args.output_dir)


if __name__ == "__main__":
    main()
