"""Development-only spectral vs compute-matched random direction shadow runner."""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading
import time

import numpy as np
import torch

from experiments import pc_shadow_spectral_closed_loop_dev_20261103_codex as base
from experiments.pc_pointmaze_durable_dispatch_20261103_codex import DurableJournal, _sync_dir

ORIGINAL = base.spectral_candidates
BASE_SHA = "e0bc75a72f49af42e3b0d67d523d6d67b45a9df413e9ea33bec43eae533c2990"
DOMAIN = b"pointmaze-shadow-matched-random-direction-v1\0"
_PATCH_LOCK = threading.Lock()


def direction_seed(noise):
    if noise.shape != (3, 16, 10) or noise.dtype != torch.float32 or not bool(torch.isfinite(noise).all()):
        raise ValueError("invalid CEM noise")
    data = noise.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    return int.from_bytes(hashlib.sha256(DOMAIN + data).digest()[:8], "little") & ((1 << 63)-1)


@contextmanager
def random_direction(noise, mean, std, event):
    """Replace only seed direction; always execute original structure calculation."""
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("random direction patch already active")
    installed = False
    try:
        if base.spectral_candidates is not ORIGINAL:
            raise RuntimeError("unexpected spectral candidate implementation")
        seed_started = time.monotonic()
        seed = direction_seed(noise)
        seed_seconds = time.monotonic() - seed_started
        def replace(response, action_mean, action_std):
            structure = ORIGINAL(response, action_mean, action_std)
            started = time.monotonic()
            random = torch.randn((10,), generator=torch.Generator(device="cpu").manual_seed(seed),
                                 dtype=action_mean.dtype).to(action_mean.device)
            magnitude = random.abs().max()
            if not bool(torch.isfinite(random).all()) or float(magnitude) <= 0:
                raise ValueError("degenerate random direction")
            direction = random / magnitude
            amplitudes = torch.tensor((.25, -.25, .5, -.5), device=direction.device, dtype=direction.dtype)
            raw = amplitudes[:, None] * direction
            normalized = (raw - action_mean.repeat(5)) / action_std.repeat(5)
            if not bool(torch.isfinite(normalized).all()):
                raise ValueError("invalid random proposals")
            if direction.is_cuda:
                torch.cuda.synchronize(direction.device)
            event("random_direction_returned", seed=seed, direction=direction.cpu().tolist(),
                  discarded_spectral_direction=structure["direction_raw"].cpu().tolist(),
                  random_generation_seconds=time.monotonic()-started,
                  seed_derivation_seconds=seed_seconds)
            return {**structure, "direction_raw": direction.detach(), "raw": raw.detach(),
                    "normalized": normalized.detach()}
        base.spectral_candidates = replace
        installed = True
        yield
    finally:
        if installed:
            base.spectral_candidates = ORIGINAL
        _PATCH_LOCK.release()


def physical_run(output_dir):
    """Runs only with explicit CLI authorization in an otherwise single-thread process."""
    if threading.active_count() != 1:
        raise RuntimeError("dedicated single-thread Python process required")
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("base executor hash changed")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    _sync_dir(output_dir.parent)
    protocol = Path(__file__).resolve().parents[2] / "docs/PC_POINTMAZE_SHADOW_RANDOM_MATCHED_DEV_PROTOCOL_20261103_CODEX.md"
    original_decide = base.decide
    def random_decide(arm, evaluate, response, mean, std, noise, synchronize=lambda: None, event=lambda *a, **k: None):
        if arm != "spectral":
            raise ValueError("unexpected base arm")
        with random_direction(noise, mean, std, event):
            return original_decide(arm, evaluate, response, mean, std, noise, synchronize, event)
    with DurableJournal(output_dir / "comparison_events.jsonl") as journal:
        try:
            journal.record("comparison_started", mapping={"spectral": "spectral", "random": "spectral"},
                           output_labels="parent subdirectory defines actual arm",
                           protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                           executor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           base_sha256=BASE_SHA, equal_total_compute=False)
            journal.record("arm_attempt", actual_arm="spectral")
            base.physical_run(("spectral",), output_dir / "spectral")
            journal.record("arm_returned", actual_arm="spectral")
            journal.record("arm_attempt", actual_arm="random")
            base.decide = random_decide
            try:
                base.physical_run(("spectral",), output_dir / "random")
            finally:
                base.decide = original_decide
            journal.record("arm_returned", actual_arm="random")
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
        parser.error("physical execution requires explicit --execute-physical authorization")
    physical_run(args.output_dir)


if __name__ == "__main__":
    main()
