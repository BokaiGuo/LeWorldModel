"""Exploratory fixed-candidate H2 kernel; no model loader or physical launcher."""
import argparse
from copy import deepcopy
import json

import numpy as np

from experiments.spectral_reachability import codex_p1_blind_bundle_consumer_20261030_v1 as blind
from experiments.spectral_reachability import p1_constructive_observed_suffix_cuda_20261027_v1 as observed

NAME = "codex_p1_h2_fixed_candidate_20261030_v1"
LABEL = "fixed_common_signed_x"


def fixed_candidates():
    # Exact float32 raw actions. Both packets fit the original logit ball.
    raw = np.zeros((2, 5, 2), dtype=np.float32)
    raw[0, :, 0] = np.float32(.125)
    raw[1, :, 0] = np.float32(-.125)
    logits = np.arctanh(raw.astype(np.float64)).reshape(2, 10)
    blind.need((np.linalg.norm(logits, axis=1) < float(.75*np.sqrt(5))).all(), "fixed action chart")
    return raw


def specification():
    return dict(schema=NAME, candidate_kind=LABEL, raw_packets=fixed_candidates().astype(float).tolist(),
                candidates=["zero_nominal", LABEL, "fresh_observed_ridge"],
                candidate_rule="fixed +0.125 x then -0.125 x, y=0; independent of all rows and outcomes",
                selection="each packet same-Q nonlinear terminal cost, stable candidate-index ties",
                feedback="one cp5 observation; packet two uses packet one predicted winner context",
                per_packet=observed.budget(),
                two_packets={k: 2*v for k, v in observed.budget().items()},
                no_dedup=True, no_early_exit=True, physical_authorized=False, P1_admission=False,
                total_budget_equality_proven=False)


def public_record(value):
    """Correct legacy names in externally visible events/results."""
    if isinstance(value, dict):
        return {("fixed_candidate" if k == "original_suffix" else k): public_record(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [public_record(x) for x in value]
    if isinstance(value, str) and value == "original_imagined_suffix":
        return LABEL
    return value


def compile_row(bundle, row_index, cp5_latent, backend_factory, sink):
    """Compose real H1 kernel; caller must separately attest encoder and backend.

    Injected backends and supplied latents are NOT CUDA/provenance authorities.
    This callable does not authorize dispatch, physical execution, or retries.
    """
    blind.validate(bundle)
    blind.need(type(row_index) is int and 0 <= row_index < 16, "existing row index")
    row = bundle["rows"][row_index]
    task = deepcopy(row["task"])
    prefix = observed.raw5(row["prefix"])
    candidates = fixed_candidates()
    context = observed.observed_context(task, cp5_latent, prefix)
    results, counts = [], []
    sink(dict(kind="fixed_candidate_contract", specification=specification(), row_index=row_index,
              cp5_pixel_sha256=row["cp5_rgb"]["pixel_sha256"]))
    for packet in range(2):
        def emit(event):
            sink(dict(packet=packet, event=public_record(event)))
        ledger = observed.Ledger(emit)
        backend = backend_factory(packet, ledger)
        result = observed.execute_branch(
            dict(task=deepcopy(task), observed_context=deepcopy(context),
                 original_suffix=candidates[packet].copy()), backend, ledger, emit)
        observed.base.exact_counts(ledger.counts(), observed.budget())
        blind.need(result["status"] == "observed_suffix_selected_not_admitted" and
                   len(result["candidates"]) == 3 and
                   result["candidates"][1]["kind"] == "original_imagined_suffix" and
                   np.array_equal(result["candidates"][1]["commands"], candidates[packet]),
                   "complete fixed-candidate kernel result")
        context = deepcopy(result["winner"]["context"])
        results.append(public_record(result))
        counts.append(deepcopy(ledger.counts()))
    return dict(schema=NAME+".result", status="exploratory_kernel_only", row_index=row_index,
                specification_sha256=blind.digest(specification()), packets=results, counts=counts,
                raw10=np.concatenate([observed.raw5(r["winner"]["commands"]) for r in results]).tolist(),
                physical_authorized=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description="CPU-only blind-bundle and fixed-rule preflight; no model calls")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    _, status = blind.read_verified(args.bundle, args.sha256)
    spec = specification()
    print(json.dumps(dict(bundle=status, specification=spec, specification_sha256=blind.digest(spec))))


if __name__ == "__main__":
    main()
