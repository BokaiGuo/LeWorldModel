"""Exploratory two-packet adapter; NOT a launcher or execution authority.

Only a separately verified, blinded model task and cp5 latent may enter here.
No physical report, simulator state, future observations, or outcome is accepted.
"""
from copy import deepcopy

import numpy as np

from experiments.spectral_reachability import p1_constructive_observed_suffix_cuda_20261027_v1 as observed


def compile_two_packets(task, cp5_latent, prefix, original_suffix10, backend_factory,
                        operation_factory, journal):
    """Compile 10 actions with one real observation, then predicted history only.

    The factories must create independent H1 budget guards. This function is a
    kernel, not an authority: it does not validate physical provenance, encode
    RGB, enforce CUDA, or authorize dispatch.
    """
    observed.base.encode.core.validate_task(task)
    original = np.asarray(original_suffix10)
    if original.shape != (10, 2):
        raise ValueError("original suffix must be exactly two packets")
    first_original, second_original = observed.raw5(original[:5]), observed.raw5(original[5:])
    context = observed.observed_context(deepcopy(task), cp5_latent, prefix)
    results = []
    for packet, original_packet in enumerate((first_original, second_original)):
        backend = backend_factory(packet)
        operation = operation_factory(packet)
        result = observed.compile_suffix(deepcopy(task), deepcopy(context), original_packet,
                                         backend, operation, journal)
        if result.get("status") != "observed_suffix_selected_not_admitted":
            raise ValueError("compiler did not return its expected non-admitted status")
        winner = result["winner"]
        commands = observed.raw5(winner["commands"])
        observed.base.d2p.validate_advance(context, commands, winner["context"])
        context = deepcopy(winner["context"])
        results.append(deepcopy(result))
    return dict(status="exploratory_kernel_only", packets=results,
                raw10=np.concatenate([observed.raw5(r["winner"]["commands"]) for r in results]).tolist(),
                planned_feedback_count=1, second_packet_context="first_packet_model_prediction",
                per_packet_budget=observed.budget(), P1_admission=False,
                physical_authorized=False)
