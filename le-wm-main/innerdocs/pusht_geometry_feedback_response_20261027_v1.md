# PushT P1 feedback response localization

Independent known-root engineering diagnostic, never P1 admission. Original model-qualified root index 1, selected/greedy/zero roles (aliases retained), original 41 zero/plus/minus suffix probes at logit epsilon .05 and .1, original saved goal0/Q. No new roots, qualification, thresholds, heldout selection, fitting, training or physics. Synthetic algebra tests validate contracts only, not neural correctness or efficacy.

## Frozen experiment

Reuse capability.prepare and physical-v2.prepare for original context, source/policy bindings and root references. Verify complete sealed capability report, protocol hash, source seals, 123 durable physical branches/journals, 246 exact RGB endpoint slots and CUDA encodings. Recover root RGB from the existing P4 round-0 archive only when its whole-stack bytes match the upstream root reference. Missing/mismatched original RGB fails closed and requires main-thread metadata recovery; this module never reconstructs an environment.

For each role, replay its exact saved five-command prefix from the original latent context once (3 predict calls). Imagined suffix context is that shifted replay context. Actual context uses p4_feedback_matched_v2.advance_observed and encode_cuda: frames=[oldframes1,oldframes2,step5RGB], between=[oldbetween1,exact_saved_prefix_packet]. Encode all three frames on CUDA (three calls, nine frames; no deduplication). Check original raw history normalized on CUDA against saved normalized history exactly. Check reencoded older two latents and actual/predicted prefix against saved vectors with absolute tolerance 2e-5, zero relative tolerance. After checking, share exact imagined older two latent vectors with actual context; both normalized between histories must be exactly identical. Save both contexts and comparison deltas.

Each role/context evaluates all 41 fixed saved tanh suffixes with one model.predict each: 246 suffix calls, 249 total. Every prediction is saved with its corresponding original physical terminal slot, not a nominal/zero proxy. Central finite differences use denominator 2*epsilon in the original saved logit chart, followed by the original Q projection. Save predicted and real 192x10 K, cosine (null for zero norm), Frobenius errors/norms, relative error (null for zero real norm), epsilon sensitivity, each probe's Q-projected prediction error and goal residual. Zero prediction residual is separately labeled and cannot substitute for nonzero probe error. No analytical/autograd Jacobian or fit.

## Commands (from le-wm-main)

Prepare is non-neural and does not allocate CUDA or instantiate physics:

```bash
.venv/bin/python -m pytest -q tests/test_pusht_geometry_feedback_response_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.pusht_geometry_feedback_response_20261027_v1 prepare --output results/pusht_geometry_feedback_response_20261027_v1_prepare_v1
```

Main thread must review protocol.json and prepare_receipt.json before separately authorizing host CUDA. Do not execute this during implementation:

```bash
.venv/bin/python -m experiments.spectral_reachability.pusht_geometry_feedback_response_20261027_v1 run --output results/pusht_geometry_feedback_response_20261027_v1_prepare_v1
```

Prepare exclusively creates the new directory and freezes input hashes, preprocessing contract, protocol digest, and source archives (including test/document, upstream sealed sources and CUDA wrappers). Runtime verifies these before allocation and after measurement, requires frozen eval CUDA with TF32 disabled, instruments actual predictor invocations, saves partial calls/contexts on failure, and exclusively creates a launch receipt to disallow retries/overwrites. No CPU neural fallback or fake success path. P1_admission=false and efficacy_verified=false even if all measurement gates pass. Figures and extra dependencies intentionally omitted per requested minimal delivery.
