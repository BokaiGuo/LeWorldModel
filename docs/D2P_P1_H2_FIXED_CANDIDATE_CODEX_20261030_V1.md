# P1 exploratory H2 fixed-candidate kernel

## Result

The existing observed H1 compiler evaluates a provided five-action argument
as candidate index 1, but does not mathematically require that the argument
come from an old imagined plan. Replacing it changes the candidate library
and method. This adapter does not claim equivalence to the sealed pilot.

The new candidate is independent of the bundle, condition and outcome:
packet one is five repetitions of `(0.125, 0)`, packet two five repetitions
of `(-0.125, 0)`, exactly float32. Each packet lies within the original
logit ball after inverse tanh. Every row uses the same rule. It does not use
or accept old imagined suffixes. This is a development rule chosen now,
not evidence it was preregistered before historical results were observable.
It must be frozen and reviewed before future physical outcomes are observed.

For each packet the unchanged observed compiler evaluates zero, fixed
candidate, and freshly solved observed ridge using its original nonlinear Q
cost and stable index tie rule; it retains all candidates even if duplicate
and independently replays the winner. The adapter changes the legacy
`original_imagined_suffix` label to `fixed_common_signed_x` in events and
results. Second-packet context is the first winner's predicted context, not
an actual cp10 observation. Each packet must complete ledger counts
5 predictor, 5 action encoder, 1 Jacobian, 1 SVD (AD forward included);
twice that for two packets. This is not proof of actual CUDA resource usage,
and suffix-only accounting does not establish prefix-to-suffix budget parity.

## CPU-only runnable checks

From `le-wm-main`, with the previous blind bundle and SHA:

```
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.codex_p1_h2_fixed_candidate_20261030_v1 --bundle /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_blind_bundle_export_20261030_v1_MAIN01.json --sha256 b5748d81de35df501cb6b7d2d325642080994701a9fa7a634e50ac1c22462856
```

This only validates the bundle and prints the fixed-rule specification and its
digest `e57024781ae4ecb6e8f7b8479903dfd879f0efa5f4d5867c7e18fdb41621726d`;
it does not run neural inference, compile plans, write a claim, or run physics.
Tests call the actual H1 compiler with a synthetic arithmetic backend and
synthetic cp5 latent across all 16 saved rows. This tests the wiring and
declared ledger only, not the real encoder/model or physical effect.

## Missing links before production

1. Eight greedy proofs and their own genuine cp5 observations are missing;
   maintain 24 conditions and do not substitute incumbent.
2. Establish an OS-enforced isolated planner process, with only a validated
   bundle and approved model/runtime files mounted, and test denial of old
   reports, physical state, future frames and outcome paths. A schema
   allowlist, or callback signatures, is not process confinement.
3. Implement actual cp5 RGB encoding on coordinated exclusive host CUDA,
   bind its input pixel hash, model identity, output latent and operation
   receipts; the current function accepts an externally supplied latent and
   does not establish that it came from RGB.
4. Wire the frozen CUDA model and observed compiler with native module hooks,
   the existing OneAD guard, synchronization, strict per-packet resource
   receipts, no CPU fallback, source census and fail-closed partial reporting.
   Arbitrary injected backends and mock ledger callbacks are not trusted.
5. Review and freeze this new candidate method and full total-resource
   comparison against the original unchanged scientific contract, including
   prefix generation and encoder; equal suffix counts alone are insufficient.
6. Implement exclusive new physical attempt claiming and a separate executor
   that replays and verifies root/warmup/prefix on the physical side, then
   executes exactly the selected ten new commands. Verify native stops and
   partial failures with a strict independent saved consumer. No such executor
   or production launch command is supplied here.

No GPU, model inference, simulator or new physical run was executed here.
Known-root exploratory engineering is not P1 admission or confirmation.

## Execution-boundary incident

The combined test command also included the earlier adapter test module. Its
fixture reads the complete historical physical report, including checkpoint
and outcome fields, even though it uses only the prefix checkpoint in its
adapter test. This was outside the current no-results/no-future/no-physical-
truth instruction. The fixed-candidate module and its own test use the blind
bundle and synthetic backend, and the candidate rule does not consume those
fields; nevertheless this overall work session cannot be certified as having
avoided all such reads. Preserve this disclosure and independently review and
freeze the method in a clean isolated context before future physical use.
