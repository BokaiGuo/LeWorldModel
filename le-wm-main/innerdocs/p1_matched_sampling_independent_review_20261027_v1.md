# Independent sampling review

Reviewed source SHA256 `42dcf7f17340e940727f292eb02d100532ee7bdb46b1287618af58b7d13a5305`.
No source/sealed changes; no neural, GPU, or simulator execution by reviewer.

## Verdict

No normal-launch blocker found. Independent tests: 5 passed, 1 strict expected
failure reproducing R1. Tests use nonzero, unequal incoming normalized tokens,
nonconstant raw histories and full three-state autoregression; neither a model
nor an environment is constructed. Preparation/source identity remains external.

R1 (failure evidence index, non-blocking for a normally completed run): in `run`,
the arm row is appended before `store.put(name+'.json',result)`, but journal_file
is attached only AFTER that put succeeds. Injecting failure of root_0_cem.json
produces partial_or_failed and retains 126/126 completed forwards, a valid durable
gzip journal and final report, yet its arm has no journal_file/journal_sha256.
Thus finalization does not hash/index that existing evidence. The independent
strict-xfail test reaches the missing journal_file assertion after checking the
decoded journal and actual partial counts. No lost candidates, false success or
normal-run numerical error is alleged. Relay to Meitner through MAIN; reviewer
has no direct thread messaging capability. Do not unseal this version to fix it.

## Protocol checks

- Each root/method: 6x10=60 full H2 candidates, exactly candidate0 zero,
  59 nonzero stochastic slots; zero consumes no PCG64 draws. Independently
  reconstructed Gaussian and IID Gaussian-direction/U**.1 radius draws agree.
- CEM: current-batch stable elite3, feasible executed logits, ddof0, sigma floor
  .05; six paid updates including unused final update. No extra mean candidates.
- Seed table root0 CEM/IID 20261027/20261127; root1 20262027/20262127.
  Root order fixed by SHA rather than outcomes; no replacement/retry.
- Both packets restart from the original full context; same goal/Q=I192,
  float32 tanh raw controls and normalized incoming tokens. Whole/segmented
  winner replays are additional independent six forwards, not free cache reuse.
- 120 sampling + 6 replay =126 predict/actionencoder per arm, aggregate504 each;
  no J/VJP/SVD/solve/image work. Candidate arrays precede selection; generation
  frozen before evaluation. Durable compressed journal retains all floats.
- This is FORWARD-CALL matching only. Witness derivatives/backprop/SVD/solve are
  not matched, nor FLOPs/walltime; cannot claim equal total compute or SOTA.

The reported actual CUDA completion and separate saved-array tests are MAIN's
work, not this review's execution. Neither mock success nor real forward counts
establish direct plateau, independent legal attainability admission, fresh roots,
physical consequences, or scientific P1 passage. Reused calibration roots and
causal-padding history retain their original limitations.

## Reproduce (small pure-array tests)

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -rx -p no:cacheprovider tests/test_p1_matched_sampling_independent_review_20261027_v1.py
```
