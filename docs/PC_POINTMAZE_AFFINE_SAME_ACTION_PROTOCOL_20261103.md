# Affine-score same-action prediction and physical decomposition

Use every one of the 36 sealed prefix and model-optimized open-loop suffix
chains from `PC_POINTMAZE_AFFINE_NONPROGRESS_ORACLE_PROTOCOL_20261103.md`,
without selecting a favorable candidate. Re-execute exactly the saved
float32 actions from the saved initial states through the real RGB bridge.
Compare cp5 and cp10 physical position with the previously saved audited
position-backend endpoints and cp5 RGB with the sealed bank. Do not change
or select actions using the observed results.

For each identical chain and goal, compute on the frozen CUDA model:

1. Open-loop predicted cp5 and cp10 tokens from the initial RGB-derived
   context, using the executed actions normalized once.
2. A cp10 token prediction conditioned on the chain's actual cp4/cp5 RGB
   and causal RGB velocity, given the identical executed suffix.
3. Encoded actual cp5 and cp10 tokens using each actual RGB and causal
   velocity. These contain RGB estimator error and are not physical truth.
4. Affine-inverse position score for all corresponding tokens, original
   proprio-token MSE, and separate actual physical goal distances.

Record actions, RGB histories, cp5/cp10 states, predicted and actual tokens,
and all scores. Check action normalization round-trip and reproduce the
sealed open-loop score within `atol=1e-6, rtol=1e-4`. For each root report
all 36 within-root candidate pairs: open vs actual encoding, refreshed vs
actual encoding, actual encoding vs physical position distance, and open
and refreshed vs physical. Count exact ties separately using comparisons,
not multiplying score differences. Report token MSE per root, and inspect
the pre-existing 8349 candidates 0, 4 and 7 individually. No assumption
that lower MSE improves ranking or control is allowed.

The cp5 actual observation for every candidate is an oracle counterfactual,
so refreshed ranking is diagnostic only; it does not evaluate a deployable
selector before executing a prefix. Physical labels may not enter scoring.
Pair comparisons share roots and trajectories, so are not independent
samples or confirmatory statistics. Bind checkpoint, normalizers, scene,
bank, selected model scores, oracle actions, and outcome artifacts by hash;
retain all failures. This diagnostic does not demonstrate P1 or PC passage.
