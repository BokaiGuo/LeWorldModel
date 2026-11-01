# P1: real-cp5 CEM development comparison

This is a known development root, not an independent confirmation or a
complete-controller matched-budget comparison. The CEM suffix is conditioned
on the already selected structured prefix. It is a local CEM implementation,
not an official reproduction.

The sealed preparation is
`le-wm-main/results/p1_cp5_cem_baseline_20261101_leibniz_PREP_MAIN01.json`,
SHA-256 `9a049ce6dd8bcb065090aa383d82a4ef93a5a6883fe01c3726a7fa6a8ab95e67`.
The host-CUDA result is
`le-wm-main/results/p1_cp5_cem_baseline_20261101_leibniz_RUN_MAIN01/report.json`,
SHA-256 `898d9a289b57259a2e9e3fec0df7f7f2a014e430f448816758a0775219ba10ca`.

All 64 H3 candidates and the winning replay completed. Candidate 61 was
selected with model-Q 0.335489286528884. This number is a prediction, not
physical task success. The actual node ledger records one RGB encoding, 195
predict calls and action encodes, and 65 replay/selection events; no gradient
or Jacobian. Upstream root, prefix, goal construction, physical cost and
model-load cost are not included in this node ledger. Synchronized elapsed
time was 3.59 seconds including model loading, and peak allocated CUDA memory
was 110062592 bytes. Frozen model parameters and buffers retained their
before/after hashes.

An independent saved-trace audit regenerated the PCG64 candidate banks,
verified all elite updates and reproduced the winner from the recorded scores.
It checked report and trace hashes but did not independently recompute the
neural scores. It made zero neural calls. Eight tests, including tampering cases
and the CEM contract tests, passed.

The sealed suffix was subsequently physically replayed along with structured
own and zero suffixes from the same structured prefix. All three actual cp5
states, actions and RGB matched, all branches completed 20 steps, and none
achieved native success. Final jointXY was CEM 345.9236, structured own
120.4565, and zero 160.0752. The physical report is
`le-wm-main/results/p1_cp5_cem_physical_20261101_leibniz_RUN_20261101_01/report.json`,
SHA-256 `5225e645130e67699fe237e050c8bf0f1a5a0c756450b9cf22b479695eb418e9`.
It is an outcome of this one known root, not a matched-cost comparison; the
ledger is dispatch accounting rather than independent hardware metering.
The saved structured-own and CEM records have identical goal, root context,
and actual cp5 latent. Both use the same centered quadratic terminal Q; their
selected predicted costs are respectively 0.1281070 and 0.3354893. The model
thus ranked the structured-own suffix better on this specific pair, consistent
with the physical jointXY ordering; this is not a general model-fidelity test.

The first follow-up same-action RGB diagnostic failed before GPU allocation or
model execution: its lock setup referenced `reverse.extra`, which is absent.
The attempted prepare SHA was
`56419af6070581df8701ca88ba9136871245da38aed023eb48584d8e50a160b1`;
no diagnostic output was produced. This is an implementation failure, not a
model or theory result. The diagnostic must be repaired, tested, resealed and
run before drawing conclusions from it.

The launcher was corrected, its lock entry was tested, and a new prepare and
host-CUDA run completed. The report is
`le-wm-main/results/p1_cp5_cem_rgb_mismatch_20261101_leibniz_RUN_MAIN02/report.json`,
SHA-256 `9445dcb9664d6681d8754eeca742a9994a2b33e41abdcc598d0e342725040a72`.
The recalculated terminal prediction Q 0.3354892856 agrees with the originally
saved scalar 0.3354892865. Predicted versus actual RGB-Q was 0.56486 versus
0.82371 at step 10, 0.53444 versus 0.77469 at step 15, and 0.33549 versus
0.46264 at step 20. Latent RMSE was respectively 0.64857, 0.97557 and
1.04910. The actual RGB-Q decreased from step 10 to step 20 while physical
jointXY worsened from 208.18 to 345.92. This single trajectory illustrates
both prediction discrepancy and representation/task misalignment; it does
not establish their population rates or a unique cause. The diagnostic used
four image encodes and three predictor/action-encoder calls, and retained the
newly recomputed prediction and actual RGB vectors.

A single conditional suffix comparison does not establish a strong
full-controller advantage or open P1; development-disjoint roots, preregistered
mechanism tests, and a comparable total-resource baseline remain required.

## Separate fixed-H3 continuation

The distinct 40-to-50-step known-root contract in
`D2P_P1_FIXED_H3_50_20261101_CODEX.md` has also run on host CUDA. Its report
is `le-wm-main/results/p1_fixed_h3_50_20261101_codex_RUN_MAIN01/report.json`,
SHA-256 `bcfa4c8c8e5c9df25ee96b1cac07af2278dd2b06e2d05162d2fed463b4981740`.
The independent saved-data verifier
`experiments/spectral_reachability/p1_fixed_h3_50_saved_20261101_codex.py`
passed on this report, including actions, RGB, carry and recorded ledgers;
it does not rerun the neural model or measure GPU kernels. Five eligible
branches completed step 50 with no native success. The random branch remains
censored at step 31; no step-50 value is imputed. Final jointXY values were
structured 96.0211, native 79.7572, immediate-progress 410.3281, zero 80.5906,
and same-bank greedy 508.8621. Object position, angle, and pusher metrics are
retained separately in the report; jointXY alone is not native success.

The new continuation dispatched ten real-RGB encodings, 3900 predict/action
encodes, 2400 backward calls, and ten Jacobian/SVD calls; earlier work and
physical replay are separately reported. This extended-horizon development run
does not retroactively change the 20- or 40-step contracts, is not independent
evidence, and does not establish equal total compute or a P1 gate pass.

## Separate full-episode CEM development roots

A distinct full-controller CEM was run from cp0 with actual RGB feedback at
five-step boundaries, using the existing cp10-target protocol on two known
roots. This is not the structured-prefix conditional experiment above. Both
root/target bindings are asserted by the runner. A saved-artifact audit checked
both event chains, candidate updates, chosen packets, returned actions and
first-success calculations; it did not independently rerun the model or
reauthenticate every prepare source. Root 0 reached native success at control step 15; the
previous structured arm also reached native success at step 15. Root 1 reached
native success at control step 6, whereas the prior structured arm stopped
successfully at step 14. Different realized control steps and controller
budgets are retained rather than forced equal.

Protocol reporting defect: the inherited report's generic `node_caps` says
171 predictor calls and `module_hooks_gate_each_node: true`, whereas the
explicit CEM boundary-0 contract caps 195 calls, and the CEM implementation
uses its own driver/journal rather than that inherited hook claim. The report
also retains unrelated old initializer descriptions. Consequently interpret
the explicit CEM contracts and actual journals, not these inherited generic
fields; the historical reports are not modified. A saved-event auditor is
now passing: its initial failure was caused by using the wrong JSON
serialization for the physical event hash, not by a proven event corruption.
An exclusive v2 runner and protocol correct the inherited metadata for future
runs: `codex_p1_full_cem_host_v2_20261101_exclusive.py` and
`D2P_P1_FULL_CEM_HOST_V2_20261101_EXCLUSIVE.md`. Its CPU contract tests and
both source preflights passed; it has not itself been exercised on CUDA or
physically. It does not alter the v1 reports.

The root0 report is
`le-wm-main/results/codex_p1_full_cem_host_20261101_exclusive_root0_RUN_MAIN01/report.json`,
SHA-256 `a973757107da8b98aed242cfb01d5a9079115150d0871c26fe255cc7b960e356`;
root1 is
`le-wm-main/results/codex_p1_full_cem_host_20261101_exclusive_root1_RUN_MAIN01/report.json`,
SHA-256 `0149183cf66149344f28eb3689f80d08149538bc8ad4143be78346c30b8efeb2`.
These are known development roots, not independently selected tasks. This
result is evidence that the candidate CEM controller can solve both observed
tasks; it does not establish a general ranking or equal total-compute protocol.
The original success calculation is reported as-is and its physical meaning
must be checked separately.
