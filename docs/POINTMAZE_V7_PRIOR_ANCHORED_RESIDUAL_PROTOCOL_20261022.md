# PointMaze V7: calibration-prior-anchored residual repair

Status: implemented; the independent host-GPU panel has not yet produced a
scientific efficacy result.

## Why V7 exists

V6 repaired the V5 runtime-safety failure: every candidate-anchored probe was
rechecked immediately before execution and unsafe remaining pulses were
rejected.  Its held-out result was nevertheless negative because the local
model was

\[
  \Delta x = b + J a.
\]

The unknown intercept plus two action coordinates needs three non-collinear
safe observations.  On the V6 root panel, safety stopping and the limited
probe budget often prevented that condition; no local rerank was exercised.
This is a model-class/information-design failure, not evidence that a looser
LOO gate would be safe.

## Predeclared V7 intervention

V7 retains the calibration-only visual block dynamics as a zero-action/drift
prior and estimates only its observed local mismatch:

\[
  \Delta x_{\rm RGB} - \widehat{\Delta x}_{\rm visual} = J_{\rm residual}a.
\]

Two non-collinear, task-safe pulses now identify the two-feature residual
Jacobian.  The intervention remains fail-closed:

- pulses are exact members of the fixed 32-action planning bank;
- each pulse is screened again from the current RGB-derived position and
  velocity immediately before physical execution;
- a nonpositive observed progress stops probing and returns unused budget;
- a candidate is eligible only in
  \(\operatorname{conv}(0,a_1,a_2)\), where zero is the declared calibration
  anchor and \(a_1,a_2\) are observed safe pulses;
- the local choice must not worsen the frozen visual-dynamics score;
- the first released action is checked against its next RGB successor and a
  mismatch quarantines the episode-local chart.

There is no fictitious two-point LOO guarantee.  The explicit support rule and
post-execution verification are the V7 release contract.

## Data and split protocol

The host launcher
`experiments/run_pointmaze_htsc_prior_anchored_residual_probe_v7_runtime_safety_cuda.sh`
creates an immutable, fresh state/action extension with roots `13128–13335`.
It holds out `13128–13191`, fits RGB/dynamics calibration on `1864–1879`, and
freezes schedule selection on `1880–1895`.  These sets are root-disjoint from
V6.  CUDA allocation preflight runs before output creation; the job is started
outside Codex's bwrap user namespace because that sandbox cannot access
`/dev/nvidia*`.

## Outcome interpretation

V7 is positive only if all of the following hold on the held-out panel:

1. raw traces pass the runtime-safety and fixed-bank contracts;
2. calibration selects at least one candidate-anchored, rank-two repairable
   schedule without reading held-out outcomes;
3. the local residual override is actually exercised;
4. paired planning performance versus HTSC does not degrade and shows a
   separately reported consequence gain.

If rank two is common but supported overrides have no downstream carry, the
failure moves from *identifiability* to *decision utility*.  If rank two is
still uncommon, the safe information-action budget itself is the limiting
factor.  Neither outcome supports a general local-repair success claim.
