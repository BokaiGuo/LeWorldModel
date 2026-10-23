# V9: passive local residual repair under a fixed action budget

## Question

V6--V8 used dedicated, candidate-anchored safety-screened probe actions.  The
probes either stopped before identifying the local chart or consumed enough of
the 25-block budget that the resulting chart had little opportunity to affect
the terminal task.  V9 tests that mechanism directly: can RGB successors of
ordinary, already-budgeted controller actions supply enough local excitation
to repair an action ranking without a probe tax?

## Fixed policy

The passive arm begins identically to `effect_adapter_rgb_htsc`.  Every
ordinary executed action is followed by an RGB observation.  It updates only
the residual chart

\[
  \Delta x_{\mathrm{RGB}}-\widehat{\Delta x}_{\mathrm{visual}}=J_{\mathrm{res}}a.
\]

No dedicated information action is permitted.  A local choice is permitted
only when all of the following hold:

1. two independent executed actions make the residual Jacobian rank two;
2. the candidate lies in `conv(0, a_1, a_2)`;
3. the global HTSC certificate declined to release;
4. the local action is successor-verified from the next RGB frame.

Before rank two, behavior is exactly the RGB visual-dynamics fallback used by
HTSC.  A failed successor verification quarantines the local chart.

## Fresh split and execution contract

`run_pointmaze_htsc_passive_local_v9_host_cuda.sh` creates state roots
13,544--13,751 with seed `202610901`.  It materializes separate renderer-only
calibration observations on roots 1,928--1,959 and evaluates held-out roots
13,544--13,607.  These ranges are disjoint from V5--V8 roots.  The result
directory is immutable and is created only after a real host CUDA/cuDNN
preflight passes.

The launcher must run through a host terminal or `systemd-run --user`; Codex's
bwrap user namespace intentionally has no NVIDIA device nodes.  GPU failure is
an execution failure, never a CPU fallback.

## Primary estimand and falsification

The primary estimand is paired passive-minus-HTSC terminal distance and
success probability over the frozen 64-root panel, with paired bootstrap
intervals.  Mechanism reporting includes passive RGB update count, rank-two
release count, local override count, support coverage and post-action
verification failures.

A result is not a control improvement unless the calibration-free held-out
panel improves without an observed safety or success loss.  A near-zero effect
with many passive updates would reject the "probe tax only" explanation; few
rank-two charts would instead identify passive action diversity as the
remaining bottleneck.

## Completed result

The fresh 64-root host-GPU panel completed with CUDA 13.0 and a real cuDNN
probe.  The passive arm made 660 RGB successor updates and 42 nominal local
releases, with no extra probe actions.  Its paired terminal-distance and
success deltas against HTSC were both exactly zero (95% paired bootstrap
interval `[0, 0]`).

This is not an inactive-chart artifact: the mechanism audit shows that every
one of the 42 releases selected the same candidate as visual dynamics.  Thus
the test rejects the hypothesis that dedicated probe budget was the principal
barrier.  In this fixed bank, the rank-two residual chart can be locally
identified and supported but does not provide a decision-divergent ranking.
The next intervention must target contrastive action information, not merely
collect more ordinary successors.
