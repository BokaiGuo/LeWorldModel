# PointMaze V8: calibrated residual-contrast release

Status: implemented; no V8 efficacy outcome is claimed until its independent
host-GPU panel and raw-trace audits complete.

## Mechanism from V7

V7 safely identified some rank-two, calibration-prior-anchored residual
charts. Its raw traces passed runtime safety and fixed-candidate membership,
but the inherited visual-consistency gate rejected almost all locally supported
corrections. This is a precise deployment bottleneck: an observed residual
cannot repair a visual ranking if it is required to agree with that ranking.

V7 remains a valid negative result because its calibration configuration was
not authorized and its active arm did not establish a task benefit. V8 does
not overwrite or reinterpret that result.

## V8 change and safeguards

V8 changes only the final local release rule. After HTSC non-release, a local
candidate can disagree with the visual controller only when the residual chart
predicts a one-block terminal-distance improvement at least equal to a
calibration-frozen threshold. It still requires:

- two safe non-collinear probe actions that are exact fixed-bank candidates;
- support in \(\operatorname{conv}(0,a_1,a_2)\), with calibrated zero-action
  drift as the anchor;
- current-state task-safety screening immediately before every pulse;
- stopping after observed nonprogress; and
- RGB successor verification and chart quarantine after a released action.

The advantage is selected only on V8 gate-calibration roots. It is not a
held-out threshold adjustment and does not remove action support or successor
verification checks.

## Independence

The V8 launcher uses a fresh extension `13336–13543`, visual calibration roots
`1896–1911`, and gate calibration roots `1912–1927`. None overlaps V7's
held-out roots, and V8 evaluates only new roots. The host CUDA preflight runs
before an immutable output directory is created.

## Decision rule

V8 earns a positive repair claim only if calibration authorizes its frozen
configuration, raw traces pass, local overrides are exercised, and an
independent held-out paired comparison improves the declared planning outcome
without a safety regression. A larger release rate alone is not a benefit.
