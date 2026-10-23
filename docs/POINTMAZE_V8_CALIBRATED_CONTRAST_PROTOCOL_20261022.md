# PointMaze V8: calibrated residual-contrast release

Status: running on a fresh host-GPU panel; no efficacy result is claimed until
the raw trace and held-out report both complete.

## V7 result and correction target

V7 passed the raw runtime-safety contract but was not authorized for deployment
on its calibration panel.  A corrected, post-execution mechanism audit of its
immutable raw traces found:

- 33 probed roots had an identifiable rank-two prior-anchored residual chart;
- 26 roots remained rank-deficient after safety stopping;
- 28 identifiable roots were rejected by the final visual-agreement gate;
- only one local override was exercised.

Thus the dominant remaining bottleneck is not only affine intercept
identification.  The V7 gate allows a local chart to choose an action only if
the frozen visual-dynamics controller already ranks it no worse, which makes a
decision-changing local correction rare.

## V8 change

V8 preserves all V7 information and safety constraints, changing one release
condition only.  A supported local residual ranking may disagree with visual
dynamics when it predicts a finite terminal-distance advantage at least
\(\delta\), where \(\delta\in\{0,0.02,0.05\}\) is selected on disjoint
calibration roots before held-out root outcomes are read.

The release condition is:

\[
 \operatorname{rank}(A)=2
 \;\land\;
 a\in\operatorname{conv}(0,a_1,a_2)
 \;\land\;
 \widehat C_{\rm visual}(a_v)-\widehat C_{\rm local}(a)\ge\delta.
\]

The next RGB successor still verifies every released correction and quarantines
the chart on a residual failure.  Therefore V8 does not use a predicted
advantage as a substitute for observed-effect verification.

## Fresh RGB calibration contract

The original PointMaze RGB archive ends at root `1899`, while V8 requires new,
root-disjoint calibration roots `1896–1927`.  The new utility
`materialize_pointmaze_rgb_calibration_extension.py` creates RGB frames for
those immutable state trajectories using a renderer calibrated only on the
declared reference roots `1800–1815`.  It records reference/source hashes and
never reads planning outcomes.  V8 uses `1896–1911` for visual calibration,
`1912–1927` for gate selection, and fresh state roots `13336–13399` for the
held-out consequence panel.

## Interpretation rule

V8 can be called a positive repair only if its calibration schedule is
authorized, raw traces pass safety/accounting audits, a local contrast override
is exercised, and the independent held-out paired planning consequence improves
without increasing failures.  If it merely increases overrides without
improving the held-out panel, that is evidence that residual advantage is not
task-transfer sufficient.
