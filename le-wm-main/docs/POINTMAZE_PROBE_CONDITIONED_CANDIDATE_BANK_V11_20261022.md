# PointMaze V11: RGB probe-conditioned candidate construction

## Motivation

V6--V8 showed that an RGB-observed local action--effect chart can be formed,
but only about 4.5--7.0% of a globally fixed candidate bank lay inside that
chart's observed action hull.  Consequently, local re-ranking had almost no
legal opportunities.  V10 separately closed the fixed endpoint-bank route:
the preregistered lower-tail oracle-headroom gate failed on every independent
bank design attempt.

V11 changes the object of intervention, not a threshold.  After four charged,
task-safe RGB probes, it constructs a compact candidate bank wholly inside the
convex hull of executed probe actions.  Its local affine chart is fitted only
to the corresponding RGB-observed action effects, and it ranks only the new
supported candidates.  If no rank-identifiable chart yields at least four
predicted goal-progressing candidates, V11 retains the root-fixed RGB fallback
bank.

## Frozen mechanism contract

- No policy input includes simulator state, oracle costs, future successors,
  or held-out task outcomes.
- Inputs to bank construction are exactly executed probe actions, their
  RGB-decoded observed effects, and the current RGB-decoded goal direction.
- Every generated candidate is rechecked against the observed action hull.
- Four probes replace four ordinary planning blocks; both arms have a
  20-block post-warm-up execution budget.
- The r1 attempt on roots `16824..17031` was invalidated before a complete
  report because an empty task-safe probe schedule raised rather than failing
  closed.  The r2 attempt on `17032..17239` then exposed the analogous
  zero-goal-direction boundary after a probe reached the current waypoint.
  Both directories are retained as invalid implementation artifacts.  The
  corrected r3 panel used fresh roots `17240..17447` and completed.  It proved
  construction support but was strongly negative because its chart was reused
  across later state transitions.  That is a scope violation, not an efficacy
  result.  The r4 panel on `17448..17655` caught a separate runtime contract
  defect: the local bank reached a legacy 32-action-only validator and was
  rejected before its first action.  The corrected r5 uses fresh roots
  `17656..17863`; a local chart may authorize exactly one action and is then
  quarantined until a new probe protocol is introduced.  Calibration roots
  `1928..1943` remain disjoint.
- CUDA is required and is preflighted on the host; no sandbox or CPU fallback
  is permitted.

## Interpretation boundary

The first V11 panel is a prospective mechanism and feasibility panel.  Its
main pre-outcome measurements are local-bank construction rate, candidate
hull coverage, predicted local headroom, and fail-closed frequency.  Its
terminal comparison is an independent consequence of the frozen mechanism,
but it cannot by itself establish a calibrated recovery policy or a general
world-model claim.  A deployment claim requires a subsequent root-disjoint
calibration/certificate stage and an untouched outcome panel.

## r5 host-CUDA outcome and stop condition

The r5 host-CUDA run completed on 64 fresh roots (`17656..17719`) with no
detector failures.  It is the first valid execution of the one-step local-bank
mechanism: 24 roots constructed a locally supported bank and consumed exactly
one local action; 40 roots failed construction closed.  Nevertheless the
mechanism is negative against the same-budget visual-dynamics control:

| arm | success | mean terminal distance |
| --- | ---: | ---: |
| visual dynamics | 34 / 64 | 0.402 |
| probe-conditioned local bank | 27 / 64 | 0.753 |

The paired local-minus-visual terminal-distance mean is `+0.352` (bootstrap
95% interval approximately `[+0.212, +0.500]`; lower is better).  The
constructed subset is also not positive (`n=24`, mean paired difference
`+0.104`).  This closes the unconditional-probe version of V11.  It does not
show that local construction is impossible: it shows that construction support
alone is insufficient to justify spending probe budget or overriding the RGB
fallback.

The next admissible protocol must freeze, on separate calibration roots, both
an information-action gate (whether the expected value of probing exceeds its
budget) and an action-release gate (whether the local candidate is predicted
to beat the visual fallback).  It must then evaluate those frozen gates once
on untouched roots; it must not retune using r5 outcomes.
