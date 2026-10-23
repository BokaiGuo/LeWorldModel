# TwoRoom verified-local calibrated V1

## Question

Can a frozen capacity readout contribute decision value beyond an identical
RGB observed-effect fallback, rather than merely avoiding the frozen LeWM's
known failures?

This is a C2/C3 test.  The teacher, checkpoint, candidate bank, and local RGB
verification implementation are frozen.  It does not train or tune GDD-WM on
the new live seeds.

## Diagnosed failure being addressed

Earlier TwoRoom results mixed two effects:

1. a capacity readout can choose a different candidate; and
2. a local RGB effect map plus fallback can repair failures even when the
   teacher provides no useful extra decision information.

Comparing teacher-plus-fallback only with bare frozen LeWM cannot separate
them.  The control in this protocol is therefore a **matched frozen-LeWM
hybrid**: the same RGB input, `vision_door16` candidate bank, CEM budget,
effect map, effect verification, and fallback rule, with only the planning
cost changed from `euclidean` to the frozen `capacity_safe_readout`.

## Three-stage sealed protocol

| stage | seeds | permitted use |
| --- | --- | --- |
| calibration | `20264032..20264095` | select one of three declared repair routes |
| outcome | `20264096..20264223` | evaluate only the selected route against its matched baseline |

The three declared routes are:

1. `verified_local`: RGB effect map, effect-policy verification, affine
   intercept correction;
2. `phase_local`: route 1 plus reactive fallback during door phase and once
   at phase transition;
3. `local_ranker`: route 1 plus the supported local RGB candidate ranker
   with frozen blend weight `0.5`.

The calibration selector authorizes a route only if the teacher hybrid has
positive paired success, at least four teacher-only wins, and a one-sided exact
paired sign-test value no larger than `0.10` against its matched frozen-LeWM
hybrid.  If no route passes, the 128 outcome seeds remain unopened.

### Panel-integrity guard

The two stages have different fixed cardinalities and are not allowed to share
an `--episodes` argument: calibration executes exactly 64 roots and the
authorized outcome executes exactly 128 roots.  The selector accepts only the
ordered calibration seeds `20264032..20264095`; the outcome auditor accepts
only `20264096..20264223`.  A truncated panel, a reordered panel, or a panel
with the right count but different roots is a protocol rejection, not a small
sample outcome or an excuse to tune the gate.

## Invariants

- Host CUDA execution only; the runner begins with a real recurrent CUDA
  allocation preflight and refuses CPU fallback.
- Both arms of every pair use identical ordered seeds and protocol fields.
- The live policy consumes rendered current/goal RGB; no simulator state is a
  policy input.
- The selector reads calibration reports only.  The outcome audit rejects an
  outcome report that appears in the calibration candidate list.
- Any authorized outcome reports the teacher contribution relative to its
  exact local-fallback control, separately from the RGB-reactive ceiling.

## Claim boundary

A passing fresh outcome would establish a scoped TwoRoom matched-arm result,
not universal transfer, action-conditional coverage, or superiority to the
RGB-reactive controller.  A calibration rejection is still informative: it
means the currently declared teacher cannot demonstrate nonempty contribution
above the local RGB repair under this deployment contract.

## Artifacts

- runner: `le-wm-main/experiments/gdd_wm/run_tworoom_verified_local_calibrated_v1_cuda.sh`
- calibration selector: `le-wm-main/experiments/gdd_wm/select_tworoom_verified_local_calibration.py`
- outcome auditor: `le-wm-main/experiments/gdd_wm/audit_tworoom_verified_local_calibrated_v1.py`
