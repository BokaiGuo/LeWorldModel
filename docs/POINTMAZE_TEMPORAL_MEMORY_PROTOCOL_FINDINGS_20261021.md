# PointMaze temporal-memory protocol findings

## What the V5 negative result did and did not establish

The original V5 recurrent result is not evidence that short visual history is
useless. Its implementation and protocol combined three distinct effects:

- The original additive readout could not let history change the relative
  effect of two candidate actions.
- Training used 16 action sequences with pulse norms `0.134--0.593`, whereas
  14/16 held sequences exceeded the training maximum and reached `0.762`.
- The runner validated and therefore parsed the held-out outcome file before
  persisting its model-selection artifact; its calibration gate also reused
  roots used for width/temperature selection.

V5 is retained as a strong action-extrapolation stress artifact only. It must
not be cited as a temporal-memory no-go result or as independent evidence for
a repaired FiLM architecture.

## Repairs and sealed evidence

V6 uses causal 8-frame latent/action histories, a pulse-aware candidate GRU,
a FiLM interaction, root-disjoint train/calibration/outcome splits, a
selection/authorization split inside calibration, exact frozen model weights,
and outcome simulation only after the frozen-selection hash exists. The GPU
runtime is also checked with an actual cuDNN GRU probe, not allocation alone.

The three predeclared numeric-action-OOD panels V6.1--V6.3 had 96 independent
outcome roots. Their root-level pooled recurrent-over-action-only regret gain
was `+0.00635`, 95% bootstrap CI `[+0.00026, +0.01363]`. This aggregate is
not a stable deployment result: V6.3 was directionally negative and two
panels failed their independent authorization gate, so the released controller
correctly fell back to action-only on those panels.

V6.4 is a separately declared candidate-in-support control: every one of its
16 held action sequences occurred verbatim among the 64 source-training
sequences, while all source, calibration, and outcome roots remained fresh and
disjoint from prior panels. It has 32 held-out roots and produced a
recurrent-over-action-only gain of `-0.00571`, 95% CI
`[-0.01808, +0.00342]`; its authorization gate also failed. Therefore action
sequence extrapolation is not sufficient to explain the lack of stable history
gain.

## Current conclusion

The corrected experiment establishes a useful separation:

1. V5's negative result had material architecture and protocol confounds.
2. A history-candidate interaction is numerically active after training, but
   that fact alone does not make it decision-relevant.
3. Under the current renderer-backed frozen JEPA representation, short history
   does not yet provide stable root-level action-ranking improvement over the
   action-only effect baseline, either under numeric action OOD or in the first
   candidate-in-support control.

This is offline PointMaze evidence with a shared privileged source-position
task-cost readout. It is neither a live RGB planning result nor a general
claim about world models. The next mechanism target is thus history-to-mode
or history-to-local-effect supervision, not another unsealed architecture
search.

## V6.5 observability control and implementation hardening

V6.5 is a fresh, root-disjoint `shared_candidate_in_support` panel built on
the host RTX 5060 Ti after the V6.4 outcome was already closed.  It freezes
four same-bank score arms before outcome simulation:

1. action-only pulse effect;
2. correctly aligned visual-history FiLM effect;
3. the identical FiLM architecture trained and evaluated with a deterministic
   cyclic root-history mismatch;
4. a diagnostic-only privileged `[x,y,vx,vy]` plus candidate effect head.

The candidate oracle has zero regret, so the fixed bank contains a genuine
selection opportunity.  On 32 newly held roots, however, the frozen visual
history arm has mean regret `0.02173`, versus `0.01199` for action-only and
`0.01037` for cyclicly mismatched history.  Its paired gains are `-0.00975`
over action-only (95% CI `[-0.02833, +0.00564]`) and `-0.01136` over the
cyclic mismatch (95% CI `[-0.02845, +0.00299]`).  The privileged state arm is
directionally but not conclusively better than action-only: `+0.00250`, 95%
CI `[-0.00391, +0.00940]`.

This is not evidence that PointMaze is intrinsically unidentifiable: the
candidate oracle remains perfect.  It does show that the current state-effect
head does not establish a strong state-conditioned ceiling on this horizon,
and that the aligned visual history fails to beat an equal-capacity
misalignment control.  Hence V6.5 cannot be used to justify a visual-memory
controller.

The runner now recomputes every candidate hash from the executed `[20,2]`
tensor, requires contiguous candidate IDs, exact root-invariant candidate
sequences and matching bank hashes before opening labels, and supports
hash-checked relocation of frozen checkpoints.  Static validation passes for
the V6.5 artifacts, so these identity checks do not explain its result.  V5
remains invalid for sealed-outcome and root-isolation reasons; V6.5's result
is an independent, strictly more controlled negative diagnostic rather than a
retroactive repair of V5.

## V6.6: the exploratory motion stratum does not transport

V6.6 tested the only predeclared response to V6.5's exploratory heterogeneity:
an input-only visual-motion gate selects recurrent or action-only scoring from
three calibration-selection quantiles and two directions, then faces a
disjoint authorization half.  The panel uses fresh roots/action hashes and
the same exact candidate-in-support condition as V6.4/V6.5.

Selection chose `recurrent_when_high` at the 25% visual-motion quantile, but
authorization rejected it: paired gain over action-only was `-0.00410`, 95%
CI `[-0.01277, +0.00421]`.  Its frozen 32-root outcome score was `0.03452`,
versus `0.03174` for action-only; the active controller therefore remained
action-only fallback.  This is decisive against promoting the V6.5 motion
stratum into an online gate.  The next temporal experiment must change the
intervention distribution itself: it should screen equal-energy action pairs
on separate design roots for state-dependent terminal-effect contrast, then
test whether visual history learns that interaction on new roots.  It must not
choose pairs by outcome task cost or reuse the design roots for calibration or
evaluation.

## V6.7: an interaction-screened bank makes aligned history diagnostic

V6.7 executed that predeclared mechanism test. Sixty-four separate design
roots screened 96 equal-energy pulse-order proposals using only terminal
effect contrast: pairs were retained for low mean pair effect and high
cross-root variation. No goal or task cost was available during screening.
The resulting sixteen held actions were included verbatim in the sixty-four
source-training actions; design, train, calibration, and outcome roots were
disjoint. All four score arms were frozen before the first outcome replay.

On 32 new outcome roots, aligned visual history reduced mean regret from
`0.00942` (action-only) to `0.00386`. The paired root-level gain was
`+0.00556`, 95% bootstrap CI `[+0.00104, +0.01131]`; it also exceeded the
equal-capacity cyclic-history mismatch control by `+0.00977`, CI
`[+0.00049, +0.02510]`. The diagnostic privileged-state arm showed the same
direction (`+0.00523`, CI `[+0.00010, +0.01187]`). The candidate oracle had
zero regret, confirming that the bank contains a genuine ranking opportunity.

This is one-panel mechanism evidence, not a controller result. Its disjoint
calibration authorization gain was `+0.00051` with CI
`[-0.00091, +0.00261]`, so the released policy correctly remained
action-only fallback. The bounded conclusion is that V6.5/V6.6 did not show
history is intrinsically useless; they used banks with weak or uncontrolled
state-action contrast. V6.7 shows aligned visual history can become
decision-relevant when the intervention bank requires that interaction. A
fresh, unchanged replication is required before changing any deployment rule.

## V6.8: independent replication keeps the mechanism but not deployment gain

V6.8 replicated the V6.7 state-interaction protocol with new root/action
seeds and excluded all V1--V6.7 root and held-action identities. It also used
the strengthened v2 panel contract: explicit shared support, identity-bound
calibration labels, explicit frozen policy, and a hash-bound current runner.
The GPU build confirmed all 16 held actions were source-supported, with zero
prior root/action-hash overlap. Outcome labels were generated only after the
frozen selection was written.

The V6.8 aligned visual arm achieved mean regret `0.01189`, compared with
`0.01404` for action-only. Its primary paired gain was `+0.00214`, 95% CI
`[-0.00893, +0.01895]`, so the primary effect did not independently clear the
predeclared threshold. It did strongly exceed the cyclic-history mismatch
control (`+0.02100`, CI `[+0.00633, +0.04037]`), which confirms the model uses
the aligned history rather than merely its parameter count. The privileged
state diagnostic was directionally positive but inconclusive (`+0.00444`, CI
`[-0.00696, +0.02144]`).

The selection/authorization gate again rejected recurrent deployment:
authorization gain `+0.00068`, CI `[-0.01275, +0.01435]`; the active policy
therefore remained action-only fallback. Pooling the two independently sealed
interaction panels yields a positive mean gain `+0.00385` across 64 outcome
roots, but the bootstrap CI remains `[-0.00259, +0.01256]`. Thus the correct
claim is now sharper: interaction-screening exposes an aligned-history signal
relative to a matched history-mismatch control, but current data do not prove
a stable advantage over the action-only baseline or an authorized controller.

## V6.9: fresh free-region replication after protocol and launcher repair

V6.9 was a new v3 panel with 64 train, 48 calibration, and 32 outcome roots;
the candidate bank used the predeclared state-interaction design under
`shared_candidate_in_support`, and the goal was the free-region `[0.8, 0.8]`.
All prior evaluation roots, action hashes, and metadata-only interaction
design roots were excluded. The builder recorded the manifest, exact root and
candidate counts, causal history ledger, and delayed outcome-label path.

This run also found and fixed two non-scientific launcher defects that could
otherwise masquerade as experiment failure: direct script execution broke
the `experiments.*` imports, and only the build stage used the cuDNN path
wrapper while calibration/score inherited incompatible system cuDNN 9.13.
The corrected launcher uses module entry points and the bundled PyTorch cuDNN
wrapper for all three stages. CUDA allocation and recurrent-kernel preflight
passed with PyTorch CUDA 13.0 and cuDNN 9.20.

The fresh result remains negative. On the independent authorization roots,
the recurrent-over-action-only mean gain was `-0.00168`, 95% CI
`[-0.00780, +0.00276]`, so recurrent deployment was rejected before outcome
simulation. On 32 held-out outcome roots, recurrent regret was `0.01713`
versus `0.01374` for action-only; the active policy therefore used the
predeclared fail-closed fallback. Outcome labels were generated only after
the frozen selection artifact was written.

V6.9 changes the diagnosis: after removing V5/V6 protocol contamination and
runtime inconsistencies, the remaining failure is not an artifact of action
extrapolation or the legacy wall-sensitive goal. The current history-
candidate interaction is not reliably transferable across fresh roots. The
supported next target is supervision or intervention design that identifies
the within-mode action effect, not another unsealed selector search.
