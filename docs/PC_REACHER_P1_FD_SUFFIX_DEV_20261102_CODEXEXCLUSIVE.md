# Reacher P1 finite-difference suffix development protocol

## Motivation and contract separation

The fixed-lattice CUDA report has SHA-256
`2cd7dec4965699d5cdc0c6852eb2f02f57f77bb12d590e99247c44acf105920d`.
The
two exposed roots each have 41 saved rows, five endpoint predictions per row,
and re-evaluated final Q scores equal to the saved selector scores. Both
have 0/40 strict eligible. Of 40, 20 per root meet the model immediate
condition; none meet every absolute-tail non-worsening condition. Two and five
respectively satisfy every adjusted-tail non-worsening condition.

As a **post-hoc description only**, 15 and 5 candidates respectively satisfy
the model immediate condition and have a best of five final costs lower than
the neutral prefix's best of those same five costs. This is not the old
strict mechanism criterion, a pre-registered result, or evidence of dynamics
improvement. It motivates an independent candidate-search diagnostic, not a
change to any prior selector or gate.

The original research plan requires a frozen-model program, observed
RGB/goal RGB/action history, action-Jacobian/SVD and action witness, followed
by root-disjoint paired physical evidence against strong resource-matched
comparators. P1 additionally requires non-greedy short-term action and improved
real downstream controllability after recompilation. This method estimates
gradients of model **scores**, not the action-to-latent Jacobian/SVD, and
does not establish that mechanism.

## Fixed model-only search

Each root begins from the neutral 10D parameter prefix. Evaluate neutral and
the 20 central-difference points `+/- 0.01 e_j`. From these model scores form
the immediate gradient and four final-score gradients: mean across all five
tails, the neutral-prefix best tail (lowest index on ties), the neutral tail,
and mean active-minus-neutral tail. Project each negative final-score gradient
onto the immediate-gradient tangent plane, normalize it, add `0.25` times the
unit immediate-gradient normal, and normalize again. Evaluate both radii `0.05`
and `0.10` for each direction, ordered by objective then radius.

Do not select finite-difference probes. A zero immediate gradient aborts the
root; a degenerate tangent produces two recorded neutral duplicates that are
explicitly ineligible for the development selector. No retries, adaptive
step size, early stop, gradient VJPs, physical signals or additional starts.
Save all probes, all nine bank points, all five suffix scores, baseline tail,
direction validity, errors and both selection outcomes.

The separate **development selector** requires immediate model Q cost
`>= neutral + 1e-6` and the candidate's minimum over the same five common
tails `<= neutral's minimum - 1e-6`. It minimizes candidate best-tail cost,
breaking candidate ties by index and tail ties by lowest index. If none
passes, both selected indices are null. This tests optimistic model
planning over a small fixed tail bank, not reconfiguration. Best-of-five is
susceptible to selection bias and model error; it is not a certificate of
successor controllability or a substitute for matched suffix replanning.

In parallel, apply the **unchanged strict mechanism selector** via the existing
`strict.row` implementation and its original sum-of-deltas/index ranking.
Report its result independently as `strict_selected_index`; it may be null
even when `development_selected_index` is not. No development selection is
automatically eligible for physical execution, strict mechanism witness, or
P1 admission. This protocol changes no existing gate.

## Resource and identity contract

Each completed root uses 29 model evaluations: 29 prefix prediction samples
and 290 suffix-token prediction samples (five tails, two tokens), zero
backward VJPs. The 20 probe actions are not part of the nine-row candidate
bank. Separately charge all RGB/goal encoding, goal-Q construction, model
loading, extra re-evaluation, synchronization, wall time and memory. The old
lattice search costs 41 prefix and 410 suffix-token samples, so no equal
total-compute claim is justified. If endpoints are independently re-evaluated,
record the exact extra sample counts.

No physical outcome or privileged state is a selector argument. A Python
callback can nevertheless close over arbitrary data, so the API alone cannot
certify absence of leakage: the future launcher must construct its evaluator
only from authenticated current RGB, goal RGB, action history, frozen model,
action bounds and fixed goal-Q; inspect and bind source provenance and prohibit
simulator state, outcome records and post-hoc labels. Root identity is for
reconstruction/accounting, not a feature supplied to the selector.

## CPU check and future execution

From `le-wm-main`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_reacher_pc_p1_fd_suffix_search_20261102_codexexclusive.py tests/test_reacher_pc_p1_constrained_common_tail_20261102_codexexclusive.py tests/test_reacher_pc_p1_common_tail_tangent_20261102_codexexclusive.py
```

No CUDA launcher or saved consumer is provided or authorized by this
protocol. Before a separately coordinated execution, implement and review a
new exclusive launcher and saved consumer; bind the new code/protocol, old
preflight inputs, root source, checkpoint, frozen parameter hash, goal-Q,
action chart and device identity. Freeze the development roots and their
order `[20263160, 20263161]` before any model run, use one search per root,
record all errors and denominator counts, save probes/parameters/decoded
actions/endpoint arrays, re-score and audit both selectors. Authenticate
hashes before and after; use a fresh non-overwriting output directory. Track
all budgets and any additional verification calls. If preflight fails, do not
run the affected root or relabel old records. No command is claimed ready
until the launcher and consumer exist and pass CPU contract tests.

The two roots and this proposal are development-exposed. Subsequent physical
execution, if proposed, needs its own frozen protocol, equal-resource strong
baselines, reconstructed-action/warmup validation, and genuinely root-disjoint
confirmation under the unchanged gate.
