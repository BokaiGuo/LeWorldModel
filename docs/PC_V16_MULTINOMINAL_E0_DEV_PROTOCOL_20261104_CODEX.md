# Multi-nominal response generation: exact-Wall development protocol

This is a new development hypothesis, not a replacement for stopped
panels: a broader task-independent nominal-action distribution may cover
useful response directions missing from the original two nominal points.
Post-hoc witness proximity motivated the hypothesis; witness actions,
states and feasibility information are not inputs to any planner.

Use four exact-Wall tasks with `x=-0.5`, `y` in `{-0.27,0.27}`, heading
in `{-1.1,1.1}` and goal `[0.5,-0.23]`. Check tuples against explicitly
enumerated previous panels, including the v15 and fixed-bank panels.
Broader provenance remains unknown. Stop on any overlap.

Run exactly three paired arms once per task with horizon 15, raw actions
in `[-1,1]`, packet length 5 and hard cap 4,095 prediction tokens:

* unchanged v13 with its existing two nominal points;
* v16: unchanged v13 algorithm with those same two points plus four
  independent constant nominals sampled uniformly in `[-0.8,0.8]^2`
  with seed `2029193000+root`; no task, witness or state input is used
  in nominal generation;
* same-global-bank full per-step CEM with paired zero-prefix comparison.

Global bank seeds and candidates are shared; all other seeds and policies
are fixed in the runner. Report all plan scores, success `<0.05`, actual
prediction tokens, SVD counts and outer wall time; token equality does not
mean equal total cost. Save all predictor batches and scored candidates,
then independently replay them with the alternate same-environment
geometry. Check shared bank, nominal generation, action bounds, ledger and
best scored full-chain selection. Failure counts in its arm's denominator.

Stop after four roots, including failures; no retry, replacement or tuning.
This is not a PC2 intervention, total-cost comparison, visual result or
independent PC3 confirmation. A positive signal requires independent
mechanism and ablation evaluation; a negative result does not establish
that wider nominals are universally ineffective.
