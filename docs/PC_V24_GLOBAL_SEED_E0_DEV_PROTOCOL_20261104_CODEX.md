# v24 global-seed exact-Wall development protocol

Before running, freeze eight tasks with starts `(-0.5, ±0.26, ±1.15)` and
goals `(0.5, ±0.14)` and reject exact overlap with the enumerated prior
development panels. Broader provenance is unknown.

Hypothesis: allowing a second distinct globally scored, unprivileged complete
chain to seed the compiler's shared full-action CEM may improve coverage
relative to passing only the best bank chain. Compare v23, v24 (two global
seed slots including the winner), and the existing shared-bank CEM baseline.
The extra v24 chain is rescored through the same metered predictor and its
cost is charged. All arms have the same initial bank, task objective,
1920 prediction-token cap, bounded float32 actions and paired zero-prefix
comparison. Each arm uses the same shared CEM implementation; differing
seed composition and computational allocation are the intervention.

Save every prediction batch and audit input bounds, common bank, token
accounting, selected full-chain score and alternate same-environment
geometry. Report all roots including failures, distance and success under the
existing 0.05 threshold, plus warm-start use and optimization allocation.
Stop after eight roots without replacement or retuning. This is an E0
development ablation, not independent PC3 confirmation or PC2 mechanism
evidence. If the hypothesis is unsupported, preserve the result.
