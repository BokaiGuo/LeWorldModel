# Fixed-bank suffix expression: development result

All four new exact-Wall tasks and 40 paired candidate comparisons
completed. The unchanged v13 generator produced ten first-layer candidates
per task. Each candidate received equal three-start, 1,440-proposal CEM
search under both four-knot and per-step (ten-knot) suffix parameterization.

| Root | Four-knot all-bank best | Per-step all-bank best | Four-knot retained best | Per-step retained best |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.500022 | 0.500006 | 0.500188 | 0.500018 |
| 1 | 0.500014 | 0.500033 | 0.500014 | 0.500134 |
| 2 | 0.500008 | 0.500043 | 0.500016 | 0.500142 |
| 3 | 0.500143 | 0.500036 | 0.500148 | 0.500240 |

Neither expression found a continuation below the existing 0.05 success
threshold, either over all candidates or retained candidates: 0/4 tasks per
arm. The all-bank best score was slightly lower in each arm on two roots.
These small, exposed-development results do not establish a population
ranking or imply that no feasible suffix exists. In this configuration,
removing the four-knot restriction alone did not rescue the fixed-bank
search; candidate coverage, optimization, task feasibility and other
causes remain possible.

The run saved the compiler's actual prediction batches as well as all
suffix batches, correcting the trace omission of the earlier budget
diagnostic for this **new** panel only. A separate audit checked all saved
actions and scores against the alternate same-environment geometry,
generation token accounting, fixed-prefix mapping, 480 samples per
start, and best-score incumbency. The four compilers used 7,678
five-step prediction tokens; the suffix stage used 115,240 full-chain
samples, or 345,720 tokens, including one zero-tail score per candidate.
This is not an equal-total-cost full-planner comparison and the alternate
geometry is not an independent environment.

This diagnostic does not establish PC2 or PC3, does not certify the
task's 15-step physical feasibility, and does not address visual
feedback or provenance independence. The predeclared four-task stop is
reached; do not rerun, replace roots, or tune this panel.

Protocol: `PC_V13_FIXEDBANK_SUFFIX_EXPRESSION_DEV_PROTOCOL_20261104_CODEX.md`.
Manifest, raw traces, results and audit:
`le-wm-main/results/pc_v13_fixedbank_suffix_expression_dev_20261104_codex/`.
