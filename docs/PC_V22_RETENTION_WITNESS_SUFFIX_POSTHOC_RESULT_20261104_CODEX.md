# v22 fixed-bank witness-prefix suffix diagnostic result

All four tasks had a previously replayed legal witness with terminal
distance below `1.4e-7`. The witness prefix was frozen, and each received
the fixed-bank diagnostic's same three 120-sample full-action suffix CEM
starts. No start recovered a successful suffix:

| Root | Witness terminal distance | Best of three suffix searches |
| --- | ---: | ---: |
| 0 | 0.000000131 | 0.495730 |
| 1 | 0.000000076 | 0.574259 |
| 2 | 0.000000117 | 0.509578 |
| 3 | 0.000000107 | 0.511424 |

The privileged witness prefixes have known successful continuations, so
these results directly show that this specific suffix optimizer and
budget are insufficient to recover those continuations on these four
exposed tasks. They do not establish a coding bug, impossibility for the
automatically generated prefixes, or failure under more search.

All scored actions were recorded; the audit verified fixed prefixes,
action bounds, allocation and scores via alternate same-environment
geometry. The witnessed actions were never used by the formal compiler.
This post-outcome diagnostic is not a PC2 or PC3 result. Data and audit:
`le-wm-main/results/pc_v22_retention_witness_suffix_posthoc_20261104_codex/`.
