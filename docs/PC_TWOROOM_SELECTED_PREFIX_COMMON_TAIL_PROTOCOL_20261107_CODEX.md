# Selected-prefix common-tail intervention: development protocol

Use all 40 previously exposed TwoRoom roots and the descriptor arm's actual
previously sealed offline selection, without selecting cases by outcome.
This is not an online executed planner or independent confirmation.

For each root, fix its selected ten-step prefix, selected B tail and all four
common B tails from the existing sealed plan. From the same reset state replay
each B tail after (1) the selected prefix, (2) ten zero actions, and
(3) the selected prefix with its action order reversed. All resulting
chains are 25 legal steps. Capture distance, termination and truncation at
every step, including the ten-step prefix boundary. Restore the same state and
verify initial RGB exactly on every replay. Outcomes must never feed selection.

First verify that selected-prefix replays reproduce the previously sealed
full-matrix physical traces; abort if termination, truncation or distances
disagree (absolute distance tolerance 1e-5). Report all-root per-tail success,
paired exclusive wins for selected versus each control, selected B success,
prefix success and prefix-end distance. These analyses diagnose whether
changing a prefix changes later physical outcomes; they do not isolate
non-immediate advantage, equal-budget reoptimization, or discovery by the
descriptor. Do not infer a PC2 pass. Keep model and truth provenance checks,
all failed cases and complete traces.
