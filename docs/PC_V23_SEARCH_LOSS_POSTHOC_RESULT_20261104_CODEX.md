# v23 search-loss post-hoc diagnostics

The baseline-prefix injection diagnostic joined the selected CEM suffix to
every recorded v23 prefix, separately at five and ten steps. For the only
successful baseline reference (root 4, terminal distance 0.046084), the
best generated-prefix-plus-reference-suffix distances were 0.614026 and
0.572536; among retained prefixes they were 0.705926 and 1.614022.
No injection on any of the eight roots met the existing 0.05 threshold.
These results expose a coverage/retention loss for this particular reference
tail, not proof that other tails cannot solve the task.

Separately, saved actions from all 16 v23/shared-CEM plans reconstructed
exactly from their seeds, recorded scores and the CEM elite-update rule.
Every start used at least two generations; the actual mean and scale before
and after each generation are recorded in the reconstruction artifact.
The compiler spent 114–131 full-chain samples in refinement per root while
the baseline spent 573; both paid the same total token cap. This identifies
a budget allocation difference, not a defective CEM update.

Both analyses are post hoc, use the same exact environment/RNG and cannot
pass a scientific gate. Source artifacts:
`le-wm-main/results/pc_v23_baseline_prefix_injection_posthoc_20261104_codex.json`
and `le-wm-main/results/pc_v23_cem_generation_reconstruction_20261104_codex.json`.
