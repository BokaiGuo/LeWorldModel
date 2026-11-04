# v19 refinement provenance: saved-trace result

On all eight stopped-panel tasks, v18 and v19 entered full-chain refinement
with exactly the same float32 action chain and the same number of
refinement calls. Both used the same task seed; their final scores were
also identical on all eight. The initial refinement chain matched a
global-bank action on two tasks and a different previously scored
full-horizon chain on six.

The diagnostic confirms that these two versions' different retained-index
records did not lead to different refinement starting actions in this
panel. It does not prove that response candidates were unnecessary:
six seeds were not global-bank actions, and the common seed can be affected
by earlier full-horizon candidate or finite-difference evaluations.
This is a saved-trace observation, not a randomized component ablation.

Of the six seeds outside the global bank, three occur in a 21-chain
successor finite-difference batch and three in the final recursive
candidate batch. These probes were deliberately included among eligible
full-chain incumbents; their prediction cost was charged. Consequently,
one must not attribute those three seeds to reachability *retention*
alone.

No new dynamics calls or search were performed. Source hashes and the
previous audit were checked; output:
`le-wm-main/results/pc_v19_retention_fix_e0_dev_20261104_codex/refinement_provenance_posthoc.json`.
Protocol: `PC_V19_REFINEMENT_PROVENANCE_POSTHOC_PROTOCOL_20261104_CODEX.md`.
