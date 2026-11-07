# Equal-CEM selection alignment: development audit

For all 40 exposed roots and both selected and reversed prefixes, this audit
recomputed the frozen-model rollout and physically replayed two chains:
the CEM-selected tail and the previously selected B tail. That B tail was
verified to be in the first CEM population. This gives 80 paired comparisons
and 160 physical trajectories, not 80 independent tasks.

All 160 complete physical traces matched their prior recorded traces at the
declared tolerance; starting RGB matched each reset. Recomputed CEM-selected
model endpoint scores matched their sealed scores within the declared
tolerance. The audit records both predicted latent states and same-packet
frozen encodings of real observations. Mean elementwise predicted-versus-real
representation MSE across the 160 chains at packets 1–5 was approximately
**0.0731, 0.0916, 0.0964, 0.1177, 0.1427**.

In the seven previously observed pairs where the CEM-selected chain failed
but the B chain succeeded, the model endpoint score favored the failing chain
in **7/7**. The real endpoint encoding score favored the successful B chain in
**5/7**, but favored the failed CEM chain in **2/7**. Both arms for seed
1888830 share a root, so these seven pairs are not seven independent tasks.
The other 73 pairs remain in the raw record, including cases where both
chains succeed and cases where the two actions are identical.

For these selected development cases, the five reversals are consistent
with predicted-versus-real latent goal-score ranking disagreement; the other
two are consistent with endpoint real-encoding score and ever-success
disagreement. They do not by themselves prove a world-model implementation
bug, model inadequacy, or a general failure rate. In particular, terminal
latent cost and success at any time are not identical targets; physical
trajectories and five packet representations are retained for further checks.
The real encodings are evaluation-only and were not used to select actions.

Two unit tests covered same-tail reconstruction, reversed-prefix mapping,
and trace-tampering detection. Gym issued the existing observation-type and
action-casting warnings. No gate was changed.

[Protocol](PC_TWOROOM_CEM_SELECTION_ALIGNMENT_PROTOCOL_20261107_CODEX.md) ·
[audit code](../le-wm-main/experiments/pc_tworoom_cem_selection_alignment_20261107_codex.py) ·
[pair traces](../le-wm-main/results/pc_tworoom_cem_selection_alignment_20261107_codex.json) ·
[latent representations](../le-wm-main/results/pc_tworoom_cem_selection_alignment_features_20261107_codex.npz)
