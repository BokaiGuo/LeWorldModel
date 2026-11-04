# State-blind primitive CEM on exposed exact wall

Post-outcome development comparison on the same single exposed exact wall
task and seeds 20261103, 20261104 and 20261105. This tests whether a generic
low-dimensional action-program parameterization, without exact prefix pose,
waypoint or wall geometry in proposal generation, can cover useful plans.
The evaluator is the exact task backend and uses its true terminal position;
this is privileged for RGB deployment and not a learned-model or P1 result.

Run two separate CEM searches using the same start, terminal Euclidean
position objective, horizon 16, raw action bounds, seeds and 4 rounds of
51 candidates, retaining 6 elites and best evaluated program. Both clip
sampled controls before scoring and elite updates, use stable score ordering,
initialize at zero with standard deviation half their coordinate bounds and
floor standard deviation at 1e-5.

* Flat arm searches all 32 raw action coordinates.
* Primitive arm searches four bounded scalars `turn1, move1, turn2, move2`;
  the fixed program is one turn1 command, five move1 commands, two turn2
  commands, five move2 commands and three zeros. Move and turn components use
  their original per-command limits. No position, wall, privileged witness,
  waypoint or selected existing action enters proposal generation.

Save rounds, all selected programs and exact replay, action clock, backend
calls, transitions, elapsed time and sources. Both arms evaluate exactly
204 complete 16-step programs (3,264 transitions); comparison is not an
equal end-to-end hardware budget because dimensions and overhead differ.
Require the same evaluator, bounds and replay agreement, and use a new
exclusive result. Seeds are repeated searches on one task, not independent
instances. This is ordinary program-parameterization ablation, not proof
of a spectral or reconfiguration mechanism. Preserve all outcomes.
