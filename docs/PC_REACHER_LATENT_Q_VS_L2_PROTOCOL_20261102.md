# Goal-Q versus unweighted latent L2: saved development comparison

Use the full 70-branch same-action RGB replay SHA
`4a0c2774ee5569476c668301a53ba8ff6ea183b6a1c3ed1255637f2a65d0f27e`
and the independently goal-encoded report SHA
`5b36c1ac7b35923fcd8f5d28e22340371ab9eb67d96acd72644cb573d8e6f066`.
Require matching seeds and goal RGB hashes. For each of the six non-neutral
prefixes and each identical tail, compare to prefix 0 at step 15. Report
strict sign reversals for predicted versus actual-RGB latent cost, and for
actual-RGB latent cost versus physical qpos distance. Do this for the
already recorded goal-Q and for unweighted latent mean squared distance to
the reconstructed goal embedding. Count exact ties separately.

The target was encoded with the same frozen model. No new planning, action
selection, model inference or physical outcome is performed. The roots,
actions and prior Q mismatch are already known; the 30 comparisons per root
share histories and are not independent observations. This is a development
diagnostic, not a pre-registered generalization claim, physical control
comparison or authorization to replace the planner's objective.
