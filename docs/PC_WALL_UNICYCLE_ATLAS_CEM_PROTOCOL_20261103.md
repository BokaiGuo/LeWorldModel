# Wall-unicycle Atlas/CEM development protocol

This exact-dynamics single-task development test combines the existing wall
detour with orientation-dependent action response. It is not a learned-world-
model result, an independent task panel, or an admission test.

State is `(x,y,theta)`, start `(-0.5,0,0)`, goal position `(0.5,0)`.
Each of 16 commands has forward displacement `v` in `[-0.3,0.3]` and heading
change `omega` in `[-pi/2,pi/2]`. First propose position displacement along
the current heading; the existing wall transition blocks a crossing of `x=0`
at `|y|<1`, then heading changes even if translation was blocked.
Terminal task cost is Euclidean position error. A successful continuous
crossing must leave the initial unit-distance circle, but this fact alone
does not establish an individual prefix's response or a P1 mechanism.

Predeclare and verify a privileged feasible witness through `(0,1.1)`, with
in-place rotations and bounded straight segments. It is only a feasibility
ceiling and is never passed to either planner.

Compare:

1. Existing `AtlasSearch`, 4 packets × 4 commands, beam width 2, 2 random
   bridges and the existing local compiler. Position-only task metric. Its
   finite-difference Jacobian uses symmetric epsilon `1e-5` on raw actions;
   each internal rollout and each command transition is counted. Collision
   discontinuities can make this approximation misleading and must not be
   attributed to a learned model.
2. Full-horizon CEM over the same bounded 16-command space. Use four rounds,
   1/8 elites, clipped normal proposals and retain best actually evaluated
   full program. Set the per-round population to
   `floor(Atlas transition count / (4*16))` for the corresponding seed. This
   uses no more exact state transitions than Atlas, including Atlas Jacobian
   finite differences, but neither makes wall time nor floating point work
   equal. The CEM is not claimed as a tuned strongest baseline.

Seeds 20261103–20261105 are repeated searches on the same start and goal,
not independent tasks. Save actions, trajectories, rejected translations,
scores, attempted and returned backend calls, all internal transitions,
population and time. Replay each selected program and the witness and check
bounds and output equality. The witness and physical/analytic state are
privileged in a real RGB-only setting; here the analytical state is the
explicit exact model, so no RGB-only claim is possible. Full P1 mechanism
and independent strong-baseline evidence remain required.
