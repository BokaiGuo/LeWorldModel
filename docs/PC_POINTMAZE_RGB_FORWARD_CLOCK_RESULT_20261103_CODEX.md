# PointMaze RGB render clock diagnostic

This is a post-hoc replay of previously selected actions, not a new planning
evaluation or independent confirmation. No historical files were modified.

On the NVIDIA host, serial replay reproduces all 1,152 sealed legacy RGB
hashes. Across 64 trajectories and 1,088 command boundaries, directly read
particle geometry under the legacy rendering path agrees with the immediately
preceding integration tick's qpos plus the fixed geometry offset to at most
6.29e-16 in Euclidean distance. Its maximum distance from the current qpos
plus that offset is 0.04252. With forward immediately before rendering, the
corresponding geometry agrees with that arm's current qpos plus offset to
6.29e-16. All 1,088 post-action images change; this pixel observation is not
used to infer qpos.

At every measured boundary the immediate qpos, qvel, and simulation time
before and after forward are bitwise equal. However, comparing all subsequent
integration ticks between the two independent simulators finds 258 differing
trajectory/tick pairs, first at zero-based trajectory 19, tick 72. Maximum
absolute component differences are 4.440892098500626e-16 in position and
6.661338147750939e-15 in velocity; time matches exactly. These are small but
nonzero and the cause is unestablished. Therefore the predeclared exact
trajectory-invariance gate fails: a direct in-place forward bridge is not
delivered as a trajectory-preserving fix. A separate rendering-only simulator
could be investigated under a new contract, including full state transfer,
render context correctness, and exact physical-trajectory preservation; it is
not validated here.

The first simultaneous-context attempt also failed the sealed-image gate; its
reason remains unknown. The serial attempt passed the image gate and then
failed the trajectory gate, and its journal is retained. A separate, declared
diagnostic replay saves all per-tick state and geometry arrays and quantifies
that failure. The saved raw arm's boundary physical states also match the
earlier native stepwise audit exactly. The test suite checks the saved arrays,
the forward-boundary condition, geometry relations, previous-state linkage,
and existing fail-closed behavior (3 tests passed).

Scope is the fixed 64 paths, model/XML, runtime and NVIDIA renderer in the
saved JSON; there were no model or optimizer calls. These measurements do not
bound numerical drift on other paths, establish a planning performance change,
or retroactively alter the sealed RGB or decisions.
