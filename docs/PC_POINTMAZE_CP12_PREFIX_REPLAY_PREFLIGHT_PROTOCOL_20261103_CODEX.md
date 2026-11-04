# CP12 fixed-prefix replay preflight

Before attempting the requested single/three-frame physical replanning,
replay the first sealed branch (raw arm, candidate 2) of each of the four
exposed cp12 roots, 8343, 8345, 8349, 8351. Use its saved reset and exactly
the first 12 saved raw commands with the current native RGB bridge. Record
RGB hashes, differing channel counts and intensity differences against
saved RGB[0:13], plus RGB-derived causal state differences at 2, 7 and 12.
No model, optimizer or new tail action is run, and simulator state is used
only for the environment reset.

The preflight passes only if every frame matches byte-for-byte. If it fails,
do not silently substitute saved RGB as current feedback, change renderer
parameters, relax tolerance after seeing outcomes, or execute the planned
suffix comparison; report the failure and leave its cause unresolved. This
is a four-branch engineering check, not an estimate of control performance.
Record source and code hashes and refuse to overwrite existing artifacts.
