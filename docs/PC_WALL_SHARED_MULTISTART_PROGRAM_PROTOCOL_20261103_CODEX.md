# Shared multistart suffix-program search on the exposed wall task

This is a new post-outcome development experiment on one exact-dynamics
unicycle wall task and the three existing seeds. It is not an RGB method,
independent root evaluation, physical experiment, or admission test.
Its exact terminal task cost and state are analytical-model quantities.

At each seed use the existing four-action compiler to compute local progress
and its first nonzero action-Jacobian spectral direction at raw zero. Form
six root prefixes: zero, that progress, signed top spectral direction with
`max(abs(action)/HIGH)=0.3`, and two seeded uniform bounded raw packets.
The candidate generator may inspect only the model context, fixed goal,
action limits and model rollouts/Jacobians, not a saved waypoint or an
observed outcome. Keep all six; score each in the same backend.

After each prefix, apply the same two-start CEM optimizer to a 12-command
program. Program parameters are four independent bounded coordinates in
`[-1,1]`: two total turns (each scaled by pi and split into the minimum
number of bounded turn-only commands) and two nonnegative lengths (each
scaled from [-1,1] to [0,1.5] and split into the minimum number of
0.3-bounded translation-only commands). Arrange turn, translation, turn,
translation; truncate beyond 12 commands or pad with zeros. Proposal
construction does not see position, wall, kinematics beyond action bounds,
or physical outcome. Each start uses three rounds ×34 candidates and 4 elites,
zero mean, std 0.5, clipped candidates used both for evaluation and elite
updates, floor 1e-5. All six prefixes within a seed use identical paired
noise arrays, with independent fixed noise for each of the two starts.
Retain the earliest best complete suffix over both starts and all rounds.

As direct controls, run a flat 16-command CEM and an identical four-parameter
program CEM, both for two starts × three rounds ×153 candidates. Thus each
direct control gets 918 full-rollout evaluations / 14,688 state transitions,
matching the six suffix searches' 6×204×12=14,688 suffix transitions. The
method also pays candidate generation, Jacobian and prefix rollouts; report
them separately rather than claiming equal total compute. Direct optimizer
population differs because the total evaluation budget differs. For each
source record seeds, all prefix/suffix actions, model costs, exact full
replays, stochastic traces, attempted/returned calls, finite-difference
transitions, and wall time.

Full-method selection is the actual minimum terminal model cost across the
six complete programs, stable by bank order. Separately report the minimum
cost between zero and same-bank greedy root candidates; a nonprogress prefix
is qualified only if its prefix cost is no lower than the root cost and its
full-chain cost strictly beats both controls. Report whether that very same
qualified index equals the actual selected index; otherwise no mechanism
claim for the executed chain. If no candidate qualifies, report abstention,
while still reporting the ordinary minimum-cost complete plan. No external
solution injection, real outcomes or gate modifications are allowed.

Trace and hash all source/protocol files, save under a new exclusive path,
and validate bounds, action binding, replay, noise pairing, optimization
accounting, and the distinct selected/qualified indices. Results on this
single exposed task do not establish statistical or generalization claims.
