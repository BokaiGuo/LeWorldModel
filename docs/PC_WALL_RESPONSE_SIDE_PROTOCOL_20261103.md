# Full-horizon wall response: prospective development protocol

Preserve prior results and gates. This exposed wall family is an exact-model
development experiment, not independent/RGB evidence. No outcome-based tasks.
Use existing pc_exact_wall_kill.step_batch (wall x=0, |y|<1). Tasks are the
Cartesian product start y=(-.25,.25), heading=(-pi/4,pi/4), goal y=(-.25,.25);
start x=-.5, goal x=.5. Horizon 16, speed bound .3, turn bound pi/2.
Distance at step 16 is the primary cost; .05 is a descriptive success threshold.

Six arms, caps 512 and 1024 complete-chain model calls:
task/gain rank-one response initialization; full no-SVD ridge initialization;
random initialization; flat zero-initialized CEM; state-blind primitive CEM.
Four shared nominal normalized action chains are uniform [-.6,.6],
seed 2026111400+task. Each response matrix is 2x32, central differences
epsilon=1e-5 in normalized coordinates. Charge 64 FD, one nominal and one
candidate score per nominal, total 264. Damping 1e-3. Choose rank-one by task
contribution or gain; ridge solves the 2x2 dual system without SVD. Clip
the infinity norm of the update to .5 before action clipping.
Random uses the same four nominals and four charged scores.
Flat and primitive use four zero starts and four scores.

All use the same four-start diagonal CEM engine, four rounds per start,
std .5, clipped actions, best eighth elites (at least two), std floor .05.
Allocate the remaining budget across 16 start-round slots, retaining remainder
samples, and preserve best scored incumbent. Noise seed is 2026111500+task.
Responses are initialization, not a hierarchical compiler; make no automatic
program-discovery claim. The primitive maps four parameters to turn, five
translations, two turns, five translations, three zeros, with signed bounded
speed. Its optimizer operates in four dimensions; other arms use all 32.

Record all selected chains, traces, prefix distance at step 4, crossings,
blocked actions, source hashes, actual rollout samples/transitions, solves,
SVD calls and elapsed time. All methods receive the same model-call cap;
this is not equal FLOPs. There is no execution feedback in this experiment.
All validation replays and rank diagnostics are separately post-selection.

Report all 8 tasks and both caps. A separately constructed geometric detour
at y=+1.1/-1.1 is feasibility-only: it is never passed to planners or used
to exclude tasks. A failure of this check invalidates feasibility claims,
not the recorded planner outputs. No mechanism or P1 passage from distance
alone. No post-outcome tuning or widening of the historical gate.
