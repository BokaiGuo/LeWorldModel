# PointMaze obstacle RGB spectral-CEM development protocol

Before inspecting outcomes, use roots 8343, 8345, 8349 and 8351, one from
each previously exposed V16 geometry stratum, at archive state index 40.
Use the fixed target position [1.8, 2.8] and render its RGB observation
through the same persistent bridge. These roots are development data.

Compare random and spectral initialization using the same frozen model,
AP-compatible CEM, score, seeds, bounds, and RGB feedback at steps 0, 5,
10, 15 and 20. Execute all 25 high-level actions without early stopping.
One high-level action comprises five MuJoCo integration steps. This is not
the old V16 24-block reset-per-action contract. Score endpoint distance to
the fixed goal after execution; a 0.25 threshold is an offline endpoint
diagnostic, not authenticated online success.

The spectral arm injects the top right singular direction of the two-JVP
repeated-action response as in the earlier eight-root development protocol.
Each arm scores 50 candidates at each decision. Report its additional JVP
and SVD work and wall time; equal candidate count does not imply equal total
budget. Planner receives only RGB-derived state, goal RGB and the frozen
model; true simulator state is used only for reset, goal rendering and
post-execution evaluation.

Save each action and RGB frame. Replay each full trajectory independently
through a fresh RGB bridge and require matching final RGB; obtain endpoint
positions from the separate audited position backend. The RGB bridge does
not expose its own state, so this does not compare two endpoint positions.
Fail closed on changed input hashes, model parameters,
invalid actions or incomplete runs; do not select roots by result. This
experiment does not establish P1, independent generalization, full official
AP reproduction, or a non-greedy mechanism.
