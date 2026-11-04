# v19 actual-candidate response error: saved-data diagnostic

Use only saved plans and prediction traces from the stopped v19 E0
same-chain panel. For each candidate in each saved recursive layer,
reconstruct its parent/nominal point from the recorded beam and find that
exact scored nominal in the saved prediction batches. Combine its saved
small-step response matrix with the candidate's actual action displacement
and compare the resulting linearized feature prediction with the saved
nonlinear candidate feature. Feature space includes the two exact-Wall
position coordinates and the scale-1 task score used by this compiler.

Report displacement maximum, absolute feature error, error relative to
actual feature change when that change has norm above `1e-8`, and cosine
only for nonzero vectors. Summarize by depth and report coverage/failures.
Verify the source hashes and match actual float32 nominal and candidate
actions. This is retrospective, only measures model-internal finite-step
linearity, and cannot establish physical task relevance, the source of
error or causal responsibility for planning failure. Do not generate new
actions, evaluate dynamics, or change the stopped panel.
