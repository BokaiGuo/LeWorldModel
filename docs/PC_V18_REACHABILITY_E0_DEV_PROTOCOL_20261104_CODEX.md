# v18 successor-reachability retention: E0 development protocol

Test the new v18 version against unchanged v17 and four-start shared-bank
CEM on eight exact-Wall tasks: start `[-0.5, y, theta]`, `y` in
`{-0.27,0.27}`, `theta` in `{-1.12,1.12}`; goal `[0.5, gy]`, `gy` in
`{-0.22,0.22}`. Check tuples against the enumerated recent panels; broader
provenance is unknown. This is a new development panel, not reopening an
earlier stopped one.

All arms use the same global proposal bank, legal actions, horizon 15 and
4,095 prediction tokens, with the same target score and per-task seed.
v18 evaluates task-augmented successor response over the *remaining*
action horizon for up to four preselected candidates per layer. It keeps
immediate-score slots, one lowest local ridge-residual candidate and then
feature diversity. Each full-horizon finite-difference probe is charged.
The unconstrained local value is a heuristic, not an action-feasibility
certificate. Budget-reserve rules may skip probes; report actual counts.
All final decisions use the same original nonlinear task score and zero
prefix pair; any scored complete chain remains an eligible incumbent.

Record all prediction actions, endpoints, scores, full compiler records,
and timing excluding trace serialization. Audit token counts, shared bank,
all endpoints with alternate same-environment geometry and selection.
Report success `<0.05`, continuous distance, probe counts and cost without
claiming equal total hardware cost or a strong baseline upper bound.

Stop after exactly eight tasks, one attempt per arm including failures.
No retries, substitutions or threshold adjustment. This panel does not
by itself establish PC2, PC3, source independence or visual performance.
