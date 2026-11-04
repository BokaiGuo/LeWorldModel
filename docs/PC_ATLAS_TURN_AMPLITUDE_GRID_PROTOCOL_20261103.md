# Fixed turn amplitude and translation grid

Use the same eight exposed exact-unicycle goals and the same root progress
operation and one-packet suffix optimizer as the matched-call ablation. For
each goal, construct a fixed 2 × 5 bank: prefix velocity is either zero or
the root progress action's velocity, and turn is one of
`[-pi/2, -0.5, 0, 0.5, pi/2]`. Use all ten candidates, regardless of goal or
outcome. Apply the same optimizer to each successor; directly replay every
complete action chain and require identical split/full context hashes.
Do not tune the grid using outcomes, or claim any candidate as a deployed
method. Store all candidate actions, prefix/terminal costs, position errors
and operation counts.

This measures how much the prior matched-call result depends on prefix
amplitude and translation on one deterministic model. Pure-turn candidates
are simultaneously ordinary steering actions and local task-response
nullspace actions here; the experiment cannot claim a unique spectral
mechanism. No learned model or physical environment is evaluated.
