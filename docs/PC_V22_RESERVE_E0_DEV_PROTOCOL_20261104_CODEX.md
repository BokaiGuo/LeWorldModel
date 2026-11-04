# v22 reserved refinement versus v21: E0 development protocol

Compare the corrected same-budget v22 reserve with unreserved v21, v17
and four-start shared-bank CEM on eight new exact-Wall tasks:
start `[-0.5,y,theta]`, `y` in `{-0.26,0.26}`, `theta` in
`{-1.19,1.19}`; goal `[0.5,gy]`, `gy` in `{-0.23,0.23}`. Exclude exact
tuples in enumerated prior panels; broader source independence is unknown.

All arms receive identical global banks, legal actions, 15-step horizon
and 1,920 prediction-token cap. v22 reserves at least 24 samples per
active refinement start, reducing successor probing if necessary; v21
uses its original unreserved policy. Count every prediction including
probe, shortlist and final comparison. CEM has the same total cap, not
a reduced budget. Final nonlinear task score and `<0.05` success are
fixed; do not interpret same tokens as equal hardware costs.

Save all calls and chains; audit common banks, scores with alternate
same-environment geometry, selected incumbent and actual token counts.
Report restart allocations and outcomes for every task. Stop after eight
attempts per arm, including failures; no retries, root replacement,
threshold adjustment or panel expansion. This is an exposed development
comparison, not a PC2 intervention or independent PC3 confirmation.
