# Wall-unicycle finite-difference branch audit

Development-only replay of the three already exposed Atlas searches in
`pc_wall_unicycle_atlas_cem_20261103_DEV01.json`. Instrument the existing
exact backend without changing candidate generation, action bounds,
finite-difference epsilon, comparator, seeds or search selection.

For every central-difference coordinate, record the blocked-translation
boolean sequence in the plus and minus internal rollouts. Count coordinate
pairs whose collision-branch sequences differ, the number of Jacobians
containing at least one such pair, and the largest absolute Jacobian entry.
Recompute each selected program and require it to match the saved action
chain and terminal error. Check Atlas compiler and transition counts also
match the old receipt. Save per-Jacobian counts and maxima.

Different branch sequences show a local nonsmoothness/finite-difference
risk for that sampled coordinate; equal sequences do not prove smoothness.
This audit cannot establish that FD caused the final failure, prove a
learned-model problem, or demonstrate any planning advantage. All replay
calls are additional to the original experiment and are reported separately.
