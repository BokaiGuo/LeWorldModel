# Turn amplitude and translation decomposition

The fixed 2 × 5 candidate grid was evaluated on the eight previously exposed
unicycle goals. Each cell used the same one-packet suffix optimizer and a
verified direct witness. The table reports the mean of the lower error from
the two signs at each amplitude; it is an offline grid summary, not an
independent or separately budgeted planning algorithm.

| Prefix velocity | Turn amplitude | Mean terminal position error |
| --- | ---: | ---: |
| zero | 0.5 | 0.211956 |
| zero | pi/2 | 0.907288 |
| root progress velocity | 0.5 | 0.308751 |
| root progress velocity | pi/2 | 0.009901 |

Translation and turn amplitude interact strongly in this exact toy model.
The previous ordinary arm used progress velocity with maximum turn, whereas
the pure-nullspace arm used zero velocity with 0.5 turn. Their observed
performance gap therefore cannot identify a special benefit or failure of
spectral generation. Pure turning is both an ordinary action-coordinate
candidate and a null direction of the position response; this case by itself
cannot distinguish those explanations.

The entire grid required 64 rollout calls and 11 Jacobian calls per goal,
including direct replay; it is not a four-candidate or equal-total-compute
method comparison. No RGB world model, physical environment, independent
task or statistical confirmation was involved. Results with every candidate,
full actions and call counts are saved in
`le-wm-main/results/pc_atlas_turn_amplitude_grid_20261103.json`.
