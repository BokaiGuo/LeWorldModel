# P3 capability cp15 and uniform-tail cp20 join

Stage B, ten-branch physical, cp15 RGB three-layer and the later uniform raw-zero tail already have saved full receipts and summaries. This new post-hoc join adds no run and does not replace their separate analyses. Inputs are the exact MAIN three-layer summary SHA `36b31985a27a16156afb3dea8c852b84199ea1814cfcba986c5b889db39bb01c` and tail summary SHA `e190a357ec02243baa635b84d28cf0382310dcec31b2048c8876ab45ac40dbb6`; their full upstream verification is inherited, not rerun.

Five fixed endpoints (root0 C/Z/P, root1 Z/P), two branches U/Z2 and cp15/20 are retained. Before joining, each available cp15 requires exact same native state, goal7, PNG file SHA, pixel SHA, and success; a missing point is null. Independent float64 `math.fsum` recalculates cp15 model/real Q and native seven-axis squared error at both times. The model and real values exist only at cp15; cp20 has no saved encoded latent or model forecast and stays null. Gains are Z2 minus U: positive is smaller U error; cp20-minus-cp15 changes are descriptive differences, not a causal test. Native success remains separate from distance, and velocity means error relative to the original nonzero target agent velocity. Zero raw actions do not imply static state or better reward.

Both known roots remain known; four logical C slots and eight C controls are retained unchanged from the three-layer summary. No new selection, gate, fresh roots, P1/P3/C3 admission, training, NN, physics or upstream strict. This method is post-hoc and is not a prespecified follow-up claim.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_capability_tail_join_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_capability_tail_join_20261029_v1 build --three /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_three_layer_summary_20261029_v1_MAIN01.json --tail /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_summary_20261029_v1_MAIN01.json --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_tail_join_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_capability_tail_join_20261029_v1 verify --three /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_three_layer_summary_20261029_v1_MAIN01.json --tail /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_summary_20261029_v1_MAIN01.json --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_tail_join_20261029_v1_MAIN01.json --output-sha ACTUAL_OUTPUT_SHA
```
