# RGB selection and velocity-intervention development results

This is a previously exposed four-start panel, not a blinded or independent
confirmation. The model runner consumed an action-only bank and RGB; it did
not load physical outcomes. The bank's direct prefix was originally
generated using privileged one-step physics. Its inclusion prevents a
claim that the complete candidate-generation method is RGB-only.

The decision was sealed before replaying the model-optimized suffixes. A
zero-action warmup generated the RGB pair, and native replay confirmed that
the warmup did not change any initial position or velocity. The model used
a single encoded current-frame context, with the preceding RGB frame used
to estimate velocity. It did **not** supply a five-action historical packet
to the predictor: the warmup was one action, not a full packet. Therefore
this is not a demonstrated full action-history interface, despite the
selection protocol's broad reference to action history. The omission and
associated history/window choice require a separately specified comparison;
do not silently reinterpret the saved decision.

All distances are position distances, lower is better. Passive change is
`distance(prefix + 10 zero actions) - distance(prefix)`. The active
response column shows all five fixed nonzero suffixes, each as
`distance(prefix + suffix) - distance(prefix + 10 zero actions)`, in the
original suffix order. The final column is a different action sequence:
the model-optimized suffix actually replayed after the selected prefix.

| Start | Model-selected prefix, rank | Passive change | Active response, common suffixes 1–5 | Model-chain terminal |
| --- | --- | ---: | --- | ---: |
| (0.8, 0.8) | direction 2, 1/12 | -0.5542 | -0.1904, +0.2770, +0.0966, +0.1280, -0.9111 | 0.5099 |
| (0.8, 1.8) | direct, 1/12 | +0.1086 | -0.2172, +0.2560, +0.1202, +0.0688, -0.5211 | 0.3861 |
| (2.8, 0.8) | random 0, 1/12 | -1.1785 | +0.2360, +0.3862, -0.1783, +0.0206, +0.6427 | 0.4565 |
| (2.8, 1.8) | random 1, 1/12 | -0.7509 | +0.2411, -0.2716, -0.0887, -0.0656, +1.0234 | 0.0664 |

All four selected prefixes physically improved immediate position distance
and all four failed the predeclared *model* reconfiguration qualification.
The model-chain terminal distances of zero/direct were 1.6471/0.4808,
0.6708/0.3861, 1.5955/0.4686 and 0.7824/0.1759, respectively. These are
model-suffix outcomes, not the earlier privileged CEM suffix outcomes.

The controlled velocity intervention reconstructs each prefix-end position
with either its recorded velocity or zero velocity, then applies identical
suffixes. Recorded-velocity reconstruction reproduced all 288 earlier
continuous-run endpoints within 4.45e-16 in terminal distance. On start
(0.8, 1.8), non-improving direction 3's passive change shifts from -0.7631
with recorded velocity to 0 with zero velocity, while its five active
responses also change from `[-0.2619,+0.2341,+0.1479,+0.0459,-0.6527]`
to `[-0.1991,-0.3330,-0.0543,+0.0437,-1.0518]`. On start (2.8, 1.8),
direction 1 has the same passive shift and active responses change from
`[+0.1863,-0.3475,-0.0529,-0.1737,+0.6217]` to
`[-0.0416,-0.2638,-0.4053,-0.3554,-0.2005]`. Thus velocity contributes
both passive drift and action response; this counterfactual does not show
that velocity-dependent reconfiguration is invalid.

On (0.8, 1.8), direction 3 ranked fourth by model terminal score; executing
its own model-optimized suffix yielded 0.3415 versus direct's 0.3861. On
(2.8, 1.8), direction 1 ranked ninth and yielded 0.2552 versus direct's
0.1759. This is a small exposed-panel indication of a possible model
ranking gap on one root, not evidence of stable selection failure or a
statistically meaningful difference. All 48 predicted chains and actual
physical outcomes are preserved for paired analysis.

The [selection](../le-wm-main/results/pc_geometry_rgb_fixed_bank_selection_20261103_codex.json),
[physical replay](../le-wm-main/results/pc_geometry_rgb_fixed_bank_physical_20261103_codex.json),
and [velocity intervention](../le-wm-main/results/pc_native_velocity_intervention_20261103_codex.json)
include full candidate scores, suffixes and trajectories. Six targeted
tests passed, including native replay of the earlier common suffixes.
Direct CEM's small differences, differing random search seeds, lack of a
predeclared practical effect size, and missing independent statistical
confirmation preclude claiming a substantive advantage or P1 admission.
