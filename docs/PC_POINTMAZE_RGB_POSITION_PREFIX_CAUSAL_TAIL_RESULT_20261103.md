# Real RGB tail replanning after RGB-position prefixes

The prewritten `PC_POINTMAZE_RGB_POSITION_PREFIX_CAUSAL_TAIL_PROTOCOL_20261103.md`
executed 16 real RGB trajectories on four exposed development roots. The
eight fixed prefixes were chosen earlier by the RGB-position planner, not by
this experiment's outcomes. Within each pair the five-action prefix and all
six observed prefix frames match exactly. Each arm uses its own causally
observed cp5 RGB, the same initial task mask, same tail CEM innovations, and
48 one-packet model samples. All 16 terminal RGBs and terminal positions
match a separate legacy replay and position-backend replay respectively.

| Root | Prefix source | Zero-velocity endpoint distance | Causal-RGB-velocity endpoint distance |
| --- | --- | ---: | ---: |
| 1884 | random CEM | 0.3703 | 0.1573 |
| 1884 | spectral CEM | 0.3266 | 0.1197 |
| 1885 | random CEM | 0.1692 | 0.0773 |
| 1885 | spectral CEM | 0.0431 | 0.0431 |
| 1886 | random CEM | 0.4018 | 0.0601 |
| 1886 | spectral CEM | 0.4108 | 0.1432 |
| 1887 | random CEM | 0.7356 | 0.1979 |
| 1887 | spectral CEM | 0.6510 | 0.0729 |

The causal arm's actual terminal distance is smaller in seven of eight
paired prefix conditions and tied in one. Under the strict offline
endpoint-distance check `<0.5`, zero velocity satisfies six conditions and
causal velocity all eight. The two improved threshold cases are both on root
1887 and are not independent successes. This measures the **selected tails
actually executed**, unlike the prior full-bank cp5 oracle.

This result does not establish independent generalization, reconfiguration,
P1 admission, or an advantage of the spectral prefix over the random prefix.
Roots were previously exposed, prefixes were fixed by a development planner,
and the threshold is evaluated at step 10 after full execution rather than
certified native online success. Both arms have equal observation access and
tail prediction counts, but this receipt does not certify equal total
end-to-end budget for random versus spectral initial planning. Causal
velocity is a quantized 0.05-second image displacement, not ground-truth
instantaneous velocity. The original task mask is fixed here; the earlier
mixed-outcome causal study used other prefixes and recomputed a cp5 mask,
so these results do not isolate which difference caused the contrast.
No simulator state was supplied to either tail planner.

The next relevant step is to freeze an end-to-end RGB-only causal-velocity
controller and compare it against matching random and spectral baselines with
full accounting and a precommitted development holdout. This result does not
relieve the remaining need for an explicit non-greedy mechanism intervention,
fair strong baselines, independently sourced roots, and source certification.

Artifacts:

- JSON `le-wm-main/results/a1_pointmaze_rgb_prefix_causal_tail_20261103_DEV01.json`,
  SHA-256 `fd20c6f83df7dd6e2c0a21cfb05f25ab9d2627dfe2958266eb5b36e1381ecb2f`.
- NPZ `le-wm-main/results/a1_pointmaze_rgb_prefix_causal_tail_20261103_DEV01.npz`,
  SHA-256 `90ed51a45b496b27cf55c1ce55546ca60eba4521f5ece6dc6d4ad5e098ca36e0`.
- Saved CEM trace, selected action, paired history, evaluation distance and
  receipt checks: **1 passed**. This test does not rerun CUDA or physics.
- Actual run: 16 ten-step trajectories and 768 tail prediction samples;
  separate 16-trajectory physical replay for verification is extra diagnostic
  cost. Wall time and per-arm feedback/tail timing are in the receipt.
