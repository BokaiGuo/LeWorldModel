# Shared-RGB B0 paired physical replay

One known-root development pair was executed in two separate native PushT
environments. Each replayed the original warmup and 15-action prefix, checked
the complete 106-event native/RGB reference before dispatching the 10-action
suffix, and completed all 25 planned actions. No new model calls were made.

The two new model winners were byte-identical to the actions already sealed
in the earlier B0 physical preparation. The bridge independently checks the
new model output and capture against both old physical trajectories, including
the entire shared prefix, three observed RGB frames, goal, root, and suffix
bytes. It does not relabel the old planner witness as a new witness.

| Arm | Native success | Object position error | Wrapped angle error | Native distance |
| --- | --- | ---: | ---: | ---: |
| joint_multistart | false | 79.061736 | 0.901563 | 150.210955 |
| prefix_chain | false | 13.234272 | 0.080992 | 137.720253 |

These are actual native output metrics from this single paired replay, not
latent objectives. The same commands and source were already observed in the
prior B0 execution, so this is a deterministic known-root repeat, not an
independent root, independent generalization, or a population effect.
Neither arm achieved native success. It does not establish equal total
compute, a causal composition mechanism, or P1/P2 admission.

Bridge contract SHA-256:
`c0397aeed4754a29e93beba303a8af8157c71f525c414847a66f40da0b6dbdc2`.
The physical report SHA-256 is
`a858d9fbe89ee997461391e6d4ea5171cfed326e010d6dc3ddd3dbceb5bba1ae`
and completion SHA-256 is
`b61f33104bfd34f70f9c17932d536f42a629732869a96dc03a2dd11cd3567fce`.
The existing saved consumer revalidated journal chronology, action
dispatch/return, native labels, all image bytes and pixels, and source seals;
the new bridge audited their connection to this model output. The consumer
shares validation implementation with the producer, not an independently
authored verifier. New bridge tests: 2 passed, including action, frame, guard,
and contract-SHA rejection.

CPU recheck:

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_shared_rgb_physical_bridge_20261102_exclusive audit \
  --contract /home/gbk/code/LeWorldModel/le-wm-main/results/p2_shared_rgb_physical_bridge_20261102_exclusive_CONTRACT_B0_01.json \
  --contract-sha256 c0397aeed4754a29e93beba303a8af8157c71f525c414847a66f40da0b6dbdc2 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p2_multistart_joint_physical_20261030_v1_shared_rgb_B0_20261102_01 \
  --report-sha256 a858d9fbe89ee997461391e6d4ea5171cfed326e010d6dc3ddd3dbceb5bba1ae \
  --completion-sha256 b61f33104bfd34f70f9c17932d536f42a629732869a96dc03a2dd11cd3567fce
```
