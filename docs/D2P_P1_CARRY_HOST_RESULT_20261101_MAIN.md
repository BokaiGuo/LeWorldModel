# P1 carry feedback implementation and host results

Both known roots completed the frozen four-arm host CUDA/physics protocol.
All models remained frozen. Saved-plan checks accepted 15 nodes for root 0
and 16 for root 1. These checks cover trace hash chains, candidate/tail origin,
selected full plans, executed action prefixes, and saved node accounting.
They do not independently replay every physical event.

| Root | structured | native-style | random | immediate-progress |
|---|---|---|---|---|
| 0 | native success at 15 | no success at 20 | no success at 20 | native success at 17 |
| 1 | no success at 20 | no success at 20 | native success at 19 | native success at 20 |

At root 0 boundary 10, the actual-RGB evaluation selected the prior plan's
tail (score 0.04100865) over all fresh candidates (best 0.05436849).
The first carried packet then reached native success at step 15.
This is evidence that the deployment selection change affected control.
It is not evidence that a non-greedy prefix changed controllability:
the structured root-0 first prefix already improved object position and angle.

Artifacts:

- Root 0 report SHA: `259fc566fa2fd27d71dd207ccf0fb389ce25a4b9771d78209e519659fcb3fd5c`
- Root 1 report SHA: `2ff0e29347996bb828a5710f081edce9f89fc854fc00c242243267ad968004ef`
- Root 0 check SHA: `97136bbced4978326ed282d4b0a8f6757dc49f15e05cb194aa7b6140b6caed0b`
- Root 1 check SHA: `15ccd998d9250d1b0008047027a74f4725df998789e557d3f9245bbc256045fe`

Reports reside in `le-wm-main/results/p1_carry_host_20261101_main_root{0,1}_RUN01`;
checks are `le-wm-main/results/p1_carry_saved_20261101_main_root{0,1}_CHECK01.json`.
The source contract and all previous runs remain unchanged.
P1 admission, fairness, and independent confirmation remain false.
