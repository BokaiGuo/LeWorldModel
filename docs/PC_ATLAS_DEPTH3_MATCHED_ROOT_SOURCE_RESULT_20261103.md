# Depth-3 matched root-candidate source: exposed development result

Under the separately written protocol, all three root-source arms selected the
shared local-progress root on all 24 paired goal/seed cases. The selected
three-action chains were exactly identical across arms in every pair. Their
mean terminal Euclidean position error was 0.003535 for random, spectral
and ordinary candidates alike. There were 8 distinct previously exposed
goals; the three seeds do not constitute independent goals.

| Goal index | Terminal error, all arms |
| ---: | ---: |
| 0 | 0.0003221 |
| 1 | 0.0012791 |
| 2 | 0.0022337 |
| 3 | 0.0031678 |
| 4 | 0.0040716 |
| 5 | 0.0049364 |
| 6 | 0.0057538 |
| 7 | 0.0065157 |

No incremental spectral-candidate benefit is detectable on this panel under
this optimizer. The shared depth-3 search and progress root were already
sufficient to reach low error; this does not show that the root source can
never matter, nor that another task or learned-model backend behaves similarly.
The evidence supports moving to a benchmark with independently verified
non-greedy need, not declaring a spectral advantage or a negative theory result.

Among the two extra root candidates, the average best terminal position
error was 0.031665 for random, 0.005755 for spectral, and 0.003535 for
ordinary. None strictly beat the progress-root terminal cost; ordinary
extras tied it within `1e-12` cost in all 24 cases. These within-bank
candidate comparisons are descriptive and do not convert the equal
final-method outcome into an advantage claim.

Every run used 108 attempted/returned rollout calls, 116 rollout action
steps, 13 attempted/returned Jacobians and 13 differentiated action steps.
The 72 arms total 7,776 rollout calls and 936 Jacobians. The mean measured
wall times were 7.08 ms random, 6.92 ms spectral and 7.29 ms ordinary; these
tiny single-process timings are descriptive, not a reliable cost ranking.
All arms computed progress geometry at all nodes, including when random or
ordinary root extras did not need it. Therefore calls are deliberately
matched by padding, not evidence of equal end-to-end natural compute.

The runner uses unmodified `AtlasSearch.search` with a subclass that changes
the root bank only; later banks are common. All selected action witnesses
were checked by direct/split replay in the search. Existing Atlas tests plus
the new tests passed: **10 passed**. No physical trajectory, trained-model
inference, new independent root, or formal admission was evaluated.

An additional independent saved-result audit replayed all 288 saved
candidate programs against the exact backend, checked root action bindings,
recomputed costs and selections, and checked the artifact hash. Two
tampering tests reject altered selection and root action. With these tests,
the relevant regression set reports **12 passed**. Replay is not a new
physical or learned-model experiment; its calls are outside the experiment
counts above.

- Protocol: `docs/PC_ATLAS_DEPTH3_MATCHED_ROOT_SOURCE_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_atlas_depth3_matched_root_source_20261103.py`
- Results: `le-wm-main/results/pc_atlas_depth3_matched_root_source_20261103_DEV01.json`
  (SHA-256 `cce9442bd7120bfcf71e8c443a3d0a3482244e46e9060dcab0d4c828fbd918c2`)
- Tests: `le-wm-main/tests/test_pc_atlas_depth3_matched_root_source_20261103.py`
- Saved-result audit:
  `le-wm-main/experiments/spectral_reachability/pc_atlas_depth3_root_saved_audit_20261103.py`
