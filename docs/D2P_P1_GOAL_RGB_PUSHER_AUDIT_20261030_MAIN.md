# P1 goal RGB pusher visibility audit

The four original goal RGB packets were read from the verified known-root
source join. No physics or neural inference was performed. In both `off` and
`on` conditions, exact canonical blue pixels `[65,105,225]` are present:

| root | goal blue pixels | goal centroid (224px image) | target pusher position (512 units) |
| --- | ---: | --- | --- |
| 0 | 41 | (161.83, 129.27) | (370.76, 297.09) |
| 1 | 32 | (62.50, 153.50) | (143.52, 351.69) |

The same centroid occurs under both overlay conditions for each root; complete
image hashes differ. The approximate positions agree with the 224/512
rendering scale, but the exact-color centroid is a raster statistic, not an
exact state estimator. The goal image does contain visible pusher-colored
pixels in these two cases; this does **not** establish that the frozen
encoder, dynamics predictor or latent cost preserve or prioritize their
location. It also does not establish visibility in other tasks or under
occlusion. The native state values here are evaluation-only and are not
introduced into the RGB planner.

The saved result is
`le-wm-main/results/codex_p1_goal_rgb_pusher_audit_20261030_MAIN01.json`,
SHA256 `822b933c9930f95d4b2d2cec7eb12f41ab69d39aabc11b3a389951303235ff81`.
The audit implementation validates the original source bindings and image
format. Two tests check centroid/empty behavior and invalid dtype rejection.
The next discriminating test should compare frozen-model encodings of
controlled pusher displacements while holding the object and goal fixed,
using host CUDA and equal input processing; this audit itself is not that
test, not a planning result, and does not open P1.

## Previously encoded real-frame pairs

A subsequent saved-data analysis reused the earlier host-CUDA
`p1_backtrack_rgb_feedback_cuda_20261030_MAIN01_report.json`; it did not
run a model on CPU or issue new neural calls. It required the same task slot,
exactly equal recorded object position and angle, different pusher positions,
and deduplicated repeated pixel/state records. Five frame pairs met those
conditions, all from the first known root, with four coming from the same
rendering slot. Every pair had a nonzero latent displacement. In three pairs
the pusher moved closer to its goal while its latent distance to the goal
increased; in two pairs both distances decreased. This demonstrates that
the encoder is sensitive to these observed image differences, but its goal
distance does not order all five pusher movements consistently.

The pairs share trajectories and were selected after the current issue was
identified; they are not five independent samples, and same object pose
alone is not a randomized intervention or proof of which pixels caused a
latent change. No actual planner was modified or evaluated from these pairs.
The result is
`le-wm-main/results/codex_p1_pusher_latent_saved_analysis_20261030_MAIN01.json`,
SHA256 `fc2653d871f4f885154127c6ed72b6577a02094dd07a8c7d4b8c5d49f7de0949`.
Two synthetic tests cover pair construction and task-identity rejection.
