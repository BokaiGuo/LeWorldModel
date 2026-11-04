# PointMaze AP-compatible RGB feedback: development result

The [fixed protocol](PC_AP_POINTMAZE_RGB_DEV_PROTOCOL_20261103.md) completed
four ten-step physical trajectories on exposed roots 1888 and 1889. Each
arm executed a five-action packet, received its own real RGB, recomputed
causal velocity from frames 4 and 5, and replanned the remaining packet.
All four full action sequences were independently replayed: terminal RGB
matched the live render exactly, and endpoint positions matched a separate
physical evaluator.

| Root | Observation-memory target: terminal distance | Final-goal target: terminal distance |
| --- | ---: | ---: |
| 1888 | 0.158442 | 0.130665 |
| 1889 | 1.208568 | 1.208941 |

The offline strict terminal-distance `<0.5` check is 1/2 for each arm; it is
not certified native online success. These two roots were already exposed,
and differences are descriptive, not independent evidence that either arm
is superior. The run neither tests D2P against AP nor establishes P1.

Every decision scored 50 candidates in four callback batches; each arm
used two decisions and ten physical steps. Memory construction encoded 63
RGB frames and took about 0.389 seconds in this execution. The memory arm
also performs retrieval and additional target encoding. Planning times
include the decision's encoding and (for memory) retrieval but do not
provide complete shared preparation and resource attribution. Matching
prediction counts therefore does **not** establish matched total budget.
The three memory episode IDs do not overlap the two query IDs; their
training-set relationship and full historical independence remain unknown.

This is an AP-compatible local adaptation with a small observation memory,
not the authors' official memory, full controller or published benchmark.
No simulator state or memory action entered the selector; reset and endpoint
evaluation used simulator state. Model parameter hashes remained unchanged.

Result [receipt](../le-wm-main/results/pc_ap_pointmaze_rgb_dev_20261103_DEV01.json)
SHA-256 `48b42f358cde910d4a66758f0c6b57a098e17ad95c87aac982083fafa7cede91`.
The saved-data verification checks action and feedback-frame bindings,
retrieval target bytes, cost counts, endpoint arithmetic and source hashes;
with the callback/kernel tests, **5 tests passed**. These tests do not
independently rerun inference or physics.
