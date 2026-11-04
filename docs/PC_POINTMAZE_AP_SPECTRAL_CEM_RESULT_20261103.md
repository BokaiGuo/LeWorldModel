# Same-CEM spectral initialization: PointMaze development result

The [fixed protocol](PC_POINTMAZE_AP_SPECTRAL_CEM_PROTOCOL_20261103.md)
executed both arms on all eight exposed roots, with two real RGB feedback
decisions and ten physical steps per arm. Sixteen saved action sequences
were independently replayed; all terminal RGB frames and independently
evaluated endpoint positions matched. The first two random-arm action
sequences and all eleven RGB frames also exactly reproduced the previous
AP-compatible final-target development run.

| Root | Random CEM terminal distance | Spectral initialization terminal distance |
| --- | ---: | ---: |
| 1888 | 0.130665 | 0.058526 |
| 1889 | 1.208941 | 1.212219 |
| 1890 | 0.534479 | 0.600352 |
| 1891 | 0.115354 | 0.109460 |
| 1892 | 0.042505 | 0.137743 |
| 1893 | 0.769128 | 0.829321 |
| 1894 | 0.417237 | 0.271191 |
| 1895 | 0.217665 | 0.043486 |

On unrounded distances, spectral is lower on four roots and higher on four.
Both satisfy the offline strict distance `<0.5` on 5/8 roots. This is not
the environment's authenticated online success rate. The roots have
previously been used in development and are not independent confirmation.

Each arm scored 50 candidates per decision; spectral additionally required
two JVPs and one SVD per decision, for 32 JVPs and 16 SVDs overall. The
measured structure time was approximately 0.038–0.070 seconds per spectral
decision; complete planning time includes that cost. Equal candidate counts
are not equal total compute, and warmup and execution order affect timing.
No claim of same-budget advantage, reconfiguration mechanism, composition,
P1 passage or complete AP replication follows.

The non-injected optimizer exactly reproduced the pinned-compatible CEM
kernel on an analytic fixture, including every evaluated candidate batch.
Six combined tests passed, including saved action/frame/endpoint bindings,
spectral vector reconstruction and resource counts; these tests do not
rerun the neural or physical computations.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_ap_spectral_cem_20261103_DEV01.json)
SHA-256 `1737a2bf2f4501b537e2fdf74da10d80c5d87bd56b9fdda09d85c1ac73954e87`.
