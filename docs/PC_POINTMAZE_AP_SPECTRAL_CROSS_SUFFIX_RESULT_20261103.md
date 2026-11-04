# PointMaze AP-compatible CEM: cross-suffix development intervention

The [fixed intervention protocol](PC_POINTMAZE_AP_SPECTRAL_CROSS_SUFFIX_PROTOCOL_20261103.md)
completed 72 ten-step physical sequences: eight exposed roots, three
prefixes and three common suffixes. All 72 terminal positions agreed with
the independent position backend; all 16 previously executed random/random
and spectral/spectral action sequences reproduced their terminal RGB and
full legacy endpoint. This was a physical intervention only, with no new
model inference.

The table reports the spectral tail under the two nonzero prefixes. The
last two columns are the change in terminal goal distance caused by this
tail relative to the **same prefix followed by zero actions**; negative is
an improvement over that prefix's passive continuation.

| Root | Initial distance | Random cp5 | Spectral cp5 | Random-prefix tail effect | Spectral-prefix tail effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1888 | 0.43596 | 0.06769 | 0.07390 | -0.16252 | -0.05483 |
| 1889 | 1.18419 | 0.28549 | 0.38546 | +0.00945 | -0.02034 |
| 1890 | 1.17662 | 0.65082 | 0.67727 | -0.27359 | -0.17517 |
| 1891 | 0.79771 | 0.21329 | 0.21329 | -0.01898 | -0.01898 |
| 1892 | 0.32978 | 0.07439 | 0.13465 | 0.00000 | 0.00000 |
| 1893 | 0.64261 | 0.49990 | 0.51892 | -0.33092 | -0.33157 |
| 1894 | 0.25290 | 0.66722 | 0.63756 | -0.10913 | -0.11113 |
| 1895 | 2.21063 | 1.11178 | 1.01731 | +0.11761 | -0.19336 |

Root 1895 shows a notably different effect of the same tail, but its
spectral prefix already reduced initial goal distance substantially, so it
does not establish a non-immediate-progress mechanism. Root 1894's spectral
prefix was worse than the initial position but was already closer to goal
than the random prefix at cp5; the tail effects differ by roughly 0.002.
This does not establish a useful reconfiguration advantage. All roots,
other common suffixes, zero controls, absolute endpoints and vector
responses remain in the receipt; none were selected for analysis.

The first execution failed the source-chain exact-endpoint assertion:
the new runner retained the source state tensor's higher precision, while
the original dataset loader converts it to float32 before reset. The
largest shown discrepancy was about `8.4e-7`. After matching that explicit
source conversion, the entire fixed matrix was rerun and all source-chain
checks passed. No partial result was saved from the failed run.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_ap_spectral_cross_suffix_20261103_DEV01.json)
SHA-256 `c626fed268254ab355e8b178710684a604b7878e697f659b6f85c8e13782d618`;
three saved-data and kernel tests passed. These are exposed development
roots with outcome-exposed actions; common suffix response is not a
comparison of independent equal-budget replanning, and neither P1 nor
independent generalization is established.
