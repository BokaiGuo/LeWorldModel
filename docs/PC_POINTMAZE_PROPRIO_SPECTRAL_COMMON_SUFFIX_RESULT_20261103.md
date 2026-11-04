# Common-suffix intervention on spectral development prefixes

The [fixed intervention protocol](PC_POINTMAZE_PROPRIO_SPECTRAL_COMMON_SUFFIX_PROTOCOL_20261103.md)
executed all 72 combinations: eight exposed roots, three prefixes
(random, spectral, zero) and three identical suffix choices (zero, random
arm's saved tail, spectral arm's saved tail). No world-model inference or
outcome-based selection was used. All 72 terminal positions matched the
independent position backend; all 16 original same-arm chains reproduced
their previously saved terminal RGB and full endpoint.

For the common *spectral-arm suffix*, distances are:

| Root | Initial | Spectral prefix at cp5 | Spectral prefix at cp10 | Zero prefix at cp10 |
| ---: | ---: | ---: | ---: | ---: |
| 1888 | 0.4360 | 0.0650 | 0.0646 | 0.0234 |
| 1889 | 1.1842 | 0.3194 | 0.5039 | 0.6705 |
| 1890 | 1.1766 | 0.6538 | 0.5426 | 0.3093 |
| 1891 | 0.7977 | 0.2317 | 0.2381 | 0.2966 |
| 1892 | 0.3298 | 0.1394 | 0.1257 | 0.3361 |
| 1893 | 0.6426 | 0.1433 | 0.3196 | 0.4996 |
| 1894 | 0.2529 | 0.5819 | 0.2447 | 0.4358 |
| 1895 | 2.2106 | 0.9805 | 0.3705 | 1.0726 |

On seven roots the spectral prefix already reduced position distance by
cp5; those examples do not establish a non-immediate-progress mechanism.
On root 1894 it increased distance from 0.2529 to 0.5819, and the same
spectral suffix gave lower terminal distance after spectral than zero
prefix. However spectral cp5 was already closer than zero-prefix cp5
(0.5819 versus 0.6316), so this observation alone cannot isolate a change
in control capability or establish a non-greedy advantage. Against each
prefix's own zero suffix, the spectral suffix changed terminal distance by
-0.0658 after spectral prefix, -0.0954 after random prefix, and +0.0487
after zero prefix; this is an outcome contrast, not separately optimized
planning. All nine outcomes per root and displacement contrasts are in
the JSON, not only this suffix.

Zero action does not necessarily mean no movement: it retains dynamics and
initial velocity. The panel is adaptive and previously exposed; the model
did not select prefixes under a preregistered non-progress criterion. It
does not establish independent generalization, equal-budget superiority or
P1 admission. Existing independent-tail-planning comparisons and this
common-suffix intervention answer different questions.

- Report:
  `le-wm-main/results/a1_pointmaze_proprio_spectral_common_suffix_20261103_DEV01.json`,
  SHA-256 `0f6ea3d1248eff0f79a433fff30002c5ec1df820e282cab5e8225e280a7515f5`.
- Arrays: same basename `.npz`, SHA-256
  `0c7c156c209560627ec6db97972c251cb857fe1d4639b1b3d8ce6717856c8e7c`.
- Saved matrix, action binding, source RGB and metric arithmetic: **1
  passed**. The test does not rerun physics.
