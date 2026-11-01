# dev303 RGB boundary witness v3: corrected connectivity

The existing `codex_p1_rgb_task_score_20261030_v2._mask` uses an all-ones
3×3 structure, i.e. eight-connected components. The earlier v2 boundary
diagnostic incorrectly used four-connectivity and its `pusher_components_exceeded`
result does not match the existing RGB chunk interface. The earlier file remains
unchanged. The independent v3 implementation uses equivalent eight-connectivity,
the same maximum of two pusher components and maximum 20-pixel bbox dimension,
and rejects clipped or disconnected object masks.

Using the same pre-existing dev303 blind NPZ and configuration as v2:

- Blind NPZ SHA-256: `092af72253c9b0463e5c7a5814052599843dc4b6a979bd040d7f318c93dcddd9`.
- Config SHA-256: `eea335f173298be862c3d7d597c2025f002cb75a1cba10f97717a07a27d14e03`.
- Exact-blue current pusher: 38 pixels, one eight-connected component, bbox
  `[65, 56, 76, 67]` (12×12 pixels).
- Static outcome: `bounded_approach_witnesses`, 4 candidates; action SHA-256:
  `a04deeb0ed64867adcb530a09ab607544dcb68d99af7da4640b90e7d375bad0c`,
  `6b43f1d7a03e3b4db1d0e45e1f8b1675c85eeb1ebd367924d1539a04a08c7526`,
  `cfeb89020d0111bf117bd5d4335989d71095a0c6867f566dbe69888f397c6131`,
  `23365c584ea11a996aa86f42d2795c07fb8454607ad6ecd5bb13144f728ff962`.
- Minimum modeled path-to-object-pixel gaps, in pixels:
  8.962221896231075, 10.36468322644298, 9.410862680103524,
  7.523894216942077.

The ten saved actions are zero; this does not establish zero physical velocity.
The source chain and motion authority were not independently reverified. These
are predicted approach witnesses only: no real contact, safety, task success,
P1 admission, or independent confirmation is demonstrated. No physical/GPU
execution or new outcome was read.

CPU tests: 12 passed, including diagonal connectivity compared with the existing
`_mask`, two-component limits, remote fragments, bbox, clipped masks, and object
fragmentation. Reproduce the static run:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m experiments.spectral_reachability.p1_dev303_rgb_boundary_static_v3_20261102_exclusive
```
