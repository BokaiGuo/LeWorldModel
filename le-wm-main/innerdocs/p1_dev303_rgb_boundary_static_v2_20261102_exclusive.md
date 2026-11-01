# dev303 RGB boundary witness v2: static development diagnostic

The v2 generator uses the existing RGB chunk interface's four-connected
pusher limit of at most two components and a maximum 20-by-20 pixel bounding
box. Current and goal object masks must each be one component and must not
touch the image edge. This is a new development protocol; the old contracts and
guards remain unchanged.

The pre-existing dev303 blind NPZ was verified against SHA-256
`092af72253c9b0463e5c7a5814052599843dc4b6a979bd040d7f318c93dcddd9`;
the corresponding manifest was verified against
`b7b5a1b2394acdad13981c1fb4258d97941e0f1863e816e9dc86d85b91b460dc`.
The full source chain was not independently reverified. The configuration SHA
is `eea335f173298be862c3d7d597c2025f002cb75a1cba10f97717a07a27d14e03`.
The 10 recorded actions are zero; this is not proof of zero physical velocity.

Actual v2 static result: `abstain`, reason `pusher_components_exceeded`,
candidate count `0`. The current exact-blue pusher mask has 38 pixels in six
four-connected components, above the unchanged limit of two. Current and goal
exact-gray object masks contain 358 and 362 pixels, respectively. No physical,
GPU, neural, or outcome operation was performed. This is not P1 admission,
an independent confirmation, a contact result, or a physical safety claim.

Tests: `10 passed` in the exclusive v2 test file, including nearby two-component
acceptance, remote-fragment, clipped-mask, oversized-bbox, excessive-component,
and object-fragmentation cases.

Reproduce the read-only static run:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m experiments.spectral_reachability.p1_dev303_rgb_boundary_static_v2_20261102_exclusive
```
