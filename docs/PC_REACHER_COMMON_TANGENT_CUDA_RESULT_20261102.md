# Common-tail tangent: host-CUDA development result

The fixed model-only protocol in `PC_REACHER_COMMON_TANGENT_CUDA_PROTOCOL_20261102.md`
ran on both previously exposed Reacher roots with the authenticated frozen
checkpoint and goal Q. No new physical outcomes were supplied to candidate
generation. Both compiler calls and all seven-candidate response-census
replays completed. The before/after model parameter hashes agree.

| Root | Eligible of six non-neutral candidates | Tangent FD relative error | Candidate response Frobenius ratios |
| --- | ---: | ---: | --- |
| 20263160 | 0 | 0.000583 | 1.0009–1.0278 |
| 20263161 | 0 | 0.000482 | 0.9970–1.0117 |

The selector returned `null` for both roots; no fallback was substituted.
Every candidate had a higher immediate Q cost than its root's neutral prefix.
On root 20263160 every candidate worsened at least four of the five common-tail
costs. On root 20263161 every candidate worsened at least two. The positive
tangent candidates had all four neutral-tail-adjusted differences negative,
but also worsened four absolute common-tail costs, so they did not satisfy
the frozen conjunction. These are model-score observations, not evidence
that the physical P1 mechanism is impossible.

The response ratios are descriptive finite model differences under the
specified five parameter-space tails, not direct task control or success.
The compiler took approximately 0.279 s and 0.068 s per root; separate
endpoint/census replays took 0.061 s and 0.053 s. These timings omit model
loading, root reconstruction and inherited goal-Q construction. Nominal per
root counts are 9 prefix and 90 suffix-token samples for compilation, plus
7 prefix and 70 suffix-token samples for the census; encoder calls,
backward costs and prior Q VJPs remain outside those counts. They are not a
total-budget comparison.

All seven prefix parameters, bounded normalized and raw actions, five tail
actions, five predicted endpoints per candidate, scalar scores and response
spectra are saved in
`le-wm-main/results/reacher_pc_common_tangent_cuda_20261102_v1/report.json`;
its SHA-256 is
`d85a12443483e2268a0849ad72b6afccf3b68804a536df06bb0f780620f41589`.
The float32 raw-prefix array SHA-256 values are
`6bd8672df7c9859150904d3a683d897b9c857feb779c506e59e15939242e1c08`
and `adc4f834fd62df5d347fe38c37f64a9de55bb680a936e9207ac2b0a2031891d8`
in root order. The common raw-tail array SHA-256 on each root is
`c9a7c2c0de90a23baf5a425d631018759a1694225fe93080d1f0b86d5d9dee04`.

No physical success, non-immediate physical progress, independent
generalization, or P1 admission was established. A different candidate
generation/selection hypothesis requires a new fixed development protocol;
the existing rejection results remain unchanged.
