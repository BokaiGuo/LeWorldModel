# FD suffix search: frozen-model development result

The predeclared model-only search ran once on each of the two exposed
Reacher roots. All probes and nine candidate rows were saved. The frozen
parameter hash and sealed plan postcheck passed.

| Root | Development selected prefix | Development selected tail | Strict selected prefix |
| --- | ---: | ---: | ---: |
| 20263160 | 4 | 1 | none |
| 20263161 | 4 | 4 | none |

Both development selections are `objective1_radius0.1`. The selected tail
has predicted Q improvement over the neutral prefix's best fixed tail of
0.0178927 and 0.0116343 respectively. Other common tails worsen; the
unchanged all-tail strict selector does not select either candidate.
Development selection is best-of-five model scoring, not the original
mechanism condition, a physical intervention, or P1 admission.

The independent saved-data consumer reconstructed the FD candidate bank
and both selectors, checked action normalization, source/plan bindings and
the saved score replay, and returned `complete_saved_consistency`. It could
not independently recompute goal-Q from endpoints because the goal latent was
not saved; GPU UUID and absence of privileged data were not independently
certified by the consumer. The model input path remains subject to the
separate source audit. The 22 CPU tests cover search, launcher and consumer.

Subsequently, a separate CPU numerical audit reused goal embeddings from the
earlier constrained-goal replay and the same goal-Q Gram matrix. It
independently recomputed all 45 saved endpoint costs per root via a NumPy
eigendecomposition; maximum absolute differences were `8.39e-9` and `2.07e-8`.
Goal RGB, frozen parameter, preflight and root identities matched. Two tests
passed, including modified score and goal identity rejection. This is
independent arithmetic, not a new encoding or physical observation. Audit
receipt SHA-256:
`7d0ecded25cc59985eafc7e90f932a53e0dc40be8a1e5dac7993e2e44faa4ca0`.

Per root the search used 29 prefix and 290 suffix-token logical samples and
the census another nine and 90; encoder, goal-Q construction, loading and
complete internal costs are not included. The measured search/census seconds
were 0.384/0.074 and 0.188/0.068; they are not total-budget measurements.

Plan SHA-256:
`12ff26c6c052e02fe487bde8affd21104b09b8950f6a5340f5538fea766a2307`.
Report SHA-256:
`b76a55c6b6f8e13e7c4cbba0887e6ec6babc062c1a3245ddabf3991d39b00401`.
Saved audit SHA-256:
`127300623499118458970a433521e5c8a3f2c1e00c55c9d2ba69986c11f7ce9c`.
No physical outcomes or independent roots were used or produced.
