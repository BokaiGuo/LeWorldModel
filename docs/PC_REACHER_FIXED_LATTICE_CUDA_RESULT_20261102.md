# Fixed lattice CUDA development result

The sealed fixed-lattice plan ran on the two previously exposed Reacher roots.
Each evaluated neutral and all 40 non-neutral directions and re-evaluated the
common-tail endpoints. Both roots completed, the frozen parameter hash was
unchanged and the plan postcheck passed.

| Root | Eligible non-neutral candidates | Selected |
| --- | ---: | --- |
| 20263160 | 0/40 | none |
| 20263161 | 0/40 | none |

Search times were 0.451 and 0.266 seconds; separate endpoint-census times were
0.305 and 0.298 seconds. These exclude model loading, inherited Q construction,
and other unmetered costs and must not be used as total-budget comparisons.
Each stage used 41 prefix and 410 suffix-token logical prediction samples per
root. No new physical outcome was used for selection or produced.

Plan SHA-256:
`c08bda9ecb65c0d8a8a2c9e23086ceb9c0cd5e54ec406d50b584c9426972cef7`.
Report SHA-256:
`2cd7dec4965699d5cdc0c6852eb2f02f57f77bb12d590e99247c44acf105920d`.
Results remain development-only, not evidence that real reconfiguration is
impossible or that P1 passed. The unchanged strict selector found no witness;
no physical follow-up can legitimately treat a non-selected candidate as its
selected output.

## Saved-score decomposition

An arithmetic recount of the 40 non-neutral rows in the saved report gives
the following numbers passing each selector condition separately. These are
overlapping subsets, not a sequential filter.

| Root | Immediate Q increase | Mean final Q improves | Every final Q non-worse | Mean adjusted Q improves | Every adjusted Q non-worse | All conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20263160 | 20 | 20 | 0 | 20 | 2 | 0 |
| 20263161 | 20 | 20 | 0 | 20 | 5 | 0 |

Thus the immediate Q and mean final-Q checks are individually satisfiable,
but the all-common-tails absolute condition rejects every tested direction.
This identifies the operational bottleneck in this fixed selector and bank;
it does not establish that the selector is erroneous, that any alternative
passes the predeclared scientific gate, or that the model correctly predicts
physical response. No outcome-conditioned action has been selected.
