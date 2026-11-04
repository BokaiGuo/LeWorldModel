# Final-chain abstention RGB development result

The predeclared development protocol is
`PC_FINAL_CHAIN_ABSTENTION_RGB_DEV_PROTOCOL_20261103_CODEX.md`. Four
previously exposed roots were run with uniform, adaptive, greedy and direct
CEM arms. Each arm executed three real-RGB feedback decisions at a fixed
terminal horizon of 15, 10 and 5 actions. This is not independent confirmation.

## Results

The native MuJoCo 2.1.2 replay gives final distances below; rows are paired by
root. Lower is better.

| Root | Uniform | Adaptive | Greedy | CEM |
|---|---:|---:|---:|---:|
| 0 | 0.03998 | 0.17954 | 0.12147 | 0.10122 |
| 1 | 0.03513 | 0.08262 | 0.08501 | 0.09916 |
| 2 | 0.06179 | 0.02123 | 0.06345 | 0.06898 |
| 3 | 0.09828 | 0.10496 | 0.01207 | 0.05452 |
| Mean | 0.05880 | 0.09709 | 0.07050 | 0.08097 |

The mean is descriptive over four exposed roots, not a generalization or
statistical advantage claim. The new fallback was selected in **0/36**
structured-arm decisions; CEM is not assigned an abstention label. Thus this
run checks the wiring in real feedback but does not measure the physical
effect of actually taking the fallback. It is not a PC2 mechanism test.

Each structured trajectory consumed 1,439 prediction tokens across three
decisions; CEM consumed 1,440. One complete-horizon prediction per decision
was reserved for the final comparison or CEM rescoring. Prediction-token
matching does not equalize SVD, encoding, device time or total compute.
`full_cost_certified=false` remains in the records.

## Audit and limits

The new adapter compares the *final* chain after joint refinement and carry
selection to the same chain with a zero first segment. The comparison uses
the current model context and its score is charged. A tie selects the zero
prefix. This is a one-counterfactual fallback, not a comparison with
independently optimized suffixes, and it does not establish reconfiguration.

All 48 decisions were checked for dependency hashes, original selected chain
in the prediction trace, correct counterfactual construction, final choice and
prediction counts. All 16 executed action sequences were replayed on the
native backend; the replay also checks RGB receipts, action timing and carry
binding. The final comparison is recorded separately from the inherited
prediction trace; its charged samples and calls were reconciled explicitly.
The first two attempted launches did not reach model execution because CUDA
was unavailable under that invocation; a direct project-environment launch
completed. No result files were produced by those failed attempts.

The new method does not implement full successor program search, does not
demonstrate same-chain PC2 intervention or PC3 strong-baseline superiority,
and does not resolve checkpoint or normalizer provenance. No PC gate is
promoted. This adapter is a development component and is not presented as the
single certified final method.
