# Live RGB cp5 CEM replanning: known-root development result

Four predeclared arms completed on host CUDA and host EGL: execute a fixed
five-step prefix, encode the actual successor RGB and action history, run the
`32 × 3` CEM tail planner, then execute its ten-step tail. The saved journal
contains the additional five-step zero-prefix diagnostic for each arm as
well, for 20 recorded physical step intent/return pairs per arm.

| Root | Prefix | Initial joint distance | After prefix | Final joint distance |
| --- | --- | ---: | ---: | ---: |
| 20263160 | neutral 0 | 1.003500 | 1.010031 | 0.511719 |
| 20263160 | selected 12 | 1.003500 | 1.025701 | 0.559084 |
| 20263161 | neutral 0 | 1.182600 | 1.245444 | 0.679627 |
| 20263161 | selected 12 | 1.182600 | 1.256223 | 0.668234 |

For this metric selected is worse than neutral at root 20263160 by about
0.04737 and better at root 20263161 by about 0.01139. These are two
development roots, not independent confirmation or a statistical
comparison. The experiment does not itself establish the same-prefix
capability mechanism or superiority to strong baselines.

The live tail actions match the previously computed archived-real-RGB tail
within `5.97e-8` maximum raw-action error, and the corresponding earlier
physical replay final distances match within `1.62e-8`. This establishes a
development replay consistency check, not an independent physical
attestation or a causal feedback benefit. Each CEM receipt records 96
candidate rollouts; full encoding, predictor, candidate construction,
control overhead and wall-clock comparability are not certified.

The saved-only verifier checks journal hashes and chain, arm and context
order, actual RGB/history bindings, executed prefix and zero control,
20 ordered step intent/return pairs per arm, stop flags, model hash, CEM
candidate counts and comparison to the earlier full-panel receipt. It does
not independently rerun the simulator. The root identity and action records
are self-reported by the executor.

- Plan SHA-256: `c9794157724c772b836daf68450efe03d6d4dec5f99638f9f26ba4058ca33764`.
- Journal SHA-256: `3230ae412f44cbbf273e03da8f84b9bf2e18c93463a7636cd3e9daf2d241ba38`.
- Saved receipt SHA-256: `d7246f653155e91a961dab0049e6ac0c16fa430416a1b8f7e03cd761a9a00e51`.
- Eleven targeted tests passed, including altered stop, action and history
  events and the earlier physical replay consumer tests.
