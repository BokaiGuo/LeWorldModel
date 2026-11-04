# P2 conditional versus joint: host CUDA development result

The sealed 16-candidate known-B0 comparison completed on host CUDA and the
independent saved-data consumer accepted the report and journal. The first
sandboxed attempt failed before model loading because UUID-filtered CUDA could
not initialize NVML; its partial report is preserved. The successful run
used the unchanged plan and a new output directory.

Both methods selected candidate 12, optimizer iteration 8. The conditional
initializer's selected model objective was 0.649799; the joint 20-dimensional
Gauss–Newton initializer's was 0.593769 (lower is better). For the 16 paired
candidate outputs, the joint control had a lower model score in all 16; the
mean conditional-minus-joint score was 0.022090. They share one known root
and candidate bank, so these are not 16 independent task outcomes. This
observation is not a physical comparison and does not establish that
composition cannot help.

The saved consumer recounts 640 total packet transitions and 256 explicit
common-optimizer gradient calls across both arms. Conditional and joint
Jacobians are 192×10 and 192×20; equal entrypoint counts do not establish
equal FLOPs or total costs. This joint control is not the historical SRA
baseline, despite a legacy ledger label. Existing upstream SRA source
authentication remains unresolved. No physical execution, independent
confirmation or P2 admission is claimed.

## Fixed-winner physical development follow-up

The two model-selected action sequences were subsequently frozen into a
separate physical contract and executed on the same known B0 root. The saved
consumer verified both branches, their 25 planned steps, and 222 PNG
artifacts; neither branch terminated early or reached native success.

| Method | Native distance | Object position error | Wrapped object angle error |
| --- | ---: | ---: | ---: |
| Conditional initialization | 133.8481 | 83.7078 | 1.1573 |
| Joint Gauss–Newton control | 103.8155 | 53.8158 | 0.1749 |

The joint control was better on these reported terminal errors at this
one exposed root, but neither succeeded. This is a same-root development
comparison, not a matched-total-compute study, independent confirmation, or
proof that program composition is ineffective in general. The saved
consumer is not an independent simulator replay.

- Physical plan SHA-256:
  `bae884415fb125d0ec22dcde7651089c0dfa0e0939c18d69b60537c34d0047f4`.
- Physical report SHA-256:
  `e391ce4743dd5d742d116f46403cf476c867c81812e011d2e2f6809222065d52`.
- Physical saved-consumer SHA-256:
  `edcaa752c3664a945f217615608c051ebb63f7be6c088fb3ab8f49233846ba84`.

- Plan SHA-256:
  `55cf41dbd6f0608a1f7ea229cc63ce040bbf5400773f1b386d8b0fca9775b22d`.
- Completed report SHA-256:
  `d991f0e57e2802ba09885eb29c18d2c95c836933db6ab5325c54cfeba11ea7c1`.
- Saved-consumer SHA-256:
  `8d26a76390b00741ce341f8f7a9d802c2bba45bee99ed305624d2dde128713b5`.
- Failed sandbox report SHA-256:
  `166a6f24e76baf8ee21c8944b11d85c5e2f6e25a1997fe2be57e7a69e69169c7`.
