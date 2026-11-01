# P2 known-root host CUDA planning exploration

The fixed four-root planning run completed on host CUDA:
`le-wm-main/results/codex_p2_knownroot_host_cuda_20261030_v1_run01/run_report.json`.
It reports `exploratory_complete`, four roots, two planning arms per root,
and frozen model state hashes identical before and after execution. The run
did not execute physical control and is not independent confirmation, equal
total compute, or P2 admission.

Each arm completed its fixed candidate census. The model-side winner costs
(lower is better) were:

| root | no-composition | conditional composition |
|---:|---:|---:|
| 0 | 1.39115 | 1.32250 |
| 1 | 0.77598 | 0.89478 |
| 2 | 1.20925 | 1.07658 |
| 3 | 0.89583 | 1.02918 |

Composition is better on roots 0 and 2 and worse on roots 1 and 3. These
model-only costs cannot establish physical composition benefit; the two arms
also have different candidate counts and actual work. The output is a
known-root engineering/exploratory result with cooperative duplicate control.

The four per-root report hashes are:
`aa2c6b35fb81d833572dd3180e001ef63784eff9c3c39e1b89ba49828726c53a`,
`9689f9526c28140385c70bd2299df86b0d01d3b4378f2dbff0e6f490a2e75c0a`,
`69b2e70f2bf118ff6566a351088099fa88b36afd13c9da60da1c7b766ef310b2`,
`01c9a568cdebd99728ab63a9ae6863fb82aa1fbed57cc01288aa82193233cb86`.
