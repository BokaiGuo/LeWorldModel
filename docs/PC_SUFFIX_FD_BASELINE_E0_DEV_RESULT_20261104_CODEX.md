# Bounded finite-difference suffix baseline: E0 development result

All four fixed tasks had a legal 15-step privileged witness. From three
shared initial tails per root, existing CEM found a successful suffix on
1/4 roots; the bounded finite-difference score-gradient baseline found
one on 0/4. Best final distances per root were:

| Root | CEM | Finite difference |
| --- | ---: | ---: |
| 0 | 0.240411 | 0.551220 |
| 1 | 0.042589 | 0.560299 |
| 2 | 0.500684 | 0.590423 |
| 3 | 0.506096 | 0.549948 |

CEM used 120 full-chain evaluations per start, or 1,080 prediction
tokens per root. Finite difference used 88 per start, or 792 tokens per
root; the remaining allowance was insufficient for another complete
40-probe gradient generation and was not filled with dummy calls. Three
initial chains were evaluated once and cost nine shared tokens per root.
This is a comparison under the same upper limit, not equal spent compute.

The audit replayed actions and scores using alternate same-environment
geometry and checked common starts and costs. A separate trace audit
reconstructed all 24 finite-difference generations, line searches and
selected scored incumbents. Witnesses are privileged diagnostic inputs;
these results neither certify this baseline as fully tuned nor establish
PC2/PC3, and they do not prove that gradient methods cannot work.

The four-root stopping rule was reached. Artifacts:
`le-wm-main/results/pc_suffix_fd_baseline_e0_dev_20261104_codex/`.
