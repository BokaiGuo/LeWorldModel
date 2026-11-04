# Goal-Q versus L2 tail planning: real-RGB development comparison

Both previously exposed roots completed all 13 fixed prefixes with real cp5
RGB feedback and their own 32-candidate Q-weighted CEM suffix. The saved
consumer checked all 26 arm action histories, RGB timelines, actual step
sequences and declared Q objective. All 26 paired cp5 joint distances exactly
match the earlier L2-tail run. Model parameters and the sealed plan passed
postchecks; no early stop was recorded. The consumer does not independently
replay the simulator or verify every underlying neural operation.

| Root | Q-tail final better than L2 for same prefix | Mean Q minus L2 final distance | Zero-prefix Q final | Zero-prefix L2 final | Q prefixes no better at cp5 yet better than Q zero at final |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20263160 | 7/13 | +0.016199 | 0.496129 | 0.546408 | 3/12 |
| 20263161 | 0/13 | +0.384012 | 0.906710 | 0.598310 | 0/12 |

Distances are wrapped joint distances; lower is better. Every Q-tail action
sequence differed from its matched L2-tail sequence. On these exposed roots,
aligning the suffix scoring objective with the Q metric did not consistently
improve physical control, and was worse on all 13 prefixes of the second
root. This comparison was motivated after the old outcomes were opened; it is
an exploratory development result, not an independent confirmation or a
causal decomposition of model error versus metric mismatch. Neither frozen
strict selector chose a prefix. The same candidate count does not certify
equal total compute, and no P1 or strong-baseline success is established.

The Q optimizer used the authenticated goal-only Gram matrix bound in the
preflight, the same bounded action chart and CEM sampling/update schedule as
the L2 run. The two actual objectives are different; the old L2 result is
preserved as such.

- Plan SHA-256:
  `093874de230fb7e95c745817a94dbbadc168fc6af4d27edc5adafe300fa64f66`.
- Journal SHA-256:
  `8c4733bceeca7b7891a713b276f557cad50575d160f7721c3611805eb4535afc`.
- Saved consumer SHA-256:
  `a66ff56d542a1728fd17d28e94ec3338bf9b6c1312695f24d0c798a567ddeedd`.

## Saved successor-context pairing

An additional consumer checked all 26 ordered seed/prefix pairs against both
source journal and receipt hashes. The three saved RGB frames, goal RGB and
float32 action histories match for each pair, as do the reported initial and
cp5 distances. The consumer separately records each final distance and suffix
equality. Missing arms, altered RGB, goal, history or cp5 distance are rejected
by fixture tests. This is a comparison of saved data, not an independent
simulator replay, a complete underlying-state identity proof, or an additional
independent sample.

- Pairing receipt: `le-wm-main/results/reacher_response_svd_q_l2_cp5_PAIRED_20261102_v1.json`.
- Receipt SHA-256:
  `3521efd0857540686f31513b4fb90b49ba60b04fe3250ae48be61b19df0ce63a`.
- Pairing and Q optimizer fixture tests: 7 passed.
