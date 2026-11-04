# Spectral predicted-tail carryover: paired development results

The fixed development protocol is
`PC_REACHER_SPECTRAL_PREDICTED_TAIL_PROTOCOL_20261102.md`. The selector's
previously discarded suffix was reconstructed per arm with the original
model, prefix, seed and successor CEM budget; all six recovered model scores
matched their archived receipts. Every physical branch reproduced its old
first raw action and observed successor RGB/action-history identity. Frozen
parameter hashes matched before and after.

| Root | Arm | Original final qpos | Carryover final qpos | Carryover tail selected |
| --- | --- | ---: | ---: | --- |
| 20263160 | selected | 0.558267 | 0.558267 | yes |
| 20263160 | zero | 0.546408 | 0.546408 | yes |
| 20263160 | same-bank greedy | 0.497367 | 0.531732 | no |
| 20263161 | selected | 0.619442 | 0.619442 | yes |
| 20263161 | zero | 0.598310 | 0.730452 | no |
| 20263161 | same-bank greedy | 0.760234 | 0.760234 | yes |

Lower qpos distance is better and is evaluator-only. On root 20263161 the
new selected-minus-zero contrast becomes negative, but selected's physical
outcome is essentially unchanged while zero's changed because inserting one
candidate also changes CEM's elite distribution and later random samples'
transformation. The selected tail was not chosen for the changed zero arm.
This cannot establish that selected's successor is more controllable, nor
that inserting a tail generally worsens zero. The already-run common-suffix
interventions remain unchanged. The roots are shared, known development roots
and outcomes were visible before this protocol.

Each arm used two CEM batches of 16 with the same actual-RGB feedback as
before. The inserted candidate replaced one random candidate per batch; thus
equal sample counts do not mean the populations are identical. The saved
source selector's per-prefix compilation and search costs must be charged;
reconstruction is extra experimental overhead, and full resource equality is
not certified. Fifteen tests covering predicted-successor CEM, spectral
guards and candidate insertion passed; these are not a P1 or PC gate.

Artifacts:

- `le-wm-main/results/reacher_pc_spectral_tail_carryover_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/run_reacher_pc_spectral_tail_carryover.py`
- `le-wm-main/experiments/spectral_reachability/reacher_pc_tail_carryover.py`
