# CEM32-selected versus neutral and same-bank greedy: development result

The fixed 13-prefix response-SVD bank was evaluated by the frozen model with
32 samples, three CEM iterations and four elites per predicted successor.
The model-only rule selected prefix 2 on root 20263160 and prefix 3 on root
20263161; same-bank immediate greedy selected prefix 6 on both. A saved-only
consumer recomputed eligibility, selection, tie ordering and cost accounting
from the saved candidate scores; it did not independently rerun the GPU
scores. The experiment was on previously exposed roots and outcomes.

All six declared physical arms completed their selected prefix and an online
real-RGB-replanned tail. The executor also ran a neutral-prefix diagnostic
within each arm. Results use wrapped joint distance, lower is better:

| Root | Role | Prefix | Initial | After prefix | Final |
| --- | --- | ---: | ---: | ---: | ---: |
| 20263160 | selected | 2 | 1.003500 | 1.046128 | 0.491628 |
| 20263160 | neutral | 0 | 1.003500 | 1.010031 | 0.511719 |
| 20263160 | same-bank greedy | 6 | 1.003500 | 0.961775 | 0.478213 |
| 20263161 | selected | 3 | 1.182600 | 1.324410 | 0.677681 |
| 20263161 | neutral | 0 | 1.182600 | 1.245444 | 0.679627 |
| 20263161 | same-bank greedy | 6 | 1.182600 | 1.171132 | 0.660184 |

Selected is better than neutral in both rows by 0.020091 and 0.001947,
but worse than same-bank greedy by 0.013415 and 0.017497. This does not
establish a general effect, a strong-baseline victory or P1 admission.
Both selected prefixes increase the joint distance from their own initial
state; this alone does not establish a real change in useful subsequent
control capability.

Earlier, before this selector was designed, the fixed five-tail physical
panel had already been opened for every prefix. When retrospectively linked
to these new selections, both selected prefixes had worse *absolute* final
joint distance than neutral for all five shared tails at their respective
roots. After subtracting each prefix's zero-tail outcome, the four active
tail differences have mixed signs on the first root and are negative on
the second. The two descriptions answer different questions; neither is
independent evidence or a general capability result. Physical intervened
tails and per-prefix replanned tails must not be conflated.

The selection's predicted future costs favor selected over greedy on both
roots. Because those costs evaluate predicted successors and model-planned
tails while the physical comparison executes tails replanned from real RGB,
this is **not** a same-action prediction-versus-physics ranking reversal.
Diagnosing that would require evaluating the same sealed action chains.

Each full-bank selector used 1,248 CEM rollout evaluations and reported
2,522 model-predict samples per root; each live tail reports 96 candidate
rollouts. Candidate generation, all encodes and total wall cost are not
certified, so shared suffix budgets do not imply equal total cost.

The saved live verifier checks source hashes and selections, root/arm
ordering, RGB/action histories, all 20 step intent/return pairs per arm,
and stop conditions, but cannot independently attest simulator execution.
Nine targeted tests passed, including altered action, history, stop and
missing-arm cases.

- Model-selection plan SHA-256:
  `9494f3eb1411cdfadc0509a696365db1390af78e738364879180539a2ad6a730`.
- Model-selection report SHA-256:
  `158dcdf2e80b6845a7c430c2befdd51f8979702d5f12d2274c5d972cdfabc1eb`.
- Live plan SHA-256:
  `9b10d01fea866e764978ceac0632e3eda8989a0ed7ba1b81234e5195f46ed088`.
- Live journal SHA-256:
  `38a832235e1f7bfd2163cefe69e5279a13b03f769ac78d1a8ebc2c27d0e70e36`.
- Live saved receipt SHA-256:
  `9a0d935a8daab2399eb3ae9900698db9d111be7b3030d979b7dd8a3313222c7a`.
