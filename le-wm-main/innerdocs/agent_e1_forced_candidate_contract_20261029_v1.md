# E1 forced-candidate comparison: prospective contract

Actual host receipt `f385e1de76ca391a7ef4d03c198211cd303eb2e4d2bc3bdd69537146e8330eba`
binds report `92a8c6d5c9a6fa60bb1738cfd5ed9abd70d5cc889ada04d62886b14c1ca5f3b7`.
Both banks were byte-identical per root; both arms selected bank index 48 on
root0 and 55 on root1, with identical selected raw10 within root. Original Q
for bank best / zero extra / witness extra is respectively
`0.31894344722083134 / 0.8348093413064602 / 0.5368562796441493` and
`0.2023436602312862 / 0.8144944740309648 / 0.6418139294353745`.
Thus this specific selection rule did not release the extra candidate; this
alone says nothing about physical execution ability or other roots.

Pre-register all two existing roots, exactly three forced raw10 candidates per
root: original bank minimum under original Q and index tie-break; all-zero
extra; compiled-witness raw5 followed by five zero actions. No selection of
roots, candidates or thresholds from physical outcomes. Same root/history/task,
original goal, same shared compiled prefix target, native action domain and
cp0/5/10; preserve every slot and early-stop null. Candidate actions are sealed
before any physical observation; replay all six from the same original context
with separately paid whole H2 and segmented 1+1 (4 packet calls per slot, 24
per each predictor/API/AE/projector axis). Seal all replay outputs before any
replay scoring; no CEM or proposal generation is rerun. Actual tokens and
whole/seg differences must be saved; no clip, fitting, retries or padding.

Native same-time execution compares the forced witness against zero and bank
winner, and compares each with its own paid predicted checkpoint and actual RGB
encoding; task goal/native7, xy, velocity, original and wrapped angle stay
separate. At cp5 the model witness target is a predicted prefix, not an
authenticated native state. A lower physical tracking error under forced
witness while the original selector chose the bank would be consistent with
selection suppressing this usable action; failure of the forced witness to
approach its target would instead limit that explanation. Neither observation
alone proves controller-wide ability, an implementation bug, causation on new
roots, or a scientific mechanism. No arrival threshold is assumed or fitted.

The old six-slot physical recipe implies, only if its exact hooks and no early
stop/extra operations apply: 120 public steps, 1,224 physics steps, 132 PNG,
six constructor/close, 12 reset/goal-render, 24 set-state, six goal-bind. A new
native adapter must prove its own hook census, resource/transport limits and
failure accounting before release; those estimates are not executed costs.
Proposal generation, controller, replay, encoder and physics costs remain
separate; FLOPs unknown, no equal-total-compute claim. A real MAIN-authorized
same-time RGB encoder, strict saved consumer and independent evidence for new
roots remain necessary for any broader claim. P1/scientific gate stays false.

This saved-only contract authenticates pinned report/receipt and old archive,
inherits the full parent receipt instead of invoking upstream fullstrict, and
provides source-bound exact actions; it is not a deployment adapter or physical
authority. MAIN only, after audit, using fresh absolute output and actual SHA:

```
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_contract_20261029_v1 prepare --output /ABS/NEW.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_contract_20261029_v1 verify --manifest /ABS/NEW.json --manifest-sha ACTUAL_SHA
```

Fixture test `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_forced_candidate_contract_20261029_v1.py` reads only sealed saved data and temporary files, never NN or physics.
