# Three-round saved mechanism diagnostics, not causal attribution

New files/output only. Preserve the runtime, outcomes, all old source/doc/seals
and journal. No model/environment instantiation, neural/physics call, action
reselection, refit, new root or outcome-dependent threshold. The experiment-code
skill supplies explicit logging, contract tests and failure receipts, not a new
experiment or figure requirement. This is one already used root, not P1/P4 admission.

## Three separate diagnostic objects

For each arm and scheduled interval 5→10,10→15,15→20:

1. Model direction/progress: original Q/goal cost at chosen start, zero-action
   nominal, and saved chosen candidate endpoint. Report candidate minus zero
   action and start reductions separately. Project the saved predicted latent
   displacement onto the chosen-context goal residual; cosine/projection are
   direction diagnostics, not physical task units or a new objective.
2. Feedback drift: observed versus own predicted boundary last latent, raw/Q
   norms, and exact original-Q cost correction. For projected drift d and
   predicted goal residual r, observed cost minus predicted cost is
   mean(d²) − 2 mean(r*d). Save both terms and their reconstruction error.
   A small drift magnitude alone does not imply goal progress. At the next
   saved boundary, compare observed latent with the prior selected prediction;
   imagined-start errors include accumulated context drift, whereas actual
   starts from the observed context. They are not identical conditional tests.
3. True saved-state progress: every primitive interval plus each round's net
   object-origin/COG and pusher translation, body-axis coordinates, wrapped
   rotation, exact fixed-goal distance/angle reduction and original pose change.
   Reuse sealed motion geometry helpers with the true T polygons/nonzero COG,
   original radius15/slop.1 contract. Static Body/Poly transforms only: no Space
   or stepping. Geometric endpoint near-contact is not force/dynamic contact.

Per-step goal/angle/pose changes must telescope to each net round. The three
complete round pose changes must telescope to original outcomes step5→20.
Official seven-state squared distance is decomposed exactly into agent XY,
object XY, unwrapped angle and pusher velocity squared terms. This explains
metric disagreements without changing either official or original pose labels.

## Missingness and evidence limits

Original runtime encodes boundaries5/10/15 only. There is NO saved step20
latent. The third round's observed-latent progress/prediction discrepancy is
explicitly unavailable; do not infer it from the image/state or encode it now.
Physical step20 progress remains measured from saved states. Partial/blocked
round slots remain present; no shortened endpoint becomes a full planned end.
No observed round is not a measured zero-progress round. All latent-Q and
physical-pose metrics remain in separate units; there is no fitted mapping.
These are trajectory diagnostics and counterexamples to naive proxy inference,
not intervention-based causal proof or independent long-horizon efficacy.

## Binding and execution

Before trajectories, verify the reviewed installed geometry contract and old
geometry source seal. Verify the outcomes audit seal/input hashes; recompute
the original read-only outcomes audit, including runtime source/data/witness,
RGB, history, action, selection and cost integrity. Require identical saved
rows, metrics and comparison. Bind the static Pymunk native binary and source.
Rehash all read inputs at exit. New module/test/doc are sealed before analysis.
Output is exclusive; report.json on success, failure.json if fail-closed.

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_multiround_mechanism_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_multiround_mechanism_20261027_v1 --outcomes-report results/p4_multiround_outcomes_20261027_v1_main_v1/report.json --output results/p4_multiround_mechanism_20261027_v1_main_v1
```

Synthetic tests prove arithmetic/history/missingness contracts only, not a
physical mechanism. Actual saved outputs must be interpreted round by round.
