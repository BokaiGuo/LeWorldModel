# Common cp5 paired terminal RGB-Q diagnostic

The fixed candidate roster is CEM then structured-own, the two distinct
previously sealed trajectories in the same known-root physical replay. Their
actual prefix actions and cp5 state/RGB hashes agree. Selection is not based
on new results; both outcomes were already observed, so this is retrospective
development evidence, not a blind or confirmatory test.

Preflight rebuilds the existing source-verified physical schedule and checks
the report/protocol hashes, executed 20-action identity, cp5 and terminal
PNG bytes/pixels and physical state hashes, common cp5, distinct 15-action
suffixes, task context/goal/normalizer/projection and model identity. It
verifies the inherited source closure, including both planning reports,
ledger and trace through the physical builder. Existing CEM and structured
terminal predicted Q values are frozen for same-action replay checks.

For each candidate, use its own already executed 15 actions and the common
cp5 model context to predict the terminal latent open-loop, and encode its
own actual cp20 RGB. The same frozen model and goal Q are used. Report
per-candidate predicted Q, actual RGB-Q, physical joint-XY, full physical
metrics and raw latents; report structured-minus-CEM differences separately.
Exact zero is a tie. Q, RGB-Q and physical distance are distinct axes; no
causal, generalization, full-controller fairness, official reproduction or
P1 admission claim follows.

Budget is two terminal RGB encodings and six action-encoder/predict
calls/samples (three per candidate), zero physical and AD. Source verification
uses existing cp5 encoded latent; its original encoding, planning and physical
acquisition costs are inherited and not free. No cp5 re-encoding or
teacher-forcing is included. Durable dispatch records attempts and returns
(`closed_unverified` does not establish model-internal coverage); failed
execution keeps a partial report. GPU peak allocation and elapsed time are
recorded; no FLOPs or energy. Replay must match each sealed terminal predicted
Q to rtol 1e-6, atol 1e-8 or fail; it may not tune tolerances after seeing
outputs.

CPU contract tests use a non-neural analytic fixture. No GPU work is started
by the following:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_p1_cp5_two_candidate_rgbq_20261101_leibniz.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cp5_two_candidate_rgbq_20261101_leibniz dry-prepare
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cp5_two_candidate_rgbq_20261101_leibniz prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_two_candidate_rgbq_20261101_leibniz_PREP_NEW.json
```

After review, use the emitted prepare SHA and a fresh output directory:

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_cp5_two_candidate_rgbq_20261101_leibniz run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_two_candidate_rgbq_20261101_leibniz_PREP_NEW.json --sha256 PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_two_candidate_rgbq_20261101_leibniz_RUN_NEW --authorization MAIN_AUTHORIZES_CP5_TWO_CANDIDATE_RGBQ_CUDA --authorize-host-cuda
```
