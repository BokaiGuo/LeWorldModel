# CEM same-action open versus actual-prefix refresh

One known development root, fixed previously executed CEM suffix. The earlier
successful RGB mismatch report is bound by SHA-256
`9445dcb9664d6681d8754eeca742a9994a2b33e41abdcc598d0e342725040a72`.
It records actual RGB latents at cp5/10/15/20 and open-loop latents at
cp10/15/20. It is not an independent root or causal proof.

Both arms start from the exact same stored cp5 three-latent context and two
normalized historical action packets. They execute the exact three actual
CEM action packets in the model, without selecting, replanning or executing
physical actions. At each boundary:

- Open: append its own predicted latent to its sliding three-frame history.
- Actual-prefix: append the corresponding sealed actual RGB latent to its
  sliding history. At cp15 its history therefore contains the actual cp5,
  cp10 and cp15 latents. Both arms advance the same actual normalized actions.

Thus cp10 is the same prediction in both arms; cp15 and cp20 compare open-loop
with predictions conditional on progressively observed real prefixes. This is
teacher-forced one-packet prediction, not an end-to-end feedback policy or a
physical intervention. Previous actual RGB embeddings are reused, not encoded
again. They were generated on host CUDA by the bound prior report.

Report both latent and projected RMSE to the same actual RGB, predicted and
actual RGB Q, and physical task metrics at each matched checkpoint. Check new
open predictions against the saved prior open latents, first predictions
between arms, and original terminal Q. Prior latent arrays, physical actions,
RGB/source closure, model hash, preparation, and durable prior ledger are
validated; no claim is made that `closed_unverified` proves module coverage.
Fixed independent new ledger: six predict calls/samples, six action-encoder
calls, zero encoder/AD/physical steps. Earlier encodings, CEM search, physical
collection and model loading remain separately paid and are not equalized.
FLOPs, energy, internal module operations and full controller cost are unknown.
Failures and partial ledgers are retained.

CPU-only contract tests use an analytic non-neural fixture. No GPU or physics
is started by prepare:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_p1_cem_actual_prefix_refresh_20261101_leibniz.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cem_actual_prefix_refresh_20261101_leibniz dry-prepare
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cem_actual_prefix_refresh_20261101_leibniz prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cem_actual_prefix_refresh_20261101_leibniz_PREP_NEW.json
```

After review, replace `PREPARE_SHA256` with the emitted SHA and use a fresh
output path for host CUDA:

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_cem_actual_prefix_refresh_20261101_leibniz run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cem_actual_prefix_refresh_20261101_leibniz_PREP_NEW.json --sha256 PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cem_actual_prefix_refresh_20261101_leibniz_RUN_NEW --authorization MAIN_AUTHORIZES_P1_CEM_ACTUAL_PREFIX_REFRESH_CUDA --authorize-host-cuda
```

No new result is asserted, no P1 gate is opened, and any improved teacher-forced
error would not establish that feedback improves closed-loop task performance.
