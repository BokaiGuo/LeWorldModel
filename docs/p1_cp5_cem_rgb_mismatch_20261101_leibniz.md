# Same-action CEM predicted versus actual RGB diagnostic

The sealed CEM report contains its actual cp5 latent, candidate scores and
selected actions, but no packet or terminal prediction latent. The trace has
four banks/updates and the selected plan; its ledger records operation
intents/returns, not prediction tensors. The physical replay contains actual
RGB and physical states at cp5/10/15/20. Accordingly the three predicted
latents here are newly recomputed on the frozen model, not original saved
predictions. The existing terminal scalar Q provides a numerical replay check.

Use only the sealed CEM arm and exactly its executed 15 suffix actions. Verify
the original report/prepare, ledger, trace, selection, physical protocol,
actual actions, cp5 identity, PNG byte/pixel hashes, state hashes and inherited
source closure before model work. Encode actual RGB at cp5/10/15/20 separately
using the frozen checkpoint and original image preprocessing. Compare cp5
encoding to the stored cp5 latent (`rtol=1e-5`, `atol=1e-6`); use the
original stored cp5 context and normalized actual actions for autoregressive
H3 rollout. Require replay terminal Q to match the sealed original scalar
(`rtol=1e-6`, `atol=1e-8`) or mark the run partial.

At cp10/15/20 retain newly predicted and actual RGB latents, latent RMSE,
Q-projected RMSE, predicted-Q and actual-RGB-Q under the same centered
quadratic Q, alongside the saved native physical metric vector. These are
different quantities: RGB-Q is neither physical state error nor task success,
and Q difference is not itself causal evidence for the physical outcome.
No physical rollout, outcome-based choice, training or CPU neural fallback.

The independent durable ledger caps 4 encoder calls/images, 3 action-encoder
calls and 3 predict calls/samples; it does not count original CEM planning,
physical acquisition, model loading, image processing, FLOPs or energy.
Report hashes, actual latent arrays, module-entry receipts, synchronized
elapsed time, CUDA allocation peak, failures and source recheck. Ledger
`closed_unverified` is not independently instrumented model-internal compute.
This is a single known-root diagnostic, not an official baseline or P1 gate.

After CPU test and review, prepare without GPU:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_p1_cp5_cem_rgb_mismatch_20261101_leibniz.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cp5_cem_rgb_mismatch_20261101_leibniz dry-prepare
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cp5_cem_rgb_mismatch_20261101_leibniz prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_cem_rgb_mismatch_20261101_leibniz_PREP_NEW.json
```

Main-thread host CUDA launch after explicit authorization, using the printed
prepare SHA and a new output path:

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_cp5_cem_rgb_mismatch_20261101_leibniz run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_cem_rgb_mismatch_20261101_leibniz_PREP_NEW.json --sha256 PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_cem_rgb_mismatch_20261101_leibniz_RUN_NEW --authorization MAIN_AUTHORIZES_CP5_CEM_RGB_MISMATCH_HOST_CUDA --authorize-host-cuda
```
