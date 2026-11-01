# Fixed RGB CUDA runner and journal consumer

This code is prepared for review; no CUDA execution is authorized by this file.
The main thread must review and explicitly decide whether to launch it. The
checkpoint is an intervention-supervised local RGB model, not JEPA, and prior
training used physical state labels. It predicts normalized six-component
model state, not images; no future observation, label, goal, physical state
or velocity is supplied. No comparisons or scientific admission are performed.

The runner pins the fixed-input artifact, input-bridge sources, model source,
loader source and checkpoint bytes; rechecks all 164 image/action inputs before
initializing CUDA, then strictly loads the checkpoint, freezes it, sets eval
and checks CUDA placement. There is no CPU neural forward or CPU fallback.
One state encoder and five recurrent state transitions are called sequentially
per branch, in fixed order. Inference mode is enabled. Predictions are copied
to CPU solely for durable serialization after CUDA execution. Each call has a
fsynced hash-chained attempt and return/failure record. Failure stops the run;
no retry or branch selection occurs. A crash may leave an attempted/unreturned
slot. The consumer rejects incomplete JSON lines rather than silently repairing
or discarding them. Once the writer has stopped, the reviewer supplies the exact
journal SHA to the consumer; consume writes a new exclusive artifact with all
164 branches and five slots, including missing/failure statuses.

The requested time limit (1–3600 seconds) is checked between calls, **not a
hard deadline**: a hung CUDA operation or filesystem sync cannot be interrupted
by it. GPU allocation, hardware and external supervision are review gates; the
exact hardware memory cap is not enforced. Calls are bounded by 164 encodes and
820 transitions; preprocessing may read only the two bound history PNGs per
branch. Preflight failures before journal creation produce no run receipt.
If a journal write fails, the consumer may reject its truncated tail; preserve
the original bytes and investigate rather than repair in place.

`off` and `on` each retain denominator 82 and separate completeness counts.
They are distinct rendering conditions, not exchangeable replicates or proof
that checkpoint training had the same visual distribution. The consumer emits
no pooled accuracy or representation score and does not access future labels.
Log hashes and CUDA metadata are internal consistency/self-attestation, not
independent proof that the stated operations ran on the stated device.
Only a complete, independently reviewed and sealed prediction artifact could
support a separately authorized downstream evaluation.

After explicit approval, the command template is:

```bash
CUDA_VISIBLE_DEVICES=<reviewed-device> .venv/bin/python -m experiments.spectral_reachability.p5_rgb_cuda_runner_20261030_codex_v1 --authorization MAIN_THREAD_APPROVED_FIXED_164_RGB_CUDA --device-index 0 --max-seconds <reviewed-cap> --output-name p5_rgb_cuda_runner_20261030_codex_v1_<unique-run>
```

Compute the stopped journal SHA-256 independently, then run the consumer with
`--run-name` set to the unique run directory name, `--journal-sha256` to that
SHA, and `--output-name p5_rgb_cuda_consumer_20261030_codex_v1_<unique>.json`.
The output directory names are restricted to the project's `results` directory,
are created exclusively, and do not touch the sealed input artifacts.
