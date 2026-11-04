# v13 isolated synthetic-observation interface smoke

This is an engineering diagnostic, not a task performance experiment. It
does not use any observation, goal, action, or seed from the stopped
contact-root experiment. Generate reproducible synthetic RGB arrays,
normalized proprio and action history, and bounded synthetic carries.
At each of the three horizon lengths 15, 10, and 5, give both the v13
and matched global-CEM worker the same input arrays (except arm identity).

Each of the six isolated workers uses the real frozen JEPA checkpoint and
the same network/filesystem sandbox as the native feedback worker, and the
usual 1920 prediction-token cap. No physical environment or action
dispatch is involved. Save inputs, worker records, exit codes, dependency
hashes, and full evaluations. Audit legal action bounds, bank identity,
carry inclusion, horizons, selected-chain binding, pairs and token counts.
Record and retain any failure; this diagnostic has exactly six attempts.
Do not substitute synthetic outputs for PC0–PC5 evidence or claim a
physical feedback loop.
