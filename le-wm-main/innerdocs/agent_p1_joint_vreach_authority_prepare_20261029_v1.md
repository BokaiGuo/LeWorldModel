# Joint Vreach authority prepare

This is a source-preparation slice only. `prepare` invokes the frozen original
winner authority's complete producer consume and joins all four original
winner charts, tasks, contexts, goals, raw bits, and source plans. It checks
the fixed joint-Vreach request roster, pins the method and callback/audit
sources, seals their tests/docs and installed model-class source paths, and
retains the producer's inherited source/model bindings. `verify` checks the
prepare file SHA and repeats the complete source rebuild. Neither command runs
a model or physics, but original producer strict reconstruction can be costly.

The result is **not** a CUDA host, execution authorization, model/precision
closure after execution, or P1 admission. Production readiness remains false;
the 4/8/160 denominator and original gates are unchanged.

From the project root, MAIN may use the actual existing mixedmode source:

```
.venv/bin/python -m experiments.spectral_reachability.agent_p1_joint_vreach_authority_prepare_20261029_v1 prepare --source-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01 --source-report-sha d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_joint_vreach_authority_prepare_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_joint_vreach_authority_prepare_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_joint_vreach_authority_prepare_20261029_v1_MAIN01.json --prepare-sha PREPARE_SHA_FROM_FIRST_COMMAND
```

Prepare uses exclusive creation and fsync; it refuses to overwrite. Tests use
only in-memory/temporary source fixtures and do not run original full strict:

```
.venv/bin/python -m pytest -q tests/test_agent_p1_joint_vreach_authority_prepare_20261029_v1.py
```
