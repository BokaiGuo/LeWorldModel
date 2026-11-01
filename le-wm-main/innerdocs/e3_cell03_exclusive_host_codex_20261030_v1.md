# E3 cell03 one-shot HOST entry

This entry binds the existing cell03 authority and original chunked HOST runner.
It adds no scientific admission and does not change frozen inputs or the runner.
The numerical control5 receipt is not a real feedback episode.

**MAIN01's claim and output already exist and must remain untouched. This
command is blocked for MAIN01 and is not a retry instruction.** A further
attempt needs a separately reviewed exclusive attempt identity and an explicit
recovery decision; neither is authorized here. The original command is retained
only to identify the entry:

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.e3_cell03_exclusive_host_codex_20261030_v1 --confirm-host-gpu-coordinated HOST_CUDA_COORDINATED_CELL03_ONCE
```

The confirmation string records the operator's assertion; it cannot verify host
identity, GPU reservation, or prevent unrelated jobs. Coordinate with P3 and
other GPU users separately, and recheck GPU, free disk and environment. The
entry first revalidates saved authority and checks the fixed output path, then
atomically creates and syncs a persistent one-shot claim before calling the
original runner in a fresh Python interpreter, without importing the wrapper or
CPU precheck there. Its source guards, CUDA check, budgets and output publication
remain in force. The claim is an attempt marker, not a success receipt. A partial
result or child exception produces a nonzero exit; zero only means publication
completion, not scientific success.

Only this entry cooperates with the claim. Direct calls to the original runner
or other output names bypass it; do not launch them concurrently. The original
runner's fixed output creation supplies an additional independent no-overwrite
guard. A crash or failure after the claim may leave a claim without an output or
a partial output. Preserve both, diagnose them and obtain explicit direction
before any retry; do not delete the claim or substitute another output name.
An output or dangling symlink at the fixed path also blocks the entry.

The original runner may return a published partial as a normal return, so check
`complete` and inspect actual report and completion hashes using the original
instructions in `e3_cell03_host_readiness_codex_20261030_v1.md`. A completed
publication is not automatically task success, full saved strict, freshness,
or scientific admission.

## Existing partial and CPU validation

MAIN01's journal contains GPU preflight, checkpoint loader receipt and binding
seal, followed by index 15 failure on the unsealed wrapper in the pre-episode
census. No physical episode event was recorded. This was not a CPU-only
attempt: the model was loaded on CUDA before census failed.

Read-only SHA256 identities:

- claim: `595a811ec66cfb7233c79c7e2f0f149f6e41cfda2194acc4f28e465e77901f66`
- report: `3f346697508d2254f7b57bf655ae53fdef1e8fa5f621f8a6c3bf4aaed8e428cb`
- completion: `42e119fbbd50075e725ee2261e96d194b914e0507e44b7082f26a46d5b3abdf2`
- journal: `b41052ec46284a00f9e63b08ef538d835a0ef35970495b6c7dae2f900a0b1fea`

The CPU subprocess integration test uses the real authority and real
`module_census.snapshot`, verifies the parent-only modules are absent, and
injects an unsealed module to verify rejection. A fixture partial and an
exception both produce nonzero child exits. It does not load the actual model,
exercise CUDA, prove the post-model or post-episode census, or validate a real
feedback run. The original runtime census still fails closed for any other
unexpected module.
