# Exploratory host Landlock wrapper

This is a development wrapper, not a production authorization or proof of
filesystem confinement against a malicious process. It invokes the existing
H2 driver only after Landlock is installed in the child before `exec`.
Nothing in the wrapper grants GPU exclusivity or approves a physical run.
Do not run it until the independent host and scientific reviews are complete.

## Manifest

Supply a JSON object with exactly these keys:

```
{
  "python": "/absolute/staged-venv/bin/python",
  "runtime_dirs": ["/usr", "/absolute/staged-venv"],
  "source_dir": "/absolute/staged-source-tree",
  "bundle": "/absolute/path/to/blind.json",
  "bundle_sha256": "<64 hex digits>",
  "model_dir": "/absolute/model-directory",
  "model_sha256": {"weights.pt": "<64 hex digits>", "config.json": "<64 hex digits>"},
  "output_dir": "/absolute/new-empty-exclusive-output-directory",
  "deny_paths": ["/absolute/historical-results-directory", "/absolute/physical-truth-directory"],
  "cuda_device": "<one coordinated CUDA_VISIBLE_DEVICES identifier>"
}
```

The source tree must contain the reviewed `experiments` package and only
approved dependencies, not a copy of the full repository. Directory grants
are recursive: the operator must inventory and approve every readable file
inside each runtime and source directory, including `.pth` files and symlink
targets. `deny_paths` are existing directories; directory grants overlapping
them are rejected. The explicitly pinned bundle may be inside a denied tree.
This list cannot identify undeclared or copied sensitive files, so it is not a
complete safety proof. Model content access is limited to the two named files;
listing the model directory is permitted.

The wrapper checks file hashes before launch; those checks do not prevent
concurrent replacement or mutation. Inputs, code and runtime require an
independently audited immutable snapshot. Paths must be absolute and canonical;
the Python executable may be a virtualenv symlink provided its target and
logical path are both in allowed runtime directories. The output directory
must be dedicated and empty, but emptiness is not a cross-entry claim or an
atomic reservation against concurrent actors.

After review and explicit coordination, a host operator could use:

```
/usr/bin/python3 /absolute/path/to/codex_p1_h2_landlock_wrapper_20261030_v1.py --manifest /absolute/reviewed-manifest.json --acknowledge EXPLORATORY_HOST_CUDA_NOT_PRODUCTION_APPROVED
```

The wrapper emits the manifest digest, applied rules, child exit status and
captured output on stdout. The operator must save this stream securely to a
new external audit file; it is not itself a durable signed receipt. A nonzero
child exit propagates as failure. No automatic retry is performed.

## Mandatory outstanding gates

- Independently inspect the complete source, Python runtime, shared-library,
  CUDA-driver and CUDA-device access paths on the actual host; run deliberate
  denial tests on representative historical outcomes, future frames and
  physical-state paths without reading their content. Validate real CUDA
  availability and required device/ioctl behavior only after coordination.
- Landlock is not a mount namespace, network or IPC boundary; it does not
  hide metadata or revoke already-open descriptors, and does not control every
  device action or ioctl. The wrapper uses fresh pipes and closes inherited
  descriptors above 2, but shared memory, sockets, other processes, privileged
  actors and kernel access remain separate threats. Install external controls
  as necessary; do not treat offline environment variables as network isolation.
- Review dynamic dependencies, `.pth` and imported modules before execution.
  An incomplete ruleset must fail rather than broaden access to the repository
  or results. `preexec_fn` assumes this wrapper is single-threaded.
- Independently establish GPU reservation, resources, attempt identity,
  cross-entry deduplication and complete output verification. A successful
  exploratory driver run does not authorize physics or resolve the eight
  missing greedy conditions, scientific gates or budget parity.
- Earlier adapter tests loaded historical outcomes and later checkpoints.
  This method is exploratory and has no preregistration or clean-blinding
  guarantee.
