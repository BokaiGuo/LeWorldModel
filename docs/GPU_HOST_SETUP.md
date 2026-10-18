# GPU host setup and permanent execution boundary

Formal learned-world-model experiments must run in a process that can see the
NVIDIA device namespace. `torch.cuda.is_available()` alone is not sufficient:
the process must also allocate a CUDA tensor and initialize the EGL/MuJoCo RGB
renderer.

## One-time platform configuration

For a managed Codex/Desktop sandbox, the platform administrator must expose
these nodes to the application process:

```text
/dev/nvidia0
/dev/nvidiactl
/dev/nvidia-uvm
/dev/nvidia-uvm-tools
/dev/nvidia-modeset
```

The CUDA driver libraries and `/proc/driver/nvidia` must also be visible. The
application must be restarted after changing this mapping; an existing process
does not acquire device nodes retroactively.

The minimum host checks are:

```bash
test -r /proc/driver/nvidia/version
ldconfig -p | grep 'libcuda.so.1'
```

For a Docker-backed deployment, use the NVIDIA container runtime (`--gpus
all`), which mounts the matching driver libraries, and additionally expose
`/proc/driver/nvidia` read-only if the runtime does not do so automatically.
For Codex Desktop, these are managed-platform settings; they cannot be enabled
by a Python package or by a process already running inside the sandbox.

If the platform cannot provide that mapping, use the host terminal. This is a
real execution boundary, not a Python dependency problem, and no repository
code can safely grant a sandbox access to a host device.

## Permanent host-side guard

From the host terminal:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
./scripts/host_gpu_exec.sh \
  ./scripts/run_cube_gpu_repair_panel.sh
```

The guard fails before scientific output if any device node, NVIDIA driver,
CUDA allocation, or EGL render check fails. It never falls back to CPU. To run
another formal command, pass it after `host_gpu_exec.sh`; for example:

```bash
./scripts/host_gpu_exec.sh \
  ./.venv/bin/python -m experiments.run_live_cube_rgb_macro_panel \
  --device cuda --output results/cube_host_gpu_panel
```

Every formal result must retain `host_gpu_preflight.json` or the command's
equivalent `cuda_preflight.json`. A failed preflight is an environment failure,
not a scientific result.
