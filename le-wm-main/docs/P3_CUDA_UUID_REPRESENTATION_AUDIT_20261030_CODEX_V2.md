# P3 CUDA UUID representation audit V2

## Observations

MAIN01 `failure.json` SHA256
`379c063a2e9ab121cab68ea352e9bfc56fbc759a9aaba8cf110937cfbd39e09c`
records stage `runtime`, message `runtime UUID`, `roster_returned=false`,
`decision_finished=false`, and `retry_authorized=false`. The user service
traceback identifies driver line 163:
`str(torch.cuda.get_device_properties(0).uuid) == plan["gpu_uuid"]`.
The service result was exit-code 1. Its native output directory was empty;
the decision journal contained one begin record, SHA256
`c13d6dd35437907777c29fbf2b99a4fd085de74b1d7c95b82fd492b6d369c377`.
The code calls `require_cuda_allocation` before the failure; it had not yet
called checkpoint loading, constructed the adapter, or opened physics.

The main thread reports the runtime attribute repr as a
`torch._C._CUuuid` object, and `u.bytes` as
`[78,137,115,255,2,120,83,243,43,174,38,21,212,97,228,65]`.
The bytes in reported order are `4e8973ff027853f32bae2615d461e441`;
standard UUID byte formatting yields
`GPU-4e8973ff-0278-53f3-2bae-2615d461e441`. Read-only `nvidia-smi` returned
that same UUID. This matching byte observation is supplied by the main thread;
the failed run did not persist the UUID bytes and this audit did not retrieve
them from CUDA. Therefore the string representation bug is supported by the
trace and reported host observation, but the failed artifact alone does not
prove the runtime byte identity.

Without CUDA initialization, the installed torch reports version
`2.12.0+cu130`, exposes `torch._C._CUuuid` with a `bytes` descriptor, and
`torch.cuda.is_initialized()` was false. Installed CUDA 13
`include/driver_types.h` defines `CUuuid_st` as a 16-byte UUID and associates
it with device properties. The current device's `_CUuuid.bytes` data itself
was not accessed in this audit.

The same faulty conversion also occurs in the old adapter's guard and in the
identity snapshot. Correcting only the driver's first comparison would leave
later failures.

## Prototype and integration boundary

`experiments/spectral_reachability/p3_cuda_uuid_bridge_20261030_codex_v2.py`
(SHA256 `e6546e4f7ac84cddfcb2de7f4196c6549284ff9df8e31635166ad2b5f4613b00`)
offers a local torch view that:

- Requires the plan UUID, independently queried nvidia-smi UUID and exact
  `CUDA_VISIBLE_DEVICES` UUID to agree, and exactly one visible device.
- Requires the exact `torch._C._CUuuid` type and 16 integer bytes in [0,255],
  compares their wire-order value against the full admitted UUID, and exposes
  a canonical string to the unchanged adapter. It checks again whenever the
  adapter requests device properties, including snapshot and guard calls.
- Does not mutate global torch; all other attributes delegate to the original.
  It records the 16-byte hex and both identity inputs in `binding_evidence`.

CPU stand-in tests cover the reported bytes, byte reversal, missing/extra and
invalid bytes, wrong object type, spoofed smi/visibility inputs, multiple
devices, wrong index, the existing adapter's guard/poisoning, and the existing
snapshot. They cannot establish real `_CUuuid` ABI or the correctness of a
full native run.

This module is **not connected to a launcher**. A separately versioned driver
must acquire and retain a fresh nvidia-smi snapshot, construct the view from
the real torch object and environment, use it for the initial check and pass
the same view to the existing adapter, persist binding evidence, and bind the
new driver and bridge SHA in a new pre-authorized plan. It must preserve all
original source, precision, resource, process supervision and failure gates.
An independent host operator must verify actual byte values and runtime behavior
in the authorized service; there is no CPU fallback and no authorization here
to retry MAIN01 or launch another run. The existing plan pins the original
driver, so it does not authorize a modified launch chain.
