# Official JEPA PushT physical bridge prerequisite

Scope: episode1 only, existing frames0/5/10 and already-observed target15.
No simulator construction/reset/physics, neural work, optimization, training,
downloads or changes to native/sealed sources. This new prerequisite is **blocked**;
there is intentionally no physics/CUDA runner or execution command.

## Exact action interface

Native dataset divides `rel_actions.pth` by100, then normalizes using
mean `[-.0087,.0068]`, std `[.2019,.2002]`. For each new normalized xy pair z,
pass `u=z*std+mean` to native `env.step`. The simulator multiplies u by100
and adds the **current** agent position once to get that tick's positional
target. It holds this target for10 substeps of .01s with PD gains100/20.
Five controls represent .5 simulation seconds, not five 30fps video ticks.
These are relative positional targets, NOT commanded velocities.
Normalized zero is raw displacement `[-.87,.68]` pixels, NOT raw zero.
No clipping, tanh, LeWM[-1,1] substitution, or precomputing all five absolute
targets from frame10. The module returns both native arguments and raw deltas.

Prepare checks all15 `abs_actions-agent_xy-rel_actions` exactly and binds the
three native 5x2 packets and proprio/RGB to the frozen v8 episode1 manifest.
Float32 native normalization roundtrip is recorded, not asserted bit-exact
after inverse conversion. Agent velocity stored float64 is cast float32 for
the native dataset proprio path; full stored values are separately retained.
The first two action packets/history are not modified.

## Concrete reset/runtime/RGB blockers

- Saved state5 is agentxy, blockxy/angle, float32; separate velocity2 is
  **agent** velocity, not block velocity (despite `_set_state` local variable
  name `vel_block`). No block linear/angular velocity/contact solver snapshot.
- `_set_state` assigns pose/agent velocity, then **advances physics .01s**.
  Legacy angle/position assignment order affects noncentral block CoG.
  `reset_to_state` is not a pure restore. Cannot infer exact frame10 root.
- Episode1 saved block poses0..15 happen to be constant; this does not prove
  hidden velocities/contact state zero. A generator-bound initial snapshot
  plus factual prefix replay could be an alternative, but generation revision,
  precision, noise, reset phase and configuration are not authenticated here.
- Shapes/goal/colors are procedural; no mesh/texture download is needed for
  this native T renderer. Missing shapes.pkl makes the dataset default to T,
  but this is not a verified episode generation/render setting.
- Native pygame canvas512 is resized by OpenCV to224. Installed episode video
  is H.264/yuv420p224 with111frames/30fps. Lossy decoded RGB is bound to v8, not
  asserted equal to freshly rendered RGB. No source-bound generator/codec
  settings or uncompressed originals establish exact RGB/time identity.
- Current interpreter lacks `skimage.transform`. Native pyproject requires
  scikit-image, gym0.23.1 and pymunk6.8.0; installed gym0.26.2/pymunk7.2.0 differ.
  The official `_setup` calls `Space.add_collision_handler`, absent on current
  pymunk7.2.0. No dependency installation/API shim is attempted.

Installed stable_worldmodel PushT source is independently bound: its current
`step` also uses relative PD targets, not pure velocity. Any separate canonical
LeWM velocity executor has a different interface; no equivalence or simple
scalar conversion to that contract is claimed. Do not conflate the package
environment with a custom executor or substitute one for the official env.

## Preparation and freeze

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_physical_bridge_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_physical_bridge_20261027_v1
```

Default command only prints a prepare JSON. Saved `manifest_frozen.json`
contains source/data hashes before/after, safe SHA-restricted tensor metadata,
all relevant episode1 values, four decoded RGB identity and ffprobe metadata,
specific prerequisite blockers and zero physics/neural call counts. It does
not import the unsafe dataset class or decode any dataset pickle. Sequence
pickle is hash-bound only; frame availability is checked by video decoding.
It never imports old experiment consumers or models. Native sources are read,
not imported. Pymunk class API inspection constructs no Space/Body/environment.

Next authorized evidence would first need isolated compatible native runtime
and authenticated generation/reset/render binding, then a separately frozen
main-run factual replay identity protocol. No blanket claim that replay is
impossible; it is not supported/authorized as exact with current inputs.
v9's delta fitted observed frame15 remains fitting diagnostics, not prospective
planning or physical/control benefit. Neither this prepare nor conversion
tests admit P5.
