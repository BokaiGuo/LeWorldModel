"""Six-slot native session adapter. No launcher, storage admission or certification.

The caller must independently verify authorities and own the journal, store,
model, device, deadlines and consumer. Creating a Factory performs no native IO.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p3_feedback_reverse_control_20261030_v1 as control
from experiments.spectral_reachability import p3_matched_order_feedback_native_v2_codex_20261029 as native
from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as native_meter
from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as reset

need = control.need
FRAME_CAP = 192
SLOT_FRAME_CAP = 32
PNG_CAP = native.capture.pub.PNG_CAP


class CaptureStore(native.capture.CaptureStore):
    """Per-slot 32 and global 192 write-before caps; store enforces byte caps."""
    def frame(self, name, image):
        need(type(name) is str and Path(name).name == name and name.endswith(".png") and
             name not in self.frames and len(self.frames) < SLOT_FRAME_CAP and
             type(self.store.frames) is int and 0 <= self.store.frames < FRAME_CAP,
             "six-slot native frame limits")
        a = np.asarray(image)
        need(a.shape == (224, 224, 3) and a.dtype == np.uint8, "native RGB224")
        stream = io.BytesIO()
        Image.fromarray(a).save(stream, format="PNG")
        data = stream.getvalue()
        pixel_sha = native.capture.pub.png_data(data)
        descriptor = dict(file=name, sha256=native.capture.pub.base.fsha(data),
                          pixel_sha256=pixel_sha, shape=[224, 224, 3], dtype="uint8")
        self.store.write(name, data, "physical_png")
        self.store.frames += 1
        self.frames[name] = descriptor
        self.store.event("physical", "record", "native_frame", self.store.index, {"frame": descriptor})
        return descriptor["sha256"]


def preflight(prepared, source, store):
    """Structural checks only; no source authentication or total admission."""
    need(type(prepared) is dict and type(prepared.get("tasks")) is list and
         len(prepared["tasks"]) == 2 and type(source) is dict and
         type(source.get("roots")) is list and len(source["roots"]) == 2 and
         type(source.get("source_plan")) is dict, "two tasks, roots and source plan")
    for task, root in zip(prepared["tasks"], source["roots"], strict=True):
        need(type(task) is dict and type(task.get("binding")) is dict and type(root) is dict and
             type(root.get("root_sha256")) is str and len(root["root_sha256"]) == 64 and
             all(c in "0123456789abcdef" for c in root["root_sha256"]) and
             task["binding"].get("root_sha256") == root["root_sha256"], "root/task binding")
    need(type(getattr(store, "frames", None)) is int and store.frames == 0 and
         type(getattr(store, "index", None)) is int and store.index >= 0 and
         all(callable(getattr(store, x, None)) for x in ("check", "event", "write")) and
         isinstance(getattr(store, "output", None), Path), "fresh store ABI")
    # This is a lower bound for PNG category only, not a combined reservation.
    store.check("physical_png", FRAME_CAP * PNG_CAP)
    return dict(roster=[list(x) for x in control.ROSTER], png_max_bytes=FRAME_CAP * PNG_CAP,
                storage_admitted=False, full_saved_strict=False, scientific_pass=False)


def source_contract():
    """Rebuild previous six-slot seal and add this adapter's exact source hash."""
    from experiments.spectral_reachability import p3_six_host_prepare_20261030_v1 as preparation
    prior = preparation.build()
    path = Path(__file__).resolve()
    need(path.is_file() and not path.is_symlink(), "native factory source")
    return dict(schema="p3_six_native_factory_20261030_v1.source", previous=prior,
                factory_file=str(path), factory_sha256=preparation.old_host.source.host.sha(path),
                frame_cap=FRAME_CAP, per_slot_frame_cap=SLOT_FRAME_CAP,
                storage_admitted=False, host_runner_ready=False, scientific_pass=False)


def verify_source_contract(value):
    need(value == source_contract(), "factory source contract drift")
    return value


class Factory:
    """Fixed order, exactly one attempt per slot, one open environment at a time."""
    def __init__(self, prepared, source, store, model, torch, device, *,
                 meter_module=native_meter, native_module=native, reset_module=reset):
        self.preflight = preflight(prepared, source, store)
        self.prepared, self.source, self.store = prepared, source, store
        self.model, self.torch, self.device = model, torch, device
        self.meter_module, self.native_module, self.reset_module = meter_module, native_module, reset_module
        self.next_slot = 0
        self.active = False
        self.poisoned = False

    def __call__(self, slot, root, policy):
        need(not self.active and not self.poisoned and type(slot) is int and type(root) is int and
             type(policy) is str and
             self.next_slot < len(control.ROSTER) and
             (slot, root, policy) == control.ROSTER[self.next_slot], "fixed ordered single attempt")
        self.next_slot += 1
        episode = f"matched_slot_{slot:02}"
        row = dict(episode_id=episode, warmup_returned=0, control_returned=0,
                   executed_raw=[], PNG=0, checkpoints=[], status="started")
        meter = self.meter_module.Meter(self.store, episode)
        capture = CaptureStore(self.store)
        original = env = None
        try:
            original = self.meter_module.native_factory(meter)
            env = self.reset_module.TracedEnv(original, self.store, episode)
            goal, goal_rgb, prior = self.native_module.setup(
                env, self.source["roots"][root], self.source["source_plan"],
                self.prepared["tasks"][root], row, meter, capture)
            session = self.native_module.Session(env, goal, goal_rgb, meter, capture, row,
                                                 self.model, self.torch, self.device)
            self.store.event("physical", "record", "six_session_ready", self.store.index,
                             dict(slot=slot, root=root, policy=policy, episode_id=episode))
        except BaseException as exc:
            target = env if env is not None else original
            if target is None:
                # Constructor may have allocated before its metering return failed.
                self.poisoned = True
            else:
                try:
                    target.close()
                except BaseException as cleanup:
                    self.poisoned = True
                    raise RuntimeError("setup failure with unsuccessful cleanup: " + repr(cleanup)) from exc
            raise
        self.active = True

        def close():
            need(self.active, "one cleanup attempt")
            self.active = False
            try:
                return env.close()
            except BaseException:
                self.poisoned = True
                raise

        return session, prior, close
