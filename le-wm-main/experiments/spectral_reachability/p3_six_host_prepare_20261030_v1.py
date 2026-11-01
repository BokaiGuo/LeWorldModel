"""Six-slot HOST preparation only. No execution authority or launcher."""
from pathlib import Path

from experiments.spectral_reachability import p3_feedback_reverse_control_20261030_v1 as control
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as old_host

NAME = "p3_six_host_prepare_20261030_v1"
need = control.need


def build():
    """Reconstruct and seal existing source closure plus the new isolated modules."""
    old = old_host.build()
    files = dict(old["implementation_sha256"])
    for path in (Path(__file__), Path(control.__file__),
                 Path(__file__).with_name("p3_six_saved_consumer_20261030_v1.py"),
                 Path(__file__).with_name("p3_six_numeric_bridge_20261030_v1.py"),
                 Path(__file__).with_name("p3_six_lifecycle_20261030_v1.py")):
        path = path.resolve()
        need(path.is_file() and not path.is_symlink(), "required source")
        old_host.source.physical.merge(files, {str(path): old_host.source.host.sha(path)})
    for name, digest in files.items():
        path = Path(name)
        need(path.is_file() and not path.is_symlink() and old_host.source.host.sha(path) == digest,
             "source drift")
    return dict(schema=NAME + ".prepare", prior_prepare=old, implementation_sha256=files,
                roster=[dict(slot=s, root=r, policy=p) for s, r, p in control.ROSTER],
                per_slot_caps=control.design.expected(), callback_caps=dict(packet=4, encode=4),
                planned_actions=20, native_observations_per_slot=32,
                frame_count_six_slots=192, storage_admitted=False,
                six_slot_storage_cap_bytes=None, six_slot_wall_cap_seconds=None,
                six_slot_cuda_caps=None, host_runner_ready=False,
                physical_steps=None, encoder_kernels=None, FLOPs=None,
                known_roots=2, fresh_roots=0, scientific_pass=False,
                preregistered=False)


def verify(value):
    need(value == build(), "six-slot prepare/source drift")
    return value


def prepare(path):
    """Exclusive, fsynced output in this module's own results namespace."""
    path = Path(path).absolute()
    need(path.parent == old_host.ROOT / "results" and path.name.startswith(NAME) and
         path.resolve() == path and not path.exists(), "exclusive preparation path")
    return old_host.write_new(path, build())
