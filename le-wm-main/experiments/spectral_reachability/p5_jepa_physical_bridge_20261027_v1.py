"""Non-neural/native-physics-free PushT bridge prerequisite preparation.

This deliberately has NO simulator execution path: existing observations are
not complete native snapshots. It binds units, inputs and exact blockers.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

NAME = "p5_jepa_physical_bridge_20261027_v1"
V8 = "p5_jepa_native_action_response_20261027_v8"
V8_MANIFEST_SHA = "0588f825c29b884350e97ed0dc47d8872bb539d0ca7a7faf045190eee5f3340f"
DATA = "data/jepa_wm_osf/extracted/pusht_noise/val"
NATIVE = "external/jepa-wms"
ENV = NATIVE + "/evals/simu_env_planning/envs/pusht_env/pusht_env.py"
DSET = NATIVE + "/app/plan_common/datasets/pusht_dset.py"
ARRAYS = {
    "states": ("7ef3ab74983415802e342642e41bf6429b927d3fff7a54db15da74745dd2fcb0", 5, "torch.float32"),
    "velocities": ("b1e4694152206f34931cc5015dfefb4456608c42d9f6a7f2688d20f2fc90d59b", 2, "torch.float64"),
    "rel_actions": ("c5e04761f261db6f96b2d16db35e5f1d7e42b8d2c507994713c98343d0f685dc", 2, "torch.float32"),
    "abs_actions": ("141231489d7dbdc8a5b90efb6ca764d89e7679aed4a892e0e9aaad0895c3234d", 2, "torch.float32"),
}
MEAN = np.array([-.0087, .0068], np.float32)
STD = np.array([.2019, .2002], np.float32)
PROPRIO_MEAN = np.array([236.6155,264.5674,-2.93032027,2.54307914], np.float32)
PROPRIO_STD = np.array([101.1202,87.0112,74.84556075,74.14009094], np.float32)


def sha(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("regular non-symlink file required: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def numeric(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fi" or not np.isfinite(a).all():
        raise ValueError("finite numeric input with exact shape required")
    return a.astype(np.float64)


def command_packet(normalized):
    """Five native normalized xy coordinates -> env arguments AND pixel deltas.

    No clipping, velocity mapping, fixed absolute targets, or tanh. Each env
    argument is relative to the actual agent position at that control step.
    """
    z = numeric(normalized, (5, 2))
    env_argument = z * STD.astype(np.float64) + MEAN.astype(np.float64)
    return {"native_relative_env_arguments": env_argument,
            "raw_target_displacements_pixels": env_argument * 100.}


def absolute_target(agent_xy, env_argument):
    """Single control-step target; never precompute five targets from frame10."""
    return numeric(agent_xy, (2,)) + 100. * numeric(env_argument, (2,))


def restricted_array(path, digest, width, dtype):
    # Exact locally bound files only; no dataset import or pickle decoder.
    if Path(path).stat().st_size > 256 * 1024 or sha(path) != digest:
        raise ValueError("array SHA/size rejected")
    import torch
    t = torch.load(path, weights_only=True, map_location="cpu")
    if (type(t) is not torch.Tensor or t.layout != torch.strided or t.device.type != "cpu"
            or tuple(t.shape) != (21, 246, width) or str(t.dtype) != dtype
            or not bool(torch.isfinite(t).all())):
        raise ValueError("array tensor contract rejected")
    return t.numpy().copy()


def source_stats(path):
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("ACTION_MEAN", "ACTION_STD", "PROPRIO_MEAN", "PROPRIO_STD"):
                if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != "torch.tensor":
                    raise ValueError("native stats are not literal tensors")
                values[node.targets[0].id] = np.asarray(ast.literal_eval(node.value.args[0]), np.float32)
    for key, expected in (("ACTION_MEAN", MEAN), ("ACTION_STD", STD),
                          ("PROPRIO_MEAN", PROPRIO_MEAN), ("PROPRIO_STD", PROPRIO_STD)):
        if key not in values or not np.array_equal(values[key], expected):
            raise ValueError("native stats drift: " + key)
    return {k: v.tolist() for k, v in values.items()}


def protocol():
    return dict(schema=NAME, episode=1, context_frames=[0,5,10], observed_target_frame=15,
        history_action_slices=[[0,5],[5,10]], new_packet_slice=[10,15],
        physics_executed=False, neural_executed=False, execution_command=None,
        reset_policy="NO reset: seven-dimensional observations are not native physics snapshots",
        action_policy="native env.step(z*ACTION_STD+ACTION_MEAN); internal scale100; relative PD target per control step",
        timing=dict(sim_hz=100, control_hz=10, substeps_per_control=10, dt=.01,
                    packet_controls=5, packet_simulation_seconds=.5, reset_extra_substeps=1),
        rendering="512 procedural pygame -> cv2.resize224 RGB; goal drawn even if white with_target=False",
        selection="episode1 only; no replacement, no fit, no prospective goal/benefit claim",
        observed_target_scope="frame15 and v9 ridge delta are already-open fitting diagnostics, not prospective planning",
        P5_admission=False)


def blockers(versions, skimage_available, collision_handler_available):
    items = [
        dict(id="incomplete_physics_snapshot", evidence="states.pth stores5 float32 pose entries; velocities.pth stores only agent vx/vy",
             missing="block linear/angular velocity, full precision pre/post-reset state and contact/solver state at frame10; or authenticated generator + initial full state permitting identity-checked prefix replay"),
        dict(id="reset_advances_physics", evidence="PushTEnv._set_state assigns agent velocity then calls space.step(1/sim_hz); legacy assignment order changes block center-of-mass placement",
             missing="bound generation legacy/CoG/damping/settings and reset phase; do not treat reset_to_state as a pure restore"),
        dict(id="generator_render_identity_unbound", evidence="local arrays/video have no episode generation settings/full snapshot; shapes.pkl absent -> dataset fallback T is a convention, not generator authentication",
             missing="episode1 generator revision/config (shape/scale/color/goal/with_target/legacy/noise), renderer/codec settings and source-linked state/RGB replay receipts; no uncompressed original RGB"),
        dict(id="video_timebase_not_physics_clock", evidence="container fps metadata is not proof of simulator action time/alignment",
             missing="generation alignment showing one saved action/observation pair per control tick including reset phase; do not resample at video30fps or infer .033s controls"),
    ]
    if not skimage_available:
        items.append(dict(id="missing_scikit_image", evidence="native pusht_env.py imports skimage.transform; unavailable in current interpreter", missing="reviewed compatible scikit-image installation in isolated native runtime; no installation attempted"))
    if versions.get("pymunk") != "6.8.0" or not collision_handler_available:
        items.append(dict(id="pymunk_api_version_mismatch", evidence=f"installed pymunk={versions.get('pymunk')}; native pyproject pins6.8.0; add_collision_handler available={collision_handler_available}", missing="native pymunk6.8.0 runtime, not a source patch translating to pymunk7"))
    if versions.get("gym") != "0.23.1":
        items.append(dict(id="gym_version_mismatch", evidence=f"installed gym={versions.get('gym')}; native pins0.23.1", missing="native gym0.23.1 runtime and reviewed wrapper API, no silent Gymnasium substitution"))
    return items


def prepare(root):
    root = Path(root).resolve()
    d = root / DATA
    native = root / NATIVE
    v8_path = root / f"results/{V8}/manifest_frozen.json"
    if sha(v8_path) != V8_MANIFEST_SHA:
        raise ValueError("frozen v8 manifest rejected")
    parent = json.loads(v8_path.read_text())
    source_paths = list((native / "evals/simu_env_planning/envs/pusht_env").rglob("*.py"))
    source_paths += [root/DSET, native/"evals/simu_env_planning/envs/pusht_gym_wrap.py", native/"pyproject.toml", native/"LICENSE",
        Path(__file__).resolve(), root/"tests"/f"test_{NAME}.py", root.parent/"docs"/f"{NAME}.md", v8_path,
        root/".venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py"]
    files = source_paths + [d/(name+".pth") for name in ARRAYS] + [d/"seq_lengths.pkl",d/"obses/episode_001.mp4"]
    before = {str(p): sha(p) for p in files}
    stats = source_stats(root / DSET)
    arrays = {k: restricted_array(d/(k+".pth"), *spec) for k,spec in ARRAYS.items()}
    s, vel, rel, absolute = (arrays[k] for k in ("states", "velocities", "rel_actions", "abs_actions"))
    # Reproduce native dataset float32 conversion before normalization exactly.
    raw_prop = np.concatenate([s[1,[0,5,10,15],:2],vel.astype(np.float32)[1,[0,5,10,15]]],axis=-1)
    prop = (raw_prop-PROPRIO_MEAN)/PROPRIO_STD
    packets = ((rel[1,:15]/100.-MEAN)/STD).reshape(3,10)
    values = dict(raw_proprio=raw_prop, normalized_proprio=prop, raw_actions=rel[1,:15], packets=packets)
    for k,v in values.items():
        saved = parent["row"]["inputs"][k]
        if array_sha(v) != saved["sha256"] or v.tolist() != saved["values"]:
            raise ValueError("v8 actual input identity differs: " + k)
    import decord
    video = decord.VideoReader(str(d/"obses/episode_001.mp4"),num_threads=1)
    rgb = video.get_batch([0,5,10,15]).asnumpy()
    if (rgb.shape != (4,224,224,3) or rgb.dtype != np.uint8
            or array_sha(rgb) != parent["row"]["rgb_sha256"] or len(video) != parent["row"]["video_length"]):
        raise ValueError("decoded RGB identity rejected")
    metadata = json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames","-of","json",str(d/"obses/episode_001.mp4")],
        check=True,capture_output=True,text=True,timeout=20).stdout)
    versions = {}
    for name in ("pymunk","gym","pygame","numpy","opencv-python","scikit-image","decord","shapely"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    # Class attribute inspection only; NO Space/Body/environment construction.
    import pymunk
    available = hasattr(pymunk.Space,"add_collision_handler")
    if (d/"shapes.pkl").exists():
        raise ValueError("unexpected shape metadata; not authorized to decode pickle")
    abs_error = float(np.max(np.abs(absolute[1,:15]-rel[1,:15]-s[1,:15,:2])))
    if abs_error != 0:
        raise ValueError("absolute/relative action identity mismatch")
    converted = command_packet(packets[-1].reshape(5,2))
    source_after = {str(p):sha(p) for p in files}
    if before != source_after:
        raise ValueError("source/data changed during preparation")
    return dict(schema=NAME,status="blocked_exact_native_physical_bridge",protocol=protocol(),seals=before,
        source_data_hashes_unchanged=True,native_stats=stats,versions=versions,
        native_source_references={"reset_and_step":ENV+":433", "reset_advances":ENV+":646", "setup":ENV+":684", "render":ENV+":578", "dataset":DSET+":38"},
        local_inventory=sorted(str(p.relative_to(d)) for p in d.iterdir()),
        seq_lengths_policy="SHA only; numpy pickle globals NOT unpickled; selected frame availability verified by video decode",
        shape_policy="absent shapes.pkl -> official dataset fallback T only; generator shape not authenticated",
        blockers=blockers(versions, importlib.util.find_spec("skimage") is not None, available),
        row=dict(episode=1,frame_indices=[0,5,10,15],state7=np.concatenate([s[1,:16],vel[1,:16]],axis=-1).tolist(),
            raw_relative_actions_0_14=rel[1,:15].tolist(),raw_absolute_targets_0_14=absolute[1,:15].tolist(),
            abs_minus_rel_minus_agent_max_abs=abs_error,inputs={k:dict(sha256=array_sha(v),values=v.tolist()) for k,v in values.items()},
            normalized_zero_raw_pixel_displacement=(MEAN.astype(np.float64)*100).tolist(),
            prepared_actual_packet={k:v.tolist() for k,v in converted.items()},
            normalized_roundtrip_pixel_max_abs=float(np.max(np.abs(converted["raw_target_displacements_pixels"]-rel[1,10:15]))),
            block_pose_0_15_constant=bool(np.all(s[1,:16,2:]==s[1,0,2:])),
            block_static_limit="saved float32 poses constant is NOT a measured zero velocity or contact cache snapshot",
            video=dict(decoded_frames=len(video),decoder_fps=float(video.get_avg_fps()),rgb_sha256=array_sha(rgb),metadata=metadata),
            v8_input_identity=True),
        canonical_boundary="installed stable_worldmodel PushT source also uses relative PD targets (env.py:312), NOT pure velocity. Any separate canonical LeWM velocity executor is a different contract; no scalar velocity conversion/equivalence asserted.",
        required_next_evidence="First establish isolated native dependency/runtime and authenticated generation/reset/render binding. Only then main may commission a separately frozen factual prefix replay identity protocol; this prepare authorizes no execution.",
        cost=dict(env_construct=0,reset=0,physics_step=0,neural_forward=0,video_frames_requested=4,
                  internal_codec_frames_decoded="not instrumented; inter-frame codec may decode additional frames"),P5_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(prepare(args.root),sort_keys=True,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
