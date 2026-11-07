"""Initial-render and teacher-forced fixed-plan diagnostic; no action selection."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_teacher_forced_rollout_dev_20261106_codex import teacher_predictions
from experiments import pc_pusht_first_plan_alignment_20261107_codex as previous

OUT = previous.RESULTS / "pc_pusht_first_plan_initial_teacher_20261107_codex"
PROTOCOL = previous.ROOT.parent / "docs/PC_PUSHT_FIRST_PLAN_INITIAL_TEACHER_PROTOCOL_20261107_CODEX.md"


def rollout(model, image, chains):
    pixels = image[None, None, None].expand(1, 2, -1, -1, -1, -1).clone()
    context = {"pixels": pixels}
    model.rollout(context, chains[None])
    return context["predicted_emb"][0]


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    prior_json = previous.OUT.with_suffix(".json")
    prior_npz = previous.OUT.with_suffix(".npz")
    report = json.loads(prior_json.read_text())
    if file_hash(prior_npz) != report["arrays_sha256"]:
        raise ValueError("prior representation hash mismatch")
    for name, digest in report["source_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"prior source changed: {name}")
    stat = previous.DATA.stat()
    if (stat.st_size, stat.st_mtime_ns) != (report["data_size"], report["data_mtime_ns"]):
        raise ValueError("dataset metadata changed")
    with np.load(prior_npz) as archive:
        prior_pred, real, goal = [archive[k].copy() for k in ("predicted", "actual", "goal")]
    audits = [previous.load_first(stem) for stem in (previous.OLD, previous.NEW)]
    traces = [load(stem.with_suffix(".trace.jsonl")) for stem in (previous.OLD, previous.NEW)]
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != report["model_sha256"]:
        raise ValueError("checkpoint differs from previous diagnostic")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    all_initial, all_rendered, all_teacher, rows = [], [], [], []
    print("Fixed plans only: compare data-frame rollout, rendered-frame rollout and evaluation-only teacher forcing.", flush=True)
    with h5py.File(previous.DATA, "r") as data:
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for index, row in enumerate(report["rows"]):
                source_row = row["data_row"]
                if row["index"] != index:
                    raise ValueError("row order mismatch")
                actions = torch.tensor(np.stack([a[index]["selected_actions"] for a in audits]),
                                       dtype=torch.float32, device="cuda")
                seed = 871001 + index
                env.reset(seed=seed)
                actor = env.unwrapped
                actor._set_state(np.asarray(data["state"][source_row], dtype=np.float64))
                actor._set_goal_state(np.asarray(data["state"][source_row + 25], dtype=np.float64))
                for trace in traces:
                    np.testing.assert_allclose(snapshot(actor), trace[seed][0]["before"], atol=1e-4, rtol=0)
                data_rgb = np.asarray(data["pixels"][source_row])
                render_rgb = native_world_pixels(actor.render())
                with torch.inference_mode():
                    data_pred = rollout(model, transform(data_rgb).to("cuda"), actions)
                    render_pred = rollout(model, transform(render_rgb).to("cuda"), actions)
                    np.testing.assert_allclose(data_pred[:, 1:].cpu().numpy(), prior_pred[index],
                                               atol=1e-4, rtol=1e-4)
                    initial = render_pred[:, 0]
                    observed = torch.from_numpy(real[index]).to("cuda")
                    teacher = torch.stack([teacher_predictions(model, torch.cat([initial[a:a+1], observed[a]], 0),
                                                               actions[a]) for a in range(2)])
                    torch.testing.assert_close(teacher[:, 0], render_pred[:, 1], atol=1e-5, rtol=1e-5)
                all_initial.append(initial.cpu().numpy())
                all_rendered.append(render_pred[:, 1:].cpu().numpy())
                all_teacher.append(teacher.cpu().numpy())
                rows.append({"index": index, "data_row": source_row,
                             "same_chain": row["same_chain"],
                             "rgb_mean_absolute_pixel_difference": float(np.abs(
                                 data_rgb.astype(np.float32) - render_rgb.astype(np.float32)).mean())})
                print(f"compared {index + 1}/50", flush=True)
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    initial, rendered, teacher = map(np.stack, (all_initial, all_rendered, all_teacher))
    with OUT.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, initial_rendered=initial, rendered_rollout=rendered, teacher=teacher)
    sources = [Path(__file__), PROTOCOL, prior_json, prior_npz]
    output = {"rows": rows, "model_sha256": fingerprint, "data": str(previous.DATA),
              "prior_json_sha256": file_hash(prior_json), "arrays_sha256": file_hash(OUT.with_suffix(".npz")),
              "source_sha256": {str(p): file_hash(p) for p in sources},
              "scope": "fixed exposed-development plans; teacher observations are evaluation-only"}
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
