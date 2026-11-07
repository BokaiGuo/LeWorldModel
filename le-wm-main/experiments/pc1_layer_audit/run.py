"""Resume the bounded E1-E6 audit without overwriting completed evidence."""
import json
from pathlib import Path

from experiments.pc_input_permissions import file_hash


def verified(path):
    data = json.loads(path.read_text())
    for source, expected in data.get("sources", {}).items():
        if file_hash(source) != expected:
            raise ValueError(f"source drift: {source}")
    print(f"verified: {path.name}", flush=True)
    return data


def main():
    from experiments.pc1_layer_audit import (
        pusht, camera, camera_runtime_bound, camera_singlebatch_analysis,
        render_alignment_comparison, summarize, figures,
    )
    if (pusht.OUT / "result.json").exists():
        data = verified(pusht.OUT / "result.json")
        if file_hash(pusht.OUT / "diagnostics.npz") != data["arrays_sha256"]:
            raise ValueError("PushT arrays changed")
    else:
        pusht.main()
    # All stages must run under one GPU-visible rendering environment.
    for stage in ("prepare", "predict", "replay"):
        camera.OUT = camera_runtime_bound.OUT
        if (camera.OUT / f"{stage}.json").exists():
            camera.read_stage(stage)
        else:
            camera_runtime_bound.main(stage)
    if (camera.OUT / "analyze.json").exists():
        camera.read_stage("analyze")
    else:
        camera_singlebatch_analysis.main()
    if (render_alignment_comparison.OUT / "result.json").exists():
        verified(render_alignment_comparison.OUT / "result.json")
    else:
        # A preceding runtime configuration must not substitute the repaired
        # worker for the intentionally cached-render counterfactual.
        camera.WORKER = Path(camera.__file__).with_name("camera_worker.py")
        render_alignment_comparison.main()
    if (summarize.OUT / "summary.json").exists():
        verified(summarize.OUT / "summary.json")
    else:
        summarize.main()
    if (figures.FIGURES / "manifest.json").exists():
        manifest = verified(figures.FIGURES / "manifest.json")
        for path, expected in manifest["artifacts"].items():
            if file_hash(path) != expected:
                raise ValueError(f"figure changed: {path}")
    else:
        figures.main()
    print("Bounded audit complete. This is not PC0-PC5 or online planning completion.", flush=True)


if __name__ == "__main__":
    main()
