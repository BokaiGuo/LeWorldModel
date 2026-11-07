"""Repeat fixed history audit with the existing precise-derivative backend."""
import json
from pathlib import Path
from unittest.mock import patch

from experiments import pc_tworoom_latent_history_gradient_20261107_codex as audit
from experiments import pc_rgb_derivative_backend_20261107_codex as backend
from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/pc_tworoom_latent_history_precise_20261107_codex.json"
MANIFEST = ROOT / "results/pc_tworoom_latent_history_precise_20261107_codex_manifest.json"
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_LATENT_HISTORY_PRECISE_BACKEND_PROTOCOL_20261107_CODEX.md"


def main():
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("output exists")
    paths = [Path(__file__), Path(audit.__file__), Path(backend.__file__), PROTOCOL]
    hashes = {str(path): file_hash(path) for path in paths}
    with backend.precise_action_derivatives(), patch.object(audit, "OUTPUT", OUTPUT):
        audit.main()
    if hashes != {str(path): file_hash(path) for path in paths}:
        raise ValueError("source changed during run")
    result = json.loads(OUTPUT.read_text())
    with MANIFEST.open("x") as stream:
        json.dump({"sources": hashes, "result_sha256": file_hash(OUTPUT),
                   "all_pass": result["all_pass"], "backend": "math SDPA, TF32 disabled",
                   "scope": "post-hoc numerical diagnosis on exposed roots"}, stream, indent=2)


if __name__ == "__main__":
    main()
