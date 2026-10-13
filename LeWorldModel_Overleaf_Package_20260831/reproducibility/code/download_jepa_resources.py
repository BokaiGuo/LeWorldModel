"""Download official JEPA resources used by latent-state spectral-transfer anchors.

The script intentionally keeps acquisition separate from experiments.  Its
manifest records exactly which official resources are present locally, so paper
experiments can distinguish downloaded checkpoints from missing feature caches
or unavailable task datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


JEPA_WMS_REPO = "facebook/jepa-wms"
JEPA_WMS_FILES = ("README.md", "jepa_wm_pusht.pth.tar")
IJEPA_REPO = "facebook/ijepa_vith16_1k"
IJEPA_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
GITHUB_REPO = "https://github.com/facebookresearch/jepa-wms.git"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_hf_file(repo_id: str, filename: str, out_dir: Path) -> dict[str, object]:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
    )
    return {
        "source": "huggingface",
        "repo_id": repo_id,
        "filename": filename,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def ensure_git_repo(url: str, target: Path) -> dict[str, object]:
    if target.exists():
        commit = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(target), "status", "--short"],
            text=True,
        ).strip()
        return {
            "source": "git",
            "url": url,
            "path": str(target),
            "commit": commit,
            "status": status,
            "action": "reused",
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", url, str(target)])
    commit = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return {
        "source": "git",
        "url": url,
        "path": str(target),
        "commit": commit,
        "status": "",
        "action": "cloned",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("external"),
        help="Directory in which JEPA resources are stored.",
    )
    parser.add_argument(
        "--include-ijepa-image",
        action="store_true",
        help="Also download the official I-JEPA ViT-H/16 image checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    records: list[dict[str, object]] = []

    records.append(ensure_git_repo(GITHUB_REPO, root / "jepa-wms"))
    for filename in JEPA_WMS_FILES:
        records.append(download_hf_file(JEPA_WMS_REPO, filename, root / "jepa_wms_hf"))

    if args.include_ijepa_image:
        for filename in IJEPA_FILES:
            records.append(download_hf_file(IJEPA_REPO, filename, root / "ijepa_vith16_1k_hf"))

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Latent State Spectral Transfer JEPA acquisition",
        "records": records,
    }
    path = root / "jepa_acquisition_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
