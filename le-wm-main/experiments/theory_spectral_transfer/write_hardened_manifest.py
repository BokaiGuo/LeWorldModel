"""Write a hash manifest for the hardened replay bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(path for path in args.bundle.rglob("*") if path.is_file())
    manifest = {
        "bundle": str(args.bundle),
        "file_count": len(files),
        "files": [
            {"path": str(path.relative_to(args.bundle)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "evidence_boundary": "Hash manifest for reproducibility; it does not add scientific evidence beyond the listed artifacts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
