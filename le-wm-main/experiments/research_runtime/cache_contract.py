"""CLI for the compatibility shell runner's input-lock contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import atomic_write_json
from .fingerprint import build_run_identity, fingerprint_inputs, git_identity


def build_contract(*, repo: Path, protocol_version: str, config: dict, inputs: list[Path]) -> dict:
    input_lock = fingerprint_inputs(inputs)
    code = git_identity(repo)
    identity = build_run_identity(
        protocol_version=protocol_version,
        config=config,
        input_lock=input_lock,
        code=code,
    )
    return {
        "artifact": "compatibility_run_contract",
        "protocol_version": protocol_version,
        "config": config,
        "input_lock": input_lock,
        "code": code,
        "run_identity": identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--input", type=Path, action="append", default=[])
    args = parser.parse_args()
    contract = build_contract(
        repo=args.repo.resolve(),
        protocol_version=args.protocol_version,
        config={"env": args.env, "policy": args.policy, "device": args.device},
        inputs=args.input,
    )
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old.get("run_identity", {}).get("sha256") != contract["run_identity"]["sha256"]:
            raise SystemExit(
                "input/code fingerprint changed; refusing to reuse compatibility cache: "
                f"{args.output}"
            )
        print(json.dumps({"status": "compatible", "identity": contract["run_identity"]["sha256"]}))
        return
    atomic_write_json(args.output, contract)
    print(json.dumps({"status": "created", "identity": contract["run_identity"]["sha256"]}))


if __name__ == "__main__":
    main()
