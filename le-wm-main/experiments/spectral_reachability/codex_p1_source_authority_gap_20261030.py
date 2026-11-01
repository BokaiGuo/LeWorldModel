"""Metadata-only gap inventory. No candidates, reset, model or authority approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CKPT = Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")
WEIGHTS = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
CONFIG = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
DOCS = {
    ROOT.parent / "docs/D2P_FRESHROOT_AUTHORITY_REQUEST_20261028_JASON.md":
        "46e9df59cb1f1d9bb344132a9a39a996c85cc26c5b79ef45c7b5d4dc4b3af39b",
    ROOT / "innerdocs/agent_p1_configuration_provenance_20261028_v1.md":
        "9d02dbea7c1de557b3fb78937151577343e2032f6294c66e5ee340b89bdf36a5",
    ROOT / "innerdocs/p1_checkpoint_historical_contract_lookup_20261027_v1.md":
        "f1ff81751babde6d514707b7d5414251e7e82da7354a2d9f08c90739b9978345",
    ROOT / "innerdocs/agent_p4_fresh_authority_gap_20261028_v1.md":
        "ad23b5e09428a58e88d317abe033f18b1edbb698572c8f07f40bc4792673abbd",
    ROOT / "config/train/data/pusht.yaml":
        "dcdd082ebdc628cdc8a88f15cb647fce6eec87343f9fe190271946ff5d52f757",
}
CALLS = {
    "A": ["call_981f76dba2b241d79588468b36e979d7"],
    "B": ["call_a41c0e8d73624b16959e29b9f05fd4ed",
          "call_3d96838206a84aba9b97fe0ce82bf8e4", "call_4b779dfb4ad045b696db7409d861c35f"],
    "C": ["call_b0f211dffc4e451fab05b0827fdfa415"],
    "D": ["call_871afe66f51744c3a8964fe49fab6b08"],
    "E": ["call_7c4638e67c234a65ad8b02be380375cb"],
    "F": ["call_e0f5ab66af7848a89e59e21a975c4165"],
    "G": ["call_a0928b48326a4f43b88a25a223834196"],
    "H": ["call_947bf4c5b1ce4004a5ab5c4e2ef3d0e5"],
}
TRAIN_FIELDS = ("checkpoint_run_binding", "training_code_revision", "resolved_config",
                "dataset_revision_and_content", "episode_split_and_filters",
                "conversion_and_units", "normalizer_run_binding")
SCOPE_FIELDS = ("machines_worktrees_cutoff", "failures_unselected_restarts",
                "post_cutoff_delta", "identity_lineage", "independent_origin_review")


def digest(path, cap):
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError("noncanonical evidence")
    before = path.stat()
    if before.st_size > cap:
        raise ValueError("evidence size cap")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("evidence changed during read")
    return h.hexdigest()


def validate_receipt(receipt):
    """Inventory assertions only. Never dereferences evidence or grants authority."""
    if type(receipt) is not dict or set(receipt) != {"schema", "checkpoint_sha256", "training", "calls", "scope"}:
        raise ValueError("exact receipt fields required")
    if receipt["schema"] != "codex_p1_source_authority_gap_20261030.receipt.v1" or receipt["checkpoint_sha256"] != WEIGHTS:
        raise ValueError("receipt schema/checkpoint mismatch")
    expected_calls = {call for group in CALLS.values() for call in group}
    for name, expected in (("training", set(TRAIN_FIELDS)), ("calls", expected_calls),
                           ("scope", set(SCOPE_FIELDS))):
        entries = receipt[name]
        if type(entries) is not dict or not set(entries) <= expected:
            raise ValueError("unknown fields prohibited: " + name)
        for key, value in entries.items():
            if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
                raise ValueError("evidence digest required: " + key)
    return {
        "training_missing": sorted(set(TRAIN_FIELDS) - set(receipt["training"])),
        "calls_missing": {source: [c for c in calls if c not in receipt["calls"]]
                          for source, calls in CALLS.items()},
        "scope_missing": sorted(set(SCOPE_FIELDS) - set(receipt["scope"])),
        "submitted_hashes_are_unverified_claims": True,
        "source_origin_review_required": True,
        "history_complete": False, "training_lineage_verified": False,
        "source_run_authorized": False, "freshness_certified": False, "p1_admitted": False,
    }


def empty_receipt():
    return {"schema": "codex_p1_source_authority_gap_20261030.receipt.v1",
            "checkpoint_sha256": WEIGHTS, "training": {}, "calls": {}, "scope": {}}


def snapshot():
    for path, expected in DOCS.items():
        if digest(path, 2 * 1024 * 1024) != expected:
            raise ValueError("fixed source document changed")
    call_doc = (ROOT / "innerdocs/agent_p4_fresh_authority_gap_20261028_v1.md").read_text()
    if any(call_doc.count(call) != 1 for calls in CALLS.values() for call in calls):
        raise ValueError("historical call IDs do not match pinned document")
    if digest(CKPT / "weights.pt", 100 * 1024 * 1024) != WEIGHTS:
        raise ValueError("checkpoint weights mismatch")
    if digest(CKPT / "config.json", 2 * 1024 * 1024) != CONFIG:
        raise ValueError("checkpoint config mismatch")
    # Only list this explicit checkpoint directory; no hidden cache traversal or candidate discovery.
    files = sorted(p.name for p in CKPT.iterdir() if p.is_file() and not p.is_symlink())
    return {
        "schema": "codex_p1_source_authority_gap_20261030.snapshot.v1",
        "checkpoint_weights_sha256": WEIGHTS, "checkpoint_config_sha256": CONFIG,
        "checkpoint_directory_files": files,
        "reviewed_source_documents": {str(p): h for p, h in DOCS.items()},
        "known_historical_source_count": 8, "known_historical_reference_count": 16,
        "known_dispatch_count": sum(map(len, CALLS.values())),
        "inventory_assessment": validate_receipt(empty_receipt()),
        "qualification": "No receipt was supplied; empty inventory is not proof that evidence does not exist.",
    }


def parse_receipt(raw):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out
    if len(raw) > 128 * 1024:
        raise ValueError("receipt too large")
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--receipt-sha256")
    args = p.parse_args()
    if (args.receipt is None) != (args.receipt_sha256 is None):
        p.error("receipt and receipt-sha256 must be supplied together")
    result = snapshot()
    if args.receipt is not None:
        path = args.receipt
        if (not path.is_absolute() or path.parent != ROOT / "results" or
                not path.name.startswith("codex_p1_source_authority_receipt_") or
                path.suffix != ".json" or path.resolve() != path):
            raise ValueError("receipt must be an explicit own-prefix results file")
        if digest(path, 128 * 1024) != args.receipt_sha256:
            raise ValueError("receipt digest mismatch")
        raw_receipt = path.read_bytes()
        if hashlib.sha256(raw_receipt).hexdigest() != args.receipt_sha256:
            raise ValueError("receipt changed during read")
        result["inventory_assessment"] = validate_receipt(parse_receipt(raw_receipt))
        result["receipt_sha256"] = args.receipt_sha256
        result["qualification"] = "Receipt hashes are declarations only; no referenced evidence was read or verified."
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"output": str(args.output.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
