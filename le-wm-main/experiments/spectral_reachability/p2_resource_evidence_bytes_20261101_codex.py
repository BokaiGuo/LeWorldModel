"""Check availability and SHA256 of P2 resource receipt evidence, CPU only.

Byte identity is not verification of evidence semantics, runtime coverage,
commitment time, or scientific admission. No dispatch or file writes.
"""
import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as gate

MAX_BYTES = 128 * 1024 * 1024


def evidence_digest(path):
    """Hash a bounded regular file without following a final-component symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        gate.require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES,
                     "bounded regular evidence required")
        digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            gate.require(count <= MAX_BYTES, "evidence too large")
            digest.update(chunk)
        after = os.fstat(fd)
        gate.require(count == before.st_size == after.st_size and
                     (before.st_mtime_ns, before.st_ctime_ns) ==
                     (after.st_mtime_ns, after.st_ctime_ns), "evidence changed during read")
        return digest.hexdigest()
    finally:
        os.close(fd)


def verify(contract, contract_sha, receipts, evidence):
    """Require exact digest-to-path coverage for every validated receipt axis."""
    comparison = gate.compare(contract, contract_sha, receipts)
    required = {entry["evidence_sha256"] for receipt in receipts
                for entry in receipt["axes"].values()}
    gate.require(type(evidence) is dict and set(evidence) == required,
                 "evidence mapping must exactly cover referenced digests")
    for digest in sorted(required):
        gate.require(evidence_digest(evidence[digest]) == digest,
                     "evidence SHA256 mismatch: " + digest)
    return {
        "schema": "p2_resource_evidence_bytes_20261101_codex.v1",
        "contract_sha256": contract_sha,
        "receipt_assertions": comparison,
        "distinct_evidence_files_verified": len(required),
        "evidence_bytes_verified": True,
        "evidence_semantics_verified": False,
        "runtime_evidence_verified": False,
        "preregistration_timing_verified": False,
        "P1_admission": False,
        "P2_admission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--receipt", nargs=2, action="append", required=True,
                        metavar=("PATH", "SHA256"))
    parser.add_argument("--evidence", nargs=2, action="append", required=True,
                        metavar=("SHA256", "PATH"))
    args = parser.parse_args()
    contract = gate.read_bound(args.contract, args.contract_sha256)
    gate.require(args.contract.read_bytes() == gate.canonical(contract),
                 "contract file must use canonical JSON bytes")
    receipts = [gate.read_bound(path, sha) for path, sha in args.receipt]
    evidence = {}
    for digest, path in args.evidence:
        gate.require(gate.hash_string(digest) and digest not in evidence,
                     "invalid or duplicate evidence digest")
        evidence[digest] = path
    print(json.dumps(verify(contract, args.contract_sha256, receipts, evidence),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
