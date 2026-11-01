#!/usr/bin/env python3
"""Experimental host Landlock wrapper. Not an approved production boundary.

Configuration is an operator-reviewed JSON manifest, not a security authority.
This program does not create a network/IPC sandbox or ensure GPU exclusivity.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys

LIBC = ctypes.CDLL(None, use_errno=True)
CREATE, ADD, RESTRICT = 444, 445, 446
EXECUTE, WRITE, READ, LIST = (1 << i for i in range(4))
MAKE_DIR, MAKE_REG = 1 << 7, 1 << 8
HANDLED = (1 << 15) - 1  # ABI 3: includes REFER and TRUNCATE
TRUNCATE = 1 << 14
DRIVER = "experiments.spectral_reachability.codex_p1_blind_h2_host_cuda_20261030_v1"
NAME = "codex_p1_blind_h2_host_cuda_20261030_v1"


class Ruleset(ctypes.Structure):
    _fields_ = [("fs", ctypes.c_uint64)]


class Rule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("rights", ctypes.c_uint64), ("fd", ctypes.c_int)]


def syscall(number, *args):
    result = LIBC.syscall(number, *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def apply_rules(rules):
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("unsupported syscall architecture")
    if syscall(CREATE, None, 0, 1) < 3:
        raise RuntimeError("Landlock ABI 3 or newer required")
    attr = Ruleset(HANDLED)
    ruleset = syscall(CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    try:
        for path, rights in rules:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                rule = Rule(rights, fd)
                syscall(ADD, ruleset, 1, ctypes.byref(rule), 0)
            finally:
                os.close(fd)
        if LIBC.prctl(38, 1, 0, 0, 0):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        syscall(RESTRICT, ruleset, 0)
    finally:
        os.close(ruleset)


def real_path(value, kind):
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("path must be absolute, canonical and symlink-free: " + value)
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if kind == "dir" else stat.S_ISREG(mode)):
        raise ValueError("wrong path type: " + value)
    return path


def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_policy(config):
    required = {"python", "runtime_dirs", "source_dir", "bundle", "bundle_sha256",
                "model_dir", "model_sha256", "output_dir", "deny_paths", "cuda_device"}
    if set(config) != required:
        raise ValueError("unexpected or missing manifest fields")
    runtimes = [real_path(p, "dir") for p in config["runtime_dirs"]]
    python = Path(config["python"])
    if (not python.is_absolute() or python.parent.resolve(strict=True) != python.parent
            or not python.is_file()):
        raise ValueError("python must have a canonical parent and resolve to a file")
    python_target = python.resolve(strict=True)
    source = real_path(config["source_dir"], "dir")
    bundle = real_path(config["bundle"], "file")
    model = real_path(config["model_dir"], "dir")
    output = real_path(config["output_dir"], "dir")
    if not runtimes or not config["deny_paths"]:
        raise ValueError("runtime and denied-path inventories are required")
    denies = [real_path(p, "dir") for p in config["deny_paths"]]
    roots = runtimes + [source, output]
    if any(overlaps(root, denied) for root in roots for denied in denies):
        raise ValueError("directory allowlist overlaps a denied tree")
    if any(overlaps(output, p) for p in runtimes + [source, model, bundle]):
        raise ValueError("output must be separate from inputs and runtime")
    if not all(any(root in path.parents for root in runtimes)
               for path in (python, python_target)):
        raise ValueError("python path and resolved executable must be in runtime directories")
    if any(overlaps(model, root) or overlaps(bundle, root) for root in runtimes + [source]):
        raise ValueError("runtime/source must not contain bundle or model")
    if any(overlaps(model, denied) for denied in denies) or any(
            bundle == denied or bundle in denied.parents for denied in denies):
        raise ValueError("model or broad bundle overlaps a denied tree")
    if any(output.iterdir()):
        raise ValueError("output directory must be empty and exclusive")
    if set(config["model_sha256"]) != {"weights.pt", "config.json"}:
        raise ValueError("exact model files required")
    files = [(bundle, config["bundle_sha256"])]
    files += [(real_path(str(model / name), "file"), expected)
              for name, expected in config["model_sha256"].items()]
    for path, expected in files:
        if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
            raise ValueError("input hash mismatch: " + str(path))
    device = config["cuda_device"]
    if not isinstance(device, str) or not device or device == "-1" or "," in device:
        raise ValueError("one explicitly coordinated CUDA device is required")
    rules = [(str(p), READ | LIST | EXECUTE) for p in runtimes + [source]]
    rules += [(str(model), LIST)]
    rules += [(str(path), READ) for path, _ in files]
    rules += [(str(output), READ | LIST | WRITE | TRUNCATE | MAKE_DIR | MAKE_REG)]
    return rules


def invoke(command, cwd, env, rules):
    # subprocess closes all inherited fds >2; stdin is /dev/null and stdout/stderr
    # are fresh pipes. Rules are installed after fd setup and before exec.
    return subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          close_fds=True, text=True,
                          preexec_fn=lambda: apply_rules(rules))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    try:
        if args.acknowledge != "EXPLORATORY_HOST_CUDA_NOT_PRODUCTION_APPROVED":
            raise ValueError("explicit exploratory acknowledgement required")
        manifest_bytes = Path(args.manifest).read_bytes()
        config = json.loads(manifest_bytes)
        rules = compile_policy(config)
        env = {"PATH": "/usr/bin:/bin", "LANG": "C", "HOME": config["output_dir"],
               "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "CUDA_VISIBLE_DEVICES": config["cuda_device"]}
        command = [config["python"], "-m", DRIVER, "--bundle", config["bundle"],
                   "--sha256", config["bundle_sha256"], "--model-dir", config["model_dir"],
                   "--output", str(Path(config["output_dir"]) / (NAME + "_attempt")),
                   "--acknowledge", "ISOLATED_HOST_CUDA_COORDINATED"]
        print(json.dumps({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                          "rules": rules, "production_authorized": False,
                          "network_isolated": False}), flush=True)
        result = invoke(command, config["source_dir"], env, rules)
        print(json.dumps({"child_exit": result.returncode, "stdout": result.stdout,
                          "stderr": result.stderr, "production_authorized": False}))
        return result.returncode if result.returncode > 0 else (1 if result.returncode else 0)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc),
                          "production_authorized": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
