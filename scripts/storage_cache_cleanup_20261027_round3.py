"""Explicit cache-only cleanup; default dry-run, identity revalidation on apply."""
import argparse
import gzip
import json
import os
from pathlib import Path
import stat
import time

BASE = Path('/home/gbk/code/LeWorldModel')
ROOTS = [Path('/home/gbk/.npm/_npx') / name for name in (
    '07e67900497e7fc9', '178212e99ffbb662', '1e7f6d9597241db0',
    '5dbd0d84e5fd82ec', '75ebcda362ef7f6c', 'e6e0ed1aca658cae')]
MANIFEST = BASE / 'docs/STORAGE_ROUND3_MANIFEST_20261027.json'
RECEIPT = BASE / 'docs/STORAGE_ROUND3_RECEIPT_20261027.json'


def identity(path):
    s = path.lstat()
    return dict(dev=s.st_dev, ino=s.st_ino, mode=s.st_mode, size=s.st_size,
                mtime_ns=s.st_mtime_ns, nlink=s.st_nlink, blocks=s.st_blocks * 512)


def check_live():
    """Check accessible processes; disclose protected OS service exceptions."""
    refs = []
    protected = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            command = proc.joinpath('cmdline').read_bytes()
            blobs = [command]
            try:
                blobs.append(proc.joinpath('maps').read_bytes())
                for item in [proc / 'cwd', proc / 'exe', *proc.joinpath('fd').iterdir()]:
                    try:
                        blobs.append(os.readlink(item).encode())
                    except FileNotFoundError:
                        pass
            except PermissionError:
                # These are OS service processes, not npm/node executors. Do not
                # extend this exception to arbitrary inaccessible applications.
                assert (command == b'/usr/lib/systemd/systemd\x00--user\x00'
                        or command.rstrip(b'\x00 ') == b'(sd-pam)'
                        or command.startswith(b'/usr/sbin/tailscaled\x00be-child\x00ssh\x00'))
                protected.append(dict(pid=proc.name, executable=command.split(b'\x00')[0].decode(),
                                      limitation='cmdline checked; maps/fds unavailable'))
            for root in ROOTS:
                if any(str(root).encode() in blob for blob in blobs):
                    refs.append([proc.name, str(root)])
        except (FileNotFoundError, ProcessLookupError):
            continue
    if refs:
        raise RuntimeError(f'Live references: {refs}')
    return protected


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def plan():
    protected = check_live()
    files = []
    dirs = []
    cutoff = time.time() - 86400
    for root in ROOTS:
        assert not root.is_symlink() and root.resolve() == root
        assert root.is_dir()
        for folder, subdirs, names in os.walk(root, followlinks=False):
            for name in list(subdirs):
                p = Path(folder) / name
                if p.is_symlink():
                    names.append(name)
                    subdirs.remove(name)
            dirs.append(str(folder))
            for name in names:
                p = Path(folder) / name
                meta = identity(p)
                assert stat.S_ISREG(meta['mode']) or stat.S_ISLNK(meta['mode'])
                assert meta['mtime_ns'] / 1e9 < cutoff, f'Recent cache: {p}'
                files.append(dict(path=str(p), **meta))
    data = dict(roots=list(map(str, ROOTS)), files=files, dirs=dirs,
                exclusive_blocks=sum(f['blocks'] for f in files if f['nlink'] == 1),
                logical_bytes=sum(f['size'] for f in files),
                rationale='Inactive npx disposable package installs; exact names/versions recoverable via npm.')
    save(MANIFEST, data)
    print(json.dumps({k: v for k, v in data.items() if k not in ('files', 'dirs')}))
    print('files', len(files))


def apply():
    assert not RECEIPT.exists()
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
    else:
        with gzip.open(str(MANIFEST) + '.gz', 'rt') as stream:
            data = json.load(stream)
    assert data['roots'] == list(map(str, ROOTS))
    protected = check_live()
    for entry in data['files']:
        p = Path(entry['path'])
        root = next(r for r in ROOTS if p.is_relative_to(r))
        assert p.parent.resolve().is_relative_to(root)
        assert identity(p) == {k: v for k, v in entry.items() if k != 'path'}
    receipt = dict(deleted_files=0, exclusive_blocks=0, removed_dirs=0, errors=[],
                   protected_system_services=protected)
    try:
        for entry in data['files']:
            p = Path(entry['path'])
            assert identity(p) == {k: v for k, v in entry.items() if k != 'path'}
            p.unlink()
            receipt['deleted_files'] += 1
            if entry['nlink'] == 1:
                receipt['exclusive_blocks'] += entry['blocks']
        for folder in sorted(data['dirs'], key=lambda p: len(Path(p).parts), reverse=True):
            try:
                Path(folder).rmdir()
                receipt['removed_dirs'] += 1
            except OSError as e:
                receipt['errors'].append(f'{folder}: {e}')
    finally:
        save(RECEIPT, receipt)
        print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    apply() if args.apply else plan()
