"""RAM ZIP fixtures only. Not real archive/model/physics certification."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
from types import SimpleNamespace
import zipfile
from unittest.mock import patch

import pytest

from experiments.spectral_reachability import agent_hilewm_asset_verify_20261028_v1 as m

PROJECT = Path(__file__).resolve().parents[1]


def fixture(names=None, compression=zipfile.ZIP_DEFLATED):
    files = names or [('package/main.ckpt', b'opaque not pickle'), ('package/config.yaml', b'seed: 42\n')]
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=compression) as z:
        for name, data in files:
            z.writestr(name, data)
    raw = out.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = m.index(z.infolist())
    b = m.Binding(len(raw), hashlib.md5(raw).hexdigest(), len(raw), m.sha(raw), entries,
                  {'package/config.yaml': m.sha(b'seed: 42\n')} if names is None else {},
                  tuple(entries), {'fixture_only': True})
    return raw, b


def inspect(raw, binding):
    r = m.new_report(binding)
    m.inspect_stream(io.BytesIO(raw), binding, r, lambda: None)
    return r


def test_real_small_pinned_references_no_archive():
    b = m.load_binding(PROJECT)
    assert b.size == 2299863924 and len(b.entries) == 414
    assert len(b.texts) == 33 and m.CONFIG in b.texts
    assert b.entries[m.OBJECT]['offset'] == 634194328
    assert b.entries[m.OBJECT]['crc32'] == 0xd59f865b
    assert sum(x['size'] for x in b.entries.values()) == 2500619520


def test_complete_ram_zip_three_layers():
    raw, b = fixture()
    r = inspect(raw, b)
    assert r['layers'] == {'download_complete': True, 'zip_safely_readable': True, 'baseline_runnable': None}
    assert r['zip_summary']['source_texts_matched'] == 1
    assert r['members'][0]['sha256_measured'] == m.sha(b'opaque not pickle')
    assert r['baseline_missing_evidence'] and not r['unpickled'] and not r['extracted']
    assert r['scientific_admission'] is False


@pytest.mark.parametrize('name', ['', '/root', '../x', 'p/../x', 'p/./x', 'p//x', 'p\\x',
                                  'C:/x', 'p\x00x', 'p\nx', '//x', 'p/\x7fx'])
def test_unsafe_names(name):
    with pytest.raises(m.Invalid) as exc:
        m.safe_name(name)
    assert exc.value.code == 'unsafe_path'


@pytest.mark.parametrize('name', ['package/', 'package/a.ckpt', '__MACOSX/package/._x'])
def test_valid_names(name):
    m.safe_name(name)


def test_duplicate_and_NUL_truncation():
    a = zipfile.ZipInfo('a')
    a.CRC, a.header_offset = 0, 0
    with pytest.raises(m.Invalid) as e:
        m.index([a, a])
    assert e.value.code == 'duplicate_path'
    a.orig_filename = 'a\x00hidden'
    with pytest.raises(m.Invalid):
        m.index([a])


@pytest.mark.parametrize('kind', [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFSOCK])
def test_no_special_members(kind):
    i = zipfile.ZipInfo('package/x')
    i.external_attr = (kind | 0o644) << 16
    with pytest.raises(m.Invalid) as e:
        m.index([i])
    assert e.value.code == 'special_member'


@pytest.mark.parametrize('flags,method', [(1, 0), (64, 0), (0, zipfile.ZIP_BZIP2), (0, zipfile.ZIP_LZMA)])
def test_encryption_or_unsupported_method(flags, method):
    i = zipfile.ZipInfo('a')
    i.flag_bits, i.compress_type = flags, method
    with pytest.raises(m.Invalid):
        m.index([i])


@pytest.mark.parametrize('field,value,code', [('size', 1, 'archive_size_mismatch'),
                                           ('md5', '0'*32, 'archive_md5_mismatch'),
                                           ('tail_sha256', '0'*64, 'tail_mismatch')])
def test_archive_integrity_failures(field, value, code):
    raw, b = fixture()
    with pytest.raises(m.Invalid) as e:
        inspect(raw, replace(b, **{field: value}))
    assert e.value.code == code


def test_truncated_zip_and_config_mismatch():
    raw, b = fixture()
    with pytest.raises(m.Invalid):
        inspect(raw[:-1], b)
    with pytest.raises(m.Invalid) as e:
        inspect(raw, replace(b, texts={'package/config.yaml': '0'*64}))
    assert e.value.code == 'source_config_mismatch'


def test_exact_missing_extra_inventory():
    raw, b = fixture()
    entries = dict(b.entries)
    entries['package/absent.ckpt'] = entries.pop('package/main.ckpt')
    report = m.new_report(b)
    with pytest.raises(m.Invalid) as e:
        m.inspect_stream(io.BytesIO(raw), replace(b, entries=entries), report, lambda: None)
    assert e.value.code == 'inventory_mismatch'
    assert report['missing_members'] == ['package/absent.ckpt']


@pytest.mark.parametrize('field,value', [('offset', 1), ('size', 100), ('packed', 0), ('crc32', 0),
                                      ('flags', 8), ('external_attr', 0)])
def test_all_metadata_axes_bound(field, value):
    raw, b = fixture()
    entries = {k: dict(v) for k, v in b.entries.items()}
    entries['package/main.ckpt'][field] = value
    with pytest.raises(m.Invalid) as e:
        inspect(raw, replace(b, entries=entries))
    assert e.value.code == 'member_metadata_mismatch'


def test_crc_corruption_not_only_metadata():
    raw, b = fixture(compression=zipfile.ZIP_STORED)
    damaged = bytearray(raw)
    offset = b.entries['package/main.ckpt']['offset']
    namelen, extralen = struct.unpack_from('<HH', raw, offset + 26)
    damaged[offset + 30 + namelen + extralen] ^= 1
    # Deliberately update only fixture transport checks to reach member CRC branch.
    damaged = bytes(damaged)
    b = replace(b, md5=hashlib.md5(damaged).hexdigest(), tail_sha256=m.sha(damaged))
    with pytest.raises(zipfile.BadZipFile, match='CRC'):
        inspect(damaged, b)


def test_local_header_name_corruption():
    raw, b = fixture()
    damaged = bytearray(raw)
    damaged[30] ^= 1
    damaged = bytes(damaged)
    b = replace(b, md5=hashlib.md5(damaged).hexdigest(), tail_sha256=m.sha(damaged))
    with pytest.raises(zipfile.BadZipFile):
        inspect(damaged, b)


@pytest.mark.parametrize('suffix', ['.part', '.part.zip', '.tmp', '.partial', '.download'])
def test_active_partial_refused_before_any_candidate_open(suffix):
    _, b = fixture()
    with patch.object(m, 'load_binding', return_value=b), patch.object(m.os, 'open', side_effect=AssertionError('opened')):
        r = m.verify_archive(Path('/never/read/archive' + suffix), PROJECT, final_file=True)
    assert r['errors'][0]['code'] == 'active_partial_forbidden'
    assert r['layers']['download_complete'] is None


def test_final_ack_and_relative_path_refused_without_candidate_open():
    _, b = fixture()
    with patch.object(m, 'load_binding', return_value=b), patch.object(m.os, 'open', side_effect=AssertionError('opened')):
        r = m.verify_archive(Path('/never/read/final.zip'), PROJECT)
        s = m.verify_archive(Path('final.zip'), PROJECT, final_file=True)
    assert r['errors'][0]['code'] == 'final_file_ack_required'
    assert s['errors'][0]['code'] == 'absolute_path_required'


def test_wall_deadline():
    raw, b = fixture()
    calls = []
    def tick():
        calls.append(1)
        if len(calls) == 3:
            raise m.Invalid('wall_cap', 'fixture wall')
    with pytest.raises(m.Invalid) as e:
        m.inspect_stream(io.BytesIO(raw), b, m.new_report(b), tick)
    assert e.value.code == 'wall_cap'


def test_plan_cli_only_references(capsys):
    with patch.object(m.os, 'open', side_effect=AssertionError('candidate opened')):
        assert m.main(['plan', '--project', str(PROJECT)]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r['members'] == 414 and not r['archive_candidate_opened']


def test_missing_archive_cli_failclosed(capsys):
    assert m.main(['verify']) == 2
    assert json.loads(capsys.readouterr().out)['code'] == 'archive_required'


def failed_receipt(http=200, exitcode=18, count=16634096):
    return {'schema': 'MAIN.hilewm_whole_archive_result.v1', 'status': 'failed_preserved',
            'actual_saved_bytes': count, 'curl_exit_code': exitcode,
            'transfer': {'url_effective': m.URL, 'http_code': http, 'exitcode': exitcode,
                         'size_download': count, 'time_total': 276.626614,
                         'errormsg': f'transfer closed with {m.ARCHIVE_SIZE-count} bytes remaining to read',
                         'filename_effective': '/forbidden/archive.part'}}


def diagnose_ram(saved, expected=None):
    raw = json.dumps(saved).encode()
    with patch.object(Path, 'open', return_value=io.BytesIO(raw)), \
         patch.object(m, 'load_binding', return_value=m.Binding(m.ARCHIVE_SIZE, m.ARCHIVE_MD5, 0, '', {}, {}, (), {})):
        # source SHA read also uses Path.read_bytes/open; mock it separately.
        with patch.object(Path, 'read_bytes', return_value=b'source fixture'):
            return m.diagnose_download_receipt(Path('/saved/terminal.json'), expected or m.sha(raw), PROJECT)


def test_receipt_precise_EOF_not_403_no_payload():
    r = diagnose_ram(failed_receipt())
    assert r['download_diagnosis'] == 'http200_early_EOF_truncated_not_403'
    assert r['archive']['missing_bytes'] == 2283229828
    assert r['archive']['sha256_measured'] is None
    assert not r['payload_opened'] and not r['automatic_retry']
    assert r['layers'] == {'download_complete': False, 'zip_safely_readable': False, 'baseline_runnable': None}


def test_receipt_403_not_relabelled_EOF():
    assert diagnose_ram(failed_receipt(403, 22))['download_diagnosis'] == 'http403_forbidden_not_classified_as_EOF'


def test_receipt_unknown_is_unknown():
    assert diagnose_ram(failed_receipt(500, 22))['download_diagnosis'] == 'incomplete_transfer_cause_unknown'


@pytest.mark.parametrize('change,code', [({'actual_saved_bytes': 1}, 'receipt_inconsistent'),
                                       ({'schema': 'wrong'}, 'receipt_schema'),
                                       ({'status': 'running'}, 'receipt_schema')])
def test_receipt_mutations(change, code):
    saved = failed_receipt()
    saved.update(change)
    with pytest.raises(m.Invalid) as e:
        diagnose_ram(saved)
    assert e.value.code == code


def test_receipt_SHA_and_missing_count_strict():
    with pytest.raises(m.Invalid) as e:
        diagnose_ram(failed_receipt(), '0'*64)
    assert e.value.code == 'receipt_sha_mismatch'
    saved = failed_receipt()
    saved['transfer']['errormsg'] = 'transfer closed with 1 bytes remaining to read'
    with pytest.raises(m.Invalid) as e:
        diagnose_ram(saved)
    assert e.value.code == 'receipt_inconsistent'


@pytest.mark.parametrize('mutation', ['none', 'fd_changed', 'path_replaced', 'truncated'])
def test_file_wrapper_cleanup_and_stability_RAM(mutation):
    raw, b = fixture()
    if mutation == 'truncated':
        raw = raw[:-1]
    class RAMFile(io.BytesIO):
        def fileno(self):
            return 123
    f = RAMFile(raw)
    before = SimpleNamespace(st_dev=1, st_ino=2, st_size=len(raw), st_mtime_ns=3,
                             st_ctime_ns=4, st_mode=stat.S_IFREG | 0o444)
    after = SimpleNamespace(**vars(before))
    path_stat = SimpleNamespace(**vars(before))
    if mutation == 'fd_changed':
        after.st_mtime_ns += 1
    if mutation == 'path_replaced':
        path_stat.st_ino += 1
    with patch.object(m, 'load_binding', return_value=b), \
         patch.object(m.os, 'open', return_value=123) as opened, \
         patch.object(m.os, 'fdopen', return_value=f), \
         patch.object(m.os, 'fstat', side_effect=[before, after]), \
         patch.object(m.os, 'stat', return_value=path_stat):
        r = m.verify_archive(Path('/RAM/final.zip'), PROJECT, final_file=True)
    assert f.closed
    assert opened.call_args.args[1] & m.os.O_NOFOLLOW
    if mutation == 'none':
        assert r['status'] == 'archive_verified_baseline_unestablished'
    elif mutation == 'truncated':
        assert r['errors'][0]['code'] == 'archive_size_mismatch'
    else:
        assert r['errors'][0]['code'] == 'archive_changed'
        assert r['layers']['download_complete'] is None
        assert r['layers']['zip_safely_readable'] is False


def test_reference_changed_before_candidate_open():
    with patch.object(Path, 'read_bytes', return_value=b'changed reference'), \
         patch.object(m.os, 'open', side_effect=AssertionError('opened candidate')):
        with pytest.raises(m.Invalid) as e:
            m.load_binding(PROJECT)
    assert e.value.code == 'reference_changed'


def test_report_exclusive_no_overwrite(capsys):
    _, b = fixture()
    r = m.new_report(b)
    r['status'] = 'archive_verified_baseline_unestablished'
    with patch.object(m, 'verify_archive', return_value=r), \
         patch.object(Path, 'open', side_effect=FileExistsError('existing saved report')) as opened:
        assert m.main(['verify', '--archive', '/RAM/final.zip', '--output', '/RAM/existing.json']) == 2
    assert opened.call_args.args == ('x',)
    assert json.loads(capsys.readouterr().out)['code'] == 'FileExistsError'
