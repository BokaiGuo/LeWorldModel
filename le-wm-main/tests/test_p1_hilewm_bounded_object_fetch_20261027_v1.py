"""Small offline fixtures only: no sockets, pickle execution or neural imports."""
import hashlib
import io
import json
from pathlib import Path
import struct
from unittest.mock import patch
import zipfile
import zlib

import pytest

from experiments.spectral_reachability import p1_hilewm_bounded_object_fetch_20261027_v1 as m


class Log:
    def __init__(self): self.rows = []
    def add(self, event, **fields): self.rows.append((event, fields))


class Response:
    def __init__(self, body, start, length, status=206, headers=None, url=m.URL):
        self.body = io.BytesIO(body)
        self.status, self.url = status, url
        self.headers = {'Content-Range': f'bytes {start}-{start+length-1}/{m.ARCHIVE_BYTES}',
                        'Content-Length': str(length)}
        self.headers.update(headers or {})
        self.reads = 0
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def geturl(self): return self.url
    def read(self, length):
        self.reads += 1
        return self.body.read(length)


class Single:
    def __init__(self, response): self.response = response
    def open(self, request, timeout):
        assert request.full_url == m.URL
        return self.response


def fixture(raw=b'hello', name='package/test.ckpt', flags=0):
    encoder = zlib.compressobj(wbits=-15)
    packed = encoder.compress(raw) + encoder.flush()
    spec = m.Spec(name, 100, len(packed), len(raw), zlib.crc32(raw)&0xffffffff, flags, 8)
    encoded = name.encode()
    header = struct.pack('<4s5H3I2H', b'PK\x03\x04',20,flags,8,0,0,
                         spec.crc if not flags&8 else 0,
                         spec.packed if not flags&8 else 0,
                         spec.expanded if not flags&8 else 0,len(encoded),0)
    return spec, header, encoded, packed


def test_real_frozen_directory_binding():
    project = Path(__file__).resolve().parents[1]
    spec, provenance = m.binding(project)
    assert spec.packed == 113104429 and spec.expanded == 122424517
    assert spec.crc == 0xd59f865b and provenance['directory_sha256'] == m.DIRECTORY_SHA


@pytest.mark.parametrize('name', ['/x','../x','package/../x','package//x','package/./x',
                                  'package/a\\b','package/\x00x'])
def test_traversal_rejected(name):
    with pytest.raises(ValueError): m.safe_member(name)


@pytest.mark.parametrize('status,headers,url', [
    (200,{},m.URL), (302,{},m.URL), (403,{},m.URL),
    (206,{'Content-Range':'bytes 1-2/999'},m.URL),
    (206,{'Content-Length':'99'},m.URL),
    (206,{'Content-Encoding':'gzip'},m.URL),
    (206,{},'https://evil.invalid/object'),
])
def test_http_contract_rejected_before_read(status, headers, url):
    response = Response(b'ab',0,2,status,headers,url)
    reader = m.RangeReader(Log(),Single(response))
    with pytest.raises(ValueError): reader.read(0,2,lambda x: None)
    assert response.reads == 0 and reader.received == 0
    assert reader.attempted == 1 and reader.completed == 0


def test_network_cap_before_open():
    reader = m.RangeReader(Log(),Single(None))
    reader.reserved = m.DOWNLOAD_CAP
    with pytest.raises(ValueError,match='network cap'): reader.small(0,1)
    assert reader.attempted == 0


def test_truncation_and_partial_counters():
    log=Log(); reader=m.RangeReader(log,Single(Response(b'a',0,2)))
    with pytest.raises(ValueError,match='Truncated'): reader.read(0,2,lambda x: None)
    assert reader.received == 1 and reader.completed == 0
    assert log.rows[-1][0] == 'range_partial'


def test_consumer_failure_not_masked_by_journal_error():
    class BadLog(Log):
        def add(self,event,**kwargs):
            if event=='range_partial': raise OSError('journal failure')
            super().add(event,**kwargs)
    reader=m.RangeReader(BadLog(),Single(Response(b'ab',0,2)))
    def consume(x): raise RuntimeError('original consumer error')
    with pytest.raises(RuntimeError,match='original consumer'): reader.read(0,2,consume)
    assert reader.received==2 and reader.journal_errors


def test_validator_change_rejected():
    reader=m.RangeReader(Log(),Single(Response(b'a',0,1,headers={'ETag':'b'})))
    reader.validators['etag']='a'
    # Production header mapping is case-insensitive HTTPMessage; fixture lowercases below.
    reader.opener.response.headers={'Content-Range':f'bytes 0-0/{m.ARCHIVE_BYTES}',
                                    'Content-Length':'1','etag':'b'}
    with pytest.raises(ValueError,match='validator'): reader.read(0,1,lambda x:None)


def test_inflate_stream_crc_sha_and_no_overexpansion(tmp_path):
    spec,_,_,packed=fixture(b'hello'*1000)
    file=io.BytesIO()
    sink=m.InflateSink(file,spec,tmp_path,check_space=lambda *x:None)
    for k in range(0,len(packed),3): sink.feed(packed[k:k+3])
    result=sink.finish()
    assert file.getvalue()==b'hello'*1000
    assert result['object_sha256']==hashlib.sha256(file.getvalue()).hexdigest()
    wrong=m.Spec(spec.name,spec.offset,spec.packed,5,spec.crc,0,8)
    bad=m.InflateSink(io.BytesIO(),wrong,tmp_path,check_space=lambda *x:None)
    with pytest.raises(ValueError,match='Expansion cap'): bad.feed(packed)


def test_inflate_rejects_crc_trailing_and_truncated(tmp_path):
    spec,_,_,packed=fixture()
    for mode in ('crc','truncated','trailing'):
        other=m.Spec(spec.name,spec.offset,spec.packed+(1 if mode=='trailing' else 0),
                     spec.expanded,spec.crc+(1 if mode=='crc' else 0),0,8)
        sink=m.InflateSink(io.BytesIO(),other,tmp_path,check_space=lambda *x:None)
        with pytest.raises(ValueError):
            sink.feed(packed[:-1] if mode=='truncated' else packed+(b'x' if mode=='trailing' else b''))
            sink.finish()


def test_disk_floor(tmp_path):
    with patch.object(m.shutil,'disk_usage',return_value=type('D',(),{'free':m.DISK_FLOOR+3})()):
        m.space(tmp_path,3)
        with pytest.raises(OSError): m.space(tmp_path,4)


def test_static_globals_no_execution_and_legacy(tmp_path):
    p=tmp_path/'mock.ckpt'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('archive/data.pkl',b'\x80\x02chi_jepa\nHiJEPA\n.')
        z.writestr('archive/data/0',b'tensor bytes not read')
    report=m.static_inventory(p)
    assert report['pickles'][0]['globals_referenced']==['hi_jepa HiJEPA']
    assert not report['tensor_storage_read'] and not report['unpickled']
    p2=tmp_path/'legacy.ckpt'; p2.write_bytes(b'\x80\x02chi_module\nLatentActionEncoder\n.')
    assert m.static_inventory(p2)['pickles'][0]['globals_referenced']==['hi_module LatentActionEncoder']


def test_stack_global_conservative():
    r=m.pickle_inventory(b'\x80\x04\x8c\x07hi_jepa\x8c\x06HiJEPA\x93.')
    assert r['globals_referenced']==['hi_jepa HiJEPA']
    r=m.pickle_inventory(b'\x80\x04NN\x93.')
    assert r['unresolved_stack_globals']==1 and not r['class_list_complete']


def test_static_traversal_refused(tmp_path):
    p=tmp_path/'bad.ckpt'
    with zipfile.ZipFile(p,'w') as z: z.writestr('../data.pkl',b'N.')
    with pytest.raises(ValueError): m.static_inventory(p)


def test_local_header_and_descriptor():
    for flags in (0,8):
        spec,head,name,packed=fixture(flags=flags)
        class R:
            def small(self,start,length): return head if length==30 else name
        assert m.local_header(spec,R())==spec.offset+30+len(name)


def fetch_fixture(tmp_path, fail=False):
    artifact=io.BytesIO()
    with zipfile.ZipFile(artifact,'w') as z:
        z.writestr('archive/data.pkl',b'\x80\x02chi_jepa\nHiJEPA\n.')
    spec,header,name,packed=fixture(artifact.getvalue())
    image=b'\x00'*spec.offset+header+name+packed
    tail=b'x'*65557
    class Opener:
        def open(self,request,timeout):
            bounds=request.get_header('Range').split('=')[1]
            start,end=map(int,bounds.split('-')); length=end-start+1
            if start==2299798367: body=tail
            else: body=image[start:end+1]
            if fail and start==spec.offset+30+len(name): body=body[:3]
            return Response(body,start,length)
    project=tmp_path/'project'; (project/'results').mkdir(parents=True)
    out=project/'results'/'exclusive'
    return spec,Opener(),tail,project,out


def test_full_mock_fetch_publishes_only_verified_and_no_overwrite(tmp_path):
    spec,opener,tail,project,out=fetch_fixture(tmp_path)
    with patch.object(m,'binding',return_value=(spec,{})), patch.object(m,'TAIL_SHA',m.digest(tail)):
        r=m.fetch_object(project,out,opener)
        assert r['status']=='object_verified_static_recorded_not_loaded'
        assert (out/r['artifact']).stat().st_ino==(out/'download.part').stat().st_ino
        with pytest.raises(FileExistsError): m.fetch_object(project,out,opener)
    previous='0'*64
    for line in (out/'events.jsonl').read_text().splitlines():
        event=json.loads(line); raw=json.dumps(event['record'],sort_keys=True,separators=(',',':')).encode()
        assert event['record']['previous_sha256']==previous
        assert m.digest(raw)==event['sha256']; previous=event['sha256']


def test_failed_mock_fetch_preserves_partial_and_report(tmp_path):
    spec,opener,tail,project,out=fetch_fixture(tmp_path,fail=True)
    with patch.object(m,'binding',return_value=(spec,{})), patch.object(m,'TAIL_SHA',m.digest(tail)):
        with pytest.raises(ValueError,match='Truncated'): m.fetch_object(project,out,opener)
    r=json.loads((out/'report.json').read_text())
    assert r['status']=='failed_preserved' and 'Truncated' in r['first_error']['message']
    assert (out/'download.part').exists()
    assert not (out/'pusht_hi_lewm_epoch15_object.ckpt').exists()


def test_final_source_failure_cannot_mask_execution_first_error(tmp_path):
    spec,opener,tail,project,out=fetch_fixture(tmp_path,fail=True)
    original=m.file_digest
    source_reads=0
    def disappearing(path):
        nonlocal source_reads
        if path==Path(m.__file__):
            source_reads+=1
            if source_reads>=2: raise FileNotFoundError('source disappeared')
        return original(path)
    with patch.object(m,'binding',return_value=(spec,{})), patch.object(m,'TAIL_SHA',m.digest(tail)), \
         patch.object(m,'file_digest',side_effect=disappearing):
        with pytest.raises(ValueError,match='Truncated'): m.fetch_object(project,out,opener)
    r=json.loads((out/'report.json').read_text())
    assert r['first_error']['type']=='ValueError'
    assert r['secondary_errors'][0]['phase']=='final_source_hash'


def test_preflight_floor_fails_before_output_or_network(tmp_path):
    spec,opener,tail,project,out=fetch_fixture(tmp_path)
    with patch.object(m,'binding',return_value=(spec,{})), \
         patch.object(m.shutil,'disk_usage',return_value=type('D',(),{'free':m.DISK_FLOOR})()):
        with pytest.raises(OSError): m.fetch_object(project,out,opener)
    assert not out.exists()
