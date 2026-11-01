"""V2 exact-bound large-protocol reader; frozen v1 arithmetic unchanged."""
import argparse
from pathlib import Path
from experiments.spectral_reachability import p3p4_goal_patch_outcomes_20261027_v1 as v1
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3p4_goal_patch_outcomes_20261027_v2'
io,require,write=v1.io,v1.require,v1.write
PROJECT=v1.PROJECT
LARGE=PROJECT/v1.PREPARED/'protocol.json'
LARGE_SHA='f7b4d26e1e52c4823a2bea095f87aa5d95efc36882b908004217069a60d0e3ad'


class Evidence(io.Evidence):
    def raw(self,path,expected=None):
        path=Path(path)
        if path!=LARGE:return super().raw(path,expected)
        require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'exact regular large protocol path')
        require(path.is_file() and path.stat().st_size<=64*1024**2,'large protocol bounded64MiB')
        raw=path.read_bytes();h=io.sha(raw)
        require(len(raw)<=64*1024**2 and h==LARGE_SHA and (expected is None or expected==h),'exact sealed protocol SHA')
        require(str(path) not in self.hashes or self.hashes[str(path)]==h,'large protocol drift');self.hashes[str(path)]=h
        return raw


def prepare(output):
    output=Path(output).absolute();require(not output.exists(),'exclusive NEW v2 preparation')
    v1.physical.validate_prepared(PROJECT/v1.PREPARED,v1.PREPARE_SHA)
    sources=v1.physical.replay.metric.old.prior.physical.source_closure(PROJECT,[Path(__file__).resolve(),PROJECT/'tests'/('test_'+NAME+'.py'),PROJECT/'innerdocs'/(NAME+'.md')])
    p=dict(schema=NAME+'.protocol',contract=v1.contract(),large_protocol_path=str(LARGE),large_protocol_sha256=LARGE_SHA,
        physical_run=str(PROJECT/v1.RUN),sources_sha256={str(s):io.sha(s.read_bytes()) for s in sources})
    output.mkdir(exist_ok=False);write(output/'protocol.json',p)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(auditor=NAME)),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    receipt=dict(status='prepared_saved_data_only',protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seals=seals)
    write(output/'prepare.json',receipt);return receipt


def validate_prepared(prepared,sha):
    prepared=Path(prepared).absolute();require(io.sha((prepared/'prepare.json').read_bytes())==sha,'v2 prepare SHA');r=v1.physical.read_json(prepared/'prepare.json')
    require(io.sha((prepared/'protocol.json').read_bytes())==r['protocol_sha256'],'v2 protocol bytes');p=v1.physical.read_json(prepared/'protocol.json')
    require(p['schema']==NAME+'.protocol' and p['contract']==v1.contract() and p['large_protocol_sha256']==LARGE_SHA and p['large_protocol_path']==str(LARGE) and p['physical_run']==str(PROJECT/v1.RUN),'v2 contract')
    for path,h in p['sources_sha256'].items():require(io.sha(Path(path).read_bytes())==h,'v2 source drift')
    for s in r['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    return p


def run(prepared,sha,report_sha,output):
    p=validate_prepared(prepared,sha);output=Path(output).absolute();require(not output.exists(),'exclusive NEW v2 report')
    reader=Evidence(PROJECT);reader.raw(Path(prepared).absolute()/'prepare.json',sha);reader.raw(Path(prepared).absolute()/'protocol.json')
    result=v1.compute(p['physical_run'],report_sha,reader);validate_prepared(prepared,sha);reader.recheck()
    result.update(schema=NAME+'.report',arithmetic_source=v1.NAME,auditor_prepare_sha256=sha,input_bindings=dict(sorted(reader.hashes.items())))
    output.mkdir(exist_ok=False);write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    for n in ('prepare','verify','run'):g.add_argument('--'+n,action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');parser.add_argument('--report-sha256');a=parser.parse_args()
    if a.prepare:require(a.output is not None,'new output');r=prepare(a.output)
    else:
        require(a.prepared is not None and a.prepare_sha256 is not None,'reviewed SHA')
        if a.verify:validate_prepared(a.prepared,a.prepare_sha256);r=dict(status='verified')
        else:require(a.output is not None and a.report_sha256 is not None,'actual report SHA');r=run(a.prepared,a.prepare_sha256,a.report_sha256,a.output)
    print(io.canonical({k:v for k,v in r.items() if k in ('status','protocol_sha256','full_same_horizon_pair_count','strata')}))


if __name__=='__main__':main()
