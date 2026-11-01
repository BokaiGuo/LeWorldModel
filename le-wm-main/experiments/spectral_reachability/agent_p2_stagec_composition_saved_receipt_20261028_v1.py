"""Exclusive persistent receipt writer for the frozen Stage-C saved consumer.

This module adds persistence only.  It does not alter or wrap the saved
contract, run neural/physics code, or turn integrity into a scientific claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from experiments.spectral_reachability import agent_p2_stagec_composition_saved_consume_20261028_v1 as consumer

NAME='agent_p2_stagec_composition_saved_receipt_20261028_v1'
CONSUMER_SHA='9255f3a34149776305b8503b3088251c6461095337eda3957555688f0c469fa2'
CAP=2*1024**2

def need(ok,msg):
    if not ok: raise ValueError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()+b'\n'

def write_receipt(directory,report_sha,protocol_sha,physical_sha,output):
    out=Path(output).absolute();need(not out.exists() and not out.is_symlink(),'receipt output must be new')
    need(out.parent.is_dir(),'receipt parent must exist')
    d=Path(directory).absolute()
    actual=consumer.consume(d,report_sha,protocol_sha,physical_sha)
    need(actual['schema']==consumer.NAME+'.consume' and actual['status']=='saved_contract_verified_not_admitted',
         'saved consumer did not return verified non-admitted status')
    need(actual['composition_benefit'] is None and actual['P1_admission'] is actual['P2_admission'] is False,
         'scientific boundary changed')
    receipt=dict(schema=NAME+'.receipt',status='saved_receipt_persisted',
        input_directory=str(d),report_sha256=report_sha,protocol_sha256=protocol_sha,
        physical_sha256=physical_sha,consumer_module=str(Path(consumer.__file__).absolute()),
        consumer_sha256=CONSUMER_SHA,consumer_result=actual,
        composition_benefit=None,scientific_claims=[],P1_admission=False,P2_admission=False,
        neural_calls=0,physics_calls=0,new_predict_calls=0)
    raw=canonical(receipt);need(len(raw)<=CAP,'receipt metadata cap')
    with out.open('xb') as f:
        need(f.write(raw)==len(raw),'short receipt write');f.flush();os.fsync(f.fileno())
    fd=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    need(sha(out)==hashlib.sha256(raw).hexdigest(),'receipt SHA after fsync')
    return dict(receipt=str(out),receipt_sha256=sha(out),bytes=len(raw),status=receipt['status'],
        branch_denominator=actual['branch_denominator'],artifact_count=actual['artifact_count'],
        composition_benefit=None,P1_admission=False,P2_admission=False)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True);p.add_argument('--report-sha',required=True)
    p.add_argument('--protocol-sha',required=True);p.add_argument('--physical-sha',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(write_receipt(a.directory,a.report_sha,a.protocol_sha,a.physical_sha,a.output),sort_keys=True))

if __name__=='__main__':main()
