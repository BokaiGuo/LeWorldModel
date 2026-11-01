"""Exclusive implementation inventory only; NOT executable run authority."""
import argparse
import os
from pathlib import Path
from experiments.spectral_reachability import p3_matched_order_feedback_source_preflight_codex_20261029 as source
from experiments.spectral_reachability import p3_matched_order_feedback_lifecycle_codex_20261029 as lifecycle
from experiments.spectral_reachability import p3_matched_order_feedback_session_factory_codex_20261029 as sessions
from experiments.spectral_reachability import p3_matched_order_feedback_context_saved_codex_20261029 as contexts
from experiments.spectral_reachability import p3_matched_order_feedback_numeric_bridge_codex_20261029 as bridge
from experiments.spectral_reachability import p3_matched_order_feedback_store_saved_codex_20261029 as saved
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as store

NAME='p3_matched_order_feedback_candidate_seal_codex_20261029'
PRIOR=source.ROOT/'results/p3_matched_order_feedback_source_preflight_codex_20261029_MAIN01.json'
PRIOR_SHA='a1c690711ef8808d6aa4b6b1d809799106623b89d2b1c031dd94d051b2d6837f'
need=source.need


def build():
    need(source.host.sha(PRIOR)==PRIOR_SHA,'previous saved source seal')
    original=source.host.design.read_pinned(PRIOR,PRIOR_SHA)
    need(original==source.build(),'independent full source reconstruction')
    modules=(source,lifecycle,sessions,contexts,bridge,saved,store,lifecycle.seam,lifecycle.native,
             source.encoder,source.encoder_saved,contexts.structural)
    files={str(Path(m.__file__).resolve()):source.host.sha(m.__file__) for m in modules}
    files[str(Path(__file__).resolve())]=source.host.sha(__file__)
    need(len(files)==len(set(files)),'unique implementation inventory')
    return dict(schema=NAME+'.candidate',source_seal_sha256=PRIOR_SHA,
        source_report_sha256=source.REPORT_SHA,source_receipt_sha256=source.RECEIPT_SHA,
        roster=original['roster'],implementation_sha256=files,
        engineering_caps=dict(output_bytes=store.CAP,final_reserve_bytes=store.RESERVE,
                              disk_floor_bytes=store.FLOOR,wall_seconds=store.WALL,categories=store.LIMITS,
                              encoder_raw_bytes_for_16=store.ENCODER_16_RAW_BYTES),
        live_reset_verified=False,runner_complete=False,complete_saved_consumer=False,
        gpu_coordinated=False,physical_authorized=False,scientific_pass=False,
        missing=['unified CUDA model loader and instance-hook lifecycle',
                 'atomic complete/partial terminal publication',
                 'saved native physical, encoder, model and context provenance join',
                 'independent full four-slot consumer and tests'],
        limitation='implementation inventory only; this is not a run prepare or GPU authorization')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);a=ap.parse_args()
    p=Path(a.output).absolute()
    need(p.parent==source.ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists(),
         'exclusive own output')
    raw=source.host.core.canonical(build())
    with p.open('xb') as f:
        need(f.write(raw)==len(raw),'complete write');f.flush();os.fsync(f.fileno())
    source.host.signal.fsync_dir(p.parent)
    print(source.host.sha(p))


if __name__=='__main__':main()
