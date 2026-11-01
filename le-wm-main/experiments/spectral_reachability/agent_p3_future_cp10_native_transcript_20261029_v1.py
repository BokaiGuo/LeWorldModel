"""Pure native-IO transcript validation for a future cp10 bridge, no execution."""
import numpy as np
from experiments.spectral_reachability import agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_future_cp10_task_projection_20261029_v1 as projection

NAME='agent_p3_future_cp10_native_transcript_20261029_v1'
require=native.require


def validate_native(p,r,events,load):
    """Validate actual native records and return own-namespace task projection.

    File/caller authority, event-file hashes, publication, failure transport and
    model/source closure are intentionally the outer consumer's responsibility.
    """
    require(len(p['rows'])==len(r['rows'])==5 and r['errors']==r['blocked']==[], 'fixed5 clean')
    require(p['expected_calls']==native.contract()['new_calls'], 'exact encoder/projector vector')
    modules=[];guards=[];want=[];latents=[];descriptors=[]
    for i,(spec,row) in enumerate(zip(p['rows'],r['rows'],strict=True)):
        require((spec['root_slot'],spec['role'])==projection.ORDER[i] and spec['index']==i, 'fixed roster')
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')} ==
                {k:v for k,v in spec.items() if k not in ('status','latent')}, 'unchanged source roster')
        require(row['status']=='encoded','all5 returned')
        pix=load(row['pixels']);native.bridge.same(pix,native.bridge.load_pixels(spec['source']),'physical PNG pixels')
        pre=load(row['preprocessed']);native.bridge.same(pre,native.bridge.pixels.preprocess_pixels(pix),'exact RGB preprocessing')
        z=load(row['latent']);native.bridge.latent(z);scope=dict(row=i,step=10,phase='encode')
        mods=row['modules'];require(len(mods)==2 and [x['resource'] for x in mods]==['encoder','projector'] and
            all(x['scope']==scope and x['completed'] is True and len(x['inputs'])==1 for x in mods), 'native module scopes')
        en,proj=mods
        buffers=[pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z]
        require(all(x.dtype==np.float32 and np.isfinite(x).all() for x in buffers),'finite native buffers')
        native.physical.saved.overlay.visual_bridge(*buffers,[1,257,192])
        native.bridge.same(buffers[1],pre.reshape(1,3,224,224),'encoder input bits')
        native.bridge.same(buffers[3],buffers[2][:,0],'encoder CLS bits')
        native.bridge.same(z,buffers[4][0],'projector returned bits')
        want.append(dict(kind='image_attempt',scope=scope,guard_image=i))
        for mod in mods:
            g=dict(index=len(guards),image=i,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),
                dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row))
        latents.append(z);descriptors.append(row['latent'])
    want += [dict(kind='all5_native_sealed',sha256=r['native_seal_sha256'],encoded=5),
             dict(kind='cp10_tasks_published',sha256=r['tasks_sha256'],all5=True)]
    require(modules==r['modules'] and [x['index'] for x in modules]==list(range(10)) and r['guards']==guards
            and events==want and r['counts']==native.bridge.counts(modules)=={
                k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()}, 'complete exact native chronology')
    result=projection.project(p['rows'],latents,descriptors)
    return dict(schema=NAME+'.validated',projection=result,native_module_count=10,event_count=len(want),
                full_saved_consumer=False,execution_authorized=False,scientific_admission=False)
