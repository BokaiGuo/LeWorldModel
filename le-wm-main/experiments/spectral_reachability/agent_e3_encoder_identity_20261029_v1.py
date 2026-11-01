"""New E3 loader/encoder runtime instance contract; not a full strict receipt."""
import secrets
import traceback
from pathlib import Path
from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
from experiments.spectral_reachability import agent_e3_encoder_io_20261029_v1 as encoder

NAME='agent_e3_encoder_identity_20261029_v1'
need=launch.need


def _event(store,kind,detail):
    store.event('physical','record','encoder_identity_'+kind,store.index,detail)


def _inventory(model):
    params=list(model.named_parameters())
    need(params and len({name for name,_ in params})==len(params),'nonempty unique parameters')
    result=[]
    for name,p in params:
        need(type(name) is str and bool(name) and p.requires_grad is False and getattr(p.device,'type',None)=='cuda',
             'frozen CUDA parameter identity')
        result.append(dict(name=name,instance=hex(id(p)),device=str(p.device)))
    return result


class Binding:
    def __init__(self,model,device,model_sha,prepare_sha,store):
        need(getattr(device,'type',None)=='cuda' and type(model_sha) is str and len(model_sha)==64
             and type(prepare_sha) is str and len(prepare_sha)==64,'HOST identity inputs')
        self.model=model;self.device=device;self.store=store;self.modules={n:getattr(model,n) for n in ('encoder','projector')}
        need(self.modules['encoder'] is not self.modules['projector'],'distinct module objects')
        self.params=_inventory(model);need(all(p['device']==str(device) for p in self.params),'exact HOST parameter device')
        self.tag=secrets.token_hex(32)
        self.descriptor=dict(schema=NAME+'.binding',tag=self.tag,prepare_sha256=prepare_sha,model_sha256=model_sha,
            model_instance=hex(id(model)),device=str(device),modules={n:hex(id(v)) for n,v in self.modules.items()},
            parameters=self.params)
        _event(store,'bind',self.descriptor)

    def check(self):
        need(all(getattr(self.model,n) is obj for n,obj in self.modules.items()),'encoder/projector object changed')
        need(_inventory(self.model)==self.params,'model parameter identities or devices changed')

    def encode(self,torch,frame,boundary,check):
        return self._encode(torch,frame,boundary,check,encoder.encode)

    def _encode(self,torch,frame,boundary,check,implementation):
        """Explicit implementation seam for fixture only; HOST calls encode()."""
        need(type(boundary) is int and boundary in (10,15),'fixed encoding boundary')
        self.check();seen=[];handles=[]
        _event(self.store,'intent',dict(tag=self.tag,boundary=boundary))
        def remove_hooks():
            errors=[]
            while handles:
                try:handles.pop().remove()
                except BaseException as exc:errors.append(exc)
            if errors:
                for extra in errors[1:]:errors[0].add_note('additional hook cleanup failure: '+repr(extra))
                raise errors[0]
        def hook(role):
            def observe(module,args):
                self.check();need(module is self.modules[role] and role not in seen,'same bound module callback')
                seen.append(role);_event(self.store,'callback',dict(tag=self.tag,boundary=boundary,role=role,
                                                                    module_instance=hex(id(module))))
            return observe
        try:
            for role in ('encoder','projector'):
                handles.append(self.modules[role].register_forward_pre_hook(hook(role)))
            result=implementation(self.model,torch,self.device,self.store,frame,boundary,check)
            self.check();need(seen==['encoder','projector'],'exact same-object encoder/projector callbacks')
            remove_hooks()
            _event(self.store,'return',dict(tag=self.tag,boundary=boundary))
            return result
        except BaseException as exc:
            try:remove_hooks()
            except BaseException as secondary:exc.add_note('identity hook cleanup: '+repr(secondary))
            try:_event(self.store,'failure',dict(tag=self.tag,boundary=boundary,type=type(exc).__name__,
                                               message=str(exc)[:2048],traceback=traceback.format_exc()[-8192:]))
            except BaseException as secondary:exc.add_note('identity failure recording: '+repr(secondary))
            raise


def load_bound(prepared,torch,device,store,prepare_sha):
    """Call the original loader and bind its returned object, not a caller substitute."""
    model,model_sha=launch.host.primitives.native.loader.loaded_model(prepared,torch,device,
                           launch.loader_bridge.LoaderBridge(store),prepare_sha)
    need(model_sha==prepared['expected_model_hash'],'loaded model hash')
    return Binding(model,device,model_sha,prepare_sha,store)


def verify(events,prepared,prepare_sha):
    """Verify authenticated event list; no old log can satisfy this contract."""
    loader_receipt=launch.loader_bridge.verify_events(events,prepared,prepare_sha,launch.host.primitives.native.loader)
    model_sha=prepared['expected_model_hash']
    records=[e for e in events if e['origin']=='physical' and e['resource'].startswith('encoder_identity_')]
    need(records and records[0]['resource']=='encoder_identity_bind' and all(e['kind']=='record' for e in records),
         'new encoder identity binding required')
    d=records[0]['detail'];need(set(d)=={'schema','tag','prepare_sha256','model_sha256','model_instance','device','modules','parameters'}
        and d['schema']==NAME+'.binding' and d['prepare_sha256']==prepare_sha and d['model_sha256']==model_sha
        and type(d['tag']) is str and len(d['tag'])==64 and all(c in '0123456789abcdef' for c in d['tag'])
        and type(d['model_instance']) is str and bool(d['model_instance'])
        and type(d['device']) is str and d['device'].startswith('cuda') and type(d['modules']) is dict
        and set(d['modules'])=={'encoder','projector'} and d['modules']['encoder']!=d['modules']['projector']
        and type(d['parameters']) is list and bool(d['parameters']),'binding schema')
    need(all(type(p) is dict and set(p)=={'name','instance','device'} and all(type(v) is str and v for v in p.values())
             and p['device']==d['device'] for p in d['parameters'])
         and len({p['name'] for p in d['parameters']})==len(d['parameters']),'bound parameter inventory')
    need(loader_receipt['tensor_hash_before']==model_sha and loader_receipt['device']==d['device'],
         'loader model hash and device binding')
    loader=[e for e in events if e['origin']=='physical' and e['resource']=='loader_receipt']
    need(len(loader)==1 and loader[0]['index']<records[0]['index'],'binding follows loader receipt')
    state=None;rows=[];previous=records[0]['index'];failed=False
    for e in records[1:]:
        x=e['detail'];kind=e['resource'].removeprefix('encoder_identity_')
        need(type(x) is dict and x.get('tag')==d['tag'] and type(x.get('boundary')) is int
             and x['boundary'] in (10,15) and e['index']>previous,'identity event binding/order')
        if kind=='intent':
            need(set(x)=={'tag','boundary'} and state is None and not failed
                 and (not rows or x['boundary']>rows[-1]['boundary']),
                 'unique ordered identity intent')
            state=dict(boundary=x['boundary'],callbacks=[],callback_indices=[],index=e['index'])
        elif kind=='callback':
            need(set(x)=={'tag','boundary','role','module_instance'} and state is not None
                 and x['boundary']==state['boundary'] and x['role'] in ('encoder','projector')
                 and x['module_instance']==d['modules'][x['role']] and
                 state['callbacks']==([] if x['role']=='encoder' else ['encoder']),'exact callback identity')
            state['callbacks'].append(x['role']);state['callback_indices'].append(e['index'])
        elif kind in ('return','failure'):
            need(state is not None and x['boundary']==state['boundary'],'matching identity outcome')
            if kind=='return':need(set(x)=={'tag','boundary'} and state['callbacks']==['encoder','projector'],
                                   'complete bound callbacks')
            else:need(set(x)=={'tag','boundary','type','message','traceback'} and
                      all(type(x[k]) is str for k in ('type','message','traceback')),'preserved identity failure')
            if kind=='failure':failed=True
            state['outcome']=kind;state['end']=e['index'];rows.append(state);state=None
        else:need(False,'unknown identity record')
        previous=e['index']
    for row in rows:
        for role,idx in zip(row['callbacks'],row['callback_indices'],strict=True):
            operations=[e for e in events if e['origin']=='encoding' and e['resource']==role
                        and e['kind']=='intent' and e['detail'].get('boundary')==row['boundary']]
            if row['outcome']=='return':need(len(operations)==1 and idx<operations[0]['index']<row['end'],
                                             'identity callback joins native encoder intent')
    return dict(schema=NAME+'.saved',binding=d,encodes=rows,pending=state,
                runtime_instance_contract_recorded=True,physical_or_neural_parity=False,full_saved_strict=False,
                limitation='source-trusted runtime observations, not cryptographic proof of process memory')


def consume(output,report_sha,completion_sha,prepared,prepare_sha,source_bindings):
    """Read transport-authenticated output and rehash the explicit source closure."""
    required=[str(Path(m.__file__).resolve()) for m in
              (launch,launch.loader_bridge,launch.host.primitives.native.loader,encoder)]
    required.append(str(Path(__file__).resolve()))
    launch.episode.old.guard.validate_sources(source_bindings,required)
    transport=launch.pub.consume(output,report_sha,completion_sha)
    p=Path(output);report=launch.pub.base.read_bytes(p/'report.json',launch.pub.FINAL_CAP)
    journal=launch.pub.base.read_bytes(p/'events.jsonl',launch.pub.base.CAP)
    need(launch.pub.base.fsha(report)==report_sha and
         launch.pub.base.fsha(journal)==launch.pub.base.decode(report)['journal_sha256'],'stable identity journal')
    result=verify([launch.pub.base.decode(line) for line in journal.splitlines()],prepared,prepare_sha)
    result['transport_complete']=transport['complete']
    return result
