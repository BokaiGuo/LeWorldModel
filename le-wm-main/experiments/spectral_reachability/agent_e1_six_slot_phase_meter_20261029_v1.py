"""Independent actual-hook meter phases and replay verification; no production runner."""
from __future__ import annotations
from copy import deepcopy
from experiments.spectral_reachability import agent_kinodynamic_matched_domain_20261028_v1 as beam

RESOURCES=beam.RESOURCES


class PhaseMeter(beam.sampling.Meter):
    """Actual pre/post module hooks, one independent phase; failure never refunds."""
    def __init__(self,phase,sink,sync=lambda:None,check=lambda:None):
        beam.require(phase in ('controller','replay'),'fixed phase')
        super().__init__(sink,sync,check)
        self.phase=phase
        self.limit=144 if phase=='controller' else 4

    def available(self):
        self.check()
        beam.require(not self.fault and all(x['attempted']<self.limit for x in self.counts().values()),
                     'phase packet cap/no retry')

    def begin(self,resource,inputs):
        self.check()
        beam.require(not self.fault and resource in RESOURCES and
                     self.counts()[resource]['attempted']<self.limit,'phase resource cap/no retry')
        # Base implementation still enforces original 144, and journals the
        # ticket before executing. Replay uses a fresh meter with cap 4.
        return super().begin(resource,inputs)


def replay(task,raw,backend,meter,sink):
    """Three live edges, four paid packet calls from literal sealed raw10."""
    beam.require(meter.phase=='replay' and meter.counts()==beam.budget(0),'fresh replay meter')
    commands=beam.commands(raw).astype(float).tolist()
    beam.require(len(commands)==10,'literal complete raw10')
    root=beam.checked_context(task['root_context'])
    d=beam.Dispatch(task,backend,meter,sink)
    whole=d(deepcopy(root),deepcopy(commands),'replay_whole',[0,1])
    a=d(deepcopy(root),deepcopy(commands[:5]),'replay_segment',[0])
    b=d(deepcopy(a[0]),deepcopy(commands[5:]),'replay_segment',[1])
    beam.require(d.index==3 and meter.counts()==beam.budget(4) and
                 meter.predictor_samples()==dict(attempted=4,completed=4),'paid replay actual module returns')
    return dict(raw=commands,whole=whole,segmented=a+b)


def verify_saved_replay(task,raw,events):
    """Authenticate all three saved edges and actual operation nesting without NN."""
    commands=beam.commands(raw).astype(float).tolist()
    beam.require(len(commands)==10,'literal complete raw10')
    root=beam.checked_context(task['root_context'])
    cursor=beam.Cursor(events)
    d=beam.SavedDispatch(task,cursor)
    whole=d(deepcopy(root),deepcopy(commands),'replay_whole',[0,1])
    a=d(deepcopy(root),deepcopy(commands[:5]),'replay_segment',[0])
    b=d(deepcopy(a[0]),deepcopy(commands[5:]),'replay_segment',[1])
    beam.require(cursor.index==len(events) and d.index==3 and len(d.operations)==16 and all(
        sum(x['resource']==resource and x['status']=='returned' for x in d.operations)==4
        for resource in RESOURCES),'exact saved replay meter and no extra events')
    return dict(raw=commands,whole=whole,segmented=a+b)
