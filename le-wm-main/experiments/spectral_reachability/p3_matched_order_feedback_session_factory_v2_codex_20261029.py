"""Four-slot actual environment session factory; caller owns full authorization.

This is not a launcher, source seal, or saved scientific consumer.
"""
from experiments.spectral_reachability import p3_matched_order_feedback_native_v2_codex_20261029 as native
from experiments.spectral_reachability import p3_matched_order_feedback_seam_codex_20261029 as seam
from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as native_meter
from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as reset


class Factory:
    def __init__(self, prepared, source, store, model, torch, device, *, meter_module=native_meter,
                 native_module=native, reset_module=reset):
        seam.need(type(prepared) is dict and len(prepared['tasks']) == 2 and
                  type(source) is dict and len(source['roots']) == 2, 'known2 inputs')
        self.prepared,self.source,self.store,self.model,self.torch,self.device = prepared,source,store,model,torch,device
        self.meter_module,self.native_module,self.reset_module = meter_module,native_module,reset_module
        self.requested = set()

    def __call__(self, slot, root, policy):
        seam.need((slot,root,policy) in seam.ROSTER and slot not in self.requested, 'fixed single-attempt slot')
        self.requested.add(slot)
        episode = f'matched_slot_{slot:02}'
        row = dict(episode_id=episode,warmup_returned=0,control_returned=0,executed_raw=[],
                   PNG=0,checkpoints=[],status='started')
        meter = self.meter_module.Meter(self.store,episode)
        capture = self.native_module.CaptureStore(self.store)
        env = None
        try:
            original = self.meter_module.native_factory(meter)
            env = self.reset_module.TracedEnv(original,self.store,episode)
            goal,goal_rgb,prior = self.native_module.setup(env,self.source['roots'][root],
                self.source['source_plan'],self.prepared['tasks'][root],row,meter,capture)
            session = self.native_module.Session(env,goal,goal_rgb,meter,capture,row,self.model,self.torch,self.device)
            self.store.event('physical','record','matched_session_ready',self.store.index,
                             dict(slot=slot,root=root,policy=policy,episode_id=episode))
            return session,prior,env.close
        except BaseException as exc:
            if env is not None:
                try: env.close()
                except BaseException as cleanup:
                    raise RuntimeError('setup failure with unsuccessful cleanup: '+repr(cleanup)) from exc
            raise
