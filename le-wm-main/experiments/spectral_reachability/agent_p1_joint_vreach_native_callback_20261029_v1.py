"""Per-call native callbacks for joint Vreach; not a standalone CUDA host."""
from copy import deepcopy

from experiments.spectral_reachability import agent_p1_joint_vreach_signal_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as components

core = bridge.core
require = core.require


class NativeCallbacks:
    """Separate observation scopes prevent nested hooks on the same model.

    `scalar_backend` is the original metered step/gradient backend. `component`
    constructs one fresh H1 component with a parent-local task. All component
    events, including failed prefixes, are forwarded before callback return.
    Source/model guards and durable journaling remain the host's responsibility.
    """
    def __init__(self, root, scalar_backend, component, sink):
        self.root = core.validate_task(deepcopy(root))
        self.scalar = scalar_backend
        self.component = component
        require(callable(component) and callable(sink), 'component factory and durable sink')
        self.sink = sink

    @classmethod
    def host(cls, root, scalar_backend, model, device, guard, sink, *, main_authorized=False):
        require(main_authorized is True and callable(guard), 'explicit MAIN authorization and model guard')
        require(getattr(device, 'type', None) == 'cuda', 'CUDA only')
        def factory(task, event):
            caps = dict.fromkeys(core.AXES, 0)
            caps.update(components.expected('postcontext_H1'))
            return components.Components.host(root=task, caps=caps,
                verification_reserve=dict.fromkeys(core.AXES, 0), model=model, device=device,
                sink=event, main_authorized=True, guard=guard)
        return cls(root, scalar_backend, factory, sink)

    def step(self, context, raw, label):
        with self.scalar.observe():
            return self.scalar.step(context, raw, label)

    def gradient(self, context, raw, label):
        with self.scalar.observe():
            return self.scalar.gradient(context, raw, 'gradient_'+label)

    def signal(self, context, label):
        local = bridge.parent.parent_task(self.root, context)
        self.sink(dict(kind='joint_signal_attempt', label=label,
            root_task_sha256=core.digest(self.root), parent_context_sha256=core.digest(context),
            local_task_sha256=core.digest(local)))
        def emit(event):
            self.sink(dict(kind='joint_signal_event', label=label, payload=event))
        c = self.component(local, emit)
        try:
            receipt = c.h1(prefix_actions=[], prefix_path=[])
            require(receipt['status']=='complete' and receipt['local_task']==local,
                'completed local signal and exact parent')
            out = bridge.from_transcript(self.root, context, receipt['native_transcript'])
            require(out['nominal_context']==receipt['result']['nominal_path'][0] and
                out['jacobian']==receipt['result']['jacobian'], 'audited callback matches component')
            require(receipt['resources']==components.expected('postcontext_H1'), 'actual H1 resources')
            self.sink(dict(kind='joint_signal_return', label=label, receipt=deepcopy(receipt)))
            return out
        except BaseException as exc:
            self.sink(dict(kind='joint_signal_failure', label=label,
                error=dict(type=type(exc).__name__, message=str(exc)),
                receipt=deepcopy(getattr(exc, 'receipt', None))))
            raise
