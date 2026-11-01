# CommonRGB：重复saved认证与内容寻址缓存边界

仅本独占新文档与文内RAM fixture；不修改任何冻结source/consumer/结果，不运行production prepare/consume、NN、physics或网络。MAIN观测prepare约300秒，本次**没有计时剖析**，不能把300秒按下面调用次数均摊，也不承诺某加速比。缓存只优化saved证据认证，不能抵扣原实验预算或放宽strict/P1。

## 1. 已定位的真实调用链

以下均位于 `experiments/spectral_reachability/`，以本次只读源码为依据：

| 函数/位置 | 直接执行的上游工作 |
|---|---|
| `agent_p2_common_resource_rgb_20261029_v1.py:125` build | :130 `producer.consume`；:132另调`producer.verify`；:146–154另验model/source绑定及实际PNG |
| `agent_p2_common_resource_physical_20261029_v1.py:617` consume | :621 `verify`→`material`；随后完整journal/state/action/resource链、486张满程PNG逐字节/像素认证及最终闭包 |
| 同physical :252 verify | :254 `material`，全prepare精确重建比较 |
| 同physical :205 material | :216 common-runtime.consume；:219 common-runtime.verify；读取六个export witness；:230 source_material；完整bindings再hash |
| `agent_p2_common_resource_grasp_cuda_20261029_v1.py:577` consume | :579自身verify；完整六臂journal/array/optimizer/whole+segmented replay等saved严格复算；末尾再验源及artifact hash |
| 同runtime :101 verify→:78 build | `wall.build`→原v2protocol.build→v1protocol.build→原planner.consume；另读任务/Q/normalizer与来源闭包 |
| `agent_p2_wallcapped_cuda_20261029_v1.py:234` build | :237 protocol.build；:242另old.verify |
| `agent_p2_compute_matched_protocol_20261029_v2.py:249` build | prior.build（包含更早完整consume），:254另old.verify |
| 对应protocol v1 :148 build | :153 old.consume，:156 old.verify；不是单纯读取manifest |

因此**单次RGB build**静态展开如下。次数不包含这些节点内部更早祖先自身的递归展开：

| 被调用节点 | 次数 |
|---|---:|
| physical.consume（完整物理journal/PNG检查） | 1 |
| physical.verify / material / source_material | 各2 |
| common-runtime.consume（完整六臂saved复算） | 2 |
| common-runtime.verify / build | 各4 |
| 通过这四次build进入wall/protocol来源链 | 4 |

这里的replay是**saved数组/控制/优化器的CPU算术认证**，不是再次神经预测或物理执行。谱侧 `wall.validate_arm:463` 会调用SavedBackend/drive重建，GRASP侧common-runtime.validate_grasp会复算Adam状态；不能把它们降成只验status/数组SHA。

RGB自己的verify(:168)又完整build一次；run(:208)先verify一次，consume(:346)也先verify一次。若MAIN分开调用prepare→verify→run→consume，累计可达8次common-runtime.consume、16次common-runtime.verify；这是四个入口各一次的调用计数，不是本次实际profile。run在创建Budget之后调用verify，因此该验证耗时计入其原600秒整个run scope；final(:248–253)还有model/source字节复核，但不再完整build。**不得为赶时间偷偷把verify移出已封存计时scope**。

只读源SHA：RGB `7964a4f1d7f7ce5e6fa0485b9100d82a5d2df6e8179c867a198f38a39c16cb56`；physical `fb660748af42fa98bcb124fcb67b5ee21059f7e10b457496521afc0a1bbb19ff`；runtime `ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d`。

## 2. 最小可落地方案：先单请求认证DAG，不先签持久缓存

未来只能新增独占版本，显式传`VerificationSession`，不patch globals、不把旧consumer替换成trust-receipt函数。建议新版本将同一原验证代码拆出返回完整认证bundle的纯节点，逐节点结果必须与旧全量入口精确相等，MAIN重新审读/测试后才可使用：

1. `runtime_prepared`：原完整prepare重建结果+完整closure；同一snapshot只重建一次。
2. `runtime_strict`：原完整六槽consume return + `runtime_prepared` + 六个原export witness；必须先真正完成全部原验证，无增删检查。
3. `native_source`：原source_material返回的原source/warmtrace+完整绑定，不能另造root。
4. `physical_prepared`：使用以上已认证对象执行原assemble/validate_execution/full-prepare equality；带原CLI args和全部SHA。
5. `physical_strict`：完整物理consume return + `physical_prepared`；原journal/PNG/未知artifact census/state/action/计量检查全做。
6. `RGB_prepare`：原full fixed_rows/model-preprocess/pixel checks使用同一bundle，不第二次material；本次读取的18个输入PNG与whole closure仍字节复核。

在同请求DAG中预期common-runtime完整strict从2→1，common-runtime prepare认证从4→1；native_source从2→1；physical严格检查仍1次。**这是结构目标，不是已实现或实测加速**。原旧接口不暴露prepared/source bundle，单靠现有return无法直接无损注入所有节点；需要新版本显式拆分受信代码，不能用memoize装饰器/monkeypatch拦截旧全局调用来假称已经解决。

所有hit返回immutable canonical bytes或重新decode的copy；不暴露可变dict作为认证对象。初次验证失败、partial拒绝或未完成的节点无成功token；已被原strict合法接受的missing/partial内容必须完整保留，不能让缓存制造成功或删分母。最终输出与原old链的full JSON精确比较是新版本上线门，不是用本fixture代替。

## 3. cache key、信任和每次命中的硬边界

内容寻址解决“是不是同一输入”，**不解决“是谁真正验过”**。外部可写的`{reportSHA,status,receiptSHA}`不是认证缓存；攻击者同时改receipt与它的SHA即可伪造。最低风险交付先只做同进程、由实际full-strict调用私有生成的认证token，不接受外部传来的“verified=true”。跨进程/CLI的持久缓存必须另行实现和审计可信builder与不可伪造发行边界（例如MAIN控制的只读CAS与可信索引，或独立保护的本地MAC/签名密钥）；本文件不提供或伪造MAIN签名。

key采用明确domain/version与canonical encoding，至少绑定：

- 请求实际report/receipt/prepare的**完整原始文件bytes SHA**、原schema/参数/完整return SHA；不同owner codec不混用，不能重新dump后当原文件SHA。
- 全部传递输入闭包：路径角色+原绝对路径+大小+SHA；模型各权重/config、Q/normalizer/task/blind/source/warmtrace、原manifest/receipt、所有artifact及journal、PNG/NPY都在内；仅report已列出artifact不够，须纳入strict实际验证的全部读取依赖。
- 每个封存输出目录的**精确名称集合/census与各项SHA/类型**，拒绝symlink/目录/extra/trailing事件的原规则照旧。新增一个未列artifact，即使report未变，也失效。
- 原validator及实际transitive helper/source/test/doc/policy/caps版本闭包；包括安装native源码、installed Lerp.h及原严格算术实现，不只顶层文件。
- 完整数值环境身份：实际解释器及NumPy/BLAS等原生库/平台/版本/二进制身份、线程与浮点策略，复用既有认证环境声明；不能随意填“同Python”。若尚不能证明闭包完备，**禁跨进程命中**，只允许一次请求内部的同解释器session复用。已有系统pytest与inner-venv结果差异提示该字段不能忽略，但本任务不诊断/运行该差异。

同字节不同路径默认不同key，除非新ABI明确允许并重新认证角色/路径语义；不能借CAS把旧root/task/动作结果换绑。只读已sealed输入才可纳入；仍在写的输出不参与。mtime/inode/文件大小只能fail-fast，永远不能代替bytes SHA。缓存不共享新模型forward/J/CPU搜索结果，不抹掉每臂实付账，也不增加freshness或科学批准。

每次hit仍须：在任何解析/使用之前核验当前闭包bytes、原目录精确census、外部真实report/receipt SHA及可信cache发行身份；验证缓存完整payload SHA/schema/key；随后只消费同一个已验snapshot。退出前再核验当前输入闭包/census，变化即拒绝并保留事件，不发布“verified”。前后hash不是对恶意并发改回的完美防护；要声称无TOCTOU，需对CAS的真实immutable snapshot或从同一已hash buffer/只读fd消费（gzip/NPY解析亦然），禁止hash路径后重新打开未受控路径。无法固定snapshot则退回原full strict，不暗降保障。

持久命中可跳过的是**已由同validator/环境对同immutable bytes完成**的昂贵saved算术；不能跳过本次bytes/census/payload/外部authority真实性。没有可靠可信缓存发行边界、依赖闭包不完整、任意SHA/策略/环境改变、缓存截断/越cap/未知字段，均miss并全量验证或failclosed；不使用TTL表示科学认证有效。

新版本另列`auth_cold_seconds/auth_hit_seconds/hash_seconds/replay_seconds/bytes_hashed/cache_status`，unknown保持null，不从调用次数推断秒数；保存旧新实际scope差异，不将缓存带来的wall减小当算法搜索收益或equal-FLOPs。现阶段没有profile，无法确定hash/解压/SVD/Adam/source rebuild各占比。

## 4. 可执行RAM fixture（不是生产缓存、模型或认证证据）

该fixture只读三份现有源码的AST，不import任何项目模块；其余全为RAM bytes。覆盖直接调用边界、2/4重复计数、请求内DAG复用、相同输出、字节/census/validator/env/receipt失效、返回值不可变、失败不缓存、验证期间改变拒绝。它**不证明真实依赖闭包已完备、不包含生产认证，也不基准300秒**。

在inner根执行下面命令即可运行本节唯一Python fence；仅读本文档与源码，不写文件、不调用旧consumer：

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c 'from pathlib import Path; p=Path("innerdocs/agent_p2_common_rgb_auth_cache_design_20261029_v1.md"); s=p.read_text().split("```python\n",1)[1].split("```",1)[0]; exec(compile(s,str(p),"exec"))'
```

```python
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def H(b): return hashlib.sha256(b).hexdigest()

base = Path('experiments/spectral_reachability')
names = {
    'rgb': 'agent_p2_common_resource_rgb_20261029_v1.py',
    'phys': 'agent_p2_common_resource_physical_20261029_v1.py',
    'model': 'agent_p2_common_resource_grasp_cuda_20261029_v1.py',
}
trees = {k: ast.parse((base / n).read_text()) for k,n in names.items()}
def calls(module, function):
    node = next(n for n in trees[module].body if isinstance(n, ast.FunctionDef) and n.name == function)
    return Counter(ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call))
assert calls('rgb','build')['producer.consume'] == calls('rgb','build')['producer.verify'] == 1
assert calls('phys','consume')['verify'] == 1
assert calls('phys','verify')['material'] == 1
assert calls('phys','material')['producer.consume'] == calls('phys','material')['producer.verify'] == 1
assert calls('model','consume')['verify'] == 1
for entry in ('run','consume'):
    assert calls('rgb',entry)['verify'] == 1

# RAM stand-ins, explicitly NOT the production transitive closure.
facts = dict(files={'report': b'report', 'source': b'source', 'array': b'array'},
    census={'output': ['report', 'array']}, validator='literal_v1', env='RAM_env1', receipt='full-return1')
def snapshot():
    return dict(files={p:dict(sha=H(b), size=len(b)) for p,b in sorted(facts['files'].items())},
        census=deepcopy(facts['census']), validator=facts['validator'], env=facts['env'], receipt=facts['receipt'])

class Session:
    def __init__(self, enabled): self.enabled, self.entries, self.counts = enabled, {}, Counter()
    def get(self, node, check):
        before = snapshot()
        key = H(canonical(dict(domain='RAM_auth_v1', node=node, closure=before)))
        if self.enabled and key in self.entries:
            blob, seal = self.entries[key]
            assert H(blob) == seal
            assert snapshot() == before
            return json.loads(blob)
        self.counts[node] += 1
        result = check()  # a real implementation MUST call full strict here
        assert snapshot() == before, 'changed during validation'
        blob = canonical(result)
        if self.enabled: self.entries[key] = (blob, H(blob))
        return json.loads(blob)

def graph(s):
    def prepared(): return s.get('runtime.verify', lambda: dict(prepared=True))
    def runtime(): return s.get('runtime.consume', lambda: dict(prepared=prepared(), full='all6_RAM'))
    def material(): return s.get('physical.material', lambda: dict(strict=runtime(), prepared=prepared()))
    def verified(): return s.get('physical.verify', material)
    def physical(): return s.get('physical.consume', lambda: dict(prepared=verified(), full='allPNG_RAM'))
    return dict(physical=physical(), prepared=verified())

old, memo = Session(False), Session(True)
expected = graph(old)
assert old.counts['runtime.consume'] == 2 and old.counts['runtime.verify'] == 4
assert graph(memo) == expected
assert memo.counts['runtime.consume'] == memo.counts['runtime.verify'] == 1
counts = memo.counts.copy()
result = graph(memo)
result['physical']['full'] = 'mutated caller copy'
assert graph(memo) == expected and memo.counts == counts

# Each independent change requires new actual validation, not status reuse.
baseline = deepcopy(facts)
for axis in ('bytes', 'extra_artifact', 'validator', 'environment', 'receipt'):
    facts.clear(); facts.update(deepcopy(baseline))
    s = Session(True); graph(s)
    if axis == 'bytes': facts['files']['array'] = b'ARRAY'  # same size, new bytes
    elif axis == 'extra_artifact': facts['census']['output'].append('unknown')
    elif axis == 'validator': facts['validator'] = 'literal_v2'
    elif axis == 'environment': facts['env'] = 'RAM_env2'
    else: facts['receipt'] = 'different-full-return'
    graph(s)
    assert s.counts['runtime.consume'] == s.counts['runtime.verify'] == 2
facts.clear(); facts.update(deepcopy(baseline))
s = Session(True)
def failure(): raise ValueError('fixture failed strict')
try: s.get('failed', failure)
except ValueError: pass
else: raise AssertionError('failure suppressed')
assert not s.entries
def change_during_check():
    facts['files']['array'] = b'changed'
    return dict(ok=True)
try: s.get('racing', change_during_check)
except AssertionError: pass
else: raise AssertionError('racing input accepted')
assert not s.entries
print('RAM fixture passed: AST calls, DAG counts, 5 invalidations, immutable copy, failure/race rejection')
```

## 5. 下一最小实现验收（未授权生产）

文内命令已由worker实际执行通过，输出`RAM fixture passed: AST calls, DAG counts, 5 invalidations, immutable copy, failure/race rejection`。只读取本文及三份source，未import项目模块、未消费真实结果；这不是性能或科学证据。

先新增单请求bundle/DAG版本，不碰冻链；用已保存fixture比较**原完整return与新完整return逐字节codec语义一致**，覆盖receipt/source/array/unknown artifact/earlystop/partial/Adam负零和FMA/PNG像素篡改，缓存冷热两路都必须拒绝同一坏输入；封存调用次数与scope账。先MAIN source-first审读，再由MAIN决定是否允许实际saved性能对比。之后才讨论跨CLI可信CAS，不先造一个未经认证的“快速verified receipt”。本任务仅交设计/RAM fixture，旧production链继续原严格行为。
