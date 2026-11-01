# P3 有效调度 / 非嵌套支持：固定已知根诊断

## 本次实装与明确边界

独占新三件套；不修改任何旧调度、模型、环境或封存结果。提供真实
`generate`、`plan_node`、`Session`、`episode` 和节点级 `audit_node`，并已按MAIN追加要求
在同一尚未冻结v1内接齐**自身8臂/24节点整作业HOST launcher与native/RGB strict consumer**。
CLI为 `prepare/verify/accounting/run/consume`，不是旧函数别名或必拒manifest。
旧6臂producer/consumer不执行、不换名、不patch globals、不伪造旧receipt。
worker没有生产prepare、模型、GPU、物理或网络执行。

## 已有证据与为何不同

检查原 `agent_p3_commitment_controls_20261029_v1/v2` 和
`agent_p3_fullcommit_live_20261028_v1` 的实际调度、候选生成、反馈滚动和计账接口。
原fixedH3为8候选、adaptive通常为4候选；同赢家/同轨迹不代表同候选支持、同成本或C3。
原fullcommit、H1/H3、真实反馈本身都已实现，不把它们重新命名为新能力。

本次新增干预是**非嵌套候选支持×固定承诺时序的2×2**，去掉候选宽度差异：

| arm | 边界5 | 边界10/15 | 每次候选fractions |
|---|---|---|---|
| commit15_endpoints | H3全15承诺 | 付费观测，不替换 | 0, 1/3, 2/3, 1 |
| commit15_midpoints | H3全15承诺 | 付费观测，不替换 | 1/8, 3/8, 5/8, 7/8 |
| feedback5_endpoints | H1执行5 | 用真实RGB重编H1 | 0, 1/3, 2/3, 1 |
| feedback5_midpoints | H1执行5 | 用真实RGB重编H1 | 1/8, 3/8, 5/8, 7/8 |

两原known roots均全保留，共8episode/24边界；这不是原adaptive政策的性能声明。
所有时序/网格在执行前固定，不读取native outcome、chi/gain或赢家来选择网格、根或预算。
按同bank比较时序、按同时序比较bank；不把跨bank比较叫纯调度因果效应。
`midpoints`去掉zero且降低最高fraction，属于明确的候选支持干预，不是相同incumbent保护。

分数集合互斥，不保证实际raw集合互斥：ridge零方向或float32塌缩时可以重复。
每个bank记录**同一局部方向**上两网格的所有16对逐字节交集；这是无额外模型调用的
算术支持诊断，不是反事实任务结果。禁止遇到重复再改网格或重采样。
不同实际context之间不得凭fraction相异宣称raw互斥。

## 不变项与实际执行接口

原任务goal、identity192 Q、checkpoint、normalizer、动作尺度、raw域和原ridge cap不变。
每个实际节点调用原 `native.generate` 的bounded_ridge、SVD和divisor；新网格仅按
`float64 fraction * direction / divisor -> float32 once` 生成候选，没有新tanh图或拟合。
全4候选完整whole/segmented replay、原terminal Q评分、stable(cost,index)选择和独立赢家replay。
没有outcome反馈选择、保旧下降guard或新重试。

`Session.plan(context, raw10, observation, factory, journal)` 实际执行以上编译，返回
`(request, result, raw5packet)`。factory为原
`lambda task,sink: live.native.NativeProgram(model, device, task, sink)`；仅允许HOST_CUDA，
`permit_ram=True`只允许标识为RAM_FIXTURE的人工数组backend，绝不是CPU神经fallback。
`acknowledge(returned, terminated=..., truncated=...)`严格绑定实际rawbits，短返回仅native stop；
未ACK不能重编，commit15未结束不能替换。失败debit不退款，后续停止。

`episode(env, ref, data, layer, base, row, visual, runtime, meter, store, factory)` 已实际接
`moving.setup/packet/encode_boundary/roll`：独立reset/place/warm10、同prefix5、同goal、
每个到达边界的真实RGB、`(warmraw10 + executedraw[:boundary])[-10:]` 与原normalizer。
编译期间native state/goal seal不得变化；完整commit15分3个raw5包释放；native stop不补齐。
`native_roster()`返回符合原moving原生字段ABI的8条新row；`roster()`是prepare未执行分母。
局部编译失败保留partial journal、request、resource receipt，不能返回完整node验证。

整作业 `run/execute` 负责每臂独立constructor/close、wall/disk/GPU/native/visual caps，
冻结模型前后检查、source seals、TF32/autocast关闭和MATH上下文。新 `Store/Meter` 显式使用
自身8臂caps/命名；仅继承旧policy-free durable写入和计量方法，不借用6臂caps。
`run` 无显式授权立即拒绝，只有MAIN在HOST CUDA可执行；不提供神经CPU路径。
full job新实现从原dispatch/census逻辑作机械适配，但schema、24节点分母、grid和commitment
绑定为自身权威。Node算法和网格在追加launcher时未改变。

`consume` 验证所有物理journal事件类型/完整hashchain/嵌套operation/逐region计量，
全24node的资源和native模块/J/VJP数组，所有PNG和RGB预处理逐float32 bits、alias原bytes、
encoder192、真实三帧与raw10滚动、immutable commitment的逐5偏移与实际命令。
每个returned state7原native reward/success/goal/goal_pose保持，精确native earlystop与未到达节点
全部保留；不把第14步终态当第20步或补齐cp15/20。模型before/after、精度、wall/GPU峰值和
分臂全部成本均消费；partial输出保留但不获得complete。
复用 `harden.audit_rgb` / `audit_resources` / `audit_trajectory` 等policy-free数组核验，
不调用旧策略的consumer来假装新策略通过。

## 预算（完整20步才成立；提前停止保留实际分母）

每episode均3边界×8signal predict =24，程序predict各27，总51；8臂408。
每episode6J/1152scalar VJP/1152grad/1152raw backward；全体48J/9216VJP。

| 每episode | commit15 | feedback5 |
|---|---:|---:|
| signal compile | 6 | 6 |
| program compile / solve | 1 | 3 |
| SVD（含诊断） | 7 | 9 |
| 完整replay调用 | 9个H3 | 27个H1 |
| metric | 4 | 12 |
| schedule decision | 1 | 3 |
| 公共执行target（含warmup） | 30 | 30 |
| encoder/projector target | 各4 | 各4 |

`costs/caps/accounting`列出全部原Ledger轴；每个node实际attempted/completed独立记录，
失败尝试不退款。只在完整20且无失败时predict数相等，**notmatched / 非等总算力、非等FLOPs、非等wall**。
inclusive operation wall不可直接相加成总wall。没有padding计算或虚构实际成本。

整作业固定caps：wall3600s、总输出1GiB（含128MiB final reserve）、磁盘floor200MiB、
CUDA allocated3GiB / reserved4GiB。8臂物理上限为constructor8/reset24/set_state48/
goal_render24/goal_setter32/goal_bind8/public240/physics2448/render256/close8；
encoder/projector各32。上限不伪称实际；全20情况下原recipe实际reset16、physics2432。
失败/timeout关闭和最终report走保留预算，不因过wall把状态假记为complete。

## Saved权威与MAIN命令

prepare通过原controls.build重建原fullcommit源/任务/模型权威。较新6臂工程receipt仅明确复用其
外部固定SHA，不声称在本地再次consume：report
`3e02de2f957253830a9122e62926563ab6ff80b8eb3c61207d4d1c367bbe044f`，receipt
`aa5c34faabae5205dbeb7dd216fa5ff0631d16239eae25de06bf2b9a2084e94a`。
其native标签不传给scheduler/generate；自有源码/tests/doc一起进入prepare SHA闭包。

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_effective_schedule_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_effective_schedule_20261029_v1 accounting
.venv/bin/python -m experiments.spectral_reachability.agent_p3_effective_schedule_20261029_v1 prepare --output results/agent_p3_effective_schedule_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_effective_schedule_20261029_v1 verify --prepared results/agent_p3_effective_schedule_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_effective_schedule_20261029_v1 run --prepared results/agent_p3_effective_schedule_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_effective_schedule_20261029_v1_MAIN01 --authorize-known-cuda-and-physics
.venv/bin/python -m experiments.spectral_reachability.agent_p3_effective_schedule_20261029_v1 consume --prepared results/agent_p3_effective_schedule_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_effective_schedule_20261029_v1_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_effective_schedule_20261029_v1_consume_MAIN01.json
```

`audit_node(request,result,events,base)`重算所有原native/J/grad模块连线、候选raw、完整路径、
原Q、赢家和成本；重用policy-free `audit_resources/audit_trajectory`，不伪造旧node schema或成功receipt。
调用前必须由 `signal.journal(path)`完整验证原生逐行hashchain及外部封存SHA/tail。
节点helper返回永远 `whole_job_verified=False`；只有独立的整作业 `consume` 可给工程complete，
也永远不是P1/P3/C3认证。receipt通过自身exclusive durable writer保存，不扫描results父目录。

## 实际轻量验证与负结果

最终轻量测试结果在交付消息给出。涵盖全4arm真实节点编译/保存数组复算、真实反馈Session、
完整成本、互斥fraction/raw退化、失败不退款、short ACK、模块/gradient/Q/赢家篡改拒绝、
纯RAM native episode完整20与rawhistory滚动、真实磁盘Store/NativeJournal一LF读写/hashchain，
完整8臂/24节点真实保存与整作业consume、模型/精度/计量/commitment篡改拒绝，
以及第14步native stop的完整strict与未到达cp15/20保留。
首轮9failed/25passed：saved重算dummy ledger未设置scope，以及测试观测未cast到literal float32；
均为本次执行前实现/fixture错误，已修复，不归因为历史P1/C3原因，不删除负结果。
追加launcher/strict首轮45passed/1failed：model-before/after fixture共用字典，修改after时
同步改了before，未构造真正的两侧不一致。已改为独立after映射，并加强最终模型/source
检查与report的绑定；没有因此修改调度、网格、预算或科学门。

尚未运行真实prepare/verify、CUDA、神经CPU或物理；真实outcome全部unknown。
P1未过，known2非独立确认；同轨迹或不同轨迹均不能单独认证C3/P3/P1。
