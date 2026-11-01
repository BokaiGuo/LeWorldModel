# P3 known2 same-feedback receding-horizon physical20

独占新stem源码/tests/本文；Pauli及所有已封存版本不改。使用experiment-code技能固定输入、规则、预算和离线变异回归；无worker CUDA/CPU神经/物理/网络，不派生下级。本版本是实际prepare/run/consume驱动，不是fullcommit状态机或只给callback的RAM骨架。MAIN审读后在主机执行，现有known-task授权足够；不新增IPC、历史cert或approval阻塞。

## 固定对照和真实控制

只有原两个known moving任务，各fixed/adaptive，共4episodes。每episode精确原两次native reset、10步非零warmup与原0/5/10 RGB/state/snapshot逐位身份核对，然后一次原goal bind；不改default goal_pose，不在控制中reset/restore。控制步0、5、10、15双方都编码本臂刚采集的真实RGB并重新计算本节点H1/H3信号及完整候选搜索，封存后执行赢家**首5步**。H3是15步模型lookahead，不是15步physical commit、不是三次child信号。总请求控制20步，terminal20另编码只评价，不成为第五次决策。

两种方法使用同样的反馈形式，不存在I/R混淆：初始older2来自原task-export真实observed历史，root当前latent重新编码；以后只滚入本臂实际RGB编码，raw10是warmup和本臂已执行动作最后两个完整packet。step15的三个latent来自本臂control5/10/15。绝不把本轮model replay successor滚作下个实际控制节点，绝不将旧根J当child J。

每个新节点以原task的goal/Q/model/normalizer加当前完整observed context和episode/boundary/observation身份重绑定，原root binding保持显式来源。复用原root-only NativeSignal/NativeProgram作为**当前实际节点的局部adapter**，每次新付两J；没有伪造predicted child或完整树。program_depth_limit保留原runtime请求，实际每节点只执行一次local extension；四次真实feedback不冒充depth3模型树。

native stop立即保留terminated/truncated，不补到20、不重试/换动作/换根；编译/编码工程失败不执行旧动作。固定4episode/16node分母在模型加载前落盘。完整20步才标completed20，native stop与未达节点单列；完整收据不等20步完成，更不等成功。PushT native_success/距离仍为原7维评估（位置含agent+object），不是object-only投影。actual_Q20与planning horizon terminal Q分栏，不把modelQ收益称物理收益。

## 现有实现复用与模型白名单

准备层沿用实际 `p1_moving_live_feedback_physical_20261027_v1_prepare_main01.json`（SHA `bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`）的严格verify与完整source/model/native绑定，以及P3 `agent_p3_local_signal_cuda_20261028_v1.prepare_material`的已封存known2任务/模型/源码闭包。逐根核旧moving task-export与P3 goal/context/Q/normalizer相等。选原GN layer仅取得相同task-export元数据，**不执行其旧prefix/plan**，不按旧结果挑臂。

物理沿用P4 moving_multiboundary.setup/packet、old.native_factory及其局部Metered subclass、capture/native_transition/Handler，constructor/reset/setter/space.step/render/goal-bind逐attempt/return。space hook在原_setup产生space后、reset首次_set_state前安装；并非宣称space在constructor之前存在。另实计close attempt/return；失败也尝试清理，不因超过deadline省略close。warmup物理是真实10步/episode，绝不把saved consumer的0新增physics说成从未物理执行。

编码复用P4 encode_boundary和原Handler。每个真实step0/5/10/15/20的当前PNG字节到唯一legacy_step5命名的兼容映射，原帧hash/实际step和observe→encode事件另行绑定；不是复制旧帧或新增render。模型侧只收到严格字段的task/context/raw10/goal192/身份及观察哈希；无env、reward、native state、goal7、success、snapshot或未来图。单host进程是受审代码输入隔离，不声称Python进程内存在安全沙箱。

复用 `agent_p3_local_signal_cuda_20261028_v1.NativeSignal`（full3latent/2tokens、192 scalar VJP/J、即时原精度buffers、instance-local hooks）、`NativeProgram.replay`（whole/segmented）、runtime_signal/decide_root及core.choose，common_budget.generate和原bounded_ridge。原identity Q、动作normalizer、damping/cap/chi=.5/gain=.05/relative SVD1e-6不改，不训练或拟合。高chi要求真实reconfiguration时明确failclosed；不偷偷替换成progress。观测节点发生unsupported type是工程/适配限制，不是科学负结果。

## 实际预算和差异

各实际节点固定独立付H1+H3 nominal/AD forward共8predict，两个J/384方向；signal两次array SVD。原scheduler选H，原equal-predict wrapper按24/(2H)取H1十二候选或H3四候选，fraction=i/(count-1)。每候选都做完整H whole和独立segmented replay，共24predict；同Q终态最小、候选index稳定tie-break。赢家首包再独立1predict验证并核完整计划首边界。原动作全部float32，不clip、二次tanh、/100或补预算。动作碰撞、零/负结果不筛掉。

每个完成节点33 predict API/33 samples/99 context tokens，AE/predictor各33calls/33samples/99tokens，pred_proj33calls/99rows；2J、384VJP/grad/raw callbacks、3array SVD、1solve、2signal_compile、1program_compile、1schedule。H1 replay=25/metric=13，H3 replay=9/metric=5，真实候选数/优化维度/嵌套wall不同。完整每臂4节点132predict、8J/1536VJP、12SVD、4solve、5encoder/projector（含terminal20）；完整全4臂528predict、32J/6144VJP、48SVD/16solve、20encoder/projector。source历史费用另外保留unknown，绝不免费计入新流程；saved verify/consume SVD另算（每完整节点consumer额外3次数组SVD），不混入live ledger。

这是controlled dual-signal fixed baseline，fixed也支付H3信号，**不声称最省成本固定H1强基线**。相同predict API/sample预算不等FLOPs、AD成本、实际wall或同planning horizon。FLOPs/CUDA ns/backward-engine次数未知null；报告按实际node/episode分轴及nested inclusive wall，不相加成总时间。native ledger单列物理/visual，不把神经ledger的physics_steps=0冒称全pipeline没物理。native stop/失败保留实际花费，不padding成33。

工程上限事前固定：wall1800s含prepare验证/加载；全输出512MiB（含所有重复buffers、frames和report），64MiB终态保留，disk floor200MiB，CUDA allocated3GiB/reserved4GiB。原完整源码/weights/config前后hash和全部persistent/nonpersistent buffer hash，实际MATH SDP/TF32关闭/autocast关闭before/after与peak memory持久。限额不足失败保留现场，不按本轮结果调宽。首次load失败也保留4/16启动分母，实际工作未知不填0。

## 严格saved消费

独占physical journal和每node独占native events.jsonl都fsync/链式保存；完整native输入/输出/raw/normalization/AE/predict/predictor/pred_proj、AD nominal/J/192方向及raw gradients立即保存。node局部Ledger故障不退款，旧模型forward包装退出恢复。实际plan/request/首包seal先durable再dispatch；完整中间RGB/state/snapshot保留，default goal_pose不变。

consume只读：外部report/prepare SHA、全source闭包、physical/native双链、逐attempt/return/失败、全数组dtype/shape及每边界端点、原动作归一化、原目标/同Q候选胜者、完整预算、实际历史与observe→encode→seal→dispatch顺序重算。缺末尾或工程partial不放行为完整神经回放。完整engineered native-stop也保留completed20=0/未达槽，而非偷换成功分母。消费校验保存数组闭合，不是独立重跑神经或GPU真实性证明。

## MAIN CLI（worker未运行prepare/run/生产consume）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_same_feedback_physical20_20261028_v1_tests.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_feedback_physical20_20261028_v1 prepare --output results/agent_p3_same_feedback_physical20_20261028_v1_prepare_MAIN01.json
# 以上实际输出sha256作为PREPARE_SHA，禁止编造未来SHA；以下全部MAIN审后host执行
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_feedback_physical20_20261028_v1 verify --prepared results/agent_p3_same_feedback_physical20_20261028_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA"
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_feedback_physical20_20261028_v1 run --prepared results/agent_p3_same_feedback_physical20_20261028_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA" --output results/agent_p3_same_feedback_physical20_20261028_v1_MAIN01 --authorize-known-cuda-and-physics
# REPORT_SHA取实际run输出
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_feedback_physical20_20261028_v1 consume --prepared results/agent_p3_same_feedback_physical20_20261028_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA" --output results/agent_p3_same_feedback_physical20_20261028_v1_MAIN01 --report-sha "$REPORT_SHA"
```

尚未运行真实模型/物理，无未来结果承诺。当前无需新增fresh cert即可做known2工程校准；独立P1准入/freshness/full-C3树及真实reconfiguration依然未完成。本worker不打开reserved4、不扩大scope、不以RAM通过宣称科学通过。

## 实际回归与冻结

最终联合回归实际179 passed in 120.36s：本版本54项，以及冻结local_signal、native_program、moving_multiboundary三套依赖测试。命令为上述pytest命令追加`tests/agent_p3_local_signal_cuda_20261028_v1_test.py tests/agent_p3_native_program_20261028_v1_test.py tests/agent_p4_moving_multiboundary_20261028_v1_tests.py`。仅RAM/临时saved fixtures与明确绑定known2只读metadata；未运行生产prepare、CUDA、physics或生产consume。

开发中曾有一项临时fixture roundtrip失败：synthetic derivative数组填充前已序列化旧零值journal；已修正测试FullRamJournal在fixture关闭时保存已填数组，未改变生产NativeJournal即时持久化。最终联合回归覆盖该路径。三个独占文件交付后冻结，后续变更仅新版本；测试通过不构成科学准入。
