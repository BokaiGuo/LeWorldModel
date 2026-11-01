# P3 same-public20 前瞻闭环最小准备（未放行生产）

仅本stem source/tests/本文。Hi stride诊断与旧P3/P4均不改。本模块不运行神经、物理、网络，
无子代；experiment-code技能用于先固定协议再做RAM失败测试，测试不是科学证据。

## 固定设计及可执行边界

固定原2 known roots，各fixed/adaptive两臂，共4分母；同原task/Q/normalizer/model/goal，
由原pairedquota prepare的真实SHA `46e191695e6e121324daf691bafe1b2862a36349e11232822957d8ca6785e6d8`
绑定完整task，而非按outcome挑根。新声明只增加原生recipe/prefix/authority的实际SHA，
两个方法独立fresh重建同源warm10+prefix5，不借旧endpoint。该原task快照不是新根认证。

共同目标public20（prefix5+新增15），warm10另计。每5步真实RGB反馈：5/10/15/20；
5/10/15两臂各自独立付费H1/H3局部谱信号，20仅终点编码/评价。原core.local_signal/choose
原封复用，physical depth绝不冒充模型树深度；本模块明确只做depth0 receding decisions，非C3树。

每次空闲调度保留原候选4/8和H1/H3，不使用旧12/4等额quota。采用明确的**full-commit**
解释：选H3就必须完整执行其15条动作；中间每5步反馈并付谱信号，但不替换已承诺动作。
这不是“每5步截断H3重规划”，也不声称反馈利用/决策频率相等。若研究想比较H3-lookahead
且每5步重规划，应另预声明一个不同协议，不能称其完整执行H3。本设计保留所有中间信号。

若后续H3装不进剩余public时域，拒绝整个节点，保留partial和四分母；不强迫H1、不裁切、
不补步、不重新选调度。原生done早停保留；任何臂未到20都不能宣称实际完整等时域比较。
因此目标public20相等不保证所有调度都可完成，失败也是该预声明合同的工程结果。

## 复用接点与确切硬缺口

- P3 runtime `local_signal/choose/Ledger`可直接复用：本模块RAM驱动已实际调用。
- `NativeChildSignal`、`NativeProgram`受旧saved child白名单约束，不能伪造新live ancestry。
- pairedquota compiler `execute_branch`仅释放selected raw5；新接口必须释放raw(5*H)、
  独立whole/segmented各候选及selected full-H重放。旧consumer不能为新15动作签证。
- 原progress编译器未认证reconfiguration/composition；当前整节点拒绝，绝不fallback或缩候选。
- P4 `setup/encode_boundary/roll/native_factory`可作为原reset/warm/观察/history模板，
  `RollingSuffix.compile_suffix`本身H1-only，不能直接当H3。它也不是已认证的隔离IPC。
- 本次检查的P3/P4实现没有可直接复用的live native↔CUDA subprocess协议；需要MAIN审读的
  两个stateful worker和durable前置资源hook IPC。物理进程不得import torch，CUDA进程不得
  import native physics；不能把现有in-process live driver伪称已隔离实现。
- 还缺此full-commit流程原生constructor/reset/warm/render/physics/close实际profile及全数组consumer。

因此`prepare/verify`是真实可执行的blocked planning artifact，`run`无条件前置拒绝且无loader。
`RamDriver.execute`是可执行的控制流最小slice（全4分母、完整H3跨3反馈、原调度、逐步earlystop、
失败清理、预算门和因果哈希）；它只接受`RAM_FIXTURE`，不能升级成真实runner或科学通过。
不会把fakecounter当实际GPU成本，也不会声称此三件套已达到HOST CUDA准入。

## 预算：只按各自单位计

各臂最多3次谱信号：24 predict API/24 samples，6J、1152 scalar VJP/grad/raw callbacks、
6 array SVD。nominal与AD-forward均计费。编译每节点H、n：候选whole/segmented `2Hn`，
selected**完整H**独立replay再付H，即`H(2n+1)`，而不是旧prefix reserve1。
最大累计已执行H<=3：fixed predict严格cap51，adaptive cap75；这是同控制时域不同自然配额，
不是同predict预算。batch固定1；API与samples仍分轴，三context token也单列。
最多4次encoder/projector；J/VJP/SVD/solve/replay/metric/schedule以及AE/predictor/proj都列在
`arm_caps`，由core.Ledger真实入口attempt/return检查；失败不退款、超cap先拒绝，不事后扣账。
callback seam的hook数字只是RAM见证，真实worker还必须提供张量/调用fullarray strict验证。

原生资源未知null（不填0）；FLOPs/total_compute/inherited acquisition/walltime equivalence均null。
900s/256MiB output/200MiB floor为后续工程上限，当前无生产runner强制执行，非已完成资源证据。
没有新模型/Q拟合、原阈值不变。P1/scientific admission/C3/freshness永远false。

## MAIN可执行命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_prospective_equal_horizon_20261028_v1_test.py
```

准备输入`SOURCES_JSON`为精确两项list，每项字段仅：source_id、root_slot(0/1)、task（原
authority.wiring.tasks对应项完整原值）、recipe_sha256、prefix_sha256、authority_sha256。
后三项必须由MAIN已有原生source精确绑定产生，不用示例/未来SHA替代。不得包含物理outcome。
这些SHA目前只作为待认证外部声明，尚不赋予physical release。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_prospective_equal_horizon_20261028_v1 prepare --sources "$SOURCES_JSON" --sources-sha "$ACTUAL_SOURCES_SHA" --output results/agent_p3_prospective_equal_horizon_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_prospective_equal_horizon_20261028_v1 verify --prepared results/agent_p3_prospective_equal_horizon_20261028_v1_prepare_MAIN01.json --prepare-sha "$ACTUAL_PREPARE_SHA"
```

不提供虚构CUDA命令：同参数`run`会明确拒绝。下一最小实现是**新live child signal/full-H
progress release适配+两进程hooks桥**，再做严格数组consumer；不是再补一层通用manifest。
若需要支持reconfiguration，需先独立实现并封存合法原算法编译ABI。旧48槽strict成功保留，
但first5物理结果不能为本合同填补完整H3物理证据，也不能证明C3。
