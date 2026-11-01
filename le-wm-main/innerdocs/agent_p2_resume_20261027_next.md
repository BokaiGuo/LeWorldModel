# P2 next：共同资源上限与 native 任务合同接线

仅新增本 stem 源码、测试、本文；旧 witness_queue 三文件及旧结果不改，不派生 agent。
使用 experiment-code 技能固定输入、计账、失败规则与测试；用户约束优先，不训练、不筛 outcome，
不运行神经/GPU/physics，不下载、不改门槛。个人完成仅指该工程组件，不是 P2 或研究完成。

## 具体补齐而非另造 planner

复用已冻结 witness_queue 的 prepare（默认 SHA
`e7b63325cb6e2995a86b5430256f289156d904bb7bece5ffea339c222675d839`），
其原 full exporter 继续验证 paired-v2/SRA-v2 全部 source/input seals 与动作见证。
再用既有 frontier 的 `bound_candidate_bank` 生成原固定四 known roots 的 joint80 bank；
逐一精确核前40 joint 完整动作和 conditional40 prefix 与旧320 witnesses 对应。
前40旧 receipt 仅作为历史接缝引用，不是新80候选已执行的凭证。
conditional suffix 仍待原 CUDA v2 在其自身 predicted-prefix history 编译，`commands=null`，
绝不拿 base bank 的第二段冒充新 suffix。旧 provenance 明确叫 original_saved_provenance；
新 bank SHA 另列。不会调用旧40 exporter 假装验证新80，也不改变原搜索/AD内核。

输出4 roots ×2 arms=8固定 arm slots，joint80/conditional40，共480未来候选。
这是 prospective 工程排程，旧 witness 数量仍320。没有按物理评分选择根/候选或重新选赢家。
`new_neural_calls=0/new_physics_steps=0`仅指本组件新增工作；旧获取/编码成本未知而非免费。

## base、edge 与共同上限

复用 `p2_full_resource_matching` 的 candidate_recipe/next_protocol，实际分配与 reservation
委托 `p2_supported_resource_guard`；8个不支持轴的任何有限值（含0）均拒绝。
共同 cap 是逐轴上限，不是跨轴加权分数，不是相等实际消耗或 FLOPs。

| 每根工作归属 | predict upper | AD/solve |
| --- | ---: | --- |
| shared graph base | 4 | 1次192×20 Jacobian、2 SVD、1 solve |
| joint 单候选一边 | 2 | 无新增 AD |
| conditional prefix 边评估 | 3 | 1次192×10 Jacobian，SVD/solve各≤1 |
| conditional suffix 边评估 | 1 | 无额外 AD，selection计于末边 |
| 每臂独立 winner replay | 4 | segmented H2 + whole H2 |
| 每臂原生执行/endpoint | 0 | 15 warmup +10 selected public steps、1 reset、1 encode/projector |

candidate 的 edge_evaluation_upper 是生成/评估费用（含 prefix 后 suffix compile），
不是遍历 saved graph 时再次收费。候选1次 selection，shared 只在 paired unique 入账一次，
但每条 standalone attribution 都付全额shared。两臂各168 predict上限，paired unique332、
四根1328；不可把168+168当真实pair332。joint 未花的40 suffix AD额度不转借、不padding。
zero-radius实际 SVD/solve 可为0，reservation 不退款。数学VJP方向不是 backward API或 FLOPs。
root collection/缓存输入历史获取/CPU生成成本未量化，`null`单列；controller中已知10个
模拟器内部substeps只描述该原生step函数，不补齐全流程未测内部工作量。

## 明确目标/成功/控制器，仍不冒充 task authority

只读安装的 native PushT env.py，严格绑定完整 SHA
`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`，
AST提取 __init__/step/eval_state 原文，不导入环境。保留原 goal latent/Q 与原 goal_state7，
task goal7 目前未由 prospective blind packet 供应，保持 null；renderer goal_pose 的 outline
不是 task goal。严禁修改目标/状态造图。

原成功为 agent+object 联合位置前4维范数<20，且原 raw-angle 公式<pi/9；不含速度，
reward却来自完整state7距离。raw-angle异常语义如实保留；wrapped仅诊断不回填标签。
object-only投影不正不是完整任务失败。controller 固定 relative、scale100、PD100/20、
dt.01/10Hz，原生step不额外clip，raw[-1,1] float32命令不再次变换。
精确原state/history/warmup/goal7/renderer/wrapper恢复未认证，禁止近似代替。
若native terminated提前出现，仍须MAIN审读固定horizon wrapper；本组件不自行执行。

## 接口与失败分母

`prepare(witness_path, witness_sha)`、`verify(prepared, canonical_sha)`、
`consume(prepared, canonical_sha, confirmatory=False)`，CLI prepare/verify/run-saved/consume。
prepare/verify复算全saved接缝，绑定源码/tests/doc、原依赖、输入银行及MAIN最新方向结果说明。
输出必须是results下本next stem的新文件；exclusive+fsync，拒绝覆盖。

`ArmQueue`是可运行的**RAM standalone归属状态机**，不是runner或durable guard：
base→全部固定候选→replay→execution。begin前reserve、一次terminal finish、上限/不支持轴
在状态更新前验证；失败保留全额reservation，未开始槽标skipped，pending attempt completion
保持null，禁止重试补候选。new_queues预建8臂；census_queues拒绝缺臂/换序/错身份/不同协议，
保留全部480候选，不把未启动记成成功或真实0成本。receipt中的task_sha仅绑定
`pending_task_contract_NOT_authority`声明，不是伪造的物理task_id。
ledger.completed只是已返回完成量的下界；任何pending attempt存在时，
attempted_completion_total保持null，不能把其未知完成量填0。

此接口不调用回调、不导入模型、不发起物理。未来真正runner必须在dispatch前持久化attempt，
绑定每次实际API/AD/encode/physics attempt/return、独立验证receipt、同步计时；本RAM对象不能
替代它，receipt hash本身也不是实际执行证明。输出一直 actual_dispatch_verified=false。

## 实际验证与后续命令

解释器cwd为 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p2_resume_20261027_next_tests.py tests/test_p2_supported_resource_guard_20261027_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resume_20261027_next prepare --output results/agent_p2_resume_20261027_next_prepared01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resume_20261027_next verify --prepared results/agent_p2_resume_20261027_next_prepared01.json --prepare-sha PREPARE_SHA
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resume_20261027_next consume --prepared results/agent_p2_resume_20261027_next_prepared01.json --prepare-sha PREPARE_SHA --output results/agent_p2_resume_20261027_next_consumed01.json
```

PREPARE_SHA使用prepare输出；已存在产物不能覆盖。实际执行结果在最终交付消息列出。
测试覆盖真实saved布局接缝/变异、8/480分母、base-edge闭合、unknown/0差异、原生规则、
部分失败/不退款/不重试、fatal pending、source seal、confirmatory读前拒绝。只属工程证据。

## 精确未决依赖

P1独立机制/任务收益MAIN authority、fresh roots与完整历史exclusions、真实prospective blind/task
与goal7/renderer精确恢复、MAIN预声明spec/资产与HOST CUDA/physics调度，仍未满足。
完整成本轴/壁钟硬cap/运行前durable逐primitive reservation尚未接到真实runner；不能称strict
resource graph或full-resource-matched实验已就绪。已有frontier CUDA v2是算法执行器，须MAIN
在新版本集成此计账合同而非猴补globals或重封旧seal。本次仅把无需独立P1的saved身份、
固定候选与共同有限轴预算工程前置接齐。confirmatory始终在读取前fail closed；
engineering_contract_ready不等于执行授权、P1/P2通过或真实控制收益。
