# P2 完整候选分母与同输入原生搜索联合回归

本轮仅新增独占文件；保留前两轮与全部既有 producer/source seal/result。
不使用子智能体，不接管 P1/主线程代码审计，不涉及 moving_overlay_capability_probe。
使用 experiment-code 技能组织固定输入验证，不训练、不运行神经或物理实验。

## 新实现

`le-wm-main/experiments/spectral_reachability/chat_p2_20261027_candidate_census.py`
是现有 H2 native result 的只读 census 适配器，不是新 planner 或新机制判据。
expected_count 必须来自外部封存的原 plan，不能从成功候选数推导。

所有预声明槽位按原 candidate ID 顺序保留，状态分别为 completed、
failed_or_interrupted、not_attempted_after_stop，记录完整 native attempt；
不能因组合搜索早停把分母从80/40缩为1。账本求和含失败 packet，失败不退款。
raw native result 的 canonical SHA 绑定完整输入，verify_census 逐项重建比较。

仅在全部槽位和 packet 完成、原生 completion 通过、原 H2 候选与独立 replay
dispatch 总量一致时给出 deployable_engineering_candidate。此字段不是执行许可，
P2_admission 和 execution_authorized 始终 false。原 native sampling 失败后可能
保留 best/witness，本适配器不修改它，但绝不把有 best 当作完整可部署结果。

dispatch 账本不是原始模型入口 instrumentation。本适配器将 actual API calls /
batch samples 保持 null，不拿 predict_samples 声明补充测量；前两轮 bundle 的
entry_counts 才投影已有真实入口记录。编译、suffix AD/solve 和共享前置成本不在
此 ledger 中，仍由原 journal 归属，不能把这里的总量冒充全预算或实际 FLOPs。

## 同输入联合验证

`le-wm-main/tests/chat_p2_20261027_test_native_search_integration.py` 复用既有
legacy_inputs/base fixture，直接调用原生 frontier joint80、conditional composition40、
SRA40、CEM40、random40，使用同一 root/context/history/metric/domain，不另造搜索。

- 每个完整候选均有10条原 raw 动作，逐段与 whole replay 完全相同，history
  shift 和最后两包动作一致；组合的两条边精确连接完整 predicted context。
- 每个组合 suffix 声明必须绑定自己的 predicted-context SHA；不同 history 拒绝。
- goal 替换产生不同 root-bound receipt；history/domain/raw action 篡改由原 witness
  compose 拒绝。不能把只改 goal 后的新 receipt 当成原 receipt。
- 常量模型目标值下 stable-first winner 不变，不用真实 outcome 决定规则。
- 五个 arm 均在第二个 packet 注入失败，首包不退款；组合按原行为停止，其余
  producer 保留40个尝试。census 在早停时仍保留原80/40分母并禁止部署。
- 验证所增加的独立 RAM replay 调用单独计入，不当成 native 搜索的免费工作。

这些都是显式 fixture_only 的 RAM 回调，不是 CPU 神经 fallback。suffix 使用原
测试 mock 而非真实 Jacobian；联合回归不验证数值导数真实性、物理可达性或收益。

单独新增测试 **21 passed in 1.94s**。首次新增测试有3项没有将原 witness 的预期
ValueError 写入断言，修正本轮新测试后通过；未修改既有源码/测试或科学门槛。

最终联合回归 **429 passed in 24.82s**：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/chat_p2_20261027_test_native_search_integration.py \
 tests/chat_p2_20261027_test_producer_bundle.py \
 tests/chat_p2_20261027_test_lifecycle.py \
 tests/test_moving_p2_saved_program_graph_20261027_v1.py \
 tests/test_moving_p2_composition_prospective_queue_20261027_v1.py \
 tests/test_p2_composition_predict_frontier_cuda_20261027_v2.py \
 tests/test_p2_no_composition_sra_20261027_v2.py \
 tests/test_p2_full20_sampling_20261027_v2.py
```

## 实际验证产物

`le-wm-main/results/chat_p2_20261027_full_candidate_census_saved.json`：先调用
原生完整 exporter 复验，再对历史 SRA 全4根和 sampling 全4根双臂生成并复验
census，共12个 arm-root、480个候选槽位。原 native attempts 全部保留，没有
读取物理 outcome 筛根/改规则。缺少的组合 v2 完整 report 不由这些480槽替代。

`le-wm-main/results/chat_p2_20261027_shared_input_ram_census.json`：五个 arm
分别正常/注入失败，共10场景；保存完整槽位状态和 native result SHA，明确不保存
为真实神经证据。RAM 槽位状态投影不包含完整 native attempts，完整结果由测试
运行中重建；不要把这个投影当原生完整 exporter 的输入。

## 当前边界及下一放行条件

本轮完成了完整分母/失败状态适配和同输入动作见证贯通测试，没有填补以下条件：

1. MAIN 提供真实 P1 通过、外部 authority 与 freshness 全历史排除认证；本线程
   不签证，不接管 P1 八计划。确认性队列继续 fail closed。
2. 组合 frontier CUDA v2 的完整、独占 report/commit、全候选数组/witness/replay
   与冻结模型/source 终检；当前 prepared 不能当已执行。
3. 全资源有效对照：共享前置、AD/JVP/VJP/backward、solve/SVD、物理、wall/CUDA/
   FLOPs 的同预算上界与实测，以及独立物理任务收益/非贪心/配对不确定性。
   原168/88/84预测配置没有因此变为严格匹配，不修改阈值或 proposal 迎合结果。

个人 goal“可执行组合及有效对照准备”仍未全部完成。新增实现是工程准备证据，
不是 P2 科学机制通过，也不是替代主线程放行的最终报告。
