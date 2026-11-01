# P2 组合候选配对数据合同

新增独立只读校验器：
`le-wm-main/experiments/spectral_reachability/d2p_p2_comparison_contract.py`。
测试：`le-wm-main/tests/test_d2p_p2_comparison_contract.py`。
不改既有 witness、CUDA adapter、runner、源码封存或实验产物。

本模块检查可执行的数据合同，不执行模型、不验证物理收益、不开放 P1/P2。
P1 尚未通过；即使结构完整，`P2_admission=false`、
`physical_efficacy_verified=false`、`p1_empirically_verified=false`。
普通 Python 上的合成测试不是神经实验；真实神经重放仍必须使用主机 CUDA，
既有 `CudaReplay` 的 CUDA / frozen / eval 限制保持不变。

## 调用与退出语义

Python cwd 为 `le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m experiments.spectral_reachability.d2p_p2_comparison_contract \
  /absolute/path/comparison.json \
  --manifest-sha256 EXTERNALLY_RETAINED_PRE_OUTCOME_MANIFEST_SHA256
```

可选 `--p1-evidence /absolute/path/p1-report.json` 仅绑定外部 P1 报告内容摘要，
不信任其中自行填写的 passed/empirical 字段。该校验器不替代独立 P1 科学审计。

- 退出 1：数据非法，输出 `invalid_contract`，不返回部分配对成功。
- 退出 2：数据一致，但缺 P1 时 `blocked_p1`；给出报告时
  `p1_requires_independent_audit`。均不是 stage pass。
- 没有自动准入的退出 0 路径。调用方需要区分
  `data_contract_valid` 与 `P2_admission`，不得把结构通过变成效益结论。
- CLI 只读输入，结果只写 stdout；不建目录、不改报告、不选候选。
  JSON 重复键、非有限数和不完整输入拒绝。

API：`validate_comparison(data, expected_manifest_sha256=..., p1_evidence=None)`，
失败抛 `ValueError`；返回值与 CLI 报告相同。输入在拥有的副本上检查。

## 输入与封存

顶层对象含 `manifest`、`rows`，可选 `physical_observations`。
manifest 摘要采用既有 witness 的 `digest`：排序键、紧凑 UTF-8 JSON、
拒绝 NaN。调用者必须保留 outcome 打开前的外部摘要；校验时重新对挑选后的
manifest 算摘要不构成预封存证据。哈希只能绑定内容，不能证明时间、来源或诚实性。

manifest 字段：

- `schema="d2p_p2_manifest.v1"`。
- `selection={"rule":"pre_outcome_frozen","outcome_accessed":false}`，严格布尔。
- `arms`：`composition`，以及至少一个 `flat_search` / `no_composition_sra`。
  支持同时纳入两个基线；只提供一个不代表两个都已验证。
- `provenance`：沿用 witness v2 的完整六项 SHA256，含 checkpoint、encoder、
  source、protocol、metric、candidate bank；所有方法必须一致。
- `action_domain`：沿用 raw 环境动作的单位、上下界、action block、归一化摘要。
  不允许把编译参数、latent waypoint 当成真实动作。
- `budget`：见下节。
- `cases`：非空列表，每项含唯一 `case_id`、完整 witness `root` 和非空 `slots`。
  root 绑定 root ID、整个 RGB 上下文摘要、goal RGB 摘要和完整 latent/action context。
  重复 root/goal 三元绑定拒绝，不能伪造独立重复。
- 每个 slot 含唯一 `slot_id`、正整数 `primitive_steps`、
  `candidate_ids={arm: candidate_id, ...}`，覆盖每个 arm。
  每个 arm 内 candidate ID 不得重复。slot 的全部候选在 outcome 前声明。

每个 row 含 `case_id`、`arm`、`budget`、`actual_wall_time_s`、`candidates`。
每个 candidate 含 `slot_id`、`candidate_id`、完整 v2 `receipt` 和原始 `edges`。
只报获胜者、缺根、缺 arm、重复 row、缺候选、额外候选、按结果排除失败样本均拒绝。
配对使用 `(case_id, arm, slot_id)`，不依赖数组顺序；候选 ID 由封存映射指定，
不同方法的动作可以不同，但对应槽位的 raw 环境步数必须完全一致。

## 精确动作见证与长度

对每个 candidate 重新调用既有 v2 compose 验证动作边界、动作拼接、完整中间上下文
连接、每边摘要、动作摘要和完整 receipt 摘要，结果须精确一致。任何裁剪、补零、
容差替换或重新排序都不会隐式发生。root、goal、context、provenance、action domain
须与 manifest 逐项一致；edge 若显式给 goal 摘要也必须一致。

`composition` 必须至少两条 primitive 边；无组合方法用一条包含完整动作串的
flat witness，即使动作串本身有多个 packet，也不把它记为组合搜索。
每个候选的 `len(receipt.actions)` 必须等于 slot 的 `primitive_steps`。
共享 action block 后，这也固定总 primitive token 数；`max_actions` 只是上限，
不等于实际长度，不能用 H2 上限掩盖 depth1 输出。

重要边界：这里使用 edges 里保存的 output context 重新构建收据，**不调用模型**。
因此验证的是动作／上下文／哈希内部一致性，而不是“模型确实生成这些状态”。
`model_replay_independently_verified=false` 始终保留。伪造者若重写所有内容及摘要，
需由外部封存、来源核验和独立 CUDA replay 发现，本模块不声称解决来源真实性。

## 完整预算

manifest 和每个 row 的 budget 必须键集合完全一致、数值精确相等：

```text
predict_calls, predict_samples,
planning_predict_calls, planning_predict_samples,
replay_predict_calls, replay_predict_samples,
encoder_calls, backward_calls, optimizer_steps, flops,
attempted_expansions, candidate_evaluations, physical_env_steps,
wall_time_limit_s
```

除正有限 wall-time 上限外，均为非负严格整数，不接受 bool。
总 predict calls/samples 必须分别等于 planning + replay；失败调用不应从调用方
账本退款。`candidate_evaluations` 必须等于每根封存的候选 slot 数，不能用 winner
数冒充完整候选评估；搜索中间扩展另记 attempted expansions。需要改变候选资格或
停止规则时，应另行预注册合同，不能事后丢弃低 depth 候选。

`actual_wall_time_s` 必须是有限非负数且不超过共同上限，不要求不同方法实际耗时
相同。数值均由外部真实 instrumentation 提供；未知 encoder/FLOPs 等不能填假零，
不能据此声称公平比较。该严格合同没有“只配 planning samples 就算等预算”的通道。
它检查所报完整账本，不独立测量算力或验证硬件记录真实性。

## 物理记录与收益边界

不提供 `physical_observations` 时，物理步预算必须为 0，输出物理配对数 0。
模型 progress/reconfiguration 标签、预测 Q、精确模型回放不能代替物理观测。

若提供物理记录，则必须覆盖全部 `(case_id, arm, slot_id)`，不得有重复或漏项。
每项包含上述三个身份字段以及：

```text
root                       与候选完整 root 相同
receipt_sha256             与候选完整 receipt 摘要相同
executed_actions           与候选 raw 动作串精确相同
execution_artifact_sha256  外部真实执行产物摘要
metric_sha256              与预声明指标摘要相同
objective                  有限实数，禁止 bool
success                    严格 bool
```

每根每 arm 的物理步预算必须等于全部候选 raw 步数之和，不得把物理执行藏在零预算。
执行摘要只是绑定声明，本模块不读取／认证物理产物，也不做显著性、收益方向或
机制判定。即使配对物理记录存在，`physical_efficacy_verified` 仍为 false。

## 对新 CUDA 搜索记录的处理

只读核对 `D2P_PROGRAM_SEARCH_CUDA_INTEGRATION_20261026.md` 与
`results/pusht_program_search4_v2_20261026_cuda/report.json`：
`integration_completed`，8 gates 为 true，320 planning + 7 full replay；
selector 是跨 depth 的 best nonempty visited，不是固定 H2。
root2 的 depth1 结果不能填入固定 H2（PushT 为 10 个 raw 环境步）slot；
不能补零、删根或按 outcome 挑剩余三根。
既有报告显式 `fair_total_budget_comparison=false`，没有本合同所需的完整
编码、反向、FLOPs、wall-time 配对账本及无组合各 arm 候选，不能当成 P2 对照通过。
不修改该已封存报告，也不将它迁移或包装成“已公平比较”。

## 验证证据

cwd `le-wm-main`，普通解释器，仅合成数据／接口回归：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_d2p_p2_comparison_contract.py \
  tests/test_d2p_composition_witness_v2.py \
  tests/test_pusht_composition_cuda_replay_v2.py \
  tests/test_run_pusht_composition_replay_cuda_v2.py
```

113 passed，其中新增合同 51 项，既有相关回归 62 项。新增模块前测试曾因模块
不存在而失败，随后实现通过。覆盖外部 seal、根／目标／上下文错配、动作篡改、
重封后的 depth1 混入 H2、断开上下文、候选缺失／重复、按身份而非顺序配对、
完整预算、漏报重放、物理动作错配、缺 P1 非零退出和自报 P1 不得准入。
没有在本任务运行新 CUDA 实验，没有据逻辑测试声称物理收益成立。

收尾复核：上述四文件回归再次得到 113 passed（0.48s）。主线程另反馈联合
176 tests 通过；该数字记录为主线程复核反馈，不冒称本模块独立执行的结果。
本任务写入范围仅本合同模块、对应新测试和本文档，交付后不再追加测试或扩展接口。
这些测试均不能证明预测重放、P1 实证或 P2 通过；当前只验证声明数据合同。
