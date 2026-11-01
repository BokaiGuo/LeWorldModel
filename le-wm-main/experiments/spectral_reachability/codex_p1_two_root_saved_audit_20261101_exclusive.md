# 两根 CEM 保存结果独立审计（只读）

## 范围与结论

审计对象是 root0 和 root1 的 `RUN_MAIN01` 保存报告、prepare、
物理 journal、CEM journal、已封存节点账本、节点 seal 和逐步 PNG
文件。原始报告未修改；未重新运行 CUDA、模型或物理环境。

两个 report 的 SHA256 分别为
`a973757107da8b98aed242cfb01d5a9079115150d0871c26fe255cc7b960e356`
和 `0149183cf66149344f28eb3689f80d08149538bc8ad4143be78346c30b8efeb2`，
与指定值一致。修复后的独立脚本和五项 CPU 测试通过。

| 指标 | root0 | root1 |
|---|---:|---:|
| 保存的原生首次成功控制步 | 15 | 6 |
| 已封存节点 | 0、5、10 | 0、5 |
| 各节点 action encode / predict / predict samples 账面值 | 195、130、65 | 195、130 |
| 终点联合四维 XY 误差 | 15.265970099414307 | 17.172400589753085 |
| 终点原生角度项 | 0.07747056792508378 | 0.0001533068597270315 |
| 终点七维距离（并非成功阈值） | 194.6973497120632 | 270.2110866780141 |

脚本核验物理日志 SHA 和末端、全链哈希，CEM 事件链，五个节点账本
SHA 和链、节点合同哈希、预算账面累加；按保存的分数重算四轮精英、
均值、标准差与跨轮传递、64 候选全局最优和重放分数；将所选第一包、
节点 seal、物理派发、返回及报告的执行动作对应；逐步按保存状态和
目标重算回报、终止和原生成功，并检查先前步未成功。物理 PNG 文件
的文件 SHA 与观察回执一致。prepare 和报告的协议、根/目标绑定及
模型身份与节点身份一致。

该审计只验证现存记录及其相互关系：未重算模型分数、未证明模型或物理
当时独立执行、未重新认证全部 prepare 来源文件、未解码 PNG 重算像素
哈希。CEM journal 自身有哈希链，但本审计没有发现并验证一个由 report
直接固定的 CEM journal 文件或末端哈希，不能说它与 report 具有同等
强度的外部哈希锚定。节点账本明确是 `closed_unverified`，实际使用和
覆盖均未独立验证；不能由此声称 FLOPs、完整成本或等成本。
两根均为已知根且报告状态为 `recorded_known_root_not_admitted`，
不是独立泛化确认。

## 物理 journal 审计器故障

首版独立审计器把物理日志的哈希输入错误编码为紧凑 JSON；实际物理
日志使用 `json.dumps(..., sort_keys=True, allow_nan=False)` 的默认
分隔符和 ASCII 转义，哈希的是不带 `event_sha256` 的对象，且不附换行。
CEM 和节点账本则用各自的紧凑 JSON；现已按格式分别计算。修复的是
新审计器，不是日志或报告。篡改和截断测试对三种日志格式均拒绝。

## 继承协议字段的范围与影响

两根 prepare、report 和物理日志开头保存了同一类混合协议：CEM
`configured().protocol()` 先取旧多臂协议，仅更新部分字段。下列
问题在两根均存在，并非 root0 独有。

1. **`node_caps` 不一致。** 旧通用 `node_caps` 对 action encode、
   predict 和 predict samples 均为 171；实际边界 0 的 CEM
   `node_costs["0"]`、已封存节点合同和账本均为 195。边界 5 为
   130，边界 10 和计划中的边界 15 为 65；实际未启动节点不能
   算作成本。旧 cap 另保留 backward 208、VJP 208、Jacobian 1、
   SVD 1、replay 和 selection 各 24 等多臂数字，而真实 CEM 节点
   合同相应轴为零。171 与 195 的冲突不可描述为同时遵循两套 cap。
   代码中 CEM 自行构建 per-boundary `node_contract`，以其 `node`
   level 建立 durable driver，结算时核对 `node_cost(boundary)`；
   保存 ledger 合同和账本与这一实际路径一致。旧字段会误导预算
   解读与自动消费，不能作为该 CEM 节点的有效门限。
2. **`module_hooks_gate_each_node=true` 无 CEM 节点证明。** 这是旧
   多臂 runner 的协议描述；旧实现有独立的 hook audit 路径，而本
   CEM 的 `execute_one` 使用 `LeafAdapter` 和 durable driver，并在
   `issue` 中结算账本和校验 CEM 事件。该适配器明确不拦截内部模块、
   不证明 hook 覆盖，receipt 的 `actual_usage_verified` 和
   `coverage_verified` 均为 false。本审计未见本 CEM 路径逐节点
   独立 hook gate 的证据；不能据继承字段宣称模块完整计量或等成本。
3. **旧多臂描述不适用。** `initializer_labels` 仍列出 structured、
   random、native、immediate_progress 的旧定义；`extra_policy`
   宣称部分 arm 的 32 次 model-winner 更新；`common_suffix` 仍写
   两步优化。实际 CEM 事件记录的是单臂，每节点四轮、每轮 16 个
   候选、四个精英、64 次打分和一次 winner replay。旧字段不能
   作为 CEM 初始化、优化算法或其他 arm 已执行的证据。
4. **合同内部遗留 arm 标签。** 各节点合同保留四个旧 arm 配置槽，
   driver / receipt 的账本 arm 名为 `native`，而 CEM 事件及 episode
   标识为 `full_cem`。这反映适配层的命名复用，不证明四个 arm
   执行，也不能把 `native` 解释为安装的原生强基线。

影响是**协议元数据和成本认证的解释边界**：不能把继承字段当作实际
CEM 配置或独立模块覆盖，不能据此声称公平匹配或完全预注册的干净
CEM 合同。这些矛盾不单独推翻本次对已保存动作、状态和早停计算的
一致性核验，也不能反过来证明物理真实性或补造历史计量。

未来如修复 runner，应使用新的独占文件、预先确定的新协议和新运行
产物，并对旧字段删除或明确标注不适用；不得覆盖本次 prepare、
report、journal 或节点账本，也不能事后声称历史运行遵守新合同。

## 复核命令

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  le-wm-main/.venv/bin/python -B \
  le-wm-main/experiments/spectral_reachability/codex_p1_two_root_saved_audit_20261101_exclusive.py

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  le-wm-main/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  le-wm-main/tests/test_codex_p1_two_root_saved_audit_20261101_exclusive.py
```
