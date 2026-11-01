# 2026-10-28 并行任务与 MAIN 实际执行

本记录不替代研究准入。保留旧源码、结果和失败；神经实验只用主机 CUDA，不训练冻结模型，不降低科学门槛。子智能体不派生子代，MAIN 负责审计和实际执行。

## 独占任务

| 执行者 | 独占任务 | 边界 |
| --- | --- | --- |
| Nash | `agent_p2_runtime_guard_20261028_v1` | 接实际 callback 与资源守卫；不以 RAM 测试宣称 P2 成立 |
| Bohr | `agent_p3_runtime_contract_20261028_v1` | 同任务程序类型、horizon、depth 合同；不以跨根路由代替 C3 |
| Pauli | `agent_moving_CEM8_physical_20261028_v1` | 八固定 anchor 精确重放与实际 CEM 后缀适配，不执行物理 |
| Hegel | `agent_CEM8_release_20261028_v1` | 真实 own-consume 持久回执，绑定实际源码与产物 |
| Jason | `agent_freshroot_authority_20261028_v1` 与独占说明 | 原历史 reset/load 证据请求与验证，不猜测历史来源、不重复全盘扫描 |
| Poincare | `agent_observed_CEM_physical_20261028_v2` | 修复 MAIN 已定位的准备文件工程尺寸限制，保留 v1 |

任务已发送，交付仍须 MAIN 审读与验证；本表不是完成声明。

## MAIN 当前已验证

- 八 anchor CEM：MAIN 52 项 RAM 测试通过；真实 prepare 1,450,521 字节，verify 通过。
- 主机 CUDA 实际完成 520 次 predict、520 次 action_encoder、520 次 pred_proj，所有 attempted=completed。报告：`le-wm-main/results/agent_moving_CEM8_20261028_v1_cuda_MAIN01/report.json`，SHA256 `660751b5784d25cce652d0ecb06ffb4cf3810a24ed135755606c97c3f957da0a`。严格 saved consume 已启动，验收结果另行记录。没有新物理结果，不能报真实控制收益。
- 四 observed-prefix 物理适配器：MAIN 74 项 RAM 测试通过，但真实 prepare 被工程尺寸门阻止，未进入物理执行。实际 material 540,309 字节，大于 PREP_CAP 524,288 字节；其中 bindings 413,851、prefixes 111,060 字节。磁盘约 60 GB 可用，不是磁盘不足。v2 仅将 prepare 上限预先改为 1 MiB 并补尺寸回归，不修改动作、Q、native 阈值或分母。

## MAIN 保存证据审计

`le-wm-main/results/audit_prefix_span_MAIN_20261028_v1_MAIN01.json` 的四个 D2P bank 均为 10 维参数、logit 提案 span rank 3。遗漏方向仅占 Jacobian Frobenius 能量约 0.56%–0.88%，不能据此断言它们能改善实际控制。

四项 winner 与 bank 内 greedy 完全相同，且模型首段 Q 均相对当前状态下降。这是当前候选库没有选出非贪心动作的具体证据，不等于理论不可行，也不是已证明的接线错误。后续若扩大候选库须事前固定并完整报告新增算力，不按物理结果挑模式。

MAIN 随后完成八 anchor 严格 saved consume：退出码 0，`errors=[]`，状态 `teacher8_CEM_commands_verified_not_admitted`，固定八项计划全部保留。持久回执封装仍由 Hegel 实现，本次 CLI 验收不能冒充已保存的独立回执文件。

上述八 anchor 仍来自两个已知校准根，不是独立确认；P1、原 C2/C3 和研究完整闭环仍未通过。
