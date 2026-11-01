# P1 分段预算现场 receipt 修复：RUN05

本轮目的仅是修复现场 CUDA / 冻结模型身份未持久化的问题，不改变
候选、动作、选择规则或 P1 判据。

## 代码级问题

RUN03 暴露出第一个问题：运行时已经检查并在 stdout 打印了
`host`、`model_before` 和 `model_after`，但 `physical.execute()` 在这些
字段加入内存对象之前就把 `run_report.json` 写盘。因此 saved audit
无法独立确认现场使用的是主机 CUDA 和同一冻结模型。

修复后又发现第二个问题：`physical.execute()` 返回的是
`status/sha256` 摘要，不是包含 `shards` 的完整报告；receipt helper
错误地把摘要当成完整报告比较，导致 RUN04 的 24 个槽都完成后，最终
receipt 发布失败。RUN04 保留为代码失败证据，不作为已消费结果。

当前实现将运行时身份字段原子写回新生成目录中的
`run_report.json`，并重新计算报告 SHA；只比较摘要中的 `status`，
不再错误读取不存在的 `shards` 字段。

## RUN05 结果

主机 CUDA 实验完成 `24/24`，冻结模型未训练，前后 receipt 一致：

- device: `cuda:0`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- `original_state_dict`:
  `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`
- `all_buffers`:
  `ca7a531449da81ea2dee6eb79716f4bf840803b9df4dd046b8a591e8f7b8bd78`
- online midpoint events: `24`
- saved event audit: `24/24` passed
- receipt consumer: `24/24` passed

RUN05 的保存消费文件与此前 RUN03 的消费文件逐字节一致。因而这次
修复只补全了证据链，没有改变控制结果或成功计数。

## 证据边界

这仍是两个已知根上的 parity / feedback 工程验证，不是独立新根、
等预算强基线或 P1 机制确认。`P1_admission`、`independent_confirmation`
和 `equal_budget_certified` 均保持 `false`。selected 的既有成功不能
仅由本轮 receipt 修复解释为后续控制能力改变。

## 文件哈希

- prepare:
  `results/p1_split_budget_online_20261031_prepare_RUN05.json`
  `0da343ecdbbebff7e419fff946c06e6ebdc3c2a3a37607f1fc450d89cb2c96e0`
- run report:
  `results/p1_split_budget_online_20261031_run_RUN05/run_report.json`
  `13f000625774ab9ff6a90161b8671b474151e6af10f58d0a3b2c63b65d481b94`
- consume:
  `results/p1_split_budget_online_20261031_consume_RUN05.json`
  `47878f5efc9514cdd97a016a1663dae6201f92e0b1aa4dc0fdcc7c9b4af77eef`
- event audit:
  `results/p1_split_budget_online_event_audit_20261031_RUN05.json`
  `4474cd14a81587bac07af6a6de38932c892bca2e023684cb5057f2bc4093ae7f`

新重选前缀的 1008 条 H10 共同后缀及真实 cp5 各自规划配对已经
封存，见 `D2P_P1_UPHILL_H10_COMPLETE_RESULT_20261031_MAIN.md`；
分段反馈 96 条共同后缀另见
`D2P_P1_SPLIT_BUDGET_CROSSOVER_RESULT_20261031.md`。这些都使用
已知根，前缀生成的完整资源和独立来源尚未认证；本修复不能替代
独立来源认证或公平基线。

## 重复执行事件

本次错误地把 RUN05 的 parity 重放当成新前缀，又准备了一个与
既有分段反馈 crossover 动作相同的 96 分支面板。发现重复后立即
中止：27 个记录中 26 个完成、1 个中断，报告状态为
`partial_or_failed`。目录保留，不能用作完整面板或新增独立证据。
后续任何重跑须先对 prefix、task、suffix 身份和动作字节排重。

2026-10-31 补充：对两份真实 prepare 直接比较，24 个完整 anchor
以及 8 个完整 bank 均相同。现在在线 crossover 入口会校验既有
prepare 的 SHA，且在完整 anchor 和 bank 都相同时拒绝生成新 prepare
或运行；实际 prepare 调用已验证拒绝。该保护只识别这一个既知面板的
严格完全相同情形，不能替代对其他来源、改序或部分重叠面板的全局排重。
修改入口后，旧的重复 prepare 源码绑定自然不再匹配；已封存结果没有改写。
去重和 receipt 相关合成测试共 5 项通过，不代表科学实验通过。
