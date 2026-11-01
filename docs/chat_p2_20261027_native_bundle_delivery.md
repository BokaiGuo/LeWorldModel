# P2 继续交付：真实 producer 绑定与原始入口记账

本轮补充上一版 saved-graph 生命周期未覆盖的真实 producer 准备：直接复用
frontier CUDA v2、联合 H2 SRA v2、full20 CEM/random v2。没有修改任何既有
producer、源码 seal、结果或先前本线程已封存文件。不使用子智能体，不运行
torch forward、CUDA allocation 或 physics，不接管主线程 P1/代码审计。
experiment-code 技能用于固定接口、保留结果和回归；不新增科学阈值或绘图。

## 新源码与接口

源码 `le-wm-main/experiments/spectral_reachability/chat_p2_20261027_producer_bundle.py`，
测试 `le-wm-main/tests/chat_p2_20261027_test_producer_bundle.py`。

它是现有 producer 的只读适配器，不重新实现组合、SRA 或采样算法：

1. prepare 分别调用原生 load_prepared（包含其封存与输入检查），严格比对原始
   root 顺序、root ID、完整 context/history、goal/root RGB、动作域、metric SHA。
   不因为 latent 相近就连接，不通过换 goal 宣称新根。独占封存三套 prepared/
   protocol canonical SHA、所有本地 source 路径和字节快照、本模块与依赖。
2. verify 需要外部提供 bundle canonical SHA，逐字节检查 current source 与快照，
   再执行三套原生 verify 并重建完整 bundle；不是仅比较自述 hash。
3. run 只执行 saved validation，绝不调用任何 native cuda()。三套 producer
   逐项保留成功或拒绝，不因组合缺失删掉该分母。不读取物理 outcome 选规则。
   原 exporter 通过后才投影完整候选分母和原始 actual_model_entries；不把候选
   dispatch 账本替代真实入口记录，也不把历史记录冒称本轮新神经测量。
4. consume 重新运行完整验证并核对除本次 wall time 外的完整审计；只有三套
   exporter 全部通过才返回 engineering_saved_bundle_verified，仍不授权执行。
5. 独占 started marker 阻断重试；异常（含 KeyboardInterrupt）保留原对象并尝试
   failure.json，失败记录写入异常不能替代首个异常。硬件/磁盘/进程死亡不能
   保证结果落盘，marker 永远不是成功证据。

新增 entry_counts 分别保存 predict 与 action_encoder 的 API calls、batch samples、
context tokens、失败 API calls。失败记录照计不退款。RAM metadata hook 测试通过
现有 actual_entries 包装器输入 batch4 成功、batch2 失败，得到 predict calls2 /
samples6 / tokens18 / failed1，action encoder calls2 / samples6 / tokens18 /
failed0，并检查 hook 恢复。这个测试没有真实神经运算。

## 原始声明差异（每根，非本轮新测量）

| Producer | 原候选分母 | 声明 predict budget |
| --- | --- | --- |
| frontier CUDA v2 | 联合80，条件组合40 | standalone 168/168，两臂共享 compile4 |
| no-composition SRA v2 | 40 | 88 calls / 88 samples |
| full20 sampling v2 | CEM40，random40 | 每臂84 calls / 84 samples |

不修改上述既有算法配置。声明表不构成实际入口证明；FLOPs 与上游 RGB 共同
前置成本仍 unknown/null，fair_total_budget_comparison=false。独立归属与共同
成本不可相加重复计费；不同 algorithm/proposal 支持也不能由动作域相同推出等价。

## 实际现场验证

真实资产 prepare/verify 通过，四根 cross-producer identity 一致，绑定100个本地
源码文件。结果是已用 engineering calibration roots，不是 prospective fresh roots。

bundle canonical SHA：
`0d332bb659c03d4ed947445e57c7da81361e00ac664390748716e6b9def70974`。

结果位置均在 `le-wm-main/results/`：

- `chat_p2_20261027_native_bundle_prepared.json`：真实资产绑定及源码快照。
- `chat_p2_20261027_native_bundle_saved.json`：完整三项分母，无筛选。
- `chat_p2_20261027_native_bundle_saved_started.json`：独占运行标记。
- `chat_p2_20261027_native_bundle_consume_blocked.json`：组合完整报告缺失的拒绝。

SRA 全4根的原 exporter 通过；从历史实际入口复算，每根 predict88 / batch88 /
tokens264，action encoder 同数，每根完整40候选。CEM/random 全4根两臂 exporter
通过；每根合计 predict168 / batch168 / tokens504，action encoder 同数，两臂各40。
这些是 saved 记录一致性，不是独立神经/物理执行真实性证明。

组合 v2 prepared01 的原 exporter 拒绝 `bounded regular saved JSON`：该目录缺
完整运行 report/commit。保留原拒绝，不替换为 v1 或 paired 的结果，不重封。
故整体状态 incomplete_native_exports，consume 确实拒绝，不把另两项成功算组合
通过。P1/P2 admission 均未获准，本轮无新神经/物理效益证据。

## 命令

最终联合回归 **408 passed in 23.55s**：新增 producer-bundle 测试、上一轮
lifecycle 测试、saved graph、prospective queue、frontier CUDA v2、SRA v2、
full20 sampling v2 原测试共同执行。测试命令与上一版说明相同，额外加入
`tests/chat_p2_20261027_test_producer_bundle.py`。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；以下 MODULE 指
`experiments.spectral_reachability.chat_p2_20261027_producer_bundle`，各命令用
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m MODULE` 执行。

```text
prepare --composition results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01 --no-composition-sra results/p2_no_composition_sra_20261027_v2_prepared01 --strong-sampling results/p2_full20_sampling_20261027_v2_cuda_v1 --output results/chat_p2_20261027_NEW_bundle.json
verify --bundle results/chat_p2_20261027_NEW_bundle.json --sha256 EXTERNAL_CANONICAL_SHA
run --bundle results/chat_p2_20261027_NEW_bundle.json --sha256 EXTERNAL_CANONICAL_SHA --output results/chat_p2_20261027_NEW_saved.json
consume --bundle results/chat_p2_20261027_NEW_bundle.json --sha256 EXTERNAL_CANONICAL_SHA --report results/chat_p2_20261027_NEW_saved.json
```

run 的 incomplete_native_exports 退出2；这不是运行 CUDA 失败，而是完整 saved
输入尚不存在。消费失败不允许重新命名/筛选结果后冒称成功。

## 未完成事项与放行

本轮填补的是三套真实 producer 的统一身份/来源绑定、完整 exported candidate
分母和 calls/samples/tokens 投影，未填补全部科学验证条件。原精确放行条件仍为
`docs/chat_p2_20261027_delivery.md` 最后一节，尤其：

- 真实外部 P1 六 gate、预声明指标统计和 MAIN authority；不得自签通过。
- freshroot 全历史排除、物理别名/lineage 审核、队列身份到真实 executor 的绑定。
- 组合 v2 完整主机 CUDA 运行、全候选 witness/边界/replay 与模型/source 不变证明。
- 共同完整资源上界、预算停止条件与 FLOPs/前置/失败/物理成本实测；本 bundle
  不声称现有168/88/84配置已是有效全预算对照，更不自动重调配置以追求好结果。
- 原协议的非贪心机制、实际任务收益与配对不确定性。

因此个人目标仍在推进，不标记全部完成；无需等待 P1 也已完成本轮可安全进行的
真实资产只读验证，但不得启动确认性队列或声称 P2 机制成立。
