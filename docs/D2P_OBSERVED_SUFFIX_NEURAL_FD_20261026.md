# Observed suffix：central neural finite difference 实现与启动交接

2026-10-26。**仅实现，未启动 CUDA、未分配模型、未产生 neural-FD 结果。**

新增 runner：`le-wm-main/experiments/spectral_reachability/run_pusht_observed_suffix_neural_fd.py`。
新增测试：`le-wm-main/tests/test_pusht_observed_suffix_neural_fd.py`。
本任务未修改父线程、物理 FD、observability guard 或任何封存源码。

## 要区分的证据

父线程物理 FD 已完成 `results/pusht_observed_suffix_physical_fd4_v1_20261026_cuda/report.json`，72 真实 suffix/72 CUDA encodes/0 predict、7 完整性门通过。该报告中的 AD 是 observed one-shot 保存的 model derivative，physical FD 则通过真实动作后的 RGB 编码得到；它们不一致本身尚不能区分微分实现问题与模型局部物理不匹配。

新增 runner 对同一 observed context 的冻结模型做 central neural FD：

```text
J_neural_FD[:,a] = [f_c(+ε e_a) − f_c(−ε e_a)] / [2*float(float32(ε))]
ε ∈ {0.01, 0.05};  a ∈ {0,1}
```

导数坐标是 atanh suffix parameter，不是 live tanh 命令；分母与父线程 physical FD 完全相同。一个 zero 加上两尺度×两轴×正负号，共9次单样本模型预测。H=1，无 backward、无重新计算 AD、无 GN、无优化。

## 固定执行面

4 个已知校准根，各自只保留原封存 selected 和 prefix_0，合计8个上下文。每个上下文：

1. 按原 root provenance 重建并 warm up；验证 goal RGB。
2. 只执行一次原封存前缀 packet，验证 prefix RGB hash 和完整7维 state 与原 execution 的 stage0 精确一致；总计8次 exact prefix replays。root 收集和 warmup 沿用现有接口，不计入这8次前缀重放。
3. 通过父线程 `run_pusht_observed_prefix_prediction.observed_context` 生成3帧/对应 incoming action 的真实观察上下文，再用 p1 `_encode_context` 编码；不使用 imagined successor。
4. 严格读取 physical FD 报告中既有 probes/float32 executed_params，以 `p1._suffix_terminal` 做9次 CUDA float32 预测。**不执行任何物理 suffix**，无需新的物理终态/RGB 编码。
5. zero 预测与绑定 observed stageplan 的 `nominal` 比较，保存 exact 与 max-abs；exact 不成立则完整性失败，不把旧 AD 与不一致上下文的 FD 混作证据。
6. 对两尺度分别报告 neural-FD vs saved AD、neural-FD vs physical-FD、saved AD vs physical-FD；同时报告固定左乘 Q/projection 后的 neural-vs-AD/physical 比较。physical FD 直接从同报告保存 vectors 重算，不调用环境重测。

总预算：72 次 predict 调用、72 predict samples、8 个 evaluation scope；8次 observed context 编码不算 predict。没有额外 nominal forward（zero 是这9次之一）。neural 预测封装在 no_grad 下；完整输出逐 probe 落盘。

## 生产路径和失败关闭

- CLI 强制 `--physical-fd --observed --execution --policy --output`；无 CPU/device 开关、无 epsilon/阈值/筛选选项。
- 核验 exact schemas、completed、非空全 true gates、P1_admission=false，校验 physical report 确实绑定给定 observed/execution、compilation 和对应 stageplans；统一4个 root 身份、顺序和原 selected，拒绝 duplicate root/candidate/probe。
- 预先校验 stageplan 的 J/nominal 与 physical report 保存数组一致，projection 与原 compiled projection 一致，全部9个 physical probe 的参数、prefix RGB/state、状态及 latent 维度完整。
- weights.pt/config.json 必须出现在已有 bindings，并在运行时校验 hash；将新输入 physical-FD report、根 bank/upstream 也纳入 bindings。现有 upstream source seals 先验证，再将实际 helper 与上游 manifest 的 source 并入新 seal。
- 新 output 目录独占；所有 bindings、预声明协议与源码 seal 完成并验证后，才调用 `require_cuda_allocation('cuda')`、加载冻结模型；无 CPU neural fallback。
- 参数哈希、全部 buffers 的名字与值、frozen/eval/CUDA 状态运行后复核，输入 hash 与 source seal 再校验。无阈值调整，无 selector 变化，无 future-physical oracle policy，无 P1 放行。
- central comparison 输出 MSE、相对误差、flattened cosine、整体/列范数；若比较基准范数为零，relative error/cosine 用 null，保留原始矩阵，不静默加 epsilon 或将不可定义 cosine 记零。
- 报告 `saved_nominal_exact` 是上游身份/数值完整性检查，不是 AD/FD 相似度阈值。AD/FD 差异作为原值保存，不按任何新成功阈值筛根。
- 每个 probe 保存 root/candidate 身份、prefix RGB hash、参数、完整 prediction vector；每上下文保存 raw matrices、zero check 和两尺度比较；逐根及总 report 保留 predict accounting。运行阶段异常会保留已有 probe 收据并写 integrity_fail；加载/源封存等更早失败可能只有 protocol/seal，无完整 report，不能称完成。

## 精确启动命令（父线程协调后执行；本任务没有执行）

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m experiments.spectral_reachability.run_pusht_observed_suffix_neural_fd \
  --physical-fd /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_observed_suffix_physical_fd4_v1_20261026_cuda/report.json \
  --observed /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_observed_oneshot_suffix_control4_v1_20261026_cuda/report.json \
  --execution /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_observed_suffix_neural_fd4_v1_20261026_cuda
```

output 必须不存在，失败结果不得覆盖或删除后重用。运行时如 upstream source 已变化，应停在 seal 检查，不能放松或跳过认证。保存图像和新物理 suffix 不在任务范围，新增磁盘主要是小型源码封存和 JSON vectors/matrices。

## 已执行验证与尚未验证的部分

使用 experiment-code 技能组织固定运行合同、分层测试和结果交接；按用户范围不生成图像或启动实验。

```bash
.venv/bin/python -m pytest -q tests/test_pusht_observed_suffix_neural_fd.py
```

13 passed。包括实际 production evaluate_probes 调用循环的9次调用与保存、float32 参数分母、零范数、不完整/重复/损坏 probe、失败传播、import/help 不载入 torch，以及读取**当前真实 physical FD 的全部8上下文与其绑定 observed stageplans**验证数组/参数合同。最后一项只对已有 vectors 跑数值比较，不做模型推理；自比较不能证明 neural-FD/AD 或物理一致。

这些不是 AST-only 检查，但也不是 live CUDA 证据：模型加载、真实 replay、72次 `_suffix_terminal` 及实际 neural FD 是否与 AD 一致仍待父线程执行。若两尺度 neural FD 接近 AD 而均与 physical FD 不同，只支持本批局部的模型—物理响应差异；不能穷尽排除所有代码错误。若 neural FD 与 AD 差异明显，应先查尺度截断、float32 消减、非光滑点及输入/执行身份，不自动宣告 AD bug；两尺度有助区分但并非证明。RGB 不可见情形和物理解释留给父线程 observability/physical-FD 分析，本页不重定义成功或机制。
