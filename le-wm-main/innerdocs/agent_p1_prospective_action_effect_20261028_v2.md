# P1 prospective action effect v2：真实 native/CUDA 生产 adapter

独占新增同 stem source/tests/本文。v1 三文件及所有旧模型、源码、结果、gate、阈值不变。本版本不是另一个 RAM 接口草稿：`run` 已直接调用原物理 `episode/place/dispatch/observe/native_hooks`、官方环境 factory、原 image transform、冻结模型 `encode/predict`。worker 不执行任何新 GPU、physics、CPU 神经或网络实验；MAIN 审读后在主机执行。无新增 IPC/审批清单；正常主机执行授权即可。

## 已实现的生产路径

固定8个 anchor = 两个 known root × off/on × Z/S；每个沿原 selector 的 stage15 model-only winner 重放到20。原选择 `[3,8,13,18,21,25,31,36]` 来自原冻结 selector，不按任何物理/误差结果重选。新时段保持同一5词 float32 raw signed bits，最终固定40槽/80对。

1. `prepare/verify` 复用 v1 的完整 saved teacher history 重建及原 physical prepare 重建。原 P1 三帧已经全是真实5/10/15，动作 history6..15 也来自实际记录；不做 mid15 替换，不借 P2 的 actual-mid/own-imagined 结果解释旧 P1 的12反转。固定原 physical prepare SHA `b125d28c4d931dca7661233d5a81e0da890ef53e2a55ceef55172134d0b08f5a`，不访问 fresh reserved。
2. 每个新分支重新构造环境并从原 reset recipe 重放 warm10 + prefix5 + 原封存选择5。前15步逐字节核原51张 RGB/15条 raw/state/snapshot；不是 state7 teleport，也不复用旧终态作为新标签。leader 的实际20状态、完整已记录 snapshot、raw20、PNG/pixel SHA 先耐久记录。其余四个环境到20必须与 leader canonical 相同（含 signed zero），才允许自己的词。
3. leader 实际20 RGB 在 CUDA 编码一次，teacher 输入为 `[z10,z15,z20]`，历史11..20为原 prefix5 + 原选中 suffix5；原 float32 mean/std 仅归一化一次。`CudaModel.predict` 严格只收 schema/anchor/word/latents/history/raw5_bits；原 native evaluator 的 state、snapshot、reward、distance、native goal 均不进入模型 request。latent goal 仅用于固定 identity192 MSE 选择。
4. 全5次 `model.predict(z3, model.action_encoder(outgoing3))` 的完整三token输入输出及模块 trace 先保存。以原 float64 `mean((p-goal)**2)` 计算成本（192维**只除192一次**），数值精确相同按最小词索引打破 tie。seal 文件 fsync 后记录 seal 事件，然后才允许任何21步动作。不能观察其它分支后换词。
5. leader 环境继续执行固定 zero 的5步；另四个独立完整重放20后分别执行其它4词。每步保原 RGB guard、native early stop、actual raw buffer、goal/renderer metadata、状态/信息/回报、triplet。选择分支的真实25结果就是此闭环决策的结果，其余4条仅研究干预；它不是部署五次试完选最优。
6. 终态复用原 `off/on/off` capture，完整 native snapshot/状态/goal/renderer before=after、恢复 off 像素完全相同，不额外 physics。原 condition 和 alternate condition 各 CUDA 编码一次。alternate 编码失败保留已经合法取得的 original endpoint，只令 renderer 标签未知；物理/guard/replay失败不回填。第25步本身 terminal 可记录，25之前 terminal 不补步。

## 实际计账和持久性

`Meter` 只在真实 callback 入口/返回计账，失败不退款；encode API 不包含前置 preprocessing，predict API 也不把先发生的 AE 调用冒充 predict。AE/predictor/projection 各自 hook 单独计数、保存原 CUDA float32 输入输出；未完成的模块尝试保留。无 AD/Jacobian/SVD、训练、Q 拟合或 CPU neural fallback。

完整成功时的事前上限（不是本 worker 已发生费用）：

| 轴 | attempt/completed 各上限 |
|---|---:|
| constructor / close | 各40 |
| reset / set_state / goal_setter / goal_bind | 80 / 160 / 120 / 40 |
| public step | 1000（warm400 + prefix200 + 原suffix200 + 新extension200） |
| internal physics step | 10160（public10000 + reset内部160） |
| goal render / explicit render | 80 / 3240（27个位置×3视图×40） |
| encode API / encoder / projector | 各88（实际20×8 + 两终态视图×40） |
| predict API / AE / predictor / pred_proj | 各40；predictor samples也是40，三token不算三次API |

完整 callback、模块计数均由 consume 从真实事件重建/交叉核对。模型配置与权重各只读成一次 immutable buffer，同 buffer SHA 与反序列化（JSON / `torch.load(BytesIO(...))`）相绑定；保原 compatible remap 和空 missing/unexpected、模型 tensor hash 前后相同。失败 constructor 的 hook 前内部工作不能凭空签0；原 native constructor 的 inert/source 契约仍绑定。这里的 snapshot 是原 runner 完整记录的 snapshot，不声称记录了任何未暴露的 solver 内存；独立 reset + 全动作重放与 RGB/原快照双核查避免依赖 snapshot restore。

输出1GiB、final reserve64MiB、journal128MiB、磁盘 floor200MiB、wall3600s（含验证/加载）。exclusive flat 文件、owned inode/size append、hashchain、PNG/NPZ/压缩NPY、fsync、每分支 JSON 和 finally 的 partial report。close 可在耗尽后清理，不能借此继续 physics。SIGKILL/磁盘完全故障不承诺能补写最终 report；此前已 fsync 的 journal/回执仍在，不自动 resume/retry。

report 给 actual API/module/native counts、实际保存字节和 wall；`total_FLOPs=null`。不把 API 匹配或固定样本量说成 FLOPs 相等，旧40库/compiler等继承成本也不记成0。failure/partial 都保固定分母，不按误差/收益删槽、改根或改词；新未知数量不被强制设成旧4。

## saved consume 与解释边界

`consume` 只读新保存文件，先验完整 prepare/source/model/sourcepack 绑定，核 artifact 全集/SHA/字节数、canonical hashchain、callback 成对 attempt/return、job scope、所有模块/image/forward 回执交叉归属。原20步用显式**只读投影视图**调用冻结的 `phys.validate_trace/validate_payloads`：只过滤新时段/模型事件并为这个派生验证视图重编号，不改新 journal、不把派生计数当总成本；原51帧/15步/history/guard/reset ledger 的验收不另写弱替代。

新部分独立核同actual20 state7/snapshot/raw/RGB、20图→编码及旧context shift、完整AE→predictor→pred_proj→endpoint桥、五个请求/预测/成本的 seal、seal 先于所有扩展；新25 raw chronology、81张图/27 triplet、guard、原 native `eval_state` 公式（full7 distance **保留速度项**、success保持其原position/angle定义）、物理终态和真实RGB→终态编码join。纯算术 native mirror 不改变任何原指标或门槛。局部 spectral、latent→pose方向和 native 的速度效应不可混称“普遍 encoder 物理不对齐”。

算术复用 v1 exact Fraction：`b0=a0-p0`，`h_w=(a_w-a0)-(p_w-p0)`，每对 margin = predicted margin + common-zero projection + response remainder，恒等残差必须**精确为0**；renderer 比较 `(a_alt,w-a_alt,0)-(a_orig,w-a_orig,0)`。全部80对保留；原76/4仍是旧面板而非新分母。缺 zero 时不捏造 b0/h；其它双端有标签的 pair margin 仍可报告。没有部署 residual 修正、拟合共享偏置、按结果挑 pair/root/selector，不能把离线项移除当可部署修复。

该校准能报告此固定动作库下共同 zero 残差、动作条件效应误差及同状态 renderer 对编码效应的影响；不能把剩余 h 唯一归因 hidden state、动力学模块或某层实现，更不自动证明/否定原 P1 非贪心物理收益。数据完整性、freshness/校准、真实非贪心机制和强基线统计仍不同门。本两known-root校准不签 fresh admission，也不替代原 gate。

## MAIN 可直接调用的 CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。下列是实际已实现命令，不是 future adapter API；`PREPARE_SHA`/`REPORT_SHA` 填前一命令真实输出，不使用文档占位 hash。新输出名必须保持本 stem、exclusive，不复用失败目录。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_prospective_action_effect_20261028_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v2 prepare --output results/agent_p1_prospective_action_effect_20261028_v2_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v2 verify --prepared results/agent_p1_prospective_action_effect_20261028_v2_prepare_MAIN01.json --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v2 run --prepared results/agent_p1_prospective_action_effect_20261028_v2_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_p1_prospective_action_effect_20261028_v2_MAIN01
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v2 consume --output results/agent_p1_prospective_action_effect_20261028_v2_MAIN01 --report-sha REPORT_SHA --receipt results/agent_p1_prospective_action_effect_20261028_v2_consume_MAIN01.json
```

MAIN 的正常 host escalation 即运行授权，不需要额外 manifest/token 或新 IPC 进程。prepare/verify/consume 不实例化环境/模型；它们仍会读取原绑定数据/模型文件核 SHA。`run` 唯一实际导入和构造 native/CUDA 的入口，CUDA 不可用即 partial，无 CPU 回退。命令摘要的 `status` 必须核实，不能只看 Python exit0（partial 也会正常返回摘要，方便消费耐久报告）。

## 本 worker 验证与明确剩余项

使用TDD先见 import/功能缺失失败，再实现真实接缝。测试含无神经 NumPy tensor/module ABI fake（不是 CPU 模型推理）、原生 orchestration 的40固定槽/封存先行/失败无重试、编码前处理与AE失败不冒充API调用、signedzero、normalization、持久失败全80未知、alternate失败保留 original、实际既存40库 schema 的只读20步 consumer 投影。测试不替代真实新 GPU/physics 数据。

未由 worker 执行新 prepare/build 或实际 run。MAIN 仍需审计本三个文件，并执行 prepare、host run、严格 consume，确认实际 CUDA模块/API shape、native determinism/guard、时空与资源上限在本主机闭合。任何真实失败须保留 partial 原因，不换根、改阈值/metric或跳过绑定继续。可实施 adapter/CLI 已交付；剩余是主机验证，不是另一个未实现的 adapter/IPC 占位。

最终联合验证：v2 **27项**（含既存40库 saved-only 投影）+ 冻结v1 **34项**，共 **61 passed**。v1 source/tests/doc 的 SHA 分别仍为 `2cfe66efa2b79b41c91fcbb678427096fbb4d06e045d04a27ed58cfc99bb7136`、`214876985e5ed336034bc84dcd5ef9c8e4b66ed14b15d89a3a0e472cb98041a0`、`f24834cea27cefac78ba4b74f141d9877f29c6a1a157c9e5e672a75a68987252`。这不构成任何新CUDA/physics结果或P1通过证据。
