# P2 library computational-matching preregistration v1

本版是独立算法探索的 **prepare/预注册工具**，不是原 P2 合同替换、物理评价或准入。worker 不读新物理结果，不执行生产 prepare、NN、AD、CUDA 或物理。本三件套之外不修改任何冻结资产。

## 可执行 joint challenger 与匹配结论

输入严格绑定原三程序 report `56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef`、strict receipt `c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f` 与 source `3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5`。prepare 调用真实原 consumer 完整重建 receipt，并重建原 prepare；绑定所有原 artifact、task/model/source closure 与本三件套。不是只信 receipt 的通过标志，也不读取新物理 report。

固定同2 off known roots、每根原42个 seed（含零方向），原 root bank/J/nominal、全30 logit L2 半径 `.75sqrt15`、原 normalizer/mean192 terminal Q 不变。joint 对每 seed 原始 full15 评分后，执行 **两轮 full30 joint** 优化，不固定前缀、不在新 context 分解成 suffix-only 程序。每轮：原 root context 下重算 incumbent full15，取得 literal raw192×30 Jacobian，按原连续 tanh chain 形成梯度；向半径一半的负梯度点走固定 alpha `[1,.5,.25,.125]`，连同 incumbent 共5个评分，stable `(cost,index)`，平局保留 incumbent；选中者 full15 replay。最终全42稳定排序，whole15 与3段分别 replay 严格一致。不新增 seed，不按收益挑根，不调 alpha/半径/Q，不用 outcome 标签。

真实可执行函数：`local_joint(context, incumbent, goal, backend, tag)` 和 `joint_candidates(slot, bank, backend)`。backend 复用冻结 planner 的 `rollout/jacobian/last_ad_contexts` ABI；调用者须先完整支付原 root nominal/J/bank（6 predict、1 root J）。本工具没有 GPU/run CLI，**这些函数的存在不是运行授权**。

## 向量预算：当前候选不匹配，fail closed

以下为每根 standalone arm，均包含原 root shared 构建和最后双 replay。predict 包含 AD primal forward。

| 实际计量项 | composition | joint 双轮 |
| --- | ---: | ---: |
| predict / action_encoder / predictor / pred_proj（各自） | 936 | 1902 |
| Jacobian | 85 | 85 |
| mathematical VJP | 16320 | 16320 |
| 局部 J 形状 | 42×192×20 + 42×192×10 | 84×192×30 |
| J 元素数（含 root192×30） | 247680 | 489600 |
| AD primal 展开步数（含 root3） | 129 | 255 |
| root SVD / ridge | 2 / 1 | 2 / 1 |

joint 单轮局部费用是21 predict = incumbent3 + AD primal3 + 4候选×3 + selected replay3；每 seed 支付 seed3 + 两轮42，总1890，加共享6与终态双replay6 =1902。composition 是 root6 + 42×(first1+H2优化14+H1优化7) + replay6 =936。

原联合运行两根总2136 predict、170 J、32640 mathematical VJP，包含两臂共享 root 费用；**不能把两个 standalone arm 相加当成旧联合总账**。旧 joint 仅138 predict/根，不能因其少量前向与新 composition 比较而宣称同总计算。原继承图像编码16 encoder/projector 属共有历史成本，不计为本 prospective 分支新增0编码；不借此抵消 AD。

矩阵计数不是 FLOPs。engine 内部 backward 调用、total FLOPs、GPU peak allocated/reserved、wall 尚未知，均 null，不伪造成0或由 VJP 推断。`compare` 要求完整向量，已知项不等或任一未知都不签 matched；`validate_ledger` 拒绝缺项、额外项、非有限值以及固定执行量偏差。失败资源无退款，不做 padding、空算、重试或只用 predict 次数折算 AD。unknown 的 wall/memory 也不构成等计算证明。

因此本版 prepare 状态固定为 `blocked_no_vector_compute_match`，`run_authorized=false`，equal_FLOPs/equal_total_compute=false。它封存一个可复算、可实现的候选规则及其 **不满足匹配** 的原因，而不是已就绪的等计算 runner。没有据此证明所有 joint 算法都不可能匹配，也没有发现科学负结果根因。要开展真正等计算对照，须另行预注册新的 joint 规则与完整计量实现；不能执行本版后按 outcome 修改定义或只消除不利账项。MAIN 可先 saved-only prepare/verify，不能拿此输出签 P2。

后续任何授权 runner 必须保存逐 call intent/return、AD形状/完整context/module张量、attempted/completed、before/after frozen model/source、实际显存/wall，保持原 native float32/TF32 off/autocast off/MATH SDP；不能把 scalar-gradient 路径当原192-row J。物理/RGB outcome 只允许在动作全封存后的独立评价模块读取。

## MAIN CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main` 运行，仅 saved-only。worker 未执行：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_compute_matched_protocol_20261029_v1 prepare \
  --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_cuda_MAIN01 \
  --planner-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_consume_MAIN01.json \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_compute_matched_protocol_20261029_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_compute_matched_protocol_20261029_v1 verify \
  --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_cuda_MAIN01 \
  --planner-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_consume_MAIN01.json \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_compute_matched_protocol_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_OWN_PREPARE_SHA
```

只允许新 own stem metadata，cap4MiB，exclusive write+fsync。verify 全量重建，不能仅改状态标志放行。无 production run CLI。所有科学 admission/freshness/release 为 false。

## 测试

TDD 初始 RED 为缺模块 ImportError。RAM 测试覆盖完整资源向量缺项/篡改、相同 J/VJP 不同展开成本、未知成本拒签、全168局部调度、full42真实算法调用分母、stable平局、原域越界拒绝不clip，以及真实 receipt schema重建接缝/字节身份绑定（RAM模拟，不当成真实消费证据）。最终命令 `.venv/bin/python -m pytest -q tests/test_agent_p2_compute_matched_protocol_20261029_v1.py -x`，结果 **14 passed in 0.30s**；无神经或物理结果。
