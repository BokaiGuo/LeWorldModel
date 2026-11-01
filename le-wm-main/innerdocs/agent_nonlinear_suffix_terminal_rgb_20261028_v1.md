# Nonlinear suffix fixed4 terminal RGB companion

## 固定范围和来源 ABI

只消费已冻结 `agent_nonlinear_suffix_physical_20261028_v1` 的真实四槽：root0/off、root0/on、root1/off、root1/on。独立 prepare/report/consume schema，绝不冒用 terminal-witness selector authority。物理 source SHA 固定 `9108512019f0a7154a5eec56dfc55c2af9035362c5687130eb48eae5d7c9dc25`；其余实际物理 output/report/prepare/receipt 及 SHA 均由 MAIN CLI 提供，没有未来默认 SHA。

build 实际调用 `physical.consume`，全量重建并逐值比较持久 receipt；该真实 consumer 内部执行 `verify→material→suffix v2.consume→verify→build→原 source.consume/verify`。不以 status/hash 替代 full build，不接受 v1 partial。实际 physical prepare 必须精确匹配 report.execution_binding 的 path/SHA；继承 strict.bindings 包含原物理 PNG/journal/dispatch/source/model、suffix v2 全168及原 paid source 闭包。额外读取原 suffix prepare/report 只做来源 joins，不调用其 neural runner。

四槽 task/blind、condition、模型/normalizer、root context/goal192/identity Q、prefix5/full raw10、选择 digest、优化 iterate、原 final context/terminal Q，以及 native root7/goal7/goal_pose/goalRGB SHA 均精确核对。保留实际 cp10 snapshot 全字段（含其保存的时间/物理状态字段），不发明缺失时间戳。cp10 raw_history10 就是实际执行的全 raw10；状态只用于证据/评估，model.encode 输入仅真实 pixels。

仅实际 cp10 PNG 可编码，要求 own_returned=10、同 condition 的 trajectory/state/snapshot/final_state 精确对应。genuine native early stop 无 cp10 时保持四分母、source/latent/actual_Q=null；不补 earlier PNG、旧 latent、其它条件图片或零。完整工程失败不作为正常 missing 放行。不按 outcome 选择根/动作/库，不 dedup。

## HOST CUDA 与完整数组

每张可用 cp10 只收费1 encoder+1 projector，最多4+4；0 predict/AE/predictor/pred_proj/AD/new physics/训练。保留继承原 allmode、suffix3376predict/1344scalar reverse、实际物理各轴计量；不加总嵌套调用、不称 equalFLOPs。消费自身0物理不表示原执行没有物理费用。

复用冻结 terminal RGB 的 instance encode_guard 和 contextual RGB Trace/Store/PNG/预处理低层，own build/verify/run/consume；不改任何模块 globals。Trace 保存 uint8 原 pixels、原 float32 预处理、encoder input/full `[1,257,192]` hidden、CLS input `[1,192]`、projector output 和最终192 latent 的原生数组、shape/dtype/数组及文件 SHA。config_shapes 从绑定224/patch14导出257，拒绝197。

run 使用原 `p0.image_transform/_image_tensor`，立即用独立 NumPy/PIL `preprocess_pixels` 逐值/逐字节复算；saved consumer 再从原 PNG 读取并独立复算，同 encoder input、full hidden CLS、projector→latent 精确校验。没有重新编码 goal 或预测末态。actual_Q 使用原 goal192/identity Q 的 float64 MSE，native 成功/奖励保留独立 evaluation，不将二者等同。

MAIN authorization 后才导入 torch/加载原冻结模型，CUDA only，无 CPU fallback。绑定模型原 tensor hash及实际 class 源码，前后 frozen/hash/source 核验。MATH SDP、matmul/cudnn TF32 off、autocast off、inference mode；实际 precision before/after 持久且相等，逐图检查。公开 AD APIs 由线程 profile probe 拒绝，禁止 predict/AE 等由实例 guard 拒绝；guard 不替换 torch globals，不冒充内部 backward/FLOPs 计量。完整 module_attempt/return、guard 和 image event 顺序逐项核对，不过滤未知事件。

预算真实 ABI：bridge 与 rgb_base 均 output64MiB、final8MiB、floor200MiB、wall600s（含 source verification，边界检查），batch1。new exclusive prefix 输出，不覆盖旧文件；partial 保留证据并报错，不重试。MAIN 如遇超时/资源上限须保留失败，不填0。

## MAIN CLI（inner 仓库根目录，全部路径绝对路径）

物理终态后 MAIN 赋真实值给 PHYSICAL_OUTPUT/PHYSICAL_REPORT_SHA/PHYSICAL_PREPARED/PHYSICAL_PREPARE_SHA/PHYSICAL_RECEIPT/PHYSICAL_RECEIPT_SHA；本版未自行 prepare/run/consume 生产资产。

```bash
M=experiments.spectral_reachability.agent_nonlinear_suffix_terminal_rgb_20261028_v1
RGB_PREPARED="$PWD/results/agent_nonlinear_suffix_terminal_rgb_20261028_v1_prepare_MAIN01.json"
RGB_OUTPUT="$PWD/results/agent_nonlinear_suffix_terminal_rgb_20261028_v1_cuda_MAIN01"
.venv/bin/python -B -m "$M" prepare --output "$RGB_PREPARED" \
  --physical-output "$PHYSICAL_OUTPUT" --physical-report-sha "$PHYSICAL_REPORT_SHA" \
  --physical-prepared "$PHYSICAL_PREPARED" --physical-prepare-sha "$PHYSICAL_PREPARE_SHA" \
  --physical-receipt "$PHYSICAL_RECEIPT" --physical-receipt-sha "$PHYSICAL_RECEIPT_SHA"
# RGB_PREPARE_SHA 取实际 prepare stdout SHA；不预填
.venv/bin/python -B -m "$M" verify --prepared "$RGB_PREPARED" --prepare-sha "$RGB_PREPARE_SHA"
.venv/bin/python -B -m "$M" run --prepared "$RGB_PREPARED" --prepare-sha "$RGB_PREPARE_SHA" \
  --output "$RGB_OUTPUT" --authorize-host-cuda
# RGB_REPORT_SHA 取实际 report.json SHA
.venv/bin/python -B -m "$M" consume --output "$RGB_OUTPUT" --report-sha "$RGB_REPORT_SHA" \
  --receipt "$PWD/results/agent_nonlinear_suffix_terminal_rgb_20261028_v1_consume_MAIN01.json"
```

## 测试与科学限制

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_nonlinear_suffix_terminal_rgb_20261028_v1.py tests/test_agent_terminal_witness_terminal_rgb_20261028_v1.py
```

测试仅 RAM metadata/native-array hooks doubles、独立预处理及冷 import，不是 CUDA/物理证据。首轮57 pass/1 fail，仅 CLI 测试替身签名不兼容真实位置参数，已定向修正，保留失败记录。最终联合94 passed in11.46s（本版59项、冻结 terminal RGB35项）。测试包括旧 authority/partial168、raw/goal/task/context/Q/native/snapshot joins、缺失 null4、197拒绝、完整事件/数组/guard/计量变异、真实 helper SHA和 budget ABI、真实 build/verify 入口完整 receipt 比较、精度 flags 和禁止 foreign runtime 冷导入。

MAIN 已通知实际四槽物理 strict 完成；由 MAIN 将真实路径/SHA 通过以上 CLI 绑定。尚未运行本 companion 的实际 prepare/verify/CUDA/consume；可用 CUDA/磁盘与 MAIN 最终审读调度仍为执行前置。worker 未执行生产 prepare、GPU、neural CPU 或物理。实现/编码或 modelQ 排序都不构成 P1/freshness/独立确认；P1/scientific_admission/freshness 始终 false，四 reserved fresh roots 不涉及。
