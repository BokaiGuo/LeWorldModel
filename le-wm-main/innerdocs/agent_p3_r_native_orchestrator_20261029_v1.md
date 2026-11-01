# Native spectral8 R tree：可执行层与原 caps 拒绝

本三件套独占，不修改旧 producer、policy、预算、协议或结果。提供 own-schema prepare/verify/run/consume。worker 仅运行 NumPy/RAM fixtures；不运行生产 prepare、神经、物理或生产 fullstrict。只有 MAIN 明确授权的 run 才导入 torch 和加载 CUDA 模型。

## 真实接线

完整遵守 `main_p3_r_spectral8_variant_protocol_20261029_v1.md`。复用冻结 NativeTree/NativeSignal、parent 双身份映射、spectral8 bank、原 bounded_ridge 和 suffix 规范化。一个共享 Ledger 连接真实回调；不将 HOST 冒作 RAM，不猴补 globals，不借 root J 给 child。root authority 和各 parent 的完整三 latent、两 action token、raw/path 都逐位绑定。模型输入不接物理 state、reward 或 outcome。

原 choose 的 horizon/width/depth/type、144 请求槽和原 modelQ beam 排序保持；R 八条 edge 加独立付费 zero 共九项。每项真实 replay 到 postcontext，独立 H1 nominal/J，再解原 ridge 并 replay bounded suffix。nominal 全 context 从已付原 signal boundary 恢复，不加隐藏 predict。suffix 只 lookahead，不拼入可执行 edge。资格严格为 immediate>zero 且 reach<zero，无 epsilon；所有 eligible 进原 beam，最低 reach 仅诊断，无 eligible 停枝、不 fallback。H3 是明确30维新变体，不签原 proposal 等价。

bank 前把已付 parent signal 完整调用保存为排他、fsync 的 `signal_s{slot}_{call}.json`，再用实际文件 SHA 与原数组/event 索引绑定 bank，不把未封口 journal 的未来 SHA 当 authority。完整 transcript、IO、192 VJP、J、各路径、银行/SVD/求解及所有失败前缀保存在链式 events 和 report。R 开始后的失败保留 zero+8 槽，未返回的不补齐；费用不退。

## 不能掩盖的静态预算冲突

生产 caps 原样复用：fixed114/adaptive562 predict 外壳及原 `replay_calls=2`，不增加 CLI cap override。R 完整组需要18 replay，最终 whole/segmented 再 reserve2，因此 unchanged production caps 必然在 R bank/control 之前原子拒绝。`production_R_group_fits_original_replay_cap=false` 明确记录；不能从本 runner 正常 exit 推断真实 R 已执行。

整组费用（不含已付 parent H1/H3 signal）由 `group_cost(h)` 精确推导：predict=9(h+3)，H1为36、H3为54；J9、VJP/grad/raw 各1728；program compile10（bank1+edge9），solve9、array SVD10（bank1+ridge9），metric18、replay18。AE/predictor/pred_proj/sample/token/row 分轴由原函数展开。另按完整 prefix+新 edge 预留 whole/segmented 两次；无 padding、退款或 FLOPs 等价声明。独立 AD primal/J 元素按真实 signal 调用派生，不与预测数混为同轴。

RAM fixture 唯一显式把 replay cap 设20，用来覆盖 dormant evaluator、全 eligible beam 和 final 验证；不是生产配置或新科学授权。四生产槽固定 known2×fixed/adaptive；cap/no_witness 按原合同继续其它槽，runtime failure 阻断后续槽但保留四分母。R/refusal 不是理论负结果。既有 root context 不是新的 actual-history 物理起点；本版本没有物理接口。

## Saved strict 与来源闭包

prepare 重建原 task/model/normalizer/Q/source authority，绑定自身与冻结依赖三件套、protocol、loader/weights/config 和 installed class 源。run 保留 MATH-only、matmul/cudnn TF32=false、CUDA/CPU autocast=false 与模型参数/buffer/class 前后不变核验，按原 wall/disk/CUDA limits 排他保存，不重试。

consume 重建 prepare、外部 SHA、artifact 全量 census、hashchain、模型/精度/来源闭包与 source capsule。逐 native 调用核完整 buffer/trajectory/AD 与资源；再以保存的实际 native 返回值运行 inert scheduler、SVD/ridge、资格、全 beam、raw/context 和 final replay 代数。复用原实测 Ledger 时钟以精确重现带时间的 bank authority hash，不替换实测 wall。未知 event、孤立 call/arm、漏收费、删 eligible 或替换 J/raw 均拒绝；不重算神经模型，不使用 neural parity tolerance。失败或来源不完整可保留 artifact，但不保证完整 consume 可认证。

## MAIN CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；SHA 必须来自实际上一条返回，不填未来值：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_r_native_orchestrator_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_orchestrator_20261029_v1 --help
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_orchestrator_20261029_v1 prepare --output results/agent_p3_r_native_orchestrator_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_orchestrator_20261029_v1 verify --prepared results/agent_p3_r_native_orchestrator_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_orchestrator_20261029_v1 run --prepared results/agent_p3_r_native_orchestrator_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_r_native_orchestrator_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_orchestrator_20261029_v1 consume --prepared results/agent_p3_r_native_orchestrator_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_r_native_orchestrator_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_r_native_orchestrator_20261029_v1_consume_MAIN01.json
```

以上树命令不会突破旧 replay cap。MAIN 新单 parent group diagnostic 协议应由后续独占 runner 实现，不能把该独立预算倒灌本树。当前不签 P1/P3/full policy/C3/physical release；无控制收益结论。测试与最终 SHA 在交付消息记录，冻结后不追加本文件。
