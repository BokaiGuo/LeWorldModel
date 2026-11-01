# P3 matched-order HOST CUDA 接线与独立 saved 验证

本次只做 CPU 源码接线和测试，没有启动 CUDA、物理或训练。新 HOST driver 为 `experiments/spectral_reachability/p3_matched_order_host_codex_20261029.py`，复用 `p3_matched_order_codex_20261029.py`、原 `NativeTree`、`RecordingBackend`、事件链和原独立 native-buffer auditor。未修改原实验代码或结果。

## 已封存输入和固定实验

CPU prepare `results/p3_matched_order_host_codex_20261029_prepare_MAIN03.json` SHA256 `85e393983b1437630bd74243afe17c6764aff9d8505c5dd9d44d0603bc0066a4`。它绑定已有 parent prepare `deb87295f3984929f4734ece0b73770220ec59ac3b621a97e85e2bade60e3599` 的两个 known task、原模型、source/installed class/weight 闭包，以及本 driver、对照实现、相关原生依赖、四槽顺序、逐轴预算和硬件/存储合同。MAIN01、MAIN02 是开发过程的旧 prepare，源码变更后已失效；保留不覆盖，不用于运行。此准备发生在既有 K17 结果之后，仅对**未来新运行**预先固定，不是对旧结果倒签预注册。

固定 root0/fixed_order、root0/signal_order、root1/fixed_order、root1/signal_order；每个最多两块，完整均 H1+H3。信号顺序首块复用原 adaptive decision，第二块使用剩余配额。其他边界见 `D2P_P3_MATCHED_ORDER_CODEX_20261029.md`。新计划是模型预测 parent，不是观察、闭环物理或原 K17 完整树。原 first-decision 要求非 progress 时完整拒绝；本 driver 停止后续槽，保留失败及固定分母，不 retry、不 padding。

## 安全门与结果

显式 `--authorize-host-cuda` 是运行开关，不能替代主线程 GPU 协调。仅 `run` 导入 torch；需 `cuda:0` 实际分配、设备节点、禁 TF32/autocast、MATH SDP、eval/frozen 模型、原权重 hash 和模型类源码。前后重验源码闭包、模型与 precision，并核恢复 forward hook。每项调用前后通过原 RecordingBackend 检查 whole-job wall、输出/磁盘与 CUDA peak memory；运行中的单个 kernel 不可中途抢占，调用返回后仍检查超限。单臂 Ledger 在真正 hook 派发前使用 27 轴 hard cap。缺源、设备、资源或失败都不释放成功 receipt。

固定工程上限：wall 3600s、输出 1GiB 含 256MiB final reserve、磁盘 floor 200MiB、CUDA allocated 3GiB/reserved 4GiB；这些是预固定工程停止门，不是实际 fit 的证明。Store 对 journal 的每项 append/fsync 并记录哈希链；异常时尽力写带失败的 `report.json`，未完成的后续槽保持 blocked。文件系统完全不可写或 final reserve 也耗尽时不能保证报告可写，不能据空目录推断成功。`run` 部分失败退出非零；即使完整，也只产生工程报告、不产生科学通过。

`consume` 必须用外部 report SHA 和 prepare SHA：检查原始锚、全部文件 hash、journal hash-chain 与四槽顺序、source/model/precision/设备证据和资源上限；由原 `resource_census` 检查尝试/返回配对及真实 tensor IO 索引，再对每个 native signal/replay 调原 saved-buffer auditor，利用惯性 `SavedBackend` 从已保存调用重新执行调度和记账（不调用模型），检查全部上下文、候选、选择与动作。两对均完整且 27 轴和记录的 hook shape 多重集匹配才产生 engineering receipt。原审计核实保存张量与模型模块连接关系，不能排除未 hook 的计算，也不是远程可信执行证明。

## 主线程可运行命令

以下命令只在主线程检查无冲突 GPU 作业并明确协调后执行；目录必须不存在，prepare SHA 必须保持如下精确值。此处没有实际启动它们。

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_matched_order_host_codex_20261029 run \
  --prepared results/p3_matched_order_host_codex_20261029_prepare_MAIN03.json \
  --prepare-sha 85e393983b1437630bd74243afe17c6764aff9d8505c5dd9d44d0603bc0066a4 \
  --output results/p3_matched_order_host_codex_20261029_run_MAIN01 --authorize-host-cuda
```

运行返回完整状态后，将输出的真实 report SHA 显式传给 CPU-only consumer（不得猜测）：

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_matched_order_host_codex_20261029 consume \
  --prepared results/p3_matched_order_host_codex_20261029_prepare_MAIN03.json \
  --prepare-sha 85e393983b1437630bd74243afe17c6764aff9d8505c5dd9d44d0603bc0066a4 \
  --output results/p3_matched_order_host_codex_20261029_run_MAIN01 \
  --report-sha "$ACTUAL_REPORT_SHA" --receipt results/p3_matched_order_host_codex_20261029_consume_MAIN01.json
```

任何源码/权重漂移会使 prepare 验证拒绝；应重新审读并生成新的独占 prepare，而非改写 MAIN03。两对只在完整返回后比较；incomplete、失败和未启动不补 0 或筛选为成功。两个 known roots 不是 fresh/独立泛化。相同 hook counts 和 shape 不等于等 FLOPs、总计算、墙钟、显存或能源；不存在科学/生产就绪声明。

## 验证

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p3_matched_order_host_codex_20261029.py tests/test_p3_matched_order_codex_20261029.py
```

14 项测试通过，包括真实 prepare 的 CPU-only 重验、未授权拒绝、失败 attempt 和后续 blocked、输出限制、源 hash 篡改、RAM 合成 transcript 的 native auditor 及完整 consumer 接线与篡改拒绝。完整 consumer fixture 中 source 验证及 journal 读取是测试替身，`HOST_CUDA` 标签也是测试注入，不是设备或实验结果。新 HOST run 和完整真实 consumer 尚未执行；实际 fit、数据有效性及科学假设均未验证。
