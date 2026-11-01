# Line-search GN：独立两根物理消费者

交付状态：只实现与静态阅读；作者没有执行本版 tests、prepare、物理、CUDA 或 CPU neural。测试通过与真实结果均须 MAIN 另行报告。没有修改 fixedsegment 或其他封存文件。

## 输入与选择合同

新 schema/name：`p1_line_search_gn_physical_20261027_v1`，arm=`line_search_GN`。固定原两根、每根一分支，共两臂；不读取 fixedsegment 物理结果，不把旧 witness 重标。

绑定真实模型报告 `results/p1_line_search_gn_cuda_20261027_v1_cuda_main01/report.json`，SHA256=`5f9825f98ed40f637c9d48e49c7c53428d1380e4234e3a1e5503a76f74c4bf1e`；绑定原 prepare01 SHA256=`e1b6d9783da6a4e32d360326869fe1b64fef8c3f4f168095be4b7f8c215b26d9`。

不是 fixedsegment 的一次插值选择：本输入从 zero restart 开始，固定四次更新，每次完整评估五尺度 `[1,.5,.25,.125,.0625]`，连同 incumbent 稳定最小化，再做独立 nominal/whole/segmented 三重重放。物理消费者只执行第四次更新后的最终 incumbent，不重新选择尺度、不修命令、不 early-stop 模型搜索。

`collect` 核验外部 prepare/report SHA、源码闭包、模型未变声明与原始资源记录、四文件 artifact manifest、完整有界 gzip、两根顺序、所有预测上下文/同 Q cost/normalized action history、每根四个完整192×20 J 与192行 raw derivative、单次 tanh 链、同 Q gradient、原点 AD 上下文、保存 delta 的 `.002` ridge 正规方程残差、原 projected endpoint、全部五尺度 float32 logits/raw commands、稳定选择、四次 selected replay、最终 winner/顶层报告相等。

每根原始模型资源：78 predict/action_encoder、4 J、768 VJP、4 SVD/solve/proposal/selection；总156/8/1536/8，另列 inherited source/image/B8direct，不与本消费者物理计账混合。这里仅保存数组算术核验，无新 J、forward、SVD、solve 或 encoder。

本地检查器按类型核对完整事件序列及不可变 operation index 配对，并核验 group/path；不是独立执行模型，也不是全事件交错的独立 cursor 重建。原 gzip 的总顺序通过外部 SHA 绑定。正规方程容差为 rtol1e-8/atol1e-10，仅用于保存证据核验，不调整搜索。

## 真实 reset / 输出

显式复用封存 `physical.branch`、`shared.Store`、`shared.object_results`，不猴补旧模块全局。新 run/prepare/schema 与 selection ID 独立。

通过 `b8.collect` 重新认证原 observed/source/native task/normalizer/Q 及 B8 模型证据；不消费 B8 物理 outcome。两根 observed0/5/10 必须逐字 stationary 且等于 parent，才允许沿原 source recipe 重建。没有把 saved poststate 作为 reset 输入，也不增加 warmup。

每臂：原 random_reset_seed 获取 native goal → 原 `sample_state(goal, shapes, slot, proposal, attempt, source_plan)` → 原 placement_reset_seed/options。比对实际 state、goal、RGB、goalRGB、root hash、render 时状态后才发第一条命令。所有十条 float32 raw 命令与模型 winner 精确相同，不 clip/project/replan。每条前要求 fresh RGB guard；初态与每次返回 step 均显式 render。

完整成功预算：4 reset /20 public command /22 explicit render /6 PNG，仅保存两根各0/5/10。reset 内部 physics=8、goal render=4 为原合同乘完成 reset 的推导计账，不冒称真实 substep hook 测量。实际内部 substeps 输出 `null`。无新神经调用。

初态失败不 dispatch；native terminated/truncated 必须停止，保留 partial，无替补/重试。保留首错及 close 等 secondary errors。native angle 原公式在 |Δ|>2π 时可负；不改 env，不覆盖 native terminal，另存 wrapped 诊断及 native-vs-wrapped 风险。object-only position/angle/pose 为描述统计，success=`null`，没有擅设任务阈值。

`journal.jsonl` 沿用 hashchain/fsync，PNG 带 file/RGB SHA；report 绑定最终 journal SHA。原 `branch_finished` 是旧 branch 原始收据；随后 NEW `object_only_result` 与 `branch_closed` 保存新增 object-only 和 close 错误信息。最终 report 为合并后的行，不将早期收据当最终 close 状态。存储错误尽力保留证据；写盘失败不保证终态 report 落盘，不能据缺 report 宣称成功。

完整输出总额≤2MiB，启动前≥2MiB+200MiB free；每次写盘继续检查200MiB reserve，失败写出的 bytes 计入 cap。小 prepare 只含路径/hash/两套命令和合同，不复制模型或大闭包；新源码+test+doc+单份 prepare≤100KiB。输出目录必须新建，不 resume/覆盖。

## MAIN 最小命令

以下从 `/home/gbk/code/LeWorldModel/le-wm-main` 运行，作者未执行。tests 只读真实 saved arrays，并用内存 Store/ArrayEnv，不创建 pytest tmp/result，不加载 neural 或 simulator。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_line_search_gn_physical_20261027_v1.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/spectral_reachability/p1_line_search_gn_physical_20261027_v1.py prepare \
  --search-prepared results/p1_line_search_gn_cuda_20261027_v1_prepare01.json \
  --search-prepare-sha e1b6d9783da6a4e32d360326869fe1b64fef8c3f4f168095be4b7f8c215b26d9 \
  --search-report results/p1_line_search_gn_cuda_20261027_v1_cuda_main01/report.json \
  --search-report-sha 5f9825f98ed40f637c9d48e49c7c53428d1380e4234e3a1e5503a76f74c4bf1e \
  --output results/p1_line_search_gn_physical_20261027_v1_prepare01.json

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/spectral_reachability/p1_line_search_gn_physical_20261027_v1.py verify \
  --prepared results/p1_line_search_gn_physical_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA

# MAIN 批准且磁盘复核通过后才执行以下真实 physics，无模型调用。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/spectral_reachability/p1_line_search_gn_physical_20261027_v1.py run \
  --prepared results/p1_line_search_gn_physical_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA \
  --output results/p1_line_search_gn_physical_20261027_v1_main01 --authorize-physics
```

用 prepare 打印的实际 SHA 替换 `PREPARE_SHA`。运行原 native env 需要 MAIN 已有 simulator 依赖，无下载步骤。

## 证据边界

同原两 calibration families，observed root=parent，不新增独立样本；global freshness unknown。完整物理执行仅提供本固定 line-search GN winner 的校准比较，不等于 P1 admission、direct plateau/stall、合法 two-stage 机制成立或普适效能。模型 cost 改善不替代真实任务成功。测试（若 MAIN 通过）只支持程序/协议合同，不能替代真实运行结果。
