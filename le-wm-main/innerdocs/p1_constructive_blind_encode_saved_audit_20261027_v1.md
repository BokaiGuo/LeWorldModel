# Constructive blind 实际编码 saved 独立审计

仅审计既有数组和调用 receipt；不导入 producer/core/model，不读取 sourcepack、journal、certificate 或 admission report。MAIN 的完整源准入 SHA 仅作为不透明外部证明标识继续绑定，不独立追认可达性。

## 实际输入与边界

- receipt：`results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/report.json`
- MAIN 外部 SHA：`684456d6d3c45b99f20db354d2a0fb7b810bb61a8ebc8185816aa3d87ff40e91`
- prepare SHA：`7470bca322e14ca7f7e94a29f9d05ef46ed0108824158a098a2760ecec58d491`
- 只追踪 prepare 指定的 whitelist02、批准的 s0/s1 blind NPZ、源代码/模型文件 hash、8NPY、2task 与编码事件。模型文件只流式 hash，不加载或执行。

## 独立 expected

不调用 `producer.task_export`、`load_task`、`core.validate_task/task_id`。独立从 literal normalizer 源 AST 解析 float32 mean/std，按十条 raw action 的时间顺序逐行归一化，再 reshape 成两条 5×2 between-state token。raw zero 的 normalized token 非零；不得换成零 padding。

四个新 NPY 对应 observed0/5/10/goal，重建整个 task、完整 states3/actions2 内容和 layout、identity192 Q、goal 像素 hash、无 target 合同、checkpoint/config、blind-source 身份和 task_contract。taskID 独立使用原合同的 sorted JSON 默认空格编码（不同于 receipt compact+newline 编码）。拒绝将 blind SHA 充当 native root SHA。

资源从完整事件顺序重建：每图 attempt → encoder attempt/return → projector attempt/return → NPY saved；每根四图后 task sealed。共 50 个事件，8encoder+8projector，重复 RGB/latent 仍保留所有付费调用。检查整个输出 inventory、文件 hash、实际 bytes、最终 TOCTOU；拒绝 partial/failed receipt，不把 partial 变成功。CLI 拒绝时仍写独占小型 failure 报告，原资产不变。

## 执行

项目根 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_blind_encode_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_encode_saved_audit_20261027_v1 --report results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/report.json --report-sha 684456d6d3c45b99f20db354d2a0fb7b810bb61a8ebc8185816aa3d87ff40e91 --output results/p1_constructive_blind_encode_saved_audit_20261027_v1_saved02.json
```

输出路径独占，不覆盖 saved01。审计报告 <128KiB，写前保留 200MiB，无 fixture 拷贝。测试为纯数组与 saved 故障注入，无模型运行；缺 MAIN 实际 receipt 时 saved 专属测试明确 skip。

## 实际结论

首次 saved01 已执行：`saved_consistency_verified`，两根 taskID 重建相同，8NPY/2task/50events/8encoder+8projector 一致。源输出总计 35,060B。两根 observed0/5/10 latent 各自数值相同，但不删帧、不去重记账。标准化 zero 每包为 `[0.043090637773275375,-0.03396603465080261]` 重复五次。

这不是第二次 CUDA 验证；无法纯 saved 独立证明 pixel→latent 的神经数值真值、运行时没有账外调用、模型参数 hash 日志真实性。它核实已绑定实际 receipt 与原始资产的内部一致性，不是物理 success、新鲜独立根确认或 P1 科学准入。
