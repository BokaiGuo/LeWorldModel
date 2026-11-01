# H15 对 ball-domain CEM 的受控 HOST 候选

本候选不改现有 H15 和 gate；使用其两条原 off task、完整 context、identity192 目标、normalizer 和 checkpoint 的桥接。CEM 为内部 bounded CEM 风格实现，非官方 CEM 复现。原 H15 与这两根结果已被观察，因此此协议虽在 CEM 新运行前固定，并非盲态预注册或独立验证。不得根据已知结果修改 seed、域、预算或根。

每根六代，每代七个独立正态 logit 样本，投影到同一个 30 维球，半径 0.75√15；与 H15 一样经 NumPy float64 tanh 转 float32 为 15×2 raw。取每代两个最低 terminal latent mean-square 作为 elite 更新 mean/std（population std，floor 0.05）；PCG64 种子 `123 + 6*root + generation`，固定平局先出现者。最终交付42个实际计分样本中的最佳者，并付费重放 whole 和 segmented，要求与获胜路径完全一致。不是未付费的分布均值或额外优化结果。

每根 CEM 42×3 + 2×3 = 132 个 predictor samples；既有 H15 也为132，但另有 192 VJP。每个实际 native 调用由账本在执行前后计量，异常不退账、整槽失败。predict API、sample、token、AD 和 wall 轴分别报告；不宣称等 FLOPs、等总算力或等算法调参成本。物理及训练均为零，未取得物理对照。候选数42与H15的40不同，因为H15的六次预测用于 Jacobian/nominal；不得将样本数相同解读为所有资源公平。

真实入口复用原 checkpoint loader、source/model/precision 前后认证、NativeTree、逐调用资源账本及原 CUDA host guard：总 wall 3600s、output 1536MiB、文件 128MiB、空闲地板512MiB、CUDA allocated 4GiB/reserved 6GiB。guard 是合作式边界检查，不能中断单次卡住的 CUDA 调用，也不保证进程崩溃后仍有最终报告。输出须不存在；失败停止后续根、已发生部分不冒充完整。consumer 重建候选和账本、核源和 model/precision 证据；不能证明未被仪器捕获的调用不存在、真实机器身份或物理表现。

已完成 CPU/RAM fixture 和原 task prepare/verify；真实 CUDA ABI、内存、位级重放及完整 saved consumer **尚未以该新候选实测**。因任务要求，在 GPU 使用前必须先与用户协调；以下命令是交接，不代表已有运行授权或成功。输出不能覆盖既有结果。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_h15_cem_matched_host_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_cem_matched_host_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_host_20261029_v1_prepare_CODEX01.json --sha256 7fa4d83320ecad6391126dc2906f8156965af98f366aa4b0f0e61715af4ed1cb
# 仅在用户协调并明确允许主机 CUDA 使用后：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_cem_matched_host_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_host_20261029_v1_prepare_CODEX01.json --sha256 7fa4d83320ecad6391126dc2906f8156965af98f366aa4b0f0e61715af4ed1cb --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_host_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_P2_H15_MATCHED_CEM_HOST_CUDA
# REPORT_SHA 用上一步实际输出，不可预填：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_cem_matched_host_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_host_20261029_v1_prepare_CODEX01.json --sha256 7fa4d83320ecad6391126dc2906f8156965af98f366aa4b0f0e61715af4ed1cb --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_host_20261029_v1_cuda_MAIN01 --report-sha256 "$REPORT_SHA"
```
