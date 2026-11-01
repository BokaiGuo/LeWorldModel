# P5 已开放面板逐根分层（独立工作位，2026-11-01）

本次只读取已保存的 `p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1_MAIN01.json`，以固定 SHA256 `6685430d507ddc808b500356e094622816fbc180bd2f3bca73c12cd3d56b51c8` 绑定；不执行模型、物理、训练或 CUDA，不依赖 P1 通过，不改现有资产。新消费者逐项重算误差，检查索引、分组、时间、任务身份、有限值及数量；输出按两个 root、两种外观、五时刻和六坐标分层，保留全部 164 枝。

一个被合并均值掩盖的差异：horizon 2 的 block-x 绝对误差，root 0 的 off/on 分别从 72.10/75.15 降至 56.44/55.46；root 1 则从 62.00/65.89 增至 114.94/115.36。每个 root/条件各有 41 枝；这些枝相关，方向计数不是独立样本数或显著性证据。

刷新使用额外真实观测，且 h>1 对比的是一步刷新与多步开环，故只支持已知面板上的描述性诊断，不是公平闭环、因果解释、新根确认、P5 准入或跨模型结论。来源比较器的上游闭包在原报告中给出；本消费者固定该报告字节并复核数值，不独立重放全部上游 journal 或认证 GPU 硬件。

执行：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_root_stratified_saved_codex_20261101_exclusive.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.p5_root_stratified_saved_codex_20261101_exclusive results/p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1_MAIN01.json results/p5_root_stratified_saved_codex_20261101_exclusive.json
```

输出采用独占创建，第二次执行必须使用新文件名。测试 6 项通过。
