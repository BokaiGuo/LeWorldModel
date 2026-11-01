# P1 full CEM host 接线审查候选

本文件及 `codex_p1_full_cem_host_20261101_exclusive.py` 为独占新文件。
未执行 CUDA、模型推理或物理环境。对照为两个 known root 的 first-feedback
cp10 目标任务；不与 Leibniz 的 cp5 开始物理重放复用或竞争。

设计沿用前一合同的 0/5/10/15、H3/H2/H1/H1、每节点四轮 16 候选、
4 elite、每次重启、全局最低成本加一次复评，种子固定 20261101 加节点索引。
每节点预测及 action encoder 调用各为 195/130/65/65，四节点合计各 455；
这是上限及完整节点目标，不是已执行结果。CEM rollout 和打分由独立持久
journal 记录，node budget ledger 只对 action encoder/predict 的调用及样本
实施硬上限；编码由原 RGB handler 的 visual meter 单独计费，原物理 meter
记录物理资源。模型加载、共享来源、设备时间及 FLOPs 等仍无完整计量，
不能主张等成本或公平。

接线复用原 host launcher 的来源重建、类来源和模型身份校验、壁钟/显存/
磁盘限制、排他路径、既有 cooperative GPU 锁及 host CUDA 策略；独立重绑
cp10 目标后，检查实际目标像素/latent 及 root 与保存的 first-feedback
绑定。原 moving setup 对 reset/warmup、RGB、state/dynamics、目标和根逐项
精确认证。每次反馈通过原 RGB handler 重新编码自身实际图像，并由已执行
真实动作构造历史；控制器先封存选中动作，原 packet 路径对每个控制步写
`authorized_dispatch` 和 `control_return`，实际原生终止不补齐。单节点
账本以尝试计费、超限拒绝。失败时不重试；部分结果保留。

此版本仍需要审查：原锁仅约束配合使用同一锁的进程；GPU 独占需要外部保证。
审查应覆盖 CPU dry-prepare 输出、真实来源重建、事件语义、source 漂移、
单节点预算、原 packet 的早停及日志，在明确授权前不要启动。旧 receipt 和
本次未来新 receipt 不可互换。当前代码/测试通过不表示模型或环境路径已实际运行。

只读 CPU dry-prepare：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_full_cem_host_20261101_exclusive --root 0 dry-prepare
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_full_cem_host_20261101_exclusive --root 1 dry-prepare
```

将来如经审查和 MAIN 明确授权，每根单独固定全新 prepare 和输出路径，先运行
`prepare --output /ABS/RESULTS/codex_p1_full_cem_host_20261101_exclusive_root0_PREP_NEW.json`，
以实际返回的 SHA 用 `run --prepared ... --sha256 ... --output ... --authorization
MAIN_AUTHORIZES_P1_FULL_CEM_HOST_CUDA_AND_PHYSICS --authorize-host-cuda-and-physics`
执行；路径与 root 必须匹配且不存在。不要把此文档或技术参数当作授权。
