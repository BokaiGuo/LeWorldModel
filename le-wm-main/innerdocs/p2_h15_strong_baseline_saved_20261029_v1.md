# H15 与既有强基线：同根同时间的 saved-only 描述

本分析使用已消费的两根 bounded GRASP、H3 bounded CEM 物理凭据和固定 H15 物理报告；它们不是新实验，也不是等预算排名。固定外部 SHA；逐根比较 task、root、native state、goal、goal pose、goal pixel、off 和实际返回步。只比较真实发生的同一步；早停后不补值。delta 为 H15 减对照。

对上游完整 consumer 的认证是信任既有凭据，不是在本程序中重跑完整闭包；H15 使用固定 report 而不是此程序自行重建其 consumer。各规划方法的原始上下文及动作绑定继承各自来源，逐条响应物理真实性不能仅由本程序独立认证。根是既用根，无独立泛化或因果推断。GRASP、CEM 均为 bounded 实现，非官方复现。CEM 全盒与 H15 联合球不同；模型调用、样本、VJP/backward、总 FLOPs/墙钟并不匹配。

已有 MPPI-inspired 对照在 full20/H2、四个 calibration roots 的另一个封存比较中，不应作为 H15 的同任务结果拼接。要完成真正共同动作域、规划时间尺度、目标与资源单位对齐的前瞻强对照，仍需事前冻结预算与公平协议、实现相应 HOST CUDA 生产和独立消费路径，再协调主机使用并实际执行；本分析不声称已经完成该工作。

复算（输出路径须不存在，以下不调用 GPU 或物理）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_h15_strong_baseline_saved_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_strong_baseline_saved_20261029_v1 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_strong_baseline_saved_20261029_v1_RECHECK01.json
```

输出含两根逐步配对及原始资源计数。six physical counts 是全部六槽的总量，不得归属于 GRASP 两槽；CEM 模型资源是两根总量。根0 CEM 在第10步原生成功早停，第11至15步均为 null。
