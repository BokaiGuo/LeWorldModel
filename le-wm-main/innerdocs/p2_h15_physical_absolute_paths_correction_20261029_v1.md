# P2 H15 物理命令路径更正

先前回复中的相对路径示例不符合 physical 入口的绝对路径要求。以下是正确的运行调用；仅在输出目录不存在且相关输入保持封存时使用，本文不代表授权或重新执行：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_physical_adapter_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_physical_adapter_20261029_v1_prepare_MAIN02.json \
  --sha256 a68ec15d03780345c1c9d2d8c68e741859a0b56e1bc8c3f6fb71be7a911d4dc8 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_physical_adapter_20261029_v1_NEW_RUN \
  --authorization MAIN_P2_H15_TWO_ROOT_PHYSICS_ONLY
```

已完成的 MAIN01 不应重跑或覆盖；消费时也使用 prepare 和 output 的绝对路径。没有修改已封存代码。
