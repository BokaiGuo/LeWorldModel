# 共同域 CEM 两根物理适配

固定读取用户确认的 matched CEM HOST report SHA `3b675cb7bb99591152704a8fe1ef1e579524e4f3c6c839f37d7a9c78934ab799`，先完整重建其 saved consumer，逐根取其唯一 `result.winner.actions`；不按分数、物理或可部署状态筛选根，不二次 tanh、不 clip。任务、上下文、目标与物理根和 warm10 按原 H15 六槽协议绑定，完整两个槽。使用原 native 环境、早停、计账及 saved-only 图像/轨迹 consumer；不执行模型或训练。

本次产物：prepare SHA `677443313c2754b74dcffc3888231e39461056242a88468a960a72b052ca3efd`；物理 report SHA `80d30187315ad70c4acdf66c4044e0529d96c612db56942486cf747134da6317`；两根均完成15步，50 public steps（含两根 warm10）、508 physics steps、162 renders。两根最后 native success 均 false。记录属于已使用的两根、不同时间的物理执行，不是 fresh/generalization 证据；H15 与 CEM 仅 predictor samples 对齐，VJP 和总计算不同。

消费在独立进程重新计算上游模型 saved audit、固定任务、物理 journal、动作字节、资源、PNG及 native 标签；不重新执行物理，不能认证未记录的外部行为。输出是 stdout，并未另行持久化 consumer receipt；可使用下面命令独立复算。已有结果不覆盖。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_h15_cem_matched_physical_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_cem_matched_physical_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 677443313c2754b74dcffc3888231e39461056242a88468a960a72b052ca3efd --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_physical_20261029_v1_physics_CODEX01 --report-sha256 80d30187315ad70c4acdf66c4044e0529d96c612db56942486cf747134da6317
```

已执行的物理命令如下（如需再执行须另选不存在的输出目录并确认授权；不要重复已有运行）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_cem_matched_physical_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 677443313c2754b74dcffc3888231e39461056242a88468a960a72b052ca3efd --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_physical_20261029_v1_physics_CODEX01 --authorization MAIN_P2_H15_MATCHED_CEM_TWO_ROOT_PHYSICS_ONLY
```
