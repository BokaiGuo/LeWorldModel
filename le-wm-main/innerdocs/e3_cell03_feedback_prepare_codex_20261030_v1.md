# E3 cell03 真实反馈 authority：CPU 准备

本节点为已知根 `r0.g0.h3.dDx` 建立独占 authority；复用原 cell02 的 host、identity、startup 实现和既有 source guard，严格重建 cell03 的 control5 report、completion、receipt 与原 resolver，继承原 source pins，不更改科学准入、资源上限或运行器。它是未来真实反馈 HOST 的输入准备，不是实际反馈、物理终态、freshness 或 E3 验收。已有数值 receipt 不能作为物理 episode。

本次固定输出 `results/e3_cell03_feedback_prepare_codex_20261030_v1_MAIN01.json` SHA256 `1facc17c2bb088c5c2af1ce4f035bc1a0c2742ab3d81ccd9d2830382fedf76e4`；prepare 自行重建验证。创建采用既有 `write_new`，拒绝覆盖；写入后验证失败可能留下无效文件，必须保留并诊断，不得视为通过。来源、依赖或源码漂移时原 guard 将拒绝，不应重新固定哈希绕过它。

依赖仓库现有 `.venv`、固定 base authority、原 resolver 和 cell03 control5 report/completion/receipt，以及被原 authority 引用的完整依赖文件；不加载模型、不创建环境。以下从 `le-wm-main` 运行：

```sh
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_e3_cell03_feedback_prepare_codex_20261030_v1.py
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.e3_cell03_feedback_prepare_codex_20261030_v1 verify --authority /home/gbk/code/LeWorldModel/le-wm-main/results/e3_cell03_feedback_prepare_codex_20261030_v1_MAIN01.json --authority-sha256 1facc17c2bb088c5c2af1ce4f035bc1a0c2742ab3d81ccd9d2830382fedf76e4
```

另一次独占准备可将 `verify` 改为 `prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/e3_cell03_feedback_prepare_codex_20261030_v1_NEW.json`；输出必须是尚不存在的该接口专属名字。真正的 HOST 后续仍需原合同要求的独立授权、硬件与源码检查；不由本文授权，也不因 authority 通过而获得科学准入。
