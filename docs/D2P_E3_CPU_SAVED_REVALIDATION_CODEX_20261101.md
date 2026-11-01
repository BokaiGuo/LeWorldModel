# E3 跨任务保存态 CPU 重验（2026-11-01）

## 范围和前提

只读重验固定 roster 中已保存的 cell 00、01、08、09；不启动模型推理、
物理环境或 GPU。联结依赖现有消费者对 report、completion、authority
及来源闭包的检查，并假设保存记录及其封存来源可信。16 格中未检查的
12 格标为 `not_inspected`，不据此判断这些格是否存在或完成。

## 本次实际结果

`e3_cross_task_identity_join_codex_20261030_v1` 退出码为 0，
四格均为 `saved_identity_joined`，状态 `completed20`；各格在控制边界
10 和 15 有身份绑定的编码返回。保存记录中的共同模型 SHA-256 为
`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`，
共同 prepare SHA-256 为
`ea8e4feae44e31876852c960db03ce4613184c55fc0745ef8f890f70485f8425`。

并行运行跨任务联结、分块传输、编码器身份以及 cell03 端到端重放
联结的既有回归测试，共 **34 passed**。测试并不等同于对 cell03 重新
执行物理或神经重放。

## 可复现命令

在 `le-wm-main` 下执行：

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m experiments.spectral_reachability.e3_cross_task_identity_join_codex_20261030_v1
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_e3_cross_task_identity_join_codex_20261030_v1.py \
  tests/test_agent_e3_identity_chunk_transport_20261029_v1.py \
  tests/test_agent_e3_encoder_identity_20261029_v1.py \
  tests/test_e3_cell03_end_to_end_replay_join_20261031_main.py
```

## 边界

这是旧资产的当前字节重验，不增加独立任务样本。哈希联结不证明跨进程
模型对象或物理 GPU 身份相同，也不证明神经或物理 parity；
`full_saved_strict=false`、`scientific_admission=false`。已有 cell03
分离式重放联结和 cell12 首步原生停止分别见相应既有报告，不能用它们
补齐本次未检查格或宣称 E3 科学门通过。
