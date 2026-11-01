# E3 跨任务保存身份联结

## 现有产物和本次范围

现有 cell00/01 和 cell08/09 比较分别覆盖两个已知根的 H1 动作域，已有对同根、同时间观测的描述性分析；不重复计算效果。cell04、12、13 的保存终态是前缀停止，不能补成完整反馈链。cell03 的唯一尝试在物理前因加载模块 census 不匹配失败，原 claim 和失败记录保持不动；cell02 的跨入口持久排他围栏仍未建立。

本次新增只读消费者，固定选择已有完成记录 00、01、08、09，使用既有 report、completion 和 authority 哈希调用现有 chunked host 保存检查，再联合检查固定模型哈希、同一 prepare 哈希和每格 10、15 两个绑定编码返回。固定 16 格分母及名称，未选择的 12 格保持 `not_inspected`，并非证明其缺失。单格来源或身份失败时保留错误，四格联结为 false；命令以非零码退出。既有源码或保存数据漂移也必须拒绝，不修改它们。

## 当前只读重验

2026-10-30 在禁用 CUDA 可见性的进程中，四份实际保存记录均通过既有来源和身份检查，模型张量哈希均为 `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`，每格均有 10、15 两个身份绑定的编码返回。这是已有输出的重新读取，不是新推理或物理实验；检查结论只对当时源码和文件字节成立。完整四格重验后另加入了固定 roster 名称检查，该检查由离线测试验证。

联结表示来源可信前提下保存记录报告相同模型哈希，不证明跨进程是同一内存对象、`cuda:0` 是同一物理 GPU，或不存在未记录的模型变化。真实物理/神经 parity、全部 16 格、独立新根、完整 saved strict 和科学准入仍未完成；这里不声称公平、泛化或科学通过。

## 安全复现命令

在 `le-wm-main` 下只读重验四份保存记录（源码闭包重算可能耗时）：

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m experiments.spectral_reachability.e3_cross_task_identity_join_codex_20261030_v1
```

运行 CPU 替身及既有身份相关回归：

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_e3_cross_task_identity_join_codex_20261030_v1.py \
  tests/test_agent_e3_identity_chunk_transport_20261029_v1.py \
  tests/test_agent_e3_encoder_identity_20261029_v1.py
```

以上均不授权或启动生产，运行命令不创建 claim、不加载模型、不运行物理。主线程若要推进缺失格，仍须独立处理跨入口排他、已有 attempt 的恢复决定、资源协调和新的来源闭包；不能把本联结作为启动许可。
