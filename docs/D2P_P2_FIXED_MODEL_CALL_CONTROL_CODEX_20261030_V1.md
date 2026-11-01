# P2 固定模型调用数对照：已知根重复

## 范围与前瞻性

本次针对每根的两个已有 planner 臂，在新执行开始前固定候选 census、
调用预算和来源，并在独占目录再运行一次。原来四根及其模型/物理结果
此前已经观察过，因此这不是独立 cohort，也不是对既有实验的事后
预注册。该实现沿用原 80/40 候选方法，不是新的等候选或等全部算力方法。
原结果未覆盖；此次只做 CUDA 规划，不训练、不运行物理。

固定合同：
`le-wm-main/results/codex_p2_fixed_call_budget_20261030_v1_PREP01.json`，
SHA-256 `0494d634e5f8d78096552e1cd7908070c93512f6060d243c2e75748b3840430e`。
输出：
`le-wm-main/results/codex_p2_fixed_call_budget_20261030_v1_RUN01/report.json`，
SHA-256 `b0df439fd8e6b96b312ecfe2efbfc93af764c8e6eb02ec7bf080355efdb3ce5c`。

## 原四根和本次执行的成本

旧 run 和本次独立重读的新 run 在以下每根计数上一致：

| 资源轴（每根） | 非组合 | 条件后缀组合 |
|---|---:|---:|
| 候选评估 | 80 | 40 |
| predict 归属调用 | 168 | 168 |
| action encoder 归属调用 | 168 | 168 |
| Jacobian 请求 | 1 | 41 |
| 数学 VJP 方向 | 192 | 7872 |
| SVD | 2 | 42 |
| solve | 1 | 41 |

每种模型入口的 168 为每臂独占 164 加共享编译 4。共享工作实际执行
一次、归属每个 standalone 臂一次，因此每根实际唯一入口为 332 次
predict、332 次 action encoder；四根各为 1328 次。不能把两个臂的
168 简单相加当作实际唯一调用。可核验的是指定入口的单样本调用数，
不是 FLOPs 或全部计算量；Jacobian/VJP/SVD/solve 明显不匹配。

新 run 的四根均完成，门控 journal 及固定候选 census 校验成功；
planner 保存 consumer 在每根执行后运行，wrapper 另重读 journal
检查固定入口计数。模型参数及 buffer hash 前后一致。新 run 的八个
winner ID 和记录的 source-Q cost 与原 run 完全相同；这是已知根上同一
方法的重复，不是额外独立证据。

CUDA 环境记录为 RTX 5060 Ti、PyTorch 2.12.0+cu130。启动前检查未见 GPU
compute 进程，但该检查和专用输出目录不能保证其他旧入口或其他进程
绝对排他。模型读取/加载、GPU 上下文创建、外部 CPU 工作、设备精确时间、
FLOPs、物理与历史 acquisition 等全成本未证明相等。科学门保持
`P1_admission=false`、`P2_admission=false`。

## 复验

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/pytest -q -p no:cacheprovider \
  tests/test_codex_p2_fixed_call_budget_20261030_v1.py \
  tests/test_codex_p2_frontier_shared_cap_gate_20261030_v1.py \
  tests/test_codex_p2_knownroot_host_cuda_20261030_v1.py
.venv/bin/python -m experiments.spectral_reachability.codex_p2_fixed_call_budget_20261030_v1 audit \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p2_fixed_call_budget_20261030_v1_RUN01/planner
```

测试结果为 19 passed；复验 audit 为 CPU 读取已存数据，不重新运行 CUDA。
