# 六 baseline 前瞻等预算候选队列（仅 planning）

本版交付三个独占文件：同 stem 源文件、`test_` 前缀测试和本文。
不修改旧 producer、sealed 预算或共享 progress/dispatch/closure 文档。
仅生成 stdout JSON 合同；无训练、adapter、Q 拟合、CPU 神经实验、模型/物理调用、删除或发布。

## 来源与真实单位

读取 `experiments/spectral_reachability/p1_moving_baseline_budget_matrix_20261027_v1.py`
及 `results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json`。
CLI 固定核验两者 SHA256，分别为
`229f8f6116bc5771d3738b5ecafe52026a00d439e3d63b0e5e9efe2f08fe94fb`、
`721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc`。
不递归打开 comparison/outcome，不以既有质量结果选择算法或调参。

| 固定队列顺序 | 两根 predict 参考 | AD forward 子集 | scalar-loss reverse | output-VJP 方向 |
|---|---:|---:|---:|---:|
| moving_gradient_open_loop | 272 | 128 | 64 | 未独立计量 |
| CEM | 268 | 0 | 未独立计量 | 未独立计量 |
| kernel_faithful_mppi_nominal | 144 | 0 | 未独立计量 | 未独立计量 |
| fixed_H2_kinodynamic_beam16x4 | 168 | 0 | 未独立计量 | 未独立计量 |
| GRASP_H3_bounded | 3604 | 3600 | 1800 | 未独立计量 |
| joint_H2_spectral_packet_mapped | 176 | 4 | 未独立计量 | 384 |

这些是配置/计量口径参考，不是未来预算预测，不比较旧不等预算实验的性能。
GRASP H3 是观测历史，未来仍为 H2。SRA 有 J=2、SVD=4、bounded solve=2
（两根 producer 参考），consumer 重构 CPU 工作另列在 sealed 配置，不能当作重复神经调用。
kinodynamic 的离散重复 packet 支持域与逐时间 logits 不同；MPPI 最终 nominal、
CEM best-ever、梯度选择、SRA bank 与各自 replay 规则不能默认为同一合同。

predict / predict API / forward samples / action encoder / pred projector 保持独立轴；
AD forward 已包含在 predict 中。backward / grad API / J / J API / VJP 既有重叠又有不同语义，
不能相加；scalar loss reverse 不能等价为 192 输出方向。SVD/solve 是 CPU 单位。
image encoder/projector、internal physics/public steps、完整 pipeline walltime 也不互换。
保存的 whole-run latency 不是逐方法延迟。共享 physics/RGB cohort 不均分给方法，
历史 encoder8/projector8 仅保留一次。只有确切 single-arm physics 回执可归属；缺失恒为 null。

## JSON 合同与门禁

`plan.candidates` 依固定六方法顺序预声明未来候选；每项保存原 sealed 配置及配置 SHA、
matrix pointer、逐轴参考合同及 `prospective_budget=null`。不增加迭代/样本、缩 bank、padding、
重复算法或猜测转换率来凑等预算。`plan_sha256` 绑定整个预声明计划，包括成本语义及共享账。

`equality` 要求全部成本轴（同时保留 attempted/completed）以及 population、history、horizon、
action/proposal support、replay/feedback、selection、device/precision/batching、encoding sharing、
pipeline/budget scope 和 derivative semantics 完整且逐项匹配；缺失、未完成或不匹配均为 `unestablished`，
`gaps` 逐项给出字段、方法和原因。即便所有 predict 相同也不够。
纯 RAM `assess()` 的 `equal` 只表示输入声明的完整向量相等，不证明真实总算力相等。
生产 CLI 不接受覆盖成本或配置的参数，固定来源在本版不满足约束，因此不会输出 equal。

没有独立 freshness authority 时输出 `blocked_planning_artifact`、`ready=false`、
`launch_authorized=false`、`executable_queue=[]`。六候选仍完整保留，不筛选旧结果。
可额外传入 `--freshness FILE --freshness-sha CANONICAL_SHA --trusted-freshness-authority MAIN`。
certificate schema 为本 stem + `.freshness`，字段见测试 `certificate()`：
绑定 plan SHA、calibration matrix SHA、两个有序独立 fresh root SHA、两个排除 calibration task SHA，
issued/expiry 纳秒时间，以及未用于设计、未见 candidate cost/outcome、完整旧根排除检查等断言。
authority 必须是 MAIN 且不同于 producer；canonical SHA 和 authority 是调用方外部信任锚，
不能把本地自算 SHA 当独立授权，也不声称密码学证明 freshness。

即使 certificate 有效，也只认证外部声明，不能补齐等预算/支持域合同。
本版没有新预算 authority、runner 或 ready 路径；不会把独立 freshness 自动升级成实验许可。
后续必须由 MAIN 审核完整新合同，再由 HOST CUDA 执行；禁止 CPU 神经/物理替代。

## 命令与验收

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.moving_equal_budget_prospective_queue_20261027_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_moving_equal_budget_prospective_queue_20261027_v1.py
```

第一条仅向 stdout 输出可机器读取的完整 blocked planning JSON，不落盘或覆盖已有文件。
第二条只用 RAM mock，不读科研 outcome，不运行神经/物理或保存 numpy 实验。
失败用例包括逐轴缺失、同 predict 不同 SVD/支持域、partial/非法计数、NaN/bool、缺失填零、
CPU 向量变标量、旧根复用、自签 authority、计划换绑、过期/未来证书、SHA 错误及来源错误。
测试成功不是科学证据，也不建立 fresh-root 或 equal-total-compute 结论。

2026-10-27 本地验证：新测试及相邻预算矩阵测试合计 111 passed（新文件 81 项，旧文件 30 项）。
联合命令为在上面的 pytest 命令末尾追加
`tests/test_p1_moving_baseline_budget_matrix_20261027_v1.py`。
真实 pinned 元数据 CLI 返回六候选、126 个逐项缺口、`unestablished`、
`blocked_planning_artifact`、`ready=false`。没有提供或生成独立 freshness authority。
两个输入 SHA 与实现前一致；未执行全库测试、真实实验或任何 HOST 作业。
