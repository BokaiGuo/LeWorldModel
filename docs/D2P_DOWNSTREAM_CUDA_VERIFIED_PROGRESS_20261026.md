# 并行任务整合后的主机 CUDA 进度

2026-10-26 22:47 后，主线程核验；原研究范围未缩小，P1/P2/P3 尚未准入。

## 已新增的真实执行证据

| 实验 | 实际计算 / 执行 | 状态与边界 |
|---|---|---|
| 真实前缀反馈固定后缀预测 | 64 次精确真实前缀重放、64 CUDA prediction samples | completed，6 项 gates true；61/64 候选预测误差下降，未改变真实终态，非效力证据 |
| 组合接口 split/full replay | 64 固定双段程序、384 CUDA predict 调用/样本 | completed，6 项 gates true；上下文精确一致，原 forecast 1e-5 匹配；无新搜索或真实执行 |
| 预算 guard integration | 64 CUDA predictions、4 次预先拦截 | cuda_integration_pass，8 项 gates true；各根第17次均未进入模型，无新物理执行或计算优势结论 |

产物分别为 `le-wm-main/results/pusht_observed_prefix_prediction4_v1_20261026_cuda/report.json`、`results/pusht_composition_replay4_v4_20261026_cuda/report.json`、`results/pusht_d2p_budget_guard_probe4x16_v1_20261026_cuda/report.json`（后两者同以 le-wm-main 为根）。三项均有来源封存、输入绑定、冻结模型检查，真实进程退出码均 0，无 CPU 神经回退或训练。

独立反馈配对复算：`le-wm-main/results/pusht_observed_prefix_prediction_analysis4_v1_20261026.json`。以根身份与候选 ID 配对，保存误差完整二次分解，不将描述统计当新根泛化或因果机制证明。第四根的 observed-prefix 终态向量误差 MSE 中位数仍约 0.414，不能宣称反馈已全面修复。

## 代码问题与科学边界分开

- 主线程审查发现 P2 context 可接受非数值／不规则输入，已由智能体修复，补上有限数值矩阵、显式布局及状态动作对齐检查。
- CUDA 组合接线捕获 NumPy scalar JSON 序列化与历史 digest 类型误用，已修复并保留两轮失败目录，不能当方法负结果。
- 原始度量和投影度量均有物理任务排序失配；官方 agent+object 任务不同于 object-only 任务，不允许事后换指标择优。
- 反馈降低预测误差并不等于真实任务收益；局部误差和度量问题仍在。

## 仍未闭环的原要求

P1：独立根上的非贪心动作、真实未来能力改变、闭环任务收益及等预算强对照仍缺完整成立证据。P2：固定双段一致性不是多程序组合搜索相对无组合 SRA 的真实增益。P3：硬预算拦截不是动力学调度优于固定或打乱信号的质量—成本曲线。P4/P5：长时闭环、三组原实验、同模型多任务与独立模型复制，以及 faithful GRASP/GN/kinodynamic 等比较仍需按原前置条件验证。

旧 frontier JSON 是生成时刻快照，不能将其作为本轮新 CUDA 接线不存在的证据。原 budget probe 说明文档属于该轮 source seal，本页另外记录最新结果，避免修改已封存说明而破坏当前来源验证。
