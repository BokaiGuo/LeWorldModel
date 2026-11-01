# D2P 并行派发 v5（2026-10-28）

状态：研究目标仍未闭环。本轮复用六个已有子智能体，未创建下级智能体；主线程负责源码审计、实验授权、主机 CUDA 和最终验收。

## 独占任务

| 子智能体 | 本轮交付范围 | 独占新文件前缀 |
| --- | --- | --- |
| Poincare | P1 prefix 前后固定能力探针合同，不以终距代替能力 | agent_mixedmode_capability_probe_20261028_v1 |
| Hegel | mixedmode222 真实执行适配；精确动作/上下文相等才可复用 | agent_mixedmode_physical_20261028_v1 |
| Bohr | 固定真实执行长度下的规划深度与预算对照合同 | agent_p3_fixed_execution_contract_20261028_v1 |
| Nash | arbitrary latent / learned prior / compiled witness 三臂统一执行接口 | agent_subgoal_gap_runner_20261028_v1 |
| Pauli | 官方整包与成员离线验证，不解 pickle、不加载模型 | agent_hilewm_asset_verify_20261028_v1 |
| Jason | GRASP、Hi-LeWM、LeWM、PlanWAM 本地运行入口与资产缺口 | D2P_BASELINE_EXECUTION_MATRIX_JASON_20261028_v1 |

共同边界：保留全部已有改动和封存证据；只写独占新文件。子智能体不得执行 CUDA、神经 CPU、物理实验或网络下载。禁止训练、按结果挑根、调低科学门槛。作者报告值与本地实验分栏，未知保留 unknown。

## 主线程实际进展

- mixedmode222 MAIN 测试：60 passed in 69.83s。
- mixedmode222 prepare 完成，SHA256：`8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b`。
- mixedmode222 主机 CUDA 已启动，输出目录 `le-wm-main/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01`，执行 session 20927。启动不代表完成；需终态报告、严格 consume 和独立复算。
- P3 common-budget CUDA 进程已正常退出；报告 SHA256：`3f7004aeb92d09a8f6010cd5a5c078a0660b19574203c38d5ae3fbe72dfab67e`。严格 consume 已启动，session 63508，尚待终态。仅等 predict 预算，不是等执行 horizon，不据此宣称物理收益。
- Hi-LeWM 唯一整包下载 session 69299 本轮核查仍运行；不重复启动、不把资产失败计为基线失败。

当前实验仍是已知两个根的四个条件，不是四个独立新根。P1、独立确认、强基线真实闭环、P2/P3 科学结论均不得因派发或单测通过而打开。
