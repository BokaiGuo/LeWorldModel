# JEPA 原生 PushT：固定预算物理可行性诊断

按[事前协议](PC_JEPA_NATIVE_FEASIBILITY_ORACLE_PROTOCOL_20261104_CODEX.md)，以评价专用目标状态和真实后端搜索同一个已暴露根的 15 步开环序列。实际执行 194 个候选：两条已执行历史链，以及零链、response 链、CEM 链三个起点各四代、每代 16 个截断高斯候选。搜索使用完整 15×2 动作坐标；每个候选均从 reset 完整重放 10 步暖身，并逐元素验证暖身后的 RGB、state 和 proprio。

没有候选达到原成功条件。按预定联合距离与角度的归一化平方和，最好候选为第 113 条，评分 `84.9280`，其 agent 位置误差 `28.5395`、物体位置误差 `126.1331`、联合位置误差 `129.3215`、角度误差 `2.2921` 弧度。原历史 response 和 CEM 链在同一重放中的评分分别为 `115.4124`、`119.2330`。该评分仅用于此特权搜索，不能替代独立的成功条件，也不能将最优候选与视觉规划器的预算公平比较。

[结果和全部动作](../le-wm-main/results/pc_jepa_native_feasibility_oracle_20261104_codex/result.json)与同目录的 `actions.npy`、`best_trajectory.npz` 封存；[独立审计](../le-wm-main/results/pc_jepa_native_feasibility_oracle_20261104_codex/audit.json)重算 194 条终点评分和成功标签，精确复现随机样本、精英更新及起点，并从 reset 重放最优链的 16 个 RGB／状态／本体边界，全部一致。审计未重跑其余 193 条物理轨迹。

**结论仅为该固定预算搜索未找到 witness；目标在合法动作域和 15 步内是否可解仍未知。** 该诊断既不能证明不可解，也不能证明响应编译方法失效、基线充分或任何 PC gate 通过；按协议停止对此暴露根追加搜索。
