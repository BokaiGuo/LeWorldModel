# P1 feedback goal contract：有界离线审计

范围固定为原 root_index=1、selected/greedy/zero × imagined/actual 六个封存计划。
读取保存的 60 个候选终点、12 个物理端点编码与五个目标变体；不新调用
model、CUDA、env/reset/step/render，不训练、不挑根、不调阈值、不改封存输入。

## 执行

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_feedback_goal_contract_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p1_feedback_goal_contract_audit_20261027_v1 --output results/p1_feedback_goal_contract_audit_20261027_v1_offline_v1
```

输出目录必须不存在。先核对上游 source/archive/args 和 protocol/input hashes，
保存新审计 protocol（含源代码片段与本模块/tests/本文档哈希），再进行计算；
结束复验全部输入。失败抛异常，不生成伪通过 report；不覆盖旧目录。
protocol 是预计算内容清单，非可信时间戳或执行证明。

## 判定与边界

独立重建原 finite-goal SVD 投影；核对 Q 对称、幂等、秩，所有候选成本及
稳定 argmin，物理 commands 与赢家、实际物理 goal 与 goal RGB 哈希、
step5 像素、旧两 latent 隔离、incoming 两 packet 的顺序/归一化，
以及每个保存端点针对其对应预测的残差。官方 evaluator 只做 AST 公式比对，
再以纯 NumPy 复算 success/distance；不实例化 simulator。

特别分开：原 Q 消去 agent-only 目标变化，是有意 surrogate；官方 success
却用前四位置坐标（含 agent），因此两者不等价。五个已保存的实际目标渲染状态
可作有限见证，不能当作新执行的终点，也不能证明物体任务保真、闭环效益或
解释六条轨迹的因果。零 nominal 误差改善不等于动作响应改善。

图像检查限定为同一运行期 render/预处理路径及实际目标几何、RGB 来源。
不声称绝对相机标定或 checkpoint 训练相机已证明；action 标准化也只能排除
本执行路径的错序/额外除100，不能证明 checkpoint 训练约定。旧 helper 的
除100不在当前 packet 路径，不当成已执行 bug。模型官方 criterion 为原坐标 SSE，
当前为 Q-MSE：固定192倍标量本身不改排序，投影 Q 则实质改变目标。

## 验证记录

2026-10-27：上述非神经测试命令实际执行，25 passed（0.19s）。
上游 collect/source 检查实际通过，预先验证1054个不同路径的来源及归档；
最终完整路径数和数值见新离线 report/input_verification。

本模块严格非神经测试覆盖目标/Q/上下文/候选/端点错接、重复/缺失候选、
非有限数值、实际渲染偏移、官方阈值改动与输入哈希变更。
真实离线结果仅以同名新目录 report.json 为准，P1_admission 与 efficacy_verified
始终为 false。本文档在离线计算前冻结，不回填或修改封存记录；终端测试记录
与新 report 共同构成本次交付，不将 fixture 通过称为真实物理证据。
