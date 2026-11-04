# PointMaze 规划器归一化与独立确认边界

## 直接核对

当前谱实验
[`a1_pointmaze_proprio_spectral_cem_20261103.py`](../le-wm-main/experiments/a1_pointmaze_proprio_spectral_cem_20261103.py)
调用
[`load_model_and_dataset`](../le-wm-main/experiments/spectral_reachability/run_sra_p0_pointmaze_cuda.py)，
后者对 `data/point_maze_tail1800_1899` 创建
`PointMazeDataset(normalize_action=True)`，并将所得动作、状态、proprio 均值和标准差
传入模型预处理器。尽管目录名含 tail，实测 `states.pth` 为 `(2000, 100, 4)`、
`actions.pth` 为 `(2000, 100, 2)`、长度数组为 `(2000,)`；原始 RGB 文件只保存了
其中一部分不改变该事实。

[`PointMazeDataset`](../le-wm-main/external/jepa-wms/app/plan_common/datasets/point_maze_dset.py)
在默认未设置 `n_rollout` 时遍历全部轨迹的有效时间步计算均值与标准差。
当前开发根 1888–1895 各有 100 个有效时间步，共 800 个状态和动作时间步参与
200,000 个时间步的统计。因此如果从同一数组中再挑根作为确认集，不能声称该根
未参与预处理统计。该观察不证明检查点训练时使用了相同数据，也不证明数值变化
一定导致规划失败。

JEPA-WM 仓库的
[`DATA_STATS`](../le-wm-main/external/jepa-wms/app/plan_common/datasets/__init__.py)
声明统计来自训练数据；本地检查点与该声明的具体训练运行、数据成员和版本仍未
取得权威绑定。不能把该声明直接升级为检查点训练溯源证明。

使用保存数组的有效长度、与 loader 相同的 `torch.mean` 和默认 `torch.std` 复算：

| 统计量 | 数值 |
| --- | --- |
| 动作均值 | `[0.00007821338658686727, 0.0006003659800626338]` |
| 动作标准差 | `[0.5769129991531372, 0.577587902545929]` |
| 状态／proprio 均值 | `[1.8130563497543335, 1.9377127885818481, -0.0039275349117815495, -0.019777977839112282]` |
| 状态／proprio 标准差 | `[1.0055010318756104, 0.9630089402198792, 1.6802769899368286, 1.9319307804107666]` |

这些数值与本地 registry 的 PointMaze 对应项目逐项相同；loader 中 proprio 是
状态的克隆。因此仅将当前这套数值换成该 registry，不会改变动态 proprio 或动作
的归一化数值。相等不证明数据成员、训练运行或外部权威来源；也不说明这些统计
对来自其他分布的输入一定合适。

## 既有对照的适用范围

旧的 [registry 对照](PC_POINTMAZE_REGISTRY_NORMALIZATION_RESULT_20261102.md)
使用根 1880–1883、固定 proprio 均值与固定动作库，报告与其先前基线的 256 个
分数完全相同。它既没有针对当前 1888–1895 真实 RGB 动态 proprio 反馈规划器
做对照，也没有认证检查点训练数据或确认根独立性。上面的独立数值复算进一步排除
了“当前 loader 数值与本地 registry 数值不同”这一特定解释；它不能排除其他输入、
时间绑定或归一化次数问题。

## 确认实验要求

确认前，必须预先固定和保存预处理统计、来源与哈希，并保证确认根未用于拟合这些
统计；不得用确认集、其后续帧或物理 outcome 重拟合。可对固定 registry 统计开展
单独标记的开发敏感性实验，但需统一动作与 proprio 两条归一化接线、目标和反馈
输入；未经独立来源绑定，训练成员仍记为 unknown。比较方法必须共享同一预处理
合同，且不能因为某套统计的结果较好就事后选为确认方案。

未来独立根生成还需分别证明未参与方法开发与历史测试；“未用于预处理拟合”、
“未参与方法开发”和“未出现在 checkpoint 训练数据”是三个不同命题。
本记录是源码及本地数组核对，不是新神经推理、物理实验或 P1 通过证据。
