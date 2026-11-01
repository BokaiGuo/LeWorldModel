# P1 动作归一化：当前源码与统计口径核查

这是对已有统计和当前源码的只读核查，没有运行模型、修改 normalizer、
选择动作或打开 P1。

当前工作树的 `le-wm-main/train.py` 先对整个已加载 dataset 的非图像列
调用 `get_column_normalizer`，随后才随机划分 train/validation；
`le-wm-main/utils.py` 使用 PyTorch 的 `data.std()` 样本标准差。
当前 `le-wm-main/eval.py` 则从其配置的数据集拟合
`sklearn.preprocessing.StandardScaler`。以上只描述当前工作树，不能
证明公开 checkpoint 的历史代码、数据版本或训练时配置。

已有归档统计 receipt
`le-wm-main/results/pusht_d2p_archive_action_statistics_v1_20261026.json`
记录 2,336,736 个有效动作行，其样本和总体标准差分别为：

| 动作维 | 样本标准差 | 总体标准差 | 差值 |
| --- | ---: | ---: | ---: |
| 0 | 0.20846748745 | 0.20846744285 | 约 4.46e-8 |
| 1 | 0.20674867061 | 0.20674862637 | 约 4.42e-8 |

同一 receipt 中样本标准差与部署常量的差分别约为 0.00657 和
0.00655。因此，**对这份归档而言，样本与总体标准差的口径差异
不能解释归档和部署常量之间约 3.3% 的差异**。这不是训练来源
认证，也没有证明两套值中哪一套是 checkpoint 的真实训练统计。
receipt 的统计是独立 float64 分块累积，不是对 PyTorch float32
归约的逐位复刻；它只哈希了 action 字段而不是整个归档。

此前已对 19 条固定动作做了两套 normalizer 的主机 CUDA 敏感性
对照，见 `D2P_P1_ACTION_NORMALIZER_SENSITIVITY_20261030_MAIN.md`。
该结果并不排除其他未测 normalizer、优化后动作变化或未认证
历史来源，也不能通过事后选择误差更小的统计解锁独立实验。
`P1_admission`、训练来源认证及独立确认均维持原状态。
