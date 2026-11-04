# 旧 ATM/MSE P1：stage、seed 与 outcome 的依赖证据

本报告只审计旧 P1 固定的 100 条 provisional 和 140 条 external 记录，不是 D2P P1。没有训练、调整 gate、alpha 或挑选新数据。原 gate 中“可辨识”表示该 stratum 内的原匹配比较可计算，**不表示 stratum 之间统计独立**。

## 可辨识 stratum 与依赖

下表沿用原匹配池；候选数不是独立样本数。复用组件是观测到的正向依赖关系的连通分组，不是独立单元或有效样本量。

| Panel | 环境 | stage | 原可辨识候选数 | 依赖证据 |
|---|---|---:|---:|---|
| provisional | PointMaze | 5 | 10 | 与 stage 10/25/50/75 通过声明源数据和 outcome episode 重叠连通；部分还共享声明训练谱系 |
| provisional | PointMaze | 10 | 8 | 同上 |
| provisional | PointMaze | 25 | 2 | 同上 |
| provisional | PointMaze | 50 | 3 | 同上 |
| provisional | PointMaze | 75 | 2 | 同上 |
| provisional | PushT | 5 | 10 | 与 stage 10 通过声明训练谱系和诊断 outcome episode 重叠连通 |
| provisional | PushT | 10 | 6 | 同上 |
| external | PointMaze | 5 | 2 | 与 stage 10 的**诊断** outcome export 共享声明数据及 episode；外部 outcome 另见下文 |
| external | PointMaze | 10 | 3 | 同上 |
| external | PushT | 75 | 2 | 在原可辨识 stratum 图中无其他节点连接；不能据此认证独立 |

Provisional PushT stage 25/50/75；external PointMaze stage 25/50/75、PushT stage 10/20/30/50/100 未通过原匹配可辨识条件，未删除：全部 240 记录仍在全记录复用目录中。原可辨识图分别有 11 和 1 条跨 stratum 边；边的例子及各候选来源在 dependency graph JSON。0 个单元被现存字段充分认证为相互独立，表示**无法认证，不是证明独立实验数量为零**。

## 全记录核验与根身份边界

- 两 panel 跨面板有 50 对相同 checkpoint 路径，同时也有 50 对字节相同的诊断 outcome export；不能把两 panel 当独立模型复制。计数是记录对，不是独立样本数。
- 原 external PointMaze 图中 stage 5/10 的直接连边仅是**诊断 outcome** 的共享，不直接证明实际外部 planning outcome 的根复用。
- 找到 PointMaze panel manifest，其列出的 50 个 evidence 路径均匹配该批记录，manifest 指向 `results/pointmaze_external_roots_20261013/jepa_external_enriched_export.json`。对当前 export、各自 fit/outcome 的 episode 和 window key 按现存 evaluator 的过滤顺序重建，并从记录的 intervention paired_roots 除以 2 推算每条 100 个源根，且 planning 根数一致：50 条记录重建出的 100 个 key 完全相同，跨五个 stage 每一对均有 100/100 key 重叠。完整 key 和逐记录路径在补充 JSON。
- 此结果**有条件**：manifest 没有保存外部 export 的运行时内容哈希、逐根 key、`max_roots` 或执行代码哈希，无法证明当前 35MB export 内容/顺序和当前 evaluator 与当时完全一致。故为条件重建，而非确证的历史外部根身份。当前 export SHA-256 为 `34383c9345ceacbd7df852e6569aa86b85a7f32ddedac893a4fbad02099c2950`。action intervention 和 observed-action planning 在现存代码使用同一 `rows`，不是两份独立根。
- 90 条 external PushT 记录都声明 planning seed `20261013` 和 local episode index 0–11。现存 evaluator 将同一 planning seed 交给规划器，但这些声明本身不是带来源和状态哈希的模拟器根身份认证。所有记录对有 4005 次该声明相同；不要当 4005 个独立观测。
- `dataset_sha256` 的语义依数据导出流程而异：旧 PointMaze 值是 frozen enriched export 的文件哈希，不是原始物理采集的完整身份证明；external PushT 的相应值是 split manifest 的哈希。不同哈希也不能证明底层根不重叠。相同 episode 编号只在声明 dataset hash 一致时作为正向连接，不同或缺失标识为未知。
- 训练谱系通过 checkpoint 所在目录、环境、架构和 seed 的声明组合识别，并未认证训练数据流、初始化或独立运行；不同路径/seed 也不自动证明训练独立。

## 新 panel 的具体记录要求

在任何新 outcome 被查看前冻结：原始数据集命名空间与内容哈希、根的不可变全局 ID 和源状态/内容哈希、逐根 fit/calibration/selection/outcome 分配与跨 stage 的根复用矩阵；逐 checkpoint 保存训练运行 manifest、训练数据与初始化及 seed、代码和权重哈希；分别记录 intervention 与 planning 的具体根、顺序、截断参数、停止/失败/删失和成本。若有意配对复用，应在预先设计中明确依赖层次与分析单位；否则先验证根及训练谱系不重叠。上述是未来设计要求，不是对旧结果的补充认证。

## 可复算产物

- 前序固定结果 SHA-256：`64e5ba46abf2aec9c9c8d26194faaab557a5ff7bb33c3b3cf96aaed6a693e287`。
- 可辨识池依赖图 `le-wm-main/results/legacy_p1_dependency_graph_20261102_exclusive_RUN01/report.json`，SHA-256 `de2e7394e42a42f6072c821911bd4c132dbfdaa4d2aa942a93c85a3d322ed500`。
- 全记录及条件根重建 `le-wm-main/results/legacy_p1_dependency_inventory_20261102_exclusive_RUN01/report.json`，SHA-256 `b3327218403fae62dfd04e71007498883ad5fbfc4dfc26cb74bd53c5312dbe65`。
- PointMaze panel manifest SHA-256 `0e316e64d792a5d18d4ad4c693e39905c172f64b1021d5214861b8b2d4e983db`。
- 独占脚本 `legacy_p1_dependency_graph_20261102_exclusive.py`、`legacy_p1_dependency_inventory_20261102_exclusive.py`；测试 `test_legacy_p1_dependency_inventory_20261102_exclusive.py`。运行输出目录必须不存在，以避免覆盖。
