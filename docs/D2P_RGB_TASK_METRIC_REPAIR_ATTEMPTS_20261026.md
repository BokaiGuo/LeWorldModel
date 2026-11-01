# RGB-only 临时任务度量修复尝试

本轮尝试修复“物体不动、推杆移动仍被计为有效响应”的任务合同缺口。方法只使用因果可得当前/目标 RGB 与冻结官方 encoder；物理状态仅作后续 evaluator 标签，不用于拟合、选参或生成度量。不训练 encoder、predictor、policy、critic、adapter 或物理 readout。

这不是已经解决 P1 的报告。以下新方法均在已知四个校准根上探索，不能作为独立确认，也不能用单测/源码封存通过替代物理控制效益。

## 尝试一：目标物体图像扰动的局部子空间

代码：`pusht_goal_object_tangent_metric.py` 与 `run_pusht_goal_object_metric_cuda.py`。

目标 RGB 中规范 LightSlateGray 可见物体像素，作固定 ±4 像素平移、±5 度旋转，共六种图像扰动。保留已知蓝色推杆像素。冻结 encoder 编码这些图像，在原五个 task-equivalent goal 的 nuisance span 正交补上构造六维 task span，再取其正交投影。此 Q 来自临时图像与矩阵分解，不是新增学习头。

产物：`le-wm-main/results/pusht_goal_object_metric4_v1_20261026_cuda/report.json`。主机 CUDA **28 次实际编码**，`construction_completed`、退出码 0、7 项完整性门全 true。四根 task rank 均 6。全部度量内容保存封存后才读完整探针进行评价；不按结果修改半径、rank、缩放或排除根。

## 尝试二：同时使用当前与目标 RGB 的推杆 nuisance span

代码：`pusht_scene_nuisance_metric.py` 与 `run_pusht_scene_nuisance_metric_cuda.py`。

在当前根 RGB 与目标 RGB，各用预定八个栅格位置移动可见蓝色推杆像素，保留可见灰色物体像素。16 个额外 embedding 差向量与原 goal nuisance 合并，构造20维 nuisance span。六维物体扰动方向投影到其正交补后生成临时 Q。没有加入物理后继或任务成功数据。

产物：`le-wm-main/results/pusht_scene_nuisance_metric4_v1_20261026_cuda/report.json`。主机 CUDA **68 次实际编码**，`construction_completed`、退出码 0、7 项完整性门全 true。每根 nuisance rank 20、task rank 6。

## 两种线性尝试的真实响应保留率

对同一批已保存的真实中心差分 J，报告 `||Q_new J||² / ||Q_old J||²`。这是响应能量保留率，不是物理任务成功率；降 rank 本来就会压低能量，不能以“小”当作修复成功。

| 根／上下文 | 物体五探针位姿完全一致 | 目标扰动 Q | 当前＋目标 nuisance Q |
|---|---|---:|---:|
| 0 selected | 是 | 4.180% | 1.405% |
| 0 zero | 否 | 4.820% | 0.602% |
| 1 selected | 是 | 5.329% | 4.404% |
| 1 zero | 否 | 1.779% | 0.509% |
| 2 selected | 否 | 5.005% | 3.536% |
| 2 zero | 否 | 3.422% | 0.550% |
| 3 selected | 是 | 6.259% | 6.288% |
| 3 zero | 否 | 1.916% | 0.393% |

三组物体静止的 selected 案例仍留有响应；第二种 Q 对它们的保留率反而都高于同根包含物体响应的 zero 对照。当前证据不支持“新线性 Q 已获得任务选择性”，不能部署后声称已修复控制。zero 探针也含推杆运动，故这个比较不是分离出的纯物体信号或因果估计。

## 所有图像扰动的边界

颜色掩码依赖默认 PushT renderer。只改可见色块、白色填补、最近邻旋转/平移不能重建遮挡后的物体或目标标记；抗锯齿边缘也不是完整物理 sprite。它们是明确披露的近似图像增强，不能冒充真实 simulator counterfactual、严格 task-equivalent states 或模型所学物理分布。

第一轮已目视查看保存的平移图像，确认边界/填补伪影存在。后续的真实响应检验正是为了防止把“增强在代数上构成 basis”误当物理有效性。这些结果最多说明本轮两种近似构造没有解决失配，不能推断所有 RGB-only 任务度量或原研究理论不可能。

## 非线性目标集合距离的实际检验

代码：`pusht_goal_set_metric.py` 与 `run_pusht_goal_set_metric_cuda.py`。

不用单一全局投影 Q，而把目标推杆位置视作任务无关变量：在固定16×16可见位置网格编码目标增强，再与原五个目标 embedding 合并。对每个当前/预测向量计算到完整等价目标目录的最小 MSE，而非到目录均值或固定 goal[0] 的距离。网格和预处理在读取完整 probe/control outcomes 前固定；仍保留上述近似图像合成限制。

该尝试不改变任何已执行动作或 selector，不能从重新评分推导出控制收益。正式接入时，谱生成、后缀优化、stage、reach 和 terminal 必须使用同一个目标集合目标，并记入全部额外编码/选择成本；不能只改最终评分后宣称完整 compiler 已验证。

产物：`le-wm-main/results/pusht_goal_set_metric4_v1_20261026_cuda/report.json`，`construction_completed`、退出0、8项完整性门全true；64个实际encoder batch calls、**1,024个CUDA编码样本**。每根261个目标向量（原5个加256个增强）。旧动作不变、没有新物理执行。

物体静止的根0/1/3 selected探针，其goal-set距离范围分别为0.029871、0.156026、0.022157，并没有消除对推杆运动的敏感。保存的10个实际终态与物体pose cost的描述性Spearman相关分别为−0.869、−0.294、−0.190、−0.513；根1/3各含一个guard中止分支，这里保留全部实际终态，不能当固定H2完整执行对照或效应统计。

## 2026-10-27：发现并修复完整sprite的实现问题

旧图像增强只移动RGB严格等于RoyalBlue的像素，没有搬移抗锯齿及轮廓。直接核对原目标的局部sprite区域，四根分别还有102、101、94、96个非白但非纯蓝像素。这些像素留在旧位置，使目标增强含残影。它是**实际的增强实现缺陷**，不能把受其影响的结果直接升级为干净的方法负结果。

新增 `pusht_clean_goal_sprite_catalog.py`：提取完整原sprite矩形（纯蓝包围盒加2像素margin），必须有全白外边界，清除原矩形，再完整搬移全部像素。仅在目标矩形原本完全白且可见的位置放置，不把白色矩形糊到物体/目标标记上。所有资格在RGB层、神经编码前固定，无物理outcome筛选。仍保留“这是RGB编辑而非一般物理等价证书”的限制。

新增runner `run_pusht_clean_goal_catalog_cuda.py`。产物：`le-wm-main/results/pusht_clean_goal_catalog4_v1_20261027_cuda/report.json`，`construction_completed`、退出0、8门全true。每根256网格位置中229个RGB合格、27个不合格，四根全部保留；每目录234个向量，含原5个目标。**916个实际CUDA编码样本、60个batch calls**。目视检查保存的source-removed图像，原推杆区域干净清除。

“全白margin拒绝”测试首次因fixture错误地把障碍放在margin外一行而失败，随后将注入点从第97行改到实际margin第98行，验证器未放宽。全部相关测试通过。

### 必须固定同一目录再配对，不能把目录变化当成纯sprite修复效果

清理版只允许229位置，旧版允许256位置，直接相减会混入目标集合范围变化。新增只读 `audit_pusht_goal_sprite_pair.py`，以稳定goal ID把旧版限制到与新版完全一致的234个向量身份（229增强＋原5目标），不靠数组位置，不删除任何物理探针。保存产物：`le-wm-main/results/pusht_goal_sprite_pair_audit4_v1_20261027.json`。

| 根／上下文 | 旧partial sprite，同一229位置的距离范围 | 完整sprite，同一229位置的距离范围 | 物体位姿静止 |
|---|---:|---:|---|
| 0 selected | 0.014511 | 0.015523 | 是 |
| 0 zero | 0.059109 | 0.070096 | 否 |
| 1 selected | 0.145921 | 0.099762 | 是 |
| 1 zero | 0.058707 | 0.058707 | 否 |
| 2 selected | 0.158699 | 0.104536 | 否 |
| 2 zero | 0.070204 | 0.064613 | 否 |
| 3 selected | 0.036156 | 0.012766 | 是 |
| 3 zero | 0.078912 | 0.059688 | 否 |

修复确实改变部分评分，特别是根1/3物体静止案例；但根0敏感度未改善，且静止物体的距离仍变动。完整sprite目录评分与物理pose cost的四根终态描述性Spearman仍为−0.719、−0.638、−0.006、−0.356（同样保留中止分支，不作固定时域胜负结论）。不能宣称任务度量已修好，也不能以这个已识别的实现错误解释全部控制失败。

## 本轮交付边界

四轮主机CUDA构造共**2,036个编码样本**；没有把纯数组CPU检查冒充神经实验，没有CPU神经回退。没有新增物理控制策略或物理执行，实际探针与终态沿用已封存校准产物。源码、输入、参数、buffers、frozen/eval和目录内容均核验；旧错误代码/报告保持封存，不覆盖。

新增纯合同/配对测试23项；修复保留了“按同一goal身份配对”的独立检查。它们与编码/物理证据分账，不能授予P1。当前仍缺部署可得且经独立验证的任务一致目标，以及非贪心程序的真实下游收益；原C1–C3、三组实验和P0–P5全范围未关闭。
