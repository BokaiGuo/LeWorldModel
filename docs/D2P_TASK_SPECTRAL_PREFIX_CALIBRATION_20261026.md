# 任务加权谱前缀已接入 CUDA 与真实执行

结论：补上了当前 PushT 固定随机前缀路径的一个核心方法缺口，但尚未得到非贪心 P1 证据。不能把新代码跑通当作原研究闭环。

## 从原方案落实的内容

新增 `pusht_task_spectral_bank.py`：在根状态的零动作 nominal 上计算完整单 token Jacobian，经同一任务投影 Q 加权后做 SVD。以前缀 atanh 参数为坐标，生成 literal zero、阻尼 DLS progress（范数上限 .75）及两个谱方向的正负 .25/.75 半径，共 10 个程序前缀。SVD 符号以最大绝对坐标规范化；不读真实后继、成功或物理标签，不训练模块。固定半径是本轮校准设置，不宣称最优、信赖域有效或动作安全。

新增 runner `run_pusht_task_spectral_bank_cuda.py`：使用同一四个已知校准根、官方冻结 checkpoint、目标集合与 Q；每个候选在自身预测 successor 重算 suffix Jacobian，并用原 stage + beta * own V 选择。完整四根计划和摘要全部保存后，才执行候选物理分支。候选不按非贪心资格筛选，不因真实结果换候选。

原编译器的 `spectral_prefix_generation=false` 仍诚实保留：它本身只接受输入 bank；本轮动力学原生 bank 的来源由新 runner 保存的根 Jacobian、nominal、谱方向、参数与封存源码共同证明。不覆盖旧封存代码或报告。

## 实际证据

`le-wm-main/results/pusht_task_spectral_bank4_v1_20261026_cuda/report.json`：`calibration_completed`，进程退出 0，9 项完整性门全 true。168 个实际 CUDA 规划预测样本（每根根 nominal/Jacobian 2 次，加 10 候选各 4 次）；40 个真实执行分支全部保留，终态编码均在主机 CUDA，无 CPU 神经降级。每根保存独立 augmented-lstsq 数组复算，共 40 次，参数、buffer、eval、输入、所有计划和源码封存核验通过。

| 根 | 选中／greedy | 选中状态 | 选中物体 pose cost | 零前缀 pose cost | 该根观测丢失分支 |
|---|---|---|---:|---:|---:|
| 0 | prefix_7 / prefix_7 | completed | 59.02849 | 46.01346 | 0 |
| 1 | prefix_6 / prefix_6 | completed | 120.02555 | 74.54012 | 1 |
| 2 | prefix_7 / prefix_7 | completed | 93.05557 | 57.31855 | 0 |
| 3 | prefix_7 / prefix_7 | completed | 99.35119 | 27.31225 | 1 |

四个选中程序都完整执行，但物体 pose cost 均差于本库零前缀。official position error 在根1/2改善、根0/3变差，不能混用两指标。全部40分支 official success 为 false。两个非选中分支观察丢失由同一逐primitive RGB guard停止，未从结果删除。当前不是独立新根、强基线等预算或多轮闭环实验，也不是一般理论负结果。

## 能力评分的含义：下降不等于可控性变强

新增只读分析 `analyze_pusht_task_spectral_bank.py`，保存产物：
`le-wm-main/results/pusht_task_spectral_bank4_decomposition_v1_20261026.json`。

令 nominal 后缀误差为 C，DLS 正则化后值为 V，局部动作改善为 A=C−V，则精确有：

`V_zero − V_selected = (C_zero − C_selected) + (A_selected − A_zero)`。

| 根 | V 下降 | nominal目标残差分量 | 增量线性动作改善分量 |
|---|---:|---:|---:|
| 0 | 0.156713 | 0.160713 | −0.004001 |
| 1 | 0.289730 | 0.207156 | +0.082574 |
| 2 | 0.238726 | 0.260035 | −0.021309 |
| 3 | 0.504688 | 0.552385 | −0.047697 |

三根的 V 虽下降，但相对零前缀的额外线性动作改善反而较小。仅凭 V 下降标注“改变后续控制能力”会把目标残差降低误读为能力增强。这是机制解释／报告的协议风险，不表示原 proposal 的 V 公式写错，也不是已经识别的因果效应；残差与 Jacobian 同时改变。

下一项科学缺口仍是同任务度量下，具有明确动作响应增量、非贪心阶段行为及真实下游收益的程序，并在新根上验证。不能用改 beta、调 margin 或删不可见分支代替这一证据。
