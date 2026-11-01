# P1：实际 cp5 上的逐动作坐标因果对照（运行前固定）

本轮不把“误差相消”解释为统一偏移，也不假设全部问题必然属于理论或代码。
主线程负责审计与执行；Nash 实现模型侧，Pasteur 实现物理侧，二者只新增独占文件。
神经计算仅主机 CUDA；冻结参数及目标、归一化、门槛，不拟合修正器。

## 已有证据与本轮增量

旧 observed suffix FD 已完成：4 个旧根、selected/zero 两种前缀、零动作附近的
两个重复 packet 参数，共 72 个探针。模型 FD 与 AD 在 h=.01 的余弦约
0.999985–0.9999996，但模型与物理响应不一致，且一条件真实 RGB 响应为零。
不能用该证据认证新优化器的十个独立原始动作坐标或新的实际历史。

新对照固定为已完成 observed recompile v2 的全部四个条件（两个已知根 ×
renderer off/on），锚点均为原 suffix，即 observed 优化 iterate0。
不使用 iterate8 的物理好坏选择锚点，不剔除失败条件，不声称独立确认。

## 固定干预

每条件 baseline 一次；raw suffix[5,2] 十坐标各 h=.01/.001、正负各一次，
合计 41 × 4 = 164 次预测与 164 个完整物理分支。每分支原 prefix5 不变，
完整原生 root/warmup/prefix 重放，在发出 step6 前认证原 cp5 的状态、
快照、51 个 prefix 图像与公开调用返回。

扰动精确定义为 `float32(float64(anchor_float32_coordinate) + sign*h)`，
其他坐标的原float32位串不变；记录 nominal h 和实际 plus-minus float32 跨度；
有限差分用后者作分母。任何越过 raw box [-1,1] 的点不得 clip、投影或替换。
此为 ambient actuator-box 诊断，点可能超出原 logit L2 球，不能宣称为原算法
可部署候选、原约束内改善或 P1 准入。运行前全体合法性检查不通过就停止。

模型使用同一实际 cp5 编码历史、原始 incoming action、原 goal192；
baseline 需与封存 iterate0 同上下文预测逐项比对。物理 baseline 需与原
完整分支复现比对。后续编码全部终态，不按响应或收益选图。

## 判别而非预设结论

1. AD vs 模型 FD：排查梯度、尺度、reshape、float32 消减；两尺度分别报告。
2. 模型 FD vs 真实 RGB FD：同动作、同历史、同时间比较局部响应。
3. 直接 scalar Q FD 与向量响应/残差：区分动作响应误差、基点偏差及交互，
   不以全向量余弦高就断言目标梯度正确。
4. 真实编码 Q vs 原生全物理指标：保留 xy、angle、velocity、reward、success，
   不把部分指标下降当作任务成功。真实物理状态仅用于诊断，禁止进入选择器。
5. 保存零响应、不可见/像素量化、接触非光滑、尺度依赖和失败分母，不加 epsilon
   伪造 cosine；不可定义的比较保留 null。

预期全体物理预算：164 env、328 reset、3280 public step、33456 physics step、
10824 PNG render。物理 wall cap1800s、输出 cap1GiB、disk floor1GiB；模型预测
164 次、无新增 AD/模型训练；终态编码由独立合同预声明后执行。

## 更广因素清单

既有因素账本继续保留：源码/序列化身份、root/freshness、动作单位与一次归一化、
历史排列与因果时间、预测输出索引、renderer/goal 一致性、重放确定性、AD链、
动作约束、候选库覆盖、选择目标、真实能力与任务指标、等预算基线。
本轮不能消除 checkpoint 对历史训练 dataset/scaler 的来源认证缺口，也不能
自动认证 fresh-root 泛化。只有定位可复现实现错误后才修改相应新版本并重跑。
P1/P2/P3 科学准入保持 false；本文件不是“所有因素已排除”的声明。
