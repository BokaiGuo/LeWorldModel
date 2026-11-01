# 固定前缀后缀重编译：上下文与动作幅度消融（探索性校准）

本协议在新编译/物理执行前固定。已有live-feedback12分支不重跑。新80预测显示teacher在小探针上更保真，故检验具体修复而非继续重复同一反馈。两个known roots，不是独立确认；此次设计受到旧校准证据启发，不能称全新无偏确认。

固定原八个root/condition/GN-D2P prefix5，不更改动作、任务或目标。不按表现删root。每prefix三个后缀编译条件：`open_full`、`teacher_full`、`teacher_small`；前二仅上下文末帧不同，后二仅新raw动作幅度不同。teacher取自己真实prefix编码；open取自己模型prefix预测。旧两帧真实latent及between-state动作相同。全24case保留，无早期择优。

全条件统一full10 logit、原ridge damping=.002、原L2参数球及两半径、zero/DLS/正负top2谱方向共10候选；按原identity mean192非线性终态代价选择，stable index tie。raw动作分别`tanh(u)`与`.1*tanh(u)`，后者收缩规划动作，不改变模型normalizer。每条件都在自身动作映射下重算零点Jacobian和谱库，不能把full库简单缩放后称同一优化。

每case固定：nominal1＋Jacobian forward1＋10候选预测＋winner精确重放1=13 predict/API/AE/predictor/pred_proj samples。总312；24 Jacobian、4608数学输出VJP方向、48真实SVD。AD引擎调用和完整FLOPs不声称相等；动作限制不同是显式消融变量。新视觉编码0，旧编码历史成本另列，不称免费。全部模块原始输入输出、Jacobian、候选与winner先封存，再放行物理。

主机CUDA，原冻结模型、原mean/std，禁TF32/训练/Q拟合/readout/CPU神经fallback；900s含verify/load，64MiB输出/4MiB最终保留/200MiB磁盘floor。失败不重试覆盖，固定24分母。没有真实执行结果前不称改善；物理执行必须在完整saved consumer验收后，保持原reset/warmup/prefix/目标/渲染，逐位核prefix5原轨迹且不在suffix前reset/setter。真实任务联合agent+object位置、角度及native flags优先，state7距离/速度/latent分栏。不得按事后结果选择更好条件作为确认方案。
