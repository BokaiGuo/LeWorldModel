# Moving-history目标轮廓：固定已知任务的协议干预

## 问题与边界

archive两episode同body显示干预已改善冻结模型factual预测，但不直接证明构造任务的
规划排序或非贪心机制。此轮固定已有moving-history两slot及其GN/D2P动作，
只改变送入编码器的轮廓显示条件，检查预测保真及目标距离排序。
这是复用已知任务的校准，不是新根确认，不由结果重选根、动作、阈值或损失。

## 不变部分

- 原placement recipe、reset、真实非零warmup10、任务root及物理goal。
- 原GN/D2P已选full10命令，float32 raw history与动作；不重新优化或选择。
- 原冻结checkpoint、native192表示、预处理、动作normalizer、history3与packet5。
- 原native成功判据（四维XY误差<20且角误差<π/9）；不以latent改善替代成功。
- 原环境保持with_target=False；仅受审计的显式渲染期间临时显示默认轮廓。
  不修改goal_pose，不把默认绿色轮廓当成构造任务的真实goal。

## 新证据链

1. 新source producer重建原moving-history构造及见证执行，真实取得两种显示下的
   RGB history、未来checkpoint和goalRGB。每个显示干预off→on→off，
   full state/body/goal/renderer前后不变，两off逐位相同且与旧封存对应。
2. 新plan producer执行旧四分支，固定plan0/5/10取得同body配对渲染，
   实际raw命令和全部原off观测/动态与旧执行逐位对齐；合法native早停保留，
   不强行补齐为full10，也不把缺测分支删出分母。
3. 新CUDA consumer独立使用同condition history及目标；保留完整请求/输出，
   对比固定H2想象预测、真实prefix条件预测与真实terminal。
   具体census与资源上限须在MAIN审读后由新prepare冻结，不能按结果增减。

部署盲输入只含RGB3、真实raw history和goalRGB。物理state、构造witness及未来真实图像
仅在离线评价域，不进入GN/D2P选择或预测输入。对factual witness的预测必须与
部署候选评估显式分开，不能将已知witness当新规划器的成功产物。

## 分析与结论门

固定四分支，不按胜负筛选。分别报告各condition的预测MSE、持久性基线、
相同目标的模型Q/真实Q、两arm排序、原生任务结果；同时报告全部反例。
两显示条件的latent坐标可能不同，绝对MSE改善须与持久性相对误差一起检查，
不能把跨condition编码距离当成相同物理误差单位。

只有MAIN完成完整源码审读、绑定/预算核验、真实主机CUDA及保存数组独立复算后，
才将该轮列为已完成校准。mock通过、已派发、已启动均不是实际证据。
即便排序改善，此轮仍不能打开P1：独立根、非贪心动作、真实能力改变与收益、
强基线和等预算等原条件必须另行满足。

## 分工与产物

MAIN负责代码审计、准入、真实执行和独立复算；不设审计子智能体。
Arendt负责新source三文件，Jason负责新plan physical三文件，Poincare负责新CUDA三文件。
不改旧sealed source/results，不训练、不拟合readout/Q、不降级CPU神经计算。
所有新实验使用独立schema、新prepare、新输出目录，原失败证据保留。

原moving sourcepack SHA：`c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。
原moving planner report SHA：`cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0`。
原moving physical report SHA：`a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf`。
这些身份引用不意味着新轮已经执行；新轮实际状态以新产物及MAIN核验为准。
