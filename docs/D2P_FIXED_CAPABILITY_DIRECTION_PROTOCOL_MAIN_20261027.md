# 下一项P1：固定真实探针的方向预测合同（未执行）

本文件在新神经实验前固定比较范围；不是已完成CUDA或科学证据。由MAIN实现、审计与运行，不与四支线任务重叠。

## 问题与固定分母

已有真实探针表明，D2P前缀改变了后续响应，但root0固定探针的object任务投影非正，root1有明显正值。下一项区分：冻结模型是否预测到真实RGB响应排序，以及真实teacher前缀与open-loop预测前缀的差别。不将模型latent映射成未观测native XY，不拟合readout、Q、统计或动作规则。

固定使用既有2 known roots×off/on×GN/D2P的8锚点，每锚点全部zero/±x/±y五探针，共40个分支。不能按响应符号删分支、选root、改幅度或补采有利根。不是fresh确认。

输入physical报告SHA `e075d3b75c7b5a16c47d8c76c846eaa2ac4d8ebd202abd18037a7242792b076a`；analysis SHA `19661bb010662dcfc7c9397723ecb085080fe18a7de069bdaac1f7c59e0e7557`；独立audit SHA `e4f668ce10f4389019f04d4fa507a6efdfa66158fc7a1609feeb74d23420d43d`。所有路径、PNG、action、root/task/goal/selection及源码SHA须进入新的prepare闭包。

## 时间与表示接口

- 真实教师prefix上下文：原warmup step5、step10、实际prefix step5三帧的原生192维编码。两条between-state action tokens依次为warmup最后5条动作和对应prefix5；outgoing token为该固定probe5。warmup不是零动作。
- open-loop上下文：同一旧编译器保存的predicted-prefix完整context，其末帧是模型自身prefix预测；过去两帧和动作历史须与教师路径逐位核验。不得用其他arm/condition的上下文替代。
- 教师与open-loop各40次H1预测，期望共80 predict/API/AE/predictor/pred_proj sample；完整模块实际计量，不能以预计预算充当实账。无AD/SVD/动作优化。
- 40条真实probe step5 PNG按对应condition选off/on/off索引0/1，不任意去重；新编码全部40条，保存原图、预处理、完整257token hidden、CLS、projector桥。原prefix/warmup/goal编码若复用必须核真实来源与完整模块收据，不称其历史成本为零；若无法严格复用则先修订并重新封存预算，不能临时增加调用。
- 真目标来自原任务goal_rgb，不是default outline goal_pose；不编辑图像、目标或渲染。

## 先封存，后分析

全部预测和真实编码在读取评分前封存。统一原生mean192/Q=原实验固定Q，不拟合新度量。每锚点保留五探针的预测/真实编码goal-cost和zero-relative差值、所有固定非zero-vs-zero有符号比较及完整pairwise排序；并列原任务objectXY方向投影与angle/velocity/native成功，不把不同轴合成新成功判据。零与反转完整保留，不事后选择更好teacher/open-loop路径。

80次预测或40真实编码缺失时保留原40分母、明确失败原因，不造padding/重试或开放P1。主机CUDA、冻结原checkpoint、禁TF32、无训练/CPU神经fallback；具体时间/存储/逐模块上限与源码必须由新prepare在执行前绑定。此次不再执行物理、不释放新动作、不修改旧结果。独立复算和真实执行验收仍由MAIN完成。
