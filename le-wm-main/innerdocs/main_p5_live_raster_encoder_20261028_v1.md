# 新 live-raster 十图 CUDA 编码

MAIN 独立新版本，不修改旧 renderer/encoder 或其失败结果。仅接受新 live renderer 的完整 strict_live_raster_complete 回执，外部 prepare/report/receipt/source SHA 全绑定，并重做完整 saved consume。保留65帧 source_comparisons；不把新鲜快照冒称与历史 full_public 相等。

输入固定五状态×visible/white，state-major；before/after 完整新鲜快照 digest 与 public digest 精确核验。仅10干预图进入冻结官方 encoder；baseline/恢复图不编码，无掩码、挑词、重试、拟合或 CPU 神经回退。原 sealed coverage 只并栏，不是新渲染 coverage，更不参与选择。

CUDA runner、meter、store、原始 full256×384 patch 及16交叉分数、全部saved消费核函数与封存旧 kernel AST一致；无全局替换、无 physics overlay 导入。新增成本严格10次batch1 encoder，predict/proprio/AD/physics均0；上游实际605物理步独立记账。模型与源码前后验证，全部10 token先fsync再计算分数；新输出32MiB cap/200MiB floor/600s cap。失败不退费、不减少分母。

原P5 v5模型/transform/source closure保持；不是规划score替换、C2机制、任务收益或独立确认。新renderer尚未主机执行通过时，prepare必须拒绝，不预填结果SHA。

入口与旧七显式authority参数相同，模块名 main_p5_live_raster_encoder_20261028_v1。先prepare/verify，再仅由MAIN在主机执行cuda --confirm-host-cuda，最后consume严格复算。RAM测试核新receipt拒绝旧status/历史等价伪签、固定10图顺序与原kernel AST，不作为模型实验证据。
