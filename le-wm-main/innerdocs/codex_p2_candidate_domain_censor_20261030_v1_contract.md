# 后验探索合同：已保存 domain-censored 物理槽

这是在已查看失败记录后制定、在新保存结果分类和完整复核之前固定的探索合同；不是原实验预注册，也不改变原报告和工程失败状态。只处理固定四 base、每 base 两槽原报告，不重跑神经或物理。

仅当全部条件成立时，在新分析视图将原工程失败标为 `domain_censored`：原首错严格为 `Rejected: fresh RGB/canvas guard`；最后公共动作确实返回、requested/actual float32 位串与固定计划一致；最后同时间原 off RGB 和 native state 均已保存并 hash 绑定；图像是 RGB224 且 agent 固定颜色像素非零；state 前四维中有一维严格在 [0,512] 外；原 guard 源码确实由「颜色像素为零或前四维越界」拒绝；该帧是最后一个 fresh observation，之后没有任何公共物理 step intent；原生 terminated/truncated 均 false。任一缺失/不符保持 engineering failure 并拒绝此合同，不归类域外。

不改写原报告。完整 saved 校验仅在新独立消费者中有条件地允许已保存的这一张越界 incident off 帧存在于 journal 但不在 producer 聚合 trajectory；仍核所有 event hash、逐操作账、原动作位、任务与 RGB 绑定、prefix 守卫、来源、全部 PNG 字节/像素、其他帧原 guard、结果与完整性。该单帧不得通过原 guard 的判定仍保留，不能冒称旧 strict 通过。原消费者内部 native 标签、检查点、早停、资源等所有其他检查不放宽。以新视图明确同时保存原 status 与新分类。

删失时刻只表示已返回但不满足视觉域的步；删失步不作为完整 checkpoint、成功结果或有效 RGB 上下文；其后所有指标为 null，不补步。其余完整槽原样验收；统计按两已知根和四 base 分母报告，不把重复 base 当独立根。无额外 GPU、神经、物理、模型选择，资源和科学门不变。
