# 规划内 RGB 编码缓存：真实反馈回归

在此前四个已曝光 TwoRoom 根、三个策略上，缓存控制器独立执行真实环境反馈；每次重新观测后新建缓存会话，按自己的选中动作实际推进。12 条 episode 均成功，与未缓存版本相同。

对全部 **57 次决策**和 **264 个实际环境步**逐项比较，候选分数最大绝对差为 **0**，保留索引、所选动作、派发步数、终止状态及物理距离均一致，最大距离差为 **0**。每次规划的图像编码器调用均为 2 次，57 次共 114 次；每次仍实际评分 680 条候选。模型和 normalizer 指纹一致，模型运行前后未改变。

这是同硬件、同模型、同任务、同随机种子的工程回归，不是独立确认。此前同观测交替顺序的配对测量显示本机规划段耗时合计下降约 26.1%；本次单独闭环运行的耗时不用于补充统计显著性或宣称整系统优势。该缓存减少重复编码，不证明行动后果描述额外改善了搜索；PC0–PC5 状态不变。

[预定协议](/home/gbk/code/LeWorldModel/docs/PC_RGB_OBSERVATION_CACHE_ACTUAL_FEEDBACK_PROTOCOL_20261107_CODEX.md) · [缓存控制器轨迹](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_rgb_cache_actual_feedback_20261107_codex.json) · [逐项对照审计](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_rgb_cache_actual_feedback_audit_20261107_codex.json) · [配对计时结果](/home/gbk/code/LeWorldModel/docs/PC_RGB_OBSERVATION_CACHE_RESULT_20261107_CODEX.md)。
