# 固定164探针的动作统计替换诊断（MAIN，运行前协议）

## 研究问题，不是部署修复

现有P1排因仍缺训练run→数据→normalizer身份链。公开std与本地官方H5归档
全action统计约差3.3%；来源差异不是权重训练统计的证书，也不能仅凭三处常数
一致排除此因素。本实验只问：这个**事先已知、来自现有文件的数值差异**是否
足以改变固定探针的预测—真实编码响应失配。不拟合/调参，不运行物理选择，
不根据结果挑统计；正负结果都不能直接认证官方训练血统。

## 固定来源与唯一变量

- 使用 `agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json` 全164 jobs，
  SHA `a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5`。
  两个已知根×off/on，baseline各1、两h(.01/.001)×10坐标×±各40，不选根/动作。
- public：原task中的mean/std，以原float32字节为准。
- archive_sample_stats：固定 `pusht_d2p_archive_action_statistics_v1_20261026.json`，
  SHA `c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda` 的
  statistics.mean/sample_std，一次cast float32。它是action-field-only统计，
  不是checkpoint-bound训练export。不另跑population std或其它统计子集。
- 每arm必须从job.observed_raw10完整重建两个incoming token，再从原suffix5
  重建outgoing token，统一float32 `(raw-mean)/std`。不只换未来token，不在旧
  token上再归一化，不加/100、不clip、不再tanh，不重编码观测或goal。
- 同一原始RGB-derived latent3、goal192、原Q=I、模型/renderer/全部raw动作不变。
  公开arm应精确重现原完整模型输出，若失败保留工程失败，不放宽一致性。

## 实际执行与验收前置

源/测试/文档与本协议先冻结，MAIN完整审读和非神经回归后才能主机CUDA。
worker不prepare生产、不加载神经模型。328预测/AE/predictor/pred_proj各328，
0 encoder/AD/physics/training；各项嵌套计量不合为统一FLOPs。固定float32、
TF32/autocast关闭、SDP MATH，冻结全部参数/buffer，前后值hash精确相同。
128MiB输出、200MiB磁盘floor、1800s边界wall、CUDA allocated3GiB/reserved4GiB。
独占新目录、异常partial/费用不删除、不退款，不自动重试或扩大面板。

consumer必须核全部328请求的输入/token/输出身份、全部80对/arm、两个h分别
保留，以及public原报告全context一致。真实终态仅在run之后做诊断评价，固定
完整RGB strict receipt SHA
`d6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2`；
其state/outcome不进入forward或动作选择。实际RGB零响应始终单列，不当同向。
报告每arm的scalar方向、192维响应、终态误差及基线误差，不挑更好h/arm。

## 结论边界

即使某arm降低误差或反转，也只说明输入合同对固定cohort有影响，不证明它是
历史P1唯一根因，更不授权其进入blind planner。若影响小或更差，只排除本次
两个明确数值方案的充分修复解释，不排除未知真实训练normalizer/数据域问题。
P1/freshness/training_provenance/deployment/scientific_admission均保持false。
本协议是后验已知根上的新校准诊断，不是前瞻独立确认；结果发布须与此绑定。
