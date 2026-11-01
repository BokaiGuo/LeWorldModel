# Six-target deployment：完整controller authority与下一阶段边界

本补记不修改已冻结协议及来源补记：

- `agent_p2_six_target_deployment_protocol_20261029_v1.md` SHA256 `97d81f6969e0ac52e808893be09dfacf0180ed04ab0955fe38d202563a32744f`。
- `agent_p2_six_target_deployment_protocol_sources_20261029_v1.md` SHA256 `4c114bf5fa6bcbf0fcbcff7177bccc0379609db8fc11979747be4cf5d4e6a8dc`。

路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main/`。MAIN已完成exit0的完整consume，实际三份authority为：

| 角色 | 路径 | SHA256 |
|---|---|---|
| prepare | results/agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json | cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9 |
| report | results/agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json | b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934 |
| receipt | results/agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json | 592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103 |

worker已只读核receipt文件SHA及顶层ABI；未重跑完整consume，也不把字段查看当完整验证。真实receipt schema为 `agent_p2_six_target_controller_cuda_20261029_v1.receipt`，status为 `all6_new_target_controller_saved_verified`。其绑定字段实际为 `manifest_sha256`、`report_sha256`、`completion_sha256`；完整检查/资源字段为 `denominator`、`local_checks`、`accounting27`、`side_axes`、`saved_consume_authentication` 等。不得猜成旧paid `.consume` schema、补造flags或仅凭status放行。

下一阶段仅模型实际部署token付费链：新own-six prepare/report/receipt与六槽全分母，导入并认证上述完整实际authority及实际数组/source/model425，按原协议固定mean→CUDA分步mul/add得到raw→sub/div得到部署tokens→whole H3与segmented[1,2]。全部六槽重新付费，计划36predict/36samples、144AE tokens、108predictor context tokens/lowproj rows；没有搜索、动作重选、encoder、训练或physics。旧四槽结果不继承，新controller的normalized-mean replay不冒充部署token replay。domain超界或位级/source/layout失配保留BLOCKED，不clip、不退款、不选择性补跑。

新实现须采用最终真实ABI逐字段接线并独立消费own-six输出，不patch旧fixed4外壳或伪装旧schema。上游 `raw_inverse`、`scientific_admission` 等真实声明保持原样；本次controller receipt不认证部署token链完成，不签训练normalizer来源，也不授予物理许可。

物理阶段必须在新paid链实际完成并完整消费后，另由MAIN审批新六分支driver与运行；当前不得构造env或派发物理。两known roots、原目标来源时距与生成预算限制不变，不签P1/P2。

本轮仅新增protocol补记、saved SHA及顶层字段核对；未实现runner或生产prepare、未运行NN/physics/fullstrict，无子代。
