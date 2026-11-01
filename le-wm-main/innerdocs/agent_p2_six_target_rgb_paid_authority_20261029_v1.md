# Six RGB：actual deployment-token 实际 authority 补充

仅新协议补充，不修改任何冻结附录、协议、源码或结果。承接 `agent_p2_six_target_rgb_abi_appendix_20261029_v1.md` SHA `9feeea8a8940ecf18d2402fa5cb992b0fc77e00917c78182dead7430b8e6e874`；其中paid结果当时未知是历史状态，本文件补入已取得的实际authority。native结果仍未在本轮认证，不推断其已存在。

## 已取得的实际绑定

以下路径相对 `/home/gbk/code/LeWorldModel/le-wm-main/`；本轮仅sha256sum与jq小metadata/六trace ABI核对，无生产fullstrict重建。

| role | 路径 | SHA256 |
|---|---|---|
| paid prepare | results/agent_p2_six_target_paid_replay_20261029_v1_prepare_MAIN01.json | 9f367dbfdf91dc69ff4ca79562868a8bf16074bff55b91699f5894356a1dc5df |
| paid report | results/agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01/report.json | c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd |
| paid full receipt | results/agent_p2_six_target_paid_replay_20261029_v1_consume_MAIN01.json | 6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db |
| paid completion | results/agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01/completion.json | bd2a281c09da05018013c5b4a5e25d0cdb719de126713f6f1bc343068124206b |

receipt实际schema=`agent_p2_six_target_paid_replay_20261029_v1.receipt`，status=`all6_actual_deployment_token_replays_verified_not_physically_released`，denominator={total:6,returned:6,blocked:0,missing:0}，predict_API/predictor_samples=36/36。其controller_authorities.receipt准确指向原six consume SHA `592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103`。这些是metadata观测及MAIN已完成严格消费的交接，不冒称worker独立fullstrict。

## 实际trace选择，不使用旧分数

实际6个 `slot0_trace.json` 至 `slot5_trace.json` 均声明backend=`host_cuda_actual_tokens`、completed=true、errors=[]；schema仍为已封低层paid trace。每个trace的输入/预测映射相同，但实现必须解析各自trace及report descriptor，不把本表硬编码当通用validator：

| trace pointer | 本次实际array key | 用途 |
|---|---|---|
| /inputs/z | array_2 | cp0观测history末帧 `[0,2,:]` |
| /inputs/a | array_0 | 原完整between-action history |
| /inputs/target | array_1 | 本槽tracking target |
| /inputs/sealed_raw | array_6 | 待与native实际有序raw15精确join |
| /actual_tokens | array_19 | actual CUDA sub/div产生的部署tokens |
| /replays/0/rollout/prediction | array_41 | whole实际token预测；cp5/10/15取 `[0,0,0/1/2,:]` |
| /replays/1/rollout/prediction | array_71 | segmented[1,2]全数组/历史逐bit一致见证 |

本轮未加载预测NPY或计算分数；shape/dtype/value/完整checkpoint-history守卫仍须在MAIN后续RGB prepare的整份paid认证中执行。每个key必须先通过该slot report.rows[i].arrays[key] descriptor读取，不能跨slot复用同名key。数组file/data SHA、trace SHA、paid seal/source/model/425闭包全部来自已绑定report并重建认证。

MAIN已指出normalized job与actual deployment-token预测的target_MSE存在小差异。本协议不测量或解释其幅度，也不把旧分数近似当新值：RGB主诊断只用上述actual-token每cp预测向量，独立f64 identity MSE对本槽target和原taskgoal重算。不得复制normalized controller的target cost、旧six预测比较器的model分数，亦不得复制paid末帧target_MSE充cp5/10。旧normalized版本只可标注旁栏，不能替代actual-token预测误差。

## 余下前置与执行边界

仍等待Pasteur实际native source/prepare/report/full receipt及逐cp canonical off PNG/state/action映射。raw15有序字节、actualstep、同root/task/goal/history必须贯穿 paid→native→RGB，paid成功不等物理部署许可；不修改其raw_deployment_authorized=false或科学字段。六槽24位置与各阶段missing/费用原合同不变，不按小分差或未来物理outcome选择动作/图像/目标。

native完整实际资产到位后，先审真实schema和函数签名、落实source-bound roster pointers，再由MAIN决定新RGB driver实现/审读/host CUDA。当前只有本新文档；无测试运行、模型/NN/physics/fullstrict/子代，无旧文件编辑，无新生产授权或未来hash。known2非fresh、prior时距unknown、生成预算不公平、P1/P2科学门false继续保留。
