# P2 intersection：实际帧 CUDA 编码与同时间排序

2026-10-29 MAIN 审读 `agent_p2_intersection_rgb_20261029_v1.py`、roster、对应测试及说明，联合 13 项测试通过。以物理 v2 MAIN receipt、原计划及同一冻结 checkpoint 为输入，真实 prepare SHA256 `fb90f7f9637ffd17f8e9ff0416b1fd9bc21aa3a29c805c6dd332e21728ad9851`，整份重建 verify 通过。旧物理 strict receipt 被继承，本次没有重跑其 full strict。

主机 CUDA 冻结编码完成，report SHA256 `9aa521a0f6befee544e6109b756a0193ee88a8624d172e839ecffae9fe748efa`，saved consume SHA256 `3797fb394ad1a693272bcf21d70ea59c9c8c344ad4aab35be3462ee298ad3087`，整份 verify-consume 通过。固定 24 位置中 23 帧实际编码；root1 joint 第15步由于原生第14步成功早停而缺失，未补帧。23 encoder、23 projector，0 新 predictor、AD 或物理调用；峰值 CUDA allocated 110062592、reserved 115343360 bytes，完整报告边界墙时 11.5544 秒。6 个 cp0 是原观测上下文，不当作预测。

MAIN 对 consume 的已观测 cp5/10/15，按同根、同检查点、不同臂枚举无序对，得到16对：模型目标 Q 与实际编码目标 Q 严格排序反向3对，分别为 root0/15 的 joint–composition 和 joint–GRASP，以及 root1/15 的 GRASP–composition；其余13对同向。这里只是读取本次 sealed consume 的描述性计算；两个根和多个共享轨迹并非独立样本。root1 joint 的 cp15 缺失，不能与其他第15步补齐比较，也不能用 cp0 作为预测准确率。

同一固定16对再分别对照同时间 native 返回：实际编码 Q 与 native7 距离排序反向11对，与物体位置误差反向5对，与物体环绕角误差反向8对；模型→实际编码和实际编码→native7 两层同时反向2对。以上按差值乘积严格小于零计数，不把平局当反向；native7 包含目标速度，而 success 判据不同。模型误差和评价语义错位必须分开核查，不能凭这一组计数证明因果或单一代码故障。

新增独立数组数值分解 `le-wm-main/results/agent_p2_intersection_rgb_pair_diagnostic_20261029_v1_MAIN01.json`，SHA `65da16658a05683599bd1f876b6af26ce100ac44c56652510050e9662294414d`；MAIN审读后10项测试通过、真实保存/整份重算verify通过。固定18对中16对RGB及native可比较，另两对保留早停缺失；逐行独立重读原float32 latent及raw动作前缀，重算模型Q、实际Q、预测误差和交叉项、native七维轴与成功。预测误差恒等式最大浮点残差 `1.1102230246251565e-16`，native距离与原reward最大重算残差 `1.4210854715202004e-14`。这排查了本批数值重建及对齐，不证明模型或编码器整体无bug；分析继承旧strict receipt，不重跑原物理。

这次工作提供实际轨迹→冻结编码→原模型同动作预测的接线和排序误差证据；编码器距离不是物理任务成功，三臂计算消耗不同，不能证明公平比较、P1/P2机制或泛化。物理与模型的前瞻性外部时间认证仍不存在；不改变 gate。
# 同时间与资源配对补充

`p2_saved_fair_pair_20261029_MAIN02.json` 的 SHA 为 `b45348e4dd8e74ee735d6c165148539b346fd21dd3b9aeb8e552ad524ec0ae1b`；主线程独立回归 7 项并重建验证通过。18 个固定同时间配对有 16 个可比较，root1 joint 第 14 步停止，对应两个第 15 步配对保留缺失，不补值。历史无组合 SRA 是 10 动作步且原生起点和目标不同，不可与此 H15 六槽严格配对。两根的 composition/joint/GRASP 每根 predictor samples 分别为 1062/1902/1902、AD primal 为 129/255/1896；FLOPs 未知，不能称等资源或公布公平胜者。该分析继承先前保存回执，并未重跑上游 full strict，也不是新物理结果。
# 后续：同根 H15 无组合谱候选的模型运行

原六槽 `joint` 已是限定协议下无组合迭代对照；新增的单次 H15 谱候选是另一种变体，不把历史 H2 SRA 混入本组。主线程独立复跑受控 HOST 测试 4 项后，以冻结模型在主机 CUDA 执行两根：两根均 sealed，报告 SHA `188fe5552b269db22739a77042a318d1f60d1c1c69cddcf4c5aead20501fe6ce`，总壁钟 70.57 秒，CUDA 峰值 allocated 142634496、reserved 146800640 字节。原 saved consumer 对模型、精度、来源、两根 native 事件及固定候选和重放重建通过，状态 `saved_native_checked_not_admitted`；每根记录 132 predictor samples 和 192 VJP directions，数组 SVD/solve 未完整计量，不能称 FLOPs 或等预算。消费结果依赖保存轨迹及来源信任，不能证明不存在未记录调用或认证硬件。

本轮没有物理执行；双根为已知根，尚缺同根物理 adapter、真实成功、无组合与组合公平资源比较及独立根。不能由已生成模型计划声称 P2 优势或阶段通过。

后续同根物理 adapter 测试 6 项通过，第一次启动因示例给出相对路径而在创建目录前被绝对路径合同拒绝，未执行物理；使用绝对路径重试后，两根实际物理执行和独立 saved consumer 完成，报告 SHA `aecf685083df2160df0d11929b2bd67f3366f2dbcc59bb111d879c1a78c9d185`。记录 50 次 public step、508 次物理步和 162 次显式渲染，均完成；两根状态为 `two_root_native_recorded_not_admitted`。独占 consumer 不等于旧六槽完整 strict verifier，不证明不存在未记录调用；同根、同目标、共同时间的描述性配对仍需完成，早停缺失不能补值，等预算和独立根仍缺。

## 状态更正：描述性同时间配对已存在

上述“描述性配对仍需完成”已过时。已保存的
`le-wm-main/results/p2_h15_physical_pairing_20261029_v1_MAIN01.json`
SHA256 `51ab421e564e099482cf0f6a92409e43ce0984f8c87903e28829203390ed6403`，
包含两根的 H15 谱候选分别对 joint 和 composition 的逐步配对。
root1 的 joint 在第 14 步原生停止，其第 15 步保留缺失。
该报告自标识为已知根上的事后描述，`equal_compute=false`、
`independent_generalization=false`、`scientific_admission=false`。
这次仅复核文件哈希及其配对状态字段，未重跑上游物理或重新认证所有来源。
它解决的是已有记录中“配对是否存在”的状态问题，不解决公平预算、
独立泛化或 P2 机制优势。

