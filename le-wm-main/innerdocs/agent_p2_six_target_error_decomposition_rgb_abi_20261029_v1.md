# saved误差分解：实际RGB终态ABI补充（严格receipt仍pending）

仅新增本文，不修改冻结的分解协议或任何已交源码/tests/doc/结果。承接 `agent_p2_six_target_error_decomposition_protocol_20261029_v1.md` SHA `74f1d02c3282ebfaf7b30828156137dd9e1a4a207527d5694c0c7abe13ff59d9`。本次是实际文件metadata/SHA接线，不是新误差分析或认证结果。

## 1. 实际取得的文件

以下路径相对 `/home/gbk/code/LeWorldModel/le-wm-main/`，本轮实际sha256sum匹配：

| 角色 | 路径 | SHA256 |
|---|---|---|
| RGB prepare | results/agent_p2_six_target_rgb_20261029_v1_prepare_MAIN01.json | 14aace0fad5c941fce475a41d45bb1dfa1bad9388cba9f767bdd4eb42aa4f393 |
| RGB report | results/agent_p2_six_target_rgb_20261029_v1_cuda_MAIN01/report.json | 26e00d955ee2c106e4452b97ddca26a7731d721f7e00605f5750e3a38981460f |
| RGB completion | results/agent_p2_six_target_rgb_20261029_v1_cuda_MAIN01/completion.json | f54b1ba7fc7657db1b738f1fe8be5ca280d790a14d5fae0a1a5326d0ac5bcb23 |
| RGB native_seal | results/agent_p2_six_target_rgb_20261029_v1_cuda_MAIN01/native_seal.json | cf8309b1ee0c8b92af53090a001f30b32007c479d9736665c1e836fa9aeff37e |

RGB report.schema=`agent_p2_six_target_rgb_20261029_v1.report`，实际metadata为completed=true、errors=[]、24行；completion.complete=true并精确绑定此report与原boundary。MAIN正在完整saved consume，**实际RGB full receipt路径/SHA尚未收到、未读取、未认证**。不能仅凭这些terminal字段生成分解器的认证输出或声称cp0/全链消费已过。

资源metadata：encoder/projector各24attempted/24completed、encoder/projector samples各24、persisted_tensor_captures=144、missing_positions=0；新predict/AD/physics均0，all_cpu_transfer_calls/FLOPs为null。这里只如实引用report，不冒独立重建。event_count=144、artifact inventory315项供未来局部join核对，不能用这些计数代完整consumer。

## 2. 从真实ABI到固定分解输入

每行 `P.rows[i]` 为prepare原封存输入，`R.rows[i]` 为对应输出；位置固定 `i=4*slot+cp_index`，尾行实际index23/slot5/cp15。未来分解器须比较两者原封存字段，不能从receipt汇总重建另一套输入：

| 符号/层 | 实际字段 | 局部核验要求 |
|---|---|---|
| p | P.rows[i].predicted192 / predicted_sha256；prediction_origin、prediction_pointer | f32[192]原字节SHA；R相同字段；cp0 observed_history，其余paid_actual_deployment_tokens；只用paid版本 |
| z | R.rows[i].latent | descriptor.file/sha256/array_sha256/shape/dtype，读取对应gzip-NPY；固定f32[192]、EOF/数据SHA |
| t / g | P.rows[i].target192 / goal192 | 原f32向量；R同字节；每root三个slot的g一致，不拿t代g |
| 原goal来源 | P.goal_sources[root]及每行goal_source | sealed原goal图/向量pointer保持；不重新goal encode |
| native | P.rows[i].frame.state / snapshot、native_goal、response | 与R封存字段及实际native authority同slot/cp/动作前缀连接；cp0 response为null合法 |
| PNG | pixel_source / frame的文件名、file/pixel SHA | 真实canonical off；不要求本算术工具重解码全部PNG或重复完整物理消费 |
| 同动作/源身份 | raw15_sha256、full_raw_bits、raw_prefix_bits、selection_digest、task_id/task_sha256、blind_sha256、original_z/a_sha256 | P/R一致，并精确指向实际paid/native；跨策略动作不要求彼此相等 |
| 时间局限 | source_time、target_origin、cp_public_step | 直接保留；prior/arbitrary unknown不填15，program源cp5不冒controller终点等价 |

实际首行latent descriptor.file=`r0_latent.npy.gz`、shape=[192]、dtype=float32；各行必须独立解析自己的descriptor，不能按首行硬编码或跨slot复用。所有新读到的array文件须属于已绑定R.artifact_sha256，descriptor.sha256一致。原reader `bridge.rgb.load_array(descriptor,read_blob)`只解析保存数组，无torch/forward；未来可复用该纯reader，但不得导入其消费入口后意外调用verify/fullstrict。

prediction_pointer实际为 `{trace:{file,file_bytes,file_sha256,root},array,index}`，指向**paid目录**的trace/array而非RGB目录同名文件。例如首行array_2/index[0,2]为cp0；cp>0必须是actual-token whole prediction相应[0,0,j]。P/R内p及SHA是本地分解的直接向量来源；上游已消费的whole/seg一致证据由最终完整receipt承接，不能仅凭指针再签完整模型认证。

report中 `metrics` 仍是未使用的输入栏，不能把它当已经算出的native或Q。三层scalar来自未来实际full receipt.rows，但分解器应从上述原向量/state独立复算，再对照receipt scalar；不把producer scalar当自己的独立计算。

## 3. 祖先费用与receipt join位置

实际prepare保存：

- imported_controller_receipt为原six完整receipt，API72/samples324及27轴；不是仅三个authority refs。
- imported_paid_receipt.report_sha256=`c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd`；actual-token36/36费用独立保留。
- imported_native_receipt.physical_report_sha256=`ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b`；native费用独立保留。

待MAIN提供RGB最终full receipt后，按已封consume真实返回字段核report_sha256、prepare_sha256、completion_sha256、rows/pairs/denominator、resources、inherited_paid_receipt、original_controller_receipt、inherited_native_counts、bindings及科学false字段；完整文件SHA必须实际给出。只读取/核对局部join与算术，不再回调RGB.consume/verify或native/paid fullstrict；认证是从MAIN已完成证据导入，非worker独立认证。

## 4. 下一步与未执行项

receipt到位并经MAIN批准后才实施新stem分解器与RAM tests：原24行/24pair、残差能量+signed cross term、tracking/target-goal几何项、native独立平方分栏、固定cp0和相邻区间差分，null及roundoff residual按上位协议不变。不读未来结果来选region/根/指标或替换归约，不产winner或P2机制结论。

本轮仅定向读取上述JSON metadata并核4文件SHA；没有加载latent/weights、计算任何新差分数值、调用fullstrict/NN/physics/子代或运行测试。未生成分解报告或认证文件。已完成CUDA是MAIN的实际工作，本附录不是worker运行证据。冻结文件保持不变。
