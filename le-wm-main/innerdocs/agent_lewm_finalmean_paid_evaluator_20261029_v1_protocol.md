# native CEM 返回均值：独立后付费评估事前协议

固定于实现前。只评两个原known/off roots 0、1的原CEM实际返回final mean，不重新选候选，
不训练、不改normalizer、模型、动作域、旧source或科学门。原slice永久保持
`final_mean_evaluated=false`；本新独立report只说明新job付费后的评估。

原实际report：`results/agent_lewm_native_cem_slice_20261029_v1_cuda_MAIN01/report.json`，
SHA `ab3313d24b6efb3539bb1b1a70417f7aa41df1602157465d1da5cd1d26e8b965`；
原prepare SHA `f8f773572326ddd52c407d14b850584228f29c5c3a14d0f44262129a676d276c`。
必须额外提供MAIN完整consume实际receipt及外部SHA；不造未来值，缺失拒绝。
prepare绑定report/prepare/receipt与旧source/model/input闭包，并逐位提取
`actions == mean == last_callback.mean`；elite平均cost不用于返回动作cost。
不重复旧fullstrict，旧完整持久receipt是明确的信任起点；新saved consumer完整验证新job。

## 两种布局固定且不因结果改变

每root先封存原returned normalized `(1,2,10)` 和cached actual0/5/10、warm2、真实RGB3、goal。
不归一化第二遍，不加tanh/clamp，不转换成物理动作，也不再次采样。

whole：官方 `LeWM.rollout`，cached emb，真实原RGB3，仅通过shape读取历史H=3；
传warm2+mean2四tokens，AE一次四tokens，predict两次，各上下文3，proj两次。
保完整predicted_emb的五帧，不只final。

segmented：不用虚构pred15对应RGB。第一次AE处理warm0/warm1/mean0三tokens，
`model.predict(actual0/5/10, action_emb3)`，保存完整full3并取末帧pred15。
第二次AE处理warm1/mean0/mean1三tokens，以actual5/10/pred15调用第二次predict，
保完整full3并取末帧pred20。第二次输入必须因果使用本分支pred15，不取whole输出。
不能把AE四token/三token布局差别藏掉或偷偷改成同布局求bitexact。

两个root全部whole+segmented预测先seal，之后独立criterion用原cached goal计算各terminal
float32 sum192；所有计分不进入预测接口。saved consumer另算float64 sum192、mean192。
criterion API共4次（每root两个结果），没有额外预测。

## 固定预算、比较规则

全job known2：8 predictor API/samples、8 pred_proj API、6 AE API/samples、20 AE tokens；
predictor context tokens24，pred_proj rows24。whole rollout API2；segmented model.predict API4；
不要把wrapper API与内部predictor再相加。encoder/projector/AD/physics/training全0。
旧CEM全部成本与原8encoder/8projector来源单列继承，不能回填equal144或宣称等FLOPs。

对两个预测帧15/20及末帧，固定全部元素比较，报告bitwise、maxabs、最大float32 ULP与非同位数；
有符号零单列bitwise差异，ULP采用单调float32位序映射，有限值才评分。
无事后容差、无allclose parity、无按根/差异筛选。只在实际bitexact时报告bitexact，
否则`parity_upgraded=false`且如实保留差异，不能修改旧receipt状态。
比对的是新whole与新segmented，不声称与旧batch300候选算子逐位相同。

整个output512MiB（含64MiB final reserve），disk floor200MiB；wall900s含verify/load，
cooperative守卫不冒充可抢占CUDA kernel。CUDA allocated3GiB/reserved4GiB。
原生输入输出全float32保存，durable intent/return，全部失败保留两root×两layout分母。
MATH是requested context，不是实际kernel证书；TF32/autocast/inference守卫另记。
权重/config文件SHA和加载后前后tensor hash不冒充同buffer loaded-bytes证书。

worker只写新source/tests/innerdoc与fixtures，不运行NN/physics/生产prepare/fullstrict。
MAIN审读和生产；工程成功不等于任务成功、论文复现、P1准入或公平baseline排名。
