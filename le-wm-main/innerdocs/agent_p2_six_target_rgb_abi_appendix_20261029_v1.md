# 六槽 RGB adapter：来源与 ABI 附录（仅协议）

上位已批准协议：`innerdocs/agent_p2_six_target_physical_comparison_protocol_20261029_v1.md`，SHA `6e45721309e9a1fe1107918c988c170daf65ebe032163d0beace73d2bcb118f0`。本附录不修改上位协议/旧源码，只落实将来的RGB接线。native adapter与raw部署authority由Pasteur负责；这里没有其替代实现。须等待真实paid receipt与actual native full strict，MAIN再确认实际schema后才实现RGB driver。

## 已核来源（本轮只读源码，不消费生产）

路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`：

| 源码 | 当前SHA256 | 真实可复用接点 |
|---|---|---|
| experiments/spectral_reachability/agent_p2_six_target_paid_replay_20261029_v1.py | 84ebc96da0f7e351132bf697a90a7eaab2e2aa296646fe62c0013db65d9157d2 | `verify(path,h)`、`validate_saved(m,d,r,folder,host=True)`、`check_completion(r,done,rh)`、`validate_files(r,folder)`、`consume(manifest,mh,report,rh,output)` |
| experiments/spectral_reachability/agent_p2_three_program_rgb_20261029_v1.py | 02a3f8f88c3ea6b3a39dbd33975d3e6e9d44e973e8a92ab262f32cf4ab10a058 | `run`内原encoder加载/预处理/Trace、`validate`内visual_bridge、`bridge.load_pixels`、`bridge.rgb.load_array` |
| experiments/spectral_reachability/agent_p2_six_target_prediction_comparison_20261029_v1.py | 5e6c1df76cc07e623c1e3cae1b1eea04ac2de10eff83d62b1ac4d702ebd6ffb4 | `resolve_goals`的sealed原goal/history来源核验、`mse64`固定f64归约；不是实际token预测来源 |

six实际consume固定：`results/agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json` SHA `592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103`，对应prepare/report沿上位协议。paid/native实际manifest/report/receipt路径和SHA **本附录均未取得/未认证**；不能从源码存在推断生产成功，也不从猜测结果目录填future hash。

## A. Paid prediction 的真实读取链

将来CLI必须给paid manifest/report/full receipt实际路径及SHA；核source闭包后用原纯saved验证链重建整份receipt并强等外部文件，不只status。`consume`本身会排他写metadata：若调用它必须用新认证路径并计账；也可显式复用其相同verify/completion/files/validate_saved/bound_sources逻辑，重建它附加的manifest/report/completion/domain/controller_authorities/inclusive-wall全部字段。不得漏字段冒充完整认证，不patch globals，不调用其run_cuda。

真实report `rows[i]`具有slot/root/condition/arm、`inputs`、`trace`和`arrays` descriptors。读取 `row.trace` JSON，按descriptor从同paid目录加载array；每文件须验证inventory/root/filename/size/fileSHA/dtype/shape/dataSHA，不能把parsed dict当Path传给旧reader。完整性沿原validate_saved检查，包括input_seal、prediction_seal和所有425before/after、模型/源hash、whole inclusive wall。六槽位置固定，不通过target值猜slot。

嵌套trace是冻结 `agent_p2_delivered_mean_paid_replay_cuda_20261029_v1` 的低层算术schema，不是旧四槽receipt：

- `tr.inputs`映射到输入array key；原`z`为 `[1,3,192]`、`a`为 `[1,2,10]`。raw/normalizer/target等完整输入以实际trace与`payloads(domain)`核，不手填字段名单替代原validator。
- `tr.actual_tokens`指向actual CUDA分步sub/div生成的f32 `[1,1,3,10]`，绝不以原normalized mean替代。对应raw15身份由Pasteur整条domain/paid/native部署链认证。
- `tr.replays[0].kind='whole'`、`[1].kind='segmented'`，两者returned；`rollout.prediction`都是 `[1,1,3,192]`。cp5/10/15分别取 `[0,0,0/1/2,:]`，cp0取原z `[0,2,:]`。完整checkpoint/final causal histories以及两种重放全数组byte一致为硬门，不只终点cost一致。
- `paid.validate_trace`重建actual-token方程/模块IO/事件/历史/费用；外层six validate_saved核全六与barrier。只用各cp预测向量重算新诊断，不将原`target_MSE_fsum`终点标量冒充cp5/10分数。

既有six normalized预测仅作明确标注的旧版本旁栏，不能用于新物理prediction-vs-encoded主join。whole/seg相同也不是原native已执行证据。paid receipt本身`raw_deployment_authorized=false`，不能自改为true；工程运行授权另由MAIN与Pasteur合同提供。

## B. Native → canonical RGB：待实际schema锁定的接口

此处定义**所需语义**，不宣称Pasteur尚未交付的schema已有某字段/函数。实际native full strict交付后才将以下字段落实到真实JSON pointer并封存map；任何缺项停止实现/prepare并报告具体pointer，不造兼容receipt。

每个 `(slot,root,condition,arm,cp)` 必须唯一连接：原task ID/SHA/goal/source/root/reset/warm身份；paid raw15完整字节与执行requested/actual相同前缀；publicstep；native state7/snapshot/reward/termination；canonical off PNG文件和pixel SHA。cp0是warm结束的真实根，不要求新增Space.step或重复render；使用实际native封存的对应帧。cp5/10/15必须是plan实际返回步，不以root+warm总步号或各臂last_state替换。

原 `three_program_rgb.fixed_rows` 固定4plan/12图、旧model_plan.edges/selection_digest；其 `build/verify/process/validate/consume`也固定旧schema，**不能直接拿新six调用**。新driver只能复用其图像核/Trace/visual_bridge等已审安全接点，另写明确six roster/reader/join；不捏造edges来兼容旧函数。native reader/consumer准确签名及返回receipt envelope待实际代码审读后确定。

固定roster为 `index=4*slot+cp_index`（slot0..5，cp_index0..3对应0/5/10/15），始终24位置。native stop/失败/缺图在原位置保留status/reason/null，未到达不encode、不沿用最后图。若native source/state/action join本身失败，整体验证失败；不能把篡改解释成正常missing。full-six物理比较是否可用按上位完整门，不因部分有图就宣称全六成功。

## C. Encoder / projector：精确计算 ABI

可复用旧RGB安全路径：`load_compatible_pretrained(policy,device=cuda:0).eval().requires_grad_(False)`，policy/model authority须与six/paid已证parity的原P2模型精确join，不借未经配对的Hi encoder。原 `preflight_pusht_d2p_p0.image_transform()`、`_image_tensor(pixels,transform,device)`和`bridge.pixels.preprocess_pixels(pixels)`逐byte核；原PNG为RGB224，由真实native reader再次验证。禁止目标/pose/contact/native指标进入encoder输入，encoder只接pixels。

原 `model.encode({'pixels':x})['emb']` 输出 `[1,1,192]`，实际latent取 `[0,0,:]`。每张保存pixels、preprocessed、encoder完整 `[1,257,192]` hidden、projector输入/输出与最终latent；`physical.saved.overlay.visual_bridge(...,[1,257,192])`核完整连接，不仅保存192末值。原Trace捕获每图encoder/projector各一次，scope固定row/step/phase，不能从24图batch合并或去重；模型425/source/expected_model_hash映射须真实parity闭包认证后before/after强等。构造与hash传输不是新forward，但费用计入其墙时。

HOST CUDA only，TF32/autocast off、MATH SDPA、frozen eval、AD0、无CPU fallback、无新predict/AE/high/physics。实现时同时绑定loader/preprocess/kernel源码及真实模型配置/权重closure，不仅上表三段源码。此为原路径复用说明，不等本附录已执行模型或认可未来接口无误。

## D. 封存及三层报告

dispatch前封存全部24 roster和可用PNG来源、任务/goal/target/history/paid prediction数组引用与SHA；先保存输入再encoder，全部实际编码与模块IO durable seal后才算诊断，不按得分选择截图。模型输出不回写原mean/目标/成本。

每位置同时报告：

1. actual-token模型到本槽tracking target/原task goal的两种MSE；
2. 实际编码到相同两个目标的MSE，及model-vs-actual编码MSE、signed model_task_Q−actual_task_Q；
3. 同时刻原native7/reward/success/done、agent/block/jointXY、agent velocity、raw/native/wrapped angle，单位各自保留。

goal192沿sealed arbitrary prepare→原parity goal图output源核验，不重新goal encode；同根三槽原taskgoal相同。target是本槽FIRST/postcondition，不偷换成原goal。归约采用已封`mse64`：f32→f64 subtract/square→math.fsum/192；native独立saved算术沿上位协议。每栏保留输入array/index/pointer/SHA；追踪不同target的loss不作跨槽排名。cp0重复图仍各encode一次并核原根latent exact；不exact保留失败，不用容差校准。

全24图可用时encoder/projector各24API/24samples，缺图按实际记录，失败attempt不退款；原six72/324、paid六槽原计划36/36及其实际debits、native计数和RGB新增费用分开导入，不用计划代实测。RGB wall900s、256MiB含32MiBreserve、floor200MiB、CUDA allocated3GiB/reserved4GiB沿已批准协议，不另扩cap。保存full arrays、journals/census、source/modelbefore/after、partial与cleanup错误，独立consumer完整重建外部strict及24位置，不省略bindings。

## 当前交付边界

本轮只新增此附录；做了定向源码/签名/SHA读取，无新测试、无生产ABI/fullstrict消费、无模型/NN/physics或子代。尚缺实际paid manifest/report/full receipt及native source/manifest/report/full receipt（包括确切frame pointers）；等待MAIN/Pasteur提供并审读，未虚填未来hash。下一可实施项是这些真实资产到位后的source-bound 24行roster fixture，再新stem RGB runner/tests/doc，不在本轮启动。

known2非fresh，prior物理时距仍unknown，生成预算不公平；三层断点不能唯一识别target质量或controller误差，不产latent/物理winner。P1前置未过，P2机制/scientific gate=false。MAIN已批准分阶段/RAM准备，不等批准生产。
