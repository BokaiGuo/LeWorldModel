# D2P 新共同库与子节点调度执行（MAIN，2026-10-28）

本页续接 v8 的实际终态；不替换旧封存结果。原始完整范围仍为 C1–C3、三组核心实验、强基线与独立确认，当前未闭环。神经工作只在主机 CUDA 执行，不训练、不拟合额外模块、不调科学门槛。

## P1：真实 teacher 锚点的共同动作库已实际完成

MAIN完整审读593行新源码与文档；冻结source/test/doc分别为 `ca80a146b602406a1ef960fd2fa064afabc9a2bfd02d5396983f328d7f443b4a`、`7cb732cef39ae7d77d66ae43fc86febfb7dba35eeaf260b631bd2d6a7cc85331`、`8282c75b9026d10ce676b03dac0bccce2e1710b56c2532463e125c8e1d69fe8c`。MAIN 45项RAM回归通过（2.18秒）；不是科学证据。

实际prepare `le-wm-main/results/agent_p1_teacher_cross_library_20261028_v1_prepare_MAIN01.json`，9,788,985bytes，SHA `5140f9c6baeadc1c9889e65797ebdb4bec9b68db530f06467556c5fb743058e1`。四原任务层各取实际zero5/selected-prefix5后15步的teacher图像，固定8 anchors；原五探针的终点与响应不进入compiler。

实际HOST CUDA完成14.74584秒：8encoder/8projector、96predict/action_encoder/pred_proj、8J、1536数学VJP、8batched grad API及8bounded-ridge/SVD helper，errors=[]，模型与源码未变。先封存全部8个suffix，再在两种上下文分别评价及独立重放共同库 `[zero,uZ,-uZ,uS,-uS]`，未挑赢家、未去重退款。

报告 `le-wm-main/results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01/report.json`，SHA `d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738`。MAIN完整strict保存回执 `le-wm-main/results/consume_teacher_cross_MAIN_20261028_v1_MAIN01.json`，SHA `d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213`；核304原生module attempt/return、原精度数组、96完整forward、AD收据、8次saved独立SVD/solve、原归一化history与全部40 literal raw10 plans。

这不是P1通过：当前S前缀仍stage-greedy，只有两known roots；共同库的模型排序与降低Q不代表真实能力改善。40分支物理adapter已独占派发，必须真实执行全分母后才能评价；不能拿旧探针代替新库结果。某些anchor的一次线性solve实际模型成本仍不及zero，需保留，不强迫为“准确率非零”改数据。

## 强基线：同身份五臂完整分栏已实际完成

MAIN完整审读271行新saved comparator和文档，31项回归通过（.33秒）。实际run保存 `le-wm-main/results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json`，450,199bytes、739bindings，SHA `eb661bfd89b75b4865b37ba517de4412f015472928a9ed3c2b76ec15367ed128`。实际consume重建原始闭包、两identity partitions各五臂通过，0新神经/物理；全部10个native失败保留。

两known roots上beam、GN、D2P、matched CEM和nominal-MPPI使用相同root/goal/history/normalizer/Q身份。beam/CEM/nominal-MPPI每根各144forward，GN78/D2P66且有不同AD/J/SVD开销，不称等总算力；数值验证域及搜索support也分栏。D2P旧球半径校验 `cap+1e-12` 与其他producer严格cap的区别保留，实际约1.11e-15越界不抹去，不新增容差或改原动作。

同Q下降不等于native任务成功；例如root1 D2P实际Q约.19895，native7终距约508.413，而beam实际Q约.70170、native7终距约135.082。这是指标不同步的具体known-root描述，不是跨方法正式排名，更不是普遍不可能性证明。作者报告值仍与本地实验分开，不以不同协议数值替代本地强对照。

## P3：新child信号实际执行及已确认的工程失败

MAIN完整审读499行child native signal源码与文档，新45项回归通过（10.77秒）；冻结三文件SHA为 `7c6d741a8973fa42ac794ca85512167b81292c35ed679b6e291c5a37ea83cee7`、`da50b5265ba2b56a09d7da1afe1cc3c491af1cd9297e9853f29af0eb4e023b2f`、`fa2d58b3ba42e765f44ad16f78697daaacef1bf574c39c2a0b59abb7310a7a58`。

新prepare提交读取此前实际24child/48方法合同，不借原root J；事前independent口径固定46 acquisitions、368predict、92J、17664 scalar VJP、46schedule，余1child/2方法not_reached。生成真实信号之前不预判chi/配额拒绝分布。新source严格绑定原weights文件SHA与tensor hash不同owner，原动作history不清零。

此接口仍不生成候选动作、不进行物理闭环。后续child compiler与whole/segmented/selected-prefix replay独占实施中，原4/8候选、H1/H3、search cap24不调；超额整节点拒绝，不能为通过而缩候选。

上述prepare已实际完成，SHA `852bcdfdecbbdc81c41e7a161fefdad826ae5f6635db25e742e7ee3811ba282d`。HOST CUDA运行200.73784秒后以 `ValueError: output engineering cap` 终止；报告 `le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_cuda_MAIN01/report.json`，SHA `c0279c6a48d86bb46ed90b1d098bd436c76a9e85daa00893d34a119afa574641`。19 child已返回、4 blocked、1 not_reached，实际304predict、76J、14592 scalar VJP、76 array SVD及38schedule；模型前后相同，source verified，但cleanup/full-success未完成，strict不得接受partial。

根因是512MiB总上限预留256MiB final reserve后，重复完整数组/队列/账本JSON使普通journal超过256MiB。属于**代码层存储预算实现失败，不是模型或调度科学负结果**。v1及失败全部封存；新v2仅做无损存储改造，不提高原cap、不降低精度、不缩科学分母、不放宽准入。

## P5：原生接触源已真实完成、严格验收与本地盲输入分离

固定两独立环境、各完整frame0..20，在任一suffix前核全部11prefix帧的五类文件字节一致。真实物理运行1.33126秒，96次原生agent-block碰撞回调、40control、402 Space.step、42 render/get_obs，global_error=null；push改变block位姿，zero保持原位。

实际报告 `le-wm-main/results/agent_p5_contact_source_native_20261028_v1_physics_MAIN01/report.json` SHA `ecd6afea8bcc9ee7f6fe32234c91a4fb4be62ecdd6e6bbc78351ed8e20d8a4ba`；strict `le-wm-main/results/agent_p5_contact_source_native_20261028_v1_consume_MAIN01.json` SHA `d13719d2a7914d7ca28a25e56f16eaf2e2fcdbd868ee388d3f162322bc7e6dc0`，状态 `all2_saved_source_traces_verified`，四项预声明source witness均true。

MAIN重验strict后实际生成独立七NPY public输入目录，参考suffix、pose、contact与来源回执留在私域；public payload `le-wm-main/results/agent_p5_contact_source_native_20261028_v1_public_MAIN01_payload.json` SHA `fdc063a6475577071fb74cf4ccd0853c4b2e4e5c061693d73efbc437686ffec3`。这里的publish只是本地数据分离，不是GitHub发布。下一步为冻结JEPA的原生context/action-response验证；当前仅构造OOD校准源，不fresh、不构成P5/C2科学准入，task_success=null。

## 其他并行工作与准入边界

- P2：全八臂actual-midpoint固定suffix条件预测已实际完成，见下；不挑根/动作、不重编码终态。
- P4：逐步actual动作与goal-pose遥测补入新producer/consumer v2；四预留fresh根不打开。历史排重证书仍缺，接口/RAM不能替代认证。
- P5：固定OOD可接触源已真实运行及strict通过，见上；冻结JEPA数值接入继续独占实现。旧12word无接触证据原样保留。

未完成部分不因“已派发”“测试通过”或“已提交”变为科学完成。原目标保持active，P1非贪心真实机制、P2组合收益、P3同预算收益、三组实验及独立确认仍须逐项真实证据。

## P2：固定后缀的真实中点替换已完成 HOST CUDA、strict 和独立复算

MAIN全读最终571行实现与innerdoc，联合143项RAM/saved测试通过（25.16秒）。最终source/test/doc SHA为 `ed4954aed9b705ddb4074c150b2b64c2ea9cd20a3abc6c7a2f7157839765eae7`、`855e8ccab76bae9dc457ec11c38fcaab725c014b9c765908ce8fcfc22ac8d63f`、`b8850bf5c7f3e67bd3dce69717aeeb7f7dc9ff4f6960c1c1119d40a2be6ace5d`。实际prepare SHA `ae2dafda6ac7bffaeb1d47155a97288a706131b2ef5128a24cf552823759f6e5`。

实际HOST CUDA 13.55331秒，8predict/8action_encoder/8predictor/8pred_proj，0新encoder/AD/physics，errors=[]、模型及来源未变。报告 `le-wm-main/results/agent_p2_midpoint_teacher_suffix_20261028_v1_cuda_MAIN01/report.json` SHA `08bec783a5791c15a03396131df4693e3d4c63c43c9523c38f3e499d65e9fe66`；完整strict回执 SHA `36864d57aa6914f0d09ff4e947ef141ccf0037a2fc23f9d7e3f414fb217d04ac`，状态 `all8_teacher_suffix_saved_verified_not_admitted`。

MAIN不导入producer/model的独立NPY算术审计完成，结果 `le-wm-main/results/audit_p2_suffix_MAIN_20261028_v1_MAIN01.json` SHA `90c79b18ecf23f1cb9bb9298e6147ecb4a3e393d8f92c51c7b7a13396af6366d`。八臂latent预测MSE均下降；四根原Q排序一致为 `[false,true,false,false]`，替换后 `[false,true,false,true]`。root3反转修复，root0/root2仍不一致；root0两臂中点相同但终态差异保留。仅支持本次替换改善了表示误差且不足以恢复全部排序，不证明组合收益或理论失败，未改变任何真实任务结果。下一步核older2 latent实际来源，不能未经审计就把剩余失败归为“历史仍有预测误差”。

## 新40分支物理适配器：发现测试缺陷，尚未派发真实执行

MAIN全读647行source与文档，独立回归终态46 passed / 3 failed：substitute动作故障保全测试、RGB guard故障测试及FakeIO缺fileno的存储测试。已回传实现worker定位fixture/实现原因，不删安全测试、不放宽检查。实际prepare MAIN01（SHA `0bf7fe33e48529e8f1934009bd6915d68be2e3645d4ff9006c8d125a35738d7e`）仅保留审计，不以其启动physics；修复后新prepare绑定新版本。新MAIN薄launcher仅减少stdout，完整报告与strict仍原样持久化，不能把本次测试失败计为科学负结果。
