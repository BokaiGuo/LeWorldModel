# D2P：共同控制器、子节点编译与后续P实际推进

本页以实际文件及终态为准，承接v10；原C1–C3、三实验与P0–P5范围不缩小。当前未完成研究闭环，目标保持active。神经执行仅主机CUDA，无训练、Q拟合、CPU降级、阈值放宽、结果筛根或公开推送；四预留新根未打开。

## 1. Experiment1共同控制器：CUDA与物理均已真实执行

MAIN全读497行共同控制器及协议，39项独立测试通过（14.22s）。固定两known根、arbitrary/compiled四dispatch、learned两unknown科学槽。共同CEM每组72项H2候选，144预测，总576；相同首轮随机draw、同原目标/动作normalizer/native192/Q。arbitrary是有限20点弱latent搜索，不是官方Hi-LeWM/SAGE。compiled继承beam proposal成本单列，不称总算力匹配。

实际prepare MAIN01 SHA `8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd`。主机CUDA22.91094秒，576 predict/API及AE/predictor/pred_proj全部attempt/return，errors=[]、模型未变。报告 `le-wm-main/results/agent_experiment1_common_controller_20261028_v1_cuda_MAIN01/report.json` SHA `bc3cd74c85f878cdfc7b170d0b5b555827e0dc4a59cdaea6fb5781e71f03767f`。

MAIN薄wrapper未改原consume，完整数组/事件链/CEM/RNG/动作/context接线严格重建通过；持久回执 `le-wm-main/results/agent_experiment1_common_controller_20261028_v1_consume_MAIN01.json` SHA `b4b72dea9d782c6d2e1b93f50523a1089e7f3d894e2e2c1171fc13f9b8cd425d`，四plan/六槽/两unknown均保留。

物理adapter全读263行及后续8行严格回执绑定增量；最终source SHA `61a16e772175627fed91d53c95e0b78566117b02676b54184d7d8e5e11e01f97`，28项MAIN回归通过（1.52s）。交接中authority新增严格回执字段曾导致MAIN01 selection被拒绝，尚未生成prepare或调用物理；原selection保留，用MAIN02显式补字段。不是科学负结果，亦未绕过准入。

实际prepare MAIN02 SHA `160517a999680baf8bc79809e36d01e0d5ed8ade69b54706ff48edb6b5786586`。四分支物理exit0，80公共控制步/816内部physics/88PNG、4construct/close、8reset、16set_state、8goal_render、4goal_bind均完成，工程失败0。报告 `le-wm-main/results/agent_experiment1_common_controller_physical_20261028_v1_physics_MAIN02/report.json` SHA `a75d97bfd5e6c19bf942ce5510ad7689ad3fd79f7bb3539b2ca96a6cda9fec6c`。

两known根原native成功均为0/2臂，即四分支0/4；learned未知不计失败率。记录中的终态native7距离：root0 arbitrary105.35273/compiled121.32344，root1 arbitrary463.33935/compiled214.18477。两根compiled物体XY误差较小，但root0全native距离较差，不能挑object-only宣布胜出。原距离含速度；模型latent tracking error仍null，待真实PNG的单独CUDA编码。该物理producer尚无独立saved consumer，已派独占实现；自计账不是独立严格验收。

## 2. P3从付费child信号推进到实际程序编译

MAIN全读632行新编译器及文档，独立69项测试通过（22.21s），冻结source SHA `f3bc4a58fb4dfc1ef640abd2ced50eafe5848a7dd9fc9110901cb98db5293e17`。prepare重算上游完整strict，不接摘要/旧root J；输入manifest SHA `f7455c34f7b1035fc8043b7500e4d70ba23b441ab21454cd3cdbded378a90ad2`；实际prepare SHA `38a1dd39a4edf0c4661a1fb8cd992faa7652e4ba44ae1c1bc06c36790abb0b94`。

实际HOST CUDA exit0，wall109.01125秒（含verify等规定范围），模型保持、sources_verified/cleanup_ok=true、errors=[]。657predict/AE/predictor/pred_proj、41compile/solve/arraySVD、369独立replay、164metric全attempt/return匹配。原48槽保41已编译/5整节点预算拒绝/2not_reached。报告 `le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_cuda_MAIN01/report.json` SHA `885f69f6de5ce9b78b8f3acdc84f14ef8891ba3ecd79912fc9af4e1920fc7ffb`。

完整strict消费已启动session29118，未落盘前不算完成。只是observed child局部extension，非完整depth3树；未定义reconfiguration仍拒绝。含付费signal后fixed391predict/adaptive634predict不同，不称等实际算力、物理收益或C3成立。后续原生物理桥/等实际预算协议继续独占推进。

## 3. P2真实历史资产：缺项有界定位，不捏造triplet

MAIN全读325行inventory及协议，18项独立测试通过（11.01s）。实际saved inventory SHA `4a11fd5d313f0feed8a9b802cef91b45ce4d9ee12d7e208b5de4a9156cb70de8`，路径 `le-wm-main/results/agent_p2_midpoint_triplet_inventory_20261028_v1_MAIN01.json`。root0两臂可认证triplet，其他六臂warmup10闭包未建立；保所有八槽，未分类匹配元数据和旧JSON解析错误不隐藏。不是宣称磁盘上绝无资产。

完整verify正在session56910；不重复神经stageA、不拿root复制成warmup10。缺三根的原collector/reset/warmup计量重建适配器已派实现，未授权其自行物理执行。

## 4. P5全四词物理桥

MAIN全读653行及最终新增prefix/suffix shape预算检查、协议；独立第一次测试发现fixture跨测试重复xb写同report、交接doc尚缺，34pass/1fail/6error。worker修复每测试独立fixture并完成文档，production物理语义不改；最终41项MAIN测试通过（27.45s）。source SHA `0a00d0f7864f03fc8e9ddf1fa7cef749a1d57f6c977357ce78c309262aa4556f`。

实际prepare SHA `46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7`；原四词numeric CUDA及严格回执完整绑定。主机物理session51463已真实启动，输出 `le-wm-main/results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01`。四prefix全部byte-gate后才执行四suffix，固定80公共步/804内部Space.step上界，callback实测。终态/strict未核不宣称成功；不以旧push/zero充当本次四词结果。

## 4.1 后续真实终态覆盖运行中快照

P3完整strict已exit0并落盘11,250,632bytes：`le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_consume_MAIN01.json` SHA `d6322717231d72bb18e26eab1acbf6afe2dbc65eaa601af1fface21083a7e91c`。全48槽、41条动作及独立whole/segmented/selected-prefix/full native IO重建通过；不是物理收益。P2 inventory完整verify亦exit0，仍stageB_execution_ready=false；已派具体缺三根warmup重建实现，而非继续重复现状文档。

P5四词真实物理2.43269秒exit0，四prefix全部身份门通过，80public/804Space.step、84render/get_obs、4construct/close、155原生collision callback；report SHA `a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633`。155是callback总数，不默认全部agent-block，真实body_roles保留。

首次strict被旧8MiB `physical.read_journal`挡住，实际9.1MiB完整journal仍在原128MiB输出/96MiBpayload预算内。MAIN新saved-only ReaderBridge用已封存96MiB payload作为有界读取上限，不改producer/module globals、字段、分母或原full validator；逐行完整性/事件链同样核验。3项MAIN测试通过（.35s）：大于8MiB完整日志、篡改链、截断/cap拒绝。使用原 `validate_saved`全量验证后再核源码/报告闭包，全部40 private suffix frame rows严格通过。

实际完整strict回执 `le-wm-main/results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json` SHA `f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee`，`all4_saved_native_words_verified`。此故障确认为消费端容量问题，未重跑物理、删回调或调低精度。仍task_success=null、scientific/action-witness/physical-release=false。下一步所有四词真实agent/block响应分析已派新stem，不把agent惯性差异当物体控制机制。

E1物理独立consumer已由MAIN完整审读218行及协议，30项MAIN独立测试通过（1.12s）；全量回执已落盘1,125,939bytes，`le-wm-main/results/agent_experiment1_common_controller_physical_20261028_v1_consume_MAIN02.json` SHA `3926373a79e5c0eb9a6295dadd4ce1403dfca13d9baad51639cd7012c913c12b`。2414事件/88PNG/12checkpoint、完整原warmup/source状态与像素、native7/目标、资源账独立核验；真实模型/物理不重跑。原完整编码差异/任务收益仍待后续真实RGB薄桥。

MAIN另全读P4窄schema修复，27项独立测试通过（.23s），真实saved执行确认conditional全部8attempt、observed两行和960unknown完整保留，discharged=0、freshness_certified=false。这修复metadata漏读，不补造历史reset事实。已把空闲实现线转向实际缺失RGB40编码，而非重复构建metadata框架；旧Poincare handle不存在且新spawn受线程上限拒绝，复用Nash继续，不把失败spawn说成已启动新智能体。

MAIN两项controller完整receipt writer另有4项测试，连同P5有界reader共7项通过（.42s）；验证全返回结构不裁剪、排他写入、validator失败不生成回执。这里只是持久化工程覆盖，不增加模型/物理证据。

后续独占实现已实际分配：Jason做E1真实八端点RGB tracking CUDA桥；Hegel做P3 child raw5物理桥及缺失的saved native源核验；Bohr做P2缺三根历史重建；Nash接旧handle失效后的P1固定40 RGB编码；Pauli做P5全四词saved response分析。MAIN继续亲自全读/验证再执行。P3源接口已定位原 `agent_p4_moving_multiboundary_20261028_v1.consume`，它已有操作账/原完整数组验证，rolling不是唯一入口；新桥仍需补其未覆盖的native起点/snapshot/PNG身份，不重复发明已有validator。

## 5. 完整闭环仍缺

P1非贪心真实能力改变与任务收益；E1 learned-prior合法资产及真正独立确认；P2组合对强对照收益与历史/API归因；P3完整树和等实际预算真实闭环；P4历史freshness原始ledger；P5接触动作响应到可执行机制。并行实现、测试、封存和实际CUDA/物理是进展，但不是上述科学命题完成。
