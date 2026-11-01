# Exp1/P2 六目标共同 controller：整批 HOST CUDA 事前协议

状态：仅协议，Nash 实际完整 receipt 已到位并绑定下述精确 SHA；待 MAIN 明确批准后才实现新 runner。没有新 CUDA、NN、physics、生产 fullstrict 或新六槽结果；旧四槽 controller、target-only delivery、raw-domain 与 paid replay 全部不修改。此协议不是生产命令或运行授权。

## 1. 已读真实接点与尚未到位的 authority

Nash 当前 companion：`experiments/spectral_reachability/agent_p2_arbitrary_latent_delivery_20261029_v1.py`。
已读完整源码和对应说明。实际 receipt 到位后复核仍相同的三件套 SHA（字节核验不冒称本 worker 重跑其数值实例或 strict）：

- source SHA `709959e6b95b4cbea65d3607b5fccdc571a85f05551a3234bcd9114d5b26f600`
- tests SHA `df801093993b3d3464202b29c53572a1a6c6f98bb65d6c16057ecca14f51e036`
- doc SHA `29048dc54d11663aa52653776dfe866e23732ba49df3fec7e31fe3f99a4090a9`

真实 API：`prepare(output, delivery, delivery_sha)`、`verify(prepared,h)`、
`run(prepared,h,output,authorization)`（纯数值，授权字符串 `MAIN_NUMERICAL_ONLY`）、
`consume(prepared,ph,output,rh,receipt)`。注意 consume 的 output 是目录，不是 report Path；return 只是 receipt 文件 ref。不得把它误作 receipt 内容。

`target_delivery(slot,result)` 取 `delivered_targets[0]`，最终 iterate64 FIRST，float64→float32 一次传输，输出 `controller_payload={z,a,target}`。新 receipt schema 为 companion NAME+`.receipt`，目标在 **`plans[0:2]`**，不是旧 delivery 的 `rows[6]`。成功门必须同时核：

- `complete is True`、`status == all2_arbitrary_FIRST_numerically_verified_NOT_tracked`、`denominator == 2`；完整 receipt 与源 consumer 精确 return 重建相等，不只测 status；
- `prepare_sha256/report_sha256`、`completion_sha256` 以及 report/prepared 路径身份和完整 completion/event/numerical 文件链；
- 两 plan 的 root0/1、off、固定 task/history/action/goal source 身份、`status=delivered`、`failure=None`、`target32_sha256` 和 payload f32 bytes；
- `parent_delivery` 与下述原 delivery 精确一致；原 optimizer/config/coordinate refs 和 inherited425 state 的精确 join；
- 原 numerical loss/gradient/update 账保留并栏；prepare 的 bindings 与实际冻结 source triple 全部绑定。认证/重算的 CPU 费用单列，不计成 controller 新调用或零成本。

MAIN 已交实际 companion authority；下列 SHA 已只读核对文件 bytes，不是未来默认值：

| role | 相对 repo 路径 | SHA256 |
|---|---|---|
| prepare | `results/agent_p2_arbitrary_latent_delivery_20261029_v1_prepare_MAIN01.json` | `3aaffe495a3cfddda0d88403a39b1c937121942c64f79be5a574499579384ad0` |
| report | `results/agent_p2_arbitrary_latent_delivery_20261029_v1_numerical_MAIN01/report.json` | `7431e2b851856fbf5e24d2db3b3762bb6298532be7316e599d6bd32de739c278` |
| receipt（整份） | `results/agent_p2_arbitrary_latent_delivery_20261029_v1_consume_MAIN01.json` | `1e47155d480c23a0bda9cae50e633f0fe7ef56c992d3f489a31e255e43286ffc` |
| completion | `results/agent_p2_arbitrary_latent_delivery_20261029_v1_numerical_MAIN01/completion.json` | `7d2edf45b88a85c9d8e62ea4de63333e3520c204b354ab2cc0f6e07c29839007` |

已只读核完整 `receipt.plans == report.rows`、parent_delivery 精确 join、completion SHA、两根 task identity、三键 shape/finite 与 z/a 对原四槽 payload 的 byte 等值；root0/1 target32 数据 SHA 分别为 `e0e4d42dca524d21aeed5ff933f75cce3d596b55bb6cd635c8773f7e02dac9fc` / `070c0b19ba9cf6674968a1b2e4316c9e58c706d12c0d0c0f27981200b8516d69`。两 plan 均 delivered、failure=null、source_time=null。receipt 的 verification_cpu_calls 为 loss130/analytic_gradient128/update128，这是上游数值验证账，不是新 controller 预算。

实现时将**整份** receipt 和以上四个实际文件身份嵌入新 manifest，不能只保存本节摘要。MAIN prepare 应完整复核 own companion 消费（用新排他临时 receipt，比较完整内容并保留认证费用），不能调用其 optimize/生成替代 prior。若认证开销需要优化，须另获 MAIN 审核，不静默省略链。本 worker 本轮未重放数值 history 或调用其 consume；当前只剩 MAIN 协议/实现授权，不虚构 receipt 缺失。

原四目标 authority：

- delivery `results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json`，SHA `82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62`；
- 原 controller prepare `results/agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json`，SHA `1a021116784dfaa5b4ce976f694fbfde377e6b282ce3553cb57d49a5e033c0aa`；
- 原 report `results/agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json`，SHA `b1358eb3cec42cdddb30b0d138c2bc2e68773f195aa985977bf9c090527382cb`；
- 原 receipt `results/agent_p2_target_controller_cuda_20261029_v1_consume_MAIN01.json`，SHA `34e0491ca05a1b381db3541997c6215c44f9927761d6123a559e604160f0be4a`。

原 receipt 仅认证旧四槽身份/来源，不充作新六槽运行证据。不能修改旧 missing、不能用付费 raw-token replay 的新 token 或预测替代原 normalized mean controller 输入。

## 2. 六槽装配与 byte gate

| slot | root/condition | target 来源 | 新 job 要做的工作 |
|---|---|---|---|
| 0 | 0/off | 原 prior_FIRST | 新 CEM + final mean 双 replay |
| 1 | 0/off | 原 program_first_postcondition | 新 CEM + final mean 双 replay |
| 2 | 0/off | companion plans[0] arbitrary FIRST | 新 CEM + final mean 双 replay |
| 3 | 1/off | 原 prior_FIRST | 新 CEM + final mean 双 replay |
| 4 | 1/off | 原 program_first_postcondition | 新 CEM + final mean 双 replay |
| 5 | 1/off | companion plans[1] arbitrary FIRST | 新 CEM + final mean 双 replay |

固定 0→5 顺序、六槽全分母，不因结果/难度调换。新六槽装配是独立 artifact，原 delivery 四 assembled/两 missing 保持历史原样。

原 slots0/1/3/4：原 row/payload 作为不变来源记录嵌入，不重写原 schema/status；派生的新调度状态放外层。必须对照原 delivery 与原 controller 已封 input NPY，逐键核 **dtype `<f4`、shape、C-order 数据 bytes/SHA**（包含 signed zero），三键全部一致，不仅 target 数值近似相同。原 NPY file SHA 也保留；新路径/文件包装不冒称原 file SHA。

新 slots2/5：只取成功 companion plan.controller_payload，不取 optimizer 第二/第三 waypoint、不重新 cast/优化/平滑；target `[1,192]` 的 C bytes SHA 必须等于 target32_sha256；z `[1,3,192]`、a `[1,2,10]` 必须与同 root 原四目标的实际 history NPY 逐位相同。核 task_id/task_sha256/blind_sha256 与原 observed-RGB/goal 源一致。companion 中 goal192 是生成任意目标时所用的原 goal，不是 controller 另一个输入键。

模型函数边界只收 `{'z','a','target'}` 三个普通 frozen CUDA float32 tensors；source/task/receipt/target64/numerical history/physical labels 等只留在认证外壳。**不能传 witness actions、program raw suffix、旧 mean、prior macro 或 privileged postcondition metadata 作为 warmstart。** a 是两段已观测 between-state action history，不是未来 witness。所有18个输入 NPY 在模型构造/首次 forward 前排他 durable 封存，并列各槽 source pointer/hash。

## 3. 共同 controller 与运行方式

复用冻结 `agent_p2_history3_tracker_cuda_20261029_v1.spec()`、
`agent_hilewm_history3_target_tracker_20261029_v1.track_cuda()` 和 `validate_saved()`：

`H3/N8/I2/topk2/var_scale1/seed0/segments(1,2)`；完整 horizon zero initialization；unclipped normalized Gaussian；identity192 terminal meanMSE float64；每槽 cap12 API/54 samples。原 CEM topk/std 更新及 callback 不改。

**同一新 HOST job、同一个冻结 graph，六个独立 solver 顺序运行。** 每槽重新构造 seed0 solver（与旧四槽 source 同 RNG 生命周期），不改成 B=6 的一次采样/求解，不把 RNG 串行推进为跨槽新 seed，不共享/去重任何 candidate 或 replay。原四槽也真实 dispatch 新求解；旧成功输出只作输入身份对照。即使输入或新结果重复也不抵扣工作。

每槽两次候选全 H3 搜索（N8），返回实际 native final mean（不是 elite-average cost、不是最佳样本）。每个 mean **另付** S=1 whole H3 与 segmented[1,2] 各一次，保留完整 z/a histories，逐 checkpoint、预测、实际 mean、module IO 与 replay cost byte-exact 验证。候选 whole traces 和 installed CEM recorder 全量保留；不把旧 mean 的 replay 或只跑新 arbitrary 两槽称为六槽新运行。

明确新增集成接点：原四槽 wrapper 的 `ACTIVE`、选择函数、slots filename regex `[0134]`、输入/分母/消费规则均只覆盖四槽，**不能直接传六槽、不能 monkeypatch globals 扩容**。新授权版本需自有六槽装配、允许0..5的 bounded array reader/store/consumer；只复用未变的 frozen core 数学、原 loader/facade/guards。原四槽 wrapper 永不 patch。

## 4. 核实后的费用账（计划，不是实测）

源公式：core.Spec.planned，predict_API=H×(I+2)，predictor_samples=H×(N×I+2)。

| 工作轴 | 每槽搜索 | 每槽 mean 双 replay | 每槽总计 | 六槽总计 |
|---|---:|---:|---:|---:|
| predict API | 6 | 6 | 12 | **72** |
| predictor samples/rows | 48 | 6 | 54 | **324** |
| AE calls | 6 | 6 | 12 | 72 |
| low projection calls | 6 | 6 | 12 | 72 |
| AE full-history tokens | 192 | 24 | 216 | 1296 |
| predictor last3 context tokens | 144 | 18 | 162 | 972 |
| low projection rows | 144 | 18 | 162 | 972 |

搜索共12次 candidate cost calls、96条 H3 candidate trajectories；另12条 mean replay trajectories。AE/predict/lowproj各72 API不混成216个world-model predict；projection rank2 rows不冒称其 time tokens。encoder/projector(image)=0、high=0、ADprimal/J/VJP/backward/physics=0；FLOPs=null。core host 内部 candidate value-preserving clone、recorder topk、同步/capture/source/model hash 费用照实计时/列出，不伪装新 derivative 或免费运算。

各槽 attempted/dispatched/module_returned/captured_returned、predict debited、输入rows/tokens和失败prefix独立保存；partial/timeout不退款、不padding、不补跑后挑成功。完整72/324仅在六槽全完成并严格消费后可写为actual；只完成部分时列真实 known/unknown/partial 六槽状态。全六成功不得借用旧48/216拼新24/108凑数。

## 5. 拟继承的工程边界，等待 MAIN 实现批准

建议直接沿原四槽 smoke 的数值/runtime设置与上限：CUDA only，allocated3GiB/reserved4GiB；确定性、TF32off、autocastoff、MATH SDPA，425 state及完整 constructor receipt/source closure before/after强等；每槽前后独立 guard。拒绝 CPU fallback/动态 foreign loader/权重拟合；模型关闭训练且参数不求导。

工程cap沿旧值：总1024MiB含128MiB footer、每root256MiB含32MiB footer、floor200MiB；每槽 core payload32MiB/core wall300s，outer slot300s含dispatch/guard/persist；全job1800s含verify/构造/source-state checks/六槽与保存。六个slot300上界相加已等1800，不能据此保证加上job前后置仍可完成；共享总wall先触发则如实partial，不自动延时、重置时钟或换配置。所有失败封存/末尾ack边界应准确披露，不声称可抢占CUDA kernel。

这里是拟固定资源，不是实际批准后的manifest；MAIN可在任何结果未知时批准或要求新协议版本。source/input/authority metadata reader cap与实际stat在实现时窄核，不能按新 outcome调整。全输出 exact census/排他写/fsync、bounded NPY dtype/header/EOF/hash 与事件链，全量可 saved consume；不得为了同job而省略IO。

## 6. 准入顺序、验收及解释边界

1. 已收到并只读核对 Nash 实际 prepare/report/complete receipt/completion SHA，如第1节；实现仍需完整最终 source/receipt 闭包校验，不能以本轮 metadata 检查替代 strict。
2. MAIN 审读/批准本协议及新六槽实现任务。此时才写新 stem runner/tests/doc，不假造未来receipt或先跑新两槽。
3. RAM tests：真实ABI shape/slot/task/source joins，old4任一byte（含零符号）变化拒绝；new2 FIRST/transport/context篡改拒绝；no witness allowlist；费用/failed debit；六槽input-first与保存/消费 mutation。fixture不冒 HOST。
4. MAIN prepare→verify：封定最终companion authority、原四目标完整不变与六槽输入/model/source closure。此前缺任一完整目标，六槽job不得启动；运行中的新失败保partial，不伪成功。
5. MAIN 单job HOST CUDA→全量 saved consume：实际六条 native CEM返回mean，全H候选、两次mean replay、所有模块/上下文数组、实际预算、冻结425及全source guards逐项闭合。外部持久receipt必须绑定新manifest/report精确SHA并完整重建；不只读status。输出新产物，不覆盖旧4controller或paidreplay。

合法结论仅为 fixed known2、三类交付目标经同一 normalized tracker 的整批工程接通。两根不是独立确认/fresh；arbitrary是geometry surrogate优化后的FIRST，不是假prior。prior physical source_time=null、program首postcondition时距与arbitrary语义不同均原样披露；同controller预算不等于三proposal生成预算相等，不称time-matched/fair baseline/equal-FLOPs。原Hi scaler仍未知，不能借P2原mean/std认证Hi scaler；没有raw inverse、native[-1,1]部署、task收益或physics许可。P1/P2/E1/scientific gate继续false。

本轮只做只读源码/实际receipt和payload ABI与预算公式核查；RAM 调用 frozen spec().planned() 独立核得72/324、1296/972/972，且确认未导入torch。未运行实际 companion prepare/optimize/consume、权重加载、NN/physics/生产fullstrict；无子代。experiment-design 技能只用于事前固定设计/预算/边界，不采用训练、调参、多数据集或按结果选择配置的通用建议。
