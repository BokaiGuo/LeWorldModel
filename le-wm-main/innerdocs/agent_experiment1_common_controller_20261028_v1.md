# E1 common controller v1：四dispatch、learned两槽unknown

仅本stem源码、RAM测试、本文；MAIN负责代码审读及实际HOST CUDA。冻结旧文件不修改。
采用experiment-code技能的固定实验/随机种子/持久日志规范；不采用其“零成功应修改”的泛化建议。
旧blocked个人goal占用API，不覆盖或假标完成；本交付条件是三文件、RAM验证、saved接线及SHA。
实现依据为MAIN接受的`innerdocs/agent_experiment1_next_step_JASON_20261028_v1.md`，SHA
`b06787399646bc0066a005e0cef10bfa4637df32170a29d0c4b9719088d372d2`。

## 固定实验，不是完整E1

原moving off task00/01，完整真实三状态与两条非零incoming action history、原goal/native192/Q、
normalizer、packet logit ball全部不变。固定root-major、arbitrary/learned/compiled六行；
只有arbitrary和compiled四行dispatch；learned两行永久unknown，commands/resources/sample
均null。无usable prior及H10官方fallback不是0%效能。两known roots不是独立确认样本。
不读physics outcome/证书选目标，不变科学门，不训练、Qfit、AD、encoder或physics。

arbitrary严格继承已接受参数：PCG64 seed2028102801/02；float64 draw20×192；
scale=norm(goal-rootlast)/sqrt192；`rootlast+scale*draw`一次转float32；按mean192对goal误差
及index稳定选一项。保存全部draw、候选和评分；scale0不重抽。是弱有限latent随机搜索优化，
不是梯度法、learned prior或所有arbitrary优化的代表，不投影到可达集。

compiled来自既存beam strict `consume`，report SHA
`d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`。
从通过完整节点/context/动作/whole/segmented核验的选中路径prefix5提取；绑定其report、
prepare、全部artifacts和源闭包。旧beam选中计划与原目标/模型保持identity；不再重选beam。
继承旧beam每根144predict、144 AE/pred_proj及120 H2选优成本，单列而不称proposal等成本。

`build`重核实际两个task及model闭包，并重新生成目标摘要。`run`先写全四目标、完整20点draw
和compiled动作见证，fsync数组与targets文件，写ALL_targets_sealed事件，再加载模型。
prepare只含小摘要/四目标/compiled prefix provenance，不复制大上游闭包或完整随机draw。

## 一个共同低层求解器

两臂调用同一个`optimize`函数。每根controller seed2028102811/12，不含arm；同根两臂首轮
24参数draw逐字相同，后轮因目标产生不同elite是预期行为。3×24完整H2；每packet5raw，
每候选从原root/context开始，不把任意臂执行结果带入另一臂。
objective=`.5*mean192(pred5-target)^2+.5*mean192(pred10-original_goal)^2`；不覆盖task.goal。
CEM elite4、初mean0/std1、ddof0、std clip[.05,1]，原投影logit球→float32 tanh→raw10×2，
原normalizer仅一次。保留全部参数、logits、raw10、两完整rolling contexts、评分、更新和winner。
选全部72项稳定全局argmin，不重放winner、不评nominal、不padding、不重试。

每成功group恰为144predict API/144 predictor samples（batch1）/144 AE/144 pred_proj；
四group各资源总576，不将API与内部模块相加。不增加encoder/image projector/AD/physics。
Meter在真正dispatch前收费、返回后同步记录；中断无退款、不补分母。共享原encoding8+8
继承成本单列；proposal不同搜索成本不能混成等FLOPs或等总算力。

重用sealed beam `NativeEdge/validate_packets`与原matched采样`NativeCuda/Meter`，不修改
globals、不冒用旧execute或旧预算声明。自身Dispatch严格72条H2，SavedDispatch消费真实
AE输入完整history→API/predictor→pred_proj→successor桥，并重建所有CEM/RNG/选择。
实际getter/hooks是原冻结transition路径；worker没有运行模型。CUDA0/MATH/TF32off/
autocastoff/frozen eval，显式CUDA allocation，无CPU神经fallback。

## 64MiB下全精度无损存储

旧JSON全展开并在attempt/return重复写各tensor，不可用压缩率担保64MiB。本版采用新公开
桥schema：所有至少10项、至多4096项的纯Python float矩形数组，按little-endian float64
原值存入`arrays.bin`；原float32可精确表示为float64，原CEM参数/draw float64不降精度。
保留signed zero。hash内容寻址去重只去重存储，不去掉任何模块调用或回执。
引用含`__f64__`字节SHA、shape、offset、nbytes；不使用pickle。数组先fsync，再写引用；
事件为带index/previous/SHA的完整hashchain，gzip Z_SYNC_FLUSH逐事件fsync。

保守资源估算：每packet模块receipt的不同数组最多包含30维incoming、576维AE输出、
576维输入z、576维predictor输出和576维API/projector输出，约2334个float；再计完整
successor history及分组候选/银行/随机draw等重复布局。按每packet总4096个float64的
保守包络，576packet约18MiB；root/proposal、参数bank及72×4候选参数远小于额外2MiB。
本版给数组32MiB硬上限，不依赖碰巧重复数值。事件结构/引用/标量及四份结果总expanded
metadata另设24MiB硬限。32+24+.5final=56.5MiB，另有gzip/zlib framing余量，始终另核
总64MiB。所有写入还核200MiB free floor。实际cap失败保留partial，绝不缩候选/精度/分母。
上述是固定192维合同的保守布局估算加运行时硬门，不是对任意模型/错误schema的无条件保证。
未知/过大shape或非finite拒绝，不能换模型后沿用该估算。

每次成功恰好12个artifact加report：binding、targets、arrays、events，以及四组各result/plan。
result使用有界单zlib流`.result.json.z`，journal单gzip流；不可当普通JSON/GZIP互换。
consume认证全部文件字节，解码总metadata≤24MiB，逐数组SHA/range/shape校验，拒绝
重叠/空洞/未引用尾部/双流/尾随垃圾，重核event链和全部计数，再运行纯saved校验。
对每张量保存完整输入/输出，不是只保存末状态或hash摘要；数组可由consume无损还原。

900秒为Store创建起dispatch/write检查，含模型加载但不含此前build/verify，不抢占阻塞native。
所有六行预置；first error保留，cleanup/model/source/journal/hash/report各独立尽力完成。
磁盘故障/进程kill不保证report可写；已有场景和pending证据保留。strict consume只接受完整
四组与两个原unknown行；partial不能发布plan，不能把缺测当0或完整E1。

## 接口与MAIN命令

在`/home/gbk/code/LeWorldModel/le-wm-main`执行，以下是交接命令，不是worker已执行实验：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_experiment1_common_controller_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_20261028_v1 prepare --output results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_20261028_v1 verify --prepared results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_20261028_v1 run --prepared results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_experiment1_common_controller_20261028_v1_cuda_MAIN01
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_20261028_v1 consume --output results/agent_experiment1_common_controller_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA
```

`prepare(output)`返回实际prepare SHA；`verify(path,sha)`完全重建闭包（包括上游beam strict
consume），只做saved/CPU数组，无神经。prepare≤256KiB且独占新文件；源/tests/doc绑定后冻结。
`run(...)`只生成模型计划，无物理授权。`consume(...)`返回四plans，每条含source_slot/arm/
task_export/task_sha/blind_sha/task_contract_sha/raw10/logits/target/两contexts/模型终距/
tracking_cost/selection_sha/whole_run_report_sha；另返回六rows和固定分母。消费者需新显式
authority接线，不能冒用旧算法whitelist；不提供physical run或E1/P1权限。

RAM可调用`optimize`、`journal_check`、`ArrayArchive/Reader`、`validate_saved`；fixture不代表
实际模型证据。测试覆盖四组完整576假transition、真实形状192receipt、全CEM重建、同首轮
draw、错误history/output/选择/计账/顺序、失败无重试、unknown保留、无损float64/signedzero、
双流/数组损坏及RAM整包strict消费。还用RAM虚拟文件系统走实际`saved_payloads/consume`
入口，破坏外层SHA重新颁发后的内层事件链仍拒绝；首错+secondary日志/close故障保留。

实际worker验证：上述专属测试 **39 passed in 14.42s**。真实saved `build()`已成功，
2slots、144bindings，首次canonical prepare为77,945bytes（仅内存，未创建prepare或输出）。
最终文档SHA绑定会导致prepare大小微变，MAIN实际prepare应重新build。真实compiled两个
prefix target semantic SHA分别为
`1b959932744b10762de67deaf841792c8fc5f720e3826b1aeb551510d36297c0`与
`d6145afdb09dc443ba924f1bdc955269b8b6ae3996b22d7debfe431c66fd67c5`。
该检查只消费封存beam和task metadata/数组，无神经调用；不是新controller实际CUDA结果。

## 科学边界

这只补“共同低层controller实装”，不是完整E1、官方SAGE/Hi-LeWM复现或P1通过。
learned仍缺认证资产/坐标配对及非fallback共同合法时域；不能从本结果删除此缺项。
compiled使用继承搜索目标，proposal成本不匹配；arbitrary仅弱20点有限搜索。
真实可实现性还需MAIN另授权own raw10物理及RGB消费；native success包含agent+object，
不能用本模型Q或object-only投影代替。保留全部失败、零/负结果，不作泛化/显著性结论。
