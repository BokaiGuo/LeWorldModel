# P2 full30 两完整程序迁移 + joint 强对照：实现交接

## 范围与真实输入

实现 MAIN 批准的协议 `agent_p2_two_program_postcontext_protocol_20261029_v1.md`，SHA
`4176575497bcf9c7395438613af0460233a9d94abe891b2023e821cf8395da99`。
新文件仅本stem runner、`_kernel.py`、tests、本文；旧源码、协议、模型、结果不改。
这不是原 full15 的重复结果，不与 full15 合并；仅两个已知off根，非独立确认。
P1/P2/scientific/physics/fresh gate 全 false；joint 外包球更大、计算更多，非公平因果或等FLOPs对照。

固定读取实际 `results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json`，SHA
`82f73b810e5cd34b12907f11d0d063914578263ae4e41163e9581ac3ce3f5567`。
其 arguments.authorities 严格等于原三程序 prepare/report/MAIN完整receipt 的 path+SHA。
完整来源168链/336capsule不删；只预指定每根joint臂的c01/p0与c02/p0，不重评全42、不按旧winner选。
所有指定程序完整planned=committed=15，三packet仍只算一个program。

prepare 继承 MAIN 已完成完整receipt+linker引用IO证据，**不调用旧build/verify/consume/fullstrict**。
独立核实际固定SHA、schema、root/task/goal/identityQ/normalizer、warm raw10因果tokens、capsule digest、
完整输入context及literal chart/rawbits，重hash原prepare bindings、linker bindings及原report全部artifacts。
`full_upstream_rebuilt_here=False` 始终明确；本轮不是独立复核旧神经数值/原J。
不同linker SHA在读取前拒绝。损坏输入拒绝，不变成负结果或另挑root/program。

新增项目源闭包复用原递归source清单，并绑定新kernel/runner/tests/doc与批准协议；
额外用PathFinder纯metadata解析8个已知torch/transformers类源，不执行torch导入。
实际模型加载后每个module的真实class source都须在prepare中已封存，未知不能skip。
loader复用未修改的原immutable-buffer loader，故loader.json保留该**loader**的真实schema；
本prepare/report/plan/consume均自有schema，绝不伪造旧planner consume。

## 算法、chart与接口

kernel公开 `chart/finite/context/legal/proposals/execute/contract/resources`；
`execute(slot,arm,backend)`只接受原task/两个固定program，不接物理标签。
backend ABI：`rollout(full_context, packets_float32, tag)`、`jacobian(context, logits60, tag)`、
`last_ad_contexts`、`event(payload)`。禁止仅传last latent。

transfer每根：A新3predict→完整CA→从**本次CA**执行B新3predict→独立whole6→独立A3+B3，合18。
所有6个完整3latent/2action-token context必须逐位重放；新A还核原A的3个保存context。
旧B输入/输出仅作inherited字段，不能回填新B；每根的新B输入为新结果contexts[2]。

joint每根：同concat60初值，两轮分别incumbent6、AD primal6、四candidate24、selected replay6；
随后whole6+segmented6，合96/2J。固定alpha=[1,.5,.25,.125]，稳定(Q,index)，平局保incumbent。
初始六context须与该根transfer结果逐位相同；两轮和最终Q均不得比自身incumbent更差。
完整4槽、不重试、不padding、不按结果调步数/半径；缺或失败原槽保留null/partial，不替补。

原A/B各自float64 logit30球`.75sqrt15`保留。新concat/joint60域为`.75sqrt30`外包球；
允许joint跨两分块重分配预算，不能冒称同支持的纯模块化干预。
原NumPy `tanh(float64).astype(float32)`生成literal raw，再native normalizer恰好一次。
不逆atanh、不clip、不静默投影、不加epsilon，不以postcast logit范数重拒历史合法见证。

新J严格为192×60 float32 **raw leaf** Jacobian；native六包primal全保存。
AD不对logits或cast求导，J返回后CPU float64连续chain使用
`(2/192)*(J*(1-tanh(u64)**2)[None,:]).T@(z64-goal64)`。
这是批准的连续chart优化约定，不声称float32量化可微或FD已验证新H6。
新HostBackend只继承原单包transition/observe/原生module IO，不继承旧H3 jacobian或全库算法；
自有H6 J/shape/计数和完整saved replay。无globals patch，predict实例观察器finally恢复。

## 真实记录、严格消费与费用

每predict durable intent→完整context/raw→AE/predictor/pred_proj每个intent/input/return→forward return。
每AD durable intent→6个完整原生forward→新NPY写入→AD return。
全部inputs/outputs按原float32原始hex+SHA保存，非只存摘要或terminal。
四个J各192×60，全float32 NPY；consumer先验NPY header/shape/EOF再分配，不能缩到局部坐标或标量梯度。
selection事件在全部候选return后、selected replay前；消费者重演全部算法并逐event比较，未知事件不滤。

新增总账：228predict，各228个AE/predictor/pred_proj，4J/768 mathematical requested VJP/
46080 requested J元素/24 AD primal packet（已含在228内），0视觉encoder/projector、SVD/solve/physics/train。
数学VJP不是engine backward；FLOPs和engine backward为null。
report的attempt/return来自durable全journal census，partial pending也不退款；row resources只是完整返回账。
原planner实际费用/原编码合同在inherited_resources另列，不能抵扣新增或补为免费B预测。

模型HostCUDA-only/TF32 off/autocast off/MATH；checkpoint config+weights从同一已验buffer反序列化，
参数全冻结eval，before/after hash相等。视觉模块入口禁止。加载及每槽allocator peak4GiB/6GiB检查，
每槽重置peak前先保留加载峰值；consumer拒绝缺省、bool或超限峰值。

whole1800秒从verify之前起，单槽450秒包含该槽CPU优化/日志/运输/保存；同步CUDA后检查时钟。
reported wall在最后report publication前采样；完成路径在report flush/fsync之后再次检查whole deadline，
若最后运输越界写terminal_failure.json（会破坏完整inventory，strict拒签），绝不靠旧wall值签成功。
seal先于report写入并由外部report SHA认证其hash，记录durable数据阶段边界及完整数据inventory，
不是一个可自行改timer但不影响report SHA的旁路文件。
source/model最终验证、journal close和census均计入whole阶段。失败finally不走普通dispatch wall门，
可用2MiB reserve写partial/report；journal已有完整IO前缀保留。verify已耗尽预算时不得import/加载NN。

输出物理cap128MiB、reserve2MiB、floor200MiB；增量字节计账，不每event扫描全目录。
所有runtime普通JSON合计强制≤4MiB；rootrow gzip和event单一gzip member，拒绝tail/额外member/空记录。
空journal仅允许partial文件解释；完整consumer的固定census+loader/event重演绝不接受空成功。
实际writer单LF，hash为本原生canonical+单LF，继承原module event格式而非另猜codec。
metadata writer文件打开时flush/fsync，随后fsync目录，xb排他不覆盖旧prepare/结果。

## RAM 验收与128MiB预算

worker只做RAM、临时磁盘夹具和保存输入ABI只读检查；无模型实例化、NN/physics/生产prepare/fullstrict。
真实保存catalog+原prepare的`material`只读smoke已通过：known2、四个固定program、8 installed源，
`torch_imported=False`。没有重新消费上游或写results。

完整4槽合成回调实际走本Store→单LF磁盘hashchain→reader→原module ABI→全部算法saved consumer，
228原生形状forward/4个192×60 NPY，所有whole/segment另付费。夹具J=0只验证接线/稳定ties，
不冒称真实神经导数；另有非零随机矩阵验证continuous chain公式。

`size_budget(rows,events)`逐字段计未压缩JSON，hex blob固定宽度、每个有限JSON float保守32字符、
每个NPY按完整46080data+128header计、runtime metadata按强制4MiB计，不依赖压缩率。
当前同shape保守预算为46,441,406 bytes（journal上界20,416,680、row上界21,645,590、
完整J NPY合184,832、强制metadata额度4,194,304），低于134217728−2097152；fixture命令逐项打印。
不降低float32精度、不删张量/候选、分母或replay。未来若字段增大触cap，必须报告并另批，不能静默改合同。

开发时逾时verify测试发现零事件partial journal被reader错误要求非空；现允许完整空gzip解释为partial，
但完整验收仍要228/4及全部loader/模块事件。此为新下游RAM接缝修复，不是历史科学失败根因。
其他回归覆盖bad shape/bool/nonfinite、J completed/坐标、module IO篡改、事件移位/未知事件、
双LF/坏hash/额外gzip、源binding变化、排他writer、未授权无IO、失败intent不退款及超wall拒签。
最终独占回归：**39 passed in 6.77s**；`git diff --check`通过。
逾时夹具在新增继承账字段后曾因mock prepare缺该字段失败，仅补齐新fixture schema后全绿，未放宽生产gate。

## MAIN 命令（worker未执行生产）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。未来实际SHA只能填实际生成值，不能提前杜撰。
MAIN生产时须按主线程排程串行使用GPU；本worker不launch，也不与正在执行的六槽job争用。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -s tests/test_agent_p2_two_program_postcontext_20261029_v1.py

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_two_program_postcontext_20261029_v1 prepare \
  --linker /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json \
  --linker-sha 82f73b810e5cd34b12907f11d0d063914578263ae4e41163e9581ac3ce3f5567 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_two_program_postcontext_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_PREPARE_SHA

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_two_program_postcontext_20261029_v1 run_cuda \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_PREPARE_SHA \
  --authorization MAIN_TWO_PROGRAM_FULL30_KNOWN2_HOST_CUDA_ONLY \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_postcontext_20261029_v1_cuda_MAIN01

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_two_program_postcontext_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_postcontext_20261029_v1_cuda_MAIN01 \
  --report-sha ACTUAL_REPORT_SHA \
  --receipt-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_postcontext_20261029_v1_consume_MAIN01.json
```

consume只输出saved receipt路径/SHA，不向stdout倒4槽完整raw/context。
当前未完成：MAIN审读/真实prepare→HostCUDA→own strict；真实迁移context和H6 AD尚无新神经验收。
没有physical runner或部署授权，本交接也不承诺任务效能、等FLOPs、P1/P2成立。
