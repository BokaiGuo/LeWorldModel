# fixed164 short-start full causal roll v1

NEW实现，只写本stem source/tests/innerdoc。遵循冻结`innerdocs/agent_p5_short_start_full_roll_protocol_20261029_v1.md`，SHA `d042320fb3270c21bd818c5d27b6b84df4b5417fa8c51a85e161c99672a711b0`。不修改probe/H1/H2或共享资料。worker仅RAM/临时文件fixtures；MAIN负责实际prepare、HOSTCUDA、consume。工程complete不是科学通过，P1/P5 admission始终false。

## 真实前置与API

ROOT为`/home/gbk/code/LeWorldModel/le-wm-main`。固定probe实际路径：

- `results/agent_p5_short_start_cuda_probe_20261029_v1_prepare_MAIN01.json` SHA `1fe5f72e9234cfa2360d5a837a68dfbc56b1a94b37fb82775f0be90f63fd8e2c`。
- `results/agent_p5_short_start_cuda_probe_20261029_v1_cuda_MAIN01/report.json` SHA `2f1f60a2f34bf329dcfcb7e5ea378dd3dbf22283be15aa640301ea1ed62974d8`。
- `results/agent_p5_short_start_cuda_probe_20261029_v1_consume_MAIN01.json` SHA `1e376b1f06ca3b04ad71822b76178308dc4ad3d700641c97726b1c7f94c5d346`。

`authority()`绑定上述工程状态、原H2 report/prepare/strict、固定164 catalog及全部原source/model文件SHA，不重跑旧fullstrict。逐枝resolver核root/condition/task/selection及六真实NPY指针，common label额外核H2第656+i调用的官方encoder输出与row.label_patch、SHA/inventory相同。绝不借文件名推身份。原H2和catalog SHA沿批准协议。

`prepare(output)`在同stem独占`<prepare文件stem>_inputs/`创建actual0+四原动作packet银行（每枝四NPY，总67,207,200有效字节）与bank.json，再写prepare文件；失败银行保留partial，不产成功prepare。prepare阶段使用冻结materializer核完整六NPY，包括actual5/10；这是离线来源认证，不宣称从未读取过未来字节。

`verify(prepared,sha)`重建metadata/source authority、原始文件哈希、银行完整性及每枝实际time0/原20动作字节。用mmap仅数值拷贝原native三个数组time0，完整源文件仅hash，不解码actual5/10 feature数值。不存在未来label评分。原full3认证依原外部strict及prepare六NPY交叉核验，未伪称重建其完整消费。

## 运行合同

实际branch-major0..163，各T=`1,2,3,3`。帧历史`[0]→[0,5]→[0,5,10]→[5,10,15]`，只有初始actual0，之后visual与proprio均追加本次新封存末token。原packet窗口`[0]、[0,1]、[0,1,2]、[1,2,3]`，来自原warm10+own10，不改变pixel32 roundtrip/normalizer。旧H2早输出、probe输出均不能作为新pred5/10替代。原warmup-before-goal-bind到plan-after-goal-bind的off/on时序保留，不能把差异唯一归因于history。

复用冻结probe.CudaBackend真实loader、attention观察器、模型receipt和全import源SHA检查；新Trace继承其实际三模型hooks，仅替换本新模块Meter，不改旧“两次”guard/global。输入/输出shape及dtype逐实际tensor读取；限制656 predictor、656 samples、1476 token-frames；任何encoder/proprio_encoder额外调用/嵌套/重试拒绝。只准HOSTCUDA，TF32off、frozen eval、inference、请求MATH并拒绝观察到的fused attention；requested_MATH不当实际kernel证书。

每次durable intent→full inputs→observed return→validated full outputs→stage seal后才滚动。attempted、实际observed returned、validated returned分开，坏返回仍保留真实return成本。模型加载前后receipt/strict loads与原H2相等；全656封存后及终态再次核模型，原权重/config/source实际文件SHA全绑定并终态复核。

全164×4输出完整后才写`predictions_seal.json`和journal seal事件；之后`labels_after_seal`核整批seal才准数值读取：每枝旧H2 context_patch batch3 slots1/2为actual5/10，旧H2 label_patch batch2 slots0/1为actual15/20。保存两个完整batch供独立重算；不重编码、不用H1异batch标签。metadata和字节hash可能此前读取，不等于向predictor反馈未来特征。任意预测失败停止本job，不重试、不缩分母、不为幸存枝先评分；164rows/80pairs/656声明槽及partial资源保留。

输出每枝四端点float64 C-order全patch MSE/actual0 persistence，全部原80对四端点plus-minus response L2/dot/cosine（严格零范数=null）、除2h的central difference L2和MSE差。无显著性、阈值拟合、挑最佳patch/候选。各pair完整full输出和label refs支持后续独立scalar分析；旧H1/H2共同标签重评分与跨协议描述性比较留独立saved分析器后续实现，本runner不宣称已完成该比较或方法排名。前5/10是共同warmup重复输入检查，不是新增物理动作效应。全164的prefix_checks固定与原41枝group的nominal逐字节比较actual0三输入、前两动作packet、pred5/10完整visual/proprio输出，不按差异筛枝；差异保留为复现性观察，不设科学通过门。label5/10与15/20来自不同batch，跨时点绝对MSE有既有批接缝限制；extra agent position/velocity非RGB-only，混合context不认证可达性，known roots非fresh。

## 存储、时间与saved consume

新native IO精确1,209,198,240B，label完整复制322,437,120B，另输入银行67,207,200B；合1,598,842,560有效tensor字节。3GiB输出cap**合计prepare文件、银行、run目录**，2GiB tensor cap（consume把银行也加回核），单NPY4MiB/metadata32MiB、磁盘floor200MiB、终态reserve64MiB、起跑CUDA free4GiB，wall900s从verify前起含加载/最终hash。终态fsync实际可能越过cap时间，elapsed为终态持久化前计时，不能据此称包括完整cleanup。实际每文件/IO/峰值GPU及host RSS单列，原H2继承模块成本报告来源另列，不称等FLOPs。

Store流式append hash避免journal二次方全重哈希，JSON/JSONL严格exactly-one-LF/fsync、独占文件、完整artifact inventory/hashchain。run目录：execution_binding.json、journal.jsonl、tensor_00000.npy等（656×5+164×2=3608文件）、predictions_seal.json、report.json。partial可缺预测seal/label，原失败不当科学负结果。

`consume(output,report_sha)`无torch/physics；重建本prepare，验证全目录/SHA/每NPY唯一顺序/完整IO、四阶段实际输入因果拼接、全部CUDA描述符/动作/计账、完整journal各字段与seal先于labels、同H2标签字节、所有每枝MSE及80pair算术、模型和源不变、聚合存储上限。complete为`all164_short_start_causal_roll_saved_verified`；partial为`blocked_partial_roll`、CLI exit2，只有工程记录通过，无science gate。

## MAIN命令（不覆盖）

在ROOT执行，SHA用实际每步产物，不填未来值：

```bash
M=experiments.spectral_reachability.agent_p5_short_start_full_roll_20261029_v1
P=results/agent_p5_short_start_full_roll_20261029_v1_prepare_MAIN01.json
O=results/agent_p5_short_start_full_roll_20261029_v1_cuda_MAIN01
.venv/bin/python -m "$M" prepare --output "$P"
.venv/bin/python -m "$M" verify --prepared "$P" --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m "$M" run --prepared "$P" --prepare-sha ACTUAL_PREPARE_SHA --output "$O" --authorization MAIN_P5_SHORT_START_FULL_ROLL_HOSTCUDA_ONLY
.venv/bin/python -m "$M" consume --output "$O" --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p5_short_start_full_roll_20261029_v1_consume_MAIN01.json
.venv/bin/python -m pytest -q tests/test_agent_p5_short_start_full_roll_20261029_v1.py tests/test_agent_p5_short_start_cuda_probe_20261029_v1.py tests/test_agent_p5_short_start_native_abi_20261029_v1.py
```

fixture是synthetic native-shaped全四阶段+真实Store/NPY/journal/prepare/verify/consume。为轻量文件测试只对本新模块N临时设1、空pair schedule；另完整656 descriptor守卫和原80 schedule预算独立测试。生产CLI无缩分母/后端注入选项，N必须164；fixture不是真实164生产证据。未运行实际prepare、NN、physics或生产fullstrict。旧goal仍属于其他目标，未覆盖或假complete。
