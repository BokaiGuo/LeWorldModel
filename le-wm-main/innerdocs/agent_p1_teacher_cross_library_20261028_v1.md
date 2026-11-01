# Teacher15共同对称库：8锚点CUDA编译合同

仅新增此stem源码/测试/本文。MAIN负责审读、prepare和实际HOST CUDA；worker只运行RAM测试及saved-only内存build，不启动模型/物理/网络，不写MAIN产物。原已封存机制合同SHA `622bc124a5b08f787acac569f7b1875ed1424508d87e9a35bc0670b819d06423` 不变。

## 固定问题与边界

四任务依次 `(0,off),(0,on),(1,off),(1,on)`，各角色Z=实际zero5、S=已封存mixed selected-prefix5。每角色固定取原zero探针分支的probe-start15 PNG；先核同角色全部五分支在15步的完整snapshot/state/pixel一致，不能用20步probe endpoint。旧40/zero20报告与MAIN已验收strict回执采用各自真实schema，外部SHA固定，不重新运行旧物理consumer或signed分析。

未发现覆盖这8个真实Z/S teacher15图的完整严格匹配既有编码。旧GN/D2P/moving terminal编码不能改名复用。因此本合同**自有8次encoder+8次projector，不dedup**；原same-condition5/10 latent及goal从原encoded task按SHA/内容复用，保历史费用。部署上下文为 `[z5,z10,z15_actual]`；incoming原warmup后5 `[0,.1]` 加本角色实际prefix5，float32原mean/std一次，不二次除100。planner whitelist仅context/goal/Q/raw_history，原probe端点/原生指标/响应方向不能进planner。

原identity192 Q、damping .002、cap .75√5、模型/normalizer/β均冻结。每anchor independent nominal1 + AD forward1/192×10 J1；用冻结 `bounded_ridge` 一次产生logits，NumPy tanh(float32)只算一次成为literal raw权威buffer。实际负动作只做float32符号翻转，不倒序、不重tanh、不clip。

8个suffix全部封存以后才评价共同库 `[zero,uZ,-uZ,uS,-uS]`。每context对5项各独立预测，再fresh同context逐项重放5次；zero重复及碰撞不退款、不去重、不挑赢家、不做earlystop/retry。release为40项完整raw10计划，不以物理效果/预测Q筛项。

## 预算及实际trace

固定总额：encoder/projector8各；predict API/samples/action_encoder/pred_proj96各；AD forward8包含在96内；Jacobian8，数学VJP1536；bounded_ridge8，成功冻结helper内SVD8。每context12 forward。AD API的grad=1是192个输出的batched VJP调用，不是一个数学方向。继承原history/goal编码及mixed搜索/物理费用另账，不称同FLOPs/完整算力匹配。

直接复用冻结 `kernel.TorchBackend` 与 `compiler.bounded_ridge`，不修改旧模块任务globals/zero gate；原始完整native精度保留。复用只读native hooks，保存实际pixels、预处理、ViT全257×192输出、CLS→projector桥、96次AE/predictor/pred_proj所有输入输出、原J与nominal/原logits/raw/context。AD作用域采用已有SuffixAD有界API跟踪并完整restore（不是替换旧任务globals）；失败保留actual attempt下界，成功还核完整单次vectorized jacobian/grad。SVD记账是源码绑定helper一次调用内唯一SVD；helper失败时仅说明已attempt该helper，不编造其内部返回。

CUDA-only、MATH、TF32=false、autocast=false、batch1；前后tensor hash及完整source/file closure。无CPU神经fallback、训练、Qfit或新物理。全部8状态保留；先封存全部B再另行物理，部分失败不可release。

工程限额：prepare64MiB（完整原43k路径SHA元数据，不删closure）；输出128MiB，final reserve32MiB，journal按未压缩64MiB，report16MiB，disk floor200MiB，wall600s。数组采用原uint8/float32/float64的NPY gzip，file/NPY/array三层SHA分开。无pickle。O(1)写账，实际部分写/fsync错误仍计费，末尾完整inode/name/size census；no-overwrite，未知文件拒收。最终model/source/census/report错误分别记录，首错不被finally覆盖。`capacity` 返回原数组+NPY开销+全journal+report保守上界，完整输出不降精度。源码绑定的AE emb_dim192及ViT257假设必须保持。

strict consume重新构建prepare，逐文件SHA/hashchain、真实模块尝试/返回、每个literal raw与原float32 token算术、同condition原context/实际15图编码桥、原J/solve独立saved算术、5+5模型输出/评分和完整40计划；不新增模型调用。saved复算SVD8另计，不能算成CUDA工作。原上游严格验收回执作为历史acquisition authority，不重复实际consume作业。

## MAIN命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_cross_library_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_20261028_v1 build-memory
.venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_20261028_v1 prepare --output results/agent_p1_teacher_cross_library_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_20261028_v1 verify --prepared results/agent_p1_teacher_cross_library_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_20261028_v1 run --prepared results/agent_p1_teacher_cross_library_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01 --authorization MAIN_TEACHER8_COMMON_LIBRARY96_CUDA_ONLY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_20261028_v1 consume --output results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA
```

SHA占位仅由MAIN实际终态计算后填，未伪造prepare/report。CLI消费结果stdout由MAIN按需要新独占保存，不覆盖任何旧产物。

## 未完成/准入

worker实际验证：45项RAM测试通过（全量8×12 analytic fixture及完整模块保存/解压/三层SHA、真实token重建、负向拒收、部分写和首错保留、硬预算predispatch）；CLI help通过。真实saved-only `build-memory`成功：8 anchors、44,096 bindings、canonical 9,788,985 bytes，未写prepare。保守原数组14,100,608 bytes，含1200文件NPY/gzip开销、完整64MiB未压缩journal及16MiBreport的存储上界99,215,488 bytes <134,217,728；未按压缩比猜容量。该数量是静态完整trace容量界，不是已运行CUDA的实测占用。

实际新8图编码、96预测/8J及strict saved消费待MAIN执行；40分支新physical adapter及其实际完整交叉执行另新版本。不得拿旧五探针端点充当本库物理结果。当前仅两个known roots，S仍为stage-greedy，C2非贪心前提缺失；动作可反向，符号变化本身不等于能力增加。即使库内支持或模型对角线改善也不隔离state/velocity/contact中介，不构成P1、独立新根确认或理论否定。P1=false。
