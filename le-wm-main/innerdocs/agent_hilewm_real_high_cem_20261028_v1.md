# Installed CEM + owned HiLeWM high-cost integration

仅本新三件套。MAIN亲自审读/主机CUDA；worker仅非神经RAM测试，无production prepare、模型、physics、pickle反序列化、官方foreign类执行、网络、训练或共享文件修改。正常安装CEM及其依赖仅由MAIN授权run导入；不导入mutable v3。

## 真接线与固定口径

不是旧analytic seam结果。复用冻结HighBatchBridge/安装CEM来源认证、CUDA tensor fingerprint、CEM reducer工程容差；`ObservedHigh.bind_high_model(info,root,...)` 将 **原owned root.get_cost** 直接作为native callback。子类仅在super前后持久记录输入/输出，未替换cost，无globals/model方法monkeypatch。构造走原可信自建loader/export NPY，不load checkpoint/pickle。原high metadata保持high，CEM只收到tensor字段，切分后bridge重附high。

B5再B2，chunk2，samples8，iterations3，topk3，H2，action_block2，macro32→dim64；CEM seed1729+B，完整B×2×64零CUDA init，不调actor。z/goal为`float32(arange%31-15+1729+B+role)/64`，role0/13；只是dyadic合成integration输入，无pixel/OOD/task性能主张。

Box[-1,1]仅合成接口声明。**安装CEM并不裁剪样本**，本实现不改solver、不clamp、不声称候选均在bounds或官方bank。候选和实际cost全保存，不按效果挑选。

固定12 cost dispatch、168 candidate rows；每cost H2×block2展开4 macro，高predictor实际48calls、672 sample rows、1680 context tokens。macro_to_condition、high_predictor、high_pred_proj各自attempt/return/sample/context分轴，projection扁平行与predictor序列行不相加。bridge原predict_API/samples unknown保持null，本runner的真实module hook另计；425 tensor copies是装载，不是预测。0pixel/macro encoder、AD、physics。失败attempt不退款，不重试，不补case；两个case从开始保留，partial报告/已写数组保留。

## 认证与严格消费

绑定MAIN已完成v2 manifest `eadbb1d11383a46b02866a1966588b482dccbb45d6684a55c812d81925782c72`、report `8f19346c99aba5c3cc5dd6efec878356938c24bcb53023444c37791f94ff6771`、receipt `0c2b3055c91cd01101774511e8dff24693ec46caa3a91a7ef43b495eeba0a4d9`；prepare/verify完整重建v2.consume并对整份持久receipt相等，不仅检查label。v2仅complete layout诊断，**NOT fullbase certificate**。继承其可信export/IR/loader来源，源/安装CEM/bridge/自身三件套及完整权重hash前后检查。

run必须MAIN显式同时给`--main-authorized --main-v3-allfour-approved`。后一flag是MAIN在v3四case冻版通过后的外部执行决定，不是此代码生成/检验的certificate；v3不由worker读取、绑定或签署。MAIN现已通报v3四case及strict通过，但本worker不自行生产执行；不得仅因为v2通过就使用此flag。即使integration通过也不宣称official parity/fullbase runnable/scientific admission。

每次root cost保存完整z_init/z_goal/candidates/cost；原callback九张量（var实际是sampleSTD）、全精确CUDA topk/gather/mean/std/sample0检查、输出actions/mean/std/elite均价、input fingerprints及expanded stride0。每次真实macro projection/high predictor/high projection hook保存原精度全输入/输出、layout和attempt/return。消费独立复算全chunk尾批来源、全部native history/macro序列/投影拼接、CEM topk/gather/mean/sampleSTD/迭代链/最终输出，检查全部数组census、NPY SHA/shape/dtype/finite/无trailing、资源、模型及export不变。

`validate_native` 返回从真实native投影记录重建的终态；消费另按原goal复算192项平方和，拒绝仅root→bridge→callback一致却cost错误的情况。CUDA记录同原float32公式精确相等；saved-only误差界独立使用unit-roundoff u=eps32/2、gamma194（差值平方最多3个rounding factor，加191项求和）×真实float64平方和，并保守加194个最小float32 subnormal。全零残差要求精确零。**不是神经重算或neural parity容差**。CEM NumPy mean/std/logging归约沿旧seam预先固定8eps rtol/atol，与cost运算误差界严格分开。

CPU仅NPY/JSON/hash及saved算术；安装solver的CPU variance初建和最终输出序列化不是神经CPU fallback。saved consumer不能独自证明硬件真实性，真实身份由MAIN受控执行提供；RAM字面量buffer夹具不是神经证据。

## 资源及MAIN CLI

32MiB新输出总cap（含4MiB最终报告保留），200MiB floor，600s cooperative wall，从run verify开始计时。排他新建结果目录/文件；每次写前计账，全部新NPY且allow_pickle=False。不解包或执行外来文件。无抢占单kernel承诺，MAIN可外部timeout；不能以超时重启掩盖partial。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。MAIN审读且v3 allfour通过前不运行`run`命令：

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_real_high_cem_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_real_high_cem_20261028_v1 prepare --output "$HILEWM_REAL_CEM_PREPARE_DIR"
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_real_high_cem_20261028_v1 verify --manifest "$HILEWM_REAL_CEM_MANIFEST" --manifest-sha256 "$HILEWM_REAL_CEM_MANIFEST_SHA"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_real_high_cem_20261028_v1 run --manifest "$HILEWM_REAL_CEM_MANIFEST" --manifest-sha256 "$HILEWM_REAL_CEM_MANIFEST_SHA" --output "$HILEWM_REAL_CEM_RUN_DIR" --main-authorized --main-v3-allfour-approved
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_real_high_cem_20261028_v1 consume --manifest "$HILEWM_REAL_CEM_MANIFEST" --manifest-sha256 "$HILEWM_REAL_CEM_MANIFEST_SHA" --report "$HILEWM_REAL_CEM_REPORT" --report-sha256 "$HILEWM_REAL_CEM_REPORT_SHA" --receipt "$HILEWM_REAL_CEM_RECEIPT"
```

目录/receipt均须绝对路径、inner results下本TAG加下划线前缀、不存在；MAIN填实际manifest/report SHA，无未来hash默认。前置：v3冻版四case pass的MAIN批准、原export文件/receipt/原始来源仍逐byte一致、CUDA可用。宏/RGB配对、官方bank、policy全链和物理/科学有效性均不在本交付内。
