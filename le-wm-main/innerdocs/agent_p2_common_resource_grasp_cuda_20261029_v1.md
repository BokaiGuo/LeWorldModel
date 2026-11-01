# P2 common-resource 三臂 HOST CUDA v1

实现范围：2 known off roots×joint/composition/history-aware bounded GRASP，共6槽；不是官方750步GRASP复现，不开放P1/P2/freshness，不称等FLOPs/等总算力。旧source/results/规格均不改。不读取新GRASP或物理outcome选配置，无训练、Q拟合或物理runner。

## API 与真实接缝

`build(protocol_prepared, protocol_sha)` / `verify(prepared, sha)` 为saved-only：完整重建冻结wall协议authority；逐源task owner task_id、task SHA、warm10、完整history、goal/Q/normalizer精确join。绑定两旧kernel固定SHA、新source/tests/doc及安装类/Adam/模型完整闭包。原wall四槽不在原地改成六槽，新prepare独立声明固定六槽。

`run_cuda(prepared, sha, output, authorization)` 仅接受 `MAIN_P2_COMMON_RESOURCE_GRASP_HOST_CUDA_ONLY`，在任何读取/torch import前检查；HOST cuda:0 only，无CPU fallback。共享加载一次原冻结模型，TF32/autocast off、MATH SDP，前后模型强等。joint/composition使用冻结wall `drive/HostBackend/Control`、原局部算法和严格 `validate_arm`，无module/global monkeypatch。原模型实例观察hooks finally移除；GRASP不替换model.predict。

GRASP只复用旧原生 `Native.step/soft/serial`、数组chart/project与逐packet saved桥，不调用旧 `optimize/consume/validate_root`。本版独立632步schedule/资源账、loss/单次chart梯度/Adam moments与proposal/投影/noise addition/最后完整动作快照/whole与segmented三边exact验证。初始及每次fresh/reset Adam memory使用当前parameter×0保留负零，不用全+0；显式复用已通过真实消费的 `agent_p2_history_joint30_grasp_saved_consumer_20261029_v2.initial_memory/native_lerp` 两个纯算术helper（源码pin `20863c44843ef76434067f3241c5b310e044160a6d21132ee8d722dae63bf34f`），绑定其installed Adam和`ATen/native/Lerp.h`精确SHA。第一moment为CUDA RN-FMA lerp，beta complements使用`1-.9`/`1-.999`，不写近似`.1`/`.001`。不调用该模块的consume/validate_root，不patch globals；原actual prefix负零与joint2/joint10消去fixture独立回归。saved算术的运算次数舍入界仅用于非神经reduction/Adam，**没有放宽tolerance**；不放宽native张量连线或神经输出parity。不能由saved消费声称独立复算神经梯度；原始梯度/数组完整保留。

## 固定预算与partial

同每槽1902 predict API上限且1902 predictor batch1 samples上限；context=3、raw15、joint30球及原normalizer/Q不变。joint自然1902，composition1062不padding；GRASP532joint+100sync=632 scalar backward、632Adam、532noise、10reset，1896优化forward+6导出=1902。最后6 whole+segmented replay包含cap。每完整优化unit先预留3+6；实际各callback进入前逐项attempt扣账、返回后才completed，失败不退款。

共同450s搜索：同步CUDA时钟，含nominal/bank/AD/CPU优化/数组capture/journal/fsync/export；initial saved验证、host setup、loader、search后seal/整row压缩落盘、最终identity检查分别记录，不隐去scope。搜索clock在seal时冻结，seal事件及整row持久化在外，两类臂一致；total_seconds为最终report写入前的实测总时长。六槽顺序 root0 joint/composition/GRASP、root1 GRASP/composition/joint。外层3600s只是安全界限，cap1536MiB/reserve96MiB/floor512MiB，allocated4GiB/reserved6GiB；每臂保存实际allocator采样峰值，整阶段另存CUDA allocator真实max并检查。不是额外搜索额度。MAIN应串行独占GPU。cooperative cap不能抢占在途kernel，late返回计真实完成/overrun，但不派发新操作/发布late快照。

GRASP每完整update才可提交last snapshot；未完成Adam/噪声/同步重置不提交，不选best iterate。缺replay、wall到期、真实失败保留None计划/原始prefix，固定六槽不删分母；不会借旧endpoint、补步或伪造返回。GAccount保留pending而不虚构失败操作的成功闭合。载入或全局fatal可能只产生failed_partial报告；完整strict拒绝，不伪签六槽成功。

`consume(output, actual_report_sha)` 验证全输出集合/字节SHA、单gzip hashchain/顺序/未知事件、loader consumed buffers、所有六槽严格算法与账本、搜索非重叠时间、三边raw/context、原任务/模型/源闭包。返回六项plans（允许None）、accounts及统一resources。谱臂原ledger的predict是packet开始，不冒充partial时实际API；统一向量另从真实predict_API/module事件提取API和sample。scalar backward、完整J、mathematical VJP requested、AD primal、J元素、SVD、模块/rows各自独立，不相加。engine内部backward/JVP/FLOPs未知=null。继承原四task编码16/16 cohort只列一次，两off task分摊unknown，不能重复退费。

## MAIN准确命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；worker不执行生产prepare/run。原v2协议实际SHA来自原冻结合同，所有NEW输出路径须不存在。准备失败保留记录，勿覆盖。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_common_resource_grasp_cuda_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_grasp_cuda_20261029_v1 prepare --protocol-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json --protocol-sha fbc151b975409eff95bffa084707531f20d1659cbcf08ea23e5da1b490db4bd2 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_grasp_cuda_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_grasp_cuda_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_P2_COMMON_RESOURCE_GRASP_HOST_CUDA_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_grasp_cuda_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json
```

RAM数据是显式synthetic saved buffers，仅测试接线/边界，不是假counter科学证据或CPU神经实验。真实CUDA loader/梯度/partial/资源和完整strict尚需MAIN审读后执行；本次不签任何科学放行或物理release。experiment-code技能只影响结构化日志与回归要求，没有采用训练/结果调参/收益下限建议。

## 实际回归与冻结边界

指定venv/OPENBLAS_NUM_THREADS=1/禁bytecode与pytest cache：本stem **53 passed in 11.62s**；与冻结wallcapped tests联合 **81 passed in 23.60s**。覆盖完整632/1902 synthetic保存链、六槽真实writer→reader→strict、partial backward/Adam/deadline/缺replay、源task join、逐轴API/sample、batch2拒绝、输出/模型/precision/clock/账户篡改及CLI参数。MAIN要求的旧实际CUDA arithmetic前缀只读核验原report/journal SHA与canonical chain，到root0 joint10即停：负零185及两个FMA/complement错误点，不访问terminal结果来选择配置。不调用旧GRASP consumer/validate_root，没有生产prepare/NN/physics/network。原常量/方法/结果均未patch。

三件套冻结后不再追加函数或改本文；任何修复另新版本。真实prepare闭包与主机CUDA仍待MAIN执行；RAM通过不能替代真实运行或科学证据。
