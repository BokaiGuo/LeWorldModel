# P5 固定164 JEPA factual H1 runtime

本版本是原真实5/10/15历史→20的条件factual诊断。与实际LeWM FD164具有同H1时距，但JEPA额外使用同期原生agent position/velocity，因此不是公平同输入，不是RGB-only，不是原root10两packet开放预测。H2另版本准备，不能混称。worker不运行NN/physics；MAIN审读后独立进程HOST CUDA生产。

## 已核准的原source/时间

实际FD prepare `results/agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json` SHA `a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5` anchor为actual cp5 context unchanged。FD source90-129继承observed_context，190-198对raw_suffix5只调用一次backend.step。midpoint ROLES为warmup5/plan0/plan5，即全局5/10/15；不是root0/5/10开放两packet。

原 `agent_mixedmode_physical_20261028_v1.episode` 先place和真实warmup10，在0/5/10保存同condition原history，再绑定constructed task goal，然后plan0及own10。绑定要求snapshot、goal_pose、renderer不变；native默认goal_pose不等于constructed task goal。故历史on帧中的原native轮廓不能解释为构造任务目标，不得重画历史或替换goal。v1桥明确使用原warmup5、plan0、plan5；此处不声称plan0与warmup10所有goal状态相同。真实target为本分支plan10/global20，不是独立witness goal图，goal不输入predictor。

继承冻结桥source SHA `d70352f2fc574ca8a226522fc3dda9f015305894c83a1d1826f05808c005da0f`，以及其全量physical prepare/report/strict、observability、源模型/normalizer/PNG/journal绑定。固定164分支、80pairs、2known roots、off/on分栏；完整原selection/condition/task身份，不筛子集、不择winner，不认证freshness/训练lineage。

## 真实模型路径

复用冻结 `p5_jepa_native_action_response_20261027_v8.load_cuda` 及其原strict checkpoint/expected model receipts、本地DINO官方实现映射。仅授权HOST CUDA，TF32off、MATH attention、eval/inference_mode、冻结参数，不训练/adapter/Qfit、无CPU neural fallback、无下载。原loader的文件SHA与模型state证据沿用，不新增同buffer loaded-bytes或权重训练来源证明。

输入RGB原224，不转换/重渲染；原make_transforms配置224、ratio/scale1、无flip/erase/autoaug。原生agent[x,y,vx,vy]从真实snapshot读取，并逐字节匹配native7索引[0,1,5,6]，不估计速度。raw15=warmup末5+原own10；用原数据投影 `(100*u64).astype(float32)/float32(100)` 与原mean/std，组成3×10 action packets。两个历史packet非零，不能normalized-zero padding。原normalizer与Q均不改。

每请求 encoder实际RGB3→patch[3,256,384]，proprio_encoder[1,3,4]→[1,3,1,16]，原predictor接完整visual/action/proprio→完整 `(prediction[1,3,256,384],None,predicted_proprio[1,3,256,16])`，完整输出全保留，只取最后token作为20的预测。不是把三token当三次自由rollout，也不调用额外best replay。

## 全预测先seal，再标签

固定164实际预测均完成、model_after_predictions等于原model_before后，durable `predictions_seal.json` 绑定164个唯一路径完整full3 NPY和桥prepare semantic SHA。所有预测文件SHA/shape/dtype/finite均验证后，才允许解码真实20 PNG，再付费编码164次。标签路径/hash预绑定不是向predictor暴露标签。异常不解锁未满足seal的标签。

每行保存全patch float64 C-order successor MSE，以及actual15到actual20的persistence MSE；不是native task success、控制收益或新Q。全部80pair固定报告plus-minus successor误差差、预测与观测response L2、cosine（任一零响应为null），不按结果挑尺度、根、overlay或方法，不作164独立样本/显著性推断。跨架构MSE大小不直接排名；P1/P5科学门始终false。

## 计账、预算、失败

| top module | API | batch samples | frame samples |
|---|---:|---:|---:|
| official_encoder |328|656|656|
| proprio_encoder |164|164|492|
| predictor |164|164|492|

共656个真实top-module dispatch，顺序164×(encoder3,proprio3,predictor)，seal后164×encoder1。透明pre/post hooks独立记录attempted/returned、完整所有args/kwargs/outputs NPY与实际API/batch/frame单位；不把内部模块和API重复相加。无encoder去重，重复图仍付费，计数不退款、不补跑/重试。原source/physical/LeWM成本单独继承，不是本次新费用。

输出cap4GiB，disk floor200MiB，terminal reserve64MiB；单metadata32MiB、单NPY4MiB。完整原float32 IO tensor cap2GiB（实际bytes单列，不是GPU激活上界），不得截断/降精度/缩分母；164个full3预测本身193,462,272 tensor bytes，另加所有module IO。CUDA初始free4GiB为工程预检，不保证峰值。报告实际CUDA peak allocated/reserved/free/total与hostRSS；初始化失败memory=null不伪填0。

wall900s起于Store创建，含load/build验证、实际dispatch、末尾model/source检查；失败封存与terminal fsync可能完成于界外。超时为工程失败。保留164rows/80pairs所有missing及原error，不把partial、接口或资源失败当科学负结论。

## 保存ABI与strict

独占NEW output目录：execution_binding.json、journal.jsonl、tensor_NNNNN.npy、predictions_seal.json、report.json。JSON/JSONL统一compact canonical exactly-one-LF、fsync、hashchain，不跳过空行/修复输入。输出cap含report，库存精确无额外文件。报告保存完整calls/counts/rows/pairs、模型前/中/后state、actual imported sources、源文件末尾rehash、字节/时间/内存。

consume只读：外部report SHA→prepare全闭包重建→所有文件SHA/字节/NPY→完整事件先后链→全部实际module IO逐位接线→seal恰对应164真实输出→独立重算164误差/80pairs。严格核原RGB/preprocess/proprio/action输入、模型严格加载/不变及全部资源收据，不运行新encoder/predictor。成功status `all164_factual_H1_and80_pairs_verified`；partial只给blocked integrity receipt。

预处理consumer NumPy reference固定为float32 reciprocal255+原mean/std；runtime调用原CUDA预处理，二者真实CUDA逐位一致性尚待MAIN实跑核准，不能用RAM测试代替。若失败应保留工程失败，不现场放宽容差或改科学阈值。

## MAIN命令

从 `le-wm-main`，现有桥prepare只读SHA已核 `1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f`。NEW目录/文件必须不存在；实际SHA由前一步返回，不伪填未来值。无PYTHONPATH overlay，用全新CUDA进程；P2占GPU时等待MAIN调度。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_raw164_jepa_cuda_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_cuda_20261029_v1 prepare --bridge-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_input_bridge_20261029_v1_prepare_MAIN01.json --bridge-prepare-sha 1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_prepare_NEW.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_cuda_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_prepare_NEW.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_cuda_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_prepare_NEW.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_cuda_NEW --authorization MAIN_P5_RAW164_JEPA_FACTUAL_H1_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_cuda_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_cuda_NEW --report-sha ACTUAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_consume_NEW.json
```

测试含真实Store→两event→完整reader/hashchain、真实NPY/codec/cap mutants、attempt/return失败不退款、seal缺分支拒绝、164/80完整编排+strict。完整编排使用明示RAM模型/tensor protocol double和marker文件，不是CUDA、真实encoder或科学产物；真实NPY路径另测。worker未生产prepare/report/神经结果；可做无持久化的真实saved build smoke验证source/schema/size，不等于生产成功。
