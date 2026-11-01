# 固定164原历史开放两packet：JEPA H2 v2

状态：供MAIN审读、prepare/verify与HOST CUDA调度；worker仅RAM/磁盘fixture及saved metadata验证，不执行模型或物理。P2生产占GPU时不得启动本runner。v1桥及未完成的H1 runtime保留不改，本版本不生产其正式公平比较。

## 时间与权限合同

固定原164分支、80个差分pair，两known roots×off/on，全部保留。输入只取原物理真实warmup0/5/10，即全局0/5/10 RGB与对应pre-action agent[x,y,vx,vy]。不是RGB-only。实际15仅作为封存后标签，不是任何预测输入。无新reset、render、encoder训练、AD、物理或下载。

有必要纠正比较对象的命名：原root-H2规划不等于后续FD164。实际FD prepare `results/agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json` SHA `a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5` 中contract.anchor为actual cp5 context unchanged；midpoint ROLES为warmup5/plan0/plan5。旧FD164实际是5/10/15→20 H1。不能回称旧FD164是两packet开放预测。

本次新H2前瞻合同只对齐原root预测时距，不认证完全同输入/同算力跨模型公平性能比较。原生proprio权限、预处理与特征维度仍不同；模型间feature MSE大小不可直接排名。164不是164独立样本，不提供显著性、freshness、训练lineage、控制收益或P1/P5准入。

## 原history / goaloverlay

`agent_mixedmode_physical_20261028_v1.episode` 先place、真实warmup0和10步（5/10存原history），核原同condition source RGB3，然后才bind constructed task goal、记录plan0并执行own10。`p1_moving_target_overlay_source_20261027_v1.source_episode` 亦在采完历史后绑定constructed goal；bind_goal逐字节保持snapshot、goal_pose与renderer。native默认goal_pose不是constructed task goal。

因此新输入明确用warmup10，不用时间相同但goal绑定之后的plan0替代。on历史是原native默认轮廓的真实可见帧；constructed task goal_rgb则来自独立原witness terminal。不得重画历史overlay、改变goal_pose、把默认轮廓解释为constructed goal，或把任何goal/terminal图像送进predictor。目标只在后续标签评分中使用。本runner不另造goal metric、不改normalizer/Q。

继承冻结桥prepare的全量绑定及full physical/observability校验，再从原physical report取同164原分支对应真实warmup0/5/10与plan5/10。无文件名猜身份、缺省proprio填充或反推速度。native7 state bits与snapshot agent position/velocity逐字节核对。缺时刻、重复时刻、overlay/selection不符均拒绝。

## 固定开放滚动

原raw20=真实warmup10+原分支own10。原float32 warmup前5 [.1,0]、后5 [0,.1]，不能零padding。所有20 raw动作按冻结JEPA数据投影 `(100*u64).astype(float32)/float32(100)` 和原mean/std归一化一次，得到四个10维packet；无新拟合。

| 调用 | visual三帧 | proprio feature三帧 | action三packet |
|---|---|---|---|
| predictor15 | actual0/5/10 | 同期原生proprio经原encoder | warm0:5,warm5:10,own0:5 |
| predictor20 | actual5/10,predicted15 | 原5/10,预测15的完整256patch proprio features | warm5:10,own0:5,own5:10 |

原native AdaLN输出 `(visual[1,3,256,384], None, proprio[1,3,256,16])`；直接滚动其最后token，第二次不调用proprio_encoder、不重编码predicted feature，不取实际15，不采用compute_new_pose，不增加normalize_reps或新normalizer。这里复用冻结v8实际loader和direct predictor数学路径，借用原wrapper的predict_proprio滚动语义；不声称逐位复现wrapper中其他预处理/零padding逻辑或官方论文完整协议。绑定参考wrapper SHA `105a6768050e4b34c95de8d7d0cc0c7ba07b69d364acf0649ba2625bacce907b`。

全164×2实际full3输出、所有module输入/输出、model不变收据完成后，写 `predictions_seal.json`。seal含328个唯一路径的原float32完整full3输出，逐项index/step/SHA/shape核后，才解码真实15/20 RGB并批量2帧编码。预先读取标签路径/hash/authority不等于解码标签或向predictor提供未来信息。

这里“不读实际15”严格指预测数据流：不解码实际15 RGB、不编码实际15标签、不将实际15 state/proprio/features送入预测。为重建冻结上游authority，prepare/loader仍会读取原完整report及H1桥中的15元数据（不会解码其图像），并核SHA；不声称文件级完全未打开含15记录的报告。predictor实际输入树独立剥离label与未来元数据，完整saved strict逐字节验证，区分+0/-0，不以allclose代替。

每行分别保存15预测误差、20预测误差、以原10作持久性基线的两误差；都是原patch float64全量C-order MSE，不改变Q。全部80pair固定输出终态20 plus-minus误差差、预测/观测响应L2与cosine（零响应null）。off/on保留在pair身份中，不挑根/尺度/响应或winner。

## 事前工程预算与实际计量

| top module | API | batch samples | frame samples |
|---|---:|---:|---:|
| official encoder |328|820|820|
| proprio encoder |164|164|492|
| predictor |328|328|984|

顺序固定164×(encoder3,proprio3,predictor15,predictor20)，seal后164×encoder2。共820个top-module实际dispatch；不把API与内部module样本相加。每个调用透明pre/post hook分别计attempted/returned，先durable intent、完整IO、真实return、完整output；异常不退计数、不重试、不补跑。所有付费编码保留，不用缓存省略重复帧。原physical/LeWM/桥source成本仅继承，不计为本次新成本或声称免费。

完整float32 IO磁盘预算上限4GiB（实际bytes单列，不是GPU激活上界），输出总cap6GiB，metadata32MiB/文件、NPY4MiB/张、final64MiB reserve、disk floor200MiB，初始CUDA free4GiB预检。full3预测每张1,179,648 tensor bytes，328张386,924,544 bytes，另加全部IO；上限不足直接失败，不降精度/缩分母。每个数组只保存一次独立NPY，不压缩/截断。job900s覆盖创建Store起至dispatch/最终source与model检查；失败封存/终端fsync可能在界外完成，超时为工程失败。

真实CUDA peak allocated/reserved/free/total和hostRSS单列。backend初始化失败时CUDA memory=null，不能伪填0；partial仅允许完整性回执，不认证预测。旧loader绑定原权重文件SHA、严格schema和expected model receipts，并前/中/后比较模型state；这是沿用其加载证据，不新增同buffer loaded-bytes或训练来源认证。

## 文件与saved-only strict

prepare绑定冻结v1桥source SHA `d70352f2fc574ca8a226522fc3dda9f015305894c83a1d1826f05808c005da0f`、真实桥prepare及其所有bindings、新三文件、native rollover参考源。prepare/verify不import torch/env，不解码实际15/20标签。

report目录为新stem独占：execution_binding.json、journal.jsonl、唯一tensor_NNNNN.npy、predictions_seal.json、report.json。JSON与JSONL统一compact canonical exactly-one-LF，journal含完整hashchain。partial保留164rows/80pairs及真实失败，不作为科学负结论。

consume独立验证外部report SHA、完整prepare重建、精确目录inventory/bytes、所有IO唯一顺序、journal chronology、实际call计数、模型与源绑定、seal恰为328个实际module输出；重算第二次visual/proprio/action输入必须恰等于第一次预测滚动，无actual15接缝；独立复算所有误差/80pair。严格预处理reference为float32 reciprocal255及原mean/std，与runtime原transform的CUDA逐位一致性需MAIN真实执行验证；若失败保留工程失败，不许临时放宽容差。无需重复encoder/预测做consume。

## CLI（MAIN审读后执行）

从 `le-wm-main`，现有桥prepare SHA已只读核实为 `1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f`。下方NEW必须尚不存在，SHA使用每步真实输出，不填未来值。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_raw164_jepa_h2_cuda_20261029_v2.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_h2_cuda_20261029_v2 prepare --bridge-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_input_bridge_20261029_v1_prepare_MAIN01.json --bridge-prepare-sha 1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_NEW.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_h2_cuda_20261029_v2 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_NEW.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_h2_cuda_20261029_v2 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_NEW.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_NEW --authorization MAIN_P5_RAW164_JEPA_OPEN_LOOP_H2_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_h2_cuda_20261029_v2 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_NEW --report-sha ACTUAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_NEW.json
```

仅MAIN可实际HOST CUDA；禁止CPU fallback，需无PYTHONPATH overlay的新进程及原本地权重，无网络下载。成功status是工程观测记录 `fixed164_root10_open_loop_H2_recorded`；strict为 `all164_open_loop_H2_and80_pairs_verified`，都不是science admission。测试含真实Store/NPY/journal与328输出seal reader、第二次visual/proprio未来替换拒绝、signed-zero逐位接线；完整820-call编排使用明确RAM模型协议double，不是真实CUDA或科学结果。真实saved build smoke已核164请求/80pairs/20837bindings，prepare约8.67MB，未持久化prepare且未import torch/物理库。未运行生产模型/物理；无生产report或receipt SHA。
