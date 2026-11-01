# Latest-slot factorial: lock actual5/10, cross only15 embeddings

本独占NEW source/tests/doc不改任何H1/H2、整history factorial、summary或结果。worker只写实现与RAM/临时文件tests，不启动神经/physics，不执行生产consume。
goal API仍是旧blocked目标，未覆盖/假complete；本次交付完成只指实现、测试、freeze，不是研究准入。

## 前瞻固定合同

这是在已见整history factorial结果后授权的探索性后续，不是独立确认，也不按前期反向与否挑root/h/overlay/坐标。完整原164分支与80pair永久保留，含全部原零响应，不选择新cohort或调阈值。
沿同原H1/H2实际prepare/report/独立strict及完整source/model/normalizer/artifact/journal闭包，来源与固定SHA直接沿已封存合同。原实际H1为actual5/10/15→20，H2第二次predictor history为actual5/10+rolled15→20。二者3token action完整逐位一致：raw6..10、11..15、16..20；本任务不改任何动作、goal、label或source轨迹。
不以整history factorial结果作为选择输入；本任务直接从原H1/H2真实native predictor输入数组取值。

| cell | visual槽5/10 | proprio槽5/10 | visual槽15 | proprio槽15 | 执行 |
|---|---|---|---|---|---|
| AA | AA原实际字节 | AA原实际字节 | AA实际 | AA实际 | 复用原H1 full3，不新增 |
| AP | AA原实际字节 | AA原实际字节 | AA实际 | H2 rolled | 新一次/branch |
| PA | AA原实际字节 | AA原实际字节 | H2 rolled | AA实际 | 新一次/branch |
| PP | AA原实际字节 | AA原实际字节 | H2 rolled | H2 rolled | 新一次/branch |

原整history PP不能当新PP；原H2第二输出在prepare只留`H2_second_output_provenance_only`指针，不参与四cell评分。新PP必须真实新predict。
输入shape：visual `(1,3,1,16,16,384)`、actions `(1,3,10)`、proprio `(1,3,256,16)`；时间轴axis1，前两槽0/1锁定，slot2为15。用float32切片concatenate做无算术拼接；保留负零等原位模式，不重新encode或normalize，不将proprio embedding当成raw state。
source_blocks只读原native输入；splice_inputs检查完整shape/dtype/finite/actions exact。runtime预测前`history_ready.latest_slot_witness`保存前两槽、最后槽与完整动作的byte SHA；原生hooks保存真正传给模型的full3输入/输出。strict不仅核准备数组，也从实际native输入NPY逐位确认前两visual/proprio槽来自AA且slot15来自指定源，拒绝整history替换或任何ULP漂移。

## 顺序、度量与边界

固定branch-major `latest_AP`→`latest_PA`→`latest_PP`，164×3=492次；不重试、不追加AA replay或padding。全部492实际full3输出及model_after_predictions先落盘seal；核完seal与全部预测后才`labels_unlocked`并读取标签特征/原AA输出。
所有四cell共同使用原H1 actual20 label，不使用各自batch标签。H2 label仅记录bitwise/maxabs接缝，不能混用解释四cell差异。full3 visual/proprio输出全存，评分仅final20 token；float64原MSE/三效应/80对response cosine算术保持原合同，不fit新Q/readout。

```
visual = ((PA-AA)+(PP-AP))/2
proprio = ((AP-AA)+(PP-PA))/2
interaction = PP-PA-AP+AA
```

0encoder/AD/training/physics，extra原生agent position/velocity权限仍披露，不是RGB-only或跨架构公平排名。此时干预确实只变模型输入第15时间槽，但rolled embedding本身保留原预测/数值历史；不是像素级15帧干预、纯horizon干预或物理可达反事实。不能称已修复、证明原因或控制收益；P1/P5 gates均false。

## 独立工程预算与原生接线

事前固定：predictor492 API=492batch samples=1476token-frame samples；official_encoder与proprio_encoder API/sample/frame均0。仍加载原完整冻结模型用于actual state receipt，但不forward两encoder，真实top-module hooks禁止其调用。原H1/H2付费成本单列inherited_costs，不宣称等FLOPs。
每call完整IO有效tensor2457720B，492call合计1209198240B（约1.126GiB）；IO cap1.5GiB，输出cap3GiB（NPY headers/journal/report在内），terminal reserve64MiB，metadata单文件32MiB，NPY单文件4MiB；disk floor200MiB、开始须cap+floor，初始CUDA free4GiB。wall900s覆盖verify/load/dispatch/score/final rehash；失败durable fsync清理可超出收尾边界。真实CUDA peak allocated/reserved/free/total、host RSS及attempt/return轴单列。
HOSTCUDA offline only，原已审loader、frozen eval/inference、TF32off/MATH；模型actual loaded bytes在预测前/后/最终与原H1/H2 receipts逐位绑定，所有source/权重/inputs输出SHA最终复核。代码从已封存接线进行独立版本化，不patch globals，不调用旧execute冒充新schema。
Store使用既有已审纯codec方法与独立所有权/cap，canonical JSON exactly1LF，journal全hashchain、每次intent/return、fullIO durable fsync；发生失败保全部164/80及未完成槽，partial不是科学负。strict只对完整pipeline返回成功。

## MAIN CLI与ABI

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_latest_slot_factorial_cuda_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_slot_factorial_cuda_20261029_v1 prepare --output results/agent_p5_latest_slot_factorial_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_slot_factorial_cuda_20261029_v1 verify --prepared results/agent_p5_latest_slot_factorial_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_slot_factorial_cuda_20261029_v1 run --prepared results/agent_p5_latest_slot_factorial_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_p5_latest_slot_factorial_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_P5_LATEST_SLOT_FACTORIAL_HOSTCUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_slot_factorial_cuda_20261029_v1 consume --output results/agent_p5_latest_slot_factorial_cuda_20261029_v1_cuda_MAIN01 --report-sha REPORT_SHA --receipt results/agent_p5_latest_slot_factorial_cuda_20261029_v1_consume_MAIN01.json
```

prepare/verify为完整真实可运行闭包重建，不输出未来receipt或SHA。成功run status=`all164_latest_slot_factorial_recorded`；strict=`all164_latest_slot_factorial_and80_pairs_verified`，带`actual_native_AA_prefix_bitwise_verified=true`；partial status=`blocked_partial_capture`且CLI exit2。旧整history consumer/summary不适用本新schema，不能骗接。
report布局：`rows[164].predictions`恰好三个new arm、score含AA/AP/PA/PP共同标签MSE/三效应/接缝；`calls[492]`真实native full3 IO树；`counts.attempted/returned`三模块API/sample/frame；`pairs[80]`完整response/cosine/效应；prediction_seal、model receipts、strict_loads、imported_sources、memory、artifacts、journal_chain_tail。`history_ready`每call均有latest_slot_witness，其source时间与native byte linkage由strict完整重建。
前置是原H1/H2封存目录/prepare/strict与原模型缓存原样可用、MAIN代码审读、授权HOSTCUDA。没有新增实验结果；tests的synthetic模型接口替身仅验证工程，不能充当研究证据。

## 工程验证

本stem55 tests passed in27.20s；首轮测试保留了旧两角末槽断言而失败，已仅改成第三角末槽，未改实验算术。测试覆盖真实Store两event→canonical/hashchain、NPY EOF、shape/time/root/task/selection/condition/raw/action错接拒绝、三个新角的前两槽锁定/最后槽来源、负零保留、任意一ULP输入漂移拒绝、encoder禁调用与真实attempt/return计账、seal前标签屏障及partial完整164/80。
全尺寸合成端到端无torch/模型，真实writer写492次完整native IO（1209198240有效tensor bytes），run→seal→score→strict并拒绝输入关联篡改。仅pytest临时文件，不是生产结果；测试建议预留约1.5GiB临时磁盘。MAIN仍须独立审读与实际HOSTCUDA/strict。
