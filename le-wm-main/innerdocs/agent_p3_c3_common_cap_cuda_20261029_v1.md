# C3 common-cap native tree v1：RAM接缝，bytes前置未通过

仅NEW独占source/tests/doc；批准协议`agent_p3_c3_common_cap_scheduler_protocol_20261029_v1.md` SHA `0ea4114c0bc88f780f2a788644ddc24851e7ba32e129029350cc5c98bad2255d`保持不变。不接secant/goalbank，不改progress/spectral8/原资格/原policy阈值。worker仅RAM/native-buffer fixture与既有保存文件的shape ABI检查，未生产prepare、NN、physics或fullstrict。

## 固定12槽与真实选择

顺序root0、root1；每根K1、K17；每K内fixedH1、staticH3_progress、adaptive。12槽各144原请求identity，总分母1728；原task/source来自冻结tree.build原known2 **root context，不是actual-cp5**。
fixedH1直接调用原core.choose(fixed)；staticH3_progress调用原fixed后只显式改变horizon=3和decision_source，保留width4/beam4/depth3/类型，是新名静态反事实，不冒充原policy；adaptive完整调用原choose，width4/8、depth1/3、beam8及R均照实执行。
每个完整model-parent付原H1/H3信号，只有选定horizon生成候选。不是双面板全枚举，不把candidate成本用于选择horizon。child用完整新3latent/2token/raw链和真实parent J；原root与local task映射仍保存。composition是已存edge真实串接，不把model预测child叫真实观测反馈。
R复用冻结tree.evaluate_R和spectral8：独立paid zero+全8，post-H1 J/原ridge/suffix诊断，严格immediate>zero且reach<zero；全部eligible进原(modelQ,depth,indices)beam，no_witness停branch，无fallback。H3是原30维新variant，不签原R等价。suffix不拼edge。

## 全shape与计费证明

`proof()`/CLI `proof`仅整数/结构计算，不读生产权威、不import torch。shape为latent3×192、between2×10、parent J192×10与192×30、post J192×10、edge最多15×2、suffix5×2、最终raw最多45×2。
本次只读既有原tree journal的前65行，确见action_encoder输入1×3×10/输出1×3×192，predict与predictor输入/输出1×3×192，pred_proj输入/输出3×192，grad basis/output192、raw10，与非神经fullshape fixture一致；这不是重建原fullstrict、读取reward或据效果选配置。
原信号收费含独立nominal与AD forward。`signal_cost()`=8predict/2J/384VJP/2SVD/1schedule；R3完整group含bank=54predict/9J/18replay/10SVD/9solve/10programcompile/18metric。逐27轴支配任意原progress(h=1/3,w=4/8)与R1。
原beam最多8，最多17parents=1+8+8；最多136外请求≤144。最长最终whole+segmented为18predict/2replay。B(K)按批准公式构造，schedule轴两层均17，不将K1强行限制成一个parent。

|上限|K1|K17|
|---|---:|---:|
|predict API/sample；AE/predictor/pred_proj单行轴|80|1072|
|context token/投影行轴|240|3216|
|J / signal_compile|11|187|
|VJP / grad / rawback各|2112|35904|
|SVD / solve / programcompile|12 / 9 / 10|204 / 153 / 170|
|replay / metric / schedule|20 / 18 / 17|308 / 306 / 17|

每K三policy共用同向量；samecap不等sameactual/wall/FLOPs。AD-primal和J元素从完整returned signal transcript的真实H派生；失败未返回signal仅保留实际ledger前缀，不给未知完整J填0。所有原始27轴、失败attempt/return、native数组仍完整保存。scalar_backward_calls/total_FLOPs为null，不从API估算。
派发前原子核完整signal pair+schedule；得到原decision后核完整4/8队列/R0+8以及当前全链最终验证reserve。cap/refusal停止整tree，保留diagnostic_best但model_plan=null；no_witness仅停该branch。全部槽继续，除runtime异常按原合同阻断后续。原depth1 child stop照实保留，不强制跑满三层。
最终验证按最佳实际raw链收费，可能短于最长reserve；不补到18、不因剩余预算多强选更深计划。原8R队列不是progress复制；所有已付R成本保留，包括全ineligible。

## 工程cap、事前bytes census与生产阻断

MAIN固定4GiB总新output（含512MiB finalreserve）、whole7200s/slot900s、CUDA allocated3GiB/reserved4GiB、diskfloor200MiB。无扩cap/退款/重试/删失败。
**NN轴的静态上界已证明；当前序列化保守上界未fit，禁止生产**。`byte_census()`输出完整12槽、全shape及重复保存费用；`proof.serialized_output_worstcase_fit_proved=false`不再仅作为警告：`run`在读prepared、创建output、import torch/加载模型之前调用`require_output_fit()`硬拒绝，consume也不能签出通过receipt。不能以超限保留partial代替事前fit。12×900不等于允许10800，whole7200同时约束；最坏wall/RAM/CUDA峰值也未证明。

字节census明确分为数值shape解释与完整格式包络，二者不能混称tight证明：

- 每predict所有原packet/boundary、action_encoder/predict/predictor/pred_proj input/output合计保守≤8192数值元素；每VJP basis192+terminal192+raw30+gradient30+rawback30=474；每J另≤192×30+192=5952。按各K原cap计算，不按已见轨迹或fixture平均值估计。
- Python有限binary64 JSON token≤24 ASCII bytes、逗号另1；原blob保存≤8byte元素再base64（不把blob误当逐float JSON）。`numeric_JSON_and_blob_bytes_upper`同时覆盖两种表示，属于故意放大的单份解释量；K1=61,415,148 bytes，K17=959,909,272 bytes。不是实际测量、不是下界，也不是加入总包络的第二份费用。
- 完整native_call transcript在journal与slot.native_calls重复；R选择父signal另写最多17份完整capsule。shard内还保存jobs/maps/R banks/paths，不能假设这些metadata免费。最坏record数取32×(predict+VJP)+128×J+256×144+4096，覆盖原native/ledger事件、队列/映射及固定元数据，K1=112,512、K17=1,248,128。
- 字符串/path/source/task字面量未有更紧的逐字段证明，因此**不冒称已经完成tight shape-only bytes证明**。完整保守包络采用写前与读时共同约束：每完整event128MiB、每source capsule64MiB、每完整shard3.5GiB；三者分别乘上述记录数、17、1。jsonfloat、base64、重复transcript以及字面metadata均包含在包络内，未假定都是小常数。写前校验新加于本stem Store，不改封存writer。
- 2roots×3policy×(K1+K17)总live包络=**1,095,790,843,920,384 bytes**，明显超过批准的**3,758,096,384 bytes**；final metadata(binding/report/completion)合计另限制512MiB，总上界1,095,791,380,791,296 bytes。很松的包络不说明实际会产生PB输出；它只说明当前实现**无法给出所需事前fit证书**。不能用这个上界宣称现有物理预算必然不足，更不能擅自扩容。

未闭合项：需要在同科学/工程cap内，进一步逐字段证明或无损去重native transcript/ledger/source metadata序列化，再重做census和strict。当前未擅自新增表示重构slice，不缩12槽、费用、保存证据或gate。MAIN只能审读/RAM，不能按下面CLI启动当前HOST版本。
避免旧inline报告/arm_return重复整棵树：各槽完整结果排他写`slot_00.json`..`slot_11.json`，顶层report和arm_return只存name/SHA/status/身份；没有丢native IO或省费用。events仍完整逐项链式fsync；R父signal在调用bank前写原`signal_s{slot}_{call}.json`source capsule。
总report/completion/binding合计保留512MiB收尾空间；live写入始终扣留该reserve。单槽JSON读取/写入上限取全live allowance=3.5GiB，不另缩科学预算；event单行128MiB、source capsule64MiB是有界解析合同。所有文件合计受4GiB硬限。这些是存储安全约束，不是fit证书；运行入口仍因上述证明失败拒绝。无法写完整槽/事件时仍保留已写journal/capsule/12槽prefix，错误不伪装settled；不保证极端I/O故障后每个未完成对象能完整落盘。
`on_start(rows)`在首event前公开同一个12槽可变prefix，HOST finally保留已发生的槽状态；不是异常后换成12个未启动来掩盖费用。输出全byte census包含binding/events、12槽shard和实际所有source capsules；unknown/多余文件拒绝。新schema不重标旧receipt。

## report持久化后的明确完成边界

`persist_completion`先完整write/flush/fsync report，再fsync目录；该调用返回之后取clock，独立排他保存`completion.json`。其schema `.completion`绑定reportSHA、journal tail及每个其他artifact的SHA/实际bytes完整census；`wall_seconds_through_report_fsync`必须≥旧report.wall且≤7200。strict检查完整目录只能是report.artifact清单+report+completion，并逐项重算completion的hash/size与边界声明。completion本身SHA加入receipt，不能沿用仅旧report.wall的验证。

精确边界是`report_file_and_directory_fsync_returned`，不是“所有落盘/进程完成”：completion自己的序列化/write/fsync和后续shutdown**显式排除**。有限文件不能记载在自身最后fsync返回后才取得的时钟值，不伪造这种递归证明。completion持久化返回后还做运行态总wall检查，超时抛错、保留文件、不能成功返回；该最后检查不是独立保存的时间证明。RAM覆盖report落盘后取clock及completion自身落盘超时，未把report fsync漏在已声明的wall边界之外。

## strict ABI

NAME=`agent_p3_c3_common_cap_cuda_20261029_v1`，有.prepare/.report/.tree/.consume。完整报告status=`twelve_slots_settled_not_admitted`，可含合法cap/refusal/no_witness；完整consume status=`saved12_common_cap_engineering_verified`。production字段是真实`full_C3_demonstrated=false`，不接受C3别名；P1/P3/physical/freshness/原R等价和fullpolicy科学声明仍false。
strict逐位重建原choose/static声明、完整queues/requests/banks、J parent双身份、所有native buffers、R资格/全部eligible beam、whole/segmented、27轴及嵌套失败。所有call transcript与journal一一相等；noncall事件也逐序列相等，unknown/orphan/漏/重复拒绝。所有slot source capsule实际字节/digest/路径使用完且无悬空，slot身份与shard SHA精确绑定。模型/installed class/原精度/source前后闭包不变。
prepare/verify复用旧tree.build的保存task/source发现，MAIN可执行；worker测试不调用。consumer不重新运行神经，只重放saved调用与调度/NumPy/SVD/ridge算术；这仍是生产fullstrict，worker未对真实artifact执行。

## MAIN CLI（当前HOST运行与成功receipt明确被bytes前置拒绝）

项目根`/home/gbk/code/LeWorldModel/le-wm-main`。prepare/report实际SHA替换P/R；所有output必须新独占、不存在。不预填未来SHA。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py
M=experiments.spectral_reachability.agent_p3_c3_common_cap_cuda_20261029_v1
.venv/bin/python -m "$M" proof
.venv/bin/python -m "$M" prepare --output results/agent_p3_c3_common_cap_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m "$M" verify --prepared results/agent_p3_c3_common_cap_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha "$P"
.venv/bin/python -m "$M" run --prepared results/agent_p3_c3_common_cap_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha "$P" --output results/agent_p3_c3_common_cap_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m "$M" consume --prepared results/agent_p3_c3_common_cap_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha "$P" --output results/agent_p3_c3_common_cap_cuda_20261029_v1_cuda_MAIN01 --report-sha "$R" --receipt results/agent_p3_c3_common_cap_cuda_20261029_v1_consume_MAIN01.json
```

## RAM验收及研究边界

覆盖真实fullshape callbacks/原choose/static显式差异；K1/K17同27轴、R H1/H3全9项、真实post J/原完整3latent/2tokens、所有eligible不只winner、depth1停止；static三层与不生成未选候选面板；12槽1728identity完整execute→saved重建，合法no_witness/cap/正常路径；native failure费用不退、后续槽阻断；事件/J/raw/资格/cap/身份/sourceflag篡改；文件census/SHA/hashchain/旧C3别名拒绝；冷import/CLI/worker授权拒绝。fixture不是神经导数或性能证据。
新增bytes census完整分母/重复计量、模型加载前拒绝、写前event包络；completion实际持久化顺序、post-report边界、wall超限、census/bytes/hash/旧wall及虚称自身fsync覆盖等篡改拒绝。所有测试使用RAM/临时文件；未生产prepare、NN、物理或fullstrict。
首轮staticH3 fixture把最长reserve误当实际final费用，已修为按最佳链长度；生产计量未改、未放宽gate。另H3测试按真实spectral bank.mapping/树signal_mappings取native J，不假设goalbank的native_response同名字段存在。
API技能用于新schema、shard/public reader与明确partial边界；未改冻结依赖，不monkeypatch生产globals。结果只能是known2模型规划/资源证据；未观测horizon切换或budget激活时必须如实报告。不同长度modelQ不可直接称同终点物理收益；P1未过不能签P3。本版不发布物理动作、不建后代、不读取secant/physical outcome选配置。
