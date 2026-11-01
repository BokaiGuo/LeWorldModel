# Public native JEPA context action-response：MAIN待执行

仅本stem source/tests/本文，保留native v1及其他旧产物。使用experiment-code技能固定预算、原值日志和RAM失败回归；不做训练/拟合/adapter/repair/新normalizer，不运行模型、physics或网络。RAM fixtures不是科学证据。

## 协议与解释边界

MAIN已发布七NPY public payload，实际FILE SHA固定为
`fdc063a6475577071fb74cf4ccd0853c4b2e4e5c061693d73efbc437686ffec3`。
本程序不接受private lineage、物理report/strict、block pose、contacts、参考suffix或source生成器参数作为输入；不读取这些私域文件来设计或筛选动作。只接受该已发布payload及固定模型资产。数值/public身份认证依赖MAIN提供的外部实际SHA；本模块不重新签发源证书。

按已封存action_words的四程序、顺序、幅度原样执行：XY（五次[.1,0]后五次[0,.1]）、YX、X−X、十个raw-zero。全四分母，不按结果筛选、追加、替换、重试；不优化目标。每程序raw float64[2,5,2]，relative原生尺度100，两个5步packet。这个构造OOD校准context不是fresh，不是新checkpoint/独立架构、任务成功或D2P贡献。

模型仅使用public RGB0/5/10、原生agent proprio[x,y,vx,vy]、两包实际prefix动作；不使用未来RGB或suffix状态。七NPY均验证SHA/大小/dtype/shape，public目录必须恰好七项；goal RGB只绑定，不编码、不评分。proprio采用原训练source统计float32 `(raw-mean)/std`，不是raw直接进入embedding，也不refit；动作严格继承owner的`(100*u64).astype(float32)/100`后原统计标准化。原raw-zero历史不能换成normalized-zero。

原生接口：官方源码DINO-small patch384 + AdaLN predictor + proprio16；严格原v8 CUDA loader，继承已认证cached-HF映射的来源限制，不改为新官方checkpoint。原双递推复用sealed owner.rollout：独立复制相同三帧context与两包history，每次新packet拼第三包，原预测visual末帧和patchwise proprio末帧同时append，history同步shift；不做teacher forcing、不共享分支预测、不裁剪/投影预测特征。两包之后只计算原patch空间XY−YX及X−X−zero的float64 MSE；这些量不是block位置/接触/物理非交换性，不能将agent惯性或feature响应说成物体操作能力。没有latent→physics拟合。

## 固定预算与失败

|资源轴|API calls|forward samples|处理context frames|
|---|---:|---:|---:|
|official RGB encoder，共享一次|1|3（batch3独立图像）|3|
|proprio encoder，共享一次|1|1（batch1三时刻序列）|3|
|predictor，四程序各2|8|8（batch1）|24|

predictor raw horizon总40步只是计划，不执行任何physics；不把这些步加到模型API/sample成本。无额外warmup/padding/replay；预算scope包含所有context编码和预测attempt，包括失败。batch和shape在真实pre-hook核验；Meter before先计attempt再放行，post-hook记录实际return，失败不退款，未返回时禁止再次dispatch。共享context只记一次，没有重复退款。AD/backward/JVP/VJP/SVD/physics均0；墙钟实测独立轴，FLOPs=null，不称equal_total_compute，也不把samples/API相加。

工程cap：64MiB输出（写前检查，含1MiB terminal reserve）、200MiB磁盘floor、600s wall+SIGALRM、4GiB CUDA free scratch，非峰值显存保证；C/CUDA执行可能延迟Python信号。所有输出exclusive create，不能恢复重试或覆盖。严格消费者按共享3项与每程序两步五数组完整检查，实际共43个NPY；共享RGB/proprio特征及归一化proprio保存，四分支每次的visual/proprio/actions输入和visual/proprio输出全保留。attempt/return和每artifact持久SHA事件链；任何失败保留四个word状态、已耗meter、原始与cleanup错误，消费者不接受partial，不以缺失填0。finally移除所有owned hooks并核模型状态不变、source/public字节未变。

## 私域/公域部署

prepare仅从旧pinned v12 manifest/report提取**模型**authority与训练source统计，不读取本次新物理outcome。v12 owner manifest使用compact digest；本模块自身继承owner.digest的默认空格语义，不能混用。投影后的单值MODEL_SPEC_SHA为
`a4757f79a4a3e699e239892bdad0273f162a6f38ef61c3df96c0e9f7a9188cb5`。
投影只保留严格加载需要的三项schema/model/weight-mapping、原外部Python/YAML源码、loader helper Python与两份权重文件；246个seal，没有旧factual arrays、数据集视频、pickle统计或results路径。prepare仅hash权重字节，绝不解pickle/加载权重。实际权重加载仅显式HOST CUDA入口、原loader weights_only/CUDA。

准备好的部署manifest不携带新源seed/geometry/contact/suffix/witness/私域路径，不携带旧factual roots/目标特征。verify/cuda只验部署manifest、public文件及投影的模型/source闭包，不重读旧v12报告，也不调用private native consumer。MAIN应在部署时只授予public目录+payload、本次部署manifest、源码/依赖和冻结模型资产；不挂载新source私域目录、private publication sibling或参考结果。Python allowlist不是OS沙箱；OS隔离由MAIN负责。

所有scientific_admission/P5_admission/action_witness_established/physical_release恒false，task_success=null。即使后续严格saved消费通过，也只表示新context下四程序的真实CUDA因果预测轨迹完整、模型未变，不是新科学admission；缺独立freshness/P1不自动放开。本轮不重跑已有fidelity/JVP、不生成新的物理证据。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。先审读并RAM：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_contact_action_response_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_response_20261028_v1 prepare --payload results/agent_p5_contact_source_native_20261028_v1_public_MAIN01_payload.json --payload-sha256 fdc063a6475577071fb74cf4ccd0853c4b2e4e5c061693d73efbc437686ffec3 --manifest results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json
sha256sum results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json
```

MAIN填上一步实际FILE SHA，审查批准后唯一HOST CUDA入口：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_response_20261028_v1 cuda --manifest results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --confirm-host-cuda --output results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01
sha256sum results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_response_20261028_v1 consume --manifest results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --report results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json --report-sha256 <ACTUAL_REPORT_SHA> --output results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json
```

verify子命令只需manifest及实际FILE SHA；无模型调用。未写未来prepare/report SHA。worker只运行RAM测试与CLI help；MAIN的实际prepare、CUDA、strict待执行。RAM用缩小维度constant token recorder验证原因果数组接线、failure不退款/批量计量、payload越域、metadata/normalizer/模型/word/预算篡改和重新封hash后的feedback mutation；不是fakecounter科学证据。真实saved回归只读已存在的模型authority JSON和公开七NPY，无私域物理outcome或模型读取。

交付回归范围：新模块55项（含record-only hook成功、encoder失败、predictor失败及finally清理），与旧native24/contract25合并104项。最终实跑结果和三文件SHA以冻结交付回复为准；初轮小维度fixture暴露动作输入30元素上限遗漏，已只在本新stem修正，未改任何封存owner/native/contract。终端报告在保留的1MiB内独立检查size/floor；即使普通wall guard已失败仍尝试落盘，若磁盘/持久化终态也失败，则保留partial文件，不能取得strict回执。
