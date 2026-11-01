# Stage C 固定十槽 cp15 RGB 桥

新独占三件套 stem：`agent_p3_same_time_cp15_rgb_bridge_20261029_v1`。不改冻结 physical v2/cp10桥/旧结果。不负责 Pasteur 的结果汇总或 native benefit 比较。worker 仅检查保存文件 ABI、RAM callbacks 和临时盘，不执行模型/physics/fullstrict。HOST CUDA 的运行与真实 consume 由 MAIN 审读后执行。

## 固定来源、位置和模型

原生 Stage C stem `agent_p3_same_time_capability_probe_physical_20261029_v2` 的以下三份 MAIN 实际权限域已到位，`actual_authorities()` 固定精确路径与 SHA。prepare CLI 仍必须显式传入，不能替换批次或填未来哈希。

| 原件（均在 `/home/gbk/code/LeWorldModel/le-wm-main/results/`） | SHA256 |
| --- | --- |
| `_prepare_MAIN01.json` | 8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f |
| `_MAIN01/batch.json` | b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b |
| `_consume_MAIN01.json` | 4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1 |

`build()` 对这三文件及十 job 的 report/completion、whole receipt row、原 prepare/source/plan、journal/PNG artifact SHA 作精确连接并封存闭包；不调用原 physical.consume/consume_job/reconstruct，不重读解析原 fullstrict。原事件完整认证继承 MAIN receipt，不能将此桥称独立重做物理认证。completion 的 journal SHA 来自实际 `artifact_sha256['journal.jsonl']`，不假设不存在的顶层字段。`source_authorities` 精确连接原 full-witness 和实际 Stage B 权限域，不用谱 summary 作 release。

五端点顺序固定 `(r0,C),(r0,Z),(r0,P),(r1,Z),(r1,P)`，各 U/Z2，共十槽。每槽选原 report 中 `phase=plan,step=15,overlay_index=0,condition=off` 的真实 RGB；cp15=public control15=absolute25。不选择其它 overlay、不换早停终态、不 dedup。实际 MAIN 本批十槽均 full15、81PNG、834events、guards=[51,66]、missing=[]；该事实只决定已存在的图是否可读，不用于重选动作/根/目标。

模型仍是冻结 P2/P3 原 runtime native192，不是 Hi。复用 `agent_p3_same_time_cp10_rgb_bridge_20261029_v1` 导出的 bridge/kernel 接点：`bridge.Trace`、`bridge.native.guard/Ledger`、`bridge.pixels.preprocess_pixels`、`old.model_inventory/precision_state`；原 `load_compatible_pretrained`、`preflight_pusht_d2p_p0.image_transform/_image_tensor`、`model.encode({'pixels':x})['emb']`。preprocess224、encoder `[1,257,192]` 全 token、CLS→projector `[1,192]`→external `[192]`，不归一化 latent、不重 encode goal/history。原 checkpoint 文件 SHA、原 source/code/class 闭包及模型张量 hash before/after 全绑定，无 monkeypatch globals。

Ledger 原底层以 `4*n` 为图容量，不能照抄旧桥的 n=2（最多8图）。本件使用 `max(1,ceil(available/4))`，实际10图时 n=3，再由本件 predispatch 精确守卫 `len(entries)<2*available`，不允许11/12图。缺图时同样只付 available 图，不把底层额外容量当授权。独立 fakeCUDA tensor RAM 测试完整走10 encoder+10 projector callbacks，并确认第21次在 callback 前拒绝；还覆盖0/1/5/8/9可用图。这里的“无 monkeypatch globals”指不替换冻结模块 protocol/run/consume 等全局依赖；原 `native.guard` 对 model/torch 调用的临时守卫与恢复仍按既有 kernel 运行，未另造替代实现。

## 固定工作与输出

完整本批实际请求10 encoder API/10 sample/10 frame，10 projector API/10 sample/10 frame。AE/predict/predictor/pred_proj/AD/physics/新求解/goal encoding 均0。API 与 sample/frame 为同一调用的不同资源轴，不相加计总 FLOPs；总 FLOPs=null。

每图七数组均完整保留：RGB uint8 `[224,224,3]`；preprocess f32 `[1,1,3,224,224]`；encoder input `[1,3,224,224]`；encoder output `[1,257,192]`；projector input/output 各 `[1,192]`；external latent `[192]`。十图70份 NPY.gz，独立 file SHA/array bytes SHA/dtype/shape。所有图及实际 module attempt/return 与 guard 双账本按时间封存；先 native seal，再派生 cp15 task。没有 Q scoring、排名、winner、threshold。

`tasks_for()` 从已认证 Stage A cp10 actual task 的 `[abs10,15,20]` 保留后两槽，再 append 新编码 abs25，得到 `[15,20,25]`；动作 history 是本槽 actual control6..15，float32 原 task mean/std 只做一次 sub/div，再 reshape `(2,10)`。保留原 goal/projection/normalizer 数值，仅更新 context 和新 namespace/task binding；不使用模型预测补 history。新 task 是下游工程输入，不授权新的搜索/部署/科学结论。

missing/transport failure 固定槽显式 null，不能伪编码。通用 schema 保留十槽并记录缺项；当前真实 pins 已锁定全十完整批次，不能拿其它缺失批次冒充。派生失败或 CUDA 错误不重试，不换配置，partial report/array 保留；complete consumer 不认证 partial 为成功。

## 预算与消费

whole wall600s；GPU allocated3GiB/reserved4GiB；output64MiB 含 final reserve8MiB，disk floor200MiB；所有 JSON 序列化副本总 quota16MiB，非单文件16MiB。census 对70数组按 generic `n+ceil(n/8)+ceil(n/64)+11+18` 计 gzip 上界，NPY实际固定128字节 header另计。压缩数组上界17,742,490 bytes，总数组＋metadata quota＋reserve为42,908,314 bytes，小于64MiB。不是压缩良好输入才成立的 tight compressBound。完整任务102个事件文件、70数组，加binding/roster/seal/tasks/report/completion共178文件。无静默扩cap。

`consume()` 是本新 RGB 桥的独立 saved consumer，不是原 physics fullstrict。它核外部 report/completion SHA、整个输出 inventory、原 prepare 重建、原文件/source/model闭包、before/after model/class/precision、GPU/墙钟/字节预算、全部 native/guard事件、七数组的实际模块连接、原像素/preprocessing、逐槽 history/task/binding。所有数组必须被消费，无 orphan。全十图 module IO 为8,010,240 bytes（每图801,024）。不计算 native 结果或预测残差，不与 Pasteur 重叠。

## RAM/临时盘测试及 MAIN 命令

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_same_time_cp15_rgb_bridge_20261029_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_cp15_rgb_bridge_20261029_v1 prepare --physical-prepare /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_prepare_MAIN01.json --physical-prepare-sha 8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f --physical-batch /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_MAIN01/batch.json --physical-batch-sha b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b --physical-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_consume_MAIN01.json --physical-receipt-sha 4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_prepare_MAIN01.json

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_cp15_rgb_bridge_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_prepare_MAIN01.json --prepare-sha "$P3_RGB_PREPARE_SHA"

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_cp15_rgb_bridge_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_prepare_MAIN01.json --prepare-sha "$P3_RGB_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_cuda_MAIN01 --authorization MAIN_P3_TEN_CP15_RGB_HOST_CUDA_ONLY

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_cp15_rgb_bridge_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_cuda_MAIN01 --report-sha "$P3_RGB_REPORT_SHA" --completion-sha "$P3_RGB_COMPLETION_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_consume_MAIN01.json
```

变量由 MAIN 用本件实际返回 SHA 填充；尚无本件 prepare/report/receipt，不生成或猜 future SHA。输出均排他创建，失败结果保留，MAIN 若重试需另命名。

测试读取明确命名原件的JSON/PNG ABI，使用 NumPy 假 tensor/module callbacks，不导入 torch、加载 weights 或实际 NN。临时盘 publish→own consume 测试只在 pytest 内替代 verify 权限边界，fixture 的 CUDA metadata 不是真实 CUDA 认证；生产 verify 不接受 fixture prepare。含全十/部分缺失/全缺失、完整数组/事件/费用重建、篡改、未知额外文件、hidden197拒绝、predict/AE/AD守卫。生产代码没有 fixture fallback。MAIN 尚需独立审读、实际prepare/verify、HOST CUDA、完整consume；RAM通过不能声称真实编码成功。known2非fresh，P1/P3科学门始终false。
