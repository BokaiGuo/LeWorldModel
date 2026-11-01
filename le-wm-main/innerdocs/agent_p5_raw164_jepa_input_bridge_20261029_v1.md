# Original164 native -> JEPA saved input bridge

独占source/tests/innerdoc；只准备输入，不冒充已经完成CUDA实验或完整run consumer。MAIN可在冻结后执行prepare/verify。本worker无torch导入、神经、物理、下载、训练或拟合。不是fresh样本，不是同预算matched收益，不打开P1/P5。

## 已核实的原始输入与时间

绑定原physical report `c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10`、原prepare、full receipt及observability四固定SHA（沿原factor.PINS）。全164plans按四panel固定顺序root0/off、root0/on、root1/off、root1/on，每panel41；完整原80 signed-pair schedule保留。不读取正在生产的capturev2，不改或消费v1partial。

每个请求是factual H1，三历史全局5/10/15分别取原warmup5/plan0/plan5的原PNG；真实successor是plan10即全局20。不把raw一步当模型packet，不拿goalRGB代替successor。三动作token各含5条xy：warmup原后5、plan原前5、plan原后5，完整非零历史不补零。on用原overlay_index1，off用0，重复off2不另挑选。

每帧原native7与snapshot state逐字节核对；`snapshot.bodies.agent.position+velocity`必须逐字节等于state[[0,1,5,6]]，这是实际保存agent x/y/vx/vy，不是轨迹差分推断。只向模型提供三历史时刻的agent proprio，物体姿态、goal、terminal proprio不进predictor。这是相对LeWM RGB-only的额外权限，必须分栏披露。缺速度/图像/历史/原支身份即prepare失败，禁止填0/插帧/替根/少算分母。

原physical full receipt是既有authority；本prepare重新核完整绑定字节、原journal文件SHA、原rows/plan结构和execution identity，不声称本轮重跑原journal strict消费或回证未封存loaded bytes。

## 不变normalizer与loader

原JEPA `pusht_dset.py`的四统计数组经AST仅解析原torch.tensor字面量，不执行torch或重新估计统计。动作沿v11公式：pixel32=(100*u64).astype(float32)，随后除float32(100)，再原float32 mean/std，最后time-major reshape(3,10)。native raw零不等于normalized零。不改Q、readout、动作或图像预处理。

真实可复用loader是 `p5_jepa_native_action_response_20261027_v8.load_cuda`，源SHA fc915ddee2e53ed58366994684fe2190da5eaaeb8323ad608ffe3292534c936a；其原parent manifest文件SHA 0588f825c29b884350e97ed0dc47d8872bb539d0ca7a7faf045190eee5f3340f。prepare绑定完整parent seals；不调用v8.prepare，因为其包含原episode1/JVP范围，不猴补或冒充原episode合同。

`load_host_cuda(prepared, expected_file_sha, 'MAIN_P5_RAW164_JEPA_HOSTCUDA_ONLY')`先完整verify再延迟import torch；拒绝PYTHONPATH overlay与无CUDA。直接调用原loader，严格冻结eval、原权重映射/expected_model_receipts/source closure、TF32off/MATH。返回原models、strict、sources、model_before和attention_context_factory；后续每次forward必须在该MATH context中。它不是一个已经计账/封存/评分的整合run命令。

## ABI与先封预测后标签

- `build()`返回完整prepare，含protocol、bindings、stats、requests[164]。每request保存index/root_slot/condition/selection_digest/task_id，三个history image收据、原proprio与normalized_proprio、原15commands与packets(3,10)，label收据独立保存。prepare读取label metadata和文件hash，但不解码label像素供预测。
- `host_inputs(q)`仅返回history RGB[3,224,224,3]、normalized_proprio[3,4]、packets[3,10]及index/root/condition；不返回label或goal。PNG mode/shape/文件及像素SHA严格，无resize、补图或conversion。
- `labels_after_seal(q, seal_path, seal_file_sha)`要求独立prediction_seal，prepared_semantic_sha256=digest(q)，predictions固定index0..163。每项path为实际完整原float32 NPY[1,3,256,384]及文件sha256；全部164均存在、有限、shape正确并核SHA后才解码任一label。不能只保存最终token或以声明completed替代文件。
- 该seal只核桥的顺序及实际数组前置，**不是**CUDA执行module ledger/loaded-buffer/源最终不变的完整consumer，不能据此签发科学或生产成功。最终实验尚需独立runtime负责真实pre/post hooks、模型/源前后核验、partial事件、全部164/80评价，且所有predictions先持久再进行label encoder与score。

建议后续MAIN审定runtime工程预算：不去重的简单实现为164×encoder(context3)+164×proprio_encoder+164×predictor，预测全封存后再164×encoder(label1)；encoder API328/sample656，proprio API164/sample492，predictor API164/sample492，0AD/physics/training。每份完整预测约1.18MiB，全部193,462,272 tensor bytes；上下文与标签patch约257,949,696 bytes，连同NPY/header和module IO须显式另算。不要照抄32MiB磁盘cap；当前32MiB只管prepare metadata。此建议不是已授权CUDA配额或已实现runtime计数，不执行额外action controls、不为凑预算padding。

特征MSE只能在各架构自身表示/固定condition内作诊断；不可把JEPA256×384与LeWM192的大小直接比较、称renderer充分因果或控制收益。保留原off/on与knownroot边界。

## CLI（inner根）

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_raw164_jepa_input_bridge_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_input_bridge_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_input_bridge_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_raw164_jepa_input_bridge_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_input_bridge_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_FILE_SHA
```

没有伪造`run/consume`命令：本次授权交付是saved输入adapter与合法loader入口准备。完整164 CUDA实验整合仍需MAIN下一步授权/审读，不能将旧v12 runner的三个seed换名为此164协议。

## 实际交付前检查

26项RAM/小文件测试通过（0.46s）。实际只读`build()`返回164requests/80pairs/20833bindings，compact单LF prepare 4,551,333 bytes，未写prepare文件；torch/pygame/pymunk/cv2/stable_worldmodel均未导入。原生agent velocity三历史时刻确有保存值，故本固定面板不存在该必需字段阻塞。没有运行loader/forward/环境；完整新CUDA module ledger、prediction seal生产与评分consumer仍是下一明确依赖，不把输入准备通过当预测或任务通过。
