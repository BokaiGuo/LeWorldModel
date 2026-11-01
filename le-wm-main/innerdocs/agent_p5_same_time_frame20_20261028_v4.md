# P5 v4：固定四词 frame20 同时间诊断

仅独占新实现；不修改 renderer、encoder、analysis v3 或任何旧失败。MAIN 审读并执行 HOST CUDA；worker 只做 RAM/保存 metadata 验证。不是 P1/C2 准入、fresh-root 证据、任务成功或颜色机制论证。

## 事前合同与输入

固定顺序 xy、yx、x_minus_x、zero_zero，全四分母。原 contact_action_words physical 的 prefix10+suffix10，取各自真实 frame20：RGB uint8[224,224,3]、state float32[7]、proprio float32[4]、完整 transition/full_public_state。prepare 绑定 report a07c2583…、prepare 46f8cf63…、dispatch 全 plans 和原完整 artifact inventory 的文件 SHA/bytes；不是用文件名猜任意输入。frame_20 布局来自原封存 producer，row.arm/frame 与连续 0..20 records 显式核对。原构造器/seed/raw20 plans 全保留，不选择词或根。

goal 为已保存真实 native goal-scene RGB（来自旧 replay prepare c2ecf013… 的显式引用）；核 goal pose 与四 endpoint 相同。模型取 MAIN live encoder v2 prepare 6c0fa942… 的原 model_spec、loader receipts 和 source seals。冻结 owner v5 的 loader/encode/MSE 数学原样纯调用，不替换 globals。现有 v5 prepare 79c3c2f1… 的全词预测三 token 数组取最后 token，原模型、whole-word 实际端点绑定逐一相同；只读保存预测，无新 predict。

新编码五张，顺序四 frame20+goal native：official_encoder 各 API/sample/frame=5；proprio_encoder/predictor/AD/physics=0，无 warmup/retry/padding/训练/拟合/阈值变化。完整 float32 patch [256,384]，不是 192 维 moving 表示或 native7。GPU/预处理/MATH SDPA/严格模型载入沿原 v5。不是 encoder CPU fallback。

旧 live frame15 不能冒充 frame20。当前没有认证的 frame20 white 干预图，四个 missing 项固定保留，颜色效应 null。已保存 goal visible/white 与原 goal RGB 全字节相同在 prepare 重核，因此 first-stage 为 null，不解释成 encoder invariance；不补画、遮挡物体或重跑原生环境。旧 renderer 失败继续失败，新协议不出 renderer release。

## 输出与分析

所有五份完整 NPY fsync、all5_tokens_sealed 事件先落盘，再载入旧预测并计分。四行分别给 observed-to-goal MSE、predicted-to-goal MSE、prediction-to-observed MSE；float64 元素差平方均值，无掩码/投影。native 同 frame20 保存完整 reward/info、final coverage、done 及 state；不将 object-only 指标或 done 自动变成完整 PushT 成功，task_success=null。

固定 12 有向比较（6 无向对的双向），分别计算 observed latent/predicted latent 成本差和 coverage/reward 差；成本低优、native 高优；严格 float 数值正负与 exact tie，不事后选择容差或词。四词共享同一已知构造根，不是四独立样本，不检验显著性或泛化，不因排序宣布科学通过。新 v4 对比以前保存模型预测与这次同时间真实编码；不是新模型预测实验，原 v5 同时间端点证据也不抹除。

独占 output 下 5 个 *_patch.npy、event_0000..0015.json（五组 intent/return/artifact 后 seal）、report.json，总 22 文件。report 固定四行、12 比较和完整失败项；consume 重放事件 hashchain、实际计账、精确目录/NPY SHA/shape/dtype、模型前后/来源闭包，独立复算全部成本/比较，missing 及 goal-first-stage 从绑定 prepare 导出。

32MiB 输出 cap、200MiB disk floor、600s wall（不含调用前 saved verify；包含 load/encode/final verification）、至少 4GiB CUDA scratch。五个 patch NPY 约 1.88MiB，full native labels 与日志另计。输出目录必须不存在且为 results 下本 stem 子目录；失败 report 保留全五输入/全四科学分母，strict consume 不接受失败。全 bindings 包括原权重文件；它们是继承资产，不计成新 forward，亦不等于新的训练或新 loaded-bytes 测量。

## MAIN CLI

在 le-wm-main 根目录，先确认所有上面真实保存资产及依赖源仍在。prepare/verify 为 saved-only，不运行神经或物理；运行时必须原 HOST CUDA 环境、无 PYTHONPATH 物理覆盖、无 pymunk import。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_same_time_frame20_20261028_v4.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_same_time_frame20_20261028_v4 prepare --manifest results/agent_p5_same_time_frame20_20261028_v4_prepare_MAIN01.json
sha256sum results/agent_p5_same_time_frame20_20261028_v4_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_same_time_frame20_20261028_v4 verify --manifest results/agent_p5_same_time_frame20_20261028_v4_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p5_same_time_frame20_20261028_v4 run --manifest results/agent_p5_same_time_frame20_20261028_v4_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --output results/agent_p5_same_time_frame20_20261028_v4_cuda_MAIN01 --confirm-host-cuda
sha256sum results/agent_p5_same_time_frame20_20261028_v4_cuda_MAIN01/report.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_same_time_frame20_20261028_v4 consume --manifest results/agent_p5_same_time_frame20_20261028_v4_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --report results/agent_p5_same_time_frame20_20261028_v4_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA --output results/agent_p5_same_time_frame20_20261028_v4_consume_MAIN01.json
```

SHA 占位符必须替换为实际文件 SHA，不预填未来报告。RAM fixture 不构成 CUDA 成功。worker 不 launch CUDA/physics，不重跑旧 consumer 或写 MAIN 结果。准备阶段任何缺依赖/字节漂移均 fail closed；新 encoder 实际 report/strict receipt 仍待 MAIN，frame20 color 因缺材料仍 null。

## 实现交付验证

31 项 RAM 测试：固定顺序/全分母、精确 ties 与双向关系、原预测/observed/native 分栏、模块 predispatch 上限、假 hook 的临时目录全 run/consume（明确非 CUDA 证据）、失败保留全四 scores 和 missing cells、score/rank/meter/seal/NPY/多余文件/补造颜色变异拒绝。实际保存资产在内存执行 prepare()+verify() 已通过：5 inputs、1163 bindings、默认 JSON semantic 大约 403431 bytes（最终 source/test/doc 字节变动不改变此结构）；未写 MAIN prepare/output。没有运行实际 encoder、环境或 physics。最终交付 SHA 以后不得原地改三件套。
