# P5 raster first-stage 与 frame15 身份诊断

这是一次针对已保存结果的独立窄诊断，不重跑consumer、物理、CUDA或任何模型，不修改封存renderer/encoder/analysis。独占工具：`le-wm-main/experiments/spectral_reachability/agent_p5_raster_first_stage_diagnosis_20261028_v1.py`；纯数组tests同stem。文档不涉及Pasteur理论或MAIN P2/P3。

## 结论

1. **goal颜色干预在最终RGB上是null first stage。** visible/white原生RGBA确实分别为`[144,238,144,255]`/`[255,255,255,255]`，但两张实际224×224×3 NPY、无损PNG解码像素完全相同，差异像素0、maxabs0、pixel MSE0；全部256×384 float32 feature bytes也相同、MSE0。不是“改变了可见像素但encoder忽略了颜色”的证据，不能从该goal案例识别encoder颜色不变性。
2. **xy与x_minus_x在frame15相同是已执行共同动作前缀的预期结果，不是已证实的wrong-state路由。** 原计划前10个zero warmup相同，11–15均`[.1,0]`；16–20才分歧：xy=`[0,.1]`、x_minus_x=`[-.1,-0.]`。本轮明确只执行每词suffix前5，到frame15；没有执行第二段。两个完整词的最终状态不能由这次中间端点区分。
3. 没有发现需要修改封存生产代码的具体dispatch/索引bug。存在**科学解释边界**：goal factor并未改变可见输入；frame15不是full-word terminal。旧frame20 coverage与新frame15 feature仍不能同状态配对，native_vs_raster_rank_agreement保持null，C2/科学准入保持false。

## 实际证据与运行

入口analysis report：`le-wm-main/results/agent_p5_live_raster_analysis_20261028_v3_saved_MAIN01.json`，SHA `6b167d6a24c2756f938a5840a02ee29211612558f8e780d1a6c905581731a928`；MAIN strict：同目录`agent_p5_live_raster_analysis_20261028_v3_consume_MAIN01.json`，SHA `28785a84f552e7276a462e852dfab74c33170d8aec1dfff7d4853fd229125547`。

由analysis.authority的真实路径/SHA追到encoder prepare/report，再由encoder render_authority追到live renderer prepare/report；只核本问题涉及的原始artifact和身份，不调用任何producer.analyze/consume来代替诊断。每个NPY/PNG/patch均对原收据SHA/bytes，PNG解码逐像素核NPY；encoder实际rgb输入路径/SHA精确join原shot；目标patch必须为对应image_key自己的文件。最终重读所有使用的saved文件SHA核稳定。

新诊断实际运行exit0，结果：`le-wm-main/results/agent_p5_raster_first_stage_diagnosis_20261028_v1_saved01.json`，SHA **`f7d89438ab63a23e6badc9c29daddfb3d7411ea4b59f003155a3291107f8100b`**。含完整bindings、5状态颜色一阶pixel/feature统计、xy/x_minus_x每帧0..15五文件字节对照、各自原source精确匹配状态、原frame15/16/20对照和结论。没有生成新图、latent或伪strict receipt。

| 固定状态 | visible/white差异像素 | RGB MSE（0..255） | patch feature MSE |
|---|---:|---:|---:|
| xy |779|114.39693611819727|0.4516983608344624|
| yx |825|123.35445232780613|0.5698911346941358|
| x_minus_x |779|114.39693611819727|0.4516983608344624|
| zero_zero |825|123.35456526360544|0.6196937512046183|
| goal |0|0|0|

四个词的差异像素bbox均为`[ymin,xmin,ymax,xmax]=[93,70,153,130]`；goal bbox=null。这些是所有固定状态的全量描述，不按效果筛选。四词feature均98304个元素不等；goal0个。它只显示这批可见颜色变化与feature变化相伴，不是机制/C2或任务收益证明。

## goal为何无像素变化：saved状态加原生绘制顺序

goal保存的block pose和goal pose精确同为`[256,256,0.7853981633974483]`，block为同一T形状，静止；color元数据确实变化。原native source `external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py` SHA `6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723`：

- `_render_frame` 592–600行先画goal polygons、使用self.goal_color。
- 602–603行随后`self.space.debug_draw(draw_options)`画实际agent/block。

因此同pose实体盖住先画的goal，是与像素完全相同一致的直接渲染解释；最终pixel first stage为零是实测事实。这里没有重新创建Body/Space、计算假world vertices或重渲染，也不把“pose相同”升级成历史cached transform认证。元数据改色成功不等于最终可见输入受干预。

## xy/x_minus_x身份与动作时间

live renderer source `agent_p5_live_raster_intervention_20261028_v1.py` 48–50行固定target_frames=15；245–247行只执行`reference_suffix_raw_u[:5]`、随后goal reset。实际meter controls60/Space.step605与此一致，不是四词fullsuffix10的80/805。

本诊断核查：两臂constructor、seed相同；每个实际command_01..15都与各自原prepare动作逐canonical一致，且两臂实际raw序列相同。所有frame0..15的**当前live**rgb/state/proprio/transition/full_public五文件两臂均逐byte相同；visible与white各自的两臂PNG像素/patch也分别全byte相同。

两臂每帧RGB/state/proprio均精确匹配原source各自同frame；不把transition/full_public历史差异抹掉：原某些frame coverage/reward末位不同，本轮live相同；诊断完整记录这些false，仍不声称历史full_public相等。

另外只读原sealed physical（report SHA `a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633`）自身的frame15/16/20：

| 原frame | xy/x_minus_x RGB bytes | state7 bytes | RGB MSE | state7 MSE（仅身份诊断） |
|---|---|---|---:|---:|
|15|相同|相同|0|0|
|16|不同|不同|22.578570099914966|402.7292164654756|
|20|不同|不同|93.44595689838435|501.121656646922|

这支持共同前缀在15相同、16以后真正分歧的解释。state7 MSE不是原native success，也不是本任务评分替代；不能据object-only、state投影或这个数值判完整PushT失败。没有对frame20重编码、调阈值或将其替换进已冻结16scores。

## 可执行复核与下一步边界

在`le-wm-main`执行（换新output避免覆盖）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_raster_first_stage_diagnosis_20261028_v1 --analysis results/agent_p5_live_raster_analysis_20261028_v3_saved_MAIN01.json --analysis-sha 6b167d6a24c2756f938a5840a02ee29211612558f8e780d1a6c905581731a928 --strict results/agent_p5_live_raster_analysis_20261028_v3_consume_MAIN01.json --strict-sha 28785a84f552e7276a462e852dfab74c33170d8aec1dfff7d4853fd229125547 --output results/agent_p5_raster_first_stage_diagnosis_20261028_v1_saved02.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_raster_first_stage_diagnosis_20261028_v1.py
```

实际轻量tests：**9 passed in0.15s**，覆盖null first stage与真实encoder invariance的区分、变化像素但feature相同分支、二者变化、相同像素却feature变化的异常分支、signed-zero byte身份、单ULP、shape/dtype/nonfinite拒绝。

未运行项：无physics/CUDA、无新encoder、无全文consumer重跑。若以后要测试完整词的颜色敏感性，需新前瞻合同明确frame20；若要识别goal颜色first stage，需新合同说明如何在合法原生场景获得真实可见像素变化（当前目标上“无变化”必须保留，不能补画、隐藏物体或事后选成功状态）。这些均是新实验设计，不是本轮自动修复授权；本交付不实现/启动。原有16score矩阵和其零goal主效应/交互仍按原数据保留，但解释为此设计下的null first stage，不是普适encoder不变性。
