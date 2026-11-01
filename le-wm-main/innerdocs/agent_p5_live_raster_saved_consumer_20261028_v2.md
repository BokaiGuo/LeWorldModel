# Live raster saved consumer v2：修正时间接线，不重跑物理

独占新三件套`agent_p5_live_raster_saved_consumer_20261028_v2`。原renderer source SHA `34876d0624373416b415dc97b4938364de4eb3156985d5df851461e665b73e84`及其run/prepare/result保持封存。MAIN实际run report SHA `0e78ceb18b7748aebd511ed90ce4fd6f93dc639773ec517181f59d495c0b0487`，605steps/85PNG/errors=[]。此值仅用于MAIN命令引用，不硬造未来回执，也不改旧consume失败。

## 唯一语义修复

原v1 consumer把各word suffix完成时的baseline render快照当作后来intervene的anchor，跨越其间原goal constructor/reset消费全局NumPy RNG。实际MAIN已定位四word仅`/global_rng/numpy/2`不同，goal行无差异；原run每张图自身before/after全等，intervene也按其真实当前anchor严格验证。这不是颜色导致RNG漂移，不能删除RNG字段，也不能重新跑物理碰运气。

v2仅新建saved消费者，完整继承v1 consume的所有校验（schema、闭包、全artifact、85PNG/65frames、原raw60动作、605dt、逐操作native delta、完整hashchain、全分母、exact/diff、原色恢复、old失败/历史null）。唯一修复的时间点：

1. baseline自身实际render before/after **全部字段**canonical相等，包括全部RNG。
2. 从该行第一cell的原始anchor文件读真正intervene时点快照；两cell anchor文件必须全量canonical相等。
3. baseline→当前anchor的public_state、topology、goal_color（含对象identity）及global Python RNG严格全等；快照schema四字段必须完整，global_rng必须保留python/numpy。四word从完整baseline MT19937 state在**独立RandomState**中仅复算原native `seed(None)`的`randint(0,25536)`一次，要求所得完整624word、游标、Gaussian flag/cache逐canonical等于当前anchor NumPy state。不是只比较游标；不允许任意global_rng变化。goal行没有这段后续constructor间隔，必须完整baseline==anchor。完整exact diff原样保存，不能将缺项填0或不记录变化。环境RNG已在public_state中，跨baseline也必须全等。
4. 每个visible/white/恢复shot自身before/after全量严格；其public_state/global_rng/topology对**当前anchor**严格全等。恢复shot额外goal_color及完整snapshot对当前anchor全等。恢复receipt也对当前anchor严格；不能退回旧baseline RNG。
5. 所有原色/恢复像素对原真实baseline仍逐byte相同，包括接触点；颜色干预图仍不放宽mask或像素门。旧失败SHA/历史null/无科学准入不变。

原native `pusht_env.py`的seed函数638行保存明确调用`np.random.randint(0,25536)`；source由原prepare闭合。这项复算证明该已知原操作能够精确解释完整状态转移，并结合原journal调度，不声称仅凭RNG state唯一识别历史函数调用。独立RandomState(0)后set_state不触碰生产global RNG，不创建Body/Space；不改任何生产状态。无函数替换、全局patch、native加载、solver、模型调用或重试。源码独立保留完整consume正文而不是catch旧异常后伪造成功。

## 新receipt ABI

`consume(original_prepare_path, external_prepare_sha, original_report_path, external_report_sha)`直接读取已有真实run。返回schema **`agent_p5_live_raster_saved_consumer_20261028_v2.receipt`**，status **`strict_live_raster_complete_v2`**，不能骗过旧encoder/v1 strict status。

保留`prepare_sha256/report_sha256/states5/cells10/physics_steps605/native_entries/encoder_inputs10/source_comparisons65/old_failure/old_renderer_release=false/historical_full_public_equivalence=null/scientific_admission=false`。`renderer_schema`明确原renderer版本；`consumer_bindings`绑定新source/tests/doc与原renderer source。

新增`baseline_to_current_anchor`固定五项，含state_id、baseline_after/current_anchor原文件收据、两完整快照digest、`exact_differences`完整JSON pointer/expected/actual/presence列表，以及`constructor_rng`证明（operation/draw/full_state_exact/before与predicted/actual after digest/production_rng_modified=false）。四word只接受上述一次原操作精确解释的差异；goal要求完整相等。encoder_inputs沿原image_key/rgb/png/pixel_sha256/before/after/fresh_snapshot_digest/fresh_public_state_digest ABI；digest来自color当前实际after，不是baseline，不删除RNG。source_comparisons保留全部原transition/fullpublic历史差异。

下游必须新encoder版本显式绑定本v2源码与receipt SHA，重算本consume并全量相等；不得改旧renderer或旧encoder的status/sourceSHA以接受v2。当前不会将旧strict失败回执改写为成功。

## CLI与测试

```bash
# le-wm-main根目录。PREP_SHA来自原实际prepare，不能新造prepare。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_saved_consumer_20261028_v2 consume --prepare results/agent_p5_live_raster_intervention_20261028_v1_prepare_MAIN01.json --prepare-sha PREP_SHA --report results/agent_p5_live_raster_intervention_20261028_v1_render_MAIN01/report.json --report-sha 0e78ceb18b7748aebd511ed90ce4fd6f93dc639773ec517181f59d495c0b0487 --output results/agent_p5_live_raster_saved_consumer_20261028_v2_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_saved_consumer_20261028_v2 verify --prepare PREP --prepare-sha PREP_SHA --report REPORT --report-sha REPORT_SHA --receipt V2_RECEIPT --receipt-sha V2_RECEIPT_SHA
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_live_raster_saved_consumer_20261028_v2.py
```

无run/prepare物理接口。输出排他、2MiB receipt cap、200MiBfloor。verify全量重算receipt并对外部SHA指定文件canonical相等。RAM tests复用原无solver内存fixture，仅在goal ctor模拟一次np.random.randint：验证v1拒绝而v2严格通过且完整保存差异；逐shot RNG篡改、当前anchor漂移、恢复错误RNG、跨baseline非RNG篡改、缺槽/PNG/命令/nativecounts/旧失败篡改均拒绝。另测624word/flags/cache/游标错误拒绝及独立RandomState不改global状态。fixture不生成研究结果，不运行物理。实际saved consume由MAIN执行。

原分析器`agent_p5_live_raster_analysis_20261028_v1`暂停，现有文件保留不变；其旧encoder authority不能自动接受本v2，后续需MAIN新encoder最终SHA与实际结果后新版本接线。
