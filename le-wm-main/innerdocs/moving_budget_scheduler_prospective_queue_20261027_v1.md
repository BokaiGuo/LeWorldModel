# Prospective blind saved queue v1

日期2026-10-27。独占source/tests/本文三个新文件。纯saved consumer，不生成模型信号，不导入旧planner/physics，不训练、重放或调用神经。遵循已冻结设计 `moving_budget_scheduler_prospective_separation_20261027_v1.md` SHA256 `8c6eee820a49dd982649df10ccffbd15412ce0117eff7c7a11553986166704c6`。无真实fresh panel与screen资产，当前仅实现/mock交付，不能称actual queue已成功。P1/P3及独立确认始终false。

使用experiment-code技能的固定接口/日志/错误保留模式；用户要求优先，未生成图、未训练、未要求非零成功。个人goal仅覆盖这个实现交付。

## 模式、顺序、冻结

MAIN必须显式选择 `population` 或 `signal_separated`，没有默认、没有自动切换。

* population：manifest恰好8个two-root panels，selected_ids始终为原8项，包括信号碰撞或失败项；只有全部完整验证后released_ids才为全部8项。任一缺材料/失败阻断整队列，不悄悄运行剩余子集。G仅描述，不筛入。
* signal_separated：本v1固定完整16个two-root panels作为分母（设计允许最多16，本实现取固定上限，不支持可变池）。必须核完16项终态，按manifest顺序选最先8个G=true且screen_completed者。少于8不补，零项status=queue_empty_no_search，无搜索授权。明确screen_failed保留分母/attempt成本，不是缺失材料；缺失/非法receipt则整队列blocked，不能靠前8项齐全跳过后8项。

同一queue禁止重复panel/root/task/blind身份；输入index可以任意顺序，稳定join后完全服从manifest次序。两模式独立freeze，不能执行后转换模式或换根。population=总体问题，signal_separated=模型信号条件化问题，不自动决定哪个是主科学问题。

freeze绑定自身source SHA、原设计SHA、完整manifest及固定policy。canonical JSON无换行，freeze-sha为实际文件字节SHA；消费时要求仍是同一source。冻结文件timestamp严格早于freshness认证及screen开始；这只是可检验的时间/身份声明，不是防恶意回溯的可信时间戳，MAIN仍应实际保管执行顺序。

## CLI（内层仓库；示意路径，尚不存在真实manifest）

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_prospective_queue_20261027_v1 freeze \
  --manifest /ABS/new_blind_panel_manifest.json --manifest-sha ACTUAL_FILE_SHA256 \
  --output /ABS/existing_parent/new_queue_freeze.json

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_prospective_queue_20261027_v1 consume \
  --freeze /ABS/new_queue_freeze.json --freeze-sha ACTUAL_FREEZE_FILE_SHA256 \
  --index /ABS/new_saved_signal_index.json --index-sha ACTUAL_INDEX_FILE_SHA256 \
  --input-root /ABS/blind_saved_receipts_root --output /ABS/existing_parent/new_queue_report.json
```

freeze/consume只进行saved输入校验和exclusive单文件发布，绝不创建研究source或执行screen。所有输出`xb`禁止覆盖，父目录须已存在。单份输出<=2MiB，发布前free>=200MiB+2MiB；输入单文件<=2MiB，不复制NPY或大报告。blocked report也发布并返回exit2；已验证/空队列exit0仅是工程状态，既不是执行授权也不是科学通过。无效外部SHA、整体schema或freeze被修改等顶层输入错误抛错且不发布正常报告；逐panel缺失/错误记录到blocked行。不能将这种输入拒绝计成模型负结果。

## 精确manifest schema

所有对象严格白名单key；未知字段不被忽略。JSON重复key、NaN/Inf拒绝。以下名称中的STEM完整为 `moving_budget_scheduler_prospective_queue_20261027_v1`。

```text
manifest = {
 schema: STEM+'.manifest', mode: 'population'|'signal_separated',
 approval: {authority: 非空MAIN认证主体, no_signal_or_outcome_seen: true},
 excluded_root_sha256: [至少2个已用根的真实SHA；MAIN应提供完整排除清单],
 panels: [{panel_id: 唯一字符串, roots: [identity_slot0, identity_slot1]}, ...]
}
identity = {
 root_sha256: 真实root身份SHA, task_id: 全队列唯一字符串,
 blind_sha256: 真实blind输入SHA,
 assets: {source,encoder,anchor,model,normalizer,q,candidate_bank_rule: 各自真实SHA}
}
```

asset字段是上游closure的SHA标识，不代表本模块已打开/重新运行这些assets；root identity整体在freshness和screen中必须逐字段相等。source/encoder生产器需要自己生成严格blind任务并保留原raw/normalizer/history合同；本consumer不从privileged state构造目标，不重新认证真实history或模型是否冻结。

excluded列表必须来自MAIN、包括旧两根及所有用于设计的根，代码仅验证显式排除和全局唯一，不能自行识别遗漏的历史根。source generation协议/seed列表/目标规则须已由source closure封存；只有新seed不构成独立性证据。

## 后续producer需要提供的新saved信号接口（不冒充现有旧schema）

```text
index = {schema: STEM+'.index', freeze_sha256, panels: [
 {panel_id, freshness: ref, screen: ref}, ...]}
ref = {file: 相对input-root路径, sha256: 文件字节SHA}

freshness = {
 schema: STEM+'.freshness', freeze_sha256, panel: 完整manifest panel,
 authority: 非空MAIN认证主体, issued_at_ns: 整数,
 independent_roots:true, not_used_for_design:true,
 no_outcome_or_candidate_cost_used:true
}

screen = {
 schema: STEM+'.screen', freeze_sha256, panel: 完整manifest panel,
 status:'screen_completed'|'screen_failed', started_at_ns, finished_at_ns,
 counters:{attempted:vector, completed:vector}, errors:[],
 roots:[root_signal_slot0,root_signal_slot1]
}
vector={predict:8, action_encoder:8, jacobian:2, vjp_directions:384, svd:2, solve:0}
root_signal={identity: 完整root identity,
 nominal:ref, goal:ref, q:ref, jacobian:ref, u:ref, sigma:ref, vt:ref}
```

screen_completed要求attempted=completed=完整vector、errors空、两根完整。screen_failed要求非空errors、roots=[]，每轴0<=completed<=attempted<=cap，禁止对partial信号挑选；错误数组仅传递诊断，不作为配额信号。失败源的partial原数组由上游独占保留，本接口不复制或删除它们。

所有ref NPY均`allow_pickle=False`，**float64原生小端/平台dtype**，finite、精确shape：nominal/goal192、q192x192、jacobian192x20、u192x20、sigma20、vt20x20。若原真实数组是float32，上游必须显式导出float64数值视图并保存原始receipt/转换来源，不能把改精度称原始字节；不允许反序列化对象。array path必须位于input-root内，拒绝绝对路径/..和逃逸symlink。

q严格为identity192；对saved U/s/Vt检查正交、非负降序和 `U diag(s) Vt = QJ`，固定rtol1e-7/atol1e-9。不调用新SVD/solve/AD；若上游导出精度不能满足先blocked，不能执行后调容差。从绑定goal/nominal计算r，rank按 `sigma>sigma_max*1e-6`，chi按原residual projection clamp，w=(1-chi)||r||²；不信任用户填入的w或candidate cost。

按原half-up分别计算D/S，U/N固定10/10。G严格为sumw>0且p<=.44或p>=.56，并再次断言D/S/U三向量不同；零sumw不可分离且不加epsilon。N只拿固定配额，不作为读取信号的算法入口。该queue会为审查生成全方法配额，但它不执行任何方法。

freshness是外部认证而非代码可独立证明的历史事实：报告始终带 `external_freshness_attestation_not_independently_proven=true`。缺证书、false声明、错误SHA/identity、时间不符、未知字段均blocked；禁止fabricate实际报告来填空。新的真实producer接口和MAIN认证尚未提供，本次没有可消费的actual fresh queue。

## 成本与release

每行保留input_refs、panel身份、status、signals/配额或错误。全部panel终态（含未入选项）汇总screen实际attempt/completed；blocked项无法认证的成本不猜0，汇总明确是lower_bound且screening_cost_complete=false。screen_failed已校验计数仍全额纳入。

population完整规划上限4160predict/AE、16J、3072方向、16SVD；signal-separated固定16分母最多8入选上限4224predict/AE、32J、6144方向、32SVD；均0solve。planned_search_predict_upper_bound=512*released数，以上均是计划而非本consumer实际调用。本consumer实际neural/physics为0；上游source/编码/anchors/真实执行另计，不把AD等价成candidate或claim等总算力。未来runner必须绑定并复用已付screen receipt，否则另封额外真实预算。

selected_ids表示策略选择，released_ids才是完整校验后的工程接口；下游必须核status、外部report SHA和原freeze，不能只看selected_ids。空队列不得启动搜索。P1_admission/P3_admission/independent_confirmation/compute_advantage始终false。不同配额不保证不同winner，更不保证真实改善。

## RAM/mock测试

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
tests/test_moving_budget_scheduler_prospective_queue_20261027_v1.py
```

fixture完全合成、NPY/CLI文件接口均BytesIO/内存mock，不落盘研究fixture，不导入torch或环境。覆盖边界、零权重、顺序、固定分母、两模式、first8而非最大信号、空队列、source失败、缺材料、freshness/chronology、身份与SHA错绑、禁止字段、分解复算、成本、reserve/独占输出及help/freeze/consume端到端。首轮43pass/1fail为CLI测试mock复用输出名保留旧结果，已修为独立blocked.json，未放宽生产门。

最终上述命令实际结果：**51 passed in 0.71s**。这仅是RAM/mock接口验证，不是新panel信号、freshness或科学准入证据。

尚缺：MAIN批准某模式；真实全新盲source/task/anchor/encoder及完整排除清单；MAIN freshness认证；在freeze之后实际执行并导出本新schema的信号producer；MAIN复读后主机运行。本次不是actual队列或P3科学完成。
