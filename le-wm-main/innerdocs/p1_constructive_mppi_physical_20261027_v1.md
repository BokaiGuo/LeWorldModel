# MPPI v2 final-nominal → generic physical adapter

仅薄接线，不新增物理循环/优化器/神经计算。唯一动作权威为已封存 `p1_constructive_mppi_cuda_20261027_v2.consume`：实际 report SHA256 `0db92a35da17b21eca3fa9d11ab90025bc8610aabe7e634af55c18ddaa9c82f9`，prepare SHA256 `139b0000336f5ed6764b46ea1ca8c8fd5882346009f8fb86829ea3f2886deccc`，compiler source SHA256 `f6bb9c882105d102c3292264f0018745c41ede8a8adcd2c3ab0d47dfcef6c920`。generic executor source SHA256 `9dea58dfc6418dd01bf9ab28de6d712189ca1154a5cf29b165efa4b8ef65ea96`。

固定原两根 source_slot0/1、taskID/blindSHA/exportSHA，单臂 `kernel_faithful_mppi_nominal`，每根完整10条float32 raw commands。consume以纯保存数组复核全kernel、原Q和history，不调用模型。adapter进一步要求selected_index34及last_updated_nominal_not_best，绝不从candidate池重选、不派发best sample、不clip/round/pad、不根据物理表现替换。generic JSON Pointer直接绑定本实际report的 `/rows/0|1/{commands,source_slot,arm,blind_sha256}`。同动作也执行全部两分支。

sourcepack SHA `9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9`；source prepare SHA `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`。两原root SHA固定在新源码，保持与原blind identity对应。selection builder只读取compiler closure和report，不读取sourcepack/witness/物理结果。prepare/run在评价器侧由generic原 `load_source` 读取已批准的goal/RGB/observed_states/root_history并按原seed与placement重建，禁止reset(savedpoststate)，不把隐藏证书回传planner。

## 预算与失败语义

原generic完整安全协议保持不变：每根10warmup+10selected，总40public steps（20+20）、4resets、44explicit renders、4goal renders、8set_state、408内部physics_step上限、2goal binds。224RGB target-free，逐步原RGB/state/canvas/dynamics guards、durable journal/PNG、terminal/guard早停与全部分母/失败留存；不补足、不替换endpoint。44PNG为两根各warmup0..10与post-goal-attach plan0..10。新neural calls为0，后续编码须MAIN另行授权。

搜索已发生每根36H2=72predict+72action_encoder（两根144各）；包含initial1、proposal32、updatednominal2、fresh replay1，不把搜索开销计为免费，也不声称sampling64/D2P总预算相等。完整search_contract与成本保存在adapter prepare；实际物理report保留原generic schema，通过execution_binding绑定新adapter prepare。ordinary project-kernel-faithful nominal baseline非官方MPPI，无exact density correction；`official_MPPI=false`、`paper_strong_MPPI_requirement_closed=false`、`P1_admission=false`。

wall600s，physical输出8MiB cap、200MiB floor；selection/prepare各<256KiB，不复制模型/上游产物。新prepare额外绑定adapter source/tests/doc、compiler完整source/model/export closure、report/prepare及generic/source closure。verify重建全部新prepare；run授权后直接委托generic.execute而非generic旧schema CLI。任何后续封存字节修改需独立新版本。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；实现者只运行RAM/mock tests与只读saved build，不prepare、不运行物理/GPU。

```bash
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_mppi_physical_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_physical_20261027_v1 selection --output results/p1_constructive_mppi_physical_20261027_v1_selection01.json
# MAIN独立核验selection SHA后填写SELECTION_SHA。
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_physical_20261027_v1 prepare --manifest results/p1_constructive_mppi_physical_20261027_v1_selection01.json --manifest-sha "$SELECTION_SHA" --output results/p1_constructive_mppi_physical_20261027_v1_prepare01.json
# MAIN独立核验prepare SHA后填写PREPARE_SHA。
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_physical_20261027_v1 verify --prepared results/p1_constructive_mppi_physical_20261027_v1_prepare01.json --prepare-sha "$PREPARE_SHA"
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_physical_20261027_v1 run --prepared results/p1_constructive_mppi_physical_20261027_v1_prepare01.json --prepare-sha "$PREPARE_SHA" --output results/p1_constructive_mppi_physical_20261027_v1_main01 --authorize-physics
```

测试仅证明接线/拒绝/纯数组mock协议，不是实际物理成功或P1科学准入。尚待MAIN review、独立prepare与真正物理执行；不预写success。
