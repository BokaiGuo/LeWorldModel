# GRASP selected physical RGB adapter v1

独占新stem `p1_constructive_grasp_rgb_cuda_20261027_v1`，只新增源码/tests/本doc。
不修改已冻结GRASP producer、physical adapter或generic selected-RGB源码/结果。
MAIN执行CUDA；实现者不运行模型、物理、actual consume或实际结果审计。

## 当前绑定状态

物理报告预期路径：
`results/p1_constructive_grasp_physical_20261027_v1_main01/report.json`。
已绑定MAIN提供的实际 physical report SHA：
`bf7f15cf9d5aad4ca7923c4a116f6703c5e559b496c50488de55a4a94394f606`。
physical prepare SHA：
`6c617a16ff889ab06673b6dc9feb5f44fb7829aeb01bc1d21b9dfcb211d6d3c9`。
selection SHA：
`ab6dcf51965091c388b0004037fc554f9fa7992011869f8cbdcbc8c31868e8f9`。
MAIN报告2 full10、initial_exact、errors=[]、0 native success；这里是来源说明，
不是实现者独立实际核验。selection SHA也在读取manifest之前精确检查。
prepare/build 在任何saved read、模型加载或输出创建之前拒绝缺失/非法SHA。
本新stem已完成最终常量绑定，旧文件始终不改。

固定GRASP physical adapter源码SHA：
`49df48cb1207459121445d2308d30928672aba3e90bad010210fd62ecd7e1c60`。
固定generic selected-RGB v2源码SHA：
`e1448739bbae827a2e3b1f62bf50f0f27aa06d47aef60163483fa289e6eb432e`。
GRASP模型report/prepare身份沿用冻结physical adapter常量；不借用gradient或MPPI的结果SHA。

## 固定输入和接线

固定 `(source_slot0,GRASP_H3_bounded,prefix5)`、`(0,GRASP_H3_bounded,terminal10)`、
`(1,GRASP_H3_bounded,prefix5)`、`(1,GRASP_H3_bounded,terminal10)`，不按native success、
物体位移或latent cost择图，不去重，不替换分支。所有分支必须完整full10、initialexact、
资源/journal/guards合规；partial failclosed。已知两根不是新独立样本。

先核物理report外部SHA，再读 execution_binding→独立physical prepare→manifest。
调用 `adapter.build(adapter.REPORT,adapter.REPORT_SHA)` 验证原模型own consume命令权限；
这两个实参不可遗漏，GRASP与旧gradient的无参build接口不同。
核对physical prepare新schema、固定source slots/contract、sourcepack SHA、模型report/prepare、
generic executor和command_authority。只hash evaluator-only closure，不解析证书/sourcepack，
不调用 adapter.verify/material、generic physical.verify/load_source。
RGB阶段不读取witness控制命令，也不把physical outcome传回planner。

generic `rgb.assemble` 只接受generic prepare schema。因此先比较原journal首项prepare与
原adapter prepare完整相等，再同时在深复制RAM视图改两个schema tag，其他字段不变。
原saved字节与hashchain绝不写回。复用generic全分母/returned commands/goal/资源/PNG join。
PNG文件和像素SHA都复验；source/end goal、blind task、原normalizer和identity192 projection不变。
模型动作约束继承原own consume，不重新选择或生成动作。

新prepare schema为本stem `.prepare`；verify用本stem build完全重建，不能调用generic verifier。
小prepare独占写入并fsync，绑定本source/tests/doc、原closure、模型、journal、四张PNG及任务。
不复制模型/PNG，不创建多余prepare/result。

## MAIN-only CUDA与输出

复用封存generic的PNG decoder、encode_images、compare、Store与Ledger/guard。
固定4encoder＋4projector，无predict/actionencoder/AD/physics/训练。
原生冻结模型batch1图像输入，经现有encoder/projector保存float32(192,) NPY；
即使PNG相同也执行4次，不从已有latent缓存冒充新编码。
CUDA唯一后端、MATH SDPA、禁TF32、inference_mode、同步计账；没有CPU神经fallback。
attempt/return/image_saved逐项durable；partial保留全分母且不重试。
前后模型tensor SHA、原始闭包最终复核；未加载模型时 model_unchanged=null。
artifact listing/hash/最终报告分别尽力保存，不用secondary错误覆盖first_error。

原Q严格为identity mean192，计算actual_Q/root_Q/actual_Q_change和相对原目标MSE。
`predicted_Q`、`predicted_vs_real_mse`保持unknown，不重算GRASP rollout或把virtual state当预测。
成功状态 `grasp_selected_rgb_encoded_not_admitted`；report独立schema并包含physical父SHA。
`official_equivalence=False`，P1/P2 admission=false；不是官方数值复现或全算力匹配证据。
compiler/physics/原始编码成本另外支付，不计入本次4张新编码以外的“免费预算”。
输出cap2MiB、磁盘reserve200MiB、final128KiB；只保存4NPY/ledger/binding/report，不复制PNG。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_grasp_rgb_cuda_20261027_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_grasp_rgb_cuda_20261027_v1 prepare --output results/p1_constructive_grasp_rgb_cuda_20261027_v1_prepare01.json
```

设置 `GRASP_RGB_PREPARE_SHA` 为上一条真实回执SHA，不伪造尚未生成的prepare：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_grasp_rgb_cuda_20261027_v1 verify --prepared results/p1_constructive_grasp_rgb_cuda_20261027_v1_prepare01.json --prepare-sha "$GRASP_RGB_PREPARE_SHA"

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_grasp_rgb_cuda_20261027_v1 run --prepared results/p1_constructive_grasp_rgb_cuda_20261027_v1_prepare01.json --prepare-sha "$GRASP_RGB_PREPARE_SHA" --output results/p1_constructive_grasp_rgb_cuda_20261027_v1_cuda_main01
```

## 验证边界

RAM/mock测试覆盖4图固定顺序、不去重、schema RAM视图不改原数据、command/goal/guard/资源
损坏拒绝、physical/模型seal和权限拒绝、GRASP显式build参数、pending SHA早拒绝、identity Q、
encoder/projector mock计账、原生NPY保存及partial attempt不推进。没有临时磁盘fixture。
复用封存generic数组fixture，不宣称独立实际审计。通过测试不等于真实CUDA或科学准入。
本版本真实physical SHA已最终绑定；实现者没有创建实际RGB prepare/result或执行模型。
