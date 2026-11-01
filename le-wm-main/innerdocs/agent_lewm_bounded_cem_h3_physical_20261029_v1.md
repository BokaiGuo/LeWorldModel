# Bounded CEM H3：actual mean 两根原生部署（NEW v1）

工程协议：完整原 known2/off/root0,root1，实际 CEM ownreceipt 的 literal actual_raw15；原 reset/place/warm10 后执行原 raw15。不是官方 CEM、fresh cohort、Hi training scaler 认证或 common-resource 公平胜负。所有科学 admission/release/equal-compute 均 false。无新 predict/encoder/AD/训练；只有 MAIN 授权 run 才导入原生环境。

## 已有权威与信任边界

所有路径基于 /home/gbk/code/LeWorldModel/le-wm-main。

- 原 producer：experiments/spectral_reachability/agent_lewm_bounded_native_cem_h3_20261029_v1.py，SHA 4a2f7bf18729f569854610d2f0b5f54c12a52d7941a82acd779a0cbdb4889e7b。
- prepare：results/agent_lewm_bounded_native_cem_h3_20261029_v1_prepare_MAIN01.json，SHA 89b091af85bb36aa36a684a438d2179e7ab0c74c85de8303d3ac3c37e10d642b。
- report：results/agent_lewm_bounded_native_cem_h3_20261029_v1_cuda_MAIN01/report.json，SHA b3e239367a482a9ff561e69f7a09657540cac690eecd2adcba08d948c210d6a1。
- 完整 ownreceipt：results/agent_lewm_bounded_native_cem_h3_20261029_v1_consume_MAIN01.json，SHA d0e215618c367049c09eed7e93accb90615973f4f8cb322fd19359fef1f7e7a4。

prepare/verify 以该已完成 ownreceipt 为数值信任起点，不递归重跑 CEM fullstrict。重建 producer.verify（纯 saved source/task ABI），校验原全部 artifact 文件 SHA/字节/roster、原 model 前后哈希、paid delivery actual_raw15 NPY 与 receipt 15×2 float32 bytes 精确一致。闭包并入 producer sources、native source material、own source/test/doc。未新增模型真实性、训练 lineage 或官方公平性认证。

原任务来自 P2 common-resource prepare（7facba500439172ced280ec0b893659715139b1e5e22073ae7313665a1c0faec）；具体原 task、goal、history、blind RGB、normalizer 的来源由该原 producer.verify/source closure 校验。root0 task_id f5145aab0b44d21d0492d591590a73b4d4c9109903d5a97f18d01618f3bd5cc7；root1 697557899923163376df67cba615f13ebf1580166ca4de40a6759d12c927292b。不再 tanh、inverse-normalize、clip、重选或按域筛根。literal 必须 lossless float32 且逐元素 [-1,1]。

## Native ABI / census

低层复用 frozen agent_mixedmode_physical_20261028_v1 的 place/bind_goal/dispatch/native_hooks，和 agent_allmode_physical_20261028_v1 的 fresh off/on/off capture/reference helpers。自有 loop/plan/schema/validator，不调用旧 GRASP runner，不 monkeypatch 其它模块。构造 swm/PushT-v1 resolution224、with_target=False、render_action=False。同原 reset seeds、placement recipe 和 warm10，不 hydrate 到 poststate。

每根：两 reset，每 reset 两次 set_state/physics step；warm10 + own15，每 control10 physics substeps。全两根上限：

|资源|整数上限|
|---|---:|
|constructor / close|2 / 2|
|reset / set_state|4 / 8|
|goal_setter / goal_render / goal_bind|6 / 4 / 2|
|public_step|2×(10+15)=50|
|physics_step|2×4+50×10=508|
|explicit_render / PNG|2×(11+16)×3=162|

原 warmup 引用只认证原 saved 17 项/根（off 全步、on 在0/5/10），恢复 off 复用对应原 off；其余 warm on 明列 source_unpaired_on_steps，不捏造历史引用。prefix51 是 fresh warm0..10 的33图 + 本次 own plan0..5 的18图；其中本次 own first5 不冒充其它 planner 的原动作。warm10 history3、native7 root/goal、renderer、全部可引用原 RGB/state/snapshot exact；任何失败保留分母，不容差。

所有控制 intent durable fsync 后才派发；嵌套原生 reset/set_state/physics/render 全计账并在 finally close。所有 actual action dtype/bits、return、fresh pre/post、goal、不变 renderer 和 native7 reward/success 精确核验。合法早停不补齐，cp0/5/10/15 缺项列 missing_checkpoints，未发生 endpoint 的 state/metric/success=null，不用零代替。完整两个槽始终存在；工程失败不当科学负结果。

## 64MiB 整数运输证明

保持同类 two-root full15 64MiB output、8MiB final reserve、200MiB floor、900s。shape_census 是整数运算；PNG224 RGB 每行包含 filter byte，scan=224×673=150752。使用保守 generic deflateBound = n+ceil(n/8)+ceil(n/64)+11（不是默认参数的 tight compressBound），再计 PNG signature/IHDR/IEND 和 65536-byte IDAT 分块开销。首轮真实不可压缩图实证不适用 tight compressBound（150968B > 150891B），故未交付前修正整数证明而不改实际 PNG 编码或 cap。Store 对每个实际 PNG enforce 保守上限和162总数；RAM fixture 将真实 Pillow 随机不可压缩 RGB 全162编码写盘核对，无降精度。

元数据不假设任意乘数：working 所有非PNG文件（journal含完整 branch_closed row、execution_binding，以及至多每根一个1MiB invalid-render NPZ诊断）合计硬限16MiB；含 report/completion/failure 总非PNG硬限24MiB。每 write 检查现存总量+实际新增 bytes，配合 output64MiB 和非final 56MiB 门，任何超限在写前失败。working PNG总上界+16MiB <56MiB；最终PNG总上界+24MiB <64MiB。没有扩cap。

具体复制槽：748 operation receipts（1496 intent/terminal events），162 fresh events，50 returned_transition，30 own_response，2 branch_closed；report/branch rows 各自有 selected54、prefix102、checkpoint8、last_observation2，以及全部748 operation receipts。所有这些重复 JSON bytes 均落在上述 aggregate quota，并非再乘一个无依据 copy 系数。consumer 对实际每份元数据/PNG/完整 closure 再计 census；fixture 给实际行/事件/输出大小。quota 是运输上界，不保证任意异常长错误文本仍可完整发布；异常预算失败必须封存 partial，不可升级成功。

report fsync 后 hash 全部 artifacts，再发布 completion；completion 绑定 report SHA、prepare SHA、全库存 SHA/bytes、publication-inclusive elapsed。completion 写完再查900s，晚到写 completion_failure 并拒绝严格成功。completion 自写不可能包含自身测量，scope 显式记录；消费者需外部 completion SHA 且拒绝 failure 文件、缺库存、非canonical journal、未知文件。wall 为 cooperative callback 边界，不能抢占已阻塞原生调用。

## MAIN canonical CLI

在 /home/gbk/code/LeWorldModel/le-wm-main 运行。以下 interpreter 使用项目 .venv；若 MAIN 审计环境选用既有同依赖解释器应明确记录。PREP_SHA/REPORT_SHA/COMPLETION_SHA 只能填本次真实输出，不能用原 CEM SHA。

```bash
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_cem_h3_physical_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_cem_h3_physical_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_prepare_MAIN01.json --prepare-sha PREP_SHA
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_cem_h3_physical_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_prepare_MAIN01.json --prepare-sha PREP_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_physics_MAIN01 --authorization MAIN_BOUNDED_CEM_H3_KNOWN2_FULL15_PHYSICS_ONLY
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_cem_h3_physical_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_physics_MAIN01 --report-sha REPORT_SHA --completion-sha COMPLETION_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_consume_MAIN01.json
```

stdout 只 compact 状态/路径/SHA，不输出完整 report。所有 NEW 文件 exclusive，不覆盖原始证据。consume 为完整本次 native 证据消费，不调用 NN/physics；生产全部由 MAIN 负责。worker 只执行 synthetic RAM/临时磁盘测试，不对实际 inputs 做生产 prepare/fullstrict。原 CEM 模型费用继承另列；物理新增资源不装入旧计账。

