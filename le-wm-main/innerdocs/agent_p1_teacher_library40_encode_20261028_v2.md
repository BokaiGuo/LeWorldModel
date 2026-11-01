# RGB40 adapter v2：真实saved ABI修复

独占source/tests/innerdoc stem `agent_p1_teacher_library40_encode_20261028_v2`。v1与所有MAIN/原始结果保持冻结。v1真实prepare失败于roster的`counts.physics_step`：误把预算contract.caps键当成实际durable resource键，原RAM fixture同错因此漏测。这是实现错误，不是神经/物理负结果。

v2保留原native ledger名，完整检查10轴：close40、constructor40、explicit_render2638、goal_bind40、goal_render80、goal_setter120、internal_physics_step8160、public_step800、reset80、set_state160；各attempted/completed都必须与实际相符，不接受physics_step别名、缺轴、额外轴或零填。原counts完整继承，不改producer/receipt。

先读5份固定JSON真实header及两份Markdown协议，native_audit没有schema（不杜撰）。physical journal是`{file,sha256}`，compiler journal是`{events,last_sha256}`，二者不混用。physical execution_binding的4字段、compiler8anchors/40plans/resources、原模型hash、prepare anchors、job/plan完整seal、checkpoint/native snapshot/PNG条件、goal192/预测数组/context/Q均走原saved闭包。源内INPUTS的7个路径/外部SHA与v1相同，不使用未来SHA。

实际输入接线核查（非仅首个错误键替换）：

| 层 | 实际字段与核验 |
| --- | --- |
| physical report / partial receipt | schema/status/40rows/counts/errors/inputs_final_verified、report/prepare/compiler两SHA、完整rows精确相等 |
| physical prepare | schema、compiler_prepared/prepare_sha256、bindings、8anchors、40jobs；anchors与compiler prepare完全一致 |
| physical rows | 固定task/condition/role/library和11项身份、原raw10字节、public warm10/prefix5/suffix5、51frame/15raw见证、last_return、checkpoints/end_state、goal/goal_pose/pixel身份 |
| terminal journal | 原canonical hashchain、run_finished/tail、同job suffix5 off/on/off、checkpoint10/absolute20、frame/pre/post/snapshot/state/renderer目标不变 |
| compiler report / receipt | 各自schema/status、source_verified、prepared/prepare_sha、before/after model hash、8rows、40plans、完整resources一致 |
| compiler prepare / arrays | bindings/model/anchors；goal数组float32[192]，evaluations.latents float32[3,192]、context.latent_history、stage/library_index/cost与sealed plan.model_Q精确相等 |
| native audit | 没有schema字段；status、40rows、known2/fresh0、原40末次native_success、index12失败XY/8blue/canvas标记；不把audit当作神经回执 |
| model/source/preprocessing | 原闭包逐SHA、冻结checkpoint config224/patch14/native192与257 shape、原预处理源及dataset_stats、本v2三文件均绑定；不加载模型 |
| protocol / correction | 原Markdown两份SHA，不当JSON解析，且不以v2重写或降低其门槛 |

未消费的原字段仍由固定外部report SHA、完整artifact SHA及原strict receipt封存，不删字段重写源文件。新增`input_fields/physical_count_fields`只记录实际已核schema，不生成另一套批准registry。

仍是known2、固定40槽、index12 final-on缺失null；8blue像素不绕过agent_y越canvas失败。default full40在任何输入读取/模型/输出创建前拒绝。仅MAIN显式partial39 CUDA可新编码39次encoder+projector，0predict/AD/physics；257token、原预处理、原192维mean Q、lossless原精度module trace、64MiB output/16MiB final reserve/200MiB floor/600s不变。未知不填0，latent误差不是物理收益；full40/P1/fresh确认均不放行。

## 回归与执行入口

测试包含真实7项INPUTS及全部prepare接线的saved-only回归，不mock upstream读取/hash/roster/数组/PNG/容量；prepare最终写入截获到RAM，随后verify只对该RAM prepare提供读取，其余输入全为真实sealed文件。此测试不调用run、不导入模型工厂、不构造环境，不创建实验结果；不是实际39张模型编码。另有10轴逐字段mutation及既有RAM hook/257vs197/null/不重试/失败账回归。

已实际运行：独占137tests通过（17.33s）；单独真实saved prepare→verify回归1passed（11.51s）。实际核48,963条来源绑定，RAM serialized prepare 9,807,422bytes，固定40槽/39available/index12missing；保守pre-final上界16,829,526bytes，低于50,331,648bytes可用。NumPy/PIL/saved hash允许，guard拒绝导入torch/gym/gymnasium/stable_worldmodel/transformers；0新神经、0新物理、0实验目录/结果写入。容量与schema测试通过不是P1科学通过。

最终联合回归：本v2+原`test_p1_moving_overlay_recompile_cuda_20261027_v1.py`+原`test_agent_p1_teacher_cross_library_20261028_v1.py`，223passed（34.48s）；CLI help退出0。v1三SHA复核仍为source`cc1486cff280567ab26704cd33a164bd780f002f894b576079600fdd476df525`、tests`f207ba1c877c3cd48bc2ff2c8f7038b7a55b24cb6b7515745f7e9cdf4eaeb9fb`、doc`6d1a0c658f9ec8442bc8dc215b742779b2f4e95281f2ebb485265c88d1bbe99a`。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_library40_encode_20261028_v2.py
.venv/bin/python -B -m pytest -q -s -p no:cacheprovider tests/test_agent_p1_teacher_library40_encode_20261028_v2.py::test_real_saved_prepare_verify_end_to_end_without_model_or_disk_write
```

MAIN从inner repo执行；仅在审读并明确授权partial39 CUDA后run。PREP_SHA与REPORT_SHA必须取实际文件外部SHA，不复用v1prepare，也不得覆盖已有输出。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v2 prepare --mode partial_observed_only --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v2_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v2 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v2_prepare_MAIN01.json --prepare-sha PREP_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v2 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v2_prepare_MAIN01.json --prepare-sha PREP_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v2_cuda_MAIN01 --authorize-main-cuda-partial39
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v2 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v2_cuda_MAIN01 --report-sha REPORT_SHA
```

冻结后三文件不再原地编辑。未执行神经、physics、网络，不触reserved4fresh。未决：MAIN审读、durable实际prepare/verify、明确partial39 CUDA授权和run/strict consume；不可复用失败v1 prepare。保留旧个人goal不覆盖，本交付只为v2修复/真实saved回归，不宣称研究goal完成。
