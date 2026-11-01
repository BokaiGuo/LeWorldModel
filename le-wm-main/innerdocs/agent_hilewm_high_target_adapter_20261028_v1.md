# Hi-LeWM high-target native capture adapter v1

独占source/tests/doc；不改三臂runner、共同controller、冻结模型或封存结果。本交付是具体的原生高层调用捕获内核+完整saved consumer，不再新增资产manifest。当前**没有可信完整loader**，生产`run_cuda()`无条件failclosed，且在任何callback/I/O/torch/pickle/GPU分配前拒绝；传入`MAIN=True`、任意loader或自填SHA均不能解锁。将来需MAIN审定真实loader、源码闭包与配对后另新版本集成，不能原地改封存v1，也不能把本fixture receipt当真实神经回执。

## 已核原生接口与未接接口

准确来源是已有离线文本，不import安装该包，不执行文本的module/import/decorator代码：

- `results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__planning__policies.py.source.txt`，SHA `d6e7d476b45e47c5015494775b4d96727b8ebb8c801f719a100ba32878f288d1`，`HierarchicalWorldModelPolicy._plan_high:791`。原始high_info含planner_level=high及z_init/z_goal；solver返回完整actions，先保留CPU warm start，截取receding_horizon后reshape `[B,keep*action_block,macro_dim]`，调用rollout_high，**仅取pred[:,0,0,:]**，steps_since_high=0；其余solver动作作为next-high init（warm_start=false则None）。不是最后waypoint、不是final-goal重命名。
- 同目录`package__code__h_le_wm__models__jepa.py.source.txt`，SHA `bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2`。`get_cost_high:440`原始candidates `[B,S,H,grouped_macro]`→reshape→完整rollout→末waypoint L2 **sum**，不换为MSE。`rollout_high:304`逐macro累积、实际quantize、截原high context、predict_high，完整返回 `[B,S,T,D]`。`predict_high:191`依次macro_to_condition→high_predictor→flatten/high_pred_proj/unflatten。

native main配置声明192/32并不等于权重现场认证；本内核维数取实际输入与model._infer_latent_action_dim，不默认填资产维度。仅捕获单query B=1，不声称支持完整50-env批评测；候选S单独计量。原生high是优化macro后生成waypoint，不是SAGE无优化generator，也没有SAGE rem<=tau神经fallback合同。标准CEM的actions通常为更新mean，不保证为已评估候选；本模块保存其全部输出，绝不另做argmin选macro或额外求cost/rollout。

## 具体可复用API

`_capture(native_plan,model,solver_factory,cfg,z_init,z_goal,next_high_init,store,meter)`是实际tensor-neutral捕获内核，不加载模型、不编码RGB、不恢复env。native_plan必须是原生_plan_high函数，solver_factory(tapped_model)须返回已原生configure的solver且solver.model精确为该facade；MAIN未来loader负责构造与认证，不在本模块发明solver/Box/bounds/宏bank。

ModelTap把原类get_cost/high rollout/predict/quantize方法绑定至局部facade，让原生内部self调用经过捕获，再委托原module与helper；不重写数学、不改变冻结模型实例方法或全局变量。三模块实际pre/post hooks抓取全部输入输出，finally移除；policy状态仅写局部SimpleNamespace，不覆盖原policy。`cfg`精确四字段horizon/receding_horizon/action_block/warm_start；本版本仅单次非staged high计划，不伪称已支持staged policy、完整CEM实现或低层物理。预编码z_init/z_goal只接受原始浮点数值，不做normalizer/resize/坐标转换。

v1唯一公开执行调用是`capture_fixture(output, *, native_plan,model,solver_factory,cfg,z_init,z_goal,next_high_init,limits)`，强制RAM标记，**仅非神经NumPy callback测试**，返回fixture_only=true / production_executed=false。它不是HOST准入入口，不能用于真实模型。原生函数体的RAM测试去掉神经decorator并注入NumPy Tensor shim，不导入torch、不跑CPU神经。没有模型加载或训练、physics、下载路径。

## 实际receipt与cost轴

每次entry先核完整预声明cap、保存输入NPY并持久entry，才调用；返回保存原始tensor数组再持久return。失败不退款不重试，保pending与完整文件，不把缺失当0；只有完整capture才允许strict。输入/输出NPY存原dtype/shape/bytes/fileSHA/rawSHA，float16/32/64不cast；bfloat16若无法无损numpy导出则明确失败，不能假装已支持。允许合法的空warm-start slice。

分轴记录high_solver、get_cost_high、rollout_high、predict_high、quantize及macro_to_condition/high_predictor/high_pred_proj，各有calls与rows两轴。cost/rollout rows为B*S，predict/quantize rows为B*S折叠batch，pred_proj rows为原flatten上下文行数；不同rows口径明确区分，不相加当samples。solver1不是predict1；每次搜索cost rollout和最后selected receding rollout分别计入，无extra replay。宏latent不是raw环境动作，不宣称raw macro bank实物已获得。latent_action_encoder的quantize不是其forward，不能冒报macro encoder forward调用。

本捕获内核不调用image encoder、low predictor、physics、AD或训练；这些明确为本模块新增0，不回填上游成本。FLOPs/kernel/peakGPU/继承encoding/backward内部成本保null；high模块与API互相嵌套重叠，不相加。实际wall独立保留，不求所有资源轴相等。96MiB整个输出（1MiB尾部reserve）/200MiBfloor/600s工程边界，检查dispatch/写入；不保证抢占阻塞native调用。每个candidate完整tensor抓取会很大：该cap不保证容纳官方900×20全量计划，超过就拒绝/保partial，不能删候选、默调cap或装作官方baseline完成。

## Saved consumer独立检查

`consume(output,report_sha256)`只读完整flat产物：外部reportSHA、当前source/tests/doc/两原生text闭包、journal连续hashchain、entry/return嵌套/父子关系、各axis实际次数/原cap、完整NPY inventory与原bytes/dtype/shape/EOF/finite。独立核solver全输出→receding reshape→最后完整rollout→首target精确字节；原warmstart切片/goal身份；全部cost候选对应原raw macro；每步实际量化输入输出、high context、模块条件/投影链到完整rollout。

所有原生高层cost保留为native_L2sum；consumer另算saved_float64_L2sum逐候选并列。**不假设CUDA reduction与NumPy逐位相同，不引入任意ULP容差或放宽已有门**，该并列不是重跑模型/证明其神经数值正确。外部SHA及原生调用链是完整性证据，不是权重推理科学认证；不宣称消费验证了solver最优性或重新实现其CEM更新。首target只是被捕获，lowlevel_target_delivered始终false；尚未实际传给三臂低层，不伪造learned_pairing成功。

## MAIN可执行命令与后续硬前置

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_high_target_adapter_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_high_target_adapter_20261028_v1 --help
```

消费由MAIN指定的实际capture布局（当前只有pytest临时RAM产物，不虚构持久成功路径）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_high_target_adapter_20261028_v1 consume --output ACTUAL_CAPTURE_DIRECTORY --report-sha256 ACTUAL_REPORT_SHA
```

`... cuda`预期ValueError拒绝；不是当前可运行HOST命令。不生成prepare/泛manifest，也不绕过P1/freshness。

待MAIN认证：完整实物checksum/许可/可信反序列化与依赖类/loader；原生solver revision及配置/bounds/bank真实来源；encoded input的RGB/预处理/goal/source身份；embedded-low与当前共同controller逐tensor/normalizer坐标。Hi原生单帧high与low零a_hist不等于现三帧非零history，需明确选择共同合同。现安装CEM源码会按info_dict首value长度取n_envs，而原high_info首项是字符串high；认证solver时须核实际依赖版本/该接口接缝，不能在此偷偷重排或修原policy后称原生通过。任何不等价须新共同backbone/合同，不训练mapper，不用旧结果换绑。

本实现不提供三臂科学或真实neural readiness；最小后续是真实可信loader绑定后的单次high捕获，由MAIN主机CUDA执行并另审saved receipt，随后才把同一target交给共同controller并补实际delivery receipt。

## 本轮实际验证

指定内层venv专属回归 **36 passed（1.97s）**；CLI --help退出0。覆盖原生_plan_high/get_cost_high/rollout_high/predict_high方法体在NumPy shim上的完整capture→真实NPY/JSON/journal布局→消费与CLI实际路径/SHA接缝；receding/action_block/空warmstart；原macro与量化macro区分；真实首target而非最后或goal；候选与高层module完整链；entry/return/cap/host-return后存储失败、hook清理与拒绝吞cap重试；对重封journal的伪counter及跨join篡改独立拒绝。模型、神经、GPU、physics、网络及pickle均未运行，只有pytest临时fixture产物。production仍明确blocked，不把测试数字当真实high资源或E1证据。
