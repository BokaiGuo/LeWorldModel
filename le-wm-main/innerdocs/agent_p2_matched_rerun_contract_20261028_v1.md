# 两根 moving / beam 与 no-composition 新执行前合同 v1

这是可运行的 prepare/verify 与旧执行接口接线清单，不是新增 planner、审计-only inventory 或 GPU launcher。遵循 experiment-code 技能的事前协议、实际输入和日志要求；本任务只读已有 metadata/source，RAM tests，不运行神经/物理/下载，不要求图表或非零结果。仅新增本stem三文件，旧源码、prepare、结果和共享文档不改。

## 固定全部任务与接口

严格读取两份原 prepare（路径、外部SHA在源码 PREPARES），它们的slots必须逐字段完全相等且顺序0/1。使用真实moving task0/1，task SHA分别 a42bfbb1… /101add21…；闭合原完整task_id、blind、task_contract、原goal192/identity Q、实际非零raw历史和normalizer、with_target=False。共四行：两个slot各含 matched_domain_H2_beam20x6、joint_H2_spectral_packet_mapped；不按旧效果选择。为保256KiB cap，完整数组只保原文件SHA+pointer和canonical value SHA，不重复内联identity192；verify始终重读原全精度数组，不量化或裁剪。

原可调用实现：

- experiments.spectral_reachability.agent_kinodynamic_matched_domain_20261028_v1：prepare(output)、verify(path,expected)、run(prepared,expected,output)、consume(output,expected)。固定20→beam6→120、whole2+segmented2，保留原seed/bank和所有动作见证。
- experiments.spectral_reachability.p1_moving_no_composition_baseline_cuda_20261027_v1：同四接口。原40个谱候选、nominal/full-H2 J、whole+segmented replay；原映射到逐packet球，不改梯度/谱bank/normalizer。

consumer 用 AST（不import两执行模块/torch）导出实际函数参数和精确源码行、SHA，生成真实CLI argv templates；NEW_*与ACTUAL_NEW_*是必须由MAIN新prepare/report回填的变量，绝非存在的结果或SHA。旧CLI能重跑算法，但**不满足新同buffer加载证明与完整模块计量**；所以本合同不直接授权这些run命令，不能仅运行旧CLI就声称补齐证据。MAIN需在独立新executor桥中接线，禁止改封存或猴补globals。

## 预算：统一axis schema，不擅自统一配额

| 每root/arm轴 | beam原声明 | no-composition原声明/固定图 |
| --- | ---: | ---: |
| predict API / batch1 samples | 144 | 88 |
| action_encoder / predictor / pred_proj | 各144 | 各88（predictor/AE旧计量；独立API/proj/sample需新hooks） |
| AD-forward（已包含上列，不再相加） | 0 | 2 |
| J / VJP | 0 / 0 | 1 / 192 |
| SVD / bounded solve | 0 / 0 | 2 / 1 |
| 新encoder/projector/physics | 0 | 0 |

88=nominal2+AD2+40candidate×2+winner whole2+segmented2。beam=20+120+2+2=144。保留两者不同AD成本，不声称等FLOPs、相同候选覆盖或等API。spectral旧predict字段实际计predictor模块入口，不能直接当已独立核过的API/proj次数。新向量中这些88上限标new_hook_required_from_fixed_call_graph，不伪装旧实测。

inherited实际来源为原moving encoder report /resources 外部SHA e7fb5543…，严格核8encoder/8projector；每root归属4+4，跨两方法共享一次，不能算成16+16真正新编码。其余上游成本未知保null，不填0。所有原零均是算法合同禁止的新操作，不是旧unknown default。

每job：beam原64MiB/900s（dispatch/write检测，不可抢占阻塞native call）；spectral原8MiB、无wall声明。不能把900s广播成每root或原spectral cap。统一资源向量的per-root wall_limit=null。prepare文件256KiB、200MiBfloor，保全精度与全部四臂。新watchdog若需要，必须由MAIN在新executor协议事前冻结。

### 必须由MAIN预先决定的144与88权衡

1. 保留原144/88：现有算法不变、能重跑；只做同task控制对照，明确API与AD预算不同。
2. 要求真等144：spectral原bank只有40候选，必须另冻结新的科学搜索协议；不能padding、额外best replay或把上限144说成实际144。
3. 要求真等88：原beam有140planning+4replay，不可能原样执行；必须另冻结减小search的协议。
4. 共同上限144不等于相同实际调用，仍是144/88；本工具不代为选择这一新决策。

默认prepare保留未决且blocked。可选decision只接受明确“保留原不等配额”；不接受任意新限额。该decision解决预算政策选择但不消除新executor/loader/hook缺口，也不开放科学门。

## 新加载字节与实际计量应如何收集

已绑定原 experiments/checkpoint_compat.py 的SHA和load_compatible_pretrained精确入口。原loader读取config文本、随后torch.load(path)另开文件，兼容键重映射后无missing/unexpected，冻结CUDA模型；没有同一已hash消费buffer的receipt。pre/post文件SHA加tensor hash只能限定一致性，不能偷换原加载字节证据。

新MAIN批准的独立loader adapter应保持原config实例化、key remap/无missing-unexpected、device/eval/frozen/插值语义，但显式读取config与weights各一次，hash其**实际将交给JSON/torch.load的同一buffer**，journal先load intent、再文件身份/长度/SHA、hash_buffer_sha与deserialize_buffer_sha、成功/失败；反序列化使用同buffer，不能hash后重开路径。CPU tensor staging是原torch.load map_location机制，不是CPU神经推理；神经操作仅HOST CUDA，无fallback。留存adapter自身最终sourceSHA、原loader语义SHA、环境/模型类版本及字节receipt，模型前后tensor hash再与原预声明hash核对。不能用新文件冒充原loader已产生receipt。

本工具提供 validate_loader_receipt 的**schema-only**严格正负验证：全路径/全文件、同buffer digest、CUDA、冻结、前后tensor hash、run/contract绑定。loader_source_sha256指绑定的原语义基线；它不代替未来adapter实现sourceSHA/完整journal审核。通过此validator不是运行证明，不产生receipt、不对缺失的executor作已实现声称。实际新执行必须另封存adapter源码与receipt到其prepare/artifact链；本合同不接受它作为已有证据。

执行时逐root/arm/API及action_encoder/predictor/pred_proj挂真实pre/post hooks，batch1样本与API分开；AD阶段2forward包含88内，192VJP/1J/2SVD/1boundedsolve另列，不把API/模块相加当FLOPs。intent在调用前durable，return/failed绑定同ticket；全四行预置，故障不删除、不补跑、不换根。source/task/goal/Q不改。最终新report以prepareSHA、adapterSHA、loader receipts、所有操作ledger、输入输出/选择动作、partial/errors、模型前后hash闭包。这样补的是**新执行**，old_runs_recertified始终false。

## 可执行命令（inner repo cwd）

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_matched_rerun_contract_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_matched_rerun_contract_20261028_v1 prepare --output results/agent_p2_matched_rerun_contract_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_matched_rerun_contract_20261028_v1 verify --output results/agent_p2_matched_rerun_contract_20261028_v1_MAIN01.json --sha ACTUAL_PREPARE_SHA
```

如MAIN明确选择保留原配额，独立预封decision内容为：

```json
{"schema":"agent_p2_matched_rerun_contract_20261028_v1.decision","budget_policy":"retain_native_144_vs_88","acknowledge_unequal_API_AD":true,"wall_policy":"retain_original_scopes_no_new_wall_claim"}
```

再对新output执行prepare，加 `--decision /ABS/decision.json --decision-sha ACTUAL_DECISION_SHA`。verify从合同内外部绑定重读decision，不允许CLI覆盖。默认无decision的blocked产物也是有效工程交付，不是假研究失败。

始终run_authorized=false，P1/P2=false、composition_benefit=null。prepare/verify和mock通过不等于新运行已发生，也不能认证旧执行。
