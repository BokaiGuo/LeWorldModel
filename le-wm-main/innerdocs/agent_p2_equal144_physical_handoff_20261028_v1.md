# P2 equal144 四槽原生物理交接

独占新source/tests/doc；不改任何旧源码、模型、Q、门槛或结果。prepare/verify/consume只读saved和NumPy；run必须MAIN明确授权，worker仅RAM测试。两根已知moving校准源，不是fresh；P1/P2/scientific_admission始终false，不宣称等FLOPs或完整D2P收益。

## 来源及实际接线

固定本次equal144 producer report SHA `7eeaa059d58895545340458455ddcaabceda4a813a41051e1a2c60f22bc10228`，MAIN全strict wrapper SHA `d8ded78d152938a3c6e5c244729f120d040c42666b44d9ed89bf2e0beca029dc`。重新执行封存wrapper.analyze（其调用producer.consume），整个persisted wrapper与实际return精确相等，再读取其native_strict_receipt，不把envelope误当原consume。绑定原producer prepare、完整source/model/task闭包、11个artifact与报告，以及本模块三件套。封存源码SHA固定，不提供换report/换plan选项。

完整root-major 2task×2arm=4槽：matched_domain_H2_beam20x6与joint_H2_spectral_top16_packet_mapped_equal144。每槽原selected plan/task/blind/context/goal身份、selection digest及原始float32 raw10 bit串保全。task_bridge核原3帧RGB、goal RGB、非零incoming raw10、task ID/contract、identity192 Q及packet cap；动作来自完整strict实际raw，不另做tanh重建。没有旧结果输入、旧endpoint回填或旧物理复用路径：全部四槽新执行，哪怕某槽动作与旧记录相同。

旧run_p2_matched_physical有模型加载/终态编码，故不调用。新execute/assemble/consume是冻结moving_prediction_matched_sampling_physical的四槽控制流程独立副本，改变仅本模块schema/arms/authority/status，未修改旧globals/execute。复用其Store/Ops/close_env及已封存moving kernel.branch/native_hooks。新material直接绑定原sourcepack与prepare，没有借用旧physical report。

每槽fresh native inert constructor（RGB224、with_target=False、render_action=False），原seed/placement/reset与原goal bind，原warm10：5×[float32 .1,0]再5×[0,float32 .1]。不是reset(savedpoststate)近似restore；按原recipe重建，并逐history0/5/10 RGB/state/dynamics snapshot门确认。goal只作原evaluator元数据/goal RGB，无goal_pose插入、不额外physics；随后严格原raw10，不padding/retry/extra replay。完整保留early native success/termination/truncation及engineering failure；不补齐短轨迹，不删分母。实际checkpoint0/5/10 RGB与7维native state（agent XY、block XY/angle、agent VX/VY）及原native labels保留；object_success未知保null，不以作者reported值回填。

## 资源边界与失败

沿用32MiB整个输出/512KiB final reserve、200MiB disk floor、600s工程wall；prepare上限独立2MiB，并有真实material尺寸回归，不删bindings省空间。wall从Store建立起包含dispatch/write，cleanup/footer豁免，不能抢占阻塞native调用。构造若失败，其内部工作unknown，不能填0；正常pristine constructor无physics经绑定源验证。hook在pristine constructor后，reset/set_state/goal_render/goal_bind/public_step/Space.step/explicit_render各自entry/return，失败不退款、close始终尝试、hook恢复。

四槽无earlystop完整预计：construct=close=4、reset8、set_state16、goal_render8、goal_bind4、public_step80（40warm+40suffix）、physics_step816（16setter+10×80）、explicit_render88。render与goal_render可能内嵌，不相加当总physics成本。实际早停用实际计数而非期待值；missing保持unknown/null。

继承模型预算单独记录：predict API576、predictor samples576、ADforward4已含在predict内、J2/VJP384/SVD4/solve2、shared encoder/projector各8只计一次；AE/pred_proj与API是重叠轴，不相加。新neural/predict/AD=0；wall不要求等，等总compute/FLOPs永不声称。

strict consumer检查外部physical report SHA→prepare重建→执行binding→有界journal链/完整嵌套实际entry-return→四槽原计划及实际raw contiguous order→checkpoint实际7维与native metrics→所有PNG文件/pixel SHA及最终closure。可验证partial仍标partial，不能变成科学通过；缺最终journal/receipt则拒绝验收。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。仅以下test由worker运行；RAM fake env不是物理证据。实际saved prepare→verify测试写pytest临时目录，不写MAIN results。以下真实prepare/run由MAIN审读后亲自执行，ACTUAL值必须现场sha256sum，不预填未来SHA。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_equal144_physical_handoff_20261028_v1.py

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_physical_handoff_20261028_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_equal144_physical_handoff_20261028_v1_prepare_MAIN01.json
sha256sum results/agent_p2_equal144_physical_handoff_20261028_v1_prepare_MAIN01.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_physical_handoff_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_equal144_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_physical_handoff_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_equal144_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_equal144_physical_handoff_20261028_v1_physics_MAIN01 --authorize-physics
sha256sum results/agent_p2_equal144_physical_handoff_20261028_v1_physics_MAIN01/report.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_physical_handoff_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_equal144_physical_handoff_20261028_v1_physics_MAIN01 --report-sha ACTUAL_PHYSICAL_REPORT_SHA
```

consume返回完整机器可读JSON到stdout；MAIN可在独占wrapper保存，不自动覆盖任何回执。剩余：MAIN审读、实际prepare/verify/run/strict；本交付不是物理结果或任务收益证据。
