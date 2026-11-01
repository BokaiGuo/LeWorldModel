# Commoncontroller 原 raw15：六槽/四分支原生物理适配器 v1

仅新 source/tests/innerdoc，旧源、结果、protocol 不变。批准协议为本目录 `agent_p2_commoncontroller_physical_protocol_20261029_v1.md`，固定 SHA `42e082498484f8bb399f819d31531c2a77f73b161e0bd2e15bb8ec900f5e8de6`。使用 experiment-code 的源先行、日志、RAM回归；不采用训练/调参/正结果选择。worker不运行生产prepare、真实环境、NN、physics、full strict；MAIN审读后执行。

## 实际 authority 与 ABI

Pauli source SHA 固定 `19b8b36337de68233e504a647b66b468e0fde95608a655e8e57803886b938481`。已存在实际 paid prepare `f0c518c6f3c5edc979a69b35dbde50b71a319b9237d8f0963623f72f18aab9a0`、report `6f1f153863d7e30b8297a03b7b10c5d5297d16cf0c5f2cefdf301b3e78772719`、receipt `a8fe4dd9c8c90ed14a5bbecdcb226ec4d7626c786fe4db1223194f7e3f6d87b1`，CLI外部显式传入实际路径/SHA，不填未来值。

实际 receipt **无 schema 字段**，`status=all_saved_paid_replays_verified_with_explicit_blocked_slots`；完整四 local_checks、6/4returned/0blocked/2missing、24predict/24samples、24AE/24projection、96AE tokens/72predictor context tokens/72projection rows。`original_Hi_scaler_certified/raw_deployment_authorized/scientific_admission` 均实际 false，严格要求这些真实键存在且 `is False`，不补 flags、不更改上游授权。

`material` 在导入 paid module 前核 source/protocol/kernel SHA；调用 paid 的真实 `verify/validate_files/analyze_saved` **无写入**重建完整 saved receipt，附其真实 manifest/report SHA 后，与已保存 receipt 的完整 JSON字节编码比较。不是调用会写文件的 `paid.consume` 假装 getter。全部四 replay完整之前不构造任何env。此 saved strict只由MAIN调用；worker没有运行。

从每槽真实 trace 的独立 CUDA add 输出 descriptor 取原 `[15,2]` float32 raw，并读真实 z/a/target/mean/std 输入NPY；取的是原raw120bytes，不再inverse、normalize或tanh。descriptor作为本adapter显式投影输入，不给原 report补造字段。归一化token差异沿原receipt保留；不复用旧mean预测成本。

原物理 prepare 固定 `a4fb18b5aedd53e1820717f827d4703774b8c41dd08b0fd3974cca58593ade93`，用于完整任务/原program context/goal/source/warmup绑定，不继承任何旧物理结果。source_material读原真实source，不构造env。核完整原task/blind/source、real history3与controller z/action2 coordinate、原goalRGB/native7/root。latent target不会写进native goal。

own spaced JSON seal 与 paid compactJSON/NPY byteSHA编码各用各的。prepare绑定完整上游sources/模型文件SHA、原物理bindings、actualreport目录全集、协议及新三件套；run/consume均重新核验，无全局patch。

## 执行与 first-step guard

自有 `.execution_plan/.prepare/.report/.consume`，六槽原序：root0 prior/program/arbitrary、root1同三槽。ACTIVE固定 `[0,1,3,4]`，2/5完整missing row且无env/operation/endpoint，不压成4个独立根。

每active槽独立原生reset/place/warm10，再原raw15全执行，原cp0/5/10/15及每时刻off/on/off。复用冻结 `kernel.observe`、native placement/hooks/dispatch、labels与metric helper；本版局部episode/execute/schema/chain/六槽validator，不调用旧 fixed4 `validate_saved/audit_chain`，不伪造其planner plans。`Meter`复用的冻结计量内核具有完全相同四分支物理caps；本版增加guard，不改变科学阈值。

`RecordingSink`只在实际写入/fsync返回后复制prefix payload。plan step1的 `Meter.call` 在父类创建 intent/调用callback **之前**，复核真实warm33+plan0三帧、十条连续实际raw返回、三帧history pixels、root/state/**完整snapshot**、renderer、goal及once-normalized action2。写/fsync accepted guard成功后才设置内存accepted标志；guard计算或写/fsync失败绝无step1 intent/callback。所有后续plan step也要求本槽accepted。

warmup用placement goal，warm10后 `bind_goal` 才挂原task goal；因此前后 **goal_state可以不同**，full physical snapshot/goal_pose/renderer必须不变，不把整个invariant强等或仅state7相同代替snapshot核验。plan0/current off pixels须等原history末帧。source只保存17个原warmup参考，本次仍真实拍33帧；原不存在的on参考标明unpaired，不造33个old match。snapshot含原生完整动力学元数据，不能以仅state比较替代。

strict重算guard，仅去已认证journal envelope三键 `index/previous_sha256/event_sha256`，不丢payload字段；核全部warm/plan0证据先于guard、guard先于所有plan intent。新hashchain按ACTIVE实际非连续slot身份和四个branch_closed顺序认证，不把slot3误当第3索引2。完整事件白名单、nested资源attempt/return、原native phase-goal/reward/stop、原rawbits、PNG/file/pixel/fullsnapshot、文件全集、原始source前后均核；unknown拒绝。

原生早停不padding、不补cp15，六分母保留；发生错误不退款，错误/清理/未启动保留。partial可严格检查其已保存部分，但不是成功release；不合法action/render/无法闭合日志继续拒绝。极端写/fsync/磁盘故障可能留下不能完整消费的partial，不以跳过完整性签receipt。

完整四槽预算保持协议：4env/4close、8reset、100public、1016physics、16set_state、12goal_setter、8goal_render、4goal_bind、324explicit_render/PNG；wall900s、64MiB含8MiBfinalreserve、floor200MiB、完整prepare128MiB。上游paid/controller成本独立完整继承，不重执行NN，不将unknown FLOPs补0。新driver 0model/encoder/AD/训练；不能声称等总预算或官方Hi scaler身份。无P1/P2科学准入、freshness、独立根确认或equalFLOPs。

## MAIN完整CLI

以下生产命令worker未执行。所有输出必须不存在，失败不覆盖。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_commoncontroller_physical_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_commoncontroller_physical_20261029_v1 prepare \
  --replay-manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_prepare_MAIN01.json \
  --replay-manifest-sha f0c518c6f3c5edc979a69b35dbde50b71a319b9237d8f0963623f72f18aab9a0 \
  --replay-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_cuda_MAIN01/report.json \
  --replay-report-sha 6f1f153863d7e30b8297a03b7b10c5d5297d16cf0c5f2cefdf301b3e78772719 \
  --replay-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_consume_MAIN01.json \
  --replay-receipt-sha a8fe4dd9c8c90ed14a5bbecdcb226ec4d7626c786fe4db1223194f7e3f6d87b1 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_prepare_MAIN01.json
# OWN_MH填实际上一命令返回SHA，不填未来hash。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_commoncontroller_physical_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$OWN_MH"
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_commoncontroller_physical_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$OWN_MH" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_MAIN01 \
  --authorization MAIN_COMMONCONTROLLER4_FULL15_PHYSICS_ONLY
# OWN_RH填实际物理report SHA。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_commoncontroller_physical_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_MAIN01 --report-sha "$OWN_RH" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_consume_MAIN01.json
```

`consume(output,report_sha)` 返回own `.consume`，rows[6]/executed_slots[0,1,3,4]/missing_slots[2,5]/counts/errors/saved_PNG_verified/planner_authority/bindings/metric_breakdown。每active row含实际cp0/5/10/15、trajectory/native responses/initial_history_guard；missing row final_state/native_success为null，不能转换为零收益。metric_breakdown六项，缺终态metrics=null。报告本身status仅工程recorded/partial；所有科学flags false。

## RAM测试与未完成证据

初轮46 tests通过22.21s；包括四分支完整RAM full15与324PNG saved consume、paid真实receipt shape/突变、raw/context/task/goal身份、首guard拒绝无plancallback、native早停/异常及重封hashchain篡改。后续新增真实saved selected-array assemble（不调用verify/consume/source_material）、真实临时Store的guard write/fsync失败回归、fullsnapshot非state-only与phase-goal区别、ACTIVE非连续闭包篡改。新增窄测试7 passed/4.69s；全量53 passed/22.55s。冻结前仅清理继承的旧单数arm标签为commoncontroller_returned_mean_raw15并加断言，重跑最终 **53 passed/22.59s**（session8564，终态exit0）；随后仅删除源码EOF多余空行和补本文记录，未更改可执行逻辑。全部使用上述pytest命令，测试不是物理证据。

实际Pauli前置已完成不等于本driver已实跑；本版生产prepare/fullstrict/四env guard/物理结果均待MAIN。没有任何未生成的预测/物理SHA或伪ready状态。
