# Constructive common-prefix physical probe v1

2026-10-27。实现依据完整`p1_constructive_common_prefix_capability_probe_design_20261027_v1.md`；仅此stem源码/tests/doc新增，旧source/planner/physical/sealed不改。不写独立auditor。不执行physics/GPU，MAIN review/prepare/run。使用experiment-code技能的先固定协议、真实输入接线、逐步日志及故障mock；不采用训练、画图或按结果调参要求。

## 固定小版本（合同不缩减）

两根、逻辑Z/G/D各三角色：Z真实5个zero prefix，G从原D2P固定10候选的stage-Q stable argmin，D原stage+own-V stable winner；同根G/D float32 raw/history必须相同，否则拒绝，不静默启动6位置。按root+goal float64 bits、原recipe/source/env/render/warmup、history合同、完整prefix float32 bits去重。保留所有alias、candidate index、原logits/hash。signed zero不规范化，不以近似状态去重，不跨根合并。

每唯一位置21probe：zero，然后j=0..9的`+.25*ej,-.25*ej`，float32 logits一次NumPy float32 tanh，time-major5×2、原`.75sqrt5`球、raw[-1,1]。不clip、不SVD/solve、不按效果选动作。当前预计M=4/84branches，若Z亦alias则2≤M≤4，逻辑角色始终6，独立根始终2。

每条从原reset seed和placement recipe重建：两reset、10真实zero warmup、绑定原goal、5prefix、5probe。仅复用旧physical的`load_source`白名单、`branch` recipe/fresh guard primitive，不调用其固定四臂runner/selection；新runner拥有计划、授权、预算/hook、prefix equality和输出。离线evaluator仅读certificate的goal/goalRGB/observed_states/root_history，不读取witness动作或future数组，不传hidden state给selector。原journal用于源history/原前5命令绑定，不以未来outcome选role。

PNG→fresh observation→guard→下一dispatch顺序不改；guard失败/原生终止立即停该条、不补试；所有计划行保留。G/D还与旧actual前5命令、step5 state64 bits/RGB原始bytes hash/已有block snapshot精确比较。所有probe与该position的zero分支step5比较两body pos/vel/unwrappedangle/angularvel/force/torque/sleep、state64、RGB、raw-prefix32、history，exact不符在suffix前记prefix_not_reproduced。完整solver/contact cache未序列化，不宣称状态setter完全重放证明。

## 预算与持久化

完整分支2reset/20public/204实际physics hook/22explicit render+PNG/2goalrender/4setter/1goalbind。M=4时168reset/1680public/17136physics/1848PNG/168goalrender/336setter/84goalbind。不是仅公式：嵌套native hooks记录每次attempt/completion、operation_index及branch身份，完整结果必须与所有caps精确相符。失败attempt不退款；嵌套调用、时间/磁盘或I/O错误停全run，不自动重试；guard/prefix错误可继续后续固定分支。未开始行不删除。

cap256MiB（所有PNG/journal/partial/manifest/report），单PNG≤1MiB，预留2MiB收尾；floor200MiB。启动至少456MiB，写入/dispatch逐次检查。wall≤600s（cleanup/最终封存允许越过dispatch wall，不保证600s内完整）。prepare≤256KiB。不降预算/guard/floor换成功。hashchain journal逐事件fsync、PNG和metadata manifest全hash；primary error不被cleanup覆盖，完全I/O失败时只能保留已写字节，不保证final report落盘。

新神经/模型加载/encoder/projector/predict/actionencoder/AD/J/SVD/solve/GPU均0；原source/planner/physical成本独立列出，其余已有encoding/feedback成本不归零、不重算。prepare仅hash本地模型身份，不加载权重。完整/partial/raw-negative分开，不从native0剔除有效endpoint。独立auditor属后续MAIN另派，不在本实现写集。

## 内置纯算术输出（不是独立auditor）

根原state/goal冻结共同位置方向u；zero范数为undefined，不换方向。所有21M条Delta4、agent/object分解、T、独立角chord响应、zero suffix drift；10列B/a/E按固定±对报告，缺zero或任一侧为null，a_L2只有全10列定义时计算。逐根跨position相同k差值完整报告；G−D标为结构同干预，不重复观察、不计probe为独立根。不拟合控制器、不选最好probe再执行。原native success/reward/state7 distance和wrapped诊断由旧branch逐步核验保留，不改多圈角bug、不以object-only替换native success。own进度表在row initial/final_metrics，与共同方向表分离。

结果仅这两knownroots/单尺度/有限5步干预描述，P1/P2=false；G=D不能提供非贪心证据，任何response大小不能追认P1。

## MAIN命令

cwd=`/home/gbk/code/LeWorldModel/le-wm-main`。本agent不创建prepare/results，不运行物理。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_common_prefix_probe_physical_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_physical_20261027_v1 prepare --sourcepack results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json --source-sha 9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9 --source-prepared results/p1_constructive_reachable_source_20261027_v1_prepare01.json --source-prepare-sha 98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086 --planner-report results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json --planner-sha 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d --physical-report results/p1_constructive_plan_physical_20261027_v1_main01/report.json --physical-sha d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 --wall-seconds 600 --output results/p1_constructive_common_prefix_probe_physical_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_physical_20261027_v1 verify --prepared results/p1_constructive_common_prefix_probe_physical_20261027_v1_prepare01.json --prepare-sha MAIN_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_physical_20261027_v1 run --prepared results/p1_constructive_common_prefix_probe_physical_20261027_v1_prepare01.json --prepare-sha MAIN_PREPARE_SHA --output results/p1_constructive_common_prefix_probe_physical_20261027_v1_main01 --authorize-physics
```

所有SHA来自显式获批旧receipt；不能glob新目录或重seal旧prepared。run缺授权在verify/环境创建前拒绝。CLI partial exit2，完整exit0仍不是科学准入。mock不是actual环境/hook或能力真实性证据。
