# All-mode prefix：固定42提案的 H2 conditional 实现

这是两known roots×off/on四项校准的实现跨度消融，P1/P3=false，不是四个独立根。
MAIN 已报告原rank3遗漏J能量仅0.56%–0.88%；本实现不把覆盖小能量方向宣称为理论修复。
不按已知物理成功、目标代理分数或谱能量选择mode/root，无训练/Qfit/readout/normalizer修改。

## 算法合同

直接复用 `pusht_full_packet_compiler_20261027.spectral_bank` 的 QJ SVD、全10个
canonical-sign right vectors 与原 bounded DLS。只将 root prefix 枚举扩为：

`zero, DLS, radius0×mode0..9×(-1,+1), radius1×mode0..9×(-1,+1)`，固定42项。

半径原值 `.25*sqrt(5)` / `.75*sqrt(5)`，cap为后者，damping=.002。
全部10mode都保留，包括零奇异值和重复动作，不按rank、能量或outcome裁剪。
重复奇异值下basis的非唯一性沿用当前SVD返回值与原符号约定，不冒称可识别科学方向。

compile_arm 移植原 `p3_goal_patch_consistent_compiler_20261027_v2.compile_arm` 流程：
每prefix实际推进后，重新计算自己的conditional nominal/J并调用原 bounded_ridge。
原 identity192 Q、goal、beta=1与normalizer不变。
选取仍为 `stage_cost + own_suffix_ridge_reach_value` 的稳定argmin，**不是**偷换成真实终点或非线性terminal最小。
全部42项保留预测、J、suffix solve、objective、nonlinear cost和独立H2重放；
winner选取后先记 selection_frozen，再做独立wholeH2与两段H1重放，完整context必须相同。

无 monkeypatch 旧module或NumPy/Torch globals。原 with_svd_receipts 的全局替换不调用。
root_bank完成意味着冻结 spectral_bank 内2次SVD完成，42个suffix_ridge各有1次SVD。
记录helper真实attempt/return；若helper内部失败，内部SVD尝试次数标unknown，不编造精确计数。
J通过显式torch.autograd.functional.jacobian(vectorize=True)，记录43次显式调用与每次192方向；
不宣称独立拦截了engine内部backward API，report该字段为null。

非零动作在NumPy只做一次 tanh(float32 params)，转成literal float32 raw，
原值封存为NPY/JSON后给模型、replay与plan派发共用，不做CUDA第二次tanh或/100。
只有零点logit Jacobian路径用CUDA tanh(u)，其实际raw buffer也保存；u=0时raw为0。
模型全部frozen eval、CUDA-only、MATH、TF32=false、autocast=false；无CPU neural fallback。

## 固定四输入与closure

使用原overlay compiler封存的 `task_00..03.json.gz`，固定slot0off/slot0on/slot1off/slot1on。
原report：`results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json`，
SHA `2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381`；
原prepare SHA `8d622196815f90476d02d7ca6750817116e2100e81945f192a46f740156a9a20`。
build 调原saved consume完整验证原16图/模块/task，再按身份取4个blind task，
保留同condition context/goal/Q/task_id/blind_sha；不读取物理outcome来定方法或分母。
没有新增编码；历史16encoder/projector与旧planning/source成本已付但不等于零。
自己的prepare绑定新source/tests/doc、原task与父prepare/report及完整传递source/model closure。

## 逐轴预算（不等总算力）

| 新调用轴 | 每root×condition | 固定四项 |
|---|---:|---:|
| predict API / predictor samples |258|1032|
| action_encoder / pred_proj，各自 |258|1032|
| Jacobian 显式API |43|172|
| 数学VJP方向 |8256|33024|
| 成功冻结helper内SVD |44|176|
| 新encoder / projector / physics |0|0|

258=`root nominal+ADforward 2 +42×(prefix1+conditional nominal1+ADforward1+suffix1+H2 replay2)+winner replay4`。
J=1+42；SVD=2+42。不是单次42shot后沿用旧66predict预算，不称与GN/CEM/旧D2P等FLOPs。
实际completed/attempted module及AD/helper记录单列，失败不退款；helper内部失败的SVD为unknown。
saved consume的bank/ridge重算、父saved consumer验证与源哈希工作另付纯算术/I/O，
不是新神经调用，不能据此宣称端到端零开销。每次prepare/verify重建也会重复付费。

## 执行与失败

实现自己的prepare/verify/run/consume CLI。prepare≤2MiB；output≤128MiB，
其中64MiB final reserve；disk floor200MiB，启动预检需floor+cap；预声明wall1800s，
是predispatch检查而非对阻塞CUDA调用的抢占，MAIN需外部watchdog。
prepare文件名必须以 `agent_allmode_prefix_20261028_v1_` 开头并以 `.json` 结尾，
自己的 write_new 先核prefix、canonical bytes类型与cap，再调用旧纯writer的独占/磁盘检查；
不借旧模块的NAME或改它的globals。interface Q显式转为原值JSON list。
完整native AE/predictor/pred_proj输入输出以原float32保存，无精度压缩或只留terminal。
API raw/input/return、AD J原NPY、全部compiler事件先持久化，四个row预建，失败不换根/重试。
初始source verify拒绝在输出创建前抛错；执行后的source/model/hash/journal/report异常保留原首错与追加错误。
report写失败时旧journal/数组不删，并向stderr输出错误；不能将工程异常当科学负结果。

strict consume验证完整artifact census/hash、每次raw→token→AE→predictor→projector、
全部258预测输入与返回、全部43J、bank/solve/selection/replay算术与日志、before/after tensor hash，
不运行模型。只在四项全齐后释放4个model_plan，status=`all4_allmode_commands_verified_not_admitted`。
plan稳定键(root_slot,condition,arm)，含task/blind/source身份、raw10、selection_digest及whole_run_report_sha256。
这是新planner schema，不能借用旧白名单consumer或旧physical endpoint；后续physical接线须另行审读实现。

## 测试与MAIN命令

使用experiment-code技能，按本任务限定不训练、不生成结果图、不派生agent。
RAM测试是NumPy/state-machine/device-labelled接口stand-in，不是CPU神经或真实native物理。
覆盖完整42提案及4项258-forward schedule、full native arrays、零/重复方向、稳定tie、
非winner篡改、独立重放、逐阶段失败与部分数组、全四项strict consume、CLI RAM prepare/verify。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_allmode_prefix_20261028_v1.py
MOD=experiments.spectral_reachability.agent_allmode_prefix_20261028_v1
.venv/bin/python -B -m "$MOD" prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m "$MOD" verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_NEW_PREPARE_SHA
.venv/bin/python -B -m "$MOD" run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_NEW_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v1_cuda_MAIN01 --authorization MAIN_ALLMODE42_PREFIX_CUDA_ONLY
.venv/bin/python -B -m "$MOD" consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_NEW_REPORT_SHA
```

worker不运行真实prepare/GPU/physics；新SHA只能由MAIN实际执行后提供。全部工程验证通过仍非P1、独立泛化或理论修复证据。

交付验证：**38 tests passed**，真实saved-only build/canonical回归通过，
全部4项prepare canonical为 **679,559 bytes**（未写prepare），小于2MiB。
包含原生shape全四项RAM saved-consume、own writer prefix/type/cap与CLI回归。
测试和saved重建通过不等于真实CUDA完成；本worker未启动任何神经或物理作业。
