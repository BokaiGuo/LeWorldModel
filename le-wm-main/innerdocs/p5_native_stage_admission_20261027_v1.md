# P5 native contact：saved-data 阶段准入 gate

2026-10-27。新增独占同 stem 的 consumer、tests、本文和 JSON 小报告；不修改既有
executor、compiler、几何审计、封存产物或共享总账。使用 scientific-critical-thinking
区分证据完整性、机制是否被实际激活、科学准入三层；没有新增效果阈值。

## 已实际交付与运行

入口：`experiments/spectral_reachability/p5_native_stage_admission_20261027_v1.py`。
只导入 Python 标准库，不导入或调用任何模型、环境、物理引擎；无 CUDA、训练、网络。

固定八个摘要/ready 文件的路径与文件 SHA，复用已存在的 outcome/geometry/ridge/FD
报告表。再逐个重验 geometry 的完整 94-file JSON 闭包：48 个 public states、45 个
commands 和 executor report；其中 executor 与摘要重复，合计 **101 个不同输入文件**。
不自动找 latest，不跨 run 回退，不把 canonical selection digest 当 receipt 文件 SHA。

实际报告：`results/p5_native_stage_admission_20261027_v1.json`，SHA256：
`68196ce549ae9debab6e6b73948d6a278bcf38dda3cfc4038bd65493f15df3de`。

```json
{
  "status": "evidence_consistent_scientific_admission_blocked",
  "all_checks": true,
  "check_count": 45,
  "input_files_checked": 101,
  "bytes_read_including_drift_check": 1732923,
  "P5_admission": false,
  "C2_established": false,
  "theory_failure": false,
  "execution_authorized": false
}
```

45 项为本 consumer 的检查数，不与上游 542/60/91 项相加冒称独立重复验证。
新 tests **40 passed in 0.11s**；与现有纯几何 tests 联合 **46 passed in 0.23s**。
fixtures 仅用于接口、算术、拒绝错接/缺失/损坏测试，不是接触或模型有效性样本。

## 真实证据逐项核对

| 项目 | 当前可确认的事实 | 不能替代的证据 |
| --- | --- | --- |
| 身份/冻结 | ready 文件 SHA 与 executor 相同；三臂同 root `033865ef3b5b6d6e4ca52db13572383eade41d4494a1ebea222a9d9e3aa55af3`、seed2026102701、decision10/terminal15；selected packet 与 compiler/outcome/ready 相同 | 三臂不是三个独立根；已有打开的 root 不是新 outcome panel |
| 编译/动作 | `cuda_main_v2` selected replay exact；13predict/10JVP/20autograd/1encoder(4RGB)/1proprio；实际15条 raw commands 各等于冻结分支 | global patch goal-MSE 是混合场景代理，不是“先接触”的约束或物体收益保证 |
| 同 chart FD | receipt 文件、compiler manifest、input digest、model state 身份一致；40predict、0encoding/AD/physics；91项原 saved-array checks 通过 | 相同模型内部 FD/JVP 一致，不是模型导数对真实 contact/pose 的校准 |
| 原生工程执行 | 三臂 completed、16帧、11个prefix byte gates、close attempted/completed/durable；45controls/453substeps；原 outcome 542项通过 | 只确认该次执行记录，不授予新物理执行权限 |
| 接触几何 | 原 geometry 60项通过；native relative=True、scale100、sim100Hz/control10Hz；decision净间隙8.4507323356 | 重建子步不是新观测，不是连续 swept-path 证明；callback计数不带精确碰撞对身份 |
| contact/物体 | control1..10 contact-point 累加24，各臂suffix11..15均0；suffix物体原点/角度逐位静止；agent路径不同 | 不是“整轨迹从未接触”；也不是“已接触但推不动” |
| 真任务指标 | 从float64 public pose重算decision/terminal XY、principal angle；compiled减两控制逐项复算为0，coverage差也0 | coverage取原生保存结果，本 consumer 不重算多边形交面积，不把native done重定义为成功 |

三臂 reconstructed suffix 最小净间隙：compiled **7.0952825597**、nominal
**7.4046015747**、factual_reference **3.1529822664**（原生坐标单位，不是224图像像素）。
终态三臂物体 XY误差 **156.8583147194**、角误差 **1.7828973626rad**、coverage **0**。
这些量解释本次没有suffix推动；不是对理论或所有候选/根的不可能性证明。

## 机器报告中的六个具体阻塞槽

这些是该固定快照尚未建立的证据义务，不声称全仓库或未来永远缺失，也不是读完
当前结果后提高/降低原门槛。gate 不接收一个可手填 `passed=true` 的新协议来放行。

| blocker ID | main 作出独立协议决定前需要的确切证据/判据 |
| --- | --- |
| `native_contact_task_witness` | 新独立封存 native task/root 协议绑定合法动作、初始几何/速度和预算；存在接触→物体响应→任务可达的见证。现有prefix接触不认证当前suffix能操纵；不以本方法未来outcome挑根，不修改本次动作重追成功 |
| `task_metric_alignment` | 在 outcome 前独立校验部署可用目标/响应量与接触下物体任务进展的对应；隔离agent-only图像变化。现有几何距离和JVP并未形成真实contact方向导数证据，不能把二者直接相乘宣称机制已对齐 |
| `same_root_C2_conjunction` | 同一准入根上，部署可选的非贪心动作同时改变未来可达能力并改善真实物体任务；与冻结greedy/nominal及强匹配控制比较，沿用原预声明margin/CI，不能调beta或contact阈值制造通过 |
| `independent_calibration_outcome` | 新起终分布、root身份、horizon/termination、动作域、全部资源单位及 calibration/outcome 分割先封存；本次已打开单根不得重标独立测试根，保留全部失败与反转 |
| `architecture_provenance_and_replication` | 独立冻结架构/权重及合法来源链、各自P0/task前提；同核心程序/机制在独立任务/根复制真实收益。严格load和相同模型状态hash已有，不再写成缺loader；它们不是官方权重来源认证或P5复制 |
| `matched_baseline_resource_contract` | 同checkpoint/root/goal/feedback/action域强基线；predictor、编码、JVP/VJP/backward、搜索、加载与墙钟分别核账。现有三臂是局部诊断控制，不等同强基线独立成套对照；不能混入paper-reported数值 |

可复用的真实执行入口已经存在：`p5_native_c1_executor_20261027_v2.py`；几何入口
`p5_native_contact_geometry_main_20261027_v1.py`；同chart FD consumer
`p5_prospective_c1_native_fd_20261027_v2.py`。无需再造wrapper才能描述缺口；真正缺的是
新独立任务/root的机制前置认证及以上绑定，不是重复当前已完成的接线工作。
本任务不创建新根、不选更有利结果、不运行这些神经/物理入口；GPU仍归main统一。

## 可复现命令与退出合同

cwd：`/home/gbk/code/LeWorldModel/le-wm-main`。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_native_stage_admission_20261027_v1.py tests/test_p5_native_contact_geometry_main_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_native_stage_admission_20261027_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_native_stage_admission_20261027_v1 --require-admission
```

前一gate命令仅 stdout，exit0 表示保存证据一致，**不表示准入**；后一命令实测
exit3，表示证据一致但科学准入阻塞。缺文件、SHA变化、JSON重复key/非有限数、
分支缺失、错误身份、计数/指标不一致均fail closed（exit2）。不允许路径越界或
symlink输入，单文件≤1MiB、含复读累计≤8MiB；结尾重验SHA防运行中漂移。

实际报告的首次落盘命令（已执行，**不要覆盖重跑**）：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_native_stage_admission_20261027_v1 --output results/p5_native_stage_admission_20261027_v1.json
```

输出采用exclusive create，只接受同stem文件名；重跑用stdout，或由main指定新独占
目录下同stem文件。保存JSON约12KiB，代码/tests/本文/报告合计远低于1MiB。

边界：这是已有SHA-pinned审计的stage consumer，不重新遍历全部NPY、journal、软件
seal或大权重；不把上游布尔值伪装成本次重做全部审计。实际重做的是JSON闭包SHA、
跨报告身份、raw actions、contact/物体静止/pose指标和paired差值算术。
P1 root-preconditions doc继续冻结，SHA仍为
`60238c499e093efd6d90c543481955405b834ecfa2543a7bdc1e6203db2fd1ac`。
