# P2 无组合 SRA consumer：严格拒绝准备版

本交付不提供可运行 baseline。它输出 `rejected_missing_native_sra_contract`，
不是把 random/CEM/flat bank 改名。仅使用原四个已用 calibration roots，
不训练、不下载、不运行 CPU/CUDA 神经 forward，不改变旧源码、封存或共同文档。
检查祖先和项目目录未找到 AGENTS.md；沿用研究计划、严格 P2 合同和 admission 边界。

科学定义：对同一完整 root predictor context 下的整个 H2 动作 U 建联合终点
Jacobian J_H2，在冻结 Q 的任务响应 Q@J_H2 的右奇异动作方向中参数化一个
完整程序，并使用同一坐标下的有界阻尼 goal-progress 更新。不得枚举 prefix、
为每个 prefix 重编译 suffix 或按中间 reachable value 选组合。H2 的两次 predictor
recurrence 不等于多程序组合；非组合 witness 应是一条含全部十步命令的边。

现有 `spectral_bank` 只接受2/10列 Jacobian；其 `compile_arm` 为每个 prefix
重算 suffix，是两阶段编译，不能原样作为无组合 SRA。`chart_matrix(horizon=2)`
确实提供20维坐标，但没有冻结 H2 名义轨迹、谱秩/退化奇异空间约定、半径/阻尼、
四十候选槽位及预算分配。故本版不擅自将 full10 超参数迁移到 joint-full20，
也不伪造任何候选或 action receipt。这是现有前置合同缺失，不是数学上无法实现 SRA。

可比较输入逐根保留：原 root SHA、完整三帧 latent/两包 between-state actions、
root/goal RGB SHA、goal_vectors[0]、Q projection/metric SHA、normalizer、原 seeds。
动作合同为 raw float32 [-1,1]^20、time-major [2,5,2]→[10,2]、两包十步。
原预算是40个完整 H2 候选＝80 planning predict，加4 independent replay，
不是80个 H2 候选。相同边界不意味着谱球与 box proposal support 相同，
旧 composition 的16-command bank 更不是共同 search support。

机器产物包括 definition、五类稳定 gaps、逐项 proposed_sra_limits=null 和
reference_prediction_only。明确列出 predict、compilation predict、Jacobian、
scalar-output VJP directions、backward、SVD、bounded solve、encoder、optimizer、
candidate/expansion/selection、FLOPs、upstream、physical steps 和共同 wall cap。
未知不是零；new_work=0 仅表示没有执行 planner/model/physics，并不声称文件哈希、
JSON校验或 prepare wall time 免费。不能据80+4宣称全资源 matched。

prepare 只核验显式历史 full20 protocol/prepared/report、原 compilation、input
bindings 和旧 source seals；纯数据验证保存的候选/见证，不执行保存源码或模型。
不打开 physical report/新 outcome，不根据分数选择根或新方法。
另绑定既有 admission 拒绝及其原 prerequisites；这不是重跑 admission 物理审计。
新的 module/test/本文、相关旧模块和严格合同镜像到独立 source seal。
verify 检查当前源/镜像/args、全部输入哈希、protocol/report绑定和原根合同。
哈希不证明执行真实性、预 outcome 时间或独立复现；四个已用根永远不是新独立根。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_no_composition_sra_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_no_composition_sra_20261027_v1 prepare --output results/p2_no_composition_sra_20261027_v1_prepared01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_no_composition_sra_20261027_v1 verify --output results/p2_no_composition_sra_20261027_v1_prepared01
```

退出2＝输入/来源一致，但科学拒绝，绝非实验成功；退出1＝非法/缺失/变化输入。
prepare 输出必须全新，失败保留 invalid.json，不能复用或修补旧目录。
`cuda --output <上述目录>` 也只验证并拒绝，在 allocation/model import 之前退出2，
不创建 started，不执行神经计算。没有用于主线程运行真实 SRA 的合法 CUDA 命令；
必须先由另行批准的冻结 joint-H2 方法/资源合同及独立实现补齐，不能覆盖本拒绝版。
selected_actions API 无条件抛错，不可交给 physical consumer。

测试覆盖现有保存输入、四根/目标/context精确保留、未知预算、独占输出、篡改、
源码镜像损坏、严格JSON以及干净子进程禁止导入 neural stack。仅验证合同拒绝，
不证明算法效果。P1/P2 admission、prediction/full-budget matched、physical efficacy
全部为 false；既有 admission 的其余缺口不因 prepare 消失。

本次实测：新增15项与原 MPPI/full20 sampling/admission/严格 P2 合同联合
148 passed（21.69秒），无神经实验。联合命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_no_composition_sra_20261027_v1.py tests/test_p2_full20_mppi_20261027_v1.py tests/test_p2_full20_sampling_20261027_v2.py tests/test_p2_composition_admission_20261027_v1.py tests/test_d2p_p2_comparison_contract.py
```
