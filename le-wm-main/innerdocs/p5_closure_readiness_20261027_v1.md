# P5 closure readiness：当前证据接线与依赖清单

结论：真实离线报告为 `saved_evidence_valid_P5_blocked`。已生成可复核的
`execution_manifest`，不是 GPU 作业清单或执行授权。129 份实际小文件的有限快照
接线通过，9 个科学要求槽仍未绑定正式完成产物。`unbound` 不等于证明全仓库没有
相关实验；本版本不自动扫描 latest，也不自动消费未来产物。没有理论失败结论。

## 要求从哪里来

先核对项目原定义与 closure，而非沿用上次 native gate 自定义的 blocker：

- 外层 `docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md:411`：
  P5 是其他冻结模型和任务类别，须预声明跨架构复制且保持 training-free。
- 外层 `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md:428`：独立冻结模型/架构复制；
  后训练 teacher、adapter、qpos readout 不能填主线。该文后续实际 C1 更新也已读，
  没有把已执行误列成 not_run。
- 外层 `docs/D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md:60`：P1及核心算法
  冻结后，每个新模型重新验证接口/P0和任务前提，再复现机制及真实任务收益；同模型
  多任务另报，预训练/额外学习成本披露。
- 外层 `docs/D2P_CVPR_CRITICAL_PATH_20261026.md:75`：P1→P2→P3→P4/E1–E3→P5
  是科学依赖，不是运行授权。只读资产/许可/接口及协议准备可以并行。

另读原附件 `pasted-text-1.txt:665–710` 的§8：原文仅命名P0/P1，P1要求已有动作动力学、
直接梯度易停滞但两阶段可达，再检验真实闭环优于单层搜索和普通 latent subgoal。
项目P2–P5为后续闭环编号，不能冒充附件逐项原定义。报告保存上述四份项目authority的
当前全文SHA、实际行号和逐字表格/依赖行；附件不复制入工作树，不作为输入依赖。

因此，native C1、contact诊断、一个checkpoint多seed、同模型多任务、跨论文reported
成绩，都不能单独替代P5。contact=0只界定当前根/后缀，不是普适P5定理门槛。

## 已执行的真实核查，不再泛泛排计划

本轮新代码不导入模型、物理环境、数组库或网络库。显式复用两份已有stdlib审计器，
并把其当前源码SHA纳入ledger，不执行manifest指定的任意源码：

1. 重跑 `p5_native_stage_admission_20261027_v1.audit()`：8份固定汇总报告以及94文件
   geometry闭包；去重共101份输入，45项算术/接线检查通过。检查compiler→ready→
   executor/outcome、FD→compiler及FD-array、ridge绑定，三臂和16帧/15controls完整性。
   不是仅信旧报告 `all_checks=true`。不重新认证NPY、所有journal或权重全传递封存。
2. 重跑 `p2_baseline_evidence_inventory_20261027_v1.build_inventory()`：保留10个方法身份，
   复用已保存原始表/本地comparison和五个physical文件的根级连接、指标与成本算术；
   GRASP保留拒绝/未配对分母。随后重新有界读取输入并严格解析JSON，纳入当前ledger。
   已知source/array认证限制仍保留，未补造预算。
3. SHA绑定成功fresh conditional compiler `64fd87…`：只验证四根×两Q×10候选完整census，
   每臂completed，独立新根为0；与五个本地H2对照的四根集合逐一相等。未重复main
   机制审计，不改β、阈值、候选、选择器。只引用冻结root-preconditions文档解释任务缺口。
4. 纳入实际native strict-load与encoder配对报告：前者
   `native_dynamics_strict_load_only`，后者 `paired_equivalence_failed`，严格保留实际状态。
   该历史配对失败不是断言之后native official encoder不能运行，也不抹去后续C1成功；
   strict load与同cached权重配对均不能认证独立远端官方权重真实性。

真实报告共129份绑定文件；本模块累计读32,423,245 bytes（含末尾drift重读，不含两个
复用审计器内部的重复读取计数）。单文件16MiB、累计64MiB上限，不读大权重/视频。
只走显式路径、不递归发现、不下载；拒绝路径逃逸、symlink、非普通文件、超限、重复
JSON键、NaN/溢出。所有输入末尾复核SHA，保证一次运行内不漂移。

## 当前实物与科学缺口逐槽接线

机器清单每槽都给 `depends_on`、authority键、`available_support_not_completion` 的真实
文件路径、`required_manifest_fields` 和 `bound_completion_artifact=null`。support必须能
在SHA ledger中解析。后一个null表示这个**科学槽**未绑定，不否定support里真实运行过。

| 槽 | 已有可消费证据 | 仍阻断什么 |
| --- | --- | --- |
| core_freeze | 当前完整Q条件编译实际报告 | 这是诊断compiler，不是已准入核心算法/复制组件冻结回执 |
| conditional_roots | 四根80候选census、冻结root-preconditions诊断 | 同任务direct-gradient平台轨迹、合法两段task-reach见证未认证；有限池/排除清单/校准确认分割须预声明，不能挑受益根 |
| local_strong_controls | 五个H2方法各4根实际记录、GRASP八槽audit、reported表 | 四根均已用校准；反馈/支持/总资源不齐，GRASP仅1完整配对且H1异协议；reported绝不补本地对照 |
| formal_P1 | 实际compiler与native物理outcome，但模型/根命名空间分开 | 独立calibration/outcome，同根非贪心＋真实own-state能力改变＋任务收益合取，及既有统计门；不能跨模型/根拼合 |
| used_components | 当前H2 prerequisite、matrix与依赖链 | P2组合增量、P3几何质量—成本、P4真实反馈及E1–E3仍须各自绑定；只复制部分组件须预声明范围，不能称全闭环 |
| architecture_contract | native冻结encoder/predictor/proprio状态SHA、strict-load及历史encoder配对实录 | 架构/源码修改/权重来源许可、RGB＋proprio信息边界、normalizer/action/history、既有学习成本须冻结披露 |
| native_P0_task | 成功C1、JVP/FD、ridge、native geometry | 新模型自己的P0及任务度量/可达前提；同模型FD不验证物理接触导数或目标进展 |
| P5_independent_replication | native三分支真实执行与outcome | 独立架构/任务预声明，准入核心机制及任务收益的独立根复制；依赖P1/适用组件/本模型P0/强对照 |
| same_model_multitask | closure及matrix明确E3要求 | 固定每模型checkpoint内goal×horizon×动作限制任务矩阵及独立真实success/cost/time/resource；与跨架构槽分开 |

native单根public-state identity为 `033865ef…`；LeWM四根RGB identity是另一命名空间。
报告保存完整SHA，不声称两边同根配对。三native后缀均contact=0、物体静止、coverage=0，
仅为当前合法已执行suffix的诊断。LeWM H2对照已知四校准根各0/4 official success不升级
为独立泛化，也不与GRASP唯一完整配对混成总体胜率。论文reported单元格保持context。

## 已可先完成的前沿与不得自动执行的部分

manifest给出拓扑顺序和4个offline frontier：资产/许可/源码修改及接口溯源，独立有限
root协议，faithful baseline反馈/动作/资源/统计合同，复制组件范围与独立E3任务矩阵。
这些字段清单已机器可读，不需要先等GPU；但本工具不会自定尚未冻结的τ、ε、n或预算，
不会自动扩池/改β来满足门。每个新模型须独立完成相同类别的任务前提，不能继承LeWM
根的合格标签。新证据需main审查后用新版本接入，旧manifest不悄悄指向latest。

真正新calibration/outcome、跨架构机制＋任务收益复制继续由main在合法准入后统一GPU/
physics。`launch_command=null`、`execution_authorized=false`，本产物仅为可执行验证的
证据依赖manifest，不是暗藏runner、作业授权或自动通过器。

## 完整路径、命令与实际结果

本轮仅apply_patch新增下列4份独立文件，不修改共享总进度/旧封存代码/结果：

- `/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p5_closure_readiness_20261027_v1.py`
- `/home/gbk/code/LeWorldModel/le-wm-main/tests/test_p5_closure_readiness_20261027_v1.py`
- `/home/gbk/code/LeWorldModel/le-wm-main/results/p5_closure_readiness_20261027_v1.json`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p5_closure_readiness_20261027_v1.md`

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_closure_readiness_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_closure_readiness_20261027_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_closure_readiness_20261027_v1 --verify-manifest le-wm-main/results/p5_closure_readiness_20261027_v1.json --require-admission
```

注意 `--verify-manifest` 路径相对**外层repo根**，与cwd不同；不写文件。
普通producer退出0表示接线可核查且P5仍blocked；最后一条实跑退出**3**，表示完整
重建后保存manifest逐字canonical结构一致，但科学准入仍阻断。坏证据/陈旧清单退出2。
若authority全文收到main新追加或任何已绑定文件改变，旧manifest应拒绝，而非忽略变化。
测试文件不纳入科学证据，当前源码自身SHA已入manifest。

本轮新增50项小型mock/纯算术及I/O边界测试通过；联合native 40项、baseline 29项，
**119 passed in 0.30s**。覆盖census缺项/重根/错臂、错baseline根、预算/论文标志污染、
false/整数伪门、strict-load/历史配对状态污染、依赖环/缺边、SHA漂移、非有限JSON、
manifest缺项/模型/来源/义务篡改、磁盘保留门及退出语义。这些是工具测试，不是复制
效能或科研准入证据。真实producer与保存manifest复核另行完成；无新神经/physics调用。
一次命令误在外层使用不存在的 `.venv/bin/python` 返回127，未执行任何审计；改用上述
准确cwd后成功，不追认该失败调用。

保存SHA256：

```text
5a6236fcc185b2014c8a28766b7b544a69ac81c8fb7ea5743096d181064b67f7  code
d8d41a32f057f2a9e2bb7f2d2cb0a59cb29a9c09d3f993c45790f5abe954759d  tests
07cd48053f31457ec390f99b03a728ece2c32376cd770c25886161b4b8381980  actual JSON report + execution_manifest
```

`git diff --check`通过。producer启动前要求磁盘≥200MiB，stdout-only不落缓存；保存JSON
也仅54KiB。末次观察可用306MiB，保留门满足；并行主线程可能继续改变空闲量，启动
每次重查。不删除、不提交、无代理、无下载。experiment-code技能仅用于自包含记录和
测试组织；用户限定优先，未采用其训练/绘图目录要求。
