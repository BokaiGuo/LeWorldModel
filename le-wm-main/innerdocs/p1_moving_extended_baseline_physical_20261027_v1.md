# Moving GRASP + no-composition 固定四分支 physical v1

仅实现新的物理执行 consumer，不改旧 physical v1 或任何封存模型/编译源。使用 experiment-code 的预声明预算、日志和故障测试模式；无 worker 实际 prepare/神经/物理执行。P1/P2 admission false。旧 moving 两根不是独立确认，不继承旧 physical outcomes、不基于预测/物理收益筛臂。RGB encoder 后续另接，本 runner 没有 torch/model。

## 精确输入 schema

manifest 必须是一个独立 JSON，exact fields（不是 JSON Pointer 插件）：

```json
{
  "schema": "p1_moving_extended_baseline_physical_20261027_v1.selection",
  "sourcepack": "/ABS/moving/sourcepack.json",
  "sourcepack_sha256": "c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6",
  "source_prepared": "/ABS/moving/prepare.json",
  "source_prepare_sha256": "MAIN_SOURCE_PREPARE_SHA",
  "roots": [
    {"root_slot": 0, "root_sha256": "ORIGINAL_MOVING_ROOT0_SHA"},
    {"root_slot": 1, "root_sha256": "ORIGINAL_MOVING_ROOT1_SHA"}
  ],
  "arms": ["GRASP_H3_bounded", "joint_H2_spectral_packet_mapped"],
  "authorities": [
    {
      "authority": "grasp",
      "output": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01",
      "report_sha256": "7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91"
    },
    {
      "authority": "no_composition",
      "output": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_no_composition_baseline_cuda_20261027_v1_cuda_main01",
      "report_sha256": "ec39956ca0c90cc5dbf955b1a65bf300b68d166627a748a82314a2176a481c34"
    }
  ],
  "selection_fixed_before_physics": true,
  "physical_outcome_selection": false,
  "wall_seconds": 600
}
```

占位 source prepare/两 root SHA 必须由 MAIN 从原 source 合同逐项填写，不生成新的 root、不从结果选根；所有路径显式 absolute。固定 authority 顺序和两个实际 report SHA 写入源码；无目录自动发现，无单臂运行。root-major 顺序 `(0,GRASP),(0,no-comp),(1,GRASP),(1,no-comp)`。四个计划全部必须通过自己冻结 producer 的 saved-only `consume(output, report_sha)`；不接受旧 zero schema、直接动作 JSON、外部 callable 或旧物理报告。调用 ownconsume 会纯 saved 数组重算，包含其已披露 CPU 算术/SVD成本，不执行模型。

authorities 的 source SHA：GRASP `52d3157574e4e2726f6d88b43bd7e5dc804ce79730b65512888e63e0dd7c7106`；no-comp `8c709bed26d88e093b0293482be07cfc34702fb095b2fe23a22edadbbb985963`。验证 own model_plan schema、selection_sha256、whole_run_report_sha256，以及原 prepare/model/export/全部 artifact closure。再用冻结 moving encoder public loader 对照 source reference：slot、blind、task_id、task_contract、原四张RGB hash、真实 float32 raw warmup10、identity native192 Q/cap、logits→float32tanh→raw10。动作不二次 /100、不重投影、不clip、不round。

两个 producer 的`core.digest`均是`p1_conditional_task_admission_20261027_v1`的`json.dumps(sort_keys=True,allow_nan=False)`再SHA256，与物理`source.digest`一致；不是 encoder compact separators + newline/bytes hash。测试直接导入两 producer 的pure core对照canonical（不加载模型）。

## 物理实现与预算

仅复用冻结 `p1_constructive_moving_history_physical_20261027_v1.branch/native_hooks` 原 recipe，不调用旧 runner/episode/witness。原两 reset（原 seed 得原goal；placement seed0 options state/goal）、原非零 raw warmup 前5`[.1,0]`后5`[0,.1]`、原 source几何 recipe不变。warmup0/5/10的RGB、全state及两body完整动态/f64bits和source独立重放精确相等后，才允许绑定派生goal及下发选中raw10。原 target-free RGB224/render_actionFalse/rawrelative native环境及原 native eval/reward/terminal统计不变。goal attachment须保持全动态/goal_pose不变且没有setter/physics/render调用；所有capture保存新PNG后再检查可见agent/canvas/state/纯render guard。未序列化 solver warmstart缓存，不能声称全隐藏引擎状态可克隆。

固定4分支全horizon上限：reset8、set_state16、goal_render8、goal_bind4、public_step80（warmup40+own40）、physics_step816（reset内部16+public800）、explicit_render/PNG88。构造attempt/return4、closeattempt/return4另计，不把native constructor当physics step；封存 native构造只配置参数，不执行reset/physics/render。所有reset后内部space.step、setter和goal-render由原hooks实际计账。env构造即使成功后的journal失败也保留引用并close；close在wall/journal故障后仍尝试，不因预算计账阻止安全清理。

构造边界由本地`.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py` SHA `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`固定：`PushT.__init__`第29行至235行仅配置，末尾space/render_buffer/latest_action=None。material执行无导入AST检查拒绝constructor直接reset/step/render/setter/_setup调用，记录该文件seal；factory返回后再检查这三个未启动字段。native_hooks随即安装，在首次reset内setter先绑定新space.step，再捕获内部physics。该保证限于封存native构造及固定CLI factory，不把任意外部factory半途失败都称为精确物理计量。导入/注册、失败前未返回的任意factory工作不具通用hook覆盖；异常仍留存，不谎称成功0physics。

600s wall是每计量dispatch前检查，不是可打断单个阻塞native调用的硬实时保证。NEW run cap16MiB（包括partial/PNG/journal/final），floor200MiB，final reserve512KiB；run前需216MiB可用。prepare≤256KiB并source-bound。预算是完整上限，native terminal/truncated/guard/error可提前停止但不得pad/retry；所有四分母保留，实际attempt/return和not_started/failed/native_stopped分别报告。部分运行不强制伪造完整816/88。

模型编译成本继承各完整ownreport原账，绑定但不与physics成本混加/不声称GRASP与谱基线等总算力。新neural/image encoding/AD/training=0。

## MAIN/Jason 输出接口

prepare含 `schema,manifest,manifest_sha256,plans,contract,bindings`，另含sourcepack SHA及source prepare digest。每plan含 MAIN要求的 `root_slot,arm,commands,task_id,task_export,task_sha256,blind_sha256,compiler_authority,compiler_report_sha256,own_plan_sha256,selection_digest`，额外保留`root_sha256,compiler_report`。四分母由manifest冻结，不从actual收益调整。

report schema `p1_moving_extended_baseline_physical_20261027_v1.report`；status成功为 `moving_extended_baseline_physical_recorded`，否则 `partial_or_failed`。包含`prepared,prepare_sha256,errors,rows,fixed_branch_denominator,counts,journal_sha256,journal_chain_tail`，及source身份/contract/准入false。成功仅代表物理记录工程完成，可能包含合法native提前停止，不意味着所有full10或成功率通过。

每row含`branch_index,root_slot,arm,selection_digest,selected_plan,status,initial_exact,full_horizon,own_attempted,own_returned,first_error,secondary_errors,checkpoints,endpoint_scope`。checkpoint exact旧schema为`phase,step,png,file_sha256,pixel_sha256,state,snapshot,observation_state`，完整plan为0/5/10，PNG名相对新output。每capture 0..10均保存；guard失败last_observation可保留PNG但不伪造成合格checkpoint。RGB后续只读每分支5/10，全四臂根分母；缺失帧明确missing，不复用旧terminal、不补值。journal为完整hashchain；`execution_binding.json`连接prepare path/SHA。新report保留全部selected_plan provenance，RGB可从prepare plans按root/arm/selection digest严格join。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，新manifest由MAIN填写并hash；worker不prepare或run：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_extended_baseline_physical_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_extended_baseline_physical_20261027_v1 prepare --manifest /ABS/selection.json --manifest-sha MANIFEST_SHA --output /ABS/new_prepared01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_extended_baseline_physical_20261027_v1 verify --prepared /ABS/new_prepared01.json --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_extended_baseline_physical_20261027_v1 run --prepared /ABS/new_prepared01.json --prepare-sha PREPARE_SHA --output /ABS/new_main01 --authorize-physics
```

无实际神经/物理证据由worker产生。RAM fakeenv只能验证dispatch/guard/计账/partial/provenance/故障留存实现，不能证明实际接触/控制能力或P1成立。没有另写独立auditor。
