# Moving-history blind encoder v1

最小端到端协议测试，不扩展benchmark，不预设moving history修复live失败。
静止history消融不是本模块有效性证据。本交付仅source/RAM tests/doc，MAIN独占CUDA/物理。
不改任何sealed源码/输出。实际NPZ、manifest SHA尚未提供，本侧不生成prepare或假成功。

## 精确审批manifest

manifest JSON只能有下列字段；`schema`为本模块自有schema。
两slots顺序必须0/1，绝对路径、外部SHA精确给定且不同；下面路径和SHA仅语法示例，
MAIN必须替换为真实值，不能把示例当已审批输入。

```json
{
  "schema": "p1_constructive_moving_history_encode_cuda_20261027_v1.approved_blind_manifest",
  "approved_by": "MAIN",
  "purpose": "calibration",
  "with_target": false,
  "history_steps": [0, 5, 10],
  "raw_action_steps": 10,
  "slots": [
    {"order": 0, "source_slot": 0, "blind_npz": "/ABSOLUTE/actual/s0_blind_inputs.npz", "blind_sha256": "REPLACE_WITH_ACTUAL_64_LOWERCASE_HEX"},
    {"order": 1, "source_slot": 1, "blind_npz": "/ABSOLUTE/actual/s1_blind_inputs.npz", "blind_sha256": "REPLACE_WITH_ACTUAL_64_LOWERCASE_HEX"}
  ]
}
```

拒绝其他字段，尤其sourcepack/certificate/witness/state/outcome入口；不扫描同目录文件。
MAIN的完整来源准入与两次replay核验在部署输入边界之外，manifest是MAIN外部审批，
本编码器不声称从blind数组证明物理执行/可达性/独立confirmation。

NPZ只能包含 `rgb_history:uint8[3,224,224,3]`、
`raw_action_history:float32[10,2]`、`goal_rgb:uint8[224,224,3]`。
压缩包字段/展开大小/原始文件SHA/像素SHA均核验，无pickle、无额外成员。
raw必须与`np.asarray([[.1,0]]*5+[[0,.1]]*5,dtype=np.float32)`逐值完全相等。
三RGB对应warmup0/5/10，不能补零、替换history或删除相同图。

## 时序与任务出口

固定每slot observed0→observed5→observed10→goal，完整8encoder/8projector，不去重。
模型仅接收单张RGB；source_slot、raw、metadata不作为图像模型输入。
原冻结PushT/LeWM native projected CLS192、identity192 Q、mean squared objective、H2、
每packet5步/2维raw、logit球半径`.75*sqrt(5)`；不训练/拟合新Q。
原normalizer float32 mean `[-.0087,.0068]`、std `[.2019,.2002]`，绑定原源码SHA。
raw已经是环境relative action，不再除100。raw按时间优先reshape(2,10)，
每包逐元素 `(raw-mean)/std` 得float32 token，三latent/twoaction严格对齐。

自有`.task_contract`/`.task_export` schema与task ID；无旧task ID、旧APPROVED_INPUTS或旧zero-only接口。
`load_task(path, external_sha)` 返回且仅返回：

```python
{"gn_task": task, "d2p_packet": {"root_context": context, "goal": goal},
 "q": identity192, "task_id": actual_new_task_id}
```

可接现有NativeCuda/conditional GN/D2P的数组接口，但未来runner必须显式导入本loader；
旧planner的硬编码旧task入口不会自动兼容，不得冒充旧SHA。
loader严格校验完整export envelope、slot/blind/task合同、model/normalizer、
goal像素身份、固定raw、归一化history、原始float32可表示latent、Q/H2/球约束及全部task hash。
即使攻击者重算task ID，也不能绕过固定raw/归一化合同。
`native_root_sha256=null`：blind SHA不是物理state root SHA。

## 准备/执行/资源

prepare封存manifest路径/外部SHA、两NPZ/像素/raw、模型/config/normalizer、
底层代码及本source/tests/doc。verify完整重建。真实8次HOST CUDA各encoder/projector，
MATH SDPA、禁matmul/cudnn TF32、frozen参数与前后tensor hash、GPU同步、无CPU fallback。
0predict/actionencoder/AD/physics/training，复用旧底层guard/Ledger/Store而非旧zero loader。
逐图durable attempt/return/error、8原始float32 NPY与2gzip task；task写后完整readback。
全两slot分母预先保留，失败保已保存NPY和partial，无重试/新候选/按outcome筛选。
未load模型时model_unchanged=null，不说已核验。
output总cap2MiB、final128KiB、disk reserve200MiB，不复制PNG/权重/大source。
prepare独占<256KiB；本身不计作新的神经证据。

MAIN从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_moving_history_encode_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_encode_cuda_20261027_v1 prepare --approved-manifest /ABSOLUTE/approved_manifest.json --manifest-sha ACTUAL_MANIFEST_SHA --output results/p1_constructive_moving_history_encode_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_encode_cuda_20261027_v1 verify --prepared results/p1_constructive_moving_history_encode_cuda_20261027_v1_prepare01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_encode_cuda_20261027_v1 run --prepared results/p1_constructive_moving_history_encode_cuda_20261027_v1_prepare01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01
```

RAM tests只验接线/算术/无去重计量/失败partial/小容量，不是CUDA/物理/科学准入。
