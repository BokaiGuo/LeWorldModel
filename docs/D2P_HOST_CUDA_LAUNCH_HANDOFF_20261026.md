# D2P 主机 CUDA 启动交接：两根 sampling calibration

日期：2026-10-26。**仅完成只读 readiness；没有启动 GPU、实验、执行器或网络请求，没有申请审批，没有删除文件。** 本次唯一新增文件是本文。

本交接选用现有 `run_reacher_d2p_sampling_calibration.py`，不是 AP 复现、正式 P1 或独立 efficacy panel。预检是文件/JSON/磁盘的时点检查，**不是主机执行授权、CUDA 可用性证明、模型正确性或 adapter 兼容性证明**。命令仅交给 parent/主机操作者，在另行明确授权和 GPU 调度后执行。

## 1. 实际只读检查结果

检查时间：`2026-10-26T21:54:04.458005+08:00`。工作目录：`/home/gbk/code/LeWorldModel/le-wm-main`。使用本项目 `.venv/bin/python`，直接调用 `d2p_experiment_readiness.check_readiness`；没有导入 runner 或 torch，没有读取/哈希权重内容。另以 stat、access 和 disk_usage 读取下列元数据。

输入：

- policy：`/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm`
- adapter：`/home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json`
- 新 output：`/home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_sampling_calibration2_host_handoff_v1_20261026_cuda`
- 最低可用空间：`minbytes=268435456`，即 256 MiB。
- 显式 sources：下表 19 个文件，与当前 runner 的封存列表对应；不是整个 Python/依赖环境的完整性扫描。

实际返回：

```json
{
  "ready_for_host_cuda_request": true,
  "issues": [],
  "scope": "Read-only host input checks only. Does NOT prove GPU availability, authorization, model correctness, adapter compatibility or experiment admission. No output/disk reservation: recheck at launch; paths may change after inspection."
}
```

磁盘实际读数（output 父目录所在文件系统）：total `982240026624` bytes，used `929322082304` bytes，free `2947428352` bytes（约 2.745 GiB）。free 比门槛多 `2678992896` bytes。total-used 不必等于普通用户可用 free（存在保留空间）；门槛比较使用 free。**256 MiB 是 caller 指定门槛，不是本检查器估算的完整运行空间需求。**

output 的 `os.path.lexists` 为 false，其父目录 `results` 存在；没有创建或预留 output。空间和路径可能随父任务/其他作业改变，启动前必须再次检查。

| 文件（源码相对于上述工作目录） | 实际 bytes | 文件结果 |
| --- | ---: | --- |
| policy/weights.pt | 72290849 | 普通、非空、可读权限；仅 stat/access，未打开 |
| policy/config.json | 1313 | 普通、非空、可读；严格非空 JSON object 通过 |
| 上述 adapter JSON | 1602 | 普通、非空、可读；严格非空 JSON object 通过 |
| experiments/spectral_reachability/run_reacher_d2p_sampling_calibration.py | 11960 | 普通、非空、可读 |
| experiments/spectral_reachability/d2p_sampling_baselines.py | 10595 | 普通、非空、可读 |
| experiments/spectral_reachability/d2p_cuda_sampling_cost.py | 5000 | 普通、非空、可读 |
| experiments/spectral_reachability/d2p_model_call_instrumentation.py | 6143 | 普通、非空、可读 |
| experiments/spectral_reachability/d2p_prerun_source_seal.py | 6312 | 普通、非空、可读 |
| experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py | 65873 | 普通、非空、可读 |
| experiments/spectral_reachability/preflight_reacher_d2p_p0.py | 21359 | 普通、非空、可读 |
| experiments/spectral_reachability/reacher_d2p_context.py | 24227 | 普通、非空、可读 |
| experiments/spectral_reachability/reacher_d2p_packet_support.py | 10192 | 普通、非空、可读 |
| experiments/spectral_reachability/reacher_d2p_qpos_readout.py | 9683 | 普通、非空、可读 |
| experiments/spectral_reachability/d2p_stage_admission.py | 9674 | 普通、非空、可读 |
| experiments/spectral_reachability/build_reacher_d2p_training_action_stats.py | 6235 | 普通、非空、可读 |
| experiments/audit_live_reacher_shared_candidate_ranking.py | 38802 | 普通、非空、可读 |
| experiments/checkpoint_compat.py | 3944 | 普通、非空、可读 |
| experiments/online_planning_protocol.py | 1680 | 普通、非空、可读 |
| experiments/run_live_cube_action_adapter_planning.py | 29736 | 普通、非空、可读 |
| experiments/run_live_reacher_planning.py | 14824 | 普通、非空、可读 |
| experiments/run_live_reacher_sequence_candidate_planning.py | 20090 | 普通、非空、可读 |
| experiments/gpu_preflight.py | 9204 | 普通、非空、可读；只读源码，不执行 CUDA probe |

readiness 对每个 source 只读取一个 byte 来检查可读性，不执行、不递归导入、不封存/证明源码实际运行身份。附加只读 JSON 检查确认 config 顶层含 `_target_`、action_encoder、encoder、pred_proj、predictor、projector；adapter 声明 schema `reacher_d2p_training_action_stats.v1`、status `training_semantics_validated`，其 Reacher report 声明 action_block=5、raw_action_dim=2。**这些是文件中的声明，不是本次重新验证训练统计或模型语义；未扫描其引用的 HDF5。**

## 2. 启动前复检命令（只读，可单独执行）

以下显式列出全部 19 个 source，不使用 glob，不扫描数据。readiness CLI exit 0 才表示本地只读检查通过；非零则停止，不进入下一段。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m experiments.spectral_reachability.d2p_experiment_readiness \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter /home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_sampling_calibration2_host_handoff_v1_20261026_cuda \
  --minbytes 268435456 \
  --source experiments/spectral_reachability/run_reacher_d2p_sampling_calibration.py \
  --source experiments/spectral_reachability/d2p_sampling_baselines.py \
  --source experiments/spectral_reachability/d2p_cuda_sampling_cost.py \
  --source experiments/spectral_reachability/d2p_model_call_instrumentation.py \
  --source experiments/spectral_reachability/d2p_prerun_source_seal.py \
  --source experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py \
  --source experiments/spectral_reachability/preflight_reacher_d2p_p0.py \
  --source experiments/spectral_reachability/reacher_d2p_context.py \
  --source experiments/spectral_reachability/reacher_d2p_packet_support.py \
  --source experiments/spectral_reachability/reacher_d2p_qpos_readout.py \
  --source experiments/spectral_reachability/d2p_stage_admission.py \
  --source experiments/spectral_reachability/build_reacher_d2p_training_action_stats.py \
  --source experiments/audit_live_reacher_shared_candidate_ranking.py \
  --source experiments/checkpoint_compat.py \
  --source experiments/online_planning_protocol.py \
  --source experiments/run_live_cube_action_adapter_planning.py \
  --source experiments/run_live_reacher_planning.py \
  --source experiments/run_live_reacher_sequence_candidate_planning.py \
  --source experiments/gpu_preflight.py
```

## 3. 完整主机 CUDA 实验命令：本次未执行

只有在 parent 明确批准本次校准、分配 GPU、确认源码与 input 不再被其他任务修改，并重新通过上段只读检查后，才可在能访问 GPU 的主机终端运行。不要把这段命令或 readiness=true 解释为已有批准；也不要将其交给当前沙箱自动提权重试。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python -B -u -m experiments.spectral_reachability.run_reacher_d2p_sampling_calibration \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact /home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_sampling_calibration2_host_handoff_v1_20261026_cuda
```

没有硬编码 GPU 编号：设备分配/可见性由 parent 的主机调度决定，命令不会替其选择空闲卡。runner 内部要求 `cuda`，在源码封存后调用 `require_cuda_allocation("cuda")`，执行真实分配探针；无 GPU/驱动失败应非零停止，禁止 CPU fallback。EGL/MuJoCo 的实际 render 可用性本次同样没有验证。

当前 runner 固定 seeds `20261040/20261041`，history=3、horizon=2、action_block=5、target_delta=(0.65,-0.65)，仅 known calibration。三方法为 clipped Gaussian MPPI heuristic、best-of-N random、CEM；前两者 pre-tanh 参数盒 [-3,3]，CEM 为原 P0 无界 Gaussian 参数域，因此不是同搜索域强基线对照。不得改名 faithful MPPI，不得由负结果排除 MPPI。

候选成本计数及 predict 收据的**预期值而非本次实测**：每根 heuristic 133 candidates / 18 calls / 266 predict samples；random 33 / 4 / 66；CEM 8 calls / 256 predict samples。三者预算不相等；`fair_budget_claim` 必须 false。即使未来 predict samples 相等，也不代表编码、反向、总 FLOPs 或 wall-clock 相等。

注意：readiness 不读取 weights 内容，但实际 runner 启动会为 input receipts 完整哈希 weights/config/adapter 并加载模型；这只在未来获准运行内发生，本次没有执行。

## 4. 成功运行应有的输出与验收

上述 output 下应出现：

```text
protocol.json
prerun_source_seal/
  args.json
  manifest.json
  sources/000.source ... 018.source
seed_20261040.clipped_gaussian_mppi_heuristic.plan.json
seed_20261040.bestofn_random.plan.json
seed_20261040.cem.plan.json
seed_20261040.json
seed_20261041.clipped_gaussian_mppi_heuristic.plan.json
seed_20261041.bestofn_random.plan.json
seed_20261041.cem.plan.json
seed_20261041.json
report.json
```

这是按当前源码推导的预期清单，不是已产生的文件。源码增加时应重新核对 source count，不沿用本文时点字节数。

验收至少要求：runner exit 0；report schema `reacher_d2p_sampling_calibration.v1`、status `calibration_completed`；全部 7 项 gates 为严格 true；`source_verification.status=verified`；两根×三方法完整、plan/report 绑定一致、物理标签有限、执行根身份一致、实际 planning predict scopes 全部 completed。`P1_admission`、`efficacy_claim`、`fair_budget_claim` 全部为 false，comparisons 为 `{}`。文件存在本身或 runner exit 0 均不足以证明模型有效或 P1 准入。

完成后 parent 可运行现有 NumPy/JSON 收据审计（只读，无新推理；本次未运行，因为新 report 不存在）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m experiments.spectral_reachability.audit_reacher_d2p_sampling_calibration \
  /home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_sampling_calibration2_host_handoff_v1_20261026_cuda/report.json
```

审计应 exit 0、status `accounting_receipts_valid`，但它只证明保存收据一致性，不是独立物理 replay 或运行真实性证明。源码封存 digest 应由 parent 独立保留；source/archive 字节一致也不能替代执行证明。

## 5. 失败处理（不删除、不复用、不降门槛）

- readiness issues 非空、空间不足、文件不可读或新 output 已存在：停止。若需新目录，由 parent 明确指定另一个未用路径后重跑同一 256 MiB 预检；不删除旧结果，不自动降低门槛，也不自动清理磁盘。
- CUDA allocation、EGL、adapter semantic validation 或模型载入失败：保持 traceback 与已有 protocol/seal，报告 parent；不切 CPU、不改 adapter/checkpoint、不重试超时审批、不绕过 host GPU 调度。
- runner 会先创建 output/seal，再做 CUDA/语义检查，故失败可能留下只有 protocol/seal 的目录。没有完整 report 不可补写成功或当作 completed；exclusive output 不支持原地复用。重启需 parent 明确另取新路径并重新预检。
- seal/source/input 哈希变化、执行身份差异、计数不符或任何 gate=false：保留所有证据，标记 integrity failure；不要编辑旧 report 或在运行期间改 shared sources。由 parent 分析原因后再决定是否批准独立新运行。
- 检查到 CUDA 不可用不证明硬件永久故障；只说明本次主机运行路径不满足要求。任何修复、重启、依赖安装或大数据下载均不在本交接授权内。

最终状态：**ready_for_host_cuda_request=true，仅文件与磁盘检查通过；主机 CUDA 未执行、未证明、未授权，研究仍未完成。**
