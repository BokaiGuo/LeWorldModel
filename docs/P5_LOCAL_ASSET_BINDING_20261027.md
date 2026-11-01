# P5 本地资产绑定与 TwoRoom 最小接口 replay

日期：2026-10-27。已交付可执行入口；**本轮没有启动 CUDA，不能写成模型 replay 已通过**。只新增独占源码、测试、本文和 P5 专属 results 目录，没有更改 sealed 文件或中央进展。Full D2P P1/P5 不开门。

## 实查结论

依据 `P45_PREREQUISITE_SNAPSHOT_20261027.md`、`D2P_CROSS_MODEL_BASELINE_ASSET_AUDIT_20261026.md`，以及原计划 `SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md` 的研究对象、冻结合同和 P0–P5 定义，本次将“跨任务接口可执行”与“独立架构科学复制”分开。

| 本地实际路径（以下相对 le-wm-main，缓存例外） | 本次验证 | 边界 |
| --- | --- | --- |
| `/home/gbk/.stable_worldmodel/checkpoints/{tworoom,reacher,pusht,cube}/lewm/{weights.pt,config.json,README.md}` | 四套官方目录均存在，逐文件完整 SHA256；README 中 TwoRoom 标注 MIT、官方模型及数据源 | 本地文件存在和模型卡声明，不是联网认证 revision 或法律意见；都是 LeWM，不能算四种架构 |
| `config/eval/{tworoom,reacher,pusht,cube}.yaml` | 文件存在并哈希 | 绑定配置不等于全部配置语义已执行 |
| TwoRoom config | `_target_=stable_worldmodel.wm.lewm.LeWM`；frames3、latent192、action-token10 | 静态满足本入口的 5×2 合同；`runtime_load_verified=false`。Cube 不符合此 **TwoRoom 专用**模板，不意味着 Cube 自己不能加载 |
| `data/jepa_wm_checkpoints/jepa_wm_{pointmaze,pusht,wall}.pth.tar` | 三个约202MiB archive 均存在并完整哈希 | 未反序列化或确认内部模型身份，不将名称 JEPA/DINO 混同，也不重复开启旧 PointMaze outcome |
| `data/dino_wm_osf/extracted/outputs/{point_maze,pusht,wall_single}/checkpoints/model_latest.pth` | 三条旧 runner 精确路径均不存在 | 仅此已知路径范围内缺失，不断言全盘或网上没有 DINO-WM 权重 |
| `experiments/dino_checkpoint_compat/runtime.py`、`external/local_dinov2/hubconf.py`、`.cache/dinov2/dinov2-small/{config.json,model.safetensors}` | 代码和约85MiB DINOv2 backbone 缓存存在并哈希 | encoder 不是 dynamics predictor；不能仅凭 encoder 凑出独立 DINO-WM 复制 |

`manifest.json` 逐项记录绝对路径、存在性、bytes、SHA256 和哈希范围；完整文件读取合计约1.013GB，只读、分块哈希，不复制大资产。二次完整哈希一致。没有扫描 outcome 报告或读取 outcome archive。repair/teacher/adapter/readout 权重不纳入候选。

## 实际接通的下一项接口

新入口 `experiments/spectral_reachability/p5_local_asset_binding_20261027_v1.py` 将以下资产连在一起，而非新增一张泛泛缺口表：

- 官方 TwoRoom 冻结 checkpoint/config；兼容 loader 和现有 source-supported context/prediction 函数均按当前实际字节绑定，既有文件不改。
- `results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json`：单步2维 action 的训练 z-score，先归一化、后 time-major 5×2 打包，所有 slot 共享物理统计。
- `results/tworoom_d2p_p1_preflight_support_coverage64_samebank_20261025_cuda/report.json` 与 `results/tworoom_d2p_p1_calibration_support_coverage64_samebank_20261025_cuda/report.json`：固定已消费 calibration seeds **202610360–202610367**；calibration 原 `status=fail` 明确保留。只读已知校准报告，不读取 outcome。
- CUDA 入口只重建固定第一根 **202610360**，不按结果选根。根据训练 packet pool 复现3个 warmup packet，比较保存的3幅 history RGB、root RGB、goal RGB、warmup actions 的全部指纹与 packet indices。
- root-anchored context 采用 frames1、2、root 的 emb3；预测时丢弃 incoming a0，拼接 between-state actions2 与新 packet1。不是将所有 incoming3 后再额外预测一步。
- 仅重放原 bank 中 index0（raw-zero，保留 OOD 诊断身份）和 index1（source-supported）。不选择 witness，不优化动作，不训练。每项预测两次检查确定性，与保存的 `immediate_goal_cost` 比较（atol/rtol均1e-4）。保存报告没有对应 terminal vector，因此不冒称逐坐标重现旧 terminal；新 terminal vector 会写入报告供后续审计。
- 记录实际 `PredictInstrumentation` calls/samples（预期4/4）、编码3幅context及1幅goal、0 backward，检查参数 hash、buffer、eval/frozen 和输入文件二次 hash。仅执行15个 warmup 环境动作，不执行 candidate 的真实物理后果。

训练 HDF5 `/home/gbk/.stable_worldmodel/datasets/tworoom.h5` 约12GB。本入口不重新读全文件哈希，而只读 `action/ep_len/ep_offset` 三列，按现有源码的逐episode sliding-window/finite过滤规则重建完整 pool，必须命中保存的 **870809×5×2** 和 pool SHA `2a745bdc0c45f94f2a1b56384c8b25bb34f5a90a78fa5d0142c96b1314e1c73a`。全HDF5 SHA保留为历史来源声明、明确 `not_reverified`；不读该训练文件的 RGB/state/success 列。动作 pool 完整哈希不能替代全文件认证。

## 完整命令

所有命令从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；解释器固定 `.venv/bin/python`，`-B` 和禁pytest缓存避免写旧源码目录。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_local_asset_binding_20261027_v1.py

# 已实跑生成；目录存在会拒绝覆盖。重新生成需换新P5专属名字。
.venv/bin/python -B -m experiments.spectral_reachability.p5_local_asset_binding_20261027_v1 --output results/p5_local_asset_binding_20261027_v1_assets

# 只供主线程排程；本轮未执行，不会自动排队。
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_local_asset_binding_20261027_v1 --replay-cuda --binding-manifest results/p5_local_asset_binding_20261027_v1_assets/manifest.json --output results/p5_local_asset_binding_20261027_v1_tworoom_calibration1_cuda
```

audit 不带 `--output` 时只输出stdout，不落盘。输出严格限于 `results/p5_local_asset_binding_20261027_v1*` 的新直接子目录，拒绝 symlink、越界、嵌套和覆盖。manifest 不可带任意输入路径；replay 重新校验固定完整 allowlist 与内容 hash，在导入模型/分配 CUDA 之前失败关闭。如主线程改动被绑定的源码，旧manifest将被拒绝，应重新生成新独占manifest，不能篡改旧hash迁就。

CUDA 资源：单模型约69MiB checkpoint，小batch、1根、4次预测；显存峰值与墙钟尚未实测，不作定量承诺。CPU仅允许 checkpoint 构造/存储 staging、训练动作列与统计处理、hash、环境模拟和结果序列化；所有神经 encode/action-encoder/predict 均在真分配成功的 CUDA 上，无 CPU 神经 fallback。预计输出远低于1MB，pool约35MB主内存，不下载、复制或删除大资产。

## 本次验证与限制

新增测试 **27 passed**，均为非神经fixture/固定校准元数据检查，涵盖重复/错序/新根、outcome flag、真假布尔、normalizer、JSON非有限数/重复键/尺寸、任意路径注入、symlink、独占输出、导入不加载torch，以及只有 action/episode 列的HDF5 fixture。测试不等于 CUDA 可加载或科学机制通过。

实产 `results/p5_local_asset_binding_20261027_v1_assets/manifest.json`：`all_hashed_bytes_stable=true`、`tworoom_replay_assets_ready=true`、`cuda_replay_executed=false`。额外非神经训练列核验产物位于独占 `results/p5_local_asset_binding_20261027_v1_training_pool/report.json`，记录实际 pool shape/hash 和未导入torch；它不是模型 replay。

范围限制：绑定的是列出的当前源文件和资产，不声称完整传递依赖/二进制环境的复现 seal，不重新认证历史 seal；历史 preflight 未绑定当前权重的完整版本身份，故本轮是“当前本地官方目录模型对旧校准上下文和分数的工程回放”，不是追认旧实验全部来源。四个官方checkpoint的跨环境资产不等于独立架构；未完成独立模型许可/revision、每模型P0/AD-FD、非贪心机制、真实任务收益和匹配预算对照。**P1/P5 admission 始终 false**，任何通过只能说明这个窄接口可回放，不解锁Full D2P或泛化结论。
