# 下一冻结复制前置：PointMaze JEPA 新权重的原生输入 P0（当前阻塞）

2026-10-28，仅离线有限路径核对。本文件为本轮唯一新写集；不修改先前 asset 工具、封存结果或共享文档。没有运行模型、CUDA、physics、反序列化、网络或训练。**目前不能交付一条已经满足输入认证、可直接运行的新 checkpoint 科学命令**，也不为此再造 manifest/wrapper。

## 去重与必要更正

- TwoRoom 已有 `experiments/spectral_reachability/preflight_tworoom_d2p_p0.py` 的 `_jacobian_and_fd`，包含真实未来 action 的 AD/central FD；已有 P0 report 为 `results/tworoom_d2p_p0_preflight_support_v3_rootanchor_20261025_cuda/report.json`。再包装原 P0 或四次 prediction replay 不是新的科学前置。
- PushT JEPA 已有 `p5_jepa_native_action_response_20261027_v8.py` 的十列 JVP/两档 FD，及实际 prospective C1 FD v2；native v12 也已有三根 factual fidelity。均不重跑充新证据。
- **更正上一份 `agent_p5_frozen_replication_assets_20261028_v1.md` 的 Cube 未发现项**：本轮沿明确 source 路径找到 `results/cube_d2p_training_source_20261025/cube_training_source.json`，SHA `328ec561545675c59ca2df311c6686e7a06a6a2e1fc890b8240ec089dbba8ee7`。其中已保存五维 raw action 的冻结 mean/std，按 time-major 重复五次；不是缺这份统计。`/home/gbk/code/LeWorldModel/data/cube_source_20261025/cube_single_expert.h5` 实际存在（仅 stat，不重新 hash 约95GiB数据）。原 source-native P0 已实际完成，report SHA `3e6cd63a3ab98179211e271cfbc901e099fc7274a28a06f5fb3b1cdc936de206`，checkpoint SHA `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`。原 report 声明无 adapter/teacher；不能仅因另一个 live runner 要求 adapter 就称本地没有原生入口。本轮不改旧冻结文档或工具；MAIN 应将旧工具 Cube 的 blocked 解读为该工具 allowlist 未覆盖，不能作全库资产不存在证明。这里也不把 stat 当完整数据认证，或把原 P0 当新实验。

TwoRoom/Cube/Reacher/PushT LeWM 是同架构不同环境权重。JEPA-WM AdaLN 相对 LeWM 是不同动力学架构候选族；它的 PushT/PointMaze/Wall 三权重仍是同一 JEPA 家族，不能数作三种独立架构。PointMaze 下一步是该独立候选族的另一环境/权重前置，不是新增第三架构，更不是 P5 已复制。

## 最小新实验，解阻后交 MAIN 预声明

选择 JEPA PointMaze checkpoint 的**原生、单窗口局部 action 微分检查**，不用 planner、teacher、readout、repair 或任何动作执行。原 temporal exporter 已有 PointMaze预测记录，但不是这个 checkpoint 的原生输入 P0 微分证据；该 exporter 本身有 CPU fallback、`weights_only=False` 和现场统计计算，不能直接作为本轮命令推荐。

固定候选窗口为已存在 tail 的原全局 episode **1800/window0**，不看 outcome 选根。必须先确认原片段至少16帧、episode本地索引0与全局1800绑定；不足即失败，不换 episode。拟用源帧0/5/10为 context、15为 factual target，真实动作0:15逐5步 time-major打包，仅对最后一包10个 normalized坐标作局部扰动，proprio完全原样冻结。帧15只评分，不进 context。需由MAIN在实现/运行前确认这与该checkpoint的原生训练/rollout causal布局相符，不能照抄PushT而假定PointMaze严格load和输出schema已经通过。

最小计划：1次原生encoder（4帧），1次proprio encoder（3帧），1次nominal predictor + 10次标准基JVP + eps=.01/.005各20次central FD，即计划51次 predictor forward、10个JVP。JVP若采用`torch.autograd.functional.jvp`须披露double-backward且独立记录实际autograd调用；上述数是**候选工程预算，不是实测**。encoder/proprio/API调用、forward samples、JVP/AD、wall/显存分别记录，不相加，不称等总算力/FLOPs。保存全部十列及两档FD数组与源身份；不只投影挑成功坐标，不按误差改epsilon或换根，不用误差自动开放P1/P5。无需新physics，所有模型计算仅MAIN批准的HOST CUDA。

## 已存在的精确材料

下列 SHA 是本轮 `sha256sum` 原始字节实测，不加载 `.pth`；相对路径均从 `/home/gbk/code/LeWorldModel/le-wm-main` 起。

|材料|路径|SHA256|
|---|---|---|
|PointMaze权重|`data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar`|`a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc`|
|tail原actions|`data/point_maze_tail1800_1899/actions.pth`|`38a0b614230aefab09365ff288e348e8c1fc730add5b25ec75bda087d6358e56`|
|tail原states|`data/point_maze_tail1800_1899/states.pth`|`e0c4e4586471506180a4412693d788d9277bf548fd98431438f57ba2a6990077`|
|tail长度|`data/point_maze_tail1800_1899/seq_lengths.pth`|`66333840ffe5223486fcf8e1fdc7317c236fe1d89a17f7ce9ad8942fe6345e20`|
|全局1800原RGB|`data/point_maze_tail1800_1899/obses/episode_1800.pth`|`a1fd0f4caa364ffc2d9f91648e7a715d0b377916f1730fc7cbaf29ece2fd29fe`|
|原生loader|`external/jepa-wms/app/plan_common/datasets/point_maze_dset.py`|`ea78143e69840d519a944e3d470301cc21aa5bc540b5397de9207ca8e1d716f7`|
|本地重索引loader|`experiments/dino_checkpoint_compat/datasets/point_maze_dset.py`|`2e94e52d63b2358ada3a50f5fd2b1e9b548c44458b5079ea534ec6de93c8474c`|

配置实际存在：`external/jepa-wms/configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml`，normalize_action=true。tail重索引loader只修RGB文件名映射，没有提供冻结normalizer。

## 硬阻塞和最小外部输入（不是再下载整包）

**第一阻塞是 PointMaze 原训练口径的冻结 normalizer 认证，而非 checkpoint。** 原生loader的 normalize_action=True 会按全部有效seq_lengths拼接数据后现场计算action/state/proprio mean/std；把现有100个tail episodes代入会重新拟合子集统计，不允许。不能从旧prediction输出反推统计，不可套PushT/Wall常数，也不可用单位统计默认补齐。

最小需要MAIN提供或定位一份**此前已经保存**、与原checkpoint原生输入一致的数值receipt：action mean/std各2维、proprio mean/std各4维（若state与proprio共用需精确声明）、dtype、std口径、action_scale、统计总体/有效长度规则、原数据与生成source的已有hash/来源、文件实际SHA。JSON或`allow_pickle=False`的NumPy均可；本轮不能为补缺现场fit。只写“official full-archive statistics”不是数值receipt。已读旧 `results/enriched_jepa_pointmaze_observed_cuda_fixed_20261011/metrics.json` 的protocol片段只给这句描述，dataset_archive_sha256=null，不能据此签normalizer。

已知原完整dataset路径 `data/dino_wm_osf/extracted/point_maze` 当前不存在；这仅限该明确路径，不声称扫描全盘后不存在。已有tail不是原统计总体。Wall loader虽有顶部四舍五入常数，但其normalize_action=True分支仍直接重算均值/方差，不能拿顶部常数自动认证checkpoint；本轮也未发现可认证Wall原始窗口闭包，不换任务绕过。

normalizer补齐后仍需MAIN核对：PointMaze原生predictor/proprio state_dict严格schema和实际CUDA严格load；原生视觉预处理与官方DINO权重映射的该任务适用性；固定窗口全局/本地索引、长度、packet/action_scale和帧轴；完整实际source inventory（本地native源码有修改，不能称干净官方）；可信checkpoint加载批准及许可/revision未知项。已有PushT loader不能通过修改其全局checkpoint路径冒充PointMaze认证。若这些不匹配，保持失败，不用adapter修补。

## 当前可执行动作与交付边界

当前只能做非神经的原始字节核对，例如：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
sha256sum data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar \
  data/point_maze_tail1800_1899/obses/episode_1800.pth \
  results/cube_d2p_training_source_20261025/cube_training_source.json
```

这是资产核对命令，**不是新科学实验命令**。本轮没有新增HOST runner、prepare或假定未来hash，因为冻结输入尚未满足。最短MAIN解阻动作是定位上述既存PointMaze冻结normalizer receipt；拿不到则本候选明确blocked，不能谎称有真正可执行的新checkpoint P0。拿到后再在新独占stem实现一次单窗口CUDA JVP/FD，不需要整套planner/物理runner或重复泛用manifest。

实际验证：仅有限source/report片段读取、指定文件stat与10项raw SHA核对；未读巨大metrics全体、未解pickle、未hash巨大HDF5、未运行科学模型。纯文档没有新增RAM可执行代码单测；文档路径与SHA另行静态检查。旧P5 assets source/tests测试记录不冒充本轮测试。P1/P5保持关闭，科学实验结果unknown；此交付只完成有界可行性/精确交接，不表示研究goal完成。
