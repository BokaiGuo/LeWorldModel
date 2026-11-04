# v7 跨架构开发覆盖与 PC5 边界

本记录核对现有结果和本地权重，不启动实验，不改变历史合同。

| 已执行开发路径 | 绑定模型 | 已有证据 | 限制 |
|---|---|---|---|
| PointMaze | JEPA-WM PointMaze；checkpoint SHA-256 `a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc` | [单根三轮 RGB 反馈、物理及预测重放](PC_V7_POINTMAZE_RGB_LIVE_DEV_RESULT_20261104_CODEX.md)；[manifest](../le-wm-main/results/pc_v7_pointmaze_rgb_live_dev_20261104_codex_manifest.json)标记 v7 | 已暴露根；训练及 normalizer 成员独立性未知；三轮动作与先前 v5 相同 |
| PushT | LeWM PushT；checkpoint SHA-256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` | [单根三轮 RGB 反馈](PC_V7_PUSHT_RGB_FEEDBACK_DEV_RESULT_20261104_CODEX.md)；[manifest](../le-wm-main/results/pc_v7_pusht_rgb_feedback_dev_20261104_codex_manifest.json)封存权重；[入口](../le-wm-main/experiments/pc_v7_pusht_rgb_feedback_dev_20261104_codex.py)接入 v7 | 已暴露根；不是来源合格的多根优势确认，也未建立接触机制 |

2026-11-04 对上述两个当前权重重新计算 SHA-256，分别匹配对应 manifest。PointMaze [入口](../le-wm-main/experiments/pc_v7_pointmaze_rgb_live_dev_20261104_codex.py)和 PushT 入口均接入同一 v7 适配器；适配器和环境接口不同，不能据此推断整个软件栈或全部超参数相同。

因此，“v7 尚未运行另一架构”过于宽泛：已有 **JEPA-WM 与 LeWM 两种模型的跨环境开发接口覆盖**。但是架构与环境同时改变，没有在相同任务和可比数据合同上复现同一机制或优势；也没有证明独立 checkpoint、未见目标及训练集外来源。PC5 仍未通过。结果中的模型代理分数和物理误差也不应跨模型直接比较。

本地还存在 JEPA-WM PushT checkpoint `le-wm-main/data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar`，本次重算 SHA-256 为 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`，与已有[静态 ABI 审计](../le-wm-main/experiments/spectral_reachability/p5_jepa_rgb_checkpoint_abi_20261030_codex_v1.py)固定值一致。该审计指出原生路径使用 agent 位置和速度；未验证等权限 RGB-only 接口，常量替代也不能声称是原生模型。本次未重新认证其训练来源或 v7 接口，也未运行 JEPA-WM PushT。它是同任务跨架构开发候选，不是已经完成的复现。历史 DINO-WM 时序诊断是另一研究线的模型诊断；[已知三个 DINO-WM checkpoint 路径](P5_LOCAL_ASSET_BINDING_20261027.md)本次仍不存在，不能用 DINOv2 编码器代替动作条件动力学模型。

下一步若使用 JEPA-WM PushT：先固定新开发协议和根，不读取未来物理结果，明确 agent 位置和速度是否在任务输入权限内；否则先验证合法的观测取得方式，并将权限不匹配的实验单列。核验权重及数据来源、RGB 预处理一次性、动作归一化及 5×2 packet、历史与物理时钟；同动作显式 rollout 一致后才评估相同 v7 核心。来源未知时只能报告开发结果，不能升级为 PC5 确认。
