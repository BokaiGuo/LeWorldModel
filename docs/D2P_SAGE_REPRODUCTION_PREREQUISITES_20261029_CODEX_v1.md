# SAGE 强基线：原文、资产与复现前置清单

2026-10-29；独立登记。本轮只读原文、已有源码缓存、配置及本地文件；未启动 GPU、加载神经模型或 checkpoint、运行模拟器、训练、安装依赖或下载权重。结论：**原文表格及主要协议已核，PushT 本地 LeWM 与数据文件存在，但 SAGE 模型组件、其与 backbone 的配对及原始评估输入尚未闭合；不能启动正式复现，也没有本地 SAGE 效能值。**

## 1. 原文与作者报告，单独记录

论文为 [SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning，arXiv:2607.17973v2](https://arxiv.org/pdf/2607.17973v2)，本次读取固定版本 PDF；SHA-256 `84ab2b0c2dc805ff7126f6c00f86f90a799425ead9e05a04e652a038ca5e94e4`。原文是预印本，会议接收状态本次未核。先前 [来源解析](p2_sage_primary_source_resolution_20261027_v2.md)只核摘要，本文进一步核原表和附录；旧附件及摘要不是原表替代。

原文 **Table 1，PDF 物理页 7**：固定 LeWM、query 与规划预算，成功率 %，均值 ± **三个固定评估组之间的样本标准差**，组索引 seeds `{32,42,52}`、每组 50 queries；不是三个独立训练种子的标准差。

| 任务 / H | Base LeWM | Subgoal Scoring | Prior Only | SAGE |
|---|---:|---:|---:|---:|
| PushT / 50 | 56.0 ± 8.0 | 75.3 ± 9.5 | 42.0 ± 6.9 | 81.3 ± 3.1 |
| PushT / 100 | 8.7 ± 3.1 | 72.7 ± 7.6 | 28.7 ± 2.3 | 72.7 ± 9.0 |
| PushT / 150 | 4.7 ± 4.6 | 58.7 ± 7.6 | 16.0 ± 5.3 | 64.7 ± 9.5 |
| Cube / 150 | 20.7 ± 5.8 | 51.3 ± 4.2 | 1.3 ± 2.3 | 67.3 ± 3.1 |

Subgoal Scoring 用 Gaussian proposal + 生成 subgoal 作评分目标并有 WM search；Prior Only 不做 WM search；SAGE 用 subgoal-conditioned proposal、相同 subgoal 评分及 WM search。作者还比较 final-goal prior/scoring，完整列见原表。原文 Figure 2 和附录 A.4 给出每决策 300 candidates、30 CEM rounds、30 elites、variance scale 1.0；这不是整个系统的等 FLOPs 或端到端墙钟证明。

**主表不是纯视觉方案。** 附录 A.5 明确 PushT 的原始组件使用当前 state 和 proprio 共 11 维；Cube 使用 28 维 observation，visual-only variants 才省去这些低维字段。Table 17 给出 PushT H150 state-assisted 64.7%、visual-only 61.3%；Table 18 的每组 50 是成功个数，pooled 是 150 queries 的百分数，不是训练种子均值。不能把 state-assisted 主表误标为 RGB-only。原文 Table 5 中 PushT success criterion 是 position < 20、angle < π/9；实际坐标单位与本地 evaluator 等价仍需核验。

原文 A.3–A.5：PushT episode split train/val/test 为 14,948/1,868/1,869；目标图像来自 held-out 轨迹 future frame。每个 H 的 3×50 固定 manifest 记录 episode、起始/目标帧和语义 hash。PushT 执行预算 2H；H150 名义日程为 15×10，未成功时按最后一个 option 时长继续直到预算耗尽。每次获取当前历史后重新规划；当剩余目标距离不超过当前 option 时直接用 final-goal latent。原文同 seed 的独立日程诊断 manifest 不等于主表 manifest；不能以 seed 相同代替 query 身份验证。

## 2. 当前资产：源码声明与真实本地文件分开

缓存来源 [资产元数据](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/report.json)：[SAGE 代码仓库](https://github.com/PKU-ML/SAGE)自称官方，所缓存固定 commit `43c0a655bd2bbd33d8047c7588360e193671e9f0`；MIT 是源码 LICENSE，不自动覆盖权重/数据许可。缓存是选定源文件的文本快照，不是已安装或完整本地 checkout。论文元数据与仓库 BibTeX 曾有作者列表差异，见来源解析；发表引用须再人工核对。

| 项目 | 当前检查 | 可复用判定 |
|---|---|---|
| SAGE 推理接口 | [eval PushT 缓存](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/sage__eval__pusht.py.source.txt)、[runtime 缓存](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/sage__runtime__lewm.py.source.txt)、[generator 缓存](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/sage__models__subgoal.py.source.txt) | 有接口和评估逻辑文本；缺完整 checkout、依赖与组件，不能直接执行 `.source.txt` 作为正式管线 |
| PushT generator | 固定源码 [checkpoint registry](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/configs__checkpoints.json.source.txt) 声明 `pusht_generator.pt`，233878994 bytes，SHA256 `0b3647a3a41435969d750ec58176ef5f92a419c4eacae2b5cda74b35e63f90da` | 声明值，未验证远端文件或本地字节；未下载 |
| PushT action prior | 同 registry 声明 `pusht_action_prior.pt`，163404922 bytes，SHA256 `03ecab8f9d757eb8b3fc15e93481e830325428c24ede813c18f8692cc9b4bd80` | 同上；原表 far-goal prior 消融另需 `pusht_far_action_prior.pt`，不是主 SAGE 的替代品 |
| 原始 backbone 对象 | 资产 registry 声明 `pusht_lewm_object.ckpt` SHA256 `4656a13b878d83e417ba97d96257e77fdadc7a9ecc997031eb611c6494a4c270`、72334304 bytes，标记 `local_only` | 这是源码登记，不是已取得对象，也不是 HF 已发布证明；序列化对象文件 hash 不可直接与下行 state 文件 hash 对比并判为不同参数 |
| 本地 PushT LeWM | [weights.pt](/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt) SHA256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`；[config.json](/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json) SHA256 `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`；本轮重新计算 | 文件存在，config 描述 192 维、3 帧、10 维 action block、224 图像；未加载，未认证其参数与 SAGE 训练时 backbone 一致 |
| 数据与统计 | [本地 H5](/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5) 存在，约 44 GiB；[缓存 train-split stats](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/data__stats__pusht_train_seed42.json.source.txt) 声明 Lance 数据及固定 split | H5 与作者 Lance 数据、episode/split/frame 映射或预处理等价未认证；stats 只是文本文件，不能用 evaluation 数据拟合补齐 |

本轮在 `le-wm-main/results` 和本地 PushT checkpoint 目录按上述三个精确文件名检索，未发现对应 generator/prior/backbone 对象；只代表检索范围，非全盘不存在。HF `CLTRAY/SAGE` 模型 API 本轮经批准只读查询 25 秒超时；实际远端库存、不可变 revision、权重许可及模型字节均 `unknown`，不把源码声明的 hash 当作远端下载成功证明。源码 registry 本身 SHA256 `9d230e63b158691831ebd71ed155fd9b4812fe6be3726174f5b2ae81c66d2ca8`；缓存 action stats SHA256 `c5b8e64d124ca09481854140820684fd90e573ee0f3aa0484ba5a327fc70e42d`。

## 3. 接口兼容：静态线索不是配对认证

- SAGE `load_lewm` 优先通过 `AutoCostModel`，失败后走 `load_pretrained`；`encode_lewm_context` 从 `encode` 取 `emb`，对 4D 表征沿第 2 维做 mean；`rollout_lewm` 使用 `[B,H,C,224,224]` 上下文、`[B,1,C,224,224]` 目标和 `[B,K,T,A]` 动作候选，调用 `get_cost`。这些是源码接口声明，不是对本地对象的运行验证。
- 本地 [jepa.py](../le-wm-main/jepa.py) 有 `encode`、`predict`、`rollout`、`get_cost`；但本地 config `_target_` 是 `stable_worldmodel.wm.lewm.LeWM`，不能仅凭本地 `jepa.py` 同名方法证明装入的是同一类或语义。需比较真实加载后的类、参数与输出，不能凭 192 维或形状认证 latent 坐标一致。
- generator `forward(history_latents, goal_latents, lowdim, goal_offset_steps, subgoal_offset_steps)`；加载器要求 `prior_type=pusht_subgoal_prior_v1`、`model_config`、`model`，并携带 stats。SAGE eval 对 prior 内 `action_mean/std` 与独立训练统计用 `atol=rtol=1e-6` 检查。图像预处理、history 次序、latent pooling、低维字段次序、归一化、5 raw actions/block、duration 与设备精度均需和训练来源一致。
- eval 的 `_normalized_lowdim` 在字段全缺时可生成零向量：因此仅“代码能跑”不足以证明使用了主表规定的 11 维观测。复现前须显式拒绝缺失、维度/顺序不一致或未获授权的特权字段；若使用 visual-only 权重，则以独立 visual-only 协议和数值报告，不能借用 state-assisted 成绩。
- 加载 generator 缓存显式 `torch.load(..., weights_only=False)`；完整 object 亦可能依赖反序列化。只有在来源、许可、完整字节和可信执行环境确认后才考虑加载；对不可信文件先做字节校验，不把 pickle 当纯数据。此次未反序列化任何文件。

## 4. 可执行前置与阻断顺序

以下是**后续执行清单**而非声称已通过；现在不发起下载、训练或 GPU 作业。

1. **选择目标并冻结身份。** 原论文 PushT state-assisted 完整复现，或同本地 moving RGB 合同的 visual-only adaptation，二者分开；固定 SAGE commit、训练版 backbone、generator、prior、配置与精度。要求发布方/可信镜像提供不可变 revision、来源许可、各文件名、字节数和 SHA256 以及组件训练时 backbone/state identity。原版对象不可得时，本地权重只能标 `candidate`，不能默认互换。
2. **资产接收和字节门。** 在明确许可和下载范围后只获取所需文件，逐个检查 size 与 registry/可信发布方 digest；任一缺失或不符即停止。实际 dataset/split、query manifests、图像与动作/lowdim stats 同样锁定版本；需要核验现有 H5 与作者 Lance 的 episode、frame、动作、终止状态及 split 映射。不可按文件名推断一致。不要直接执行 `scripts/reproduce_main.sh`：它含下载和环境评估；`download_checkpoints.py` 默认 `main` 且循环 registry，非精确最小资产请求。
3. **环境门。** 缓存 [pyproject](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/pyproject.toml.source.txt) 指定 Python `>=3.10,<3.12`；`core` 有 torch `>=2.5,<2.6`、gymnasium `>=1.3,<1.4`、transformers `5.12.0` 等。当前本地 `.venv` 只读元数据：Python 3.12.3、torch 2.12.0、gymnasium 1.2.3、transformers 5.9.0，存在明确版本不匹配。准备独立环境并核完整锁文件、stable-worldmodel 与模拟器兼容，不覆盖现有环境；此次未安装。
4. **可信、非 GPU 合同检查。** 在资产与安全门后进行 CPU 字节/配置检查及小形状接口测试，记录 `prior_type`、state dict 键、encoder/backbone 配对、normalizer、lowdim 键与顺序、图像规范、action 单位和 3 帧/5 步约束；确认 generator 输入实际是相同编码坐标，验证 seed、query、goal-offset、duration 与 `remaining≤option` 最终目标替换逻辑。形状测试、CPU smoke 或旧结果合同只验证接线，不是性能结果。转换 object 与 state weights 时，只有可信环境内逐参数等值或可追溯 state hash 加前处理/配置吻合，才可认证等价。
5. **原协议冻结。** 对 Table 1 使用 PushT H50/H100/H150 的 3×50 manifest、原始 success、动作反馈与选项 schedule、延长到 2H；保留方法标签与全部逐 query 结果。记录独立 generator/prior 调用、编码、CEM forward、反向、物理反馈、墙钟与硬件，并验证真实输出文件的 checkpoint SHA 与每次计划日程。候选数和 CEM rounds 相同不等于总资源相同。
6. **本地比较另行预注册。** 为 D2P 与 SAGE-style 固定共同 checkpoint/坐标、独立 roots 与 goal、观察权限、重规划频率、动作范围、预算轴和 native evaluator；分层对照仅 subgoal scoring、prior-only 和 full SAGE，记录低维字段的可用性、原生成功、实际端点和完整资源。不同观察权限/反馈/训练成本无法对齐时以异协议资源–结果对照呈现，不能称同协议 superiority。任何物理或 GPU 执行另经安全和任务准入，不由本文自动授权。

## 5. 本地结果与验证边界

本地 SAGE 作者组件复现数值：**unknown，未运行**。现有 [moving comparison](../le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json) 是其他方法的 2-root 校准，不是 SAGE，也不能与作者 150-query 值混排。已有 [subgoal-gap 合同记录](D2P_EXECUTABLE_SUBGOAL_GAP_NASH_20261028_v1.md) 的 saved 诊断与测试只是接口/旧轨迹证据，其 learned-prior 真实调用和物理准入仍未完成。

本次验证为：原 PDF Table 1、Table 5、Table 17/18、附录 A.4/A.5 与静态源码阅读；本地两文件 SHA 及存在性、限定范围文件名检索、Python 包版本元数据读取。PDF 文本抽取有字体警告，所列 Table 1 与附录数字经版面文本复核；未进行图形目视渲染。模型文件、HF 库存、许可证、manifest 与数据字节映射未认证。结论可用于锁定前置，不是官方复现或效能声明。
