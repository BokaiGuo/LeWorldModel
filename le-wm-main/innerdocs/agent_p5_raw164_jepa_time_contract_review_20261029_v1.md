# 固定164：时间、source history 与 goal overlay 接线核对

状态：仅保存材料与源码核对；未执行 torch、模型或物理。P2生产占用GPU期间不启动P5。现有 `agent_p5_raw164_jepa_cuda_20261029_v1` H1源码/测试保留，尚未作为正式公平比较交付；本文不授权其生产。

## 已核实的区别

原root-H2规划与后来的actual-prefix FD164不是同一预测合同。不可把前者的时距归给后者。

* 实际FD prepare：`results/agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json`，SHA256 `a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5`。本次实际读取：164 jobs、4 points；contract.anchor 明写 `actual cp5 context unchanged`。
* `agent_observed_raw_coordinate_fd_20261028_v1.py:90-129` 复制 observed_context；`:190-198` 对每个 raw_suffix5 仅调用一次 backend.step。source SHA256 `83bf2420f5555f892ec1222bd158dde0634b096fa19f8a4bc626a2e07347970e`。
* `agent_nonlinear_suffix_midpoint_rgb_20261028_v1.py:31` 的 ROLES 为 warmup5、plan0、plan5，即全局5/10/15；source SHA256 `b1371a0de4d805bac1b4b64cfb625b7d327a3772d61d68c092804430f94a7ece`。
* `agent_nonlinear_suffix_observed_recompile_20261028_v2.py:139-149` 从实际 actual_full3 构造 observed_context，历史动作是 warmup末5+实际prefix5，不是全零。

所以冻结JEPA桥v1也是5/10/15→20的H1 factual successor，但增加原生agent position/velocity权限。即使H1时间相同，也不能声称完全同输入或跨架构MSE数值可直接排名。用户指定的新0/5/10→15→20应明确作为前瞻H2合同。

## Source history / goal时序

`agent_mixedmode_physical_20261028_v1.py:211-240`：原生place/reset；warmup0；10次原非零warmup；在5/10存历史；核对同condition原source RGB3；随后才bind constructed task goal；plan0；own raw10。source SHA256 `e4fac2479254db07c34de5d9f88fbe0d19606b8ce2b1f8d7d5e42cbad8a042ae`。

`p1_moving_target_overlay_source_20261027_v1.py:303-345` 同样先采0/5/10，再在reconstruction阶段绑定constructed goal。`:275-282` bind_goal要求 snapshot、goal_pose、renderer 不变。native默认goal_pose不是constructed task goal；on历史是原native默认轮廓可见的真实帧，不得重画成constructed goal。goal_rgb则是独立真实witness terminal的同condition图，不是历史帧的替代品。

新H2必须读取保存的 warmup0/5/10 的原condition RGB与各自pre-action agent position/velocity，逐项绑定PNG/像素/state/source SHA。不能仅因物理时间相同便将plan0替换warmup10，须有原字节证据。缺帧或缺原生proprio则拒绝，不能用0、反推速度或真实15补齐。

## 优先新H2接线（待新版本实现与MAIN审读）

固定164及80pairs，off/on分栏，不选根/尺度/响应。warmup动作前5为float32 [.1,0]，后5为[0,.1]；own10沿原每分支动作。

| 调用 | visual context | proprio feature context | 3个outgoing action packet |
|---|---|---|---|
| first | actual0, actual5, actual10 | 对应实际原生proprio经原encoder | warm0:5, warm5:10, own0:5 |
| second | actual5, actual10, predicted15 | 原5, 原10, predicted15 proprio feature | warm5:10, own0:5, own5:10 |

`external/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py:338-365` 展示 predict_proprio 分支直接滚动预测proprio features；source SHA256 `105a6768050e4b34c95de8d7d0cc0c7ba07b69d364acf0649ba2625bacce907b`。新adapter需显式固定该分支并验证checkpoint输出shape，不准切换compute_new_pose或读取实际15。该wrapper另有缺历史时零padding逻辑，本合同不使用：原真实非零动作历史必须完整传入。这里只确认源码可接线，不声称已运行验证。

固定328次predictor dispatch；第一、第二调用完整visual/proprio输出全部保存。全164两次预测和模型不变收据先durable seal，之后才允许解码实际15/20标签。标签路径/hash可作为权威绑定预先核验，但不向predictor或selector暴露图像、状态或label features。采用原normalizer，不拟合、不改Q；编码、proprio、预测API/sample/frame成本分别事前列出，不能把328predict称为与旧164H1等预算。

H2还不等于完全公平跨模型：JEPA原生proprio额外权限、预处理、特征维度与计算量须独立披露。不得以模型间feature MSE大小做直接性能排序；无freshness、training lineage、反事实物理或P1/P5科学准入结论。

## 当前交付边界

本次只做上述指定文件的定点只读检查，未扫描历史库存，未改冻结bridge、源、结果或共享文档。H1 runtime缺最终innerdoc且未冻结；保留，不生产其正式公平比较。新H2尚未实现prepare/run/consume，因此没有可授权的H2生产命令、prepare SHA或成功receipt。
