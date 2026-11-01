# P1 配置/normalizer 溯源 manifest

独占新增 source/test/本文；不修改模型、normalizer、源码、gate或旧封存。仅本地文件读/hash、AST字面量提取、safe YAML读取和JSON比较，无网络、GPU、物理、训练、scaler.fit；不导入训练脚本/torch/环境，不加载state_dict/pickle，不读取dataset数组。

## 工具回答什么

manifest 将以下证据等级分开：

1. **checkpoint文件身份**：现冻结 config SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`；weights SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`。streaming重hash字节，不反序列化；它不是本工具重算的tensor hash。
2. **数值绑定**：AST读取run_pusht_counterfactual、JEPA PushTDataset、当前P1 compiler的mean/std，以显式IEEE float32 bits和原字面量共同记录；与固定封存 task_00.json.gz 的normalization及checkpoint字段精确连接。全部数值一致是值合同，不是训练血统。JEPA字面量转float32用于比较，不声称核验了历史torch默认dtype。
3. **当前训练实现/config快照**：对train.py、utils.py、train lewm/data/model YAML、eval YAML、安装版save_pretrained源码封存SHA，并给出实际函数source/行号；不导入执行。YAML interpolation与Hydra composition不求值，不把当前train.py或其当前config冒充发布checkpoint的历史训练run。
4. **下载/声明证据**：README的dataset名称、checkpoint cache共同revision、weights LFS SHA256和config Git blob ID，明确只认证本地下载对象；不是训练dataset revision。
5. **归档统计证据**：固定已封存archive_action_statistics报告SHA `c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda`，保留action_field_only_not_full_archive、统计值与原source身份。只重hash统计报告，不打开46GB H5，不重算或冒称新验证action hash，不把本地H5自动认作训练Lance。

封存 task 的 `training_provenance_verified=False` 保留；没有任何matching值、相同数据集名字、相同模型形状或下载revision可以将其提升为True。此任务复用原normalization证据，不重新计算训练normalizer，不替换任何planner输入。

## 缺失身份字段

本地有界材料不能建立 checkpoint→training run→不可变training dataset→normalizer 链。输出训练身份verified=False，并逐项给出：checkpoint_to_training_run、training_code_revision、resolved_training_config、training_dataset_immutable_revision、training_dataset_action_content_identity、training_episode_split_and_filter_identity、training_action_units_and_conversion_identity、training_normalizer_export、normalizer_to_checkpoint_binding、h5_to_lance_conversion_identity。

这是在指定checkpoint目录、命名当前源码/config、固定封存task与归档统计报告范围内的“尚未建立”，不是遍历全机/检查binary内部后宣称不存在。现有差异不能证明public常数错误，也不能证明它是checkpoint真实训练scaler，更不能独自解释P1失败。

## CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
# 默认stdout，完全只读（除Python运行本身；-B禁止pyc）。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_configuration_provenance_20261028_v1 manifest

# 如需封存，只允许独占新stem结果文件，xb禁止覆盖。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_configuration_provenance_20261028_v1 manifest --output results/agent_p1_configuration_provenance_20261028_v1_MAIN01.json
sha256sum results/agent_p1_configuration_provenance_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_configuration_provenance_20261028_v1 verify --manifest results/agent_p1_configuration_provenance_20261028_v1_MAIN01.json --manifest-sha REPLACE_WITH_SHA

.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_configuration_provenance_20261028_v1.py
```

verify重建完整manifest精确比较，不能仅看status/verified字段；输入或source/test/doc发生变化则需新manifest，不可覆盖。原文件完整SHA均在生成结束复核。该工具不接收任意“verified=true”来源声明，也不凭补齐字段签发训练身份认证。MAIN负责正式落盘与独立审读。

## 验证记录

TDD先写缺实现失败测试，再实现；实际 **12 passed in 0.67s**，含真实封存值绑定、非执行AST、非字面量/重复/维度/std非法拒绝、相同cache revision不提升训练身份、完整manifest篡改拒绝、CLI verify全量重算、输出不可覆盖、无神经/训练/环境模块导入。真实 `manifest` stdout CLI 已exit0，没有生产结果文件；状态为 `value_bound_training_lineage_unestablished`。本工具只新增source/test/本文；最终冻结SHA在交付消息中给出。
