# P1 新根来源：已保存 checkpoint state 投影补充核查

本次没有生成、选择或比较候选根，也没有运行模型、训练、GPU、reset 或物理。
读取固定 SHA 的三份已有投影和 identity checker 源码；没有重新加载权重或读取数据集数组。
脚本不解析历史结果的 outcome；结果输出使用独占创建，不修改旧封存或科学门。

## 增量发现

既有训练配置来源 manifest 将本地 checkpoint 字节身份与当前源码、配置和公开
normalizer 值合同相连，但明确没有建立历史训练 run 血缘。较新的 SRA CPU identity
proof 对相同 weights SHA
`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`
和 config SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`
记录了 303 个 remapped 持久 state tensor，checker 源码记录了安全加载后的
checkpoint tensor 与 CPU reference persistent state 的逐项比较。
本次以固定 proof、manifest、checker 和 reset recovery map 的完整 SHA 连接这些
投影，不重新执行该比较，也不把它称为独立执行认证。

对 proof 的 303 个持久项名称按 `normal|scaler|mean|std|dataset|train|split`
不区分大小写匹配，只有 `pred_proj.net.1.running_mean` 和
`projector.net.1.running_mean`，各为 float32、长度 2048；没有匹配到显式命名
的 action normalizer 或 split 项。这是**名称检索**，不是证明任意编码或别名下
没有相关数据，也不证明外部文件不存在、提取内容的语义或发布方来源。
checkpoint 的 tensor 字典以及模型配置也不因此构成训练 run receipt。

已有 `p1_constructive_action_stats_provenance_audit_20261027_v1.md` 已分析：
当前训练源码先在加载的数据列上拟合 normalizer 再 random_split，当前 eval 则
对其加载的数据拟合 scaler。该源码快照不能证明此 checkpoint 的实际训练实现；
H5 和 Lance 的身份关联、统计成员及历史 normalizer 仍未认证。故没有重复扫描
数据或把 public 常数／本地 H5 统计提升为 checkpoint 训练真值。

历史方面复用固定 LOCAL04 recovery map，它涵盖已知十次调用的派发与返回证据映射，
但没有证明各次 reset 穷尽，也没有证明全部开发历史的范围完整。本次未重新扫描
日志或重复 pose 碰撞计算。

## 两种独立性不是同一个命题

- **训练外**：针对精确定义的数据身份，证明候选未被冻结模型参数训练使用，并
  单列训练数据衍生的预处理／normalizer 统计是否使用过该数据；仅训练 split
  排除不能自动证明统计拟合排除。需要可信训练运行、完整成员及转换映射；如果
  使用模型冻结后独立采集路线，还需可信时序、no-copy 和冻结证明。当前 unknown。
- **开发外**：针对预先固定的开发范围，证明候选及相关信息未用于探索、调参、
  筛选或反馈，包括失败与未保存尝试，并验证身份映射及范围完整性。训练外不
  蕴含开发外，开发外也不蕴含训练外。不同 seed、不同事件和物理状态不重叠
  分别是更细的命题，不能互相替代。当前 unknown。

下一步需训练方提供 checkpoint 绑定的实际 run／解析配置、不可变数据与完整成员、
转换及 normalizer 拟合记录；历史责任方需补全十次调用和范围内其他运行的 reset、
失败、重启及开发接触记录，再由独立方审阅真实性、完整性和候选身份映射。
尚未取得这些证据，不授权 P1 或新采集。

## 产物与验证

结果 `results/codex_p1_checkpoint_state_source_join_20261030_LOCAL01.json`，
SHA256 `af9867acc0ad4125e56991355f1f081ae8309e884563f866140e3081e759e91c`。
脚本及其测试为 `experiments/spectral_reachability/codex_p1_checkpoint_state_source_join_20261030.py`
和 `tests/test_codex_p1_checkpoint_state_source_join_20261030.py`。
本轮新测试联合既有 reset recovery、pose collision、source authority 测试共 35 passed；
测试通过仅验证这些合同与投影，不提供新根认证。
