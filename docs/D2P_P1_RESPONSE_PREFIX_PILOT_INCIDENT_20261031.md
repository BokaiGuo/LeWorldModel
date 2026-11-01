# 结构化前缀 cp5 采集：首次执行中断

探索协议 `D2P_P1_RESPONSE_PREFIX_PILOT_PROTOCOL_20261031.md` 固定
8 个 slot/arm 各取响应多样性列表前两项，共 16 行。来源和准备文件
分别为：

- `le-wm-main/results/p1_response_prefix_pilot_20261031_source_RUN01.json`
  SHA-256 `16c86645bffcc01737e259810e749064d36e9d6a56b94d373e2725efbce71224`。
- `le-wm-main/results/p1_response_prefix_pilot_20261031_prepare_RUN01.json`
  SHA-256 `9a6b0b55a647e1c0e43872008aa59bbe49a20c5ecd53b9745e740fb91e8bf27f`。
- 执行目录 `le-wm-main/results/p1_response_prefix_pilot_20261031_RUN01`，
  `report.json` SHA-256
  `8090993eaa8a9211c594dcdc6903234152ad2b3fc6c619418f3d2baac2d90a60`。

第 0 行候选 index 0 完成 15 步。第 1 行候选 index 14 在第四步
返回后触发 `fresh RGB/canvas guard`；其保存的同一步渲染前后状态
均记录推杆 y=520.4985249270607，超过画布上限 512。记录的首个
对应帧为 `b1_plan_04_0.png`，因此这里有越界证据，不宜为了凑齐
cp5 删除检查或补造后继。其余 14 行未启动；16 行数组不是 16 条
完成轨迹。报告整体 `partial_or_failed`，不是完整面板，不能用现有
完整消费者验收，也不能视作方法的科学负结果或独立确认。

这是已知根上的固定候选探索；零后缀只用于采集，不是后缀优化。
后续应另立版本，在保留此失败的前提下，对经过原 guard 来源、
动作、状态与帧证据核验的越界分支明确记为删失，继续其余固定行，
同时仍对其他错误 fail-closed。新版本须配套独立保存态消费者并
复核停止状态；不得用已观察的越界 outcome 改候选清单。

本次采集不含神经调用。来源重建和执行封存测试 3 项通过，只证明
固定名单的基本绑定与拒绝逻辑，不证明物理报告完整、P1 通过或
资源公平。
