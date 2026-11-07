# PC1 现场渲染双相机诊断 v2：明确协议分支，不修订 v1

继承 `PC1_CAMERA_LAYER_TRANSFER_PROTOCOL_20261107_CODEX.md` 的32 roots、
16/16划分、动作、5/10/25步时域、模型、输入权限、拟合边界、统计单位、
物理比较与退出条件，仅改变观察协议的声明。

v1 archive逐像素等价前检失败，未产生模型预测和动作后果。失败证据保存
于 `results/pc1_camera_layer_transfer_20261107/archive_parity_failure`。
原网格worker和状态文件哈希匹配旧记录，但本次重跑1884–1891旧初态也
产生非零像素误差；反复渲染相同状态六次输出不变，单/双相机实现输出
一致。故不能将问题直接归因于新root或新循环，也尚未证实底层原因。
不降低v1阈值，不改写旧相机认证，不将v2记为v1通过。

## v2 唯一观察协议变更

- 当前同一legacy renderer生成初态、目标与真实后继。
- 与archive帧的差异逐root、逐frame完整保存；该差异是诊断结果，
  不作为本版本的筛选条件。所有32 roots都保留，无按差异筛选。
- 仍要求同物理状态、相机切换不改变动力学；replay初态逐像素等于
  prepare初态；目标、真实后继和预测使用同一预处理。
- 明确标记 `controlled_current_renderer` 和
  `archive_pixel_equivalence=false`。只证明声明现场观察合同下的
  固定动作诊断，不证明官方archive协议复现。
- 当前前检像素已经曝光，动作后果尚未产生；未依据物理结果修改候选。
- 原协议、前检和重复渲染记录哈希绑定到v2，保留完整来源链。

此分支用于独立定位表示/预测问题，不是将未解释的渲染差异“修复为通过”。
E1–E6和PC0–PC5仍分别验收；state-assisted不升级为RGB-only。
