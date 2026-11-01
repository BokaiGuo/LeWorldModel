# Prospective goal-only 可见 body 填充色支持合同（未授权生产）

仅独占 source/tests/本协议；旧 patchbridge、旧 mask 拒绝、旧 renderer 失败及科学门完全不改。目的只是回答能否清楚定义一个更窄的 goal 条件支持集合，不是新 Q/方法验证，不评分候选，不读取 candidate RGB/features/native state/coverage/reward。

## 因素 → 原因 → 本次准备修改

1. 原 LightGreen overlay 被后画的 body 遮挡：旧完整 overlay mask 拒绝仍成立；新定义明确不是 overlay 分割。
2. body 原始 shape.color 不是内部填充色：封存 `pusht_env.py:DrawOptions.draw_polygon` 先画 `light_color(fill_color).as_int()`，后画原 fill_color 的 fat-segment 边缘，且内部强制 radius=2。native `light_color` 是 NumPy float32×1.2、逐通道 min255；实际 `.jepa-physics-site/pymunk/space_debug_draw_options.py:SpaceDebugColor.as_int` 是 **round，不是 trunc/int**。实际 goal 两个 block Poly 的原色 `[119,136,153,255]` 对应内部 `[143,163,184,255]`，边缘仍是原色。本版匹配前者，不把原色边缘当内部。
3. 原 `_render_frame` 顺序：白背景 → goal polygons → live space.debug_draw（含各 body/碰撞标记）→ RGB 转置 → cv2.resize 至224。后画遮挡和 resize 混色意味着“精确填充色”不等于完整物体几何内部，也不恢复隐藏 goal。本版只定义输出 RGB 上的可观察颜色集合，不造隐藏标签。

native env/source、实际原 goal report/RGB/public capture 和实际物理 overlay SDK 均用显式 SHA 绑定；不误用普通 `.venv` 的另一 pymunk 版本。SDK collision_point_color 从其 literal AST 解析，无动态执行/环境导入。saved goal public 只提取 shapes 的 body/type/color，不将 body pose/coverage/geometry witness 用于选 mask。独立 scene 的 ctor/seed 不改，原goal RGB 不补画、不遮挡物体。

named-color 文本表只读 `.venv/.../pygame/colordict.py` 并绑定 SHA，用于当前规则的 native LightGreen/LightSlateGray 名称语义一致性。body 颜色本身来自保存 snapshot。**读取当前文本表不是认证旧 compiled pygame 的 loaded bytes**，也不据此认证 resize 后某个像素的历史来源。

## 事前固定精确定义

输入仅原 goalRGB uint8[224,224,3] 和上述颜色语义。令 C=实际 polygon 内部填充 RGB。

`M(y,x) := [goalRGB(y,x,:) == C 全三个通道精确相等]`。

这可执行、无阈值，命名为 **goal-visible exact body-fill-color support**。它至多是可见颜色内核的操作性代理；`geometric_interior_certified=null`、`overlay_mask=null`。已知其它原生 primitive 的直接颜色（agent Circle 的原/亮色、static Segment 原色、白背景、goal命名色、SDK碰撞点色）若与 C 相同则 fail closed。即便它们都不同，也没有证明 bilinear 混色不可能碰巧等于 C；此剩余像素来源不确定性明确保留，不冒充纯几何 interior。

固定16×16、row-major，每格14×14。唯一 prospective patch 支持规则为 **格内全部196像素 M=true**，即 count==196。不是 count>0、中心点、任意比例阈值、腐蚀/膨胀或最近 patch。全部256 count/support 布尔值均保留，不按候选效果选格，不生成 normalized weights。若全196支持为空，保留空并阻断未来 support-mean metric；不因空而换规则、不将空均值填0。

像素 M 以 C-order/little-bitorder packbits+base64 保存，另给 unpacked uint8 SHA/count。协议无 normalization/Q/权重。输出状态始终 production_allowed=false；任何后续 metric 都需 MAIN 新批准并冻结 normalization、空支持处理、配套全四词对照及资源合同。不得在本文件加入候选 feature 评分。

## 可识别与未识别

- 可识别：原图上的精确填充色集合、各固定 patch 精确匹配像素数、是否全部196匹配；输入没有效果标签，规则不拟合。
- 不识别：完整 goal overlay、遮挡后的完整 object silhouette、真实几何边界、缩放混色像素的来源、token 的局部因果感受野（attention 是全局）、这个支持是否更贴近任务成功。
- 不授权：从候选 native label 选区域/阈值、换色/藏物体重画、跑模型、加权候选重排、使用原生结果验证同批新方法、宣称 C2/P1。

因此可交付一个严格可复验的 **窄操作性支持定义**，不能交付几何 interior/全 overlay 认证或已验证任务权重。若后续任务要求这些更强语义，仍缺独立遮挡/分割 provenance（尤其 pre-resize 像素来源）；本版不会填补猜测。

## 仅 MAIN 运行保存 prepare/verify；不是 production run

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_goal_object_interior_contract_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_goal_object_interior_contract_20261028_v1 prepare --manifest results/agent_p5_goal_object_interior_contract_20261028_v1_prepare_MAIN01.json
sha256sum results/agent_p5_goal_object_interior_contract_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_goal_object_interior_contract_20261028_v1 verify --manifest results/agent_p5_goal_object_interior_contract_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA
```

没有 run/CUDA/physics/feature scoring 接口。写入仅 results 下独占新 stem，exclusive create，2MiB cap、200MiB free floor。worker 不产生 MAIN prepare，不写生产结果。准备缺颜色/源/完整原 goal 时 fail closed，不猜默认色。RAM 测试不构成任务桥有效性或科学通过。

## 本次交付检查

21 项 RAM 测试已通过：native float32/round 调色、原色边缘与一通道近似色排除、固定 all196（195明确不通过）、空集合不填0或放宽、所有256格/packbits还原、已知 palette 冲突及未知形状拒绝、输入 ABI、无候选字段/环境导入、协议完整重算及排他写入。

在规则代码及协议写出后，只读原 goal 做了一次内存 prepare 可行性检查：base `[119,136,153,255]` → fill `[143,163,184,255]`；精确颜色像素988，固定 all196 整 patch 数0，256格全保留。未导入 torch/pymunk/pygame/cv2，未写生产/MAIN prepare，未运行模型/物理。这个结果只说明操作性色彩集合非空而严格整格支持为空；不改变 all196，不换占比/中心/any规则，不外推新的方法有效性。

当前不能释放一个非空整 patch mean 任务桥；几何 interior 和混色来源仍未认证。若 MAIN 要探索其它 goal-only pixel→patch 聚合，需另立前瞻合同且不可依据候选效果选规则；本版不自动提供替代权重。
