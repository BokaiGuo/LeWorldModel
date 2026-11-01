# P5 已保存 temporal 与原生方向交叉表

只读两个明确绑定的已完成分析，固定 164 枝、80 对、两 known roots、四层各 20 对；不打开保留根，不运行神经或物理，不更改旧封存。该问题是在结果已知后提出的描述性对照，不是预注册、独立确认或科学准入。

输入 temporal 独立标量文件 SHA256 `9552571dc77ff819d54143dcdf402b39c08e684945325d3448ee39c3d01768cb`；fixed80 分解 SHA256 `b74240a0c9a10f7c9b938f799b91be090d61a2f027f522c33c1743047582ddfc`。代码核固定顺序、root/condition、h/coord、正负枝、两端 task 身份，独立重算 native distance 与 LeWM actual-Q 的方向类别及 JEPA cosine 符号，检查缺失、非有限数、重复 JSON 键和文件 SHA。它继承上游完整 strict、数组复算与身份连接，不重新运行这些核验；共同索引和 task 核对不能独立证明上游所有 raw 动作或模型载入字节。

每角均保存 5×4 表，包括 same/opposite/both_zero/one_zero/missing 与 positive/negative/zero/null，不按非零筛对。四角 AA/AR/RA/RR 分别是时槽5、10的 actual/rolled 联合模态；时槽15固定。JEPA 符号是预测与观测特征响应的 cosine 符号，另一轴是原生距离与 LeWM **实际编码** goal Q 的差分方向关系；不是 LeWM 预测与 JEPA 预测直接比较。两模型输入权限及特征空间不同，列联不可解释成公平性能、物理因果或控制收益。

实际结果 `results/p5_temporal_native_cross_tab_20261030_codex_v1_MAIN01.json` SHA256 `5302fd93f1fb35cb3e0f1c578327464f24e8d169144ea22557b6c04bbed06739`。例如 AA 中 native/Q 同向 35 对分为 JEPA 正22/负13；反向 21 对分为正14/负7；单方零24对全部 JEPA null。其他三角及四层完整表和逐对记录在 JSON；不对相关性或独立性作统计推断。

运行新增9项测试全部通过；另对相同两个固定输入用当前实现输出到 `/tmp`，SHA 与上述结果一致。这是同一实现的重建，不是独立实现，也不是上游 fullstrict。P1/P5 admission 均 false。未执行新模型/物理；没有认证任何新根。

下一步：若需独立根确认，须先获得完整历史排除/coverage approval 与事前 P5 协议、N、模型和输入权限/预算；如需真正同协议跨模型效果，需要冻结同一物理面板和输入桥、分别计量模型，并在单独授权后产生真实证据。现有表不能代替这些材料。
