# P3 K17 六槽候选接缝：保存验证边界

新增 `agent_p3_c3_k17_candidate_join_20261029_v1.py` 是离线候选接缝，不是物理 runner。它调用现有 `outer.material()` 重建并封存两根、六槽及其源材料，再以 fixture 执行 outer，将 outer 终态**事后投影**到既有 lifecycle。lifecycle 事件不是原始调度事件，不可用于证明实时执行顺序。输出路径使用独占创建；外部持有 manifest SHA 是校验前提。

独立 consumer 可检查被外部 SHA 锚定的文件、规范编码、journal 链、六槽顺序、branch_closed 与 outer 结果一致性、lifecycle 投影及已登记 PNG 的哈希；可选 `verify_sources=True` 重新调用现有真实来源构建并比较完整来源快照。这继承上游 pinned receipt 和来源读取的信任边界，不是对所有上游训练或模型执行历史的独立认证。没有外部可信锚点时自洽的重写不可检测。

它不能证明 callback 的真实环境身份、PNG 由原生环境实时生成、动作实际生效、guard 实时成立、完整预算或墙钟正确，也不能证明物理结果、等实际计算量、科学 gate 或可生产部署。`run_fixture` 接受回调只是测试接缝，调用者必须仅使用 fixture；绝不可用于真实环境。已有 outer 的 `production_authorized` 保持 false。写入失败可能留下未完成目录，不构成完成凭据；此模块不恢复或覆盖它。

要成为可派发的生产候选仍需事先批准的物理执行范围和预算、不可替换的原生采集与计量绑定、完整独立动作/guard/资源/失败链 consumer，以及针对存储失败和原生来源伪造的对抗测试。当前未修改 gate，未做物理或训练。
