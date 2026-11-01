# MAIN：优化锚点 × 保旧的真实物理消融

仅从封存的32模型选择、原始control真实执行、incumbent真实执行、新zero真实
执行读取数据，不导入模型或环境模块。新zero侧尚未完成时不能生成完整结果。

全量8分支、32cell固定顺序，不删失败或按真实收益挑动作。复用分支只由预先
固定的factor及完整float32动作位串决定，并验证同task/root/goal/arm、原prefix5
位串、真实中点完整snapshot及PNG身份。zero guard若与zero无guard不同，只能
精确匹配原control；不能搜索更有利的历史outcome。各cohort必须完整返回10步，
早停不能补成相同终态。若今后真实早停，当前工具明确拒绝，须预先定义新协议。

直接从native state7复算XY、velocity、wrapped angle、native distance及原native
成功条件。原成功角规则忠实保留（不悄悄改环境门槛）；wrapped angle另列。
各误差轴越小越好，避免把native reward与error符号混用。报告逐分支anchor效果、
guard效果和交互差分，不将8分支当8独立根，不把2known根升级为泛化。

输入使用外部SHA及已接受strict receipt身份；它是独立state算术，明确
`full_upstream_rebuilt_here=false`，不是又一次全环境消费。单元测试是合成接线
测试，不是CUDA/物理证据。所有P1/P2 admission保持false。
