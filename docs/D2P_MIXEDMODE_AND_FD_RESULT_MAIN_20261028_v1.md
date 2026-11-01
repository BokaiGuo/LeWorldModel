# 混合谱方向与导数实现核查：实际证据

## 新候选是否改变程序

MAIN 实际运行 mixedmode222：保留旧42候选，追加预注册180个等半径两谱方向组合；四个条件完整执行5352predict、892 Jacobian、171264数学VJP方向、896成功helper SVD。冻结模型和源码未变；无新物理或图像编码。

报告 `le-wm-main/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json`
SHA256 `d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2`。

严格消费与MAIN独立stage/reach/nonlinear-terminal复算完成，旧42全部候选完整字典、原nominal与Jacobian精确复现。持久分析：
`le-wm-main/results/analyze_mixedmode_MAIN_20261028_v1_MAIN01.json`
SHA256 `0788833b58af36e49c4794be90def317688c3590c2d873f81cfc631f2226c830`。

| 已知根/条件 | 所选index | 相对旧raw | 初始预测Q | 首段预测Q | 终态预测Q |
| --- | --- | --- | --- | --- | --- |
| 0/off | 51 | 改变 | 1.15418728 | 0.60416867 | 0.36837627 |
| 0/on | 51 | 改变 | 1.14792416 | 0.64872874 | 0.42065298 |
| 1/off | 23 | 不变 | 1.38880453 | 0.45925001 | 0.16606681 |
| 1/on | 23 | 不变 | 1.55651265 | 0.41091101 | 0.24178323 |

新增方向确实改变root0动作，不是只增加日志；但四项所选均为stage-greedy且立即改善，没有非贪心机制见证。各条件(stage,own-reach)前沿仍只有对应所选点。只否定此固定候选库的非贪心见证，不否定连续动作空间或研究理论。root0新动作须新物理执行；root1仅在完整上下文/来源/raw及旧结果strict consume全部一致时可继承，不混拼轨迹。

## Jacobian 数值正确性，不只程序重放一致

MAIN新增并实际运行中心差分审计：四条件各取root与所选prefix后完整上下文，全部10坐标、预先固定epsilon=.01/.005、正负侧与nominal。8上下文、328次CUDA预测、160项完整192维比较；无新AD、训练、RGB编码或物理。

实际报告 `le-wm-main/results/audit_mixedmode_fd_MAIN_20261028_v1_cuda_MAIN01/report.json`
SHA256 `b2371343e58536585f90a4708e380a36b81db7c63d70e05c61182834364cf4fc`。

独立消费者逐条核656项attempt/return journal、328个raw请求及结果、160项差分公式、原nominal精确复现和源码/模型闭包；回执：
`le-wm-main/results/consume_mixedmode_fd_MAIN_20261028_v1_MAIN01.json`
SHA256 `8bb78a6f1fc7caedc35cdd9e557b1f44955e15d133737f0eac14214e83fcf913`。

全比较最大相对L2误差0.0004381352；.005下最大0.0001195449。没有事后拟合pass阈值或改选择。MAIN实际28项算术/损坏拒绝测试通过。该结果显著削弱这些位置的导数接线/尺度错误解释，但不证明模型物理/RGB正确、有限幅线性近似准确或P1成立。

首份FD prepare_MAIN01未执行：MAIN在运行前将资源hook标签改为准确的predictor_module（不冒充外层API计数），保留旧prepare。实际仅执行prepare_MAIN02 SHA `6ed9d67de447f1e95c9c942ada03a27689dbdb975f1eb6e144870cbec52779ae`，没有覆盖或重用漂移seal。

## 状态

真实物理新动作验收、能力探针、独立新根、强基线同协议资源对照与三实验族仍未闭环。P1保持未准入；研究goal仍active。

MAIN进一步实际执行能力探针v1准备，43项fixture单测通过，但完整build在 `complete prepare8MiB` 断言失败，exit1，未生成prepare、未执行物理。完整closure真实尺寸大于8MiB，精确值未测；这是元数据容量实现缺陷，不是能力探针科学失败。保留source v1和 `le-wm-main/results/MAIN_mixedmode_probe_prepare_failure_20261028_v1.json`，已派发独占v2修复，不裁减原证据、不更改探针或阈值。
