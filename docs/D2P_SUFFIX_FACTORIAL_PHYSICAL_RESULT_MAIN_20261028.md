# 后缀重编译实际物理结果：代码修复后的探索性校准

## 完整执行与验收

MAIN完整审读Pauli交付源码、测试、innerdoc，41项RAM测试通过。实际prepare与verify退出0；主机物理执行退出0，186.1468秒，24分支固定保留，工程错误0，23条完成5步后缀、1条在第2步原生成功而停止，不补步。全51前缀观察、原目标/渲染/动作buffer逐位核验；实际477 public steps、4866 physics substeps、1575 renders，attempted/completed一致。未训练，无新增神经调用；编译证据为此前主机CUDA312调用。

saved-only物理consumer验证journal/hashchain、全部PNG/pixel、逐操作收据、24身份、实际动作bits与前缀恢复。MAIN分析器8项测试通过，实际全24原生指标复算退出0。

| 条件 | 原生成功/固定分母 | 联合agent+object位置误差相对原后缀均值变化(px) |
| --- | ---: | ---: |
| open_full | 0/8 | -39.91193 |
| teacher_full | 1/8 | -54.68591 |
| teacher_small | 0/8 | +7.14731 |

负值表示误差更低。全部条件均报告，不事后挑方案。teacher_full相对open_full在5/8前缀上降低联合位置误差、3/8增加；相对均值为-14.77398px。其唯一成功位于root0/on/full10_conditional_D2P，suffix第2步terminated=True/truncated=False。wrapped独立诊断也成功，不是负raw-angle漏洞。

## 解释边界

v1实际CUDA动作与NumPy重建的ULP接口错误已经由v2修复，不能归入理论负结果。观察到teacher上下文在此固定校准集合中会改变真实控制并出现原生成功；但部分前缀仍变差，small幅度整体更差。以上不是“理论成立”或“所有问题均为代码”的证明。

MAIN另从原八分支实际checkpoint0/5按官方联合位置轴复算：四个D2P前缀均立即改善该误差，而不是暂时退步。root0/off为179.73780→61.39528、root0/on为179.73780→62.28799、root1/off为223.01842→74.59258、root1/on为223.01842→73.02630px。唯一成功分支的首段也已明显接近目标；因此该成功不能充当C2要求的“先退步以改变未来能力”机制见证。GN前缀中root0/on和root1/off联合位置确有短期变差，但不能将基线的退步事后改记为D2P重配置机制。

仍只有两个known roots、八个相关prefix，非24个独立样本；方案受旧校准启发。不重放成功分支挑根、不按结果选条件。不宣称非贪心机制、P1独立通过、P2/P3科学准入、等总算力或泛化。原后缀作为历史参考编译成本另列，不冒充等预算强基线。

## 实际证据

内仓le-wm-main/results：

- capability_suffix_factorial_physical_20261027_v1_prepare_MAIN01.json：443fcd05fe761b80bdafdf6eabace46977c156340d6b5fd5bedcb26428c13a88
- capability_suffix_factorial_physical_20261027_v1_MAIN01/report.json：cc92465141b400659e024e052b05683153b4dc6c013b81bdf1e3e29355fd2663
- analyze_suffix_factorial_MAIN_20261027_v1_MAIN01.json：970bb7071d3bfb25ab5d01e5b4e87ebc07377b01e52c704799372739d2fa6c8f

完整原生指标包括joint/object/agent位置、原生与wrapped角、速度、state7距离，分栏不混合；所有失败未知保留null，原生成功率允许真实为零。分析源码在experiments/spectral_reachability/analyze_suffix_factorial_MAIN_20261027_v1.py。

并行P2新合同MAIN审读、109项RAM测试与actual saved verify通过；它仍只有saved合同而无新物理执行。P3新冷启动对照MAIN审读、33项RAM测试及actual prepare verify通过，实际CUDA运行状态以独占结果目录为准，不能将启动当完成。
