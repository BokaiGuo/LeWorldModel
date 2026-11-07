# TwoRoom RGB 非线性字典：位移诊断开发结果

依照预先记录的 [协议](PC_TWOROOM_RGB_NONLINEAR_CHART_PROTOCOL_20261107_CODEX.md)，使用当前及目标 RGB 合成平移模板，冻结编码器后以最近模板估计末帧相对目标位移。预测先于本阶段读取真实末帧编码及物理位移而封存；但这些任务和既有链的结果先前已暴露，因此不是盲法或独立确认。

10 根中 8 根有效，2 根因既有图像重叠条件弃权；每根三条既有链。按根先平均再汇总，零位移基线误差为 35.3972 像素，真实末帧编码解码误差为 12.2903，模型预测解码误差为 11.5853；后两者分别在 6/8 根优于零基线。满足本诊断事前开发出口，不构成控制收益或 PC1 通过。

有效根的字典分别包含 289、255、204、210、289、204、238、255 张模板，冻结编码器同步时间分别约 0.337、0.189、0.151、0.156、0.212、0.159、0.177、0.188 秒，合计 1.569 秒；不包含完整规划与环境执行成本。模板平移可能不可达或处于分布外，且本地任务和候选已暴露，不能据此推断泛化。

预测、字典及分析原始文件分别为 `le-wm-main/results/pc_tworoom_nonlinear_chart_prediction_20261107_codex.json`、`pc_tworoom_nonlinear_chart_dictionary_20261107_codex.npz`、`pc_tworoom_nonlinear_chart_result_20261107_codex.json`。同池决策检验另见 [选择结果](PC_TWOROOM_NONLINEAR_CHART_SELECTION_RESULT_20261107_CODEX.md)。
