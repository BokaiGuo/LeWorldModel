# MAIN实际中点独立复算

本adapter接真实midpoint runner的完整consume并与外部SHA固定的实际receipt
逐字段比较。它只读saved arrays，无模型、神经、物理或重新选择。输入需给
实际report/receipt/source SHA，不预填未来值。旧168源由已认证的authority
固定，再按slot与原selected_index读取精确原mid，检查原full10动作一致。

主要/次要指标仍调用已在结果前封存的primary协议：endpoint MSE及goal Q
绝对误差、带符号交叉项。matched-old2补充控制单独调用history_control，
比较新真实history前两项与原root后两项，不加epsilon或事后阈值，不把微小
漂移归因于batch-layout，也不把整三帧替换称为last-only因果干预。

本版要求原四条件完整预测；缺项即拒绝分析，不删行、补造或改分母。它不
发阶段准入，无独立根、成功率或控制收益结论。unit fixtures不是实际数据。
运行入口为该module，参数--midpoint-output/--report-sha/--receipt/
--receipt-sha/--source-sha/--output。源冻结与实际完整严格验收后由MAIN运行。
