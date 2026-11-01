# LeWorldModel 四个独立对话交接

用户要求四个可在侧边栏独立接续的对话，不使用子智能体替代。当前主对话工具缺少 create_thread/send_message_to_thread；以下只是已准备的任务文件，**不是已经创建或启动的对话**。在 LeWorldModel 项目手动新建四个对话，分别发送“读取并执行 /home/gbk/code/LeWorldModel/docs/parallel_chats_20261027/01_strong_baselines.md”等对应文件即可。

| 建议对话名 | 启动文件 | 独占新文件前缀 |
| --- | --- | --- |
| 强基线对比 | 01_strong_baselines.md | chat_baselines_20261027_ |
| 独立新根 | 02_independent_roots.md | chat_freshroots_20261027_ |
| P2 程序组合 | 03_p2_composition.md | chat_p2_20261027_ |
| P3 计算调度 | 04_p3_scheduling.md | chat_p3_20261027_ |

主线程负责 P1、代码审读、主机 CUDA、结果验收与共享总账。各独立对话只改自己的新文件；不改其他对话、当前六个历史 worker 或已封存 producer 的文件，不在公共日志覆盖别人的记录。不将“派发”“运行”“测试通过”称为科学通过。不发布 GitHub、不删除数据。

## 主线程当前已验证证据

- 八计划 off/on 全重新优化 CUDA：`le-wm-main/results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA `2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381`；576 predict，8计划 ownconsume通过。
- 八计划真实物理：`le-wm-main/results/p1_moving_overlay_recompile_physical_20261027_v1_MAIN01/report.json`，SHA `56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521`；8/8 horizon完整、native success 0/8，不能推导理论不可行。
- saved物理比较：`le-wm-main/results/p1_moving_overlay_recompile_comparison_20261027_v1_MAIN01.json`，SHA `569bd194b1aae31d0362b0cb444068e9f4765cf6c9b34c96822047a640e0fd32`；主线程另独立复算24checkpoint/144标量及5578事件资源计数，产物 `p1_moving_overlay_native_metrics_independent_MAIN01.json` SHA `98e1e0620d1aab41f68a2002e1289814aba3c46f697239aec65514ffc396d588`。
- 等预测预算采样 CUDA：`le-wm-main/results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA `60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`；四组各144 API/144 predictor samples，总576，各576 AE/pred_proj，errors=[]、模型不变，24.315秒；48测试和实际ownconsume均通过。不宣称与GN/D2P等FLOPs，MPPI为明确标注的local nominal-guided sampled-winner变体。
- 八分支真实RGB CUDA：`le-wm-main/results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01/report.json`，SHA `18d2f192a75e397c01e6c7eb7f6422e9bb073af865107ef469debe5b8ae32d1e`；28 encoder/28 projector、无新predict/AD/physics，errors=[]、模型不变，47测试通过；实际saved consume已通过8行/24checkpoint。

本轮捕获并在新实验前修复197/257 hidden token混淆、moving非零history误当全零等实现问题，不能当理论负结果。两个root仍是known calibration roots，独立性未认证。D2P on−off终态native_distance均值−15.06636，但object_xy误差+6.32193；GN分别+137.92594、+15.98985。不同指标不一致，不称控制已修复、P1通过或独立泛化。

所有旧状态先以真实文件/SHA复核，时间变化可能使“进行中”过时。研究总goal仍active，不能由某个交接模块的完成替代完整闭环。
