# P3 同任务真实 H1/H3 信号：MAIN 主机 CUDA 与独立复算

MAIN 完整审读冻结 signal adapter 与新增 runtime bridge、对应测试及 innerdoc，核对 source SHA `6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b`。实际联合 91 tests 通过（7.77s），prepare02 saved verify 退出 0。此前六项联合导入测试污染已修复：精准 import 拦截与独立冷进程，不删断言或清空 sys.modules。

实际主机 CUDA 完成，报告 `le-wm-main/results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02/report.json`，SHA `5c776ea604f615081b9f50b92648e22e2612ef46d8443bf1949f69341ed4864d`；errors=[]，source/model 前后不变，wall43.97473s。因与 allmode v2 并发，不据此作运行时间优势比较。

固定两任务×H1/H3：16 predict/AE/predictor/proj，4J、768数学VJP/grad/raw-hook、4array SVD、4schedule。无新encoder/projector/physics/训练。全部实际计费轴 attempted=completed；backward-engine/FLOPs与硬件总成本未知，不用数学方向数替代。

| 已知 task slot | H1共同残差 χ | H3共同残差 χ | 有符号覆盖增益 | 固定决策 | 自适应决策 |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 0.36189714 | 0.24978922 | +0.11210792 | progress/H1 | progress/H3 |
| 1 | 0.45664032 | 0.36961563 | +0.08702469 | progress/H1 | progress/H3 |

两方法均4候选请求、depth limit3，沿用原chi=.5/gain=.05政策，无事后调阈值。H1 rank10、H3 rank30；H3自身nominal残差χ与表中的共同短H残差不同，不能混用或把nominal移动算成覆盖增益。

MAIN 实际 strict consume 已退出0，完整校验所有真实数组、768方向、normalizer/ABI/模块桥、四决策与队列。另用独立 `np.linalg.lstsq(rcond=1e-6)` 计算残差投影，与producer的SVD投影在1e-12内一致，并独立按冻结规则复算type/horizon/depth。持久严格回执与独立复算：`le-wm-main/results/consume_P3_signal_MAIN_20261028_v1_MAIN01.json`，SHA `c2e520e1e03d464a52d735bcb83cf45cfe74beb4392474f32732bc758e065dc4`。

这次确实有同任务horizon决策差异，不能再把旧跨根rounding造成的零差当本实验结论。但两根未触发reconfiguration或depth差异，且本轮仅产生待compiler请求，model_plan/program_plan均null，尚无完整树搜索、可执行程序及真实物理收益。H3/depth3不是总共三packet；未来子节点必须新付局部信号，不能复用root信号冒充。不是等预算对照、fresh确认、P1或完整C3通过。
