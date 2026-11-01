# P3：真实动作程序与 CUDA 双路径回放

MAIN 已完整审读 native program source/tests/innerdoc 及复用 solver/观察器关键路径。联合回归127项通过（13.17秒）。原冻结源码未改；新 prepare SHA256 为 `8a14691a4df2f8f90513fedb0588c6dd1be250f51a46ab42d4ebacc8ff547b26`。

实际主机 CUDA 已退出0，报告 `le-wm-main/results/agent_p3_native_program_20261028_v1_cuda_MAIN01/report.json`，SHA256 `7f5f16ffe9d440a5cd2731e1e87c2325b1b39de9c18508ebdcb28409cbe858d3`。errors=[]；模型前后不变，源码闭包通过。壁钟67.207929秒含验证等开销，不是纯GPU时间。

16候选全部完成，32条 whole/segmented replay 的每个完整context边界精确一致。新增64 predict/AE/predictor/proj调用，4次生成solve/SVD；未重新采集J、未新增视觉编码、未训练或执行物理。上游J与验证附加数组费用另列，不算免费。

| 已知根 | fixed H1 模型终距Q | adaptive H3 模型终距Q |
| --- | ---: | ---: |
| 0 | 0.9454683415079467 | 0.9086686598914356 |
| 1 | 1.2824975011868995 | 1.2831926496736858 |

四分支均选固定候选index3，各自四个不同raw动作buffer。fixed总16次预测，adaptive48次预测，执行horizon也不同；不能由表格宣称等预算优势或真实收益。

严格消费器已实际通过，MAIN用独立dot-product形式复算16个终态Q，误差容限1e-12。持久回执 `le-wm-main/results/consume_P3_program_MAIN_20261028_v1_MAIN01.json`，SHA256 `189208fa11b7ff51ea0f1f72e04647b89220257bc808889ff59c3272278b73a6`。

本轮从“只输出调度请求”推进为真实root动作生成及模型回放；不是完整depth3树，未采集child-local信号，未实现真实reconfiguration路由、等预算闭环或独立确认。P1、P3科学准入、C3和研究总goal仍未完成。
