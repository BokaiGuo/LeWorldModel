# 全 packet 计划的 80 分支物理执行

独立 driver `pusht_full_packet_physical_20261027.py`，不改旧封存文件，不重新
规划，不训练/readout，不按物理后果筛根/候选。P1 admission=false，无显著性结论。

输入 planning report 必须为 `status=plans_sealed`，含 protocol 的 bindings、
compilation、policy、roots；plans 四根各含 constant/full 两臂，每臂全部十候选。
report 必须提供 `source_seals` 或单个 `source_seal`，验证规划前源码封存。
physical 端把最终 report 文件 SHA 写入 protocol，再将 protocol SHA 与 report SHA
写入自己的 execution source seal；所有封存完成后才 reset/warmup/执行。
上游 compilation/checkpoint/config/bank/root history RGB/goal 全部严格复核。

candidate_id 支持非负整数或安全字串；实际执行 method=arm+'_'+candidate_id，
repetition=0、H2、10 raw steps。commands 必须精确等于 prefix5+suffix5，有限且
在 [-1,1]，预测两向量均192维。constant suffix 五步相同；full 动作逐值原样消费，
不通过旧2D参数重建，不重复/投影/裁剪 full。参数签名可由 compiler 自己绑定，
consumer 以 report SHA 与实际 commands box/concat 为准。

selected_candidate_id/greedy_candidate_id 必须在封存十候选中；zero 对照由唯一
**kind=zero 且命令为零的 prefix** 识别（不是零 suffix/全零十步），若声明
zero_candidate_id 必须一致。
三类 ID 只做后验索引，不影响物理执行。80 分支全部执行、失败全部留分母。

消费端逐项复算 compiler 原 chart、spectral bank、suffix bounded ridge、原 Q/goal
stage/reach/objective/nonlinear terminal cost，并复核原候选顺序的稳定 selected/greedy。
norm cap/params/raw mapping 使用1e-6 float32数值容差，不是成功阈值；raw 动作本体
精确拼接且以文件SHA封存，physical不重建动作。报告所有 gates 必须 true，且每候选
独立 model replay exact=true；这不是新跑神经 replay，也不独立证明旧执行完整。

## 持久化与编码

复用已冻结 P3 physical_branch 的 fsync journal、每步 PNG、RoyalBlue RGB guard。
每次 env.step 前记 attempt，返回后记 returned；guard 缺失立即停止，不补动作。
每分支 physical.json 以及全体 physical_report.json 先持久化，之后才可加载 CUDA。
setup 失败时 80 个未执行槽均保留 blocked receipt。无任何 model.predict 调用。

`--encode-terminals` 同时申请 prefix(step5) 和 terminal(step10) 两类编码，各最多
80 张。编码前验证 journal SHA 及对应 PNG SHA，只读已保存图像。guard<5 时
prefix blocked；<10 时 H2 terminal blocked（中止终态仍保存在物理记录，但不能
拿来冒充十步预测对齐）。prefix 达5但后续终止仍可做 prefix 对齐。
每类分开保存 attempted samples、failure、comparable 和192维 vector、prediction MSE。
另报同一原Q/goal的 observed_Q_goal_cost 和 projected_prediction_mse，不混淆物体任务结果。
编码失败继续记录其余分支，最终 fail closed；CUDA setup失败也写160条编码收据。
无 CPU 神经 fallback。物理数据/guard/官方状态距离/物体pose与Q代理误差分列，
不宣称当前表示或任务 efficacy 修复。

## 主线程执行（不由本任务启动）

Python cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.pusht_full_packet_physical_20261027 \
  --planning-report /ABSOLUTE/PATH/TO/SEALED_PLANNING/report.json \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_full_packet_physical_20261027_v1_cuda \
  --encode-terminals --preflight-only
```

收到主线程稳定 report 路径后替换第一路径；预检通过后去掉 `--preflight-only`
即为 HOST CUDA 物理执行命令。缺少 --encode-terminals 只运行物理，不算神经 fallback。
输出 exclusive 创建，不复用/覆盖目录。最大800 selected-path env.step（warmup另计）、
880 PNG、160 batch1 encode、0 predict，预计预留200 MB磁盘、4 GiB显存、15分钟，
估计非实测；不复制模型权重，2.4GB余量下无需删除文件。

测试仅非神经接口/保存轨迹 fixture：80 census、不可变输入、命令拼接/范围、full
非恒定保真、缺图/guard前缀不足、encoder异常及 setup 异常持久化、模型原选择的
selected/greedy/zero 后验汇总。还须稳定 compiler report 互操作预检与主线程真实执行。
