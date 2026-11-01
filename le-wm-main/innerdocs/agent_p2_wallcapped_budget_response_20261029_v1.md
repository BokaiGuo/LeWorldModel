# P2 wall-capped saved-only预算响应consumer

本NEW三件套实现冻结分析字段声明（SHA `31f2c6da78a86626b3ee1358868dea81d48291b218e66927d97705e533b7edf1`），不修改runtime、协议、候选或科学gate。作者实现/测试阶段未读取当前生产report/partial/strict或物理结果；MAIN通知实际report SHA仅作为已收到的外部信息，未用其结果改变分析方法。本版无NN/torch/物理/训练/参数搜索，RAM fixture不是实验证据。

## Authority与ABI

`consume(output, report_sha, strict, strict_sha)`接受实际输出目录、report外部SHA、完整strict文件路径与外部SHA。不硬编码未来strict SHA，不导入runtime/driver/protocol，也不调用其选择计算。先验证完整strict schema/status、固定四项accounts/plans、report/prepare/artifact绑定、冻结runtime源码及全部strict bindings字节哈希。strict是MAIN提供的既有工程认证authority；本工具不重新实现模型加载/原优化算法认证，更不会通过一个自签receipt替代MAIN完整strict。缺失或失败strict拒绝分析，不能以producer自报完成绕过。

从原gzip journal逐行核canonical compact JSON+newline、序号、previous/event SHA、gzip EOF、完整四arm封口，提取所有commit原witness及真实ledger_init时钟。不凭commit数量推测时钟，不信导出Q列。将raw accounting完整匹配strict summary（事件digest），逐条独立NumPy float64 `mean((terminal192-goal192)**2)`重算Q，严格同原float标量比较，无新epsilon。复算accepted/late、稳定早者tie、capped incumbent、nominal减incumbent gain、导出Q、全部配对差值。逐条验原goal、task/root/context、raw chart bits、导出contexts与incumbent精确匹配；原优化最优性由已绑定strict认证，不在这里重新挑动作。

原始候选/动作上下文不复制到巨大分析报告：保存全部commit的Q/tag/时刻/accepted/late、witness digest与相对nominal gain，原完整输入保留在绑定journal。失败commit命令单列error和可复算Q，不伪作成功commit。

## 固定输出

固定 `(0,joint),(0,composition),(1,composition),(1,joint)`四行、2 known roots；无incumbent/export时保留null，不补0或删行。每行含原caps、attempted/completed全14维、请求闭合差额、utilization（cap0为null/not_applicable）、elapsed、overrun、stop/errors/pending/active/finished units、五阶段完成计数、显存峰值（null保留）、全部commit、Q/gain、export missing。

两配对统一 `composition - joint`，完整双方原值及elapsed/overrun/Q/gain差值、逐资源attempted/completed/差额delta、阶段与stop、cap deviation。Q比较better/tie/worse/missing精确分栏，export配对可用分母另列0..2/固定2。共同cap不是等实耗、equalFLOPs或物理优势；VJP/J元素/horizon为请求工作，engine backward/FLOPs仍null。无置信区间、bootstrap、显著性或候选伪重复。全部unknown/missing/失败保留，P1/P2/fresh/release均不准入。除account.errors外，journal原arm_exception与ledger_failure全部另列，不能被自然停止/最终导出覆盖。

输入caps：report/prepare各4MiB、strict128MiB、单arm展开512MiB、journal展开1GiB/单行16MiB；新derivative16MiB。只写不存在的本stem输出，verify完整确定性重建，旧封存不写入。

## MAIN CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。MAIN已提供下列实际report/strict SHA；worker仅把外部信息填入CLI，没有读取生产报告/receipt内容或运行生产分析。代码不硬编码未来receipt；所有SHA仍是CLI外部入参。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_budget_response_20261029_v1 analyze \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_cuda_MAIN01 \
  --report-sha 1359439a994f91f8eb18e0128fd53e0b0aea800ff281876c6cbf84e222f6e3ba \
  --strict /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_consume_MAIN01.json \
  --strict-sha 7b3eba575a9cf39a565c1956a26a937828509b313be7e4aecb9f95e7f0b67ac1 \
  --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_budget_response_20261029_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_budget_response_20261029_v1 verify \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_cuda_MAIN01 \
  --report-sha 1359439a994f91f8eb18e0128fd53e0b0aea800ff281876c6cbf84e222f6e3ba \
  --strict /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_consume_MAIN01.json \
  --strict-sha 7b3eba575a9cf39a565c1956a26a937828509b313be7e4aecb9f95e7f0b67ac1 \
  --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_budget_response_20261029_v1_MAIN01.json \
  --analysis-sha ACTUAL_ANALYSIS_SHA
```

测试：`.venv/bin/python -m pytest -q tests/test_agent_p2_wallcapped_budget_response_20261029_v1.py`。覆盖RAM四行/两配对、全部commit/late/tie/missing、float64 mean192、语义篡改拒绝、真实gzip/hashchain磁盘读写及完整saved入口，未运行模型。

## 最终验证与冻结

新增34项测试；与冻结runtime原28项联合运行：`62 passed in 13.40s`。初次新增测试31 passed，补齐全导出/失败commit/codec后34 passed；最后联合回归为上述62项。磁盘篡改测试在更改journal后连report/strict/plan的外层SHA一起重新封存，确认不能仅靠自洽字节绕过语义检查。所有运行均为RAM/NumPy与临时文件，不加载torch/模型或环境。没有生产analyze，实际结果待MAIN审读本三件套后运行。

冻结runtime source/tests/doc SHA分别仍为 `5e733e75f56b3dd2f5598d6e9c4ae7878c1e3dd0d613c5b4777c1ecd891b9776` / `95138a646e8459aad810127c6ba191ffcfd7a117d44ead239ebb53406c171146` / `2628a145a8b52332e319775baba61bc70a3e6a5a4bec30f28b35a844363a5ac1`，无编辑。
