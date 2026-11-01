# P3 budget-gated HOST CUDA lifecycle / interleaved fullarray consumer

新增独占stem，不改任何冻结producer、authorizer、native adapter或MAIN结果。prepare固定绑定MAIN实际native prepare：
`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_native_20261028_v1_prepare_MAIN01.json`
SHA `22f162fb493b6b83b30bddd66ec1b7f900f4bf36ea73c6249a5a97258002e50d`。
完整调用其verify，继承真实signal/compiler consumer与源闭包，再绑定新三文件。

## 完整actual journal消费

`consume`先重建prepare与全源闭包，读取单一gzip的canonical/hash chain，严格按48槽顺序逐槽：

1. 重放冻结v2 gate的原始事件与cap-vector identity；gate attempt必须紧邻且先于native attempt，native return必须紧邻且先于gate completed。不是事后用cap补actual。
2. 保留一个槽的完整signal/native trace，调用冻结`paid.validate_trace`，检查原float32 fullarray、H1/H3 nominal/AD、384 raw scalar VJP、所有AE/predict/predictor/pred_proj/grad IO及真实shape、nonzero history、normalizer、同child/task/goal/model/source。独立saved SVD/chi/gain复算不计入新CUDA信号账。
3. 原冻结chooser复算并核paid schedule/seal；超原配额或不支持的节点保持整节点paid拒绝，不缩候选、不挑子集、不借root J。
4. 对合法compiler调用冻结`validate_branch`，独立saved ridge/actions/Q/stable tie selection、完整whole/segmented/selected raw5 prefix和所有native数组链逐项核验。
5. 聚合原signal/compiler ledger逐轴attempted/completed，必须与重放gate完全一致。比较的是actual，不把同上限或391/634说成等算力。

24 child×fixed/adaptive全48槽，未到达不补零。本次census由actual得出，不预填历史41/5/2。consumer只接受complete终态；失败日志保留但不被认证为结果。未知FLOPs保持null；native metadata不是硬件真实性证明，也不重新证明AD数学正确性。

## 工程边界（不是科学配额调整）

- 总输出512MiB，normal gzip+binding≤256MiB，terminal report≤256MiB。
- gzip单member，原canonical事件无损压缩；逐事件sync flush+fsync；decoded≤1GiB、单行≤64MiB；consumer每槽native trace≤64MiB。
- wall1800s：从prepare verify前开始，至同步/model/source guards、journal close结束；不包括最终report写入。原始signal/compiler inclusive wall不相加。
- CUDA allocated≤3GiB、reserved≤4GiB；disk floor200MiB；模型导入前须有整个512MiB输出额度+200MiB余量。
- saved receipt独立上限256MiB，不裁剪full rows；不是实验输出预算变更。
- 原始science `2*h*c<=24`、4/8候选、H1/H3、chooser/threshold/normalizer/模型不改。fixed/adaptive相同vector cap，真实cost可不同。不padding、不retry。
- 这些是实际failclosed工程上限，不是容量必足的数学证明。资源不足保留engineering failure、所有48身份和已付attempt；final report超cap则尝试独占compact failure metadata，原完整gzip不删。若磁盘/短写连failure metadata也不可写，CLI失败且原已写日志保留，不覆写。

## MAIN命令

以下命令均在 `/home/gbk/code/LeWorldModel/le-wm-main` 执行。prepare/verify/consume不加载模型，worker没有执行run。未来SHA只由实际CLI输出取得，不预造。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_host_cuda_20261028_v1 prepare --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_native_20261028_v1_prepare_MAIN01.json --prepare-sha256 22f162fb493b6b83b30bddd66ec1b7f900f4bf36ea73c6249a5a97258002e50d --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_prepare_MAIN01.json

/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_host_cuda_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_HOST_PREPARE_SHA

# 仅MAIN审读批准后HOST执行，默认缺authorization拒绝。
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_host_cuda_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_HOST_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_cuda_MAIN01 --authorization MAIN_APPROVED_BUDGET_CHILD_CUDA

/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_host_cuda_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_cuda_MAIN01 --report-sha256 ACTUAL_HOST_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_consume_MAIN01.json
```

## 测试与科学边界

```bash
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_budget_host_cuda_20261028_v1_test.py
```

RAM完整48槽测试mock数组validator，仅检验资源/交错协议；另用真实saved首槽full native arrays调用原validator及负向篡改回归，不把重新包装旧日志当新CUDA。无新神经/physics/网络执行。测试不是科学证据，P1/P3/fullC3/scientific gates始终false；即使MAIN实际run/consume成功，也不是等实际预算或完整树比较的科学闭合。

实际联合回归：新host/consumer39项、冻结native11项、budget v2六项，共56 passed in49.81s；源码/tests语法检查通过。未执行本stem真实prepare、verify、run或consume CLI；真实完整新journal尚不存在，不预填report/receipt SHA。已有saved数组回归只覆盖真实首槽，完整48槽fullarray闭包仍须MAIN实际run后的strict consumer通过。
