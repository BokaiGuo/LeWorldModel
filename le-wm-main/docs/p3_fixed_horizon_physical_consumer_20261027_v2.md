# Fixed-H3 physical consumer v2

独立新文件，消费主线程实际CUDA八条H3路径，不修改既有scheduler、runner、
tests或封存结果，不重选动作、不改chi→K公式、不沿用旧10步/可变时域终点。
本轮只有实现、非神经tests与prepare；physics/CUDA须主线程显式执行。

## 封存输入和离线校验

唯一planning输入：
`results/p3_fixed_horizon_budget_schedule_20261027_v2_cuda_main_v1/report.json`。
精确SHA256为 `1d7f9da67e545af1cfe7a081b78fb5e0bbde918ce368f01914433a7e892046c5`。
同时校验原scheduler prepare SHA、protocol ID、58份原source seal、每arm JSON和
planning journal的hash及内容。四根×uniform/dynamic全部保留，按完整root SHA join。

每arm必须有144 planning+3 independent verification，B=1/T=3，共147 actual
predict samples、147 action encoder samples、441 processed context tokens。
全cohort1152+24=1176，物理consumer不将它再计成新增预测或复制到每个执行阶段。
原backward/VJP、视觉encoder、失败及blocked预测的零计数逐坐标核验。

从packet原始16bank与三个整数primitive indices重建完整15×2 raw commands；
不接受bool索引、H2、改命令、缺arm、重复arm、错goal/model/root/protocol。
用保存的edge context作为非神经callback，独立重算所有144事件的父顺序、成本、
H1目标相关finite-response SVD、既定allocation及H3-only winner；完整结果须与
保存search逐字段一致，emit顺序须与journal一致。这是保存内容重算，不是新模型
forward，不证明模型预测符合物理。每arm verification receipt另行保留。

physical source只取原root来源、whole-history RGB SHA和goal RGB SHA；旧物理终点
不读、不拼接、不外推。AST静态读取本地实验及相关安装包依赖，分块source seal，
不为prepare导入torch/gymnasium/pymunk、不创建env、不加载权重。
外部模型hash在prepare中保持上游断言，可选CUDA编码时才全量重验。

## 实际执行合同

按原先交替arm顺序，四根两臂各一次，无额外repetition，总planned selected-path
steps=120。每分支重新生成同root并reset/warmup，核验完整RGB history、goal和
同root两臂context一致后才dispatch。reset/warmup开销单独排除，不冒称零成本。

直接复用旧durable `physical_branch` / `guarded_execute`（已支持15步），每raw step
都经fresh RoyalBlue RGB appearance guard。evaluator state只用于端点评估，不进入
动作选择或guard；此合同仍是open-loop command + safety presence guard，不是MPC。
prepare先归档consumer、guard、reset、root collector、installed evaluator等来源。

attempt/result、每张RGB、每arm物理JSON立即durable写出；补充的endpoint JSON区分
full_H3、guard_partial_or_terminated、unknown_or_stale_endpoint，不覆盖原branch JSON。
执行后独立核对journal hash、全部命令前缀、attempt/return计数和图像hash。
setup失败也物化全部8槽位；所有guard、termination、异常及未执行分支均保留。
guard中止不是完整15步结果；不同执行步数不能被endpoint数值相同掩盖。

## 可选终态编码：八分支至多八次，不是四次

默认physics-only：无模型加载或神经前向。添加 `--encode-terminals` 才启用HOST CUDA。
必须先落盘全部branch及physical_report.json，再加载冻结模型。
每个fresh actual endpoint（包括guard实际中止端点）至多编码一次，输出endpoint_scope；
全8分支都有fresh endpoint时8次encode。env.step异常、render失败或stale image不编码，
保留blocked_stale_endpoint收据，不能用上一帧充当终态，也不补执行/重试。
CUDA setup失败、encode异常仍保存对应receipt，绝不覆盖physics。

编码入口重验权重/config、CUDA allocation、冻结模型，禁止model.predict；
实际encode calls/samples/failed calls与外层attempted_encode_samples分开报告，不能
将图像读取失败算成已完成神经forward。末尾复核模型和source bytes。
report落盘后才以非零exit报告完整性或可选编码未完成。

## 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
.venv/bin/python -m pytest -q tests/test_p3_fixed_horizon_physical_consumer_20261027_v2.py
.venv/bin/python -m experiments.spectral_reachability.p3_fixed_horizon_physical_consumer_20261027_v2 --prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_fixed_horizon_physical_consumer_20261027_v2_prepared
```

主线程审查prepare并独立确认交付的prepare.json SHA后，才可运行：

```bash
.venv/bin/python -m experiments.spectral_reachability.p3_fixed_horizon_physical_consumer_20261027_v2 --execute-physics --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p3_fixed_horizon_physical_consumer_20261027_v2_prepared --prepare-sha256 REVIEWED_PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_fixed_horizon_physical_consumer_20261027_v2_main_v1 --encode-terminals
```

去掉 `--encode-terminals` 即仅物理，不进行神经执行；其余合同不变。
所有output exclusive NEW，不覆盖、续写或重新封存旧目录。

## 分母和结论边界

始终4个既有calibration roots、8 arms、每arm1次物理执行，新增独立根0。
相同命令也不合并分支，不凭选中路径或已开model outcomes筛根。
未来physical报告只提供本批描述性full-H3/partial/unmatched配对，P1/P3均不准入；
上游两臂模型cost已开，但不据此改公式、换bank、改goal或选择性执行。
非神经fixtures验证控制流/收据合同，不是physics或CUDA efficacy证据。
