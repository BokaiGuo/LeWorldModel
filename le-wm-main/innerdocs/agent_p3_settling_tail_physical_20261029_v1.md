# P3 十支原生统一 settling 尾段

本件为工程诊断实现，不签 P1/P3/C3、freshness、独立根、equal FLOPs 或 physical release。
已知已有结果后固定5步尾段，是探索性时间延长，不替换原 gate、原 cp15 或原预算结论。
worker仅RAM/临时盘验证，MAIN审读后决定是否运行真实物理；不加载模型或执行NN。

## 输入和固定动作

继承已封存 Stage C 的 prepare SHA `8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f`、
batch SHA `b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b`、
full receipt SHA `4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1`；
原receipt作为已认证 saved 起点，不重新执行原 upstream fullstrict。自身 prepare 重建十支
whole job、原源闭包、原goal payload、五个 full witness 和新source/tests/doc SHA。

固定五端点(r0C,r0Z,r0P,r1Z,r1P)各U/Z2，不选根或分支；每支新构造环境、原seed reset、
原placement reset、warm10、原raw15、正float32零raw5；没有cp15 hydrate、clip、重算或补动作。
relative raw零令每步PD目标为当前agent位置，阻尼和接触仍可改变状态，不保证静止或收益。
原goal7含非零目标agent速度，velocity误差是相对goal速度而非速度幅值；success不含速度。

在control6/11前复用原source-bound 51/66 guard；control16前独立核全81原观测、cp15三帧RGB、
25个原public和此前373个操作、动作历史、goal、完整state/snapshot。每项guard须完整写入
journal、flush和fsync返回后才能dispatch；第16步前重新读取81 PNG文件并核字节和解码像素SHA。
失败不补动作。RNG/contact缓存未知，不说是恢复。
消费时重放事件顺序和guard证明，逐PNG验证byte SHA、解码pixel SHA、全操作和计数。

## 预算和失败

每支完整444操作、888操作事件、96观测、3 guard、3 lifecycle共990事件；public30、physics304、
render96。十支300/3040/960，不能只支付尾段增量。每支64MiB含8MiB reserve、磁盘floor200MiB、
墙时900秒；PNG单张172056字节上界。96张+原38MiB metadata quota+8MiB reserve为64751872字节。
batch派发前一次性要求可用磁盘至少10×64MiB加200MiB floor；每支运行仍检查floor。
prepare根据实际来源字段结构推算journal/report上界，超额拒绝；运行时继续字节和事件上界检查。
原生early-stop后不续跑，缺失cp20为null；失败保留partial、unknown计数不写0、不重试。
成功report/completion全文件SHA与事件链由独立consumer重建；未能出版的failure以独立记录封存
已有文件和unknown，不能冒称完整消费。completion墙时记录范围为report哈希结束，自写入排除。

## MAIN 命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；以下路径均canonical绝对且必须独占。
变量填写MAIN实际打印SHA，不能预填或重用其他协议。worker未执行prepare/run/consume。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_settling_tail_physical_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_settling_tail_physical_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_settling_tail_physical_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$SETTLING_PREPARE_SHA"
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_settling_tail_physical_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$SETTLING_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_MAIN01 --authorization MAIN_P3_SETTLING_10_FRESH_ENVS_V1
sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_MAIN01/batch.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_settling_tail_physical_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$SETTLING_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_MAIN01 --batch-sha "$SETTLING_BATCH_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_consume_MAIN01.json
```

本件测试用inert回调和临时PNG/事件日志，不构造物理环境；fixture不能被生产consumer接受。
experiment-code技能只用于固定调度和失败记账，不执行训练或调参。
