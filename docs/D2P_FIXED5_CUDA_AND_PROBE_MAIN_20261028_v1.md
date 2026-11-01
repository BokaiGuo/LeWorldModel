# 固定五步 P3 CUDA 与 P1 前后能力探针

## 已实际完成

- probe v2完整saved prepare成功：`le-wm-main/results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json`，SHA `e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5`，约8.9MiB。旧8MiB容量拒绝是工程问题，失败版本不改；64MiB v2保留全部闭包，没有减少分母或丢弃绑定。
- 新40分支物理执行器完整源码/文档审读，含逐动作真实 RGB guard；最新联合回归70 passed、2 skipped（RAM/saved测试，不是实际物理结果）。own prepare实际8,452,251 bytes，SHA `8226f8a77a7584c598ec3ce137926703a99029ec58160a1024c8c1b76630f394`。
- P3 fixed5源码/测试/文档MAIN审读及101项联合回归通过，新增MAIN独立Q/稳定tie/raw5复算器另5项测试通过。
- fixed5实际HOST CUDA完成：report `le-wm-main/results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01/report.json`，SHA `8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5`，errors=[]，32候选及4个赢家首包全部返回。设备为RTX5060Ti，CUDA allocation_sum=4.0，无CPU神经回退。报告墙钟127.3038秒（最终写盘开销另计）。

## 预算与解释范围

新执行100predict API/sample；继承已付信号后四臂各33predict/sample、2J/384VJP。各臂请求实际执行5步，模型规划H为1/3/1/3，未在此CUDA作业执行物理。不能声称FLOPs、wall、最廉价fixed-H1成本或闭环收益相同。

报告原规划终点Q：root0 fixed/adaptive为0.9454683415/0.9086686599；root1为1.2824975012/1.2831926497。规划终点不相同，不能用此直接证明共同五步任务收益。

MAIN严格消费者及独立全部32候选Q、稳定tie-break、四条raw5复算已完成，回执 `le-wm-main/results/consume_p3_fixed_execution_MAIN_20261028_v1_MAIN01.json`，SHA `a8770ed8242ed8d2f7f27eac0002d7f0bfabbd8ed5084c5d7d933f8793bc0388`。共同五步模型Q为root0 fixed/adaptive 0.9454683415/0.8897802376，root1为1.2824975012/1.6456318951；一好一坏，仍不是物理任务收益。完整producer consume及独立算术均无新模型调用。

## 已启动、尚未验收

- fixed5严格consume+MAIN独立复算已完成，见上述实际回执；不再标为运行中。
- 真实40分支before/after物理已退出0且全部完成，report SHA `57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8`，errors=[]，40/40 completed，墙钟82.6490秒，700 public/7160内部physics/2340 render。固定before/after各5种probe，四条件、两个known根，完整40分母，没有神经模型调用。
- strict全闭包physical/probe消费者已成功，持久回执 `results/consume_mixedmode_capability_MAIN_20261028_v1_MAIN01.json`，SHA `d1efceb6aa0eb4e72a586bb0c1ba297ecb433f12e048e7a0c966450b71156537`，status `saved_probe_responses_not_admitted`、errors=[]、分母40。此前session22057运行中快照已被终态取代。真实响应已验收，任务有符号分析仍待交付；不是P1科学准入。
- MAIN另写不导入任何producer/model/env的独立原生算术审计，3项测试通过后实际重算全部26,322事件hashchain/700原始NPZ奖励与终止/状态7维info join/资源计数，全部一致。实际回执 `results/audit_mixedmode_capability_native_MAIN_20261028_v1_MAIN01.json` SHA `4e90b88ffb135aeafdd63b68cda57b75f6315638bf1f6829a75822d8d07b622e`。该原生算术核对不能取代完整producer consume或有符号任务响应分析。

P1仍未通过，P3仍不是完整child-local树或科学准入，两个known根不是独立泛化。前后探针存在时间/速度/接触混杂，不能单凭无符号谱变大或终距下降认定有利能力改变。P2–P5保持原科学门槛并行补实际实现，不把代码/测试/prepare当研究闭环。

最后联合回归（fixed5 CUDA/合同/独立算术、40probe物理/v2合同/独立原生算术）实际179 passed、2 skipped in50.69s。未启用两个显式saved-build测试，不冒称已运行；完整实际prepare和严格消费者已另行完成，证据按上述产物分账。
