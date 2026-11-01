# control15 O/I/F v2：prepare库存集合比较接缝修复

## 原因与最小变动

MAIN实际v1 prepare失败，未产生结果文件，未进入模型候选。根因是新runner build把Python set传给
JSON canonical等价器equal，触发 `TypeError: Object of type set is not JSON serializable`。
此前47项RAM测试覆盖material与完整saved native链，但没有调用完整build路径；这是代码/测试覆盖缺口，
不是理论失败、模型结果或科学负结果。没有更改v1源、kernel、测试、文档或旧prepare/结果。

仅新增本stem v2 runner/tests/本文。集合库存比较改为
`require(actual_names == expected_names, 'entire inherited encoding inventory')`。
其余算法复用冻结v1 kernel，SHA
`2ea1e14447031865eef617c932da90ad2b25f72f23ddcbedff52f117f87c3fcf`，build显式校验该pin。
v2有独立NAME、prepare/contract/report/plan/completion/consume schema、输出prefix、授权字面值和own files闭包；
contract仅schema改为v2，所有算法域/alpha/阈值/资源字段与v1逐项相等，不猴补旧globals。
files绑定v2 runner/tests/doc及冻结v1 kernel，继承源闭包与8类installed源存在检查不变。

## 保持的固定实验与预算

原两根×transfer/joint四base，O/I/F固定12策略槽，I/F合8新job。
每job27predict/1J：全216predict/8J/1536 requested VJP/46080 J元素/24已含AD primal。
0新encoder/projector/physics/train/SVD/solve；未知engine backward/FLOPs不填0。
标签白名单仍cp≤15；不按outcome选根、调候选或改变原生成功。future真实cp20..30不进入生成器。
原float64父witness→tanh64→cast32 raw，transfer独立球/joint剩余域、alpha1/.5/.25/.125与incumbent guard不变。
同context nonlinear terminalQ稳定选优只保证模型非劣，不保证真实任务或科学门。

128MiB输出/2MiB reserve、floor200MiB、whole1800s/job450s、CUDA4GiB allocated/6GiB reserved不变。
完整216IO/8J/3068events保守运输上界43,677,726 bytes；包含逐event压缩上界、每条64-byte SYNC_FLUSH余量、
128-byte封装尾部、NPY headers、重复row/plan与全部4MiB runtime metadata额度。
report fsync/hash后独立completion记录elapsed；consumer必须同时绑定外部report SHA及completion SHA。
completion后运输再runtime wall检查，超时failure marker使完整库存拒签，不用出版前wall冒充inclusive。
源/权重前后hash、模型class source、precision前后、完整事件时序、partial首因及失败不退款均保留。

## 输入权威与验收边界

默认固定实际RGB三权威，路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`：

- `results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json`：
  `20a248dc940188ff3858fe71769cdc2d726ec23d2e77e8e50783ff777db63527`。
- `results/agent_p2_two_program_full30_rgb_encode_20261029_v1_cuda_MAIN01/report.json`：
  `915b6760db8bf30abe16e4bfba347604e6359c17d08c1ff3f2a7a076b7714df4`。
- `results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json`：
  `42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166`。

继承MAIN完整上游model/native/RGB receipt，显式full_upstream_rebuilt_here=False，独立重hash闭包；
不回调旧build/verify/consume/fullstrict，不把指针“已付”当任意新artifact认证。
P1/P2mechanism/C3/scientific/physics/fresh门全false。不是物理部署器，也不声称等FLOPs或跨根确认。

v2重跑全部47项v1等价RAM/磁盘回归，并新增完整build临时文件ABI夹具：实际创建prepare/report/receipt、
artifact/completion目录、全部已付binding，通过完整build执行库存比较而非只material smoke。
测试不替换files/source闭包生成器；用独立临时authority pins模拟已认证输入，禁止嵌套上游fullstrict。
另验多余库存文件、继承binding篡改拒绝、v2 own源码/测试/doc/kernel全部纳入hash闭包，
以及contract除schema外与v1逐项相同。夹具不是实际模型结果或生产prepare。
最终完整RAM/临时磁盘回归：51 passed in 14.01s；完整运输上界仍43,677,726 bytes。

## MAIN完整CLI

worker仅RAM/临时磁盘，无生产prepare/fullstrict/NN/physics。MAIN工作目录：
`/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
.venv/bin/python -m pytest -q -s tests/test_agent_p2_full30_control15_feedback_20261029_v2.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v2 prepare \
  --output results/agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json
```

从真实prepare返回取PREP_SHA；无未来SHA：

```bash
PREP_SHA=实际prepare_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v2 verify \
  --prepared results/agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json --prepare-sha "$PREP_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v2 run \
  --prepared results/agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json --prepare-sha "$PREP_SHA" \
  --output results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01 --authorize-host-cuda
```

从run实际返回取report sha256及completion_sha256：

```bash
REPORT_SHA=实际report_SHA
COMPLETION_SHA=实际completion_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v2 consume \
  --output results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01 \
  --report-sha "$REPORT_SHA" --completion-sha "$COMPLETION_SHA" \
  --receipt results/agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json
```

剩余MAIN最终审读/独立测试/实际prepare→verify→HOSTCUDA→own consume。v1失败保留为工程首因，
不覆写旧文件，不重启物理协议或增加实验范围。
