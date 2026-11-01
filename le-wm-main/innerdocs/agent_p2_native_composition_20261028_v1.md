# P2 native composition：已有等价实现，直接复用而非新runner

本轮只新增本stem source/tests/本文。未改runtime、native kernel、saved consumer、MAIN
文件、封存结果、共享总账；无后代agent、下载、训练、Q/readout/normalizer拟合、阈值/根筛选。
祖先与项目树未发现适用AGENTS.md。完整读取experiment-code技能，采用其先读既有代码与
真实资产、定点接线测试流程；不适用训练、择优或强制正结果要求。

## 结论与精确复用入口

现有 `p2_composition_predict_frontier_cuda_20261027_v2.run_arm` 已实现最小条件两程序
组合与无组合SRA对照。本source仅导出原函数对象，零算法/dispatch wrapper，不另造
manifest、runner、消费者或新prepare schema。导出别名如下：

| 本模块入口 | 原函数 | 用途 |
| --- | --- | --- |
| prepare_known / verify_known | runtime_guard.prepare / verify | 已有全闭包prepare与只读verify |
| execute_known / run_known_root | runtime_guard同名函数 | MAIN专用CUDA已durable预算执行路径 |
| run_native_arm | frontier_cuda_v2.run_arm | 原80/40候选、动作见证、winner replay |
| compile_root_cuda / compile_suffix_cuda | frontier_cuda_v2.compile_cuda / suffix_cuda | 原root及真实中间context编译 |
| candidate_bank | frontier_cuda_v2.candidate_bank | 原joint80和其前40个prefix生成 |
| resource_allocation | resume_next.allocation | 原base/edge/replay/未来physical预算合同 |
| compose_action_witness | d2p_composition_witness_v2.compose | 原逐边完整上下文/动作重放合同 |

缩写模块路径均为 `experiments.spectral_reachability` 下封存原文件；runtime_guard为
`agent_p2_runtime_guard_20261028_v1`。低层入口不是执行授权，生产集成只走MAIN审过的
execute_known，不直接把RAM callback注入实际神经路径。别名没有放宽原CUDA adapter检查。
旧saved消费仍只使用 `agent_p2_guard_saved_consume_20261028_v1.consume`，不重复实现/运行。

## 真实组合语义与明确边界

每个条件候选按固定序号执行：

1. 原root全context执行固定5-step prefix，得到mid的3 latent states与2 between-state动作。
2. 在此mid用零suffix的nominal与192×10 Jacobian编译原Q/goal残差；半径严格为原joint
   logit球扣除prefix范数，damping原.002。零余量仍付nominal/AD，SVD/solve真实0不伪补。
3. 执行新suffix的5 raw actions，保存root→mid、mid→end两条动作边；第二边输入精确等于
   第一边输出。上下文history保留、动作原float32 tanh映射、goal/Q/model/normalizer不变。
4. 固定40完整候选全计分、稳定first-min选择，winner独立分段与整段重放4 predicts。

无组合臂执行原joint-H2 SRA 80个完整20-logit提案，每个2 packets，但见证为一条full10边。
前40 prefix相同；不调用suffix compiler、不按中间分数剪枝/缓存/补根。
这是真实“先执行prefix预测，再依其完整context重新编译第二程序”，不是对旧动作重评分。
它不等同于“两个独立预编译程序库元素直接拼接”，也没有任意深度、非贪心多程序搜索。
若研究目标要求后两者，当前并无等价实现；不能因存在两条边就宣称广义组合理论成立。
saved_witness内部以已有contexts绑定并非独立神经重放，真实winner重放在run_arm执行。

## 预算和故障分母

复用guard.run_known_root已有primitive reserve→entry→return接线，不用本source重计或替代。
每根shared4 predicts，joint80×2、conditional40×4，各winner replay4：standalone168/168、
unique332/root，总4 known roots 1328；AE同数。shared在unique一次，standalone各全额。
每根Jacobian API1 vs41，数学VJP192 vs7872；不是backward-engine调用数或FLOPs。
每根shared SVD2/solve1，conditional suffix额外≤40，selection80/40。
边级上限之和闭合到原candidate上限；future warmup/physics/endpoint预算未执行，不能称已付。
未知FLOPs/同步device时间/峰值/历史获取等保留null，common cap不代表实际总资源相等。

native.run_arm失败时保留部分attempts，不伪装40/80条都完成；outer runtime.DispatchGuard/
report才负责固定4 roots、8 arms、480候选的failed/pending/skipped完整分母。
失败无deployable winner、无退款/替根/重试，durable断裂仍fail closed。本别名不改变异常策略。

## 已有真实资产（只读核对）

- runtime prepared: `results/agent_p2_runtime_guard_20261028_v1_prepared01.json`，SHA
  `02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48`。
- native prepared canonical SHA
  `1d79f5b17002334434454dfb5a94e6d918de1d603afdf84918e5ded6eb980d1b`。
- MAIN CUDA report `results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/report.json`，SHA
  `0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0`。
- MAIN saved receipt `results/agent_p2_guard_saved_consume_20261028_v1_MAIN02.json`，SHA
  `c7d183f5b18d1bfacd81f855025efdbbb403f97e5442650970080766db17088b`（已有MAIN验收，不重消费）。

本tests逐项读取4根480条真实动作见证，检查320单边/160双边与中间context、prefix、
suffix compile digest接缝。不是新增consumer，也不再计算全套Q/ledger或新保存receipt。
已有MAIN02严格消费负责完整saved数组/评分/资源真实性一致性检查；本轮不替代其范围。

## 可执行命令与未完成项

只读verify（无模型加载、无新prepare、无CUDA）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_runtime_guard_20261028_v1 verify --prepared results/agent_p2_runtime_guard_20261028_v1_prepared01.json --prepare-sha 02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p2_native_composition_20261028_v1_tests.py tests/test_p2_composition_predict_frontier_cuda_20261027_v2.py tests/agent_p2_runtime_guard_20261028_v1_tests.py
```

已存在的execute_known生产入口不需worker重跑；MAIN已有真实4根工程结果。
未来若MAIN批准独立工程重跑，需独占新runtime_guard输出、外部review SHA，按旧CLI
execute-known接原prepared，不更改本alias或旧prepare闭包。无confirmatory/physical命令。
execute_known(confirmatory=True)在读取/加载前直接拒绝；本source没有任何物理放行接口。

精确未完成：独立P1的真实非贪心/能力变化/原task收益与MAIN authority；fresh roots及完整
排除/goal7/history/blind认证；完整硬件成本和共同停止规则；exact physical恢复、原agent+
object native success/controller/renderer合同与MAIN放行；广义预编译程序库组合/多深度搜索。
这些不是另一个metadata层能补齐，当前不为凑并行重复实现。工程可复用≠P1/P2成立。

测试仅RAM callbacks/metadata/已保存数组。首次测试9 passed/1 failed：新测试把40次suffix
编译中的nominal+ADforward回调共80条误写40；只修正新测试计数，未修改任何旧kernel。

最终实际联合回归 **88 passed in 35.84s**（本stem10 + native v2 58 + guard20）；
上述原prepared实际verify退出0，stdout为`verified_no_dispatch`。旧guard/native源码与
MAIN02字节SHA均与冻结值相等。未执行新CUDA/physics或独立saved消费；测试不产生科学证据。
