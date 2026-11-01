# P4 actual response：独立 saved-array 核验与 live 接口评估

仅新增本stem源码/tests/本文及小报告；不修改任何sealed core、旧结果或共享进度文档。使用experiment-code组织实现/测试，但仅做saved-data与mock，无神经/physics/下载/删除。原始receipt：`results/p4_feedback_root_cuda_response_20261027_v1_cuda_main01/receipt.json`，外部固定SHA `a7fc0c38e17bf379d663ad639d1e2d9ae170b279f172eeda8c71d3b8f07936a8`。

## 实际saved结果

独立核验通过；原prepare SHA `c00f6d8fdc5f59bd0dc49cd7d0f9f54867aeb64cd552b72e00628f71747f60ff`，sourcepack prepare SHA `4de418b03fb7e4a664d9e5d04d8c78b7f7ac467e89f04677329b463fb6489fe2`。输入为同一known root0的step5真实prefix对应imagined/actual两个条件上下文，不是两批实验，更不是live三轮。

| arm | nonlinear原Q最优候选 | 重算Q cost | native J Frobenius |
| --- | ---: | ---: | ---: |
| imagined | 7 | 1.2607695953433102 | 15.135975140326563 |
| actual | 9 | 1.3842826622672255 | 15.676883279488214 |

不同输入context的预测cost不可当物理收益或反馈效能比较。这次确实得到不同的预测候选选择，但没有新物理轨迹或任务success证据。

实际记录资源：26predict、26action_encoder、2 Jacobian API、**384 grad API和384 engine Python入口分别计数**，不是“所有API合计384”；数学标量VJP方向384、处理context tokens78。完整822条计量entry、1673条journal事件精确核对。运行记录wall6.709753s，peak allocated140898304bytes/reserved144703488bytes，均在MAIN给定600s/3GiB/4GiB内。它们是对原运行记录的独立核账，不是本auditor再次实测GPU峰值。

## 独立性与边界

auditor不import response producer、其Ledger、compiler或原candidate checker。只在provenance层调用冻结sourcepack.verify（saved-only），并核对original source seal/archive/args、model bytes、PNG/NPY来源链。56份原文件清单逐项SHA复核；receipt/prepare SHA外部固定，数组审计后再次检查输出字节未变。

独立算法逐项检查：

- 原root/arm/metric/context identity、实际源与预测源历史，以及nominal和AD forward完全相等。
- J为native float32有限非零192×10；独立QJ SVD奇异值、谱向量正交/特征关系/符号。
- 不调用原SVD ridge solver：以10维normal-equation和二分Lagrange乘子重算受限ridge；核对两半径×两模式×正负及完整十槽。
- 原float32 tanh动作、time-major5×2、两条原动作token归一化与三个latent滚动，无元数据错接。
- 所有20候选的完整保存数组、非线性原Q cost、稳定winner、独立replay完整context相等。
- 不调用原Ledger：独立生成nested journal grammar，逐字匹配每个arm/phase/entry index、192组grad→engine attempt/return嵌套及最终finish，拒绝增删、退款、重排或错臂。

**限制：不运行新AD/FD，故没有独立数值重导J的每个元素；不声称仅凭saved arrays重新证明神经执行或导数真值。** 封存源码、实际CUDA收据、J forward和API账本是来源证据；若需要独立导数数值认证，必须另获MAIN CUDA授权。本轮P1/P4仍false。

38项纯mock测试覆盖对角解析J/独立ridge、budget/journal、J/候选/历史/cost/identity篡改、SHA与完整临时文件包；mock不替代实际来源。本轮独立真实audit输出：`results/p4_feedback_root_cuda_response_saved_audit_20261027_v1_main01/report.json`。

## 下一NEW live driver的可执行接口评估

结论：**来源和H1响应kernel具备复用基础，但尚无可启动live driver；不准把本saved response直接admit旧runtime。**

原parameter-root runtime有可借鉴的fail-closed状态机、一次性授权、persist确认、实际history与step20仅评价约束；其bind_sources明确要求旧123-probe物理capability格式。当前P3 sourcepack不满足该格式，必须NEW bridge/driver，不能跳过constructor校验或改旧schema冒充。

已审读P3 sealed physical consumer：它有root seed/reference RGB绑定、`_reset_and_warmup(history_len=3)`、每臂15步warmup、逐步RGB guard和reset/step计量实现可引用。不是“根来源完全不存在”，而是这些材料尚未装配/封存为本NEW live合同。原response Native.bind可接收上下文，但旧material固定保存step5请求、Ledger固定双臂13阶段，run不能直接重用于step10/15。NEW接口必须改在调用层，不修改sealed kernel。

拟定输入/顺序已写入报告的机器可读`live_driver_interface`：

1. 固定root/model/goalQ/reference封存身份；每臂实际reset+15步warmup，核对rootRGB3/raw history/goal与原初始latent，执行真实guarded prefix5并单独重算prefix预测。
2. step5/10/15获取**本臂新轨迹**真实RGB3及两个实际packet；原exact transform→冻结encoder+existingprojector编码，保存native192和来源账。重叠历史校验，不借旧未来帧。actual用observed编译，imagined仅用自身predicted编译；两臂均保留真实观测作审计。
3. 用root+arm+boundary+previous_execution_SHA+context_SHA+goalQ/model_SHA唯一键，重新运行AD/十候选/独立replay；先durable封存选择，再给一次性5步dispatch。
4. 逐步guard与attempt/return落盘，partial立即中止，不补步、重选、退款；step20只做新终图编码和真实任务评价，不倒灌step15。

如prefix预测也重新运行，则全局候选预算应为**80predict/80action_encoder**（6轮×13加两臂prefix各1），不是忽略prefix的78；Jacobian6、grad/engine各1152、处理context tokens240。反馈6次三帧+2终评单帧=20frame，8encoder/8projector调用。真实计划动作40步（含两臂prefix），另外沿P3路径warmup30步，**总public env.step cap至少70**；collector step0，collector/per-arm reset cap必须另行前瞻固定。不能把warmup当免费或把旧单次response的600s/显存cap直接当全局实测上限。

还缺：NEW root reset/reference绑定、动态单请求compiler/ledger bridge、六轮加prefix/encoder的全局预算supervisor、真实guard/partial journal和终评器，及其MAIN审查/source seal/实际运行。现阶段仅评估接口，不写假protocol或启动作业。

## 复现

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_root_cuda_response_saved_audit_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_cuda_response_saved_audit_20261027_v1 \
  --output results/p4_feedback_root_cuda_response_saved_audit_20261027_v1_main01
```

结果目录独占，已存在不得覆盖；复核可换全新后缀。CLI无skip-provenance开关；测试内显式mock模式不验真实来源，其status独立标为mock_arrays_verified_without_provenance，不会得到实际provenance通过。
