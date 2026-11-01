# Actual-cp5 R：复用既有标量分解，不新增 wrapper

只新增本说明，不修改已冻结代码/协议/结果。已读取全部四槽32候选+4zero，调用既有 `main_p3_R_gate_decomposition_20261029_v1.scalar(row, goal)`及其`close()`完成标准库标量复算；不调用producer、resolver生产prepare、神经、physics或fullstrict，不写分析receipt。不因结果修改bank、gate或阈值。

## 实际文件与兼容范围

- prepare：`results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json`，SHA `b4b2f9d3ca1867a86edbb84c753cef3e52d757582299b6f851c3e8c9244a042d`。
- report：`results/agent_p3_actual_cp5_r_delivery_20261029_v1_cuda_MAIN01/report.json`，SHA `deb69d78579b4cdc0ffb3ae4c671dd71f9ecd9b9e02729484085cb3d0c0dd19d`。
- MAIN strict receipt：`results/agent_p3_actual_cp5_r_delivery_20261029_v1_consume_MAIN01.json`，SHA `8d55a55e108aca1695a550861d125a0b0baac2b225cba914d0af4b47d30159a5`。
- 复用源码：`experiments/spectral_reachability/main_p3_R_gate_decomposition_20261029_v1.py`，SHA `910b3186748bb13f4a574dac58c3bfb30418fbffb7c79b9e0b539f423c1b57ef`。

`scalar`需要的postcontext.latent_history[-1]、signal.nominal/jacobian、solve.params/linear_terminal_cost/ridge_penalty/own_reach_value、immediate/reach与cp5报告完全同构。goal取own prepare.tasks[row.root].goal；所有postcontext J均192×10，H3 edge不改变其postcontext H1 suffix的10维接口。identity192 Q合同不变，因此旧scalar的MSE分解适用。

旧`analyze()`不能直接用：AUTH硬绑定control0的prepare/report/receipt SHA。旧`explain()`也不能直接用：要求旧状态`four_groups_complete_not_admitted`与`all4_native_R_groups_verified_not_admitted`；实际cp5状态分别为`four_slots_settled_not_admitted`和`actual_cp5_R_model_deliveries_verified_NOT_physical`。禁止改状态、换schema或替换旧AUTH以冒充旧结果。

建议MAIN直接调用现有`scalar()`，在其独立分析中对实际cp5 own schema、三SHA join及完整四槽身份做显式认证，然后逐组分解；不新造wrapper。现有14项纯RAM回归通过（0.29s）：

```bash
.venv/bin/python -m pytest -q tests/test_main_p3_R_gate_decomposition_20261029_v1.py
```

## 实际资格断点（四槽完整保留）

| cp5组 | immediate严格大于zero | reach严格小于zero | 同时满足 | 退步项中gain增大 |
| --- | ---: | ---: | ---: | ---: |
| root0 H1 | 4 | 4 | 0 | 2 |
| root0 H3 | 4 | 3 | 0 | 0 |
| root1 H1 | 6 | 2 | 0 | 1 |
| root1 H3 | 6 | 2 | 0 | 3 |

20个即时退步候选全部reach更差，11个reach改善候选全部不满足即时退步。另root0 H3/index7同时不满足两条件：immediate改善但reach损害。全部32项不是“已有eligible被delivery排序丢弃”；四组eligible皆空，因此whole/segmented未派发、delivery为null。上述计数与control0相同不代表context、J或数值路径相同。

严格gate使用原封存immediate/reach的`>`/`<`，不使用算术close容差。标量复算验证的恒等式为：

`delta_reach = delta_immediate + delta_nominal_drift - delta_bounded_linear_gain`。

gain是同postcontext的H1局部有界线性ridge代价降低，不是物理可控性完全度量。20个退步项中6个gain改善仍不足抵偿nominal任务残差增长；其余14个gain未改善。不能把reach单独变化全称为“能力变化”。

例如root1 H1/index2：delta_immediate=+0.04042331835655966，delta_nominal_drift=+0.00414864349565458，delta_gain=+0.03390209983543491，合成delta_reach=+0.01066986201677933。虽局部gain增大，仍未满足reach改善。

root0 H3/index7：delta_immediate=−0.05992813087459048，delta_nominal_drift=−0.008445012642126493，delta_gain=−0.14389125623272925，合成delta_reach=+0.07551811271601228，因此两条件均失败。

## 边界与后续建议

本说明只是MAIN已认证receipt之上的保存标量算术，不重复或自签fullstrict，不验证神经输出parity。没有新增witness可交物理driver；driver应保留四槽no_witness，不dispatch、不借旧计划替代。0/32仅约束这两已见根、actual-cp5与事前固定spectral8 bank；不能推出全动作空间无解、一般理论不可能、P1/P3/C3或物理收益结论。

下一研究选择应由MAIN另行事前固定，不以本次失败偷偷换bank/阈值。眼前无需新增工程wrapper：原scalar接口已足够解释这次拒绝，完整物理线仍由其冻结guard合同约束。本文件交付后冻结，不追加生产结果。
