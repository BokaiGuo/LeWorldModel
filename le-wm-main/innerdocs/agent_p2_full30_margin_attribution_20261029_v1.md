# Full30 同时间分项精确 margin：固定 saved 分析

唯一实际输入是`main_p2_full30_same_time_summary_20261029_v1_MAIN01.json`，SHA `d9b97b5c351bcee90f1d2bde5598d2d2c6469dd7bcceb09b9a66117cad00fa16`。仅消费该summary，不访问其上游prepare/report/receipt，不导入producer数学，不重跑fullstrict/NN/physics。保留原参数/源码SHA及资源向量；unknown不补0。旧文件不改。

固定四branch（root0/1各transfer、joint），control0/5/10/15/20/25/30＝28cp，absolute=control+10；两根每时刻joint−transfer＝14pair。保留完整panel、每pair正负/tie/null、缺帧位置；不以terminal补缺cp，不按结果筛指标/根。原native success和angle栏原样保留，wrapped不替换原指标。真实all_four_full30=true时不允许缺项；其他合法早停ABI允许null。

记保存float64标量为q。Fraction精确的是**保存标量值**，不是精确动力学/物理收益。

- E_k=Fraction(q_right,k)−Fraction(q_left,k)。T/P/A/V为native7平方/位置平方/原始角平方/速度平方。
- D_k=float64(q_right,k−q_left,k)，S_k=Fraction(D_k)−E_k为逐项相减舍入。
- R=E_T−E_P−E_A−E_V为保存标量归约差；位置另分agent/object及其归约差。
- 严格验证Fraction(D_T)−ΣFraction(D_axis)=R+S_T−ΣS_axis，左右有理数完全相等，无容差。

每cp的保存标量归约差另核float64误差界（非科学门）。沿原summary的math.fsum核其float decomposition_roundoff，不能把这个再次舍入后的数当精确R。轴项与total同向/反向/tie或total_tie分列；total为tie仍保留非零相消项。位置与总距/总平方的反向索引只是完整14pair表的导航，不是新分母或selector。只作算术归因，不声称因果控制、唯一根因、部署修复或P1/P2成立。

## MAIN CLI（所有路径canonical绝对）

在`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_full30_margin_attribution_20261029_v1.py
MOD=experiments.spectral_reachability.agent_p2_full30_margin_attribution_20261029_v1
.venv/bin/python -B -m "$MOD" analyze \
 --summary /home/gbk/code/LeWorldModel/le-wm-main/results/main_p2_full30_same_time_summary_20261029_v1_MAIN01.json \
 --summary-sha d9b97b5c351bcee90f1d2bde5598d2d2c6469dd7bcceb09b9a66117cad00fa16 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_margin_attribution_20261029_v1_MAIN01.json
# A为上步实际输出绝对路径，AH为实际SHA。
.venv/bin/python -B -m "$MOD" verify --analysis "$A" --analysis-sha "$AH"
```

排他写、16MiBcap、文件及目录fsync；verify整份重算强等。worker仅RAM和指定summary的只读检查；未写真实analysis产物，MAIN审读后执行。

## 测试与只读全表结果

首次RAM为19 passed/1 failed（0.19s），唯一失败是fixture所选二进制浮点数恰好相减精确，没有产生测试要求的非零相减舍入。只将fixture右侧位置平方从0.3改为1.0、总平方从1.2改为1.9；严格有理数恒等式和实际分析未变。最终20 passed/0.20s，exit0。包含全28cp/14pair、null、total-tie非零相消、双层非零舍入、ABI/指标/分母篡改、RAM持久化与整份verify回归。本三件套至此冻结。

指定SHA输入的只读算术：native7平方margin负5/正1/tie8/null0；位置平方负1/正5/tie8/null0；角平方负5/正0/tie9/null0；速度平方负5/正1/tie8/null0。没有改变14pair分母。

位置与总距/总平方反向共4项，均root0，方向joint−transfer；以下仅为完整表导航，数值显示四舍五入，精确值见脚本Fraction栏：

| control cp | Δ位置² | Δ原始角² | Δ速度² | Δnative7² |
|---|---:|---:|---:|---:|
| 5 | +10170.430 | 0 | −33788.797 | −23618.367 |
| 10 | +13343.971 | −0.086 | −20812.175 | −7468.290 |
| 15 | +10865.183 | −0.108 | −37910.726 | −27045.652 |
| 25 | +3778.393 | −0.306 | −4330.984 | −552.897 |

这4项速度平方下降在算术上超过位置平方上升；cp10/15/25角平方下降也与总平方同向。不是因果干预，不说明较差位置的来源或速度部署修复可行，也不把总距较小等同任务成功。root1和其余时刻、ties都在原全表保留。
