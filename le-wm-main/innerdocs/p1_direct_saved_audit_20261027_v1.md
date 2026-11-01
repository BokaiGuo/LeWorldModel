# P1 direct CUDA saved-array independent audit

审计对象：`results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01`。
顶层 report SHA256：`fad2c2ff4f409762fa5925f142f9b4d3edc371645fc4c114bfa69c996483f3d4`。

结论：本批 4 组件保存的程序/协议收据一致；没有发现可由这些数组复现的
Q/尺度/完整 H2 history/梯度/接收/计账错误。**不是 plateau 或 P1 准入证明**。
主线程执行了真实 CUDA；本审计线程只读取文件、使用 NumPy 重算，没有运行模型、
autograd、reset/render 或物理。旧封存文件没有修改。

## 独立数字

同一原生 projected CLS192、Q=I192，每个组件一个 zero restart，B1=1、B2=3。
cost 为全192维 projected terminal residual 的 mean square。

| root | method | cost0 | cost1(B1) | cost2 | cost3(B2) | B1−B2 | 三次接收 |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | GN | 1.381570742 | 1.290451595 | 1.155789711 | 1.155789711 | 0.134661884 | 是/是/否 |
| 0 | gradient | 1.381570742 | 1.353354346 | 1.314002327 | 1.273638113 | 0.079716233 | 是/是/是 |
| 1 | GN | 2.191056314 | 2.068284459 | 1.968030272 | 1.968030272 | 0.100254187 | 是/是/否 |
| 1 | gradient | 2.191056314 | 1.920785889 | 1.869796051 | 1.830458096 | 0.090327793 | 是/是/是 |

各步**求导点**的 objective gradient L2（iteration0/1/2 incumbent）：

- root0 GN：0.158310233 / 0.326810633 / 0.114468221。
- root0 gradient：0.158310233 / 0.196259038 / 0.202862149。
- root1 GN：0.659553466 / 0.555358646 / 0.261431236。
- root1 gradient：0.659553466 / 0.235257277 / 0.203840438。

gradient 方法最后接受的新 terminal 没有第四次 J，不能把以上最后一个数冒称其最终梯度。
raw-control J 第一包/第二包 Frobenius 范围分别为
[5.229888104,21.590522431] / [5.224827541,17.680970375]；12 个 J 均双包非零。
这是保存导数的支持性证据，不是 independently differentiated 正确性证明，也不是
显著性阈值、可控性下界或真实物理灵敏度。

## 核验内容

独立测试只导入 stdlib/NumPy/pytest，不导入被审计 checker 或模型代码：

- task digest 固定完整 Q/goal/normalizer/root context；root/raw-source/plan SHA 与源包关联；
  8 次 image receipt 的 RGB SHA 和 encoded CLS 与实际 source NPZ/task 一一匹配。
- 每个候选 logits `(2,10)` 按 float32 tanh 精确形成 time-major raw commands `(10,2)`；
  每包 logit L2 半径 `0.75*sqrt(5)`，raw 逐元素界及每包球界均通过。
- 全部16个候选均有两包 nominal、whole replay、segmented replay，逐包检查 latent shift、
  用 float32 mean/std 重算 normalized action token、完整 `(2,10)` action history shift。
  journal 的48条 prediction paths/96条 native packets 与对应 report 完全一致。
- 每个组件3次 J，各192行，共2304条 raw-control derivative rows。直接从 journal 堆叠
  `(192,20)`，仅乘一次 `(1-command²)`，与保存 logit J 精确相等；AD forward 的两包
  contexts/terminal 与对应 incumbent 完全一致。不能用 receipt consistency 代替独立 AD/FD。
- 重算 `2*(QJ).T@Q(z-goal)/192`；GN 重算 MSE ridge SVD、damping=.002 和投影 proposal；
  gradient 重算负梯度 proposal。float32 proposal receipt 绝对容差2e-7；此容差非科学阈值。
- strict `<` 接收规则、incumbent 延续、configuration SHA、trajectory/proposal/derivative
  journal 与 report 均交叉匹配。嵌套 J 内部工作按 index 配对，不能假定完成序等于派发序。
- 每组件30 predict/30 action_encoder/3 J/576数学 VJP directions/3 proposal/3 selection；
  GN另3 SVD和3 solve。总计120/120/12/2304/12/12/6/6；实际 image encoder/projector各8。
  全部 attempted=completed。VJP directions不是 kernel/FLOP 等价计量。
- 顶层 model before/after hash 相等且 errors=[]；这是报告收据一致性，不是本审计重载模型
  独立证明 GPU 参数未变。journal 无外部加密运行认证，不能把文件一致性提升为独立运行证明。

## 理论及科学边界

四条 B1→B2 均继续下降，GN只末步固定proposal拒绝，gradient仍连续接受。
没有授权 plateau threshold、足够长预算稳定性、独立多起点/line-search 收敛证据；
不得把三步预算、单次拒绝或小奇异值称为 stationarity/plateau。
原始 RGB3 是同一帧因果 padding，不是三次真实 observed transitions；零 raw padding 经
normalizer 变换后不是零 token，不应抹掉。来源仍是 calibration-only，全球 freshness unknown。
尚缺合法独立 two-stage witness/replay、独立 calibration 后冻结 confirmation 阈值、
真实非贪心机制与任务收益。`P1_admission=false`、`model_task_eligible=false` 正确保留。

## 复现

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_direct_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_p1_direct_saved_audit_20261027_v1.py
```

第一条运行实际保存数组审计及六种篡改负测试（gradient/history/acceptance/J/Q/ball）；
第二条打印独立完整精度数字，不写文件、不运行神经或物理。
