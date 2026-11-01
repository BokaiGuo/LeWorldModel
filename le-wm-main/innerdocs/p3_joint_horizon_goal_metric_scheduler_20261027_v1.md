# P3 joint-horizon goal-metric scheduler v1

独立实现 outer `docs/p3_goal_patch_consistent_compiler_spec_20261027_v1.md` 的 root spectral-ridge + random-tail 四臂提案。**不是 Arendt 的 full-packet conditional fixed-H2 P1 compiler**；不导入或改动其文件。所有 P admission 保持 false。仅 main 在 review 后可执行 CUDA；无 CPU 神经、训练、physics 或 outcome 参数选择入口。

## 输入前置条件

复用 `p3p4_goal_patch_replay_20261027_v1.validate_metric` 的完整 model-free 验证：固定四根 v4 source context/goal/旧 Q，sealed goal RGB catalog/legacy vectors/新 Q、metric journal 与完整旧候选并集；不是只读有利 winner。当前材料真实存在，独立只读验证通过（4 packets/8 metric plans）。该 validator 固定本 panel 的 80 encoder +80 existing projector +8 metric SVD 历史记录；本模块不重新构造 Q，不读物理结果，也不把这些 inherited 成本当零。

原 source input hash 保持不变。active metric id 额外绑定 root、原 input、Q 的 little-endian float64 shape/dtype/bytes SHA、goal、metric report/metric SHA 与模型绑定。root join 用 root SHA 字典；原 packet 的位置只冻结为 seed_index，不作身份 join。prepare 将所有材料和源码 closure 封存到全新目录，verify 重新构造；缺失或漂移严格拒绝，不 fallback。

normalization另以AST读取封存 `experiments/run_pusht_counterfactual.py` 的两个 `np.asarray` 字面量，绑定源码SHA与float32 mean/std，不导入main固定H2 auditor/conditional compiler。public常量的training provenance未验证，不冒称checkpoint训练统计。

## 算法与坐标

四臂 `old_Q__uniform / old_Q__dynamic / goal_patch_Q__uniform / goal_patch_Q__dynamic`。四根按循环置换平衡四臂顺序，prepare 前固定。

共享零 joint logits 处 native raw J_H `[192,20/30]` 和 nominal `[192]`，H2/H3 各一次 fresh Jacobian；全滚动图无 detach。每 Q/H 独立求 `K=QJ`、`r=Q(goal0−nominal)`，用旧 residual chi 和 bounded ridge `min ||Kd−r||²+.002||d||²`、`||d||<=.75sqrt(5H)`。signal 与 ridge 各一次数组 SVD；ridge 不做 chi 的 rank 截断。展示代价统一除以192，正则问题不能将 K/r Gram 除192后仍用 .002。

候选0 zero；候选1 own-Q ridge logits 的 tanh；k>=2 **原 pool[k]**，不左移随机尾部。seed `20261027+100*sealed_seed_index+H`。新 Q 真实改变候选1，不仅改排序。旧 Q 也用同一新生成器；历史随机池不能冒充本实验对照。v4 random pool 本身不是 spectral generation，本实现是 SVD-ridge root proposal，非多模式 spectral bank、非 iterative MPC、非 conditional suffix。

logits 为 joint time-major 10H，tanh 后 `[5H,2]` raw packet controls；不是逐包 ball、hard clipping 或 `/100`。复用 frozen `make_transition`/`CudaReplay` 的 public action normalization 与 full3 context。模型 frozen eval/同 CUDA device；AD、搜索、整段/分段回放都在 SDPA MATH backend。double replay 必须完整 context 精确等于候选。没有误差放宽或多次重试。

各 metric 自己的 chi 驱动 dynamic；uniform 5/5 blocks，dynamic 仍为旧公式。每 block6 predict，H2有3候选/H3有2候选：每臂正好60 search predict。zero/duplicate 也执行付费，不去重。保留全部候选和稳定 `(cost,H,index)` argmin。

## 预算与失败

每根共享10 predict（H2 nominal2+ADforward2，H3 nominal3+ADforward3）；每臂60+2H winner replay，独占上限66。四臂总实际 `250+2sum(Hwinner)`，上限274/root、1096/four roots。action_encoder 单独同上限；两类 module entries 总上限2192。无 padding、失败退款或把剩余额度转搜索。

每根2 Jacobian、384数学 VJP方向、2 jacobian API调用；backward engine calls未知。AD forward已计入共享10，backward开销另列。每根8执行SVD（2Q×2H×signal/ridge）；模型验证每次另16 saved-array SVD，export另外32 response重建SVD。cached history成本未知，Q构造历史成本另列；**不称等总预算/FLOPs**。不同winner H导致实际回放/未来physical长度不同，same-H与mixed-H必须分层；此版本无physical consumer。

NeuralCap预dispatch计attempt，每次结果记completed；失败仍占额。原始J、nominal、K/r、solve、候选与回放结果立即挂到根结果对象，并即时写fsync数组journal：nominal及AD-forward每个已完成prefix boundary都保存完整context，nominal terminal/J/K/r/solve各自产生即保存；候选context在评分前保存；whole/各segment回放各自产生即保存。prefix快照使用detach的CPU副本，仅用于保存，下一transition仍接原state/between图，不detach递推路径，不增加模型调用。

root execution异常先记为首错，再分别尝试journal SHA/root JSON，二者失败分别追加 `root_journal_hash/root_write`，不从finally抛出覆盖首错。hook恢复/terminal journal失败、final模型哈希/source验证各自留存；根finalization失败阻断后续根，不自动重跑。进程被硬杀/磁盘全部写失败时只能以已有journal为下界，不能保证新数组/最终report已落盘，也不能宣称完整成功。

export 是只读重建：四根四臂census、own-Q响应/ridge、bank/代价/argmin、实际entry账与journal SHA、完整double replay；不运行模型。每个candidate/whole/segmented/nominal与AD-forward prefix context均以独立NumPy float32减法、除法、time-major reshape重新生成action token并滚动末两段history（支持H1 segment、H2、H3丢弃最早token），逐位校验，不只是重复receipt相等。检查META类型/值、完整metadata保持、有限latent维度和仍在窗口的latent顺序；不能验证未重跑的预测latent值本身来自模型。`physical_ready=False`，未来消费者需独立新版本封存，不能套旧两臂36replay/90commands合同。

输出体积为JSON重复保存的工程估计，非实际CUDA结果：四根execution目录预计约40–70 MB，建议预留100 MB；包括report、四root JSON及含即时数组的journal，不包含模型weights/未来physics。prepare协议含多份192×192 Q和源码快照，另计其实际磁盘体积。数组副本增大CPU/磁盘开销，不改变forward/AD/SVD预算，也不称等总资源。

## main review 后的命令

下列为本次交付使用的独占prepare路径。prepare只做输入/数组/源码工作；prepare的实际SHA由命令输出并另行交给main；不把尚未完成prepare写作已封存。只有显式execute-cuda能加载模型，agent不执行它。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p3_joint_horizon_goal_metric_scheduler_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p3_joint_horizon_goal_metric_scheduler_20261027_v1 --prepare --output results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01
sha256sum results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01/prepare.json
```

main独立审阅prepare并记录SHA后，用真实SHA替换下列占位值；这不是本agent执行授权：

```bash
.venv/bin/python -m experiments.spectral_reachability.p3_joint_horizon_goal_metric_scheduler_20261027_v1 --execute-cuda --prepared results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01 --prepare-sha256 ACTUAL_PREPARE_SHA --output results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01
```

本次38项测试通过；仅NumPy与constant-interface mocks，验证预算、own-Q候选、随机尾部、stable tie、失败attempt/部分数组与source seal。覆盖root journal SHA/root write分别或同时失败时保留execution首错；nominal/AD-prefix失败前已有数组journal；H1/H2/H3 token算术、倒序、错误metadata、float64替代float32、额外/100均拒绝；所有重复receipt同改为伪history也被export拒绝。mock完整替换loader、device/preflight、torch接口，不创建神经模型或执行真实tensor/AD。测试不认证真实CUDA导数或物理质量。P1/P3未通过，本次仅非神经验证及prepare，等待main CUDA。
