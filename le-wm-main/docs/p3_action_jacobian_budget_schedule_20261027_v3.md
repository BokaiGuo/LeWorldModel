# Fixed-H3 action-Jacobian scheduler v3

新模块是明确语义修复，不改动或追认v2。v2使用未去自主漂移的H1有限位移span，
不能称action-Jacobian；v3只改变chi来源，不按已开outcome调整阈值或公式。

## 信号与合法复用

唯一继承源是 `results/pusht_full_packet_compiler_20261027_cuda_v1/report.json`，
SHA256 `05eb37c9b1fc3fc4976da702239b13dce018eb9bb4067660687ada4e6d3988dc`。
读取全部四根的 `root_jacobian_full` (192×10) 与 `root_nominal_terminal` (192)。
它们来自原始root context下一包五步、十维time-major action logits零点：
`J = d terminal(tanh(u).reshape(5,2))/du | u=0`，零点tanh导数为1，
因此此点也等于raw-action导数；不是constant2、successor/suffix或H2导数。

使用原task的Q和goal0，`r=Q(goal0−nominal)`、`K=QJ`，SVD(K)的左奇异向量
定义col(K)。相对奇异值截断仍1e-6，`chi=||(I−UUᵀ)r||²/||r||²`。
拒绝非有限数组、零residual/零rank和非对称幂等Q，不fallback到uniform。
这是root-H1局部线性可达像，不是H3整体可控性或物理保证。

prepare校验继承report精确hash、源seal live/archive、四份独立plan与report一致、
finite-difference≤.1与chain-rule≤5e-5原完整性合同，并核验同root全集、同原始
compilation路径/hash、同冻结weights/config。v2 packet须与该compilation重建完全
一致，含root latent/action history、goal、Q和16bank；不从继承源选择胜者或bank。
不读取本批physical outcome作为调度输入。若任何绑定失效，prepare fail closed；
不自动换根/改用suffix Jacobian，也不暗中计算fresh Jacobian。

## 调度与成本

仍uniform=4/4，dynamic `K2=2+floor(4chi+.5), K3=8−K2`；全16bank、H3-only
winner、原Q/goal、cost后lexicographic path稳定tie、原交替arm顺序不变。
两臂都在完整16个fresh H1预测后各重算一次同源Jacobian-array SVD；uniform忽略chi。
每arm 144 planning + 3独立整路径verification =147 predict samples，cohort1176；
action encoder1176、context token3528；全完成8次新数组SVD，新增Jacobian0。
prepare另有4次数组SVD，独立记录为prepare_numpy_svd_calls，不混入执行期8次。
所有重复边、失败、guard以外的模型异常、blocked slots与journal保留，无补预算/重试。
本模块无physics或终态image encoding；旧v2 physical consumer不适用于v3协议。

四份继承Jacobian/nominal属于共享已获取输入，cohort只计一次继承，不能将每臂
receipt中的描述性引用重复相加。根级获取成本为 `inherited_unknown_cost`：predict、
action-encoder、backward API calls、VJP方向、walltime全部null，而不是0。
源report包含更大compiler工作流的总体计数，不能精确拆成所用四份root数组，故
不以该总体计数伪装根级精确成本。新增runtime no-grad预测无backward是代码策略，
不是autograd engine invocation测量。只能称本次搜索预测坐标匹配，不能称全pipeline
equivalent或FLOPs相等；模型加载、RGB上游编码、bank编译仍不在147预测内。

## 执行

仅prepare和数组/metadata tests可在本轮运行；CUDA由主线程review后显式启动。
prepare不导入torch/模拟器、不加载权重；模型文件全量hash在CUDA执行入口验证。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_p3_action_jacobian_budget_schedule_20261027_v3.py
.venv/bin/python -m experiments.spectral_reachability.p3_action_jacobian_budget_schedule_20261027_v3 --prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_budget_schedule_20261027_v3_prepared
.venv/bin/python -m experiments.spectral_reachability.p3_action_jacobian_budget_schedule_20261027_v3 --execute-cuda --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_budget_schedule_20261027_v3_prepared --prepare-sha256 REVIEWED_PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_budget_schedule_20261027_v3_cuda_main_v1
```

所有output须独占新目录。四根仍全是calibration、独立新根0，P1/P3不准入。
fixtures证明合同，不是新CUDA/physics实验证据。技能使用：experiment-code促使
分离prepare/显式CUDA与收据；TDD覆盖漂移混淆反例和非法复用/成本回归。
