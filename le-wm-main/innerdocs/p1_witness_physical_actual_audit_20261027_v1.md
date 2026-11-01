# P1 witness actual physical：保存证据独立审计及24帧编码交接

## 结论

主线程真实物理完成，独立saved-only审计通过。8/8分支全部10步，16 resets、80 steps、
88 explicit renders的attempted=completed。整条hash-chain、固定动作、返回/实际state、
初始root/RGB、24个PNG、88个fresh guard receipt、逐步/终态指标一致。
本审查没有运行模拟、神经或GPU，没有修改producer、prepare、结果或共享文档。

官方success为0/8，独立wrapped同allxy阈值success也为0/8。实际80步中：
负native角度项0次、native/wrapped success分歧0次、提前termination/truncation0次。
native角度公式的多圈缺陷仍是一般协议风险，但没有在本批已保存步骤触发，不能用来
解释这些实际负结果。2π表示差不作为几何错位。

不是equal-compute胜率、独立确认或P1pass。两根校准源的重复臂不是8个独立研究样本。

## 证据与复算范围

根目录 `/home/gbk/code/LeWorldModel/le-wm-main`。

| 文件 | SHA256 |
| --- | --- |
| `results/p1_witness_physical_20261027_v1_main01/report.json` | `48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747` |
| 同目录 `journal.jsonl` | `908265c892eb8142ef56b7ffd9bc531ac628a76427a094a87e36ba3b9546ee75` |
| `results/p1_witness_physical_20261027_v1_prepared01/prepare.json` | `e4fc7b7adeac3a23036196adaadfbbe69d8873f1e8649ebbf5fe9e8fb1eca0ff` |
| `results/p1_witness_physical_actual_audit_20261027_v1_audit01/report.json` | `05054e0d40a0f857ec004df02f25d735c21b5647ac271d345a3e50b1bb39f1c8` |
| 同审计目录 `acquisition.json` | `97bc72a35d7a7184efc09f2dd261c306f15e54a12fd90eef14a05955c47f2d64` |

新审计模块 `experiments/spectral_reachability/p1_witness_physical_actual_audit_20261027_v1.py`
先重验原physical prepare及direct/witness/source完整绑定，再独立用模运算计算角差，
不调用producer.pose_diagnostics作为审计真值。浮点receipt核对用2e-12绝对误差，
只用于表示/算术一致性，不修改success或研究门槛。

具体核对：

- 完整journal SHA、每行previous/event SHA、run起止、8个branch起止/close、16reset、
  80step、88render及24 checkpoint事件全分母；branch_closed的identity位于其receipt
  内，不能错误要求顶层selection_sha256。按稳定root/arm/selection join，逆序report
  rows仍通过，不以数组位置判断根。
- 原random seed、placement seed、request options和记录的original placement request
  一致；8个placement returned state/root与封存NPZ精确对应；24 PNG的8个initial帧
  精确等于原source RGB。此审查没有重新执行sample_state或physics。
- 80个实际dispatch动作等于prepare固定float32 rawcommands；journal attempt/return
  及successor observation/state/info均与trace相同；每个command前已有新guard.observe，
  顺序为fresh guard→attempt→return→下一fresh RGB。最终guard授权10条、观察11帧，
  无abstention，canonical pixels始终为正。
- 24 PNG重新校验文件SHA、PNG/RGB/224×224×3/uint8、解码像素SHA、state/step、
  canonical agent像素数；与相应fresh frame日志一致。其余64帧没有长期保存像素，
  只能核对哈希/guard日志与时序，不能假称独立重看了它们的RGB。
- 全80个state重新计算native原公式、wrapped/object/agent/allxy/state指标、风险flags；
  终态与第10个实际return一致。Internal reset physics仍是按封存env源码推得32步，
  不是新增native内部step instrumentation。

## 根级真实差异

根0完整身份 `02a139165f6e8465ec882926c2ceb13a33ca5547f736ed1e3817242902050d06`。
根1完整身份 `05844ca2c3ab8ba09846f9084edd71c8365ad74f7ab5961adaa5b0fded6289a7`。
下表均是实际第10步终态；误差越低越好，角度为rad。不同指标不擅自合成单一pose分数。

| 根 | 臂 | 物体位置误差 | wrapped物体角误差 | allxy误差 | native state距离 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | zero | 9.167074 | 3.017657 | 242.217669 | 242.239681 |
| 0 | GN | 9.167074 | 3.017657 | 170.862167 | 304.469437 |
| 0 | gradient | 9.167074 | 3.017657 | 176.102883 | 181.639371 |
| 0 | two_stage | 15.388567 | 2.970557 | 157.428615 | 257.203400 |
| 1 | zero | 13.691348 | 1.782513 | 319.276476 | 319.308196 |
| 1 | GN | 20.552924 | 1.804490 | 315.364254 | 320.324197 |
| 1 | gradient | 13.691348 | 1.782513 | 257.991421 | 266.409165 |
| 1 | two_stage | 13.691348 | 1.782513 | 281.289476 | 288.698952 |

根0：two_stage相对zero/GN/gradient的物体位置误差都增加6.221492633，角误差减少
0.047099262；这不是物体位置/角度的共同改善。其allxy优于其余三臂，主要因agent
更接近目标；native state包含末态agent velocity，故与allxy排序不同。
two_stage native state相对GN减少47.266037278，但相对gradient增加75.564028878，
相对zero增加14.963718961。不能只挑它胜GN的一项说物理机制成立。

根1：two_stage与zero/gradient的物体pose完全相同；GN物体位置误差增加6.861576637、
角误差增加0.021976211。因此two_stage比GN好的物体指标，是未发生GN的损害，并非
优于zero/gradient的物体进展。two_stage native state相对zero改善30.609244127、相对
GN改善31.625244482，但差于gradient22.289786907。

观察到的逐步object state：根0 two_stage在step1后相对初态改变，其余三臂10个保存
successor的object pose均与初态相同；根1 GN从step4起改变，其余三臂均未变。
这描述保存的successor，不声称内部每个physics substep。原official n_contacts字段：
根0 zero在1–10为正、two_stage在1/2为正；根1 GN在4为正，其余所述臂为零。
该字段没有body-pair身份，不能据此将全部接触归因agent↔object，更不能从终态
邻近/接触倒推整轨迹接触。物体不动和正接触日志也并不矛盾。

目前没有真实终点native latent，故不报告“预测Q对实际Q的排序反转”或latent成功。
上述全是state/pose consequence；下一步24帧编码才能检查表示/评分错配。

## 24帧native CUDA编码交接（MAIN编写/执行新driver）

已输出小型、无模型执行的 `acquisition.json`，完整枚举root/arm/step=0/5/10共24项，
含selection SHA、PNG文件/像素SHA、shape/dtype、原task文件路径/SHA、实际raw动作前缀。
原task固定goal、projection、normalizer、root history和模型SHA；其文件hash已由physical
prepare绑定。物理图像不复制、不重新render；不能借用旧endpoint。

建议MAIN新driver在CUDA前prepare固定这24项及manifest外部SHA，重新验证physical
report/journal/PNG/task哈希、原direct prepare中的checkpoint/config SHA和原image_transform
源码。加载原权重、eval+requires_grad_(False)、CUDA only；按原
`screen.native_image_encoder`的native model.encode与原ImageNet/Resize224 transform
逐图编码，保留192维原数组及输入像素SHA，记录encoder/projector实际entry attempts
和完成数；predict/action_encoder必须为0。后续可复用函数，但driver自身必须即时保存
每图attempt/部分数组、独立保护hookremove/finalhash异常，不让它们覆盖执行首错。

新预算应固定24encoder+24projector样本，包含重复initial帧，不去重或只选有利终点。
原goal latent来自原task，沿用时明确标记继承的8image acquisition成本，不冒充本次
重新编码或免费新goal；若要重编码goal，须在新prepare另行预声明并记账。冻结模型
前后hash应相同，不训练、不调动作、不按编码结果重选。新输出与旧physical seal分离。

## 本轮命令与验证类别

纯mock/PNG/算术测试及actual saved审计测试：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_witness_physical_actual_audit_20261027_v1.py
```

已执行的saved-only审计命令（审计目录已存在，不覆盖；复跑需NEW output）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_witness_physical_actual_audit_20261027_v1 --prepared results/p1_witness_physical_20261027_v1_prepared01 --prepare-sha e4fc7b7adeac3a23036196adaadfbbe69d8873f1e8649ebbf5fe9e8fb1eca0ff --report results/p1_witness_physical_20261027_v1_main01/report.json --report-sha 48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747 --output results/p1_witness_physical_actual_audit_20261027_v1_audit01
```

mock验证工程，actual saved验证保存证据与算术，科学准入仍false；没有因为审计通过
开放P1，也没有修改任何研究门槛、原方法或预算。
