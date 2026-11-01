# P5 既有物理面板跨模型输入身份绑定

本版本以已开放的 fixed164 面板为范围，新增离线身份绑定器；不改变旧 bridge、physical receipt 或 sealed 资产，不扫描或打开保留根。已有 `agent_p5_raw164_jepa_input_bridge_20261029_v1` 已实现 JEPA native 输入适配；已有 fixed164 join 已绑定 LeWM actual-Q、JEPA 既有结果与 physical receipt。本增量检查 **bridge 请求与共同物理面板** 的逐枝内容，而不是再次实现 loader 或结果 join。

## 固定输入及权限

仅直接读取以下三个固定文件，不自动跟随其大量 bindings 路径：

- JEPA bridge prepare SHA256 `1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f`。
- 物理完整 receipt SHA256 `8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb`。
- 既有 fixed164 模型／物理 join SHA256 `25cb6846263027b1ceedd39b9060c3158c6cc6a095d323b012be30728e7511c3`。

另只访问这份物理 receipt 中每枝固定的 warmup5、plan0、plan5、plan10 四个 PNG，且路径必须在固定 physical 目录内并与原 basename 精确匹配。核 PNG 文件 SHA，不解码像素：pixel SHA、上游物理执行时序与完整 strict 仍继承 receipt。终帧仅作 successor 文件身份，不读取像素或输入模型。

JEPA 请求是 RGB 加三时刻的 agent 位置和速度；LeWM 使用 RGB，既有 physical plan 保存 LeWM model task binding。两者并非相同观察权限或相同输入张量；本文只绑定共同物理来源与动作，不认证公平同预算比较。也不证明 LeWM 模型预测与 JEPA 在相同推理时间窗下已按新合同运行。

## 离线合同

- 固定 164 枝、四组各 41 枝、80 对及 baseline `[0,41,82,123]`；顺序、root slot、condition、task、selection、所有正负坐标和两 h 必须一致，缺失或置换拒绝，不缩分母。
- 每枝 bridge 的后 10 条 raw action 按 little-endian float32 与 physical plan、existing join 哈希一致；前 5 条固定原 warmup ABI。3 个 5-step token 重算旧 JEPA float32 归一化，逐字节匹配已保存 packets。
- history 为 global 5/10/15，即 warmup5/plan0/plan5；successor 为 global20，即 plan10。检查 condition、overlay、路径、文件 SHA、receipt pixel SHA 和 frame 唯一性。分别将每个 history 的 snapshot state、agent position/velocity 与 bridge raw proprio 做 float64 字节检查，并重算 float32 proprio 归一化。
- 结果保存每枝完整 RGB 文件身份、原始与归一化动作／本体的 hash，以及 LeWM task binding；不保存 successor 像素或做任何目标评分。已知两根并非独立 fresh cohort。
- 输入文件的整个 SHA 在计算前后检查，结果排他创建；如遇源漂移、缺图、非有限值、时槽/overlay/动作/normalizer/身份错误直接拒绝。SHA 确认本轮读取与已知文件相同，不证明过去模型实际加载字节或认证 external timestamp。

## 实际执行与复验

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_crossmodel_panel_binding_20261030_codex_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_crossmodel_panel_binding_20261030_codex_v1 verify --artifact results/p5_crossmodel_panel_binding_20261030_codex_v1_MAIN01.json --sha256 80ed10d3b8f623d361cc25e2764d390bc9cc758178c89e7467e99da3d785ef98
```

真实资产绑定输出 `results/p5_crossmodel_panel_binding_20261030_codex_v1_MAIN01.json`，文件 SHA256 `80ed10d3b8f623d361cc25e2764d390bc9cc758178c89e7467e99da3d785ef98`，164/80；15 项测试通过，完整 verify 由同一实现重建一致，不是另一实现或 fullstrict。仅 NumPy/CPU 数组与文件核验，0 新神经、物理、GPU 和训练调用；P1/P5 均 false。

下一步若要构建真正的可比较跨模型协议，需事前明确两模型共同允许的信息、模型自身 preprocessing 和观测/动作时间 ABI、指标及预算；本资产中 JEPA 本体信息特权必须作为条件披露，不能由共同 RGB hash 抹去。需要真实模型推理或新根时另行授权，并先解决独立根覆盖审批与采集合同；本次没有产生上述证据。
