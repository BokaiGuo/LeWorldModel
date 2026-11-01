# P5 本地 RGB 因果视觉模型与固定物理面板输入准备

## 候选和真实 CPU 加载

候选为项目内 `results/causal_visual_pusht_phaseutility_formaltrain30_20261009_seed20261009/causal_visual_world_model.pt`，SHA256 `ca9b51ed2b98923bebc59941ba24612bc5ab8f59d7afdf6239334b550350d0dd`。这是先前本地训练的 `VisualCausalWorldModel`，不是发布的 DINO-WM，也不是 LeWM 或 JEPA；其 mask-moments 几何前端只看 RGB。原训练报告明确使用物理状态作为**训练监督标签和 evaluator**，运行策略使用图像；必须披露既有额外监督成本，不能称无监督预训练或无训练成本。本轮没有训练。

严格限定 `CUDA_VISIBLE_DEVICES=''`，对这份 hash 匹配的 checkpoint 使用 `torch.load(..., weights_only=True, map_location='cpu')`，按原源码和原 loader 的旧 checkpoint 推断规则实例化 phase/legacy 架构，`load_state_dict(strict=True)`：26 个 state_dict 项、93,842 参数、缺失／多余键均空，CPU eval 且参数 `requires_grad=false`。保存 checkpoint 原有六维状态 mean/std，未重新拟合。这里的加载是实际本轮操作；并未做神经 forward 或测试预测能力。原本地代码与原训练真实性由文件绑定和原报告支持，不等同官方远端权重或独立来源认证。

## 固定可运行输入准备

输入是已开放原 physical receipt 与其既有 fixed164 panel binding，不读任何 reserved root。每枝固定：

| 用途 | 已保存 physical 时间 | 输入 |
|---|---|---|
| previous | plan4/global14 | 本枝原始 RGB |
| current | plan5/global15 | 本枝原始 RGB，与 JEPA H1 第三帧文件 SHA 相同 |
| action | plan 第 6 条命令，驱动 plan5→plan6 | 原 float32 2D raw，不做 JEPA packet 归一化 |
| label | plan6/global16 | 只记录原 receipt 中的文件及 pixel SHA，不打开或解码 |

读取 history PNG 时核 224×224 RGB mode、文件和像素 SHA，按该模型源码的 OpenCV `INTER_AREA` 缩放到 64×64、HWC→CHW uint8、float32 除 255；保存 uint8 和 float32 array bytes SHA。`materialize_inputs(row)` 可重新检查并返回模型接口用的 `[1,3,64,64]` previous/current 及 `[1,2]` action，只返回上述三个字段，不返回 goal、物理状态或标签。它不调用模型，metadata hash 不代替未来调用时的 runtime hook。

严格校验 164 条身份及 80 对 schedule、四 baseline、图像 identity 和 raw10 SHA；缺图、漏帧、overlay/动作/身份/归一化不一致直接拒绝，不缩分母。全部 80 对必须保留：实际保存输入中 16 对（坐标0/1）单步 action 不同、64 对 action 相同；全部 80 对的 previous/current 预处理 RGB 在 pair 内一致。这只说明固定单步接口的干预覆盖，不能把 64 对的相同输入包装成有效动作干预。

此桥是**一个动作的 global15→16 预测准备**，不是 JEPA H1 的五动作 global15→20；目标、goal权限、输入历史和实际计算预算也尚未配齐，因此不能称跨模型匹配预测或控制评价。训练数据与本 panel 分布、同构包装的动作单位、RGB颜色阈值对 on/off 渲染的适用性尚需独立验证；不得凭尺寸相同宣称分布一致。两个已知根不是 fresh confirmation，P1/P5 均 false。

## 实际交付与验证

源码 `experiments/spectral_reachability/p5_rgb_causal_panel_bridge_20261030_codex_v1.py`；保存结果 `results/p5_rgb_causal_panel_bridge_20261030_codex_v1_MAIN01.json`，SHA256 `559c7c12667e39a320e27151e08c9022ba719f19444b78b08e8a392aed1ba324`。真实固定 164/80 输入准备、CPU strict load 均完成；14 项测试通过（包括身份、动作、缺帧、overlay、输入篡改、未来图未读取、CPU 隔离），同一实现对完整输入重新计算 verify 一致，不是独立实现或上游 fullstrict。无新 GPU、模型前向、物理或训练。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_rgb_causal_panel_bridge_20261030_codex_v1.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_rgb_causal_panel_bridge_20261030_codex_v1 verify --artifact results/p5_rgb_causal_panel_bridge_20261030_codex_v1_MAIN01.json --sha256 559c7c12667e39a320e27151e08c9022ba719f19444b78b08e8a392aed1ba324
```

下一步须先明确跨模型共同时间目标与信息权限、该候选动作/渲染支持、两模型各自可实证运行的预算和评估指标，再封存真实运行协议；若希望以五步结果比较，必须另外设计、授权并检验连续滚动，不得将单步接口改名为五步或隐含观察未来帧。
