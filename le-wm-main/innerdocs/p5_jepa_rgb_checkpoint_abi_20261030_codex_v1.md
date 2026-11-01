# P5 PushT JEPA checkpoint：纯 RGB 接口离线核查

结论：在本次绑定的原生 loader、checkpoint schema 和既有输入桥中，**没有已验证可直接使用的纯 RGB 同权限预测接口**。这是固定资产集合与源码静态分析的结论，不是对其他 checkpoint、未来适配器或所有可能实现的穷尽证明。没有模型反序列化、神经前向、GPU、物理、训练或保留根访问。

## 证据与实际 ABI

- checkpoint `data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar` 的实际文件 SHA 为 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`；它与原 frozen manifest 的 seal 匹配。本轮仅计算文件 hash，不解析权重。
- 原 `p5_jepa_native_action_response_20261027_v8.py::load_cuda` 明确构造 `predictor_embed_dim=384`、`proprio_emb_dim=16`、`proprio_encoding=feature`、`proprio_encoder_inpred=False`，没有传 `use_proprio`；固定 `AdaLN_vit.py` 的构造默认值为 `True`。模型输入的 proprio feature 是单独 encoder 的输出，而不是视觉 feature。
- 原 frozen manifest 保存的 predictor schema 包括 `action_encoder.weight=[400,10]`、`predictor_blocks.0.norm1.weight=[400]`、`predictor_blocks.0.adaLN_modulation.1.weight=[2400,400]`；proprio encoder 的 `patch_embed.weight=[16,4,1]`。这些是先前记录的 schema，本轮未执行 strict load。
- 既有 H1 完成 receipt SHA `4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759`，自身记录 `RGB_only=false`、164 requests、80 pairs；本轮继承该回执，不重跑其 full strict。

既有 H1 predictor 保存接口：visual float32 `[1,3,1,16,16,384]`，action float32 `[1,3,10]`，proprio feature float32 `[1,3,256,16]`；输出 visual `[1,3,256,384]`、None、proprio feature `[1,3,256,16]`。JEPA 的历史 agent position/velocity 构成本体输入；动作是另一个控制通道。三帧 global 5/10/15 和 successor20 的物理绑定见上一合同，本次没有重复打开图像。

## 为什么签名可选不等于可用纯 RGB

`forward(x, actions, proprio=None)` 的 Python 签名允许传 None，但固定配置下其无 proprio 分支仅 flatten 384 维视觉特征；第一层 block 的 `norm1` 由总宽度 `384+16=400` 构造，随后的 action 条件也是 400 维。因此可由源码和 schema 静态判断这一组合维度不匹配；本轮**没有实际 forward 验证报错**。

将配置改为 `use_proprio=False` 会把总宽度变为 384，连带 action encoder、block 和 normalization 参数形状改变，不能严格加载固定的 400 宽权重；本轮未构造新模型或实际尝试加载。另行提供固定 16 维常量可能构成一个不同的输入干预或填充方案，但目前既无真实运行回执也无训练支持或有效性证明；不把它冒充原生 RGB-only JEPA、原生结果或经验证同权限对照，更不伪造训练。

## 可复验交付

新增源码 `experiments/spectral_reachability/p5_jepa_rgb_checkpoint_abi_20261030_codex_v1.py`：固定并双次 hash loader、模型源码、manifest、checkpoint、H1 receipt 和权限合同；对 constructor 做 AST 字面量检查，并核已保存权重 schema 和 H1 标注。`check_descriptor` 对 native H1 三输入的字段、shape、dtype fail closed；它只检查描述符，不是 live tensor hook 或 GPU 测试。

实际保存结果 `results/p5_jepa_rgb_checkpoint_abi_20261030_codex_v1_MAIN01.json`，SHA256 `e50df3f1f8c139dea00405adaff7f03df6b7d5b3d2d00101298dd6271866ee13`。新测试 15 项，与权限合同联合 **37 passed**；同一实现的重建 verify 一致，不是独立验证或原 full strict。P1/P5 admission=false。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_jepa_rgb_checkpoint_abi_20261030_codex_v1.py tests/test_p5_observation_permission_contract_20261030_codex_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_rgb_checkpoint_abi_20261030_codex_v1 verify --artifact results/p5_jepa_rgb_checkpoint_abi_20261030_codex_v1_MAIN01.json --sha256 e50df3f1f8c139dea00405adaff7f03df6b7d5b3d2d00101298dd6271866ee13
```

后续若要同权限比较，需先选定并封存确有证据的合法接口／模型：例如 separately specified、明确披露非原生及分布外风险的固定输入干预，或具有合法来源和验证记录的 RGB-only checkpoint；随后单独核实际加载、运行时输入、时间和 goal 权限、固定分母及资源预算。前者是否可用尚未知，后者目前未在本固定证据集合中提供；两者均需另行授权才能运行，不自动打开 P5。
