# Saved-only 强基线汇总 v2

本版本是 v1 的独占修正版，不修改 v1、旧结果或封存报告。它只读取已保存的 kinodynamic、no-composition、sampling 报告，作者值与本地值严格分栏；不加载模型或模拟器，不运行神经、物理、网络、训练或拟合。

## 修正点

比较结果使用 JSON-safe 字符串键：`kinodynamic__vs__no_composition`、`kinodynamic__vs__sampling`、`no_composition__vs__sampling`。v1 的 tuple key 会在 mutation test 进入校验前触发 JSON 序列化错误；v2 保留相同输入 SHA、分母、协议指纹和 fail-closed 规则。

三组协议不一致，因此每一项均为 `protocol_mismatch_no_direct_comparison`，`delta` 为 `null`。不得把不同 horizon、search domain、资源计量、physical/model-only 结果或作者值拼成共同差值；不宣称 P1/P2 admission 或科学收益。

## 绑定资产与命令

v2 复用三份已封存输入：kinodynamic report SHA `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`、no-composition consume SHA `a05aa97f6c2688d7ae7a1fd8b961a23ccba1f4e45b14d95159cca63c5ba23167`、sampling report SHA `c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6`。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_strong_baseline_saved_summary_20261028_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_strong_baseline_saved_summary_20261028_v2 build --output results/agent_p2_strong_baseline_saved_summary_20261028_v2_MAIN01.json
P2_SUMMARY_SHA=$(sha256sum results/agent_p2_strong_baseline_saved_summary_20261028_v2_MAIN01.json | cut -d' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_strong_baseline_saved_summary_20261028_v2 verify --summary results/agent_p2_strong_baseline_saved_summary_20261028_v2_MAIN01.json --sha "$P2_SUMMARY_SHA"
```

作者原始报告未绑定时保持 `author.status=unavailable`；测试通过不是科学证据。实际 saved 输入存在、协议可比性和 P1/fresh-root 前置仍由 MAIN 审核。
