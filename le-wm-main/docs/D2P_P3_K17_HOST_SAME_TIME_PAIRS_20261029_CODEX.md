# P3 K17 HOST 六槽：同时间描述性配对

此记录只消费已保存数据，不执行 GPU、模型或物理。输入为 `results/p3_k17_host_bundle_20261029_MAIN01` 和 `results/p3_k17_host_bundle_20261029_consume_MAIN01.json`，外部 completion SHA256 为 `41ed9feb5fdfaaa1edbb9edf6b2dc1f28df130826b5129dba82c3173bceb2818`，receipt SHA256 为 `20caa3ba066ca86299c59192505aba5a13dce7a6037af207df87c8281e01a155`。结果文件 `results/p3_k17_host_same_time_pairs_20261029_CODEX01.json` SHA256 `0a8fa2c7a822aa702f5a5ed1bbbe6d168dfd24deddc33feaa251da39ba09debb`。

分析程序核对上述外部锚、completion→manifest→全部列举文件、journal 链、两个上游消费审计的 journal hash、原始动作请求与实际位串、返回状态与保存帧，以及逐槽资源总数。上游 receipt 是既有独立消费结果；本分析不重新认证来源。上游明确 `native_provenance_verified=false`、`scientific_pass=false`。结果是工程记录，不是科学通过或生产授权。

## 定义和真实结果

两个 root 各三槽：0 对应 3 fixedH1、4 staticH3_progress、5 adaptive；1 对应 9、10、11。仅同 root 且同 plan step 配对；六对均逐一核对 plan 0 的完整 snapshot、绑定目标向量、目标 pose 和目标图像 hash 相等。步 1–5 的动作来自 journal 实际执行的 `<f4` 位串；结果文件保存每步双轴动作、七维状态、返回 reward 及 B−A 差值。没有为 step 0 捏造 reward。动作前缀指从 step 1 连续逐位相同的步数。

| root | A | B | 相同动作前缀 | 第 5 步 reward B−A |
|---|---|---|---:|---:|
| 0 | fixedH1 | staticH3_progress | 0 | -20.6100 |
| 0 | fixedH1 | adaptive | 0 | -53.7953 |
| 0 | staticH3_progress | adaptive | 0 | -33.1853 |
| 1 | fixedH1 | staticH3_progress | 0 | +332.2450 |
| 1 | fixedH1 | adaptive | 0 | +332.2450 |
| 1 | staticH3_progress | adaptive | 5 | 0 |

第 5 步实际 reward 分别为 root 0：-209.4809、-230.0909、-263.2762；root 1：-638.9133、-306.6683、-306.6683，顺序均为 fixedH1、staticH3、adaptive。六槽第 5 步均未 terminated 或 truncated。root 1 后两槽五步动作及每步 reward 相同；root 0 的三对首步动作已经不同。时间过程并非单调：例如 root 0 staticH3−fixedH1 步 1 为 +93.5536、步 5 为 -20.6100。不能将某个终点差异解释成跨根稳定排名。

环境源码中原生 reward 是负的七维状态欧氏距离；termination 的位置和角度判据另行计算。故 reward 不仅反映方块位置，而且包含 agent 位置、方块角度和 agent 速度；非终止也不等同于任务质量的完整结论。此处报告的 reward 是实际日志返回值，不重算或替代原生任务。

## 资源和环境边界

每槽返回的接口资源向量相同：constructor 1、close 1、reset 2、set_state 4、goal_render 2、goal_setter 3、goal_bind 1、public_step 15、physics_step 154、explicit_render 51；包括 10 步 warmup 和 5 步 plan。全局数分别乘六，尝试和完成数一致、无失败。记录中的六个环境证据均为 `swm/PushT-v1`、`stable_worldmodel.envs.pusht.env.PushT`、`rgb_array`、224、render_action false。整个 outer 墙钟 8.568656 秒，不是单方法或单槽耗时。规划/神经计算及其分方法墙钟未认证，因此接口计数相同**不意味着总体等预算**。

只有两个既有 root，未证明独立泛化、因果优势或统计显著性；真实来源认证仍未闭合。初始状态与目标一致只在上述每个 root 内成立。结果文件保留全部逐步差异和原始值供复核；缺失数据不填补。测试使用真实已保存 bundle 的只读重算，以及内存篡改和临时 journal fixture；后者不是真实新采集或科学证据。

## 复核

在 `le-wm-main` 下执行：

```sh
python experiments/spectral_reachability/p3_k17_host_pairs_codex_20261029.py results/p3_k17_host_bundle_20261029_MAIN01 results/p3_k17_host_bundle_20261029_consume_MAIN01.json results/p3_k17_host_same_time_pairs_20261029_CODEX01.json --verify
pytest -q -p no:cacheprovider tests/test_p3_k17_host_pairs_codex_20261029.py
```

`--verify` 从两个外部锚重新计算并逐字比对结果，而不是只核对输出自身散列。任何锚或必要步骤不符会报错；独占写入模式不覆盖既有结果。
