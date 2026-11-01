# H15 GN 两根物理结果

固定两根和 HOST 最终动作执行，prepare SHA `feb2078a5070d043e52b1ac58ffd8c77c86e3e4677b89fccc4d0e433b974e9a3`；HOST report SHA `2d41a16c0ad568c79c9b627fffe6e9cd608b0a38a2c26503542b3b935b7a1ceb`；物理 report SHA `e158ae8de24543dfd7200f37676dc7121803a2de8fdce7efa7c40295ca93dbc0`。独立进程 saved consumer 成功；逐根15步均完成，无 native success。50 public steps（含每根 warm10）、508 physics steps、162 renders；未调用 GPU 或新模型，无训练。

| root | native distance | object position error | wrapped angle error (rad) | native success |
|---|---:|---:|---:|---|
| 0 | 203.958 | 26.728 | 0.155198 | false |
| 1 | 513.902 | 143.868 | 0.482462 | false |

物理是原已知根的另次执行，不是 fresh roots 或实时配对；两根不能支持泛化结论，也不等于科学准入。saved consumer 复核保存轨迹和动作/任务/账本/PNG，但不是独立物理重演，不能证明未记录的外部行为。GN HOST 每根 117 predictor samples、768 VJP；不可称与 CEM 或 H15 等总算力。

完整复核命令：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_physical_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 feb2078a5070d043e52b1ac58ffd8c77c86e3e4677b89fccc4d0e433b974e9a3 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_physics_CODEX01 --report-sha256 e158ae8de24543dfd7200f37676dc7121803a2de8fdce7efa7c40295ca93dbc0
```
