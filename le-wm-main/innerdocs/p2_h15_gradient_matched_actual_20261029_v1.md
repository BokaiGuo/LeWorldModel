# H15 projected-gradient：主机与固定两根物理结果

新增 CPU 数值/边界/saved-consumer 负控 12 项通过；封存 HOST prepare CODEX02 复核通过。首次 MAIN01 在 sandbox 中因 `/dev/nvidia*` 不可见而 failed_partial，两个根均 not_started，没有模型或候选；该目录保留。获准主机执行后，同一封存协议在独占 MAIN02 完成两根，HOST report SHA `c544804ab9579fea0ef72fccac61c36ed7cc9450f74c551679f8d34b7b059ee1`，独立 saved consumer 通过；每根 117 predictor samples、768 VJP，未科学准入。没有更改门槛或降 CPU。

因物理原 adapter 固定指向 MAIN01，新建 v2 只绑定 MAIN02，7 项 CPU fixture 测试通过；未改变原封存文件。物理 prepare SHA `cb35597351babcbdfce7ab299e2dc62bea09c9b2429e5ccb3a7eb3a47834deda`，物理 report SHA `f9d81f99760368579fca0d252c1ea40e192aa2af052900612fae45c38145e463`，独立 saved consumer 通过。两根均完成15控制步，native success 均 false；50 public steps（含 warm10）、508 physics steps、162 renders。

| root | native distance | object position error | wrapped angle error radians | native success |
|---|---:|---:|---:|---|
| 0 | 136.335 | 43.677 | 0.267256 | false |
| 1 | 173.257 | 59.445 | 0.059996 | false |

已知根、另次物理执行，不构成 fresh-root 泛化或与其他方法等总算力；saved consumer 审计保存证据并非独立物理重演，不能证明未记录行为。未训练模型。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gradient_matched_physical_20261029_v2 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_physical_20261029_v2_prepare_CODEX01.json --sha256 cb35597351babcbdfce7ab299e2dc62bea09c9b2429e5ccb3a7eb3a47834deda --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_physical_20261029_v2_physics_CODEX01 --report-sha256 f9d81f99760368579fca0d252c1ea40e192aa2af052900612fae45c38145e463
```
