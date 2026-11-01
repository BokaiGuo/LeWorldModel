# P1 独立来源线：开发根重放与盲输入接线

本接线只处理已打开的 development root，不运行 confirmation slot，不授予
P1 admission。未知训练来源、未命中已知 seed 和新环境实例均不能被解释为
独立性证明。

## 接线范围

`p1_source_replay_blind_20261101_exclusive.py` 复用已封存的 future-root queue、
queue-bound reset/RGB history producer 和保存验收器，增加以下三层：

1. 在新的原生 PushT 实例中按同一 development seed 重放 reset 与十步实际
   zero action，逐字节核对 goal RGB、0/5/10 RGB、state 和刚体 snapshot。
2. 仅在精确重放通过后导出只含 `rgb_history`、`raw_action_history` 和
   `goal_rgb` 的 NPZ；seed、物理 state、snapshot 和 outcome 不进入盲载荷。
3. 将盲输入 manifest 与固定历史 inventory、checkpoint 来源 authority
   snapshot 联合审计。工程链可 ready，但训练 lineage、历史穷尽和外部保管
   未闭合时，确认授权固定为 false。

重放是同一已知开发根的确定性复验，不是第二个独立样本。当前历史 inventory
仅覆盖四组固定 sourcepack/journal 的 44 条 reset 和 overlay 的 12 条记录；
训练来源仍缺 checkpoint-run、数据版本与 split、转换及 normalizer 绑定。

## 禁止提升

- confirmation slot 在生产入口和测试中均 fail-closed。
- `engineering_ready=true` 只表示来源、重放、盲输入接口连通。
- `development_seed_is_not_independence=true` 和
  `unknown_training_source_is_not_independence=true` 是硬边界。
- 未取得历史 authority、训练 lineage、独立 origin review 和外部 custody
  前，不得启动正式确认或打开 P1 gate。
