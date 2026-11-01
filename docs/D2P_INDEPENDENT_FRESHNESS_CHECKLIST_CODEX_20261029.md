# D2P 新根 freshness：只读排重核查清单

本清单不选择根、不采样、不读取 planner outcome、不改封存产物；测试中的状态是合成例子，不是新根。适用范围是现有 PushT moving 协议，不替其他环境或未封存的新协议签发认证。

## 现有证据与缺口

- 2026-10-28 封存 P4 审计报告 SHA `15e3ece87e36eba469a0cc2188efbb7edaa94fa00f1562f9e6adc2052e4e4ae0`：无 certified roots，历史覆盖不完整，原执行边界认证数为 0，source run 未获授权。报告的 `certificate_count=2` 是两个 construction-only 候选，不是两个 freshness 证书。该报告是当时的固定快照，不证明此后没有外部新证据。
- 原来源请求文档列 8 来源、16 引用、10 次已定位调用；B 有三次调用。已恢复的 bytes/命令和过程状态不等于原加载闭包或全部 reset 尝试。需要每次穷尽 ledger，或原运行绑定的加载闭包与可验证有限包络，另加历史范围和重启完整性来源审查。
- 历史身份恢复保留 52 个 selected 记录；它们不是所有尝试，也不是 52 个独立物理根。缺失的失败/未选/重启和真实物理身份不能由 seed、现有哈希或当前文件推断。
- 两个 construction NPZ 和 reused calibration sourcepack 不构成新根或历史认证；未获外部范围批准时 global freshness 保持 unknown。

## 执行前后逐项记录（不按 outcome 挑根）

1. 绑定事前冻结协议、批准的 cutoff/范围、来源证据路径和 SHA；确认覆盖已用、失败、未选、reserved、派生及所有重启。未知身份或原 reset 完整性缺口须逐条保留，不能自签 approval。
2. 固定四个 slot：root seed `203027101..104`、reset seed `203027201..204`，每个只有一次 proposal；检查全部已用 seed 和半开 reserved span。若冲突则协议阻断，不能换 seed 或补样。
3. 每个 slot 保留全部尝试、失败和拒绝，记录 phase、reset 和实际 post-warmup state7、snapshot、RGB3、raw10 的来源及 SHA。对历史和批内均检查：state7 字节、物理 pose、snapshot、history；换 goal、速度、signed zero 或角度加 2π 均不能使同一物理 pose 变新根。
4. 按冻结规则，前四维逐维差不超过 `1e-9` 且 wrapped angle 差不超过 `1e-12` 即排除；相同物理 pose key 或完整 snapshot/history 碰撞也需记录并按协议排除。身份字段缺失为 unknown，而非不重复。碰撞保留原 slot 和分母，不替换。
5. 记录检查发生在 witness dispatch 前；construction 得到 goal 后额外检查 root-goal hash。禁止看 planner cost、success、future outcome 或难度再改变保留规则。审查前不得宣布 global freshness、独立确认或 P1 准入。

## 本地验证

在 `le-wm-main` 运行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_d2p_independent_freshness_checklist_20261029.py
```

测试固定协议、coverage、blocked 报告字节，并检验现有 pose 实现对 2π、signed zero、速度、容差及非法 state 的处理；不构造清单资产、不核实完整历史、不授权采集或证明科学效果。报告是静态快照，若未来获得新证据应另做有界审查，不改旧报告。
