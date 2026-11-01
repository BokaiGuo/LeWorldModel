# 固定开发根 witness 构造：前瞻性预注册

状态：仅约定未来探索的输入与时序；producer、完整运行前封存及物理执行尚未完成。
本预注册发生在 v4 历史及其重放已被观察之后，不是对这些历史或根的事前盲注册。
它不是原 moving-fresh 四新根协议的执行，不改变该协议及任何 gate。

## 固定来源和范围

固定 v4 来源的 slot 0、1，按原顺序、分母 2；不按物理、像素、模型或任何其他
结果选择、重试或补槽。它们是已用开发根，不是独立开发根。训练来源、完整历史
覆盖和 freshness 均 unknown，不能用换 goal 或关闭 target overlay 消除已有 lineage。
来源绑定：

- `le-wm-main/results/p1_exploration_seed_preseal_20261101_hypatia_request.json`：
  SHA256 `bcc5d39dcb85b0f7fba1c66f718ec00acf1909871568023ee809338a10814b53`。
- 对应 run01 的 `source_0.npz`、`source_1.npz` SHA256 分别为
  `1163eb2b5efa1897c5ed106351e5f8e65ada27041a3eacbb3c9d5f1232d92b19`、
  `356fe60e6499ee5a1957a3c01189826607ea5cf72afd630186ca71569cb7c0f1`。
- v4 的 `PREP01.json` SHA256
  `9cbab1d0eef2fcf9c067d332e89f12b77178f17c8f3b0c37cce30aa0e39c6dca`。
  使用其中固定 input 及 reset recipe；不得以像素一致冒充 v3 target-on 来源。

上述 hash 是已验证的既有身份，不是对未来代码或运行的封存。未来执行前还必须
以独占 prepare 绑定本文、producer、测试、所有传递依赖及运行环境、上述输入，
以及已有结果记录；校验 hash、容量与操作预算，并先完成含失败和中途写盘的测试。
任何输入漂移均停止，不覆盖或重写既有产物。

## 事先固定的构造

环境 `swm/PushT-v1`，原 relative、action_scale=100、dt=.01、control_hz=10，
RGB 224，`with_target=False`、`render_action=False`；保留原生 eval_state 和停止，
额外分栏严格环绕角，只作诊断。每个 phase 用新环境实例依 v4 同一 request 和
source metadata 做原 seed reset 和 seed=0 placement reset；核对实际初态与 source
state/goal，不能 restore 保存的物理状态或替换 slot。

每槽 `construction → replay0 → replay1`，每 phase 实际执行前五
float32 `[.1,0]`、后五 `[0,.1]`，保留每次实际动作 bits，保存每一步 state、
RGB 及含 agent/block position、velocity、angle、angular velocity、force、torque、
sleep 和 float64 bits 的 body snapshot。observed 0/5/10 是实际渲染。原生提前停止、
truncation 或异常时停止该槽，保存部分数据和缺失点，不补步或重试。

从 warmup 后实际 root 的 objectXY-agentXY 算单位 radial 与逆时针 tangent；
槽 0 sign=+1、槽 1 sign=-1。前五 logits 为 `.5*radial`，后五为
`.25*radial+sign*.5*tangent`；先转 float32 后 tanh，每 5 步包的 logit L2
上限 `.75*sqrt(5)`。这两个 sign 是此已用开发根探索方案的固定参数，不是原四新根。
首次 witness dispatch **之前** durable 写出全部 logits、动作 bits、root snapshot；
零范数、不满足 cap 或任何 guard 拒绝时原地失败，不重选。

construction 必须正常完成 10 witness 步，其实际 terminal state 和 target-off RGB
才是唯一 derived goal 和 goal RGB；禁止把早停终态、原 reset goal 或 H5 替代。
replay0、replay1 分别真实重建 warmup，逐项比较 0/5/10 state、RGB、动作 bits
和 body snapshot；通过后按原合同离线 bind derived goal/goal RGB，核验 bind 不
改变 body snapshot，不改 goal_pose，不多做物理步或 render。随后重算并核对相同
witness，逐步比较 construction 的 state、RGB 和 snapshot。native success 可提前
停止，保留真实步数和缺失 checkpoints；truncation 或非成功停止不能称重放可达。
若 root 已 native success，标记 `degenerate_goal`，不改阈值、不重造 goal。
每步分开保存 native 角结果与 strict wrapped 诊断；后者不改变 native gate。
未导出的 solver cache 一律 unknown。

## 资源、失败和发布边界

每槽最大 3 phase、6 reset、30 warmup public step、30 witness public step、
63 explicit render；两槽上限分别 6、12、60、60、126。按原合同记录实际 reset、
state setter、goal setter/bind、internal physics、render、CPU 时间、磁盘和所有
attempt/completion，提前持久化 journal 和每步数据；原生早停与工程失败分开。
总预算和失败写盘测试须在新 producer prepare 中可验证地固定，不以这里的
上限代替实现。最终必须留两行，含未开始、首错、次错、phase、实际次数及缺失点。

仅当 construction 和两次重放通过，才可制作 evaluator 隔离格式样本：
`rgb_history:uint8[3,224,224,3]`（replay0），
`raw_action_history:float32[10,2]`（replay0），
`goal_rgb:uint8[224,224,3]`（construction）。不得含 privileged state、seed、
witness、未来轨迹或来源路径；不得发布给 planner、生成获批 manifest 或声称
moving-fresh 正式 source。模型、encoder、训练、planner 调用均为 0。

原四新根合同仍要求其固定 seed/recipe、全部历史排除和 MAIN 授权；本文件不提供
这些条件。未来独立根或训练独立性、P1 及 planner 授权均不得从本探索推断。
