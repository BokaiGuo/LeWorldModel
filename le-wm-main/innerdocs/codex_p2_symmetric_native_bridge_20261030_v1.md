# P2 对称内核：四根来源及原生后端接缝

新增 `codex_p2_symmetric_native_bridge_20261030_v1.py` 和同名测试，不修改已有内核、旧结果或其他线程资产。`known_roots()` 重建既有四根 prepare 并核查 source closure，按 root SHA 精确 join 原 compilation；比较实际编译 `emb` 和根 latent history、编译 action history 末两帧与根 between-state history，并核对每根 goal 和投影的原 SHA、动作基础域。测试使用实际已封存四根，未加载模型。对 goal 篡改和非法 L2 logits 均拒绝。

`NativeCudaBackend` 使用原 `make_transition`、原 action normalization、CUDA float32 模型、3 packet recurrence，以及真实 `torch.autograd.functional.jacobian(...,vectorize=True)`；投影乘到终态和 Jacobian，可供调用方以投影后的 goal 使用同一均方形式。同步时钟调用 CUDA synchronize，packet/J 前后通过注入 sink 发事件。此类尚未执行 CUDA，且没有自身的预测模块 hooks 或计量硬上限；sink 的持久性也依赖未来调用方，不能把它当实际 GPU 资源认证或生产入口。

H3 是新的探索而非原四根 H2 实验：旧四根投影 rank 4，旧内核目标为未投影的 192 维均方，两者不能混同。本桥的投影适配是明确的探索方案，未被前一内核合同事前固定；不得称预注册。原内核 `clip(-3,3)` 不保障任何联合 L2 ball；桥提供 `validate_logits` 拒绝越界，但还没有解决内核如何保证每个候选合法。即使使用历史 H3 的 `.75*sqrt(15)`，也不能把它假称原 H2 约束或通过事后更改科学门。

尚缺生产所需的独占 prepare/run 及耐久 journal、模型和实际加载模块的完整身份固定与前后校验、模块真实 predispatch 硬计量、SVD/内核 dispatch 耐久事件与预算门、完整三段 replay 的 context 等值验证和 saved consumer、H3 任务定义及全加载成本。缺任一项均不运行；已封存旧实验不覆盖。FLOPs、engine 内部 backward 和加载全成本保持 unknown；不宣称等总算力、P1/P2 准入或物理成功。

CPU 复验（无模型、GPU或物理）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p2_symmetric_budget_20261030_v1.py tests/test_codex_p2_symmetric_native_bridge_20261030_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p2_symmetric_native_bridge_20261030_v1 preflight
```

测试 6 passed；不提供生产命令，因为真实执行入口尚未完成。本轮仅查询 GPU 状态，未启动 CUDA 模型、训练或物理。
