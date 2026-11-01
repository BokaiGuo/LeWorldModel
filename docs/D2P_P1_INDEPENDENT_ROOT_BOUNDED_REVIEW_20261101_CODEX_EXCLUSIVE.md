# P1 独立根与历史排重：有界复核

本次只检查既存来源文档、固定身份投影、候选提案和排重代码；没有打开新
outcome、采样候选、执行模型、GPU、环境 reset 或物理实验，也没有修改 gate。
本文件是独占复核记录，不是来源授权或新鲜性认证。

## 当前依据

- `results/chat_freshroots_20261027_root_manifest.json` 的 SHA256 为
  `3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838`。
  四个槽是提案，实际根身份均为 null，source run、coverage 和 freshness
  均未授权或认证。它不是实际确认根池。
- `docs/D2P_P1_SOURCE_CANDIDATE_DISCOVERY_20261101_HYPATIA.md` 的有界
  检索也未找到具备来源授权、实际身份和固定顺序的确认候选；该检索不证明
  其他位置不存在补证。
- 固定历史投影 `results/p1_known_history_exclusion_20261101_hypatia_LOCAL01.json`
  SHA256 `c89b869189c45c63bb86b2afa09f5ba67f079d0e4ebed84396fc0acd70345136`；
  已知开发根投影 `results/p1_coherent_development_identity_20261101_hypatia_LOCAL01.json`
  SHA256 `e084cd3d1ed3e0fd8e216ddc776b238ce8e5b829a165d55d153c789ba69a23f7`。
  本次复算两者字节 SHA，不重新读取开发根 witness/target 原结果。
- 既存历史复核记录 80 条 prefix、3160 对比较、360 对碰撞和八个各十条的
  clique；prefix 不是独立采样根，八组不能解释为八个独立根。这些数字引用
  `docs/D2P_HISTORICAL_POSE_COLLISION_INDEPENDENT_20261030.md`，不是本次重算。

## 工具与验证边界

既有 `experiments/spectral_reachability/p1_explicit_collision_check_20261101_hypatia.py`
已实现有界 seed、保留区间、state7、容差 pose 和批内碰撞检查，拒绝额外
outcome 字段，缺失身份维持 unknown，输出不认证 freshness 或 P1 admission。
它不认证候选输入与实际来源的映射，也不覆盖全部历史；无碰撞不能证明独立。
其完整 `check` / `verify` 会通过 coherent 验证器重新读取既有 witness 和
target 结果，因此本轮没有执行此路径。

使用固定身份投影和纯比较测试，排除会调用完整生成/验收的测试：

```
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_explicit_collision_check_20261101_hypatia.py -k 'not create_verify'
```

在 `le-wm-main` 下结果为 10 passed、1 deselected。该测试包含合成候选和
已知开发根回放，不是对实际确认候选的排重。未新造另一套同质工具：目前没有
经来源授权的实际确认根及可校验的来源映射，工具不能凭空把自述变成权威。

## 仍需补齐

1. 来源负责人提供并独立审验固定候选池的授权范围、顺序、不可变 manifest
   SHA、实际每槽 root/reset 身份和 source-to-state/episode 原始映射；保留
   所有尝试及失败，不依结果筛选。若字段缺失则保持 unknown。
2. 获取十次已知 dispatch 的穷尽 reset ledger，或可信实际加载闭包及可证明
   穷尽的尝试范围，并覆盖失败、重启、其他机器/工作树和 cutoff 后活动；
   现有固定投影不是全历史。
3. 获取 checkpoint 到真实训练 run、不可变数据及 split、过滤转换和
   normalizer 的可审验绑定。缺乏映射时训练独立性 unknown。
4. 在不访问 outcome 的授权身份输入就绪后，选择符合当时保密与来源要求的
   有界碰撞检查路径；现有完整验证会读 witness/target，不能直接用于仍禁止
   读取这些文件的操作约束。独立保管和原来源真实性需另行审查。

以上缺口解除前不能声称 global freshness、独立确认或 P1 准入。
