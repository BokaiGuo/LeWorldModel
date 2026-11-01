# P2 H15 双根独立物理候选

固定重建 H15 HOST 独立消费、六槽 physical prepare 及其来源文件；两根 task/context/goal 和原物理 root 相同，各取自己已审计 winner 的 raw15 和 float32 位串。物理执行直接复用六槽的 native `episode`、观察、reset/place、goal 绑定和 stop 路径；两根单独建环境和账本，不混入六槽或继承其物理结果。仍是已观察过的两个 off 根，不是独立新鲜验证。native terminated/truncated 或 warmup/root 已终止均不补动作。

两根上限 constructor/close 2、reset 4、public_step 50、physics_step 508、explicit_render 162；文件 cap、final reserve、disk floor、wall 继承旧六槽实现。物理回调按 intent/return/failure 计数，不含 HOST 账本；不运行神经网络、训练、规划或额外物理筛选。失败后第二根不启动；写盘或进程故障可能没有最终报告。

saved consumer 重建 prepare，验证源报告、动作位串、链/资源数、原生 placement/goal、PNG/轨迹/早停和 native 标签；只消费完整两根结果，partial 不作为完成。它是本独占两根实现，不等同完整旧六槽 strict verifier，也不能独立证明未记录的物理调用或真实宿主；两根 fixture 完整和截断路径通过，但本实现尚未跑真实物理。没有统计显著、相同总预算、科学 admission 或 physical release 主张。
