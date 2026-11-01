# D2P E1 同目标保存消费与预算边界

本消费者只读两个固定路径、固定 SHA 的既有保存资产，向 stdout 输出结果，不写资产、
不运行模型或物理，也不修改科学门槛。

- 强制候选物理 paired summary SHA
  `9d15df249e17ecc35a99ad57db38ef6fc581ca0f1f7e5e7a40bdf406a6431efd`：
  两个已知根，三臂，cp0/5/10，固定 18 个检查点。消费者检查格子唯一完整、
  同根的已观察 native goal 七维精确一致、缺失状态和原始 report/receipt 哈希声明。
  输出保存的各格原值，保留缺失；不计算排名或合并不同根。
- 同目标四槽 controller 保存 receipt SHA
  `f385e1de76ca391a7ef4d03c198211cd303eb2e4d2bc3bdd69537146e8330eba`：
  每槽每模型轴 controller 144、extra 2、replay 4；四槽每轴 600。模型轴互相嵌套，
  不能相加当 FLOPs。此 receipt 是先前候选选择阶段的四槽账，不能当作强制候选六槽
  的每臂完整预算；六槽物理 panel 的总计数另行输出，不能分摊为各臂已证成本。
- 另一个六槽 common controller receipt 记录各槽 controller 144、replay 4，但
  中间目标因方法而异，且历史 proposal/生成费用不同。跨面板虽有同任务最终目标，
  不等于同中间目标、等信息或等端到端预算，因此不纳入本消费者的比较。

工具不输出任何排名；状态固定为 `blocked_equal_budget_comparison`。原因是每臂
proposal 和继承开销、完整端到端逐轴账及 FLOPs 未闭合。代码中 `check_comparable`
只对未来人工构造的声明做必要条件检查：根、步骤、任务、目标、指标及所有规定
预算轴（attempted/completed）须一致；缺项或不一致拒绝。该函数不是 receipt
验证器，数值相等也不证明计量单位相同、来源真实、信息量公平或可作科学排名。
当前 CLI 不接入该声明接口，也不从固定资产凭空补全账本。

信任边界：原物理 summary 的同 target 结论和 strict receipt 验证由上游继承；
本消费者只对固定文件 bytes 和选定字段复核，不重做上游 fullstrict，不重新验证
latent target 字节、物理轨迹或模型 IO。已观察 native goal 比对不能替代 latent
target 证明；已知两根不是独立新根。

验证命令（CPU、只读既有来源；测试只使用固定来源和内存中的篡改副本）：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_e1_strict_saved_consumer_20261030_v1.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_e1_strict_saved_consumer_20261030_v1
```
