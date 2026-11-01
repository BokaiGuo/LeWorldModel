# P5 历史未选尝试与 freshness：定向只读追溯

## 结论与范围

截至本次检查，**不能认证新的 P5 fresh cohort 或完整历史排重**。可定位原八来源与十条命令意图，且旧 authority 已投影部分历史身份；但未得到原进程载入字节证明、所有失败/未选/重启的完整 reset ledger，或覆盖旧 cutoff 后全部活动的外部批准。未选根的数量和身份因此是 unknown，不是零。只核现有身份元数据、指向的十条命令日志和当前源码；没有打开预留 outcome 文件、运行环境/模型，或按结果选根。引用 D 的命令意图不表示读取 D 的 outcome 报告或其内容。本次不是全盘穷尽搜索。

## 已可核的历史定位

旧 P5 前置结果 `results/agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json` SHA `d453e4cf70518839844916566fa274959c515d05310ae090fb5699e589cf3133`：`blocked_no_certified_fresh_cohort`、认证根为空、P5 确认分母 null、`reserved_outcomes_opened=false`。旧 catalog `results/agent_freshroots_resume_20261027_coverage.json` SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb` 保存每源原 report 路径/SHA、命令、调用ID、候选恢复源码 SHA、条件 seed 范围及 unresolved。这里继承其报告元数据，不读取其 outcome。

本次直接核以下日志**指定命令行**，路径 `/home/gbk/.codex/sessions/2026/10/23/rollout-2026-10-23T21-46-56-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a16903-c194-71a3-972b-7a268203b373.jsonl`，本次文件 SHA `0a2c097548ef4bd9818affcf82ecb53102ddfa965498b966c13aa7067efdf70c`。SHA 为当前完整日志快照，不声称历史进程自证；只投影指定行的 dispatch，而非打开其他 outcome。

| 来源 | 指定行：call_id 后缀 | catalog 中重放补丁得到的候选 producer SHA |
|---|---|---|
| A | 58279: `981f76dba2b241d79588468b36e979d7` | `5215f0b49b772d8def14997996774bfd49db614b52b9f615c164271591f7b83b` |
| B | 22791: `a41c0e8d73624b16959e29b9f05fd4ed` | `22518d203282b7b1a3aed99446f2262c11a9b42be4f6f2eab394e8c20801b9cd` |
| B | 22814: `3d96838206a84aba9b97fe0ce82bf8e4` | `2830d8ace84341e48423b2fdb635f19ea4449c2b5edfc99806e5d1581a6c372a` |
| B | 22850: `4b779dfb4ad045b696db7409d861c35f` | `6170c69352462b8755b729b0e8dd32cdb873364944a60be9756f6d8935ad1962` |
| C | 23599: `b0f211dffc4e451fab05b0827fdfa415` | `567bcbe1b4edc9241806f1b60882de5d971cab694620153d992199d984b5f3ac` |
| D | 23773: `871afe66f51744c3a8964fe49fab6b08` | `24171e996addc6910d30f800f8ad7c03817f9296b95ae4c675cf9dd3168006f3` |
| E | 58307: `7c4638e67c234a65ad8b02be380375cb` | `80b9674063360a2f51d2e146c830c29a5399bb53b9086b177f3b3f2dc3b0d5e4` |
| F | 23658: `e0f5ab66af7848a89e59e21a975c4165` | `bc3b93478c3a29bf6f3dc0b0f4b478b6be9f963e1fb0384ec5776b9cf5d4b2d6` |
| G | 23730: `a0928b48326a4f43b88a25a223834196` | `06d2c0d2d279f43bdd094006d8d4095a800366cfdc2fef36342b8ca261359908` |
| H | 58363: `947bf4c5b1ce4004a5ab5c4e2ef3d0e5` | `6dc263e6f69840479daedd92a064bead928b1b1b80e1628f46e4c44964525806` |

实际 call_id 均为表中后缀加 `call_`。B 三次命令指向同一输出但分别出现在修复前后：不能只按最终 report 计算一次尝试。catalog 对十条均标 `dispatch_intention_only=true`、`process_return_attested=false`、`execution_source_attested=false`，root-factory 已重放路径均为空；候选 SHA 不是原 Python 进程 loaded-bytes attestation，也不是每次成功启动的证明。A/E/H 三个工具调用行包含多条命令/补丁，行号不是单独进程日志。

## 当前代码能解释的机制，不是历史证明

当前 P0 `experiments/spectral_reachability/preflight_pusht_d2p_p0.py` SHA `745a13d47934f1862a2b43a62b20bc07fc7308075b7b6fea26c3236d9ffa4c98`，P1 `experiments/spectral_reachability/preflight_pusht_d2p_p1_reconfiguration.py` SHA `57bfd63b078b8df26fefae75786156ddb611c271b2630cacdfadbf179008e7c7`，都调用 `collect_nonterminal_boundary_contact_roots`。当前调用链为：

1. `experiments/audit_pusht_reachability_horizon.py:84` SHA `73871055f10b09aea15f03acf87b04e79e30d546668d44bb8160c7be3c439a67`：生成 `count*4` 候选，再按 root fingerprint 和非终态条件选择到 count；未选候选不是最终 rows。
2. `experiments/run_pusht_mode_aware_shared_candidate.py:387` SHA `1b027be83024ea1d4b868c86f2c1340032596520256e5eb5d7a9be869ad203e6`：默认从 terminal roots 生成 boundary-contact roots；使用独立 RNG 和每个候选最多64个几何尝试，失败可中断。
3. `experiments/run_pusht_signed_torque_audit.py:189` SHA `f35fe78b70237fcb7f10cabc950b30c163318c5eb5121cbaff8a5081e5ad3493` 调用 `experiments/run_causal_visual_pusht_world_model.py:443` SHA `261f609e91da9110f8ba5254033529527314d6bcf38e058c3a237c920d3c4190` 的 `terminal_pose_cases`；当前默认 attempt budget 为候选数的两倍，随机 reset 使用 `seed+index`，显式 state/goal placement reset 用 seed 0。

例如当前函数、默认参数且确实完整进入此路径时，4个目标会请求16个 base roots，随机 reset **最多**32次，并有显式 placement reset；不等于历史实际执行了32次，更不能把 seed 0 当新随机根。不同路径、异常、重启、goal render 和历史依赖均须原始证据。catalog 的 A–H 条件范围均标未认证：A `[202753000,202753031]`、B `[202610241,202610272]`、C `[202710241,202710272]`、D `[202710281,202710472]`、E/H `[202753010,202753041]`、F/G `[202710251,202710282]`，这里照 catalog 的端点逐字列示，**不是**已执行集合或获准范围；不能自行转成另一端点约定进行排重。当前源码和这些候选重建版本并非所有历史调用的已载入同一版本。

## 可排除的已知身份与不可恢复项

`results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json` SHA `032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00` 的固定快照记录 480 captured files、173 带直接 root identity、614 inherited refs 无直接 identity、16 个 reset-bound 未认证 refs、6 个 mapped original records；这些是该快照统计，不是全历史穷尽。其 `physical_identity_bridges` 有四条明确投影：reachable root0/root1 和 moving root0/root1，分别映射 source rows 与两个 moving tasks；源 journal SHA 分别为 `786c92cf19c525923dc79c121701bf8fdc5c36a12d08602f2d98b258bd5cc138`、`f30790272a2d2bd1eaa0d533e1767577c69c8608d98b4fb3a3705bd9b70490e9`。四记录共享两个 seed 对，不能当四个独立根；导出的 state/snapshot/RGB3/raw10 可作为相应已知身份的有限排除输入，不能覆盖所有未选根或 solver cache。

`results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json` SHA `a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060` 提供旧固定 inventory 的 seed/span、身份与 unresolved 投影；旧文档称四候选在该**不完整清单**无已知冲突，非 freshness 认证。resolver 中 `MAIN_coverage_approval=false`。原8来源的未选 attempt→state/goal/episode/snapshot 对应、运行实际加载的 factory/env/wrapper 字节、进程失败前进度及未记录重启，当前证据均未闭合；不得从最终 root_count、当前代码、source 路径、mtime、不同 seed 或检索无命中推导出来。

## 下一步：向原运行保管者逐调用索取，不生成新根

1. 复用 `agent_freshroot_authority_20261028_v1` 已有 request/validate 接口，将上述十个 call_id 和原 report path/SHA 逐一交原运行保管者。请求其提供带来源 path/SHA/JSON pointer 的完整按序 reset/attempt/restart ledger，包含随机、显式 placement、goal-render/replay、失败/未选/拒绝，以及每个成功实际身份和失败 unknown；或提供实际 loaded producer、四层 factory、environment/wrapper 及 P1→P0 依赖的原字节闭包，加可证覆盖所有重启的有限执行包络。B 的三调用分别回答，不能只给最后结果。可明确回复不可恢复，不能补造。
2. MAIN 在不读取 outcome 前固定历史 scope/cutoff，提供 cutoff 后全部已接触、失败、未选和预留 prepare 的 delta inventory 与可验证身份桥；沿旧 manifest 合并而非抹除 unknown。随后由 MAIN 对完整内容 SHA/scope/cutoff 作外部 coverage approval；缺任一条仍 blocked。
3. 独立 P5 确认须另冻结 N、候选顺序、采样分布、模型和全 ABI、失败分母与预算。只有另授权后取得实际新候选全 attempt journal 和 state7/pose/snapshot/RGB3/raw10，才按原预注册规则做历史与批内排除；没有身份的尝试保持 unknown，不能自动换根或解封 outcome。不同 seed 不是独立采样证明。

本文件不修改旧 gate，不批准物理执行，不签 P1/P5/freshness，也不证明项目中不存在其他历史资产。
