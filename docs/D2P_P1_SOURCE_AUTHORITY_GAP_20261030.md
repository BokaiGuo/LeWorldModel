# P1 checkpoint 训练来源与新 reset 历史 authority：只读核查

本次不选根、不读取候选和 outcome、不访问 P4 保留槽、不运行模型、环境、reset 或 GPU。
只读指定 checkpoint 文件、目录文件名及五份固定来源文档／配置；没有全盘搜索，
没有读取训练数据数组，也没有反序列化权重。

## checkpoint 训练来源

本轮实际流式复算 `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt`
SHA256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`、
`config.json` SHA256 `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`，
均匹配既有 P1 来源审计。该目录可见文件 `.gitattributes`、`README.md`、`config.json`、
`weights.pt`，没有独立训练来源清单文件；这不是其他目录或隐藏子目录不存在的证明。
README 的数据集名称和当前训练配置的 `pusht_expert_train.lance` 是来源线索，
不能证明 checkpoint 当时的训练数据字节或 split。

既有有界审计也未建立 checkpoint → 训练 run → 不可变数据版本／episode split →
转换／normalizer 的链条；先前公开历史查询得到的源码和权重下载 revision 不等于
训练时实际加载的版本。需要训练方提供与上述权重 SHA 绑定的 run receipt、
实际代码／已解析配置、数据版本及完整训练成员和过滤、转换与 normalizer 来源，
再独立审验真实性及完整范围。仅一个自述 manifest 或匹配 hash 也不能证明穷尽。
没有训练集排除证明不代表一个新模拟实例必然在训练集内；它表示训练独立性 unknown。

## 生成新 reset 前的历史排除 authority

按既有固定范围，A–H 八来源有 16 条继承引用，已定位 10 次 dispatch，B 包含三次。
逐次需要原始穷尽 reset ledger，或者原实际加载闭包加可证明穷尽的有限包络；
均须涵盖失败、未选、显式重建、replay 和相关重启。还需原负责人可追溯地裁定
机器／工作树／cutoff、未保存失败、cutoff 后新增活动，以及身份到原 episode 的映射。
当前检查未获得新响应，不能认为十次均已补齐；也不能推断未来没有新证据。
可复用的正式 response 语法和字节验收是
`agent_freshroot_authority_20261028_v1`；通过其语法检查仍需来源真实性独立审读，
不能自动解除继承 unknown。

## 本次独占工具及验证

`le-wm-main/experiments/spectral_reachability/codex_p1_source_authority_gap_20261030.py`
固定检查上述权重、配置及文档 SHA，对已知十个 call ID 对照固定文档；
输出七类训练来源、逐调用和五类 scope 缺项。可选 receipt 仅接受精确字段及
证据 SHA，不接收 seed、state、score、outcome 或额外字段，不跟随或读取证据；
它是**声明库存**，不是 evidence validator，所有证据哈希及真实性均未核验，
即使字段齐全也固定不认证、不授权。实际本次没有传 receipt；输出缺项是
“本次未提交”的意思，不是每项在其他地方不存在。

脚本 SHA `03bcc3665b17c9ebee57022a9d927909c032e38c3943407ef51dececae49696b`；
测试 SHA `ecc3643a00b4c807b89ef298ad96517120fa0859ee4f487961e1c27641b7b73a`；
最终结果 `le-wm-main/results/codex_p1_source_authority_gap_20261030_LOCAL02.json`
SHA `da197b3a6586e31b1c11f0353831be64bb3ecdb6e74adf972a42acd155ba1b24`。
初版 LOCAL01 先于补充固定 call 文档绑定而生成，保留但不作为最终复核结果。
最终测试 10 passed，含 B 三调用、伪完整输入不提升 authority、额外/outcome 字段、
错误权重、非法 hash、重复键和非有限 JSON 拒绝。

在 `le-wm-main` 使用新的输出名复现：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_source_authority_gap_20261030 --output results/codex_p1_source_authority_gap_20261030_LOCAL03.json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p1_source_authority_gap_20261030.py
```

收到来源方 receipt 时，可将元数据文件放在 `results` 下，
命名为 `codex_p1_source_authority_receipt_*.json`，并附 `--receipt` 的绝对路径和
`--receipt-sha256`；先用该工具盘点声明，再将真实附件交已有 authority 验收工具和
责任方审读。不要把 receipt 内容、字段齐全或本报告当作模拟采集许可。
