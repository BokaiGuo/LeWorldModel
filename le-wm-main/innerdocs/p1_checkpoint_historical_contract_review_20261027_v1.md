# 历史合同最终只读 review

输入：本轮独立 public lookup 与 GitHub API follow-up 的 report/protocol、source seals、原始响应和解码文本。不再联网，不执行 transform fixture、神经或物理；不改旧 seal，P1=false。

验证所有 raw body、官方 Git blob 身份、两父报告/协议和 source seals。对已取回官方 pinned tree 中固定文件，尝试将本地完整文件逐字节 Git blob SHA 与远端树比较；仅相同才列为公开源码字节 witness，不靠代码相似猜测。新增三项纯 hash/边界 tests。

已查明官方历史：2026-03-23 initial commit；2026-05-14 19:33 将训练格式转为 Lance；同日19:37 make normalizer pickleable；2026-05-22 fix data config 为查询时官方 head。HF 本轮不可达；公开 GitHub releases/tags 请求成功但列表为空。GitHub API 得到历史/树和部分文本后遇匿名共享配额403，停止API查询，不把失败等同于无元数据。公开 GitHub网页 HEAD 也失败；未使用镜像或认证，也不联系作者。

正常情况下官方 utils/train 公开代码最多给出算法定义；要把精确均值/方差、数据 camera/codec 归到固定 checkpoint，仍须 weights→run/source/dependency/dataset 的明确绑定。本次连官方 utils/train 正文也因限流未取齐，只记录准确 blob SHA/URL，不能用本地相似实现冒充已取得官方历史代码。

现有具体可检验点：公开树支持两个 source-history revision；data config 的 `.lance` 与模型 config 若完整 Git blob 匹配，可确认为该公开版本的内容。已证公开历史在5月14日转Lance，但旧版本具体reader/codec不能只由commit message确定；也不能据 checkpoint 本地缓存时间断言用了哪一侧，更不能将当前 Lance JPEG95 默认值当作历史 codec。依赖 main 即使抓取成功也不提供训练依赖版本身份。

```bash
.venv/bin/python -m pytest -q tests/test_p1_checkpoint_historical_contract_lookup_20261027_v1.py tests/test_p1_checkpoint_historical_contract_github_api_20261027_v1.py tests/test_p1_checkpoint_historical_contract_review_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_checkpoint_historical_contract_review_20261027_v1 --out-dir results/p1_checkpoint_historical_contract_review_20261027_v1_readonly_v1
```

report 中 decisive_witness_needed/next_readonly_targets 是精确缺口，不是新实验授权。找不到保持 unknown，不替换动作stats、不修改camera、不声称P1代码全排除或收益。
