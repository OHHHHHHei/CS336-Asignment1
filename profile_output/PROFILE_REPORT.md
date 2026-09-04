# TinyStories BPE 性能分析报告

## 分析对象

目标函数：`cs336_basics.train_bpe.train_bpe`

全量运行使用 TinyStories 训练文本、10,000 个 token 的词表以及已有的特殊 token `<|endoftext|>`。性能分析 wrapper 只返回词表和 merge 数量，不会写入 `vocab.json` 或 `merges.txt`，因此已有 tokenizer 产物保持不变。

## 全量数据资源测量

| 指标 | 数值 |
| --- | ---: |
| 输入大小 | 2,227,753,162 bytes |
| 词表大小 | 10,000 |
| merge 操作数 | 9,743 |
| 墙钟时间 | 14:28.24 |
| 预分词阶段 | 约 13:34 |
| BPE merge 阶段 | 约 00:35 |
| User CPU time | 855.53 s |
| System CPU time | 7.91 s |
| CPU 利用率 | 99% |
| 峰值常驻内存 | 11,381,472 KiB（约 10.85 GiB） |
| Major page faults | 8 |
| Swap 操作 | 0 |
| 退出状态 | 0 |

## 函数级 profile

5 MiB fixture 的 profile 运行耗时 20.46 秒，峰值 RSS 为 53,540 KiB。主要计算集中在 merge selection 循环：

- 反复执行的 `max(stats.items(), key=...)` 累计耗时 15.724 秒，共约 9,970 次调用。
- 该 selection 使用的 comparison lambda 被调用约 5,370 万次，自身耗时累计 5.253 秒。
- Regex 预分词在 profile 中约耗时 0.443 秒。

Fixture profile 表明，遍历完整 `stats` 字典来选择 pair 是 merge 阶段内部的主要热点。全量数据的时间测量则说明，端到端瓶颈更早出现在 regex 预分词和原始计数构建：训练文件包含约 272 万个 document segments，这两个阶段约占总墙钟时间的 94%。倒排索引可以把 pair 更新限制在局部范围，但无法消除每次 merge 都要寻找最高频 pair 的全局扫描。

## 结果解释

对于 TinyStories 训练文件，当前实现的资源消耗是可接受的：运行在 15 分钟内成功完成，峰值内存也明显低于教案给出的 30 GB 指导值。内存峰值主要来自同时保留完整文本，以及 token frequency、word list、pair statistics 和 inverted index 等数据结构。

如果目标是提升全语料吞吐，优先考虑并行或流式预分词，以及更紧凑的原始计数构建，这两项预计带来最大的收益。对于 merge 阶段，可以使用带 lazy invalidation 的 heap 或 priority queue，减少反复进行的全局 pair selection；同时保留当前倒排索引的增量更新策略。上述优化没有在本次测量中实现，也没有用于提交结果。

## 分析产物

- `profile_output/profile_tinystories_bpe.py`
- `profile_output/tinystories_bpe_sample.prof`
- `profile_output/tinystories_bpe_sample_profile.txt`
- `profile_output/tinystories_bpe_sample_time.txt`
- `profile_output/tinystories_bpe_full_run.log`
- `profile_output/tinystories_bpe_full_time.txt`

## 插桩变更记录

| 文件 | 变更类型 | 新增内容 | 行数 |
| --- | --- | --- | --- |
| `profile_output/profile_tinystories_bpe.py` | 新建 | 可复现的 cProfile 和全量 BPE 资源测量 wrapper；不会修改 tokenizer 输出 | - |
| `profile_output/tinystories_bpe_sample.prof` | 生成 | 5 MiB fixture 的 cProfile 二进制结果 | - |
| `profile_output/tinystories_bpe_sample_profile.txt` | 生成 | 可读的累计耗时 profile 摘要 | - |
| `profile_output/tinystories_bpe_sample_time.txt` | 生成 | fixture 的 `/usr/bin/time -v` 资源摘要 | - |
| `profile_output/tinystories_bpe_full_run.log` | 生成 | 全量数据的进度和完成日志 | - |
| `profile_output/tinystories_bpe_full_time.txt` | 生成 | 全量数据的 `/usr/bin/time -v` 资源摘要 | - |
