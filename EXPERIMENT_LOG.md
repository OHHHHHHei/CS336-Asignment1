# CS336 Assignment 1 实验日志

本文档记录当前工作区中已经完成的 TinyStories 训练、架构消融和超参数实验。下表中的数值均从对应运行目录的 `metrics.jsonl` 读取。

## 1. 共同实验配置

- 数据集：TinyStories 训练集/验证集划分
- 词表大小：10,000
- 上下文长度：256
- Transformer：4 层、16 个 attention heads、`d_model=512`、`d_ff=1344`
- 位置编码：基线使用 RoPE
- 归一化：基线使用 Pre-Norm RMSNorm
- 前馈网络：基线使用 SwiGLU
- 优化器：AdamW，weight decay 为 `0.01`
- 学习率策略：1,000 步 warmup，随后使用 cosine decay
- 梯度裁剪：全局 L2 norm 上限为 `1.0`
- 训练长度：20,000 个 optimizer steps
- 验证：每 500 步使用 50 个随机验证 batch
- 随机种子：受控基线和本次新增运行均使用 `42`

本次新增实验统一使用 `tinystories_lr3e-4` 作为公平对照：seed 为 `42`、batch size 为 `64`、peak learning rate 为 `3e-4`，最终验证集 loss 为 `1.4360`。更早的 baseline 运行最终达到 `1.4499`，但其配置中没有记录 seed，因此这里只把它作为历史参考，不作为受控对照。

## 2. TinyStories BPE

### 数据与 tokenizer 统计

- 训练文本：2,227,753,162 bytes
- 验证文本：22,502,601 bytes
- 词表：10,000 个 token
- merge 规则：9,743 条
- 最长 token：` accomplishment`
- 最长 token 长度：15 个 UTF-8 bytes
- 编码后的训练集：540,796,778 个 token
- 编码后的验证集：5,461,210 个 token
- 训练集平均 bytes/token：4.1194
- 验证集平均 bytes/token：4.1204
- 训练集和验证集编码耗时：2,903.47 秒（48.39 分钟）
- 序列化 token 类型：`uint16`

最长 token 是一个带前导空格的常见英文单词。这符合 GPT-2 风格 byte-level BPE 的行为：预分词模式会把单词前的空格附着在单词上，因此高频的“空格 + 单词”组合可以被学习成一个 token。

### 性能分析

在 5 MiB TinyStories fixture 上进行的 `cProfile` 运行耗时 20.46 秒，峰值常驻内存为 53,540 KiB。在这个小规模 fixture 中，BPE merge 阶段的主要函数级热点是反复执行 `max(stats.items(), key=...)`；regex 预分词的耗时相对较小。全量数据的进度日志提供了端到端视角：预分词约耗时 13:34，merge 循环约耗时 00:35，总耗时为 14:28.24。因此，全语料运行的主要瓶颈是预分词和原始计数构建，而全局 pair selection 是 merge 阶段内部的主要热点。

全量数据资源测量（没有覆盖已有 tokenizer 文件）结果为：墙钟时间 14:28.24，user CPU time 855.53 秒，system CPU time 7.91 秒，CPU 利用率 99%，峰值 RSS 为 11,381,472 KiB（约 10.85 GiB）。进度条显示预分词约耗时 13:34，BPE merge 约耗时 00:35。运行成功完成全部 9,743 次 merge，退出码为 0，没有发生 swap，也没有明显的 major page-fault 压力。已有 tokenizer 文件未被修改。

性能分析产物：

- `profile_output/tinystories_bpe_sample_profile.txt`
- `profile_output/tinystories_bpe_sample_time.txt`
- `profile_output/tinystories_bpe_sample.prof`
- `profile_output/profile_tinystories_bpe.py`

## 3. 基线与架构消融

| 运行名称 | 改动 | 最终验证集 loss | PPL | 相对对照 | 运行时间（分钟） |
| --- | --- | ---: | ---: | ---: | ---: |
| `tinystories_lr3e-4` | 受控基线 | 1.4360 | 4.2039 | 0.0000 | 75.45 |
| `tinystories_no_rmsnorm` | 移除 RMSNorm | 1.4655 | 4.3297 | +0.0295 | 71.87 |
| `tinystories_post_norm` | Pre-Norm -> Post-Norm | 1.4357 | 4.2024 | -0.0004 | 76.36 |
| `tinystories_nope` | 移除 RoPE | 1.5091 | 4.5224 | +0.0730 | 71.51 |
| `tinystories_silu` | SwiGLU -> SiLU FFN | 1.4582 | 4.2983 | +0.0222 | 74.85 |
| `tinystories_no_rmsnorm_lr1e-4` | 移除 RMSNorm，并降低 peak lr | 1.6744 | 5.3357 | +0.2384 | 71.18 |

### 结果分析

1. 在测试的架构改动中，移除 RoPE 带来的性能下降最大。NoPE 运行最终验证集 loss 为 `1.5091`，比受控基线高 `0.0730`。
2. 在原始学习率下移除 RMSNorm 也会损害性能。后续使用更低学习率的运行结果更差，说明在当前配置下，`1e-4` 没有解决去掉 RMSNorm 后的优化问题。
3. 在这个规模较小的 TinyStories 配置上，Post-Norm 与 Pre-Norm 几乎持平。这只是当前 seed 和计算预算下的观测结果，不能据此认为两种归一化方式通常等价。
4. 在 hidden dimension 和参数量大致匹配的情况下，不带 gated linear unit 的 SiLU 略差于 SwiGLU。

## 4. 学习率 sweep

每个运行的 minimum learning rate 都设置为 peak learning rate 的十分之一。

| Peak learning rate | 最终验证集 loss | PPL | 相对对照 | 状态 |
| ---: | ---: | ---: | ---: | --- |
| `1e-4` | 1.6294 | 5.1006 | +0.1933 | 完成 |
| `2e-4` | 1.4923 | 4.4474 | +0.0563 | 完成 |
| `3e-4` | 1.4360 | 4.2039 | 0.0000 | 完成 |
| `6e-4` | 1.3727 | 3.9462 | -0.0633 | 完成 |
| `1e-3` | 1.3447 | 3.8371 | -0.0913 | 完成 |
| `3e-3` | 1.3171 | 3.7327 | -0.1189 | 完成；未发散 |

### 结果分析

1. 在测试范围内，peak learning rate 越高，最终验证集 loss 越低；当前测试过的最好值是 `3e-3`。
2. 所有运行都没有发散，因此还没有触及稳定性边界。教案中要求的发散运行仍未完成，需要继续尝试更高的 peak learning rate。
3. 所有运行使用相同的 seed、架构、batch size 和 step 数，因此这个 sweep 能给出清晰的同 seed 排名，但不能证明 `3e-3` 在所有设置下都是全局最优值。

## 5. Batch size sweep

所有运行均使用 20,000 个 optimizer steps 和基线 peak learning rate `3e-4`。因此，batch 越大，实际处理的总 token 越多；下面的 loss 差异同时包含 batch size 影响和 token budget 差异。

| Batch size | 已处理 token 数 | 最终验证集 loss | PPL | 相对对照 | 运行时间（分钟） | 状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5.12M | 2.3195 | 10.1703 | +0.8835 | 13.30 | 完成 |
| 32 | 163.84M | 1.4986 | 4.4754 | +0.0626 | 39.66 | 完成 |
| 64 | 327.68M | 1.4360 | 4.2039 | 0.0000 | 75.39 | 完成 |
| 128 | 655.36M | 1.4007 | 4.0582 | -0.0353 | 147.29 | 完成 |
| 192 | - | - | - | - | - | 第 1 步 OOM |

### 结果分析

1. 在固定 step 数的协议下，batch size 从 1 增加到 128 时，验证集 loss 持续改善。
2. 这个趋势不能完全归因于 batch size：batch 128 看到的 token 数是 batch 64 的两倍，而 batch 1 只看到 batch 64 的 1/64。
3. Batch 192 在完整训练配置下超过了 24 GB RTX 3090 的显存限制。Batch 128 是目前成功完成的最大设置。

## 6. TinyStories 文本生成

使用 baseline 在第 20,000 次迭代保存的 checkpoint 进行解码，prompt 为 `Hello, she said`，temperature 为 `0.8`，top-p 为 `0.9`，随机种子为 `42`。模型在第一次生成 `<|endoftext|>` 后停止，包含 prompt 在内共生成 113 个 token。生成结果形成了一个连贯的儿童故事续写，能够保持人物和动物实体的一致，并且正常结束。文本整体较简单且有重复，这与小模型和单一的 TinyStories 数据领域相符。

完整生成文本及必要的文字分析见 `outputs/tinystories_base/generation_report.md`。

## 7. KV cache 推理性能

使用 baseline 第 20,000 次迭代的 checkpoint，固定 prompt `Hello, she said`、上下文长度 256 和贪心解码，在 6 号 RTX 3090 上比较 KV cache 与每一步全量重算。每个长度 warmup 2 次、正式测量 3 次，5 个生成长度的 token 序列均一致。

| 生成 token 数 | KV cache 总耗时 (ms) | 无 cache 总耗时 (ms) | KV cache decode (ms/token) | 无 cache decode (ms/token) | KV cache 吞吐 (token/s) | 无 cache 吞吐 (token/s) | 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 63.00 | 69.21 | 3.86 | 4.30 | 259.02 | 232.30 | 1.10x |
| 32 | 123.18 | 138.12 | 3.80 | 4.30 | 263.04 | 232.64 | 1.12x |
| 64 | 247.44 | 275.92 | 3.83 | 4.29 | 261.17 | 232.86 | 1.12x |
| 128 | 493.51 | 558.76 | 3.83 | 4.35 | 261.01 | 229.95 | 1.13x |
| 192 | 737.43 | 840.14 | 3.82 | 4.36 | 261.85 | 229.39 | 1.14x |

在当前 17M 模型上，KV cache 的优势主要体现在后续 token 的延迟和吞吐。生成长度从 16 增加到 192 时，总耗时加速比从 1.10x 增加到 1.14x。静态 cache 路径的峰值 allocated memory 固定为 362.64 MB，无 cache 路径为 359.90--377.10 MB。

原始测量结果见 `outputs/tinystories_base/kv_cache_benchmark.json`，对比图见 `outputs/tinystories_base/kv_cache_benchmark.png`，测试说明见 `profile_output/kv_cache_benchmark.md`。

## 8. 局限与后续工作

- 尚未进行多 seed 重复实验，因此 Post-Norm 这类差异很小的结果需要谨慎解读。
- 学习率 sweep 仍需要一次真正发散的运行，例如继续提高 peak learning rate。
- Batch size 对比最好在匹配 token budget 的条件下重跑，或者像本文一样明确标注这是固定 step 数的实验。
- 当前日志还没有记录 OpenWebText tokenizer 训练、不同 tokenizer 的对比、OpenWebText 语言模型训练，以及自定义 leaderboard 改动。
