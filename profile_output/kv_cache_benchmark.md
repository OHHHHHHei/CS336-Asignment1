# KV Cache 推理性能对比

本次测试使用同一个 TinyStories checkpoint、同一个 prompt 和贪心解码。
`KV cache` 使用预分配的静态 K/V 空间；`No KV cache` 每一步重新计算最近上下文。

## 测试设置

- checkpoint：`/data/leejt/cs336_assignment1/runs/tinystories_base/checkpoint.pt`
- cache 实现：`preallocated_static_kv_cache`
- prompt：`Hello, she said`
- 上下文长度：`256`
- 生成长度：`[16, 32, 64, 128, 192]`
- 设备：`cuda`
- warmup：`2` 次，正式测量：`3` 次

## 结果

| 生成 token 数 | KV cache 总耗时 (ms) | 无 cache 总耗时 (ms) | KV cache 平均 decode (ms/token) | 无 cache 平均 decode (ms/token) | KV cache 吞吐 (token/s) | 无 cache 吞吐 (token/s) | 输出一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 16 | 63.00 | 69.21 | 3.86 | 4.30 | 259.02 | 232.30 | True |
| 32 | 123.18 | 138.12 | 3.80 | 4.30 | 263.04 | 232.64 | True |
| 64 | 247.44 | 275.92 | 3.83 | 4.29 | 261.17 | 232.86 | True |
| 128 | 493.51 | 558.76 | 3.83 | 4.35 | 261.01 | 229.95 | True |
| 192 | 737.43 | 840.14 | 3.82 | 4.36 | 261.85 | 229.39 | True |

`avg decode token latency` 从第二个生成 token 开始统计，排除了 prompt prefill。
峰值显存包含模型参数、临时张量和 KV cache 本身。

## 性能测试文件

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `cs336_basics/generate.py` | 修改 | 增加 KV cache / 全量重算开关 |
| `cs336_basics/nn.py` | 修改 | 使用每层预分配的静态 K/V 空间 |
| `scripts/benchmark_kv_cache.py` | 新建 | 可复现的延迟、吞吐和显存测量脚本 |
| `outputs/tinystories_base/kv_cache_benchmark.json` | 生成 | 原始测量结果 |
| `outputs/tinystories_base/kv_cache_benchmark.png` | 生成 | 四项指标对比图 |
