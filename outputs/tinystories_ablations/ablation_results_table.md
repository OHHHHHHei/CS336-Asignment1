# TinyStories 实验结果对比

除特别说明外，所有已完成运行均使用 20,000 个 optimizer steps、50 个验证 batch 和 seed 42。相对值均以受控基线为参照：`batch_size=64`、peak learning rate `3e-4`，最终验证集 loss 为 `1.4360`。

| 类别 | 实验 | 关键改动 | 最终验证集 loss | PPL | 相对对照 | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 架构消融 | 受控基线 | Pre-Norm + RMSNorm + RoPE + SwiGLU | 1.4360 | 4.2039 | 0.0000 | 完成 |
| 架构消融 | No RMSNorm | 移除 RMSNorm | 1.4655 | 4.3297 | +0.0295 | 完成 |
| 架构消融 | Post-Norm | Pre-Norm -> Post-Norm | 1.4357 | 4.2024 | -0.0004 | 完成 |
| 架构消融 | NoPE | 移除 RoPE | 1.5091 | 4.5224 | +0.0730 | 完成 |
| 架构消融 | SiLU | SwiGLU -> SiLU FFN | 1.4582 | 4.2983 | +0.0222 | 完成 |
| RMSNorm 补充实验 | No RMSNorm，lr=1e-4 | 降低 learning rate | 1.6744 | 5.3357 | +0.2384 | 完成 |
| 学习率 sweep | lr=1e-4 | Peak lr 1e-4 | 1.6294 | 5.1006 | +0.1933 | 完成 |
| 学习率 sweep | lr=2e-4 | Peak lr 2e-4 | 1.4923 | 4.4474 | +0.0563 | 完成 |
| 学习率 sweep | lr=3e-4 | Peak lr 3e-4 | 1.4360 | 4.2039 | 0.0000 | 完成 |
| 学习率 sweep | lr=6e-4 | Peak lr 6e-4 | 1.3727 | 3.9462 | -0.0633 | 完成 |
| 学习率 sweep | lr=1e-3 | Peak lr 1e-3 | 1.3447 | 3.8371 | -0.0913 | 完成 |
| 学习率 sweep | lr=3e-3 | Peak lr 3e-3 | 1.3171 | 3.7327 | -0.1189 | 完成；未发散 |
| Batch size sweep | batch=1 | Batch size 1 | 2.3195 | 10.1703 | +0.8835 | 完成 |
| Batch size sweep | batch=32 | Batch size 32 | 1.4986 | 4.4754 | +0.0626 | 完成 |
| Batch size sweep | batch=64 | Batch size 64 | 1.4360 | 4.2039 | 0.0000 | 完成 |
| Batch size sweep | batch=128 | Batch size 128 | 1.4007 | 4.0582 | -0.0353 | 完成 |
| Batch size sweep | batch=192 | Batch size 192 | - | - | - | 第 1 步 OOM |

## 备注

- 原始 baseline 运行记录的最终验证集 loss 为 `1.4499`；为了公平比较新增 sweep，表中使用 seed-42 的受控运行作为基线。
- 测试范围内，较高的 peak learning rate 持续带来更低的最终 loss，但 `3e-3` 没有发散。教案要求的发散运行仍需使用更大的 learning rate 补充。
- Batch size 128 成功完成；Batch size 192 在完整训练配置下超过了 24 GB RTX 3090 的显存限制。
